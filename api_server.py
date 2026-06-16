
import os
import re
import asyncio
import io
import base64
import uuid
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import librosa
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time

from indextts.infer_vllm import IndexTTS

tts = None

# ===== Voice clone API constants and helpers =====

# CWD at import time; mutated by _init_paths() once `args` is parsed so that
# helpers below can find the cloned_voices directory regardless of argv layout.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLONED_VOICES_DIR = os.path.join(CURRENT_DIR, "cloned_voices")
CLONED_VOICES_MANIFEST = os.path.join(CLONED_VOICES_DIR, "manifest.json")
VOICE_ID_REGEX = re.compile(r"^indextts-([a-z0-9_]{1,10})-([a-f0-9]{8})$")
PREFIX_REGEX = re.compile(r"^[a-z0-9_]{1,10}$")
registry_lock = asyncio.Lock()


def _ensure_cloned_voices_dir():
    os.makedirs(CLONED_VOICES_DIR, exist_ok=True)


def _load_manifest() -> list:
    if not os.path.exists(CLONED_VOICES_MANIFEST):
        return []
    try:
        with open(CLONED_VOICES_MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(manifest: list):
    tmp_path = CLONED_VOICES_MANIFEST + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CLONED_VOICES_MANIFEST)


def _generate_voice_id(prefix: str) -> str:
    for _ in range(8):
        suffix = uuid.uuid4().hex[:8]
        voice_id = f"indextts-{prefix}-{suffix}"
        if not any(entry["voice_id"] == voice_id for entry in _load_manifest()):
            return voice_id
    raise RuntimeError("Failed to generate unique voice_id after 8 attempts")


def _decode_audio_to_wav(audio_base64: str, dest_path: str) -> None:
    try:
        raw = base64.b64decode(audio_base64, validate=True)
    except Exception as ex:
        raise ValueError(f"invalid base64 payload: {ex}") from ex
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as ex:
        raise ValueError(f"audio cannot be decoded by soundfile: {ex}") from ex
    sf.write(dest_path, data, sr, format="WAV", subtype="PCM_16")


def _save_conditioning_cache(voice_id: str, entry: dict) -> None:
    cond_path = os.path.join(CLONED_VOICES_DIR, voice_id, "conditioning.pt")
    blob = {
        "auto_conditioning": [t.detach().cpu() for t in entry["auto_conditioning"]],
        "speech_conditioning_latent": entry["speech_conditioning_latent"].detach().cpu(),
    }
    torch.save(blob, cond_path)


def _load_conditioning_cache(voice_id: str, device: str):
    cond_path = os.path.join(CLONED_VOICES_DIR, voice_id, "conditioning.pt")
    blob = torch.load(cond_path, map_location="cpu")
    return {
        "auto_conditioning": [t.to(device) for t in blob["auto_conditioning"]],
        "speech_conditioning_latent": blob["speech_conditioning_latent"].to(device),
    }


def _wav_bytes_to_base64(wav_data: np.ndarray, sampling_rate: int) -> str:
    with io.BytesIO() as buf:
        sf.write(buf, wav_data, sampling_rate, format="WAV", subtype="PCM_16")
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ===== rate / volume post-processing =====

OUTPUT_SAMPLE_RATE = 16000  # Hardcoded API output sample rate (IndexTTS-vLLM native is 24kHz)

_RATE_MIN = 0.25
_RATE_MAX = 4.0
_VOLUME_MIN = 0
_VOLUME_MAX = 100
_RATE_DEFAULT = 1.0
_VOLUME_DEFAULT = 50


def _resolve_speed_or_rate(data: dict, prefer_speed: bool) -> float:
    """Resolve the speed/rate multiplier from a request body.

    prefer_speed=True (/audio/speech): prefer OpenAI 'speed', fall back to 'rate'.
    prefer_speed=False (/api/tts): only 'rate'.
    Out-of-range values fall back to default with a printed warning.
    """
    raw = None
    field_name = None
    if prefer_speed and "speed" in data:
        raw = data["speed"]
        field_name = "speed"
    elif "rate" in data:
        raw = data["rate"]
        field_name = "rate"
    if raw is None:
        return _RATE_DEFAULT
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        print(f"WARNING: ignoring non-numeric {field_name}={raw!r}, using {_RATE_DEFAULT}")
        return _RATE_DEFAULT
    if rate < _RATE_MIN or rate > _RATE_MAX:
        print(f"WARNING: {field_name}={rate} out of range [{_RATE_MIN}, {_RATE_MAX}], using {_RATE_DEFAULT}")
        return _RATE_DEFAULT
    return rate


def _resolve_volume(data: dict, default: int = _VOLUME_DEFAULT) -> int:
    raw = data.get("volume", default)
    if raw is None:
        return default
    try:
        volume = int(raw)
    except (TypeError, ValueError):
        print(f"WARNING: ignoring non-integer volume={raw!r}, using {default}")
        return default
    if volume < _VOLUME_MIN or volume > _VOLUME_MAX:
        print(f"WARNING: volume={volume} out of range [{_VOLUME_MIN}, {_VOLUME_MAX}], using {default}")
        return default
    return volume


def _apply_rate_volume(wav_data: np.ndarray, sampling_rate: int,
                       rate: float = _RATE_DEFAULT,
                       volume: int = _VOLUME_DEFAULT) -> np.ndarray:
    """Post-process synthesized wav: librosa time_stretch + amplitude scaling.

    Input:  int16 array, shape (samples, 1) — what `infer_with_ref_audio_embed` returns.
    Output: int16 array, shape (samples', 1). When rate=1.0 and volume=50 the
    output equals the input up to dtype conversion.
    """
    # Convert to float32 in [-1, 1] for librosa
    y = wav_data.astype(np.float32).reshape(-1) / 32768.0

    if abs(rate - 1.0) > 1e-6:
        # librosa.effects.time_stretch requires at least a few samples; pad if needed
        if y.size < 8:
            y = np.pad(y, (0, 8 - y.size), mode="constant")
        y = librosa.effects.time_stretch(y, rate=rate)

    if volume != _VOLUME_DEFAULT:
        gain = volume / float(_VOLUME_DEFAULT)
        y = y * gain

    y = np.clip(y, -1.0, 1.0)
    out = (y * 32767.0).astype(np.int16).reshape(-1, 1)
    return out


def _resample_to_16k(wav_data: np.ndarray, orig_sr: int) -> tuple:
    """Resample int16 wav to OUTPUT_SAMPLE_RATE. Returns (wav, sr).

    Applied AFTER `_apply_rate_volume` so that time_stretch runs on the
    model's native 24 kHz signal (where the default hop_length gives
    ~21 ms frames appropriate for speech).
    """
    if orig_sr == OUTPUT_SAMPLE_RATE:
        return wav_data, orig_sr
    y = wav_data.astype(np.float32).reshape(-1) / 32768.0
    y = librosa.resample(y, orig_sr=orig_sr, target_sr=OUTPUT_SAMPLE_RATE)
    y = np.clip(y, -1.0, 1.0)
    out = (y * 32767.0).astype(np.int16).reshape(-1, 1)
    return out, OUTPUT_SAMPLE_RATE


def _validate_voice_id(voice_id: str) -> bool:
    """Return True iff voice_id exists in either manifest or assets/speaker.json."""
    if not VOICE_ID_REGEX.match(voice_id):
        # Static speaker names from assets/speaker.json are not required to match the regex
        speaker_path = os.path.join(CURRENT_DIR, "assets/speaker.json")
        if os.path.exists(speaker_path):
            try:
                with open(speaker_path, "r", encoding="utf-8") as f:
                    return voice_id in json.load(f)
            except (json.JSONDecodeError, OSError):
                return False
        return False
    for entry in _load_manifest():
        if entry["voice_id"] == voice_id:
            return True
    return False


async def _load_cloned_voices_on_startup():
    _ensure_cloned_voices_dir()
    manifest = _load_manifest()
    for entry in manifest:
        voice_id = entry["voice_id"]
        audio_path = os.path.join(CURRENT_DIR, entry["audio_path"])
        voice_dir = os.path.dirname(audio_path)
        cond_path = os.path.join(voice_dir, "conditioning.pt")
        try:
            if os.path.exists(cond_path) and os.path.exists(audio_path):
                cached = await asyncio.to_thread(_load_conditioning_cache, voice_id, tts.device)
                tts.speaker_dict[voice_id] = cached
                print(f">> Loaded cached conditioning for {voice_id}")
            elif os.path.exists(audio_path):
                await asyncio.to_thread(tts.registry_speaker, voice_id, [audio_path])
                await asyncio.to_thread(_save_conditioning_cache, voice_id, tts.speaker_dict[voice_id])
                print(f">> Extracted and cached conditioning for {voice_id}")
            else:
                entry["status"] = "failed"
                print(f">> Missing audio file for {voice_id}, marked failed")
        except Exception as ex:
            entry["status"] = "failed"
            tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            print(f">> Failed to load {voice_id}: {tb_str}")
    _save_manifest(manifest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS(model_dir=args.model_dir, gpu_memory_utilization=args.gpu_memory_utilization)

    speaker_path = os.path.join(CURRENT_DIR, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))

        for speaker, audio_paths in speaker_dict.items():
            audio_paths_ = []
            for audio_path in audio_paths:
                audio_paths_.append(os.path.join(CURRENT_DIR, audio_path))
            tts.registry_speaker(speaker, audio_paths_)

    await _load_cloned_voices_on_startup()
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        global tts
        if tts is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "TTS model not initialized"
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "message": "Service is running",
                "timestamp": time.time()
            }
        )
    except Exception as ex:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(ex)
            }
        )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]
        seed = data.get("seed", 8)

        global tts
        sr, wav = await tts.infer(audio_paths, text, seed=seed)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )



@app.get("/audio/voices")
async def tts_voices():
    """ additional function to provide the list of available voices, in the form of JSON """
    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))
        return speaker_dict
    else:
        return []



@app.post("/audio/speech", responses={
    200: {"content": {"application/octet-stream": {}}},
    400: {"content": {"application/json": {}}},
    404: {"content": {"application/json": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_openai(request: Request):
    """ OpenAI competible API, see: https://api.openai.com/v1/audio/speech

    Accepts optional `speed` (OpenAI standard) and `volume` (extension).
    `speed` takes precedence over `rate` if both are present.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "error": "invalid JSON body"})

    text = data.get("input")
    character = data.get("voice")
    if not isinstance(text, str) or not text:
        return JSONResponse(status_code=400, content={"status": "error", "error": "input is required"})
    if not isinstance(character, str) or not character:
        return JSONResponse(status_code=400, content={"status": "error", "error": "voice is required"})
    # `model` is required by OpenAI but not used here; accept silently
    _model = data.get("model")

    if not _validate_voice_id(character):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": f"voice '{character}' not found"},
        )

    rate = _resolve_speed_or_rate(data, prefer_speed=True)
    volume = _resolve_volume(data)

    try:
        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(status_code=500, content={"status": "error", "error": tb_str})

    wav = _apply_rate_volume(wav, sr, rate=rate, volume=volume)
    wav, sr = _resample_to_16k(wav, sr)

    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, wav, sr, format='WAV', subtype='PCM_16')
        wav_bytes = wav_buffer.getvalue()

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/clone-voice")
async def api_clone_voice(request: Request):
    """Clone a voice from a base64-encoded audio payload.

    Request:  {"audio_base64": str, "prefix": str}
    Response: {"voice_id": str, "status": "ready"}
    Errors:   400 (invalid input), 500 (extraction failed)
    """
    global tts
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "error": "invalid JSON body"})

    audio_b64 = data.get("audio_base64")
    prefix = data.get("prefix")
    if not isinstance(audio_b64, str) or not audio_b64:
        return JSONResponse(status_code=400, content={"status": "error", "error": "audio_base64 is required"})
    if not isinstance(prefix, str) or not PREFIX_REGEX.match(prefix):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "prefix must match [a-z0-9_]{1,10}"},
        )

    _ensure_cloned_voices_dir()
    manifest = _load_manifest()
    voice_id = _generate_voice_id(prefix)
    voice_dir = os.path.join(CLONED_VOICES_DIR, voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    audio_path = os.path.join(voice_dir, "audio.wav")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".audio", dir=voice_dir)
    os.close(tmp_fd)
    try:
        _decode_audio_to_wav(audio_b64, tmp_path)
    except ValueError as ex:
        os.remove(tmp_path)
        shutil.rmtree(voice_dir, ignore_errors=True)
        return JSONResponse(status_code=400, content={"status": "error", "error": str(ex)})

    os.replace(tmp_path, audio_path)

    try:
        async with registry_lock:
            await asyncio.to_thread(tts.registry_speaker, voice_id, [audio_path])
            await asyncio.to_thread(_save_conditioning_cache, voice_id, tts.speaker_dict[voice_id])
    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        shutil.rmtree(voice_dir, ignore_errors=True)
        return JSONResponse(status_code=500, content={"status": "error", "error": tb_str})

    entry = {
        "voice_id": voice_id,
        "prefix": prefix,
        "audio_path": os.path.relpath(audio_path, CURRENT_DIR),
        "audio_filename": "audio.wav",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ready",
    }
    manifest.append(entry)
    _save_manifest(manifest)
    return JSONResponse(content={"voice_id": voice_id, "status": "ready"})


@app.get("/api/voices")
async def api_list_voices():
    """List all cloned voices with their metadata."""
    manifest = _load_manifest()
    return JSONResponse(content=[
        {
            "voice_id": entry["voice_id"],
            "prefix": entry.get("prefix", ""),
            "audio_filename": entry.get("audio_filename", "audio.wav"),
            "created_at": entry.get("created_at", ""),
            "status": entry.get("status", "ready"),
        }
        for entry in manifest
    ])


@app.delete("/api/voices/{voice_id}")
async def api_delete_voice(voice_id: str):
    """Delete a cloned voice by voice_id."""
    if not VOICE_ID_REGEX.match(voice_id):
        return JSONResponse(status_code=400, content={"status": "error", "error": "invalid voice_id format"})

    manifest = _load_manifest()
    target = next((e for e in manifest if e["voice_id"] == voice_id), None)
    if target is None:
        return JSONResponse(status_code=404, content={"status": "error", "error": "voice_id not found"})

    async with registry_lock:
        if hasattr(tts, "speaker_dict") and voice_id in tts.speaker_dict:
            tts.speaker_dict.pop(voice_id, None)
        manifest = [e for e in manifest if e["voice_id"] != voice_id]
        _save_manifest(manifest)
        voice_dir = os.path.join(CLONED_VOICES_DIR, voice_id)
        if os.path.isdir(voice_dir):
            shutil.rmtree(voice_dir, ignore_errors=True)
    return JSONResponse(content={"message": "voice deleted", "voice_id": voice_id})


@app.post("/api/tts")
async def api_tts(request: Request):
    """Synthesize speech with a cloned voice_id.

    Request:  {"text": str, "voice": str, "rate"?: float, "volume"?: int}
              rate: OpenAI-style multiplier, 1.0 = normal, range [0.25, 4.0]
              volume: Aliyun-style 0–100, 50 = normal
    Response: {"audio_base64": str}  (16kHz mono PCM_16 WAV)
    Errors:   400, 404 (voice_id unknown), 425 (voice in failed state), 500
    """
    global tts
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"status": "error", "error": "invalid JSON body"})

    text = data.get("text")
    voice_id = data.get("voice")
    if not isinstance(text, str) or not text:
        return JSONResponse(status_code=400, content={"status": "error", "error": "text is required"})
    if not isinstance(voice_id, str) or not voice_id:
        return JSONResponse(status_code=400, content={"status": "error", "error": "voice is required"})

    manifest = _load_manifest()
    entry = next((e for e in manifest if e["voice_id"] == voice_id), None)
    if entry is None:
        return JSONResponse(status_code=404, content={"status": "error", "error": f"voice_id '{voice_id}' not found"})
    if entry.get("status") == "failed":
        return JSONResponse(status_code=425, content={"status": "error", "error": f"voice_id '{voice_id}' is in failed state"})

    rate = _resolve_speed_or_rate(data, prefer_speed=False)
    volume = _resolve_volume(data)

    try:
        sr, wav = await tts.infer_with_ref_audio_embed(voice_id, text)
    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": tb_str})

    wav = _apply_rate_volume(wav, sr, rate=rate, volume=volume)
    wav, sr = _resample_to_16k(wav, sr)
    return JSONResponse(content={"audio_base64": _wav_bytes_to_base64(wav, sr)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()

    uvicorn.run(app=app, host=args.host, port=args.port)
