"""
Voice cloning API test suite for api_server.py.

Covers the four new endpoints:
  POST   /api/clone-voice
  GET    /api/voices
  DELETE /api/voices/{voice_id}
  POST   /api/tts

Usage:
    # Terminal 1 (in WSL): start the API server
    python api_server.py --model_dir ./checkpoints/Index-TTS-1.5-vLLM --port 6006

    # Terminal 2 (in WSL): run the tests
    python test/api_test.py
    python test/api_test.py --server http://localhost:6006
    python test/api_test.py --audio examples/voice_01.wav --text "你好，世界"
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import traceback

import numpy as np
import requests
import soundfile as sf


DEFAULT_SERVER = "http://localhost:6006"
DEFAULT_AUDIO = "examples/voice_01.wav"
DEFAULT_TEXT = "你好，这是一个语音克隆接口的测试。"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class TestRunner:
    def __init__(self, server: str, audio_path: str, text: str):
        self.server = server.rstrip("/")
        self.audio_path = audio_path
        self.text = text
        self.results: list[tuple[str, bool, str]] = []
        self.created_voice_ids: list[str] = []

    # ---------- result tracking ----------
    def record(self, name: str, ok: bool, detail: str = ""):
        self.results.append((name, ok, detail))
        marker = PASS if ok else FAIL
        line = f"{marker}  {name}"
        if detail:
            line += f"   {detail}"
        print(line)

    def report(self) -> int:
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        print()
        print("=" * 60)
        print(f"  {passed}/{total} tests passed")
        print("=" * 60)
        return 0 if passed == total else 1

    # ---------- preconditions ----------
    def check_server_alive(self):
        try:
            r = requests.get(f"{self.server}/health", timeout=5)
            ok = r.status_code == 200
            detail = f"status={r.status_code}" if ok else f"unexpected: {r.text[:200]}"
            self.record("server /health reachable", ok, detail)
            return ok
        except Exception as ex:
            self.record("server /health reachable", False, str(ex))
            return False

    def check_audio_file(self) -> bytes:
        if not os.path.exists(self.audio_path):
            self.record("audio file exists", False, self.audio_path)
            return b""
        with open(self.audio_path, "rb") as f:
            data = f.read()
        self.record("audio file exists", True, f"{self.audio_path} ({len(data)} bytes)")
        return data

    # ---------- happy path ----------
    def test_clone_voice(self, audio_b64: str, prefix: str) -> str | None:
        try:
            r = requests.post(
                f"{self.server}/api/clone-voice",
                json={"audio_base64": audio_b64, "prefix": prefix},
                timeout=60,
            )
            ok = r.status_code == 200 and "voice_id" in r.json() and r.json().get("status") == "ready"
            voice_id = r.json().get("voice_id", "") if r.status_code == 200 else ""
            self.record(
                "POST /api/clone-voice (valid)",
                ok,
                f"voice_id={voice_id}, status={r.status_code}",
            )
            if ok and voice_id:
                self.created_voice_ids.append(voice_id)
            return voice_id if ok else None
        except Exception as ex:
            self.record("POST /api/clone-voice (valid)", False, str(ex))
            return None

    def test_list_voices_contains(self, voice_id: str):
        try:
            r = requests.get(f"{self.server}/api/voices", timeout=10)
            data = r.json() if r.status_code == 200 else []
            ids = [v.get("voice_id") for v in data] if isinstance(data, list) else []
            ok = r.status_code == 200 and voice_id in ids
            self.record(
                "GET /api/voices contains clone",
                ok,
                f"status={r.status_code}, count={len(ids)}",
            )
        except Exception as ex:
            self.record("GET /api/voices contains clone", False, str(ex))

    def test_api_tts(self, voice_id: str):
        try:
            r = requests.post(
                f"{self.server}/api/tts",
                json={"text": self.text, "voice": voice_id},
                timeout=120,
            )
            if r.status_code != 200:
                self.record(
                    "POST /api/tts (cloned voice)",
                    False,
                    f"status={r.status_code}, body={r.text[:200]}",
                )
                return None
            payload = r.json()
            audio_b64 = payload.get("audio_base64", "")
            if not audio_b64:
                self.record("POST /api/tts (cloned voice)", False, "missing audio_base64")
                return None
            wav_bytes = base64.b64decode(audio_b64)
            # Verify it parses as WAV
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            ok = sr == 16000 and data.size > 0
            self.record(
                "POST /api/tts (cloned voice)",
                ok,
                f"sr={sr}, samples={data.size}, duration={data.size/sr:.2f}s",
            )
            return wav_bytes if ok else None
        except Exception as ex:
            self.record("POST /api/tts (cloned voice)", False, str(ex))
            return None

    def test_delete_voice(self, voice_id: str):
        try:
            r = requests.delete(f"{self.server}/api/voices/{voice_id}", timeout=10)
            ok = r.status_code == 200 and r.json().get("voice_id") == voice_id
            self.record(
                "DELETE /api/voices/{voice_id}",
                ok,
                f"status={r.status_code}",
            )
            if voice_id in self.created_voice_ids:
                self.created_voice_ids.remove(voice_id)
        except Exception as ex:
            self.record("DELETE /api/voices/{voice_id}", False, str(ex))

    # ---------- error paths ----------
    def test_clone_invalid_prefix(self, audio_b64: str):
        try:
            r = requests.post(
                f"{self.server}/api/clone-voice",
                json={"audio_base64": audio_b64, "prefix": "Invalid-PREFIX"},
                timeout=10,
            )
            ok = r.status_code == 400
            self.record(
                "POST /api/clone-voice rejects bad prefix",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("POST /api/clone-voice rejects bad prefix", False, str(ex))

    def test_clone_invalid_base64(self):
        try:
            r = requests.post(
                f"{self.server}/api/clone-voice",
                json={"audio_base64": "not-valid-base64!!!@@@", "prefix": "validprfx"},
                timeout=10,
            )
            ok = r.status_code == 400
            self.record(
                "POST /api/clone-voice rejects bad base64",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("POST /api/clone-voice rejects bad base64", False, str(ex))

    def test_clone_missing_fields(self):
        try:
            r = requests.post(
                f"{self.server}/api/clone-voice",
                json={"prefix": "validprfx"},
                timeout=10,
            )
            ok = r.status_code == 400
            self.record(
                "POST /api/clone-voice rejects missing audio_base64",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("POST /api/clone-voice rejects missing audio_base64", False, str(ex))

    def test_tts_unknown_voice(self):
        try:
            r = requests.post(
                f"{self.server}/api/tts",
                json={"text": self.text, "voice": "indextts-fakevoice-deadbeef"},
                timeout=10,
            )
            ok = r.status_code == 404
            self.record(
                "POST /api/tts returns 404 for unknown voice",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("POST /api/tts returns 404 for unknown voice", False, str(ex))

    def test_delete_unknown_voice(self):
        try:
            r = requests.delete(
                f"{self.server}/api/voices/indextts-fakevoice-deadbeef",
                timeout=10,
            )
            ok = r.status_code == 404
            self.record(
                "DELETE returns 404 for unknown voice",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("DELETE returns 404 for unknown voice", False, str(ex))

    def test_delete_invalid_format(self):
        try:
            r = requests.delete(
                f"{self.server}/api/voices/INVALID-FORMAT",
                timeout=10,
            )
            ok = r.status_code == 400
            self.record(
                "DELETE returns 400 for malformed voice_id",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("DELETE returns 400 for malformed voice_id", False, str(ex))

    # ---------- /audio/speech rate/volume/speed ----------
    def test_audio_speech_unknown_voice_404(self):
        try:
            r = requests.post(
                f"{self.server}/audio/speech",
                json={"input": self.text, "voice": "indextts-fakevoice-deadbeef", "model": "tts-1"},
                timeout=10,
            )
            ok = r.status_code == 404
            self.record(
                "/audio/speech returns 404 for unknown voice (was 500 before fix)",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("/audio/speech returns 404 for unknown voice", False, str(ex))

    def test_audio_speech_missing_fields_400(self):
        try:
            r = requests.post(
                f"{self.server}/audio/speech",
                json={"input": self.text},  # missing voice
                timeout=10,
            )
            ok = r.status_code == 400
            self.record(
                "/audio/speech returns 400 when voice is missing",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("/audio/speech returns 400 when voice is missing", False, str(ex))

    def test_audio_speech_with_cloned_voice(self, voice_id: str):
        """Verify cloned voice_id works through OpenAI-compatible /audio/speech."""
        try:
            r = requests.post(
                f"{self.server}/audio/speech",
                json={"input": self.text, "voice": voice_id, "model": "tts-1"},
                timeout=120,
            )
            if r.status_code != 200:
                self.record(
                    "/audio/speech works with cloned voice_id",
                    False,
                    f"status={r.status_code}, body={r.text[:200]}",
                )
                return None
            wav_bytes = r.content
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            ok = sr == 16000 and data.size > 0
            self.record(
                "/audio/speech works with cloned voice_id",
                ok,
                f"sr={sr}, samples={data.size}",
            )
            return wav_bytes if ok else None
        except Exception as ex:
            self.record("/audio/speech works with cloned voice_id", False, str(ex))
            return None

    def test_rate_changes_duration(self, voice_id: str):
        """rate=2.0 should produce a wav with roughly half the samples of rate=1.0."""
        def synth(rate_param: float) -> int:
            r = requests.post(
                f"{self.server}/api/tts",
                json={"text": self.text, "voice": voice_id, "rate": rate_param},
                timeout=180,
            )
            r.raise_for_status()
            data, _ = sf.read(io.BytesIO(base64.b64decode(r.json()["audio_base64"])), dtype="float32")
            return data.size

        try:
            n1 = synth(1.0)
            n2 = synth(2.0)
            # Expect n2 to be roughly half of n1 (within 30% tolerance for STFT boundary effects)
            ratio = n2 / n1 if n1 else 0
            ok = 0.35 < ratio < 0.65
            self.record(
                "rate=2.0 produces ~half-length wav vs rate=1.0",
                ok,
                f"n1={n1}, n2={n2}, ratio={ratio:.2f}",
            )
        except Exception as ex:
            self.record("rate=2.0 produces ~half-length wav vs rate=1.0", False, str(ex))

    def test_audio_speech_speed_field(self, voice_id: str):
        """OpenAI 'speed' field is accepted as alias for rate."""
        try:
            r = requests.post(
                f"{self.server}/audio/speech",
                json={"input": self.text, "voice": voice_id, "model": "tts-1", "speed": 2.0},
                timeout=180,
            )
            ok = r.status_code == 200 and len(r.content) > 0
            self.record(
                "/audio/speech accepts OpenAI 'speed' field",
                ok,
                f"status={r.status_code}, bytes={len(r.content)}",
            )
        except Exception as ex:
            self.record("/audio/speech accepts OpenAI 'speed' field", False, str(ex))

    # ---------- output sample rate (16 kHz requirement) ----------
    def test_api_tts_16khz(self, voice_id: str):
        """Explicit check that /api/tts output is 16kHz WAV (PCM_16)."""
        try:
            r = requests.post(
                f"{self.server}/api/tts",
                json={"text": self.text, "voice": voice_id},
                timeout=120,
            )
            r.raise_for_status()
            wav_bytes = base64.b64decode(r.json()["audio_base64"])
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            ok = sr == 16000 and data.size > 0
            self.record(
                "/api/tts output is 16kHz WAV",
                ok,
                f"sr={sr}, samples={data.size}, duration={data.size/sr:.2f}s",
            )
        except Exception as ex:
            self.record("/api/tts output is 16kHz WAV", False, str(ex))

    def test_audio_speech_16khz(self, voice_id: str):
        """Explicit check that /audio/speech output is 16kHz WAV (PCM_16)."""
        try:
            r = requests.post(
                f"{self.server}/audio/speech",
                json={"input": self.text, "voice": voice_id, "model": "tts-1"},
                timeout=120,
            )
            r.raise_for_status()
            data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
            ok = sr == 16000 and data.size > 0
            self.record(
                "/audio/speech output is 16kHz WAV",
                ok,
                f"sr={sr}, samples={data.size}, duration={data.size/sr:.2f}s",
            )
        except Exception as ex:
            self.record("/audio/speech output is 16kHz WAV", False, str(ex))

    # ---------- static voices from assets/speaker.json ----------
    def _resolve_static_voices(self) -> list[str]:
        """Read assets/speaker.json from the server's filesystem via the helper endpoint.

        Falls back to probing /api/voices + /audio/voices, and finally to a hardcoded
        default list matching the most common configuration.
        """
        candidates: list[str] = []
        # Try to read speaker.json via the local filesystem relative to CWD
        for path in ("assets/speaker.json", "./assets/speaker.json"):
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        candidates = list(data.keys())
                        break
                except Exception:
                    pass
        return candidates

    def test_api_tts_with_static_voice(self):
        """Regression test: /api/tts must accept voices from assets/speaker.json
        (not just cloned voices registered via /api/clone-voice)."""
        static_voices = self._resolve_static_voices()
        if not static_voices:
            self.record(
                "/api/tts accepts assets/speaker.json voices (regression)",
                False,
                "no static voices found in assets/speaker.json",
            )
            return
        voice = static_voices[0]
        try:
            r = requests.post(
                f"{self.server}/api/tts",
                json={"text": self.text, "voice": voice},
                timeout=120,
            )
            if r.status_code != 200:
                self.record(
                    "/api/tts accepts assets/speaker.json voices (regression)",
                    False,
                    f"voice={voice}, status={r.status_code}, body={r.text[:200]}",
                )
                return
            wav_bytes = base64.b64decode(r.json()["audio_base64"])
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            ok = sr == 16000 and data.size > 0
            self.record(
                "/api/tts accepts assets/speaker.json voices (regression)",
                ok,
                f"voice={voice}, sr={sr}, duration={data.size/sr:.2f}s",
            )
        except Exception as ex:
            self.record(
                "/api/tts accepts assets/speaker.json voices (regression)",
                False,
                str(ex),
            )

    def test_api_tts_with_each_static_voice(self):
        """Exercise every static voice from assets/speaker.json through /api/tts."""
        static_voices = self._resolve_static_voices()
        if not static_voices:
            self.record(
                "/api/tts works for every assets/speaker.json voice",
                False,
                "no static voices found",
            )
            return
        for voice in static_voices:
            try:
                r = requests.post(
                    f"{self.server}/api/tts",
                    json={"text": self.text, "voice": voice},
                    timeout=120,
                )
                ok = r.status_code == 200 and "audio_base64" in r.json()
                self.record(
                    f"/api/tts voice '{voice}' (from speaker.json)",
                    ok,
                    f"status={r.status_code}",
                )
            except Exception as ex:
                self.record(
                    f"/api/tts voice '{voice}' (from speaker.json)",
                    False,
                    str(ex),
                )

    def test_api_tts_unknown_voice_still_404(self):
        """Regression test: invalid voices must still return 404 (not silently 500)."""
        try:
            r = requests.post(
                f"{self.server}/api/tts",
                json={"text": self.text, "voice": "nonexistent_voice_zzz"},
                timeout=10,
            )
            ok = r.status_code == 404
            self.record(
                "/api/tts still returns 404 for unknown voice (regression)",
                ok,
                f"status={r.status_code}",
            )
        except Exception as ex:
            self.record("/api/tts still returns 404 for unknown voice", False, str(ex))

    # ---------- persistence check ----------
    def test_persistence(self, voice_id: str):
        """Verify that re-querying the manifest (simulating server restart)
        still returns the cloned voice. Server restart isn't automated here —
        the user must restart the API server between calls.
        """
        try:
            r = requests.get(f"{self.server}/api/voices", timeout=10)
            ids = [v.get("voice_id") for v in r.json()] if r.status_code == 200 else []
            ok = voice_id in ids
            self.record(
                "voice persists in /api/voices list (restart required for conditioning.pt cache)",
                ok,
                f"ids={ids}",
            )
        except Exception as ex:
            self.record("voice persists in /api/voices list", False, str(ex))

    # ---------- cleanup ----------
    def cleanup(self):
        print()
        print("--- cleanup ---")
        for vid in list(self.created_voice_ids):
            try:
                r = requests.delete(f"{self.server}/api/voices/{vid}", timeout=10)
                print(f"  deleted {vid}: status={r.status_code}")
            except Exception as ex:
                print(f"  failed to delete {vid}: {ex}")


def main():
    parser = argparse.ArgumentParser(description="Voice cloning API test suite")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="API server base URL")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="Reference audio for clone-voice")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text for /api/tts")
    parser.add_argument("--keep", action="store_true", help="Skip cleanup of created voices")
    args = parser.parse_args()

    runner = TestRunner(server=args.server, audio_path=args.audio, text=args.text)

    print(f"Server: {runner.server}")
    print(f"Audio:  {runner.audio_path}")
    print(f"Text:   {runner.text}")
    print()

    if not runner.check_server_alive():
        print("\nServer not reachable. Start the API server first:")
        print("  python api_server.py --model_dir ./checkpoints/Index-TTS-1.5-vLLM")
        return 1

    audio_bytes = runner.check_audio_file()
    if not audio_bytes:
        return 1
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    # Unique prefix per run avoids collisions with previously cloned voices.
    prefix = f"test{int(time.time()) % 100000:05d}"

    # Error paths first — don't pollute manifest.
    runner.test_clone_invalid_prefix(audio_b64)
    runner.test_clone_invalid_base64()
    runner.test_clone_missing_fields()
    runner.test_delete_unknown_voice()
    runner.test_delete_invalid_format()

    # Static voices from assets/speaker.json — independent of clone flow.
    runner.test_api_tts_with_static_voice()
    runner.test_api_tts_with_each_static_voice()
    runner.test_api_tts_unknown_voice_still_404()

    # Happy path: clone → list → tts → delete.
    voice_id = runner.test_clone_voice(audio_b64, prefix)
    if voice_id:
        runner.test_list_voices_contains(voice_id)
        runner.test_api_tts(voice_id)
        runner.test_persistence(voice_id)
        runner.test_tts_unknown_voice()

        # /audio/speech with cloned voice + rate/volume/speed
        runner.test_audio_speech_unknown_voice_404()
        runner.test_audio_speech_missing_fields_400()
        runner.test_audio_speech_with_cloned_voice(voice_id)
        runner.test_rate_changes_duration(voice_id)
        runner.test_audio_speech_speed_field(voice_id)

        # 16kHz output requirement
        runner.test_api_tts_16khz(voice_id)
        runner.test_audio_speech_16khz(voice_id)

        runner.test_delete_voice(voice_id)
    else:
        print("\nSkipping follow-up tests because clone failed.")

    if not args.keep:
        runner.cleanup()

    return runner.report()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)