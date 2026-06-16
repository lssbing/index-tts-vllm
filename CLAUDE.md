# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

IndexTTS-vLLM 是 [index-tts](https://github.com/index-tts/index-tts) 的 vLLM 加速版本，将 GPT 模型的推理通过 vLLM 库重写。核心目标是将 GPT decode 速度从 ≈90 token/s 提升到 ≈280 token/s（RTF 从 ≈0.3 降到 ≈0.1）。

支持三个版本：
- **Index-TTS v1.0**（`webui.py` / `api_server.py` + `indextts/infer_vllm.py`）
- **Index-TTS v1.5**（`webui.py --version 1.5`）
- **IndexTTS-2**（`webui_v2.py` / `api_server_v2.py` + `indextts/infer_vllm_v2.py`，支持情感控制）

## 常用命令

### 环境配置
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
# pytorch 2.8.0（对应 vllm 0.10.2）
pip install -r requirements.txt
```

### 模型权重下载
```bash
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

权重目录里需要包含 `config.yaml`、`gpt.pth`、`bpe.model`、`bigvgan_generator.pth`（v1/v1.5），以及 IndexTTS-2 需要的 `s2mel.pth`、`wav2vec2bert_stats.pt` 等。vLLM 用的 GPT 模型需要先用 `convert_hf_format.sh` 转换到 `gpt/` 子目录下。

### 运行

WebUI（首次启动会编译 bigvgan 的 CUDA kernel，会比较慢）：
```bash
python webui.py                    # Index-TTS v1.0
python webui.py --version 1.5      # Index-TTS v1.5
python webui_v2.py                 # IndexTTS-2
```

API 服务：
```bash
python api_server.py    --model_dir ./checkpoints/Index-TTS-1.5-vLLM --gpu_memory_utilization 0.25
python api_server_v2.py --model_dir ./checkpoints/IndexTTS-2-vLLM --gpu_memory_utilization 0.25 --qwenemo_gpu_memory_utilization 0.10
```

CLI 单次合成：
```bash
python -m indextts.cli "你好世界" -v assets/jay_promptvn.wav -o out.wav --model_dir ./checkpoints/Index-TTS-vLLM
```

Docker：
```bash
docker compose up --build
```

### 性能基准测试
```bash
# GPT 模型推理基准（多并发对比，输出 token/s）
python test/gpt_vllm.py

# TTS 服务压测（默认 16 并发 × 5 请求）
python test/simple_test.py --urls http://localhost:6006/tts_url --concurrency 16 --requests 5
```

## 架构与代码组织

### 推理流水线
```
text → TextNormalizer/TextTokenizer (BPE)
     → vLLM GPT model (batched async)            # 主要瓶颈，已加速
     → s2mel (v2 only，DiT 25 步迭代，未加速)     # v2 主要瓶颈
     → BigVGAN vocoder
     → wav @ 24kHz (v1/1.5) 或 22050Hz (v2)
```

### 关键文件

**入口**：
- [webui.py](webui.py) / [webui_v2.py](webui_v2.py) — Gradio UI
- [api_server.py](api_server.py) / [api_server_v2.py](api_server_v2.py) — FastAPI 服务
- [indextts/cli.py](indextts/cli.py) — 命令行接口

**vLLM 集成核心**：
- [indextts/infer_vllm.py](indextts/infer_vllm.py) — v1/v1.5 推理封装类 `IndexTTS`
- [indextts/infer_vllm_v2.py](indextts/infer_vllm_v2.py) — v2 推理封装类 `IndexTTS2`（含 QwenEmotion）
- [indextts/gpt/model_vllm.py](indextts/gpt/model_vllm.py) — `UnifiedVoice`（v1/v1.5），保留 conformer + perceiver 等非 GPT 部分，GPT 部分委托给 vLLM
- [indextts/gpt/model_vllm_v2.py](indextts/gpt/model_vllm_v2.py) — v2 的 `UnifiedVoice`，增加情感条件融合
- [indextts/gpt/index_tts_gpt2_vllm_v1.py](indextts/gpt/index_tts_gpt2_vllm_v1.py) — 注册到 vLLM 的 `GPT2TTSModel`，使用 multi-modal 接口传入 audio embeds（`PLACEHOLDER_TOKEN`/`PLACEHOLDER_TOKEN_ID`）
- [patch_vllm.py](patch_vllm.py) — **必须导入**的 monkey patch：在 `GPUModelRunner._prepare_inputs` 中为 GPT2TTSModel 调整 position_ids（减去 prefill 长度再加 1），并注册 `GPT2TTSModel` 到 `ModelRegistry`。每个 `model_vllm*.py` 模块顶部都有 `import patch_vllm`，不要删除。

**模型组件**：
- `indextts/BigVGAN/` — vocoder（`bigvgan.py`、CUDA kernel 在 `alias_free_activation/`）
- `indextts/s2mel/` — v2 声学模型（含 DiT/CFM 扩散）
- `indextts/utils/front.py` — `TextNormalizer`、`TextTokenizer`（BPE）
- `indextts/utils/maskgct_utils.py` — v2 用 w2v-BERT 提取 semantic feature

### vLLM 工作机制要点
1. 原 GPT 通过 `convert_hf_format.sh` 转换为 HF 格式存到 `<model_dir>/gpt/`，启动时 vLLM 加载这个目录
2. 推理时，`UnifiedVoice` 把 conditioning mel（通过 conformer + perceiver 编码）和文本 token 拼成 `inputs_embeds`，作为 multi-modal `audio_embeds` 传给 vLLM
3. 推理 batch 化在 vLLM `AsyncLLM` 内部完成（`gpu_memory_utilization` 0.25 即可支持 ~16 并发，参考 `test/gpt_vllm.md` 的基准）

### API 端点
v1/v1.5 (`api_server.py`)：
- `POST /tts_url` — 用文件路径列表作为参考音频
- `POST /tts` — 用 `assets/speaker.json` 注册的 speaker 名
- `POST /audio/speech` — OpenAI 兼容
- `GET /audio/voices` — 列出 speaker
- `GET /health`

v2 (`api_server_v2.py`)：
- `POST /tts_url` — 完整参数，支持 `emo_control_method` 0/1/2/3（相同 / 情感参考音频 / 情感向量 / 情感文本）

完整参数示例：[api_example.py](api_example.py)、[api_example_v2.py](api_example_v2.py)

## 注意事项

- **不要删除 `import patch_vllm`**：v1 和 v2 的 `model_vllm*.py` 都依赖这个 monkey patch 来正确计算 position embedding，否则推理结果错误
- **首次启动慢**：BigVGAN 的 CUDA fused activation kernel 要现场编译
- **v2 并发瓶颈在 s2mel**（DiT 扩散），不是 GPT —— 进一步加速需要优化 s2mel（项目 TODO）
- **多参考音频（v1/v1.5）**：传入多个 prompt 音频会混合声线但不稳定，需要"抽卡"
- **中文 WebUI 文本含特殊语气词**：`嗯`(EN4)、`嘿`(HEI1)、`嗨`(HAI4)、`哈哈`(HA1HA1) 会被自动替换为对应韵律标记
- 大模型权重（`checkpoints/*`）已在 `.gitignore` 中，不会入库