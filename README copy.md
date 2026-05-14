# Qwen3-TTS + Megakernel + Pipecat

Real-time voice agent: mic → STT → LLM → Qwen3-TTS (CUDA megakernel decode) → speaker.

```
Mic → [Deepgram STT] → [OpenAI LLM] → [Qwen3-TTS talker ← megakernel] → [vocoder] → Speaker
```

## Performance Targets

| Metric | Target | Baseline (HF, no megakernel) |
|--------|--------|------------------------------|
| RTF | < 0.15 | 0.879 |
| TTFC | < 60 ms | — |

---

## Requirements

- GPU: RTX 5090 (Blackwell, sm_120a), CUDA 12.8+, driver 570+
- GPU RAM: ~8 GB (Qwen3-TTS 0.6B in bfloat16)
- Recommended: Vast.ai RTX 5090 instance
- API keys: OpenAI + Deepgram

---

## GPU Server Setup

### 1. SSH into the Vast.ai instance and clone

```bash
git clone <your-repo-url> /workspace/qwen-megakernel-pipecat
cd /workspace/qwen-megakernel-pipecat
```

### 2. Run one-shot setup

```bash
bash scripts/setup_server.sh
```

This does:
- installs system deps (libsndfile1, ffmpeg)
- creates `.venv`
- installs PyTorch with CUDA 12.8
- installs all Python packages
- clones `qwen_megakernel` repo
- patches `kernel.cu` (LDG_VOCAB_SIZE: 151936 → 3072)
- builds the megakernel extension
- verifies the ops load

Takes ~10 minutes first run (PyTorch download is ~820MB).

### 3. Set env vars

```bash
cp .env.example .env
nano .env   # fill OPENAI_API_KEY and DEEPGRAM_API_KEY
```

### 4. Start the server

```bash
source .venv/bin/activate
set -a && source .env && set +a
uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
```

Server exposes:
- `GET /` — health check
- `WS /ws` — Pipecat WebSocket endpoint

---

## Frontend

On your local machine (or any machine with Node 18+):

```bash
cd client
npm install
# Point at the GPU server:
VITE_WS_URL=ws://<vast-ai-ip>:8000/ws npm run dev
```

Open http://localhost:5173, click CONNECT, speak.

---

## Benchmarking

```bash
source .venv/bin/activate
python scripts/benchmark.py --backend megakernel --trials 5
```

Outputs mean ± std for TTFC, RTF, tok/s across 4 test sentences.

---

## Architecture

### Megakernel integration strategy

HuggingFace's `Qwen3TTSModel.generate_custom_voice()` internally:
1. Runs prefill (text → KV cache, via HF transformer forward pass)
2. Calls `talker.model.generate()` for the autoregressive decode loop
3. Passes codec tokens to code predictor + vocoder → audio

We patch `talker.model.generate()` at call time:
1. Let HF run prefill normally → get `past_key_values` (DynamicCache)
2. Intercept `generate()`: copy KV cache into megakernel tensors
3. Run decode loop via `torch.ops.qwen_megakernel_C.decode()` (one call per token)
4. Return token tensor to HF in the same format HF expects
5. HF runs code predictor + vocoder unchanged → audio

This means:
- prefill: HF (no change needed)
- decode: megakernel (full speedup)
- vocoder: HF (no change needed, no API needed)

### Weight remapping

The megakernel expects weights in `LDGLayerWeights` format (11 pointers per layer).
`_extract_talker_weights()` in `tts_backend_mk.py` strips the `talker.` prefix and
remaps:
- `talker.model.codec_embedding.weight` → `embed_weight` [3072, 1024]
- `talker.codec_head.weight` → `lm_head_weight` [3072, 1024]
- 28 × 11 layer weights as a packed pointer buffer

### MRope tables

Qwen3-TTS uses interleaved MRope with `mrope_section=[24,20,20]` and `rope_theta=1e6`.
The kernel's standard RoPE table is replaced by `_build_mrope_tables()` which
builds the correct [MAX_SEQ_LEN, HEAD_DIM] cos/sin tables.

### Kernel patches required

| Constant | Default | Required | Where |
|----------|---------|----------|-------|
| `LDG_VOCAB_SIZE` | 151936 | 3072 | `csrc/kernel.cu` — requires rebuild |
| `MAX_SEQ_LEN` | 2048 | 32768 | Python constant only |
| `rope_theta` | 10000 | 1,000,000 | Python only |
| RoPE type | standard | interleaved MRope | Python only |

---

## Project Structure

```
server/
  pipeline/voice_agent.py               FastAPI + Pipecat pipeline
  backend/tts_backend_mk.py             Megakernel backend — weights, decode loop
  pipecat_services/qwen_tts_service.py  Pipecat TTSService adapter

scripts/
  setup_server.sh                       One-shot GPU server setup
  benchmark.py                          TTFC / RTF / tok/s measurement

client/
  src/components/Dashboard.tsx          Voice UI with live metrics
  src/lib/pipecatClient.ts              WebSocket transport config
```

---

## Known Limitations

- `synthesize_streaming()` yields audio in ~100ms chunks after full generation completes.
  True token-by-token streaming requires access to the vocoder's per-frame decode API,
  which is not exposed by `Qwen3TTSModel`. TTFC reflects time-to-first-chunk after full
  generation, not time-to-first-token.

- Megakernel targets `sm_120a` (RTX 5090 Blackwell) only. Will not compile on other GPUs.

- If the megakernel build fails, verify `LDG_VOCAB_SIZE` was patched before building
  and that CUDA 12.8 toolkit is installed (`nvcc --version`).
