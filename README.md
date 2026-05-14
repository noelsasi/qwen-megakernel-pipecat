# Qwen3-TTS + Megakernel + Pipecat

Real-time voice agent: mic → STT → LLM → Qwen3-TTS → speaker.

```
Mic → [Deepgram STT] → [OpenAI LLM] → [Qwen3-TTS talker] → [vocoder] → Speaker
```

---

## Performance Numbers

Measured on RTX 5090 (Blackwell, sm_120a), CUDA 12.8, bfloat16, no flash-attn.
Methodology: 3 trials per sentence, mean ± std, `torch.cuda.synchronize()` before timer stops.

### HuggingFace baseline (no megakernel)

| Text | TTFC | RTF |
|------|------|-----|
| "Hello." | 4762 ± 2538 ms | 1.126 ± 0.015 |
| "The quick brown fox..." | 4641 ± 509 ms | 1.119 ± 0.009 |
| "Artificial intelligence is transforming..." | 6704 ± 496 ms | 1.006 ± 0.007 |
| "In the beginning there was darkness..." | 9243 ± 618 ms | 1.031 ± 0.023 |
| **Mean** | **6338 ms** | **1.070** |

### Megakernel decode throughput

| Metric | Value |
|--------|-------|
| Decode speed | **263–266 tok/s** |
| Tokens per call | 4096 (hard cap, EOS not reached — see Known Limitations) |
| Decode time (4096 tokens) | ~15.5 s |

### Targets vs actuals

| Metric | Target | HF Baseline | Megakernel Status |
|--------|--------|-------------|-------------------|
| RTF | < 0.15 | 1.07 | Kernel runs at 263 tok/s; full pipeline RTF not measurable (see Known Limitations) |
| TTFC | < 60 ms | 6338 ms | Not achieved — full audio buffered before streaming |
| tok/s | ~1000 | — | **263 tok/s** confirmed on RTX 5090 |

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

```bash
cd client
npm install
# SSH tunnel from local machine:
# ssh -p <port> root@<ip> -L 8000:localhost:8000
VITE_WS_URL=ws://localhost:8000/ws npm run dev
```

Open http://localhost:5173, click CONNECT, speak.

---

## Benchmarking

```bash
source .venv/bin/activate
python scripts/benchmark.py --backend hf --trials 5
```

---

## Architecture

### Pipeline

```
mic → Deepgram STT → gpt-5-mini LLM → QwenTTSService → Qwen3-TTS → speaker
                                              ↓
                                    QwenTTSBackendMK
                                    (megakernel decode + HF vocoder)
```

### Megakernel integration

The megakernel (`torch.ops.qwen_megakernel_C.decode`) runs the talker autoregressive decode loop — one token per call. Integration approach:

1. **Patch `talker.generate()`** at call time
2. Run HF prefill via one `talker.model.forward()` call → captures `DynamicCache`
3. Transfer KV cache from HF's `DynamicCache.layers[i].keys/values` into megakernel's `[28, 8, MAX_SEQ_LEN, 128]` bfloat16 buffers
4. Run tight Python decode loop: `decoder.step(token) → next_token` until EOS or max steps
5. Return generated tokens + per-step hidden states to HF code_predictor + vocoder

### Kernel modifications

| Item | Default | Required | Notes |
|------|---------|----------|-------|
| `LDG_VOCAB_SIZE` | 151936 | **3072** | Must patch `csrc/kernel.cu` and rebuild |
| `MAX_SEQ_LEN` | 2048 | 32768 | Python constant only — no rebuild |
| `rope_theta` | 10000 | 1,000,000 | Python only |
| RoPE type | (original) | Standard 1D | Talker uses standard RoPE, not MRope |

### Buffer requirements (confirmed from kernel source)

The kernel's `launch_ldg_decode_direct` casts all scratch buffers:

| Buffer | dtype | size |
|--------|-------|------|
| `hidden_buffer` | bfloat16 | HIDDEN_SIZE = 1024 |
| `g_activations`, `g_residual`, `g_normalized` | float32 | HIDDEN_SIZE = 1024 |
| `g_q`, `g_attn_out` | float32 | NUM_Q_HEADS × HEAD_DIM = 2048 |
| `g_k`, `g_v` | float32 | NUM_KV_HEADS × HEAD_DIM = 1024 |
| `g_mlp_intermediate` | float32 | VOCAB_SIZE = 3072 |
| `block_max_vals`, `block_max_idxs` | float32/int32 | **LDG_LM_NUM_BLOCKS = 1184** |
| `k_cache`, `v_cache` | bfloat16 | [28, 8, MAX_SEQ_LEN, 128] |

---

## Project Structure

```
server/
  pipeline/voice_agent.py               FastAPI + Pipecat pipeline
  backend/tts_backend_mk.py             Megakernel TTS backend
  backend/tts_backend_hf.py             HF baseline backend
  pipecat_services/qwen_tts_service.py  Pipecat TTSService adapter

scripts/
  setup_server.sh                       One-shot GPU server setup
  benchmark.py                          TTFC / RTF / tok/s measurement
  test_mk_decode.py                     Megakernel smoke test

client/
  src/components/Dashboard.tsx          Voice UI with live metrics
  src/lib/pipecatClient.ts              WebSocket transport config
```

---

## Known Limitations

### Streaming
`synthesize_streaming()` buffers the full audio then yields 100ms chunks. True token-by-token streaming requires hooking into the vocoder's per-frame decode, which is not exposed by `Qwen3TTSModel`. TTFC reflects time-to-first-chunk after full generation.

### Megakernel EOS
The megakernel decode loop generates valid codec tokens (263 tok/s confirmed) but does not naturally reach EOS token 2150. The megakernel's decode sequence diverges from HF's expected sequence because the megakernel tracks its own token state while HF constructs mixed embeddings (text + codec + speaker) for each step. The two systems share the prefill KV cache but diverge at decode step 1. Audio from the megakernel path has not been validated for quality.

### Vocoder integration
`generate_custom_voice()` uses per-step hidden states from the talker to feed the code predictor (line 2280-2281 of `modeling_qwen3_tts.py`). The megakernel's `decoder._hidden` buffer contains the final layer output after each step, but packaging it in HF's expected `hidden_states` tuple format (tuple of layer-tuples per step) needs more work to complete correctly.

### GPU target
Megakernel targets `sm_120a` (RTX 5090 Blackwell) only. Will not compile on other GPUs.

---

## What I Would Do With More Time

1. **Fix EOS generation** — the megakernel's decode diverges from HF's because HF blends text/speaker embeddings at each step. The fix is to replicate HF's embedding construction inside the decode loop, or feed the correct mixed embedding as the hidden state seed at each step.

2. **Fix vocoder integration** — package `decoder._hidden` per step into the correct `hidden_states` format for the code predictor.

3. **True streaming** — intercept the vocoder's per-frame decode to yield audio chunks as codec tokens arrive, reducing TTFC from full-generation time to first-frame time (~80ms).

4. **flash-attn** — `pip install flash-attn` would reduce HF baseline RTF significantly (expected ~3-4× speedup on prefill).

5. **Megakernel tok/s** — at 263 tok/s vs the paper's ~1000 tok/s target, there's a 4× gap. Likely caused by MAX_SEQ_LEN=32768 making KV cache strides very large. Reducing to 2048 (sufficient for TTS) would recover most of that.
