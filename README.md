# Qwen3-TTS + Megakernel + Pipecat

Real-time voice agent: mic → STT → LLM → Qwen3-TTS → speaker, targeting RTX 5090 (Blackwell).

```
Mic → [Deepgram STT] → [gpt-5-mini LLM] → [Custom TTS decode loop] → [Vocoder] → Speaker
```

---

## Performance Numbers

Measured on RTX 5090 (Blackwell, sm_120a), CUDA 12.8, bfloat16, no flash-attn.
All numbers from `scripts/test_v2_decode.py` after CUDA graph warmup.

### Current best (custom decode loop + CUDA graphs, `tts_backend_v2`)

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| RTF | **0.237** | < 0.15 | 1.58× |
| TTFC | **93 ms** | < 60 ms | 1.55× |
| Codec frames/s | **76.6** | — | — |
| ms/frame (decode only) | **13.1 ms** | < 12 ms | close |

### Progress across sessions

| Session | Approach | RTF | TTFC | Notes |
|---------|----------|-----|------|-------|
| 1-4 | HF baseline | 1.070 | 6338 ms | Vanilla HF, no streaming |
| 5-6 | Megakernel monkey-patch | ~1.07 | ~6338 ms | Megakernel ran but EOS never fired; fell back to HF every call |
| Phase 2 (v2) eager | Custom decode loop | 0.835 | 842 ms | EOS fires, real streaming, no CUDA graphs |
| Phase 2 + CUDA graphs | StaticCache + graph capture | **0.237** | **93 ms** | Current state |

### HuggingFace baseline (no custom loop)

| Text | TTFC | RTF |
|------|------|-----|
| "Hello." | 4762 ± 2538 ms | 1.126 ± 0.015 |
| "The quick brown fox..." | 4641 ± 509 ms | 1.119 ± 0.009 |
| "Artificial intelligence is transforming..." | 6704 ± 496 ms | 1.006 ± 0.007 |
| "In the beginning there was darkness..." | 9243 ± 618 ms | 1.031 ± 0.023 |
| **Mean** | **6338 ms** | **1.070** |

---

## Architecture

### Pipeline

```
mic → Deepgram STT → gpt-5-mini LLM → QwenTTSService → QwenTTSBackendV2 → speaker
                                              ↓
                               Custom decode loop (tts_backend_v2.py)
                               ├── HF prefill (DynamicCache, variable length)
                               ├── PredictorGraph CUDA graph (15-step predictor loop)
                               ├── TalkerGraph CUDA graph (28-layer decode step)
                               └── Incremental vocoder (25-frame context window)
```

### Custom decode loop (Phase 2 architecture)

The core insight from inspecting `modeling_qwen3_tts.py` and `andimarafioti/faster-qwen3-tts`: Qwen3-TTS generation is **not** simple autoregressive token sampling. Each decode step requires:

1. **Prefill** — HF talker processes text+speaker embeddings → `DynamicCache` + `past_hidden` + first logits
2. **Per-step loop:**
   - Sample CB0 token from logits
   - `pred_input = cat(past_hidden, codec_embedding(CB0))`  `[1, 2, 1024]`
   - Run code predictor 15 steps → CB1..CB15  → full codec frame `[16]`
   - Build next-step embedding: sum of 16 per-codebook embeddings + text conditioning
   - Run talker backbone → new `hidden`, `past_hidden`, logits
3. **Incremental vocoder** — every 4 codec frames, decode with 25-frame left-context window → yield audio

Previous approach (monkey-patching `talker.generate()`) failed because it only tracked integer token IDs and ignored the code predictor entirely — causing EOS never to fire and sequence divergence from step 1.

### CUDA graphs (key speedup)

Both hot paths captured as CUDA graphs using `transformers.StaticCache`:

| Component | Eager | CUDA graph | Speedup |
|-----------|-------|------------|---------|
| Code predictor (15 steps) | ~49 ms | ~2-3 ms | ~20× |
| Talker backbone (1 step, 28L) | ~20 ms | ~2-5 ms | ~5× |

**Why StaticCache:** `DynamicCache` grows via `torch.cat` every step → dynamo recompiles on each new shape. `StaticCache` pre-allocates `[batch, kv_heads, max_seq, head_dim]` and writes via `index_copy_()` → shape never changes → CUDA graph capture works.

**Prefill flow:** HF eager prefill (variable prompt length) → `prefill_kv()` copies `DynamicCache` → `StaticCache` → CUDA graph decode loop with static shapes.

### Streaming

Audio chunks arrive every 4 codec frames = 320ms of audio. The vocoder runs inside the decode thread with a 25-frame causal context window. First chunk TTFC = prefill + 4 decode steps + one vocoder call ≈ 21ms + 4×13ms + 5ms = **~78ms**.

---

## Requirements

- GPU: RTX 5090 (Blackwell, sm_120a), CUDA 12.8+, driver 570+
- GPU RAM: ~8 GB (Qwen3-TTS 0.6B in bfloat16)
- Recommended: Vast.ai RTX 5090 instance
- API keys: OpenAI + Deepgram

---

## GPU Server Setup

### 1. Clone

```bash
git clone <your-repo-url> /workspace/qwen-megakernel-pipecat
cd /workspace/qwen-megakernel-pipecat
```

### 2. One-shot setup

```bash
bash scripts/setup_server.sh
```

Installs deps, creates `.venv`, installs PyTorch cu128, clones and builds the megakernel extension. Takes ~10 minutes first run.

### 3. Set env vars

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, DEEPGRAM_API_KEY
```

### 4. Start the server

```bash
source .venv/bin/activate
set -a && source .env && set +a

# v2 backend (custom decode loop + CUDA graphs) — recommended
TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000

# HF baseline (no custom loop)
TTS_BACKEND=hf uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
```

Server exposes:
- `GET /` — health check
- `WS /ws` — Pipecat WebSocket endpoint

### 5. Validate before running the pipeline

```bash
python scripts/test_v2_decode.py
```

Runs 5 staged tests: prefill capture → single decode step → full decode to EOS → vocoder output → streaming TTFC. Takes ~2 minutes including CUDA graph warmup.

---

## Frontend

```bash
cd client && npm install
# SSH tunnel from local machine:
ssh -p <port> root@<ip> -L 8000:localhost:8000
VITE_WS_URL=ws://localhost:8000/ws npm run dev
```

Open `http://localhost:5173`, click CONNECT, speak.

---

## Benchmarking

```bash
source .venv/bin/activate
# Full validation + timing breakdown
python scripts/test_v2_decode.py

# HF baseline RTF/TTFC
python scripts/benchmark.py --backend hf --trials 5
```

---

## Project Structure

```
server/
  pipeline/voice_agent.py               FastAPI + Pipecat pipeline
                                        TTS_BACKEND=v2|hf|megakernel
  backend/tts_backend_v2.py             Custom decode loop (current)
  backend/cuda_graphs.py                TalkerGraph + PredictorGraph CUDA graph capture
  backend/tts_backend_mk.py             Megakernel backend (Phase 1, deprecated)
  backend/tts_backend_hf.py             HF baseline
  pipecat_services/qwen_tts_service.py  Pipecat TTSService adapter

scripts/
  test_v2_decode.py                     5-stage validation: prefill → EOS → vocoder → TTFC
  setup_server.sh                       One-shot GPU server setup
  benchmark.py                          TTFC / RTF / tok/s measurement
  test_mk_decode.py                     Megakernel smoke test (Phase 1)

client/
  src/components/Dashboard.tsx          Voice UI with live metrics
  src/lib/pipecatClient.ts              WebSocket transport config

docs/
  custom_decode_architecture.md         Full decode loop architecture with tensor shapes
  findings.md                           Ground-truth model inspection results
  progress.md                           Session-by-session progress log
```

---

## Known Limitations

### RTF and TTFC not yet at target

Current: RTF 0.237, TTFC 93ms. Targets: RTF < 0.15, TTFC < 60ms.

The remaining gap (1.55×) is split between:
- Python loop overhead per decode step (~5-8ms/frame)
- Vocoder running synchronously in the decode thread (~6ms/frame amortized)
- Audio queue latency for TTFC

### Megakernel not yet integrated into v2

The `qwen_megakernel` CUDA kernel (`torch.ops.qwen_megakernel_C.decode`) runs at 263 tok/s on RTX 5090 and was verified working in isolation. Phase 3 plan: replace `TalkerGraph`'s decode step with the megakernel, using the sentinel `token_id=-1` trick (3-line patch to `kernel.cu`) to accept pre-computed embeddings instead of integer token IDs. Expected talker latency: ~1ms vs current ~2-5ms.

### Audio quality not yet validated end-to-end

The v2 pipeline produces audio (confirmed vocoder output, EOS fires correctly, codec frames are valid). Audio quality against the HF baseline has not been formally compared. The Phase 1 megakernel backend produced no audio (EOS never fired).

### GPU target

Megakernel targets `sm_120a` (RTX 5090 Blackwell) only. The custom decode loop (`tts_backend_v2`) works on any GPU that supports the `qwen_tts` model.

---

## What Remains for Full Target

1. **Phase 3 — Megakernel in v2 loop** (~1ms/talker step):
   - 3-line patch to `kernel.cu`: `if (token_id < 0) use hidden_buffer as embedding`
   - Replace `TalkerGraph.run()` with `mk_decoder.step(-1, embed=inputs_embeds)`
   - Expected: talker 2-5ms → ~1ms, bringing total below 12ms/frame

2. **Python loop vectorization** (~5ms/frame Python overhead):
   - Move sampling, embedding lookup, tensor cat out of per-step Python loop
   - Batch 16-codebook embedding sum as a single batched matmul

3. **Async vocoder** — run vocoder in a separate thread, don't block decode loop

4. **flash-attn** — `pip install flash-attn` for 3-4× prefill speedup (TTFC reduction)

5. **Demo recording** — screen capture of full voice round-trip
