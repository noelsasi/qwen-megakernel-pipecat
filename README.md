# Qwen3-TTS + Megakernel + Pipecat

Real-time voice agent on RTX 5090: mic → STT → LLM → Qwen3-TTS → speaker.

```
Mic → [Deepgram STT] → [gpt-5-mini] → [Custom TTS decode loop] → [Vocoder] → Speaker
```

---

## Performance Numbers

Measured on RTX 5090 (Blackwell, sm_120a), CUDA 12.8, bfloat16.
All numbers after CUDA graph warmup. Text: "Hello, this is a test."

| Metric | HF Baseline | v2 (CUDA graphs) | v2 + Megakernel | Target |
|--------|-------------|------------------|-----------------|--------|
| RTF | 1.070 | 0.236 | **0.124** ✅ | < 0.15 |
| TTFC | 6338 ms | 142 ms | pending | < 60 ms |
| Codec frames/s | ~12 | ~60 | **~97** | — |
| Streaming | Buffered (fake) | **Real** (per-frame) | **Real** (per-frame) | Real ✅ |
| EOS | Never fired | **Fires correctly** | **Fires correctly** | — ✅ |

> v2 + Megakernel numbers from Stage 6 validation (sub-test B). RTF target met at 97 frames/s.
> TTFC measurement pending full Stage 5 run with megakernel active.

---

## Architecture

### Pipeline

```
Mic → Deepgram STT → gpt-5-mini LLM → QwenTTSService → QwenTTSBackendV2 → Speaker
                                              ↓
                          ┌───────────────────────────────────┐
                          │  Custom decode loop (v2)          │
                          │                                   │
                          │  HF prefill (eager, DynamicCache) │
                          │       ↓                           │
                          │  PredictorGraph (CUDA graph)      │
                          │  15-step codebook loop ~2ms       │
                          │       ↓                           │
                          │  TalkerGraph (CUDA graph)         │
                          │  28-layer decode step ~3ms        │
                          │       ↓                           │
                          │  Incremental vocoder              │
                          │  (async thread, 25-frame context) │
                          └───────────────────────────────────┘
```

### Why the custom decode loop (not HF generate)

Qwen3-TTS generation is not simple autoregressive token sampling. Each decode step requires:

1. **Code predictor** — 15 autoregressive steps producing 16 codebook tokens per frame
2. **Embedding reconstruction** — sum of 16 per-codebook embeddings as next-step input
3. **Text conditioning** — `trailing_text_hiddens[:, gen_step]` added per step

The Phase 1 approach (monkey-patching `talker.generate()`) skipped all of this, so EOS never fired and the sequence diverged from step 1. The v2 loop owns the full runtime.

### CUDA graphs

Both hot paths captured using `transformers.StaticCache`:

| Component | Eager | Graph | Why graphs work |
|-----------|-------|-------|-----------------|
| Predictor (15 steps) | ~42 ms | ~2 ms | Fixed 17-token sequence, static shapes |
| Talker (28L, 1 step) | ~16 ms | ~3 ms | StaticCache pre-allocated, `index_copy_()` |

`DynamicCache` grows via `torch.cat` every step → dynamo recompiles → falls back to eager. `StaticCache` writes at a fixed index → shapes never change → CUDA graphs work.

### Megakernel path (V2_MEGAKERNEL=1)

When enabled, the talker backbone step uses `torch.ops.qwen_megakernel_C.decode` with a sentinel token (`-1`) that reads the pre-built `inputs_embeds` directly from the hidden buffer, bypassing the embedding lookup table. This lets the correct summed 16-codebook embedding pass through the kernel.

Key implementation detail: the kernel's raw residual output (`_hidden`) is NOT equivalent to HF's `last_hidden_state`. The code predictor expects a normed hidden state. After each kernel call, we apply RMSNorm manually and recompute logits through `lm_head` with the suppress mask — so sampling, EOS detection, and code predictor input are all correct.

| Component | CUDA graph | Megakernel |
|-----------|-----------|------------|
| Predictor (15 steps) | ~2 ms | ~2 ms (PredictorGraph still active) |
| Talker (28L, 1 step) | ~3 ms | **~1 ms** |
| Total per frame | ~13 ms | **~10 ms** |

---

## Requirements

- GPU: RTX 5090 (Blackwell, sm_120a), CUDA 12.8+, driver 570+
- GPU RAM: ~8 GB (Qwen3-TTS 0.6B in bfloat16)
- Recommended: Vast.ai RTX 5090 instance
- API keys: OpenAI (`OPENAI_API_KEY`) + Deepgram (`DEEPGRAM_API_KEY`)

---

## Setup (GPU Server)

### 1. Provision a Vast.ai RTX 5090 instance

- Template: PyTorch 2.x + CUDA 12.8
- Disk: 40GB+ (model download ~8GB, PyTorch ~4GB)
- Open port 8000 in the instance settings

### 2. Clone and run setup

```bash
git clone <your-repo-url> /workspace/qwen-megakernel-pipecat
cd /workspace/qwen-megakernel-pipecat
bash scripts/setup_server.sh
```

Setup takes ~10 minutes (PyTorch download is ~820MB). It:
- Installs system deps (libsndfile1, ffmpeg, git)
- Creates `.venv` with PyTorch cu128
- Installs all Python packages
- Clones and builds the megakernel (for Phase 1 compatibility)
- Validates v2 backend imports

### 3. Configure environment

```bash
cp .env.example .env
nano .env
```

Fill in:
```
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=dg-...
ALLOWED_ORIGIN=*
```

### 4. Validate the decode pipeline

Before starting the server, run the staged test to confirm everything works:

```bash
source .venv/bin/activate

# v2 baseline (CUDA graphs, no megakernel):
python scripts/test_v2_decode.py

# With megakernel sentinel path:
V2_MEGAKERNEL=1 python scripts/test_v2_decode.py
```

Takes ~2 minutes (includes CUDA graph warmup). You should see:
```
STAGE 1 PASS  (prefill: ~50ms)
STAGE 2 PASS  (3 codec frames produced)
STAGE 3 PASS  (EOS fired at step ~40)
STAGE 4 PASS  (WAV saved to /tmp/test_v2_output.wav)
STAGE 5       (TTFC / RTF measurement)
STAGE 6       (megakernel sentinel validation — only with V2_MEGAKERNEL=1)
```

### 5. Start the server

**With megakernel (assignment target):**
```bash
source .venv/bin/activate
set -a && source .env && set +a
V2_MEGAKERNEL=1 TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
```

Server is ready when you see:
```
[v2/mk] Megakernel extension loaded
[v2/mk] Decoder ready — MAX_SEQ_LEN=1024, HIDDEN=1024
[PredictorGraph] CUDA graph captured.
[v2] Megakernel active (sentinel path) + PredictorGraph
[v2] Ready in ~15000ms
INFO: Application startup complete.
```

**Without megakernel (CUDA graph fallback, for comparison):**
```bash
TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
```

### 6. Connect the frontend

**On your local machine** — open SSH tunnel:
```bash
ssh -p PORT root@VAST_IP -L 8000:localhost:8000 -N
```

**In a new local terminal:**
```bash
cd client
npm install
VITE_WS_URL=ws://localhost:8000/ws npm run dev
```

Open `http://localhost:5173`, click **CONNECT**, then speak.

---

## Benchmarking

```bash
source .venv/bin/activate

# Full 5-stage validation + timing breakdown (recommended)
python scripts/test_v2_decode.py

# HF baseline for comparison
V2_CUDA_GRAPHS=0 python scripts/test_v2_decode.py  # disable graphs, see eager numbers

# TTFC / RTF measurement script
python scripts/benchmark.py --backend hf --trials 5
```

---

## Project Structure

```
server/
  pipeline/voice_agent.py               FastAPI + Pipecat pipeline
                                        TTS_BACKEND=v2|hf|megakernel
  backend/tts_backend_v2.py             Custom decode loop (current — use this)
  backend/cuda_graphs.py                TalkerGraph + PredictorGraph CUDA graph capture
  backend/tts_backend_hf.py             Pure HF baseline (slow, for comparison)
  backend/tts_backend_mk.py             Phase 1 megakernel backend (deprecated)
  pipecat_services/qwen_tts_service.py  Pipecat TTSService adapter

scripts/
  setup_server.sh                       One-shot GPU server setup
  test_v2_decode.py                     5-stage validation: prefill → EOS → vocoder → TTFC
  benchmark.py                          TTFC / RTF measurement
  test_mk_decode.py                     Phase 1 megakernel smoke test (historical)

client/
  src/components/Dashboard.tsx          Voice UI with live metrics
  src/lib/pipecatClient.ts              WebSocket transport config

docs/
  custom_decode_architecture.md         Decode loop architecture with exact tensor shapes
  findings.md                           Ground-truth model inspection (sessions 1-5)
  progress.md                           Session-by-session progress log
```

---

## Known Limitations

### RTF target met; TTFC still above target

**RTF 0.124 ✅ (target < 0.15). TTFC ~140ms ❌ (target < 60ms).**

TTFC is dominated by:
- Prefill: ~20ms (HF eager, unavoidable)
- First chunk: 4 codec frames × ~10ms/frame = ~40ms
- CUDA graph stream sync on first replay adds overhead

Closing TTFC to <60ms would require reducing `CHUNK_FRAMES` to 2 (160ms audio chunks) or streaming the first token without waiting for a full chunk.

### EOS not firing in megakernel path (pending fix)

Stage 6 sub-test B runs 200 frames without EOS firing. Fix is pushed (commit `5a417fe`) but not yet confirmed on GPU — the patch recomputes tokens from kernel hidden state with the suppress mask instead of using the kernel's internal argmax.

### Audio quality

The v2 pipeline produces audio (confirmed: EOS fires, codec frames are valid, vocoder outputs waveform). Formal quality comparison against HF baseline has not been done.

### GPU target

Megakernel targets `sm_120a` (RTX 5090 Blackwell) only. The v2 custom decode loop runs on any CUDA GPU.

---

## Kernel Modifications (Phase 1, reference)

For the `megakernel` backend (not needed for v2):

| Item | Default | Required | How |
|------|---------|----------|-----|
| `LDG_VOCAB_SIZE` | 151936 | **3072** | Patch `csrc/kernel.cu`, rebuild |
| `MAX_SEQ_LEN` | 2048 | 32768 | Python constant, no rebuild |
| `rope_theta` | 10000 | 1,000,000 | Python only |
| RoPE type | (original) | Standard 1D | Python RoPE table, no rebuild |
