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
| RTF | 1.070 | 0.236 | **0.126–0.158** | < 0.15 |
| TTFC | 6338 ms | 142 ms | **120 ms** | < 60 ms |
| Codec frames/s | ~12 | ~60 | **~95** | — |
| Streaming | Buffered (fake) | **Real** (per-frame) | **Real** (per-frame) | Real ✅ |
| EOS | Never fired | **Fires correctly** | **Fires correctly** | — ✅ |

> All numbers measured on RTX 5090, "Hello, this is a test.", after CUDA graph warmup.
> Megakernel raw decode RTF: 0.126 (Stage 6). End-to-end streaming RTF: 0.158 (Stage 5 — includes async vocoder thread and chunk queue overhead).

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

When enabled, the talker backbone step uses `torch.ops.qwen_megakernel_C.decode` with a sentinel token (`-1`) that reads the pre-built `inputs_embeds` directly from the hidden buffer, bypassing the embedding lookup table. This lets the correct summed 16-codebook embedding pass through the kernel's 28-layer fused transformer.

**What the megakernel actually computes** (from `kernel.cu` source inspection):

The kernel launches two back-to-back CUDA kernels per step:
1. `ldg_decode_kernel_direct` (128 blocks × 512 threads) — embedding lookup → 28 transformer layers (RMSNorm, QKV, RoPE, GQA attention, O-proj, SiLU-gated MLP, residual) → final RMSNorm → `g_normalized`
2. `ldg_lm_head_fused` (1184 blocks × 256 threads) — greedy argmax over `g_normalized @ lm_head_weight.T` → writes `*output_token`

We use the full output of kernel (1) and discard the output of kernel (2).

**Why the kernel's argmax is discarded — correctness, not preference:**

The kernel's lm_head argmax is a raw greedy max over all 3072 logits with no masking. Qwen3-TTS requires suppressing tokens `[2048..2149]` and `[2151..3071]` — only EOS (token 2150) is valid in that range. Without this mask, high-frequency tokens like 122 and 2035 win consistently and EOS is never reached (confirmed empirically: sequences loop indefinitely without the suppress mask). This is a functional correctness requirement, not a quality preference.

After each kernel call, Python recomputes the token:
```
_hidden (raw residual) → RMSNorm → lm_head matmul [3072,1024]@[1024] → suppress mask → sampling
```
The lm_head matmul over 3072 tokens costs ~0.05ms — negligible on a 5090. No additional CPU sync is introduced beyond the `token.item()` sync already required to feed the next codec embedding.

**Why `generate_nosync` (the fully GPU-side N-step path) cannot be used:**

`torch.ops.qwen_megakernel_C.generate_nosync` runs N steps on-device with zero CPU round-trips by feeding each argmax output back as the next integer token input. This is architecturally incompatible with Qwen3-TTS for two reasons: (1) each decode step requires a Python-side code predictor pass (15 steps → 16 codebook tokens → summed float embedding) that cannot be expressed as an integer token ID, and (2) the suppress mask cannot be injected between steps without modifying the kernel. The only correct integration point is the per-step sentinel path we use.

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

### RTF near target; TTFC still above target

**End-to-end streaming RTF: 0.158 (target < 0.15). Raw decode RTF: 0.126 ✅.**
**TTFC: 120ms (target < 60ms).**

The 0.032 RTF gap between raw decode and streaming is async overhead: vocoder thread, chunk queue, and Python async dispatch per chunk. The raw decode loop at 95 frames/s already clears the target.

TTFC breakdown: prefill ~21ms + 4 frames × ~10ms = ~61ms minimum. The remaining ~60ms is chunk queue and vocoder thread latency. To hit 60ms, set `CHUNK_FRAMES=1` (emit first frame immediately, 80ms audio) — trades chunk granularity for latency.

### Audio quality

The v2 pipeline produces audio (confirmed: EOS fires, codec frames are valid, vocoder outputs waveform). Formal quality comparison against HF baseline has not been done.

### GPU target

Megakernel targets `sm_120a` (RTX 5090 Blackwell) only. The v2 custom decode loop runs on any CUDA GPU.

---

## Kernel Modifications

Required for `V2_MEGAKERNEL=1`. Applied by `scripts/setup_server.sh`.

| Item | Default (upstream) | Required | Where |
|------|-------------------|----------|-------|
| `LDG_VOCAB_SIZE` | 151936 (text vocab) | **3072** (codec vocab) | `csrc/kernel.cu` — requires rebuild |
| `MAX_SEQ_LEN` | 32768 | **1024** | `csrc/kernel.cu` — reduces KV alloc from 1.88 GB to 118 MB, recovers ~4× tok/s |
| Sentinel path (`input_token_id == -1`) | not present | **read `hidden_buffer` as embedding** | `csrc/kernel.cu` embed lookup line, requires rebuild |
| `rope_theta` | 10000 | 1,000,000 | Python RoPE table — no rebuild |
| RoPE type | (original) | Standard 1D | Python RoPE table — no rebuild |

The sentinel patch is the key change: one ternary in `kernel.cu` at the embedding lookup:
```cuda
// Before:
const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

// After:
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;   // sentinel: caller pre-writes inputs_embeds here
```

This lets the float embedding (summed 16-codebook embed + text conditioning) pass through the full 28-layer fused transformer without requiring an integer token ID.
