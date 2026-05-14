# Qwen3-TTS + Megakernel + Pipecat Voice Agent

Real-time voice agent: microphone → STT → LLM → Qwen3-TTS (optionally accelerated by CUDA megakernel) → speaker.

```
Mic → [STT] → [LLM] → [Qwen3-TTS Talker] → [Code Predictor] → [Vocoder] → Speaker
                              ↑
                    CUDA megakernel replaces
                    the autoregressive decode loop
```

## Performance Targets

| Metric | Target | Baseline (HF, no megakernel) |
|--------|--------|------------------------------|
| RTF (real-time factor) | < 0.15 | 0.879 |
| TTFC (time to first chunk) | < 60 ms | — |

RTF < 0.15 means audio is generated 6.7× faster than real-time.

---

## Requirements

- **GPU server:** RTX 5090 (Blackwell, sm_120a), CUDA 12.8+, driver 570+
- **GPU RAM:** ~8 GB (Qwen3-TTS 0.6B in bfloat16)
- **Recommended:** Vast.ai RTX 5090 instance
- **Frontend:** Any machine with Node.js 18+
- **API keys:** OpenAI (LLM) + optionally Deepgram (STT)

---

## GPU Server Setup (Vast.ai)

### 1. SSH into the instance and clone the repo

```bash
git clone <your-repo-url> /workspace/qwen-megakernel-pipecat
cd /workspace/qwen-megakernel-pipecat
```

Or push from local:

```bash
# On your local machine
rsync -avz --exclude .venv --exclude __pycache__ --exclude node_modules \
  /path/to/qwen-megakernel-pipecat/ \
  root@<vast-ip>:/workspace/qwen-megakernel-pipecat/
```

### 2. Run the setup script

```bash
cd /workspace/qwen-megakernel-pipecat
bash scripts/setup_server.sh
```

This installs system deps, creates `.venv`, installs PyTorch (cu128), installs all Python packages, and clones the megakernel repo. Takes ~10 minutes on first run.

Or do it manually step by step:

```bash
# System deps
apt update && apt install -y libsndfile1 ffmpeg git

# Python venv
python3 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA 12.8 (install FIRST — other packages depend on it)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Project dependencies
pip install -r requirements.txt

# Clone megakernel
git clone https://github.com/AlpinDale/qwen_megakernel
```

### 3. Patch and build the megakernel

The megakernel's default vocab size (151936) targets the text LLM. Qwen3-TTS uses a codec vocabulary of 3072 — patch it before compiling:

```bash
# Patch (only needed once — check current value first)
grep 'LDG_VOCAB_SIZE' qwen_megakernel/csrc/kernel.cu
# If it shows 151936, patch it:
sed -i 's/LDG_VOCAB_SIZE = 151936/LDG_VOCAB_SIZE = 3072/' qwen_megakernel/csrc/kernel.cu

# Build (JIT compile via torch.utils.cpp_extension — takes ~2 min)
python -c "import sys; sys.path.insert(0, 'qwen_megakernel'); from qwen_megakernel.build import get_extension; get_extension()"
```

JIT cache is stored at `/root/.cache/torch_extensions/py3xx_cu128/qwen_megakernel_C/` — subsequent starts are instant.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
#   ALLOWED_ORIGIN   — your Vercel app URL (or * for local testing)
#   HF_TOKEN         — HuggingFace token (needed to download the model)
#   OPENAI_API_KEY   — for LLM (gpt-4o-mini by default)
#   DEEPGRAM_API_KEY — optional, lower-latency STT than Whisper
#   TTS_BACKEND      — "hf" or "megakernel"
```

### 5. Verify the setup

```bash
source .venv/bin/activate

# GPU check
python -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
# Expected: NVIDIA GeForce RTX 5090  12.8

# HF baseline smoke test (generates output.wav, prints RTF)
python scripts/phase_a_baseline.py
```

### 6. Start the server

```bash
source .venv/bin/activate

# HF backend (default)
TTS_BACKEND=hf make server

# Megakernel backend
TTS_BACKEND=megakernel make server
```

Server listens on `0.0.0.0:8000`.
- `GET /` — health check, returns active backend name
- `WS /ws` — WebSocket endpoint for Pipecat

---

## Frontend Setup (Local or Vercel)

### Local dev

```bash
cd client
npm install
cp ../.env.example .env.local
# Set VITE_WS_URL=ws://<vast-ip>:8000/ws
npm run dev
# Opens at http://localhost:5173
```

### Vercel deploy

```bash
cd client
npm install
# Set in Vercel dashboard:
#   VITE_WS_URL = wss://<your-vast-ip>/ws
vercel deploy --prod
```

---

## TTS Backends

| Backend | Env value | Description |
|---------|-----------|-------------|
| Dev (edge-tts) | `dev` | No GPU needed. For local frontend development only. |
| HuggingFace | `hf` | Vanilla Qwen3-TTS. Requires GPU. Baseline for benchmarking. |
| Megakernel | `megakernel` | Megakernel decode loop replaces HF autoregressive generate. Requires RTX 5090 + patched build. |

Switch at runtime via env var — no code changes needed:

```bash
TTS_BACKEND=megakernel uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
```

---

## Benchmarking

```bash
source .venv/bin/activate

# HF baseline
python scripts/benchmark.py --backend hf --trials 5

# Megakernel
python scripts/benchmark.py --backend megakernel --trials 5
```

Reports: RTF (mean ± std), TTFC (mean ± std), tok/s. Each trial uses `torch.cuda.synchronize()` before stopping the timer.

Monitor GPU utilization during a benchmark run:

```bash
nvidia-smi dmon -s u -d 1
```

---

## Architecture

### Megakernel integration strategy

The megakernel cannot directly replace `model.generate()` because:
- HF prefill uses `inputs_embeds` (mixed float tensors) — megakernel only accepts integer token IDs
- The vocoder has no public API outside `generate_custom_voice()`

**Solution:** monkey-patch `talker.model.generate()` at call time.

```
HF prefill runs normally  →  past_key_values (DynamicCache)
                                      ↓
              copy KV cache into megakernel tensors
                                      ↓
              megakernel decode loop (token-by-token, argmax)
                                      ↓
              return token tensor to HF in expected shape
                                      ↓
              HF code_predictor + vocoder run unchanged
                                      ↓
                              audio waveform
```

The patch installs before `generate_custom_voice()` and restores in `finally` — thread-safe per call.

### KV cache compatibility (confirmed)

| | HF DynamicCache | Megakernel |
|-|----------------|------------|
| Shape | list of 28 × `(k[1,8,seq,128], v[1,8,seq,128])` | `[28, 8, MAX_SEQ_LEN, 128]` |
| Transfer | `load_kv_cache_from_hf()` copies layer-by-layer | pre-allocated, written in-place |

### Talker architecture (confirmed from config + weight inspection)

| Parameter | Value |
|-----------|-------|
| Layers | 28 |
| Hidden size | 1024 |
| Attention heads | 16 (Q), 8 (KV) |
| Head dim | 128 |
| Intermediate size | 3072 |
| Codec vocab size | 3072 |
| RoPE theta | 1,000,000 |
| RoPE type | Interleaved MRope, sections [24, 20, 20] |
| Audio sample rate | 24,000 Hz |
| EOS token | 2150 |

---

## Project Structure

```
.
├── server/
│   ├── backend/
│   │   ├── tts_backend_dev.py      # edge-tts, no GPU, local dev only
│   │   ├── tts_backend_hf.py       # HuggingFace Qwen3-TTS baseline
│   │   └── tts_backend_mk.py       # megakernel decode backend
│   ├── pipecat_services/
│   │   └── qwen_tts_service.py     # Pipecat TTSService subclass
│   └── pipeline/
│       └── voice_agent.py          # FastAPI app, WebSocket endpoint
├── client/                         # React + Vite frontend
├── scripts/
│   ├── setup_server.sh             # one-shot GPU server setup
│   ├── phase_a_baseline.py         # HF inference smoke test
│   ├── phase_a_inspect_model.py    # print model config + weight shapes
│   ├── phase_b_streaming_probe.py  # streaming feasibility check
│   ├── phase_d_compat_check.py     # megakernel constants vs model config
│   └── benchmark.py               # RTF + TTFC + tok/s benchmark suite
├── docs/
│   ├── implementation_plan.md      # full engineering plan
│   ├── progress.md                 # session-by-session progress log
│   └── findings.md                 # confirmed ground truth (API, shapes, numbers)
├── qwen_megakernel/                # cloned on GPU server — not committed
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Performance Numbers

| Metric | Baseline (HF) | With Megakernel |
|--------|---------------|-----------------|
| RTF | 0.879 | — |
| TTFC | — | — |
| tok/s | — | — |
| E2E latency | — | — |

*Megakernel numbers pending full benchmark run. Target RTF < 0.15.*

---

## Known Limitations

- Both backends fake-stream: full audio is generated before any chunk is sent. True streaming (yield audio as tokens decode) requires intercepting the vocoder per-frame, which has no public API in `qwen_tts`.
- TTFC is therefore bounded by full generation time, not first-token time.
- The megakernel monkey-patch approach means prefill still runs on HF — only the decode loop is accelerated.
- RTX 5090 (sm_120a) is required for the megakernel. No CPU or older GPU fallback.

## What Would Come Next

- Hook the vocoder per-frame to enable true streaming and hit the TTFC < 60ms target
- flash-attn install to improve HF baseline RTF before megakernel comparison
- Expose a `/benchmark` HTTP endpoint so results can be recorded without SSH access
