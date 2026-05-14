#!/usr/bin/env bash
# One-shot setup on RTX 5090 Vast.ai instance.
# Usage:
#   bash scripts/setup_server.sh
#
# After setup:
#   cp .env.example .env && nano .env   # fill OPENAI_API_KEY and DEEPGRAM_API_KEY
#   source .venv/bin/activate
#   set -a && source .env && set +a
#   TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== [1/7] GPU check ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | grep release

echo ""
echo "=== [2/7] System deps ==="
apt-get update -q
apt-get install -y -q libsndfile1 ffmpeg git

echo ""
echo "=== [3/7] Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

echo ""
echo "=== [4/7] PyTorch (CUDA 12.8) ==="
# Must install torch first — qwen-tts and pipecat depend on it
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check driver and CUDA version'
print(f'  GPU:  {torch.cuda.get_device_name(0)}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')
"

echo ""
echo "=== [5/7] Python packages ==="
pip install -r requirements.txt --quiet
# qwen-tts installed separately to ensure it gets the already-installed torch
pip install -U qwen-tts --quiet

echo ""
echo "=== [6/7] Megakernel (optional — only needed for TTS_BACKEND=megakernel) ==="
# The v2 backend (TTS_BACKEND=v2) does NOT require the megakernel.
# Build it anyway so the option is available.
if [ ! -d "qwen_megakernel" ]; then
    git clone https://github.com/AlpinDale/qwen_megakernel
    echo "Cloned qwen_megakernel"
else
    echo "qwen_megakernel already present — skipping clone"
fi

# Patch kernel.cu: default LDG_VOCAB_SIZE=151936 (text LLM vocab),
# talker needs 3072 (codec token vocab). Idempotent — sed no-ops if already patched.
sed -i 's/LDG_VOCAB_SIZE = 151936/LDG_VOCAB_SIZE = 3072/' qwen_megakernel/csrc/kernel.cu
echo "kernel.cu: LDG_VOCAB_SIZE → 3072 (patched)"

# JIT-build the extension (sm_120a = RTX 5090 Blackwell)
# pip install -e . is NOT required — get_extension() compiles via torch.utils.cpp_extension
cd qwen_megakernel && python build.py 2>&1 || true && cd ..

python -c "
import sys
sys.path.insert(0, 'qwen_megakernel')
try:
    from qwen_megakernel.build import get_extension
    get_extension()
    import torch
    ops = [x for x in dir(torch.ops.qwen_megakernel_C) if not x.startswith('_')]
    print(f'  megakernel ops: {ops}')
except Exception as e:
    print(f'  megakernel build warning (non-fatal for v2 backend): {e}')
"

echo ""
echo "=== [7/7] Validate v2 backend imports ==="
python -c "
from server.backend.tts_backend_v2 import QwenTTSBackendV2
from server.backend.cuda_graphs import TalkerGraph, PredictorGraph
print('  v2 backend imports OK')
"

echo ""
echo "================================================================"
echo " SETUP COMPLETE"
echo "================================================================"
echo ""
echo " NEXT STEPS:"
echo ""
echo " 1. Set environment variables:"
echo "    cp .env.example .env"
echo "    nano .env   # fill OPENAI_API_KEY and DEEPGRAM_API_KEY"
echo ""
echo " 2. Validate the v2 decode pipeline (takes ~2 min, includes CUDA graph warmup):"
echo "    source .venv/bin/activate"
echo "    python scripts/test_v2_decode.py"
echo ""
echo " 3. Start the server (v2 backend — custom decode loop + CUDA graphs):"
echo "    source .venv/bin/activate"
echo "    set -a && source .env && set +a"
echo "    TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000"
echo ""
echo " 4. On your local machine, open SSH tunnel:"
echo "    ssh -p PORT root@IP -L 8000:localhost:8000 -N"
echo ""
echo " 5. Run frontend:"
echo "    cd client && npm install"
echo "    VITE_WS_URL=ws://localhost:8000/ws npm run dev"
echo "    # Open http://localhost:5173 → click CONNECT → speak"
echo ""
echo " Expected server startup time: ~30s (CUDA graph capture for talker + predictor)"
echo " Expected performance: TTFC ~135ms, RTF ~0.21"
echo ""
echo " Backend options (TTS_BACKEND env var):"
echo "   v2          — custom decode loop + CUDA graphs (recommended)"
echo "   hf          — pure HuggingFace baseline (slow, for comparison)"
echo "   megakernel  — Phase 1 monkey-patch (deprecated, use v2)"
echo "================================================================"
