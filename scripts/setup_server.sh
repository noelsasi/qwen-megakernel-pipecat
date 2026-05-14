#!/usr/bin/env bash
# One-shot setup on RTX 5090 Vast.ai instance.
# Usage: bash scripts/setup_server.sh
set -euo pipefail

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | grep release

# System deps
apt-get update -q && apt-get install -y -q libsndfile1 ffmpeg git

# Python venv
python3 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA 12.8 first — other packages depend on torch
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}')
"

# Project deps
pip install -r requirements.txt

# Clone megakernel
if [ ! -d "qwen_megakernel" ]; then
    git clone https://github.com/AlpinDale/qwen_megakernel
fi

# Patch kernel.cu: default vocab=151936 (text LLM), talker needs 3072 (codec tokens)
sed -i 's/LDG_VOCAB_SIZE = 151936/LDG_VOCAB_SIZE = 3072/' qwen_megakernel/csrc/kernel.cu
echo "Patched LDG_VOCAB_SIZE → 3072"

# Build megakernel (targets sm_120a — RTX 5090 only)
cd qwen_megakernel && pip install -e . && cd ..

# Verify
python -c "
from qwen_megakernel.build import get_extension
get_extension()
import torch
print('megakernel ops:', dir(torch.ops.qwen_megakernel_C))
"

echo ""
echo "=== Setup done ==="
echo ""
echo "Copy and fill env:"
echo "  cp .env.example .env && nano .env"
echo ""
echo "Run server:"
echo "  source .venv/bin/activate"
echo "  uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000"
echo ""
echo "Run benchmark:"
echo "  python scripts/benchmark.py --backend megakernel"
