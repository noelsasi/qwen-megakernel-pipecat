#!/usr/bin/env bash
# Run this on the Vast.ai RTX 5090 instance after SSHing in.
# Sets up Python venv, installs all dependencies, clones megakernel repo.
# Usage: bash scripts/setup_server.sh

set -euo pipefail

echo "=== Vast.ai GPU server setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'unknown')"

# System deps
apt-get update -q && apt-get install -y -q libsndfile1 ffmpeg git curl

# Python venv (isolated — does not touch system packages)
python3 -m venv .venv
source .venv/bin/activate
echo "venv: $(which python)"

# PyTorch with CUDA 12.8 (must be first — other packages depend on it)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify GPU is accessible from PyTorch
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch CUDA OK: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
"

# Project dependencies
pip install -r requirements.txt

# Clone megakernel (Phase D)
if [ ! -d "qwen_megakernel" ]; then
    git clone https://github.com/AlpinDale/qwen_megakernel
    echo "Cloned qwen_megakernel"
else
    echo "qwen_megakernel already exists, skipping clone"
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Activate venv:  source .venv/bin/activate"
echo "  2. Inspect model:  python scripts/phase_a_inspect_model.py 2>&1 | tee inspect_output.txt"
echo "  3. Baseline TTS:   python scripts/phase_a_baseline.py"
echo "  4. Start server:   uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000"
