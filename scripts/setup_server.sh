#!/usr/bin/env bash
# One-shot setup on RTX 5090 Vast.ai instance.
# Usage:
#   bash scripts/setup_server.sh
#
# After setup:
#   cp .env.example .env && nano .env   # fill OPENAI_API_KEY and DEEPGRAM_API_KEY
#   source .venv/bin/activate
#   set -a && source .env && set +a
#   V2_MEGAKERNEL=1 TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000

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
echo "=== [6/7] Megakernel (required for V2_MEGAKERNEL=1, the recommended path) ==="
# The v2 backend uses the megakernel for the 28-layer talker forward when V2_MEGAKERNEL=1.
# Falls back to TalkerGraph (CUDA graph of HF forward) if not built or flag not set.
if [ ! -d "qwen_megakernel" ]; then
    git clone https://github.com/AlpinDale/qwen_megakernel
    echo "Cloned qwen_megakernel"
else
    echo "qwen_megakernel already present — skipping clone"
fi

# Patch 1: LDG_VOCAB_SIZE 151936→3072 (text LLM vocab → codec token vocab). Idempotent.
sed -i 's/LDG_VOCAB_SIZE = 151936/LDG_VOCAB_SIZE = 3072/' qwen_megakernel/csrc/kernel.cu
echo "kernel.cu: LDG_VOCAB_SIZE → 3072 (patched)"

# Patch 2: Sentinel support — allow token_id=-1 to read from hidden_buffer.
# This lets the v2 decode loop pass inputs_embeds directly to the kernel
# instead of going through an embed table lookup.
# The patch is idempotent (grep checks before applying).
KERNEL_FILE="qwen_megakernel/csrc/kernel.cu"
if grep -q "input_token_id >= 0" "$KERNEL_FILE"; then
    echo "kernel.cu: sentinel patch already applied"
else
    # Find the embed_row assignment and add the sentinel conditional.
    # The original line looks like:
    #   const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    python3 - <<'PYEOF'
import re, sys

path = "qwen_megakernel/csrc/kernel.cu"
with open(path) as f:
    src = f.read()

old = r'const __nv_bfloat16 \*embed_row = embed_weight \+ input_token_id \* HIDDEN_SIZE;'
new = (
    "const __nv_bfloat16 *embed_row =\n"
    "        (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE\n"
    "                              : hidden_buffer;"
)

patched = re.sub(old, new, src)
if patched == src:
    print("ERROR: Could not find embed_row line to patch. Inspect kernel.cu manually.", file=sys.stderr)
    # Show context so user can find it
    for i, line in enumerate(src.splitlines()):
        if 'embed_row' in line:
            print(f"  Line {i+1}: {line.strip()}", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(patched)
print("kernel.cu: sentinel patch applied (embed_row ternary)")
PYEOF
fi

# Patch 3: Reduce MAX_SEQ_LEN from 32768 to 1024.
# At 32768 the KV cache is 1.88GB per cache (2 caches = 3.76GB) — tanks tok/s.
# TTS sequences are at most ~300 tokens. 1024 is safe and recovers ~4× tok/s.
# Idempotent.
if grep -q "MAX_SEQ_LEN = 32768" "$KERNEL_FILE" || grep -q "MAX_SEQ_LEN=32768" "$KERNEL_FILE"; then
    sed -i 's/MAX_SEQ_LEN = 32768/MAX_SEQ_LEN = 1024/' "$KERNEL_FILE"
    sed -i 's/MAX_SEQ_LEN=32768/MAX_SEQ_LEN=1024/' "$KERNEL_FILE"
    echo "kernel.cu: MAX_SEQ_LEN → 1024 (patched)"
else
    echo "kernel.cu: MAX_SEQ_LEN already at target (or different format — check manually)"
fi

# JIT-build the extension (sm_120a = RTX 5090 Blackwell)
# pip install -e . is NOT required — get_extension() compiles via torch.utils.cpp_extension
cd qwen_megakernel/qwen_megakernel && python build.py 2>&1 || true && cd ../..

python -c "
import sys
sys.path.insert(0, 'qwen_megakernel/qwen_megakernel')
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
echo "    V2_MEGAKERNEL=1 python scripts/test_v2_decode.py   # all 6 stages including mk"
echo "    # (without flag: stages 1-5 only, uses TalkerGraph fallback)"
echo ""
echo " 3. Start the server (v2 + megakernel — recommended):"
echo "    source .venv/bin/activate"
echo "    set -a && source .env && set +a"
echo "    V2_MEGAKERNEL=1 TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000"
echo ""
echo "    Without megakernel (CUDA graph fallback, slower):"
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
echo " Expected server startup time: ~15s (megakernel load + PredictorGraph CUDA graph capture)"
echo " Expected performance (V2_MEGAKERNEL=1): raw RTF ~0.126, streaming RTF ~0.158, TTFC ~120ms"
echo " Expected performance (TalkerGraph fallback): RTF ~0.236, TTFC ~142ms"
echo ""
echo " Backend options (TTS_BACKEND env var):"
echo "   v2          — custom decode loop, use with V2_MEGAKERNEL=1 (recommended)"
echo "   hf          — pure HuggingFace baseline (slow, for comparison)"
echo "   megakernel  — Phase 1 monkey-patch (deprecated, use v2)"
echo "================================================================"
