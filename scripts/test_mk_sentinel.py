"""
Isolated megakernel sentinel test — run BEFORE the full test suite.
Tests the decode(-1) sentinel path in complete isolation.

Usage:
    V2_MEGAKERNEL=1 V2_CUDA_GRAPHS=0 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
        python scripts/test_mk_sentinel.py
"""
import sys, os, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import torch

from server.backend.tts_backend_v2 import (
    QwenTTSBackendV2, _MK_HIDDEN_SIZE, _MK_MAX_SEQ_LEN
)
from server.backend.tts_backend_v2 import _build_prefill_inputs_and_run

logger.info("Loading backend (V2_MEGAKERNEL=1 required)...")
b = QwenTTSBackendV2()
mk = b._mk_decoder

if mk is None:
    logger.error("mk_decoder is None — megakernel init failed. Check V2_MEGAKERNEL=1 and kernel build.")
    sys.exit(1)

logger.info(f"mk_decoder OK — position={mk._position}")

# ── Test 1: sentinel with zero embedding and empty KV cache ──────────────────
logger.info("Test 1: step_with_embed(zeros), empty KV cache...")
mk.reset()
zero_embed = torch.zeros(_MK_HIDDEN_SIZE, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()
try:
    tok = mk.step_with_embed(zero_embed)
    torch.cuda.synchronize()
    valid = 0 <= tok < 3072
    logger.info(f"Test 1 {'PASS' if valid else 'FAIL'}: token={tok}, in_range={valid}")
    if not valid:
        sys.exit(1)
except Exception as e:
    logger.error(f"Test 1 FAIL: {e}", exc_info=True)
    sys.exit(1)

# ── Test 2: sentinel with real prefill KV and real embedding ─────────────────
logger.info("Test 2: step_with_embed with real prefill KV...")
past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
    _build_prefill_inputs_and_run(b._hf, "Hello.", b._speaker, b._language)

mk.reset()
prefill_len = mk.load_kv_from_hf(past_kv)
logger.info(f"KV loaded — prefill_len={prefill_len}, mk.position={mk._position}")

token = first_logits.squeeze(0).argmax()
codec_embedding = b._talker.get_input_embeddings()
last_id_hidden = codec_embedding(token.unsqueeze(0).unsqueeze(0))  # [1, 1, 1024]
inputs_embeds = last_id_hidden  # simplified — no predictor, just CB0 embed

torch.cuda.synchronize()
try:
    tok = mk.step_with_embed(inputs_embeds)
    torch.cuda.synchronize()
    valid = 0 <= tok < 3072
    logger.info(f"Test 2 {'PASS' if valid else 'FAIL'}: token={tok}, in_range={valid}")
    if not valid:
        sys.exit(1)
except Exception as e:
    logger.error(f"Test 2 FAIL: {e}", exc_info=True)
    sys.exit(1)

# ── Test 3: 5 consecutive steps ──────────────────────────────────────────────
logger.info("Test 3: 5 consecutive sentinel steps...")
mk.reset()
mk.load_kv_from_hf(past_kv)
token = first_logits.squeeze(0).argmax()

for i in range(5):
    last_id_hidden = codec_embedding(token.unsqueeze(0).unsqueeze(0))
    inputs_embeds = last_id_hidden
    try:
        tok = mk.step_with_embed(inputs_embeds)
        torch.cuda.synchronize()
        valid = 0 <= tok < 3072
        logger.info(f"  step {i}: token={tok}, valid={valid}, pos={mk._position-1}")
        if not valid:
            logger.error(f"  step {i} out of range — FAIL")
            sys.exit(1)
        token = torch.tensor(tok, dtype=torch.long, device='cuda')
    except Exception as e:
        logger.error(f"  step {i} FAIL: {e}", exc_info=True)
        sys.exit(1)

logger.info("Test 3 PASS")
logger.info("All sentinel tests passed — kernel is working correctly")
