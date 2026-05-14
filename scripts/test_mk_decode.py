"""
Smoke test for the megakernel decode path — run this BEFORE starting the server.

Usage (on GPU server, from repo root):
    source .venv/bin/activate
    python scripts/test_mk_decode.py

Checks in order:
  1. Megakernel op loads
  2. Inspect decode op schema (prints arg names + types)
  3. Load HF model and extract weights
  4. Run one decode step — confirm no crash
  5. Run the full megakernel generate() patch end-to-end with a short sentence
  6. Write output.wav — play it to verify audio quality
"""

import sys
import time
import torch
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "./qwen_megakernel")


def check_ops():
    from qwen_megakernel.build import get_extension
    get_extension()
    ops = dir(torch.ops.qwen_megakernel_C)
    print(f"[1] Registered ops: {[o for o in ops if not o.startswith('_')]}")
    assert "decode" in ops, "decode op missing — build failed"
    print("    decode op: OK")

    # Print schema so we can verify arg count / types
    try:
        schema = torch.ops.qwen_megakernel_C.decode._schema
        print(f"    decode schema: {schema}")
    except Exception as e:
        print(f"    (schema unavailable: {e})")


def check_single_step():
    from server.backend.tts_backend_mk import (
        NUM_LAYERS, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        HIDDEN_SIZE, VOCAB_SIZE, MAX_SEQ_LEN, ROPE_THETA,
        _build_rope_tables,
    )

    print("\n[2] Allocating minimal buffers for single decode step ...")
    embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    lm_head = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    final_norm = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Minimal layer weights (zeros — just checking the op doesn't crash on shapes)
    n_keys = 11
    import struct
    layer_tensors = []
    layer_shapes = [
        (HIDDEN_SIZE,),          # input_layernorm
        (NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE),  # q_proj  [2048, 1024]
        (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE), # k_proj  [1024, 1024]
        (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE), # v_proj  [1024, 1024]
        (HEAD_DIM,),             # q_norm
        (HEAD_DIM,),             # k_norm
        (HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM),  # o_proj  [1024, 2048]
        (HIDDEN_SIZE,),          # post_attn_norm
        (VOCAB_SIZE, HIDDEN_SIZE), # gate_proj [3072, 1024]
        (VOCAB_SIZE, HIDDEN_SIZE), # up_proj   [3072, 1024]
        (HIDDEN_SIZE, VOCAB_SIZE), # down_proj [1024, 3072]
    ]
    for _ in range(NUM_LAYERS):
        for shape in layer_shapes:
            t = torch.zeros(*shape, dtype=torch.bfloat16, device="cuda").contiguous()
            layer_tensors.append(t)

    buf = bytearray(NUM_LAYERS * n_keys * 8)
    for i, t in enumerate(layer_tensors):
        struct.pack_into("Q", buf, i * 8, t.data_ptr())
    layer_weights_packed = torch.frombuffer(buf, dtype=torch.uint8).cuda()

    cos_table, sin_table = _build_rope_tables()

    k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    hidden = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    activations = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    q = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    attn_out = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    mlp_intermediate = torch.zeros(VOCAB_SIZE * 2, dtype=torch.bfloat16, device="cuda")
    normalized = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    block_max_vals = torch.full((8,), float("-inf"), dtype=torch.float32, device="cuda")
    block_max_idxs = torch.zeros(8, dtype=torch.int32, device="cuda")
    output_token = torch.zeros(1, dtype=torch.int32, device="cuda")

    attn_scale = float(HEAD_DIM ** -0.5)

    print("    Calling decode op with zero weights ...")
    try:
        torch.ops.qwen_megakernel_C.decode(
            output_token, 0,
            embed, layer_weights_packed, final_norm, lm_head,
            cos_table, sin_table,
            k_cache, v_cache,
            hidden, activations, residual,
            q, k, v, attn_out, mlp_intermediate, normalized,
            block_max_vals, block_max_idxs,
            NUM_LAYERS, 0, MAX_SEQ_LEN, attn_scale,
        )
        torch.cuda.synchronize()
        print(f"    decode op: OK — output_token={output_token.item()}")
    except Exception as e:
        print(f"    decode op FAILED: {e}")
        print("    Check arg count/types against torch_bindings.cpp")
        raise


def check_end_to_end():
    print("\n[3] Loading HF model + running megakernel generate() patch ...")
    from server.backend.tts_backend_mk import QwenTTSBackendMK
    import asyncio

    backend = QwenTTSBackendMK()

    async def run():
        chunks = []
        t_start = time.perf_counter()
        t_first = None
        async for audio_bytes, sr in backend.synthesize_streaming("Hello, this is a test."):
            if t_first is None:
                t_first = time.perf_counter()
                print(f"    TTFC: {(t_first - t_start)*1000:.0f}ms")
            chunks.append(np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        audio = np.concatenate(chunks)
        audio_dur = len(audio) / sr
        gen_time = t_end - t_start
        print(f"    RTF: {gen_time/audio_dur:.3f}  (target < 0.15)")
        print(f"    Audio duration: {audio_dur*1000:.0f}ms  Gen time: {gen_time*1000:.0f}ms")

        import soundfile as sf
        sf.write("output_mk_test.wav", audio, sr)
        print("    Saved output_mk_test.wav — play to verify audio quality")

    asyncio.run(run())


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    check_ops()
    check_single_step()
    check_end_to_end()
    print("\n=== All checks passed ===")
