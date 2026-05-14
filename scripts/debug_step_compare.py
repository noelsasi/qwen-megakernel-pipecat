"""
Minimal repro: compare direct decode() call vs _MKDecoder.step()
with identical inputs in the same process.

If direct call works but step() fails → arg order or buffer issue in _MKDecoder.
If both fail → something else.
"""
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "./qwen_megakernel")

from qwen_megakernel.build import get_extension
get_extension()

from qwen_tts import Qwen3TTSModel
from server.backend.tts_backend_mk import (
    _extract_talker_weights, _pack_layer_weights, _build_rope_tables,
    _MKDecoder, NUM_LAYERS, NUM_KV_HEADS, NUM_Q_HEADS,
    HEAD_DIM, HIDDEN_SIZE, VOCAB_SIZE, MAX_SEQ_LEN,
)

print("Loading model...")
m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda", dtype=torch.bfloat16
)

# Capture KV cache from real prefill
orig = m.model.talker.model.forward
pkv_obj = [None]
done = [False]
def spy(*a, **kw):
    out = orig(*a, **kw)
    if not done[0]:
        pkv_obj[0] = out.past_key_values
        done[0] = True
    return out
m.model.talker.model.forward = spy
m.generate_custom_voice(text="Hi.", language="English", speaker="Ryan", max_new_tokens=3)
m.model.talker.model.forward = orig

pkv = pkv_obj[0]
prefill_len = pkv.layers[0].keys.shape[2]
print(f"prefill_len={prefill_len}")

# Build weights
state = m.model.state_dict()
weights = _extract_talker_weights(state)
packed = _pack_layer_weights(weights["layer_weights"])
inv_freq = m.model.talker.model.rotary_emb.inv_freq.detach().cpu()
cos_table, sin_table = _build_rope_tables(inv_freq=inv_freq)

# Build k/v cache with HF data
k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
v_cache = torch.zeros_like(k_cache)
for i, layer in enumerate(pkv.layers):
    seq = layer.keys.shape[2]
    k_cache[i, :, :seq, :] = layer.keys[0].to(torch.bfloat16)
    v_cache[i, :, :seq, :] = layer.values[0].to(torch.bfloat16)

# ── Test 1: direct positional call (known working) ──────────────────────────
print("\n=== Test 1: Direct decode() call ===")
hidden = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
activations = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
residual = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
q = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
k = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
v = torch.zeros_like(k)
attn_out = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
mlp_intermediate = torch.zeros(VOCAB_SIZE * 2, dtype=torch.bfloat16, device="cuda")
normalized = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
block_max_vals = torch.full((8,), float("-inf"), dtype=torch.float32, device="cuda")
block_max_idxs = torch.zeros(8, dtype=torch.int32, device="cuda")
output_token = torch.zeros(1, dtype=torch.int32, device="cuda")

torch.ops.qwen_megakernel_C.decode(
    output_token, 1995,
    weights["embed_weight"], packed, weights["final_norm_weight"], weights["lm_head_weight"],
    cos_table, sin_table, k_cache, v_cache,
    hidden, activations, residual, q, k, v, attn_out, mlp_intermediate, normalized,
    block_max_vals, block_max_idxs,
    NUM_LAYERS, prefill_len, MAX_SEQ_LEN, float(HEAD_DIM ** -0.5),
)
torch.cuda.synchronize()
t1 = output_token.item()
print(f"direct decode(1995, pos={prefill_len}) → {t1}  valid={0<=t1<VOCAB_SIZE}")

# ── Test 2: _MKDecoder.step() with same inputs ───────────────────────────────
print("\n=== Test 2: _MKDecoder.step() ===")
decoder = _MKDecoder(weights, inv_freq=inv_freq)
decoder.reset()
decoder.load_kv_cache_from_hf(pkv)
print(f"decoder._position after load: {decoder._position}")

# Print ptrs to check they match
print(f"Direct k_cache ptr:    {k_cache.data_ptr():#x}")
print(f"decoder._k_cache ptr:  {decoder._k_cache.data_ptr():#x}")
print(f"decoder cos_table ptr: {decoder._cos_table.data_ptr():#x}")
print(f"direct  cos_table ptr: {cos_table.data_ptr():#x}")

# Check KV cache values match
print(f"KV match layer0 head0: {torch.allclose(k_cache[0,0,:prefill_len,:], decoder._k_cache[0,0,:prefill_len,:], atol=1e-3)}")

t2 = decoder.step(1995)
print(f"decoder.step(1995) → {t2}  valid={0<=t2<VOCAB_SIZE}")

# ── Test 3: _call_args() order vs direct order ───────────────────────────────
print("\n=== Test 3: _call_args() order verification ===")
call_args = decoder._call_args()
direct_args = (
    weights["embed_weight"], packed, weights["final_norm_weight"], weights["lm_head_weight"],
    cos_table, sin_table, k_cache, v_cache,
    hidden, activations, residual, q, k, v, attn_out, mlp_intermediate, normalized,
    block_max_vals, block_max_idxs,
)
names = [
    "embed_weight", "layer_weights_packed", "final_norm_weight", "lm_head_weight",
    "cos_table", "sin_table", "k_cache", "v_cache",
    "hidden", "activations", "residual", "q", "k", "v", "attn_out",
    "mlp_intermediate", "normalized", "block_max_vals", "block_max_idxs",
]
print(f"{'idx':<4} {'name':<22} {'direct_shape':<25} {'decoder_shape':<25} {'ptr_match'}")
for i, (name, da, ca) in enumerate(zip(names, direct_args, call_args)):
    ds = str(tuple(da.shape))
    cs = str(tuple(ca.shape))
    pm = da.data_ptr() == ca.data_ptr()
    print(f"{i:<4} {name:<22} {ds:<25} {cs:<25} {pm}")
