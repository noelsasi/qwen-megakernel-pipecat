"""
Diagnostic: KV cache RoPE compatibility between HF and megakernel.

Answers two questions:
  Q1. Does HF store pre-RoPE or post-RoPE keys in DynamicCache?
  Q2. Does HF use adjacent-pair rotation or split-half rotation?
  Q3. Does the megakernel re-apply RoPE during attention, or expect pre-rotated keys?
  Q4. Numeric comparison of HF cached K vs megakernel K after one decode step.

Run from repo root:
    python scripts/debug_rope_kvcache.py
"""

import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "./qwen_megakernel")

from qwen_megakernel.build import get_extension
get_extension()

from qwen_tts import Qwen3TTSModel
from server.backend.tts_backend_mk import (
    _extract_talker_weights, _pack_layer_weights, _build_rope_tables,
    NUM_LAYERS, NUM_KV_HEADS, NUM_Q_HEADS, HEAD_DIM, HIDDEN_SIZE,
    VOCAB_SIZE, MAX_SEQ_LEN,
)

print("Loading model...")
m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda", dtype=torch.bfloat16
)

# ── Q1/Q2: Capture keys BEFORE and AFTER RoPE inside HF attention ──────────
print("\n=== Q1/Q2: HF RoPE application order ===")

attn = m.model.talker.model.layers[0].self_attn
orig_attn_fwd = attn.forward
captured = {}

def spy_attn(*args, **kwargs):
    # Patch inside the attention to capture pre/post RoPE keys
    orig_apply_rope = None
    import qwen_tts.core.models.modeling_qwen3_tts as mod

    # Capture the raw K projection output before RoPE
    orig_k_proj = attn.k_proj.forward
    def spy_k_proj(x):
        out = orig_k_proj(x)
        captured["k_raw"] = out.detach().clone()
        return out
    attn.k_proj.forward = spy_k_proj

    result = orig_attn_fwd(*args, **kwargs)
    attn.k_proj.forward = orig_k_proj
    return result

attn.forward = spy_attn

# Also capture what goes INTO the cache
orig_model_fwd = m.model.talker.model.forward
call_n = [0]
def spy_model_fwd(*args, **kwargs):
    out = orig_model_fwd(*args, **kwargs)
    if call_n[0] == 0:
        pkv = out.past_key_values
        captured["cached_k_layer0"] = pkv.layers[0].keys.detach().clone()
        captured["cached_v_layer0"] = pkv.layers[0].values.detach().clone()
    call_n[0] += 1
    return out
m.model.talker.model.forward = spy_model_fwd

m.generate_custom_voice(text="Hi.", language="English", speaker="Ryan", max_new_tokens=3)
m.model.talker.model.forward = orig_model_fwd
attn.forward = orig_attn_fwd

k_raw = captured.get("k_raw")          # [1, seq, num_kv_heads*head_dim] or [1, seq, kv_heads, head_dim]
cached_k = captured.get("cached_k_layer0")  # [1, kv_heads, seq, head_dim]

print(f"k_raw shape:    {k_raw.shape if k_raw is not None else 'NOT CAPTURED'}")
print(f"cached_k shape: {cached_k.shape}")
seq_len = cached_k.shape[2]

if k_raw is not None:
    # Reshape k_raw to [seq, kv_heads, head_dim] for comparison
    if k_raw.dim() == 3:
        k_raw_r = k_raw[0].view(seq_len, NUM_KV_HEADS, HEAD_DIM)  # [seq, heads, dim]
    else:
        k_raw_r = k_raw[0]  # already [seq, heads, dim]

    cached_k_r = cached_k[0].permute(1, 0, 2)  # [seq, heads, dim]

    # Compare first token, first head
    raw_h0 = k_raw_r[0, 0].float()
    cached_h0 = cached_k_r[0, 0].float()

    print(f"\nLayer 0, head 0, position 0:")
    print(f"  k_raw[:8]:    {raw_h0[:8].tolist()}")
    print(f"  cached_k[:8]: {cached_h0[:8].tolist()}")

    are_same = torch.allclose(raw_h0, cached_h0, atol=1e-2)
    print(f"  k_raw == cached_k: {are_same}")
    if are_same:
        print("  → HF stores PRE-RoPE keys (raw projection, no rotation applied)")
    else:
        print("  → HF stores POST-RoPE keys (rotation applied before caching)")

        # Check if it's adjacent-pair or split-half rotation
        # Adjacent-pair: pairs are (x[0],x[1]), (x[2],x[3])...
        # Split-half:    pairs are (x[0],x[64]), (x[1],x[65])...
        half = HEAD_DIM // 2
        # Try to recover cos/sin from the relationship
        # For adjacent-pair: cached[i] = raw[i]*cos - raw[i+1]*sin  (i even)
        # For split-half:    cached[i] = raw[i]*cos - raw[i+half]*sin
        print(f"\n  Checking rotation layout:")
        print(f"  raw[0]={raw_h0[0].item():.4f} raw[1]={raw_h0[1].item():.4f}")
        print(f"  raw[{half}]={raw_h0[half].item():.4f}")
        print(f"  cached[0]={cached_h0[0].item():.4f} cached[1]={cached_h0[1].item():.4f}")

        # Build cos/sin for pos=0 from HF inv_freq
        inv_freq = m.model.talker.model.rotary_emb.inv_freq.float().cpu()
        cos_0 = torch.cos(inv_freq)  # [64] — cos at position 0
        sin_0 = torch.sin(inv_freq)  # [64] — sin at position 0 (all zeros since freq*0=0)
        # At pos=0, sin=0 and cos=1, so cached should equal raw regardless of layout
        print(f"\n  At pos=0: sin=0, cos=1 → cached should equal raw")
        print(f"  Checking pos=1 instead:")
        raw_h0_p1 = k_raw_r[1, 0].float()
        cached_h0_p1 = cached_k_r[1, 0].float()
        # pos=1 frequencies
        cos_1 = torch.cos(inv_freq * 1.0)   # [64]
        sin_1 = torch.sin(inv_freq * 1.0)   # [64]

        # Test adjacent-pair: cached[2i] = raw[2i]*cos[i] - raw[2i+1]*sin[i]
        adj_recon = torch.zeros(HEAD_DIM)
        for i in range(half):
            adj_recon[2*i]   = raw_h0_p1[2*i]   * cos_1[i] - raw_h0_p1[2*i+1] * sin_1[i]
            adj_recon[2*i+1] = raw_h0_p1[2*i+1] * cos_1[i] + raw_h0_p1[2*i]   * sin_1[i]

        # Test split-half: cached[i] = raw[i]*cos[i] - raw[i+half]*sin[i]
        split_recon = torch.zeros(HEAD_DIM)
        cos_full = torch.cat([cos_1, cos_1])
        sin_full = torch.cat([sin_1, sin_1])
        split_recon[:half] = raw_h0_p1[:half] * cos_1 - raw_h0_p1[half:] * sin_1
        split_recon[half:] = raw_h0_p1[half:] * cos_1 + raw_h0_p1[:half] * sin_1

        adj_err   = (adj_recon   - cached_h0_p1).abs().mean().item()
        split_err = (split_recon - cached_h0_p1).abs().mean().item()
        print(f"  adjacent-pair reconstruction error: {adj_err:.6f}")
        print(f"  split-half    reconstruction error: {split_err:.6f}")
        if adj_err < split_err:
            print("  → HF uses ADJACENT-PAIR rotation")
        else:
            print("  → HF uses SPLIT-HALF rotation")

# ── Q3: Does megakernel re-apply RoPE during attention? ─────────────────────
print("\n=== Q3: Megakernel RoPE in attention ===")
print("Checking kernel source...")
import subprocess
result = subprocess.run(
    ["grep", "-n", "cos\|sin\|rope\|rotat\|RoPE\|k_cache\[", "qwen_megakernel/csrc/kernel.cu"],
    capture_output=True, text=True
)
lines = [l for l in result.stdout.splitlines()
         if any(x in l for x in ["cos_pos", "sin_pos", "k_cache[", "kl[", "cos_table", "sin_table"])]
for l in lines[:20]:
    print(" ", l)

# ── Q4: Numeric comparison after real decode step ───────────────────────────
print("\n=== Q4: Megakernel K cache vs HF K cache at layer 0, head 0 ===")
state = m.model.state_dict()
weights = _extract_talker_weights(state)
packed = _pack_layer_weights(weights["layer_weights"])
inv_freq = m.model.talker.model.rotary_emb.inv_freq.detach().cpu()
cos_table, sin_table = _build_rope_tables(inv_freq=inv_freq)

k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
v_cache = torch.zeros_like(k_cache)

# Copy HF prefill KV
pkv = captured["cached_k_layer0"]  # already have from spy
for layer_idx, layer in enumerate(captured.get("pkv_layers", [])):
    pass  # re-run below

# Re-capture with pkv object
call_n[0] = 0
pkv_obj = [None]
def spy2(*args, **kwargs):
    out = orig_model_fwd(*args, **kwargs)
    if call_n[0] == 0:
        pkv_obj[0] = out.past_key_values
    call_n[0] += 1
    return out
m.model.talker.model.forward = spy2
m.generate_custom_voice(text="Hi.", language="English", speaker="Ryan", max_new_tokens=2)
m.model.talker.model.forward = orig_model_fwd

pkv = pkv_obj[0]
for layer_idx, layer in enumerate(pkv.layers):
    k = layer.keys; v = layer.values
    seq = k.shape[2]
    k_cache[layer_idx, :, :seq, :] = k[0].to(torch.bfloat16)
    v_cache[layer_idx, :, :seq, :] = v[0].to(torch.bfloat16)

prefill_len = pkv.layers[0].keys.shape[2]
print(f"prefill_len: {prefill_len}")
print(f"HF cached K layer0 head0 pos0 [:8]: {pkv.layers[0].keys[0,0,0,:8].float().tolist()}")
print(f"MK k_cache   layer0 head0 pos0 [:8]: {k_cache[0,0,0,:8].float().tolist()}")
print(f"Values match: {torch.allclose(pkv.layers[0].keys[0,:,:seq,:].to(torch.bfloat16), k_cache[0,:,:seq,:], atol=1e-3)}")

# Run one decode step and check if output is valid
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
tok = output_token.item()
print(f"\ndecode(1995, pos={prefill_len}) with HF KV cache → token {tok}")
print(f"Valid: {0 <= tok < VOCAB_SIZE}")

# Now try with ZERO KV cache for comparison
k_cache2 = torch.zeros_like(k_cache)
v_cache2 = torch.zeros_like(v_cache)
output_token2 = torch.zeros(1, dtype=torch.int32, device="cuda")
block_max_vals.fill_(float("-inf"))
torch.ops.qwen_megakernel_C.decode(
    output_token2, 1995,
    weights["embed_weight"], packed, weights["final_norm_weight"], weights["lm_head_weight"],
    cos_table, sin_table, k_cache2, v_cache2,
    hidden, activations, residual, q, k, v, attn_out, mlp_intermediate, normalized,
    block_max_vals, block_max_idxs,
    NUM_LAYERS, 0, MAX_SEQ_LEN, float(HEAD_DIM ** -0.5),
)
torch.cuda.synchronize()
print(f"decode(1995, pos=0)  with zero KV cache → token {output_token2.item()} (baseline)")
