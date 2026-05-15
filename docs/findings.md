# Findings & Observations

> Ground truth only — confirmed by running actual code.
> Last updated: 2026-05-15 (Session 10)

---

## Model Package

| Finding | Value |
|---------|-------|
| pip package | `pip install -U qwen-tts` |
| High-level class | `Qwen3TTSModel` from `qwen_tts` |
| Low-level class | `Qwen3TTSForConditionalGeneration` from `qwen_tts.core.models` — do NOT call directly |
| Tokenization | Internal to `Qwen3TTSModel` — no separate processor needed |
| HF model ID | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| Audio sample rate | **24000 Hz** — "12Hz" in model name refers to codec frame rate, not audio |
| EOS token ID | **2150** (confirmed from pad_token_id warning) |

---

## Inference API

```python
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda",
    dtype=torch.bfloat16,   # not torch_dtype — that's deprecated
)

wavs, sr = model.generate_custom_voice(
    text="Hello world",
    language="English",
    speaker="Ryan",         # must be in supported list
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
)
# wavs: list[np.ndarray], sr: 24000
audio = wavs[0]
```

**Valid speakers:** Ryan, Aiden (EN), Vivian, Serena, Uncle_Fu, Dylan, Eric (ZH), Ono_Anna (JA), Sohee (KO)

**Valid languages:** English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**Lessons from failed attempts:**
- `speakers=["default"]` → `NotImplementedError: Speaker default not implemented`
- `speakers=[None]` → crashes at `input_id[:, :3]` (1D vs 2D tensor inside model)
- Calling `generate()` directly → incompatible — use `generate_custom_voice()`
- `torch_dtype=` kwarg is deprecated — use `dtype=`
- `transformers` from source HEAD does NOT support qwen3_tts — `qwen-tts` is a separate package

---

## Architecture (Confirmed from config + weight inspection)

### Talker (autoregressive decoder)

| Parameter | Value | Source |
|-----------|-------|--------|
| `num_hidden_layers` | 28 | config.json |
| `num_attention_heads` | 16 | config.json + q_proj shape [2048,1024] |
| `num_key_value_heads` | 8 | config.json + k_proj shape [1024,1024] |
| `hidden_size` | 1024 | config.json |
| `head_dim` | **128** | config.json + q_norm shape [128] |
| `intermediate_size` | 3072 | config.json + gate_proj shape [3072,1024] |
| `vocab_size` | **3072** | codec tokens (NOT text tokens) |
| `max_position_embeddings` | 32768 | config.json |
| `rope_theta` | 1,000,000 | config.json |
| `rope_scaling.interleaved` | true | config.json |
| `mrope_section` | [24, 20, 20] | config.json — sums to 64 = HEAD_DIM//2 |
| `rms_norm_eps` | 1e-6 | config.json |

### Weight shapes (confirmed from state_dict)

| Key | Shape |
|-----|-------|
| `talker.model.layers.{i}.self_attn.q_proj.weight` | [2048, 1024] |
| `talker.model.layers.{i}.self_attn.k_proj.weight` | [1024, 1024] |
| `talker.model.layers.{i}.self_attn.v_proj.weight` | [1024, 1024] |
| `talker.model.layers.{i}.self_attn.o_proj.weight` | [1024, 2048] |
| `talker.model.layers.{i}.self_attn.q_norm.weight` | [128] |
| `talker.model.layers.{i}.self_attn.k_norm.weight` | [128] |
| `talker.model.layers.{i}.mlp.gate_proj.weight` | [3072, 1024] |
| `talker.model.layers.{i}.mlp.up_proj.weight` | [3072, 1024] |
| `talker.model.layers.{i}.mlp.down_proj.weight` | [1024, 3072] |
| `talker.model.layers.{i}.input_layernorm.weight` | [1024] |
| `talker.model.layers.{i}.post_attention_layernorm.weight` | [1024] |
| `talker.model.codec_embedding.weight` | [3072, 1024] — input embed |
| `talker.model.norm.weight` | [1024] — final norm |
| `talker.codec_head.weight` | [3072, 1024] — output logits, **NOT tied** |
| `talker.model.text_embedding.weight` | [151936, 2048] — text encoder (separate) |
| `talker.text_projection.linear_fc1.weight` | [2048, 2048] |
| `talker.text_projection.linear_fc2.weight` | [1024, 2048] |

### Code predictor

- Path: `model.talker.code_predictor`
- 16 codebook heads (`lm_head.0` through `lm_head.14`, each [2048, 1024])
- `num_code_groups: 16`

---

## Performance (RTX 5090, bfloat16, no flash-attn, "Hello, this is a test.")

| Path | Frames/s | RTF (raw decode) | RTF (streaming) | TTFC | Notes |
|------|----------|-----------------|-----------------|------|-------|
| HF baseline (eager, no graphs) | ~12 | 1.070 | 1.070 | 6338ms | |
| v2 + CUDA graphs (TalkerGraph + PredictorGraph) | ~60 | 0.236 | 0.236 | 142ms | confirmed Session 10 |
| v2 + Megakernel sentinel (V2_MEGAKERNEL=1) | **~95** | **0.126** | **0.158** | **120ms** | confirmed Session 10 |

Eager timing breakdown (Stage 3b):
- Code predictor (15 steps): 42ms/frame
- Talker backbone (28L, 1 step): 16ms/frame
- Total eager: ~58ms/frame → 17 frames/s → RTF 0.73

---

## Megakernel (qwen_megakernel)

### Build

- No `setup.py` or `pyproject.toml` — `pip install -e .` **fails** (non-fatal)
- Real build trigger: `from qwen_megakernel.build import get_extension; get_extension()`
- Uses `torch.utils.cpp_extension.load` JIT compilation on first call
- JIT cache: `/root/.cache/torch_extensions/py314_cu128/qwen_megakernel_C/`
- Compiled for: sm_120a (RTX 5090 Blackwell only)
- `setup_server.sh` now uses `python build.py || true` instead of `pip install -e .`

### Ops (confirmed from server — Session 3)

```
Registered ops: ['decode', 'name']
```

**`generate_nosync` is NOT present** in the built extension. Only `decode` exists.
`generate_n()` in `tts_backend_mk.py` falls back to calling `step()` in a loop.

```python
# Single decode step — writes output token into pre-allocated tensor
# Arg order inferred from torch_bindings.cpp — NOT yet verified on GPU with real weights
# Run scripts/test_mk_decode.py stage 2 to confirm schema
torch.ops.qwen_megakernel_C.decode(
    output_token,         # int32 tensor [1] — written in-place
    input_token_id,       # int
    embed_weight,         # [VOCAB_SIZE=3072, HIDDEN=1024] bfloat16
    layer_weights_packed, # packed pointer buffer (uint8 on CUDA)
    final_norm_weight,    # [HIDDEN=1024] bfloat16
    lm_head_weight,       # [VOCAB_SIZE=3072, HIDDEN=1024] bfloat16
    cos_table,            # [MAX_SEQ_LEN, HEAD_DIM=128] bfloat16
    sin_table,            # same shape
    k_cache,              # [NUM_LAYERS=28, NUM_KV_HEADS=8, MAX_SEQ_LEN, HEAD_DIM=128] bfloat16
    v_cache,              # same shape
    hidden_buffer,        # [HIDDEN=1024] bfloat16
    activations,          # [HIDDEN=1024] bfloat16
    residual,             # [HIDDEN=1024] bfloat16
    q,                    # [NUM_Q_HEADS=16 * HEAD_DIM=128 = 2048] bfloat16
    k,                    # [NUM_KV_HEADS=8 * HEAD_DIM=128 = 1024] bfloat16
    v,                    # same as k
    attn_out,             # [2048] bfloat16
    mlp_intermediate,     # [VOCAB_SIZE*2 = 6144] bfloat16  ← gate+up proj each 3072
    normalized,           # [HIDDEN=1024] bfloat16
    block_max_vals,       # [8] float32  (LDG_ATTN_BLOCKS=8)
    block_max_idxs,       # [8] int32
    num_layers,           # int = 28
    position,             # int (plain integer, NOT a tensor)
    max_seq_len,          # int = 32768
    attn_scale,           # float = HEAD_DIM**-0.5 ≈ 0.0884
) -> ()
```

**⚠️ `mlp_intermediate` buffer correction (Session 3):** Was incorrectly sized at
`HIDDEN_SIZE*2=2048`. Correct size is `VOCAB_SIZE*2=6144` — gate_proj and up_proj
each project to the intermediate dimension (3072), not HIDDEN_SIZE.

### Compatibility with TTS talker

| Parameter | Kernel | TTS talker | Match? |
|-----------|--------|------------|--------|
| NUM_Q_HEADS | 16 | 16 | ✅ |
| NUM_KV_HEADS | 8 | 8 | ✅ |
| HIDDEN_SIZE | 1024 | 1024 | ✅ |
| HEAD_DIM | 128 | 128 | ✅ |
| INTERMEDIATE_SIZE | 3072 | 3072 | ✅ |
| LDG_VOCAB_SIZE | ~~151936~~ → **3072** | 3072 | ✅ patched |
| rope_theta | 10000 → **1,000,000** | 1,000,000 | ✅ fixed Python-side |
| RoPE type | ~~MRope~~ → **standard 1D** | standard 1D | ✅ fixed Python-side |
| MAX_SEQ_LEN | 2048 → **32768** | 32768 | ✅ fixed Python-side |

### RoPE — confirmed standard 1D (Session 4)

**Was wrong in Sessions 1-3:** We assumed the talker uses interleaved MRope (`mrope_section=[24,20,20]`) based on the model config. This was incorrect.

**Confirmed from attention layer spy:** `position_ids` shape is `[1, seq_len]` — standard sequential `[0,1,...,N-1]`. NOT `[3, 1, seq_len]`. The talker uses `Qwen3TTSTalkerRotaryEmbedding` with standard 1D RoPE.

**Fix:** Replaced `_build_mrope_tables()` with `_build_rope_tables()`. `inv_freq` is copied directly from `model.rotary_emb.inv_freq` to guarantee bit-for-bit match with HF prefill.

### DynamicCache API — confirmed (Session 4)

This version of transformers uses a different DynamicCache API than documented:

```python
# WRONG (older transformers):
pkv.key_cache[i]    # AttributeError
pkv.value_cache[i]  # AttributeError

# CORRECT (installed version):
pkv.layers[i].keys    # [1, NUM_KV_HEADS, seq_len, HEAD_DIM]
pkv.layers[i].values  # [1, NUM_KV_HEADS, seq_len, HEAD_DIM]
len(pkv.layers)       # 28
```

### talker.generate() call signature — confirmed (Session 4)

```python
# generate() receives:
inputs_embeds=[1, 13, 1024]  # NOT input_ids — always embeddings
past_key_values=None          # no KV cache passed in
# + trailing_text_hidden, tts_pad_embed, and other custom kwargs
```

Every decode step also uses `inputs_embeds=[1,1,1024]` — never `input_ids`. HF does codec_embedding lookup internally before calling `talker.model.forward()`. The embedding at decode steps is a blend (not a pure codec lookup) — cosine similarity to any single codec token ≈ 0.48 max.

### Integration approach — current (Session 4)

Patch `talker.model.forward()`:
1. **Prefill** (first call): run HF normally → capture `DynamicCache` via `.layers[i].keys/values` → transfer to megakernel k/v cache buffers → compute first decode token from `last_hidden_state[-1]` via manual RMSNorm + `lm_head_weight` argmax
2. **Decode steps**: ignore `inputs_embeds` entirely → feed internally-tracked token to megakernel `step()` → return `SimpleNamespace(logits, past_key_values=None, hidden_states, last_hidden_state, attentions=None)` → HF embeds the one-hot argmax token for next step (we ignore it)

### Root cause found and fixed — Session 5

All garbage token issues traced to **two wrong buffer sizes**:

**Bug 1: `block_max_vals` / `block_max_idxs` size = 8 (wrong) → 1184 (correct)**

`ldg_lm_head_fused` iterates `block_max_vals[0..num_blocks-1]` where `num_blocks = LDG_LM_NUM_BLOCKS = 1184`. We allocated 8 entries (guessed from `LDG_ATTN_BLOCKS`). Reads at indices 8-1183 returned garbage floats interpreted as token indices.

**Bug 2: All scratch buffers (`g_q`, `g_k`, `g_v`, `g_attn_out`, `g_activations`, `g_residual`, `g_mlp_intermediate`, `g_normalized`) must be `float32`, not `bfloat16`.**

Confirmed from `launch_ldg_decode_direct` cast: `(float*)g_activations`, `(float*)g_q` etc. Only `hidden_buffer` is `bfloat16`. Allocating as `bfloat16` gave half the byte size, causing out-of-bounds writes.

**Verification:** Position sweep `pos=0..19` with zero KV cache — all return valid token 505 after fix. Previously had garbage at most positions.

**Buffer sizes — final confirmed values:**

| Buffer | dtype | size |
|--------|-------|------|
| `hidden_buffer` | bfloat16 | HIDDEN_SIZE=1024 |
| `g_activations` | float32 | HIDDEN_SIZE=1024 |
| `g_residual` | float32 | HIDDEN_SIZE=1024 |
| `g_q` | float32 | NUM_Q_HEADS×HEAD_DIM=2048 |
| `g_k` | float32 | NUM_KV_HEADS×HEAD_DIM=1024 |
| `g_v` | float32 | NUM_KV_HEADS×HEAD_DIM=1024 |
| `g_attn_out` | float32 | NUM_Q_HEADS×HEAD_DIM=2048 |
| `g_mlp_intermediate` | float32 | VOCAB_SIZE=3072 |
| `g_normalized` | float32 | HIDDEN_SIZE=1024 |
| `block_max_vals` | float32 | LDG_LM_NUM_BLOCKS=1184 |
| `block_max_idxs` | int32 | LDG_LM_NUM_BLOCKS=1184 |
| `k_cache` | bfloat16 | [28, 8, MAX_SEQ_LEN, 128] |
| `v_cache` | bfloat16 | [28, 8, MAX_SEQ_LEN, 128] |

---

## Environment

| Item | Value |
|------|-------|
| GPU | RTX 5090 (Vast.ai) |
| CUDA | 12.8 |
| Python | 3.14 |
| venv | `/workspace/qwen-megakernel-pipecat/.venv/` |
| qwen-tts | latest (pip install -U) |
| flash-attn | NOT installed |
| HF auth | unauthenticated (set HF_TOKEN to avoid rate limits) |
| Megakernel JIT cache | `/root/.cache/torch_extensions/py314_cu128/qwen_megakernel_C/` |
