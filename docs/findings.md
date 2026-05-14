# Findings & Observations

> Ground truth only — confirmed by running actual code.
> Last updated: 2026-05-14 (Session 1)

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

## Baseline Performance (RTX 5090, bfloat16, no flash-attn)

| Metric | Value |
|--------|-------|
| Model load | ~5800 ms |
| Generation time | 8582 ± 853 ms |
| Audio duration | ~9760 ms |
| **RTF** | **0.879** |
| Target RTF | < 0.15 |
| Gap | 5.9× speedup needed |

---

## Megakernel (qwen_megakernel)

### Build

- No `setup.py` — uses JIT compilation via `torch.utils.cpp_extension.load`
- Build trigger: `from qwen_megakernel.build import get_extension; get_extension()`
- JIT cache: `/root/.cache/torch_extensions/py314_cu128/qwen_megakernel_C/`
- Compiled for: sm_120a (RTX 5090 Blackwell only)

### Ops (confirmed from torch_bindings.cpp)

```python
# Single decode step — writes output token into pre-allocated tensor
torch.ops.qwen_megakernel_C.decode(
    output_token,        # int32 tensor [1] — written in-place
    input_token_id,      # int
    embed_weight,        # [VOCAB, HIDDEN] bfloat16
    layer_weights_packed,# packed struct blob
    final_norm_weight,   # [HIDDEN] bfloat16
    lm_head_weight,      # [VOCAB, HIDDEN] bfloat16
    cos_table,           # [MAX_SEQ_LEN, HEAD_DIM] bfloat16
    sin_table,           # [MAX_SEQ_LEN, HEAD_DIM] bfloat16
    k_cache,             # [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM] bfloat16
    v_cache,             # same shape
    hidden_buffer,       # [HIDDEN] bfloat16
    activations,         # [HIDDEN] bfloat16
    residual,            # [HIDDEN] bfloat16
    q,                   # [NUM_Q_HEADS * HEAD_DIM] bfloat16
    k,                   # [NUM_KV_HEADS * HEAD_DIM] bfloat16
    v,                   # same
    attn_out,            # [NUM_Q_HEADS * HEAD_DIM] bfloat16
    mlp_intermediate,    # [HIDDEN * 2] bfloat16
    normalized,          # [HIDDEN] bfloat16
    block_max_vals,      # [8] float32  (LDG_ATTN_BLOCKS=8)
    block_max_idxs,      # [8] int32
    num_layers,          # int
    position,            # int (plain integer, NOT a tensor)
    max_seq_len,         # int
    attn_scale,          # float
) -> ()

# Multi-step without sync — returns int32 tensor of token IDs
torch.ops.qwen_megakernel_C.generate_nosync(
    first_token_id, num_steps, ...same buffers...,
    num_layers, start_position, max_seq_len, attn_scale
) -> Tensor[num_steps, int32]
```

### Compatibility with TTS talker

| Parameter | Kernel | TTS talker | Match? |
|-----------|--------|------------|--------|
| NUM_Q_HEADS | 16 | 16 | ✅ |
| NUM_KV_HEADS | 8 | 8 | ✅ |
| HIDDEN_SIZE | 1024 | 1024 | ✅ |
| HEAD_DIM | 128 | 128 | ✅ |
| INTERMEDIATE_SIZE | 3072 | 3072 | ✅ |
| LDG_VOCAB_SIZE | ~~151936~~ → **3072** | 3072 | ✅ patched |
| rope_theta | 10000 | 1,000,000 | ❌ Python-side fix |
| RoPE type | standard | interleaved MRope | ❌ Python-side fix |
| MAX_SEQ_LEN | 2048 | 32768 | ❌ Python constant only |

**All kernel.cu mismatches are now fixed.** Remaining fixes are Python-only (cos/sin table builder).

### Single decode step — confirmed working

```
step(0) → token 112  ✅
```

### Integration blocker

The megakernel decode loop cannot be wired to the HF model because:

1. **Prefill uses `inputs_embeds`** — HF talker prefill constructs a mixed float embedding (text projections + codec special tokens + speaker embeddings). The megakernel only accepts integer token IDs and does its own embedding lookup. No clean interface exists between them.

2. **Vocoder has no public API** — The speech tokenizer (vocoder) that converts codec token sequences to audio is called internally inside `Qwen3TTSModel.generate_custom_voice()`. There is no exposed method to run it on a custom codec sequence.

### KV cache layout (confirmed)

HF `DynamicCache`: list of 28 × `(k[1, 8, seq, 128], v[1, 8, seq, 128])`  
Megakernel: `[28, 8, MAX_SEQ_LEN, 128]` pre-allocated  
→ Layouts are compatible. A KV cache transfer is straightforward IF the vocoder blocker is resolved.

---

## MRope Table Builder

Implemented in `server/backend/tts_backend_mk.py: _build_mrope_tables()`.

- Interleaved layout: T₀H₀A₀T₁H₁A₁... (round-robin across 3 position streams)
- During TTS autoregressive decode all 3 streams share the same step index
- Output: `[MAX_SEQ_LEN, HEAD_DIM]` bfloat16, compatible with kernel indexing

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
