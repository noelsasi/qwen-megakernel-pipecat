# Findings & Observations

> Ground truth discovered by running actual code. All values here are confirmed, not assumed.
> Updated as each investigation step completes.

---

## Model Package

| Finding | Value | Source |
|---------|-------|--------|
| pip package | `qwen-tts` (`pip install -U qwen-tts`) | Trial and error — NOT in transformers |
| **High-level class** | `Qwen3TTSModel` from `qwen_tts` | Official repo — use this |
| Low-level class | `Qwen3TTSForConditionalGeneration` from `qwen_tts.core.models` | Do NOT call directly |
| No separate processor | Tokenization is internal to `Qwen3TTSModel` | Confirmed from source |
| `model_type` in config | `qwen3_tts` | `config.json` |
| HF model ID | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Model card |
| **Sample rate** | **24000 Hz** | Confirmed from baseline output — model card "12Hz" refers to codec frame rate, not audio sample rate |

---

## Architecture (Confirmed from config.json + model inspection)

### Talker (the autoregressive LLM decoder)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_hidden_layers` | 28 | Transformer blocks in talker |
| `num_attention_heads` | 16 | NOT 32 — megakernel default wrong |
| `num_key_value_heads` | 8 | GQA ratio 2:1 |
| `hidden_size` | 1024 | Confirmed from `lm_head` weight shape [2048, 1024] |
| `intermediate_size` | 3072 | FFN expansion |
| `vocab_size` | **3072** | Codec tokens (NOT text tokens) — megakernel default 151936 is wrong |
| `max_position_embeddings` | 32768 | Megakernel default 2048 is too small |
| `rope_theta` | 1,000,000 | Non-standard — megakernel must match |
| `rms_norm_eps` | 1e-6 | Standard |
| `use_sliding_window` | false | |

### Text Encoder (separate from talker)

| Parameter | Value |
|-----------|-------|
| `text_hidden_size` | 2048 |
| `text_vocab_size` | 151936 |

### Code Predictor (`model.talker.code_predictor`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Module path | `model.talker.code_predictor` | Confirmed from named_modules |
| `num_code_groups` | 16 | 16 parallel codebook heads |
| `lm_head` count | 15 visible (0–14) | Each weight: [2048, 1024] |
| Output | 16 codebook tokens per talker frame | One per codebook group |

### Positional Encoding — CRITICAL RISK

| Parameter | Value | Risk |
|-----------|-------|------|
| `rope_scaling.type` | `default` | |
| `rope_scaling.interleaved` | **true** | Interleaved MRope — non-standard |
| `mrope_section` | `[24, 20, 20]` | Multimodal RoPE sections |
| `position_id_per_seconds` | 13 | ~13 codec frames/sec → ~77ms/frame |

**MRope is the biggest Phase D risk.** The megakernel almost certainly has standard RoPE hardcoded. Interleaved MRope applies different rotation frequencies to different head dimension slices. If the megakernel ignores this, output will be garbage even if everything else matches.

---

## generate() API (Corrected — use high-level wrapper)

**Do NOT call `model.generate()` directly.** Use the high-level wrapper methods:

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda", dtype=torch.bfloat16)

# CustomVoice (our model variant):
wavs, sr = model.generate_custom_voice(
    text="Hello world",      # str or list[str] for batch
    language="English",      # str or list[str]
    speaker="Ryan",          # str or list[str] — must be in supported list
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
)
# wavs: list[np.ndarray], sr: int (12000)
audio = wavs[0]  # first batch item
```

**Valid speakers:** Ryan, Aiden (EN), Vivian, Serena, Uncle_Fu, Dylan, Eric (ZH), Ono_Anna (JA), Sohee (KO)

**Valid languages:** English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**Lessons from failed attempts:**
- `speakers=["default"]` → `NotImplementedError: Speaker default not implemented`
- `speakers=[None]` → crashes at `input_id[:, :3]` (1D vs 2D tensor mismatch inside model)
- Calling `model.generate()` directly with `input_ids` as list of 1D tensors → IndexError
- The internal `generate()` expects 2D input_ids — but this is abstracted by `generate_custom_voice()`

**Streaming:** `generate_custom_voice()` is blocking — returns full audio. Real streaming requires hooking the internal `generate()` loop. Deferred to Phase B.

---

## Module Hierarchy (Key paths)

```
model
└── talker
    ├── model              (the transformer body)
    │   ├── embed_tokens
    │   └── layers.0–27    (28 transformer blocks)
    └── code_predictor
        └── lm_head.0–14   (16 codebook heads, weights [2048, 1024])
```

---

## Megakernel Compatibility Matrix

Source: `qwen_megakernel/csrc/kernel.cu` (confirmed from source) vs model config.

| Parameter | Megakernel (kernel.cu) | Model actual | Match? | Action |
|-----------|----------------------|--------------|--------|--------|
| `NUM_Q_HEADS` | 16 | 16 | ✅ | — |
| `NUM_KV_HEADS` | 8 | 8 | ✅ | — |
| `HIDDEN_SIZE` | 1024 | 1024 | ✅ | — |
| `INTERMEDIATE_SIZE` | 3072 | 3072 | ✅ | — |
| `LDG_RMS_EPS` | 1e-6 | 1e-6 | ✅ | — |
| `HEAD_DIM` | 128 | **128** (confirmed from weight shapes) | ✅ | — |
| `Q_SIZE` | 16×128=2048 | 2048 (q_proj out dim confirmed) | ✅ | — |
| `KV_SIZE` | 8×128=1024 | 1024 (k_proj out dim confirmed) | ✅ | — |
| `LDG_VOCAB_SIZE` | **151936** | **3072** | ❌ | Change to 3072 |
| NUM_LAYERS | 28 (from model.py) | 28 | ✅ | — |
| rope_theta | 10000 (from model.py) | **1,000,000** | ❌ | Change in Python table builder |
| RoPE type | standard RoPE | **interleaved MRope** | ❌ | Python-side fix only |
| MAX_SEQ_LEN | 2048 | 32768 | ❌ | Increase in model.py constant |

> Earlier HEAD_DIM=64 estimate was wrong. Confirmed from weight shapes:  
> `q_proj [2048, 1024]` → 16 heads × 128 = 2048. HEAD_DIM=128 all along.  
> `mrope_section [24,20,20]` sums to 64 = HEAD_DIM//2 — consistent with Qwen3-VL pattern.

**Confirmed mismatches (reduced from earlier estimate):**
1. `LDG_VOCAB_SIZE 151936 → 3072` — one constant change in kernel.cu
2. `rope_theta 10000 → 1,000,000` — change in Python `load_weights()` in model.py
3. `MAX_SEQ_LEN 2048 → 32768` — change in model.py constant (affects KV cache allocation)
4. **Interleaved MRope** — replace `build_rope_tables()` in Python; kernel inner loop unchanged

**RoPE situation — better than feared:**
- Kernel RoPE reads from precomputed `cos_pos` / `sin_pos` pointer args — theta is NOT hardcoded
- Inner loop is standard half-dimension rotation: `(i < HEAD_DIM/2) ? cos*x - sin*y : sin*y + cos*x`
- **MRope does NOT require rewriting the kernel's inner loop**
- MRope only requires computing the right `cos_pos`/`sin_pos` tables before calling the kernel
- For interleaved MRope with sections [24, 20, 20]: different frequencies apply to dims [0:24], [24:44], [44:64]
- This means we compute MRope cos/sin in Python/PyTorch before each decode step and pass them in
- The kernel inner loop stays unchanged — only the cos/sin table generation changes

**Revised assessment: MRope is a Python-side fix, not a CUDA rewrite.** Risk drops from CRITICAL to MEDIUM.

**Confirmed cos/sin table mechanics (from kernel.cu + model.py):**
- Tables are `[MAX_SEQ_LEN, HEAD_DIM]` bfloat16 on GPU
- Built once at init: `freqs = torch.outer(positions, inv_freq)`, then `cos(freqs).repeat(1,2)` and `sin(freqs).repeat(1,2)`
- `.repeat(1,2)` duplicates the half-dim freqs to fill full HEAD_DIM (standard RoPE pattern)
- Kernel indexes: `cos_pos = cos_table + position * HEAD_DIM` — single integer position, not multi-dim
- **MRope fix:** Replace `torch.outer(positions, inv_freq)` with a concatenation of three frequency bands for mrope_sections [24,20,20], each with the same theta but different position_id streams (time, text, audio)
- This is purely Python — the CUDA kernel is unchanged

**Weight key mapping (confirmed from state_dict inspection):**

| Megakernel expects | HF state_dict key | Shape |
|-------------------|-------------------|-------|
| `model.layers.{i}.input_layernorm.weight` | `talker.model.layers.{i}.input_layernorm.weight` | [1024] |
| `model.layers.{i}.self_attn.q_proj.weight` | `talker.model.layers.{i}.self_attn.q_proj.weight` | [2048, 1024] |
| `model.layers.{i}.self_attn.k_proj.weight` | `talker.model.layers.{i}.self_attn.k_proj.weight` | [1024, 1024] |
| `model.layers.{i}.self_attn.v_proj.weight` | `talker.model.layers.{i}.self_attn.v_proj.weight` | [1024, 1024] |
| `model.layers.{i}.self_attn.q_norm.weight` | `talker.model.layers.{i}.self_attn.q_norm.weight` | [128] |
| `model.layers.{i}.self_attn.k_norm.weight` | `talker.model.layers.{i}.self_attn.k_norm.weight` | [128] |
| `model.layers.{i}.self_attn.o_proj.weight` | `talker.model.layers.{i}.self_attn.o_proj.weight` | [1024, 2048] |
| `model.layers.{i}.post_attention_layernorm.weight` | `talker.model.layers.{i}.post_attention_layernorm.weight` | [1024] |
| `model.layers.{i}.mlp.gate_proj.weight` | `talker.model.layers.{i}.mlp.gate_proj.weight` | [3072, 1024] |
| `model.layers.{i}.mlp.up_proj.weight` | `talker.model.layers.{i}.mlp.up_proj.weight` | [3072, 1024] |
| `model.layers.{i}.mlp.down_proj.weight` | `talker.model.layers.{i}.mlp.down_proj.weight` | [1024, 3072] |
| `embed_weight` | `talker.model.codec_embedding.weight` | [3072, 1024] |
| `final_norm_weight` | `talker.model.norm.weight` | [1024] |
| `lm_head_weight` | `talker.codec_head.weight` | [3072, 1024] — **NOT tied** |

**LM head is NOT tied:** megakernel text model used `lm_head_weight = embed_weight` (tied embeddings). TTS talker has a separate `codec_head` — must pass it explicitly.

**o_proj shape:** [1024, 2048] — non-square (in=2048=16heads×128, out=1024). Verify megakernel kernel handles this correctly (it may assume square or [out, in] layout).

**Summary:** 5 confirmed mismatches, 1 critical (MRope). Simple `#define` changes fix 4 of them. MRope requires actual kernel code changes.

---

## Environment

| Item | Value |
|------|-------|
| GPU | RTX 5090 (Vast.ai) |
| CUDA | 12.8+ |
| Python | 3.14 |
| `qwen-tts` version | latest (pip install -U) |
| flash-attn | NOT installed (performance only, not blocking) |
| HF auth | unauthenticated (set HF_TOKEN to avoid rate limits) |

---

## Baseline Performance (Phase A.4 — RTX 5090, no megakernel)

| Metric | Value | Notes |
|--------|-------|-------|
| Model load time | ~5800 ms | Cold load, bfloat16 |
| Generation time | 8582 ± 853 ms | 5 trials, ~9.7s audio |
| Audio duration | ~9760 ms | Default test sentence |
| Sample rate | 24000 Hz | Confirmed — "12Hz" = codec frame rate |
| **RTF** | **0.879** | Near real-time, NOT faster than real-time |
| Target RTF < 0.15 | **FAIL** | Needs ~6× speedup from megakernel |
| EOS token ID | **2150** | From pad_token_id warning |
| flash-attn | NOT installed | Expected to give meaningful speedup |

**The 6× speedup gap is what the megakernel must close.** RTF 0.879 → 0.15 is a 5.9× improvement needed.

---

## Phase D Integration — Critical Blocker (2026-05-14)

### The Prefill Incompatibility Problem

The megakernel's `decode` op signature:
```
decode(output_token, input_token_id, embed_weight, ...)
```
It takes an **integer token ID** and does its own embedding lookup via `codec_embedding`.

The HF talker's prefill constructs `inputs_embeds` — a **mixed float tensor** combining:
- Text token projections via `text_projection` MLP
- Codec special token embeddings (codec_think_id, codec_bos_id, language_id, etc.)
- Speaker embeddings (if provided)
- Padding embeddings
- `trailing_text_hiddens` (text continuation)

These are concatenated and passed to `self.talker.generate(inputs_embeds=...)`. The megakernel has no way to accept this mixed embedding — it only understands integer codec token IDs.

**Consequence:** The megakernel cannot replace the prefill. The prefill must always run via HF.

### What the Megakernel CAN Replace

The decode loop inside `self.talker.generate()` autoregressively generates codec tokens one at a time. After the prefill KV cache is established, each decode step takes the previous codec token ID and produces the next one — **this** is the hot loop. But:

1. HF's `generate()` method runs prefill + decode as one atomic call — no exposed hook to intercept between them
2. The talker's `generate()` is called with `inputs_embeds` not `input_ids` — HF generate doesn't support splitting prefill/decode when `inputs_embeds` is used
3. `trailing_text_hidden` is a custom kwarg unique to this model — further complicates any monkey-patching

### Viable Paths Forward

**Option A — Monkey-patch talker.model.forward():**
Replace the inner transformer forward pass with a megakernel call after the KV cache is populated. Requires:
- Intercepting after the first forward pass (prefill)
- Reading the KV cache from HF tensors and writing them to megakernel's cache format
- Running subsequent steps via megakernel decode op
- Risk: KV cache format may differ (head ordering, layout)

**Option B — Replace only the transformer layers with custom CUDA kernels:**
Keep HF generate() but replace the attention+MLP compute with megakernel ops. Requires hooking into each layer's forward() — very complex.

**Option C — Pure HF with flash-attn (pragmatic):**
Skip megakernel decode loop integration. Install flash-attn, benchmark improvement. Document why full megakernel integration is blocked by the prefill embedding incompatibility. This is honest and shippable.

**Option D — Rewrite generate() from scratch:**
Implement the full prefill + decode loop manually in Python, handling the mixed embeddings for prefill then switching to megakernel for decode. Very high effort but technically correct.

**Recommended path:** Option A (monkey-patch) + Option C as fallback if A fails. Document everything honestly.

## Open Questions

1. **EOS token** — confirmed as 2150 from `pad_token_id` warning ✅
2. **Sample rate** — confirmed 24000 Hz ✅ ("12Hz" in model name = codec frame rate)
3. **KV cache format** — does HF talker's past_key_values layout match megakernel's [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]? (needed for Option A)
4. **flash-attn impact** — how much does installing flash-attn improve RTF on RTX 5090?
