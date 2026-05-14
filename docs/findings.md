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
| `HEAD_DIM` | **128** | **64** (=1024/16) | ❌ | Change to 64; ripples into Q_SIZE, KV_SIZE |
| `LDG_VOCAB_SIZE` | **151936** | **3072** | ❌ | Change to 3072 |
| `Q_SIZE` | 16×128=2048 | 16×64=1024 | ❌ | Derived — fixed when HEAD_DIM fixed |
| `KV_SIZE` | 8×128=1024 | 8×64=512 | ❌ | Derived — fixed when HEAD_DIM fixed |
| NUM_LAYERS | not found yet | 28 | ? | Verify in source |
| rope_theta | not found yet | 1,000,000 | ? | Verify in source |
| RoPE type | **standard RoPE only** | **interleaved MRope** | ❌ | Critical — needs new impl |

> `HEAD_DIM = HIDDEN_SIZE / NUM_Q_HEADS = 1024 / 16 = 64`  
> The kernel was written for Qwen3-0.6B text model (HEAD_DIM=128, VOCAB=151936).  
> The TTS talker has the same layer structure but different HEAD_DIM and vocab.

**Confirmed mismatches requiring kernel changes:**
1. `HEAD_DIM 128 → 64` — all attention buffer sizing must be recalculated
2. `LDG_VOCAB_SIZE 151936 → 3072` — LM head output projection size
3. `Q_SIZE / KV_SIZE` — derived, auto-fixed when HEAD_DIM changes
4. **Interleaved MRope** — standard RoPE in kernel must be replaced (see RoPE section below)

**RoPE situation (critical):**
- Kernel has standard RoPE only (grep confirms: no `mrope`, no `interleave`, no `mrope_section`)
- Talker uses interleaved MRope with sections [24, 20, 20] and theta=1,000,000
- MRope applies different rotation frequencies to different slices of each head dimension
- This is not a constant change — requires rewriting the RoPE block in the kernel

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

## Open Questions

1. **EOS token** — confirmed as 2150 from `pad_token_id` warning ✅
2. **Sample rate** — confirmed 24000 Hz ✅ ("12Hz" in model name = codec frame rate)
3. **Streaming internals** — `generate_custom_voice()` is blocking; need to hook internal `generate()` to get per-chunk audio for real TTFC measurement (Phase B)
4. **MRope in megakernel** — does `kernel.cu` have any MRope code, or is it purely standard RoPE? (Phase D.1)
5. **Weight key names** — what keys does `qwen_megakernel/model.py` expect vs `model.model.state_dict()` keys? (Phase D.2)
6. **flash-attn impact** — how much does installing flash-attn improve RTF on RTX 5090? Worth measuring before megakernel.
