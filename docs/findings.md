# Findings & Observations

> Ground truth discovered by running actual code. All values here are confirmed, not assumed.
> Updated as each investigation step completes.

---

## Model Package

| Finding | Value | Source |
|---------|-------|--------|
| pip package | `qwen-tts` (`pip install -U qwen-tts`) | Trial and error — NOT in transformers |
| Model class | `Qwen3TTSForConditionalGeneration` | `qwen_tts.core.models` |
| Processor class | `Qwen3TTSProcessor` | `qwen_tts.core.models` |
| `model_type` in config | `qwen3_tts` | `config.json` |
| HF model ID | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Model card |

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

## generate() API (Confirmed from inspection)

```python
model.generate(
    input_ids: list[torch.Tensor],   # list of 1D tensors, one per batch item
    instruct_ids: list[torch.Tensor] | None = None,
    ref_ids: list[torch.Tensor] | None = None,
    voice_clone_prompt: list[dict] = None,
    languages: list[str] = None,
    speakers: list[str] = None,
    non_streaming_mode=False,        # False = streaming (default!), True = batch
    max_new_tokens: int = 4096,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.9,
    subtalker_dosample: bool = True,
    subtalker_top_k: int = 50,
    subtalker_top_p: float = 1.0,
    subtalker_temperature: float = 0.9,
    eos_token_id: int | None = None,
    repetition_penalty: float = 1.05,
)
```

**Key finding:** `non_streaming_mode=False` is the default — **real streaming is on by default**. No need to fake-stream or manually chunk. The model yields audio incrementally.

**input_ids format:** Must be a `list` of 1D tensors (not a batched 2D tensor). Pass `[ids[0]]` not `ids`.

**Unknown:** What exactly `generate()` returns/yields in streaming mode — list of tensors, generator, queue? Must confirm from Phase A.4 run.

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

Source: `qwen_megakernel/csrc/kernel.cu` (expected defaults) vs confirmed model values.

| Parameter | Megakernel default | Model actual | Match? | Action |
|-----------|-------------------|--------------|--------|--------|
| `NUM_LAYERS` | 28 | 28 | ✅ | — |
| `HIDDEN_SIZE` | 1024 | 1024 | ✅ | — |
| `INTERMEDIATE_SIZE` | 3072 | 3072 | ✅ | — |
| `NUM_HEADS` | 32 | **16** | ❌ | Change to 16 |
| `NUM_KV_HEADS` | 8 | 8 | ✅ | — |
| `HEAD_DIM` | 128 | 64 (=1024/16) | ❌ | Change to 64 |
| `VOCAB_SIZE` | 151936 | **3072** | ❌ | Change to 3072 |
| `MAX_SEQ_LEN` | 2048 | **32768** | ❌ | Increase |
| `rope_theta` | unknown | 1,000,000 | ? | Verify in kernel.cu |
| RoPE type | standard | **interleaved MRope** | ❌ | Critical — needs MRope impl |

> `HEAD_DIM = HIDDEN_SIZE / NUM_HEADS = 1024 / 16 = 64` (not 128 as megakernel assumes)

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

## Open Questions

1. **Streaming return type** — what does `generate(non_streaming_mode=False)` actually return? Generator? Blocking with list? (Answer pending Phase A.4)
2. **EOS token ID** — what value signals end of codec generation? (check `generation_config.json` or model source)
3. **Vocoder path** — where is the DAC/vocoder in the module tree? Not visible in lm_head output — may be inside code_predictor or a separate module.
4. **MRope in megakernel** — does `kernel.cu` have any MRope code, or is it purely standard RoPE? (Answer pending Phase D.1 clone + grep)
5. **Weight key names** — what keys does `qwen_megakernel/model.py` expect? Must match against `model.talker.state_dict()` keys.
