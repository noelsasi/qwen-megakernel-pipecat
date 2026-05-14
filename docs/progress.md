# Progress Log

> Updated as each phase step completes. Most recent entry at top.

---

## 2026-05-14

### Phase A.3 — Model Inspection ✅

- `qwen-tts` pip package installs and imports correctly on Vast.ai
- `phase_a_inspect_model.py` ran successfully and produced full module hierarchy
- All critical architecture values confirmed (see findings.md)
- `generate()` signature confirmed — real streaming supported via `non_streaming_mode=False`

### Phase A.2 — Package Discovery ✅

- `qwen3_tts` is NOT part of HuggingFace `transformers` (any released version or source HEAD)
- It is a standalone pip package: `pip install qwen-tts`
- Class: `Qwen3TTSForConditionalGeneration` from `qwen_tts.core.models`
- Processor: `Qwen3TTSProcessor` from `qwen_tts.core.models`

### Environment ✅

- Vast.ai instance: RTX 5090 (confirmed running)
- Python 3.14, venv at `.venv/`
- `qwen-tts` installed and importing correctly
- flash-attn not installed (warning only — not blocking)

### Scaffold ✅

- All scripts, server, and client files created and pushed to GitHub
- `git remote`: `git@github.com:noelsasi/qwen-megakernel-pipecat.git`

---

### Phase A.4 — Baseline Inference ✅

- `generate_custom_voice()` works correctly with speaker="Ryan", language="English"
- WAV saved: `output_baseline.wav`, sr=24000 Hz (model returns 24000, not 12000 as assumed)
- EOS token confirmed: 2150 (from pad_token_id warning)

**Baseline numbers (RTX 5090, bfloat16, no flash-attn, no megakernel):**

| Metric | Value |
|--------|-------|
| Generation time | 8582 ± 853 ms |
| Audio duration | ~9760 ms |
| Sample rate | 24000 Hz |
| **RTF** | **0.879** |
| Target RTF < 0.15 | **FAIL** (5.9× too slow) |

RTF 0.879 means the model generates audio almost in real-time but NOT faster — this is the baseline we need to beat with the megakernel. The target (RTF < 0.15) requires ~6× speedup.

---

## Pending

- [ ] **Phase B** — Real streaming: hook `generate()` internals to yield audio chunks before full generation completes (reduces TTFC)
- [ ] **Phase C** — Pipecat pipeline end-to-end (STT → LLM → TTS → speaker)
- [ ] **Phase D** — Megakernel integration
  - [x] Clone megakernel repo
  - [x] Compat matrix complete — HEAD_DIM=128 ✅, NUM_HEADS=16 ✅, only VOCAB_SIZE wrong
  - [x] Weight extraction code written (tts_backend_mk.py)
  - [x] MRope cos/sin table builder written (interleaved, theta=1e6)
  - [x] Patch kernel.cu: `LDG_VOCAB_SIZE 151936 → 3072` ✅
  - [x] JIT build succeeds (`qwen_megakernel_C.so`) ✅
  - [x] Single decode step works: `step(0) → token 112` ✅
  - [ ] **BLOCKER:** Prefill uses mixed `inputs_embeds` (text + codec + speaker) — megakernel only accepts integer token IDs. Cannot directly hand off after HF prefill.
  - [ ] Investigate Option A: monkey-patch talker.model.forward() post-prefill
  - [ ] Fallback: flash-attn baseline + document integration blocker honestly
- [ ] flash-attn install — likely gives meaningful speedup on RTX 5090
- [ ] HF_TOKEN set on server (currently unauthenticated — hitting rate limits)
