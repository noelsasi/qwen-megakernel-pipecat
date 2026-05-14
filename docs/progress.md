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

## Pending

- [ ] **Phase A.4** — Run `phase_a_baseline.py`, get WAV output + RTF measurement
- [ ] **Phase B** — Probe streaming: does `non_streaming_mode=False` yield chunks or full audio?
- [ ] **Phase C** — Pipecat pipeline end-to-end (STT → LLM → TTS → speaker)
- [ ] **Phase D** — Megakernel: clone repo, run compat check, resolve MRope risk
- [ ] flash-attn install (optional performance improvement for baseline)
- [ ] HF_TOKEN set on server (currently unauthenticated — hitting rate limits)
