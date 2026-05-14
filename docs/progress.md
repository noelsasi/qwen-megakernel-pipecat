# Progress Log

> Updated as each phase step completes. Most recent entry at top.

---

## 2026-05-14 — Session 1 Complete

### Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Environment setup | ✅ | RTX 5090, Python 3.14, venv, qwen-tts installed |
| Phase A — Baseline inference | ✅ | WAV output working, RTF measured |
| Phase D — Megakernel build | ✅ | Kernel patched, JIT built, single decode step confirmed |
| Phase D — Full integration | ❌ Blocked | Prefill/vocoder boundary (see findings.md) |
| Phase C — Pipecat pipeline | ⏳ Not started | Next priority |
| Phase B — Real streaming | ⏳ Not started | After Phase C |

### Key Numbers

| Metric | Value |
|--------|-------|
| Baseline RTF (HF, no megakernel, no flash-attn) | **0.879** |
| Target RTF | < 0.15 |
| Speedup needed | ~6× |
| Audio sample rate | 24000 Hz |
| EOS token ID | 2150 |
| Model load time | ~5800 ms |
| GPU | RTX 5090, CUDA 12.8 |

### Phase D Blocker

The megakernel single decode step works (`step(0) → token 112` confirmed). Full loop integration is blocked by:

1. **Prefill incompatibility** — HF talker prefill uses `inputs_embeds` (mixed text + codec + speaker float tensors). Megakernel only accepts integer token IDs. No clean handoff point.
2. **No public vocoder API** — `Qwen3TTSModel.generate_custom_voice()` buries the speech tokenizer (vocoder) call internally. No exposed method to run vocoder on custom codec token sequences.

Both require deep reverse-engineering of `qwen_tts` internals. Deferred to next session.

---

---

## 2026-05-14 — Session 2

### Changes

| File | What changed |
|------|-------------|
| `server/pipeline/voice_agent.py` | Added `OpenAILLMContext` + system prompt; proper `context_aggregator` pipeline wiring |
| `server/pipecat_services/qwen_tts_service.py` | Added TTFC/RTF/E2E metrics emission after each utterance; timing instrumentation |
| `server/backend/tts_backend_mk.py` | **Phase D unblocked**: replaced speculative `forward_sub_talker` approach with `talker.model.generate()` monkey-patch — megakernel runs the decode loop, HF downstream (code_predictor + vocoder) runs unchanged |
| `scripts/benchmark.py` | Fixed `--trials` wiring through to `measure_ttfc` / `measure_rtf` |

### Phase D Integration Strategy (new)

Instead of calling `forward_sub_talker` (which doesn't exist in the public API),
we now patch `talker.model.generate()` at call time:
1. HF prefill runs normally via `talker.model(inputs_embeds=..., use_cache=True)`
2. HF then calls `talker.model.generate()` for the decode loop — we intercept this
3. Our patch copies the HF `DynamicCache` into megakernel tensors, runs megakernel
   decode until EOS, returns a `[1, N]` token tensor
4. HF receives that tensor and runs code_predictor + vocoder as normal
5. Auto-fallback to full HF if megakernel fails (safety net for debugging)

The patch is thread-safe per-call (install → run → restore in try/finally).

---

## 2026-05-14 — Session 3

### Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Environment setup | ✅ | Megakernel built, ops confirmed: `decode` + `name` only |
| Phase A — Baseline inference | ✅ | RTF 0.879 confirmed |
| Phase D — Megakernel build | ✅ | `decode` op loads. `generate_nosync` NOT present in this build |
| Phase D — Full integration | ⚠️ Written, untested | Monkey-patch approach coded; not yet run end-to-end on GPU |
| Phase C — Pipecat pipeline | ⚠️ Written, untested | `voice_agent.py` complete; not yet started on server |
| Phase B — Real streaming | ❌ Not implemented | Fake streaming only — full audio buffered before chunking |

### Confirmed from server output

- `pip install -e .` fails on megakernel (no `setup.py` / `pyproject.toml`) — **non-fatal**, `get_extension()` JIT-builds the `.so` directly
- Registered ops: `['decode', 'name']` — **`generate_nosync` does NOT exist** in this build
- Setup script `set -euo pipefail` was causing exit on the `pip install -e .` error — fixed to `|| true`

### Fixes shipped (Session 3)

| File | Fix |
|------|-----|
| `server/backend/tts_backend_mk.py` | Removed `generate_nosync` reference (crashes on init). `generate_n()` now calls `step()` in a loop |
| `server/backend/tts_backend_mk.py` | Fixed `_mlp_intermediate` buffer: was `HIDDEN_SIZE*2=2048`, must be `VOCAB_SIZE*2=6144` (gate + up proj each 3072) |
| `scripts/setup_server.sh` | Replaced `pip install -e .` with `python build.py \|\| true` — no-op if already built |
| `scripts/test_mk_decode.py` | **New**: staged smoke test — op schema, single decode step with zero weights, full end-to-end with WAV output |

### Known gaps vs assignment requirements

1. **Fake streaming** — `synthesize_streaming()` buffers full audio then chunks it. Assignment requires token-by-token push. The monkey-patch approach makes true streaming hard: the patched `generate()` must return a complete token tensor to HF before the vocoder runs. True streaming would require intercepting the vocoder too.

2. **End-to-end never run** — The full `_run_with_megakernel_decode` → `generate_custom_voice` path has not been executed on GPU. Still needs verification.

3. **No performance numbers** — README table is blank. Need to run `scripts/benchmark.py` and fill in real values.

4. **No demo recording** — Required deliverable. Needs the pipeline running end-to-end first.

### Next session: immediate priorities (in order)

1. `git pull && python scripts/test_mk_decode.py` — confirm decode op signature, single step, end-to-end
2. If stage 2 fails with arg error: paste the schema line + error, fix arg order
3. If stage 3 fails with `AttributeError` on `DynamicCache`: paste traceback, fix KV cache extraction
4. Once `test_mk_decode.py` passes: `uvicorn server.pipeline.voice_agent:app` — check for import errors
5. Connect React client, confirm STT→LLM→TTS→audio round-trip works
6. `python scripts/benchmark.py --backend both --trials 5` — get real numbers
7. Record demo, fill README table
