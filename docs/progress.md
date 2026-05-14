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

---

## 2026-05-14 — Session 4

### Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase D — Megakernel decode firing | ✅ Prefill works | KV cache transfers, first token extracted correctly |
| Phase D — Full decode loop | ❌ Blocked | Garbage tokens at pos>0 — KV RoPE compatibility issue |
| Phase C — Pipecat pipeline | ⏳ Not started | Blocked on Phase D |

### What was discovered and fixed

| Finding | Impact |
|---------|--------|
| `talker.generate()` receives `inputs_embeds`, no `past_key_values` | Needed to patch `talker.model.forward()` instead |
| All forward calls use `inputs_embeds`, never `input_ids` | Can't recover token from embedding (cosine sim ~0.48) |
| `DynamicCache` uses `.layers[i].keys/.values` API | Previous `.key_cache/.value_cache` was silently wrong |
| Talker uses **standard 1D RoPE**, NOT interleaved MRope | MRope tables replaced with standard RoPE matching HF inv_freq |
| `BaseModelOutputWithPast` has no `.logits` | First token computed via manual RMSNorm + lm_head argmax on last_hidden_state |
| HF generate loop accesses `.attentions` on forward output | Added `attentions=None` to SimpleNamespace |

### Current blocker — KV cache RoPE format

**Symptom:** `decode(token, pos=0)` → valid token ✅. `decode(token, pos=18)` with real HF prefill KV cache → garbage token ❌

**Root cause hypothesis:** HF stores keys post-RoPE in DynamicCache. The megakernel also expects to attend over post-RoPE keys — but the rotation format may differ. Specifically: HF's `apply_rotary_pos_emb` uses the standard complex rotation `(x1*cos - x2*sin, x1*sin + x2*cos)`, while the kernel may use a different layout (interleaved pairs vs split half).

**Next step:** Check whether HF stores pre-RoPE or post-RoPE keys, and confirm the rotation format matches the kernel's attention code.

```bash
grep -n "k_cache\|past_key\|rotary\|apply_rot" \
  .venv/lib/python3.14/site-packages/qwen_tts/core/models/modeling_qwen3_tts.py \
  | grep "1[0-9][0-9][0-9]:" | head -30
```

### All commits squashed

All session 3+4 fix commits were squashed into one clean commit (`272afde` + `9f1b320`) before pushing. Git history is clean.
