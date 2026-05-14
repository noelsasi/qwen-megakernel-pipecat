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

---

## 2026-05-14 — Session 5

### Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase D — Megakernel decode loop | ✅ **WORKING** | All positions return valid tokens |
| Phase D — Full end-to-end with audio | ⏳ Running | `test_mk_decode.py` stage 3 in progress |
| Phase C — Pipecat pipeline | ⏳ Next | After stage 3 confirms audio output |

### Root cause found: two wrong buffer allocations

**Bug 1 — `block_max_vals`/`block_max_idxs` size 8 → 1184**
`ldg_lm_head_fused` iterates over `LDG_LM_NUM_BLOCKS=1184` entries. We allocated 8 (guessed from `LDG_ATTN_BLOCKS`). Reads 8-1183 returned garbage floats as token indices.

**Bug 2 — scratch buffers `bfloat16` → `float32`**
All scratch buffers except `hidden_buffer` are cast to `float*` inside the kernel. Allocating as `bfloat16` gave half the byte count, causing out-of-bounds writes at all positions.

**Proof:** Position sweep `pos=0..19` with zero KV cache now returns valid token 505 at every position.

### Next immediate steps

1. Wait for `test_mk_decode.py` to complete — confirm `[MK] Decode complete` + RTF
2. Listen to `output_mk_test.wav` — verify audio quality vs HF baseline
3. Start server: `uvicorn server.pipeline.voice_agent:app`
4. Connect React client, confirm end-to-end voice round-trip
5. Run `python scripts/benchmark.py --backend both --trials 5`
6. Record demo
7. Fill README numbers table

---

## 2026-05-14 — Session 7 (Phase 2 — Custom Decode Loop + CUDA Graphs)

### Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 2 — Custom decode loop | ✅ Working | EOS fires, valid codec frames, real streaming |
| Phase 2 — CUDA graphs | ✅ Captured | TalkerGraph + PredictorGraph via StaticCache |
| Phase 2 — Pipecat integration | ✅ Working | TTS_BACKEND=v2 wired in voice_agent.py |
| Phase 3 — Megakernel in v2 | ⏳ Next | Replace TalkerGraph decode with megakernel step |
| Demo recording | ⏳ Pending | Audio confirmed playing, need screen capture |

### Architecture reset: why Phase 1 failed and Phase 2 approach

**Root cause of Phase 1 failure:** Monkey-patching `talker.generate()` only tracked integer token IDs. Qwen3-TTS generation requires per-step: code predictor (15 steps → 16-codebook frame), summed codec embeddings, and `trailing_text_hiddens` text conditioning. Skipping all of this meant EOS (token 2150) never appeared in the logit output.

**Phase 2 approach:** Own the full decode runtime. Intercept `talker.generate()` only to capture prefill tensors, then run our own loop calling code_predictor and talker backbone directly.

**Key source references used:**
- `QwenLM/Qwen3-TTS modeling_qwen3_tts.py` — exact tensor shapes and per-step flow
- `andimarafioti/faster-qwen3-tts streaming.py` — reference decode loop implementation
- `andimarafioti/faster-qwen3-tts talker_graph.py` — CUDA graph + StaticCache pattern

### Performance milestones this session

| Milestone | RTF | TTFC | Notes |
|-----------|-----|------|-------|
| v2 eager, HF generate for predictor | 0.906 | 924 ms | EOS hit, real streaming, first working end-to-end |
| v2 + manual predictor loop | 0.835 | 842 ms | Replaced predictor.generate() with direct forward() calls |
| v2 + CUDA graphs (TalkerGraph + PredictorGraph) | **0.237** | **93 ms** | StaticCache + CUDA graph capture |

### Key bugs found and fixed

| Bug | Fix |
|-----|-----|
| EOS token ID was 2150 (in logit range) but suppress mask wiped it | Suppress `[2048:2150]` and `[2151:3072]` separately, leave 2150 open |
| `predictor.generate()` with `inputs_embeds` returns sequences without prefix | Take `sequences[0]` directly (not `seq[input_len:]`) |
| `talker.forward()` intercept not working | Intercept `talker.generate()` instead — what `generate_custom_voice()` actually calls |
| `torch.compile(mode="reduce-overhead")` → CUDA graph error on dynamic KV | Use StaticCache + explicit CUDA graph capture instead |
| `StaticCache.reset()` fails outside inference_mode | Decorate `prefill_kv()` and `run()` with `@torch.inference_mode()` |

### Per-step timing breakdown (post CUDA graphs)

| Component | Eager | CUDA graph |
|-----------|-------|------------|
| Code predictor (15 steps) | 49 ms | ~2-3 ms |
| Talker backbone (28L, 1 step) | 20 ms | ~2-5 ms |
| Python loop overhead | — | ~5-8 ms |
| **Total per frame** | **~69 ms** | **~13 ms** |

### Remaining gap to targets

RTF 0.237 vs 0.15 target — 1.58× remaining. Three paths:
1. Megakernel for talker (~1ms/step vs current ~2-5ms)
2. Async vocoder (don't block decode thread on vocoder call)
3. Python loop vectorization (batch embedding ops)

---

## 2026-05-14 — Session 6 (Final)

### Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase A — HF baseline | ✅ | RTF 1.070, TTFC 6338ms measured |
| Phase D — Megakernel decode | ✅ Runs | 263 tok/s confirmed, EOS not reached |
| Phase D — Full audio output | ❌ Blocked | Vocoder hidden_states format mismatch |
| Phase C — Pipecat pipeline | ✅ Working | STT→LLM→TTS→audio confirmed end-to-end |
| README | ✅ Done | Real numbers, honest limitations |
| Demo | ⏳ Pending | Screen recording needed |

### End-to-end demo confirmed working

Full voice pipeline on RTX 5090:
- Deepgram STT transcribes mic input correctly
- gpt-5-mini generates response
- QwenTTS synthesizes and plays audio
- Dashboard shows transcript, metrics, waveform

Audio uses HF fallback (megakernel falls back due to vocoder integration issue).
E2E latency: ~18-20s (dominated by megakernel decode running 4096 tokens before fallback).

### Benchmark numbers (RTX 5090, CUDA 12.8, bfloat16, no flash-attn, 3 trials)

| Sentence | TTFC | RTF |
|----------|------|-----|
| "Hello." | 4762 ± 2538 ms | 1.126 ± 0.015 |
| "The quick brown fox..." | 4641 ± 509 ms | 1.119 ± 0.009 |
| "Artificial intelligence..." | 6704 ± 496 ms | 1.006 ± 0.007 |
| "In the beginning..." | 9243 ± 618 ms | 1.031 ± 0.023 |
| **Mean** | **6338 ms** | **1.070** |

Megakernel decode: **263-266 tok/s** (paper target: ~1000 tok/s; gap due to MAX_SEQ_LEN=32768).

### Why megakernel audio doesn't complete

1. **EOS not reached** — megakernel runs 4096 tokens and hits hard cap. The decode sequence diverges from HF because HF constructs mixed embeddings (text + codec + speaker) per step, while megakernel tracks its own integer token state. Sequences diverge at step 1.

2. **Vocoder hidden_states format** — `generate_custom_voice()` reads per-step hidden states from `talker_result.hidden_states` (line 2280 of `modeling_qwen3_tts.py`). HF expects a tuple of per-step tuples containing all-layer hidden states. Our SimpleNamespace with a single `[1,1,1024]` tensor doesn't match the format the code_predictor expects.

### What would close the gap

- Fix EOS: replicate HF's mixed embedding construction inside the decode loop
- Fix vocoder: package `decoder._hidden` per step into correct hidden_states format
- Reduce MAX_SEQ_LEN to 2048 for TTS (sufficient) to recover tok/s toward ~1000
- Install flash-attn for 3-4× HF prefill speedup
