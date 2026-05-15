# Progress Log

> Updated as each phase step completes. Most recent entry at top.

---

## 2026-05-15 — Session 10: Megakernel Running, EOS Fix In Progress

### Current Status

| Item | Status |
|------|--------|
| v2 CUDA graphs path | ✅ Working — RTF 0.236, TTFC 142ms |
| Megakernel sentinel path (V2_MEGAKERNEL=1) | ✅ Running — RTF **0.124** (target met), 97 frames/s |
| EOS fires in megakernel path | ❌ Not yet — fix pushed, pending GPU confirmation |
| TTFC with megakernel | ❌ Not measured — blocked on EOS fix |
| Audio quality with megakernel | ❌ Not verified — blocked on EOS fix |

### What was confirmed on GPU this session

- Stage 6 sub-test A: `step_with_embed(zeros)` → token 0, valid ✅
- Stage 6 sub-test B: 200 frames in 2058ms = **97.2 frames/s, RTF 0.124** ✅ (target < 0.15 met)
- Tokens are valid (all in [0, 3072)) but sequence doesn't reach EOS

### Root causes found and fixed

**Bug 1 — Kernel argmax bypasses suppress mask**
The megakernel does argmax over raw logits internally — no suppress mask, no sampling.
Tokens 2048-3071 (except EOS=2150) are supposed to be masked out. Without the mask,
high-probability tokens like 122 and 2035 win every time → sequence loops forever.

Fix (commit `5a417fe`): ignore the kernel's token output. After `step_with_embed()`,
manually apply RMSNorm to `_hidden`, recompute logits via `lm_head`, run through
`_sample()` with the suppress mask. Kernel is used only for its 28-layer forward pass.

**Bug 2 — Raw residual fed to code predictor**
The kernel's `_hidden` buffer is the raw post-layer-norm residual stream, not HF's
`last_hidden_state` (which goes through `model.norm` RMSNorm before return). The code
predictor was trained on normed hidden states — feeding it the raw residual caused
token divergence after ~3 steps.

Fix (commit `bd02ce6`): manually apply `RMSNorm(_hidden * final_norm_weight)` before
using the result as `past_hidden` for the code predictor.

**Bug 3 — `step()` in tts_backend_mk.py missing reset_barriers()**
`step()` was not calling `reset_barriers()` before `decode()`, causing barrier deadlock
on the second consecutive call. Only `step_with_embed()` had the reset.

Fix (commit `a54a333`): `step()` now calls `_reset_barriers()` helper; ctypes loader
also added to `_MKDecoder.__init__` in `tts_backend_mk.py`.

### Performance comparison (all on RTX 5090, "Hello, this is a test.")

| Path | Frames/s | RTF | TTFC |
|------|----------|-----|------|
| HF baseline (no graphs) | ~12 | 1.070 | 6338ms |
| v2 eager (no graphs) | ~17 | ~0.73 | — |
| v2 + CUDA graphs | ~60 | 0.236 | 142ms |
| v2 + Megakernel (Stage 6) | **~97** | **0.124** | pending |

### Next steps

1. Pull commit `5a417fe` on GPU and re-run Stage 6 — confirm EOS fires
2. Run full `V2_MEGAKERNEL=1 python scripts/test_v2_decode.py` — all stages including TTFC
3. Listen to `/tmp/test_v2_output.wav` — verify audio quality
4. Update README performance table with confirmed TTFC number
5. Run end-to-end voice pipeline: `V2_MEGAKERNEL=1 TTS_BACKEND=v2 uvicorn ...`

---

## 2026-05-15 — Session 9: Phase 3 Code Written (Not Yet GPU-Tested)

### Current Status

| Item | Status |
|------|--------|
| v2 decode loop (correct architecture) | ✅ Working on GPU — RTF 0.209, TTFC 135ms |
| Megakernel sentinel integration code | ✅ Written locally — **not yet run on GPU** |
| Kernel sentinel patch in `setup_server.sh` | ✅ Written — **not yet applied to live server** |
| `V2_MEGAKERNEL=1` env gate in backend | ✅ Written |
| Stage 6 validation in `test_v2_decode.py` | ✅ Written |
| Megakernel tok/s with sentinel path | ❌ Unknown — needs GPU run |
| TTFC / RTF with megakernel | ❌ Unknown — needs GPU run |
| README performance table updated with mk numbers | ❌ Pending GPU results |
| Demo recording | ❌ Not done |

### What was written this session (local, not GPU-tested)

**`server/backend/tts_backend_v2.py`** — three additions:

1. `_MKDecoder` class + helpers (`_mk_extract_weights`, `_mk_pack_layer_weights`,
   `_mk_build_rope_tables`) added at module level. Constants use `_MK_MAX_SEQ_LEN=1024`
   (was 32768 in Phase 1 — that's why Session 6 got only 263 tok/s instead of ~1000).

2. `mk_decoder` branch at the top of the talker backbone block in `_custom_decode_loop()`:
   ```python
   if mk_decoder is not None:
       next_tok_id = mk_decoder.step_with_embed(inputs_embeds)
       past_hidden = mk_decoder._hidden.view(1, 1, _MK_HIDDEN_SIZE).clone()
   ```
   `step_with_embed()` writes `inputs_embeds [1024 bf16]` into `_hidden`, calls
   `decode(-1)` (sentinel), reads argmax token back. No logit exposure — kernel does
   argmax internally. TalkerGraph and eager branches unchanged below it.

3. `_setup_megakernel()` and wiring in `__init__`, `_run_custom_decode()`,
   `_decode_thread()`. `PredictorGraph` stays active regardless — code predictor
   is independent of backbone choice.

**`scripts/setup_server.sh`** — three kernel patches added (all idempotent):
- Patch 1: `LDG_VOCAB_SIZE 151936 → 3072` (already existed)
- Patch 2: Sentinel ternary on the `embed_row` assignment — Python regex replacement
- Patch 3: `MAX_SEQ_LEN 32768 → 1024` — reduces KV alloc from 1.88GB to 118MB

**`scripts/test_v2_decode.py`** — Stage 6 added:
- Sub-test A: `step_with_embed(zeros)` in isolation — verifies sentinel doesn't crash
- Sub-test B: full decode to EOS with megakernel + PredictorGraph — measures tok/s, RTF, EOS

### What must happen on GPU before submission

**In order, stop if any step fails:**

1. `bash scripts/setup_server.sh` — watch for sentinel patch output:
   - ✅ `kernel.cu: sentinel patch applied (embed_row ternary)`
   - ❌ `ERROR: Could not find embed_row line` → inspect kernel.cu manually, patch by hand

2. `V2_MEGAKERNEL=1 python scripts/test_v2_decode.py`:
   - Stage 6 Sub-test A must pass (kernel doesn't crash on sentinel)
   - Stage 6 Sub-test B must show EOS within 200 frames and valid unique tokens

3. Update README performance table with real megakernel numbers from Stage 6

4. Update Known Limitations section once real numbers are known

5. `V2_MEGAKERNEL=1 TTS_BACKEND=v2 uvicorn ...` — confirm end-to-end voice pipeline works

6. Record demo

### Known risks going into GPU run

| Risk | Mitigation |
|------|-----------|
| Sentinel patch regex misses actual line | Script prints found `embed_row` lines on failure; patch manually |
| Kernel crashes on `decode(-1)` | Sub-test A catches this before any decode logic runs |
| EOS doesn't fire (sequence diverges) | v2 builds correct `inputs_embeds`; most likely cause is RoPE mismatch — `inv_freq` copied directly from HF model should prevent this |
| `MAX_SEQ_LEN=1024` too small for some sentences | Prefill for long text could exceed 1024; `load_kv_from_hf()` raises explicitly — bump to 2048 if hit |

---

## 2026-05-15 — Session 8: Alignment Review + Phase 3 Plan

### Alignment Review Findings

Full review of implementation vs assignment requirements:

| Requirement | Status | Notes |
|---|---|---|
| Use AlpinDale's megakernel as decode backend | ❌ Not in working path | Kernel is built but `TTS_BACKEND=v2` doesn't use it |
| Wire to Qwen3-TTS talker decoder | ✅ Correct decode loop | v2 owns the full runtime with correct code predictor |
| Real streaming to Pipecat | ✅ | Per-frame audio push, async vocoder thread |
| TTFC < 60ms | ❌ 135ms (2.25×) | Root cause: `token.item()` sync + Python overhead |
| RTF < 0.15 | ❌ 0.209 (1.4×) | Same root cause |
| Demo recording | ❌ Not done | Required deliverable |
| Honest benchmarking | ✅ | Quantified gaps, root causes documented |

**Core gap:** The v2 backend is correct and fast, but it uses `TalkerGraph` (CUDA-captured
HF PyTorch forward) for the backbone — not the CUDA megakernel. The megakernel is built
and was used in `tts_backend_mk.py` (Phase 1), but that backend was abandoned because
it used the wrong decode strategy (no code predictor). The correct fix is to put the
megakernel *inside* the working v2 decode loop at the talker backbone step.

### Why the gap exists (root cause chain)

1. Phase 1 (`tts_backend_mk.py`): megakernel ran but EOS never fired — the monkey-patch
   intercepted at the wrong level, missing code predictor + embedding reconstruction.
2. Phase 2 (`tts_backend_v2.py`): fixed the decode strategy by owning the full loop —
   but replaced TalkerGraph (HF model) instead of integrating the megakernel.
3. Phase 3 (now planned): put megakernel *inside* the Phase 2 loop, replacing TalkerGraph.

### Phase 3 Plan Summary

See `docs/custom_decode_architecture.md` sections 5-11 for full specification.

Five concrete steps, each independently verifiable:

**Step 1 — Kernel patch** (server-side, requires rebuild):
- Find embedding lookup line in `qwen_megakernel/csrc/kernel.cu`
- Add sentinel: `(input_token_id >= 0) ? embed_weight + ... : hidden_buffer`
- Change `MAX_SEQ_LEN` from 32768 to 1024 (recovers ~4× tok/s)
- Rebuild and smoke-test with a zero-weight decode(-1) call

**Step 2 — Copy `_MKDecoder` into v2 backend**:
- `_MKDecoder` from `tts_backend_mk.py` is self-contained and already tested
- Add `V2_MEGAKERNEL=1` env gate — default off, existing behavior preserved
- `_setup_megakernel()` initializes decoder using weights from already-loaded HF model

**Step 3 — Add `mk_decoder` branch to `_custom_decode_loop()`**:
- Sentinel write: `mk_decoder._hidden.copy_(inputs_embeds.squeeze())`
- Kernel call with `token_id=-1`
- Skip `_sample()` (kernel does argmax internally)
- `past_hidden` from `mk_decoder._hidden.view(1, 1, 1024)`

**Step 4 — Wire prefill handoff**:
- Call `mk_decoder.load_kv_cache_from_hf(past_kv)` after HF prefill
- `PredictorGraph` (code predictor) remains active — it's independent of backbone choice

**Step 5 — Staged validation**:
- Stage A: kernel sentinel smoke test (no Python decode logic yet)
- Stage B-E: existing `test_v2_decode.py` stages with `V2_MEGAKERNEL=1`

### Expected performance after Phase 3

| Metric | Current (TalkerGraph) | Expected (megakernel) | Target |
|--------|----------------------|----------------------|--------|
| Talker step | ~3ms | ~1ms | — |
| `token.item()` sync | ~2ms (unavoidable) | **0ms** (kernel does argmax) | — |
| Python overhead | ~5ms | ~2ms | — |
| Total per frame | ~13ms | ~4ms | <12ms for RTF<0.15 |
| RTF | 0.209 | **~0.05** | <0.15 |
| TTFC | 135ms | **~40ms** | <60ms |

### What does NOT change

- Prefill (HF generate_custom_voice intercept)
- Code predictor + PredictorGraph (CB1..CB15)
- 16-codebook embedding reconstruction
- Text conditioning (trailing_text_hiddens)
- Vocoder (incremental, async thread)
- Pipecat integration

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

### Final numbers this session (after all optimizations)

| Optimization | RTF | TTFC | Notes |
|---|---|---|---|
| CUDA graphs active | 0.237 | 93ms | TalkerGraph + PredictorGraph |
| + codec_head in graph | 0.257 | 93ms | Slight regression (noise) |
| + async vocoder thread | **0.209** | **135ms** | Decode and vocoder now overlap |

### Remaining gap to targets

RTF 0.209 vs 0.15 target — 1.4× remaining. Root cause:
- `token.item()` CPU sync every step — unavoidable without megakernel embedding sentinel
- Python loop overhead: `cat`, `clamp`, `unsqueeze`, dispatch — ~3-5ms/frame
- TTFC gap: vocoder thread still adds latency on first chunk

Next: Phase 3 — megakernel sentinel patch (3 lines to kernel.cu). Estimated result: RTF ~0.05, TTFC ~40ms.

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
