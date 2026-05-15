# Custom Decode Architecture: Megakernel-Backed Qwen3-TTS

> Ground truth from source inspection of QwenLM/Qwen3-TTS + andimarafioti/faster-qwen3-tts.
> Every tensor shape verified from actual code. No speculation.

---

## 1. Why the Previous Strategy Failed

HF's `talker.forward()` is NOT a simple "embedding lookup → transformer → logits" loop.
At every decode step it:
1. Calls `code_predictor.generate()` internally to get 15 RVQ codebook tokens
2. Builds a **summed 16-codebook embedding** as the next-step input
3. Adds `trailing_text_hiddens[:, gen_step]` (text conditioning) or `tts_pad_embed`
4. Returns `hidden_states[1]` = `codec_ids` ([1, 16]) — the full codebook frame

Feeding only integer token IDs to the megakernel and ignoring this process means:
- EOS token 2150 is never reached (sequence diverges from step 1)
- The next-step embedding is wrong every step
- The code predictor never runs → no codec frames → vocoder has nothing

---

## 2. Correct Architecture

```
Text
  │
  ▼
[prefill_embeds construction]
  - tts_bos/eos/pad embed via text_projection(text_embeddings(token_id))
  - trailing_text_hiddens via text_projection(text_embeddings(text_token_ids[4:-5]))
  - Input to talker prefill: [1, prefill_len, 1024]
  │
  ▼
[talker.forward() — PREFILL]
  - inputs_embeds: [1, prefill_len, 1024]
  - kwargs: trailing_text_hidden, tts_pad_embed, generation_step=None, past_hidden=None
  - Outputs:
      * past_key_values  → DynamicCache (28 layers)
      * past_hidden      → [1, 1, 1024] (last hidden state)
      * generation_step  → starts at 0
      * logits[:, -1, :] → [1, 3072] → sample first codec token CB0
  │
  ▼
[DECODE LOOP — each step produces one codec frame = 16 codebook tokens]
  │
  ├─ token: [1] scalar (CB0 codec token from previous logits)
  │
  ├─ last_id_hidden = talker.codec_embedding(token.unsqueeze(1))
  │   shape: [1, 1, 1024]
  │
  ├─ pred_input = cat(past_hidden, last_id_hidden, dim=1)
  │   shape: [1, 2, 1024]
  │
  ├─ [code_predictor] ← runs 15-step autoregressive loop internally
  │   input: pred_input [1, 2, 1024]
  │   output: codebook_token_ids [15] (CB1..CB15)
  │
  ├─ all_cb = cat(token.view(1), codebook_token_ids)
  │   shape: [16] ← THIS IS ONE CODEC FRAME → buffer for vocoder
  │
  ├─ Build next-step embedding:
  │   codec_hiddens[0] = last_id_hidden                         [1, 1, 1024]
  │   codec_hiddens[1..15] = predictor.codec_embedding[i](CB_i) [1, 1, 1024] each
  │   inputs_embeds = cat(codec_hiddens, dim=1).sum(1, keepdim=True)
  │   shape: [1, 16, 1024] → sum → [1, 1, 1024]
  │
  ├─ Add text conditioning:
  │   if gen_step < trailing_text_hiddens.shape[1]:
  │       inputs_embeds += trailing_text_hiddens[:, gen_step].unsqueeze(1)
  │   else:
  │       inputs_embeds += tts_pad_embed
  │
  ├─ [TALKER BACKBONE STEP] — currently: TalkerGraph (CUDA graph of HF model.forward)
  │   OR [MEGAKERNEL STEP]  — target: torch.ops.qwen_megakernel_C.decode with sentinel
  │
  │   Key difference:
  │   - HF path: talker_model.forward(inputs_embeds=[1,1,1024], ...) → hidden [1,1,1024]
  │   - MK path: decoder._hidden.copy_(inputs_embeds.squeeze())
  │              torch.ops.qwen_megakernel_C.decode(output_token, -1, ...)
  │              → decoder._hidden now holds new [1024] hidden state
  │
  ├─ hidden_states = talker step output → [1, 1, 1024]
  │
  ├─ logits = lm_head_weight @ RMSNorm(hidden[:, -1])  [3072]
  ├─ sample next CB0 token
  ├─ past_hidden = hidden[:, -1:, :]                   [1, 1, 1024]
  ├─ gen_step += 1
  │
  └─ [CHUNK VOCODING — every chunk_size frames]
      all_codes: [total_frames, 16]
      window = all_codes[max(0, n-25):n]  ← 25-frame left context
      audio_list, sr = speech_tokenizer.decode({"audio_codes": window.unsqueeze(0)})
      trim audio to only emit new_frames * 80ms samples
      yield audio_bytes to Pipecat
```

---

## 3. Key Tensors

| Tensor | Shape | Source | Used by |
|--------|-------|--------|---------|
| `prefill_embeds` | [1, prefill_len, 1024] | text_projection(text_embeddings) | talker prefill |
| `trailing_text_hiddens` | [1, text_len-9, 1024] | text_projection(text_embeddings(input[4:-5])) | decode loop per step |
| `tts_pad_embed` | [1, 1, 1024] | text_projection(text_embeddings(pad_id)) | decode loop after text runs out |
| `past_hidden` | [1, 1, 1024] | talker last hidden state | code predictor input |
| `last_id_hidden` | [1, 1, 1024] | codec_embedding(CB0_token) | code predictor + next embed |
| `pred_input` | [1, 2, 1024] | cat(past_hidden, last_id_hidden) | code predictor |
| `codebook_token_ids` | [15] | code_predictor output | next embed + codec frame |
| `all_cb` | [16] | cat(CB0, codebook_token_ids) | codec frame buffer |
| `inputs_embeds` (next step) | [1, 1, 1024] | sum of 16 codec embeds + text cond | megakernel / talker |
| `hidden_buffer` | [1024] bfloat16 | megakernel output | logits, past_hidden |

---

## 4. Why the Current v2 Loop is Correct (and what it's missing)

`tts_backend_v2.py` implements everything above correctly:
- Prefill via `_build_prefill_inputs_and_run()` captures `trailing_text_hiddens`, `tts_pad_embed`, `past_hidden`, `first_logits`
- `_custom_decode_loop()` calls code predictor, reconstructs 16-codebook embedding, applies text conditioning per step
- EOS fires at token 2150 correctly
- `TalkerGraph` captures `talker.model.forward(inputs_embeds=..., past_kv=...)` as a CUDA graph — ~3ms/step

**What it's missing:** the megakernel is not in the hot path. `TalkerGraph` wraps HF's PyTorch forward, not `torch.ops.qwen_megakernel_C.decode`. The kernel exists, is built by `setup_server.sh`, and is tested in `tts_backend_mk.py` — but `tts_backend_mk.py` was abandoned because it used the wrong decode strategy (no code predictor).

The correct integration point is inside `_custom_decode_loop()` at the talker backbone step (lines ~343-361 in `tts_backend_v2.py`).

---

## 5. Megakernel Integration: The Sentinel Patch

### The problem: megakernel only accepts integer token IDs

The kernel signature (confirmed from `tts_backend_mk.py` docstring):
```python
torch.ops.qwen_megakernel_C.decode(
    output_token,        # [1] int32, written by kernel
    input_token_id,      # int: index into embed_weight table
    embed_weight,        # [vocab, hidden] — embed lookup table
    layer_weights_packed,
    final_norm_weight,
    lm_head_weight,
    cos_table, sin_table,
    k_cache, v_cache,
    hidden_buffer,       # [hidden] bfloat16 — kernel reads AND writes this
    activations, residual, q, k, v, attn_out, mlp_intermediate, normalized,
    block_max_vals, block_max_idxs,
    num_layers, position, max_seq_len, attn_scale
)
```

When `input_token_id >= 0`, the kernel reads embedding from `embed_weight[input_token_id]`.
Our decode loop has `inputs_embeds` — a float tensor [1, 1, 1024] that is NOT a simple embed lookup
(it's the sum of 16 codebook embeddings + text conditioning). No integer ID exists for it.

### The fix: sentinel token_id = -1 reads from hidden_buffer

In `kernel.cu`, the embedding lookup line is approximately:
```cuda
// Current (before patch):
const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

// After patch:
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;   // sentinel: use pre-filled hidden_buffer
```

Then Python writes `inputs_embeds` into `hidden_buffer` before calling `decode(-1)`:
```python
# Before kernel call:
decoder._hidden.copy_(inputs_embeds.view(HIDDEN_SIZE))  # [1024] bfloat16

# Kernel call with sentinel:
torch.ops.qwen_megakernel_C.decode(
    decoder._output_token,
    -1,          # sentinel: read from hidden_buffer instead of embed_weight
    *decoder._call_args(),
    NUM_LAYERS,
    decoder._position,
    MAX_SEQ_LEN,
    decoder._attn_scale,
)
# After: decoder._hidden now holds the NEW hidden state [1024]
# decoder._output_token holds the predicted next token [1] int32
```

**Why hidden_buffer is safe to overwrite:** The kernel reads it at the very start
(first layer input), then overwrites it with the final hidden state at the end.
Since we write before calling, the kernel gets our embedding; after return,
`decoder._hidden` contains the new hidden state ready for the next iteration.

### Consequences for the decode loop

After the megakernel step:
```python
# hidden_buffer = decoder._hidden = [1024] bfloat16 — raw residual (NOT normed)
# output_token  = decoder._output_token = [1] int32 — kernel argmax (DO NOT USE directly)

# Correct extraction for next iteration:
_h = decoder._hidden.float()
_h = _h * torch.rsqrt(_h.pow(2).mean() + 1e-6)      # RMSNorm
_h = (_h * decoder._final_norm_weight.float()).to(torch.bfloat16)
past_hidden = _h.view(1, 1, HIDDEN_SIZE)              # [1, 1, 1024] — normed, for code predictor

# Recompute token with suppress mask:
logits = (decoder._lm_head_weight @ _h).squeeze()    # [vocab_size]
token = _sample(logits, suppress_eos=...)             # same path as HF/TalkerGraph
```

**Why NOT use decoder._output_token:** The kernel's argmax has no suppress mask.
Tokens [2048..3071] except EOS=2150 should be suppressed — without this, high-frequency
tokens win every step and EOS is never reached.

**Important implication (confirmed Session 10):** The megakernel does argmax internally with NO suppress mask.
Accepting the kernel's argmax token directly causes infinite loops — tokens 122, 2035 etc. win every step
and EOS=2150 is never reached.

**Correct approach (implemented):** Ignore the kernel's `output_token`. After `step_with_embed()`:
1. Apply RMSNorm to `_hidden`: `h = hidden * rsqrt(hidden.pow(2).mean() + 1e-6) * final_norm_weight`
2. Recompute logits: `logits = lm_head_weight @ h`
3. Run `_sample(logits, suppress_eos=...)` — same suppress mask as HF path
4. Use sampled token as next CB0; use normed `h` as `past_hidden` for code predictor

The kernel is used only for its 28-layer transformer forward pass (KV cache update + hidden state).
Token selection always goes through Python `_sample()` with the correct suppress mask.

---

## 6. What Changes in `_custom_decode_loop()` (Phase 3)

Current talker backbone section (tts_backend_v2.py lines ~342-362):

```python
# CURRENT (TalkerGraph / HF eager)
if talker_graph is not None:
    hidden, logits = talker_graph.run(inputs_embeds, position=prefill_len + step_idx)
    past_hidden = hidden[:, -1:, :]
    if logits is None:
        logits = codec_head(hidden[:, -1, :])
else:
    backbone_out = talker_model(
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=True, output_hidden_states=False, return_dict=True,
    )
    hidden = backbone_out.last_hidden_state
    past_key_values = backbone_out.past_key_values
    logits = codec_head(hidden[:, -1, :])
    past_hidden = hidden[:, -1:, :].clone()
```

After Phase 3, a third branch replaces the above when megakernel is active:

```python
# TARGET (megakernel sentinel path)
elif mk_decoder is not None:
    # Write inputs_embeds into hidden_buffer — kernel reads it as embedding
    mk_decoder._hidden.copy_(inputs_embeds.view(HIDDEN_SIZE))
    torch.ops.qwen_megakernel_C.decode(
        mk_decoder._output_token,
        -1,                          # sentinel
        *mk_decoder._call_args(),
        NUM_LAYERS,
        mk_decoder._position,
        MAX_SEQ_LEN,
        mk_decoder._attn_scale,
    )
    mk_decoder._position += 1
    # hidden_buffer now holds new hidden state
    past_hidden = mk_decoder._hidden.view(1, 1, HIDDEN_SIZE).clone()
    # output_token is argmax — use directly as next step's token
    # (no logit sampling — megakernel does argmax internally)
    token = mk_decoder._output_token.to(torch.long).squeeze()
    # Skip the _sample() call at the bottom of the loop
    continue  # → next codec frame
```

**Note:** Because the kernel outputs the token directly (not logits), the normal
`token = _sample(logits, ...)` at the bottom of the decode loop must be skipped
for the megakernel path. The loop structure needs a flag or early-continue.

---

## 7. KV Cache: Prefill → Megakernel Handoff

The v2 prefill runs via HF and produces a `DynamicCache` (28 layers of KV tensors).
The megakernel uses its own pre-allocated `k_cache`/`v_cache` flat tensors.
`tts_backend_mk.py` already has `_MKDecoder.load_kv_cache_from_hf()` which does this copy:

```python
def load_kv_cache_from_hf(self, past_key_values) -> int:
    self._k_cache.zero_()
    self._v_cache.zero_()
    for layer_idx, layer in enumerate(past_key_values.layers):
        k = layer.keys   # [1, 8, seq_len, 128]
        v = layer.values
        seq = k.shape[2]
        self._k_cache[layer_idx, :, :seq, :] = k[0].to(torch.bfloat16)
        self._v_cache[layer_idx, :, :seq, :] = v[0].to(torch.bfloat16)
    prefill_len = past_key_values.layers[0].keys.shape[2]
    self._position = prefill_len
    return prefill_len
```

This code is correct and reusable. Phase 3 will call it after the HF prefill instead of
calling `talker_graph.prefill_kv()`.

---

## 8. MAX_SEQ_LEN Impact on tok/s

From Session 6 (progress.md): megakernel achieved **263 tok/s** when `MAX_SEQ_LEN=32768`.
The target is ~1000 tok/s (paper result). The gap is likely because:

The kernel allocates `k_cache` and `v_cache` at `[28, 8, MAX_SEQ_LEN, 128]` bfloat16.
At MAX_SEQ_LEN=32768: `28 × 8 × 32768 × 128 × 2 bytes = 1.88 GB per cache`. This doesn't fit.
At MAX_SEQ_LEN=2048: `28 × 8 × 2048 × 128 × 2 bytes = 118 MB per cache` — fine.

For TTS the talker generates ~40-80 codec frames. Prefill is ~18-50 tokens.
Total sequence length needed: prefill_len + max_frames ≈ 50 + 200 = 250 tokens.
MAX_SEQ_LEN=512 is safe, 1024 is generous. **Use 1024 for Phase 3.**

With MAX_SEQ_LEN=1024, the kernel's attention computation touches only 1024 positions
per head vs 32768 — the bandwidth reduction will materially improve tok/s.

---

## 9. Phase 3 Action Plan

> This is the concrete sequence of changes to make on the GPU server.
> Each step is independently verifiable before proceeding to the next.
> Do NOT run all steps at once — validate each before continuing.

### Step 1 — Patch kernel.cu (on server, requires rebuild)

**File:** `qwen_megakernel/csrc/kernel.cu`

Find the embedding lookup line. It will look like:
```cuda
const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
```

Replace with:
```cuda
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;
```

**Also change MAX_SEQ_LEN** in the kernel constants from 32768 to 1024.
Look for: `#define MAX_SEQ_LEN 32768` or similar constant definition.

Then rebuild:
```bash
cd qwen_megakernel && python build.py
```

**Validate before any Python changes:**
```python
import sys; sys.path.insert(0, 'qwen_megakernel')
from qwen_megakernel.build import get_extension; get_extension()
import torch
# Allocate minimal buffers and call decode with token_id=-1
# Verify no CUDA error and output_token is in [0, 3072)
```

### Step 2 — Add MKDecoder to v2 backend (surgical addition, no existing code deleted)

**File:** `server/backend/tts_backend_v2.py`

Copy `_MKDecoder` class from `tts_backend_mk.py` verbatim. It is self-contained.
Also copy `_extract_talker_weights`, `_pack_layer_weights`, `_build_rope_tables`,
and the constant definitions (`NUM_LAYERS`, `HEAD_DIM`, etc.) — or import them.

In `QwenTTSBackendV2.__init__()`, add megakernel initialization after model load:
```python
self._mk_decoder = None
if os.environ.get("V2_MEGAKERNEL", "0") == "1":
    self._setup_megakernel(model)
```

```python
def _setup_megakernel(self, model):
    try:
        sys.path.insert(0, "./qwen_megakernel")
        from qwen_megakernel.build import get_extension
        get_extension()

        state = model.state_dict()
        weights = _extract_talker_weights(state)
        inv_freq = model.talker.model.rotary_emb.inv_freq.detach().cpu()
        self._mk_decoder = _MKDecoder(weights, inv_freq=inv_freq)
        # Load prefill KV cache will be done per-call — reset here
        logger.info("[v2] Megakernel decoder initialized (MAX_SEQ_LEN=1024)")
    except Exception as e:
        logger.warning(f"[v2] Megakernel init failed ({e}), falling back to TalkerGraph/eager")
        self._mk_decoder = None
```

**Key:** controlled by `V2_MEGAKERNEL=1` env var. Default off — v2 behavior unchanged.

### Step 3 — Patch `_custom_decode_loop()` to accept mk_decoder argument

Add `mk_decoder=None` parameter to `_custom_decode_loop()`.

Inside the loop, replace the talker backbone block:

```python
# --- Talker backbone: single decode step ---
if mk_decoder is not None:
    # Megakernel sentinel path: write inputs_embeds into hidden_buffer
    mk_decoder._hidden.copy_(inputs_embeds.squeeze())   # [1024] bfloat16
    torch.ops.qwen_megakernel_C.decode(
        mk_decoder._output_token,
        -1,
        *mk_decoder._call_args(),
        NUM_LAYERS,
        mk_decoder._position,
        MAX_SEQ_LEN,
        mk_decoder._attn_scale,
    )
    mk_decoder._position += 1
    past_hidden = mk_decoder._hidden.view(1, 1, HIDDEN_SIZE).clone()
    # output_token is argmax — bypass _sample()
    token = mk_decoder._output_token.to(torch.long).squeeze()
    if token.item() == eos_id:
        logger.info(f"[v2/mk] EOS fired at step {step_idx}")
        break
    gen_step += 1
    # Buffer the codec frame and yield chunk
    if len(chunk_buffer) >= chunk_size and on_chunk is not None:
        on_chunk(torch.stack(chunk_buffer)); chunk_buffer = []
    continue   # skip _sample() at bottom

elif talker_graph is not None:
    # ... existing TalkerGraph code unchanged ...
else:
    # ... existing eager code unchanged ...
```

### Step 4 — Wire mk_decoder into synthesize_streaming

In `_decode_thread()` inside `synthesize_streaming()`:
```python
# After prefill_kv call, before _custom_decode_loop:
if self._mk_decoder is not None:
    self._mk_decoder.reset()
    self._mk_decoder.load_kv_cache_from_hf(past_kv)

frames = _custom_decode_loop(
    ...
    talker_graph=self._talker_graph if self._mk_decoder is None else None,
    predictor_graph=self._predictor_graph,  # still used — code predictor unchanged
    mk_decoder=self._mk_decoder,
)
```

Note: `PredictorGraph` (CUDA graph for code predictor) remains active regardless of
whether megakernel or TalkerGraph handles the backbone. They are independent.

### Step 5 — Validate on server (staged, each stage must pass before next)

**Stage A: Kernel smoke test**
```bash
python - <<'EOF'
import sys; sys.path.insert(0, 'qwen_megakernel')
from qwen_megakernel.build import get_extension; get_extension()
import torch
# Create minimal decoder with zero weights
# Call decode(-1) with sentinel, verify output_token in [0, 3072)
print("Kernel sentinel test passed")
EOF
```

**Stage B: Prefill + megakernel step 1**
```bash
V2_MEGAKERNEL=1 python scripts/test_v2_decode.py
```
Look for: Stage 1 PASS (prefill), then first mk step log showing valid token.

**Stage C: EOS fires within 200 frames**
Stage 3 in `test_v2_decode.py` already checks this. EOS should fire at ~40-80 frames.

**Stage D: Audio output quality**
Stage 4 saves `/tmp/test_v2_output.wav`. Download and listen. Should match HF baseline quality.

**Stage E: Performance numbers**
Stage 5 measures TTFC and RTF. Expected with megakernel + PredictorGraph:
- TTFC: ~40-60ms (vs 135ms current)
- RTF: ~0.05-0.10 (vs 0.209 current)

---

## 10. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Kernel sentinel line not found at expected location | Medium | High | grep for `embed_weight` in kernel.cu; location confirmed from AlpinDale source |
| Rebuilt kernel crashes with `token_id=-1` | Low | Medium | Test kernel in isolation (Stage A) before any Python changes |
| KV cache layout mismatch (DynamicCache → MK tensors) | Low | High | `load_kv_cache_from_hf()` from `tts_backend_mk.py` already tested in Session 6 |
| EOS doesn't fire with megakernel (sequence diverges) | Low | High | The decode loop builds inputs_embeds correctly (v2 verified); megakernel just replaces the forward pass, not embedding construction |
| MAX_SEQ_LEN=1024 too small for long sentences | Low | Medium | Set to 2048 to be safe; still 16× smaller than 32768 |
| Output token from kernel always same value (argmax collapse) | Low | Medium | Log first 10 output tokens; compare unique token count vs v2 TalkerGraph path |
| `past_hidden` shape mismatch after megakernel | Low | High | Kernel outputs [1024] flat; reshape to [1, 1, 1024] explicitly with `.view()` |

---

## 11. What Changes, What Doesn't

| Component | Status after Phase 3 |
|-----------|---------------------|
| Prefill (HF `generate_custom_voice` intercept) | **Unchanged** |
| Code predictor (CB1..CB15) | **Unchanged** — PredictorGraph still active |
| Codec embedding reconstruction (16 codebooks summed) | **Unchanged** |
| Text conditioning (trailing_text_hiddens) | **Unchanged** |
| Talker backbone step | **Replaced**: TalkerGraph → megakernel `decode(-1)` |
| Logit computation | **Eliminated**: megakernel does argmax internally |
| Temperature sampling for CB0 | **Lost**: megakernel is argmax-only |
| Vocoder (incremental, async thread) | **Unchanged** |
| Pipecat integration | **Unchanged** |
| `TTS_BACKEND=v2` env var | **Unchanged** — `V2_MEGAKERNEL=1` added as a sub-flag |
