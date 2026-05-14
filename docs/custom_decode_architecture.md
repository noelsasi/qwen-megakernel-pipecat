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
  ├─ [MEGAKERNEL STEP] — replace this forward call with megakernel.step()
  │   BUT: megakernel takes integer token_id, not inputs_embeds
  │
  │   *** THE KEY PATCH (from jayanth-kumar-morem/qwen-megakernel-tts) ***
  │   Modify kernel.cu line:
  │     const __nv_bfloat16 *embed_row =
  │       (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
  │                             : hidden_buffer;   // <-- use pre-filled buffer
  │   When token_id == -1, kernel reads embed from hidden_buffer instead.
  │   Python writes inputs_embeds [1024] into hidden_buffer before calling step(-1).
  │
  ├─ hidden_states = megakernel.step(-1, prefilled_embed=inputs_embeds)
  │   OR: talker.model.forward(inputs_embeds=inputs_embeds, past_kv=...) [HF fallback]
  │   output: hidden_buffer [1024] = last layer hidden state
  │
  ├─ logits = lm_head_weight @ RMSNorm(hidden_buffer)  [3072]
  ├─ sample next CB0 token
  ├─ past_hidden = hidden_buffer.view(1, 1, 1024).clone()
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

## 4. Megakernel Integration Strategy

### Option A — Kernel Patch (jayanth-kumar-morem approach, ~50ms TTFC)

Patch `kernel.cu` to accept sentinel `token_id = -1`:
```cuda
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;   // hidden_buffer pre-filled by Python
```

Python side before each `step(-1)`:
```python
# Write the [1024] bfloat16 inputs_embeds into decoder._hidden
decoder._hidden.copy_(inputs_embeds.squeeze())
torch.ops.qwen_megakernel_C.decode(
    output_token,
    -1,            # sentinel: use hidden_buffer as embedding
    *args
)
```

Requires kernel rebuild. The `hidden_buffer` the kernel reads for `token_id == -1`
is the same `hidden_buffer` argument the kernel also writes its output into.
So: write inputs_embeds into `hidden_buffer` → kernel reads it → runs all 28 layers → writes new hidden into `hidden_buffer`.

### Option B — HF Talker Forward (no kernel change, safer)

Use `talker.model.forward(inputs_embeds=..., past_key_values=...)` directly.
This is slower than megakernel but gives correct results and allows incremental vocoding.
Then benchmark and compare vs megakernel path.

### Recommended approach: Build Option B first (correct), then add Option A (fast).

---

## 5. Prefill Construction

From `generate_custom_voice()` source (QwenLM/Qwen3-TTS):

```python
# Special tokens:
tts_bos_token_id = config.tts_bos_token_id
tts_eos_token_id = config.tts_eos_token_id  
tts_pad_token_id = config.tts_pad_token_id

# Compute bos/eos/pad embeddings:
tts_bos_embed, tts_eos_embed, tts_pad_embed = text_projection(
    text_embeddings(tensor([[bos_id, eos_id, pad_id]]))
).chunk(3, dim=1)   # each [1, 1, 1024]

# trailing_text_hiddens = text conditioning for decode steps:
trailing_text_hidden = cat(
    text_projection(text_embeddings(input_id[:, 4:-5])),  # text tokens
    tts_eos_embed
)   # shape: [1, text_len-9+1, 1024]

# prefill input = bos + codec_embedding(speaker_tokens) + trailing_text:
prefill_embeds = cat([tts_bos_embed, speaker_codec_embeds, trailing_text_hidden, ...], dim=1)
```

---

## 6. Vocoder Streaming

The `speech_tokenizer` (Qwen3-TTS-Tokenizer-12Hz) uses a causal ConvNet decoder.
It can decode incrementally with a left-context window.

Per-chunk decode:
```python
CHUNK_FRAMES = 12   # 12 codec frames = 960ms audio at 12.5Hz
CONTEXT_FRAMES = 25  # 25-frame left context for codec ConvNet

# After accumulating chunk_frames new frames:
window = all_codes[max(0, n_total - CONTEXT_FRAMES):n_total]  # [ctx+chunk, 16]
audio_out = speech_tokenizer.decode({"audio_codes": window.unsqueeze(0)})
# Trim to only new samples: audio_out[-CHUNK_FRAMES * samples_per_frame:]
samples_per_frame = SAMPLE_RATE // 12  # = 24000 // 12 = 2000 samples = 80ms
new_audio = audio_out[-(CHUNK_FRAMES * samples_per_frame):]
yield new_audio.tobytes(), SAMPLE_RATE
```

First chunk TTFC:
- Prefill: ~20-50ms
- 1 megakernel decode step: ~1ms  
- Code predictor (5-layer, 15 steps): ~5ms
- Vocoder decode first chunk: ~10ms
- **Realistic TTFC with megakernel: ~35-70ms** (vs 6338ms current)

---

## 7. Implementation Order

1. `server/backend/tts_backend_v2.py` — Option B (HF talker forward, custom loop, real streaming)
   - `_build_prefill_embeds()` — construct tts_bos/pad/trailing correctly
   - `_decode_loop()` — owns the loop, calls code_predictor, builds codec_hiddens
   - `_vocode_chunk()` — incremental decode with left context
   - `synthesize_streaming()` — yields audio chunks as they arrive

2. Validate: does it produce audio? Does EOS fire? Compare vs HF baseline.

3. `server/backend/tts_backend_v3.py` — Option A (megakernel + kernel patch)
   - Patch `kernel.cu` with sentinel -1 support
   - Replace `talker.model.forward()` calls with `mk_decoder.step(-1, embed=inputs_embeds)`

4. Benchmark: TTFC, RTF, tok/s on RTX 5090.

5. Wire v2/v3 into Pipecat via existing `QwenTTSService` — it already accepts any backend.
