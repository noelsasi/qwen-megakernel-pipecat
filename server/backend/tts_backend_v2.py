"""
Phase 2 — Custom decode loop (Option B): correct architecture, real streaming.

Architecture (verified from QwenLM/Qwen3-TTS + andimarafioti/faster-qwen3-tts):

    Text
      → prefill_embeds (text_projection + trailing_text_hiddens)
      → talker.forward(prefill)                  → past_kv, past_hidden, gen_step=0
      → DECODE LOOP per codec frame:
          CB0_token = sample(logits)
          last_id_hidden = codec_embedding(CB0_token)   [1, 1, 1024]
          pred_input = cat(past_hidden, last_id_hidden) [1, 2, 1024]
          CB1..15 = code_predictor(pred_input)          [15]
          all_cb = cat(CB0, CB1..15)                    [16]  ← codec frame
          inputs_embeds = sum(16 codec embeds) + text_cond  [1, 1, 1024]
          hidden = talker.model.forward(inputs_embeds, past_kv)
          logits = lm_head(RMSNorm(hidden[:, -1]))      [3072]
          past_hidden = hidden[:, -1:, :]               [1, 1, 1024]
          gen_step += 1
      → CHUNK VOCODING every CHUNK_FRAMES frames:
          window = all_codes[max(0, n-CONTEXT_FRAMES):n]
          audio = speech_tokenizer.decode(window)
          yield trimmed_new_audio

DO NOT modify: existing HF pipeline, voice_agent.py, tts_backend_mk.py, tts_backend_hf.py.
This is a new isolated module. Wire it in by setting TTS_BACKEND=v2 in voice_agent.py.
"""

import asyncio
import os
import sys
import time
import logging
from collections.abc import AsyncGenerator

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
CHUNK_FRAMES = 12      # yield audio every 12 codec frames = 960ms of audio
CONTEXT_FRAMES = 25    # left-context window for causal codec decoder
SAMPLES_PER_FRAME = SAMPLE_RATE // 12   # = 2000 samples = 80ms per codec frame
# codec_eos_token_id from the actual model's config.json (Qwen3-TTS-12Hz-0.6B-CustomVoice):
#   talker_config.codec_eos_token_id = 2150
# The Python class default is 4198 but the shipped checkpoint overrides it to 2150.
# 2150 is within [0, 3072) so it's a valid logit index.
# Confirmed from: QwenLM/Qwen3-TTS modeling + faster-qwen3-tts which reads it from config.
EOS_TOKEN_ID = 2150
MAX_NEW_TOKENS = 4096
MIN_NEW_TOKENS = 2

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


# ---------------------------------------------------------------------------
# Prefill helpers
# ---------------------------------------------------------------------------


def _build_prefill_inputs_and_run(hf_model, text: str, speaker: str, language: str):
    """
    Run the prefill pass and return prefill outputs + captured conditioning tensors.

    Strategy: patch talker.generate() (what generate_custom_voice() calls).
    When generate() is called, we intercept its kwargs (which include inputs_embeds,
    trailing_text_hidden, tts_pad_embed, etc.), run the prefill forward() manually,
    then raise to abort the rest of generate_custom_voice().

    generate_custom_voice() calls:
        talker.generate(
            inputs_embeds=prefill_embeds,       [1, prefill_len, 1024]
            trailing_text_hidden=...,           [1, T, 1024]
            tts_pad_embed=...,                  [1, 1, 1024]
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=...,
            ...
        )

    Returns:
        past_key_values: DynamicCache (28 layers of KV)
        past_hidden: [1, 1, 1024] — last hidden state
        gen_step: int (0 after prefill)
        trailing_text_hiddens: [1, T, 1024]
        tts_pad_embed: [1, 1, 1024]
        logits: [1, 3072] — logits from last prefill position (for first token)
    """
    model = hf_model.model
    talker = model.talker

    captured = {}
    orig_generate = talker.generate

    class _AbortAfterPrefill(Exception):
        pass

    def _intercept_generate(**kwargs):
        # Extract the conditioning tensors generate_custom_voice passes to talker.generate()
        inputs_embeds = kwargs.get("inputs_embeds")
        trailing = kwargs.get("trailing_text_hidden")
        tts_pad = kwargs.get("tts_pad_embed")
        attention_mask = kwargs.get("attention_mask")

        if inputs_embeds is None:
            raise RuntimeError("talker.generate() called without inputs_embeds")

        # Clone conditioning tensors before the generate call modifies any state
        captured["trailing_text_hidden"] = trailing.detach().clone() if trailing is not None else None
        captured["tts_pad_embed"] = tts_pad.detach().clone() if tts_pad is not None else None

        # Run the prefill forward pass directly on talker (not via generate)
        # talker.forward() with inputs_embeds.shape[1] > 1 is the prefill
        prefill_result = talker.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        captured["prefill_result"] = prefill_result
        raise _AbortAfterPrefill()

    talker.generate = _intercept_generate
    try:
        with torch.inference_mode():
            hf_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
    except _AbortAfterPrefill:
        pass
    except Exception as e:
        logger.warning(f"Unexpected exception during prefill capture: {e}", exc_info=True)
    finally:
        talker.generate = orig_generate

    if "prefill_result" not in captured:
        raise RuntimeError(
            "Prefill capture failed — talker.generate() was not intercepted. "
            "generate_custom_voice() may have a different call structure."
        )

    result = captured["prefill_result"]

    # past_hidden comes from talker.forward() output field
    past_hidden = getattr(result, "past_hidden", None)
    if past_hidden is None:
        # Fallback: use last hidden state directly
        past_hidden = result.last_hidden_state[:, -1:, :]

    gen_step = getattr(result, "generation_step", 0) or 0

    return (
        result.past_key_values,                 # DynamicCache
        past_hidden,                            # [1, 1, 1024]
        gen_step,                               # int
        captured["trailing_text_hidden"],       # [1, T, 1024]
        captured["tts_pad_embed"],              # [1, 1, 1024]
        result.logits[:, -1, :],               # [1, 3072]
    )


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _custom_decode_loop(
    talker,
    past_key_values,
    past_hidden: torch.Tensor,
    gen_step: int,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    first_logits: torch.Tensor,
    config,
    max_new_tokens: int = MAX_NEW_TOKENS,
    min_new_tokens: int = MIN_NEW_TOKENS,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    chunk_size: int = CHUNK_FRAMES,
    on_chunk=None,  # callback(codec_chunk: torch.Tensor [chunk, 16]) called per chunk
):
    """
    Custom autoregressive decode loop owning the full generation runtime.

    Calls talker.model.forward() directly (bypassing talker.forward() / talker.generate()).
    Handles: code predictor, codec embedding reconstruction, text conditioning.
    Calls on_chunk(codec_chunk) every chunk_size frames.

    Returns: list of all [16]-tensors (one per codec frame)
    """
    device = past_hidden.device
    talker_model = talker.model
    codec_head = talker.codec_head
    codec_embedding = talker.get_input_embeddings()

    predictor = talker.code_predictor
    pred_model = predictor.model
    pred_lm_heads = predictor.lm_head           # ModuleList of 15 Linear heads
    pred_embeds = predictor.get_input_embeddings()  # ModuleList of 15 Embeddings(2048, 1024)
    # predictor_codec_embeds used for building next-step talker embedding
    predictor_codec_embeds = pred_embeds

    num_code_groups = config.num_code_groups   # 16
    eos_id = EOS_TOKEN_ID
    vocab_size = config.vocab_size   # 3072
    suppress_start = max(0, vocab_size - 1024)   # 2048

    # Pre-build suppress mask on GPU — avoids repeated masked_fill calls per step
    # Suppresses [2048..3071] except EOS=2150, matching official suppress_tokens list
    _suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    _suppress_mask[suppress_start:eos_id] = True
    _suppress_mask[eos_id + 1:] = True

    def _sample(logits_1d: torch.Tensor, suppress_eos: bool) -> torch.Tensor:
        logits_1d = logits_1d.clone()
        logits_1d[_suppress_mask] = float("-inf")
        if suppress_eos:
            logits_1d[eos_id] = float("-inf")
        if do_sample and temperature > 0:
            logits_1d = logits_1d / temperature
            if top_k > 0:
                topk_vals = torch.topk(logits_1d, min(top_k, logits_1d.size(-1))).values
                logits_1d = logits_1d.masked_fill(logits_1d < topk_vals[-1], float("-inf"))
            return torch.multinomial(torch.softmax(logits_1d, dim=-1), 1).squeeze()
        return logits_1d.argmax()

    def _run_predictor(pred_input: torch.Tensor) -> torch.Tensor:
        """
        Run code predictor for 15 codebook steps. Returns [15] int64 tensor.
        pred_input: [1, 2, hidden] = cat(past_hidden, last_id_hidden)
        """
        cb_tokens = []
        # Prefill: 2-token sequence through 5-layer predictor, build KV cache
        pred_out = pred_model(inputs_embeds=pred_input, use_cache=True, return_dict=True)
        pred_pkv = pred_out.past_key_values
        cb_logits = pred_lm_heads[0](pred_out.last_hidden_state[:, -1, :])
        cb_tok = (torch.multinomial(torch.softmax(cb_logits / temperature, -1), 1).squeeze()
                  if do_sample and temperature > 0 else cb_logits.argmax(-1).squeeze())
        cb_tokens.append(cb_tok)

        # Steps CB2..CB15: single-token decode with KV cache
        for cb_idx in range(1, num_code_groups - 1):
            cb_emb = pred_embeds[cb_idx - 1](cb_tok.unsqueeze(0).unsqueeze(0))
            pred_out = pred_model(inputs_embeds=cb_emb, past_key_values=pred_pkv,
                                  use_cache=True, return_dict=True)
            pred_pkv = pred_out.past_key_values
            cb_logits = pred_lm_heads[cb_idx](pred_out.last_hidden_state[:, -1, :])
            cb_tok = (torch.multinomial(torch.softmax(cb_logits / temperature, -1), 1).squeeze()
                      if do_sample and temperature > 0 else cb_logits.argmax(-1).squeeze())
            cb_tokens.append(cb_tok)

        return torch.stack(cb_tokens)  # [15]

    # First token from prefill logits
    token = _sample(first_logits.squeeze(0), suppress_eos=(min_new_tokens > 0))

    all_frames = []
    chunk_buffer = []
    step = 0

    for step_idx in range(max_new_tokens):
        tok_val = token.item()   # CPU sync — needed to check EOS
        if tok_val == eos_id:
            logger.info(f"[v2] EOS fired at step {step_idx}")
            break
        if step_idx < 3:
            logger.debug(f"[v2] step {step_idx}: token={tok_val}")

        # --- Code predictor: CB1..CB15 ---
        last_id_hidden = codec_embedding(token.unsqueeze(0).unsqueeze(0))   # [1, 1, 1024]
        pred_input = torch.cat([past_hidden, last_id_hidden], dim=1)         # [1, 2, 1024]
        codebook_token_ids = _run_predictor(pred_input)                      # [15]

        # Full codec frame: CB0 + CB1..CB15
        all_cb = torch.cat([token.view(1), codebook_token_ids])   # [16]
        chunk_buffer.append(all_cb.detach())
        all_frames.append(all_cb.detach())

        # --- Build next-step input embedding ---
        # Sum of 16 codec embeddings: CB0 from talker embed, CB1..15 from predictor embeds
        codec_hiddens = [last_id_hidden]   # [1, 1, 1024]
        for i in range(num_code_groups - 1):
            cb_tok = codebook_token_ids[i].unsqueeze(0).unsqueeze(0)   # [1, 1]
            codec_hiddens.append(predictor_codec_embeds[i](cb_tok))    # [1, 1, 1024]

        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)   # [1, 16, 1024] → [1, 1, 1024]

        # Add text conditioning
        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        # --- Talker backbone forward (single decode step) ---
        # Call talker.model (the Qwen3Model backbone) directly, bypassing talker.forward()
        # so we control KV cache and embedding — no code predictor called inside
        backbone_out = talker_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

        hidden = backbone_out.last_hidden_state    # [1, 1, 1024]
        past_key_values = backbone_out.past_key_values

        # Next token logits
        logits = codec_head(hidden[:, -1, :])      # [1, 3072]

        # Update recurrent state
        past_hidden = hidden[:, -1:, :].clone()    # [1, 1, 1024]
        gen_step += 1
        step = step_idx + 1

        # Sample next CB0 token
        suppress_eos = step < min_new_tokens
        token = _sample(logits.squeeze(0), suppress_eos=suppress_eos)

        if step_idx < 5 or step_idx % 20 == 0:
            logger.debug(f"[v2] step={step_idx} token={token.item()} gen_step={gen_step} eos_suppressed={suppress_eos}")

        # --- Yield chunk if buffer full ---
        if len(chunk_buffer) >= chunk_size and on_chunk is not None:
            chunk_tensor = torch.stack(chunk_buffer)   # [chunk_size, 16]
            on_chunk(chunk_tensor)
            chunk_buffer = []

    # Final partial chunk
    if chunk_buffer and on_chunk is not None:
        chunk_tensor = torch.stack(chunk_buffer)
        on_chunk(chunk_tensor)

    logger.info(f"[v2] Decode complete — {step} codec frames, {step * SAMPLES_PER_FRAME / SAMPLE_RATE:.2f}s audio")
    # all_frames is populated incrementally via all_frames.append() above
    return all_frames  # list of [16] tensors, one per codec frame


# ---------------------------------------------------------------------------
# Incremental vocoder
# ---------------------------------------------------------------------------

class _IncrementalVocoder:
    """
    Wraps speech_tokenizer to emit audio incrementally with left-context windowing.

    Codec decoder is a causal ConvNet — needs ~25 frames of left context for
    accurate reconstruction. We re-decode a window of (context + new) frames
    and trim to emit only the new samples.
    """

    def __init__(self, speech_tokenizer, context_frames: int = CONTEXT_FRAMES):
        self._tokenizer = speech_tokenizer
        self._context_frames = context_frames
        self._all_codes: list[torch.Tensor] = []  # list of [16] tensors
        self._emitted_frames = 0

    def _decode_window(self, window_codes: torch.Tensor) -> np.ndarray:
        """
        Call speech_tokenizer.decode() on window_codes [n_frames, 16].
        Returns float32 numpy array of waveform samples.

        API: speech_tokenizer.decode([{"audio_codes": codes}]) returns (wav_list, sr)
        where wav_list[0] is a float32 numpy array of shape [n_samples].
        """
        window = window_codes.long()  # ensure int64
        result = self._tokenizer.decode([{"audio_codes": window}])
        if isinstance(result, (tuple, list)) and len(result) == 2:
            audio_data, sr = result
            if isinstance(audio_data, (list, tuple)):
                audio_arr = audio_data[0] if audio_data else np.zeros(0, dtype=np.float32)
            else:
                audio_arr = audio_data
        else:
            audio_arr = result

        if hasattr(audio_arr, "cpu"):
            audio_arr = audio_arr.cpu().float().numpy()
        return np.asarray(audio_arr, dtype=np.float32).squeeze()

    def add_chunk(self, codec_chunk: torch.Tensor) -> bytes:
        """
        Add codec_chunk ([N, 16]) and return new audio bytes (int16 PCM, 24kHz).
        """
        for i in range(codec_chunk.shape[0]):
            self._all_codes.append(codec_chunk[i])

        n_total = len(self._all_codes)
        n_new = n_total - self._emitted_frames
        if n_new <= 0:
            return b""

        ctx_start = max(0, n_total - self._context_frames - n_new)
        window_frames = torch.stack(self._all_codes[ctx_start:n_total])   # [ctx+new, 16]

        try:
            audio_arr = self._decode_window(window_frames)

            # Trim to only new samples (the tail of the decoded audio)
            new_samples = n_new * SAMPLES_PER_FRAME
            if len(audio_arr) > new_samples:
                audio_arr = audio_arr[-new_samples:]

            self._emitted_frames = n_total
            return (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        except Exception as e:
            logger.error(f"[v2] Vocoder error: {e}", exc_info=True)
            self._emitted_frames = n_total
            return bytes(n_new * SAMPLES_PER_FRAME * 2)   # silence

    def flush(self, full_codes: list[torch.Tensor]) -> bytes:
        """
        Final flush: vocode remaining frames not yet emitted.
        Uses the full code sequence for best tail quality.
        """
        n_total = len(full_codes)
        n_remaining = n_total - self._emitted_frames
        if n_remaining <= 0:
            return b""

        try:
            all_frames = torch.stack(full_codes)   # [T, 16]
            audio_arr = self._decode_window(all_frames)

            tail_samples = n_remaining * SAMPLES_PER_FRAME
            if len(audio_arr) > tail_samples:
                audio_arr = audio_arr[-tail_samples:]

            return (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"[v2] Vocoder flush error: {e}", exc_info=True)
            return bytes(n_remaining * SAMPLES_PER_FRAME * 2)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class QwenTTSBackendV2:
    """
    Phase 2 backend — custom decode loop with correct codec frame generation.

    Differences from QwenTTSBackendMK (v1):
    - Owns the FULL decode loop (no HF generate() call)
    - Calls code_predictor correctly per step
    - Builds codec embedding reconstruction (16 codebooks summed)
    - Applies trailing_text_hiddens conditioning per step
    - Streams audio chunks from vocoder incrementally (true streaming)

    Drop-in compatible with QwenTTSService (same synthesize_streaming() interface).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        speaker: str = "Ryan",
        language: str = "English",
        megakernel_path: str = "./qwen_megakernel",
    ):
        from qwen_tts import Qwen3TTSModel

        logger.info("[v2] Loading Qwen3-TTS model...")
        t0 = time.perf_counter()

        self._hf = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        self._hf.model.eval()
        self._speaker = speaker
        self._language = language
        self.sample_rate = SAMPLE_RATE

        # Cache handles to modules we need directly
        model = self._hf.model
        self._talker = model.talker
        self._config = model.talker.config
        self._speech_tokenizer = model.speech_tokenizer

        # torch.compile for kernel fusion — set V2_COMPILE=0 to disable
        if os.environ.get("V2_COMPILE", "1") != "0":
            logger.info("[v2] Applying torch.compile (set V2_COMPILE=0 to skip)...")
            try:
                model.talker.model = torch.compile(
                    model.talker.model, mode="reduce-overhead", fullgraph=False
                )
                model.talker.code_predictor.model = torch.compile(
                    model.talker.code_predictor.model, mode="reduce-overhead", fullgraph=False
                )
                logger.info("[v2] torch.compile applied")
            except Exception as e:
                logger.warning(f"[v2] torch.compile failed ({e}), running uncompiled")

        logger.info(f"[v2] Ready in {(time.perf_counter()-t0)*1000:.0f}ms")
        logger.info(f"[v2] num_code_groups={self._config.num_code_groups}, vocab_size={self._config.vocab_size}")

    def _run_custom_decode(self, text: str) -> tuple[list, float]:
        """
        Run full custom decode. Returns (all_frames, decode_time_s).
        Raises on error — caller handles fallback.
        """
        logger.info(f"[v2] Starting custom decode for: {text[:60]!r}")
        t0 = time.perf_counter()

        past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
            _build_prefill_inputs_and_run(
                self._hf, text, self._speaker, self._language
            )

        t_prefill = time.perf_counter() - t0
        logger.info(f"[v2] Prefill done in {t_prefill*1000:.0f}ms, gen_step={gen_step}")

        frames = _custom_decode_loop(
            talker=self._talker,
            past_key_values=past_kv,
            past_hidden=past_hidden,
            gen_step=gen_step,
            trailing_text_hiddens=trailing,
            tts_pad_embed=tts_pad,
            first_logits=first_logits,
            config=self._config,
        )

        t_decode = time.perf_counter() - t0
        return frames, t_decode

    def _run_hf_fallback(self, text: str) -> np.ndarray:
        """Pure HF inference — unchanged backup path."""
        wavs, sr = self._hf.generate_custom_voice(
            text=text,
            language=self._language,
            speaker=self._speaker,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.9,
        )
        return np.array(wavs[0], dtype=np.float32).squeeze()

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) in ~CHUNK_FRAMES*80ms chunks.
        Uses custom decode loop with real streaming.
        Falls back to HF batch if custom decode fails.
        """
        loop = asyncio.get_event_loop()

        # Run the blocking decode in a thread executor
        # We use a queue to stream results from the executor thread to the async generator
        audio_queue: asyncio.Queue = asyncio.Queue()
        vocoder = _IncrementalVocoder(self._speech_tokenizer)
        all_frames_ref = []
        decode_error = None

        def _decode_thread():
            nonlocal decode_error
            try:
                t_start = time.perf_counter()

                past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
                    _build_prefill_inputs_and_run(
                        self._hf, text, self._speaker, self._language
                    )

                t_prefill = time.perf_counter() - t_start
                logger.info(
                    f"[v2] Prefill {t_prefill*1000:.0f}ms, gen_step={gen_step}, "
                    f"trailing_len={trailing.shape[1] if trailing is not None else 0}"
                )

                def _on_chunk(chunk: torch.Tensor):
                    # chunk: [CHUNK_FRAMES, 16] or smaller for final chunk
                    audio_bytes = vocoder.add_chunk(chunk)
                    if audio_bytes:
                        asyncio.run_coroutine_threadsafe(
                            audio_queue.put(audio_bytes), loop
                        ).result()

                frames = _custom_decode_loop(
                    talker=self._talker,
                    past_key_values=past_kv,
                    past_hidden=past_hidden,
                    gen_step=gen_step,
                    trailing_text_hiddens=trailing,
                    tts_pad_embed=tts_pad,
                    first_logits=first_logits,
                    config=self._config,
                    chunk_size=CHUNK_FRAMES,
                    on_chunk=_on_chunk,
                )
                all_frames_ref.extend(frames)

                # Flush tail using the full frame list for best quality
                tail = vocoder.flush(all_frames_ref)
                if tail:
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put(tail), loop
                    ).result()

            except Exception as e:
                logger.error(f"[v2] Decode thread error: {e}", exc_info=True)
                decode_error = e
            finally:
                asyncio.run_coroutine_threadsafe(
                    audio_queue.put(None), loop  # sentinel — signals end of stream
                ).result()

        try:
            executor_fut = loop.run_in_executor(None, _decode_thread)

            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield chunk, self.sample_rate

            await executor_fut  # propagate any thread exception

            if decode_error is not None:
                logger.warning("[v2] Custom decode failed, falling back to HF")
                audio = await loop.run_in_executor(None, self._run_hf_fallback, text)
                chunk_samples = int(self.sample_rate * 100 / 1000)
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i+chunk_samples]
                    yield (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), self.sample_rate
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"[v2] synthesize_streaming error: {e}", exc_info=True)
            # Last resort: HF fallback
            audio = await loop.run_in_executor(None, self._run_hf_fallback, text)
            chunk_samples = int(self.sample_rate * 100 / 1000)
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i+chunk_samples]
                yield (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), self.sample_rate
                await asyncio.sleep(0)
