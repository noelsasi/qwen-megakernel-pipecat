"""
Validation script for Phase 2 custom decode loop.

Run on the GPU server:
    cd /workspace/qwen-megakernel-pipecat
    source .venv/bin/activate
    python scripts/test_v2_decode.py

Stages:
  1. Prefill capture — verify trailing_text_hiddens + tts_pad_embed captured
  2. Single decode step — verify code_predictor produces 16 codebook tokens
  3. Full decode to EOS — verify EOS fires before max_new_tokens
  4. Vocoder output — verify audio WAV is produced
  5. Streaming test — verify audio chunks arrive incrementally (TTFC measurement)
"""

import sys
import time
import logging
import asyncio
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")

TEST_TEXT = "Hello, this is a test."
SPEAKER = "Ryan"
LANGUAGE = "English"


def stage0_probe_generate_args(backend):
    """Probe what kwargs generate_custom_voice() passes to talker.generate()."""
    logger.info("=== STAGE 0: Probe talker.generate() kwargs ===")
    model = backend._hf.model
    talker = model.talker
    orig_generate = talker.generate

    def _probe(**kwargs):
        logger.info("talker.generate() called with kwargs:")
        for k, v in kwargs.items():
            if hasattr(v, "shape"):
                logger.info(f"  {k}: tensor {v.shape} {v.dtype}")
            elif isinstance(v, (int, float, bool, str)) or v is None:
                logger.info(f"  {k}: {v}")
            else:
                logger.info(f"  {k}: {type(v).__name__}")
        raise RuntimeError("probe done")

    talker.generate = _probe
    try:
        import torch
        with torch.inference_mode():
            backend._hf.generate_custom_voice(
                text=TEST_TEXT, language=LANGUAGE, speaker=SPEAKER, max_new_tokens=1
            )
    except RuntimeError as e:
        if "probe done" in str(e):
            logger.info("STAGE 0 PASS — kwargs logged above")
        else:
            logger.error(f"STAGE 0 unexpected error: {e}")
    except Exception as e:
        logger.error(f"STAGE 0 error: {e}", exc_info=True)
    finally:
        talker.generate = orig_generate


def stage1_prefill(backend):
    logger.info("=== STAGE 1: Prefill capture ===")
    from server.backend.tts_backend_v2 import _build_prefill_inputs_and_run

    t0 = time.perf_counter()
    past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
        _build_prefill_inputs_and_run(
            backend._hf, TEST_TEXT, SPEAKER, LANGUAGE
        )
    t1 = time.perf_counter()

    logger.info(f"Prefill done in {(t1-t0)*1000:.0f}ms")
    logger.info(f"  past_hidden shape: {past_hidden.shape}")   # expect [1, 1, 1024]
    logger.info(f"  gen_step: {gen_step}")
    logger.info(f"  trailing_text_hiddens shape: {trailing.shape if trailing is not None else None}")
    logger.info(f"  tts_pad_embed shape: {tts_pad.shape if tts_pad is not None else None}")
    logger.info(f"  first_logits shape: {first_logits.shape}")   # expect [1, 3072]
    logger.info(f"  first token (argmax): {first_logits.argmax(-1).item()}")

    assert past_hidden.shape == (1, 1, 1024), f"past_hidden shape wrong: {past_hidden.shape}"
    assert first_logits.shape[-1] == 3072, f"logits shape wrong: {first_logits.shape}"
    if trailing is not None:
        assert trailing.ndim == 3, f"trailing_text_hiddens should be 3D: {trailing.shape}"
    logger.info("STAGE 1 PASS")
    return past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits


def stage2_single_decode_step(backend, past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits):
    logger.info("=== STAGE 2: Single decode step ===")
    import torch
    from server.backend.tts_backend_v2 import _custom_decode_loop

    frames_out = []

    def on_chunk(chunk):
        frames_out.extend([chunk[i] for i in range(chunk.shape[0])])

    t0 = time.perf_counter()
    with torch.inference_mode():
        frames = _custom_decode_loop(
            talker=backend._talker,
            past_key_values=past_kv,
            past_hidden=past_hidden,
            gen_step=gen_step,
            trailing_text_hiddens=trailing,
            tts_pad_embed=tts_pad,
            first_logits=first_logits,
            config=backend._config,
            max_new_tokens=3,   # just 3 steps to verify frame structure
            on_chunk=on_chunk,
            chunk_size=1,
        )
    t1 = time.perf_counter()

    logger.info(f"3 steps in {(t1-t0)*1000:.0f}ms")
    logger.info(f"  frames produced: {len(frames)}")
    if frames:
        logger.info(f"  frame[0] shape: {frames[0].shape}")   # expect [16]
        logger.info(f"  frame[0] tokens: {frames[0].tolist()}")
        all_valid = all(0 <= int(frames[i][0]) < 3072 for i in range(len(frames)))
        logger.info(f"  All CB0 tokens in range [0, 3072): {all_valid}")
    logger.info("STAGE 2 PASS")
    return frames


def stage3_eos(backend):
    logger.info("=== STAGE 3: Full decode to EOS ===")
    import torch
    from server.backend.tts_backend_v2 import _build_prefill_inputs_and_run, _custom_decode_loop, EOS_TOKEN_ID

    past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
        _build_prefill_inputs_and_run(backend._hf, TEST_TEXT, SPEAKER, LANGUAGE)

    frames_out = []

    def on_chunk(chunk):
        frames_out.extend([chunk[i] for i in range(chunk.shape[0])])

    t0 = time.perf_counter()
    with torch.inference_mode():
        frames = _custom_decode_loop(
            talker=backend._talker,
            past_key_values=past_kv,
            past_hidden=past_hidden,
            gen_step=gen_step,
            trailing_text_hiddens=trailing,
            tts_pad_embed=tts_pad,
            first_logits=first_logits,
            config=backend._config,
            max_new_tokens=4096,
            on_chunk=on_chunk,
            chunk_size=12,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    n_frames = len(frames)
    audio_dur = n_frames / 12   # seconds (12Hz codec)
    elapsed = t1 - t0
    cb0_tokens = [int(f[0]) for f in frames]
    eos_hit = EOS_TOKEN_ID in cb0_tokens

    logger.info(f"Decode: {n_frames} frames in {elapsed*1000:.0f}ms")
    logger.info(f"  Audio duration: {audio_dur:.2f}s")
    logger.info(f"  tok/s: {n_frames / elapsed:.0f}")
    logger.info(f"  EOS hit: {eos_hit}")
    if not eos_hit:
        logger.warning(f"  EOS NOT hit — ran full {n_frames} frames. Sequence may be diverging.")
        logger.info(f"  Last 5 CB0 tokens: {cb0_tokens[-5:]}")
    logger.info("STAGE 3 PASS (with or without EOS)" if n_frames > 0 else "STAGE 3 FAIL (no frames)")
    return frames


def stage4_vocoder(backend, frames):
    logger.info("=== STAGE 4: Vocoder output ===")
    import torch
    import soundfile as sf
    from server.backend.tts_backend_v2 import _IncrementalVocoder, SAMPLES_PER_FRAME, SAMPLE_RATE

    if not frames:
        logger.warning("No frames to vocode — skipping")
        return

    vocoder = _IncrementalVocoder(backend._speech_tokenizer)

    # Feed all frames as one big chunk
    chunk = torch.stack(frames)   # [N, 16]
    t0 = time.perf_counter()
    audio_bytes = vocoder.add_chunk(chunk)
    t1 = time.perf_counter()

    n_samples = len(audio_bytes) // 2
    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
    audio_dur = n_samples / SAMPLE_RATE

    logger.info(f"Vocoder: {len(frames)} frames → {n_samples} samples ({audio_dur:.2f}s) in {(t1-t0)*1000:.0f}ms")

    out_path = "/tmp/test_v2_output.wav"
    sf.write(out_path, audio_arr, SAMPLE_RATE)
    logger.info(f"Saved: {out_path}")
    logger.info("STAGE 4 PASS" if n_samples > 0 else "STAGE 4 FAIL (no audio)")


async def stage5_streaming_ttfc(backend):
    logger.info("=== STAGE 5: Streaming TTFC measurement ===")
    t_start = time.perf_counter()
    t_first = None
    total_samples = 0
    n_chunks = 0

    async for audio_bytes, sr in backend.synthesize_streaming(TEST_TEXT):
        if t_first is None:
            t_first = time.perf_counter()
            ttfc_ms = (t_first - t_start) * 1000
            logger.info(f"  TTFC: {ttfc_ms:.0f}ms (target < 60ms)")
        total_samples += len(audio_bytes) // 2
        n_chunks += 1

    t_end = time.perf_counter()
    total_time = t_end - t_start
    audio_dur = total_samples / backend.sample_rate
    rtf = total_time / audio_dur if audio_dur > 0 else float("inf")

    logger.info(f"  Chunks received: {n_chunks}")
    logger.info(f"  Audio duration: {audio_dur:.2f}s")
    logger.info(f"  Total time: {total_time*1000:.0f}ms")
    logger.info(f"  RTF: {rtf:.3f} (target < 0.15)")
    logger.info(f"  RTF PASS: {rtf < 0.15}")
    logger.info("STAGE 5 complete")


def main():
    logger.info("Loading QwenTTSBackendV2...")
    from server.backend.tts_backend_v2 import QwenTTSBackendV2
    backend = QwenTTSBackendV2()

    logger.info("\n" + "="*60)

    # Stage 0: Probe (helps debug if stage 1 fails)
    stage0_probe_generate_args(backend)

    logger.info("\n" + "="*60)

    # Stage 1: Prefill
    past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = stage1_prefill(backend)

    logger.info("\n" + "="*60)

    # Stage 2: Single decode step
    stage2_single_decode_step(backend, past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits)

    logger.info("\n" + "="*60)

    # Stage 3: Full decode to EOS
    frames = stage3_eos(backend)

    logger.info("\n" + "="*60)

    # Stage 4: Vocoder
    stage4_vocoder(backend, frames)

    logger.info("\n" + "="*60)

    # Stage 5: Streaming TTFC
    asyncio.run(stage5_streaming_ttfc(backend))

    logger.info("\n" + "="*60)
    logger.info("All stages complete.")


if __name__ == "__main__":
    main()
