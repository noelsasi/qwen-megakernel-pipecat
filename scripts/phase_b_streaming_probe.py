"""
Phase B.1 — Probe whether Qwen3-TTS supports real streaming.

Answers:
  1. Does generate() accept a TextIteratorStreamer?
  2. What is the codec frame rate / tokens-per-frame?
  3. Is there a per-frame decode path (code predictor + vocoder separately)?

Run AFTER phase_a_inspect_model.py has confirmed the model API.

Usage:
    python scripts/phase_b_streaming_probe.py 2>&1 | tee streaming_probe_output.txt
"""

import time
import torch
import numpy as np

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEST_TEXT = "Hello world, this is a streaming test."


def probe_streamer_support(model, processor):
    """Test whether generate() accepts a TextIteratorStreamer."""
    print("\n--- Probe: TextIteratorStreamer support ---")
    try:
        from transformers import TextIteratorStreamer
        import threading

        streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=False)

        inputs = processor(text=TEST_TEXT, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        tokens_received = []
        timestamps = []
        t_start = time.perf_counter()

        def generate():
            model.generate(**inputs, streamer=streamer, max_new_tokens=512)

        thread = threading.Thread(target=generate)
        thread.start()

        for tok in streamer:
            timestamps.append(time.perf_counter() - t_start)
            tokens_received.append(tok)
            if len(tokens_received) == 1:
                print(f"  First token arrived at: {timestamps[0]*1000:.1f}ms")

        thread.join()

        print(f"  Total tokens: {len(tokens_received)}")
        if len(timestamps) > 1:
            intervals = np.diff(timestamps) * 1000
            print(f"  Token interval: {np.mean(intervals):.1f} ± {np.std(intervals):.1f} ms")

        print("  RESULT: TextIteratorStreamer IS supported")
        return True, tokens_received

    except Exception as e:
        print(f"  TextIteratorStreamer FAILED: {e}")
        print("  RESULT: Streamer not supported — will use fake streaming or manual loop")
        return False, []


def probe_codec_frame_rate(model):
    """Look for codec/frame-rate config values."""
    print("\n--- Probe: Codec frame rate ---")
    config = model.config
    for attr in ["codec_frame_rate", "frame_rate", "sampling_rate", "audio_frame_rate",
                 "tokens_per_second", "codebook_size", "num_codebooks"]:
        val = getattr(config, attr, None)
        if val is not None:
            print(f"  config.{attr} = {val}")

    # Also check nested configs
    for sub_name in ["talker_config", "decoder_config", "speech_config"]:
        sub = getattr(config, sub_name, None)
        if sub is not None:
            print(f"  {sub_name} found: {sub}")


def probe_vocoder_separate(model):
    """Check if vocoder/code_predictor are accessible as separate modules."""
    print("\n--- Probe: Separate vocoder/code_predictor ---")
    for attr in ["vocoder", "code_predictor", "speech_tokenizer", "dac", "codec",
                 "talker", "decoder", "lm_head", "audio_head"]:
        val = getattr(model, attr, None)
        if val is not None:
            print(f"  model.{attr}: {type(val).__name__}")
            # Try to find a decode method
            for method in ["decode", "forward", "synthesize", "generate"]:
                if hasattr(val, method):
                    print(f"    .{method}() exists")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    probe_codec_frame_rate(model)
    probe_vocoder_separate(model)
    streamer_ok, tokens = probe_streamer_support(model, processor)

    print("\n" + "=" * 60)
    print("STREAMING PROBE COMPLETE")
    print(f"Streamer supported: {streamer_ok}")
    print("""
Fill into phase_b_streaming.py:
  STREAMER_SUPPORTED = ???
  CODEC_FRAME_RATE = ???   (Hz, e.g. 12.5)
  SAMPLES_PER_FRAME = sample_rate / frame_rate = 24000 / ??? = ???
  TOKENS_PER_FRAME = ???
""")


if __name__ == "__main__":
    main()
