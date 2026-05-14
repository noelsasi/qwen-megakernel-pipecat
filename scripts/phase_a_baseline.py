"""
Phase A.4 — Baseline TTS to WAV file with latency measurements.

Run AFTER phase_a_inspect_model.py has confirmed the correct class/API.
Update MODEL_CLASS and LOAD_KWARGS based on inspect output.

Usage:
    python scripts/phase_a_baseline.py --text "Hello, this is a test."
    python scripts/phase_a_baseline.py  # uses default text
"""

import argparse
import time
import numpy as np
import soundfile as sf
import torch


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_TEXT = "Hello, this is a test of the Qwen TTS system. The quick brown fox jumps over the lazy dog."
OUTPUT_PATH = "output_baseline.wav"


def load_model():
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    processor = Qwen3TTSProcessor.from_pretrained(MODEL_ID)

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load: {load_ms:.0f}ms")
    print(f"Model type: {type(model).__name__}")

    return model, processor


def run_inference(model, processor, text: str):
    """
    Run single inference. Returns (audio_np, sample_rate).
    FILL IN after phase_a_inspect_model confirms the correct generate() signature.
    """
    # Tokenize
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        # Adjust generate() kwargs from inspect output
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.9,
        )

    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000

    # Extract audio — format TBD from A.3 inspection
    # Common patterns:
    #   outputs.audio          (tensor)
    #   outputs.waveform       (tensor)
    #   outputs[0]             (tensor)
    #   processor.decode(outputs) → numpy
    #
    # Attempt auto-detection:
    if hasattr(outputs, "audio"):
        audio = outputs.audio.cpu().float().numpy()
    elif hasattr(outputs, "waveform"):
        audio = outputs.waveform.cpu().float().numpy()
    elif isinstance(outputs, torch.Tensor):
        audio = outputs.cpu().float().numpy()
    else:
        # Last resort: ask processor to decode
        audio = processor.batch_decode(outputs, skip_special_tokens=True)
        print(f"WARNING: unexpected output type: {type(outputs)} — decoded as: {type(audio)}")

    # Flatten to 1D if (1, N) or (1, 1, N)
    if isinstance(audio, np.ndarray):
        audio = audio.squeeze()

    # Sample rate — most likely 24000 from model card
    sr = getattr(processor, "sampling_rate", None) or getattr(model.config, "sampling_rate", 24000)

    return audio, sr, gen_ms


def warmup(model, processor):
    print("Warming up (first inference compiles CUDA ops) ...")
    try:
        run_inference(model, processor, "warmup")
        torch.cuda.synchronize()
        print("Warmup done.")
    except Exception as e:
        print(f"Warmup failed (may be API mismatch — check inspect output): {e}")
        raise


def benchmark(model, processor, text: str, trials: int = 5):
    print(f"\nBenchmarking {trials} trials:")
    gen_times = []
    last_audio, last_sr = None, None

    for i in range(trials):
        audio, sr, gen_ms = run_inference(model, processor, text)
        gen_times.append(gen_ms)
        last_audio, last_sr = audio, sr
        audio_ms = len(audio) / sr * 1000
        rtf = gen_ms / audio_ms
        print(f"  Trial {i+1}: gen={gen_ms:.0f}ms  audio={audio_ms:.0f}ms  RTF={rtf:.3f}")

    mean = np.mean(gen_times)
    std = np.std(gen_times)
    audio_ms = len(last_audio) / last_sr * 1000
    rtf = mean / audio_ms

    print(f"\nResults (mean ± std over {trials} trials):")
    print(f"  Generation time: {mean:.0f} ± {std:.0f} ms")
    print(f"  Audio duration:  {audio_ms:.0f} ms")
    print(f"  RTF:             {rtf:.3f}")
    print(f"  Target RTF < 0.15: {'PASS' if rtf < 0.15 else 'FAIL'}")

    return last_audio, last_sr, mean, rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, processor = load_model()
    warmup(model, processor)

    audio, sr, gen_mean, rtf = benchmark(model, processor, args.text, trials=args.trials)

    sf.write(args.output, audio, sr)
    print(f"\nSaved: {args.output} (sr={sr}Hz)")
    print("\nPhase A DONE — fill in compatibility_check.py with config values from inspect output.")


if __name__ == "__main__":
    main()
