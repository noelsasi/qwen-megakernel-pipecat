"""
Phase A.4 — Baseline TTS to WAV file with latency measurements.

Confirmed API (from QwenLM/Qwen3-TTS repo + source inspection):
  - Use Qwen3TTSModel (high-level wrapper), NOT Qwen3TTSForConditionalGeneration directly
  - Call model.generate_custom_voice() — handles tokenization internally
  - Speaker must be one of: Ryan, Aiden (English), Vivian, Serena, Uncle_Fu,
    Dylan, Eric (Chinese), Ono_Anna (Japanese), Sohee (Korean)
  - Sample rate returned by the model is 12000 Hz (12Hz tokenizer)
  - Returns (wavs: list[np.ndarray], sr: int)

Usage:
    python scripts/phase_a_baseline.py
    python scripts/phase_a_baseline.py --text "Hello world" --speaker Ryan
"""

import argparse
import time
import numpy as np
import soundfile as sf
import torch


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_TEXT = "Hello, this is a test of the Qwen TTS system. The quick brown fox jumps over the lazy dog."
OUTPUT_PATH = "output_baseline.wav"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"


def load_model():
    from qwen_tts import Qwen3TTSModel

    print(f"Loading model: {MODEL_ID}")
    t0 = time.perf_counter()

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        dtype=torch.bfloat16,
    )

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load: {load_ms:.0f}ms")
    print(f"Model type: {type(model).__name__}")
    print(f"Supported speakers: {list(model.get_supported_speakers())}")

    return model


def run_inference(model, text: str, speaker: str, language: str):
    """
    Run single non-streaming inference. Returns (audio_np, sample_rate, gen_ms).
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
    )

    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000

    audio = wavs[0]  # first (only) batch item
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()
    audio = np.array(audio, dtype=np.float32).squeeze()

    return audio, sr, gen_ms


def warmup(model, speaker, language):
    print("Warming up (first inference compiles CUDA ops) ...")
    run_inference(model, "warmup test", speaker, language)
    torch.cuda.synchronize()
    print("Warmup done.")


def benchmark(model, text: str, speaker: str, language: str, trials: int = 5):
    print(f"\nBenchmarking {trials} trials | speaker={speaker} | language={language}")
    gen_times = []
    last_audio, last_sr = None, None

    for i in range(trials):
        audio, sr, gen_ms = run_inference(model, text, speaker, language)
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
    print(f"  Sample rate:     {last_sr} Hz")
    print(f"  RTF:             {rtf:.3f}")
    print(f"  Target RTF < 0.15: {'PASS' if rtf < 0.15 else 'FAIL'}")

    return last_audio, last_sr, mean, std, rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model()
    warmup(model, args.speaker, args.language)

    audio, sr, gen_mean, gen_std, rtf = benchmark(
        model, args.text, args.speaker, args.language, trials=args.trials
    )

    sf.write(args.output, audio, sr)
    print(f"\nSaved: {args.output} (sr={sr}Hz)")
    print("Phase A DONE.")


if __name__ == "__main__":
    main()
