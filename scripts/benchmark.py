"""
Final benchmark suite — TTFC, RTF, tok/s, E2E latency.

Run AFTER all phases are complete. Tests both HF baseline and megakernel backend.

Usage:
    python scripts/benchmark.py --backend hf
    python scripts/benchmark.py --backend megakernel
    python scripts/benchmark.py --backend both
"""

import argparse
import asyncio
import time
import numpy as np
import torch


BENCHMARK_TEXTS = [
    "Hello.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we interact with computers and devices.",
    "In the beginning there was darkness, and then there was light, and everything changed forever after that moment.",
]


async def measure_ttfc(backend, text: str, trials: int = 5) -> tuple[float, float]:
    """Returns (mean_ms, std_ms)."""
    ttfc_list = []
    for _ in range(trials):
        t_start = time.perf_counter()
        first = True
        async for chunk, sr in backend.synthesize_streaming(text):
            if first:
                torch.cuda.synchronize()
                ttfc_list.append((time.perf_counter() - t_start) * 1000)
                first = False
    mean = float(np.mean(ttfc_list))
    std = float(np.std(ttfc_list))
    return mean, std


async def measure_rtf(backend, text: str, trials: int = 5) -> tuple[float, float]:
    """Returns (mean_rtf, std_rtf)."""
    rtf_list = []
    for _ in range(trials):
        t_start = time.perf_counter()
        total_samples = 0
        sr_val = 24000
        async for chunk, sr in backend.synthesize_streaming(text):
            total_samples += len(chunk) // 2  # int16
            sr_val = sr
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t_start
        audio_dur = total_samples / sr_val
        if audio_dur > 0:
            rtf_list.append(gen_time / audio_dur)
    mean = float(np.mean(rtf_list))
    std = float(np.std(rtf_list))
    return mean, std


def measure_toks_per_second(decoder, warmup_steps=10, measure_steps=100, trials=5):
    """
    Measure megakernel decode step throughput (tok/s).
    Only valid for the megakernel backend which exposes decoder.step().
    """
    # Warmup
    dummy_token = 0
    for _ in range(warmup_steps):
        decoder.step(dummy_token)
    torch.cuda.synchronize()

    tok_s_list = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(measure_steps):
            decoder.step(dummy_token)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tok_s_list.append(measure_steps / elapsed)

    mean = float(np.mean(tok_s_list))
    std = float(np.std(tok_s_list))
    return mean, std


async def run_benchmark(backend_name: str, backend, trials: int = 5):
    print(f"\n{'=' * 60}")
    print(f"Backend: {backend_name}")
    print("=" * 60)

    results = {}
    for text in BENCHMARK_TEXTS:
        label = text[:40] + ("..." if len(text) > 40 else "")
        print(f"\nText: '{label}'")

        ttfc_mean, ttfc_std = await measure_ttfc(backend, text, trials=trials)
        rtf_mean, rtf_std = await measure_rtf(backend, text, trials=trials)

        print(f"  TTFC: {ttfc_mean:.1f} ± {ttfc_std:.1f} ms  (target < 60ms: {'PASS' if ttfc_mean < 60 else 'FAIL'})")
        print(f"  RTF:  {rtf_mean:.3f} ± {rtf_std:.3f}     (target < 0.15: {'PASS' if rtf_mean < 0.15 else 'FAIL'})")

        results[text[:20]] = {"ttfc": ttfc_mean, "rtf": rtf_mean}

    print(f"\n{'=' * 60}")
    print(f"Summary — {backend_name}")
    print(f"  Mean TTFC: {np.mean([r['ttfc'] for r in results.values()]):.1f} ms")
    print(f"  Mean RTF:  {np.mean([r['rtf'] for r in results.values()]):.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "megakernel", "both"], default="hf")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Trials per measurement: {args.trials}")

    import sys
    sys.path.insert(0, ".")

    if args.backend in ("hf", "both"):
        from server.backend.tts_backend_hf import QwenTTSBackendHF
        hf_backend = QwenTTSBackendHF(model_id=args.model_id)
        asyncio.run(run_benchmark("HuggingFace baseline", hf_backend, trials=args.trials))

    if args.backend in ("megakernel", "both"):
        from server.backend.tts_backend_mk import QwenTTSBackendMK
        mk_backend = QwenTTSBackendMK(model_id=args.model_id, megakernel_path="./qwen_megakernel")
        asyncio.run(run_benchmark("Megakernel", mk_backend, trials=args.trials))

        print("\n--- tok/s (megakernel decode step) ---")
        tok_mean, tok_std = measure_toks_per_second(mk_backend.mk_decoder)
        print(f"  tok/s: {tok_mean:.1f} ± {tok_std:.1f}")


if __name__ == "__main__":
    main()
