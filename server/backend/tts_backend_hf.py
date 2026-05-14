"""
Phase A/B/C — HuggingFace TTS backend.

Wraps Qwen3-TTS model inference as an async streaming generator.
This is a best-effort implementation; exact generate() kwargs and output
extraction may need adjustment after running phase_a_inspect_model.py.

Interface:
    backend = QwenTTSBackendHF(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    async for audio_bytes, sample_rate in backend.synthesize_streaming("Hello"):
        # audio_bytes: int16 PCM bytes
        # sample_rate: int (e.g. 24000)
"""

import asyncio
import time
import numpy as np
import torch
from collections.abc import AsyncGenerator


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Chunk size for fake-streaming fallback (100ms chunks at 24kHz)
_CHUNK_MS = 100


def _audio_to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio.tobytes()


class QwenTTSBackendHF:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        # Requires transformers from source: pip install git+https://github.com/huggingface/transformers.git
        from transformers import Qwen3TtsForConditionalGeneration, Qwen3TtsProcessor

        print(f"[QwenTTSBackendHF] Loading {model_id} ...")
        t0 = time.perf_counter()

        self.model = Qwen3TtsForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.processor = Qwen3TtsProcessor.from_pretrained(model_id)
        self.sample_rate = getattr(self.processor, "sampling_rate", 24000) or 24000

        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[QwenTTSBackendHF] Loaded in {load_ms:.0f}ms  sr={self.sample_rate}")

    def _run_inference(self, text: str) -> np.ndarray:
        """Blocking inference — runs full generate() and returns float32 audio."""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.9,
            )

        # Extract audio — auto-detect output format from inspect results
        if hasattr(outputs, "audio"):
            audio = outputs.audio.cpu().float().numpy().squeeze()
        elif hasattr(outputs, "waveform"):
            audio = outputs.waveform.cpu().float().numpy().squeeze()
        elif isinstance(outputs, torch.Tensor):
            audio = outputs.cpu().float().numpy().squeeze()
        else:
            # Processor-based decode fallback
            audio = self.processor.batch_decode(outputs, skip_special_tokens=True)
            if isinstance(audio, list):
                audio = np.array(audio[0], dtype=np.float32)

        return audio

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) in chunks.

        Uses fake streaming (full inference then chunk) until Phase B confirms
        a real streaming path is available.
        """
        loop = asyncio.get_event_loop()

        # Run blocking inference on a thread so we don't block the event loop
        audio = await loop.run_in_executor(None, self._run_inference, text)

        chunk_samples = int(self.sample_rate * _CHUNK_MS / 1000)

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            yield _audio_to_int16_bytes(chunk), self.sample_rate
            await asyncio.sleep(0)  # yield to event loop between chunks
