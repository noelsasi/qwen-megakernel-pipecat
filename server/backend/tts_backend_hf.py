"""
Phase A/B/C — HuggingFace TTS backend.

Confirmed API (from QwenLM/Qwen3-TTS repo):
  - Class: Qwen3TTSModel from qwen_tts (high-level wrapper)
  - Call: model.generate_custom_voice(text, language, speaker, ...)
  - Returns: (wavs: list[np.ndarray], sr: int) — sr is 12000 Hz
  - Streaming: model.generate_custom_voice() is blocking (no native chunk streaming)
  - We fake-stream by chunking the returned audio — Phase B will investigate
    whether the underlying generate() can be hooked for real streaming.

  Valid speakers: Ryan, Aiden (EN), Vivian, Serena, Uncle_Fu, Dylan, Eric (ZH),
                  Ono_Anna (JA), Sohee (KO)
  Valid languages: English, Chinese, Japanese, Korean, German, French,
                   Russian, Portuguese, Spanish, Italian

Interface:
    backend = QwenTTSBackendHF()
    async for audio_bytes, sample_rate in backend.synthesize_streaming("Hello"):
        ...  # audio_bytes: int16 PCM, sample_rate: 12000
"""

import asyncio
import time
import numpy as np
import torch
from collections.abc import AsyncGenerator


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
CHUNK_MS = 100  # chunk size for fake-streaming


def _to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio.tobytes()


class QwenTTSBackendHF:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        speaker: str = "Ryan",
        language: str = "English",
    ):
        from qwen_tts import Qwen3TTSModel

        print(f"[QwenTTSBackendHF] Loading {model_id} ...")
        t0 = time.perf_counter()

        self.model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self.speaker = speaker
        self.language = language
        self.sample_rate = 24000  # confirmed: "12Hz" in model name = codec frame rate, audio is 24kHz

        print(f"[QwenTTSBackendHF] Loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
        print(f"[QwenTTSBackendHF] Speaker={speaker} Language={language}")

    def run_batch(self, text: str) -> tuple[np.ndarray, int]:
        """Blocking full inference. Returns (float32 audio, sample_rate)."""
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
        )
        self.sample_rate = sr
        audio = np.array(wavs[0], dtype=np.float32).squeeze()
        return audio, sr

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) in ~100ms chunks.

        Currently fake-streams: runs full inference then chunks output.
        Real streaming requires hooking into the underlying generate() loop —
        deferred to Phase B investigation.
        """
        loop = asyncio.get_event_loop()
        audio, sr = await loop.run_in_executor(None, self.run_batch, text)

        chunk_samples = int(sr * CHUNK_MS / 1000)
        for i in range(0, len(audio), chunk_samples):
            yield _to_int16_bytes(audio[i : i + chunk_samples]), sr
            await asyncio.sleep(0)
