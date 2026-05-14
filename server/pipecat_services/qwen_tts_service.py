"""
Pipecat 1.1.0 TTSService adapter for Qwen3-TTS (and LocalDevTTSBackend).

Verified API (pipecat-ai 1.1.0):
  - run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]
  - TTSStartedFrame(context_id=...)
  - TTSAudioRawFrame(audio, sample_rate, num_channels, context_id=...)
  - TTSStoppedFrame(context_id=...)
"""

import time
from collections.abc import AsyncGenerator

import numpy as np

from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)


class QwenTTSService(TTSService):
    """
    Pipecat TTSService adapter.
    Works with QwenTTSBackendHF, QwenTTSBackendMK, or LocalDevTTSBackend.
    """

    def __init__(self, backend, sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.backend = backend
        self._sample_rate = sample_rate

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        t_start = time.perf_counter()
        t_first_chunk: float | None = None
        total_samples = 0

        try:
            yield TTSStartedFrame(context_id=context_id)

            async for audio_bytes, sr in self.backend.synthesize_streaming(text):
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()

                total_samples += len(audio_bytes) // 2  # int16 = 2 bytes/sample
                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=sr,
                    num_channels=1,
                    context_id=context_id,
                )

            yield TTSStoppedFrame(context_id=context_id)

        except Exception as e:
            yield ErrorFrame(error=f"QwenTTS error: {e}")
            return

        # Log metrics
        t_end = time.perf_counter()
        if t_first_chunk is not None and total_samples > 0:
            ttfc_ms = (t_first_chunk - t_start) * 1000
            gen_time = t_end - t_start
            audio_dur = total_samples / self._sample_rate
            rtf = gen_time / audio_dur if audio_dur > 0 else 0.0
            import logging
            logging.getLogger(__name__).info(
                f"TTS metrics — TTFC={ttfc_ms:.0f}ms RTF={rtf:.3f} "
                f"audio={audio_dur*1000:.0f}ms e2e={gen_time*1000:.0f}ms"
            )
