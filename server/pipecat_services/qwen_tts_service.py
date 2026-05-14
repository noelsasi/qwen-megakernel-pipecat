"""
Phase C — Pipecat TTSService adapter for Qwen3-TTS.

Wraps QwenTTSBackendHF (or MK) as a Pipecat TTSService subclass.
Emits `tts-metrics` custom frames after each utterance so the React
dashboard can display TTFC, RTF, tok/s, and E2E latency live.

NOTE: The exact TTSService API varies by pipecat-ai version. If you get
AttributeError or signature mismatches, run:
    python -c "import pipecat.services.tts_service as m; print(m.__file__)"
and read that file to find the correct base class API.
"""

import time
from collections.abc import AsyncGenerator

import numpy as np

try:
    from pipecat.services.tts_service import TTSService
    from pipecat.frames.frames import (
        Frame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        ErrorFrame,
    )
except ImportError as e:
    raise ImportError(
        f"pipecat-ai not installed or wrong version: {e}\n"
        "Install with: pip install pipecat-ai[silero]"
    ) from e


class QwenTTSService(TTSService):
    """
    Pipecat TTSService backed by Qwen3-TTS.

    Args:
        backend: QwenTTSBackendHF or QwenTTSBackendMK instance (pre-loaded)
        sample_rate: audio sample rate in Hz (default 24000)
    """

    def __init__(self, backend, sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.backend = backend
        self._sample_rate = sample_rate

    async def run_tts(self, text: str) -> AsyncGenerator[Frame | None, None]:
        """
        Called by Pipecat when LLM produces text.
        Yields TTSAudioRawFrame per audio chunk, then emits tts-metrics.

        The exact signature and return type may differ across pipecat versions.
        If this breaks, read pipecat.services.tts_service.TTSService source.
        """
        t_start = time.perf_counter()
        t_first_chunk: float | None = None
        total_samples = 0

        try:
            yield TTSStartedFrame()

            async for audio_bytes, sr in self.backend.synthesize_streaming(text):
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()

                total_samples += len(audio_bytes) // 2  # int16 = 2 bytes/sample
                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=sr,
                    num_channels=1,
                )

            yield TTSStoppedFrame()

        except Exception as e:
            yield ErrorFrame(error=f"QwenTTS error: {e}")
            return

        # Emit metrics after utterance completes
        t_end = time.perf_counter()
        if t_first_chunk is not None and total_samples > 0:
            ttfc_ms = (t_first_chunk - t_start) * 1000
            gen_time = t_end - t_start
            audio_duration = total_samples / self._sample_rate
            rtf = gen_time / audio_duration if audio_duration > 0 else None
            e2e_ms = gen_time * 1000

            # Best-effort: emit custom metrics event if transport supports it
            # The React dashboard listens for "tts-metrics" via useRTVIClientEvent
            try:
                if hasattr(self, "_transport") and hasattr(self._transport, "send_message"):
                    await self._transport.send_message({
                        "type": "tts-metrics",
                        "ttfc_ms": round(ttfc_ms, 1),
                        "rtf": round(rtf, 4) if rtf is not None else None,
                        "toks_per_s": None,  # filled by benchmark, not live pipeline
                        "e2e_ms": round(e2e_ms, 1),
                    })
            except Exception:
                pass  # metrics emission is best-effort — never break the audio path
