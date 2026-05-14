"""
Phase C — Pipecat TTSService adapter for Qwen3-TTS.

Wraps QwenTTSBackendHF (or MK) as a Pipecat TTSService subclass.
The actual TTSService API is read from source at runtime — see C.1 in the plan.

NOTE: The exact TTSService API varies by pipecat-ai version. If you get
AttributeError or signature mismatches, run:
    python -c "import pipecat.services.tts_service as m; print(m.__file__)"
and read that file to find the correct base class API.
"""

import asyncio
from collections.abc import AsyncGenerator

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
        Yields TTSAudioRawFrame per audio chunk from the backend.

        The exact signature and return type may differ across pipecat versions.
        If this breaks, read pipecat.services.tts_service.TTSService source.
        """
        try:
            yield TTSStartedFrame()

            async for audio_bytes, sr in self.backend.synthesize_streaming(text):
                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=sr,
                    num_channels=1,
                )

            yield TTSStoppedFrame()

        except Exception as e:
            yield ErrorFrame(error=f"QwenTTS error: {e}")
