"""
Local dev TTS backend — uses edge-tts (Microsoft, no key, no GPU).

Same interface as QwenTTSBackendHF: synthesize_streaming() yields
(audio_bytes: bytes, sample_rate: int) so the Pipecat service layer
is identical between dev and production.

edge-tts produces 24000 Hz mono PCM via in-memory synthesis.

Usage:
    backend = LocalDevTTSBackend()
    async for audio_bytes, sr in backend.synthesize_streaming("Hello"):
        ...  # PCM int16 bytes, sr=24000
"""

import asyncio
import io
import struct
from collections.abc import AsyncGenerator

import edge_tts
import numpy as np

SAMPLE_RATE = 24000
CHUNK_MS = 100
VOICE = "en-US-GuyNeural"  # swap to any edge-tts voice


def _pcm_from_mp3_bytes(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 bytes → float32 PCM array using stdlib only."""
    # edge-tts returns MP3; we need raw PCM for Pipecat
    # Use pydub if available, else fall back to soundfile+ffmpeg
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1)
        raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        return raw
    except ImportError:
        pass

    try:
        import soundfile as sf
        import subprocess
        # Use ffmpeg to convert MP3 → raw PCM via stdout
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ar", str(SAMPLE_RATE),
             "-ac", "1", "pipe:1", "-loglevel", "quiet"],
            input=mp3_bytes, capture_output=True
        )
        if proc.returncode == 0 and proc.stdout:
            raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return raw
    except (FileNotFoundError, Exception):
        pass

    raise RuntimeError(
        "Cannot decode MP3: install pydub (`pip install pydub`) or ffmpeg. "
        "On Mac: `brew install ffmpeg`"
    )


class LocalDevTTSBackend:
    """
    CPU-only TTS backend for local development.
    Uses edge-tts (Microsoft Neural voices, no API key).
    Drop-in replacement for QwenTTSBackendHF.
    """

    def __init__(self, voice: str = VOICE, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._voice = voice
        print(f"[LocalDevTTSBackend] voice={voice} sr={sample_rate}")

    async def _synthesize_raw(self, text: str) -> np.ndarray:
        """Run edge-tts and return float32 PCM array."""
        communicate = edge_tts.Communicate(text, self._voice)
        mp3_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])
        mp3_bytes = mp3_buf.getvalue()
        if not mp3_bytes:
            return np.zeros(0, dtype=np.float32)
        return _pcm_from_mp3_bytes(mp3_bytes)

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) in ~100ms chunks.
        Same interface as QwenTTSBackendHF.synthesize_streaming().
        """
        audio = await self._synthesize_raw(text)
        if len(audio) == 0:
            return

        chunk_samples = int(self.sample_rate * CHUNK_MS / 1000)
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            chunk_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            yield chunk_bytes, self.sample_rate
            await asyncio.sleep(0)
