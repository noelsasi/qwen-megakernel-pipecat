"""
Phase A/B/C — HuggingFace TTS backend.

Confirmed API (from phase_a_inspect_model.py output):
  Class:     Qwen3TTSForConditionalGeneration (qwen_tts.core.models)
  Processor: Qwen3TTSProcessor
  generate() key params:
    input_ids: list[torch.Tensor]   — list of 1D tensors, one per batch item
    non_streaming_mode: bool        — False = streaming (default), True = batch
    max_new_tokens, do_sample, temperature, top_k, top_p, repetition_penalty
    subtalker_dosample, subtalker_temperature, subtalker_top_k, subtalker_top_p
  Talker hidden size: 1024  (from lm_head weight shapes)
  Code predictor: model.talker.code_predictor (16 lm_head outputs → 16 codebooks)
  Vocab size: 3072 codec tokens
  Sample rate: 24000 Hz
  Frame rate: ~13 frames/sec (position_id_per_seconds=13)

Interface:
    backend = QwenTTSBackendHF()
    async for audio_bytes, sample_rate in backend.synthesize_streaming("Hello"):
        ...  # audio_bytes: int16 PCM, sample_rate: 24000
"""

import asyncio
import queue
import threading
import time
import numpy as np
import torch
from collections.abc import AsyncGenerator


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SAMPLE_RATE = 24000


def _to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio.tobytes()


class QwenTTSBackendHF:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda"):
        from qwen_tts.core.models import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

        print(f"[QwenTTSBackendHF] Loading {model_id} ...")
        t0 = time.perf_counter()

        self.model = Qwen3TTSForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.processor = Qwen3TTSProcessor.from_pretrained(model_id)
        self.sample_rate = SAMPLE_RATE
        self.device = device

        print(f"[QwenTTSBackendHF] Loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    def _prepare(self, text: str) -> dict:
        ids = self.processor(text, return_tensors="pt").input_ids.to(self.model.device)
        return dict(
            input_ids=[ids[0]],
            languages=["English"],
            speakers=["default"],
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_temperature=0.9,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
        )

    def run_batch(self, text: str) -> np.ndarray:
        """Blocking full inference. Returns float32 audio array."""
        with torch.inference_mode():
            outputs = self.model.generate(
                non_streaming_mode=True,
                **self._prepare(text),
            )
        audio = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()
        return audio.squeeze()

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Real streaming: generate() runs with non_streaming_mode=False (default),
        which yields audio chunks incrementally via a queue on a background thread.

        If the model's streaming API yields tensors/arrays directly, we forward them.
        Falls back to chunked batch output if streaming API is not iterable.
        """
        loop = asyncio.get_event_loop()
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def _stream():
            try:
                with torch.inference_mode():
                    result = self.model.generate(
                        non_streaming_mode=False,
                        **self._prepare(text),
                    )
                # If result is iterable (generator), forward each chunk
                if hasattr(result, "__iter__") and not isinstance(result, (torch.Tensor, np.ndarray)):
                    for chunk in result:
                        if isinstance(chunk, torch.Tensor):
                            chunk = chunk.cpu().float().numpy().squeeze()
                        q.put(chunk)
                else:
                    # non_streaming_mode=False returned full audio — chunk it
                    audio = result[0] if isinstance(result, (list, tuple)) else result
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().float().numpy()
                    audio = audio.squeeze()
                    chunk_samples = SAMPLE_RATE // 10  # 100ms chunks
                    for i in range(0, len(audio), chunk_samples):
                        q.put(audio[i : i + chunk_samples])
            except Exception as e:
                q.put(e)
            finally:
                q.put(_DONE)

        thread = threading.Thread(target=_stream, daemon=True)
        thread.start()

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield _to_int16_bytes(item), self.sample_rate
