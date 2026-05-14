"""
Phase D — Megakernel TTS backend.

Replaces only the talker autoregressive decode loop with the CUDA megakernel.
Code predictor + vocoder remain as HF operations.

Same interface as QwenTTSBackendHF — drop-in swap.

NOTE: This is a skeleton. Fill in the blanks after:
  - phase_a_inspect_model.py confirms: talker module path, weight key names
  - phase_d_compat_check.py confirms: kernel constants match model config
  - qwen_megakernel/ is cloned and built (make kernel)

Usage:
    backend = QwenTTSBackendMK(
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        megakernel_path="./qwen_megakernel",
    )
    async for audio_bytes, sr in backend.synthesize_streaming("Hello"):
        ...
"""

import asyncio
import sys
import time
import numpy as np
import torch
from collections.abc import AsyncGenerator


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Fill these in after phase_a_inspect_model.py + phase_b_streaming_probe.py
# These are placeholders — do NOT use until verified.
_EOS_TOKEN_ID = None        # e.g. 2 or model-specific — verify from A.3
_FRAME_TOKENS = None        # how many talker tokens = one vocoder call — verify from B.1
_MAX_DECODE_STEPS = 2048    # max tokens before forcing EOS


def _audio_to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio.tobytes()


def _extract_talker_weights(hf_model) -> dict:
    """
    Extract talker weights in the format expected by qwen_megakernel/model.py.

    PLACEHOLDER — actual key mapping requires:
      1. Running phase_a_inspect_model.py to get HF weight names
      2. Reading qwen_megakernel/model.py to get expected keys
      3. Building the mapping below

    Until then, this raises NotImplementedError to make the gap explicit.
    """
    raise NotImplementedError(
        "Weight extraction mapping not yet implemented.\n"
        "Steps:\n"
        "  1. Run: python scripts/phase_a_inspect_model.py | grep 'talker'\n"
        "  2. Read: qwen_megakernel/qwen_megakernel/model.py\n"
        "  3. Fill in the key mapping in _extract_talker_weights()"
    )

    # Example (speculative — replace with actual keys from inspection):
    # talker = hf_model.talker  # verify path from A.3
    # state = talker.state_dict()
    # return {
    #     "embed_weight": state["model.embed_tokens.weight"],
    #     "norm_weight": state["model.norm.weight"],
    #     # ... per layer weights ...
    # }


class QwenTTSBackendMK:
    def __init__(self, model_id: str = MODEL_ID, megakernel_path: str = "./qwen_megakernel"):
        sys.path.insert(0, megakernel_path)

        from transformers import AutoModel, AutoProcessor

        print(f"[QwenTTSBackendMK] Loading {model_id} ...")
        t0 = time.perf_counter()

        self.hf_model = AutoModel.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.hf_model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.sample_rate = getattr(self.processor, "sampling_rate", 24000) or 24000

        # Extract weights and build megakernel decoder
        print("[QwenTTSBackendMK] Extracting talker weights for megakernel ...")
        weights = _extract_talker_weights(self.hf_model)

        from qwen_megakernel import Decoder
        self.mk_decoder = Decoder(weights=weights)

        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[QwenTTSBackendMK] Ready in {load_ms:.0f}ms")

    def _prefill(self, text: str):
        """
        Run prefill pass to get initial KV cache and first token.

        PLACEHOLDER — exact call depends on whether the HF model exposes
        a separate prefill path (to be determined from A.3 inspection).
        """
        raise NotImplementedError(
            "Prefill path not yet implemented.\n"
            "After phase_a_inspect_model.py, check if model has:\n"
            "  - model.talker_prefill(input_ids) → (first_token, kv_cache)\n"
            "  - model.talker.forward(input_ids, use_cache=True) → BaseModelOutput\n"
            "Fill this in once the actual API is confirmed."
        )

    def _run_code_predictor(self, codec_tokens: list[int]) -> torch.Tensor:
        """Run code predictor (still HF) to expand codec tokens → codebook tokens."""
        raise NotImplementedError(
            "Fill in after phase_a_inspect_model.py identifies the code predictor module path.\n"
            "Expected: model.code_predictor.forward(codec_tokens) or similar."
        )

    def _run_vocoder(self, codebook_tokens: torch.Tensor) -> np.ndarray:
        """Run vocoder/DAC (still HF) to convert codebook tokens → audio waveform."""
        raise NotImplementedError(
            "Fill in after phase_a_inspect_model.py identifies the vocoder module path.\n"
            "Expected: model.vocoder.decode(codebook_tokens) → float32 waveform."
        )

    def _decode_loop(self, text: str):
        """
        Megakernel decode loop — yields (audio_bytes, sample_rate) per frame.

        This is a synchronous generator that blocks; called via run_in_executor.
        """
        # Step 1: Prefill (still HF)
        first_token, kv_cache = self._prefill(text)

        # Step 2: Decode loop with megakernel
        codec_tokens = []
        last_token = first_token

        for step in range(_MAX_DECODE_STEPS):
            token_id = self.mk_decoder.step(last_token)
            last_token = token_id

            if _EOS_TOKEN_ID is not None and token_id == _EOS_TOKEN_ID:
                break

            codec_tokens.append(token_id)

            # Every _FRAME_TOKENS tokens, decode a frame
            if _FRAME_TOKENS is not None and len(codec_tokens) % _FRAME_TOKENS == 0:
                frame = codec_tokens[-_FRAME_TOKENS:]
                codebook_tokens = self._run_code_predictor(frame)
                audio_chunk = self._run_vocoder(codebook_tokens)
                yield _audio_to_int16_bytes(audio_chunk), self.sample_rate

        # Flush any remaining tokens
        if codec_tokens and (_FRAME_TOKENS is None or len(codec_tokens) % _FRAME_TOKENS != 0):
            codebook_tokens = self._run_code_predictor(codec_tokens)
            audio_chunk = self._run_vocoder(codebook_tokens)
            yield _audio_to_int16_bytes(audio_chunk), self.sample_rate

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) per decoded audio frame.
        Uses megakernel for talker decode; HF for code predictor + vocoder.
        """
        loop = asyncio.get_event_loop()

        # Run the blocking decode loop in an executor thread
        # and surface each chunk as it arrives
        queue: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for chunk in self._decode_loop(text):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        executor_future = loop.run_in_executor(None, producer)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        await executor_future
