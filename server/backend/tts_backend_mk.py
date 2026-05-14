"""
Phase D — Megakernel TTS backend.

Replaces the talker autoregressive decode loop with the CUDA megakernel.
Code predictor + vocoder remain as HF (unchanged).

Confirmed weight shapes (from state_dict inspection 2026-05-14):
  q_proj:           [2048, 1024]  (16 heads × 128 HEAD_DIM, in 1024)
  k_proj:           [1024, 1024]  (8 KV heads × 128, in 1024)
  v_proj:           [1024, 1024]
  o_proj:           [1024, 2048]  (in 2048 = 16 heads × 128, out 1024)
  q_norm/k_norm:    [128]
  gate/up_proj:     [3072, 1024]
  down_proj:        [1024, 3072]
  input/post_norm:  [1024]
  codec_embedding:  [3072, 1024]  → embed_weight (input tokens)
  codec_head:       [3072, 1024]  → lm_head_weight (output logits)
  model.norm:       [1024]        → final_norm_weight

Changes vs original megakernel model.py:
  - LDG_VOCAB_SIZE: 151936 → 3072  (kernel.cu — requires rebuild)
  - MAX_SEQ_LEN: 2048 → 32768      (Python constant — no rebuild)
  - rope_theta: 10000 → 1,000,000  (Python — no rebuild)
  - RoPE tables: standard → interleaved MRope sections [24,20,20]
  - embed_weight: text_embedding → codec_embedding [3072, 1024]
  - lm_head_weight: tied embed → codec_head [3072, 1024] (untied)

PREREQUISITES before using this backend:
  1. Patch kernel constant:
       sed -i 's/LDG_VOCAB_SIZE = 151936/LDG_VOCAB_SIZE = 3072/' qwen_megakernel/csrc/kernel.cu
  2. Build:
       cd qwen_megakernel && pip install -e . && cd ..
  3. Verify:
       python -c "import qwen_megakernel; print('megakernel ok')"

Interface (drop-in for QwenTTSBackendHF):
    backend = QwenTTSBackendMK()
    async for audio_bytes, sample_rate in backend.synthesize_streaming("Hello"):
        ...
"""

import asyncio
import struct
import sys
import time
import numpy as np
import torch
from collections.abc import AsyncGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 28
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HIDDEN_SIZE = 1024
VOCAB_SIZE = 3072        # codec tokens — kernel.cu LDG_VOCAB_SIZE must match
MAX_SEQ_LEN = 32768      # Python only — no kernel rebuild needed
ROPE_THETA = 1_000_000.0
MROPE_SECTION = [24, 20, 20]  # of HEAD_DIM//2=64; sums to 64

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SAMPLE_RATE = 24000
EOS_TOKEN_ID = 2150  # confirmed from generation warning


# ---------------------------------------------------------------------------
# MRope cos/sin table builder
# ---------------------------------------------------------------------------

def _build_mrope_tables(
    max_seq_len: int = MAX_SEQ_LEN,
    head_dim: int = HEAD_DIM,
    theta: float = ROPE_THETA,
    mrope_section: list = MROPE_SECTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build [max_seq_len, head_dim] bfloat16 cos/sin tables for interleaved MRope.

    MRope interleaved layout (Qwen3-VL/TTS pattern):
      - half_dim = head_dim // 2 = 64 frequency slots
      - mrope_section [24, 20, 20] partitions these 64 slots across 3 position streams
      - Interleaved: T0 H0 A0 T1 H1 A1 ... (round-robin across streams)
      - During TTS autoregressive decode all 3 streams share the same step index
      - Final table uses .repeat(1,2) to fill full head_dim (standard RoPE convention)
    """
    half_dim = head_dim // 2  # 64
    assert sum(mrope_section) == half_dim, \
        f"mrope_section {mrope_section} must sum to HEAD_DIM//2={half_dim}"

    # inv_freq per section — same theta, section-relative indexing
    inv_freqs = []
    for section_dim in mrope_section:
        idx = torch.arange(0, section_dim, 2, dtype=torch.float32)
        inv_freqs.append(1.0 / (theta ** (idx / section_dim)))

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    section_freqs = [torch.outer(positions, inv_f) for inv_f in inv_freqs]
    # shapes: [max_seq_len, 12], [max_seq_len, 10], [max_seq_len, 10]

    section_pairs = [s // 2 for s in mrope_section]  # [12, 10, 10]
    max_pairs = max(section_pairs)

    interleaved = []
    for pair_idx in range(max_pairs):
        for axis_idx, n_pairs in enumerate(section_pairs):
            if pair_idx < n_pairs:
                interleaved.append(section_freqs[axis_idx][:, pair_idx : pair_idx + 1])

    all_freqs = torch.cat(interleaved, dim=1)  # [max_seq_len, 32]

    cos_table = torch.cos(all_freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(all_freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table  # each [max_seq_len, head_dim]


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

_LAYER_KEYS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def _extract_talker_weights(hf_state: dict) -> dict:
    """
    Remap HF talker state_dict to megakernel weight format.
    Strip 'talker.' prefix; codec_embedding → embed_weight; codec_head → lm_head_weight.
    """
    layer_weights = []
    for i in range(NUM_LAYERS):
        prefix = f"talker.model.layers.{i}."
        for key in _LAYER_KEYS:
            full_key = prefix + key
            if full_key not in hf_state:
                raise KeyError(f"Missing talker weight: {full_key}")
            layer_weights.append(hf_state[full_key].cuda().contiguous())

    return dict(
        embed_weight=hf_state["talker.model.codec_embedding.weight"].cuda().contiguous(),
        layer_weights=layer_weights,
        final_norm_weight=hf_state["talker.model.norm.weight"].cuda().contiguous(),
        lm_head_weight=hf_state["talker.codec_head.weight"].cuda().contiguous(),
    )


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    ptr_size = 8
    n_ptrs = len(_LAYER_KEYS)
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


# ---------------------------------------------------------------------------
# Megakernel decoder
# ---------------------------------------------------------------------------

class _MKDecoder:
    """Stateful single-token decoder using the CUDA megakernel."""

    def __init__(self, weights: dict):
        self._weights = weights  # keep references — prevents GC of GPU tensors
        self._position = 0
        self._attn_scale = HEAD_DIM ** -0.5

        self._cos_table, self._sin_table = _build_mrope_tables()
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        # Working buffers
        self._hidden = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._act = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._res = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._q = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._k = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._v = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._attn_out = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._mlp_inter = torch.zeros(HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda")
        self._norm_out = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        n_attn = NUM_Q_HEADS
        self._bmax_vals = torch.full((n_attn,), float("-inf"), device="cuda")
        self._bmax_idxs = torch.zeros(n_attn, dtype=torch.int32, device="cuda")
        self._d_position = torch.zeros(1, dtype=torch.int32, device="cuda")

        self._step_fn = torch.ops.qwen_megakernel_C.step

    def step(self, token_id: int) -> int:
        self._d_position[0] = self._position
        self._step_fn(
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._d_position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        # Output token is the argmax of lm_head(norm_out)
        # The kernel writes logits into a buffer — read from norm_out and project
        with torch.no_grad():
            logits = self._norm_out.float() @ self._lm_head_weight.float().T
        return int(logits.argmax().item())

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._bmax_vals.fill_(float("-inf"))
        self._bmax_idxs.zero_()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class QwenTTSBackendMK:
    """
    Megakernel TTS backend — drop-in for QwenTTSBackendHF.

    Architecture:
      - Prefill: HF model processes text input → produces first codec token + KV cache
      - Decode loop: megakernel steps one codec token at a time
      - Code predictor + vocoder: HF model, unchanged
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        megakernel_path: str = "./qwen_megakernel",
    ):
        sys.path.insert(0, megakernel_path)

        from qwen_tts import Qwen3TTSModel

        print(f"[QwenTTSBackendMK] Loading HF model ...")
        t0 = time.perf_counter()

        self._hf = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        self._hf.model.eval()
        self.sample_rate = SAMPLE_RATE

        print(f"[QwenTTSBackendMK] Extracting weights + building decoder ...")
        state = self._hf.model.state_dict()
        weights = _extract_talker_weights(state)
        self._decoder = _MKDecoder(weights)

        print(f"[QwenTTSBackendMK] Ready in {(time.perf_counter()-t0)*1000:.0f}ms")

    def _run_batch(self, text: str) -> np.ndarray:
        """
        Fallback: full HF inference (used until megakernel decode loop is wired).
        Remove once Phase D decode loop integration is complete.
        """
        wavs, sr = self._hf.generate_custom_voice(
            text=text,
            language="English",
            speaker="Ryan",
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.9,
        )
        return np.array(wavs[0], dtype=np.float32).squeeze()

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) in ~100ms chunks.
        Currently falls back to HF until megakernel decode loop is wired.
        """
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(None, self._run_batch, text)

        chunk_samples = int(self.sample_rate * 100 / 1000)
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            chunk_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            yield chunk_bytes, self.sample_rate
            await asyncio.sleep(0)
