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
    """
    Stateful single-token decoder using the CUDA megakernel.

    Confirmed ops from torch_bindings.cpp:
      decode(output_token, input_token_id, embed_weight, layer_weights_packed,
             final_norm_weight, lm_head_weight, cos_table, sin_table,
             k_cache, v_cache, hidden_buffer, activations, residual,
             q, k, v, attn_out, mlp_intermediate, normalized,
             block_max_vals, block_max_idxs,
             num_layers, position, max_seq_len, attn_scale) -> ()

      generate_nosync(first_token_id, num_steps, ..same buffers..,
                      num_layers, start_position, max_seq_len, attn_scale) -> Tensor[num_steps]
    """

    def __init__(self, weights: dict):
        self._weights = weights  # keep references — prevents GC of GPU tensors
        self._position = 0
        self._attn_scale = float(HEAD_DIM ** -0.5)

        self._cos_table, self._sin_table = _build_mrope_tables()
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        # Working buffers (exact names match torch_bindings.cpp parameter names)
        self._hidden = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._activations = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._residual = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._q = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._k = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._v = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._attn_out = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self._mlp_intermediate = torch.zeros(HIDDEN_SIZE * 2, dtype=torch.bfloat16, device="cuda")
        self._normalized = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        # block_max_vals/idxs: one entry per attention block (LDG_ATTN_BLOCKS=8 from build.py)
        n_attn_blocks = 8
        self._block_max_vals = torch.full((n_attn_blocks,), float("-inf"), dtype=torch.float32, device="cuda")
        self._block_max_idxs = torch.zeros(n_attn_blocks, dtype=torch.int32, device="cuda")
        # output_token: single int32 tensor written by decode op
        self._output_token = torch.zeros(1, dtype=torch.int32, device="cuda")

        self._decode_op = torch.ops.qwen_megakernel_C.decode
        self._generate_op = torch.ops.qwen_megakernel_C.generate_nosync

    def _call_args(self):
        """Common buffer args shared by decode and generate_nosync."""
        return (
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._activations,
            self._residual,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_intermediate,
            self._normalized,
            self._block_max_vals,
            self._block_max_idxs,
        )

    def step(self, token_id: int) -> int:
        """Run one decode step. Returns next token id."""
        self._decode_op(
            self._output_token,
            token_id,
            *self._call_args(),
            NUM_LAYERS,
            self._position,   # plain int
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return int(self._output_token.item())

    def generate_n(self, first_token_id: int, num_steps: int) -> list[int]:
        """Run num_steps decode steps without sync. Returns list of token ids."""
        output = self._generate_op(
            first_token_id,
            num_steps,
            *self._call_args(),
            NUM_LAYERS,
            self._position,   # start_position
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += num_steps
        return output.cpu().tolist()

    def load_kv_cache_from_hf(self, past_key_values, prefill_len: int):
        """
        Copy HF DynamicCache into megakernel's pre-allocated KV cache tensors.

        HF format: list of (k, v) tuples, each [1, NUM_KV_HEADS, seq_len, HEAD_DIM]
        Megakernel format: [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]

        Called after HF prefill, before megakernel decode loop starts.
        """
        self._k_cache.zero_()
        self._v_cache.zero_()
        for layer_idx, (k, v) in enumerate(past_key_values):
            # k, v: [1, NUM_KV_HEADS, seq_len, HEAD_DIM]
            seq = k.shape[2]
            self._k_cache[layer_idx, :, :seq, :] = k[0].to(torch.bfloat16)
            self._v_cache[layer_idx, :, :seq, :] = v[0].to(torch.bfloat16)
        self._position = prefill_len
        self._block_max_vals.fill_(float("-inf"))
        self._block_max_idxs.zero_()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._block_max_vals.fill_(float("-inf"))
        self._block_max_idxs.zero_()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class QwenTTSBackendMK:
    """
    Megakernel TTS backend — drop-in for QwenTTSBackendHF.

    Strategy: patch talker.model.generate() so each decode step uses the
    megakernel instead of HF transformers. The generate() wrapper returns
    the same tensor format HF expects, so code_predictor + vocoder run
    unchanged downstream. This avoids the vocoder-API-discovery problem entirely.

    Prefill flow:
      1. HF calls talker.model(inputs_embeds=..., use_cache=True) for prefill
         → we let that run normally via HF (gets past_key_values)
      2. HF then calls talker.model.generate() for the decode loop
         → we intercept: copy KV cache into megakernel tensors, run decode
           steps via megakernel, return the token tensor to HF
      3. HF passes the token tensor to code_predictor + vocoder as normal

    KV cache layout compatibility confirmed (findings.md):
      HF:          list of 28 × (k[1,8,seq,128], v[1,8,seq,128])
      Megakernel:  [28, 8, MAX_SEQ_LEN, 128] pre-allocated, written in-place
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        megakernel_path: str = "./qwen_megakernel",
        speaker: str = "Ryan",
        language: str = "English",
    ):
        sys.path.insert(0, megakernel_path)

        # Trigger JIT compilation and op registration before any torch.ops calls
        from qwen_megakernel.build import get_extension
        get_extension()

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
        self._speaker = speaker
        self._language = language

        print(f"[QwenTTSBackendMK] Extracting weights + building decoder ...")
        state = self._hf.model.state_dict()
        weights = _extract_talker_weights(state)
        self._decoder = _MKDecoder(weights)
        # mk_decoder exposed for benchmark access
        self.mk_decoder = self._decoder

        print(f"[QwenTTSBackendMK] Ready in {(time.perf_counter()-t0)*1000:.0f}ms")

    def _run_with_megakernel_decode(self, text: str) -> np.ndarray:
        """
        Run generate_custom_voice with the talker decode loop replaced by megakernel.

        We patch talker.model.generate() so HF's pipeline calls our function
        instead of its autoregressive generate loop. HF's generate() is only
        called during the decode phase (after prefill), so patching it here
        is safe — prefill runs via talker.model() forward pass as normal.

        The patch function receives (input_ids, past_key_values, ...) from HF,
        copies the KV cache into megakernel tensors, runs megakernel decode,
        and returns a token_ids tensor in the shape HF expects.
        """
        decoder = self._decoder
        orig_generate = self._hf.model.talker.model.generate

        def _mk_generate(input_ids=None, past_key_values=None,
                         max_new_tokens=4096, **kwargs):
            # input_ids: [1, prefill_len] (last token after prefill)
            # past_key_values: HF DynamicCache from prefill forward pass
            if past_key_values is None:
                # Fallback: no KV cache available — run HF normally
                return orig_generate(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )

            # Extract KV cache — DynamicCache exposes .key_cache / .value_cache
            # or can be iterated as list of (k, v) tuples
            try:
                pkv_list = [
                    (past_key_values.key_cache[i], past_key_values.value_cache[i])
                    for i in range(len(past_key_values.key_cache))
                ]
            except AttributeError:
                # Older pipecat/transformers: past_key_values is list of tuples
                pkv_list = list(past_key_values)

            prefill_len = pkv_list[0][0].shape[2]
            decoder.reset()
            decoder.load_kv_cache_from_hf(pkv_list, prefill_len)

            # Get first decode token from input_ids (last prefill token)
            first_token = int(input_ids[0, -1].item())
            print(f"[MK] generate() intercepted — prefill={prefill_len}, first_tok={first_token}")

            # Megakernel decode loop
            tokens = []
            current_token = first_token
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    next_token = decoder.step(current_token)
                    if next_token == EOS_TOKEN_ID:
                        break
                    tokens.append(next_token)
                    current_token = next_token

            print(f"[MK] Decoded {len(tokens)} codec tokens")
            # Return shape [1, num_tokens] int64 — same as HF generate() output
            return torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)

        # Install patch, run, restore
        self._hf.model.talker.model.generate = _mk_generate
        try:
            with torch.inference_mode():
                wavs, sr = self._hf.generate_custom_voice(
                    text=text,
                    language=self._language,
                    speaker=self._speaker,
                    max_new_tokens=4096,
                    do_sample=False,  # megakernel uses argmax internally
                )
        finally:
            self._hf.model.talker.model.generate = orig_generate

        audio = np.array(wavs[0], dtype=np.float32).squeeze()
        return audio

    def _run_batch(self, text: str) -> np.ndarray:
        """Full HF inference fallback (no megakernel)."""
        wavs, sr = self._hf.generate_custom_voice(
            text=text,
            language=self._language,
            speaker=self._speaker,
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
        Uses megakernel decode if available; falls back to HF on error.
        """
        loop = asyncio.get_event_loop()
        try:
            audio = await loop.run_in_executor(None, self._run_with_megakernel_decode, text)
        except Exception as e:
            print(f"[QwenTTSBackendMK] Megakernel decode failed ({e}), falling back to HF")
            audio = await loop.run_in_executor(None, self._run_batch, text)

        chunk_samples = int(self.sample_rate * 100 / 1000)
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            chunk_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            yield chunk_bytes, self.sample_rate
            await asyncio.sleep(0)
