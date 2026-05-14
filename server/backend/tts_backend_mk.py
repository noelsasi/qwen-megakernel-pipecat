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
# Note: talker uses standard 1D RoPE (position_ids shape [1, seq_len]),
# NOT interleaved MRope — confirmed from attention layer inspection.

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SAMPLE_RATE = 24000
EOS_TOKEN_ID = 2150  # confirmed from generation warning


# ---------------------------------------------------------------------------
# MRope cos/sin table builder
# ---------------------------------------------------------------------------

def _build_rope_tables(
    max_seq_len: int = MAX_SEQ_LEN,
    head_dim: int = HEAD_DIM,
    theta: float = ROPE_THETA,
    inv_freq: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build [max_seq_len, head_dim] bfloat16 cos/sin tables matching HF's RoPE.

    Confirmed from inspection (2026-05-14): talker uses standard 1D position_ids
    [0,1,...,N-1] with Qwen3TTSTalkerRotaryEmbedding (theta=1e6, standard RoPE).
    NOT interleaved MRope — position_ids shape is [1, seq_len], not [3, 1, seq_len].

    If inv_freq is provided (copied from model.rotary_emb.inv_freq), use it directly
    to guarantee exact match with HF. Otherwise compute from theta.
    """
    if inv_freq is not None:
        inv_f = inv_freq.float().cpu()
    else:
        idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_f = 1.0 / (theta ** (idx / head_dim))  # [head_dim//2]

    positions = torch.arange(max_seq_len, dtype=torch.float32)  # [max_seq_len]
    freqs = torch.outer(positions, inv_f)  # [max_seq_len, head_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]

    cos_table = torch.cos(emb).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(emb).to(torch.bfloat16).cuda().contiguous()
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

    Confirmed ops (from server verification 2026-05-14):
      torch.ops.qwen_megakernel_C.decode(
          output_token, input_token_id, embed_weight, layer_weights_packed,
          final_norm_weight, lm_head_weight, cos_table, sin_table,
          k_cache, v_cache, hidden_buffer, activations, residual,
          q, k, v, attn_out, mlp_intermediate, normalized,
          block_max_vals, block_max_idxs,
          num_layers, position, max_seq_len, attn_scale) -> ()

      generate_nosync was NOT present in the built extension — generate_n() falls
      back to calling step() in a loop.
    """

    def __init__(self, weights: dict, inv_freq: torch.Tensor = None):
        self._weights = weights  # keep references — prevents GC of GPU tensors
        self._position = 0
        self._attn_scale = float(HEAD_DIM ** -0.5)

        self._cos_table, self._sin_table = _build_rope_tables(inv_freq=inv_freq)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        # Dtypes from kernel.cu launch_ldg_decode_direct (lines 1197-1201):
        #   hidden_buffer       → bfloat16
        #   g_activations       → float32
        #   g_residual          → float32
        #   g_q, g_k, g_v       → float32
        #   g_attn_out          → float32
        #   g_mlp_intermediate  → float32
        #   g_normalized        → float32
        self._hidden = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._activations = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self._residual = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self._q = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.float32, device="cuda")
        self._k = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32, device="cuda")
        self._v = torch.zeros(NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32, device="cuda")
        self._attn_out = torch.zeros(NUM_Q_HEADS * HEAD_DIM, dtype=torch.float32, device="cuda")
        self._mlp_intermediate = torch.zeros(VOCAB_SIZE, dtype=torch.float32, device="cuda")
        self._normalized = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        # block_max_vals/idxs are read by ldg_lm_head_fused over LDG_LM_NUM_BLOCKS=1184 entries
        # (kernel.cu line 1157: for i in range(num_blocks) where num_blocks=LDG_LM_NUM_BLOCKS)
        LDG_LM_NUM_BLOCKS = 1184
        self._block_max_vals = torch.full((LDG_LM_NUM_BLOCKS,), float("-inf"), dtype=torch.float32, device="cuda")
        self._block_max_idxs = torch.zeros(LDG_LM_NUM_BLOCKS, dtype=torch.int32, device="cuda")
        # output_token: single int32 tensor written by decode op
        self._output_token = torch.zeros(1, dtype=torch.int32, device="cuda")

        self._decode_op = torch.ops.qwen_megakernel_C.decode
        # generate_nosync is not present in all builds — use step() loop instead

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
        result = int(self._output_token.item())
        if self._position < 3:  # log first 2 steps only
            print(f"[MK] step: in={token_id} pos={self._position} out={result} valid={0<=result<VOCAB_SIZE}")
        self._position += 1
        return result

    def generate_n(self, first_token_id: int, num_steps: int) -> list[int]:
        """Run num_steps decode steps. Returns list of token ids."""
        tokens = []
        current = first_token_id
        for _ in range(num_steps):
            current = self.step(current)
            tokens.append(current)
        return tokens

    def load_kv_cache_from_hf(self, past_key_values) -> int:
        """
        Copy HF DynamicCache into megakernel's pre-allocated KV cache tensors.

        HF format (confirmed 2026-05-14): DynamicCache with .layers list of
        DynamicLayer objects, each with .keys [1, NUM_KV_HEADS, seq_len, HEAD_DIM]
        and .values [1, NUM_KV_HEADS, seq_len, HEAD_DIM].

        Megakernel format: [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]

        Returns prefill_len (seq_len from layer 0).
        """
        self._k_cache.zero_()
        self._v_cache.zero_()
        for layer_idx, layer in enumerate(past_key_values.layers):
            k = layer.keys   # [1, NUM_KV_HEADS, seq_len, HEAD_DIM]
            v = layer.values
            seq = k.shape[2]
            self._k_cache[layer_idx, :, :seq, :] = k[0].to(torch.bfloat16)
            self._v_cache[layer_idx, :, :seq, :] = v[0].to(torch.bfloat16)
        prefill_len = past_key_values.layers[0].keys.shape[2]
        self._position = prefill_len
        self._block_max_vals.fill_(float("-inf"))
        self._block_max_idxs.zero_()
        return prefill_len

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._block_max_vals.fill_(float("-inf"))
        self._block_max_idxs.zero_()
        # Zero all working buffers — stale values from previous calls cause garbage tokens
        self._hidden.zero_()
        self._activations.zero_()
        self._residual.zero_()
        self._q.zero_()
        self._k.zero_()
        self._v.zero_()
        self._attn_out.zero_()
        self._mlp_intermediate.zero_()
        self._normalized.zero_()
        self._output_token.zero_()


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
        # Copy inv_freq from HF rotary embedding to guarantee exact RoPE match
        inv_freq = self._hf.model.talker.model.rotary_emb.inv_freq.detach().cpu()
        self._decoder = _MKDecoder(weights, inv_freq=inv_freq)
        # mk_decoder exposed for benchmark access
        self.mk_decoder = self._decoder

        print(f"[QwenTTSBackendMK] Ready in {(time.perf_counter()-t0)*1000:.0f}ms")

    def _run_with_megakernel_decode(self, text: str) -> np.ndarray:
        """
        Run generate_custom_voice with talker decode steps replaced by megakernel.

        Confirmed from spy (2026-05-14):
          - talker.generate() receives inputs_embeds=[1,13,1024], no past_key_values
          - Every decode step also uses inputs_embeds=[1,1,1024] — never input_ids
          - inputs_embeds at decode steps is NOT a pure codec_embedding lookup
            (cosine sim ~0.48 max) — HF blends in trailing_text_hidden etc.
          - Cannot reliably recover token ID from inputs_embeds

        Strategy: patch talker.model.forward().
          - Prefill (first call): run HF normally, capture KV cache + prefill logits.
            Extract first decode token = argmax of last prefill logit position.
          - Decode steps: ignore inputs_embeds entirely. Run megakernel step with
            the token we chose last step. Return fake output with one-hot logits
            so HF picks the right next token to embed. HF embeds it and passes
            it back — we ignore that embedding and use our own token tracking.
        """
        decoder = self._decoder
        orig_forward = self._hf.model.talker.model.forward
        lm_head_weight = decoder._lm_head_weight      # [3072, 1024]
        final_norm_weight = decoder._final_norm_weight  # [1024]
        _prefill_done = [False]
        _step_count = [0]
        _current_token = [0]  # tracks the token the megakernel should process next

        def _mk_forward(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=True,
            **kwargs,
        ):
            from types import SimpleNamespace

            # Prefill: first call — run HF normally to populate KV cache
            if not _prefill_done[0]:
                out = orig_forward(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **kwargs,
                )
                pkv = out.past_key_values
                decoder.reset()
                prefill_len = decoder.load_kv_cache_from_hf(pkv)

                # Compute first decode token from prefill last hidden state.
                # talker.model (inner) returns last_hidden_state, not logits —
                # the outer Qwen3TTSTalkerForConditionalGeneration applies codec_head.
                # We replicate that: final_norm → lm_head → argmax.
                last_h = out.last_hidden_state[0, -1].float()  # [HIDDEN_SIZE]
                variance = last_h.pow(2).mean()
                normed = (last_h * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16)
                normed = normed * final_norm_weight
                logits_1d = lm_head_weight @ normed  # [3072]
                first_token = int(logits_1d.argmax().item())
                _current_token[0] = first_token
                _prefill_done[0] = True
                print(f"[MK] Prefill done — seq_len={prefill_len}, first_token={first_token}")
                return out

            # Decode step: run megakernel with the token we tracked from last step.
            # Ignore inputs_embeds — HF constructed it but we don't need it.
            current_token = _current_token[0]
            next_token = decoder.step(current_token)
            _step_count[0] += 1

            # Tell HF the next token via one-hot logits so it embeds the right token
            _current_token[0] = next_token

            logits = torch.zeros(1, 1, VOCAB_SIZE, dtype=torch.bfloat16, device="cuda")
            logits[0, 0, next_token] = 1.0

            last_hidden = decoder._hidden.detach().clone().view(1, 1, HIDDEN_SIZE)

            return SimpleNamespace(
                logits=logits,
                past_key_values=None,
                hidden_states=(last_hidden,),
                last_hidden_state=last_hidden,
                attentions=None,
            )

        self._hf.model.talker.model.forward = _mk_forward
        try:
            with torch.inference_mode():
                wavs, sr = self._hf.generate_custom_voice(
                    text=text,
                    language=self._language,
                    speaker=self._speaker,
                    max_new_tokens=4096,
                    do_sample=False,
                )
            print(f"[MK] Decode complete — {_step_count[0]} megakernel steps")
        finally:
            self._hf.model.talker.model.forward = orig_forward

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
