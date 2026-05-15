"""
Phase 2 — Custom decode loop (Option B): correct architecture, real streaming.

Architecture (verified from QwenLM/Qwen3-TTS + andimarafioti/faster-qwen3-tts):

    Text
      → prefill_embeds (text_projection + trailing_text_hiddens)
      → talker.forward(prefill)                  → past_kv, past_hidden, gen_step=0
      → DECODE LOOP per codec frame:
          CB0_token = sample(logits)
          last_id_hidden = codec_embedding(CB0_token)   [1, 1, 1024]
          pred_input = cat(past_hidden, last_id_hidden) [1, 2, 1024]
          CB1..15 = code_predictor(pred_input)          [15]
          all_cb = cat(CB0, CB1..15)                    [16]  ← codec frame
          inputs_embeds = sum(16 codec embeds) + text_cond  [1, 1, 1024]
          hidden = talker.model.forward(inputs_embeds, past_kv)
          logits = lm_head(RMSNorm(hidden[:, -1]))      [3072]
          past_hidden = hidden[:, -1:, :]               [1, 1, 1024]
          gen_step += 1
      → CHUNK VOCODING every CHUNK_FRAMES frames:
          window = all_codes[max(0, n-CONTEXT_FRAMES):n]
          audio = speech_tokenizer.decode(window)
          yield trimmed_new_audio

DO NOT modify: existing HF pipeline, voice_agent.py, tts_backend_mk.py, tts_backend_hf.py.
This is a new isolated module. Wire it in by setting TTS_BACKEND=v2 in voice_agent.py.
"""

import asyncio
import os
import struct
import sys
import time
import logging
from collections.abc import AsyncGenerator

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Free ~10% speedup on Blackwell/Ampere via TF32 tensor cores for bfloat16 matmuls
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
CHUNK_FRAMES = 4       # 4 frames = 320ms audio per chunk; balances TTFC vs vocoder overhead
CONTEXT_FRAMES = 25    # left-context window for causal codec decoder
SAMPLES_PER_FRAME = SAMPLE_RATE // 12   # = 2000 samples = 80ms per codec frame
# codec_eos_token_id from the actual model's config.json (Qwen3-TTS-12Hz-0.6B-CustomVoice):
#   talker_config.codec_eos_token_id = 2150
# The Python class default is 4198 but the shipped checkpoint overrides it to 2150.
# 2150 is within [0, 3072) so it's a valid logit index.
# Confirmed from: QwenLM/Qwen3-TTS modeling + faster-qwen3-tts which reads it from config.
EOS_TOKEN_ID = 2150
MAX_NEW_TOKENS = 4096
MIN_NEW_TOKENS = 2

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# ---------------------------------------------------------------------------
# Megakernel constants (talker model dimensions)
# ---------------------------------------------------------------------------
_MK_NUM_LAYERS = 28
_MK_HEAD_DIM = 128
_MK_NUM_Q_HEADS = 16
_MK_NUM_KV_HEADS = 8
_MK_HIDDEN_SIZE = 1024
_MK_VOCAB_SIZE = 3072
_MK_MAX_SEQ_LEN = 1024   # sufficient for TTS; 32768 tanks tok/s due to huge KV alloc
_MK_ROPE_THETA = 1_000_000.0
_MK_LM_NUM_BLOCKS = 1184  # ldg_lm_head_fused block count — must match kernel.cu


def _mk_build_rope_tables(max_seq_len, inv_freq=None):
    if inv_freq is not None:
        inv_f = inv_freq.float().cpu()
    else:
        idx = torch.arange(0, _MK_HEAD_DIM, 2, dtype=torch.float32)
        inv_f = 1.0 / (_MK_ROPE_THETA ** (idx / _MK_HEAD_DIM))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_f)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_table = torch.cos(emb).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(emb).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


_MK_LAYER_KEYS = [
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


def _mk_extract_weights(hf_state):
    layer_weights = []
    for i in range(_MK_NUM_LAYERS):
        prefix = f"talker.model.layers.{i}."
        for key in _MK_LAYER_KEYS:
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


def _mk_pack_layer_weights(layer_weights):
    ptr_size = 8
    n_ptrs = len(_MK_LAYER_KEYS)
    buf = bytearray(_MK_NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(_MK_NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class _MKDecoder:
    """
    Stateful megakernel decoder for one talker backbone step.

    Normal path: step(token_id) — kernel does embed lookup + 28-layer forward + argmax.
    Sentinel path: step_with_embed(inputs_embeds [1024 bf16]) — caller writes inputs_embeds
    into _hidden before calling decode(-1). Kernel reads _hidden as the embedding row
    instead of indexing embed_weight. This is the Phase 3 integration point.

    Requires kernel.cu patch:
        const __nv_bfloat16 *embed_row =
            (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                                  : hidden_buffer;
    """

    def __init__(self, weights, inv_freq=None):
        self._weights = weights
        self._position = 0
        self._attn_scale = float(_MK_HEAD_DIM ** -0.5)
        self._cos_table, self._sin_table = _mk_build_rope_tables(_MK_MAX_SEQ_LEN, inv_freq)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _mk_pack_layer_weights(weights["layer_weights"])

        self._hidden = torch.zeros(_MK_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self._activations = torch.zeros(_MK_HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self._residual = torch.zeros(_MK_HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self._q = torch.zeros(_MK_NUM_Q_HEADS * _MK_HEAD_DIM, dtype=torch.float32, device="cuda")
        self._k = torch.zeros(_MK_NUM_KV_HEADS * _MK_HEAD_DIM, dtype=torch.float32, device="cuda")
        self._v = torch.zeros(_MK_NUM_KV_HEADS * _MK_HEAD_DIM, dtype=torch.float32, device="cuda")
        self._attn_out = torch.zeros(_MK_NUM_Q_HEADS * _MK_HEAD_DIM, dtype=torch.float32, device="cuda")
        self._mlp_intermediate = torch.zeros(_MK_VOCAB_SIZE, dtype=torch.float32, device="cuda")
        self._normalized = torch.zeros(_MK_HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self._k_cache = torch.zeros(
            _MK_NUM_LAYERS, _MK_NUM_KV_HEADS, _MK_MAX_SEQ_LEN, _MK_HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros(
            _MK_NUM_LAYERS, _MK_NUM_KV_HEADS, _MK_MAX_SEQ_LEN, _MK_HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._block_max_vals = torch.full((_MK_LM_NUM_BLOCKS,), float("-inf"), dtype=torch.float32, device="cuda")
        self._block_max_idxs = torch.zeros(_MK_LM_NUM_BLOCKS, dtype=torch.int32, device="cuda")
        self._output_token = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._decode_op = torch.ops.qwen_megakernel_C.decode
        # Load reset_barriers via ctypes — bypasses torch dispatch (no tensor args).
        # reset_barriers() zeros d_barrier_counter/sense/kv_flag/attn_flag on host
        # before each decode() call to prevent the barrier race in consecutive calls.
        # Must load with RTLD_GLOBAL | RTLD_NOLOAD since torch already loaded the .so.
        self._reset_barriers_fn = None
        try:
            import ctypes, ctypes.util, glob
            so_files = glob.glob(
                os.path.expanduser("~/.cache/torch_extensions/*/qwen_megakernel_C/qwen_megakernel_C.so")
            )
            if so_files:
                # RTLD_NOLOAD: don't re-load, just get handle to already-loaded .so
                _lib = ctypes.CDLL(so_files[0], mode=ctypes.RTLD_GLOBAL)
                _lib.reset_barriers.restype = None
                _lib.reset_barriers.argtypes = []
                self._reset_barriers_fn = _lib.reset_barriers
                logger.info("[v2/mk] reset_barriers loaded via ctypes")
        except Exception as e:
            logger.warning(f"[v2/mk] reset_barriers not available ({e}), using cuda.synchronize()")

    def _call_args(self):
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

    def step_with_embed(self, inputs_embeds: torch.Tensor) -> int:
        """
        Sentinel path: write inputs_embeds [1024] into hidden_buffer, call decode(-1).
        Kernel reads hidden_buffer as the embedding row (sentinel patch required).
        Returns next token id (argmax from kernel).

        Barriers (d_barrier_counter/sense/kv_flag/attn_flag) must be zeroed before
        each decode() call because the direct kernel's on-device reset races with
        blocks 1-127 reading stale values from the prior call — deadlocking on the
        second consecutive launch.  The GPU is idle here (previous step drained via
        item()), so the synchronous memsets in reset_barriers() complete instantly.
        """
        # Reset barriers. ctypes path preferred (direct symbol, no dispatch overhead).
        # PyTorch op is an equivalent fallback. A bare cuda.synchronize() is NOT a
        # valid fallback — it stalls but does not zero the barrier memory, so the
        # second kernel launch would still deadlock.
        if self._reset_barriers_fn is not None:
            self._reset_barriers_fn()
        else:
            torch.ops.qwen_megakernel_C.reset_barriers()
        self._hidden.copy_(inputs_embeds.view(_MK_HIDDEN_SIZE))
        self._decode_op(
            self._output_token,
            -1,
            *self._call_args(),
            _MK_NUM_LAYERS,
            self._position,
            _MK_MAX_SEQ_LEN,
            self._attn_scale,
        )
        # item() below forces the required CPU-GPU sync — no explicit synchronize needed.
        self._position += 1
        return int(self._output_token.item())

    def load_kv_from_hf(self, past_key_values) -> int:
        """Copy HF DynamicCache into megakernel KV tensors. Returns prefill seq_len."""
        self._k_cache.zero_()
        self._v_cache.zero_()
        for layer_idx, layer in enumerate(past_key_values.layers):
            k = layer.keys   # [1, kv_heads, seq_len, head_dim]
            v = layer.values
            seq = k.shape[2]
            if seq > _MK_MAX_SEQ_LEN:
                raise RuntimeError(f"Prefill seq_len {seq} > MK_MAX_SEQ_LEN {_MK_MAX_SEQ_LEN}")
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
# Prefill helpers
# ---------------------------------------------------------------------------


def _build_prefill_inputs_and_run(hf_model, text: str, speaker: str, language: str):
    """
    Run the prefill pass and return prefill outputs + captured conditioning tensors.

    Strategy: patch talker.generate() (what generate_custom_voice() calls).
    When generate() is called, we intercept its kwargs (which include inputs_embeds,
    trailing_text_hidden, tts_pad_embed, etc.), run the prefill forward() manually,
    then raise to abort the rest of generate_custom_voice().

    generate_custom_voice() calls:
        talker.generate(
            inputs_embeds=prefill_embeds,       [1, prefill_len, 1024]
            trailing_text_hidden=...,           [1, T, 1024]
            tts_pad_embed=...,                  [1, 1, 1024]
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=...,
            ...
        )

    Returns:
        past_key_values: DynamicCache (28 layers of KV)
        past_hidden: [1, 1, 1024] — last hidden state
        gen_step: int (0 after prefill)
        trailing_text_hiddens: [1, T, 1024]
        tts_pad_embed: [1, 1, 1024]
        logits: [1, 3072] — logits from last prefill position (for first token)
    """
    model = hf_model.model
    talker = model.talker

    captured = {}
    orig_generate = talker.generate

    class _AbortAfterPrefill(Exception):
        pass

    def _intercept_generate(**kwargs):
        # Extract the conditioning tensors generate_custom_voice passes to talker.generate()
        inputs_embeds = kwargs.get("inputs_embeds")
        trailing = kwargs.get("trailing_text_hidden")
        tts_pad = kwargs.get("tts_pad_embed")
        attention_mask = kwargs.get("attention_mask")

        if inputs_embeds is None:
            raise RuntimeError("talker.generate() called without inputs_embeds")

        # Clone conditioning tensors before the generate call modifies any state
        captured["trailing_text_hidden"] = trailing.detach().clone() if trailing is not None else None
        captured["tts_pad_embed"] = tts_pad.detach().clone() if tts_pad is not None else None

        # Run the prefill forward pass directly on talker (not via generate)
        # talker.forward() with inputs_embeds.shape[1] > 1 is the prefill
        prefill_result = talker.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        captured["prefill_result"] = prefill_result
        raise _AbortAfterPrefill()

    talker.generate = _intercept_generate
    try:
        with torch.inference_mode():
            hf_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
    except _AbortAfterPrefill:
        pass
    except Exception as e:
        logger.warning(f"Unexpected exception during prefill capture: {e}", exc_info=True)
    finally:
        talker.generate = orig_generate

    if "prefill_result" not in captured:
        raise RuntimeError(
            "Prefill capture failed — talker.generate() was not intercepted. "
            "generate_custom_voice() may have a different call structure."
        )

    result = captured["prefill_result"]

    # past_hidden comes from talker.forward() output field
    past_hidden = getattr(result, "past_hidden", None)
    if past_hidden is None:
        # Fallback: use last hidden state directly
        past_hidden = result.last_hidden_state[:, -1:, :]

    gen_step = getattr(result, "generation_step", 0) or 0

    return (
        result.past_key_values,                 # DynamicCache
        past_hidden,                            # [1, 1, 1024]
        gen_step,                               # int
        captured["trailing_text_hidden"],       # [1, T, 1024]
        captured["tts_pad_embed"],              # [1, 1, 1024]
        result.logits[:, -1, :],               # [1, 3072]
    )


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _custom_decode_loop(
    talker,
    past_key_values,
    past_hidden: torch.Tensor,
    gen_step: int,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    first_logits: torch.Tensor,
    config,
    max_new_tokens: int = MAX_NEW_TOKENS,
    min_new_tokens: int = MIN_NEW_TOKENS,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    chunk_size: int = CHUNK_FRAMES,
    on_chunk=None,
    talker_graph=None,      # TalkerGraph instance (CUDA graph path)
    predictor_graph=None,   # PredictorGraph instance (CUDA graph path)
    mk_decoder=None,        # _MKDecoder instance (megakernel path — highest priority)
):
    """
    Custom autoregressive decode loop owning the full generation runtime.

    Calls talker.model.forward() directly (bypassing talker.forward() / talker.generate()).
    Handles: code predictor, codec embedding reconstruction, text conditioning.
    Calls on_chunk(codec_chunk) every chunk_size frames.

    Returns: list of all [16]-tensors (one per codec frame)
    """
    device = past_hidden.device
    talker_model = talker.model
    codec_head = talker.codec_head
    codec_embedding = talker.get_input_embeddings()

    predictor = talker.code_predictor
    pred_model = predictor.model
    pred_lm_heads = predictor.lm_head           # ModuleList of 15 Linear heads
    pred_embeds = predictor.get_input_embeddings()  # ModuleList of 15 Embeddings(2048, 1024)
    # predictor_codec_embeds used for building next-step talker embedding
    predictor_codec_embeds = pred_embeds

    num_code_groups = config.num_code_groups   # 16
    eos_id = EOS_TOKEN_ID
    vocab_size = config.vocab_size   # 3072
    suppress_start = max(0, vocab_size - 1024)   # 2048

    # Pre-build suppress mask on GPU — avoids repeated masked_fill calls per step
    # Suppresses [2048..3071] except EOS=2150, matching official suppress_tokens list
    _suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    _suppress_mask[suppress_start:eos_id] = True
    _suppress_mask[eos_id + 1:] = True

    def _sample(logits_1d: torch.Tensor, suppress_eos: bool) -> torch.Tensor:
        logits_1d = logits_1d.clone().float()  # float32 for numerical stability
        logits_1d[_suppress_mask] = float("-inf")
        if suppress_eos:
            logits_1d[eos_id] = float("-inf")
        if do_sample and temperature > 0:
            logits_1d = logits_1d / temperature
            if top_k > 0:
                topk_vals = torch.topk(logits_1d, min(top_k, logits_1d.size(-1))).values
                logits_1d = logits_1d.masked_fill(logits_1d < topk_vals[-1], float("-inf"))
            probs = torch.softmax(logits_1d, dim=-1)
            # Guard: if all probs are zero/nan (all-inf logits), fall back to argmax
            if not torch.isfinite(probs).any() or probs.sum() == 0:
                return logits_1d.nan_to_num(nan=0.0, neginf=0.0).argmax()
            return torch.multinomial(probs, 1).squeeze()
        return logits_1d.argmax()

    def _run_predictor(pred_input: torch.Tensor) -> torch.Tensor:
        """
        Run code predictor for 15 codebook steps. Returns [15] int64 tensor.
        pred_input: [1, 2, hidden] = cat(past_hidden, last_id_hidden)
        """
        cb_tokens = []
        # Prefill: 2-token sequence through 5-layer predictor, build KV cache
        pred_out = pred_model(inputs_embeds=pred_input, use_cache=True, return_dict=True)
        pred_pkv = pred_out.past_key_values
        cb_logits = pred_lm_heads[0](pred_out.last_hidden_state[:, -1, :])
        cb_tok = (torch.multinomial(torch.softmax(cb_logits / temperature, -1), 1).squeeze()
                  if do_sample and temperature > 0 else cb_logits.argmax(-1).squeeze())
        cb_tokens.append(cb_tok)

        # Steps CB2..CB15: single-token decode with KV cache
        for cb_idx in range(1, num_code_groups - 1):
            cb_emb = pred_embeds[cb_idx - 1](cb_tok.unsqueeze(0).unsqueeze(0))
            pred_out = pred_model(inputs_embeds=cb_emb, past_key_values=pred_pkv,
                                  use_cache=True, return_dict=True)
            pred_pkv = pred_out.past_key_values
            cb_logits = pred_lm_heads[cb_idx](pred_out.last_hidden_state[:, -1, :])
            cb_tok = (torch.multinomial(torch.softmax(cb_logits / temperature, -1), 1).squeeze()
                      if do_sample and temperature > 0 else cb_logits.argmax(-1).squeeze())
            cb_tokens.append(cb_tok)

        return torch.stack(cb_tokens)  # [15]

    # Pre-stack predictor embedding weights for vectorized lookup — [15, vocab, hidden]
    # Avoids 15-iteration Python loop per decode step.
    _pred_cb_weights = torch.stack(
        [predictor_codec_embeds[i].weight for i in range(num_code_groups - 1)]
    )  # [15, 2048, 1024]
    _cb_indices = torch.arange(num_code_groups - 1, device=device)  # [15]

    # First token from prefill logits
    token = _sample(first_logits.squeeze(0), suppress_eos=(min_new_tokens > 0))

    all_frames = []
    chunk_buffer = []
    step = 0
    # prefill_len needed for TalkerGraph position tracking
    if talker_graph is not None:
        prefill_len = talker_graph.static_cache.get_seq_length()
    else:
        prefill_len = 0  # not used in eager path

    # Track EOS on GPU to avoid .item() sync every step.
    # We check EOS only when we'd sync anyway (chunk yield or end of loop).
    # Between chunk boundaries the GPU pipeline runs unblocked.
    _eos_id_t = torch.tensor(eos_id, dtype=token.dtype, device=device)
    _eos_found_at = None   # set when EOS detected

    for step_idx in range(max_new_tokens):
        # CPU sync required here — we need token value to build codec_embedding input.
        # This is unavoidable: codec_embedding(token) needs the integer index on CPU.
        # Minimize by keeping everything else GPU-side.
        tok_val = token.item()
        if tok_val == eos_id:
            logger.info(f"[v2] EOS fired at step {step_idx}")
            break

        # --- Code predictor: CB1..CB15 ---
        last_id_hidden = codec_embedding(token.unsqueeze(0).unsqueeze(0))   # [1, 1, 1024]
        pred_input = torch.cat([past_hidden, last_id_hidden], dim=1)         # [1, 2, 1024]

        if predictor_graph is not None:
            codebook_token_ids = predictor_graph.run(pred_input)             # [15] graph path
        else:
            codebook_token_ids = _run_predictor(pred_input)                  # [15] eager path

        # Full codec frame: CB0 + CB1..CB15
        all_cb = torch.cat([token.view(1), codebook_token_ids])   # [16]
        chunk_buffer.append(all_cb.detach())
        all_frames.append(all_cb.detach())

        # --- Build next-step input embedding ---
        # Vectorized: index into pre-stacked [15, 2048, 1024] weight tensor.
        # Clamp tokens to valid range [0, pred_vocab-1] before indexing — graph output
        # buffer may contain stale values on first replay that cause CUDA asserts.
        pred_vocab = _pred_cb_weights.shape[1]  # 2048
        safe_cb_ids = codebook_token_ids.clamp(0, pred_vocab - 1)
        cb_embeds = _pred_cb_weights[_cb_indices, safe_cb_ids]  # [15, 1024]
        # Sum CB0 (last_id_hidden squeezed) + CB1..15
        inputs_embeds = last_id_hidden.squeeze(1) + cb_embeds.sum(0, keepdim=True)  # [1, 1024]
        inputs_embeds = inputs_embeds.unsqueeze(1)  # [1, 1, 1024]

        # Add text conditioning
        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        # --- Talker backbone: single decode step ---
        if mk_decoder is not None:
            # Megakernel sentinel path.
            # inputs_embeds [1, 1, 1024] → write into hidden_buffer → decode(-1)
            # Kernel reads hidden_buffer as embedding row, runs 28-layer forward,
            # writes new hidden to hidden_buffer, writes argmax token to output_token.
            next_tok_id = mk_decoder.step_with_embed(inputs_embeds)
            # _hidden is the raw residual stream — apply final RMSNorm to match
            # what HF talker.model returns as last_hidden_state (normed output).
            # Code predictor was trained on normed hidden states; feeding it the
            # raw residual causes divergence within a few steps.
            _h = mk_decoder._hidden.float()
            _h = _h * torch.rsqrt(_h.pow(2).mean() + 1e-6)
            _h = (_h * mk_decoder._final_norm_weight.float()).to(torch.bfloat16)
            past_hidden = _h.view(1, 1, _MK_HIDDEN_SIZE)
            gen_step += 1
            step = step_idx + 1
            if step_idx < 10 or step_idx % 20 == 0:
                logger.info(f"[v2/mk] step={step_idx} mk_token={next_tok_id} valid={0<=next_tok_id<vocab_size} pos={mk_decoder._position-1}")
            suppress_eos = step < min_new_tokens
            if next_tok_id == eos_id and not suppress_eos:
                logger.info(f"[v2/mk] EOS fired at step {step_idx}")
                if chunk_buffer and on_chunk is not None:
                    on_chunk(torch.stack(chunk_buffer))
                    chunk_buffer = []
                break
            if not (0 <= next_tok_id < vocab_size):
                logger.error(f"[v2/mk] Out-of-range token {next_tok_id} at step {step_idx} — aborting decode")
                break
            token = torch.tensor(next_tok_id, dtype=torch.long, device=device)
        elif talker_graph is not None:
            hidden, logits = talker_graph.run(inputs_embeds, position=prefill_len + step_idx)
            # No clone needed: past_hidden is consumed by predictor_graph.run() at the top
            # of the NEXT iteration, before talker_graph.run() overwrites output_buf again.
            past_hidden = hidden[:, -1:, :]
            if logits is None:
                logits = codec_head(hidden[:, -1, :])
            gen_step += 1
            step = step_idx + 1
            suppress_eos = step < min_new_tokens
            token = _sample(logits.squeeze(0), suppress_eos=suppress_eos)
        else:
            backbone_out = talker_model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = backbone_out.last_hidden_state
            past_key_values = backbone_out.past_key_values
            logits = codec_head(hidden[:, -1, :])
            past_hidden = hidden[:, -1:, :].clone()
            gen_step += 1
            step = step_idx + 1
            suppress_eos = step < min_new_tokens
            token = _sample(logits.squeeze(0), suppress_eos=suppress_eos)

        # --- Yield chunk if buffer full ---
        if len(chunk_buffer) >= chunk_size and on_chunk is not None:
            chunk_tensor = torch.stack(chunk_buffer)   # [chunk_size, 16]
            on_chunk(chunk_tensor)
            chunk_buffer = []

    # Final partial chunk
    if chunk_buffer and on_chunk is not None:
        chunk_tensor = torch.stack(chunk_buffer)
        on_chunk(chunk_tensor)

    logger.info(f"[v2] Decode complete — {step} codec frames, {step * SAMPLES_PER_FRAME / SAMPLE_RATE:.2f}s audio")
    # all_frames is populated incrementally via all_frames.append() above
    return all_frames  # list of [16] tensors, one per codec frame


# ---------------------------------------------------------------------------
# Incremental vocoder
# ---------------------------------------------------------------------------

class _IncrementalVocoder:
    """
    Wraps speech_tokenizer to emit audio incrementally with left-context windowing.

    Codec decoder is a causal ConvNet — needs ~25 frames of left context for
    accurate reconstruction. We re-decode a window of (context + new) frames
    and trim to emit only the new samples.
    """

    def __init__(self, speech_tokenizer, context_frames: int = CONTEXT_FRAMES):
        self._tokenizer = speech_tokenizer
        self._context_frames = context_frames
        self._all_codes: list[torch.Tensor] = []  # list of [16] tensors
        self._emitted_frames = 0

    def _decode_window(self, window_codes: torch.Tensor) -> np.ndarray:
        """
        Call speech_tokenizer.decode() on window_codes [n_frames, 16].
        Returns float32 numpy array of waveform samples.

        API: speech_tokenizer.decode([{"audio_codes": codes}]) returns (wav_list, sr)
        where wav_list[0] is a float32 numpy array of shape [n_samples].
        """
        window = window_codes.long()  # ensure int64
        result = self._tokenizer.decode([{"audio_codes": window}])
        if isinstance(result, (tuple, list)) and len(result) == 2:
            audio_data, sr = result
            if isinstance(audio_data, (list, tuple)):
                audio_arr = audio_data[0] if audio_data else np.zeros(0, dtype=np.float32)
            else:
                audio_arr = audio_data
        else:
            audio_arr = result

        if hasattr(audio_arr, "cpu"):
            audio_arr = audio_arr.cpu().float().numpy()
        return np.asarray(audio_arr, dtype=np.float32).squeeze()

    def add_chunk(self, codec_chunk: torch.Tensor) -> bytes:
        """
        Add codec_chunk ([N, 16]) and return new audio bytes (int16 PCM, 24kHz).
        """
        for i in range(codec_chunk.shape[0]):
            self._all_codes.append(codec_chunk[i])

        n_total = len(self._all_codes)
        n_new = n_total - self._emitted_frames
        if n_new <= 0:
            return b""

        ctx_start = max(0, n_total - self._context_frames - n_new)
        window_frames = torch.stack(self._all_codes[ctx_start:n_total])   # [ctx+new, 16]

        try:
            audio_arr = self._decode_window(window_frames)

            # Trim to only new samples (the tail of the decoded audio)
            new_samples = n_new * SAMPLES_PER_FRAME
            if len(audio_arr) > new_samples:
                audio_arr = audio_arr[-new_samples:]

            self._emitted_frames = n_total
            return (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        except Exception as e:
            logger.error(f"[v2] Vocoder error: {e}", exc_info=True)
            self._emitted_frames = n_total
            return bytes(n_new * SAMPLES_PER_FRAME * 2)   # silence

    def flush(self, full_codes: list[torch.Tensor]) -> bytes:
        """
        Final flush: vocode remaining frames not yet emitted.
        Uses the full code sequence for best tail quality.
        """
        n_total = len(full_codes)
        n_remaining = n_total - self._emitted_frames
        if n_remaining <= 0:
            return b""

        try:
            all_frames = torch.stack(full_codes)   # [T, 16]
            audio_arr = self._decode_window(all_frames)

            tail_samples = n_remaining * SAMPLES_PER_FRAME
            if len(audio_arr) > tail_samples:
                audio_arr = audio_arr[-tail_samples:]

            return (np.clip(audio_arr, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"[v2] Vocoder flush error: {e}", exc_info=True)
            return bytes(n_remaining * SAMPLES_PER_FRAME * 2)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class QwenTTSBackendV2:
    """
    Phase 2 backend — custom decode loop with correct codec frame generation.

    Differences from QwenTTSBackendMK (v1):
    - Owns the FULL decode loop (no HF generate() call)
    - Calls code_predictor correctly per step
    - Builds codec embedding reconstruction (16 codebooks summed)
    - Applies trailing_text_hiddens conditioning per step
    - Streams audio chunks from vocoder incrementally (true streaming)

    Drop-in compatible with QwenTTSService (same synthesize_streaming() interface).
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        speaker: str = "Ryan",
        language: str = "English",
        megakernel_path: str = "./qwen_megakernel",
    ):
        from qwen_tts import Qwen3TTSModel

        logger.info("[v2] Loading Qwen3-TTS model...")
        t0 = time.perf_counter()

        self._hf = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
        )
        self._hf.model.eval()
        self._speaker = speaker
        self._language = language
        self.sample_rate = SAMPLE_RATE

        # Cache handles to modules we need directly
        model = self._hf.model
        self._talker = model.talker
        self._config = model.talker.config
        self._speech_tokenizer = model.speech_tokenizer

        # Megakernel decoder — activated by V2_MEGAKERNEL=1.
        # When active, replaces TalkerGraph for backbone step.
        # PredictorGraph remains active regardless (code predictor is independent).
        self._mk_decoder = None
        self._megakernel_path = megakernel_path
        if os.environ.get("V2_MEGAKERNEL", "0") == "1":
            self._setup_megakernel(model)

        # CUDA graphs for talker and predictor.
        # Skipped for talker if megakernel is active (it handles the backbone instead).
        self._talker_graph = None
        self._predictor_graph = None
        if os.environ.get("V2_CUDA_GRAPHS", "1") != "0":
            self._setup_cuda_graphs(model)

        logger.info(f"[v2] Ready in {(time.perf_counter()-t0)*1000:.0f}ms")
        logger.info(f"[v2] num_code_groups={self._config.num_code_groups}, vocab_size={self._config.vocab_size}")
        if self._mk_decoder is not None:
            logger.info("[v2] Megakernel active (sentinel path) + PredictorGraph")
        elif self._talker_graph:
            logger.info("[v2] TalkerGraph + PredictorGraph CUDA graphs active")
        else:
            logger.info("[v2] Running uncompiled")

    def _setup_megakernel(self, model):
        try:
            sys.path.insert(0, self._megakernel_path)
            from qwen_megakernel.build import get_extension
            get_extension()
            logger.info("[v2/mk] Megakernel extension loaded")

            state = model.state_dict()
            weights = _mk_extract_weights(state)
            inv_freq = model.talker.model.rotary_emb.inv_freq.detach().cpu()
            self._mk_decoder = _MKDecoder(weights, inv_freq=inv_freq)
            logger.info(f"[v2/mk] Decoder ready — MAX_SEQ_LEN={_MK_MAX_SEQ_LEN}, HIDDEN={_MK_HIDDEN_SIZE}")
        except Exception as e:
            logger.warning(f"[v2/mk] Megakernel init failed ({e}) — falling back to TalkerGraph/eager")
            self._mk_decoder = None

    def _setup_cuda_graphs(self, model):
        from server.backend.cuda_graphs import TalkerGraph, PredictorGraph
        try:
            talker = model.talker
            pred_config = talker.code_predictor.config
            talker_hidden = self._config.hidden_size  # 1024

            self._predictor_graph = PredictorGraph(
                code_predictor=talker.code_predictor,
                pred_config=pred_config,
                talker_hidden_size=talker_hidden,
                do_sample=True,
                temperature=0.9,
                top_k=50,
            )
            self._predictor_graph.capture()

            # Skip TalkerGraph when megakernel handles the backbone
            if self._mk_decoder is None:
                self._talker_graph = TalkerGraph(
                    talker_model=talker.model,
                    talker_config=self._config,
                    codec_head=talker.codec_head,
                    max_seq_len=2048,
                )
                self._talker_graph.capture()

        except Exception as e:
            logger.warning(f"[v2] CUDA graph capture failed ({e}), falling back to eager", exc_info=True)
            self._talker_graph = None
            self._predictor_graph = None

    def _run_custom_decode(self, text: str) -> tuple[list, float]:
        """
        Run full custom decode. Returns (all_frames, decode_time_s).
        Raises on error — caller handles fallback.
        """
        logger.info(f"[v2] Starting custom decode for: {text[:60]!r}")
        t0 = time.perf_counter()

        past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
            _build_prefill_inputs_and_run(
                self._hf, text, self._speaker, self._language
            )

        t_prefill = time.perf_counter() - t0
        logger.info(f"[v2] Prefill done in {t_prefill*1000:.0f}ms, gen_step={gen_step}")

        if self._mk_decoder is not None:
            self._mk_decoder.reset()
            self._mk_decoder.load_kv_from_hf(past_kv)
        elif self._talker_graph is not None:
            self._talker_graph.prefill_kv(past_kv)

        frames = _custom_decode_loop(
            talker=self._talker,
            past_key_values=past_kv,
            past_hidden=past_hidden,
            gen_step=gen_step,
            trailing_text_hiddens=trailing,
            tts_pad_embed=tts_pad,
            first_logits=first_logits,
            config=self._config,
            talker_graph=self._talker_graph if self._mk_decoder is None else None,
            predictor_graph=self._predictor_graph,
            mk_decoder=self._mk_decoder,
        )

        t_decode = time.perf_counter() - t0
        return frames, t_decode

    def _run_hf_fallback(self, text: str) -> np.ndarray:
        """Pure HF inference — unchanged backup path."""
        wavs, sr = self._hf.generate_custom_voice(
            text=text,
            language=self._language,
            speaker=self._speaker,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.9,
        )
        return np.array(wavs[0], dtype=np.float32).squeeze()

    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """
        Yields (audio_bytes: bytes, sample_rate: int) in ~CHUNK_FRAMES*80ms chunks.
        Uses custom decode loop with real streaming.
        Falls back to HF batch if custom decode fails.
        """
        loop = asyncio.get_event_loop()

        # Run the blocking decode in a thread executor
        # We use a queue to stream results from the executor thread to the async generator
        audio_queue: asyncio.Queue = asyncio.Queue()
        vocoder = _IncrementalVocoder(self._speech_tokenizer)
        all_frames_ref = []
        decode_error = None

        # Vocoder runs in a separate thread so decode loop is never blocked.
        import queue as _queue
        _codec_queue: _queue.Queue = _queue.Queue()  # codec chunks → vocoder thread
        _VOCODER_DONE = object()  # sentinel

        def _vocoder_thread():
            while True:
                item = _codec_queue.get()
                if item is _VOCODER_DONE:
                    break
                chunk, is_final_flush = item
                if is_final_flush:
                    audio_bytes = vocoder.flush(chunk)  # chunk = all_frames_ref here
                else:
                    audio_bytes = vocoder.add_chunk(chunk)
                if audio_bytes:
                    asyncio.run_coroutine_threadsafe(
                        audio_queue.put(audio_bytes), loop
                    ).result()
            asyncio.run_coroutine_threadsafe(audio_queue.put(None), loop).result()

        import threading as _threading
        _vt = _threading.Thread(target=_vocoder_thread, daemon=True)
        _vt.start()

        def _decode_thread():
            nonlocal decode_error
            try:
                t_start = time.perf_counter()

                past_kv, past_hidden, gen_step, trailing, tts_pad, first_logits = \
                    _build_prefill_inputs_and_run(
                        self._hf, text, self._speaker, self._language
                    )

                t_prefill = time.perf_counter() - t_start
                logger.info(
                    f"[v2] Prefill {t_prefill*1000:.0f}ms, gen_step={gen_step}, "
                    f"trailing_len={trailing.shape[1] if trailing is not None else 0}"
                )

                def _on_chunk(chunk: torch.Tensor):
                    # Push to vocoder thread — decode loop is never blocked
                    _codec_queue.put((chunk.clone(), False))

                # Handoff prefill KV cache to whichever backbone is active
                if self._mk_decoder is not None:
                    self._mk_decoder.reset()
                    self._mk_decoder.load_kv_from_hf(past_kv)
                elif self._talker_graph is not None:
                    self._talker_graph.prefill_kv(past_kv)

                frames = _custom_decode_loop(
                    talker=self._talker,
                    past_key_values=past_kv,
                    past_hidden=past_hidden,
                    gen_step=gen_step,
                    trailing_text_hiddens=trailing,
                    tts_pad_embed=tts_pad,
                    first_logits=first_logits,
                    config=self._config,
                    chunk_size=CHUNK_FRAMES,
                    on_chunk=_on_chunk,
                    talker_graph=self._talker_graph if self._mk_decoder is None else None,
                    predictor_graph=self._predictor_graph,
                    mk_decoder=self._mk_decoder,
                )
                all_frames_ref.extend(frames)

                # Final flush via vocoder thread, then signal done
                _codec_queue.put((list(all_frames_ref), True))
                _codec_queue.put(_VOCODER_DONE)

            except Exception as e:
                logger.error(f"[v2] Decode thread error: {e}", exc_info=True)
                decode_error = e
                # Signal vocoder thread to stop and emit sentinel
                _codec_queue.put(_VOCODER_DONE)

        try:
            executor_fut = loop.run_in_executor(None, _decode_thread)

            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield chunk, self.sample_rate

            await executor_fut  # propagate any thread exception

            if decode_error is not None:
                logger.warning("[v2] Custom decode failed, falling back to HF")
                audio = await loop.run_in_executor(None, self._run_hf_fallback, text)
                chunk_samples = int(self.sample_rate * 100 / 1000)
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i+chunk_samples]
                    yield (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), self.sample_rate
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"[v2] synthesize_streaming error: {e}", exc_info=True)
            # Last resort: HF fallback
            audio = await loop.run_in_executor(None, self._run_hf_fallback, text)
            chunk_samples = int(self.sample_rate * 100 / 1000)
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i+chunk_samples]
                yield (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), self.sample_rate
                await asyncio.sleep(0)
