"""
CUDA graph capture for talker backbone and code predictor.

Adapted from andimarafioti/faster-qwen3-tts (talker_graph.py, predictor_graph.py).
Strategy:
  - Prefill runs in HF eager with DynamicCache (variable prompt length)
  - prefill_kv() copies DynamicCache → StaticCache (fixed shape)
  - Decode step / predictor loop captured as CUDA graphs (zero CPU overhead per step)

Both TalkerGraph and PredictorGraph use transformers.StaticCache, which writes
via index_copy_() into pre-allocated [batch, kv_heads, max_seq, head_dim] buffers.
Shape never changes → CUDA graph capture works without recompilation.
"""

import torch
from transformers import StaticCache

try:
    from transformers.masking_utils import create_causal_mask
    _HAS_MASKING_UTILS = True
except ImportError:
    _HAS_MASKING_UTILS = False


def _make_causal_mask(config, input_embeds, cache_position, static_cache):
    """Build causal attention mask for given position. Returns tensor or None."""
    if not _HAS_MASKING_UTILS:
        return None
    try:
        return create_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=static_cache,
        )
    except Exception:
        return None


def _init_static_cache_layers(static_cache, config, dtype, device):
    """Force lazy initialization of StaticCache layers before graph capture."""
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    for layer in static_cache.layers:
        if hasattr(layer, "is_initialized") and not layer.is_initialized:
            layer.lazy_initialization(dummy_k)
        elif hasattr(layer, "key_cache") and layer.key_cache.numel() == 0:
            layer.lazy_initialization(dummy_k)


class TalkerGraph:
    """
    Captures the talker backbone single-token decode step as a CUDA graph.

    Usage:
        tg = TalkerGraph(talker.model, talker.config, max_seq_len=512)
        tg.capture()
        prefill_len = tg.prefill_kv(past_key_values_from_hf)
        for step in range(n):
            hidden = tg.run(inputs_embeds, position=prefill_len + step)
    """

    def __init__(self, talker_model, talker_config, codec_head=None,
                 device="cuda", dtype=torch.bfloat16, max_seq_len=512):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers
        self.config = talker_config
        self.model = talker_model
        self.codec_head = codec_head  # nn.Linear(hidden, vocab) — included in graph if provided

        self.static_cache = StaticCache(config=talker_config, max_cache_len=max_seq_len)

        # Static I/O buffers — shapes never change during graph replay
        self.input_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        # logits_buf: only allocated if codec_head is captured in the graph
        vocab_size = getattr(talker_config, "vocab_size", 3072)
        self.logits_buf = torch.zeros(1, vocab_size, dtype=dtype, device=device) if codec_head else None

        # Attention mask lookup table — one per position, built at capture time
        self.attn_mask_table = None
        self.attn_mask = None

        self.graph = None
        self.captured = False

    def _build_attention_masks(self):
        dummy = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self.attn_mask_table = []
        for i in range(self.max_seq_len):
            pos = torch.tensor([i], device=self.device)
            mask = _make_causal_mask(self.config, dummy, pos, self.static_cache)
            self.attn_mask_table.append(mask)
        # Static buffer updated via copy_() before each replay
        if self.attn_mask_table[0] is not None:
            self.attn_mask = self.attn_mask_table[0].clone()

    def _decode_step(self):
        kwargs = dict(
            inputs_embeds=self.input_buf,
            past_key_values=self.static_cache,
            cache_position=self.cache_position,
            use_cache=True,
            return_dict=True,
        )
        if self.attn_mask is not None:
            kwargs["attention_mask"] = self.attn_mask
        out = self.model(**kwargs)
        self.output_buf.copy_(out.last_hidden_state)
        # Include codec_head in graph if provided — avoids separate Python matmul
        if self.codec_head is not None:
            self.logits_buf.copy_(self.codec_head(out.last_hidden_state[:, -1, :]))

    @torch.inference_mode()
    def capture(self, prefill_len=50, num_warmup=3):
        """Warmup and capture CUDA graph for single-token decode."""
        print("[TalkerGraph] Initializing StaticCache...")
        _init_static_cache_layers(self.static_cache, self.config, self.dtype, self.device)
        self._build_attention_masks()

        self.cache_position[0] = prefill_len
        if self.attn_mask is not None and self.attn_mask_table[prefill_len] is not None:
            self.attn_mask.copy_(self.attn_mask_table[prefill_len])

        print(f"[TalkerGraph] Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()

        print("[TalkerGraph] Capturing CUDA graph...")
        self.graph = torch.cuda.CUDAGraph()
        self._stream = torch.cuda.Stream()
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._decode_step()  # warmup in capture stream
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph):
                self._decode_step()
        torch.cuda.current_stream().wait_stream(self._stream)
        torch.cuda.synchronize()
        self.captured = True
        print("[TalkerGraph] CUDA graph captured.")

    @torch.inference_mode()
    def prefill_kv(self, past_key_values) -> int:
        """
        Copy HF DynamicCache → StaticCache on the capture stream.
        Must run on the same stream the graph was captured on — otherwise the
        graph replay and the KV writes race across streams.
        """
        with torch.cuda.stream(self._stream):
            self.static_cache.reset()
            seq_len = 0
            for li in range(self.num_layers):
                layer = past_key_values.layers[li]
                k = layer.keys   # [1, kv_heads, seq_len, head_dim]
                v = layer.values
                seq_len = k.shape[2]
                cache_pos = torch.arange(seq_len, device=self.device)
                self.static_cache.update(k.to(self.dtype), v.to(self.dtype), li,
                                         {"cache_position": cache_pos})
        return seq_len

    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int):
        """
        Run one decode step via CUDA graph replay.
        input_embeds: [1, 1, hidden_size]
        position: current sequence position (prefill_len + step_idx)
        Returns: (hidden [1,1,hidden], logits [1,vocab]) if codec_head captured,
                 else (hidden [1,1,hidden], None)
        """
        with torch.cuda.stream(self._stream):
            self.input_buf.copy_(input_embeds)
            self.cache_position[0] = position
            if self.attn_mask is not None and position < len(self.attn_mask_table) and self.attn_mask_table[position] is not None:
                self.attn_mask.copy_(self.attn_mask_table[position])
            self.graph.replay()
        self._stream.synchronize()
        return self.output_buf, self.logits_buf


class PredictorGraph:
    """
    Captures the full code predictor 15-step loop as a single CUDA graph.

    The entire loop (prefill + 14 decode steps) is captured once with fixed
    StaticCache shape (max_seq = 17). Replay takes ~1ms vs ~49ms uncompiled.

    Usage:
        pg = PredictorGraph(talker.code_predictor, pred_config, talker_hidden_size=1024)
        pg.capture()
        cb_tokens = pg.run(pred_input)  # pred_input: [1, 2, 1024]
    """

    def __init__(self, code_predictor, pred_config, talker_hidden_size=1024,
                 device="cuda", dtype=torch.bfloat16,
                 do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
        self.device = device
        self.dtype = dtype
        self.num_codebooks = pred_config.num_code_groups - 1  # 15
        self.max_seq = 2 + self.num_codebooks                 # 17
        self.hidden_size = pred_config.hidden_size
        self.config = pred_config
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        cp = code_predictor
        # small_to_mtp_projection: Linear(talker_hidden, pred_hidden) or Identity
        self.small_to_mtp = getattr(cp, "small_to_mtp_projection", None)
        self.pred_model = cp.model
        self.lm_heads = cp.lm_head           # ModuleList[15]
        self.codec_embeds = cp.get_input_embeddings()  # ModuleList[15] Embedding(2048, 1024)

        self.static_cache = StaticCache(config=pred_config, max_cache_len=self.max_seq)

        # Pre-allocated cache positions — avoid CPU→GPU transfers inside graph
        self.prefill_cache_pos = torch.arange(2, device=device)
        self.decode_cache_positions = [
            torch.tensor([2 + i], device=device) for i in range(self.num_codebooks - 1)
        ]

        # I/O buffers
        self.input_buf = torch.zeros(1, 2, talker_hidden_size, dtype=dtype, device=device)
        self.output_tokens = torch.zeros(self.num_codebooks, dtype=torch.long, device=device)

        # Attention masks (built at capture time)
        self.prefill_attn = None
        self.decode_attn = []

        self.graph = None
        self.captured = False

    def _build_attention_masks(self):
        dummy_prefill = torch.zeros(1, 2, self.hidden_size, dtype=self.dtype, device=self.device)
        dummy_decode = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self.prefill_attn = _make_causal_mask(
            self.config, dummy_prefill, self.prefill_cache_pos, self.static_cache
        )
        self.decode_attn = [
            _make_causal_mask(self.config, dummy_decode, pos, self.static_cache)
            for pos in self.decode_cache_positions
        ]

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.small_to_mtp is not None:
            return self.small_to_mtp(x)
        return x

    def _sample_tok(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample single token from [1, vocab] logits."""
        if self.do_sample and self.temperature > 0:
            logits = logits / self.temperature
            if self.top_k > 0:
                top_vals = torch.topk(logits, min(self.top_k, logits.size(-1))).values
                logits = logits.masked_fill(logits < top_vals[..., -1:], float("-inf"))
            return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
        return logits.argmax(dim=-1)

    def _full_loop(self):
        """Full 15-step loop on static buffers — captured as CUDA graph."""
        h = self._project(self.input_buf)  # [1, 2, pred_hidden]

        kw0 = dict(inputs_embeds=h, past_key_values=self.static_cache,
                   cache_position=self.prefill_cache_pos, use_cache=True, return_dict=True)
        if self.prefill_attn is not None:
            kw0["attention_mask"] = self.prefill_attn
        out = self.pred_model(**kw0)

        logits = self.lm_heads[0](out.last_hidden_state[:, -1, :])  # [1, vocab]
        tok = self._sample_tok(logits)
        self.output_tokens[0] = tok.squeeze()

        for cb_idx in range(1, self.num_codebooks):
            emb = self.codec_embeds[cb_idx - 1](tok.unsqueeze(0))  # [1, 1, codec_hidden]
            emb = self._project(emb)
            kw = dict(inputs_embeds=emb, past_key_values=self.static_cache,
                      cache_position=self.decode_cache_positions[cb_idx - 1],
                      use_cache=True, return_dict=True)
            if self.decode_attn[cb_idx - 1] is not None:
                kw["attention_mask"] = self.decode_attn[cb_idx - 1]
            out = self.pred_model(**kw)
            logits = self.lm_heads[cb_idx](out.last_hidden_state[:, -1, :])
            tok = self._sample_tok(logits)
            self.output_tokens[cb_idx] = tok.squeeze()

    @torch.inference_mode()
    def capture(self, num_warmup=3):
        """Warmup and capture the 15-step CUDA graph."""
        print("[PredictorGraph] Initializing StaticCache...")
        _init_static_cache_layers(self.static_cache, self.config, self.dtype, self.device)
        self._build_attention_masks()

        print(f"[PredictorGraph] Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()

        print("[PredictorGraph] Capturing CUDA graph...")
        self.graph = torch.cuda.CUDAGraph()
        self._stream = torch.cuda.Stream()
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self.static_cache.reset()
            self._full_loop()
            torch.cuda.synchronize()
            self.static_cache.reset()
            with torch.cuda.graph(self.graph):
                self._full_loop()
        torch.cuda.current_stream().wait_stream(self._stream)
        torch.cuda.synchronize()
        self.captured = True
        print("[PredictorGraph] CUDA graph captured.")

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """
        Run captured 15-step loop on the capture stream.
        pred_input: [1, 2, talker_hidden_size]
        Returns: [15] long tensor — CB1..CB15 token IDs, clamped to [0, 2047]
        """
        with torch.cuda.stream(self._stream):
            self.input_buf.copy_(pred_input)
            self.static_cache.reset()
            self.graph.replay()
        self._stream.synchronize()
        result = self.output_tokens.clone()
        result.clamp_(0, 2047)
        return result
