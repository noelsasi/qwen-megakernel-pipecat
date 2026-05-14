# Implementation Plan: Qwen3-TTS + Megakernel + Pipecat

> **Engineering principle:** Inspect first. Verify before building. Unknowns are labeled explicitly.
> This document is structured as investigation → evidence → implementation, not assumption → code.

---

## What We're Building

A real-time voice agent:
```
Microphone → STT → LLM → Qwen3-TTS (talker accelerated by CUDA megakernel) → Speaker
```

Performance targets from the brief:
- **TTFC** (time to first audio chunk): < 60 ms
- **RTF** (real-time factor): < 0.15  
- Audio must stream frame-by-frame — no buffering full utterance

---

## Status (Final — 2026-05-14)

| Phase | Status |
|-------|--------|
| A — Baseline HF inference | ✅ Done — RTF 1.070 mean, TTFC 6338ms mean (3 trials) |
| B — Streaming | ⚠️ Fake streaming only — full audio buffered then chunked |
| C — Pipecat pipeline | ✅ Working end-to-end — STT→LLM→TTS→audio confirmed |
| D — Megakernel decode | ✅ Kernel runs at 263 tok/s — EOS and vocoder integration incomplete |

**Final state:** Full voice pipeline works with HF fallback. Megakernel decode loop runs and produces valid tokens at 263 tok/s but does not complete to audio due to EOS divergence and vocoder hidden_states format mismatch.

---

## Honest Risk Register

| Risk | Severity | Status |
|------|----------|--------|
| Qwen3-TTS Python package API is unverified | HIGH | ✅ Resolved — `qwen-tts` pip package, `Qwen3TTSModel` |
| Internal talker module path is unknown | HIGH | ✅ Resolved — `model.talker`, `model.talker.model` |
| Streaming/incremental audio decode may not be supported | HIGH | ⚠️ Confirmed unsupported natively — fake streaming implemented |
| Megakernel constant compatibility is unverified | HIGH | ✅ All constants match after LDG_VOCAB_SIZE patch |
| RTX 5090 (sm_120) required — no fallback | HIGH | ✅ Running on Vast.ai RTX 5090 |
| TTFC < 60ms may be unreachable with full model load overhead | MEDIUM | ⏳ Not yet measured with working megakernel |
| KV cache RoPE format compatibility | HIGH | 🔴 **Active blocker** — HF post-RoPE keys may not match kernel format |
| Pipecat version and TTSService API may differ from docs | LOW | ✅ Resolved — pipecat 1.1.0 API confirmed |

---

## ML Concepts for Full-Stack Developers

### Autoregressive Decoding

LLMs (and the Qwen3-TTS talker) generate one token at a time. Each step:
1. Feed the previous token into the model
2. Run all transformer layers (matrix multiplications on GPU)
3. Get a probability distribution over the vocabulary
4. Pick the most likely token (argmax or sampling)
5. Repeat

Each step is ~1ms on a fast GPU. 100 tokens = ~100ms. This is the loop the megakernel accelerates.

### What a CUDA Megakernel Is

Normal inference: launch a new GPU function (kernel) per layer, per operation. Overhead adds up.  
Megakernel: one persistent GPU function that handles ALL 28 layers in one launch, with thread blocks prefetching the next layer's weights while the current layer runs. Result: 71% of theoretical memory bandwidth vs ~20% normally.

### Qwen3-TTS Architecture (conceptual, not verified API)

```
Text → [Talker LLM] → codec token stream (one token per audio frame, autoregressively)
                          ↓
               [Code Predictor] → 32 codebook tokens per frame
                          ↓
               [Vocoder/Tokenizer] → waveform (24kHz audio)
```

The megakernel targets the **Talker LLM** decode loop. The code predictor and vocoder are left as-is.

### Real-Time Factor (RTF)

```
RTF = time_spent_generating / duration_of_generated_audio
```
- RTF = 1.0: generates audio exactly as fast as it plays
- RTF = 0.15: generates audio 6.7x faster than real-time (target)
- RTF > 1.0: too slow, playback will stutter

### TTFC (Time to First Chunk)

Time from when text is submitted to when the first audio byte is ready to play. Dominated by:
1. Model prefill time (processing the input text prompt)
2. First autoregressive decode step
3. Minimum tokens needed before the codec can decode any audio

---

## Phase A — Baseline TTS (Verify Everything First)

**Goal:** Run vanilla Qwen3-TTS, generate a WAV file, measure latency. No megakernel yet.

### A.1 — Environment Setup

```bash
# On RTX 5090 Vast.ai instance (CUDA 12.8+, driver 570+)
apt update && apt install -y libsndfile1 ffmpeg

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
# Expected: NVIDIA GeForce RTX 5090, 12.8
```

### A.2 — Discover the Real Qwen3-TTS Package (DO THIS BEFORE WRITING ANY CODE)

**Unknown:** The actual pip package name and import path for Qwen3-TTS.

```bash
# Option 1: check if it's in transformers
pip install transformers --upgrade
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers import AutoModelForCausalLM; help(AutoModelForCausalLM)"

# Option 2: check HuggingFace model card for install instructions
# Visit: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
# Look for: pip install instructions, import statements, usage examples

# Option 3: check if standalone package exists
pip search qwen-tts 2>/dev/null || pip install qwen-tts 2>&1 | head -20

# Option 4: use HF hub to inspect files
pip install huggingface_hub
python -c "
from huggingface_hub import list_repo_files
files = list(list_repo_files('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'))
print('\n'.join(files))
"
```

**What to look for in the repo files:**
- `modeling_qwen3_tts.py` — the actual model class definition
- `configuration_qwen3_tts.py` — config class
- `README.md` — usage examples
- `requirements.txt` — dependencies

```bash
# Download and inspect the modeling file directly
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', 'modeling_qwen3_tts.py')
print(path)
" 
# Then: cat <path> | head -100
# Look for: class names, __init__ signatures, generate() method
```

### A.3 — Load and Inspect the Model

Only run this AFTER A.2 confirms the correct import path. Replace `ACTUAL_CLASS` with what you find.

```python
# inspect_model.py — run this, capture the output, use it to inform all subsequent code

import torch
import json

# PLACEHOLDER — replace with actual import from A.2
# from transformers import ACTUAL_CLASS
# model = ACTUAL_CLASS.from_pretrained(
#     "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
#     device_map="cpu",          # CPU first to avoid OOM during inspection
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,    # likely needed for custom model code
# )

# Step 1: Print config
print("=== CONFIG ===")
print(model.config)

# Step 2: Print all named modules (this is the module hierarchy)
print("\n=== NAMED MODULES ===")
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")

# Step 3: Print all named parameters with shapes
print("\n=== PARAMETER SHAPES ===")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} {param.dtype}")

# Step 4: Identify the talker decoder
# Look for: modules with many layers, attention heads, MLP — these are the transformer blocks
# Look for: a generate() method or forward() that does autoregressive decode

# Step 5: Identify vocab size and output projection
# Look for: lm_head, codec_head, or similar Linear layers at the output
```

**Evidence required before proceeding:**
- [ ] Actual class name(s)
- [ ] Actual module path to talker decoder (e.g. `model.talker.layers` or `model.decoder.blocks`)
- [ ] Number of layers in talker
- [ ] Hidden size, intermediate size, num heads, num KV heads
- [ ] Vocab size at talker output
- [ ] Whether there is a code predictor and where it sits
- [ ] How generation is triggered (method name, signature)

### A.4 — Minimal Inference Script

Write this ONLY after A.3 produces the evidence above. Fill in the blanks.

```python
# phase_a_baseline.py
import torch
import time
import soundfile as sf
import numpy as np

# Fill in after A.3:
# from ??? import ???
# TALKER_LAYERS = ???  (from config inspection)
# SAMPLE_RATE = ???    (from config or model card — expected 24000)

def run_baseline(text: str, output_path: str = "output.wav"):
    t_load_start = time.perf_counter()
    
    model = ...  # actual load
    processor = ...  # actual processor load
    
    t_load_end = time.perf_counter()
    print(f"Model load time: {(t_load_end - t_load_start)*1000:.0f}ms")
    
    # Warmup (see benchmarking section)
    _ = run_inference(model, processor, "warmup text")
    torch.cuda.synchronize()
    
    # Timed run
    t_start = time.perf_counter()
    
    result = run_inference(model, processor, text)
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    # Extract audio from result (format TBD from A.3)
    # audio = ???  (numpy array or tensor)
    # sr = ???
    
    audio_duration_s = len(audio) / sr
    total_time_s = t_end - t_start
    rtf = total_time_s / audio_duration_s
    
    print(f"Text: '{text}'")
    print(f"Audio duration: {audio_duration_s*1000:.0f}ms")
    print(f"Generation time: {total_time_s*1000:.0f}ms")
    print(f"RTF: {rtf:.3f}")
    
    sf.write(output_path, audio, sr)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    run_baseline("Hello, this is a test of the Qwen TTS system.")
```

**Phase A deliverables:**
- WAV file that plays correctly
- Measured: model load time, generation time, RTF, audio duration
- Documented: actual class names, module paths, config values

---

## Phase B — Streaming Audio

**Goal:** Stream partial audio chunks as they decode, not buffer the full utterance.

### B.1 — Determine if Streaming is Even Possible

**This is a critical unknown.** Qwen3-TTS may require the full token sequence before the vocoder can decode audio. Or it may support frame-by-frame decoding.

Investigation steps:

```python
# streaming_probe.py

# Question 1: Does the vocoder/speech tokenizer decode per-frame?
# Look at the speech tokenizer's decode method
# from huggingface_hub import hf_hub_download
# path = hf_hub_download(..., 'speech_tokenizer/...')
# Inspect: does decode() take a full sequence or single frames?

# Question 2: Does the generation loop expose intermediate tokens?
# Look for: streamer= argument in generate(), or hooks, or callbacks
# Check if model.generate() accepts TextIteratorStreamer

# Question 3: What is the codec frame rate?
# Expected from research: 12.5 Hz = one frame per 80ms of audio
# Verify: check config for 'frame_rate', 'codec_frame_rate', or similar

# Question 4: How many talker tokens = one audio frame?
# This determines the minimum chunk size for streaming
```

**If streaming is NOT supported natively:**

Option A — Fake streaming: generate fully, then chunk the output audio into frames
```python
# chunk_size_ms = 100  (100ms chunks)
# chunk_samples = sample_rate * chunk_size_ms // 1000
# for i in range(0, len(audio), chunk_samples):
#     yield audio[i:i+chunk_samples]
```
This gives Pipecat streaming semantics but doesn't reduce TTFC.

Option B — Hook into HuggingFace streamer:
```python
from transformers import TextIteratorStreamer
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False)
# Pass to generate() as streamer=streamer
# Then: for token in streamer: decode_to_audio(token)
```
This gives real streaming but only works if the model's generate() accepts a streamer.

Option C — Manual decode loop (required for megakernel integration anyway):
```python
# Replace model.generate() with a manual loop:
# past_kv_cache = None
# for step in range(max_tokens):
#     logits, past_kv_cache = model.talker_forward(input_ids, past_kv_cache)
#     token = logits.argmax(-1)
#     codec_tokens.append(token)
#     if len(codec_tokens) >= frame_size:
#         audio_chunk = vocoder.decode(codec_tokens[-frame_size:])
#         yield audio_chunk
```

**UNKNOWN:** Whether `model.talker_forward()` exists or whether the talker forward pass is accessible without calling full `model.generate()`. Must verify from A.3 module inspection.

### B.2 — Chunking Strategy (After B.1 Evidence)

Fill this in after B.1 investigation:

```
Codec frame rate: ??? Hz  (expected ~12.5)
Samples per frame: ??? = sample_rate / frame_rate = 24000 / 12.5 = 1920 samples
Bytes per frame (int16 mono): ??? = 1920 * 2 = 3840 bytes
Talker tokens per frame: ???  (unknown — depends on model)
```

**Target for TTFC < 60ms:**
- Must emit first audio chunk within 60ms of receiving text
- If prefill takes 30ms and first decode step takes 1ms, minimum TTFC ≈ 31ms + vocoder overhead
- If vocoder requires N frames before decoding, TTFC = 31ms + N * (1ms/token * tokens_per_frame)

### B.3 — Async Generator

```python
# phase_b_streaming.py
import asyncio
import numpy as np
from typing import AsyncGenerator, Tuple

async def synthesize_streaming(
    model,
    processor,
    text: str,
) -> AsyncGenerator[Tuple[bytes, int], None]:
    """
    Yields (audio_chunk: bytes, sample_rate: int) incrementally.
    Implementation depends on B.1 findings — fill in the actual approach.
    """
    # PLACEHOLDER — replace with actual implementation after B.1
    
    # Option A (fake streaming — always works):
    audio, sr = run_full_inference(model, processor, text)
    chunk_samples = sr // 10  # 100ms chunks
    audio_int16 = (audio * 32767).astype(np.int16)
    for i in range(0, len(audio_int16), chunk_samples):
        chunk = audio_int16[i:i+chunk_samples].tobytes()
        yield chunk, sr
        await asyncio.sleep(0)  # yield control to event loop
    
    # Option B/C (real streaming — implement after B.1 confirms feasibility)

# Test
async def test_streaming():
    t_first_chunk = None
    t_start = time.perf_counter()
    total_samples = 0
    
    async for chunk, sr in synthesize_streaming(model, processor, "Hello world"):
        if t_first_chunk is None:
            t_first_chunk = time.perf_counter()
            print(f"TTFC: {(t_first_chunk - t_start)*1000:.1f}ms")
        total_samples += len(chunk) // 2  # int16 = 2 bytes per sample
    
    total_time = time.perf_counter() - t_start
    audio_duration = total_samples / sr
    print(f"RTF: {total_time / audio_duration:.3f}")
```

**Phase B deliverables:**
- Streaming generator that yields audio chunks
- Measured TTFC from this approach
- Documented: whether streaming is real or fake, and why

---

## Phase C — Pipecat Integration

**Goal:** Working end-to-end voice pipeline: mic → STT → LLM → TTS → speaker.

### C.1 — Pin Pipecat Version and Read Source

```bash
pip install pipecat-ai[silero,local]
python -c "import pipecat; print(pipecat.__version__)"
pip show pipecat-ai

# Find TTSService source
python -c "import pipecat.services.tts_service as m; print(m.__file__)"
# Then read that file to get the ACTUAL base class API
```

**Read the actual TTSService source before writing the subclass.** The public docs may lag the implementation. What to confirm:
- Does `run_tts` return `AsyncGenerator[Frame | None, None]` or something else?
- What is the exact signature of `TTSAudioRawFrame`?
- Is `context_id` a real parameter or does it differ by version?
- Does the base class handle `TTSStartedFrame`/`TTSStoppedFrame` automatically?

### C.2 — Custom TTSService

Write this ONLY after C.1 confirms the actual API.

```python
# pipecat_service/qwen_tts_service.py

from pipecat.services.tts_service import TTSService
# Import Frame types — verify exact names from C.1
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from collections.abc import AsyncGenerator
from pipecat.frames.frames import Frame

class QwenTTSService(TTSService):
    def __init__(self, model, processor, sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model = model
        self.processor = processor
        self._sample_rate = sample_rate

    async def run_tts(self, text: str, **kwargs) -> AsyncGenerator[Frame | None, None]:
        # context_id parameter — verify from C.1 whether this exists in this version
        context_id = kwargs.get("context_id", "")
        
        try:
            yield TTSStartedFrame()  # or TTSStartedFrame(context_id=context_id) — verify
            
            async for audio_chunk, sr in synthesize_streaming(
                self.model, self.processor, text
            ):
                yield TTSAudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=sr,
                    num_channels=1,
                    # context_id=context_id,  # only if verified from C.1
                )
            
            yield TTSStoppedFrame()
            
        except Exception as e:
            yield ErrorFrame(error=str(e))
```

### C.3 — Full Pipeline

```python
# pipeline/voice_agent.py
import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

# STT options (pick one based on what you have):
# - pipecat-ai[silero] for local Silero VAD + Whisper (no API key needed)
# - DeepgramSTTService (needs API key)
# - WhisperSTTService (local, slower)

# LLM options:
# - AnthropicLLMService (needs API key)
# - OpenAILLMService (needs API key)
# - OllamaLLMService (local)

async def main():
    # Transport: local audio (mic + speaker)
    # Verify: does pipecat-ai[local] provide LocalAudioTransport?
    # Alternative: use Daily.co transport if local doesn't work

    transport = ...   # fill after C.1
    stt = ...         # fill after verifying pipecat STT options
    llm = ...         # fill after confirming API key access
    tts = QwenTTSService(model=model, processor=processor)

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
```

**Phase C deliverables:**
- End-to-end pipeline working (speak → hear response)
- Measured: end-to-end latency, audio quality, any frame drops

---

## Phase D — Megakernel Integration

**Goal:** Replace only the autoregressive talker decode loop with the megakernel. Keep everything else (code predictor, vocoder) unchanged.

### D.1 — Clone and Inspect the Megakernel

```bash
git clone https://github.com/AlpinDale/qwen_megakernel
cd qwen_megakernel

# Read these files carefully:
# csrc/kernel.cu — all hardcoded constants
# csrc/torch_bindings.cpp — Python-facing API
# qwen_megakernel/model.py — Python wrapper
# qwen_megakernel/__init__.py — public API

# Extract all hardcoded constants:
grep -E '#define [A-Z_]+ [0-9]+' csrc/kernel.cu
```

**Expected output (verify against actual):**
```
#define NUM_LAYERS      28
#define NUM_KV_HEADS    8
#define NUM_HEADS       32
#define HEAD_DIM        128
#define HIDDEN_SIZE     1024
#define INTERMEDIATE_SIZE 3072
#define VOCAB_SIZE      151936
#define MAX_SEQ_LEN     2048
```

### D.2 — Compatibility Matrix

Run this inspection to get real numbers from the Qwen3-TTS talker config. Fill the table from actual output.

```python
# compatibility_check.py
import json

# Load talker config from HuggingFace
from huggingface_hub import hf_hub_download
# Look for: config.json, talker_config.json, or similar
path = hf_hub_download("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "config.json")
with open(path) as f:
    config = json.load(f)
print(json.dumps(config, indent=2))
```

**Compatibility matrix (fill from actual inspection):**

| Parameter | Megakernel (kernel.cu) | Qwen3-TTS Talker | Match? | Action Required |
|-----------|----------------------|------------------|--------|-----------------|
| NUM_LAYERS | 28 | ??? | ??? | Change to talker depth |
| HIDDEN_SIZE | 1024 | ??? | ??? | — |
| INTERMEDIATE_SIZE | 3072 | ??? | ??? | Change if different |
| NUM_HEADS | 32 | ??? | ??? | Change if different |
| NUM_KV_HEADS | 8 | ??? | ??? | Change if different |
| HEAD_DIM | 128 | ??? | ??? | — |
| VOCAB_SIZE | 151936 | ??? | ??? | Change to codec vocab |
| MAX_SEQ_LEN | 2048 | ??? | ??? | Increase if needed |
| Rope theta | ??? | ??? | ??? | Change if different |
| Attention type | ??? | ??? | ??? | GQA vs MHA |
| Norm type | RMSNorm | ??? | ??? | — |
| Activation | SiLU/GeGLU | ??? | ??? | Critical — affects MLP kernel |

**Additional compatibility checks:**

```python
# Check weight layout compatibility
# The megakernel expects weights in a specific format (LDGLayerWeights struct)
# 11 pointers per layer: q, k, v, o projections + norms + MLP

# Verify from model.py which keys it loads:
# grep -n "load\|state_dict\|weight" qwen_megakernel/model.py

# Verify HF model weight keys:
for name, param in model.named_parameters():
    if 'talker' in name and 'layer' in name:
        print(name, param.shape)
# Compare output to what megakernel/model.py expects
```

**KV cache layout check:**
```python
# Megakernel KV cache shape:
# [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]
# Verify this matches HF model's cache format:
# outputs = model.talker_forward(..., use_cache=True)
# print(type(outputs.past_key_values))
# print(outputs.past_key_values[0][0].shape)  # (batch, heads, seq, head_dim) typically
```

### D.3 — Weight Extraction

The megakernel needs weights extracted from the HF model in a specific layout.

```python
# weight_extraction.py
# PLACEHOLDER — fill in after D.1 and D.2 identify the actual weight keys

def extract_talker_weights(hf_model) -> dict:
    """
    Extract talker weights in the format expected by qwen_megakernel/model.py.
    The megakernel expects specific key names — read model.py to find them.
    """
    talker = hf_model.talker  # VERIFY this path from A.3
    state = talker.state_dict()
    
    # Map HF weight keys → megakernel weight keys
    # This mapping is UNKNOWN until D.1 + D.2 are done
    # Example (speculative, do not use until verified):
    # megakernel_weights = {
    #     'embed_weight': state['model.embed_tokens.weight'],
    #     'layers.0.input_layernorm': state['model.layers.0.input_layernorm.weight'],
    #     ...
    # }
    
    return megakernel_weights
```

### D.4 — Integration

Only implement after D.2 matrix is complete and all mismatches are resolved.

```python
# megakernel_tts_backend.py

import sys
sys.path.insert(0, './qwen_megakernel')
from qwen_megakernel import Decoder

class MegakernelTTSBackend:
    def __init__(self, hf_model, processor):
        self.hf_model = hf_model
        self.processor = processor
        # Extract weights and build megakernel decoder
        weights = extract_talker_weights(hf_model)
        self.mk_decoder = Decoder(weights=weights, tokenizer=processor.tokenizer)
    
    async def synthesize_streaming(self, text: str):
        """
        Uses megakernel for talker decode.
        Uses original HF code predictor + vocoder for audio decoding.
        """
        # Step 1: Prefill — run the full model's prefill pass to get initial KV cache
        # This part still uses HF — the megakernel only accelerates the decode loop
        # UNKNOWN: how to separate prefill from decode in this model
        
        # Step 2: Decode loop with megakernel
        codec_tokens = []
        for step in range(max_steps):
            # Single decode step via megakernel
            token_id = self.mk_decoder.step(last_token_id)
            codec_tokens.append(token_id)
            
            if token_id == EOS_TOKEN:  # verify EOS token ID from D.2
                break
            
            # Step 3: Every frame_size tokens, run code predictor + vocoder
            # UNKNOWN: frame_size — must verify from B.1
            if len(codec_tokens) % frame_size == 0:
                # Run code predictor (still HF)
                codes = run_code_predictor(
                    self.hf_model, codec_tokens[-frame_size:]
                )
                # Decode to audio (still HF vocoder)
                audio_chunk = run_vocoder(self.hf_model, codes)
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                yield audio_bytes, 24000
```

**Phase D deliverables:**
- Megakernel decode loop running correctly
- Correctness validated: compare output audio against Phase A baseline
- tok/s measured and reported

---

## Benchmarking Methodology

The assignment explicitly evaluates benchmarking rigor. Do it properly.

### General Rules

1. **Always warmup** before measuring — first inference includes model JIT compilation, cache warming
2. **Always `torch.cuda.synchronize()`** before stopping the timer — GPU work is async
3. **Report variance** — run 5+ trials, report mean ± std
4. **Report GPU utilization** — use `nvidia-smi dmon` or `nvitop`

### tok/s Measurement

```python
import torch
import time
import numpy as np

def measure_toks_per_second(decoder, warmup_steps=10, measure_steps=100):
    # Warmup
    for _ in range(warmup_steps):
        decoder.step(some_token_id)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for trial in range(5):
        t0 = time.perf_counter()
        for _ in range(measure_steps):
            decoder.step(some_token_id)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(measure_steps / (t1 - t0))
    
    mean = np.mean(times)
    std = np.std(times)
    print(f"tok/s: {mean:.1f} ± {std:.1f}")
    return mean, std
```

### TTFC Measurement

```python
def measure_ttfc(backend, text: str, trials: int = 5):
    ttfc_times = []
    
    for _ in range(trials):
        t_start = time.perf_counter()
        first_chunk = True
        
        async for chunk, sr in backend.synthesize_streaming(text):
            if first_chunk:
                torch.cuda.synchronize()  # ensure GPU work is done
                ttfc = (time.perf_counter() - t_start) * 1000
                ttfc_times.append(ttfc)
                first_chunk = False
            break  # only need the first chunk time
    
    print(f"TTFC: {np.mean(ttfc_times):.1f} ± {np.std(ttfc_times):.1f} ms")
    print(f"Target: < 60ms | {'PASS' if np.mean(ttfc_times) < 60 else 'FAIL'}")
```

### RTF Measurement

```python
def measure_rtf(backend, text: str, trials: int = 5):
    rtf_values = []
    
    for _ in range(trials):
        t_start = time.perf_counter()
        total_samples = 0
        sample_rate = None
        
        async for chunk, sr in backend.synthesize_streaming(text):
            total_samples += len(chunk) // 2  # int16 = 2 bytes
            sample_rate = sr
        
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        audio_duration = total_samples / sample_rate
        gen_time = t_end - t_start
        rtf_values.append(gen_time / audio_duration)
    
    print(f"RTF: {np.mean(rtf_values):.3f} ± {np.std(rtf_values):.3f}")
    print(f"Target: < 0.15 | {'PASS' if np.mean(rtf_values) < 0.15 else 'FAIL'}")
```

### GPU Utilization

```bash
# Run during inference in a separate terminal:
nvidia-smi dmon -s u -d 1 > gpu_util.txt &
# (run your benchmark)
kill %1
cat gpu_util.txt

# Or use nvitop for interactive monitoring:
pip install nvitop && nvitop
```

---

## Execution Order (Fastest Path to Demo)

```
Day 1 (target):

Morning:
  [1] Provision Vast.ai RTX 5090 instance (30min)
  [2] Phase A.1 — environment setup (30min)
  [3] Phase A.2 — discover actual Qwen3-TTS package API (60min)
  [4] Phase A.3 — inspect model, capture evidence (30min)
  [5] Phase A.4 — baseline inference working, WAV saved (60min)

Afternoon:
  [6] Phase B.1 — determine streaming feasibility (45min)
  [7] Phase B.3 — streaming generator (30min)
  [8] Phase C.1 — pin Pipecat, read TTSService source (30min)
  [9] Phase C.2 — custom TTSService (30min)
  [10] Phase C.3 — full pipeline working (60min)
  [11] Demo recording with vanilla TTS (done — ship this if needed)

Evening (if time):
  [12] Phase D.1 — inspect megakernel constants (30min)
  [13] Phase D.2 — compatibility matrix (45min)
  [14] Phase D.3/D.4 — megakernel integration (2-3h)
  [15] Benchmarks for all phases (30min)
  [16] README + numbers (30min)
```

**The demo is shippable after step 11.** The megakernel integration is a performance upgrade, not a requirement for a working demo. Prioritize correctness first.

---

## Deployment

### Server (Vast.ai RTX 5090)

```bash
# 1. Install Caddy (auto-HTTPS)
apt install -y caddy

# 2. Caddyfile — replace <your-domain> with Vast.ai IP or a domain pointing to it
cat > /etc/caddy/Caddyfile <<'EOF'
<your-domain> {
    reverse_proxy /ws localhost:8000
    reverse_proxy localhost:8000
}
EOF
systemctl restart caddy

# 3. Start the FastAPI server
cd /workspace/qwen-megakernel-pipecat
uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
```

**FastAPI app must expose:**
- `GET /` — health check
- `WS /ws` — WebSocket endpoint for Pipecat `WebsocketServerTransport`

**CORS config in voice_agent.py:**
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ["ALLOWED_ORIGIN"]],  # e.g. https://your-app.vercel.app
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Frontend (Vercel)

```bash
cd client
npm install
# Set env var in Vercel dashboard or .env.local:
# VITE_WS_URL=wss://<your-vast-ai-host>/ws
vercel deploy --prod
```

The React app reads `VITE_WS_URL` at build time and passes it to `WebSocketTransport`.

### Required env vars

| Location | Variable | Value |
|----------|----------|-------|
| Vast.ai server | `ALLOWED_ORIGIN` | `https://<your-vercel-app>.vercel.app` |
| Vast.ai server | `HF_TOKEN` | HuggingFace token (gated model access) |
| Vast.ai server | `OPENAI_API_KEY` | or whichever LLM provider |
| Vast.ai server | `DEEPGRAM_API_KEY` | if using Deepgram STT |
| Vercel | `VITE_WS_URL` | `wss://<your-vast-ai-host>/ws` |

---

## README Outline (Fill in with Real Numbers)

```markdown
## Architecture

[diagram]

## Setup

[exact commands that work on RTX 5090]

## Running the Demo

[exact command]

## Performance Numbers

| Metric | Baseline (HF) | With Megakernel |
|--------|---------------|-----------------|
| tok/s  | ???           | ???             |
| TTFC   | ???ms         | ???ms           |
| RTF    | ???           | ???             |
| E2E latency | ???ms   | ???ms           |

## Benchmarking Methodology

- Warmup: N steps before timing
- Trials: 5 runs, reported as mean ± std
- torch.cuda.synchronize() before all timer stops
- GPU: RTX 5090, CUDA 12.8, Driver ???

## Known Limitations

[be honest here]

## What I Would Do With More Time

[be honest here too]
```

---

## What This Plan Does NOT Assume

- The exact Qwen3-TTS class name or import path
- Internal module hierarchy of the model
- Whether streaming is natively supported
- Exact kernel constants for the megakernel
- Which Pipecat version is current and what its API looks like

All of the above must be discovered by running the inspection steps before writing implementation code.
