# Architecture: Qwen3-TTS + Megakernel + Pipecat Voice Agent

> **Purpose:** Reference architecture for implementation. All component boundaries are
> explicit. All unknowns are labeled. Review this before writing any code.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER DEVICE (Browser)                              │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │              React UI  (@pipecat-ai/client-react)                    │  │
│   │                                                                      │  │
│   │   <PipecatClientProvider client={pipecatClient}>                     │  │
│   │     <PipecatClientAudio />          ← handles bot audio playback     │  │
│   │     <PipecatClientMicToggle />      ← mic on/off button              │  │
│   │     <VoiceVisualizer />             ← real-time input level          │  │
│   │     <TranscriptDisplay />           ← custom component               │  │
│   │     <MetricsPanel />                ← TTFC, RTF, tok/s               │  │
│   │   </PipecatClientProvider>                                           │  │
│   │                                                                      │  │
│   │   PipecatClient(@pipecat-ai/client-js)                               │  │
│   │     transport: WebSocketTransport (ws://<server>:PORT)               │  │
│   │     enableMic: true                                                  │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                    ↑ WebSocket (RTVI protocol over ws://)                   │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────────────────┐
│                   VAST.AI RTX 5090 SERVER                                   │
│                          │                                                  │
│   ┌──────────────────────▼──────────────────────────────────────────────┐   │
│   │               Pipecat Pipeline (asyncio)                            │   │
│   │                                                                     │   │
│   │  WebSocketTransport ──► VoiceActivityDetector ──► WhisperSTT        │   │
│   │         (input)              (Silero VAD)        (local/Deepgram)   │   │
│   │                                                       │             │   │
│   │                                              TranscriptFrame        │   │
│   │                                                       │             │   │
│   │                                               ┌───────▼───────┐    │   │
│   │                                               │  LLM Service  │    │   │
│   │                                               │ (OpenAI/Ollama│    │   │
│   │                                               │  /Anthropic)  │    │   │
│   │                                               └───────┬───────┘    │   │
│   │                                                       │             │   │
│   │                                              TextFrame (streamed)   │   │
│   │                                                       │             │   │
│   │                                          ┌────────────▼──────────┐  │   │
│   │                                          │   QwenTTSService      │  │   │
│   │                                          │  (TTSService subclass) │  │   │
│   │                                          └────────────┬──────────┘  │   │
│   │                                                       │             │   │
│   │                                          TTSAudioRawFrame (chunked) │   │
│   │                                                       │             │   │
│   │                                          WebSocketTransport ◄───────┘   │
│   │                                               (output)                  │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              TTS Inference Layer (QwenTTSBackend)                   │   │
│   │                                                                     │   │
│   │   Text prompt                                                       │   │
│   │       │                                                             │   │
│   │       ▼                                                             │   │
│   │  ┌──────────────────────────────────────────────────────────────┐  │   │
│   │  │              Qwen3-TTS HuggingFace Model                     │  │   │
│   │  │                                                              │  │   │
│   │  │  [Prefill Pass]  →  [Talker Decoder]  →  [Code Predictor]   │  │   │
│   │  │   (HF standard)      (REPLACED by        (HF, unchanged)    │  │   │
│   │  │                       megakernel)                           │  │   │
│   │  │                            │              [Vocoder/DAC]     │  │   │
│   │  │                            └─────────────►  (HF, unchanged) │  │   │
│   │  └──────────────────────────────────────────────────────────────┘  │   │
│   │                                    │                                │   │
│   │                    async generator: yield (bytes, sample_rate)     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CUDA Megakernel (Phase D)                        │   │
│   │                                                                     │   │
│   │   qwen_megakernel/Decoder                                           │   │
│   │   ├── 128 persistent thread blocks × 512 threads                   │   │
│   │   ├── Single non-cooperative kernel launch                         │   │
│   │   ├── Weight prefetch: layer N+1 loads while layer N runs          │   │
│   │   ├── 71% GDDR7 bandwidth utilization                              │   │
│   │   └── ~1,000 tok/s on RTX 5090 (sm_120 / Blackwell)               │   │
│   │                                                                     │   │
│   │   GPU memory layout:                                                │   │
│   │   ├── Model weights (bfloat16, ~1.2GB for 0.6B)                    │   │
│   │   ├── KV cache [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]  │   │
│   │   └── Activation buffers (double-buffered per layer)               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Map

### Layer 1 — Web UI (@pipecat-ai/client-react)

Pipecat ships official client libraries that handle WebSocket transport, mic access, and audio
playback natively. No custom audio plumbing needed in the browser.

| Package        | NPM name                          | Purpose                                  |
| -------------- | --------------------------------- | ---------------------------------------- |
| Core client    | `@pipecat-ai/client-js`           | `PipecatClient` — transport, mic, events |
| React bindings | `@pipecat-ai/client-react`        | Components + hooks                       |
| Transport      | `@pipecat-ai/websocket-transport` | WebSocket to Pipecat server              |

**Client setup (one-time, outside React tree):**

```ts
import { PipecatClient } from "@pipecat-ai/client-js";
import { WebSocketTransport } from "@pipecat-ai/websocket-transport";

const client = new PipecatClient({
  transport: new WebSocketTransport(),
  enableMic: true,
  params: { baseUrl: "ws://<vast-ai-ip>:<port>" },
});
```

**React component tree:**

```tsx
<PipecatClientProvider client={client}>
  <PipecatClientAudio /> {/* bot audio playback — automatic */}
  <MicToggleButton /> {/* uses usePipecatClientMicControl */}
  <VoiceVisualizer /> {/* real-time mic level bars */}
  <TranscriptLog /> {/* uses usePipecatConversation */}
  <MetricsPanel /> {/* TTFC, RTF, tok/s — custom state */}
</PipecatClientProvider>
```

**Key hooks used:**

- `usePipecatClientMicControl` → `{ enableMic, isMicEnabled }` — mic toggle
- `usePipecatConversation` → message stream (user + bot transcripts)
- `usePipecatClientTransportState` → connection status
- `useRTVIClientEvent` → subscribe to custom metric events from server

**UI layout (minimal):**

```
┌─────────────────────────────────────────────┐
│         Qwen3-TTS Voice Agent                │
├──────────────────────┬──────────────────────┤
│  [Mic Toggle]        │  [VoiceVisualizer]    │
│  [Connect Button]    │  (waveform bars)      │
├──────────────────────┴──────────────────────┤
│  Transcript                                  │
│  You: ____________                           │
│  Agent: __________                           │
├─────────────────────────────────────────────┤
│  TTFC: ___ms   RTF: ___   tok/s: ___         │
└─────────────────────────────────────────────┘
```

**Why this over Gradio:**

- `PipecatClientAudio` handles audio output streaming natively — this is exactly what we need
- The client speaks RTVI protocol, which Pipecat server already understands
- No manual WebSocket audio chunking or MediaRecorder wiring
- `VoiceVisualizer` is ready-made mic level display
- This is the officially supported path — less risk of transport mismatch

---

### Layer 2 — Pipecat Pipeline

Pipecat is the **orchestration layer**. It connects transport → STT → LLM → TTS as a frame-based async pipeline.

```
Frame types that flow through the pipeline:

AudioRawFrame      — raw PCM audio bytes from mic
TranscriptionFrame — text output of STT
LLMFullResponseFrame — LLM text (may come in chunks)
TTSAudioRawFrame   — audio PCM bytes from TTS
```

**Pipeline topology:**

```python
Pipeline([
    transport.input(),          # WebSocket → AudioRawFrame
    SileroVADAnalyzer(),        # detect speech boundaries
    DeepgramSTTService(),       # OR WhisperSTTService()
    LLMService(),               # OpenAI / Ollama / Anthropic
    QwenTTSService(),           # custom — see Layer 3
    transport.output(),         # TTSAudioRawFrame → WebSocket
])
```

**Pipecat version pin:** `pipecat-ai==0.0.x` — read from source after install. Do not assume API from docs.

**Transport choice:**

- Primary: `WebsocketServerTransport` (Pipecat built-in, works with Gradio's WebSocket connection)
- Fallback: `DailyTransport` (Daily.co managed, needs API key but handles all audio I/O)

---

### Layer 3 — QwenTTSService (custom TTSService subclass)

This is the **adapter** between Pipecat and the TTS inference layer.

```python
class QwenTTSService(TTSService):
    """
    Pipecat TTSService subclass.
    Delegates to QwenTTSBackend for actual inference.
    Yields TTSAudioRawFrame per chunk as they arrive.
    """

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame()
        async for audio_bytes, sr in self.backend.synthesize_streaming(text):
            yield TTSAudioRawFrame(audio=audio_bytes, sample_rate=sr, num_channels=1)
        yield TTSStoppedFrame()
```

**Key constraint:** This class must NOT do any model inference itself. It only wraps the backend generator. This keeps Pipecat's async event loop clean — all blocking GPU work happens in the backend on a thread/process.

---

### Layer 4 — QwenTTSBackend (inference layer)

This is where the model runs. Two implementations:

#### Phase A/B/C — HuggingFace baseline:

```
QwenTTSBackend (HF)
├── model: Qwen3TTS (HF AutoModel, loaded once at startup)
├── processor: AutoProcessor
└── synthesize_streaming(text) → AsyncGenerator[bytes, int]
    ├── tokenize input
    ├── run model.generate() or manual decode loop
    ├── for each codec frame:
    │   ├── run vocoder/code_predictor
    │   └── yield (audio_bytes, 24000)
    └── done
```

#### Phase D — Megakernel backend:

```
QwenTTSBackend (Megakernel)
├── hf_model: loaded for prefill + code_predictor + vocoder only
├── mk_decoder: qwen_megakernel.Decoder (talker decode loop)
│   └── weights extracted from hf_model.talker at load time
└── synthesize_streaming(text) → AsyncGenerator[bytes, int]
    ├── tokenize input
    ├── prefill: hf_model.talker_prefill(input_ids) → initial KV cache
    ├── decode loop (megakernel):
    │   ├── mk_decoder.step(last_token) → next codec token (~1ms)
    │   └── every frame_size tokens:
    │       ├── hf_model.code_predictor(codec_tokens) → 32 codebook tokens
    │       ├── hf_model.vocoder(codebook_tokens) → audio waveform
    │       └── yield (audio_bytes, 24000)
    └── done on EOS token
```

---

### Layer 5 — CUDA Megakernel

Source: `github.com/AlpinDale/qwen_megakernel`

**Internal structure:**

```
qwen_megakernel/
├── csrc/
│   ├── kernel.cu          ← 1,200-line CUDA kernel (do NOT modify lightly)
│   │   ├── #define NUM_LAYERS, NUM_HEADS, HIDDEN_SIZE, etc.
│   │   ├── LDGLayerWeights struct (11 weight pointers per layer)
│   │   ├── decode_step() ← single token, single call
│   │   └── persistent thread blocks (128 blocks × 512 threads)
│   └── torch_bindings.cpp ← pybind11 → Python-callable
├── qwen_megakernel/
│   ├── __init__.py        ← Decoder class
│   └── model.py           ← weight loading + forward
└── setup.py               ← builds with nvcc for sm_120
```

**What must be verified before Phase D (see compatibility_check.py in implementation_plan.md):**

- `NUM_LAYERS` matches Qwen3-TTS talker depth
- `HIDDEN_SIZE`, `INTERMEDIATE_SIZE` match
- `VOCAB_SIZE` — talker outputs codec tokens (NOT text tokens), so this will likely differ from 151936
- KV cache layout matches HF model's `past_key_values` shape
- Weight key names match what `model.py` expects to load

**If talker ≠ 0.6B architecture:** Document which `#define` lines changed and why.

---

## Data Flow: One Full Turn

```
1. User speaks into browser microphone
        │
        ▼
2. Browser MediaRecorder → PCM audio → WebSocket to server

3. Pipecat WebSocketTransport receives AudioRawFrame
        │
        ▼
4. SileroVAD detects end of speech → emits complete utterance

5. WhisperSTT / DeepgramSTT transcribes → TranscriptionFrame("what is AI?")
        │
        ▼
6. LLMService sends to LLM → streams response text
   ("Artificial intelligence is...")
        │
        ▼  (text arrives in chunks as LLM streams)
7. QwenTTSService.run_tts("Artificial intelligence is...")
        │
        ▼
8. QwenTTSBackend.synthesize_streaming(text)
   ├── [~5-15ms]  tokenize + prefill
   ├── [~1ms]     megakernel decode step → codec token 1
   ├── [~1ms]     decode step → codec token 2
   ├── ...
   ├── [N tokens] codec frame complete → code_predictor → vocoder
   └── yield (80ms of audio PCM, 24000Hz)  ← FIRST CHUNK (TTFC measured here)
        │
        ▼
9. TTSAudioRawFrame → WebSocketTransport.output()
        │
        ▼
10. WebSocket → browser AudioContext → speaker playback begins
    (while steps 8-9 continue generating next frames)
```

---

## File Structure

```
qwen-megakernel-pipecat/
│
├── README.md                        ← deliverable: setup + numbers + demo link
│
├── docs/
│   ├── architecture.md              ← this file
│   ├── implementation_plan.md       ← phase-by-phase plan
│   └── takehome_project.docx.md    ← original brief
│
├── qwen_megakernel/                 ← cloned from AlpinDale/qwen_megakernel
│   ├── csrc/
│   │   ├── kernel.cu
│   │   └── torch_bindings.cpp
│   ├── qwen_megakernel/
│   │   ├── __init__.py
│   │   └── model.py
│   └── setup.py
│
├── server/                              ← Python, runs on GPU server
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── tts_backend_hf.py        ← Phase A/B: HuggingFace baseline backend
│   │   └── tts_backend_mk.py        ← Phase D: megakernel backend
│   │
│   ├── pipecat_services/
│   │   ├── __init__.py
│   │   └── qwen_tts_service.py      ← TTSService subclass (Pipecat adapter)
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── voice_agent.py           ← assembles and runs Pipecat pipeline + FastAPI
│
├── client/                              ← React app, runs in browser
│   ├── package.json
│   │   # deps: @pipecat-ai/client-js, @pipecat-ai/client-react,
│   │   #       @pipecat-ai/websocket-transport, react, typescript, vite
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx                  ← PipecatClientProvider + layout
│   │   ├── components/
│   │   │   ├── MicToggle.tsx        ← usePipecatClientMicControl
│   │   │   ├── TranscriptLog.tsx    ← usePipecatConversation
│   │   │   └── MetricsPanel.tsx     ← TTFC / RTF / tok/s display
│   │   └── lib/
│   │       └── pipecatClient.ts     ← PipecatClient + WebSocketTransport init
│   └── index.html
│
├── scripts/
│   ├── phase_a_inspect_model.py     ← A.3: dumps model structure, fills blanks
│   ├── phase_a_baseline.py          ← A.4: WAV output + RTF measurement
│   ├── phase_b_streaming_probe.py   ← B.1: test streaming feasibility
│   ├── phase_d_compat_check.py      ← D.2: diff kernel constants vs model config
│   └── benchmark.py                 ← final benchmark: TTFC, RTF, tok/s, E2E
│
├── requirements.txt
└── Makefile                         ← build kernel, run dev, run benchmark
```

---

## Build & Runtime Dependencies

```
# GPU requirement
CUDA 12.8+, Driver 570+, RTX 5090 (sm_120)

# Python (server — requirements.txt)
torch>=2.4.0          (cu128 build)
transformers>=4.47    (Qwen3-TTS support — verify exact version)
huggingface_hub
pipecat-ai[silero]    (VAD + local audio support)
fastapi               (serves the WebSocket endpoint the React client connects to)
uvicorn
soundfile
numpy

# Optional STT (pick one)
openai-whisper        (local, no API key)
deepgram-sdk          (cloud, needs key — lower latency)

# Build (megakernel)
nvcc (from CUDA toolkit)
pybind11

# JavaScript (client/ — package.json)
@pipecat-ai/client-js
@pipecat-ai/client-react
@pipecat-ai/websocket-transport
react
react-dom
typescript
vite                  (dev server + build)
```

---

## Latency Budget (Target: TTFC < 60ms)

```
Step                        Target time    Notes
─────────────────────────────────────────────────────────
Mic → server (WebSocket)         ~5ms      local or Vast.ai
VAD endpoint detection           ~0ms      runs in real-time
STT (Deepgram cloud)            ~200ms     ← bottleneck, but out of TTFC scope
STT (Whisper local)             ~500ms     ← worse
LLM first token                 ~200ms     depends on model/API
─────────────────────────────────────────────────────────
TTS prefill                     ~10-20ms   tokenize + HF prefill pass
TTS first decode step            ~1ms      megakernel
Min tokens before vocoder        ~5ms      depends on codec frame size (TBD)
Code predictor + vocoder         ~5-10ms   single frame
─────────────────────────────────────────────────────────
TTS TTFC subtotal               ~20-35ms   this is what we optimize
Audio → browser                  ~5ms

TOTAL E2E (excl. STT+LLM)      ~30-45ms   reasonable to hit < 60ms
```

> Note: The < 60ms TTFC target is measured from text-in to first audio chunk out — it does NOT
> include STT or LLM latency. That is standard practice for TTS benchmarking.

---

## Component Interfaces (Contract Summary)

```python
# src/backend/tts_backend_hf.py
class QwenTTSBackendHF:
    def __init__(self, model_id: str, device: str = "cuda"): ...
    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]:
        """Yields (pcm_bytes: bytes, sample_rate: int) per audio frame."""

# src/backend/tts_backend_mk.py
class QwenTTSBackendMK:
    """Same interface as QwenTTSBackendHF. Drop-in swap."""
    def __init__(self, model_id: str, megakernel_path: str): ...
    async def synthesize_streaming(
        self, text: str
    ) -> AsyncGenerator[tuple[bytes, int], None]: ...

# src/pipecat_services/qwen_tts_service.py
class QwenTTSService(TTSService):
    def __init__(self, backend: QwenTTSBackendHF | QwenTTSBackendMK, **kwargs): ...
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]: ...

# src/ui/gradio_app.py
def create_app(pipeline_runner) -> gr.Blocks:
    """Returns Gradio Blocks app. Connects mic input to pipeline, streams audio output."""
```

---

## Phase Dependency Graph

```
Phase A (Baseline TTS — WAV output)
    │   depends on: nothing, just a GPU + HF access
    │   delivers: actual class names, module paths, config values
    ▼
Phase B (Streaming audio generator)
    │   depends on: A's model inspection evidence
    │   delivers: async generator, measured TTFC baseline
    ▼
Phase C (Pipecat integration)
    │   depends on: B's streaming generator
    │   delivers: end-to-end pipeline, working demo
    │
    │   ← SHIPPABLE DEMO POINT (megakernel not required for passing)
    ▼
Phase D (Megakernel integration)
    │   depends on: A's module inspection + C's working pipeline
    │   delivers: accelerated backend, benchmarked tok/s, RTF, TTFC
    ▼
Final benchmark + README
    depends on: D (or C if D fails)
    delivers: numbers table, demo recording
```

---

## Key Decisions & Tradeoffs

| Decision              | Chosen                              | Alternative    | Why                                                                                                                                                                                                                                                                               |
| --------------------- | ----------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI framework          | React + `@pipecat-ai/client-react`  | Gradio         | Pipecat's own client library handles WebSocket transport, mic, and audio playback via `<PipecatClientAudio>` — no custom plumbing. Speaks RTVI protocol the server already understands. Gradio would require manual WebSocket audio wiring that the Pipecat client does for free. |
| STT provider          | Deepgram (cloud) or Whisper (local) | AssemblyAI     | Deepgram has lowest latency; Whisper works offline. Choice after testing.                                                                                                                                                                                                         |
| LLM provider          | OpenAI/Anthropic via Pipecat        | Local Ollama   | Simplest for demo. Ollama is fallback if no API key.                                                                                                                                                                                                                              |
| TTS backend interface | `AsyncGenerator[bytes, int]`        | `Queue[bytes]` | Generator is simpler, plays well with Pipecat's async pipeline.                                                                                                                                                                                                                   |
| Megakernel scope      | Talker decode loop only             | Full model     | Brief specifies this. Code predictor + vocoder stay as HF.                                                                                                                                                                                                                        |
| Audio format          | PCM int16, 24kHz, mono              | float32        | Pipecat's `TTSAudioRawFrame` expects int16 PCM by convention. 24kHz = Qwen3-TTS output.                                                                                                                                                                                           |
| Phase ordering        | A → B → C → D                       | A → D → C      | Working demo first, then optimize. Reduces risk of blocking on megakernel integration.                                                                                                                                                                                            |

---

## Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER DEVICE (Browser)                        │
│                                                                  │
│   React + @pipecat-ai/client-react                               │
│   Hosted on: Vercel (CDN-served static build)                    │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WSS (secure WebSocket)
                           │ wss://<vast-ai-host>/ws
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   VAST.AI RTX 5090 SERVER                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           Reverse Proxy  (Caddy or Nginx)                  │  │
│  │                                                            │  │
│  │   :443  → TLS termination → :8080 (FastAPI/uvicorn)        │  │
│  │   WSS upgrade handled here                                 │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │ ws://localhost:8080
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │       FastAPI + uvicorn  (server/pipeline/voice_agent.py)  │  │
│  │                                                            │  │
│  │   POST /connect  → creates Pipecat pipeline task           │  │
│  │   WS   /ws       → WebsocketServerTransport               │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │       Pipecat Pipeline (asyncio)                           │  │
│  │   STT → LLM → QwenTTSService → WebSocketTransport         │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │   GPU Inference  (CUDA, RTX 5090)                          │  │
│  │   Qwen3-TTS HF model + optional megakernel                 │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

| Layer          | Technology                         | Host             |
| -------------- | ---------------------------------- | ---------------- |
| Frontend       | React + `@pipecat-ai/client-react` | Vercel           |
| Transport      | Secure WebSocket (WSS)             | —                |
| Reverse proxy  | Caddy (preferred) or Nginx         | Vast.ai instance |
| API + pipeline | FastAPI + uvicorn + Pipecat        | Vast.ai instance |
| GPU inference  | Qwen3-TTS + optional megakernel    | Vast.ai RTX 5090 |

**Why Caddy:** auto-HTTPS via Let's Encrypt with zero config. One `Caddyfile` line handles TLS + WSS upgrade. Nginx works but requires manual cert management.

**CORS:** Vercel domain must be in FastAPI's `allow_origins`. Set `ALLOWED_ORIGIN=https://<your-vercel-app>.vercel.app` as an env var.

**Env vars needed on server:**

```bash
ALLOWED_ORIGIN=https://<vercel-app>.vercel.app
HF_TOKEN=<huggingface token for gated model>
OPENAI_API_KEY=<or whichever LLM provider>
DEEPGRAM_API_KEY=<if using Deepgram STT>
```

**Env vars needed on Vercel:**

```bash
VITE_WS_URL=wss://<vast-ai-ip-or-domain>/ws
```

---

## Open Questions (Require Investigation Before Coding)

These are unknowns that Phase A/B inspection must resolve:

1. **Qwen3-TTS class name** — `AutoModel`? Custom class? `trust_remote_code=True`?
2. **Talker module path** — `model.talker`? `model.decoder`? Something else?
3. **Talker layer count** — Is it actually 28 (same as 0.6B text model) or different for TTS?
4. **Codec frame size** — How many talker tokens = one vocoder call? What is the codec frame rate?
5. **EOS token ID** — What token signals end of speech generation?
6. **Streaming support** — Does `model.generate()` accept a `TextIteratorStreamer`?
7. **Prefill separation** — Can we call talker prefill without running the full decode loop?
8. **Pipecat TTSService API** — What exactly does `run_tts` return in the installed version?
9. **Kernel VOCAB_SIZE** — 151936 is text vocab; talker outputs codec tokens (likely much smaller vocab)

All 9 must be answered from actual code inspection before Phase D code is written.
