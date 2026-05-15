"""
Qwen3-TTS + megakernel voice agent.

Flow: mic → Deepgram STT → OpenAI LLM → QwenTTS → speaker

Env vars (required):
  OPENAI_API_KEY      gpt-4o-mini for LLM
  DEEPGRAM_API_KEY    Deepgram for STT (fast, low-latency)
  ALLOWED_ORIGIN      CORS origin, e.g. https://your-app.vercel.app (default: *)

TTS_BACKEND options (set as env var):
  megakernel  (default) — Phase 1 monkey-patch backend (HF fallback in practice)
  v2                    — Phase 2 custom decode loop (correct architecture, real streaming)
  hf                    — Pure HF baseline, no megakernel

Start:
  TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_tts_backend = None

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep answers concise — one to three sentences. "
    "Avoid markdown, bullet points, or characters that don't read well aloud."
)


def _load_backend():
    global _tts_backend
    backend_name = os.environ.get("TTS_BACKEND", "v2").lower()

    if backend_name == "v2":
        from server.backend.tts_backend_v2 import QwenTTSBackendV2
        _tts_backend = QwenTTSBackendV2()
        logger.info("v2 custom-decode backend ready")
    elif backend_name == "hf":
        from server.backend.tts_backend_hf import QwenTTSBackendHF
        _tts_backend = QwenTTSBackendHF()
        logger.info("HF baseline backend ready")
    else:
        from server.backend.tts_backend_mk import QwenTTSBackendMK
        _tts_backend = QwenTTSBackendMK()
        logger.info("Megakernel backend ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_backend)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok", "backend": "megakernel"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"Client connected: {websocket.client}")

    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketTransport,
            FastAPIWebsocketParams,
        )
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.serializers.protobuf import ProtobufFrameSerializer
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

        from server.pipecat_services.qwen_tts_service import QwenTTSService

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=ProtobufFrameSerializer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
        llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-5-mini")
        tts = QwenTTSService(backend=_tts_backend, sample_rate=_tts_backend.sample_rate)

        context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
        context_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        runner = PipelineRunner()
        task = PipelineTask(pipeline)
        await runner.run(task)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        logger.info(f"Client disconnected: {websocket.client}")
