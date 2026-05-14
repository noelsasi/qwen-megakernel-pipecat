"""
FastAPI + Pipecat voice agent pipeline.

Exposes:
  GET  /          health check
  WS   /ws        WebSocket endpoint for Pipecat WebsocketServerTransport

Env vars:
  ALLOWED_ORIGIN      e.g. https://your-app.vercel.app  (required for CORS)
  OPENAI_API_KEY      LLM provider (required unless Ollama fallback)
  DEEPGRAM_API_KEY    STT provider (optional — falls back to Whisper)
  HF_TOKEN            HuggingFace token for gated model access (GPU only)
  TTS_BACKEND         "dev" (default) | "hf" | "megakernel"

Start with:
  uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import logging
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


def _load_tts_backend():
    global _tts_backend
    backend_type = os.environ.get("TTS_BACKEND", "dev")

    if backend_type == "megakernel":
        from server.backend.tts_backend_mk import QwenTTSBackendMK
        _tts_backend = QwenTTSBackendMK()
    elif backend_type == "hf":
        from server.backend.tts_backend_hf import QwenTTSBackendHF
        _tts_backend = QwenTTSBackendHF()
    else:
        from server.backend.tts_backend_dev import LocalDevTTSBackend
        _tts_backend = LocalDevTTSBackend()

    logger.info(f"TTS backend: {type(_tts_backend).__name__}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_tts_backend)
    yield


app = FastAPI(lifespan=lifespan)

allowed_origin = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {
        "status": "ok",
        "backend": type(_tts_backend).__name__ if _tts_backend else None,
        "tts_backend_env": os.environ.get("TTS_BACKEND", "dev"),
    }


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
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

        from server.pipecat_services.qwen_tts_service import QwenTTSService

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                serializer=ProtobufFrameSerializer(),
            ),
        )

        llm = _build_llm()
        stt = _build_stt()
        tts = QwenTTSService(
            backend=_tts_backend,
            sample_rate=_tts_backend.sample_rate,
        )

        context = LLMContext(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
        )
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


def _build_stt():
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    if deepgram_key:
        from pipecat.services.deepgram import DeepgramSTTService
        logger.info("STT: Deepgram")
        return DeepgramSTTService(api_key=deepgram_key)

    logger.info("STT: Whisper (no DEEPGRAM_API_KEY set)")
    try:
        from pipecat.services.whisper import WhisperSTTService
        return WhisperSTTService()
    except (ImportError, Exception):
        from pipecat.services.openai.stt import OpenAISTTService
        return OpenAISTTService(api_key=os.environ["OPENAI_API_KEY"])


def _build_llm():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from pipecat.services.openai.llm import OpenAILLMService
        logger.info("LLM: OpenAI gpt-4o-mini")
        return OpenAILLMService(api_key=openai_key, model="gpt-4o-mini")

    logger.info("LLM: Ollama llama3.2 (no OPENAI_API_KEY)")
    from pipecat.services.ollama import OllamaLLMService
    return OllamaLLMService(model="llama3.2")
