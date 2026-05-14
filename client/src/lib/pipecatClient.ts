import { PipecatClient } from "@pipecat-ai/client-js";
import { WebSocketTransport } from "@pipecat-ai/websocket-transport";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws";

export const pipecatClient = new PipecatClient({
  transport: new WebSocketTransport(),
  enableMic: true,
  params: {
    baseUrl: WS_URL,
  },
});
