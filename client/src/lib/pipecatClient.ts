import { PipecatClient } from "@pipecat-ai/client-js";
import { WebSocketTransport } from "@pipecat-ai/websocket-transport";

export function createPipecatClient(wsUrl: string): PipecatClient {
  return new PipecatClient({
    transport: new WebSocketTransport({ wsUrl }),
    enableMic: true,
  });
}

export const DEFAULT_WS_URL =
  import.meta.env.VITE_WS_URL ?? "ws://localhost:8080/ws";
