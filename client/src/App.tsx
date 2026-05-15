import { useState, useRef } from "react";
import {
  PipecatClientProvider,
  PipecatClientAudio,
} from "@pipecat-ai/client-react";
import { PipecatClient } from "@pipecat-ai/client-js";
import { createPipecatClient, DEFAULT_WS_URL } from "./lib/pipecatClient";
import Dashboard from "./components/Dashboard";

export default function App() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  // Stable client ref — recreated only when wsUrl changes via onConnect
  const clientRef = useRef<PipecatClient>(createPipecatClient(wsUrl));

  // Called by Dashboard before connecting — rebuilds client if URL changed
  function handleConnect(url: string) {
    if (url !== wsUrl) {
      setWsUrl(url);
      clientRef.current = createPipecatClient(url);
    }
    clientRef.current.connect();
  }

  return (
    <PipecatClientProvider client={clientRef.current}>
      <PipecatClientAudio />
      <Dashboard wsUrl={wsUrl} onConnect={handleConnect} />
    </PipecatClientProvider>
  );
}
