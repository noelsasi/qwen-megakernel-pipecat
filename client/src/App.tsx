import { useState, useRef } from "react";
import {
  PipecatClientProvider,
  PipecatClientAudio,
} from "@pipecat-ai/client-react";
import { PipecatClient } from "@pipecat-ai/client-js";
import { createPipecatClient, DEFAULT_WS_URL } from "./lib/pipecatClient";
import Dashboard from "./components/Dashboard";
import DocsPage from "./components/DocsPage";

export default function App() {
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [page, setPage] = useState<"home" | "docs">("home");
  const [docsInitialId, setDocsInitialId] = useState<string | undefined>();
  const clientRef = useRef<PipecatClient>(createPipecatClient(wsUrl));

  function handleConnect(url: string) {
    if (url !== wsUrl) {
      setWsUrl(url);
      clientRef.current = createPipecatClient(url);
    }
    clientRef.current.connect();
  }

  function openDocs(initialId?: string) {
    setDocsInitialId(initialId);
    setPage("docs");
  }

  if (page === "docs") {
    return <DocsPage onBack={() => setPage("home")} initialId={docsInitialId} />;
  }

  return (
    <PipecatClientProvider client={clientRef.current}>
      <PipecatClientAudio />
      <Dashboard wsUrl={wsUrl} onConnect={handleConnect} onOpenDocs={openDocs} />
    </PipecatClientProvider>
  );
}
