import { PipecatClientProvider, PipecatClientAudio } from "@pipecat-ai/client-react";
import { pipecatClient } from "./lib/pipecatClient";
import MicToggle from "./components/MicToggle";
import TranscriptLog from "./components/TranscriptLog";
import MetricsPanel from "./components/MetricsPanel";

export default function App() {
  return (
    <PipecatClientProvider client={pipecatClient}>
      <PipecatClientAudio />
      <div style={styles.root}>
        <h1 style={styles.title}>Qwen3-TTS Voice Agent</h1>
        <div style={styles.controls}>
          <MicToggle />
        </div>
        <TranscriptLog />
        <MetricsPanel />
      </div>
    </PipecatClientProvider>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    maxWidth: 720,
    margin: "0 auto",
    padding: "24px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: 600,
    color: "#ffffff",
  },
  controls: {
    display: "flex",
    gap: 12,
    alignItems: "center",
  },
};
