import {
  PipecatClientProvider,
  PipecatClientAudio,
} from "@pipecat-ai/client-react";
import { pipecatClient } from "./lib/pipecatClient";
import Dashboard from "./components/Dashboard";

export default function App() {
  return (
    <PipecatClientProvider client={pipecatClient}>
      <PipecatClientAudio />
      <Dashboard />
    </PipecatClientProvider>
  );
}
