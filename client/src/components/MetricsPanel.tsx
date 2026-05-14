import { useState } from "react";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";

interface Metrics {
  ttfc_ms: number | null;
  rtf: number | null;
  toks_per_s: number | null;
}

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<Metrics>({ ttfc_ms: null, rtf: null, toks_per_s: null });

  // Listen for custom metric events emitted by the server
  // The server sends these via RTVIMessage with type "tts-metrics"
  useRTVIClientEvent("tts-metrics", (data: Metrics) => {
    setMetrics(data);
  });

  const fmt = (v: number | null, unit: string) =>
    v == null ? "—" : `${v.toFixed(unit === "ms" ? 0 : unit === "rtf" ? 3 : 1)}${unit === "ms" ? " ms" : unit === "rtf" ? "" : " tok/s"}`;

  return (
    <div style={styles.panel}>
      <Metric label="TTFC" value={fmt(metrics.ttfc_ms, "ms")} target="< 60 ms" pass={metrics.ttfc_ms != null && metrics.ttfc_ms < 60} />
      <Metric label="RTF" value={fmt(metrics.rtf, "rtf")} target="< 0.15" pass={metrics.rtf != null && metrics.rtf < 0.15} />
      <Metric label="tok/s" value={fmt(metrics.toks_per_s, "tok/s")} target="~1000" pass={null} />
    </div>
  );
}

function Metric({ label, value, target, pass }: { label: string; value: string; target: string; pass: boolean | null }) {
  return (
    <div style={styles.metric}>
      <div style={styles.metricLabel}>{label}</div>
      <div style={styles.metricValue}>{value}</div>
      <div style={styles.metricTarget(pass)}>
        target {target}{pass != null ? (pass ? " ✓" : " ✗") : ""}
      </div>
    </div>
  );
}

const styles = {
  panel: {
    display: "flex",
    gap: 12,
    background: "#1a1a1a",
    borderRadius: 8,
    padding: "12px 16px",
  } as React.CSSProperties,
  metric: {
    flex: 1,
    display: "flex",
    flexDirection: "column" as const,
    gap: 2,
  },
  metricLabel: { fontSize: 11, color: "#666", textTransform: "uppercase" as const, letterSpacing: 1 },
  metricValue: { fontSize: 22, fontWeight: 700, color: "#e0e0e0", fontVariantNumeric: "tabular-nums" },
  metricTarget: (pass: boolean | null): React.CSSProperties => ({
    fontSize: 11,
    color: pass == null ? "#555" : pass ? "#27ae60" : "#c0392b",
  }),
};
