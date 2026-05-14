import { useState, useEffect, useRef, useCallback } from "react";
import {
  usePipecatClientTransportState,
  usePipecatClientMicControl,
  usePipecatConversation,
  useRTVIClientEvent,
} from "@pipecat-ai/client-react";
import { pipecatClient } from "../lib/pipecatClient";
import Waveform from "./Waveform";
import LatencyChart from "./LatencyChart";

// ── types ──────────────────────────────────────────────────────────────────
interface Metrics {
  ttfc_ms:   number | null;
  rtf:       number | null;
  toks_per_s: number | null;
  e2e_ms:    number | null;
}

interface LogEntry {
  ts:    string;
  level: "info" | "warn" | "error" | "debug";
  msg:   string;
}

// ── helpers ────────────────────────────────────────────────────────────────
function now() {
  return new Date().toISOString().slice(11, 23); // HH:MM:SS.mmm
}

function fmt(v: number | null, decimals = 0, unit = "") {
  if (v == null) return "—";
  return v.toFixed(decimals) + unit;
}

function stateLabel(s: string): { label: string; color: string; pulse: boolean } {
  switch (s) {
    case "authenticating":
    case "connecting":    return { label: "CONNECTING",      color: "var(--amber)",  pulse: true };
    case "connected":     return { label: "CONNECTED",       color: "var(--green)",  pulse: false };
    case "ready":         return { label: "LISTENING",       color: "var(--green)",  pulse: true };
    case "disconnecting": return { label: "DISCONNECTING",   color: "var(--amber)",  pulse: false };
    case "disconnected":  return { label: "OFFLINE",         color: "var(--text)",   pulse: false };
    case "error":         return { label: "ERROR",           color: "var(--red)",    pulse: false };
    default:              return { label: s.toUpperCase(),   color: "var(--text)",   pulse: false };
  }
}

// ── main component ─────────────────────────────────────────────────────────
export default function Dashboard() {
  const transportState = usePipecatClientTransportState();
  const { isMicEnabled, enableMic, disableMic } = usePipecatClientMicControl();
  const { messages } = usePipecatConversation();

  const [metrics, setMetrics] = useState<Metrics>({
    ttfc_ms: null, rtf: null, toks_per_s: null, e2e_ms: null,
  });
  const [logs, setLogs] = useState<LogEntry[]>([
    { ts: now(), level: "info",  msg: "Dashboard initialized" },
    { ts: now(), level: "debug", msg: "WebSocket transport ready" },
    { ts: now(), level: "info",  msg: "Waiting for connection…" },
  ]);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((level: LogEntry["level"], msg: string) => {
    setLogs(prev => [...prev.slice(-199), { ts: now(), level, msg }]);
  }, []);

  // ── pipecat event hooks ──────────────────────────────────────────────────
  useRTVIClientEvent("tts-metrics" as never, (data: Metrics) => {
    setMetrics(data);
    if (data.e2e_ms != null) {
      setLatencyHistory(h => [...h.slice(-59), data.e2e_ms!]);
    }
    addLog("debug", `metrics ttfc=${data.ttfc_ms?.toFixed(0)}ms rtf=${data.rtf?.toFixed(3)} toks/s=${data.toks_per_s?.toFixed(0)}`);
  });

  useRTVIClientEvent("bot-started-speaking" as never, () => {
    setIsStreaming(true);
    addLog("info", "Bot started speaking — audio streaming");
  });

  useRTVIClientEvent("bot-stopped-speaking" as never, () => {
    setIsStreaming(false);
    addLog("info", "Bot finished speaking");
  });

  useRTVIClientEvent("user-started-speaking" as never, () => {
    addLog("info", "User speech detected");
  });

  useRTVIClientEvent("user-stopped-speaking" as never, () => {
    addLog("debug", "User speech ended — sending to STT");
  });

  // ── track transport state changes ────────────────────────────────────────
  const prevState = useRef(transportState);
  useEffect(() => {
    if (prevState.current !== transportState) {
      const level = transportState === "error" ? "error" : "info";
      addLog(level, `Transport → ${transportState}`);
      prevState.current = transportState;
    }
  }, [transportState, addLog]);

  // ── auto-scroll logs ─────────────────────────────────────────────────────
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // ── connect / disconnect ─────────────────────────────────────────────────
  const connected = transportState === "ready" || transportState === "connected";
  const handleToggle = async () => {
    if (!connected) {
      addLog("info", "Initiating WebSocket connection…");
      await pipecatClient.connect();
    } else {
      addLog("info", "Disconnecting…");
      await pipecatClient.disconnect();
    }
  };

  const { label: stLabel, color: stColor, pulse: stPulse } = stateLabel(transportState);

  return (
    <div style={s.root}>
      {/* ── hairline grid overlay ── */}
      <div style={s.gridOverlay} aria-hidden />

      {/* ── top bar ── */}
      <header style={s.topbar}>
        <div style={s.topbarLeft}>
          <span style={s.logo}>
            <span style={{ color: "var(--cyan)" }}>Q</span>WEN3
            <span style={s.logoDivider}>/</span>
            <span style={{ color: "var(--text)" }}>PIPECAT</span>
          </span>
          <span style={s.badge}>TTS INFERENCE</span>
        </div>

        <div style={s.topbarCenter}>
          {/* WS streaming indicator */}
          <div style={s.wsIndicator}>
            <span style={{ ...s.dot, background: connected ? "var(--cyan)" : "var(--text)",
              boxShadow: connected ? "0 0 6px var(--cyan)" : "none",
              animation: connected ? "pulse-dot 1.2s ease-in-out infinite" : "none" }} />
            <span style={{ ...s.mono, fontSize: 11, color: connected ? "var(--cyan)" : "var(--text)" }}>
              WS {connected ? "STREAMING" : "IDLE"}
            </span>
          </div>

          {/* state pill */}
          <div style={{ ...s.statePill, borderColor: stColor + "44", background: stColor + "0f" }}>
            {stPulse && (
              <span style={{ ...s.dot, background: stColor, animation: "pulse-dot 1s ease-in-out infinite" }} />
            )}
            <span style={{ ...s.mono, fontSize: 11, color: stColor, fontWeight: 600 }}>
              {stLabel}
            </span>
          </div>
        </div>

        <div style={s.topbarRight}>
          <button
            onClick={handleToggle}
            style={{ ...s.connectBtn,
              background: connected ? "var(--red-dim)" : "var(--cyan-dim)",
              borderColor: connected ? "var(--red)" : "var(--cyan)",
              color: connected ? "var(--red)" : "var(--cyan)",
            }}
          >
            {connected ? "DISCONNECT" : "CONNECT"}
          </button>
          {connected && (
            <button
              onClick={() => isMicEnabled ? disableMic() : enableMic()}
              style={{ ...s.micBtn,
                background: isMicEnabled ? "var(--green-dim)" : "var(--bg-2)",
                borderColor: isMicEnabled ? "var(--green)" : "var(--border-2)",
                color: isMicEnabled ? "var(--green)" : "var(--text)",
              }}
            >
              {isMicEnabled ? "⏺ MIC ON" : "⏺ MIC OFF"}
            </button>
          )}
        </div>
      </header>

      {/* ── main grid ── */}
      <main style={s.main}>

        {/* ── col 1: metrics + waveform + chart ── */}
        <div style={s.col1}>

          {/* metrics cards */}
          <section style={s.metricsGrid}>
            <MetricCard
              label="TTFC"
              value={fmt(metrics.ttfc_ms, 0)}
              unit="ms"
              target="< 60"
              pass={metrics.ttfc_ms != null ? metrics.ttfc_ms < 60 : null}
              accent="var(--cyan)"
            />
            <MetricCard
              label="RTF"
              value={fmt(metrics.rtf, 3)}
              unit=""
              target="< 0.15"
              pass={metrics.rtf != null ? metrics.rtf < 0.15 : null}
              accent="var(--cyan)"
            />
            <MetricCard
              label="TOK/S"
              value={fmt(metrics.toks_per_s, 0)}
              unit=""
              target="~1000"
              pass={null}
              accent="var(--green)"
            />
            <MetricCard
              label="E2E LAT"
              value={fmt(metrics.e2e_ms, 0)}
              unit="ms"
              target="< 200"
              pass={metrics.e2e_ms != null ? metrics.e2e_ms < 200 : null}
              accent="var(--amber)"
            />
          </section>

          {/* waveform */}
          <section style={s.panel}>
            <PanelHeader label="AUDIO STREAM" extra={
              <span style={{ ...s.mono, fontSize: 10, color: isStreaming ? "var(--green)" : "var(--text)",
                display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ ...s.dot, background: isStreaming ? "var(--green)" : "var(--text)",
                  animation: isStreaming ? "pulse-dot 0.8s ease-in-out infinite" : "none" }} />
                {isStreaming ? "STREAMING" : "IDLE"}
              </span>
            } />
            <div style={s.waveformWrap}>
              <Waveform active={isStreaming} />
            </div>
          </section>

          {/* latency chart */}
          <section style={s.panel}>
            <PanelHeader label="E2E LATENCY HISTORY" extra={
              <span style={{ ...s.mono, fontSize: 10, color: "var(--text)" }}>
                last {latencyHistory.length} calls
              </span>
            } />
            <div style={{ padding: "8px 16px 16px" }}>
              <LatencyChart data={latencyHistory} />
            </div>
          </section>
        </div>

        {/* ── col 2: transcript + logs ── */}
        <div style={s.col2}>

          {/* GPU / backend status */}
          <section style={s.panel}>
            <PanelHeader label="BACKEND STATUS" />
            <div style={s.backendGrid}>
              <BackendChip icon="⬛" label="RTX 5090" status="online" />
              <BackendChip icon="▦" label="CUDA 12.8" status="active" />
              <BackendChip icon="◈" label="MEGAKERNEL" status="online" />
              <BackendChip icon="◎" label="VLLM ENGINE" status={connected ? "active" : "idle"} />
            </div>
          </section>

          {/* transcript */}
          <section style={{ ...s.panel, flex: 1, minHeight: 0 }}>
            <PanelHeader label="TRANSCRIPT" extra={
              <span style={{ ...s.mono, fontSize: 10, color: "var(--text)" }}>
                {messages?.length ?? 0} messages
              </span>
            } />
            <div style={s.transcriptScroll}>
              {(!messages || messages.length === 0) ? (
                <div style={s.emptyState}>
                  <span style={{ ...s.mono, fontSize: 12, color: "var(--text)" }}>
                    — connect and speak to begin —
                  </span>
                </div>
              ) : (
                messages.map((msg, i) => (
                  <TranscriptEntry key={i} role={msg.role} content={msg.content as string} />
                ))
              )}
            </div>
          </section>

          {/* debug log */}
          <section style={{ ...s.panel, flex: "0 0 220px" }}>
            <PanelHeader label="SESSION LOG" extra={
              <button
                onClick={() => setLogs([{ ts: now(), level: "info", msg: "Log cleared" }])}
                style={s.clearBtn}
              >
                CLEAR
              </button>
            } />
            <div style={s.logScroll}>
              {logs.map((l, i) => (
                <LogLine key={i} entry={l} />
              ))}
              <div ref={logsEndRef} />
            </div>
          </section>
        </div>

      </main>

      {/* ── footer ── */}
      <footer style={s.footer}>
        <span style={{ ...s.mono, fontSize: 10, color: "var(--text)" }}>
          qwen3-tts · pipecat-ai · ws://localhost:8000/ws
        </span>
        <span style={{ ...s.mono, fontSize: 10, color: "var(--text)" }}>
          {new Date().toISOString().slice(0, 19).replace("T", " ")} UTC
        </span>
      </footer>
    </div>
  );
}

// ── sub-components ─────────────────────────────────────────────────────────

function PanelHeader({ label, extra }: { label: string; extra?: React.ReactNode }) {
  return (
    <div style={ph.wrap}>
      <span style={ph.label}>{label}</span>
      {extra}
    </div>
  );
}
const ph = {
  wrap:  { display: "flex", justifyContent: "space-between", alignItems: "center",
           padding: "10px 14px", borderBottom: "1px solid var(--border)" } as React.CSSProperties,
  label: { fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600,
           letterSpacing: "0.12em", color: "var(--text)", textTransform: "uppercase" as const },
};

function MetricCard({ label, value, unit, target, pass, accent }:
  { label: string; value: string; unit: string; target: string; pass: boolean | null; accent: string }) {
  const passColor = pass == null ? "var(--text)" : pass ? "var(--green)" : "var(--red)";
  return (
    <div style={{ ...mc.card, borderColor: "var(--border)" }}>
      <span style={{ ...mc.label }}>{label}</span>
      <div style={mc.valueRow}>
        <span style={{ ...mc.value, color: accent }}>{value}</span>
        {unit && <span style={{ ...mc.unit, color: accent + "aa" }}>{unit}</span>}
      </div>
      <span style={{ ...mc.target, color: passColor }}>
        {pass != null ? (pass ? "✓" : "✗") : "○"} {target}
      </span>
    </div>
  );
}
const mc = {
  card:     { background: "var(--bg-1)", border: "1px solid", borderRadius: 6, padding: "14px 16px",
              display: "flex", flexDirection: "column" as const, gap: 4, minWidth: 0 },
  label:    { fontFamily: "var(--mono)", fontSize: 9, fontWeight: 600,
              letterSpacing: "0.14em", color: "var(--text)", textTransform: "uppercase" as const },
  valueRow: { display: "flex", alignItems: "baseline", gap: 4 },
  value:    { fontFamily: "var(--mono)", fontSize: 28, fontWeight: 600,
              fontVariantNumeric: "tabular-nums", lineHeight: 1 },
  unit:     { fontFamily: "var(--mono)", fontSize: 13, fontWeight: 400 },
  target:   { fontFamily: "var(--mono)", fontSize: 10, marginTop: 2 },
};

function BackendChip({ icon, label, status }:
  { icon: string; label: string; status: "online" | "active" | "idle" | "error" }) {
  const colors = { online: "var(--green)", active: "var(--cyan)", idle: "var(--text)", error: "var(--red)" };
  const c = colors[status];
  return (
    <div style={{ ...bc.chip, borderColor: c + "33", background: c + "08" }}>
      <span style={{ fontSize: 12 }}>{icon}</span>
      <div style={bc.info}>
        <span style={{ ...bc.name }}>{label}</span>
        <span style={{ ...bc.status, color: c }}>{status.toUpperCase()}</span>
      </div>
      <span style={{ ...bc.dot, background: c,
        boxShadow: status !== "idle" ? `0 0 5px ${c}` : "none",
        animation: status === "active" ? "pulse-dot 1.2s ease-in-out infinite" : "none" }} />
    </div>
  );
}
const bc = {
  chip:   { display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
            border: "1px solid", borderRadius: 6, background: "transparent" } as React.CSSProperties,
  info:   { display: "flex", flexDirection: "column" as const, gap: 1, flex: 1 },
  name:   { fontFamily: "var(--mono)", fontSize: 11, fontWeight: 500, color: "var(--text-2)" },
  status: { fontFamily: "var(--mono)", fontSize: 9, fontWeight: 600, letterSpacing: "0.1em" },
  dot:    { width: 6, height: 6, borderRadius: "50%", flexShrink: 0 } as React.CSSProperties,
};

function TranscriptEntry({ role, content }: { role: string; content: string }) {
  const isUser = role === "user";
  return (
    <div style={{ ...te.wrap, animation: "fade-in 0.2s ease" }}>
      <span style={{ ...te.role, color: isUser ? "var(--cyan)" : "var(--green)" }}>
        {isUser ? "USR" : "BOT"}
      </span>
      <span style={te.text}>{content}</span>
    </div>
  );
}
const te = {
  wrap: { display: "flex", gap: 12, padding: "8px 14px",
          borderBottom: "1px solid var(--border)", alignItems: "flex-start" } as React.CSSProperties,
  role: { fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600,
          letterSpacing: "0.1em", paddingTop: 1, flexShrink: 0, width: 28 } as React.CSSProperties,
  text: { fontSize: 13, color: "var(--text-2)", lineHeight: 1.6 } as React.CSSProperties,
};

function LogLine({ entry }: { entry: LogEntry }) {
  const colors = { info: "var(--text)", warn: "var(--amber)", error: "var(--red)", debug: "#4a5568" };
  return (
    <div style={{ ...ll.row, animation: "slide-in-right 0.15s ease" }}>
      <span style={ll.ts}>{entry.ts}</span>
      <span style={{ ...ll.level, color: colors[entry.level] }}>{entry.level.toUpperCase().slice(0,4)}</span>
      <span style={{ ...ll.msg, color: colors[entry.level] }}>{entry.msg}</span>
    </div>
  );
}
const ll = {
  row:   { display: "flex", gap: 8, padding: "3px 12px", alignItems: "baseline",
           fontFamily: "var(--mono)", fontSize: 10, lineHeight: 1.6 } as React.CSSProperties,
  ts:    { color: "#374151", flexShrink: 0, letterSpacing: "0.02em" } as React.CSSProperties,
  level: { flexShrink: 0, width: 30, fontWeight: 600, letterSpacing: "0.08em" } as React.CSSProperties,
  msg:   { wordBreak: "break-all" as const },
};

// ── layout styles ──────────────────────────────────────────────────────────
const s: Record<string, React.CSSProperties> = {
  root: {
    position: "relative",
    display: "flex",
    flexDirection: "column",
    height: "100dvh",
    overflow: "hidden",
    background: "var(--bg)",
  },
  gridOverlay: {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    zIndex: 0,
    backgroundImage: `
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px)
    `,
    backgroundSize: "40px 40px",
    maskImage: "radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%)",
    opacity: 0.4,
  },
  topbar: {
    position: "relative", zIndex: 10,
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0 20px", height: 52,
    borderBottom: "1px solid var(--border)",
    background: "rgba(8,10,15,0.95)",
    backdropFilter: "blur(8px)",
    flexShrink: 0,
  },
  topbarLeft:   { display: "flex", alignItems: "center", gap: 12 },
  topbarCenter: { display: "flex", alignItems: "center", gap: 12 },
  topbarRight:  { display: "flex", alignItems: "center", gap: 8 },
  logo: {
    fontFamily: "var(--mono)", fontSize: 14, fontWeight: 600,
    letterSpacing: "0.06em", color: "var(--text-hi)",
  },
  logoDivider: { color: "var(--border-2)", margin: "0 4px" },
  badge: {
    fontFamily: "var(--mono)", fontSize: 9, fontWeight: 600,
    letterSpacing: "0.14em", color: "var(--text)",
    border: "1px solid var(--border-2)", borderRadius: 3,
    padding: "2px 7px", textTransform: "uppercase",
  },
  wsIndicator: {
    display: "flex", alignItems: "center", gap: 6,
    padding: "4px 10px", border: "1px solid var(--border)",
    borderRadius: 4, background: "var(--bg-1)",
  },
  statePill: {
    display: "flex", alignItems: "center", gap: 6,
    padding: "4px 10px", border: "1px solid",
    borderRadius: 4,
  },
  dot: { width: 5, height: 5, borderRadius: "50%", flexShrink: 0, display: "inline-block" },
  connectBtn: {
    fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600,
    letterSpacing: "0.1em", padding: "6px 14px",
    border: "1px solid", borderRadius: 4, cursor: "pointer",
    transition: "all 0.15s",
  },
  micBtn: {
    fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600,
    letterSpacing: "0.1em", padding: "6px 12px",
    border: "1px solid", borderRadius: 4, cursor: "pointer",
    transition: "all 0.15s",
  },
  main: {
    position: "relative", zIndex: 1,
    display: "flex", gap: 1,
    flex: 1, minHeight: 0,
    overflow: "hidden",
  },
  col1: {
    display: "flex", flexDirection: "column", gap: 1,
    flex: "0 0 420px", minWidth: 0,
    borderRight: "1px solid var(--border)",
    overflow: "hidden",
  },
  col2: {
    display: "flex", flexDirection: "column", gap: 1,
    flex: 1, minWidth: 0,
    overflow: "hidden",
  },
  metricsGrid: {
    display: "grid", gridTemplateColumns: "1fr 1fr",
    gap: 1, padding: 1,
    background: "var(--border)",
    flexShrink: 0,
  },
  panel: {
    display: "flex", flexDirection: "column",
    background: "var(--bg-1)",
    border: "1px solid var(--border)",
    borderRadius: 0,
    minHeight: 0,
  },
  waveformWrap: { padding: "16px 16px 20px", flexShrink: 0 },
  backendGrid: {
    display: "grid", gridTemplateColumns: "1fr 1fr",
    gap: 8, padding: "12px 14px",
    flexShrink: 0,
  },
  transcriptScroll: {
    flex: 1, overflowY: "auto", minHeight: 0,
  },
  emptyState: {
    display: "flex", alignItems: "center", justifyContent: "center",
    padding: "32px 16px", color: "var(--text)",
  },
  logScroll: {
    flex: 1, overflowY: "auto", padding: "6px 0",
  },
  clearBtn: {
    fontFamily: "var(--mono)", fontSize: 9, fontWeight: 600,
    letterSpacing: "0.1em", padding: "3px 8px",
    border: "1px solid var(--border-2)", borderRadius: 3,
    background: "transparent", color: "var(--text)", cursor: "pointer",
  },
  footer: {
    position: "relative", zIndex: 10,
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "0 20px", height: 32,
    borderTop: "1px solid var(--border)",
    background: "rgba(8,10,15,0.95)",
    flexShrink: 0,
  },
};
