import { useState, useEffect, useRef, useCallback } from "react";
import {
  usePipecatClientTransportState,
  usePipecatClientMicControl,
  usePipecatConversation,
  useRTVIClientEvent,
} from "@pipecat-ai/client-react";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { pipecatClient } from "../lib/pipecatClient";
import Waveform from "./Waveform";

interface Metrics {
  ttfc_ms: number | null;
  rtf: number | null;
  e2e_ms: number | null;
}
interface LogEntry {
  ts: string;
  level: "info" | "warn" | "error" | "debug";
  msg: string;
}

function now() { return new Date().toISOString().slice(11, 19); }
function fmtMs(v: number | null) { return v == null ? "—" : `${v.toFixed(0)}`; }
function fmtRTF(v: number | null) { return v == null ? "—" : v.toFixed(3); }

// ─── Orb ──────────────────────────────────────────────────────────────────
function Orb({ state }: { state: "idle" | "listening" | "speaking" | "connecting" | "off" }) {
  const configs = {
    off:         { bg: "#e4e4e7", shadow: "none",                              ring: false, anim: "none" },
    idle:        { bg: "#d4d4d8", shadow: "none",                              ring: false, anim: "none" },
    connecting:  { bg: "#fbbf24", shadow: "0 0 0 0 rgba(251,191,36,0)",        ring: false, anim: "spin 1s linear infinite" },
    listening:   { bg: "#16a34a", shadow: "0 8px 32px rgba(22,163,74,0.3)",    ring: true,  ringColor: "22,163,74", anim: "breathe 2s ease-in-out infinite" },
    speaking:    { bg: "#2563eb", shadow: "0 8px 40px rgba(37,99,235,0.35)",   ring: true,  ringColor: "37,99,235", anim: "breathe 1.6s ease-in-out infinite" },
  };
  const c = configs[state];

  return (
    <div style={{ position: "relative", width: 88, height: 88, flexShrink: 0 }}>
      {/* pulse rings */}
      {c.ring && (
        <>
          <div style={{
            position: "absolute", inset: -16, borderRadius: "50%",
            border: `1.5px solid rgba(${(c as { ringColor: string }).ringColor},0.25)`,
            animation: "pulse-ring 1.8s ease-out infinite",
          }} />
          <div style={{
            position: "absolute", inset: -16, borderRadius: "50%",
            border: `1.5px solid rgba(${(c as { ringColor: string }).ringColor},0.15)`,
            animation: "pulse-ring 1.8s ease-out infinite",
            animationDelay: "0.6s",
          }} />
        </>
      )}
      {/* orb */}
      <div style={{
        width: 88, height: 88, borderRadius: "50%",
        background: state === "connecting"
          ? `conic-gradient(${c.bg} 0deg 90deg, #e4e4e7 90deg 360deg)`
          : `radial-gradient(circle at 35% 32%, ${c.bg}, ${c.bg}cc 100%)`,
        boxShadow: c.shadow,
        animation: c.anim,
        transition: "background 0.5s ease, box-shadow 0.5s ease",
      }} />
    </div>
  );
}

// ─── Message ──────────────────────────────────────────────────────────────
function Message({ role, content }: { role: string; content: string }) {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      animation: "msg-in 0.25s ease both",
    }}>
      <div style={{
        maxWidth: "75%",
        padding: "10px 14px",
        borderRadius: isUser ? "16px 16px 4px 16px" : "4px 16px 16px 16px",
        fontSize: 14,
        lineHeight: 1.6,
        fontWeight: 400,
        letterSpacing: "-0.005em",
        ...(isUser ? {
          background: "#2563eb",
          color: "#ffffff",
        } : {
          background: "#f4f4f5",
          color: "#09090b",
          border: "1px solid #e4e4e7",
        }),
      }}>
        {content}
      </div>
    </div>
  );
}

// ─── Metric card ──────────────────────────────────────────────────────────
function MetricCard({ label, value, unit, pass }: {
  label: string; value: string; unit: string; pass?: boolean | null;
}) {
  const valueColor = pass == null ? "#09090b" : pass ? "#16a34a" : "#dc2626";
  const badgeColor = pass == null ? null : pass ? { bg: "#f0fdf4", border: "#bbf7d0", text: "#15803d" } : { bg: "#fef2f2", border: "#fecaca", text: "#dc2626" };
  return (
    <div style={{
      background: "#fafafa",
      border: "1px solid #e4e4e7",
      borderRadius: 12,
      padding: "14px 16px",
      display: "flex",
      flexDirection: "column",
      gap: 6,
    }}>
      <span style={{ fontSize: 11, fontWeight: 500, color: "#a1a1aa", letterSpacing: "0.04em", textTransform: "uppercase" as const }}>
        {label}
      </span>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{ fontSize: 26, fontWeight: 600, color: valueColor, fontVariantNumeric: "tabular-nums", letterSpacing: "-0.02em", fontFamily: "var(--mono)" }}>
          {value}
        </span>
        {value !== "—" && (
          <span style={{ fontSize: 12, color: "#a1a1aa", fontWeight: 400 }}>{unit}</span>
        )}
      </div>
      {badgeColor && (
        <span style={{
          display: "inline-flex", alignSelf: "flex-start",
          fontSize: 10, fontWeight: 500, padding: "2px 7px", borderRadius: 99,
          background: badgeColor.bg, border: `1px solid ${badgeColor.border}`, color: badgeColor.text,
        }}>
          {pass ? "on target" : "above target"}
        </span>
      )}
    </div>
  );
}

// ─── Log entry ────────────────────────────────────────────────────────────
function LogEntry({ entry }: { entry: LogEntry }) {
  const col = { info: "#52525b", warn: "#d97706", error: "#dc2626", debug: "#a1a1aa" }[entry.level];
  return (
    <div style={{ display: "flex", gap: 10, fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.6, animation: "fade-up 0.1s ease both" }}>
      <span style={{ color: "#d4d4d8", flexShrink: 0 }}>{entry.ts}</span>
      <span style={{ color: col, wordBreak: "break-all" as const }}>{entry.msg}</span>
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const transportState = usePipecatClientTransportState();
  const { isMicEnabled, enableMic } = usePipecatClientMicControl();
  const { messages } = usePipecatConversation();

  const [metrics, setMetrics] = useState<Metrics>({ ttfc_ms: null, rtf: null, e2e_ms: null });
  const [logs, setLogs] = useState<LogEntry[]>([{ ts: now(), level: "info", msg: "Waiting for connection" }]);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [panel, setPanel] = useState<"none" | "metrics" | "logs">("none");

  const transcriptEnd = useRef<HTMLDivElement>(null);
  const logsEnd = useRef<HTMLDivElement>(null);

  const addLog = useCallback((level: LogEntry["level"], msg: string) => {
    setLogs(p => [...p.slice(-199), { ts: now(), level, msg }]);
  }, []);

  useRTVIClientEvent(RTVIEvent.Metrics, (data: unknown) => {
    const raw = data as Record<string, { processor: string; value: number }[]>;
    const ttfb = raw?.ttfb?.find(d => d.processor === "QwenTTSService");
    const proc = raw?.processing?.find(d => d.processor === "QwenTTSService");
    if (!ttfb && !proc) return;
    setMetrics(prev => ({
      ...prev,
      ttfc_ms: ttfb ? ttfb.value * 1000 : prev.ttfc_ms,
      e2e_ms: proc ? proc.value * 1000 : prev.e2e_ms,
    }));
  });

  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => { setIsBotSpeaking(true);  addLog("info", "Bot speaking"); });
  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking,  () => { setIsBotSpeaking(false); addLog("info", "Bot done"); });
  useRTVIClientEvent(RTVIEvent.UserStartedSpeaking, () => { setIsUserSpeaking(true);  addLog("debug", "User speaking"); });
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking,  () => { setIsUserSpeaking(false); });

  const prevState = useRef(transportState);
  useEffect(() => {
    if (prevState.current !== transportState) {
      addLog("info", transportState);
      prevState.current = transportState;
    }
  }, [transportState, addLog]);

  useEffect(() => { transcriptEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  useEffect(() => { if (panel === "logs") logsEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [logs, panel]);

  const connected = transportState === "ready" || transportState === "connected";
  const isConnecting = transportState === "connecting" || transportState === "authenticating";

  const handleToggle = async () => {
    if (!connected) { addLog("info", "Connecting…"); await pipecatClient.connect(); }
    else { addLog("info", "Disconnecting…"); await pipecatClient.disconnect(); }
  };

  const orbState = isConnecting ? "connecting"
    : isBotSpeaking  ? "speaking"
    : isUserSpeaking ? "listening"
    : connected      ? "idle"
    : "off";

  const statusLabel = isConnecting ? "Connecting" : isBotSpeaking ? "Speaking" : isUserSpeaking ? "Listening" : connected ? "Ready" : "Disconnected";
  const statusDot   = isConnecting ? "#fbbf24" : isBotSpeaking ? "#2563eb" : isUserSpeaking ? "#16a34a" : connected ? "#22c55e" : "#d4d4d8";

  const togglePanel = (p: "metrics" | "logs") => setPanel(prev => prev === p ? "none" : p);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100dvh", background: "#ffffff", overflow: "hidden" }}>

      {/* ── TOPBAR ── */}
      <header style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 24px", height: 52, flexShrink: 0,
        borderBottom: "1px solid #e4e4e7",
        background: "rgba(255,255,255,0.9)", backdropFilter: "blur(12px)",
        zIndex: 20,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: "#09090b", letterSpacing: "-0.02em" }}>Qwen3 TTS</span>
          <span style={{
            fontSize: 10, fontWeight: 500, color: "#6366f1",
            background: "#eef2ff", border: "1px solid #c7d2fe",
            padding: "2px 7px", borderRadius: 99, letterSpacing: "0.02em",
          }}>megakernel</span>
        </div>

        {/* status */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%", background: statusDot,
            animation: (isBotSpeaking || isUserSpeaking || isConnecting) ? "blink 1.4s ease infinite" : "none",
            transition: "background 0.3s",
          }} />
          <span style={{ fontSize: 13, color: "#52525b", fontWeight: 450 }}>{statusLabel}</span>
        </div>

        {/* actions */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {connected && (
            <button onClick={() => enableMic(!isMicEnabled)} style={{
              height: 32, padding: "0 12px", borderRadius: 8, fontSize: 13, fontWeight: 500,
              cursor: "pointer", transition: "all 0.15s",
              background: isMicEnabled ? "#f0fdf4" : "#fafafa",
              border: `1px solid ${isMicEnabled ? "#86efac" : "#e4e4e7"}`,
              color: isMicEnabled ? "#15803d" : "#52525b",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <span style={{ fontSize: 16, lineHeight: 1 }}>{isMicEnabled ? "🎙️" : "🔇"}</span>
              {isMicEnabled ? "Mic on" : "Mic off"}
            </button>
          )}
          <button onClick={handleToggle} style={{
            height: 32, padding: "0 16px", borderRadius: 8, fontSize: 13, fontWeight: 500,
            cursor: "pointer", transition: "all 0.15s",
            background: connected ? "#fef2f2" : "#2563eb",
            border: `1px solid ${connected ? "#fecaca" : "#2563eb"}`,
            color: connected ? "#dc2626" : "#ffffff",
          }}>
            {connected ? "Disconnect" : "Connect"}
          </button>
        </div>
      </header>

      {/* ── BODY ── */}
      <div style={{ flex: 1, display: "flex", minHeight: 0, overflow: "hidden" }}>

        {/* conversation column */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>

          {/* orb zone */}
          <div style={{
            flexShrink: 0, display: "flex", flexDirection: "column",
            alignItems: "center", gap: 14, paddingTop: 40, paddingBottom: 12,
          }}>
            <Orb state={orbState} />
            {/* waveform */}
            <div style={{
              width: 180, height: 28,
              opacity: isBotSpeaking ? 1 : 0,
              transition: "opacity 0.4s ease",
            }}>
              <Waveform active={isBotSpeaking} color="37,99,235" />
            </div>
            {/* state label under orb */}
            <span style={{
              fontSize: 13, color: "#a1a1aa", fontWeight: 400,
              animation: "fade-up 0.3s ease both",
              minHeight: 20,
            }}>
              {isBotSpeaking ? "Speaking…" : isUserSpeaking ? "Listening…" : isConnecting ? "Connecting…" : connected ? "" : ""}
            </span>
          </div>

          {/* transcript */}
          <div style={{
            flex: 1, overflowY: "auto", padding: "8px 0 24px",
            display: "flex", flexDirection: "column", gap: 8,
            width: "100%", maxWidth: 640, margin: "0 auto", padding: "8px 24px 24px",
          }}>
            {(!messages || messages.length === 0) ? (
              <div style={{
                flex: 1, display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                gap: 8, paddingTop: 20,
                animation: "fade-up 0.5s ease both",
              }}>
                <span style={{ fontSize: 14, color: "#a1a1aa" }}>
                  {connected ? "Start speaking to begin" : "Connect to start a conversation"}
                </span>
              </div>
            ) : (
              messages.map((msg, i) => (
                <Message
                  key={i}
                  role={msg.role}
                  content={msg.parts.map(p =>
                    typeof p.text === "string" ? p.text
                      : (p.text as { spoken?: string })?.spoken ?? ""
                  ).join("")}
                />
              ))
            )}
            <div ref={transcriptEnd} />
          </div>
        </div>

        {/* side panel — metrics or logs */}
        {panel !== "none" && (
          <div style={{
            width: 300, flexShrink: 0,
            borderLeft: "1px solid #e4e4e7",
            background: "#fafafa",
            display: "flex", flexDirection: "column",
            overflow: "hidden",
            animation: "panel-in 0.2s ease both",
          }}>
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "14px 16px 12px", borderBottom: "1px solid #e4e4e7",
              flexShrink: 0,
            }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#09090b", letterSpacing: "0.01em" }}>
                {panel === "metrics" ? "Performance" : "Logs"}
              </span>
              <button onClick={() => setPanel("none")} style={{
                background: "none", border: "none", cursor: "pointer",
                color: "#a1a1aa", fontSize: 18, lineHeight: 1, padding: "0 2px",
              }}>×</button>
            </div>

            {panel === "metrics" && (
              <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10, overflowY: "auto" }}>
                <MetricCard
                  label="Time to First Chunk"
                  value={fmtMs(metrics.ttfc_ms)}
                  unit="ms"
                  pass={metrics.ttfc_ms != null ? metrics.ttfc_ms < 60 : null}
                />
                <MetricCard
                  label="Real-time Factor"
                  value={fmtRTF(metrics.rtf)}
                  unit=""
                  pass={metrics.rtf != null ? metrics.rtf < 0.15 : null}
                />
                <MetricCard
                  label="End-to-End Latency"
                  value={fmtMs(metrics.e2e_ms)}
                  unit="ms"
                  pass={metrics.e2e_ms != null ? metrics.e2e_ms < 500 : null}
                />
                {/* targets legend */}
                <div style={{
                  marginTop: 4, padding: "10px 12px", borderRadius: 8,
                  background: "#f0f9ff", border: "1px solid #bae6fd",
                  fontSize: 11, color: "#0369a1", lineHeight: 1.7,
                  fontFamily: "var(--mono)",
                }}>
                  targets: TTFC &lt;60ms · RTF &lt;0.15 · E2E &lt;500ms
                </div>
              </div>
            )}

            {panel === "logs" && (
              <div style={{ flex: 1, overflowY: "auto", padding: "10px 14px", display: "flex", flexDirection: "column", gap: 2 }}>
                {logs.map((l, i) => <LogEntry key={i} entry={l} />)}
                <div ref={logsEnd} />
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── BOTTOM BAR ── */}
      <div style={{
        flexShrink: 0, height: 44,
        borderTop: "1px solid #e4e4e7",
        background: "rgba(255,255,255,0.95)",
        backdropFilter: "blur(8px)",
        display: "flex", alignItems: "center",
        padding: "0 24px", gap: 0,
        zIndex: 20,
      }}>
        {/* inline metric chips — always visible */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 16 }}>
          {[
            { label: "TTFC", value: metrics.ttfc_ms != null ? `${fmtMs(metrics.ttfc_ms)} ms` : "—", pass: metrics.ttfc_ms != null ? metrics.ttfc_ms < 60 : null },
            { label: "RTF",  value: metrics.rtf != null ? fmtRTF(metrics.rtf) : "—",              pass: metrics.rtf != null ? metrics.rtf < 0.15 : null },
            { label: "E2E",  value: metrics.e2e_ms != null ? `${fmtMs(metrics.e2e_ms)} ms` : "—", pass: metrics.e2e_ms != null ? metrics.e2e_ms < 500 : null },
          ].map(({ label, value, pass }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ fontSize: 10, fontWeight: 600, color: "#a1a1aa", letterSpacing: "0.06em", fontFamily: "var(--mono)" }}>
                {label}
              </span>
              <span style={{
                fontSize: 12, fontWeight: 500, fontFamily: "var(--mono)",
                color: pass == null ? "#52525b" : pass ? "#16a34a" : "#dc2626",
                fontVariantNumeric: "tabular-nums",
              }}>
                {value}
              </span>
            </div>
          ))}
        </div>

        {/* panel toggle buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <button onClick={() => togglePanel("metrics")} style={{
            height: 28, padding: "0 10px", borderRadius: 6,
            fontSize: 12, fontWeight: 500, cursor: "pointer", transition: "all 0.15s",
            background: panel === "metrics" ? "#eff6ff" : "transparent",
            border: `1px solid ${panel === "metrics" ? "#bfdbfe" : "transparent"}`,
            color: panel === "metrics" ? "#2563eb" : "#71717a",
          }}>
            Metrics
          </button>
          <button onClick={() => togglePanel("logs")} style={{
            height: 28, padding: "0 10px", borderRadius: 6,
            fontSize: 12, fontWeight: 500, cursor: "pointer", transition: "all 0.15s",
            background: panel === "logs" ? "#fafafa" : "transparent",
            border: `1px solid ${panel === "logs" ? "#e4e4e7" : "transparent"}`,
            color: panel === "logs" ? "#09090b" : "#71717a",
          }}>
            Logs
          </button>
        </div>
      </div>
    </div>
  );
}
