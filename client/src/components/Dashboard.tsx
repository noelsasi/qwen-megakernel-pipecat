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

// ── types ──────────────────────────────────────────────────────────────────
interface Metrics {
  ttfc_ms:    number | null;
  rtf:        number | null;
  toks_per_s: number | null;
  e2e_ms:     number | null;
}

interface LogEntry {
  ts:    string;
  level: "info" | "warn" | "error" | "debug";
  msg:   string;
}

function now() { return new Date().toISOString().slice(11, 23); }

function fmtMs(v: number | null) {
  if (v == null) return "—";
  return v < 1000 ? `${v.toFixed(0)}ms` : `${(v / 1000).toFixed(2)}s`;
}

function fmtNum(v: number | null, d = 2) {
  if (v == null) return "—";
  return v.toFixed(d);
}

// ── voice orb ─────────────────────────────────────────────────────────────
function VoiceOrb({ state }: { state: "idle" | "listening" | "speaking" | "connecting" }) {
  const size = 72;

  const coreColor = state === "speaking"
    ? "rgba(37,99,235,0.9)"
    : state === "listening"
    ? "rgba(22,163,74,0.85)"
    : state === "connecting"
    ? "rgba(217,119,6,0.75)"
    : "rgba(180,180,195,0.6)";

  const glowColor = state === "speaking"
    ? "rgba(37,99,235,0.18)"
    : state === "listening"
    ? "rgba(22,163,74,0.15)"
    : state === "connecting"
    ? "rgba(217,119,6,0.12)"
    : "transparent";

  const ringColor = state === "speaking"
    ? "rgba(37,99,235,0.12)"
    : state === "listening"
    ? "rgba(22,163,74,0.1)"
    : "transparent";

  const pulseAnim = state === "speaking"
    ? "orb-pulse-ring 1.4s ease-out infinite"
    : state === "listening"
    ? "orb-listen-ring 2s ease-out infinite"
    : "none";

  const breatheAnim = (state === "speaking" || state === "listening")
    ? "orb-breathe 2s ease-in-out infinite"
    : "none";

  const connectAnim = state === "connecting"
    ? "connecting-spin 1.2s linear infinite"
    : "none";

  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      {(state === "speaking" || state === "listening") && (
        <>
          <div style={{
            position: "absolute",
            inset: -size * 0.4,
            borderRadius: "50%",
            border: `1px solid ${ringColor}`,
            animation: pulseAnim,
          }} />
          <div style={{
            position: "absolute",
            inset: -size * 0.4,
            borderRadius: "50%",
            border: `1px solid ${ringColor}`,
            animation: pulseAnim,
            animationDelay: "0.5s",
          }} />
        </>
      )}
      <div style={{
        position: "absolute",
        inset: -20,
        borderRadius: "50%",
        background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`,
        pointerEvents: "none",
      }} />
      <div style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: `radial-gradient(circle at 38% 35%, ${coreColor}, ${
          coreColor.replace("0.9", "0.55").replace("0.85", "0.5").replace("0.75", "0.4").replace("0.6", "0.25")
        } 100%)`,
        boxShadow: state !== "idle"
          ? `0 4px 24px ${glowColor}, 0 1px 4px rgba(0,0,0,0.1)`
          : "0 1px 4px rgba(0,0,0,0.08)",
        animation: state === "connecting" ? connectAnim : breatheAnim,
        transition: "background 0.6s ease, box-shadow 0.6s ease",
        cursor: "pointer",
        position: "relative",
        zIndex: 1,
        border: state !== "idle"
          ? `1px solid ${ringColor.replace("0.12", "0.25").replace("0.1", "0.22")}`
          : "1px solid rgba(0,0,0,0.08)",
      }} />
    </div>
  );
}

// ── message bubble ─────────────────────────────────────────────────────────
function MessageBubble({ role, content }: { role: string; content: string }) {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      animation: "msg-enter 0.3s cubic-bezier(0.16,1,0.3,1) both",
    }}>
      <div style={{
        maxWidth: "72%",
        padding: "11px 15px",
        borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
        background: isUser ? "rgba(37,99,235,0.09)" : "#ffffff",
        border: isUser
          ? "1px solid rgba(37,99,235,0.16)"
          : "1px solid rgba(0,0,0,0.07)",
        color: isUser ? "#1e3a8a" : "#111114",
        fontSize: 15,
        lineHeight: 1.65,
        fontWeight: 400,
        letterSpacing: "-0.01em",
        boxShadow: isUser
          ? "none"
          : "0 1px 3px rgba(0,0,0,0.05)",
      }}>
        {content}
      </div>
    </div>
  );
}

// ── metric pill ────────────────────────────────────────────────────────────
function MetricPill({ label, value, pass }: {
  label: string; value: string; pass?: boolean | null;
}) {
  const valueColor = pass == null
    ? "#5a5a6a"
    : pass ? "#16a34a" : "#dc2626";
  return (
    <span style={{
      fontFamily: "var(--mono)",
      fontSize: 11,
      display: "inline-flex",
      alignItems: "center",
      gap: 4,
    }}>
      <span style={{ color: "#9a9aaa" }}>{label}</span>
      <span style={{ color: valueColor, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </span>
  );
}

// ── log line ───────────────────────────────────────────────────────────────
function LogLine({ entry }: { entry: LogEntry }) {
  const colors: Record<string, string> = {
    info:  "#5a5a6a",
    warn:  "#d97706",
    error: "#dc2626",
    debug: "#9a9aaa",
  };
  return (
    <div style={{
      display: "flex",
      gap: 10,
      fontFamily: "var(--mono)",
      fontSize: 10.5,
      lineHeight: 1.7,
      color: colors[entry.level],
      animation: "log-slide-in 0.15s ease both",
    }}>
      <span style={{ color: "#c0c0cc", flexShrink: 0 }}>{entry.ts}</span>
      <span style={{ wordBreak: "break-all" }}>{entry.msg}</span>
    </div>
  );
}

// ── connect button ─────────────────────────────────────────────────────────
function ConnectButton({ connected, onClick }: { connected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: connected ? "rgba(220,38,38,0.06)" : "rgba(37,99,235,0.08)",
        border: `1px solid ${connected ? "rgba(220,38,38,0.2)" : "rgba(37,99,235,0.2)"}`,
        color: connected ? "#dc2626" : "#2563eb",
        padding: "7px 16px",
        borderRadius: 8,
        fontSize: 13,
        fontFamily: "var(--sans)",
        fontWeight: 500,
        cursor: "pointer",
        transition: "all 0.2s ease",
      }}
    >
      {connected ? "Disconnect" : "Connect"}
    </button>
  );
}

// ── main ───────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const transportState = usePipecatClientTransportState();
  const { isMicEnabled, enableMic } = usePipecatClientMicControl();
  const { messages } = usePipecatConversation();

  const [metrics, setMetrics] = useState<Metrics>({
    ttfc_ms: null, rtf: null, toks_per_s: null, e2e_ms: null,
  });
  const [logs, setLogs] = useState<LogEntry[]>([
    { ts: now(), level: "info", msg: "Ready" },
  ]);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);

  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((level: LogEntry["level"], msg: string) => {
    setLogs(prev => [...prev.slice(-99), { ts: now(), level, msg }]);
  }, []);

  useRTVIClientEvent(RTVIEvent.Metrics, (data: unknown) => {
    const raw = data as Record<string, { processor: string; value: number }[]>;
    const ttfb = raw?.ttfb?.find(d => d.processor === "QwenTTSService");
    const proc = raw?.processing?.find(d => d.processor === "QwenTTSService");
    if (!ttfb && !proc) return;
    const ttfc_ms = ttfb ? ttfb.value * 1000 : null;
    const e2e_ms  = proc ? proc.value * 1000 : null;
    setMetrics(prev => ({ ...prev, ttfc_ms, e2e_ms }));
  });

  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => {
    setIsBotSpeaking(true);
    addLog("info", "Speaking");
  });
  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking, () => {
    setIsBotSpeaking(false);
    addLog("info", "Done speaking");
  });
  useRTVIClientEvent(RTVIEvent.UserStartedSpeaking, () => {
    setIsUserSpeaking(true);
    addLog("debug", "User speaking");
  });
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, () => {
    setIsUserSpeaking(false);
    addLog("debug", "User stopped");
  });

  const prevState = useRef(transportState);
  useEffect(() => {
    if (prevState.current !== transportState) {
      addLog("info", transportState);
      prevState.current = transportState;
    }
  }, [transportState, addLog]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (logsOpen) logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, logsOpen]);

  const connected = transportState === "ready" || transportState === "connected";
  const isConnecting = transportState === "connecting" || transportState === "authenticating";

  const handleToggle = async () => {
    if (!connected) { addLog("info", "Connecting…"); await pipecatClient.connect(); }
    else            { addLog("info", "Disconnecting…"); await pipecatClient.disconnect(); }
  };

  const orbState = isConnecting ? "connecting"
    : isBotSpeaking   ? "speaking"
    : isUserSpeaking  ? "listening"
    : "idle";

  const statusText = isConnecting ? "Connecting…"
    : isBotSpeaking  ? "Speaking"
    : isUserSpeaking ? "Listening"
    : connected      ? "Ready"
    : "Disconnected";

  const statusColor = isConnecting  ? "#d97706"
    : isBotSpeaking  ? "#2563eb"
    : isUserSpeaking ? "#16a34a"
    : connected      ? "#9a9aaa"
    : "#c0c0cc";

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100dvh",
      background: "#f7f7f8",
      overflow: "hidden",
      position: "relative",
    }}>
      {/* ── ambient glow ── */}
      <div style={{
        position: "absolute",
        top: 0,
        left: "50%",
        transform: "translateX(-50%)",
        width: 560,
        height: 260,
        background: isBotSpeaking
          ? "radial-gradient(ellipse, rgba(37,99,235,0.07) 0%, transparent 70%)"
          : isUserSpeaking
          ? "radial-gradient(ellipse, rgba(22,163,74,0.06) 0%, transparent 70%)"
          : "none",
        pointerEvents: "none",
        transition: "background 0.8s ease",
        zIndex: 0,
      }} />

      {/* ── top bar ── */}
      <header style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 28px",
        height: 56,
        flexShrink: 0,
        borderBottom: "1px solid rgba(0,0,0,0.07)",
        background: "rgba(247,247,248,0.85)",
        backdropFilter: "blur(16px)",
        position: "relative",
        zIndex: 10,
      }}>
        {/* brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: "#111114", letterSpacing: "-0.01em" }}>
            Qwen3 TTS
          </span>
          <span style={{ width: 1, height: 13, background: "rgba(0,0,0,0.1)" }} />
          <span style={{ fontSize: 11, color: "#9a9aaa", fontFamily: "var(--mono)" }}>
            megakernel
          </span>
        </div>

        {/* status pill */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          padding: "5px 12px",
          borderRadius: 20,
          background: "#ffffff",
          border: "1px solid rgba(0,0,0,0.08)",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}>
          <span style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: statusColor,
            boxShadow: (connected || isConnecting) ? `0 0 5px ${statusColor}` : "none",
            animation: (isBotSpeaking || isUserSpeaking || isConnecting)
              ? "status-fade 1.5s ease-in-out infinite" : "none",
            transition: "background 0.4s ease",
            flexShrink: 0,
          }} />
          <span style={{
            fontSize: 12,
            color: statusColor,
            fontWeight: 500,
            transition: "color 0.4s ease",
          }}>
            {statusText}
          </span>
        </div>

        {/* actions */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {connected && (
            <button
              onClick={() => enableMic(!isMicEnabled)}
              style={{
                background: isMicEnabled ? "rgba(22,163,74,0.08)" : "#ffffff",
                border: `1px solid ${isMicEnabled ? "rgba(22,163,74,0.25)" : "rgba(0,0,0,0.1)"}`,
                color: isMicEnabled ? "#16a34a" : "#5a5a6a",
                padding: "6px 13px",
                borderRadius: 8,
                fontSize: 12,
                fontFamily: "var(--sans)",
                fontWeight: 500,
                cursor: "pointer",
                transition: "all 0.2s ease",
                display: "flex",
                alignItems: "center",
                gap: 5,
                boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
              }}
            >
              <span style={{ fontSize: 9 }}>{isMicEnabled ? "●" : "○"}</span>
              Mic
            </button>
          )}
          <ConnectButton connected={connected} onClick={handleToggle} />
        </div>
      </header>

      {/* ── main ── */}
      <main style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        minHeight: 0,
        overflow: "hidden",
        position: "relative",
        zIndex: 1,
      }}>
        {/* orb + waveform */}
        <div style={{
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 18,
          paddingTop: 36,
          paddingBottom: 16,
        }}>
          <VoiceOrb state={orbState} />
          <div style={{
            width: 220,
            height: 32,
            opacity: isBotSpeaking ? 1 : 0,
            transition: "opacity 0.5s ease",
          }}>
            <Waveform active={isBotSpeaking} color="37,99,235" />
          </div>
        </div>

        {/* transcript */}
        <div style={{
          flex: 1,
          width: "100%",
          maxWidth: 660,
          overflowY: "auto",
          padding: "4px 24px 24px",
          display: "flex",
          flexDirection: "column",
          gap: 10,
          maskImage: "linear-gradient(to bottom, transparent 0%, black 5%, black 100%)",
        }}>
          {(!messages || messages.length === 0) ? (
            <div style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#c0c0cc",
              fontSize: 14,
              paddingTop: 40,
              animation: "fade-in 0.8s ease both",
            }}>
              {connected ? "Say something…" : "Connect to begin"}
            </div>
          ) : (
            messages.map((msg, i) => (
              <MessageBubble
                key={i}
                role={msg.role}
                content={msg.parts.map(p =>
                  typeof p.text === "string" ? p.text
                    : (p.text as { spoken?: string })?.spoken ?? ""
                ).join("")}
              />
            ))
          )}
          <div ref={transcriptEndRef} />
        </div>
      </main>

      {/* ── footer ── */}
      <footer style={{
        flexShrink: 0,
        borderTop: "1px solid rgba(0,0,0,0.07)",
        background: "rgba(247,247,248,0.9)",
        backdropFilter: "blur(16px)",
        position: "relative",
        zIndex: 10,
      }}>
        {/* collapsible logs */}
        {logsOpen && (
          <div style={{
            borderBottom: "1px solid rgba(0,0,0,0.06)",
            padding: "12px 28px",
            maxHeight: 150,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 1,
            background: "#ffffff",
            animation: "soft-slide-up 0.2s ease both",
          }}>
            {logs.slice(-20).map((l, i) => (
              <LogLine key={i} entry={l} />
            ))}
            <div ref={logsEndRef} />
          </div>
        )}

        {/* metrics strip */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 28px",
          height: 44,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <MetricPill label="TTFC" value={fmtMs(metrics.ttfc_ms)}
              pass={metrics.ttfc_ms != null ? metrics.ttfc_ms < 60 : null} />
            <span style={{ color: "rgba(0,0,0,0.12)", fontSize: 11 }}>·</span>
            <MetricPill label="RTF" value={fmtNum(metrics.rtf, 3)}
              pass={metrics.rtf != null ? metrics.rtf < 0.15 : null} />
            <span style={{ color: "rgba(0,0,0,0.12)", fontSize: 11 }}>·</span>
            <MetricPill label="E2E" value={fmtMs(metrics.e2e_ms)}
              pass={metrics.e2e_ms != null ? metrics.e2e_ms < 500 : null} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <button
              onClick={() => setLogsOpen(o => !o)}
              style={{
                background: "none",
                border: "none",
                color: logsOpen ? "#5a5a6a" : "#9a9aaa",
                fontSize: 11,
                fontFamily: "var(--mono)",
                cursor: "pointer",
                padding: "3px 0",
                transition: "color 0.2s ease",
              }}
            >
              {logsOpen ? "hide logs" : "logs"}
            </button>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "#c0c0cc" }}>
              pipecat-ai
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
