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
    ? "rgba(110,181,255,0.95)"
    : state === "listening"
    ? "rgba(74,222,128,0.9)"
    : state === "connecting"
    ? "rgba(251,191,36,0.8)"
    : "rgba(90,90,110,0.5)";

  const glowColor = state === "speaking"
    ? "rgba(110,181,255,0.25)"
    : state === "listening"
    ? "rgba(74,222,128,0.2)"
    : state === "connecting"
    ? "rgba(251,191,36,0.15)"
    : "transparent";

  const ringColor = state === "speaking"
    ? "rgba(110,181,255,0.15)"
    : state === "listening"
    ? "rgba(74,222,128,0.12)"
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
      {/* outer pulse ring */}
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
      {/* glow */}
      <div style={{
        position: "absolute",
        inset: -16,
        borderRadius: "50%",
        background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`,
        pointerEvents: "none",
      }} />
      {/* core orb */}
      <div style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: `radial-gradient(circle at 38% 35%, ${coreColor}, ${coreColor.replace("0.9", "0.5").replace("0.95", "0.55").replace("0.8", "0.45").replace("0.5", "0.2")} 100%)`,
        boxShadow: state !== "idle"
          ? `0 0 32px ${glowColor}, 0 0 60px ${glowColor.replace("0.25", "0.1").replace("0.2", "0.08").replace("0.15", "0.06")}`
          : "none",
        animation: state === "connecting" ? connectAnim : breatheAnim,
        transition: "background 0.6s ease, box-shadow 0.6s ease",
        cursor: "pointer",
        position: "relative",
        zIndex: 1,
        border: state !== "idle"
          ? `1px solid ${ringColor.replace("0.15", "0.3").replace("0.12", "0.25")}`
          : "1px solid rgba(255,255,255,0.06)",
      }} />
    </div>
  );
}

// ── message bubble ─────────────────────────────────────────────────────────
function MessageBubble({ role, content, isLatest }: {
  role: string; content: string; isLatest?: boolean;
}) {
  const isUser = role === "user";
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      animation: "msg-enter 0.3s cubic-bezier(0.16,1,0.3,1) both",
      gap: 4,
    }}>
      <div style={{
        maxWidth: "72%",
        padding: isUser ? "10px 14px" : "12px 16px",
        borderRadius: isUser
          ? "18px 18px 4px 18px"
          : "18px 18px 18px 4px",
        background: isUser
          ? "rgba(110,181,255,0.1)"
          : "rgba(255,255,255,0.04)",
        border: isUser
          ? "1px solid rgba(110,181,255,0.18)"
          : "1px solid rgba(255,255,255,0.06)",
        color: isUser ? "rgba(200,220,255,0.95)" : "rgba(240,240,242,0.9)",
        fontSize: 15,
        lineHeight: 1.6,
        fontWeight: 400,
        letterSpacing: "-0.01em",
        backdropFilter: "blur(12px)",
        transition: "all 0.2s ease",
        ...(isLatest && !isUser ? {
          borderColor: "rgba(110,181,255,0.14)",
        } : {}),
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
  const color = pass == null
    ? "rgba(255,255,255,0.35)"
    : pass
    ? "rgba(74,222,128,0.7)"
    : "rgba(248,113,113,0.7)";
  return (
    <span style={{
      fontFamily: "var(--mono)",
      fontSize: 11,
      color: "rgba(255,255,255,0.4)",
      letterSpacing: "0.01em",
      display: "inline-flex",
      alignItems: "center",
      gap: 4,
    }}>
      <span style={{ color: "rgba(255,255,255,0.22)", fontSize: 10 }}>{label}</span>
      <span style={{ color, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </span>
  );
}

// ── log line ───────────────────────────────────────────────────────────────
function LogLine({ entry }: { entry: LogEntry }) {
  const colors: Record<string, string> = {
    info:  "rgba(255,255,255,0.35)",
    warn:  "rgba(251,191,36,0.6)",
    error: "rgba(248,113,113,0.7)",
    debug: "rgba(255,255,255,0.15)",
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
      <span style={{ color: "rgba(255,255,255,0.12)", flexShrink: 0 }}>{entry.ts}</span>
      <span style={{ wordBreak: "break-all" }}>{entry.msg}</span>
    </div>
  );
}

// ── connection button ──────────────────────────────────────────────────────
function ConnectButton({ connected, onClick }: { connected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: connected ? "rgba(248,113,113,0.08)" : "rgba(110,181,255,0.1)",
        border: `1px solid ${connected ? "rgba(248,113,113,0.25)" : "rgba(110,181,255,0.25)"}`,
        color: connected ? "rgba(248,113,113,0.85)" : "rgba(110,181,255,0.9)",
        padding: "7px 16px",
        borderRadius: 8,
        fontSize: 12,
        fontFamily: "var(--sans)",
        fontWeight: 500,
        letterSpacing: "0.01em",
        cursor: "pointer",
        transition: "all 0.2s ease",
        backdropFilter: "blur(8px)",
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLButtonElement).style.background = connected
          ? "rgba(248,113,113,0.14)" : "rgba(110,181,255,0.16)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLButtonElement).style.background = connected
          ? "rgba(248,113,113,0.08)" : "rgba(110,181,255,0.1)";
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

  // ── events ─────────────────────────────────────────────────────────────
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
      addLog("info", `${transportState}`);
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
    if (!connected) {
      addLog("info", "Connecting…");
      await pipecatClient.connect();
    } else {
      addLog("info", "Disconnecting…");
      await pipecatClient.disconnect();
    }
  };

  const orbState = isConnecting ? "connecting"
    : isBotSpeaking ? "speaking"
    : isUserSpeaking ? "listening"
    : connected ? "idle"
    : "idle";

  const statusText = isConnecting ? "Connecting…"
    : isBotSpeaking ? "Speaking"
    : isUserSpeaking ? "Listening"
    : connected ? "Ready"
    : "Disconnected";

  const statusColor = isConnecting ? "rgba(251,191,36,0.7)"
    : isBotSpeaking ? "rgba(110,181,255,0.8)"
    : isUserSpeaking ? "rgba(74,222,128,0.8)"
    : connected ? "rgba(255,255,255,0.3)"
    : "rgba(255,255,255,0.2)";

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100dvh",
      background: "var(--bg)",
      overflow: "hidden",
      position: "relative",
    }}>

      {/* ── ambient background glow ── */}
      <div style={{
        position: "absolute",
        top: "5%",
        left: "50%",
        transform: "translateX(-50%)",
        width: 600,
        height: 300,
        background: isBotSpeaking
          ? "radial-gradient(ellipse, rgba(110,181,255,0.045) 0%, transparent 70%)"
          : isUserSpeaking
          ? "radial-gradient(ellipse, rgba(74,222,128,0.035) 0%, transparent 70%)"
          : "radial-gradient(ellipse, rgba(110,181,255,0.02) 0%, transparent 70%)",
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
        borderBottom: "1px solid rgba(255,255,255,0.045)",
        backdropFilter: "blur(20px)",
        background: "rgba(10,10,12,0.7)",
        position: "relative",
        zIndex: 10,
      }}>
        {/* brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{
            fontSize: 13,
            fontWeight: 500,
            color: "rgba(255,255,255,0.55)",
            letterSpacing: "0.02em",
          }}>
            Qwen3 TTS
          </span>
          <span style={{
            width: 1,
            height: 12,
            background: "rgba(255,255,255,0.1)",
          }} />
          <span style={{
            fontSize: 11,
            color: "rgba(255,255,255,0.22)",
            letterSpacing: "0.02em",
            fontFamily: "var(--mono)",
          }}>
            megakernel
          </span>
        </div>

        {/* center status */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          padding: "5px 12px",
          borderRadius: 20,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <span style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: statusColor,
            boxShadow: connected || isConnecting ? `0 0 6px ${statusColor}` : "none",
            animation: (isBotSpeaking || isUserSpeaking || isConnecting)
              ? "status-fade 1.5s ease-in-out infinite"
              : "none",
            transition: "background 0.4s ease, box-shadow 0.4s ease",
            flexShrink: 0,
          }} />
          <span style={{
            fontSize: 12,
            color: statusColor,
            fontWeight: 450,
            letterSpacing: "0.01em",
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
                background: isMicEnabled
                  ? "rgba(74,222,128,0.1)"
                  : "rgba(255,255,255,0.04)",
                border: `1px solid ${isMicEnabled ? "rgba(74,222,128,0.22)" : "rgba(255,255,255,0.08)"}`,
                color: isMicEnabled
                  ? "rgba(74,222,128,0.8)"
                  : "rgba(255,255,255,0.3)",
                padding: "6px 12px",
                borderRadius: 8,
                fontSize: 12,
                fontFamily: "var(--sans)",
                fontWeight: 450,
                cursor: "pointer",
                transition: "all 0.2s ease",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span style={{ fontSize: 10 }}>
                {isMicEnabled ? "●" : "○"}
              </span>
              Mic
            </button>
          )}
          <ConnectButton connected={connected} onClick={handleToggle} />
        </div>
      </header>

      {/* ── main conversation area ── */}
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

        {/* orb + waveform zone */}
        <div style={{
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 20,
          paddingTop: 36,
          paddingBottom: 20,
        }}>
          <VoiceOrb state={orbState} />

          {/* waveform strip — only visible when bot speaking */}
          <div style={{
            width: 240,
            height: 36,
            opacity: isBotSpeaking ? 1 : 0,
            transition: "opacity 0.5s ease",
          }}>
            <Waveform active={isBotSpeaking} />
          </div>
        </div>

        {/* transcript */}
        <div style={{
          flex: 1,
          width: "100%",
          maxWidth: 680,
          overflowY: "auto",
          padding: "0 24px 20px",
          display: "flex",
          flexDirection: "column",
          gap: 10,
          maskImage: "linear-gradient(to bottom, transparent 0%, black 6%, black 100%)",
        }}>
          {(!messages || messages.length === 0) ? (
            <div style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "rgba(255,255,255,0.15)",
              fontSize: 14,
              fontWeight: 400,
              letterSpacing: "0.01em",
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
                isLatest={i === messages.length - 1}
              />
            ))
          )}
          <div ref={transcriptEndRef} />
        </div>
      </main>

      {/* ── bottom bar — metrics + log toggle ── */}
      <footer style={{
        flexShrink: 0,
        borderTop: "1px solid rgba(255,255,255,0.045)",
        background: "rgba(10,10,12,0.8)",
        backdropFilter: "blur(20px)",
        position: "relative",
        zIndex: 10,
      }}>
        {/* collapsible log panel */}
        {logsOpen && (
          <div style={{
            borderBottom: "1px solid rgba(255,255,255,0.045)",
            padding: "12px 28px",
            maxHeight: 160,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 1,
            animation: "soft-slide-up 0.2s ease both",
          }}>
            {logs.slice(-20).map((l, i) => (
              <LogLine key={i} entry={l} />
            ))}
            <div ref={logsEndRef} />
          </div>
        )}

        {/* footer strip */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 28px",
          height: 44,
        }}>
          {/* metrics pills */}
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}>
            <MetricPill
              label="TTFC"
              value={fmtMs(metrics.ttfc_ms)}
              pass={metrics.ttfc_ms != null ? metrics.ttfc_ms < 60 : null}
            />
            <span style={{ color: "rgba(255,255,255,0.08)", fontSize: 10 }}>·</span>
            <MetricPill
              label="RTF"
              value={fmtNum(metrics.rtf, 3)}
              pass={metrics.rtf != null ? metrics.rtf < 0.15 : null}
            />
            <span style={{ color: "rgba(255,255,255,0.08)", fontSize: 10 }}>·</span>
            <MetricPill
              label="E2E"
              value={fmtMs(metrics.e2e_ms)}
              pass={metrics.e2e_ms != null ? metrics.e2e_ms < 500 : null}
            />
          </div>

          {/* right side: wordmark + log toggle */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <button
              onClick={() => setLogsOpen(o => !o)}
              style={{
                background: "none",
                border: "none",
                color: logsOpen ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.18)",
                fontSize: 11,
                fontFamily: "var(--mono)",
                cursor: "pointer",
                padding: "3px 0",
                letterSpacing: "0.02em",
                transition: "color 0.2s ease",
              }}
            >
              {logsOpen ? "hide logs" : "logs"}
            </button>
            <span style={{
              fontFamily: "var(--mono)",
              fontSize: 10,
              color: "rgba(255,255,255,0.1)",
              letterSpacing: "0.04em",
            }}>
              pipecat-ai
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
