import { useState, useEffect, useRef, useCallback } from "react";
import {
  usePipecatClient,
  usePipecatClientTransportState,
  usePipecatClientMicControl,
  useRTVIClientEvent,
} from "@pipecat-ai/client-react";
import { RTVIEvent } from "@pipecat-ai/client-js";
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
interface Turn {
  id: number;
  role: "user" | "bot";
  text: string;
  final: boolean;
}

function now() {
  return new Date().toISOString().slice(11, 19);
}
function fmtMs(v: number | null) {
  return v == null ? "—" : `${v.toFixed(0)}`;
}
function fmtRTF(v: number | null) {
  return v == null ? "—" : v.toFixed(3);
}

// ─── Orb ──────────────────────────────────────────────────────────────────
function Orb({
  state,
}: {
  state: "idle" | "listening" | "speaking" | "connecting" | "off";
}) {
  const configs = {
    off: { bg: "#e4e4e7", shadow: "none", ring: false, anim: "none" },
    idle: { bg: "#d4d4d8", shadow: "none", ring: false, anim: "none" },
    connecting: {
      bg: "#fbbf24",
      shadow: "0 0 0 0 rgba(251,191,36,0)",
      ring: false,
      anim: "spin 1s linear infinite",
    },
    listening: {
      bg: "#16a34a",
      shadow: "0 8px 32px rgba(22,163,74,0.3)",
      ring: true,
      ringColor: "22,163,74",
      anim: "breathe 2s ease-in-out infinite",
    },
    speaking: {
      bg: "#2563eb",
      shadow: "0 8px 40px rgba(37,99,235,0.35)",
      ring: true,
      ringColor: "37,99,235",
      anim: "breathe 1.6s ease-in-out infinite",
    },
  };
  const c = configs[state];

  return (
    <div style={{ position: "relative", width: 88, height: 88, flexShrink: 0 }}>
      {/* pulse rings */}
      {c.ring && (
        <>
          <div
            style={{
              position: "absolute",
              inset: -16,
              borderRadius: "50%",
              border: `1.5px solid rgba(${(c as { ringColor: string }).ringColor},0.25)`,
              animation: "pulse-ring 1.8s ease-out infinite",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: -16,
              borderRadius: "50%",
              border: `1.5px solid rgba(${(c as { ringColor: string }).ringColor},0.15)`,
              animation: "pulse-ring 1.8s ease-out infinite",
              animationDelay: "0.6s",
            }}
          />
        </>
      )}
      {/* orb */}
      <div
        style={{
          width: 88,
          height: 88,
          borderRadius: "50%",
          background:
            state === "connecting"
              ? `conic-gradient(${c.bg} 0deg 90deg, #e4e4e7 90deg 360deg)`
              : `radial-gradient(circle at 35% 32%, ${c.bg}, ${c.bg}cc 100%)`,
          boxShadow: c.shadow,
          animation: c.anim,
          transition: "background 0.5s ease, box-shadow 0.5s ease",
        }}
      />
    </div>
  );
}

// ─── Message ──────────────────────────────────────────────────────────────
function Message({ role, content }: { role: string; content: string }) {
  const isUser = role === "user" || role === "human";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        animation: "msg-in 0.25s ease both",
      }}
    >
      <div
        style={{
          maxWidth: "75%",
          padding: "10px 14px",
          borderRadius: isUser ? "16px 16px 4px 16px" : "4px 16px 16px 16px",
          fontSize: 14,
          lineHeight: 1.6,
          fontWeight: 400,
          letterSpacing: "-0.005em",
          ...(isUser
            ? {
                background: "#2563eb",
                color: "#ffffff",
              }
            : {
                background: "#f4f4f5",
                color: "#09090b",
                border: "1px solid #e4e4e7",
              }),
        }}
      >
        {content}
      </div>
    </div>
  );
}

// ─── Metric card ──────────────────────────────────────────────────────────

// ─── Log entry ────────────────────────────────────────────────────────────
function LogEntry({ entry }: { entry: LogEntry }) {
  const col = {
    info: "#27272a",
    warn: "#d97706",
    error: "#dc2626",
    debug: "#71717a",
  }[entry.level];
  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        fontFamily: "var(--mono)",
        fontSize: 11,
        lineHeight: 1.6,
        animation: "fade-up 0.1s ease both",
      }}
    >
      <span style={{ color: "#71717a", flexShrink: 0 }}>{entry.ts}</span>
      <span style={{ color: col, wordBreak: "break-all" as const }}>
        {entry.msg}
      </span>
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────
export default function Dashboard({
  wsUrl,
  onConnect,
}: {
  wsUrl: string;
  onConnect: (url: string) => void;
}) {
  const transportState = usePipecatClientTransportState();
  const { isMicEnabled, enableMic } = usePipecatClientMicControl();
  const client = usePipecatClient();

  const [metrics, setMetrics] = useState<Metrics>({
    ttfc_ms: null,
    rtf: null,
    e2e_ms: null,
  });
  const [logs, setLogs] = useState<LogEntry[]>([
    { ts: now(), level: "info", msg: "Waiting for connection" },
  ]);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [urlInput, setUrlInput] = useState(wsUrl);
  const turnIdRef = useRef(0);

  const transcriptEnd = useRef<HTMLDivElement>(null);
  const logsEnd = useRef<HTMLDivElement>(null);

  const addLog = useCallback((level: LogEntry["level"], msg: string) => {
    setLogs((p) => [...p.slice(-199), { ts: now(), level, msg }]);
  }, []);

  useRTVIClientEvent(RTVIEvent.Metrics, (data: unknown) => {
    const raw = data as Record<string, { processor: string; value: number }[]>;
    const ttfb = raw?.ttfb?.find((d) => d.processor === "QwenTTSService");
    const proc = raw?.processing?.find((d) => d.processor === "QwenTTSService");
    if (!ttfb && !proc) return;
    setMetrics((prev) => ({
      ...prev,
      ttfc_ms: ttfb ? ttfb.value * 1000 : prev.ttfc_ms,
      e2e_ms: proc ? proc.value * 1000 : prev.e2e_ms,
    }));
  });

  // Build transcript from raw transcript events — correct ordering guaranteed
  useRTVIClientEvent(RTVIEvent.UserTranscript, (data: unknown) => {
    const { text, final } = data as { text: string; final: boolean };
    if (!text?.trim()) return;
    setTurns((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "user" && !last.final) {
        // update in-progress user turn
        return [...prev.slice(0, -1), { ...last, text, final }];
      }
      if (!final && last?.role === "user") return prev; // skip interim if already final
      return [...prev, { id: ++turnIdRef.current, role: "user", text, final }];
    });
  });

  useRTVIClientEvent(RTVIEvent.BotOutput, (data: unknown) => {
    const text =
      typeof data === "string"
        ? data
        : ((data as { text?: string })?.text ?? "");
    if (!text?.trim()) return;
    setTurns((prev) => {
      const last = prev[prev.length - 1];
      // Append to in-progress bot turn or start new one
      if (last && last.role === "bot" && !last.final) {
        return [...prev.slice(0, -1), { ...last, text: last.text + text }];
      }
      return [
        ...prev,
        { id: ++turnIdRef.current, role: "bot", text, final: false },
      ];
    });
  });

  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => {
    setIsBotSpeaking(true);
    addLog("info", "Bot speaking");
  });
  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking, () => {
    setIsBotSpeaking(false);
    addLog("info", "Bot done");
    // Mark last bot turn as final
    setTurns((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "bot" && !last.final) {
        return [...prev.slice(0, -1), { ...last, final: true }];
      }
      return prev;
    });
  });
  useRTVIClientEvent(RTVIEvent.UserStartedSpeaking, () => {
    setIsUserSpeaking(true);
    addLog("debug", "User speaking");
  });
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, () => {
    setIsUserSpeaking(false);
  });

  const prevState = useRef(transportState);
  useEffect(() => {
    if (prevState.current !== transportState) {
      addLog("info", transportState);
      prevState.current = transportState;
    }
  }, [transportState, addLog]);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);
  useEffect(() => {
    logsEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const connected =
    transportState === "ready" || transportState === "connected";
  const isConnecting =
    transportState === "connecting" || transportState === "authenticating";

  const handleToggle = async () => {
    if (!connected) {
      setTurns([]);
      addLog("info", `Connecting to ${urlInput}…`);
      onConnect(urlInput.trim());
    } else {
      addLog("info", "Disconnecting…");
      await client!.disconnect();
    }
  };

  const orbState = isConnecting
    ? "connecting"
    : isBotSpeaking
      ? "speaking"
      : isUserSpeaking
        ? "listening"
        : connected
          ? "idle"
          : "off";

  const statusLabel = isConnecting
    ? "Connecting"
    : isBotSpeaking
      ? "Speaking"
      : isUserSpeaking
        ? "Listening"
        : connected
          ? "Ready"
          : "Disconnected";
  const statusDot = isConnecting
    ? "#fbbf24"
    : isBotSpeaking
      ? "#2563eb"
      : isUserSpeaking
        ? "#16a34a"
        : connected
          ? "#22c55e"
          : "#d4d4d8";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100dvh",
        background: "#ffffff",
        overflow: "hidden",
      }}
    >
      {/* ── TOPBAR ── */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          padding: "0 20px",
          height: 48,
          flexShrink: 0,
          borderBottom: "1px solid #e4e4e7",
          background: "rgba(255,255,255,0.95)",
          backdropFilter: "blur(12px)",
          zIndex: 20,
          gap: 12,
        }}
      >
        {/* left: branding */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          <span style={{ fontSize: 14, fontWeight: 600, color: "#09090b", letterSpacing: "-0.02em" }}>
            Qwen3 TTS
          </span>
          <span style={{
            fontSize: 10, fontWeight: 500, color: "#6366f1",
            background: "#eef2ff", border: "1px solid #c7d2fe",
            padding: "2px 7px", borderRadius: 99, letterSpacing: "0.02em",
          }}>
            megakernel
          </span>
        </div>

        {/* right: status + actions */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 12, flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%", background: statusDot, flexShrink: 0,
              animation: isBotSpeaking || isUserSpeaking || isConnecting ? "blink 1.4s ease infinite" : "none",
              transition: "background 0.3s",
            }} />
            <span style={{ fontSize: 13, color: "#09090b", fontWeight: 500 }}>{statusLabel}</span>
          </div>
          {connected && (
            <button
              onClick={() => enableMic(!isMicEnabled)}
              style={{
                height: 30, padding: "0 12px", borderRadius: 8,
                fontSize: 12, fontWeight: 500, cursor: "pointer", transition: "all 0.15s",
                background: isMicEnabled ? "#f0fdf4" : "#fff0f0",
                border: `1px solid ${isMicEnabled ? "#86efac" : "#fca5a5"}`,
                color: isMicEnabled ? "#15803d" : "#dc2626",
                display: "flex", alignItems: "center", gap: 5,
              }}
            >
              {isMicEnabled ? "🎙️ Mic on" : "🔇 Mic off"}
            </button>
          )}
          {connected && (
            <button
              onClick={handleToggle}
              style={{
                height: 30, padding: "0 14px", borderRadius: 8,
                fontSize: 12, fontWeight: 500, cursor: "pointer",
                background: "#fef2f2", border: "1px solid #fecaca", color: "#dc2626",
              }}
            >
              Disconnect
            </button>
          )}
        </div>
      </header>

      {/* ── BODY: left = conversation, right = always-on panel ── */}
      <div
        style={{ flex: 1, display: "flex", minHeight: 0, overflow: "hidden" }}
      >
        {/* ── LEFT: orb + transcript/connect card ── */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
            overflow: "hidden",
          }}
        >
          {/* orb */}
          <div
            style={{
              flexShrink: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              width: "100%",
              gap: 12,
              paddingTop: 36,
              paddingBottom: 8,
            }}
          >
            <Orb state={orbState} />
            <div
              style={{
                width: 180,
                height: 28,
                opacity: isBotSpeaking ? 1 : 0,
                transition: "opacity 0.4s ease",
              }}
            >
              <Waveform active={isBotSpeaking} color="37,99,235" />
            </div>
            <span
              style={{
                fontSize: 13,
                color: "#a1a1aa",
                fontWeight: 400,
                minHeight: 20,
              }}
            >
              {isBotSpeaking
                ? "Speaking…"
                : isUserSpeaking
                  ? "Listening…"
                  : isConnecting
                    ? "Connecting…"
                    : ""}
            </span>
          </div>

          {/* connect card — shown when disconnected */}
          {!connected && !isConnecting && (
            <div
              style={{
                flexShrink: 0,
                display: "flex",
                justifyContent: "center",
                padding: "8px 24px 0",
                animation: "fade-up 0.3s ease both",
              }}
            >
              <div
                style={{
                  width: "100%",
                  maxWidth: 400,
                  border: "1px solid #e4e4e7",
                  borderRadius: 12,
                  padding: "16px 16px",
                  background: "#fafafa",
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <label
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#71717a",
                    letterSpacing: "0.05em",
                    textTransform: "uppercase" as const,
                  }}
                >
                  GPU Server URL
                </label>
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleToggle()}
                    placeholder="ws://your-gpu-ip:8000/ws"
                    spellCheck={false}
                    style={{
                      flex: 1,
                      height: 36,
                      padding: "0 12px",
                      borderRadius: 8,
                      border: "1px solid #e4e4e7",
                      background: "#ffffff",
                      fontSize: 13,
                      fontFamily: "var(--mono)",
                      color: "#09090b",
                      outline: "none",
                      minWidth: 0,
                    }}
                  />
                  <button
                    onClick={handleToggle}
                    style={{
                      height: 36,
                      padding: "0 18px",
                      borderRadius: 8,
                      fontSize: 13,
                      fontWeight: 600,
                      cursor: "pointer",
                      background: "#2563eb",
                      border: "none",
                      color: "#fff",
                      flexShrink: 0,
                      whiteSpace: "nowrap" as const,
                    }}
                  >
                    Connect
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* transcript */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              width: "100%",
              maxWidth: 640,
              margin: "0 auto",
              padding: "16px 24px 24px",
            }}
          >
            {turns.length === 0 ? (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span style={{ fontSize: 13, color: "#d4d4d8" }}>
                  {connected ? "Start speaking to begin" : ""}
                </span>
              </div>
            ) : (
              turns.map((turn) => (
                <Message key={turn.id} role={turn.role} content={turn.text} />
              ))
            )}
            <div ref={transcriptEnd} />
          </div>
        </div>

        {/* ── RIGHT: always-on metrics + logs panel ── */}
        <div
          style={{
            width: 272,
            flexShrink: 0,
            borderLeft: "1px solid #e4e4e7",
            background: "#fafafa",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* metrics section */}
          <div style={{ flexShrink: 0, borderBottom: "1px solid #e4e4e7" }}>
            <div
              style={{
                padding: "12px 14px 10px",
                borderBottom: "1px solid #f0f0f0",
              }}
            >
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#09090b",
                  letterSpacing: "0.01em",
                }}
              >
                Performance
              </span>
            </div>
            <div
              style={{
                padding: "10px 14px 12px",
                display: "flex",
                flexDirection: "column",
                gap: 8,
              }}
            >
              {[
                {
                  label: "TTFC",
                  value: metrics.ttfc_ms != null ? `${fmtMs(metrics.ttfc_ms)} ms` : "—",
                  target: "target < 60 ms",
                  pass: metrics.ttfc_ms != null ? metrics.ttfc_ms < 60 : null,
                },
                {
                  label: "RTF",
                  value: metrics.rtf != null ? fmtRTF(metrics.rtf) : "—",
                  target: "target < 0.15",
                  pass: metrics.rtf != null ? metrics.rtf < 0.15 : null,
                },
                {
                  label: "E2E",
                  value: metrics.e2e_ms != null ? `${fmtMs(metrics.e2e_ms)} ms` : "—",
                  target: "target < 500 ms",
                  pass: metrics.e2e_ms != null ? metrics.e2e_ms < 500 : null,
                },
              ].map(({ label, value, target, pass }) => (
                <div key={label} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                      <span style={{
                        fontSize: 11, fontWeight: 700, color: "#71717a",
                        letterSpacing: "0.06em", fontFamily: "var(--mono)",
                      }}>
                        {label}
                      </span>
                      <span style={{
                        fontSize: 16, fontWeight: 700, fontFamily: "var(--mono)",
                        color: pass == null ? "#09090b" : pass ? "#16a34a" : "#dc2626",
                        fontVariantNumeric: "tabular-nums",
                      }}>
                        {value}
                      </span>
                    </div>
                    {pass != null && (
                      <span style={{
                        fontSize: 10, fontWeight: 600, padding: "1px 6px", borderRadius: 99,
                        background: pass ? "#f0fdf4" : "#fef2f2",
                        color: pass ? "#15803d" : "#dc2626",
                        border: `1px solid ${pass ? "#bbf7d0" : "#fecaca"}`,
                      }}>
                        {pass ? "✓" : "↑"}
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: 10, color: "#52525b", fontFamily: "var(--mono)" }}>
                    {target}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* logs section — always visible, scrollable */}
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "10px 14px 8px",
                borderBottom: "1px solid #f0f0f0",
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#09090b",
                  letterSpacing: "0.01em",
                }}
              >
                Logs
              </span>
            </div>
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                padding: "8px 14px",
                display: "flex",
                flexDirection: "column",
                gap: 2,
              }}
            >
              {logs.map((l, i) => (
                <LogEntry key={i} entry={l} />
              ))}
              <div ref={logsEnd} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
