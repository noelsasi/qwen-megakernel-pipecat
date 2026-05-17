import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import readmeRaw from "../../../README.md?raw";
import architectureRaw from "../../../docs/architecture.md?raw";
import customDecodeRaw from "../../../docs/custom_decode_architecture.md?raw";
import findingsRaw from "../../../docs/findings.md?raw";
import progressRaw from "../../../docs/progress.md?raw";
import takehomeRaw from "../../../docs/takehome_project.docx.md?raw";

// ─── Setup Guide (moved from Dashboard) ────────────────────────────────────
function CodeBlock({ children }: { children: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ position: "relative" }}>
      <pre
        style={{
          margin: 0,
          padding: "10px 14px",
          background: "#09090b",
          borderRadius: 8,
          fontSize: 12,
          lineHeight: 1.7,
          fontFamily: "var(--mono)",
          color: "#e4e4e7",
          overflowX: "auto",
          whiteSpace: "pre-wrap" as const,
          wordBreak: "break-all" as const,
        }}
      >
        {children}
      </pre>
      <button
        onClick={() => {
          navigator.clipboard.writeText(children);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
        style={{
          position: "absolute",
          top: 6,
          right: 6,
          padding: "2px 7px",
          borderRadius: 5,
          border: "none",
          background: copied ? "#16a34a" : "#3f3f46",
          color: "#fff",
          fontSize: 10,
          fontWeight: 600,
          cursor: "pointer",
          transition: "background 0.2s",
        }}
      >
        {copied ? "✓" : "copy"}
      </button>
    </div>
  );
}

function SetupGuide() {
  const steps = [
    {
      n: "1",
      title: "GPU requirements",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: "#52525b",
              lineHeight: 1.6,
            }}
          >
            You need a machine (local, cloud, or rented) with:
          </p>
          <ul
            style={{
              margin: 0,
              paddingLeft: 18,
              fontSize: 13,
              color: "#52525b",
              lineHeight: 2,
            }}
          >
            <li>
              GPU: <strong>RTX 5090</strong> (Blackwell, sm_120a) — required for
              the megakernel
            </li>
            <li>
              CUDA: <strong>12.8+</strong>
            </li>
            <li>
              Disk: <strong>40 GB+</strong> free
            </li>
            <li>
              Port <strong>8080</strong> reachable from your browser
            </li>
          </ul>
          <p
            style={{
              margin: 0,
              fontSize: 12,
              color: "#71717a",
              lineHeight: 1.5,
            }}
          >
            Works on Vast.ai, RunPod, Lambda Labs, or your own RTX 5090 rig —
            the commands below are the same everywhere.
          </p>
        </div>
      ),
    },
    {
      n: "2",
      title: "Clone & setup (~10 min)",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <CodeBlock>
            {
              "git clone https://github.com/noelsasi/qwen-megakernel-pipecat\ncd qwen-megakernel-pipecat\nbash scripts/setup_server.sh"
            }
          </CodeBlock>
          <p
            style={{
              margin: 0,
              fontSize: 12,
              color: "#71717a",
              lineHeight: 1.5,
            }}
          >
            Installs PyTorch cu128, all deps, clones + patches + builds the
            megakernel extension.
          </p>
        </div>
      ),
    },
    {
      n: "3",
      title: "Set API keys",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <CodeBlock>{"cp .env.example .env\nnano .env"}</CodeBlock>
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: "#52525b",
              lineHeight: 1.6,
            }}
          >
            Replace the placeholder values:
          </p>
          <CodeBlock>
            {
              "OPENAI_API_KEY=sk-...\nDEEPGRAM_API_KEY=dg-...\nALLOWED_ORIGIN=https://qwen-megakernel-pipecat.vercel.app"
            }
          </CodeBlock>
        </div>
      ),
    },
    {
      n: "4",
      title: "Validate pipeline (optional)",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <CodeBlock>
            {
              "source .venv/bin/activate\nV2_MEGAKERNEL=1 python scripts/test_v2_decode.py"
            }
          </CodeBlock>
          <p
            style={{
              margin: 0,
              fontSize: 12,
              color: "#71717a",
              lineHeight: 1.5,
            }}
          >
            ~2 min. Should show 6 PASS stages ending with RTF ~0.126.
          </p>
        </div>
      ),
    },
    {
      n: "5",
      title: "Start the server",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <CodeBlock>
            {
              "source .venv/bin/activate\nset -a && source .env && set +a\nV2_MEGAKERNEL=1 TTS_BACKEND=v2 uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8080"
            }
          </CodeBlock>
          <p
            style={{
              margin: 0,
              fontSize: 12,
              color: "#71717a",
              lineHeight: 1.5,
            }}
          >
            Ready when you see{" "}
            <code
              style={{
                fontSize: 11,
                background: "#f4f4f5",
                padding: "1px 4px",
                borderRadius: 4,
              }}
            >
              Application startup complete
            </code>{" "}
            (~15 s).
          </p>
        </div>
      ),
    },
    {
      n: "6",
      title: "Connect & speak",
      content: (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: "#52525b",
              lineHeight: 1.6,
            }}
          >
            Enter your server address in the URL field:
          </p>
          <CodeBlock>{"ws://<SERVER_IP>:8080/ws"}</CodeBlock>
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: "#52525b",
              lineHeight: 1.6,
            }}
          >
            Click <strong>Connect</strong>, allow microphone, and speak. Switch
            to <strong>Metrics & Logs</strong> to see live RTF and TTFC.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "40px 56px 80px" }}>
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            margin: 0,
            fontSize: 24,
            fontWeight: 700,
            color: "#09090b",
            letterSpacing: "-0.02em",
          }}
        >
          Setup Guide
        </h1>
        <p style={{ margin: "8px 0 0", fontSize: 14, color: "#71717a" }}>
          Get the Qwen3 megakernel voice pipeline running on any RTX 5090
          machine.
        </p>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 0,
          border: "1px solid #e4e4e7",
          borderRadius: 12,
          overflow: "hidden",
          background: "#fff",
        }}
      >
        {steps.map((step, i) => (
          <div
            key={step.n}
            style={{
              padding: "20px 24px",
              borderBottom: i < steps.length - 1 ? "1px solid #f0f0f0" : "none",
            }}
          >
            <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
              <span
                style={{
                  flexShrink: 0,
                  width: 24,
                  height: 24,
                  borderRadius: "50%",
                  background: "#2563eb",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 11,
                  fontWeight: 700,
                  color: "#fff",
                  marginTop: 1,
                }}
              >
                {step.n}
              </span>
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <span
                  style={{ fontSize: 14, fontWeight: 600, color: "#09090b" }}
                >
                  {step.title}
                </span>
                {step.content}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Docs config ───────────────────────────────────────────────────────────
type DocEntry =
  | {
      id: string;
      label: string;
      tag: string;
      kind: "markdown";
      content: string;
    }
  | {
      id: string;
      label: string;
      tag: string;
      kind: "custom";
      component: React.ReactNode;
    };

const DOCS: DocEntry[] = [
  {
    id: "readme",
    label: "Overview",
    tag: "README",
    kind: "markdown",
    content: readmeRaw,
  },
  {
    id: "setup",
    label: "Setup Guide",
    tag: "guide",
    kind: "custom",
    component: <SetupGuide />,
  },
  {
    id: "takehome",
    label: "Take-Home Brief",
    tag: "brief",
    kind: "markdown",
    content: takehomeRaw,
  },
  {
    id: "architecture",
    label: "Architecture",
    tag: "docs",
    kind: "markdown",
    content: architectureRaw,
  },
  {
    id: "custom-decode",
    label: "Custom Decode",
    tag: "docs",
    kind: "markdown",
    content: customDecodeRaw,
  },
  {
    id: "findings",
    label: "Findings",
    tag: "docs",
    kind: "markdown",
    content: findingsRaw,
  },
  {
    id: "progress",
    label: "Progress Log",
    tag: "docs",
    kind: "markdown",
    content: progressRaw,
  },
];

const TAG_COLORS: Record<
  string,
  { bg: string; color: string; border: string }
> = {
  README: { bg: "#eef2ff", color: "#4f46e5", border: "#c7d2fe" },
  guide: { bg: "#ecfdf5", color: "#059669", border: "#6ee7b7" },
  brief: { bg: "#fff7ed", color: "#c2410c", border: "#fed7aa" },
  docs: { bg: "#f0fdf4", color: "#15803d", border: "#bbf7d0" },
};

export default function DocsPage({
  onBack,
  initialId,
}: {
  onBack: () => void;
  initialId?: string;
}) {
  const [activeId, setActiveId] = useState(initialId ?? "readme");
  const active = DOCS.find((d) => d.id === activeId) ?? DOCS[0];

  return (
    <div
      style={{
        display: "flex",
        height: "100dvh",
        background: "#ffffff",
        overflow: "hidden",
      }}
    >
      {/* ── Sidebar ── */}
      <div
        style={{
          width: 220,
          flexShrink: 0,
          borderRight: "1px solid #e4e4e7",
          background: "#fafafa",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "0 14px",
            height: 48,
            display: "flex",
            alignItems: "center",
            gap: 8,
            borderBottom: "1px solid #e4e4e7",
            flexShrink: 0,
          }}
        >
          <button
            onClick={onBack}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              background: "none",
              border: "none",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 500,
              color: "#71717a",
              padding: 0,
            }}
          >
            ← Back
          </button>
          <span style={{ fontSize: 12, color: "#d4d4d8" }}>|</span>
          <span style={{ fontSize: 12, fontWeight: 600, color: "#09090b" }}>
            Docs
          </span>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "8px 8px" }}>
          {DOCS.map((doc) => {
            const isActive = doc.id === activeId;
            const tagStyle = TAG_COLORS[doc.tag];
            return (
              <button
                key={doc.id}
                onClick={() => setActiveId(doc.id)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 8,
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: "none",
                  cursor: "pointer",
                  textAlign: "left" as const,
                  background: isActive ? "#f4f4f5" : "transparent",
                  transition: "background 0.1s",
                  marginBottom: 2,
                }}
              >
                <span
                  style={{
                    fontSize: 13,
                    fontWeight: isActive ? 600 : 400,
                    color: isActive ? "#09090b" : "#3f3f46",
                  }}
                >
                  {doc.label}
                </span>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 500,
                    padding: "1px 6px",
                    borderRadius: 99,
                    background: tagStyle.bg,
                    color: tagStyle.color,
                    border: `1px solid ${tagStyle.border}`,
                    flexShrink: 0,
                  }}
                >
                  {doc.tag}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Content ── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: 48,
            flexShrink: 0,
            borderBottom: "1px solid #e4e4e7",
            display: "flex",
            alignItems: "center",
            padding: "0 32px",
            gap: 10,
          }}
        >
          <span style={{ fontSize: 14, fontWeight: 600, color: "#09090b" }}>
            {active.label}
          </span>
          <span
            style={{
              fontSize: 10,
              fontWeight: 500,
              padding: "2px 7px",
              borderRadius: 99,
              ...TAG_COLORS[active.tag],
              border: `1px solid ${TAG_COLORS[active.tag].border}`,
            }}
          >
            {active.tag}
          </span>
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          {active.kind === "custom" ? (
            active.component
          ) : (
            <div style={{ padding: "40px 56px 80px" }}>
              <div
                className="prose"
                style={{ maxWidth: 860, margin: "0 auto" }}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {active.content}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
