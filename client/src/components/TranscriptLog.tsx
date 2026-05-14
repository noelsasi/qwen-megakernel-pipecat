import { usePipecatConversation } from "@pipecat-ai/client-react";

export default function TranscriptLog() {
  const { messages } = usePipecatConversation();

  if (!messages || messages.length === 0) {
    return (
      <div style={styles.empty}>
        Connect and speak to start a conversation.
      </div>
    );
  }

  return (
    <div style={styles.log}>
      {messages.map((msg, i) => (
        <div key={i} style={styles.msg(msg.role === "user")}>
          <span style={styles.role}>{msg.role === "user" ? "You" : "Agent"}</span>
          <span style={styles.text}>{msg.content}</span>
        </div>
      ))}
    </div>
  );
}

const styles = {
  log: {
    background: "#1a1a1a",
    borderRadius: 8,
    padding: 16,
    display: "flex",
    flexDirection: "column" as const,
    gap: 10,
    minHeight: 120,
    maxHeight: 400,
    overflowY: "auto" as const,
  },
  msg: (isUser: boolean): React.CSSProperties => ({
    display: "flex",
    gap: 10,
    alignItems: "flex-start",
    flexDirection: isUser ? "row-reverse" : "row",
  }),
  role: { fontSize: 12, color: "#666", minWidth: 40, paddingTop: 2 } as React.CSSProperties,
  text: { fontSize: 14, lineHeight: 1.5, color: "#ddd" } as React.CSSProperties,
  empty: { color: "#555", fontSize: 14, padding: 16 } as React.CSSProperties,
};
