import { usePipecatClientMicControl, usePipecatClientTransportState } from "@pipecat-ai/client-react";
import { pipecatClient } from "../lib/pipecatClient";

export default function MicToggle() {
  const { isMicEnabled, enableMic, disableMic } = usePipecatClientMicControl();
  const transportState = usePipecatClientTransportState();
  const connected = transportState === "ready";

  const handleConnect = async () => {
    if (!connected) {
      await pipecatClient.connect();
    } else {
      await pipecatClient.disconnect();
    }
  };

  return (
    <div style={styles.row}>
      <button onClick={handleConnect} style={styles.btn(connected ? "#c0392b" : "#27ae60")}>
        {connected ? "Disconnect" : "Connect"}
      </button>
      {connected && (
        <button
          onClick={() => (isMicEnabled ? disableMic() : enableMic())}
          style={styles.btn(isMicEnabled ? "#e67e22" : "#2980b9")}
        >
          {isMicEnabled ? "Mute" : "Unmute"}
        </button>
      )}
      <span style={styles.status}>
        {transportState}
      </span>
    </div>
  );
}

const styles = {
  row: { display: "flex", gap: 10, alignItems: "center" } as React.CSSProperties,
  btn: (bg: string): React.CSSProperties => ({
    padding: "8px 18px",
    background: bg,
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
  }),
  status: { fontSize: 13, color: "#888" } as React.CSSProperties,
};
