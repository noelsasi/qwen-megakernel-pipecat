interface Props { data: number[] }

const W = 388;
const H = 72;
const PAD = { t: 8, r: 8, b: 20, l: 44 };

export default function LatencyChart({ data }: Props) {
  if (data.length < 2) {
    return (
      <div style={{ height: H, display: "flex", alignItems: "center",
        fontFamily: "var(--mono)", fontSize: 10, color: "var(--text)" }}>
        — awaiting data —
      </div>
    );
  }

  const iW = W - PAD.l - PAD.r;
  const iH = H - PAD.t - PAD.b;

  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;

  const xStep = iW / (data.length - 1);
  const pts = data.map((v, i) => ({
    x: PAD.l + i * xStep,
    y: PAD.t + iH - ((v - min) / range) * iH,
  }));

  const linePath = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${pts[pts.length-1].x},${PAD.t + iH} L${PAD.l},${PAD.t + iH} Z`;

  const last = data[data.length - 1];
  const pass = last < 200;

  // y-axis ticks
  const ticks = [0, 0.5, 1].map(f => ({
    val: min + f * range,
    y: PAD.t + iH - f * iH,
  }));

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible", display: "block" }}>
      <defs>
        <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={pass ? "#00d4ff" : "#ffb547"} stopOpacity="0.25" />
          <stop offset="100%" stopColor={pass ? "#00d4ff" : "#ffb547"} stopOpacity="0"   />
        </linearGradient>
      </defs>

      {/* grid lines */}
      {ticks.map((tk, i) => (
        <g key={i}>
          <line x1={PAD.l} y1={tk.y} x2={W - PAD.r} y2={tk.y}
            stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
          <text x={PAD.l - 5} y={tk.y + 3.5}
            fontSize={8} fill="rgba(255,255,255,0.25)"
            textAnchor="end" fontFamily="var(--mono)">
            {tk.val.toFixed(0)}
          </text>
        </g>
      ))}

      {/* 200ms target line */}
      {max > 0 && (
        <line
          x1={PAD.l} y1={PAD.t + iH - ((200 - min) / range) * iH}
          x2={W - PAD.r} y2={PAD.t + iH - ((200 - min) / range) * iH}
          stroke="rgba(255,71,87,0.3)" strokeWidth={1} strokeDasharray="3 3"
        />
      )}

      {/* area fill */}
      <path d={areaPath} fill="url(#area-grad)" />

      {/* line */}
      <path
        d={linePath}
        fill="none"
        stroke={pass ? "#00d4ff" : "#ffb547"}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* last point dot */}
      <circle
        cx={pts[pts.length - 1].x}
        cy={pts[pts.length - 1].y}
        r={3}
        fill={pass ? "#00d4ff" : "#ffb547"}
      />
      <circle
        cx={pts[pts.length - 1].x}
        cy={pts[pts.length - 1].y}
        r={6}
        fill="none"
        stroke={pass ? "#00d4ff" : "#ffb547"}
        strokeWidth={1}
        strokeOpacity={0.35}
      />

      {/* x-axis */}
      <line x1={PAD.l} y1={PAD.t + iH} x2={W - PAD.r} y2={PAD.t + iH}
        stroke="rgba(255,255,255,0.08)" strokeWidth={1} />

      {/* current value label */}
      <text
        x={W - PAD.r} y={pts[pts.length - 1].y - 6}
        fontSize={9} fill={pass ? "#00d4ff" : "#ffb547"}
        textAnchor="end" fontFamily="var(--mono)" fontWeight="600">
        {last.toFixed(0)}ms
      </text>
    </svg>
  );
}
