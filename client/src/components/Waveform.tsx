import { useEffect, useRef } from "react";

interface Props { active: boolean; color?: string }

const BAR_COUNT = 36;

export default function Waveform({ active, color = "110,181,255" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef  = useRef<number>(0);
  const barsRef   = useRef<number[]>(Array(BAR_COUNT).fill(0));
  const timeRef   = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas!.offsetWidth * dpr;
      const H = canvas!.offsetHeight * dpr;
      canvas!.width = W;
      canvas!.height = H;
      ctx.clearRect(0, 0, W, H);

      timeRef.current += active ? 0.045 : 0.012;
      const t = timeRef.current;
      const bars = barsRef.current;

      for (let i = 0; i < BAR_COUNT; i++) {
        const wave = active
          ? 0.08 + 0.92 * Math.pow(Math.abs(
              Math.sin(t * 1.8 + i * 0.42) * 0.55 +
              Math.sin(t * 3.1 + i * 0.27) * 0.28 +
              Math.sin(t * 0.9 + i * 0.68) * 0.17
            ), 0.7)
          : 0.02 + 0.05 * Math.abs(Math.sin(t * 0.4 + i * 0.6));
        bars[i] += (wave - bars[i]) * (active ? 0.18 : 0.06);
      }

      const barW = W / BAR_COUNT;
      const gap  = barW * 0.35;
      const bW   = Math.max(1, barW - gap);
      const cx   = W / 2;

      for (let i = 0; i < BAR_COUNT; i++) {
        const h = bars[i] * H * 0.85;
        const x = i * barW + gap / 2;
        const y = (H - h) / 2;
        const distFromCenter = Math.abs((x + bW / 2) - cx) / cx;
        const edgeFade = Math.pow(1 - distFromCenter * 0.6, 1.5);
        const alpha = active
          ? (0.15 + bars[i] * 0.75) * edgeFade
          : 0.12 * edgeFade;

        ctx.fillStyle = `rgba(${color},${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.roundRect(x, y, bW, Math.max(2, h), 2);
        ctx.fill();
      }

      frameRef.current = requestAnimationFrame(draw);
    }

    frameRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frameRef.current);
  }, [active, color]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
