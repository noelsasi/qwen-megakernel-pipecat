import { useEffect, useRef } from "react";

interface Props { active: boolean }

const BAR_COUNT = 48;

export default function Waveform({ active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef  = useRef<number>(0);
  const barsRef   = useRef<number[]>(Array(BAR_COUNT).fill(0.05));
  const timeRef   = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    function draw() {
      const W = canvas!.width = canvas!.offsetWidth * window.devicePixelRatio;
      const H = canvas!.height = canvas!.offsetHeight * window.devicePixelRatio;
      ctx.clearRect(0, 0, W, H);

      timeRef.current += 0.04;
      const t = timeRef.current;

      const bars = barsRef.current;
      for (let i = 0; i < BAR_COUNT; i++) {
        const target = active
          ? 0.1 + 0.9 * Math.abs(
              Math.sin(t * 2.1 + i * 0.38) * 0.5 +
              Math.sin(t * 3.7 + i * 0.22) * 0.3 +
              Math.sin(t * 1.3 + i * 0.61) * 0.2
            )
          : 0.04 + 0.02 * Math.abs(Math.sin(t * 0.3 + i * 0.5));
        bars[i] = bars[i] + (target - bars[i]) * 0.12;
      }

      const barW = W / BAR_COUNT;
      const gap  = barW * 0.25;
      const bW   = barW - gap;

      for (let i = 0; i < BAR_COUNT; i++) {
        const h = bars[i] * H * 0.9;
        const x = i * barW + gap / 2;
        const y = (H - h) / 2;

        const alpha = active ? 0.15 + bars[i] * 0.85 : 0.25;
        if (active) {
          ctx.fillStyle = `rgba(0,212,255,${alpha})`;
        } else {
          ctx.fillStyle = `rgba(60,80,100,${alpha})`;
        }
        ctx.beginPath();
        ctx.roundRect(x, y, bW, h, 1);
        ctx.fill();

        // peak glow on active high bars
        if (active && bars[i] > 0.6) {
          ctx.fillStyle = `rgba(0,212,255,${(bars[i] - 0.6) * 0.8})`;
          ctx.beginPath();
          ctx.roundRect(x, y, bW, 2, 1);
          ctx.fill();
        }
      }

      frameRef.current = requestAnimationFrame(draw);
    }

    frameRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frameRef.current);
  }, [active]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: 64, display: "block" }}
    />
  );
}
