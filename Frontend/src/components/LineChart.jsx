import { useState } from "react";

const W = 420;
const H = 300;
const PAD = { top: 12, right: 12, bottom: 40, left: 46 };
const PW = W - PAD.left - PAD.right;
const PH = H - PAD.top - PAD.bottom;

// Linear interpolation of y at x along a sorted polyline
function yAt(points, x) {
  if (!points.length) return null;
  if (x <= points[0][0]) return points[0][1];
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    if (x <= x1) return x1 === x0 ? y1 : y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  }
  return points[points.length - 1][1];
}

/**
 * Square 0..max line chart with a dashed diagonal reference line.
 * series: [{ key, name, color, points: [[x, y], ...] }]
 */
export default function LineChart({ series, xLabel, yLabel, max = 1, format = (v) => v.toFixed(2),
  referenceLabel, markers = false, ariaLabel }) {
  const [hoverX, setHoverX] = useState(null);
  const sx = (v) => PAD.left + (v / max) * PW;
  const sy = (v) => PAD.top + PH - (v / max) * PH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);

  const onMove = (e) => {
    const svg = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - svg.left) / svg.width) * W;
    const v = ((x - PAD.left) / PW) * max;
    setHoverX(v < 0 || v > max ? null : v);
  };

  return (
    <figure>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={ariaLabel}
          onMouseMove={onMove} onMouseLeave={() => setHoverX(null)}>
          {/* grid + axes */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(0)} x2={sx(max)} y1={sy(t)} y2={sy(t)} stroke="#e5e5e3" />
              <text x={PAD.left - 6} y={sy(t) + 4} textAnchor="end" fontSize="11" fill="#52514e">{format(t)}</text>
              <text x={sx(t)} y={H - PAD.bottom + 16} textAnchor="middle" fontSize="11" fill="#52514e">{format(t)}</text>
            </g>
          ))}
          <text x={PAD.left + PW / 2} y={H - 6} textAnchor="middle" fontSize="12" fill="#0b0b0b">{xLabel}</text>
          <text x={12} y={PAD.top + PH / 2} textAnchor="middle" fontSize="12" fill="#0b0b0b"
            transform={`rotate(-90 12 ${PAD.top + PH / 2})`}>{yLabel}</text>
          <line x1={sx(0)} y1={sy(0)} x2={sx(max)} y2={sy(max)} stroke="#a3a29d" strokeDasharray="4 4" />

          {series.map((s) => (
            <g key={s.key}>
              <polyline fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round"
                points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
              {markers && s.points.map(([x, y], i) => (
                <circle key={i} cx={sx(x)} cy={sy(y)} r="4" fill={s.color} stroke="#fff" strokeWidth="2" />
              ))}
            </g>
          ))}

          {hoverX !== null && (
            <g pointerEvents="none">
              <line x1={sx(hoverX)} x2={sx(hoverX)} y1={sy(0)} y2={sy(max)} stroke="#52514e" strokeWidth="1" />
              {series.map((s) => {
                const y = yAt(s.points, hoverX);
                return y === null ? null : (
                  <circle key={s.key} cx={sx(hoverX)} cy={sy(y)} r="4" fill={s.color} stroke="#fff" strokeWidth="2" />
                );
              })}
            </g>
          )}
        </svg>

        {hoverX !== null && (
          <div className="absolute top-2 left-14 bg-white/95 border rounded-md shadow px-2 py-1 text-xs pointer-events-none">
            <p className="font-semibold">{xLabel}: {format(hoverX)}</p>
            {series.map((s) => {
              const y = yAt(s.points, hoverX);
              return y === null ? null : (
                <p key={s.key}>
                  <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ background: s.color }} />
                  {s.name}: {format(y)}
                </p>
              );
            })}
          </div>
        )}
      </div>

      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 mt-1 text-xs text-gray-700">
        {series.map((s) => (
          <span key={s.key}>
            <span className="inline-block w-4 h-0.5 align-middle mr-1" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
        {referenceLabel && (
          <span><span className="inline-block w-4 border-t-2 border-dashed border-gray-400 align-middle mr-1" />{referenceLabel}</span>
        )}
      </figcaption>
    </figure>
  );
}
