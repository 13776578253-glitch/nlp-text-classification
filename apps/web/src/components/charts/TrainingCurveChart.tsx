export interface CurveSeries {
  id: string;
  label: string;
  color: string;
  bestEpoch?: number;
  points: Array<{ epoch: number; value: number }>;
}

interface TrainingCurveChartProps {
  title: string;
  note?: string;
  series: CurveSeries[];
  compact?: boolean;
}

export function TrainingCurveChart({
  title,
  note,
  series,
  compact = false,
}: TrainingCurveChartProps) {
  const width = 560;
  const height = compact ? 172 : 240;
  const pad = { top: 16, right: 18, bottom: 32, left: 52 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const allPoints = series.flatMap((s) => s.points);
  const maxEpoch = Math.max(2, ...allPoints.map((p) => p.epoch));
  const minEpoch = Math.min(1, ...allPoints.map((p) => p.epoch));
  const vals = allPoints.map((p) => p.value).filter((v) => Number.isFinite(v));
  const rawMin = vals.length ? Math.min(...vals) : 0;
  const rawMax = vals.length ? Math.max(...vals) : 1;
  const span = Math.max(0.02, rawMax - rawMin);
  const yMin = Math.max(0, rawMin - span * 0.08);
  const yMax = rawMax + span * 0.08;

  const X = (ep: number) =>
    pad.left + ((ep - minEpoch) / Math.max(1, maxEpoch - minEpoch)) * plotW;
  const Y = (val: number) =>
    pad.top + (1 - (val - yMin) / Math.max(1e-6, yMax - yMin)) * plotH;

  const yTicks = (compact ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map(
    (f) => yMin + (yMax - yMin) * f
  );
  const xStep = maxEpoch <= 8 ? 1 : maxEpoch <= 16 ? 2 : 4;
  const xTicks: number[] = [];
  for (let ep = minEpoch; ep <= maxEpoch; ep += xStep) xTicks.push(ep);
  if (xTicks[xTicks.length - 1] !== maxEpoch) xTicks.push(maxEpoch);

  return (
    <figure className="figure-box">
      <figcaption>
        <span className="figure-title">{title}</span>
        {note && <span className="figure-note mono">{note}</span>}
      </figcaption>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={note ? `${title} · ${note}` : title}>
        {yTicks.map((tick, idx) => (
          <g key={idx}>
            <line x1={pad.left} x2={width - pad.right} y1={Y(tick)} y2={Y(tick)} stroke="var(--chart-grid)" />
            <text
              x={pad.left - 8}
              y={Y(tick) + 3.5}
              textAnchor="end"
              fontSize={10}
              fill="var(--muted)"
              fontFamily="var(--mono)"
            >
              {tick.toFixed(3)}
            </text>
          </g>
        ))}

        {xTicks.map((ep) => (
          <g key={ep}>
            <line x1={X(ep)} x2={X(ep)} y1={pad.top} y2={height - pad.bottom} stroke="var(--chart-row)" />
            <text
              x={X(ep)}
              y={height - pad.bottom + 14}
              textAnchor="middle"
              fontSize={10}
              fill="var(--muted)"
              fontFamily="var(--mono)"
            >
              {ep}
            </text>
          </g>
        ))}

        <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="none" stroke="var(--chart-border)" />

        {series.map((s) => {
          if (!s.points.length) return null;
          const d = s.points
            .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${X(p.epoch).toFixed(1)} ${Y(p.value).toFixed(1)}`)
            .join(' ');
          const bestPt = s.bestEpoch !== undefined ? s.points.find((p) => p.epoch === s.bestEpoch) : undefined;
          return (
            <g key={s.id}>
              {bestPt && (
                <line
                  x1={X(bestPt.epoch)}
                  x2={X(bestPt.epoch)}
                  y1={pad.top}
                  y2={height - pad.bottom}
                  stroke={s.color}
                  strokeDasharray="3 3"
                  opacity={0.55}
                />
              )}
              <path
                d={d}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
              />
              {s.points.map((p) => (
                <circle
                  key={p.epoch}
                  cx={X(p.epoch)}
                  cy={Y(p.value)}
                  r={s.bestEpoch === p.epoch ? 4.2 : 2.4}
                  fill={s.bestEpoch === p.epoch ? 'var(--surface)' : s.color}
                  stroke={s.color}
                  strokeWidth={s.bestEpoch === p.epoch ? 2 : 0}
                >
                  <title>{`${s.label} · Epoch ${p.epoch}: ${p.value.toFixed(4)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}

        <text
          x={pad.left + plotW / 2}
          y={height - 4}
          textAnchor="middle"
          fontSize={10}
          fill="var(--muted)"
        >
          Epoch
        </text>
      </svg>
    </figure>
  );
}
