import { MODEL_LABELS_BY_CODE } from '../../data/inference-models';
import type { AdjustedBootstrapComparison } from '../../types/dashboard';

interface BootstrapIntervalChartProps {
  comparisons: AdjustedBootstrapComparison[];
  title: string;
  note: string;
  axisLabel: string;
  showRandomVariant?: boolean;
}

export function BootstrapIntervalChart({
  comparisons,
  title,
  note,
  axisLabel,
  showRandomVariant = false,
}: BootstrapIntervalChartProps) {
  const modelName = (code: string) =>
    showRandomVariant && code === 'E2'
      ? 'TextCNN-Random'
      : MODEL_LABELS_BY_CODE.get(code)?.name ?? code;
  const rows = comparisons.map((comparison) => ({
    label: `${modelName(comparison.model_a)} − ${modelName(comparison.model_b)}${comparison.originally_planned ? '' : ' †'}`,
    diff: comparison.difference,
    lo: comparison.ci_low,
    hi: comparison.ci_high,
  }));

  const width = 920;
  const rowH = 34;
  const pad = { top: 18, right: 192, bottom: 32, left: 245 };
  const height = pad.top + pad.bottom + rows.length * rowH;
  const plotW = width - pad.left - pad.right;

  const allVals = [0, ...rows.flatMap((r) => [r.lo, r.diff, r.hi])];
  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  const span = Math.max(0.02, rawMax - rawMin);
  const xMin = rawMin - span * 0.1;
  const xMax = rawMax + span * 0.1;
  const X = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * plotW;

  const leftPx = X(0) - X(xMin);
  const rightPx = X(xMax) - X(0);
  // Center edge labels on their ticks so the gap to zero is shared by both labels.
  const ticks: number[] = [];
  if (leftPx >= 50) {
    ticks.push(Number(xMin.toFixed(3)));
  }
  if (leftPx >= 130) {
    ticks.push(Number((xMin / 2).toFixed(3)));
  }
  ticks.push(0);
  if (rightPx >= 130) {
    ticks.push(Number((xMax / 2).toFixed(3)));
  }
  if (rightPx >= 50) {
    ticks.push(Number(xMax.toFixed(3)));
  }

  return (
    <figure className="figure-box">
      <figcaption>
        <span className="figure-title">{title}</span>
        <span className="figure-note mono">{note}</span>
      </figcaption>
      <div className="capability-plot-scroll">
        <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
          {ticks.map((value, idx) => {
            const tickX = Math.max(pad.left, Math.min(width - pad.right, X(value)));
            const label = `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
            return (
              <g key={`${value}-${idx}`}>
                <line
                  x1={tickX}
                  x2={tickX}
                  y1={pad.top}
                  y2={height - pad.bottom}
                  stroke={value === 0 ? 'var(--chart-zero)' : 'var(--chart-grid)'}
                  strokeDasharray={value === 0 ? '4 3' : undefined}
                  strokeWidth={value === 0 ? 1.3 : 1}
                />
                <text
                  x={tickX}
                  y={height - pad.bottom + 14}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--muted)"
                  fontFamily="var(--mono)"
                >
                  {label}
                </text>
              </g>
            );
          })}

          <rect
            x={pad.left}
            y={pad.top}
            width={plotW}
            height={height - pad.top - pad.bottom}
            fill="none"
            stroke="var(--chart-border)"
          />

          {rows.map((row, idx) => {
            const y = pad.top + idx * rowH + rowH / 2;
            const crossesZero = row.lo <= 0 && row.hi >= 0;
            const color = crossesZero ? 'var(--muted)' : row.diff > 0 ? 'var(--pos)' : 'var(--neg)';
            return (
              <g key={row.label}>
                <line
                  x1={pad.left}
                  x2={width - pad.right}
                  y1={y}
                  y2={y}
                  stroke="var(--chart-row)"
                  strokeWidth={1}
                />
                <text
                  x={pad.left - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize={11}
                  fill="var(--text)"
                >
                  {row.label}
                </text>
                <line x1={X(row.lo)} x2={X(row.hi)} y1={y} y2={y} stroke={color} strokeWidth={2.2} />
                <line x1={X(row.lo)} x2={X(row.lo)} y1={y - 4.5} y2={y + 4.5} stroke={color} strokeWidth={1.6} />
                <line x1={X(row.hi)} x2={X(row.hi)} y1={y - 4.5} y2={y + 4.5} stroke={color} strokeWidth={1.6} />
                <circle cx={X(row.diff)} cy={y} r={3.8} fill={color} />
                <text
                  x={width - pad.right + 8}
                  y={y + 3.5}
                  fontSize={10}
                  fill="var(--text)"
                  fontFamily="var(--mono)"
                >
                  {`${row.diff >= 0 ? '+' : ''}${row.diff.toFixed(4)} [${row.lo.toFixed(4)}, ${row.hi.toFixed(4)}]`}
                </text>
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
            {axisLabel}
          </text>
        </svg>
      </div>
    </figure>
  );
}
