import { MODEL_LABELS_BY_CODE } from '../../data/inference-models';
import type { LeaderboardEntry } from '../../types/dashboard';

interface ErrorBarChartProps {
  entries: LeaderboardEntry[];
  title: string;
  note: string;
}

const BATCH_COLOR: Record<string, string> = {
  'Batch A': 'var(--series-e1)',
  'Batch B · Encoder': 'var(--series-b3)',
  'Batch B · QLoRA': 'var(--series-qwen)',
};

export function ErrorBarChart({ entries, title, note }: ErrorBarChartProps) {
  const rows = entries.map((entry) => {
    const stats = entry.three_seed?.test_macro_f1;
    return {
      id: entry.model_id,
      name: MODEL_LABELS_BY_CODE.get(entry.model_code)?.name ?? entry.display_name,
      score: stats?.mean ?? entry.seed42.test_macro_f1,
      sd: stats?.sd,
      seeds: Object.entries(entry.three_seed?.by_seed ?? { 42: entry.seed42.test_macro_f1 }),
      color: BATCH_COLOR[entry.batch] ?? 'var(--series-b3)',
    };
  });
  const width = 860;
  const rowH = 23;
  const pad = { top: 8, right: 140, bottom: 23, left: 166 };
  const height = pad.top + pad.bottom + Math.max(1, rows.length) * rowH;
  const plotW = width - pad.left - pad.right;

  const vals = rows.flatMap(({ score, sd = 0, seeds }) => [
    score - sd, score + sd, ...seeds.map(([, score]) => score),
  ]);
  const rawMin = vals.length ? Math.min(...vals) : 0;
  const rawMax = vals.length ? Math.max(...vals) : 1;
  const span = Math.max(0.05, rawMax - rawMin);
  const xMin = Math.max(0, Math.floor((rawMin - span * 0.14) * 20) / 20);
  const xMax = Math.min(1, Math.ceil((rawMax + span * 0.14) * 20) / 20);
  const X = (v: number) => pad.left + ((v - xMin) / Math.max(1e-6, xMax - xMin)) * plotW;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => xMin + (xMax - xMin) * f);

  return (
    <figure className="figure-box">
      <figcaption>
        <span className="figure-title">{title}</span>
        <span className="figure-note mono">{note}</span>
      </figcaption>
      <div className="capability-plot-scroll">
        <svg
          className="chart-svg"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={title}
        >
          {ticks.map((tick) => {
            const x = X(tick);
            return (
              <g key={tick}>
                <line
                  x1={x}
                  x2={x}
                  y1={pad.top}
                  y2={height - pad.bottom}
                  stroke="var(--chart-grid)"
                  strokeWidth={0.75}
                />
                <text
                  x={x}
                  y={height - pad.bottom + 14}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--muted)"
                  fontFamily="var(--mono)"
                >
                  {tick.toFixed(2)}
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
            const lo = Math.max(xMin, row.score - (row.sd ?? 0));
            const hi = Math.min(xMax, row.score + (row.sd ?? 0));
            const value = row.sd === undefined
              ? `${row.score.toFixed(4)} (n=1)`
              : `${row.score.toFixed(4)} ± ${row.sd.toFixed(4)}`;
            const seedSummary = row.seeds
              .map(([s, v]) => `seed ${s}: ${v.toFixed(4)}`)
              .join(' · ');

            return (
              <g key={row.id}>
                <line
                  x1={pad.left}
                  x2={width - pad.right}
                  y1={y}
                  y2={y}
                  stroke="var(--chart-border)"
                  strokeWidth={0.7}
                  strokeDasharray="2 3"
                />
                <text
                  x={pad.left - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize={11}
                  fill="var(--text)"
                  fontWeight={500}
                >
                  {row.name}
                </text>
                {hi > lo && (
                  <path
                    d={`M ${X(lo)} ${y} H ${X(hi)} M ${X(lo)} ${y - 3.5} v 7 M ${X(hi)} ${y - 3.5} v 7`}
                    fill="none"
                    stroke={row.color}
                    strokeWidth={1.25}
                  />
                )}
                <circle
                  cx={X(row.score)}
                  cy={y}
                  r={3}
                  fill={row.sd === undefined ? 'var(--surface)' : row.color}
                  stroke={row.sd === undefined ? row.color : 'var(--surface)'}
                  strokeWidth={row.sd === undefined ? 1.25 : 0.7}
                >
                  <title>{`${row.name} · ${value} · ${seedSummary}`}</title>
                </circle>
                <text
                  x={width - pad.right + 10}
                  y={y + 3.5}
                  fontSize={10.5}
                  fill="var(--text)"
                  fontFamily="var(--mono)"
                >
                  {value}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </figure>
  );
}
