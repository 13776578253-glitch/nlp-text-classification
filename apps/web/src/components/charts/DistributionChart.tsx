import type { LabelSupportRow } from '../../types/dashboard';

export type DistSortOrder = 'freq_desc' | 'freq_asc' | 'alpha';

interface DistributionChartProps {
  rows: LabelSupportRow[];
  scale: 'log' | 'linear';
  sortOrder?: DistSortOrder;
  title: string;
  note?: string;
}

export function DistributionChart({
  rows,
  scale,
  sortOrder = 'freq_desc',
  title,
  note,
}: DistributionChartProps) {
  const sorted = [...rows].sort((a, b) => {
    if (sortOrder === 'freq_asc') return a.v1_train - b.v1_train;
    if (sortOrder === 'alpha') return a.name.localeCompare(b.name);
    return b.v1_train - a.v1_train;
  });

  const width = 920;
  const height = 326;
  const pad = { top: 20, right: 16, bottom: 108, left: 56 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const maxVal = Math.max(...sorted.map((r) => r.v1_train), 1);
  const Y = (val: number) => {
    if (scale === 'log') {
      const lo = Math.log10(10);
      const hi = Math.log10(Math.max(20000, maxVal * 1.15));
      const v = Math.log10(Math.max(10, val));
      return pad.top + (1 - (v - lo) / (hi - lo)) * plotH;
    }
    return pad.top + (1 - val / (maxVal * 1.05)) * plotH;
  };

  const yTicks = scale === 'log' ? [50, 200, 1000, 5000, 15000] : [0, 4000, 8000, 12000, 15000];
  const barStep = plotW / sorted.length;
  const barW = Math.max(8, barStep * 0.68);

  return (
    <figure className="figure-box">
      <figcaption>
        <span className="figure-title">{title}</span>
        <span className="figure-note mono">
          {note ?? 'Train support bars (<400 rare amber) · Dot = Test ×8'}
        </span>
      </figcaption>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {yTicks.map((tick) => {
          const y = Y(tick);
          return (
            <g key={tick}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="var(--chart-grid)" />
              <text
                x={pad.left - 8}
                y={y + 3.5}
                textAnchor="end"
                fontSize={10}
                fill="var(--muted)"
                fontFamily="var(--mono)"
              >
                {tick.toLocaleString('en-US')}
              </text>
            </g>
          );
        })}

        <rect
          x={pad.left}
          y={pad.top}
          width={plotW}
          height={plotH}
          fill="none"
          stroke="var(--chart-border)"
        />

        {sorted.map((row, idx) => {
          const cx = pad.left + idx * barStep + barStep / 2;
          const barY = Y(row.v1_train);
          const barH = Math.max(1, pad.top + plotH - barY);
          const testScaledY = Y(row.test * 8);
          const isRare = row.v1_train < 400;
          return (
            <g key={row.id}>
              <rect
                x={cx - barW / 2}
                y={barY}
                width={barW}
                height={barH}
                rx={2}
                fill={isRare ? 'var(--series-qwen)' : 'var(--series-b3)'}
                opacity={0.86}
              >
                <title>{`${row.name}: train ${row.v1_train}, val ${row.v1_val}, test ${row.test}`}</title>
              </rect>
              <circle cx={cx} cy={testScaledY} r={2.3} fill="var(--text)" />
              <text
                x={cx}
                y={pad.top + plotH + 10}
                transform={`rotate(-42 ${cx} ${pad.top + plotH + 10})`}
                textAnchor="end"
                fontSize={10}
                fill={isRare ? 'var(--rare-text)' : 'var(--text)'}
                fontWeight={isRare ? 600 : 400}
              >
                {`${row.name} (${row.v1_train})`}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
