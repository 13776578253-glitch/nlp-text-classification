import { modelShortName } from '../../data/inference-models';
import type { PerLabelGoEmotionsRow } from '../../types/dashboard';

interface LongTailF1ChartProps {
  rows: PerLabelGoEmotionsRow[];
  selectedModels: string[];
  title: string;
}

export const LONG_TAIL_COLORS: Record<string, string> = {
  E1: 'var(--series-e1)',
  E2: 'var(--series-e2)',
  'E3-RNN': 'var(--series-e3r)',
  'E3-LSTM': 'var(--series-e3l)',
  'E3-BiLSTM': 'var(--series-e3)',
  'E5-TUNE': 'var(--series-e5)',
  'E5-FROZEN': 'var(--series-e5f)',
  B1: 'var(--series-b1)',
  B2: 'var(--series-b2)',
  B3: 'var(--series-b3)',
  'Qwen3.5-4B': 'var(--series-qwen)',
};

export function LongTailF1Chart({ rows, selectedModels, title }: LongTailF1ChartProps) {
  const width = 980;
  const height = 356;
  const pad = { top: 22, right: 20, bottom: 130, left: 56 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const step = plotW / Math.max(1, rows.length);
  const X = (idx: number) => pad.left + idx * step + step / 2;
  const Y = (v: number) => pad.top + (1 - Math.max(0, Math.min(1, v))) * plotH;

  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <figure className="figure-box">
      <figcaption>
        <span className="figure-title">{title}</span>
        <span className="figure-note" style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {selectedModels.map((m) => (
            <span key={m} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
              <span
                style={{
                  width: 11,
                  height: 3,
                  borderRadius: 2,
                  background: LONG_TAIL_COLORS[m] ?? 'var(--series-b3)',
                  display: 'inline-block',
                }}
              />
              <span className="mono">{modelShortName(m)}</span>
            </span>
          ))}
        </span>
      </figcaption>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line x1={pad.left} x2={width - pad.right} y1={Y(tick)} y2={Y(tick)} stroke="var(--chart-grid)" />
            <text
              x={pad.left - 8}
              y={Y(tick) + 3.5}
              textAnchor="end"
              fontSize={10}
              fill="var(--muted)"
              fontFamily="var(--mono)"
            >
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="none" stroke="var(--chart-border)" />

        {rows.map((row, idx) => (
          <g key={row.id}>
            {row.train_support < 400 && (
              <rect
                x={pad.left + idx * step}
                y={pad.top}
                width={step}
                height={plotH}
                fill="var(--rare-band)"
              />
            )}
            <text
              x={X(idx)}
              y={pad.top + plotH + 10}
              transform={`rotate(-45 ${X(idx)} ${pad.top + plotH + 10})`}
              textAnchor="end"
              fontSize={9.8}
              fill={row.train_support < 400 ? 'var(--rare-text)' : 'var(--text)'}
              fontWeight={row.train_support < 400 ? 600 : 400}
            >
              {`${row.name} (${row.test_support})`}
            </text>
          </g>
        ))}

        {selectedModels.map((modelCode) => {
          const color = LONG_TAIL_COLORS[modelCode] ?? 'var(--series-b3)';
          const pathD = rows
            .map((r, idx) => {
              const f1 = r.models[modelCode]?.f1 ?? 0;
              return `${idx === 0 ? 'M' : 'L'} ${X(idx).toFixed(1)} ${Y(f1).toFixed(1)}`;
            })
            .join(' ');
          return (
            <g key={modelCode}>
              <path d={pathD} fill="none" stroke={color} strokeWidth={2} />
              {rows.map((r, idx) => {
                const f1 = r.models[modelCode]?.f1 ?? 0;
                return (
                  <circle key={r.id} cx={X(idx)} cy={Y(f1)} r={2.9} fill={color}>
                    <title>{`${modelShortName(modelCode)} · ${r.name}: F1 ${f1.toFixed(4)} (train ${r.train_support}, test ${r.test_support})`}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
