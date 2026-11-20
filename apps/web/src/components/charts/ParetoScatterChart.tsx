import { MODEL_LABELS_BY_CODE } from '../../data/inference-models';
import type { LeaderboardEntry } from '../../types/dashboard';

export interface TradeoffPoint {
  entry: LeaderboardEntry;
  cost: number;
  f1: number;
  detail: string;
}

interface ParetoScatterChartProps {
  points: TradeoffPoint[];
  title: string;
  note: string;
  quadrantLabel: string;
  paretoLabel: string;
  xAxisLabel: string;
  yAxisLabel: string;
  xCandidates: readonly number[];
  formatTick: (cost: number) => string;
}

interface PlotPoint {
  code: string;
  name: string;
  cost: number;
  f1: number;
  paramLabel: string;
  color: string;
  detail: string;
  px: number;
  py: number;
  onFrontier: boolean;
}

interface PlacedLabel {
  x: number;
  y: number;
  anchor: 'start' | 'end' | 'middle';
  boxX: number;
  boxY: number;
  w: number;
  h: number;
}

function pointColor(entry: LeaderboardEntry): string {
  if (entry.batch === 'Batch B · QLoRA') return 'var(--series-qwen)';
  if (entry.batch === 'Batch B · Encoder') return 'var(--series-b3)';
  if (entry.family === 'Linear') return 'var(--series-e1)';
  if (entry.family === 'Recurrent') return 'var(--series-e3)';
  return 'var(--series-e5)';
}

const MODEL_LEGEND = [
  { label: 'Encoder', color: 'var(--series-b3)' },
  { label: 'Qwen QLoRA', color: 'var(--series-qwen)' },
  { label: 'CNN', color: 'var(--series-e5)' },
  { label: 'RNN', color: 'var(--series-e3)' },
  { label: 'LinearSVC', color: 'var(--series-e1)' },
];

function rectsOverlap(
  a: { x: number; y: number; w: number; h: number },
  b: { x: number; y: number; w: number; h: number },
  padX = 8,
  padY = 5
): boolean {
  return !(
    a.x + a.w + padX <= b.x ||
    b.x + b.w + padX <= a.x ||
    a.y + a.h + padY <= b.y ||
    b.y + b.h + padY <= a.y
  );
}

function segmentIntersectsBox(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  box: { x: number; y: number; w: number; h: number },
  margin = 4
): boolean {
  const left = box.x - margin;
  const right = box.x + box.w + margin;
  const top = box.y - margin;
  const bottom = box.y + box.h + margin;
  const dx = x2 - x1;
  const dy = y2 - y1;
  let t0 = 0;
  let t1 = 1;
  const checks: Array<[number, number]> = [
    [-dx, x1 - left],
    [dx, right - x1],
    [-dy, y1 - top],
    [dy, bottom - y1],
  ];
  for (const [p, q] of checks) {
    if (Math.abs(p) < 1e-9) {
      if (q < 0) return false;
    } else {
      const r = q / p;
      if (p < 0) {
        if (r > t1) return false;
        if (r > t0) t0 = r;
      } else {
        if (r < t0) return false;
        if (r < t1) t1 = r;
      }
    }
  }
  return t0 <= t1;
}

export function ParetoScatterChart({
  points,
  title,
  note,
  quadrantLabel,
  paretoLabel,
  xAxisLabel,
  yAxisLabel,
  xCandidates,
  formatTick,
}: ParetoScatterChartProps) {
  const width = 960;
  const height = 390;
  const pad = { top: 28, right: 48, bottom: 48, left: 62 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const rawPoints = points.map(({ entry: e, cost, f1, detail }) => ({
    code: e.model_code,
    name: MODEL_LABELS_BY_CODE.get(e.model_code)?.name ?? e.display_name,
    cost,
    f1,
    paramLabel: e.param_label ?? '',
    color: pointColor(e),
    detail,
  }));

  // Log10 X bounds & Linear Y bounds
  const logVals = rawPoints.map((p) => Math.log10(p.cost));
  const f1Vals = rawPoints.map((p) => p.f1);
  const logMin = (logVals.length ? Math.min(...logVals) : -1.5) - 0.25;
  const logMax = (logVals.length ? Math.max(...logVals) : 1.2) + 0.3;

  const minF1 = f1Vals.length ? Math.min(...f1Vals) : 0.2;
  const maxF1 = f1Vals.length ? Math.max(...f1Vals) : 0.95;
  const ySpan = Math.max(0.06, maxF1 - minF1);
  const yMin = Math.max(0, Math.floor((minF1 - ySpan * 0.14) * 20) / 20);
  const yMax = Math.min(1, Math.ceil((maxF1 + ySpan * 0.14) * 20) / 20);

  const X = (cost: number) =>
    pad.left + ((Math.log10(cost) - logMin) / Math.max(1e-6, logMax - logMin)) * plotW;
  const Y = (v: number) => pad.top + (1 - (v - yMin) / Math.max(1e-6, yMax - yMin)) * plotH;

  // Sort by increasing cost and keep each improvement in Macro-F1.
  const sortedByCost = [...rawPoints].sort((a, b) => a.cost - b.cost || b.f1 - a.f1);
  let bestSoFar = -Infinity;
  const frontierCodes = new Set<string>();
  const frontierPoints: Array<{ px: number; py: number; code: string }> = [];
  for (const pt of sortedByCost) {
    if (pt.f1 > bestSoFar + 1e-9) {
      bestSoFar = pt.f1;
      frontierCodes.add(pt.code);
      frontierPoints.push({ px: X(pt.cost), py: Y(pt.f1), code: pt.code });
    }
  }

  const plotPoints: PlotPoint[] = rawPoints.map((pt) => ({
    ...pt,
    px: X(pt.cost),
    py: Y(pt.f1),
    onFrontier: frontierCodes.has(pt.code),
  }));

  // Split the drawn area in half on both axes, including the logarithmic X axis.
  const quadMidX = pad.left + plotW / 2;
  const quadMidY = pad.top + plotH / 2;
  const activeColors = new Set(plotPoints.map(({ color }) => color));

  // Clean candidate X ticks (log scale)
  const xTicks = xCandidates.filter((v) => Math.log10(v) >= logMin && Math.log10(v) <= logMax);
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => yMin + (yMax - yMin) * f);

  // Greedy collision-free label placement (avoids other labels, points, and Pareto frontier segments)
  const placedBoxes: PlacedLabel[] = [];
  const labelPositions = new Map<string, PlacedLabel>();
  const sortedForLabels = [...plotPoints].sort((a, b) => b.f1 - a.f1);
  for (const pt of sortedForLabels) {
    const w = Math.max(34, pt.name.length * 6.25 + 8);
    const h = 15;
    const candidates: Array<{ dx: number; dy: number; anchor: 'start' | 'end' | 'middle' }> = [
      { dx: 0, dy: -12, anchor: 'middle' },
      { dx: 11, dy: 4, anchor: 'start' },
      { dx: -11, dy: 4, anchor: 'end' },
      { dx: 0, dy: 19, anchor: 'middle' },
      { dx: 12, dy: -11, anchor: 'start' },
      { dx: -12, dy: -11, anchor: 'end' },
      { dx: 12, dy: 17, anchor: 'start' },
      { dx: -12, dy: 17, anchor: 'end' },
      { dx: 16, dy: -18, anchor: 'start' },
      { dx: -16, dy: -18, anchor: 'end' },
      { dx: 16, dy: 24, anchor: 'start' },
      { dx: -16, dy: 24, anchor: 'end' },
      { dx: 0, dy: -26, anchor: 'middle' },
      { dx: 0, dy: 32, anchor: 'middle' },
      { dx: 20, dy: 4, anchor: 'start' },
      { dx: -20, dy: 4, anchor: 'end' },
    ];

    let chosen: PlacedLabel | null = null;
    let secondaryChosen: PlacedLabel | null = null;

    for (const c of candidates) {
      const tx = pt.px + c.dx;
      const ty = pt.py + c.dy;
      const boxLeft = c.anchor === 'start' ? tx : c.anchor === 'end' ? tx - w : tx - w / 2;
      const boxTop = ty - h + 2;
      const box = { x: boxLeft, y: boxTop, w, h };
      const inBounds =
        boxLeft >= pad.left + 2 &&
        boxLeft + w <= width - pad.right - 2 &&
        boxTop >= pad.top + 2 &&
        boxTop + h <= pad.top + plotH - 2;
      if (!inBounds) continue;

      const hitsPoint = plotPoints.some(
        (other) =>
          other.code !== pt.code &&
          Math.abs(other.px - Math.max(boxLeft, Math.min(other.px, boxLeft + w))) < 9 &&
          Math.abs(other.py - Math.max(boxTop, Math.min(other.py, boxTop + h))) < 9
      );
      if (hitsPoint) continue;

      const overlapsLabel = placedBoxes.some((b) =>
        rectsOverlap(box, { x: b.boxX, y: b.boxY, w: b.w, h: b.h })
      );
      if (overlapsLabel) continue;

      const hitsFrontier = frontierPoints.some((fp, idx) => {
        if (idx === 0) return false;
        const prev = frontierPoints[idx - 1];
        return segmentIntersectsBox(prev.px, prev.py, fp.px, fp.py, box, 4);
      });

      const candidateLabel: PlacedLabel = {
        x: tx,
        y: ty,
        anchor: c.anchor,
        boxX: boxLeft,
        boxY: boxTop,
        w,
        h,
      };

      if (!hitsFrontier) {
        chosen = candidateLabel;
        break;
      }
      if (!secondaryChosen) {
        secondaryChosen = candidateLabel;
      }
    }

    if (!chosen) {
      chosen = secondaryChosen;
    }

    if (!chosen) {
      const fx = Math.min(width - pad.right - w / 2 - 4, Math.max(pad.left + w / 2 + 4, pt.px));
      const fy = Math.max(pad.top + 14, pt.py - 12);
      chosen = {
        x: fx,
        y: fy,
        anchor: 'middle',
        boxX: fx - w / 2,
        boxY: fy - h + 2,
        w,
        h,
      };
    }

    placedBoxes.push(chosen);
    labelPositions.set(pt.code, chosen);
  }

  const frontierPath = frontierPoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.px.toFixed(1)} ${p.py.toFixed(1)}`)
    .join(' ');

  return (
    <figure className="figure-box">
      <figcaption>
        <span className="figure-title">{title}</span>
        <span className="figure-note mono">{note}</span>
      </figcaption>

      <div className="aa-legend">
        <span className="aa-legend-item">
          <span className="aa-swatch-box" />
          <span>{quadrantLabel}</span>
        </span>
        <span className="aa-legend-item">
          <span className="aa-swatch-line" />
          <span>{paretoLabel}</span>
        </span>
        {MODEL_LEGEND.filter(({ color }) => activeColors.has(color)).map(({ label, color }) => (
          <span className="aa-legend-item" key={label}>
            <span className="aa-swatch-dot" style={{ background: color }} />
            <span>{label}</span>
          </span>
        ))}
      </div>

      <div className="pareto-plot-scroll">
        <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
          {/* Quadrant Shaded Zones */}
          <rect
            x={pad.left}
            y={pad.top}
            width={plotW / 2}
            height={plotH / 2}
            fill="var(--pareto-good-bg)"
          />
          <rect
            x={quadMidX}
            y={quadMidY}
            width={plotW / 2}
            height={plotH / 2}
            fill="var(--pareto-muted-bg)"
          />

          {/* Grid & Ticks */}
          {yTicks.map((tick) => {
            const y = Y(tick);
            return (
              <g key={tick}>
                <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="var(--chart-grid)" />
                <text
                  x={pad.left - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize={10.5}
                  fill="var(--muted)"
                  fontFamily="var(--mono)"
                >
                  {tick.toFixed(2)}
                </text>
              </g>
            );
          })}

          {xTicks.map((tick) => {
            const x = X(tick);
            return (
              <g key={tick}>
                <line x1={x} x2={x} y1={pad.top} y2={pad.top + plotH} stroke="var(--chart-grid)" />
                <text
                  x={x}
                  y={pad.top + plotH + 16}
                  textAnchor="middle"
                  fontSize={10.5}
                  fill="var(--muted)"
                  fontFamily="var(--mono)"
                >
                  {formatTick(tick)}
                </text>
              </g>
            );
          })}

          <path
            d={`M ${quadMidX} ${pad.top} V ${pad.top + plotH} M ${pad.left} ${quadMidY} H ${pad.left + plotW}`}
            fill="none"
            stroke="var(--chart-border)"
            strokeWidth={0.8}
          />
          <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="none" stroke="var(--chart-border)" />

          {/* Dotted Pareto Frontier */}
          {frontierPoints.length >= 2 && (
            <path
              d={frontierPath}
              fill="none"
              stroke="var(--pareto-frontier)"
              strokeWidth={1.3}
              strokeDasharray="0.8 4"
              strokeLinecap="round"
              opacity={0.85}
            />
          )}

          {/* Data Points & Non-Overlapping Labels */}
          {plotPoints.map((pt) => {
            const pos = labelPositions.get(pt.code)!;
            return (
              <g key={pt.code}>
                <circle
                  cx={pt.px}
                  cy={pt.py}
                  r={pt.onFrontier ? 4 : 3.3}
                  fill={pt.color}
                  stroke="var(--surface)"
                  strokeWidth={0.8}
                >
                  <title>
                    {`${pt.name} · Macro-F1 ${pt.f1.toFixed(4)} · ${pt.detail}${pt.paramLabel ? ` · ${pt.paramLabel}` : ''}`}
                  </title>
                </circle>
                <text
                  x={pos.x}
                  y={pos.y}
                  textAnchor={pos.anchor}
                  fontSize={10.5}
                  fontWeight={pt.onFrontier ? 600 : 400}
                  fill={pt.onFrontier ? 'var(--text)' : 'var(--muted)'}
                  stroke="var(--surface)"
                  strokeWidth={3}
                  strokeLinejoin="round"
                  paintOrder="stroke"
                >
                  {pt.name}
                </text>
              </g>
            );
          })}

          {/* Axis Titles */}
          <text
            x={pad.left + plotW / 2}
            y={height - 8}
            textAnchor="middle"
            fontSize={11}
            fill="var(--muted)"
          >
            {xAxisLabel}
          </text>
          <text
            x={16}
            y={pad.top + plotH / 2}
            transform={`rotate(-90 16 ${pad.top + plotH / 2})`}
            textAnchor="middle"
            fontSize={11}
            fill="var(--muted)"
          >
            {yAxisLabel}
          </text>
        </svg>
      </div>
    </figure>
  );
}
