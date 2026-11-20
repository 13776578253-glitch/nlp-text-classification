import { MODEL_LABELS_BY_CODE } from '../data/inference-models';
import { useI18n } from '../i18n/I18nContext';
import type { DatasetId, LeaderboardEntry } from '../types/dashboard';
import { ParetoScatterChart, type TradeoffPoint } from './charts/ParetoScatterChart';
import { SegmentedPill } from './SegmentedPill';
import { SortableTable, type TableColumn } from './SortableTable';

type TradeoffView = 'chart' | 'table';

interface TradeoffPanelProps {
  mode: 'inference' | 'training';
  entries: LeaderboardEntry[];
  runSeconds: ReadonlyMap<string, number>;
  dataset: DatasetId;
  onDatasetChange: (dataset: DatasetId) => void;
  view: TradeoffView;
  onViewChange: (view: TradeoffView) => void;
}

const INFERENCE_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20];
const TRAINING_TICKS = [2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200];

function durationLabel(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds % 1 === 0 ? 0 : 1)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    const remainder = Math.floor(seconds % 60);
    return remainder ? `${minutes} min ${remainder} s` : `${minutes} min`;
  }
  const remainder = minutes % 60;
  return remainder ? `${Math.floor(minutes / 60)} h ${remainder} min` : `${minutes / 60} h`;
}

function modelName({ entry }: TradeoffPoint): string {
  return MODEL_LABELS_BY_CODE.get(entry.model_code)?.name ?? entry.display_name;
}

export function TradeoffPanel({
  mode, entries, runSeconds, dataset, onDatasetChange, view, onViewChange,
}: TradeoffPanelProps) {
  const { t } = useI18n();
  const training = mode === 'training';
  const title = training
    ? `${dataset === 'goemotions' ? 'GoEmotions' : t('datasetZhName')} · ${t('trainingTradeoffTitle')}`
    : t(dataset === 'goemotions' ? 'paretoTitleEn' : 'paretoTitleZh');
  const points = entries.flatMap((entry): TradeoffPoint[] => {
    if (training) {
      const cost = runSeconds.get(entry.run_id);
      if (cost === undefined || !Number.isFinite(cost) || cost <= 0) return [];
      return [{ entry, cost, f1: entry.seed42.test_macro_f1, detail: `${cost.toFixed(2)} s · Seed 42` }];
    }
    const benchmark = entry.inference_benchmark;
    if (!benchmark || !Number.isFinite(benchmark.ms_per_sample) || benchmark.ms_per_sample <= 0) {
      return [];
    }
    return [{
      entry,
      cost: benchmark.ms_per_sample,
      f1: entry.three_seed?.test_macro_f1.mean ?? entry.seed42.test_macro_f1,
      detail: `${benchmark.ms_per_sample.toFixed(3)} ms/sample · ${benchmark.device}`,
    }];
  });

  const columns: TableColumn<TradeoffPoint>[] = [
    { key: 'model', label: t('colModel'), isRowHeader: true, sortValue: modelName, render: modelName },
    {
      key: 'cost', label: t(training ? 'trainingDuration' : 'colLatency'), align: 'right',
      type: 'number', sortValue: (row) => row.cost,
      render: ({ cost }) => (
        <span className="mono" title={training ? `${cost.toFixed(2)} s` : undefined}>
          {training ? durationLabel(cost) : `${cost.toFixed(3)} ms`}
        </span>
      ),
    },
    {
      key: 'f1', label: 'Test Macro-F1', align: 'right', type: 'number',
      sortValue: (row) => row.f1, render: (row) => <strong>{row.f1.toFixed(4)}</strong>,
    },
  ];

  return (
    <div className="stage-card stable-card pareto-card">
      <div className="stage-bar">
        <SegmentedPill
          size="sm"
          ariaLabel={t('colDataset')}
          options={[
            { id: 'goemotions', label: t('datasetEnFull') },
            { id: 'movie_reviews_zh', label: t('datasetZhFull') },
          ]}
          activeId={dataset}
          onChange={onDatasetChange}
        />
        <SegmentedPill
          size="sm"
          ariaLabel={t('tradeoffViewLabel')}
          options={[
            { id: 'chart', label: t('tradeoffViewChart') },
            { id: 'table', label: t('tradeoffViewTable') },
          ]}
          activeId={view}
          onChange={onViewChange}
        />
      </div>

      <div className="stage-card-body">
        {view === 'chart' ? (
          <ParetoScatterChart
            points={points}
            title={title}
            note={t(training ? 'trainingTradeoffNote' : 'paretoNote')}
            quadrantLabel={t('paretoQuadrant')}
            paretoLabel={t('paretoFrontierLabel')}
            xAxisLabel={t(training ? 'trainingTradeoffXAxis' : 'paretoXAxis')}
            yAxisLabel={t('paretoYAxis')}
            xCandidates={training ? TRAINING_TICKS : INFERENCE_TICKS}
            formatTick={training ? durationLabel : String}
          />
        ) : (
          <div className="tradeoff-table">
            <SortableTable
              caption={title}
              hideCaption
              columns={columns}
              rows={points}
              getRowKey={(row) => row.entry.model_id}
              defaultSortKey="f1"
              defaultDirection="descending"
              responsiveStack={false}
            />
          </div>
        )}
      </div>

      <div className="stage-footnote">
        <span>{t(training ? 'trainingTradeoffFoot' : 'inferenceTradeoffFoot')}</span>
        <span className="mono">{t('tradeoffModelCount').replace('{count}', String(points.length))}</span>
      </div>
    </div>
  );
}
