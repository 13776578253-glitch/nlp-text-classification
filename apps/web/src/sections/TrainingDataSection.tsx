import { useMemo, useState } from 'react';
import { useI18n } from '../i18n/I18nContext';
import { SegmentedPill } from '../components/SegmentedPill';
import { DistributionChart, type DistSortOrder } from '../components/charts/DistributionChart';
import { TrainingCurveChart, type CurveSeries } from '../components/charts/TrainingCurveChart';
import { MODEL_LABELS_BY_CODE, modelShortName } from '../data/inference-models';
import type { DashboardBundle, DatasetId, DatasetOverview, TrainingRunCurve } from '../types/dashboard';

interface TrainingDataSectionProps {
  data: DashboardBundle;
}

type DataModule = 'overview' | 'distribution' | 'curves';
type CurveMode = 'single' | 'compare';
type CompareScope = DatasetId | 'all';

const COMPARE_COLORS = ['var(--series-b3)', 'var(--series-qwen)', 'var(--series-b2)'];
const MODEL_ORDER = new Map([...MODEL_LABELS_BY_CODE.keys()].map((code, index) => [code, index]));

function hasEpochHistory(run: TrainingRunCurve) {
  return run.epochs.some(({ epoch, val_macro_f1 }) => epoch > 0 && Number.isFinite(val_macro_f1));
}

function curveSeries(
  run: TrainingRunCurve,
  label: string,
  color: string,
  metric: 'val_macro_f1' | 'loss',
): CurveSeries {
  return {
    id: run.id,
    label,
    color,
    bestEpoch: run.best_epoch ?? undefined,
    points: run.epochs.flatMap(({ epoch, [metric]: value }) =>
      epoch > 0 && typeof value === 'number' && Number.isFinite(value) ? [{ epoch, value }] : []
    ),
  };
}

export function TrainingDataSection({ data }: TrainingDataSectionProps) {
  const { t } = useI18n();
  const [moduleTab, setModuleTab] = useState<DataModule>('overview');

  // Module 2: Sample Distribution
  const [sortOrder, setSortOrder] = useState<DistSortOrder>('freq_desc');
  const [scale, setScale] = useState<'log' | 'linear'>('log');

  // Module 3: Training Convergence
  const runs = useMemo(
    () => [...(data.training_curves.runs ?? [])].sort((a, b) =>
      (MODEL_ORDER.get(a.model) ?? MODEL_ORDER.size) - (MODEL_ORDER.get(b.model) ?? MODEL_ORDER.size)
    ),
    [data.training_curves.runs]
  );
  const [curveMode, setCurveMode] = useState<CurveMode>('single');
  const [singleDataset, setSingleDataset] = useState<DatasetId>('goemotions');
  const [singleModel, setSingleModel] = useState<string>('B3');
  const [singleSeed, setSingleSeed] = useState<number>(42);

  const [compareScope, setCompareScope] = useState<CompareScope>('goemotions');
  const [compareIds, setCompareIds] = useState<string[]>([
    'goemotions:B3:seed-42',
    'goemotions:Qwen3.5-4B:seed-42',
    'goemotions:B1:seed-42',
  ]);

  // Single-run derived options
  const singleDatasetRuns = useMemo(
    () => runs.filter((r) => r.dataset === singleDataset),
    [runs, singleDataset]
  );

  const singleModels = useMemo(
    () => [...new Map(singleDatasetRuns.map((run) => [
      run.model,
      MODEL_LABELS_BY_CODE.get(run.model)?.name ?? run.display_name,
    ] as const))],
    [singleDatasetRuns]
  );

  const effectiveSingleModel = useMemo(
    () =>
      singleModels.some(([model]) => model === singleModel)
        ? singleModel
        : singleModels[0]?.[0] ?? '',
    [singleModels, singleModel]
  );

  const availableSeeds = useMemo(
    () =>
      [...new Set(singleDatasetRuns
        .filter((r) => r.model === effectiveSingleModel)
        .map((r) => r.seed))]
        .sort((a, b) => a - b),
    [singleDatasetRuns, effectiveSingleModel]
  );

  const effectiveSingleSeed = useMemo(
    () => (availableSeeds.includes(singleSeed) ? singleSeed : availableSeeds[0] ?? 42),
    [availableSeeds, singleSeed]
  );

  const activeSingleRun = useMemo(
    () =>
      singleDatasetRuns.find(
        (r) => r.model === effectiveSingleModel && r.seed === effectiveSingleSeed
      ),
    [singleDatasetRuns, effectiveSingleModel, effectiveSingleSeed]
  );

  const handleSingleDatasetChange = (nextDs: DatasetId) => {
    setSingleDataset(nextDs);
    const defaultModel = nextDs === 'goemotions' ? 'B3' : 'B4';
    setSingleModel(defaultModel);
    setSingleSeed(42);
  };

  // Multi-model comparison derived options (max 3 curves)
  const compareCandidates = runs.filter(
    (r) => r.seed === 42 && (compareScope === 'all' || r.dataset === compareScope) && hasEpochHistory(r)
  );

  const handleScopeChange = (nextScope: CompareScope) => {
    setCompareScope(nextScope);
    if (nextScope === 'goemotions') {
      setCompareIds([
        'goemotions:B3:seed-42',
        'goemotions:Qwen3.5-4B:seed-42',
        'goemotions:B1:seed-42',
      ]);
    } else if (nextScope === 'movie_reviews_zh') {
      setCompareIds([
        'movie_reviews_zh:B4:seed-42',
        'movie_reviews_zh:E5-TUNE:seed-42',
      ]);
    } else {
      setCompareIds([
        'goemotions:B3:seed-42',
        'movie_reviews_zh:B4:seed-42',
      ]);
    }
  };

  const toggleCompareRun = (runId: string) => {
    setCompareIds((prev) => {
      if (prev.includes(runId)) {
        return prev.length > 1 ? prev.filter((id) => id !== runId) : prev;
      }
      if (prev.length >= 3) return prev;
      return [...prev, runId];
    });
  };

  const selectedCompareRuns = useMemo(
    () =>
      compareIds
        .map((id, idx) => {
          const run = runs.find((r) => r.id === id);
          return run ? { run, color: COMPARE_COLORS[idx % COMPARE_COLORS.length] } : null;
        })
        .filter((x): x is { run: TrainingRunCurve; color: string } => x !== null),
    [compareIds, runs]
  );

  const compareRunLabel = (run: TrainingRunCurve) =>
    `${modelShortName(run.model)}${compareScope === 'all' ? ` (${run.dataset === 'goemotions' ? 'EN' : 'ZH'})` : ''}`;

  const compareF1Series = selectedCompareRuns.map(({ run, color }) =>
    curveSeries(run, compareRunLabel(run), color, 'val_macro_f1')
  );

  // Group training loss by distinct loss_function so different objective functions are never mixed
  const lossGroups = new Map<string, CurveSeries[]>();
  for (const { run, color } of selectedCompareRuns) {
    const series = curveSeries(run, compareRunLabel(run), color, 'loss');
    if (!series.points.length) continue;
    const group = lossGroups.get(run.loss_function) ?? [];
    group.push(series);
    lossGroups.set(run.loss_function, group);
  }

  const datasetOptions: Array<{ id: DatasetId; label: string }> = [
    { id: 'goemotions', label: 'GoEmotions' },
    { id: 'movie_reviews_zh', label: t('dsZhCardTitle') },
  ];
  const singleModelLabel = singleModels.find(([model]) => model === effectiveSingleModel)?.[1] ?? '';
  const singleF1 = activeSingleRun && curveSeries(activeSingleRun, singleModelLabel, 'var(--series-b3)', 'val_macro_f1');
  const singleLoss = activeSingleRun && curveSeries(activeSingleRun, singleModelLabel, 'var(--series-qwen)', 'loss');

  const renderDatasetCard = (ds: DatasetOverview) => {
    const isEn = ds.dataset === 'goemotions';
    const trPct = (ds.splits.train.ratio * 100).toFixed(2);
    const vaPct = (ds.splits.validation.ratio * 100).toFixed(2);
    const tePct = (ds.splits.test.ratio * 100).toFixed(2);

    return (
      <article key={ds.dataset} className="dataset-card">
        <div className="dataset-card-head">
          <div>
            <h3>{isEn ? t('dsEnCardTitle') : t('dsZhCardTitle')}</h3>
            <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
              {isEn ? t('dsEnCardType') : t('dsZhCardType')}
            </div>
          </div>
        </div>

        <div className="dataset-total-row">
          <div>
            <span className="faint" style={{ fontSize: 11 }}>{t('colFrozenTotal')}</span>
            <div className="dataset-total-num mono">{ds.total_frozen.toLocaleString('en-US')}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span className="faint" style={{ fontSize: 11 }}>{t('labelCountTitle')}</span>
            <div className="mono" style={{ fontSize: 14, fontWeight: 600 }}>
              {ds.label_names.length}
            </div>
          </div>
        </div>

        <div className="split-bar" aria-label="Split ratios">
          <div className="split-seg train" style={{ width: `${trPct}%` }} title={`Train ${trPct}%`} />
          <div className="split-seg val" style={{ width: `${vaPct}%` }} title={`Val ${vaPct}%`} />
          <div className="split-seg test" style={{ width: `${tePct}%` }} title={`Test ${tePct}%`} />
        </div>

        <div className="split-list">
          <div className="split-list-row">
            <span>
              <span className="split-dot" style={{ background: 'var(--series-b3)' }} />
              Train
            </span>
            <span className="mono">
              {`${ds.splits.train.frozen_count.toLocaleString('en-US')} (${trPct}%)`}
            </span>
          </div>
          <div className="split-list-row">
            <span>
              <span className="split-dot" style={{ background: 'var(--series-b2)' }} />
              Validation
            </span>
            <span className="mono">
              {`${ds.splits.validation.frozen_count.toLocaleString('en-US')} (${vaPct}%)`}
            </span>
          </div>
          <div className="split-list-row">
            <span>
              <span className="split-dot" style={{ background: 'var(--series-qwen)' }} />
              Test
            </span>
            <span className="mono">
              {`${ds.splits.test.frozen_count.toLocaleString('en-US')} (${tePct}%)`}
            </span>
          </div>
        </div>
      </article>
    );
  };

  return (
    <section id="data" className="chapter" tabIndex={-1}>
      <div className="chapter-head">
        <p className="chapter-eyebrow">{t('dataEyebrow')}</p>
        <h2 className="chapter-title">{t('dataTitle')}</h2>
      </div>

      {/* 3 Modules: 数据集概览 | 样本分布 | 训练收敛 */}
      <SegmentedPill
        ariaLabel={t('chapterData')}
        centered
        options={[
          { id: 'overview', label: t('dataModOverview') },
          { id: 'distribution', label: t('dataModDistribution') },
          { id: 'curves', label: t('dataModCurves') },
        ]}
        activeId={moduleTab}
        onChange={setModuleTab}
      />

      {moduleTab === 'overview' && (
        <div className="stage-card">
          <div className="dataset-overview-grid">
            {renderDatasetCard(data.datasets.goemotions)}
            {renderDatasetCard(data.datasets.movie_reviews_zh)}
          </div>
        </div>
      )}

      {moduleTab === 'distribution' && (
        <div className="stage-card stable-card">
          <div className="stage-bar">
            <div className="toolbar-cluster">
              <SegmentedPill
                size="sm"
                ariaLabel="Sort label distribution"
                options={[
                  { id: 'freq_desc', label: t('distSortDesc') },
                  { id: 'freq_asc', label: t('distSortAsc') },
                  { id: 'alpha', label: t('distSortAlpha') },
                ]}
                activeId={sortOrder}
                onChange={setSortOrder}
              />
            </div>

            <SegmentedPill
              size="sm"
              ariaLabel="Scale mode"
              options={[
                { id: 'log', label: t('scaleLog') },
                { id: 'linear', label: t('scaleLinear') },
              ]}
              activeId={scale}
              onChange={setScale}
            />
          </div>

          <div className="stage-card-body">
            <DistributionChart
              rows={data.datasets.goemotions.label_counts}
              scale={scale}
              sortOrder={sortOrder}
              title={t('distTitle')}
              note={t('distNote')}
            />
          </div>
        </div>
      )}

      {moduleTab === 'curves' && (
        <div className="stage-card stable-card">
          <div className="stage-bar">
            <SegmentedPill
              size="sm"
              ariaLabel="Convergence view mode"
              options={[
                { id: 'single', label: t('curvesModeSingle') },
                { id: 'compare', label: t('curvesModeCompare') },
              ]}
              activeId={curveMode}
              onChange={setCurveMode}
            />

            {curveMode === 'single' ? (
              <div className="toolbar-cluster">
                <SegmentedPill
                  size="sm"
                  ariaLabel={t('colDataset')}
                  options={datasetOptions}
                  activeId={singleDataset}
                  onChange={handleSingleDatasetChange}
                />

                <select
                  aria-label={t('colModel')}
                  className="compact-select themed-select training-model-select"
                  value={effectiveSingleModel}
                  disabled={!singleModels.length}
                  onChange={(e) => setSingleModel(e.target.value)}
                >
                  {singleModels.map(([model, label]) => (
                    <option key={model} value={model}>
                      {label}
                    </option>
                  ))}
                </select>

                <select
                  aria-label={t('seedLabel')}
                  className="compact-select themed-select training-seed-select"
                  value={effectiveSingleSeed}
                  disabled={!availableSeeds.length}
                  onChange={(e) => setSingleSeed(Number(e.target.value))}
                >
                  {availableSeeds.map((s) => (
                    <option key={s} value={s}>
                      {`Seed ${s}`}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="toolbar-cluster">
                <SegmentedPill
                  size="sm"
                  ariaLabel="Compare scope"
                  options={[
                    ...datasetOptions,
                    { id: 'all', label: t('compareScopeAll') },
                  ]}
                  activeId={compareScope}
                  onChange={handleScopeChange}
                />
                <span className="mono curve-selection-count">{`Seed 42 · ${selectedCompareRuns.length}/3`}</span>
              </div>
            )}
          </div>

          {curveMode === 'single' && !activeSingleRun && <p className="curve-empty">{t('noEpochHistory')}</p>}
          {curveMode === 'single' && activeSingleRun && (
            <div className="stage-card-body">
              <dl className="curve-metrics">
                {activeSingleRun.best_epoch !== null && (
                  <div>
                    <dt>{t('bestEpochLabel')}</dt>
                    <dd>{`${activeSingleRun.best_epoch} / ${activeSingleRun.epochs_completed}`}</dd>
                  </div>
                )}
                <div>
                  <dt>{t('curveValF1')}</dt>
                  <dd>{activeSingleRun.val_macro_f1.toFixed(4)}</dd>
                </div>
                <div>
                  <dt>{t('curveTestF1')}</dt>
                  <dd>{activeSingleRun.test_macro_f1.toFixed(4)}</dd>
                </div>
              </dl>

              {singleF1?.points.length ? (
                <div className="curves-pair">
                  <TrainingCurveChart title={t('curveValF1')} series={[singleF1]} />
                  {singleLoss?.points.length ? (
                    <TrainingCurveChart
                      title={t('curveTrainLoss')}
                      note={activeSingleRun.loss_function}
                      series={[singleLoss]}
                    />
                  ) : null}
                </div>
              ) : <p className="curve-empty">{t(activeSingleRun.model === 'E1' ? 'svmFitNote' : 'noEpochHistory')}</p>}
            </div>
          )}
          {curveMode === 'compare' && (
            <div className="stage-card-body">
              <div className="model-toggle-pills" style={{ marginBottom: 12 }}>
                {compareCandidates.map((r) => {
                  const active = compareIds.includes(r.id);
                  const idx = compareIds.indexOf(r.id);
                  const disabled = !active && compareIds.length >= 3;
                  return (
                    <button
                      key={r.id}
                      type="button"
                      className="filter-pill-btn"
                      aria-pressed={active}
                      disabled={disabled}
                      onClick={() => toggleCompareRun(r.id)}
                      style={
                        active
                          ? {
                              borderColor: COMPARE_COLORS[idx % COMPARE_COLORS.length],
                              color: COMPARE_COLORS[idx % COMPARE_COLORS.length],
                              background: 'var(--surface-subtle)',
                            }
                          : undefined
                      }
                    >
                      {compareRunLabel(r)}
                    </button>
                  );
                })}
              </div>

              <div className="shared-legend">
                {compareF1Series.map((s) => (
                  <span key={s.id} className="legend-item">
                    <span className="legend-swatch" style={{ background: s.color }} />
                    <span className="mono">{s.label}</span>
                  </span>
                ))}
              </div>

              <div className="curves-pair">
                <TrainingCurveChart title={t('curveValF1')} series={compareF1Series} />

                {lossGroups.size > 0 && (
                  <div className="loss-stack">
                    {[...lossGroups].map(([lossFn, lossSeries]) => (
                      <TrainingCurveChart
                        key={lossFn}
                        title={t('curveTrainLoss')}
                        note={lossFn}
                        series={lossSeries}
                        compact={lossGroups.size > 1}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
