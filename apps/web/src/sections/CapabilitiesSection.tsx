import { useMemo, useState } from 'react';
import { useI18n } from '../i18n/I18nContext';
import { SegmentedPill } from '../components/SegmentedPill';
import { TradeoffPanel } from '../components/TradeoffPanel';
import { LONG_TAIL_COLORS, LongTailF1Chart } from '../components/charts/LongTailF1Chart';
import { BootstrapIntervalChart } from '../components/charts/BootstrapIntervalChart';
import { MODEL_LABELS_BY_CODE, modelShortName } from '../data/inference-models';
import type { BootstrapGroup, DashboardBundle, DatasetId, LeaderboardEntry } from '../types/dashboard';

interface CapabilitiesSectionProps {
  data: DashboardBundle;
}

type CapTab = 'pareto' | 'training' | 'longtail' | 'bootstrap';
type TailSort = 'support_asc' | 'support_desc' | 'gain_desc';

const TAIL_MODEL_CHOICES = ['E1', 'E2', 'E3-BiLSTM', 'B1', 'B2', 'B3', 'Qwen3.5-4B'];
function leaderboardName(row: LeaderboardEntry): string {
  return MODEL_LABELS_BY_CODE.get(row.model_code)?.name ?? row.display_name;
}

export function CapabilitiesSection({ data }: CapabilitiesSectionProps) {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState<CapTab>('pareto');
  const [dataset, setDataset] = useState<DatasetId>('goemotions');
  const [tradeoffView, setTradeoffView] = useState<'chart' | 'table'>('chart');
  const [bootstrapGroup, setBootstrapGroup] = useState<BootstrapGroup>('architecture');
  const [tailSort, setTailSort] = useState<TailSort>('support_asc');
  const [tailModels, setTailModels] = useState<string[]>(['E1', 'B1', 'B3', 'Qwen3.5-4B']);

  const enBest = data.leaderboard.goemotions[0];
  const qwenEn = data.leaderboard.goemotions.find((m) => m.model_code === 'Qwen3.5-4B');
  const zhBest = data.leaderboard.movie_reviews_zh[0];
  const summaries = [
    { label: t('metricEnBestLabel'), model: enBest },
    { label: t('metricQwenLabel'), model: qwenEn },
    { label: t('metricZhBestLabel'), model: zhBest },
  ];

  const rows = data.leaderboard[dataset];
  const runSeconds = useMemo(() => new Map(
    (data.training_curves.runs ?? []).flatMap((run) =>
      run.dataset === dataset && run.seed === 42 && typeof run.elapsed_seconds === 'number'
        ? [[run.run_id, run.elapsed_seconds] as const]
        : [],
    ),
  ), [data.training_curves.runs, dataset]);
  const bootstrap = data.paired_comparisons.movie_reviews_zh_families;
  const confidence = (bootstrap.confidence_level * 100).toFixed(3);

  const sortedTailRows = useMemo(() => {
    const list = [...data.per_label_goemotions];
    if (tailSort === 'support_asc') {
      return list.sort((a, b) => a.train_support - b.train_support);
    }
    if (tailSort === 'support_desc') {
      return list.sort((a, b) => b.train_support - a.train_support);
    }
    return list.sort(
      (a, b) =>
        (b.models['B3']?.f1 ?? 0) -
        (b.models['E1']?.f1 ?? 0) -
        ((a.models['B3']?.f1 ?? 0) - (a.models['E1']?.f1 ?? 0))
    );
  }, [data.per_label_goemotions, tailSort]);

  const toggleTailModel = (code: string) => {
    setTailModels((prev) =>
      prev.includes(code)
        ? prev.length > 1
          ? prev.filter((m) => m !== code)
          : prev
        : [...prev, code]
    );
  };

  return (
    <section id="capabilities" className="chapter" tabIndex={-1}>
      <div className="chapter-top-row">
        <div className="chapter-head compact">
          <p className="chapter-eyebrow">{t('capEyebrow')}</p>
          <h2 className="chapter-title">{t('capTitle')}</h2>
        </div>

        <div className="capabilities-summary">
          <div className="capabilities-summary-label">{t('capMetricTitle')}</div>
          <dl className="capabilities-metrics">
            {summaries.map(({ label, model }) => {
              const stats = model?.three_seed?.test_macro_f1;
              const modelName = model && leaderboardName(model);
              return (
                <div className="capabilities-metric" key={label}>
                  <dt>
                    {label}
                    {modelName && <span>{modelName}</span>}
                  </dt>
                  <dd className="mono">
                    {stats ? (
                      <>
                        <strong>{stats.mean.toFixed(4)}</strong>{' '}
                        <span>±{stats.sd.toFixed(4)}</span>
                      </>
                    ) : '—'}
                  </dd>
                </div>
              );
            })}
          </dl>
        </div>
      </div>

      <SegmentedPill
        ariaLabel={t('chapterCapabilities')}
        centered
        options={[
          { id: 'pareto', label: t('capTabPareto') },
          { id: 'training', label: t('capTabTraining') },
          { id: 'longtail', label: t('capTabLongTail') },
          { id: 'bootstrap', label: t('capTabBootstrap') },
        ]}
        activeId={activeTab}
        onChange={setActiveTab}
      />

      {(activeTab === 'pareto' || activeTab === 'training') && (
        <TradeoffPanel
          mode={activeTab === 'training' ? 'training' : 'inference'}
          entries={rows}
          runSeconds={runSeconds}
          dataset={dataset}
          onDatasetChange={setDataset}
          view={tradeoffView}
          onViewChange={setTradeoffView}
        />
      )}

      {activeTab === 'longtail' && (
        <div className="stage-card stable-card">
          <div className="stage-bar">
            <div className="model-toggle-pills" role="group" aria-label={t('colModel')}>
              {TAIL_MODEL_CHOICES.map((code) => {
                const active = tailModels.includes(code);
                return (
                  <button
                    key={code}
                    type="button"
                    className="filter-pill-btn"
                    aria-pressed={active}
                    onClick={() => toggleTailModel(code)}
                    style={
                      active
                        ? {
                            borderColor: LONG_TAIL_COLORS[code],
                            color: LONG_TAIL_COLORS[code],
                            background: 'var(--surface-subtle)',
                          }
                        : undefined
                    }
                  >
                    {modelShortName(code)}
                  </button>
                );
              })}
            </div>

            <select
              aria-label={t('sortEmotionsLabel')}
              className="compact-select themed-select longtail-sort"
              value={tailSort}
              onChange={(e) => setTailSort(e.target.value as TailSort)}
            >
              <option value="support_asc">{t('sortSupportAsc')}</option>
              <option value="support_desc">{t('sortSupportDesc')}</option>
              <option value="gain_desc">{t('sortGainDesc')}</option>
            </select>
          </div>

          <div className="stage-card-body">
            <LongTailF1Chart
              rows={sortedTailRows}
              selectedModels={TAIL_MODEL_CHOICES.filter((m) => tailModels.includes(m))}
              title={t('longTailTitle')}
            />
          </div>

          <div className="stage-footnote">
            <span>{t('longTailFoot')}</span>
            <span className="mono">28 labels · GoEmotions</span>
          </div>
        </div>
      )}

      {activeTab === 'bootstrap' && (
        <div className="stage-card bootstrap-families-card">
          <div className="stage-bar">
            <SegmentedPill
              size="sm"
              ariaLabel={t('bootstrapGroupLabel')}
              options={[
                { id: 'architecture', label: t('bootstrapArchitecture') },
                { id: 'ablation', label: t('bootstrapAblation') },
              ]}
              activeId={bootstrapGroup}
              onChange={setBootstrapGroup}
            />
            <span className="muted mono">
              {t('datasetZhName')} · Seed {bootstrap.seed} · n={bootstrap.n}
            </span>
          </div>
          <BootstrapIntervalChart
            comparisons={bootstrap.groups[bootstrapGroup]}
            title={t(bootstrapGroup === 'architecture' ? 'bootstrapArchitectureTitle' : 'bootstrapAblationTitle')}
            note={'Bonferroni · ' + confidence + '% CI'}
            axisLabel={t('bootstrapAxis')}
            showRandomVariant={bootstrapGroup === 'ablation'}
          />
          <div className="stage-footnote bootstrap-method">
            <span>
              {t('bootstrapMethod')
                .replace('{resamples}', bootstrap.n_resamples.toLocaleString('en-US'))
                .replace('{count}', String(bootstrap.comparison_count))
                .replace('{alpha}', String(bootstrap.familywise_alpha))}
            </span>
            {bootstrapGroup === 'architecture' && <span>{t('bootstrapExploratory')}</span>}
          </div>
        </div>
      )}
    </section>
  );
}
