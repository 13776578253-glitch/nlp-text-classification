import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { INFERENCE_MODEL_GROUPS } from '../data/inference-models';
import { useI18n } from '../i18n/I18nContext';
import type { TranslationKey } from '../i18n/translations';
import type {
  ApiHealth,
  ApiModelInfo,
  DashboardBundle,
  DatasetId,
  LeaderboardEntry,
  PredictionResult,
} from '../types/dashboard';

interface InferenceSectionProps {
  data: DashboardBundle;
  apiHealth: ApiHealth | null;
  apiModels: ApiModelInfo[];
  onRefreshApi: () => void;
}

interface CardState {
  status: 'loading' | 'done' | 'error';
  result?: PredictionResult;
  pending?: boolean;
  showSpinner?: boolean;
  error?: { message: string; code?: string; startCommand?: string };
}

interface InferenceModel {
  model_id: string;
  name: string;
  shortName: string;
  detail?: TranslationKey;
  testMacroF1: number;
  runtime: ApiModelInfo['runtime'];
  paramLabel?: string;
}

const DEFAULT_MODELS: Record<DatasetId, string[]> = {
  goemotions: ['goemotions/B3', 'goemotions/B1', 'goemotions/E1'],
  movie_reviews_zh: ['movie_reviews_zh/B4', 'movie_reviews_zh/E1', 'movie_reviews_zh/E2'],
};

const QUICK_SAMPLE_LABELS: Partial<Record<string, TranslationKey>> = {
  eelgwd1: 'sampleGratitude',
  ee3yyz3: 'sampleAmusement',
  eeccgbb: 'sampleAdmiration',
  edk4e66: 'sampleFear',
  'test:200': 'sampleCrimeDrama',
  'test:6': 'sampleStrongNegative',
  'test:190': 'sampleMixedLanguage',
  'test:2': 'sampleNuancedNegative',
};

function useAnchoredPopover(
  triggerRef: React.RefObject<HTMLButtonElement | null>,
  popoverRef: React.RefObject<HTMLDivElement | null>,
  alignEnd = false
) {
  useEffect(() => {
    const trigger = triggerRef.current;
    const pop = popoverRef.current;
    if (!trigger || !pop) return;

    const handleToggle = (event: Event) => {
      const toggleEv = event as Event & { newState?: 'open' | 'closed' };
      const isOpen = toggleEv.newState === 'open' || pop.matches(':popover-open');
      trigger.setAttribute('aria-expanded', isOpen ? 'true' : 'false');

      if (!isOpen) return;
      const supportsAnchor =
        typeof CSS !== 'undefined' &&
        typeof CSS.supports === 'function' &&
        CSS.supports('position-area: block-end');
      if (supportsAnchor) return;

      const tRect = trigger.getBoundingClientRect();
      const pRect = pop.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let left = alignEnd ? tRect.right - pRect.width : tRect.left;
      left = Math.max(12, Math.min(left, vw - pRect.width - 12));
      let top = tRect.bottom + 8;
      if (top + pRect.height > vh - 12 && tRect.top - pRect.height - 8 >= 12) {
        top = tRect.top - pRect.height - 8;
      }
      pop.style.left = `${Math.round(left)}px`;
      pop.style.top = `${Math.round(top)}px`;
    };

    pop.addEventListener('toggle', handleToggle);
    return () => pop.removeEventListener('toggle', handleToggle);
  }, [triggerRef, popoverRef, alignEnd]);
}

function useMenuArrowNavigation(menuRef: React.RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const menu = menuRef.current;
    if (!menu) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
      const items = Array.from(menu.querySelectorAll<HTMLButtonElement>('button')).filter(
        (btn) => !btn.disabled && btn.getAttribute('aria-disabled') !== 'true'
      );
      if (!items.length) return;
      const current = items.indexOf(document.activeElement as HTMLButtonElement);
      const next =
        event.key === 'Home'
          ? 0
          : event.key === 'End'
          ? items.length - 1
          : current < 0
          ? 0
          : (current + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
      event.preventDefault();
      items[next]?.focus();
    };

    menu.addEventListener('keydown', onKeyDown);
    return () => menu.removeEventListener('keydown', onKeyDown);
  }, [menuRef]);
}

export function InferenceSection({
  data,
  apiHealth,
  apiModels,
  onRefreshApi,
}: InferenceSectionProps) {
  const { locale, t } = useI18n();
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');

  const presetTriggerId = `preset-trig-${uid}`;
  const presetPopoverId = `preset-pop-${uid}`;
  const modelTriggerId = `model-trig-${uid}`;
  const modelPopoverId = `model-pop-${uid}`;
  const modelMenuLabelId = `model-title-${uid}`;
  const paramsTriggerId = `params-trig-${uid}`;
  const paramsPopoverId = `params-pop-${uid}`;

  const presetTriggerRef = useRef<HTMLButtonElement | null>(null);
  const presetPopoverRef = useRef<HTMLDivElement | null>(null);
  const presetMenuRef = useRef<HTMLDivElement | null>(null);

  const modelTriggerRef = useRef<HTMLButtonElement | null>(null);
  const modelPopoverRef = useRef<HTMLDivElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  const paramsTriggerRef = useRef<HTMLButtonElement | null>(null);
  const paramsPopoverRef = useRef<HTMLDivElement | null>(null);
  const composerInputRef = useRef<HTMLTextAreaElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);

  const [dataset, setDataset] = useState<DatasetId>('goemotions');
  const firstPreset = data.sample_presets.goemotions[0];
  const [text, setText] = useState<string>(firstPreset?.text ?? '');
  const [activePresetId, setActivePresetId] = useState<string | null>(firstPreset?.sample_id ?? null);
  const [activePresetGold, setActivePresetGold] = useState<string[] | null>(
    firstPreset?.gold_labels ?? null
  );
  const [showGold, setShowGold] = useState(false);
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_MODELS.goemotions);
  const [focusedModelId, setFocusedModelId] = useState<string | null>(DEFAULT_MODELS.goemotions[0]);
  const [seed, setSeed] = useState<number>(42);
  const [topK, setTopK] = useState<number>(8);
  const [cards, setCards] = useState<Record<string, CardState>>({});
  const [resultModels, setResultModels] = useState<InferenceModel[]>([]);
  const [resultGold, setResultGold] = useState<string[] | null>(null);
  const [resultsExpanded, setResultsExpanded] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [showRunSpinner, setShowRunSpinner] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const runSeqRef = useRef(0);
  const runSpinnerTimerRef = useRef<number | null>(null);
  const cardTimersRef = useRef<Record<string, number>>({});

  const clearPendingTimers = () => {
    if (runSpinnerTimerRef.current !== null) {
      window.clearTimeout(runSpinnerTimerRef.current);
      runSpinnerTimerRef.current = null;
    }
    for (const timerId of Object.values(cardTimersRef.current)) {
      window.clearTimeout(timerId);
    }
    cardTimersRef.current = {};
  };

  useAnchoredPopover(presetTriggerRef, presetPopoverRef, false);
  useAnchoredPopover(modelTriggerRef, modelPopoverRef, false);
  useAnchoredPopover(paramsTriggerRef, paramsPopoverRef, true);
  useMenuArrowNavigation(presetMenuRef);
  useMenuArrowNavigation(modelMenuRef);

  const apiOnline = Boolean(apiHealth && apiHealth.status === 'ok');
  const qwenOnline = Boolean(apiHealth?.qwen_worker?.online);

  const { catalog, modelGroups } = useMemo(() => {
    const liveMap = new Map(apiModels.map((m) => [m.model_id, m]));
    const entries = new Map(data.leaderboard[dataset].map((entry) => [entry.model_code, entry]));
    const toModel = (
      entry: LeaderboardEntry,
      label?: { name: string; shortName?: string; detail: TranslationKey },
    ): InferenceModel => ({
      model_id: entry.model_id,
      name: label?.name ?? entry.display_name,
      shortName: label?.shortName ?? label?.name ?? entry.short_label,
      detail: label?.detail,
      testMacroF1: entry.seed42.test_macro_f1,
      runtime:
        liveMap.get(entry.model_id)?.runtime ??
        (entry.model_code === 'Qwen3.5-4B' ? 'wsl2_worker' : 'local'),
      paramLabel: entry.param_label,
    });

    const groups = INFERENCE_MODEL_GROUPS.map(({ label, models }) => ({
      label,
      models: models.flatMap((model) => {
        const entry = entries.get(model.code);
        if (!entry) return [];
        entries.delete(model.code);
        return [toModel(entry, model)];
      }),
    })).filter(({ models }) => models.length > 0);
    if (entries.size > 0) {
      groups.push({
        label: 'modelGroupOther',
        models: [...entries.values()].map((entry) => toModel(entry)),
      });
    }
    return { catalog: groups.flatMap(({ models }) => models), modelGroups: groups };
  }, [apiModels, data.leaderboard, dataset]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      clearPendingTimers();
    };
  }, []);

  const returnToInput = useCallback(() => {
    if (window.scrollY > 1) window.scrollTo({ top: 0, behavior: 'instant' });
    if (resultsRef.current?.contains(document.activeElement)) {
      composerInputRef.current?.focus({ preventScroll: true });
    }
    setResultsExpanded(false);
  }, []);

  const clearResults = () => {
    abortRef.current?.abort();
    clearPendingTimers();
    runSeqRef.current += 1;
    returnToInput();
    setCards({});
    setResultModels([]);
    setResultGold(null);
    setIsRunning(false);
    setShowRunSpinner(false);
  };

  useEffect(() => {
    if (!resultsExpanded) return;

    const popoverIsOpen = () =>
      [presetPopoverRef, modelPopoverRef, paramsPopoverRef].some((ref) =>
        ref.current?.matches(':popover-open'),
      );

    const onWheel = (event: WheelEvent) => {
      if (
        event.deltaY >= 0 ||
        event.ctrlKey ||
        Math.abs(event.deltaX) > -event.deltaY ||
        popoverIsOpen()
      ) return;
      if (event.target instanceof HTMLTextAreaElement && event.target.scrollTop > 0) return;
      const wheelUnit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1;
      const topRange = Math.min(80, -event.deltaY * wheelUnit);
      if (window.scrollY > Math.max(1, topRange)) return;
      if (event.cancelable) event.preventDefault();
      returnToInput();
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented || popoverIsOpen()) return;
      const inInference = event.target instanceof Element && event.target.closest('#inference');
      if (window.scrollY <= 1 || inInference) returnToInput();
    };

    let swipeStart: { x: number; y: number } | null = null;
    const onTouchStart = (event: TouchEvent) => {
      swipeStart = null;
      if (event.touches.length !== 1 || window.scrollY > 80 || popoverIsOpen()) return;
      if (event.target instanceof HTMLTextAreaElement && event.target.scrollTop > 0) return;
      const touch = event.touches[0];
      swipeStart = { x: touch.clientX, y: touch.clientY };
    };
    const onTouchEnd = (event: TouchEvent) => {
      const start = swipeStart;
      swipeStart = null;
      if (!start || event.touches.length > 0 || window.scrollY > 1) return;
      const touch = event.changedTouches[0];
      if (!touch) return;
      const deltaY = touch.clientY - start.y;
      if (deltaY >= 48 && deltaY > Math.abs(touch.clientX - start.x)) returnToInput();
    };

    window.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchend', onTouchEnd, { passive: true });
    return () => {
      window.removeEventListener('wheel', onWheel);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchend', onTouchEnd);
    };
  }, [resultsExpanded, returnToInput]);

  const handleDatasetSwitch = (nextDs: DatasetId) => {
    if (nextDs === dataset) return;
    abortRef.current?.abort();
    clearPendingTimers();
    runSeqRef.current += 1;
    setDataset(nextDs);
    const preset = data.sample_presets[nextDs][0];
    setText(preset?.text ?? '');
    setActivePresetId(preset?.sample_id ?? null);
    setActivePresetGold(preset?.gold_labels ?? null);
    setSelectedModels(DEFAULT_MODELS[nextDs]);
    setResultsExpanded(false);
    setCards((previous) => Object.fromEntries(
      Object.entries(previous)
        .filter(([, card]) => card.status !== 'loading')
        .map(([id, card]) => [id, { ...card, pending: false, showSpinner: false }]),
    ));
    setIsRunning(false);
    setShowRunSpinner(false);
  };

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((m) => m !== modelId) : [...prev, modelId]
    );
  };

  const orderedSelected = useMemo(
    () => catalog.filter((m) => selectedModels.includes(m.model_id)),
    [catalog, selectedModels]
  );

  const runInference = async (overrideText?: string, overrideGold?: string[]) => {
    const cleaned = (overrideText ?? text).trim();
    if (!cleaned || selectedModels.length === 0 || !apiOnline) return;

    abortRef.current?.abort();
    clearPendingTimers();
    const mySeq = ++runSeqRef.current;
    const controller = new AbortController();
    abortRef.current = controller;

    const targetModels = orderedSelected.map((m) => m.model_id);
    const preferredFocus =
      focusedModelId && targetModels.includes(focusedModelId)
        ? focusedModelId
        : DEFAULT_MODELS[dataset].find((id) => targetModels.includes(id)) ?? targetModels[0];
    const gold = overrideGold ?? activePresetGold;

    const wasExpanded = resultsExpanded;
    const settled: Record<string, CardState> = {};
    let committedInitial = wasExpanded;

    const fetchModel = async (modelId: string) => {
      let nextState: CardState;
      try {
        const resp = await fetch('/api/predict/single', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: cleaned, model_id: modelId, seed, top_k: topK }),
          signal: controller.signal,
        });
        const body = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          nextState = {
            status: 'error',
            pending: false,
            showSpinner: false,
            error: {
              message: body.detail || `HTTP ${resp.status}`,
              code: body.code,
              startCommand: body.start_command,
            },
          };
        } else {
          nextState = {
            status: 'done',
            result: body.result as PredictionResult,
            pending: false,
            showSpinner: false,
          };
        }
      } catch (err) {
        if ((err as Error).name === 'AbortError') return;
        nextState = {
          status: 'error',
          pending: false,
          showSpinner: false,
          error: {
            message: (err as Error).message || t('cardOfflineError'),
          },
        };
      }

      if (mySeq !== runSeqRef.current || controller.signal.aborted) return;

      if (cardTimersRef.current[modelId] !== undefined) {
        window.clearTimeout(cardTimersRef.current[modelId]);
        delete cardTimersRef.current[modelId];
      }

      settled[modelId] = nextState;
      if (committedInitial) {
        setCards((prev) => ({
          ...prev,
          [modelId]: nextState,
        }));
      }
    };

    const allDonePromise = Promise.all(targetModels.map(fetchModel));

    if (!wasExpanded) {
      setIsRunning(true);
      // Give fast models a short window to populate the result panel before it opens.
      await Promise.race([
        allDonePromise,
        new Promise<void>((resolve) => window.setTimeout(resolve, 140)),
      ]);
      if (mySeq !== runSeqRef.current || controller.signal.aborted) return;

      committedInitial = true;
      const initialSnapshot: Record<string, CardState> = {};
      let anyPending = false;
      for (const id of targetModels) {
        if (settled[id]) {
          initialSnapshot[id] = settled[id];
        } else {
          anyPending = true;
          initialSnapshot[id] = { status: 'loading', pending: true, showSpinner: true };
        }
      }

      setCards(initialSnapshot);
      setResultModels(orderedSelected);
      setResultGold(gold);
      setFocusedModelId(preferredFocus);
      setIsRunning(anyPending);
      setShowRunSpinner(anyPending);
      setResultsExpanded(true);
    } else {
      // Already expanded: incremental in-place update without blanking results or flashing spinners <180ms
      setIsRunning(true);
      setShowRunSpinner(false);
      setResultModels(orderedSelected);
      setResultGold(gold);
      setCards((prev) => {
        const next: Record<string, CardState> = {};
        for (const id of targetModels) {
          const old = prev[id];
          if (old && old.status === 'done' && old.result) {
            next[id] = {
              status: 'done',
              result: old.result,
              pending: true,
              showSpinner: false,
            };
          } else {
            next[id] = { status: 'loading', pending: true, showSpinner: false };
          }
        }
        return next;
      });
      if (preferredFocus !== focusedModelId) {
        setFocusedModelId(preferredFocus);
      }

      runSpinnerTimerRef.current = window.setTimeout(() => {
        if (mySeq === runSeqRef.current && !controller.signal.aborted) {
          setShowRunSpinner(true);
        }
      }, 180);

      for (const modelId of targetModels) {
        cardTimersRef.current[modelId] = window.setTimeout(() => {
          if (mySeq !== runSeqRef.current || controller.signal.aborted) return;
          setCards((prev) => {
            const cur = prev[modelId];
            if (!cur || !cur.pending) return prev;
            return { ...prev, [modelId]: { ...cur, showSpinner: true } };
          });
        }, 180);
      }
    }

    await allDonePromise;

    if (mySeq === runSeqRef.current && !controller.signal.aborted) {
      clearPendingTimers();
      setIsRunning(false);
      setShowRunSpinner(false);
      onRefreshApi();
    }
  };

  const activeCarouselModels = useMemo(
    () => resultModels.filter((m) => Boolean(cards[m.model_id])),
    [resultModels, cards]
  );

  const activeIndex = useMemo(() => {
    if (!activeCarouselModels.length) return 0;
    const idx = activeCarouselModels.findIndex((m) => m.model_id === focusedModelId);
    return idx >= 0 ? idx : 0;
  }, [activeCarouselModels, focusedModelId]);

  const activeModel = activeCarouselModels[activeIndex] ?? null;
  const activeCard = activeModel ? cards[activeModel.model_id] : undefined;

  const stepCarousel = (delta: -1 | 1) => {
    if (activeCarouselModels.length <= 1) return;
    const nextIdx =
      (activeIndex + delta + activeCarouselModels.length) % activeCarouselModels.length;
    setFocusedModelId(activeCarouselModels[nextIdx].model_id);
  };

  const selectedNames = orderedSelected.map((m) => m.shortName);
  const compactModelSummary = selectedNames.length > 2
    ? `${selectedNames.slice(0, 2).join(' · ')} +${selectedNames.length - 2}`
    : selectedNames.join(' · ') || t('modelSummaryEmpty');
  const modelSummary = selectedNames.length === 3
    ? selectedNames.join(' · ')
    : compactModelSummary;

  const consensus = useMemo(() => {
    const doneCards = Object.values(cards).filter(
      (c): c is CardState & { status: 'done'; result: PredictionResult } =>
        c.status === 'done' && Boolean(c.result)
    );
    if (!doneCards.length) return null;
    const counts = new Map<string, number>();
    for (const c of doneCards) {
      for (const lbl of c.result.predicted_labels) {
        counts.set(lbl, (counts.get(lbl) ?? 0) + 1);
      }
    }
    const items = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    return { total: doneCards.length, items };
  }, [cards]);

  const copyQwenCmd = async () => {
    try {
      await navigator.clipboard.writeText('wsl bash scripts/start_qwen_worker_wsl.sh');
      setCopiedCmd(true);
      window.setTimeout(() => setCopiedCmd(false), 1800);
    } catch {
      // Clipboard may be unavailable in restricted contexts
    }
  };

  const quickPresets = data.sample_presets[dataset].slice(0, 4);

  return (
    <section
      id="inference"
      className="hero-stage"
      data-view={resultsExpanded ? 'results' : 'input'}
      tabIndex={-1}
    >
      <div className="hero-aura" aria-hidden="true" />

      <div className="hero-input-region">
        <div className="hero-workspace">
          <div className="hero-intro">
            <h1 className="hero-display-title">{t('heroTitle')}</h1>
          </div>

          <div className="hero-composer-wrap">
            <div className="composer-card">
              <label htmlFor={`input-${uid}`} className="visually-hidden">
                {t('inputAriaLabel')}
              </label>
              <textarea
                ref={composerInputRef}
                id={`input-${uid}`}
                className="composer-input"
                rows={3}
                maxLength={6000}
                value={text}
                placeholder={dataset === 'goemotions' ? t('inputPlaceholderEn') : t('inputPlaceholderZh')}
                onChange={(e) => {
                  setText(e.target.value);
                  setActivePresetId(null);
                  setActivePresetGold(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    runInference();
                  }
                }}
              />

              <div className="composer-toolbar">
                <div className="toolbar-cluster">
                  {/* + Presets Popover (exact mathematically centered SVG) */}
                  <button
                    ref={presetTriggerRef}
                    id={presetTriggerId}
                    type="button"
                    className="harness-trigger icon-only"
                    popoverTarget={presetPopoverId}
                    aria-haspopup="menu"
                    aria-expanded="false"
                    aria-label={t('presetPopoverTitle')}
                    style={{ anchorName: `--p-preset-${uid}` } as React.CSSProperties}
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 16 16"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      aria-hidden="true"
                    >
                      <path d="M8 3.25v9.5M3.25 8h9.5" />
                    </svg>
                  </button>
                  <div
                    ref={presetPopoverRef}
                    id={presetPopoverId}
                    popover="auto"
                    className="harness-popover model-popover"
                    style={{ positionAnchor: `--p-preset-${uid}` } as React.CSSProperties}
                  >
                    <div className="popover-head">{t('presetPopoverTitle')}</div>
                    <div
                      ref={presetMenuRef}
                      role="menu"
                      aria-labelledby={presetTriggerId}
                      className="menu-list"
                    >
                      {data.sample_presets[dataset].map((preset) => (
                        <button
                          key={preset.sample_id}
                          type="button"
                          role="menuitem"
                          className="model-row"
                          onClick={() => {
                            setText(preset.text);
                            setActivePresetId(preset.sample_id);
                            setActivePresetGold(preset.gold_labels);
                            (presetPopoverRef.current as HTMLElement | null)?.hidePopover?.();
                          }}
                        >
                          <span className="model-row-left">
                            <span className="model-row-code">
                              {locale === 'zh' ? preset.title_zh : preset.title_en}
                            </span>
                          </span>
                          <span className="menu-check" aria-hidden="true">
                            {activePresetId === preset.sample_id ? '✓' : ''}
                          </span>
                        </button>
                      ))}
                      <hr className="popover-divider" />
                      <button
                        type="button"
                        role="menuitemcheckbox"
                        aria-checked={showGold}
                        className="model-row"
                        onClick={() => setShowGold((v) => !v)}
                      >
                        <span>{t('presetRevealGold')}</span>
                        <span className="menu-check" aria-hidden="true">
                          {showGold ? '✓' : ''}
                        </span>
                      </button>
                    </div>
                  </div>

                  {/* Lightweight Single-Line Model Picker */}
                  <button
                    ref={modelTriggerRef}
                    id={modelTriggerId}
                    type="button"
                    className="harness-trigger model-picker-trigger"
                    popoverTarget={modelPopoverId}
                    aria-haspopup="menu"
                    aria-expanded="false"
                    style={{ anchorName: `--p-model-${uid}` } as React.CSSProperties}
                  >
                    <span className="mono model-summary-wide">{modelSummary}</span>
                    <span className="mono model-summary-compact">{compactModelSummary}</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" aria-hidden="true">
                      <path d="m6 9 6 6 6-6" />
                    </svg>
                  </button>
                  <div
                    ref={modelPopoverRef}
                    id={modelPopoverId}
                    popover="auto"
                    className="harness-popover model-picker-popover"
                    style={{ positionAnchor: `--p-model-${uid}` } as React.CSSProperties}
                  >
                    <div className="popover-head model-picker-head">
                      <span id={modelMenuLabelId} className="model-picker-title">
                        {t('modelPopoverTitle')}
                      </span>
                      <span className="popover-head-actions">
                        <button
                          type="button"
                          className="text-link-btn"
                          onClick={() => setSelectedModels(DEFAULT_MODELS[dataset])}
                        >
                          {t('modelResetDefault')}
                        </button>
                        <button
                          type="button"
                          className="text-link-btn"
                          onClick={() => setSelectedModels(catalog.map((m) => m.model_id))}
                        >
                          {t('modelSelectAll')}
                        </button>
                        <button
                          type="button"
                          className="text-link-btn"
                          onClick={() => setSelectedModels([])}
                        >
                          {t('modelClear')}
                        </button>
                        <span className="model-picker-count">
                          {`${t('modelSelectedCount')} ${selectedModels.length} / ${catalog.length}`}
                        </span>
                      </span>
                    </div>
                    <div
                      ref={modelMenuRef}
                      role="menu"
                      aria-labelledby={modelMenuLabelId}
                      className="model-picker-menu"
                    >
                      {modelGroups.map(({ label, models }) => {
                        const groupLabelId = `${modelPopoverId}-${label}`;
                        return (
                          <div key={label} role="group" aria-labelledby={groupLabelId}>
                            <div id={groupLabelId} className="model-group-label">{t(label)}</div>
                            {models.map((model) => {
                              const checked = selectedModels.includes(model.model_id);
                              const detail = [
                                model.detail && t(model.detail),
                                model.runtime === 'wsl2_worker' && !qwenOnline && t('modelUnavailableBadge'),
                              ].filter(Boolean).join(' · ');

                              return (
                                <button
                                  key={model.model_id}
                                  type="button"
                                  role="menuitemcheckbox"
                                  aria-checked={checked}
                                  className="model-row model-picker-row"
                                  onClick={() => toggleModel(model.model_id)}
                                >
                                  <span className="model-picker-row-label">
                                    <span className="model-picker-name">{model.name}</span>
                                    {detail && <span className="model-picker-detail">{detail}</span>}
                                  </span>
                                  <span className="model-picker-row-score mono">
                                    {model.testMacroF1?.toFixed(3) ?? '—'}
                                  </span>
                                  <span className="menu-check" aria-hidden="true">
                                    {checked ? '✓' : ''}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        );
                      })}
                    </div>
                    {dataset === 'goemotions' && !qwenOnline && (
                      <>
                        <hr className="popover-divider" />
                        <div className="popover-foot">
                          <button type="button" className="copy-cmd-btn" onClick={copyQwenCmd}>
                            <span>{t('qwenCopyShort')}</span>
                            <span className="mono">{copiedCmd ? t('copiedCommand') : t('copyCommand')}</span>
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                <div className="toolbar-cluster">
                  <div className="dataset-pill-group" role="group" aria-label={t('colDataset')}>
                    <button
                      type="button"
                      className="dataset-pill-btn"
                      aria-pressed={dataset === 'goemotions'}
                      onClick={() => handleDatasetSwitch('goemotions')}
                    >
                      {t('datasetEnShort')}
                    </button>
                    <button
                      type="button"
                      className="dataset-pill-btn"
                      aria-pressed={dataset === 'movie_reviews_zh'}
                      onClick={() => handleDatasetSwitch('movie_reviews_zh')}
                    >
                      {t('datasetZhShort')}
                    </button>
                  </div>

                  {/* Params Dialog Popover */}
                  <button
                    ref={paramsTriggerRef}
                    id={paramsTriggerId}
                    type="button"
                    className="harness-trigger mono"
                    popoverTarget={paramsPopoverId}
                    aria-haspopup="dialog"
                    aria-expanded="false"
                    style={{ anchorName: `--p-params-${uid}` } as React.CSSProperties}
                  >
                    <span>{dataset === 'goemotions' ? `Seed ${seed} · Top ${topK}` : `Seed ${seed}`}</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" aria-hidden="true">
                      <path d="m6 9 6 6 6-6" />
                    </svg>
                  </button>
                  <div
                    ref={paramsPopoverRef}
                    id={paramsPopoverId}
                    role="dialog"
                    aria-label={t('paramsPopoverTitle')}
                    popover="auto"
                    className="harness-popover align-end"
                    style={{ positionAnchor: `--p-params-${uid}` } as React.CSSProperties}
                  >
                    <div className="popover-head">{t('paramsPopoverTitle')}</div>
                    <div className="params-group">
                      <span className="params-label">{t('seedLabel')}</span>
                      <div className="params-pills">
                        {[42, 123, 2026].map((sVal) => (
                          <button
                            key={sVal}
                            type="button"
                            className="param-pill"
                            aria-pressed={seed === sVal}
                            onClick={() => setSeed(sVal)}
                          >
                            {sVal}
                          </button>
                        ))}
                      </div>
                    </div>
                    {dataset === 'goemotions' && (
                      <div className="params-group">
                        <span className="params-label">{t('topKLabel')}</span>
                        <div className="params-pills">
                          {[5, 8, 12, 28].map((kVal) => (
                            <button
                              key={kVal}
                              type="button"
                              className="param-pill"
                              aria-pressed={topK === kVal}
                              onClick={() => setTopK(kVal)}
                            >
                              {kVal}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <button
                    type="button"
                    className="run-circle-btn"
                    aria-label={t('runBtnAria')}
                    disabled={!apiOnline || !text.trim() || selectedModels.length === 0 || isRunning}
                    onClick={() => runInference()}
                  >
                    {showRunSpinner ? (
                      <svg
                        className="btn-spinner"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        aria-hidden="true"
                      >
                        <circle cx="8" cy="8" r="5.5" stroke="currentColor" strokeOpacity="0.28" strokeWidth="2" />
                        <path d="M13.5 8A5.5 5.5 0 0 0 8 2.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    ) : (
                      <svg
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.1"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M8 12.75V3.25M4.25 7 8 3.25 11.75 7" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="sample-strip">
            {quickPresets.map((preset) => {
              const shortLabel = QUICK_SAMPLE_LABELS[preset.sample_id];
              return (
                <button
                  key={preset.sample_id}
                  type="button"
                  className="sample-chip"
                  aria-pressed={activePresetId === preset.sample_id}
                  onClick={() => {
                    setText(preset.text);
                    setActivePresetId(preset.sample_id);
                    setActivePresetGold(preset.gold_labels);
                    if (apiOnline && selectedModels.length > 0) {
                      runInference(preset.text, preset.gold_labels);
                    }
                  }}
                >
                  {shortLabel ? t(shortLabel) : locale === 'zh' ? preset.title_zh : preset.title_en}
                </button>
              );
            })}
          </div>

          {!apiOnline && (
            <div className="inline-notice" role="status">
              <span>{t('offlineNotice')}</span>
              <code className="mono">uvicorn apps.api.main:app --host 127.0.0.1 --port 8000</code>
            </div>
          )}
        </div>

        <a
          href={activeCard ? '#inference-results' : '#capabilities'}
          className="scroll-cue"
          inert={resultsExpanded}
        >
          <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M8 3.25v9.5M4.5 9.25 8 12.75l3.5-3.5" />
          </svg>
          <span>{t(activeCard ? 'scrollToResults' : 'scrollDownCue')}</span>
        </a>
      </div>

      <div id="inference-results" className="results-stage" ref={resultsRef} hidden={!activeCard} tabIndex={-1}>
        {activeModel && activeCard && (
          <>
              <div className="consensus-strip">
                <div className="consensus-items">
                  <span className="muted" style={{ fontSize: 12, fontWeight: 500 }}>
                    {t('consensusPrefix')}
                  </span>
                  {consensus && consensus.items.length === 0 && (
                    <span className="muted">{t('noThresholdHit')}</span>
                  )}
                  {consensus?.items.map(([lbl, cnt]) => (
                    <span key={lbl} className="result-consensus-label mono">
                      {`${lbl} (${cnt}/${consensus.total})`}
                    </span>
                  ))}
                  {showGold && resultGold && (
                    <span className="result-gold-label mono">
                      {`${t('goldTagPrefix')}: ${resultGold.join(', ')}`}
                    </span>
                  )}
                </div>
                <div className="toolbar-cluster">
                  {consensus && (
                    <span className="faint mono" style={{ fontSize: 11.5 }}>
                      {`${consensus.total} ${t('completedSuffix')}`}
                    </span>
                  )}
                  <button
                    type="button"
                    className="result-return"
                    aria-label={t('returnToInput')}
                    title={t('clearResultsAndReturn')}
                    onClick={clearResults}
                  >
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M8 12.75V3.25M4.5 6.75 8 3.25l3.5 3.5" />
                    </svg>
                  </button>
                </div>
              </div>

              <article className="result-carousel">
                {/* Switcher tabs + prev/next arrows */}
                <div className="carousel-head-bar">
                  <div className="carousel-tabs" role="tablist" aria-label={t('modelPopoverTitle')}>
                    {activeCarouselModels.map((m) => {
                      const st = cards[m.model_id];
                      const selected = m.model_id === activeModel.model_id;
                      return (
                        <button
                          key={m.model_id}
                          type="button"
                          role="tab"
                          aria-selected={selected}
                          className="carousel-tab mono"
                          onClick={() => setFocusedModelId(m.model_id)}
                        >
                          <span>{m.shortName}</span>
                          {st?.result && (
                            <span style={{ opacity: 0.72, fontSize: 11 }}>
                              {`${st.result.latency_ms} ms`}
                            </span>
                          )}
                          {st?.status === 'loading' && st.showSpinner && (
                            <span style={{ opacity: 0.72 }}>…</span>
                          )}
                          {st?.status === 'error' && <span style={{ color: 'var(--warn)' }}>!</span>}
                        </button>
                      );
                    })}
                  </div>

                  <div className="carousel-arrows">
                    <button
                      type="button"
                      className="carousel-arrow-btn"
                      aria-label={t('prevModel')}
                      disabled={activeCarouselModels.length <= 1}
                      onClick={() => stepCarousel(-1)}
                    >
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M12.5 8H3.5M7 4.5 3.5 8 7 11.5" />
                      </svg>
                    </button>
                    <span className="faint mono" style={{ fontSize: 11.5, minWidth: 36, textAlign: 'center' }}>
                      {`${activeIndex + 1} / ${activeCarouselModels.length}`}
                    </span>
                    <button
                      type="button"
                      className="carousel-arrow-btn"
                      aria-label={t('nextModel')}
                      disabled={activeCarouselModels.length <= 1}
                      onClick={() => stepCarousel(1)}
                    >
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M3.5 8h9M9 4.5 12.5 8 9 11.5" />
                      </svg>
                    </button>
                  </div>
                </div>

                {activeCard.status === 'loading' && !activeCard.result && activeCard.showSpinner && (
                  <div style={{ padding: '24px 0' }}>
                    <div className="muted" style={{ fontSize: 13, marginBottom: 10 }}>
                      {`${activeModel.name} · ${t('cardWaiting')}`}
                    </div>
                    <progress aria-label={`${activeModel.name} · ${t('cardWaiting')}`} style={{ width: '100%' }} />
                  </div>
                )}

                {activeCard.status === 'error' && !activeCard.result && (
                  <div className="tone-badge warn" style={{ display: 'block', padding: '14px 16px', borderRadius: 12 }}>
                    <div style={{ fontWeight: 600 }}>{t('cardOfflineError')}</div>
                    <div style={{ marginTop: 4 }}>{activeCard.error?.message ?? t('cardOfflineError')}</div>
                  </div>
                )}

                {activeCard.result && (
                  <div>
                    <div className="carousel-meta-head">
                      <div className="carousel-model-title">
                        <strong className="mono">{activeModel.name}</strong>
                        {activeModel.detail && (
                          <span className="muted" style={{ fontSize: 13 }}>{t(activeModel.detail)}</span>
                        )}
                        <span className="result-meta mono">
                          {[
                            activeModel.paramLabel,
                            `${activeCard.result.latency_ms} ms`,
                            activeCard.result.device,
                            `seed ${activeCard.result.seed}`,
                          ].filter(Boolean).join(' · ')}
                        </span>
                      </div>

                      <div className="result-meta mono">
                        {[
                          activeCard.result.score_kind === 'decision'
                            ? t('scoreKindDecision')
                            : activeCard.result.score_kind === 'softmax_positive'
                            ? t('scoreKindSoftmax')
                            : t('scoreKindSigmoid'),
                          `${t('thresholdPrefix')}${activeCard.result.threshold.toFixed(2)}`,
                          activeCard.result.dataset === 'movie_reviews_zh' &&
                          activeCard.result.family !== 'batch_b_encoder' &&
                          activeCard.result.preprocessing.segmentation_applied && t('segApplied'),
                        ].filter(Boolean).join(' · ')}
                      </div>
                    </div>

                    <div className="carousel-pred-strip">
                      <span className="muted" style={{ fontSize: 12 }}>{t('predictedStripLabel')}</span>
                      {activeCard.result.predicted_labels.length > 0 ? (
                        activeCard.result.predicted_labels.map((lbl) => (
                          <span
                            key={lbl}
                            className={`tone-badge ${lbl === 'negative' ? 'neg' : 'pos'} mono`}
                          >
                            {lbl}
                          </span>
                        ))
                      ) : (
                        <span className="tone-badge mono">{t('belowThreshold')}</span>
                      )}
                    </div>

                    <div
                      className={`wide-score-list${
                        activeCard.result.top_k.length > 10 ? ' two-col' : ''
                      }`}
                    >
                      {activeCard.result.top_k.map((item, idx) => {
                        const isMargin = activeCard.result!.score_kind === 'decision';
                        const pct = isMargin
                          ? Math.max(0, Math.min(100, ((item.score + 2) / 4) * 100))
                          : Math.max(0, Math.min(100, item.score * 100));
                        const threshPct = isMargin
                          ? Math.max(0, Math.min(100, ((item.threshold + 2) / 4) * 100))
                          : Math.max(0, Math.min(100, item.threshold * 100));
                        const deltaStr = `${item.delta >= 0 ? '+' : ''}${item.delta.toFixed(3)}`;

                        return (
                          <div key={idx} className="wide-score-row">
                            <span
                              className="mono"
                              style={{
                                fontWeight: item.active ? 600 : 400,
                                color: item.active ? 'var(--text)' : 'var(--muted)',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {item.label}
                            </span>
                            <div className="t3-track" aria-hidden="true">
                              <div
                                className="t3-fill"
                                style={{
                                  width: `${pct.toFixed(1)}%`,
                                  background: item.active ? 'var(--bar-active)' : 'var(--bar-muted)',
                                }}
                              />
                              <div
                                className="t3-threshold"
                                style={{ left: `calc(${threshPct.toFixed(1)}% - 1px)` }}
                              />
                            </div>
                            <span
                              className="mono"
                              style={{
                                textAlign: 'right',
                                color: item.active ? 'var(--text)' : 'var(--muted)',
                                fontWeight: item.active ? 600 : 400,
                              }}
                            >
                              {isMargin
                                ? `${item.score >= 0 ? '+' : ''}${item.score.toFixed(3)}`
                                : `${(item.score * 100).toFixed(1)}%`}
                            </span>
                            <span
                              className="mono"
                              style={{
                                textAlign: 'right',
                                fontSize: 11,
                                color: item.active ? 'var(--pos)' : 'var(--faint)',
                              }}
                            >
                              {`Δ${deltaStr}`}
                            </span>
                          </div>
                        );
                      })}
                    </div>

                    {activeCard.result.preprocessing.preview_tokens.length > 0 && (
                      <div className="token-preview mono">
                        {`${activeCard.result.preprocessing.token_count} ${t('tokensSuffix')}: ${activeCard.result.preprocessing.preview_tokens
                          .slice(0, 16)
                          .join(' · ')}`}
                      </div>
                    )}
                  </div>
                )}
              </article>
          </>
        )}
      </div>

    </section>
  );
}
