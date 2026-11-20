import { useCallback, useEffect, useState } from 'react';
import rawData from './data/dashboard-data.json';
import { ThemeProvider } from './theme/ThemeContext';
import { I18nProvider, useI18n } from './i18n/I18nContext';
import { FloatingDock } from './components/FloatingDock';
import { ChapterRail } from './components/ChapterRail';
import { InferenceSection } from './sections/InferenceSection';
import { CapabilitiesSection } from './sections/CapabilitiesSection';
import { TrainingDataSection } from './sections/TrainingDataSection';
import type { ApiHealth, ApiModelInfo, ChapterId, DashboardBundle } from './types/dashboard';

const dashboardData = rawData as unknown as DashboardBundle;
const CHAPTER_IDS: ChapterId[] = ['inference', 'capabilities', 'data'];

function AppShell() {
  const { locale, t } = useI18n();
  const [apiHealth, setApiHealth] = useState<ApiHealth | null>(null);
  const [apiModels, setApiModels] = useState<ApiModelInfo[]>([]);
  const [activeChapter, setActiveChapter] = useState<ChapterId>('inference');
  const [railVisible, setRailVisible] = useState<boolean>(false);

  const refreshApi = useCallback(async () => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 10000);
    try {
      const [hRes, mRes] = await Promise.allSettled([
        fetch('/api/health', { signal: controller.signal }),
        fetch('/api/models', { signal: controller.signal }),
      ]);
      if (hRes.status === 'fulfilled' && hRes.value.ok) {
        setApiHealth((await hRes.value.json()) as ApiHealth);
      } else {
        setApiHealth(null);
      }
      if (mRes.status === 'fulfilled' && mRes.value.ok) {
        const body = (await mRes.value.json()) as { models?: ApiModelInfo[] };
        setApiModels(body.models ?? []);
      }
    } catch {
      setApiHealth(null);
    } finally {
      window.clearTimeout(timer);
    }
  }, []);

  useEffect(() => {
    refreshApi();
  }, [refreshApi]);

  useEffect(() => {
    let rafId = 0;

    const updateScrollState = () => {
      const scrollY = window.scrollY;
      setRailVisible(scrollY >= 240);

      const atBottom =
        window.innerHeight + scrollY >= document.documentElement.scrollHeight - 24;
      if (atBottom) {
        setActiveChapter('data');
        return;
      }

      let current: ChapterId = 'inference';
      const threshold = Math.min(320, window.innerHeight * 0.4);
      for (const id of CHAPTER_IDS) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= threshold) {
          current = id;
        }
      }
      setActiveChapter(current);
    };

    const onScroll = () => {
      if (rafId) return;
      rafId = window.requestAnimationFrame(() => {
        rafId = 0;
        updateScrollState();
      });
    };

    updateScrollState();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    window.addEventListener('hashchange', updateScrollState);
    return () => {
      if (rafId) window.cancelAnimationFrame(rafId);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      window.removeEventListener('hashchange', updateScrollState);
    };
  }, []);

  useEffect(() => {
    document.title =
      locale === 'zh'
        ? 'EmotionLab · 多代文本分类实时对比与评测'
        : 'EmotionLab · Multi-Generation Text Classification';
  }, [locale]);

  return (
    <>
      <a href="#main-content" className="skip-link visually-hidden">
        {t('skipToContent')}
      </a>
      <FloatingDock apiHealth={apiHealth} onRefreshApi={refreshApi} />

      <div className="editorial-shell">
        <ChapterRail visible={railVisible} activeChapter={activeChapter} />
        <main id="main-content" className="main-col" tabIndex={-1}>
          <InferenceSection
            data={dashboardData}
            apiHealth={apiHealth}
            apiModels={apiModels}
            onRefreshApi={refreshApi}
          />
          <CapabilitiesSection data={dashboardData} />
          <TrainingDataSection data={dashboardData} />
          <footer className="site-foot">{t('footerLabel')}</footer>
        </main>
      </div>
    </>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <I18nProvider>
        <AppShell />
      </I18nProvider>
    </ThemeProvider>
  );
}
