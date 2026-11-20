import { useI18n } from '../i18n/I18nContext';
import type { ChapterId } from '../types/dashboard';

interface ChapterRailProps {
  visible: boolean;
  activeChapter: ChapterId;
}

const CHAPTERS: Array<{
  id: ChapterId;
  labelKey: 'chapterInference' | 'chapterCapabilities' | 'chapterData';
}> = [
  { id: 'inference', labelKey: 'chapterInference' },
  { id: 'capabilities', labelKey: 'chapterCapabilities' },
  { id: 'data', labelKey: 'chapterData' },
];

export function ChapterRail({ visible, activeChapter }: ChapterRailProps) {
  const { t } = useI18n();
  const supportsCssScrollTarget =
    typeof CSS !== 'undefined' &&
    typeof CSS.supports === 'function' &&
    CSS.supports('scroll-target-group: auto');

  return (
    <aside className="rail-col">
      <nav
        className="chapter-rail"
        data-visible={visible ? 'true' : 'false'}
        aria-label={t('chapterNavLabel')}
      >
        {CHAPTERS.map((ch) => {
          const active = activeChapter === ch.id;
          const useJsFallbackClass = active && !supportsCssScrollTarget;
          return (
            <a
              key={ch.id}
              href={`#${ch.id}`}
              className={`chapter-link${useJsFallbackClass ? ' target-current' : ''}`}
              aria-current={active ? 'true' : 'false'}
              tabIndex={visible ? undefined : -1}
            >
              {t(ch.labelKey)}
            </a>
          );
        })}
      </nav>
    </aside>
  );
}
