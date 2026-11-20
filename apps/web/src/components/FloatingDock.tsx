import { useI18n } from '../i18n/I18nContext';
import { useTheme } from '../theme/ThemeContext';
import type { ApiHealth } from '../types/dashboard';

interface FloatingDockProps {
  apiHealth: ApiHealth | null;
  onRefreshApi: () => void;
}

export function FloatingDock({ apiHealth, onRefreshApi }: FloatingDockProps) {
  const { locale, toggleLocale, t } = useI18n();
  const { theme, toggleTheme } = useTheme();

  const apiOnline = Boolean(apiHealth && apiHealth.status === 'ok');
  const qwenOnline = Boolean(apiHealth?.qwen_worker?.online);
  const apiState = !apiOnline ? 'offline' : qwenOnline ? 'online' : 'partial';

  return (
    <aside className="edge-dock" aria-label={t('utilityDockLabel')}>
      <button
        type="button"
        className="status-ghost-btn"
        data-state={apiState}
        onClick={onRefreshApi}
        title={t('refreshApiTitle')}
      >
        <span className="status-dot" aria-hidden="true" />
        <span>{apiOnline ? t('apiOnlineShort') : t('apiOfflineShort')}</span>
      </button>

      <span className="dock-sep" aria-hidden="true" />

      <button
        type="button"
        className="ghost-tool-btn mono"
        onClick={toggleLocale}
        aria-label={t('langSwitchAria')}
      >
        {locale === 'zh' ? 'EN' : '中文'}
      </button>

      <span className="dock-sep" aria-hidden="true" />

      <button
        type="button"
        className="ghost-tool-btn icon"
        onClick={toggleTheme}
        aria-label={theme === 'light' ? t('switchToDark') : t('switchToLight')}
        title={theme === 'light' ? t('switchToDark') : t('switchToLight')}
      >
        {theme === 'light' ? (
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
          </svg>
        ) : (
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.9"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
          </svg>
        )}
      </button>
    </aside>
  );
}
