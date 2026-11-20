import React, { useId, useRef } from 'react';

export interface PillOption<K extends string> {
  id: K;
  label: string;
}

interface SegmentedPillProps<K extends string> {
  ariaLabel: string;
  options: PillOption<K>[];
  activeId: K;
  onChange: (next: K) => void;
  size?: 'md' | 'sm';
  centered?: boolean;
}

export function SegmentedPill<K extends string>({
  ariaLabel,
  options,
  activeId,
  onChange,
  size = 'md',
  centered = false,
}: SegmentedPillProps<K>) {
  const baseId = useId();
  const btnRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const handleKeyDown = (idx: number, event: React.KeyboardEvent<HTMLButtonElement>) => {
    let target = idx;
    if (event.key === 'ArrowRight') {
      target = (idx + 1) % options.length;
    } else if (event.key === 'ArrowLeft') {
      target = (idx - 1 + options.length) % options.length;
    } else if (event.key === 'Home') {
      target = 0;
    } else if (event.key === 'End') {
      target = options.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    const next = options[target];
    if (next) {
      onChange(next.id);
      btnRefs.current[target]?.focus();
    }
  };

  return (
    <div className={centered ? 'pill-row centered' : undefined}>
      <div
        className={`pill-switch${size === 'sm' ? ' sm' : ''}`}
        role="tablist"
        aria-label={ariaLabel}
      >
        {options.map((opt, idx) => {
          const selected = opt.id === activeId;
          return (
            <button
              key={opt.id}
              ref={(el) => {
                btnRefs.current[idx] = el;
              }}
              id={`${baseId}-pill-${opt.id}`}
              type="button"
              role="tab"
              aria-selected={selected}
              tabIndex={selected ? 0 : -1}
              className="pill-tab"
              onClick={() => onChange(opt.id)}
              onKeyDown={(e) => handleKeyDown(idx, e)}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
