import React, { useId, useMemo, useState } from 'react';

export interface TableColumn<T> {
  key: string;
  label: string;
  align?: 'left' | 'right';
  type?: 'text' | 'number';
  sortable?: boolean;
  isRowHeader?: boolean;
  sortValue?: (row: T) => string | number | null | undefined;
  render: (row: T, rowIndex: number) => React.ReactNode;
}

interface SortableTableProps<T> {
  caption: string;
  hideCaption?: boolean;
  columns: TableColumn<T>[];
  rows: T[];
  getRowKey: (row: T) => string;
  defaultSortKey?: string;
  defaultDirection?: 'ascending' | 'descending';
  highlightRow?: (row: T) => boolean;
  responsiveStack?: boolean;
  emptyText?: string;
}

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

export function SortableTable<T>({
  caption,
  hideCaption = false,
  columns,
  rows,
  getRowKey,
  defaultSortKey,
  defaultDirection = 'descending',
  highlightRow,
  responsiveStack = true,
  emptyText = 'No matching rows',
}: SortableTableProps<T>) {
  const captionId = useId();
  const [sortKey, setSortKey] = useState<string | null>(defaultSortKey ?? null);
  const [direction, setDirection] = useState<'ascending' | 'descending'>(defaultDirection);
  const [statusMessage, setStatusMessage] = useState('');

  const sortedRows = useMemo(() => {
    if (!sortKey) return rows;
    const col = columns.find((c) => c.key === sortKey);
    if (!col || col.sortable === false) return rows;

    const factor = direction === 'ascending' ? 1 : -1;
    const getValue = col.sortValue ?? ((r: T) => (r as Record<string, unknown>)[col.key] as string | number);
    return [...rows].sort((a, b) => {
      const va = getValue(a);
      const vb = getValue(b);
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      if (col.type === 'number') {
        return factor * (Number(va) - Number(vb));
      }
      return factor * collator.compare(String(va), String(vb));
    });
  }, [rows, columns, sortKey, direction]);

  const handleSort = (col: TableColumn<T>) => {
    const nextDir: 'ascending' | 'descending' =
      sortKey === col.key && direction === 'ascending' ? 'descending' : 'ascending';
    setSortKey(col.key);
    setDirection(nextDir);
    setStatusMessage(`Sorted by ${col.label}, ${nextDir}, ${rows.length} rows`);
  };

  return (
    <div>
      <div role="status" className="visually-hidden" aria-live="polite">
        {statusMessage}
      </div>
      <div
        tabIndex={0}
        role="region"
        aria-labelledby={captionId}
        className="table-wrapper"
      >
        <table className={`data-table${responsiveStack ? ' responsive-stack' : ''}`}>
          <caption id={captionId} className={hideCaption ? 'visually-hidden' : undefined}>{caption}</caption>
          <thead>
            <tr>
              {columns.map((col) => {
                const isSorted = sortKey === col.key;
                const sortable = col.sortable !== false;
                return (
                  <th
                    key={col.key}
                    scope="col"
                    className={col.align === 'right' ? 'num' : undefined}
                    aria-sort={isSorted ? direction : undefined}
                  >
                    {sortable ? (
                      <button
                        type="button"
                        className={`sort-btn${col.align === 'right' ? ' num' : ''}`}
                        onClick={() => handleSort(col)}
                      >
                        <span>{col.label}</span>
                        <span className="sort-indicator" aria-hidden="true" />
                      </button>
                    ) : (
                      col.label
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sortedRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="muted" style={{ textAlign: 'center', padding: '18px' }}>
                  {emptyText}
                </td>
              </tr>
            ) : (
              sortedRows.map((row, rIdx) => {
                const isHot = highlightRow ? highlightRow(row) : false;
                return (
                  <tr key={getRowKey(row)} className={isHot ? 'highlight' : undefined}>
                    {columns.map((col) =>
                      col.isRowHeader ? (
                        <th key={col.key} scope="row">
                          {col.render(row, rIdx)}
                        </th>
                      ) : (
                        <td
                          key={col.key}
                          className={col.align === 'right' ? 'num' : undefined}
                          data-col-label={col.label}
                        >
                          {col.render(row, rIdx)}
                        </td>
                      )
                    )}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
