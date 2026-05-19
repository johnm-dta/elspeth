/**
 * AuditReadinessRow — single row of the audit-readiness panel.
 *
 * Extracted from AuditReadinessPanel (Phase 6B FIX-C, plan Step 0)
 * so the same row primitive can render inside both the live composer
 * panel and the read-only `SharedAuditReadinessPanel`. The panel
 * passes a fully-prepared `RowPresentation` so the row component
 * stays a pure renderer — all the LLM-interpretations / inline-blob /
 * provenance overrides remain in the parent (where they belong: they
 * depend on multiple stores).
 *
 * Read-only signalling: the component consumes `useReadOnly()` and
 * forces the static (non-actionable) variant when true, regardless of
 * what `onSelect` the caller supplied. This is the load-bearing
 * contract that wraps a `SharedAuditReadinessPanel` subtree under
 * `<ReadOnlyProvider value={true}>` and gets actionability disabled
 * everywhere downstream. The live AuditReadinessPanel never wraps in
 * a provider (the default `false` value flows through) so its
 * actionable buttons render as today.
 */

import { useReadOnly } from "../../contexts/ReadOnlyContext";
import type { ReadinessRowId, ReadinessStatus } from "../../types/api";

/**
 * Prepared presentation values for a single row. Callers pre-compute
 * any glyph / aria / summary overrides (e.g. the llm_interpretations
 * stylised formatter, the inline-blob provenance override) and pass
 * the final values in. This keeps the row a pure renderer.
 */
export interface RowPresentation {
  id: ReadinessRowId;
  status: ReadinessStatus;
  /** Pre-resolved heading text (label or rowHeading() fallback). */
  heading: string;
  /** Pre-resolved summary text (after any overrides). */
  summaryText: string;
  /** Pre-resolved glyph character. */
  glyph: string;
  /** Accessible status label, read by SRs before the heading. */
  ariaStatusLabel: string;
  /**
   * Optional extra CSS modifier appended to the row's class list (e.g.
   * "audit-readiness-row--llm-interpretations"). Optional — most rows
   * don't need one.
   */
  extraClassName?: string;
  /**
   * Optional test id for the wrapping <li>. The live panel sets this
   * on the llm-interpretations row; other rows omit it.
   */
  testId?: string;
}

export interface AuditReadinessRowProps {
  /** The presentation values to render. */
  row: RowPresentation;
  /**
   * Click handler for actionable rows. When omitted OR when
   * `useReadOnly()` returns true, the row renders in its static
   * (non-clickable) variant. This is the read-only contract.
   */
  onSelect?: (rowId: ReadinessRowId) => void;
}

export function AuditReadinessRow({
  row,
  onSelect,
}: AuditReadinessRowProps): JSX.Element {
  const readOnly = useReadOnly();
  // Clickability requires (a) the row's status is actionable, (b) a
  // handler was provided, AND (c) the read-only signal is false. In
  // read-only mode every row renders static regardless of status —
  // shared inspect surfaces have no detail-drawer affordance.
  const isActionable = row.status === "warning" || row.status === "error";
  const clickable = isActionable && onSelect !== undefined && !readOnly;
  const baseClassName = `audit-readiness-row audit-readiness-row--${row.status}`;
  const className = row.extraClassName
    ? `${baseClassName} ${row.extraClassName}`
    : baseClassName;

  if (clickable) {
    return (
      <li
        key={row.id}
        className={className}
        data-testid={row.testId}
      >
        <button
          type="button"
          className="audit-readiness-row-btn"
          onClick={() => onSelect(row.id)}
        >
          <span className="audit-readiness-glyph" aria-hidden="true">
            {row.glyph}
          </span>
          <span className="sr-only">{row.ariaStatusLabel}.</span>
          <span className="audit-readiness-row-label">{row.heading}</span>
          <span className="audit-readiness-row-summary">{row.summaryText}</span>
        </button>
      </li>
    );
  }

  return (
    <li key={row.id} className={className} data-testid={row.testId}>
      <div
        className="audit-readiness-row-static"
        role="group"
        aria-label={row.heading}
      >
        <span className="audit-readiness-glyph" aria-hidden="true">
          {row.glyph}
        </span>
        <span className="sr-only">{row.ariaStatusLabel}.</span>
        <span className="audit-readiness-row-label">{row.heading}</span>
        <span className="audit-readiness-row-summary">{row.summaryText}</span>
      </div>
    </li>
  );
}
