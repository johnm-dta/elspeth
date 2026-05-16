/**
 * Placeholder shipped by 14b. 14c replaces this with the full implementation
 * (per-row warning detail + jump-to-component, with proper dialog semantics:
 * focus trap, aria-modal, initial focus, focus restoration).
 *
 * This placeholder is intentionally NOT a dialog. It renders a minimal stub
 * tagged with `data-testid="readinessrowdetail-placeholder"` so the panel's
 * tests can assert the component mounted, lint/typecheck pass, and the W2
 * accessibility defect (role="dialog" without focus management) does not ship.
 * The `onClose` callback is wired up so the parent's close-on-trigger flow is
 * already exercised — 14c only needs to add the modal semantics.
 *
 * DO NOT extend the placeholder. Extensions (real dialog markup with focus
 * trap, aria-modal, escape-to-close, backdrop dismiss) belong in 14c.
 */
import type { ReadinessRow } from "../../types/api";

export interface ReadinessRowDetailProps {
  row: ReadinessRow;
  onClose: () => void;
}

export function ReadinessRowDetail({ row, onClose }: ReadinessRowDetailProps) {
  return (
    <div
      data-testid="readinessrowdetail-placeholder"
      aria-label={row.label}
      className="readiness-row-detail"
    >
      <h3>{row.label}</h3>
      <p>{row.summary}</p>
      {row.detail && <pre>{row.detail}</pre>}
      <button type="button" onClick={onClose}>
        Close
      </button>
    </div>
  );
}
