/**
 * SharedAuditReadinessPanel — read-only six-row audit panel for the
 * shared-inspect view.
 *
 * Phase 6B FIX-C: the live `AuditReadinessPanel` is store-coupled
 * (fetches via API on composition-version change, layers in inline-
 * source override, layers in the Phase 5b interpretation-events
 * stylising for the LLM row). The shared-inspect view receives a
 * FROZEN `AuditReadinessSnapshot` — captured at mark-time — and must
 * render the same six-row layout without any live overlays. This
 * component is the read-only renderer that takes the snapshot as a
 * prop, wraps in `<ReadOnlyProvider value={true}>`, and renders one
 * `AuditReadinessRow` per snapshot row.
 *
 * The read-only signal flows through context to every row, forcing
 * the static (non-clickable) variant regardless of row status. No
 * detail-drawer affordance exists in shared mode — the snapshot
 * already includes the row's `detail` text, and reviewers see the
 * same information the owner saw at mark-time.
 */

import { ReadOnlyProvider } from "../../contexts/ReadOnlyContext";
import {
  AuditReadinessRow,
  type RowPresentation,
} from "../audit/AuditReadinessRow";
import type {
  AuditReadinessSnapshot,
  ReadinessRow,
  ReadinessRowId,
  ReadinessStatus,
} from "../../types/api";

/**
 * Default glyph + accessible label for each status. Mirrors
 * AuditReadinessPanel's `statusGlyph` — kept local because the shared
 * panel never applies the Phase 5b stylising and never imports the
 * formatLlmInterpretationsRow path (which exists only for the live
 * composer panel's interpretation-events-coupled rendering).
 */
function statusGlyph(status: ReadinessStatus): { glyph: string; aria: string } {
  switch (status) {
    case "ok":
      return { glyph: "✓", aria: "OK" };
    case "warning":
      return { glyph: "⚠", aria: "Warning" };
    case "error":
      return { glyph: "✗", aria: "Error" };
    case "not_applicable":
      return { glyph: "—", aria: "Not applicable" };
    default: {
      const _exhaustive: never = status;
      throw new Error(`unknown readiness status: ${String(_exhaustive)}`);
    }
  }
}

/**
 * Linda-vocabulary heading per row id. Used only as a fallback when
 * the backend-supplied label is empty (Phase 2A's `Field(min_length=1)`
 * rules this out for live snapshots — but a frozen snapshot from an
 * older mark-time might have any shape, and the exhaustive switch
 * documents which row ids the renderer knows about).
 */
function rowHeading(id: ReadinessRowId): string {
  switch (id) {
    case "validation":
      return "Validation";
    case "plugin_trust":
      return "Plugin trust";
    case "provenance":
      return "Provenance";
    case "retention":
      return "Retention";
    case "llm_interpretations":
      return "LLM interpretations";
    case "secrets":
      return "Secrets";
    default: {
      const _exhaustive: never = id;
      throw new Error(`unknown readiness row id: ${String(_exhaustive)}`);
    }
  }
}

function presentationForSharedRow(row: ReadinessRow): RowPresentation {
  const { glyph, aria } = statusGlyph(row.status);
  return {
    id: row.id,
    status: row.status,
    heading: row.label || rowHeading(row.id),
    summaryText: row.summary,
    glyph,
    ariaStatusLabel: aria,
    // Per-row test ids match the pre-FIX-C inline table to keep any
    // upstream consumer / contract tests stable.
    testId: `shared-inspect-readiness-row-${row.id}`,
  };
}

interface SharedAuditReadinessPanelProps {
  snapshot: AuditReadinessSnapshot;
}

export function SharedAuditReadinessPanel({
  snapshot,
}: SharedAuditReadinessPanelProps): JSX.Element {
  return (
    <ReadOnlyProvider value={true}>
      <section
        aria-label="Audit readiness (read-only shared view)"
        className="audit-readiness audit-readiness--shared"
        data-testid="shared-audit-readiness-panel"
      >
        <header className="audit-readiness-header">
          <h2 className="audit-readiness-title">Audit readiness</h2>
          <p className="audit-readiness-freshness">
            Frozen at the moment of marking ready for review. Composition v
            {snapshot.composition_version}; checked at{" "}
            <time dateTime={snapshot.checked_at}>
              {new Date(snapshot.checked_at).toLocaleString()}
            </time>
            .
          </p>
          {/* Gate legibility (elspeth-088bf83922 T-2, option (a)): a reviewer
              opening this frozen snapshot has no ExecuteButton in view, so
              the per-row "Blocks Run" / "Advisory" badges (rendered by
              AuditReadinessRow below) need this standalone explanation —
              past tense, since the run this snapshot describes may already
              have happened. Same classification the live panel uses; no
              gating behaviour is described or implied here (this view has
              no Run control at all). */}
          <p className="audit-readiness-freshness">
            Rows marked "Blocks Run" had to be clear before this pipeline
            could run; the rest are advisory and did not stop it.
          </p>
        </header>
        <ul
          className="audit-readiness-rows"
          aria-label="Audit readiness rows"
        >
          {snapshot.rows.map((row) => (
            <AuditReadinessRow
              key={row.id}
              row={presentationForSharedRow(row)}
            />
          ))}
        </ul>
      </section>
    </ReadOnlyProvider>
  );
}
