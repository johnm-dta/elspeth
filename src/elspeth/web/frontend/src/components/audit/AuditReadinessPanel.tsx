/**
 * AuditReadinessPanel (Phase 2)
 *
 * Persistent right-rail panel showing six rows of audit-readiness state.
 * Auto-fetches on compositionState.version change; collapses to a single
 * "Audit ready ✓" summary when nothing actionable is present.
 *
 * Design spec: docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md
 *
 * The renderer is intentionally exhaustive on ReadinessRowId — the `never`
 * default arm fails the build if a new row is added to the wire schema
 * without a UI case.
 */
import { useEffect, useMemo, useState } from "react";

import { useSessionStore } from "../../stores/sessionStore";
import { useAuditReadinessStore } from "../../stores/auditReadinessStore";
import type {
  ReadinessRow,
  ReadinessRowId,
  ReadinessStatus,
} from "../../types/api";
import { ReadinessRowDetail } from "./ReadinessRowDetail";
import { ExplainDialog } from "./ExplainDialog";

/** Glyph + accessible label for each row status. */
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

/** Linda-vocabulary heading for each row id. The wire schema's `label` is
 *  authoritative; this map is the fallback when the backend label is empty,
 *  which Phase 2A's `Field(min_length=1)` rules out — but the renderer must
 *  be exhaustive on the id type regardless. */
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

function isActionable(status: ReadinessStatus): boolean {
  return status === "warning" || status === "error";
}

export function AuditReadinessPanel() {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);

  const snapshot = useAuditReadinessStore((s) =>
    activeSessionId ? s.snapshotsBySession[activeSessionId] : undefined,
  );
  const isLoading = useAuditReadinessStore((s) =>
    activeSessionId ? !!s.isLoadingBySession[activeSessionId] : false,
  );
  const error = useAuditReadinessStore((s) =>
    activeSessionId ? s.errorBySession[activeSessionId] ?? null : null,
  );
  const loadSnapshot = useAuditReadinessStore((s) => s.loadSnapshot);

  const hasCompositionContent =
    !!compositionState &&
    (compositionState.source !== null ||
      compositionState.nodes.length > 0 ||
      compositionState.outputs.length > 0);

  useEffect(() => {
    if (!activeSessionId || !compositionState || !hasCompositionContent) return;
    // Fire and forget; store handles errors.
    void loadSnapshot(activeSessionId, compositionState.version);
    return () => {
      // Unmount-during-fetch cleanup: abort the in-flight controller for this
      // session. The store's AbortError catch arm clears
      // isLoadingBySession[activeSessionId] and preserves cached snapshot/error
      // (see issue elspeth-f018ea84c6 — the prior synchronous setState here
      // was a workaround for signal-blind test mocks; Task 4A Step 2 fixed
      // that at the mock layer).
      const ctrl = useAuditReadinessStore.getState().abortControllers[activeSessionId];
      if (ctrl) {
        ctrl.abort();
      }
    };
  // Intentional: `compositionState?.version` is the dep, not the compositionState reference.
  // Using the reference would re-run the effect on every render-cycle that re-creates the object
  // without changing the version. The linter flags `compositionState` as missing; suppress here.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId, compositionState?.version, hasCompositionContent, loadSnapshot]);

  const anyActionable = useMemo(
    () => snapshot?.rows.some((r) => isActionable(r.status)) ?? false,
    [snapshot],
  );

  const [expanded, setExpanded] = useState(false);
  const [selectedRowId, setSelectedRowId] = useState<ReadinessRowId | null>(null);
  const [explainOpen, setExplainOpen] = useState(false);

  // When the snapshot changes and contains a warning/error, force expansion.
  useEffect(() => {
    if (anyActionable) setExpanded(true);
  }, [anyActionable]);

  if (!activeSessionId || !hasCompositionContent) {
    return null;
  }

  if (isLoading && !snapshot) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--loading"
      >
        <span className="audit-readiness-loading">
          Checking audit readiness…
        </span>
      </section>
    );
  }

  if (error && !snapshot) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--error"
      >
        <div role="alert" className="audit-readiness-error">
          {error}
        </div>
      </section>
    );
  }

  if (!snapshot) {
    return null;
  }

  // Collapsed view — single summary line when nothing is actionable.
  if (!expanded && !anyActionable) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--collapsed"
      >
        <button
          type="button"
          className="audit-readiness-summary"
          onClick={() => setExpanded(true)}
          aria-label="Audit ready. Show details."
        >
          <span aria-hidden="true">{"✓"}</span> Audit ready
        </button>
      </section>
    );
  }

  return (
    <>
      <section aria-label="Audit readiness" className="audit-readiness">
        <header className="audit-readiness-header">
          <h2 className="audit-readiness-title">Audit readiness</h2>
          <div className="audit-readiness-actions">
            <button
              type="button"
              className="btn audit-readiness-action-btn"
              onClick={() => setExplainOpen(true)}
              aria-label="Explain what this pipeline will record"
            >
              Explain →
            </button>
            {!anyActionable && (
              <button
                type="button"
                className="btn audit-readiness-action-btn audit-readiness-action-btn--ghost"
                onClick={() => setExpanded(false)}
                aria-label="Collapse audit readiness"
              >
                Collapse
              </button>
            )}
          </div>
        </header>

        <ul id="audit-readiness-rows" className="audit-readiness-rows">
          {snapshot.rows.map((row: ReadinessRow) => {
            const { glyph, aria } = statusGlyph(row.status);
            const heading = row.label || rowHeading(row.id);
            const clickable = isActionable(row.status);
            return (
              <li
                key={row.id}
                className={`audit-readiness-row audit-readiness-row--${row.status}`}
              >
                {clickable ? (
                  <button
                    type="button"
                    className="audit-readiness-row-btn"
                    onClick={() => setSelectedRowId(row.id)}
                    aria-label={heading}
                  >
                    <span
                      className="audit-readiness-glyph"
                      aria-hidden="true"
                    >
                      {glyph}
                    </span>
                    <span className="sr-only">{aria}.</span>
                    <span className="audit-readiness-row-label">{heading}</span>
                    <span className="audit-readiness-row-summary">{row.summary}</span>
                  </button>
                ) : (
                  <div
                    className="audit-readiness-row-static"
                    role="group"
                    aria-label={heading}
                  >
                    <span
                      className="audit-readiness-glyph"
                      aria-hidden="true"
                    >
                      {glyph}
                    </span>
                    <span className="sr-only">{aria}.</span>
                    <span className="audit-readiness-row-label">{heading}</span>
                    <span className="audit-readiness-row-summary">{row.summary}</span>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {selectedRowId && (
        <ReadinessRowDetail
          row={snapshot.rows.find((r) => r.id === selectedRowId)!}
          onClose={() => setSelectedRowId(null)}
        />
      )}

      {explainOpen && (
        <ExplainDialog
          sessionId={activeSessionId}
          compositionVersion={snapshot.composition_version}
          onClose={() => setExplainOpen(false)}
        />
      )}
    </>
  );
}
