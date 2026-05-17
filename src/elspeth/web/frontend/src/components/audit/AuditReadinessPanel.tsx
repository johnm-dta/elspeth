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
import { relativeTime } from "../../utils/time";
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
      // isLoadingBySession[activeSessionId] and preserves cached snapshot/error.
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

  // Tracks the user's explicit expand/collapse intent. Auto-expansion on
  // actionable snapshots is computed atomically as `anyActionable ||
  // userExpanded` rather than synced through a useEffect — this avoids the
  // extra render cycle a derived-state effect would cause, and makes the
  // panel auto-collapse when a later snapshot returns all-green (unless the
  // user explicitly clicked Expand).
  //
  // Stored per-session in auditReadinessStore so the preference survives the
  // Phase 3B remount of <AuditReadinessPanel /> from InspectorPanel to
  // SideRail.auditReadinessSlot. Component-local useState would reset on
  // remount.
  const userExpanded = useAuditReadinessStore((s) =>
    activeSessionId ? (s.userExpandedBySession[activeSessionId] ?? false) : false,
  );
  const setUserExpandedInStore = useAuditReadinessStore((s) => s.setUserExpanded);
  const [selectedRowId, setSelectedRowId] = useState<ReadinessRowId | null>(null);
  const [explainOpen, setExplainOpen] = useState(false);

  const showExpanded = anyActionable || userExpanded;

  if (!activeSessionId || !hasCompositionContent) {
    return null;
  }
  if (!compositionState) {
    throw new Error("compositionState missing after audit-readiness content guard");
  }

  if (isLoading && !snapshot) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--loading"
      >
        <div
          className="audit-readiness-live-region"
          aria-live="polite"
          aria-atomic="false"
        >
          <span className="audit-readiness-loading">
            Checking audit readiness…
          </span>
        </div>
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

  const checkedText = relativeTime(snapshot.checked_at);
  const freshnessLabel = `Audit readiness checked ${checkedText} as of v${snapshot.composition_version}`;

  // Collapsed view — single summary line when nothing is actionable.
  if (!showExpanded) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--collapsed"
        aria-busy={isLoading ? "true" : undefined}
      >
        <button
          type="button"
          className="audit-readiness-summary"
          onClick={() => setUserExpandedInStore(activeSessionId, true)}
          aria-expanded={false}
          aria-label="Audit ready. Show details."
        >
          <span aria-hidden="true">{"✓"}</span> Audit ready
          <span
            className="audit-readiness-summary-meta"
            aria-label={freshnessLabel}
          >
            Checked {checkedText} · as of v{snapshot.composition_version}
          </span>
        </button>
      </section>
    );
  }

  return (
    <>
      <section
        aria-label="Audit readiness"
        className="audit-readiness"
        aria-busy={isLoading ? "true" : undefined}
      >
        <header className="audit-readiness-header">
          <div>
            <h2 className="audit-readiness-title">Audit readiness</h2>
            <p
              className="audit-readiness-freshness"
              aria-label={freshnessLabel}
            >
              Checked {checkedText} · as of v{snapshot.composition_version}
            </p>
          </div>
          <div className="audit-readiness-actions">
            <button
              type="button"
              className="btn audit-readiness-action-btn audit-readiness-action-btn--ghost"
              onClick={() =>
                void loadSnapshot(activeSessionId, compositionState.version, {
                  force: true,
                })
              }
              aria-label="Refresh audit check now"
            >
              Refresh
            </button>
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
                onClick={() => setUserExpandedInStore(activeSessionId, false)}
                aria-label="Collapse audit readiness"
              >
                Collapse
              </button>
            )}
          </div>
        </header>

        <ul
          id="audit-readiness-rows"
          className="audit-readiness-rows"
          aria-live="polite"
          aria-atomic="false"
        >
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
