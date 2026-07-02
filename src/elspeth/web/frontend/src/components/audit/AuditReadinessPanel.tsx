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
import { useExecutionStore } from "../../stores/executionStore";
import { useInlineSourceStore } from "../../stores/inlineSourceStore";
import { useInterpretationEventsStore } from "../../stores/interpretationEventsStore";
import { hasCompositionContent } from "../../utils/compositionState";
import { relativeTime } from "../../utils/time";
import type {
  AuditReadinessSnapshot,
  CompositionState,
  ReadinessRow,
  ReadinessRowId,
  ReadinessStatus,
  ValidationResult,
} from "../../types/api";
import { ReadinessRowDetail } from "./ReadinessRowDetail";
import { ExplainDialog } from "./ExplainDialog";
import { AuditReadinessRow, type RowPresentation } from "./AuditReadinessRow";

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

/**
 * Phase 5b.18b.7 — LLM-interpretations row text format.
 *
 * The backend returns one of three statuses (`not_applicable`, `warning`,
 * `ok`) for the `llm_interpretations` row. The frontend re-formats the
 * summary to a stylised form (spec lines 657-674 of
 * `docs/composer/ux-redesign-2026-05/18b-phase-5b-frontend.md`):
 *
 *   | Backend status   | Frontend summary text                                     |
 *   |------------------|-----------------------------------------------------------|
 *   | `not_applicable` | (row hidden) OR "not yet surfaced" when LLM transform     |
 *   |                  | exists but no events yet (frontend-derived F-14 state)    |
 *   | `warning`        | "{P} pending review ({R} resolved)"                       |
 *   | `ok`             | "all {N} resolved"                                        |
 *
 * Opt-out override (separate from backend status):
 *
 *   When `optedOutBySession[sessionId]` is true, the summary becomes
 *   "opted out for this session ({N} drafted, not reviewed)" regardless
 *   of backend status.
 *
 * Counts are sourced from `interpretationEventsStore`:
 *   - P = `Object.keys(pendingBySession[sid]).length`
 *   - R = sum of `resolvedCountBySession[sid]` fields
 *   - N = P + R (total drafted)
 *
 * CLOSED switch over `ReadinessStatus`; the `never` arm prevents silent
 * fallthrough if a future backend extension adds a new status value.
 *
 * @returns null when the row should be HIDDEN (not_applicable with no
 *   LLM-context to surface). The caller must skip rendering the row
 *   entirely on null.
 */
interface LlmInterpretationsRenderInputs {
  status: ReadinessStatus;
  pendingCount: number;
  resolvedCount: number;
  optedOut: boolean;
  hasLlmTransform: boolean;
}

interface LlmInterpretationsRenderOutput {
  /** Visible summary text (replaces backend row.summary). */
  summaryText: string;
  /** Visible glyph (replaces backend status glyph). */
  glyph: string;
  /** Accessible status label (read by SRs before the heading). */
  ariaStatusLabel: string;
}

function formatLlmInterpretationsRow(
  inputs: LlmInterpretationsRenderInputs,
): LlmInterpretationsRenderOutput | null {
  const { status, pendingCount, resolvedCount, optedOut, hasLlmTransform } =
    inputs;
  const total = pendingCount + resolvedCount;

  // Opt-out override is unconditional — it suppresses the status mapping
  // entirely. Per spec lines 665-669, the opt-out summary surfaces even
  // when the backend status is `not_applicable` as long as the session
  // has at least one event in the store. When opt-out is true and there
  // are no events at all, we still show the opt-out line because the
  // user has explicitly chosen to silence the surface — hiding it would
  // create the impression the opt-out preference didn't land.
  if (optedOut) {
    return {
      summaryText: `Opted out for this session (${total} drafted, not reviewed)`,
      glyph: "◎", // ◎ — neutral "circled dot" per spec table row
      ariaStatusLabel: "Opted out",
    };
  }

  switch (status) {
    case "warning":
      return {
        summaryText: `${pendingCount} pending review (${resolvedCount} resolved)`,
        glyph: "⚠", // ⚠
        ariaStatusLabel: "Warning",
      };
    case "ok":
      return {
        summaryText: `all ${total} resolved`,
        glyph: "✓", // ✓
        ariaStatusLabel: "OK",
      };
    case "not_applicable":
      // F-14 frontend-derived state: an LLM transform is present but no
      // events have been surfaced yet. The backend returns
      // `not_applicable` in this case (its summary string says
      // "No interpretation events yet for this composition"); the
      // frontend re-words to match the spec table.
      if (hasLlmTransform && total === 0) {
        return {
          summaryText: "Not yet surfaced",
          glyph: "—", // — (em-dash)
          ariaStatusLabel: "Not yet surfaced",
        };
      }
      // No LLM transform AND no events: hide the row entirely (return
      // null). The caller skips rendering.
      return null;
    case "error":
      // The backend never emits `error` for this row today. Render with a
      // generic error frame so a future backend extension that adds it
      // produces visible output rather than silent emptiness.
      return {
        summaryText: "Interpretation review error",
        glyph: "✗", // ✗
        ariaStatusLabel: "Error",
      };
    default: {
      const _exhaustive: never = status;
      throw new Error(
        `unknown readiness status for llm_interpretations: ${String(_exhaustive)}`,
      );
    }
  }
}

/** True when the composition contains at least one `llm`-plugin transform. */
function compositionHasLlmTransform(state: CompositionState | null): boolean {
  if (state === null) return false;
  return state.nodes.some(
    (n) => n.node_type === "transform" && n.plugin === "llm",
  );
}

function validationResultFromSnapshot(snapshot: AuditReadinessSnapshot): ValidationResult {
  return snapshot.validation_result;
}

function projectMatchingSnapshotToExecution(
  sessionId: string,
  compositionVersion: number,
  setValidationResult: (result: ValidationResult) => void,
): void {
  const currentSnapshot =
    useAuditReadinessStore.getState().snapshotsBySession[sessionId];
  const activeSessionId = useSessionStore.getState().activeSessionId;
  const activeVersion =
    useSessionStore.getState().compositionState?.version ?? null;
  if (
    activeSessionId !== sessionId ||
    activeVersion !== compositionVersion ||
    currentSnapshot?.composition_version !== compositionVersion
  ) {
    return;
  }
  setValidationResult(validationResultFromSnapshot(currentSnapshot));
}

export function AuditReadinessPanel() {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);

  const currentCompositionVersion = compositionState?.version ?? null;
  const snapshot = useAuditReadinessStore((s) => {
    if (!activeSessionId || currentCompositionVersion === null) return undefined;
    const cached = s.snapshotsBySession[activeSessionId];
    return cached?.composition_version === currentCompositionVersion ? cached : undefined;
  });
  const isLoading = useAuditReadinessStore((s) =>
    activeSessionId ? !!s.isLoadingBySession[activeSessionId] : false,
  );
  const error = useAuditReadinessStore((s) =>
    activeSessionId ? s.errorBySession[activeSessionId] ?? null : null,
  );
  const loadSnapshot = useAuditReadinessStore((s) => s.loadSnapshot);
  const setValidationResult = useExecutionStore((s) => s.setValidationResult);

  // Phase 5a Task 7: when an inline_blob source is bound to the active
  // composition, the Provenance row's summary text is replaced with an
  // "Inline content hashed (SHA-256: <prefix>…)" line. The `provenance`
  // discriminant is a PROJECTION of the server-recorded `creation_modality`
  // (Task 2.5), not a frontend computation — the override here only
  // changes the displayed text. Status/heading/clickability still come
  // from the backend-supplied row.
  const inlineSummary = useInlineSourceStore((s) =>
    activeSessionId ? s.getSummary(activeSessionId) : null,
  );

  // Phase 5b.18b.7 — interpretation-event counts feed the `llm_interpretations`
  // row's frontend-stylised summary. We subscribe to the per-session
  // sub-maps (rather than reading via store.getState()) so a resolve / opt-out
  // mutation triggers a re-render of this panel. The selector returns the
  // whole Record so the equality check fires on identity change; the
  // derived counts are computed inside the render below.
  const pendingInterpretationsBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const resolvedInterpretationCountsBySession = useInterpretationEventsStore(
    (s) => s.resolvedCountBySession,
  );
  const optedOutInterpretationsBySession = useInterpretationEventsStore(
    (s) => s.optedOutBySession,
  );

  const compositionHasContent = hasCompositionContent(compositionState);

  useEffect(() => {
    if (!activeSessionId || !compositionState || !compositionHasContent) return;
    let cancelled = false;
    // Fire and forget; store handles errors.
    void loadSnapshot(activeSessionId, compositionState.version).then(() => {
      if (cancelled) return;
      projectMatchingSnapshotToExecution(
        activeSessionId,
        compositionState.version,
        setValidationResult,
      );
    });
    return () => {
      cancelled = true;
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
  }, [activeSessionId, compositionState?.version, compositionHasContent, loadSnapshot, setValidationResult]);

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
  // Stored per-session in auditReadinessStore so the preference survives
  // right-rail remounts. Component-local useState would reset on remount.
  const userExpanded = useAuditReadinessStore((s) =>
    activeSessionId ? (s.userExpandedBySession[activeSessionId] ?? false) : false,
  );
  const setUserExpandedInStore = useAuditReadinessStore((s) => s.setUserExpanded);
  const [selectedRowId, setSelectedRowId] = useState<ReadinessRowId | null>(null);
  const [explainOpen, setExplainOpen] = useState(false);

  const showExpanded = anyActionable || userExpanded;

  if (!activeSessionId || !compositionHasContent) {
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
        aria-busy="true"
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
        aria-busy={isLoading ? "true" : undefined}
      >
        <div role="alert" className="audit-readiness-error">
          {error}
        </div>
        <button
          type="button"
          className="btn audit-readiness-action-btn audit-readiness-action-btn--ghost"
          onClick={() =>
            void loadSnapshot(
              activeSessionId,
              compositionState.version,
              { force: true },
            ).then(() =>
              projectMatchingSnapshotToExecution(
                activeSessionId,
                compositionState.version,
                setValidationResult,
              ),
            )
          }
          disabled={isLoading}
          aria-label="Retry audit readiness check"
        >
          Retry
        </button>
      </section>
    );
  }

  if (!snapshot) {
    return null;
  }

  const checkedText = relativeTime(snapshot.checked_at);

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
          {/* No aria-label here: on a role-less span it is never exposed
              (elspeth-37293a3b7c), and the parent button's aria-label wins
              the name computation anyway. The freshness detail is the
              visible text. */}
          <span className="audit-readiness-summary-meta">
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
            {/* No aria-label: naming is not exposed (and is prohibited) on
                a paragraph role (elspeth-37293a3b7c); the visible text IS
                the freshness statement. */}
            <p className="audit-readiness-freshness">
              Checked {checkedText} · as of v{snapshot.composition_version}
            </p>
          </div>
          <div className="audit-readiness-actions">
            <button
              type="button"
              className="btn audit-readiness-action-btn audit-readiness-action-btn--ghost"
              onClick={() =>
                void loadSnapshot(
                  activeSessionId,
                  compositionState.version,
                  {
                    force: true,
                  },
                ).then(() =>
                  projectMatchingSnapshotToExecution(
                    activeSessionId,
                    compositionState.version,
                    setValidationResult,
                  ),
                )
              }
              aria-label="Refresh audit check now"
            >
              Refresh
            </button>
            <button
              type="button"
              className="btn btn-primary audit-readiness-action-btn"
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
            // Phase 5b.18b.7 — llm_interpretations row uses a
            // frontend-stylised renderer driven by interpretationEventsStore
            // counts (pending / resolved) and the opt-out flag. The
            // formatter returns null when the row should be HIDDEN (no LLM
            // transform + no events); we skip rendering entirely in that
            // case so the row is removed from the list (parallel to the
            // backend's "not_applicable" semantics but with the
            // frontend-derived F-14 "not yet surfaced" override layered on).
            if (row.id === "llm_interpretations") {
              const pendingCount = activeSessionId
                ? Object.keys(
                    pendingInterpretationsBySession[activeSessionId] ?? {},
                  ).length
                : 0;
              const counts = activeSessionId
                ? resolvedInterpretationCountsBySession[activeSessionId]
                : undefined;
              const resolvedCount = counts
                ? counts.accepted_as_drafted + counts.amended + counts.opted_out
                : 0;
              const optedOut = activeSessionId
                ? (optedOutInterpretationsBySession[activeSessionId] ?? false)
                : false;
              const formatted = formatLlmInterpretationsRow({
                status: row.status,
                pendingCount,
                resolvedCount,
                optedOut,
                hasLlmTransform: compositionHasLlmTransform(compositionState),
              });
              if (formatted === null) {
                // Row hidden — no LLM transform AND no events AND not
                // opted out. Skip rendering so the row does not appear.
                return null;
              }
              const heading = row.label || rowHeading(row.id);
              const presentation: RowPresentation = {
                id: row.id,
                status: row.status,
                heading,
                summaryText: formatted.summaryText,
                glyph: formatted.glyph,
                ariaStatusLabel: formatted.ariaStatusLabel,
                extraClassName: "audit-readiness-row--llm-interpretations",
                testId: "audit-readiness-row-llm-interpretations",
              };
              // Clickability mirrors the generic `isActionable` semantics
              // — only warning/error open the detail drawer. The opt-out
              // and "not yet surfaced" overrides land on `not_applicable`
              // statuses, which are not clickable. Users access the
              // session-level opt-out via the chat widget, not via the
              // audit panel.
              return (
                <AuditReadinessRow
                  key={row.id}
                  row={presentation}
                  onSelect={setSelectedRowId}
                />
              );
            }
            const { glyph, aria } = statusGlyph(row.status);
            const heading = row.label || rowHeading(row.id);
            // Phase 5a Task 7: inline-source provenance override. The
            // backend `summary` is replaced (not appended to) with a
            // hash-prefix line when an inline_blob source is bound. The
            // 12-char prefix is a display convenience; the full hash
            // stays in the Tier-1 audit trail (Landscape) — this row is
            // a UI affordance, not the legal record.
            const summaryText =
              row.id === "provenance" && inlineSummary !== null
                ? `Inline content hashed (SHA-256: ${inlineSummary.contentHash.slice(0, 12)}…)`
                : row.summary;
            const presentation: RowPresentation = {
              id: row.id,
              status: row.status,
              heading,
              summaryText,
              glyph,
              ariaStatusLabel: aria,
            };
            return (
              <AuditReadinessRow
                key={row.id}
                row={presentation}
                onSelect={setSelectedRowId}
              />
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
