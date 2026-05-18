import { useId } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";

/**
 * Run-button tooltip text used when a pending interpretation event blocks
 * execution. Exported for the corresponding test so the assertion is pinned
 * against the exact string the spec table requires (18b-phase-5b-frontend.md
 * line 705: `title="Resolve pending interpretation first."`).
 */
export const INTERPRETATION_PENDING_RUN_BLOCK_TITLE =
  "Resolve pending interpretation first.";

/**
 * Run-pipeline button (Phase 2C, with Phase 5b.18b.7 interpretation gating).
 *
 * Gating predicate (spec 18b lines 698-722):
 *
 *   isRunBlocked = !optedOut && Object.keys(pendingBySession[sessionId] ?? {}).length > 0
 *
 * When `isRunBlocked` is true the button:
 *   - sets `disabled` so it cannot be activated by pointer or Enter,
 *   - sets `aria-disabled="true"` so AT users hear the disabled state,
 *   - sets `title` to the spec-required tooltip string,
 *   - sets `aria-describedby` pointing to a visually-hidden span with the
 *     same text so screen readers receive the affordance text (a
 *     `title` attribute alone is not reliably announced).
 *
 * The opt-out path is the gate's complement: a session that has opted out
 * of interpretation review runs freely (the backend bakes auto-
 * interpretations directly into prompt templates and records them as
 * `interpretation_source='auto_interpreted_opt_out'` in the audit trail).
 */
export function ExecuteButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const isExecuting = useExecutionStore((s) => s.isExecuting);
  const progress = useExecutionStore((s) => s.progress);
  const execute = useExecutionStore((s) => s.execute);

  // Phase 5b.18b.7 — interpretation-review run-button gating. Subscribe to
  // the per-session sub-maps (not the whole store state) so the button
  // re-renders precisely when its inputs change.
  const pendingInterpretationsBySession = useInterpretationEventsStore(
    (s) => s.pendingBySession,
  );
  const optedOutInterpretationsBySession = useInterpretationEventsStore(
    (s) => s.optedOutBySession,
  );

  const reactId = useId();
  const describedById = `${reactId}-run-block-reason`;

  if (!activeSessionId) return null;

  const optedOut = optedOutInterpretationsBySession[activeSessionId] ?? false;
  const pendingCount = Object.keys(
    pendingInterpretationsBySession[activeSessionId] ?? {},
  ).length;
  const isRunBlocked = !optedOut && pendingCount > 0;

  const canExecute =
    validationResult?.is_valid === true &&
    !isExecuting &&
    progress?.status !== "running" &&
    !isRunBlocked;

  return (
    <>
      <button
        type="button"
        className={`btn side-rail-execute-btn ${canExecute && !isExecuting ? "btn-primary" : ""}`}
        onClick={() => execute(activeSessionId)}
        disabled={!canExecute}
        aria-disabled={!canExecute ? true : undefined}
        aria-label="Run pipeline"
        // `title` only when blocked by pending interpretations — the
        // pre-existing disabled-on-invalid-validation case has its own
        // surface (the audit-readiness panel and inline error banners) and
        // a tooltip here would be redundant.
        title={isRunBlocked ? INTERPRETATION_PENDING_RUN_BLOCK_TITLE : undefined}
        aria-describedby={isRunBlocked ? describedById : undefined}
      >
        {isExecuting ? (
          <>
            <span
              className="spinner"
              role="status"
              aria-label="Starting pipeline"
            />
            Starting...
          </>
        ) : (
          "Run pipeline"
        )}
      </button>
      {/*
        Visually-hidden description for AT users. The `title` attribute on
        the button alone is not reliably announced by all screen readers
        (NVDA reads it; VoiceOver and some JAWS configurations ignore it).
        `aria-describedby` pointing at a hidden span is the WCAG-canonical
        way to surface a "why is this button disabled?" reason.
      */}
      {isRunBlocked && (
        <span id={describedById} className="sr-only">
          {INTERPRETATION_PENDING_RUN_BLOCK_TITLE}
        </span>
      )}
    </>
  );
}
