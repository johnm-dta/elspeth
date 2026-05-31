/**
 * CompletionBar — three-button completion gesture surface (Phase 6B Task 3).
 *
 * Renders three co-equal buttons:
 *
 *   * Save for review  → POSTs mark-ready-for-review, opens dialog with the
 *                        signed share URL.
 *   * Run pipeline     → reuses the existing ExecuteButton primitive (which
 *                        carries the Phase 5b interpretation-gating logic).
 *   * Export YAML      → reuses the existing ExportYamlButton primitive.
 *
 * Per plan 19b §"Scope boundaries": no primary emphasis — all three are
 * co-equal verbs. The "Save for review" button is disabled when the
 * current composition's validation result is invalid (the backend would
 * 409 anyway; the disabled state makes the precondition visible).
 */

import { useShareableReviewStore } from "@/stores/shareableReviewStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { ExecuteButton } from "@/components/sidebar/ExecuteButton";
import { ExportYamlButton } from "@/components/sidebar/ExportYamlButton";

const SAVE_FOR_REVIEW_DISABLED_TITLE =
  "Fix validation errors before sharing for review.";

export function CompletionBar(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const openAndMark = useShareableReviewStore((s) => s.openAndMark);
  const inFlight = useShareableReviewStore((s) => s.inFlight);

  if (!activeSessionId) return null;

  // Validation gate mirrors the backend's mark-time gate (CompositionNotRunnableError).
  // ``validationResult`` is null when no validation has been run yet; treat as
  // not-ready (the user must explicitly run validation first).
  const isValidationReady = validationResult !== null && validationResult.is_valid === true;
  const saveDisabled = !isValidationReady || inFlight;

  return (
    <div
      className="completion-bar"
      role="group"
      aria-label="Composition completion gestures"
      data-testid="completion-bar"
    >
      <button
        type="button"
        className="btn completion-bar-save-for-review"
        onClick={() => {
          // openAndMark resolves asynchronously and persists outcome in the
          // store; no need to await here at the click site.
          void openAndMark(activeSessionId);
        }}
        disabled={saveDisabled}
        aria-disabled={saveDisabled || undefined}
        title={saveDisabled && !isValidationReady ? SAVE_FOR_REVIEW_DISABLED_TITLE : undefined}
        data-testid="completion-bar-save-for-review"
      >
        Save for review
      </button>
      <div data-testid="completion-bar-run-pipeline">
        <ExecuteButton />
      </div>
      <div data-testid="completion-bar-export-yaml">
        <ExportYamlButton />
      </div>
    </div>
  );
}
