import { useRef, useState } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import { RecoveryDiff } from "./RecoveryDiff";
import { RecoveryTranscript } from "./RecoveryTranscript";
import type { CompositionState, ComposerRecoveryError } from "@/types/api";

interface ApplyRecoveredStateOptions {
  confirmed?: boolean;
}

interface ApplyRecoveredStateResult {
  applied: boolean;
  needsConfirmation: boolean;
}

interface RecoveryPanelProps {
  activeSessionId: string | null;
  currentState: CompositionState | null;
  recoveryError: ComposerRecoveryError | null;
  onApply: (options?: ApplyRecoveredStateOptions) => ApplyRecoveredStateResult;
  onDiscard: () => void;
}

function reasonLabel(errorType: string | undefined): string {
  if (!errorType) {
    return "Composer failed";
  }
  const normalized = errorType.replace(/^composer_/, "composer_").replace(/_/g, " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function countLabel(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function RecoveryPanel({
  activeSessionId,
  currentState,
  recoveryError,
  onApply,
  onDiscard,
}: RecoveryPanelProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [needsConfirmation, setNeedsConfirmation] = useState(false);
  const [showTranscriptControls, setShowTranscriptControls] = useState(false);
  useFocusTrap(dialogRef, recoveryError !== null, ".recovery-panel-apply");

  if (recoveryError === null || activeSessionId === null) {
    return null;
  }

  const reason = reasonLabel(recoveryError.error_type);
  const failedTurn = recoveryError.failed_turn;

  function requestApply() {
    const result = onApply();
    setNeedsConfirmation(result.needsConfirmation);
  }

  function confirmApply() {
    const result = onApply({ confirmed: true });
    if (result.applied) {
      setNeedsConfirmation(false);
    }
  }

  return (
    <>
      <div
        className="recovery-panel-backdrop"
        role="presentation"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="recovery-panel-title"
        className="recovery-panel"
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Enter" && event.target === event.currentTarget) {
            event.preventDefault();
          }
        }}
      >
        <header className="recovery-panel-header">
          <div>
            <h2 id="recovery-panel-title">Recover partial composer draft</h2>
            <p>{recoveryError.detail}</p>
          </div>
          {/* The "Recovery reason:" context is a visually-hidden prefix —
              the previous aria-label sat on a role-less span, which AT
              ignores (WCAG 1.3.1, elspeth-37293a3b7c). */}
          <span className="recovery-panel-reason">
            <span className="sr-only">Recovery reason: </span>
            {reason}
          </span>
        </header>

        <section className="recovery-panel-evidence" aria-label="Recovery evidence">
          <span>
            {countLabel(
              failedTurn.tool_calls_attempted,
              "tool call attempted",
              "tool calls attempted",
            )}
          </span>
          <span>
            {countLabel(
              failedTurn.tool_responses_persisted,
              "tool response persisted",
              "tool responses persisted",
            )}
          </span>
        </section>

        {needsConfirmation ? (
          <div className="recovery-panel-confirm" role="alert">
            <p>
              The current pipeline changed after this failed turn started.
              Applying the partial draft will replace the current draft.
            </p>
            <div className="recovery-panel-confirm-actions">
              <button
                className="btn btn-secondary"
                type="button"
                onClick={() => setNeedsConfirmation(false)}
              >
                Cancel
              </button>
              <button
                className="btn btn-danger"
                type="button"
                onClick={confirmApply}
              >
                Apply anyway
              </button>
            </div>
          </div>
        ) : null}

        <div className="recovery-panel-body">
          <RecoveryDiff
            currentState={currentState}
            recoveredState={recoveryError.partial_state}
          />
          <div className="recovery-panel-transcript-controls">
            <button
              className="btn btn-secondary"
              type="button"
              onClick={() =>
                setShowTranscriptControls((currentlyShown) => !currentlyShown)
              }
            >
              View raw transcript controls
            </button>
            {showTranscriptControls ? (
              <p>
                Transcript rows are loaded from the audit view with tool rows
                only; raw provider payloads are not requested.
              </p>
            ) : null}
          </div>
          <RecoveryTranscript
            sessionId={activeSessionId}
            failedTurn={failedTurn}
          />
        </div>

        <footer className="recovery-panel-actions">
          <button
            className="btn btn-danger recovery-panel-discard"
            type="button"
            onClick={onDiscard}
          >
            Discard recovery
          </button>
          <button
            className="btn btn-primary recovery-panel-apply"
            type="button"
            onClick={requestApply}
          >
            Apply partial draft
          </button>
        </footer>
      </div>
    </>
  );
}
