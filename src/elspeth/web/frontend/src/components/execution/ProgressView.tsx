// ============================================================================
// ProgressView
//
// Live progress display for an active execution run. Shows:
// - Indeterminate progress bar (using .progress-bar CSS classes from styles/animations.css)
// - Explicit source/token counters
// - Recent errors list (scrolling, newest first, capped at 50)
// - Cancel button (disabled once run reaches terminal state)
// - "Pipeline execution was cancelled" message on cancelled event
// - WebSocket disconnect banner with reconnect status
// ============================================================================

import { useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useWebSocket } from "@/hooks/useWebSocket";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { Button, StatusBadge } from "@/components/ui";
import type { RunAccounting, RunProgress } from "@/types/index";

function formattedCount(value: number): string {
  return value.toLocaleString();
}

// M07 (WCAG 4.1.3): phase-driven announcement for the polite live region.
// Derived from ``progress.status`` (the phase) rather than the counters, so the
// string is stable across counter ticks while running and only changes on a
// terminal transition — the live region then announces the run once per phase
// instead of on every update.
function buildStatusAnnouncement(progress: RunProgress, cancelRequested: boolean): string {
  if (cancelRequested) return "Cancelling pipeline.";
  const totals =
    `${formattedCount(progress.source_rows_processed)} rows, ` +
    `${formattedCount(progress.tokens_succeeded)} succeeded, ` +
    `${formattedCount(progress.tokens_failed)} failed`;
  switch (progress.status) {
    case "completed":
      return `Pipeline completed — ${totals}.`;
    case "completed_with_failures":
      return `Pipeline completed with failures — ${totals}.`;
    case "failed":
      return `Pipeline failed — ${totals}.`;
    case "empty":
      return "Pipeline completed — no rows processed.";
    case "cancelled":
      return "Pipeline execution was cancelled.";
    case "pending":
      // Distinct from "running" so the polite live region announces the
      // pending→running transition (a queued run has not started yet).
      return "Pipeline queued.";
    case "running":
      return "Pipeline running.";
  }
}

function ProgressAccountingDetails({ accounting }: { accounting: RunAccounting }) {
  const accountingCounters = [
    ["Tokens Emitted", accounting.tokens.emitted],
    ["Tokens Terminal", accounting.tokens.terminal],
    ["Tokens Structural", accounting.tokens.structural],
    ["Tokens Pending", accounting.tokens.pending],
    ["Rows Discarded", accounting.routing.discarded],
  ] as const;
  const integrityWarnings = [
    ["Missing Terminal", accounting.integrity.missing_terminal_outcomes],
    ["Duplicate Terminal", accounting.integrity.duplicate_terminal_outcomes],
  ] as const;

  return (
    <div role="group" aria-label="Run accounting" className="progress-accounting">
      <dl className="progress-accounting-grid">
        {accountingCounters.map(([label, value]) => (
          <div key={label} className="progress-accounting-item">
            <dt>{label}</dt>
            <dd>{formattedCount(value)}</dd>
          </div>
        ))}
      </dl>
      <div className="progress-accounting-integrity">
        <span className="progress-accounting-integrity-item">
          <span className="progress-accounting-integrity-label">Audit Closure</span>
          <strong>{accounting.integrity.closure}</strong>
        </span>
        {integrityWarnings.map(([label, value]) =>
          value > 0 ? (
            <span key={label} className="progress-accounting-integrity-item progress-accounting-integrity-item--warning">
              <span className="progress-accounting-integrity-label">{label}</span>
              <strong>{formattedCount(value)}</strong>
            </span>
          ) : null,
        )}
      </div>
    </div>
  );
}

export function ProgressView() {
  const { progress, wsDisconnected, activeRunId } = useWebSocket();
  const cancel = useExecutionStore((s) => s.cancel);
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);

  if (!progress || !activeRunId) return null;

  // Phase 2.2 (elspeth-0de989c56d): the operator-completion subset
  // (completed / completed_with_failures / empty) is also terminal — the
  // older 3-value tuple ("completed", "cancelled", "failed") would leave the
  // progress view appearing active for the two new terminal statuses.
  const isTerminal =
    progress.status === "completed" ||
    progress.status === "completed_with_failures" ||
    progress.status === "empty" ||
    progress.status === "cancelled" ||
    progress.status === "failed";
  const cancelRequested = progress.cancel_requested === true && !isTerminal;
  const statusAnnouncement = buildStatusAnnouncement(progress, cancelRequested);
  const displayStatus = cancelRequested ? ("cancelling" as const) : progress.status;

  return (
    <div className="progress-container">
      {/* Single phase-driven live region (M07 / WCAG 4.1.3): announces the run
          phase (running → each terminal status) once per transition. The
          visible cancelled/failed messages below are visual-only (no live
          role) so a screen reader hears each terminal status exactly once,
          politely, via this region rather than from a competing live region. */}
      <div role="status" aria-live="polite" className="sr-only">
        {statusAnnouncement}
      </div>

      {/* WebSocket disconnect banner */}
      {wsDisconnected && !isTerminal && (
        <div
          role="status"
          className="progress-ws-banner"
        >
          Live progress connection lost. Reconnecting...
        </div>
      )}

      {/* Status header with cancel button. StatusBadge carries the a11y glyph
          map (⚠ completed_with_failures / ∅ empty) so the distinction is not
          colour-only (elspeth-e1c5ad0b53); the underscored identifier is
          rendered space-separated for human reading and the badge CSS handles
          uppercasing. */}
      <div className="progress-status-header">
        <StatusBadge status={displayStatus}>
          {displayStatus.replace(/_/g, " ")}
        </StatusBadge>
        {!isTerminal && !cancelRequested && (
          <Button
            variant="danger"
            onClick={() => setShowCancelConfirm(true)}
            aria-label="Cancel pipeline execution"
            className="progress-cancel-btn"
          >
            Cancel
          </Button>
        )}
      </div>

      {showCancelConfirm && (
        <ConfirmDialog
          title="Cancel pipeline"
          message="Cancel the running pipeline? This cannot be undone."
          confirmLabel="Cancel pipeline"
          variant="danger"
          onConfirm={() => {
            cancel(activeRunId);
            setShowCancelConfirm(false);
          }}
          onCancel={() => setShowCancelConfirm(false)}
        />
      )}

      {/* Progress bar -- indeterminate mode (no percentage, animated stripe) */}
      <div
        className="progress-bar progress-bar-outer"
        role="progressbar"
        aria-label="Pipeline execution in progress"
      >
        <div
          className={isTerminal ? "progress-bar-complete" : "progress-bar-stripe"}
          style={
            isTerminal
              ? {
                  // Phase 2.2 colour mapping:
                  //   completed                  → success (green)
                  //   completed_with_failures    → warning (the run produced
                  //                                output but had failures)
                  //   empty                      → muted text (clean run that
                  //                                consumed no rows)
                  //   failed                     → error (red)
                  //   cancelled                  → warning (orange)
                  backgroundColor:
                    progress.status === "completed"
                      ? "var(--color-success)"
                      : progress.status === "failed"
                        ? "var(--color-error)"
                        : progress.status === "empty"
                          ? "var(--color-text-muted)"
                          : "var(--color-warning)",
                }
              : {}
          }
        />
      </div>

      {/* Source/token counters -- large and prominent */}
      <div className="progress-counters">
        <div>
          <div className="progress-counter-label">
            Source Rows
          </div>
          <div className="progress-counter-value">
            {progress.source_rows_processed.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="progress-counter-label">
            Tokens Succeeded
          </div>
          <div className="progress-counter-value">
            {progress.tokens_succeeded.toLocaleString()}
          </div>
        </div>
        <div>
          <div className="progress-counter-label">
            Tokens Failed
          </div>
          <div
            className="progress-counter-value"
            style={{
              color:
                progress.tokens_failed > 0
                  ? "var(--color-error)"
                  : "var(--color-text)",
            }}
          >
            {progress.tokens_failed.toLocaleString()}
          </div>
        </div>
      </div>

      {(progress.tokens_quarantined > 0 ||
        progress.tokens_routed_success > 0 ||
        progress.tokens_routed_failure > 0) && (
        <div className="progress-routing-summary">
          {progress.tokens_routed_success > 0 && (
            <span>{progress.tokens_routed_success.toLocaleString()} routed success</span>
          )}
          {progress.tokens_routed_failure > 0 && (
            <span>{progress.tokens_routed_failure.toLocaleString()} routed failure</span>
          )}
          {progress.tokens_quarantined > 0 && (
            <span>{progress.tokens_quarantined.toLocaleString()} quarantined</span>
          )}
        </div>
      )}

      {progress.accounting && <ProgressAccountingDetails accounting={progress.accounting} />}

      {/* Cancellation message — visual-only; the announcement is carried by
          the polite live region at the top of the container. */}
      {progress.status === "cancelled" && (
        <div className="progress-cancelled-msg">
          Pipeline execution was cancelled.
        </div>
      )}

      {/* Failure message — visual-only; announced via the live region above. */}
      {progress.status === "failed" && progress.recent_errors.length === 0 && (
        <div className="progress-failed-msg">
          Pipeline execution failed.
        </div>
      )}

      {/* Recent errors */}
      {progress.recent_errors.length > 0 && (
        <div>
          <div className="progress-errors-title">
            Recent errors ({progress.recent_errors.length})
          </div>
          <div className="progress-errors-container">
            {progress.recent_errors.map((err, i) => (
              <div
                key={`${err.node_id}-${i}`}
                className="progress-error-item"
                style={{
                  borderBottom:
                    i < progress.recent_errors.length - 1
                      ? "1px solid var(--color-error-border)"
                      : "none",
                }}
              >
                {err.node_id && <strong>{err.node_id}</strong>}
                {err.node_id && ": "}
                {err.message}
                {err.row_id && (
                  <span className="progress-error-row-id">
                    {" "}
                    (row: {err.row_id})
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
