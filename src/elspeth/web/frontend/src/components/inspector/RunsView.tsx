// ============================================================================
// RunsView
//
// List of runs for the current session. Each run entry shows:
// - Status badge (colour + text label, not colour alone)
// - Explicit source/token accounting
// - Duration (elapsed time or "running..." for active runs)
// - Composition state version
//
// Click an active run to expand and show ProgressView inline.
//
// Empty state: "No runs yet. Validate your pipeline, then click Execute
// to run it."
// ============================================================================

import { useEffect, useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { ProgressView } from "@/components/execution/ProgressView";
import type { RunAccounting, RunDiagnostics, RunDiagnosticsWorkingView, RunProgress, RunStatus } from "@/types/index";

// ── Status badge CSS class mapping ───────────────────────────────────────────
// Uses .status-badge + .status-badge-{status} classes from App.css.
//
// The `Record<RunStatus, string>` type makes this map exhaustive: when a new
// status is added to RUN_STATUS_VALUES (in types/index.ts), this entry must
// be filled in or `tsc --noEmit` fails with TS2741. Mirrors the backend
// single-source-of-truth pattern at sessions/protocol.py:35-36.

const STATUS_BADGE_CLASSES: Record<RunStatus, string> = {
  pending: "status-badge status-badge-pending",
  running: "status-badge status-badge-running",
  completed: "status-badge status-badge-completed",
  completed_with_failures: "status-badge status-badge-completed-with-failures",
  failed: "status-badge status-badge-failed",
  empty: "status-badge status-badge-empty",
  cancelled: "status-badge status-badge-cancelled",
};

// Human-readable badge labels.  Same exhaustiveness guarantee as the class
// map: a missing entry is a compile error.  Underscored identifiers like
// `completed_with_failures` become "completed with failures" for display
// (CSS then uppercases to "COMPLETED WITH FAILURES").
const STATUS_DISPLAY_LABELS: Record<RunStatus, string> = {
  pending: "pending",
  running: "running",
  completed: "completed",
  completed_with_failures: "completed with failures",
  failed: "failed",
  empty: "empty",
  cancelled: "cancelled",
};

// ── Duration formatting ──────────────────────────────────────────────────────

function formatDuration(startedAt: string, finishedAt: string | null): string {
  const start = new Date(startedAt).getTime();
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now();
  const diffSec = Math.max(0, Math.floor((end - start) / 1000));

  if (diffSec < 60) return `${diffSec}s`;
  const min = Math.floor(diffSec / 60);
  const sec = diffSec % 60;
  return `${min}m ${sec}s`;
}

function counted(label: string, count: number): string {
  return count === 1 ? `1 ${label}` : `${count.toLocaleString()} ${label}s`;
}

function summarizeCounts(prefix: string, counts: Record<string, number>): string | null {
  const entries = Object.entries(counts).sort(([left], [right]) => left.localeCompare(right));
  if (entries.length === 0) {
    return null;
  }
  return `${prefix} include ${entries.map(([name, count]) => `${name}=${count}`).join(", ")}.`;
}

function buildVisibleEvidence(diagnostics: RunDiagnostics): string[] {
  const evidence: string[] = [];
  const tokenCount = diagnostics.summary.token_count;
  if (tokenCount > 0) {
    evidence.push(`${counted("token", tokenCount)} ${tokenCount === 1 ? "is" : "are"} visible in the runtime trace.`);
    if (diagnostics.summary.preview_truncated) {
      evidence.push(`The preview is limited to the first ${counted("token", diagnostics.summary.preview_limit)}.`);
    }
  }

  const stateSummary = summarizeCounts("Node states", diagnostics.summary.state_counts);
  if (stateSummary) {
    evidence.push(stateSummary);
  }

  const operationSummary = summarizeCounts("Operation records", diagnostics.summary.operation_counts);
  if (operationSummary) {
    evidence.push(operationSummary);
  }

  diagnostics.artifacts.slice(0, 3).forEach((artifact) => {
    evidence.push(`Saved output is visible at ${artifact.path_or_uri}.`);
  });

  if (evidence.length === 0) {
    evidence.push("No tokens, operations, or saved outputs are visible yet.");
  }
  return evidence;
}

function buildPendingWorkingView(diagnostics: RunDiagnostics): RunDiagnosticsWorkingView {
  return {
    headline: "Reading current run evidence",
    evidence: buildVisibleEvidence(diagnostics),
    meaning: "The LLM is reading the same run records shown here and preparing a plain-English explanation.",
    next_steps: [],
  };
}

interface VisibleRunAccounting {
  sourceRowsProcessed: number | null;
  tokensEmitted: number | null;
  tokensSucceeded: number | null;
  tokensFailed: number | null;
  tokensStructural: number | null;
  tokensPending: number | null;
  tokensQuarantined: number;
  tokensRoutedSuccess: number;
  tokensRoutedFailure: number;
  tokensDiscarded: number;
}

function visibleRunAccounting(
  accounting: RunAccounting | null,
  progress: RunProgress | null,
): VisibleRunAccounting {
  return {
    sourceRowsProcessed:
      accounting?.source.rows_processed ?? progress?.source_rows_processed ?? null,
    tokensEmitted: accounting?.tokens.emitted ?? null,
    tokensSucceeded:
      accounting?.tokens.succeeded ?? progress?.tokens_succeeded ?? null,
    tokensFailed:
      accounting?.tokens.failed ?? progress?.tokens_failed ?? null,
    tokensStructural: accounting?.tokens.structural ?? null,
    tokensPending: accounting?.tokens.pending ?? null,
    tokensQuarantined:
      accounting?.routing.quarantined ?? progress?.tokens_quarantined ?? 0,
    tokensRoutedSuccess:
      accounting?.routing.routed_success ?? progress?.tokens_routed_success ?? 0,
    tokensRoutedFailure:
      accounting?.routing.routed_failure ?? progress?.tokens_routed_failure ?? 0,
    tokensDiscarded: accounting?.routing.discarded ?? 0,
  };
}

// ── RunsView component ───────────────────────────────────────────────────────

export function RunsView() {
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const runs = useExecutionStore((s) => s.runs);
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);
  const diagnosticsByRunId = useExecutionStore((s) => s.diagnosticsByRunId);
  const diagnosticsLoadingByRunId = useExecutionStore((s) => s.diagnosticsLoadingByRunId);
  const diagnosticsEvaluatingByRunId = useExecutionStore((s) => s.diagnosticsEvaluatingByRunId);
  const diagnosticsErrorByRunId = useExecutionStore((s) => s.diagnosticsErrorByRunId);
  const diagnosticsExplanationByRunId = useExecutionStore((s) => s.diagnosticsExplanationByRunId);
  const diagnosticsWorkingViewByRunId = useExecutionStore((s) => s.diagnosticsWorkingViewByRunId);
  const loadRuns = useExecutionStore((s) => s.loadRuns);
  const loadRunDiagnostics = useExecutionStore((s) => s.loadRunDiagnostics);
  const evaluateRunDiagnostics = useExecutionStore((s) => s.evaluateRunDiagnostics);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const hasActiveRun = runs.some((run) => run.status === "pending" || run.status === "running");
  const expandedRun = expandedRunId ? runs.find((run) => run.id === expandedRunId) : undefined;
  const expandedRunIsActive = expandedRun?.status === "pending" || expandedRun?.status === "running";

  // Load runs when the view mounts or session changes
  useEffect(() => {
    if (activeSessionId) {
      loadRuns(activeSessionId);
    }
  }, [activeSessionId, loadRuns]);

  useEffect(() => {
    if (!activeSessionId || !hasActiveRun) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadRuns(activeSessionId);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [activeSessionId, hasActiveRun, loadRuns]);

  useEffect(() => {
    if (!expandedRunId || !expandedRunIsActive) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadRunDiagnostics(expandedRunId);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [expandedRunId, expandedRunIsActive, loadRunDiagnostics]);

  // Empty state
  if (runs.length === 0) {
    return (
      <div
        className="empty-state"
        style={{
          padding: 24,
          fontSize: 14,
        }}
      >
        No runs yet. Validate your pipeline, then click Execute to run it.
      </div>
    );
  }

  return (
    <div style={{ padding: 8 }}>
      {runs.map((run) => {
        const isActive =
          run.id === activeRunId && progress?.status === "running";
        const activeProgress = run.id === activeRunId ? progress : null;
        const accounting = visibleRunAccounting(run.accounting, activeProgress);
        const discardTotal = run.discard_summary?.total ?? 0;
        const discardTitle = run.discard_summary
          ? [
              `Validation ${run.discard_summary.validation_errors.toLocaleString()}`,
              `transform ${run.discard_summary.transform_errors.toLocaleString()}`,
              `sink ${run.discard_summary.sink_discards.toLocaleString()}`,
            ].join(", ")
          : undefined;

        return (
          <div key={run.id}>
            <div
              className="component-card"
              style={{
                padding: "10px 12px",
                marginBottom: 4,
                fontSize: 13,
              }}
            >
              {/* Top row: status badge + version */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 8,
                  flexWrap: "wrap",
                  marginBottom: 4,
                }}
              >
                {/* Status badge: uses CSS class from App.css */}
                <span className={STATUS_BADGE_CLASSES[run.status]}>
                  {STATUS_DISPLAY_LABELS[run.status]}
                </span>
                <span
                  style={{
                    fontSize: 11,
                    color: "var(--color-text-muted)",
                    whiteSpace: "nowrap",
                  }}
                >
                  v{run.composition_version}
                </span>
              </div>

              {/* Source/token accounting + duration */}
              <div
                style={{
                  display: "flex",
                  gap: 12,
                  flexWrap: "wrap",
                  fontSize: 12,
                  color: "var(--color-text-muted)",
                }}
              >
                {accounting.sourceRowsProcessed === null ? (
                  <span>Accounting pending</span>
                ) : (
                  <span>{counted("source row", accounting.sourceRowsProcessed)}</span>
                )}
                {accounting.tokensEmitted !== null && (
                  <span>{counted("token", accounting.tokensEmitted)} emitted</span>
                )}
                {accounting.tokensSucceeded !== null && accounting.tokensSucceeded > 0 && (
                  <span title="Tokens that reached a successful terminal disposition">
                    {counted("token", accounting.tokensSucceeded)} succeeded
                  </span>
                )}
                {accounting.tokensFailed !== null && accounting.tokensFailed > 0 && (
                  <span title="Tokens that reached a failure terminal disposition">
                    <span style={{ color: "var(--color-error)" }}>
                      {counted("token", accounting.tokensFailed)} failed
                    </span>
                  </span>
                )}
                {accounting.tokensStructural !== null && accounting.tokensStructural > 0 && (
                  <span title="Closed structural parent/delegation tokens">
                    {counted("structural token", accounting.tokensStructural)}
                  </span>
                )}
                {accounting.tokensPending !== null && accounting.tokensPending > 0 && (
                  <span
                    title="Tokens without a completed terminal disposition"
                    style={{ color: "var(--color-warning)" }}
                  >
                    {counted("pending token", accounting.tokensPending)}
                  </span>
                )}
                {accounting.tokensQuarantined > 0 && (
                  <span
                    title="Tokens quarantined for investigation"
                    style={{ color: "var(--color-warning)" }}
                  >
                    {counted("token", accounting.tokensQuarantined)} quarantined
                  </span>
                )}
                {accounting.tokensRoutedSuccess > 0 && (
                  <span title="Gate route_to_sink MOVE tokens (intentional success-path routing)">
                    {accounting.tokensRoutedSuccess.toLocaleString()} routed (success)
                  </span>
                )}
                {accounting.tokensRoutedFailure > 0 && (
                  <span
                    title="Transform on_error DIVERT tokens (failure-path routing)"
                    style={{ color: "var(--color-warning)" }}
                  >
                    {accounting.tokensRoutedFailure.toLocaleString()} routed (failure)
                  </span>
                )}
                {accounting.tokensDiscarded > 0 && (
                  <span
                    title="Tokens discarded by sink or validation disposition"
                    style={{ color: "var(--color-warning)" }}
                  >
                    {counted("token", accounting.tokensDiscarded)} discarded
                  </span>
                )}
                {discardTotal > 0 && (
                  <span
                    title={discardTitle}
                    style={{ color: "var(--color-warning)" }}
                  >
                    {discardTotal.toLocaleString()} discarded
                  </span>
                )}
                <span>
                  {run.status === "running"
                    ? "running..."
                    : formatDuration(run.started_at, run.finished_at)}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    const nextRunId = expandedRunId === run.id ? null : run.id;
                    setExpandedRunId(nextRunId);
                    if (nextRunId) {
                      void loadRunDiagnostics(run.id);
                    }
                  }}
                  style={{
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-sm)",
                    background: "var(--color-bg)",
                    color: "var(--color-text)",
                    fontSize: 12,
                    padding: "2px 7px",
                    cursor: "pointer",
                  }}
                >
                  {expandedRunId === run.id ? "Hide" : "Inspect"}
                </button>
              </div>

              {/*
                Failure-evidence alert.

                Phase 2.2 (elspeth-0de989c56d): the gate uses the backend's
                authoritative status classification rather than re-deriving
                from accounting counts.  The backend always populates `error` for
                failure-like terminal runs:
                  - FAILED  → structural failure message
                  - COMPLETED_WITH_FAILURES → _partial_completion_message
                    (count summary; per-token detail lives behind "Inspect")
                A count-based predicate would duplicate the L0
                `failure_indicator` invariant (web/execution/schemas.py)
                and would also miss FAILED-with-zero-counts (engine crash
                before any accounting was available).
              */}
              {(run.status === "failed" || run.status === "completed_with_failures") &&
                run.error && (
                <div
                  role="alert"
                  style={{
                    marginTop: 8,
                    padding: "7px 9px",
                    border: "1px solid var(--color-error-border)",
                    borderRadius: "var(--radius-sm)",
                    backgroundColor: "var(--color-error-bg)",
                    color: "var(--color-error)",
                    fontSize: 12,
                    lineHeight: 1.4,
                    overflowWrap: "anywhere",
                  }}
                >
                  {run.error}
                </div>
              )}

              {expandedRunId === run.id && (
                <RunDiagnosticsPanel
                  diagnostics={diagnosticsByRunId[run.id]}
                  error={diagnosticsErrorByRunId[run.id] ?? null}
                  explanation={diagnosticsExplanationByRunId[run.id] ?? null}
                  isEvaluating={diagnosticsEvaluatingByRunId[run.id] ?? false}
                  isLoading={diagnosticsLoadingByRunId[run.id] ?? false}
                  workingView={diagnosticsWorkingViewByRunId[run.id] ?? null}
                  onExplain={() => void evaluateRunDiagnostics(run.id)}
                  onRefresh={() => void loadRunDiagnostics(run.id)}
                />
              )}
            </div>

            {/* Show live progress inline for the active running run */}
            {isActive && <ProgressView />}
          </div>
        );
      })}
    </div>
  );
}

interface RunDiagnosticsPanelProps {
  diagnostics: RunDiagnostics | undefined;
  error: string | null;
  explanation: string | null;
  isEvaluating: boolean;
  isLoading: boolean;
  workingView: RunDiagnosticsWorkingView | null;
  onExplain: () => void;
  onRefresh: () => void;
}

function RunDiagnosticsPanel({
  diagnostics,
  error,
  explanation,
  isEvaluating,
  isLoading,
  workingView,
  onExplain,
  onRefresh,
}: RunDiagnosticsPanelProps) {
  const visibleWorkingView =
    workingView ?? (isEvaluating && diagnostics ? buildPendingWorkingView(diagnostics) : null);

  return (
    <div
      style={{
        marginTop: 10,
        paddingTop: 10,
        borderTop: "1px solid var(--color-border)",
        fontSize: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
          marginBottom: 8,
        }}
      >
        <span style={{ color: "var(--color-text-muted)" }}>
          {diagnostics
            ? `${diagnostics.summary.token_count.toLocaleString()} tokens`
            : isLoading
              ? "Loading diagnostics..."
              : "Diagnostics not loaded"}
          {diagnostics?.summary.preview_truncated ? `, first ${diagnostics.summary.preview_limit}` : ""}
        </span>
        <span style={{ display: "flex", gap: 6 }}>
          <button type="button" onClick={onRefresh} disabled={isLoading}>
            Refresh
          </button>
          <button type="button" onClick={onExplain} disabled={isEvaluating || isLoading || !diagnostics}>
            {isEvaluating ? "Explaining..." : "Explain"}
          </button>
        </span>
      </div>

      {error && (
        <div role="alert" style={{ color: "var(--color-error)", marginBottom: 8 }}>
          {error}
        </div>
      )}

      {diagnostics && (
        <>
          {diagnostics.operations.length > 0 && (
            <div style={{ marginBottom: 8, color: "var(--color-text-muted)" }}>
              {diagnostics.operations.map((operation) => (
                <span key={operation.operation_id} style={{ marginRight: 10 }}>
                  {operation.operation_type} {operation.status}
                </span>
              ))}
            </div>
          )}

          {diagnostics.tokens.length > 0 && (
            <div style={{ display: "grid", gap: 6, marginBottom: 8 }}>
              {diagnostics.tokens.map((token) => (
                <div
                  key={token.token_id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(90px, 1fr) minmax(80px, 1.2fr)",
                    gap: 8,
                    alignItems: "start",
                  }}
                >
                  <span style={{ overflowWrap: "anywhere" }}>{token.token_id}</span>
                  <span style={{ color: "var(--color-text-muted)", overflowWrap: "anywhere" }}>
                    row {token.row_index ?? "-"}
                    {token.terminal_outcome ? `, ${token.terminal_outcome}` : ""}
                    {token.states.map((state) => (
                      <span key={state.state_id} style={{ marginLeft: 8 }}>
                        {state.node_id} {state.status}
                      </span>
                    ))}
                  </span>
                </div>
              ))}
            </div>
          )}

          {diagnostics.artifacts.length > 0 && (
            <div style={{ display: "grid", gap: 4, marginBottom: 8 }}>
              {diagnostics.artifacts.map((artifact) => (
                <span key={artifact.artifact_id} style={{ overflowWrap: "anywhere" }}>
                  {artifact.path_or_uri}
                </span>
              ))}
            </div>
          )}
        </>
      )}

      {visibleWorkingView && (
        <div
          style={{
            padding: "8px 10px",
            borderLeft: "3px solid var(--color-border)",
            backgroundColor: "var(--color-surface-hover)",
            borderRadius: "var(--radius-sm)",
            lineHeight: 1.45,
            overflowWrap: "anywhere",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 5 }}>
            {visibleWorkingView.headline}
          </div>
          {visibleWorkingView.evidence.length > 0 && (
            <ul style={{ margin: "0 0 6px 16px", padding: 0 }}>
              {visibleWorkingView.evidence.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
          <div>{visibleWorkingView.meaning}</div>
          {visibleWorkingView.next_steps.length > 0 && (
            <ul style={{ margin: "6px 0 0 16px", padding: 0 }}>
              {visibleWorkingView.next_steps.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {!visibleWorkingView && explanation && (
        <div
          style={{
            padding: "7px 9px",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-sm)",
            lineHeight: 1.45,
            overflowWrap: "anywhere",
          }}
        >
          {explanation}
        </div>
      )}
    </div>
  );
}
