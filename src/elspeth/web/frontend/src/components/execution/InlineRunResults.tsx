// ============================================================================
// InlineRunResults
//
// Mounts in the chat column and renders the active or most-recent run's
// progress and outputs. Historical access lives in RunsHistoryDrawer.
// ============================================================================

import { useEffect, useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { ProgressView } from "@/components/execution/ProgressView";
import { RunOutputsPanel } from "@/components/inspector/RunOutputsPanel";
import { NarrativeResults } from "@/components/composer/NarrativeResults";
import { useNarrativeMode } from "@/hooks/useNarrativeMode";
import {
  isTerminalRunStatus,
  type DiscardStageSummary,
  type DiscardSummary,
  type Run,
  type RunProgress,
  type RunStatus,
} from "@/types/index";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";

function statusLabel(status: RunStatus | null): string {
  if (!status) return "No run";
  switch (status) {
    case "completed":
      return "Completed";
    case "completed_with_failures":
      return "Completed with failures";
    case "failed":
      return "Failed";
    case "empty":
      return "Empty";
    case "cancelled":
      return "Cancelled";
    case "running":
      return "Running";
    case "pending":
      return "Pending";
  }
}

function runSummaryParts(
  status: RunStatus | null,
  progress: RunProgress | null,
  run: Run | null,
): string[] {
  const accounting = progress?.accounting ?? run?.accounting ?? null;
  const rows =
    progress?.source_rows_processed ?? accounting?.source.rows_processed ?? null;
  const succeeded =
    progress?.tokens_succeeded ?? accounting?.tokens.succeeded ?? null;
  const failed = progress?.tokens_failed ?? accounting?.tokens.failed ?? null;
  const parts = [statusLabel(status)];
  if (rows !== null) parts.push(`${rows} ${rows === 1 ? "row" : "rows"}`);
  if (succeeded !== null) {
    parts.push(`${succeeded} succeeded`);
  }
  if (failed !== null) {
    parts.push(`${failed} failed`);
  }
  const discardTotal = run?.discard_summary?.total ?? 0;
  if (discardTotal > 0) {
    parts.push(`${discardTotal} discarded`);
  }
  return parts;
}

function pluralRows(count: number): string {
  return `${count} ${count === 1 ? "row" : "rows"}`;
}

function sourceValidationFallbackStage(summary: DiscardSummary): DiscardStageSummary | null {
  if (summary.validation_errors <= 0) return null;
  return {
    stage: "source_validation",
    node_id: null,
    count: summary.validation_errors,
  };
}

function transformValidationFallbackStage(summary: DiscardSummary): DiscardStageSummary | null {
  if (summary.transform_errors <= 0) return null;
  return {
    stage: "transform_validation",
    node_id: null,
    count: summary.transform_errors,
  };
}

function sinkDiscardFallbackStage(summary: DiscardSummary): DiscardStageSummary | null {
  if (summary.sink_discards <= 0) return null;
  return {
    stage: "sink_discard",
    node_id: null,
    count: summary.sink_discards,
  };
}

function primaryDiscardStage(summary: DiscardSummary): DiscardStageSummary | null {
  const stages = summary.stages ?? [];
  return (
    stages.find((stage) => stage.stage === "source_validation") ??
    stages.find((stage) => stage.stage === "transform_validation") ??
    stages.find((stage) => stage.stage === "sink_discard") ??
    sourceValidationFallbackStage(summary) ??
    transformValidationFallbackStage(summary) ??
    sinkDiscardFallbackStage(summary)
  );
}

function discardStageLabel(stage: DiscardStageSummary): string {
  switch (stage.stage) {
    case "source_validation":
      return "source validation";
    case "transform_validation":
      return "transform validation";
    case "sink_discard":
      return "sink discard handling";
  }
}

function discardCauseText(stage: DiscardStageSummary): string {
  switch (stage.stage) {
    case "source_validation":
      return "Common causes: source schema declares fields the input data does not contain; CSV header mismatch; on_validation_failure: \"discard\" is dropping rows.";
    case "transform_validation":
      return "Common causes: transform output did not match its declared schema; error routing sends invalid rows to discard; upstream fields changed shape.";
    case "sink_discard":
      return "Common causes: sink write failure handling routed rows to discard before an output artifact was produced.";
  }
}

function DiscardSummaryWarning({ run }: { run: Run | null }): JSX.Element | null {
  const summary = run?.discard_summary;
  if (!run || !summary || summary.total <= 0) return null;
  const stage = primaryDiscardStage(summary);
  if (!stage) return null;
  const nodeSuffix = stage.node_id ? ` (${stage.node_id})` : "";
  const terminalNote =
    run.status === "empty"
      ? "Run terminated empty."
      : "Run completed with discarded rows.";
  return (
    <div role="alert" className="discard-summary-warning">
      <strong>
        {pluralRows(stage.count)} discarded at {discardStageLabel(stage)}
        {nodeSuffix}. {terminalNote}
      </strong>
      <span>{discardCauseText(stage)} View diagnostics for the first failed row's error message.</span>
    </div>
  );
}

export function InlineRunResults(): JSX.Element | null {
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);
  const runs = useExecutionStore((s) => s.runs);
  const loadRuns = useExecutionStore((s) => s.loadRuns);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [showHistory, setShowHistory] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    if (!activeSessionId) return;
    void loadRuns(activeSessionId);
  }, [activeSessionId, loadRuns]);

  const visibleRuns = activeSessionId
    ? runs.filter((run) => !run.session_id || run.session_id === activeSessionId)
    : runs;
  const hasActiveOrPendingRun =
    visibleRuns.some((run) => run.status === "pending" || run.status === "running") ||
    (progress !== null && !isTerminalRunStatus(progress.status));

  useEffect(() => {
    if (!activeSessionId || !hasActiveOrPendingRun) return;
    const timer = window.setInterval(() => {
      void loadRuns(activeSessionId);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [activeSessionId, hasActiveOrPendingRun, loadRuns]);

  const activeRun = activeRunId
    ? (visibleRuns.find((run) => run.id === activeRunId) ?? null)
    : null;
  const mostRecentRun = !activeRunId ? (visibleRuns[0] ?? null) : null;
  const displayRun = activeRun ?? mostRecentRun;
  const historyRuns = visibleRuns.filter(
    (run) => isTerminalRunStatus(run.status) && run.id !== displayRun?.id,
  );
  const progressBelongsToActiveRun = activeRunId !== null && progress !== null;
  const displayStatus = progressBelongsToActiveRun ? progress.status : displayRun?.status ?? null;
  const showProgress = progressBelongsToActiveRun;
  const outputRunId =
    activeRunId && progress && isTerminalRunStatus(progress.status)
      ? activeRunId
      : displayRun && displayStatus && isTerminalRunStatus(displayStatus)
        ? displayRun.id
        : null;
  const hasHistory = historyRuns.length > 0;
  const summaryParts = runSummaryParts(
    displayStatus,
    progressBelongsToActiveRun ? progress : null,
    displayRun,
  );

  if (!showProgress && !outputRunId && !hasHistory) {
    return null;
  }

  return (
    <section
      className={`inline-run-results${isCollapsed ? " inline-run-results--collapsed" : ""}`}
      aria-label="Pipeline run results"
    >
      <div className="inline-run-results-toolbar">
        <div className="inline-run-results-heading">
          <span className="inline-run-results-title">Run results</span>
          {isCollapsed && (
            <span className="inline-run-results-summary">
              {summaryParts.join(" · ")}
            </span>
          )}
        </div>
        <div className="inline-run-results-actions">
          <button
            type="button"
            onClick={() => setIsCollapsed((value) => !value)}
            className="btn-compact inline-run-results-collapse-btn"
            aria-label={isCollapsed ? "Show run results" : "Hide run results"}
            title={isCollapsed ? "Show run results" : "Hide run results"}
          >
            <span aria-hidden="true">{isCollapsed ? "\u25B2" : "\u25BC"}</span>
          </button>
          {hasHistory && (
            <button
              type="button"
              onClick={() => setShowHistory(true)}
              className="btn-compact inline-run-results-history-btn"
            >
              Past runs ({historyRuns.length})
            </button>
          )}
        </div>
      </div>

      {!isCollapsed && <DiscardSummaryWarning run={displayRun} />}
      {!isCollapsed && showProgress && <ProgressView />}
      {!isCollapsed && outputRunId && <NarrativeResultsBranch runId={outputRunId} />}

      {showHistory && (
        <RunsHistoryDrawer
          onClose={() => setShowHistory(false)}
          runsOverride={historyRuns}
        />
      )}
    </section>
  );
}

/**
 * NarrativeResultsBranch — Phase 6B Task 7 dispatcher.
 *
 * XOR switch between the narrative-mode `NarrativeResults` panel and the
 * default tabular `RunOutputsPanel`, based on `useNarrativeMode`. Per
 * plan 19b:365 the literal pattern is
 * `return narrativeMode ? <NarrativeResults /> : <ExistingTablePreview />`
 * and per plan 19b:359 the rendered output is `<NarrativeResults />`
 * **rather than** the existing table preview — the two views are
 * mutually exclusive at the composition level, not stacked.
 *
 * Per plan §"Scope boundaries": narrative mode is binary at the
 * composition level. If any pipeline plugin carries the
 * `"narrative-summary"` capability tag (per Phase 6A Task 8), the
 * narrative panel renders; otherwise the existing tabular view renders.
 */
function NarrativeResultsBranch({ runId }: { runId: string }): JSX.Element {
  const { narrativeMode } = useNarrativeMode();
  return narrativeMode ? <NarrativeResults /> : <RunOutputsPanel runId={runId} />;
}
