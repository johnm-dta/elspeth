// ============================================================================
// InlineRunResults
//
// Mounts in the chat column and renders the active or most-recent run's
// progress and outputs. Historical access lives in RunsHistoryDrawer.
// ============================================================================

import { useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { ProgressView } from "@/components/execution/ProgressView";
import { RunOutputsPanel } from "@/components/inspector/RunOutputsPanel";
import { isTerminalRunStatus } from "@/types/index";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";

export function InlineRunResults(): JSX.Element | null {
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);
  const runs = useExecutionStore((s) => s.runs);
  const [showHistory, setShowHistory] = useState(false);

  const activeRun = activeRunId
    ? (runs.find((run) => run.id === activeRunId) ?? null)
    : null;
  const mostRecentRun = !activeRunId ? (runs[0] ?? null) : null;
  const displayRun = activeRun ?? mostRecentRun;
  const progressBelongsToDisplayRun =
    activeRunId !== null &&
    displayRun !== null &&
    activeRunId === displayRun.id &&
    progress !== null;
  const displayStatus = progressBelongsToDisplayRun
    ? progress.status
    : displayRun?.status ?? null;
  const showProgress =
    progressBelongsToDisplayRun && !isTerminalRunStatus(progress.status);
  const outputRunId =
    displayRun && displayStatus && isTerminalRunStatus(displayStatus)
      ? displayRun.id
      : null;
  const hasHistory = runs.length > 0;

  if (!showProgress && !outputRunId && !hasHistory) {
    return null;
  }

  return (
    <section className="inline-run-results" aria-label="Pipeline run results">
      {showProgress && <ProgressView />}
      {outputRunId && <RunOutputsPanel runId={outputRunId} />}

      {hasHistory && (
        <div className="inline-run-results-history-cta">
          <button
            type="button"
            onClick={() => setShowHistory(true)}
            className="btn"
          >
            Past runs ({runs.length})
          </button>
        </div>
      )}

      {showHistory && (
        <RunsHistoryDrawer onClose={() => setShowHistory(false)} />
      )}
    </section>
  );
}
