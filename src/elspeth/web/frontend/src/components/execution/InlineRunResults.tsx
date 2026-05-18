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
import { isTerminalRunStatus } from "@/types/index";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";

export function InlineRunResults(): JSX.Element | null {
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);
  const runs = useExecutionStore((s) => s.runs);
  const loadRuns = useExecutionStore((s) => s.loadRuns);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const [showHistory, setShowHistory] = useState(false);

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
  const progressBelongsToActiveRun = activeRunId !== null && progress !== null;
  const displayStatus = progressBelongsToActiveRun ? progress.status : displayRun?.status ?? null;
  const showProgress = progressBelongsToActiveRun;
  const outputRunId =
    activeRunId && progress && isTerminalRunStatus(progress.status)
      ? activeRunId
      : displayRun && displayStatus && isTerminalRunStatus(displayStatus)
        ? displayRun.id
        : null;
  const hasHistory = visibleRuns.length > 0;

  if (!showProgress && !outputRunId && !hasHistory) {
    return null;
  }

  return (
    <section className="inline-run-results" aria-label="Pipeline run results">
      {showProgress && <ProgressView />}
      {outputRunId && <NarrativeResultsBranch runId={outputRunId} />}

      {hasHistory && (
        <div className="inline-run-results-history-cta">
          <button
            type="button"
            onClick={() => setShowHistory(true)}
            className="btn"
          >
            Past runs ({visibleRuns.length})
          </button>
        </div>
      )}

      {showHistory && (
        <RunsHistoryDrawer onClose={() => setShowHistory(false)} />
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
