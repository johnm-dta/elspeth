// ============================================================================
// ELSPETH Execution Store
//
// Zustand store managing validation results, execution runs, live progress,
// and the WebSocket connection lifecycle.
//
// Key behaviour: auto-clear. When the session store's compositionState.version
// changes, the validation result is cleared and the Execute button becomes
// disabled. This is implemented via a cross-store subscription initialised
// in stores/subscriptions.ts (called once from App.tsx at startup).
// ============================================================================

import { create } from "zustand";
import type {
  Run,
  RunDiagnostics,
  RunDiagnosticsWorkingView,
  RunProgress,
  RunEvent,
  RunEventProgress,
  RunEventError,
  RunEventCompleted,
  RunEventCancelled,
  RunEventFailed,
  RunStatus,
  ValidationResult,
  ApiError,
} from "@/types/index";
import * as api from "@/api/client";
import { connectToRun, type WebSocketConnection } from "@/api/websocket";
import { useAuthStore } from "./authStore";
import { useBlobStore } from "./blobStore";
import { useSessionStore } from "./sessionStore";


const MAX_RECENT_ERRORS = 50;

interface ExecutionState {
  runs: Run[];
  activeRunId: string | null;
  progress: RunProgress | null;
  diagnosticsByRunId: Record<string, RunDiagnostics>;
  diagnosticsLoadingByRunId: Record<string, boolean>;
  diagnosticsEvaluatingByRunId: Record<string, boolean>;
  diagnosticsErrorByRunId: Record<string, string | null>;
  diagnosticsExplanationByRunId: Record<string, string>;
  diagnosticsWorkingViewByRunId: Record<string, RunDiagnosticsWorkingView>;
  validationResult: ValidationResult | null;
  isValidating: boolean;
  isExecuting: boolean;
  wsDisconnected: boolean;
  error: string | null;

  validate: (sessionId: string) => Promise<void>;
  execute: (sessionId: string) => Promise<string | null>;
  cancel: (runId: string) => Promise<void>;
  loadRuns: (sessionId: string) => Promise<void>;
  loadRunDiagnostics: (runId: string) => Promise<void>;
  evaluateRunDiagnostics: (runId: string) => Promise<void>;
  connectWebSocket: (runId: string) => void;
  clearValidation: () => void;
  reset: () => void;
}

// The WebSocket connection handle is held outside Zustand state
// because it's not serialisable and components don't need to read it.
let wsConnection: WebSocketConnection | null = null;

/**
 * Derive the run status from a RunEvent.
 *
 * Phase 2.2 (elspeth-0de989c56d): for terminal events the backend now
 * supplies an explicit `data.status` discriminator on each terminal payload.
 * The frontend MUST consume it verbatim — re-deriving from row counts would
 * duplicate the L0 `failure_indicator` predicate
 * (`web/execution/schemas.py:_check_status_row_count_invariant`) and create
 * dual-source-of-truth drift between backend and frontend classification.
 *
 * The exhaustiveness `assertNever` guard makes a future Phase 2.3 widening
 * (a new RunEvent.event_type) a compile error here rather than a silent
 * fall-through to "running".
 */
function deriveStatus(event: RunEvent): RunStatus {
  switch (event.event_type) {
    case "progress":
    case "error":
      // Non-terminal events: pipeline is still in flight.
      return "running";
    case "completed":
      return (event.data as RunEventCompleted).status;
    case "cancelled":
      return (event.data as RunEventCancelled).status;
    case "failed":
      return (event.data as RunEventFailed).status;
    default: {
      const _exhaustive: never = event.event_type;
      throw new Error(`Unhandled RunEvent.event_type: ${String(_exhaustive)}`);
    }
  }
}

/**
 * Apply a RunEvent to the current progress state.
 * Accumulates exceptions (keeping the most recent N) and updates
 * row counters. Terminal events ("completed", "cancelled", "failed") update
 * the run status in the runs list.
 */
function applyRunEvent(
  state: ExecutionState,
  event: RunEvent,
): Partial<ExecutionState> {
  const data = event.data;

  // Accumulate recoverable row errors and terminal failure detail, keeping
  // the most recent N. Terminal failure detail arrives as RunEventFailed.detail
  // rather than a RunEventError payload, but it is the user-facing reason.
  // New errors come first (newest-first display).
  const terminalFailure: RunEventError | null =
    event.event_type === "failed"
      ? {
          message: (data as RunEventFailed).detail,
          node_id: (data as RunEventFailed).node_id,
          row_id: null,
        }
      : null;
  const newErrors =
    event.event_type === "error"
      ? [data as RunEventError, ...(state.progress?.recent_errors ?? [])]
      : terminalFailure
        ? [terminalFailure, ...(state.progress?.recent_errors ?? [])]
      : [...(state.progress?.recent_errors ?? [])];
  const recentErrors = newErrors.slice(0, MAX_RECENT_ERRORS);

  // Extract row counts from data payload (progress, completed, cancelled all have them)
  const rowsProcessed =
    "rows_processed" in data ? (data as RunEventProgress).rows_processed : (state.progress?.rows_processed ?? 0);
  const rowsFailed =
    "rows_failed" in data ? (data as RunEventProgress).rows_failed : (state.progress?.rows_failed ?? 0);
  // elspeth-5069612f3c — preserve the rows_routed split from progress /
  // completed / cancelled payloads (failed terminal events do not carry
  // these counters today, so the previous progress values are kept on
  // failure-side terminal events).
  const rowsRoutedSuccess =
    "rows_routed_success" in data
      ? (data as RunEventProgress).rows_routed_success
      : (state.progress?.rows_routed_success ?? 0);
  const rowsRoutedFailure =
    "rows_routed_failure" in data
      ? (data as RunEventProgress).rows_routed_failure
      : (state.progress?.rows_routed_failure ?? 0);

  const newProgress: RunProgress = {
    rows_processed: rowsProcessed,
    rows_failed: rowsFailed,
    rows_routed_success: rowsRoutedSuccess,
    rows_routed_failure: rowsRoutedFailure,
    recent_errors: recentErrors,
    status: deriveStatus(event),
  };

  // Update the run in the list on every event that carries fresh row counts.
  //
  // elspeth-0c076ad374 — `progress` events also patch the four row counters
  // so the runs-list row stays in lockstep with the live ProgressView slot.
  // `error` events are deliberately excluded: RunEventError carries no row
  // counters, so the rowsProcessed/etc. above fall back to
  // `state.progress?.* ?? 0`, which would clobber a real REST snapshot in
  // the reconnect-before-first-progress case. Status / error detail /
  // finished_at remain gated on terminal events to avoid out-of-order
  // progress downgrading a `failed`/`completed` run back to `running`.
  const isTerminal =
    event.event_type === "completed" ||
    event.event_type === "cancelled" ||
    event.event_type === "failed";
  const isProgress = event.event_type === "progress";
  let updatedRuns = state.runs;
  if (isTerminal || isProgress) {
    updatedRuns = state.runs.map((r) =>
      r.id === event.run_id
        ? {
            ...r,
            rows_processed: rowsProcessed,
            rows_failed: rowsFailed,
            rows_routed_success: rowsRoutedSuccess,
            rows_routed_failure: rowsRoutedFailure,
            ...(isTerminal
              ? {
                  status: newProgress.status,
                  error:
                    event.event_type === "failed"
                      ? (data as RunEventFailed).detail
                      : r.error,
                  finished_at: event.timestamp,
                }
              : {}),
          }
        : r,
    );
  }

  return {
    progress: newProgress,
    runs: updatedRuns,
    wsDisconnected: false,
  };
}

const initialExecutionState = {
  runs: [] as Run[],
  activeRunId: null as string | null,
  progress: null as RunProgress | null,
  diagnosticsByRunId: {} as Record<string, RunDiagnostics>,
  diagnosticsLoadingByRunId: {} as Record<string, boolean>,
  diagnosticsEvaluatingByRunId: {} as Record<string, boolean>,
  diagnosticsErrorByRunId: {} as Record<string, string | null>,
  diagnosticsExplanationByRunId: {} as Record<string, string>,
  diagnosticsWorkingViewByRunId: {} as Record<string, RunDiagnosticsWorkingView>,
  validationResult: null as ValidationResult | null,
  isValidating: false,
  isExecuting: false,
  wsDisconnected: false,
  error: null as string | null,
};

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  ...initialExecutionState,

  async validate(sessionId: string) {
    set({ isValidating: true, validationResult: null, error: null });
    try {
      const result = await api.validatePipeline(sessionId);
      set({ validationResult: result, isValidating: false });
    } catch (err) {
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 500
          ? "Validation encountered an internal error. Please try again."
          : apiErr.detail ?? "Validation failed. Please try again.";
      set({
        isValidating: false,
        error: message,
      });
    }
  },

  async execute(sessionId: string) {
    set({ isExecuting: true, error: null });
    try {
      const { run_id } = await api.executePipeline(sessionId);
      set({
        activeRunId: run_id,
        isExecuting: false,
        progress: {
          rows_processed: 0,
          rows_failed: 0,
          rows_routed_success: 0,
          rows_routed_failure: 0,
          recent_errors: [],
          status: "running",
        },
      });

      // Refresh runs list so the new run appears in the Runs tab immediately
      get().loadRuns(sessionId);

      // Connect WebSocket for live progress
      get().connectWebSocket(run_id);
      return run_id;
    } catch (err) {
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 409
          ? "A run is already in progress for this pipeline."
          : apiErr.detail ??
            "Pipeline execution failed. Check the Runs tab for error details.";
      set({
        isExecuting: false,
        error: message,
      });
      return null;
    }
  },

  connectWebSocket(runId: string) {
    // Close any existing WebSocket connection
    wsConnection?.close();

    // Open a WebSocket for live progress, passing JWT as query parameter
    const token = useAuthStore.getState().token ?? "";
    wsConnection = connectToRun(runId, token, {
      onProgress(event: RunEvent, _data: RunEventProgress) {
        set((state) => applyRunEvent(state, event));
      },
      onError(event: RunEvent, _data: RunEventError) {
        // Non-terminal: per-row exception. Accumulate into progress.
        set((state) => applyRunEvent(state, event));
      },
      onComplete(event: RunEvent, _data: RunEventCompleted) {
        set((state) => applyRunEvent(state, event));
        // Refresh blob list — pipeline outputs are finalized on completion
        const sessionId = useSessionStore.getState().activeSessionId;
        if (sessionId) {
          useBlobStore.getState().loadBlobs(sessionId);
        }
      },
      onCancelled(event: RunEvent, _data: RunEventCancelled) {
        set((state) => applyRunEvent(state, event));
        // Refresh blob list — partial outputs may exist after cancellation
        const sessionId = useSessionStore.getState().activeSessionId;
        if (sessionId) {
          useBlobStore.getState().loadBlobs(sessionId);
        }
      },
      onFailed(event: RunEvent, _data: RunEventFailed) {
        set((state) => applyRunEvent(state, event));
        // Refresh blob list — partial outputs may exist after failure
        const sessionId = useSessionStore.getState().activeSessionId;
        if (sessionId) {
          useBlobStore.getState().loadBlobs(sessionId);
        }
      },
      onAuthFailure() {
        // Close code 4001 -- do not reconnect, trigger logout
        useAuthStore.getState().logout();
      },
    });
  },

  async cancel(runId: string) {
    try {
      await api.cancelRun(runId);
    } catch (err) {
      const apiErr = err as ApiError;
      set({ error: apiErr.detail ?? "Failed to cancel run." });
    }
  },

  async loadRuns(sessionId: string) {
    try {
      const runs = await api.fetchRuns(sessionId);
      set({ runs });
    } catch {
      // Non-critical -- runs list can be stale temporarily
    }
  },

  async loadRunDiagnostics(runId: string) {
    set((state) => ({
      diagnosticsLoadingByRunId: {
        ...state.diagnosticsLoadingByRunId,
        [runId]: true,
      },
      diagnosticsErrorByRunId: {
        ...state.diagnosticsErrorByRunId,
        [runId]: null,
      },
    }));
    try {
      const diagnostics = await api.fetchRunDiagnostics(runId);
      set((state) => ({
        diagnosticsByRunId: {
          ...state.diagnosticsByRunId,
          [runId]: diagnostics,
        },
        diagnosticsLoadingByRunId: {
          ...state.diagnosticsLoadingByRunId,
          [runId]: false,
        },
        diagnosticsErrorByRunId: {
          ...state.diagnosticsErrorByRunId,
          [runId]: null,
        },
      }));
    } catch (err) {
      const apiErr = err as ApiError;
      set((state) => ({
        diagnosticsLoadingByRunId: {
          ...state.diagnosticsLoadingByRunId,
          [runId]: false,
        },
        diagnosticsErrorByRunId: {
          ...state.diagnosticsErrorByRunId,
          [runId]: apiErr.detail ?? "Failed to load run diagnostics.",
        },
      }));
    }
  },

  async evaluateRunDiagnostics(runId: string) {
    set((state) => ({
      diagnosticsEvaluatingByRunId: {
        ...state.diagnosticsEvaluatingByRunId,
        [runId]: true,
      },
      diagnosticsErrorByRunId: {
        ...state.diagnosticsErrorByRunId,
        [runId]: null,
      },
    }));
    try {
      const result = await api.evaluateRunDiagnostics(runId);
      set((state) => ({
        diagnosticsExplanationByRunId: {
          ...state.diagnosticsExplanationByRunId,
          [runId]: result.explanation,
        },
        diagnosticsWorkingViewByRunId: {
          ...state.diagnosticsWorkingViewByRunId,
          [runId]: result.working_view,
        },
        diagnosticsEvaluatingByRunId: {
          ...state.diagnosticsEvaluatingByRunId,
          [runId]: false,
        },
        diagnosticsErrorByRunId: {
          ...state.diagnosticsErrorByRunId,
          [runId]: null,
        },
      }));
    } catch (err) {
      const apiErr = err as ApiError;
      set((state) => ({
        diagnosticsEvaluatingByRunId: {
          ...state.diagnosticsEvaluatingByRunId,
          [runId]: false,
        },
        diagnosticsErrorByRunId: {
          ...state.diagnosticsErrorByRunId,
          [runId]: apiErr.detail ?? "Failed to explain run diagnostics.",
        },
      }));
    }
  },

  clearValidation() {
    set({ validationResult: null });
  },

  reset() {
    wsConnection?.close();
    wsConnection = null;
    set(initialExecutionState);
  },
}));
