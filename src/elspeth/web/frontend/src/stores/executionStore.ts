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
  ExecutionFanoutAck,
  ExecutionFanoutGuard,
} from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import * as api from "@/api/client";
import { connectToRun, type WebSocketConnection } from "@/api/websocket";
import { useAuthStore } from "./authStore";
import { useBlobStore } from "./blobStore";
import { useInterpretationEventsStore } from "./interpretationEventsStore";
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
  pendingFanoutGuard: ExecutionFanoutGuard | null;
  pendingFanoutSessionId: string | null;
  isValidating: boolean;
  isExecuting: boolean;
  wsDisconnected: boolean;
  error: string | null;

  validate: (sessionId: string, options?: ValidateOptions) => Promise<boolean>;
  setValidationResult: (result: ValidationResult | null) => void;
  execute: (sessionId: string, fanoutAck?: ExecutionFanoutAck) => Promise<string | null>;
  confirmFanoutExecution: () => Promise<string | null>;
  dismissFanoutGuard: () => void;
  cancel: (runId: string) => Promise<void>;
  loadRuns: (sessionId: string) => Promise<void>;
  loadRunDiagnostics: (runId: string) => Promise<void>;
  evaluateRunDiagnostics: (runId: string) => Promise<void>;
  connectWebSocket: (runId: string) => void;
  clearValidation: () => void;
  reset: () => void;
}

interface ValidateOptions {
  expectedVersion?: number;
}

// The WebSocket connection handle is held outside Zustand state
// because it's not serialisable and components don't need to read it.
let wsConnection: WebSocketConnection | null = null;
let validationRequestSeq = 0;
let executionRequestSeq = 0;

function currentCompositionVersion(): number | null {
  return useSessionStore.getState().compositionState?.version ?? null;
}

function shouldApplyValidationResult(
  sessionId: string,
  expectedVersion: number | null,
): boolean {
  const sessionState = useSessionStore.getState();
  if (sessionState.activeSessionId !== sessionId) {
    return false;
  }
  if (
    expectedVersion !== null &&
    sessionState.compositionState?.version !== expectedVersion
  ) {
    return false;
  }
  return true;
}

function shouldApplyExecutionResult(
  sessionId: string,
  requestSeq: number,
): boolean {
  return (
    requestSeq === executionRequestSeq &&
    useSessionStore.getState().activeSessionId === sessionId
  );
}

function assertNever(value: never): never {
  throw new Error(`Unhandled interpretation kind: ${String(value)}`);
}

function describePendingInterpretation(event: InterpretationEvent): string {
  const nodeLabel = event.affected_node_id ?? "this transform";
  switch (event.kind) {
    case "invented_source":
      return `Resolve invented source data for ${nodeLabel} before running.`;
    case "llm_prompt_template":
      return `Resolve the LLM prompt template for ${nodeLabel} before running.`;
    case "pipeline_decision":
      return `Resolve the pipeline decision for ${nodeLabel} before running.`;
    case "llm_model_choice":
      return `Set the LLM model choice for ${nodeLabel} before running.`;
    case "vague_term":
    case null:
      return `Resolve the pending interpretation for ${nodeLabel} before running.`;
    default: {
      return assertNever(event.kind);
    }
  }
}

function getRunBlockError(sessionId: string): string | null {
  const interpretationState = useInterpretationEventsStore.getState();
  if (interpretationState.optedOutBySession[sessionId] === true) {
    return null;
  }
  const pending = interpretationState.pendingBySession[sessionId];
  if (!pending || Object.keys(pending).length === 0) {
    return null;
  }
  const firstPending = Object.values(pending)[0];
  return describePendingInterpretation(firstPending);
}

/**
 * Derive the run status from a RunEvent.
 *
 * Phase 2.2 (elspeth-0de989c56d): for terminal events the backend now
 * supplies an explicit `data.status` discriminator on each terminal payload.
 * The frontend MUST consume it verbatim — re-deriving from accounting counts would
 * duplicate the L0 `failure_indicator` predicate
 * (`web/execution/schemas.py:_check_status_accounting_invariant`) and create
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

function hasLiveCounters(
  data: RunEvent["data"],
): data is RunEventProgress | RunEventCancelled {
  return "source_rows_processed" in data;
}

/**
 * Apply a RunEvent to the current progress state.
 * Accumulates exceptions (keeping the most recent N) and updates
 * explicit source/token counters. Terminal events update run status; completed
 * events also attach the backend's closed Landscape accounting projection.
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

  const liveCounters = hasLiveCounters(data) ? data : null;
  const completedAccounting =
    event.event_type === "completed"
      ? (data as RunEventCompleted).accounting
      : null;
  const accounting = completedAccounting ?? state.progress?.accounting ?? null;
  const isTerminal =
    event.event_type === "completed" ||
    event.event_type === "cancelled" ||
    event.event_type === "failed";
  const cancelRequested = isTerminal ? false : (state.progress?.cancel_requested ?? false);

  // Progress and cancelled events carry best-known live counters. Completed
  // events carry the canonical Landscape accounting projection instead.
  // Error/failed events carry no counters, so they preserve the previous
  // progress snapshot through the event-type contract.
  const sourceRowsProcessed =
    liveCounters?.source_rows_processed ??
    completedAccounting?.source.rows_processed ??
    state.progress?.source_rows_processed ??
    0;
  const tokensSucceeded =
    liveCounters?.tokens_succeeded ??
    completedAccounting?.tokens.succeeded ??
    state.progress?.tokens_succeeded ??
    0;
  const tokensFailed =
    liveCounters?.tokens_failed ??
    completedAccounting?.tokens.failed ??
    state.progress?.tokens_failed ??
    0;
  const tokensQuarantined =
    liveCounters?.tokens_quarantined ??
    completedAccounting?.routing.quarantined ??
    state.progress?.tokens_quarantined ??
    0;
  const tokensRoutedSuccess =
    liveCounters?.tokens_routed_success ??
    completedAccounting?.routing.routed_success ??
    state.progress?.tokens_routed_success ??
    0;
  const tokensRoutedFailure =
    liveCounters?.tokens_routed_failure ??
    completedAccounting?.routing.routed_failure ??
    state.progress?.tokens_routed_failure ??
    0;

  const newProgress: RunProgress = {
    source_rows_processed: sourceRowsProcessed,
    tokens_succeeded: tokensSucceeded,
    tokens_failed: tokensFailed,
    tokens_quarantined: tokensQuarantined,
    tokens_routed_success: tokensRoutedSuccess,
    tokens_routed_failure: tokensRoutedFailure,
    cancel_requested: cancelRequested,
    accounting,
    recent_errors: recentErrors,
    status: deriveStatus(event),
  };

  // Run records mirror the REST shape: terminal completed runs receive
  // accounting; in-flight counters stay in `progress` rather than being
  // reintroduced as legacy rows_* fields on Run.
  let updatedRuns = state.runs;
  if (isTerminal) {
    updatedRuns = state.runs.map((r) =>
      r.id === event.run_id
        ? {
            ...r,
            status: newProgress.status,
            cancel_requested: false,
            accounting: completedAccounting ?? r.accounting,
            error:
              event.event_type === "failed"
                ? (data as RunEventFailed).detail
                : r.error,
            finished_at: event.timestamp,
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
  pendingFanoutGuard: null as ExecutionFanoutGuard | null,
  pendingFanoutSessionId: null as string | null,
  isValidating: false,
  isExecuting: false,
  wsDisconnected: false,
  error: null as string | null,
};

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  ...initialExecutionState,

  async validate(sessionId: string, options: ValidateOptions = {}) {
    const requestSeq = ++validationRequestSeq;
    const expectedVersion = options.expectedVersion ?? currentCompositionVersion();
    set({ isValidating: true, validationResult: null, error: null });
    try {
      const result = await api.validatePipeline(sessionId);
      if (requestSeq !== validationRequestSeq) return false;
      if (!shouldApplyValidationResult(sessionId, expectedVersion)) {
        set({ isValidating: false });
        return false;
      }
      set({ validationResult: result, isValidating: false });
      return true;
    } catch (err) {
      if (requestSeq !== validationRequestSeq) return false;
      if (!shouldApplyValidationResult(sessionId, expectedVersion)) {
        set({ isValidating: false });
        return false;
      }
      const apiErr = err as ApiError;
      const message =
        apiErr.status === 500
          ? "Validation encountered an internal error. Please try again."
          : apiErr.detail ?? "Validation failed. Please try again.";
      set({
        isValidating: false,
        error: message,
      });
      // false = caller must not record this version as validated.
      return false;
    }
  },

  setValidationResult(result: ValidationResult | null) {
    set({ validationResult: result });
  },

  async execute(sessionId: string, fanoutAck?: ExecutionFanoutAck) {
    const blockedByInterpretation = getRunBlockError(sessionId);
    if (blockedByInterpretation !== null) {
      set({ isExecuting: false, error: blockedByInterpretation });
      return null;
    }
    const requestSeq = ++executionRequestSeq;
    set({ isExecuting: true, error: null });
    try {
      const { run_id } =
        fanoutAck === undefined
          ? await api.executePipeline(sessionId)
          : await api.executePipeline(sessionId, fanoutAck);
      if (!shouldApplyExecutionResult(sessionId, requestSeq)) {
        if (requestSeq === executionRequestSeq) {
          set({ isExecuting: false });
        }
        return null;
      }
      set({
        activeRunId: run_id,
        isExecuting: false,
        pendingFanoutGuard: null,
        pendingFanoutSessionId: null,
        progress: {
          source_rows_processed: 0,
          tokens_succeeded: 0,
          tokens_failed: 0,
          tokens_quarantined: 0,
          tokens_routed_success: 0,
          tokens_routed_failure: 0,
          cancel_requested: false,
          accounting: null,
          recent_errors: [],
          status: "running",
        },
      });

      // Refresh runs list so the new run appears in InlineRunResults immediately
      get().loadRuns(sessionId);

      // Connect WebSocket for live progress
      get().connectWebSocket(run_id);
      return run_id;
    } catch (err) {
      if (!shouldApplyExecutionResult(sessionId, requestSeq)) {
        if (requestSeq === executionRequestSeq) {
          set({ isExecuting: false });
        }
        return null;
      }
      const apiErr = err as ApiError;
      if (
        apiErr.status === 428 &&
        apiErr.error_type === "execution_fanout_ack_required" &&
        apiErr.fanout_guard
      ) {
        set({
          isExecuting: false,
          pendingFanoutGuard: apiErr.fanout_guard,
          pendingFanoutSessionId: sessionId,
          error: null,
        });
        return null;
      }
      const message =
        apiErr.status === 409
          ? "A run is already in progress for this pipeline."
          : apiErr.detail ??
            "Pipeline execution failed. Check the run results panel for error details.";
      set({
        isExecuting: false,
        error: message,
      });
      return null;
    }
  },

  async confirmFanoutExecution() {
    const pendingGuard = get().pendingFanoutGuard;
    const pendingSessionId = get().pendingFanoutSessionId;
    if (!pendingGuard || !pendingSessionId) {
      return null;
    }
    return get().execute(pendingSessionId, {
      accepted: true,
      token: pendingGuard.ack_token,
    });
  },

  dismissFanoutGuard() {
    set({
      isExecuting: false,
      pendingFanoutGuard: null,
      pendingFanoutSessionId: null,
    });
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
        // Refresh terminal route projections: run rows carry discard summaries,
        // and blob rows carry finalized output inventory.
        const sessionId = useSessionStore.getState().activeSessionId;
        if (sessionId) {
          void get().loadRuns(sessionId);
          void useBlobStore.getState().loadBlobs(sessionId);
        }
      },
      onCancelled(event: RunEvent, _data: RunEventCancelled) {
        set((state) => applyRunEvent(state, event));
        // Refresh terminal route projections: run rows carry discard summaries,
        // and blob rows may carry partial outputs after cancellation.
        const sessionId = useSessionStore.getState().activeSessionId;
        if (sessionId) {
          void get().loadRuns(sessionId);
          void useBlobStore.getState().loadBlobs(sessionId);
        }
      },
      onFailed(event: RunEvent, _data: RunEventFailed) {
        set((state) => applyRunEvent(state, event));
        // Refresh terminal route projections: run rows carry discard summaries,
        // and blob rows may carry partial outputs after failure.
        const sessionId = useSessionStore.getState().activeSessionId;
        if (sessionId) {
          void get().loadRuns(sessionId);
          void useBlobStore.getState().loadBlobs(sessionId);
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
      const result = await api.cancelRun(runId);
      set((state) => ({
        runs: state.runs.map((run) =>
          run.id === runId
            ? {
                ...run,
                status: result.status,
                cancel_requested: result.cancel_requested,
              }
            : run,
        ),
        progress:
          state.activeRunId === runId && state.progress
            ? {
                ...state.progress,
                status: result.status,
                cancel_requested: result.cancel_requested,
              }
            : state.progress,
        error: null,
      }));
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
    validationRequestSeq += 1;
    executionRequestSeq += 1;
    wsConnection?.close();
    wsConnection = null;
    set(initialExecutionState);
  },
}));
