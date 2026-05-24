import { describe, it, expect, vi, beforeEach } from "vitest";
import { useExecutionStore } from "./executionStore";
import { useSessionStore } from "./sessionStore";
import { connectToRun } from "@/api/websocket";
import type { Run, RunAccounting, RunDiagnostics, RunEvent, ValidationResult } from "@/types/index";

// Mock the API client
vi.mock("@/api/client", () => ({
  validatePipeline: vi.fn(),
  executePipeline: vi.fn(),
  cancelRun: vi.fn(),
  fetchRuns: vi.fn().mockResolvedValue([]),
  fetchRunDiagnostics: vi.fn(),
  evaluateRunDiagnostics: vi.fn(),
}));

vi.mock("@/api/websocket", () => ({
  connectToRun: vi.fn(),
}));

const READY_READINESS = {
  authoring_valid: true,
  execution_ready: true,
  completion_ready: true,
  blockers: [],
};
const BLOCKED_READINESS = {
  authoring_valid: false,
  execution_ready: false,
  completion_ready: false,
  blockers: [],
};

describe("executionStore.validate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: { version: 1, sources: {}, nodes: [], edges: [], outputs: [] },
    } as never);
  });

  it("stores validation result on success", async () => {
    const result: ValidationResult = {
      is_valid: true,
      summary: "All checks passed",
      checks: [],
      errors: [],
      warnings: [],
      readiness: READY_READINESS,
    };

    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockResolvedValue(result);

    await useExecutionStore.getState().validate("session-1");

    const state = useExecutionStore.getState();
    expect(state.validationResult).toEqual(result);
    expect(state.isValidating).toBe(false);
  });

  it("stores validation result on failure without side effects", async () => {
    const failedResult: ValidationResult = {
      is_valid: false,
      summary: "Validation failed",
      checks: [],
      errors: [
        {
          component_id: "llm_extract",
          component_type: "transform",
          message: "Missing required option: model",
          suggestion: "Add a model option",
        },
      ],
      warnings: [],
      readiness: {
        ...BLOCKED_READINESS,
        blockers: [
          {
            code: "settings_load",
            component_id: "llm_extract",
            component_type: "transform",
            detail: "llm_extract",
          },
        ],
      },
    };

    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockResolvedValue(failedResult);

    await useExecutionStore.getState().validate("session-1");

    // validate() should only store the result — no cross-store side effects.
    // Orchestration (system messages, LLM feedback) is handled by subscriptions.
    const state = useExecutionStore.getState();
    expect(state.validationResult).toEqual(failedResult);
    expect(state.isValidating).toBe(false);
    expect(state.error).toBeNull();
  });

  it("sets error state when API call fails", async () => {
    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockRejectedValue({
      status: 500,
      detail: "Internal server error",
    });

    const result = await useExecutionStore.getState().validate("session-1");

    const state = useExecutionStore.getState();
    expect(state.validationResult).toBeNull();
    expect(state.isValidating).toBe(false);
    expect(state.error).toContain("internal error");
    // Catch path must return false so the caller does not cache this version.
    expect(result).toBe(false);
  });

  it("does not store a validation result that resolves after the user switches sessions", async () => {
    const staleResult: ValidationResult = {
      is_valid: true,
      summary: "Stale session result",
      checks: [],
      errors: [],
      warnings: [],
      readiness: READY_READINESS,
    };
    let resolveValidation: (result: ValidationResult) => void = () => {};
    const pendingValidation = new Promise<ValidationResult>((resolve) => {
      resolveValidation = resolve;
    });
    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockReturnValue(pendingValidation);

    const validatePromise = useExecutionStore.getState().validate("session-1");
    useSessionStore.setState({ activeSessionId: "session-2" } as never);
    resolveValidation(staleResult);
    await validatePromise;

    const state = useExecutionStore.getState();
    expect(state.validationResult).toBeNull();
    expect(state.isValidating).toBe(false);
  });

  it("does not store a validation result after the composition version changes", async () => {
    const staleResult: ValidationResult = {
      is_valid: false,
      summary: "Stale version result",
      checks: [],
      errors: [
        {
          component_id: "source",
          component_type: "source",
          message: "Missing path on the old snapshot",
          suggestion: null,
        },
      ],
      warnings: [],
      readiness: {
        ...BLOCKED_READINESS,
        blockers: [
          {
            code: "settings_load",
            component_id: "source",
            component_type: "source",
            detail: "source",
          },
        ],
      },
    };
    let resolveValidation: (result: ValidationResult) => void = () => {};
    const pendingValidation = new Promise<ValidationResult>((resolve) => {
      resolveValidation = resolve;
    });
    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockReturnValue(pendingValidation);

    const validatePromise = useExecutionStore.getState().validate("session-1");
    useSessionStore.setState({
      compositionState: { version: 2, sources: {}, nodes: [], edges: [], outputs: [] },
    } as never);
    resolveValidation(staleResult);
    const applied = await validatePromise;

    const state = useExecutionStore.getState();
    expect(applied).toBe(false);
    expect(state.validationResult).toBeNull();
    expect(state.isValidating).toBe(false);
  });
});

function makeRun(overrides: Partial<Run> & { error?: string | null } = {}): Run {
  return {
    id: "run-1",
    session_id: "session-1",
    status: "running",
    accounting: null,
    started_at: "2026-04-26T05:31:57.000Z",
    finished_at: null,
    composition_version: 1,
    ...overrides,
  } as Run;
}

function makeAccounting(overrides: Partial<RunAccounting> = {}): RunAccounting {
  return {
    source: { rows_processed: 1 },
    sources: { source: { rows_processed: 1 } },
    tokens: {
      emitted: 9_324,
      terminal: 9_324,
      succeeded: 9_323,
      failed: 0,
      structural: 1,
      pending: 0,
    },
    routing: {
      routed_success: 0,
      routed_failure: 0,
      quarantined: 0,
      discarded: 0,
    },
    integrity: {
      closure: "closed",
      missing_terminal_outcomes: 0,
      duplicate_terminal_outcomes: 0,
    },
    ...overrides,
  };
}

function makeDiagnostics(overrides: Partial<RunDiagnostics> = {}): RunDiagnostics {
  return {
    run_id: "run-1",
    landscape_run_id: "run-1",
    run_status: "running",
    cancel_requested: false,
    summary: {
      token_count: 1,
      preview_limit: 50,
      preview_truncated: false,
      state_counts: { completed: 1 },
      operation_counts: { source_load: 1 },
      latest_activity_at: null,
    },
    tokens: [
      {
        token_id: "token-1",
        row_id: "row-1",
        row_index: 0,
        branch_name: null,
        fork_group_id: null,
        join_group_id: null,
        expand_group_id: null,
        step_in_pipeline: null,
        created_at: "2026-04-26T05:31:58.000Z",
        terminal_outcome: "completed",
        states: [
          {
            state_id: "state-1",
            token_id: "token-1",
            node_id: "extract",
            step_index: 1,
            attempt: 0,
            status: "completed",
            duration_ms: 125,
            started_at: "2026-04-26T05:31:58.000Z",
            completed_at: "2026-04-26T05:31:59.000Z",
            error: null,
            success_reason: null,
          },
        ],
      },
    ],
    operations: [],
    artifacts: [],
    failure_detail: null,
    ...overrides,
  };
}

describe("executionStore failed run events", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  it("preserves terminal failed event detail in progress and run list", () => {
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
      progress: {
        source_rows_processed: 0,
        tokens_succeeded: 0,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    useExecutionStore.getState().connectWebSocket("run-1");
    const handlers = (connectToRun as ReturnType<typeof vi.fn>).mock.calls[0][2];
    const failedEvent: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:31:58.000Z",
      event_type: "failed",
      data: {
        status: "failed",
        detail: "Pipeline execution failed (FrameworkBugError)",
        node_id: null,
      },
    };

    handlers.onFailed(failedEvent, failedEvent.data);

    const state = useExecutionStore.getState();
    expect(state.progress?.status).toBe("failed");
    expect(state.progress?.recent_errors[0]).toEqual({
      message: "Pipeline execution failed (FrameworkBugError)",
      node_id: null,
      row_id: null,
    });
    expect(state.runs[0]).toMatchObject({
      status: "failed",
      error: "Pipeline execution failed (FrameworkBugError)",
    });
  });
});

describe("executionStore fanout guard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  const guard = {
    ack_token: "ack-line-explode",
    risk_level: "high" as const,
    summary: "LLM transform 'classify_line' may make an unknown number of OpenRouter calls.",
    risks: [
      {
        node_id: "classify_line",
        provider: "openrouter",
        model: "openai/gpt-4o-mini",
        credential_ref: "secret_ref:OPENROUTER_API_KEY",
        estimated_provider_calls: null,
        provider_calls_per_row: 1,
        upstream_fanout: ["transform:explode_lines:line_explode"],
        risk_level: "high" as const,
        message: "LLM transform 'classify_line' may make one OpenRouter call per expanded row.",
      },
    ],
  };

  it("holds a 428 fanout guard for explicit user confirmation", async () => {
    const { executePipeline } = await import("@/api/client");
    (executePipeline as ReturnType<typeof vi.fn>).mockRejectedValue({
      status: 428,
      detail: guard.summary,
      error_type: "execution_fanout_ack_required",
      fanout_guard: guard,
    });

    const runId = await useExecutionStore.getState().execute("session-1");

    const state = useExecutionStore.getState();
    expect(runId).toBeNull();
    expect(executePipeline).toHaveBeenCalledWith("session-1");
    expect(state.pendingFanoutGuard).toEqual(guard);
    expect(state.pendingFanoutSessionId).toBe("session-1");
    expect(state.isExecuting).toBe(false);
    expect(state.error).toBeNull();
  });

  it("retries execution with the accepted fanout guard token", async () => {
    const { executePipeline } = await import("@/api/client");
    (executePipeline as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce({
        status: 428,
        detail: guard.summary,
        error_type: "execution_fanout_ack_required",
        fanout_guard: guard,
      })
      .mockResolvedValueOnce({ run_id: "run-1" });

    await useExecutionStore.getState().execute("session-1");
    const runId = await useExecutionStore.getState().confirmFanoutExecution();

    const state = useExecutionStore.getState();
    expect(runId).toBe("run-1");
    expect(executePipeline).toHaveBeenLastCalledWith("session-1", {
      accepted: true,
      token: "ack-line-explode",
    });
    expect(state.pendingFanoutGuard).toBeNull();
    expect(state.pendingFanoutSessionId).toBeNull();
    expect(state.activeRunId).toBe("run-1");
  });
});

describe("executionStore.cancel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  it("marks an active run as cancelling while the backend drains work", async () => {
    const { cancelRun } = await import("@/api/client");
    (cancelRun as ReturnType<typeof vi.fn>).mockResolvedValue({
      status: "running",
      cancel_requested: true,
    });
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
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

    await useExecutionStore.getState().cancel("run-1");

    const state = useExecutionStore.getState();
    expect(state.runs[0].cancel_requested).toBe(true);
    expect(state.progress?.cancel_requested).toBe(true);
  });
});

describe("executionStore progress events advance live accounting", () => {
  // The API run record no longer carries best-known live counters. Progress
  // events update state.progress, while completed events attach closed
  // accounting to the matching run record.
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  it("advances live source/token counters without reintroducing legacy run rows", () => {
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
      progress: {
        source_rows_processed: 0,
        tokens_succeeded: 0,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    useExecutionStore.getState().connectWebSocket("run-1");
    const handlers = (connectToRun as ReturnType<typeof vi.fn>).mock.calls[0][2];

    const firstProgress: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:00.000Z",
      event_type: "progress",
      data: {
        source_rows_processed: 50,
        tokens_succeeded: 40,
        tokens_failed: 1,
        tokens_quarantined: 0,
        tokens_routed_success: 7,
        tokens_routed_failure: 2,
      },
    };
    handlers.onProgress(firstProgress, firstProgress.data);

    let state = useExecutionStore.getState();
    expect(state.runs[0]).not.toHaveProperty("rows_processed");
    expect(state.runs[0]).toMatchObject({
      id: "run-1",
      status: "running",
      accounting: null,
      finished_at: null,
    });
    expect(state.progress).toMatchObject({
      source_rows_processed: 50,
      tokens_succeeded: 40,
      tokens_failed: 1,
      tokens_routed_success: 7,
      tokens_routed_failure: 2,
    });

    const secondProgress: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:05.000Z",
      event_type: "progress",
      data: {
        source_rows_processed: 120,
        tokens_succeeded: 95,
        tokens_failed: 3,
        tokens_quarantined: 0,
        tokens_routed_success: 18,
        tokens_routed_failure: 4,
      },
    };
    handlers.onProgress(secondProgress, secondProgress.data);

    state = useExecutionStore.getState();
    expect(state.progress).toMatchObject({
      source_rows_processed: 120,
      tokens_succeeded: 95,
      tokens_failed: 3,
      tokens_routed_success: 18,
      tokens_routed_failure: 4,
    });
  });

  it("leaves state.runs unchanged when progress arrives for an unknown run_id", () => {
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    const otherRun = makeRun({
      id: "run-other",
    });
    useExecutionStore.setState({
      runs: [otherRun],
      activeRunId: "run-1",
      progress: {
        source_rows_processed: 0,
        tokens_succeeded: 0,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    useExecutionStore.getState().connectWebSocket("run-1");
    const handlers = (connectToRun as ReturnType<typeof vi.fn>).mock.calls[0][2];

    const event: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:00.000Z",
      event_type: "progress",
      data: {
        source_rows_processed: 99,
        tokens_succeeded: 72,
        tokens_failed: 9,
        tokens_quarantined: 0,
        tokens_routed_success: 9,
        tokens_routed_failure: 9,
      },
    };
    handlers.onProgress(event, event.data);

    const state = useExecutionStore.getState();
    expect(state.runs).toHaveLength(1);
    expect(state.runs[0]).toEqual(otherRun);
  });

  it("does not zero state.runs[i] counters on error events with null progress", () => {
    // RunEventError carries no accounting. The store must preserve the REST
    // snapshot instead of fabricating zeros during reconnect-before-progress.
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    const restSnapshot = makeRun({ accounting: makeAccounting() });
    useExecutionStore.setState({
      runs: [restSnapshot],
      activeRunId: "run-1",
      progress: null,
    });

    useExecutionStore.getState().connectWebSocket("run-1");
    const handlers = (connectToRun as ReturnType<typeof vi.fn>).mock.calls[0][2];

    const errorEvent: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:00.000Z",
      event_type: "error",
      data: {
        message: "Row-level transform exception",
        node_id: "extract",
        row_id: "row-7",
      },
    };
    handlers.onError(errorEvent, errorEvent.data);

    const state = useExecutionStore.getState();
    expect(state.runs[0]).toMatchObject({
      accounting: makeAccounting(),
    });
  });

  it("stores completed event accounting on progress and matching run", () => {
    const close = vi.fn();
    const accounting = makeAccounting();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
      progress: {
        source_rows_processed: 1,
        tokens_succeeded: 9_000,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    useExecutionStore.getState().connectWebSocket("run-1");
    const handlers = (connectToRun as ReturnType<typeof vi.fn>).mock.calls[0][2];

    const completedEvent: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:08.000Z",
      event_type: "completed",
      data: {
        status: "completed",
        accounting,
        landscape_run_id: "landscape-run-1",
      },
    };
    handlers.onComplete(completedEvent, completedEvent.data);

    const state = useExecutionStore.getState();
    expect(state.progress).toMatchObject({
      source_rows_processed: 1,
      tokens_succeeded: 9_323,
      tokens_failed: 0,
      accounting,
      status: "completed",
    });
    expect(state.runs[0]).toMatchObject({
      status: "completed",
      accounting,
      finished_at: "2026-04-26T05:32:08.000Z",
    });
  });

  it("refreshes the run list after terminal completion so discard summaries reach the UI", async () => {
    const close = vi.fn();
    const accounting = makeAccounting({
      source: { rows_processed: 0 },
      tokens: {
        emitted: 0,
        terminal: 0,
        succeeded: 0,
        failed: 0,
        structural: 0,
        pending: 0,
      },
    });
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    const { fetchRuns } = await import("@/api/client");
    (fetchRuns as ReturnType<typeof vi.fn>).mockResolvedValue([
      makeRun({
        id: "run-1",
        status: "empty",
        accounting,
        discard_summary: {
          total: 2,
          validation_errors: 2,
          transform_errors: 0,
          sink_discards: 0,
          stages: [
            {
              stage: "source_validation",
              node_id: "source_csv_upload",
              count: 2,
            },
          ],
        },
      }),
    ]);
    useSessionStore.setState({ activeSessionId: "session-1" } as never);
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
      progress: {
        source_rows_processed: 0,
        tokens_succeeded: 0,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    useExecutionStore.getState().connectWebSocket("run-1");
    const handlers = (connectToRun as ReturnType<typeof vi.fn>).mock.calls[0][2];
    const completedEvent: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:08.000Z",
      event_type: "completed",
      data: {
        status: "empty",
        accounting,
        landscape_run_id: "landscape-run-1",
      },
    };
    handlers.onComplete(completedEvent, completedEvent.data);

    expect(fetchRuns).toHaveBeenCalledWith("session-1");
  });
});

describe("executionStore run diagnostics", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  it("stores diagnostics snapshots by run id", async () => {
    const diagnostics = makeDiagnostics();
    const { fetchRunDiagnostics } = await import("@/api/client");
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(diagnostics);

    await useExecutionStore.getState().loadRunDiagnostics("run-1");

    const state = useExecutionStore.getState();
    expect(state.diagnosticsByRunId["run-1"]).toEqual(diagnostics);
    expect(state.diagnosticsLoadingByRunId["run-1"]).toBe(false);
    expect(state.diagnosticsErrorByRunId["run-1"]).toBeNull();
  });

  it("stores LLM diagnostics explanations by run id", async () => {
    const { evaluateRunDiagnostics } = await import("@/api/client");
    (evaluateRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "run-1",
      generated_at: "2026-04-26T05:32:00.000Z",
      explanation: "The run is still working and has loaded one token.",
      working_view: {
        headline: "The run is processing data",
        evidence: ["1 token is visible in the runtime trace."],
        meaning: "The run is still working and has loaded one token.",
        next_steps: ["Refresh diagnostics if this does not change soon."],
      },
    });

    await useExecutionStore.getState().evaluateRunDiagnostics("run-1");

    const state = useExecutionStore.getState();
    expect(state.diagnosticsExplanationByRunId["run-1"]).toContain("still working");
    expect(state.diagnosticsWorkingViewByRunId["run-1"]).toEqual({
      headline: "The run is processing data",
      evidence: ["1 token is visible in the runtime trace."],
      meaning: "The run is still working and has loaded one token.",
      next_steps: ["Refresh diagnostics if this does not change soon."],
    });
    expect(state.diagnosticsEvaluatingByRunId["run-1"]).toBe(false);
  });
});
