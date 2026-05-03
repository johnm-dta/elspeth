import { describe, it, expect, vi, beforeEach } from "vitest";
import { useExecutionStore } from "./executionStore";
import { connectToRun } from "@/api/websocket";
import type { Run, RunDiagnostics, RunEvent, ValidationResult } from "@/types/index";

// Mock the API client
vi.mock("@/api/client", () => ({
  validatePipeline: vi.fn(),
  fetchRuns: vi.fn().mockResolvedValue([]),
  fetchRunDiagnostics: vi.fn(),
  evaluateRunDiagnostics: vi.fn(),
}));

vi.mock("@/api/websocket", () => ({
  connectToRun: vi.fn(),
}));

describe("executionStore.validate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  it("stores validation result on success", async () => {
    const result: ValidationResult = {
      is_valid: true,
      summary: "All checks passed",
      checks: [],
      errors: [],
      warnings: [],
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
    };

    const { validatePipeline } = await import("@/api/client");
    (validatePipeline as ReturnType<typeof vi.fn>).mockResolvedValue(failedResult);

    await useExecutionStore.getState().validate("session-1");

    // validate() should only store the result — no cross-store side effects.
    // Orchestration (system messages, LLM feedback) is handled by InspectorPanel.
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

    await useExecutionStore.getState().validate("session-1");

    const state = useExecutionStore.getState();
    expect(state.validationResult).toBeNull();
    expect(state.isValidating).toBe(false);
    expect(state.error).toContain("internal error");
  });
});

function makeRun(overrides: Partial<Run> & { error?: string | null } = {}): Run {
  return {
    id: "run-1",
    session_id: "session-1",
    status: "running",
    rows_processed: 0,
    rows_succeeded: 0,
    rows_failed: 0,
    rows_quarantined: 0,
    rows_routed_success: 0,
    rows_routed_failure: 0,
    started_at: "2026-04-26T05:31:57.000Z",
    finished_at: null,
    composition_version: 1,
    ...overrides,
  } as Run;
}

function makeDiagnostics(overrides: Partial<RunDiagnostics> = {}): RunDiagnostics {
  return {
    run_id: "run-1",
    landscape_run_id: "run-1",
    run_status: "running",
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
        rows_processed: 0,
        rows_succeeded: 0,
        rows_failed: 0,
        rows_quarantined: 0,
        rows_routed_success: 0,
        rows_routed_failure: 0,
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

describe("executionStore progress events advance runs list", () => {
  // elspeth-0c076ad374 — pre-CR-3 the runs-list row counters froze at the
  // REST snapshot until the terminal event arrived, while the live
  // ProgressView ticked from state.progress. The fix patches state.runs[i]
  // on `progress` events too, so both surfaces advance in lockstep.
  beforeEach(() => {
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
  });

  it("advances state.runs[i] row counters on progress events", () => {
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    useExecutionStore.setState({
      runs: [makeRun()],
      activeRunId: "run-1",
      progress: {
        rows_processed: 0,
        rows_succeeded: 0,
        rows_failed: 0,
        rows_quarantined: 0,
        rows_routed_success: 0,
        rows_routed_failure: 0,
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
        rows_processed: 50,
        rows_succeeded: 40,
        rows_failed: 1,
        rows_quarantined: 0,
        rows_routed_success: 7,
        rows_routed_failure: 2,
      },
    };
    handlers.onProgress(firstProgress, firstProgress.data);

    let state = useExecutionStore.getState();
    expect(state.runs[0]).toMatchObject({
      id: "run-1",
      status: "running",
      rows_processed: 50,
      rows_failed: 1,
      rows_routed_success: 7,
      rows_routed_failure: 2,
      finished_at: null,
    });
    // ProgressView slot must continue to mirror the same counts.
    expect(state.progress).toMatchObject({
      rows_processed: 50,
      rows_failed: 1,
      rows_routed_success: 7,
      rows_routed_failure: 2,
    });

    const secondProgress: RunEvent = {
      run_id: "run-1",
      timestamp: "2026-04-26T05:32:05.000Z",
      event_type: "progress",
      data: {
        rows_processed: 120,
        rows_succeeded: 95,
        rows_failed: 3,
        rows_quarantined: 0,
        rows_routed_success: 18,
        rows_routed_failure: 4,
      },
    };
    handlers.onProgress(secondProgress, secondProgress.data);

    state = useExecutionStore.getState();
    expect(state.runs[0]).toMatchObject({
      rows_processed: 120,
      rows_failed: 3,
      rows_routed_success: 18,
      rows_routed_failure: 4,
      finished_at: null,
    });
  });

  it("leaves state.runs unchanged when progress arrives for an unknown run_id", () => {
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    const otherRun = makeRun({
      id: "run-other",
      rows_processed: 42,
      rows_failed: 1,
      rows_routed_success: 3,
      rows_routed_failure: 0,
    });
    useExecutionStore.setState({
      runs: [otherRun],
      activeRunId: "run-1",
      progress: {
        rows_processed: 0,
        rows_succeeded: 0,
        rows_failed: 0,
        rows_quarantined: 0,
        rows_routed_success: 0,
        rows_routed_failure: 0,
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
        rows_processed: 99,
        rows_succeeded: 72,
        rows_failed: 9,
        rows_quarantined: 0,
        rows_routed_success: 9,
        rows_routed_failure: 9,
      },
    };
    handlers.onProgress(event, event.data);

    const state = useExecutionStore.getState();
    expect(state.runs).toHaveLength(1);
    expect(state.runs[0]).toEqual(otherRun);
  });

  it("does not zero state.runs[i] counters on error events with null progress", () => {
    // Locks in a deliberate divergence from the issue's wording
    // (elspeth-0c076ad374). RunEventError carries no row counters; if
    // applyRunEvent patched runs[] on error events too, it would fall back
    // to `state.progress?.* ?? 0` and clobber a real REST snapshot in the
    // reconnect-before-first-progress case. Progress events are the only
    // safe surface for live runs[] updates.
    const close = vi.fn();
    (connectToRun as ReturnType<typeof vi.fn>).mockReturnValue({ close });
    const restSnapshot = makeRun({
      rows_processed: 100,
      rows_failed: 5,
      rows_routed_success: 12,
      rows_routed_failure: 3,
    });
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
      rows_processed: 100,
      rows_failed: 5,
      rows_routed_success: 12,
      rows_routed_failure: 3,
    });
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
