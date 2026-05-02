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
    rows_failed: 0,
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
        rows_failed: 0,
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
