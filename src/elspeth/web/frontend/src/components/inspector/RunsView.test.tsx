import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RunsView } from "./RunsView";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import type { Run, RunAccounting, RunDiagnostics } from "@/types/index";

vi.mock("@/api/client", () => ({
  fetchRuns: vi.fn().mockResolvedValue([]),
  fetchRunDiagnostics: vi.fn(),
  evaluateRunDiagnostics: vi.fn(),
  // RunOutputsPanel mounts whenever a row is expanded — give it an empty
  // manifest by default so tests that don't care about outputs don't fail
  // on an unmocked fetch. Tests that DO care override this.
  fetchRunOutputs: vi.fn().mockResolvedValue({
    run_id: "run-1",
    landscape_run_id: "run-1",
    artifacts: [],
  }),
  fetchRunOutputPreview: vi.fn(),
  downloadRunOutputContent: vi.fn(),
}));

function makeRun(overrides: Partial<Run> & { error?: string | null } = {}): Run {
  return {
    id: "run-1",
    session_id: "session-1",
    status: "failed",
    accounting: null,
    error: null,
    started_at: "2026-04-26T05:31:58.000Z",
    finished_at: "2026-04-26T05:31:59.000Z",
    composition_version: 1,
    ...overrides,
  } as Run;
}

function makeAccounting(overrides: Partial<RunAccounting> = {}): RunAccounting {
  return {
    source: { rows_processed: 1 },
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
      latest_activity_at: "2026-04-26T05:32:00.000Z",
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
    operations: [
      {
        operation_id: "op-1",
        node_id: "source",
        operation_type: "source_load",
        status: "completed",
        duration_ms: 15,
        started_at: "2026-04-26T05:31:57.000Z",
        completed_at: "2026-04-26T05:31:58.000Z",
        error_message: null,
      },
    ],
    artifacts: [
      {
        artifact_id: "artifact-1",
        sink_node_id: "json_out",
        artifact_type: "json",
        path_or_uri: "/tmp/out.json",
        size_bytes: 42,
        created_at: "2026-04-26T05:31:59.000Z",
      },
    ],
    failure_detail: null,
    ...overrides,
  };
}

describe("RunsView", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
    useSessionStore.setState({ activeSessionId: null });
  });

  it("renders the stored failure reason for failed runs", () => {
    useExecutionStore.setState({
      runs: [
        makeRun({
          error: "Pipeline execution failed (FrameworkBugError)",
        }),
      ],
    });

    render(<RunsView />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Pipeline execution failed (FrameworkBugError)",
    );
  });

  it("never renders a negative duration for terminal runs", () => {
    useExecutionStore.setState({
      runs: [
        makeRun({
          started_at: "2026-04-26T05:31:59.500Z",
          finished_at: "2026-04-26T05:31:59.000Z",
        }),
      ],
    });

    render(<RunsView />);

    expect(screen.getByText("0s")).toBeInTheDocument();
    expect(screen.queryByText("-1s")).not.toBeInTheDocument();
  });

  it("renders fan-out accounting without treating token success as source rows", () => {
    useExecutionStore.setState({
      runs: [
        makeRun({
          status: "completed",
          accounting: makeAccounting(),
          error: null,
        }),
      ],
    });

    render(<RunsView />);

    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("1 source row")).toBeInTheDocument();
    expect(screen.getByText("9,324 tokens emitted")).toBeInTheDocument();
    expect(screen.getByText("9,323 tokens succeeded")).toBeInTheDocument();
    expect(screen.getByText("1 structural token")).toBeInTheDocument();
    expect(screen.queryByText("9,323 rows")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText(/No row reached/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Pipeline execution failed/i)).not.toBeInTheDocument();
  });

  it("renders status badge symbols as aria-hidden DOM text", () => {
    useExecutionStore.setState({
      runs: [
        makeRun({
          status: "completed",
          error: null,
        }),
      ],
    });

    render(<RunsView />);

    const label = screen.getByText("completed");
    expect(label).toHaveClass("status-badge-label");
    const badge = label.closest(".status-badge");
    expect(badge).not.toBeNull();

    const icon = badge!.querySelector(".status-badge-icon");
    expect(icon).toHaveAttribute("aria-hidden", "true");
    expect(icon).toHaveTextContent("\u2713");
  });

  it("renders rows routed to the virtual discard sink", () => {
    useExecutionStore.setState({
      runs: [
        makeRun({
          status: "completed",
          accounting: makeAccounting({
            source: { rows_processed: 3 },
            tokens: {
              emitted: 3,
              terminal: 3,
              succeeded: 3,
              failed: 0,
              structural: 0,
              pending: 0,
            },
          }),
          discard_summary: {
            total: 3,
            validation_errors: 1,
            transform_errors: 1,
            sink_discards: 1,
          },
        }),
      ],
    });

    render(<RunsView />);

    expect(screen.getByText("3 discarded")).toBeInTheDocument();
  });

  it("polls session runs while a run is active", async () => {
    vi.useFakeTimers();
    useSessionStore.setState({ activeSessionId: "session-1" });
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
    });
    const { fetchRuns } = await import("@/api/client");
    (fetchRuns as ReturnType<typeof vi.fn>).mockResolvedValue([
      makeRun({ status: "running", error: null }),
    ]);

    render(<RunsView />);

    expect(fetchRuns).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(fetchRuns).toHaveBeenCalledTimes(2);
  });

  it("polls expanded diagnostics while an inspected run is active", async () => {
    vi.useFakeTimers();
    const { fetchRunDiagnostics } = await import("@/api/client");
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(makeDiagnostics());
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
    });

    render(<RunsView />);
    fireEvent.click(screen.getByRole("button", { name: /show detail/i }));

    expect(fetchRunDiagnostics).toHaveBeenCalledTimes(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(fetchRunDiagnostics).toHaveBeenCalledTimes(2);
  });

  it("shows token states and artifacts when diagnostics are opened", async () => {
    const { fetchRunDiagnostics, fetchRunOutputs } = await import("@/api/client");
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(makeDiagnostics());
    // Artifact rendering moved from the diagnostics panel (capped at 3,
    // text-only) to the new RunOutputsPanel (full manifest + actions).
    // The manifest call returns the same artifact the diagnostics
    // payload used to surface, so the visible string assertion still
    // proves the output is reachable through the expanded row.
    (fetchRunOutputs as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "run-1",
      landscape_run_id: "run-1",
      artifacts: [
        {
          artifact_id: "artifact-1",
          sink_node_id: "json_out",
          artifact_type: "file",
          path_or_uri: "file:///tmp/out.json",
          content_hash: "a".repeat(64),
          size_bytes: 42,
          created_at: "2026-04-26T05:31:59.000Z",
          exists_now: true,
          downloadable: true,
        },
      ],
    });
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
    });

    render(<RunsView />);
    await userEvent.click(screen.getByRole("button", { name: /show detail/i }));

    expect(await screen.findByText("token-1")).toBeInTheDocument();
    expect(screen.getByText(/extract completed/)).toBeInTheDocument();
    // Outputs panel renders the basename, not the full path, with a
    // Download anchor available.
    expect(await screen.findByText("out.json")).toBeInTheDocument();
    expect(screen.getByText("Download")).toBeInTheDocument();
  });

  it("renders the LLM explanation for diagnostics", async () => {
    const { fetchRunDiagnostics, evaluateRunDiagnostics } = await import("@/api/client");
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(makeDiagnostics());
    (evaluateRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue({
      run_id: "run-1",
      generated_at: "2026-04-26T05:32:00.000Z",
      explanation: "The run is still working and has saved /tmp/out.json.",
      working_view: {
        headline: "The run has saved output",
        evidence: ["Saved output is visible at /tmp/out.json."],
        meaning: "The run is still working and has saved /tmp/out.json.",
        next_steps: ["Open the saved file when the run completes."],
      },
    });
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
    });

    render(<RunsView />);
    await userEvent.click(screen.getByRole("button", { name: /show detail/i }));
    await userEvent.click(await screen.findByRole("button", { name: /explain/i }));

    expect(await screen.findByText("The run has saved output")).toBeInTheDocument();
    expect(screen.getByText("Saved output is visible at /tmp/out.json.")).toBeInTheDocument();
    expect(await screen.findByText(/saved \/tmp\/out\.json/)).toBeInTheDocument();
  });

  it("renders failure_detail.error_message in the diagnostics panel when present", async () => {
    // Regression for run 8294aab2 on 2026-05-13: the run banner showed only
    // "Pipeline execution failed (RuntimePreflightFailedError)" — the actual
    // cause (HTTP 400, "max_output_tokens below minimum") lived in the audit
    // DB but was invisible in the UI. failure_detail now exposes the chain.
    const { fetchRunDiagnostics } = await import("@/api/client");
    const diagnostics = makeDiagnostics({
      run_status: "failed",
      failure_detail: {
        operation_id: "op-2",
        node_id: "rate_colours",
        operation_type: "runtime_preflight",
        failed_at: "2026-05-13T15:03:00.866Z",
        error_message:
          "pre_flight_failed: llm provider openrouter failed runtime preflight: " +
          "LLMClientError: HTTP 400 | body: " +
          '{"error":{"message":"max_output_tokens below minimum value"}}',
      },
    });
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(diagnostics);
    useExecutionStore.setState({
      runs: [
        makeRun({
          status: "failed",
          error: "Pipeline execution failed (RuntimePreflightFailedError)",
        }),
      ],
      diagnosticsByRunId: { "run-1": diagnostics },
    });

    render(<RunsView />);
    await userEvent.click(screen.getByRole("button", { name: /show detail/i }));

    const detail = await screen.findByTestId("run-failure-detail");
    expect(detail).toHaveTextContent("runtime_preflight failed");
    expect(detail).toHaveTextContent("rate_colours");
    expect(detail).toHaveTextContent("max_output_tokens below minimum value");
    expect(detail).toHaveTextContent("HTTP 400");
  });

  it("does not render failure_detail block when none is set", async () => {
    const { fetchRunDiagnostics } = await import("@/api/client");
    const diagnostics = makeDiagnostics(); // failure_detail: null by default
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(diagnostics);
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
      diagnosticsByRunId: { "run-1": diagnostics },
    });

    render(<RunsView />);
    await userEvent.click(screen.getByRole("button", { name: /show detail/i }));

    expect(screen.queryByTestId("run-failure-detail")).toBeNull();
  });

  it("shows concrete run evidence while the LLM read is pending", async () => {
    const { fetchRunDiagnostics } = await import("@/api/client");
    const diagnostics = makeDiagnostics({
      summary: {
        token_count: 2,
        preview_limit: 50,
        preview_truncated: false,
        state_counts: { completed: 1, running: 1 },
        operation_counts: { source_load: 1 },
        latest_activity_at: "2026-04-26T05:32:00.000Z",
      },
    });
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(diagnostics);
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
      diagnosticsByRunId: {
        "run-1": diagnostics,
      },
      diagnosticsEvaluatingByRunId: { "run-1": true },
    });

    render(<RunsView />);
    await userEvent.click(screen.getByRole("button", { name: /show detail/i }));

    expect(screen.getByText("Reading current run evidence")).toBeInTheDocument();
    expect(screen.getByText("2 tokens are visible in the runtime trace.")).toBeInTheDocument();
    expect(screen.getByText("Node states include completed=1, running=1.")).toBeInTheDocument();
  });
});

describe("RunsView Inspect button a11y", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
    useSessionStore.setState({ activeSessionId: null });
  });

  it("declares aria-expanded reflecting diagnostics panel state", async () => {
    const { fetchRunDiagnostics } = await import("@/api/client");
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(makeDiagnostics());
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
    });
    const user = userEvent.setup();

    render(<RunsView />);

    const inspect = screen.getByRole("button", { name: /show detail/i });
    expect(inspect.getAttribute("aria-expanded")).toBe("false");

    await user.click(inspect);

    const hide = screen.getByRole("button", { name: /hide/i });
    expect(hide.getAttribute("aria-expanded")).toBe("true");
  });

  it("aria-controls IDREF resolves both before and after the panel is expanded", async () => {
    const { fetchRunDiagnostics } = await import("@/api/client");
    (fetchRunDiagnostics as ReturnType<typeof vi.fn>).mockResolvedValue(makeDiagnostics());
    useExecutionStore.setState({
      runs: [makeRun({ status: "running", error: null })],
    });
    const user = userEvent.setup();

    render(<RunsView />);

    const inspect = screen.getByRole("button", { name: /show detail/i });
    const controlsId = inspect.getAttribute("aria-controls");
    expect(controlsId).toBe("run-diagnostics-run-1");
    // Option A: the wrapper div is always in the DOM — IDREF resolves when collapsed
    expect(document.getElementById(controlsId!)).not.toBeNull();

    await user.click(inspect);

    // IDREF must continue to resolve after expansion (panel is now mounted inside the wrapper)
    const hide = screen.getByRole("button", { name: /hide/i });
    const expandedControlsId = hide.getAttribute("aria-controls");
    expect(document.getElementById(expandedControlsId!)).not.toBeNull();
  });
});

describe("RunsView cancelling badge", () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    useExecutionStore.getState().reset();
    useSessionStore.setState({ activeSessionId: null });
  });

  it("uses dedicated cancelling badge class (visual differentiation from fully cancelled)", () => {
    useExecutionStore.setState({
      runs: [
        makeRun({
          status: "running",
          cancel_requested: true,
          error: null,
        }),
      ],
    });

    render(<RunsView />);

    // Get all status badge elements (excluding the duration label which also contains "cancelling")
    const badges = screen.getAllByText(/cancelling/i);
    // The first one is the badge; the second one is the duration span
    const badge = badges[0].closest(".status-badge");
    expect(badge).not.toBeNull();
    // Distinct class — NOT status-badge-cancelled.  This pins the visual
    // differentiation invariant: cancel-pending uses a pulsing-dot glyph,
    // cancelled uses an em-dash glyph (App.css).
    expect(badge).toHaveClass("status-badge-cancelling");
    expect(badge).not.toHaveClass("status-badge-cancelled");
  });
});
