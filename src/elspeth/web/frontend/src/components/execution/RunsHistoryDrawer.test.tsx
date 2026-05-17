import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";
import { useExecutionStore } from "@/stores/executionStore";
import type { RunDiagnostics } from "@/types/index";

vi.mock("@/components/inspector/RunOutputsPanel", () => ({
  RunOutputsPanel: ({ runId }: { runId: string }) => (
    <div data-testid="run-outputs-panel" data-run-id={runId} />
  ),
}));

function makeDiagnostics(overrides: Partial<RunDiagnostics> = {}): RunDiagnostics {
  return {
    run_id: "r2",
    landscape_run_id: "r2",
    run_status: "failed",
    cancel_requested: false,
    summary: {
      token_count: 1,
      preview_limit: 50,
      preview_truncated: false,
      state_counts: { failed: 1 },
      operation_counts: { runtime_preflight: 1 },
      latest_activity_at: "2026-05-17T00:00:00Z",
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
        created_at: "2026-05-17T00:00:00Z",
        terminal_outcome: "failed",
        states: [
          {
            state_id: "state-1",
            token_id: "token-1",
            node_id: "rate_colours",
            step_index: 0,
            attempt: 0,
            status: "failed",
            duration_ms: 12,
            started_at: "2026-05-17T00:00:00Z",
            completed_at: "2026-05-17T00:00:01Z",
            error: null,
            success_reason: null,
          },
        ],
      },
    ],
    operations: [
      {
        operation_id: "op-1",
        node_id: "rate_colours",
        operation_type: "runtime_preflight",
        status: "failed",
        duration_ms: 12,
        started_at: "2026-05-17T00:00:00Z",
        completed_at: "2026-05-17T00:00:01Z",
        error_message: "HTTP 400",
      },
    ],
    artifacts: [],
    failure_detail: {
      operation_id: "op-1",
      node_id: "rate_colours",
      operation_type: "runtime_preflight",
      error_message: "HTTP 400: max_output_tokens below minimum value",
      failed_at: "2026-05-17T00:00:01Z",
    },
    ...overrides,
  };
}

describe("RunsHistoryDrawer", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      runs: [
        { id: "r1", status: "completed" } as never,
        { id: "r2", status: "failed" } as never,
      ],
      activeRunId: null,
      progress: null,
      diagnosticsByRunId: {},
      diagnosticsLoadingByRunId: {},
      diagnosticsEvaluatingByRunId: {},
      diagnosticsErrorByRunId: {},
      diagnosticsExplanationByRunId: {},
      diagnosticsWorkingViewByRunId: {},
    } as never);
  });

  it("lists every run from the store", () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByText(/r1/)).toBeInTheDocument();
    expect(screen.getByText(/r2/)).toBeInTheDocument();
  });

  it("calls onClose when the Close button is clicked", async () => {
    const onClose = vi.fn();
    render(<RunsHistoryDrawer onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose when Escape is pressed", async () => {
    const onClose = vi.fn();
    render(<RunsHistoryDrawer onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("renders 'No prior runs' when the runs list is empty", () => {
    useExecutionStore.setState({ runs: [] } as never);
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByText(/no prior runs/i)).toBeInTheDocument();
  });

  it("loads and renders diagnostics detail for a selected run", async () => {
    const loadRunDiagnostics = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({
      loadRunDiagnostics,
      diagnosticsByRunId: { r2: makeDiagnostics() },
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /show detail for r2/i }));

    expect(loadRunDiagnostics).toHaveBeenCalledWith("r2");
    expect(screen.getByTestId("run-failure-detail")).toHaveTextContent(
      "max_output_tokens below minimum value",
    );
    expect(screen.getByText("token-1")).toBeInTheDocument();
    expect(screen.getByTestId("run-outputs-panel")).toHaveAttribute("data-run-id", "r2");
  });

  it("renders the diagnostics working view while explanation is pending", async () => {
    useExecutionStore.setState({
      diagnosticsByRunId: { r2: makeDiagnostics() },
      diagnosticsEvaluatingByRunId: { r2: true },
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /show detail for r2/i }));

    expect(screen.getByText("Reading current run evidence")).toBeInTheDocument();
    expect(screen.getByText("1 token is visible in the runtime trace.")).toBeInTheDocument();
  });

  it("requests an LLM diagnostics explanation for a selected run", async () => {
    const evaluateRunDiagnostics = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({
      diagnosticsByRunId: { r2: makeDiagnostics() },
      evaluateRunDiagnostics,
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /show detail for r2/i }));
    await userEvent.click(screen.getByRole("button", { name: /explain/i }));

    expect(evaluateRunDiagnostics).toHaveBeenCalledWith("r2");
  });

  it("moves focus into the drawer on open (Close button receives focus)", () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByRole("button", { name: /close/i })).toHaveFocus();
  });

  it("traps Tab and Shift+Tab inside the drawer", async () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    const closeBtn = screen.getByRole("button", { name: /close/i });
    const firstDetail = screen.getByRole("button", { name: /show detail for r1/i });
    closeBtn.focus();
    await userEvent.tab();
    expect(firstDetail).toHaveFocus();
    await userEvent.tab({ shift: true });
    expect(closeBtn).toHaveFocus();
  });
});
