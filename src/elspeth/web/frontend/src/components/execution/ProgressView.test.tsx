import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProgressView } from "./ProgressView";
import { useWebSocket } from "@/hooks/useWebSocket";

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(),
}));

describe("ProgressView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders live progress with explicit source and token units", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: {
        source_rows_processed: 1,
        tokens_succeeded: 9_323,
        tokens_failed: 2,
        tokens_quarantined: 1,
        tokens_routed_success: 7,
        tokens_routed_failure: 2,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    render(<ProgressView />);

    expect(screen.getByText("Source Rows")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("Tokens Succeeded")).toBeInTheDocument();
    expect(screen.getByText("9,323")).toBeInTheDocument();
    expect(screen.getByText("Tokens Failed")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("7 routed success")).toBeInTheDocument();
    expect(screen.getByText("2 routed failure")).toBeInTheDocument();
    expect(screen.getByText("1 quarantined")).toBeInTheDocument();
  });

  it("shows a cancelling state after cancel is requested", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: {
        source_rows_processed: 1,
        tokens_succeeded: 0,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        cancel_requested: true,
        accounting: null,
        recent_errors: [],
        status: "running",
      },
    });

    render(<ProgressView />);

    expect(screen.getByText("cancelling")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel pipeline execution" })).not.toBeInTheDocument();
  });

  it("shows closed accounting totals for structural-token DAG completions", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: {
        source_rows_processed: 1,
        tokens_succeeded: 9_323,
        tokens_failed: 0,
        tokens_quarantined: 0,
        tokens_routed_success: 0,
        tokens_routed_failure: 0,
        cancel_requested: false,
        accounting: {
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
        },
        recent_errors: [],
        status: "completed",
      },
    });

    render(<ProgressView />);

    expect(screen.getByLabelText("Run accounting")).toBeInTheDocument();
    expect(screen.getByText("Tokens Emitted")).toBeInTheDocument();
    expect(screen.getAllByText("9,324")).toHaveLength(2);
    expect(screen.getByText("Tokens Terminal")).toBeInTheDocument();
    expect(screen.getByText("Tokens Structural")).toBeInTheDocument();
    expect(screen.getByText("Audit Closure")).toBeInTheDocument();
    expect(screen.getByText("closed")).toBeInTheDocument();
  });
});
