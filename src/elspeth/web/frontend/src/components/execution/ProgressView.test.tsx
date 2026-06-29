import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProgressView } from "./ProgressView";
import { useWebSocket } from "@/hooks/useWebSocket";

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(),
}));

// Minimal RunProgress-shaped fixture; tests override only the fields they care
// about. The useWebSocket mock returns it untyped, mirroring the inline objects
// used by the other cases in this file.
function progressFixture(overrides: Record<string, unknown> = {}) {
  return {
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
    ...overrides,
  };
}

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

  // M07 (WCAG 4.1.3): a single polite live region announces the run phase for
  // every terminal status, not just cancelled.
  it("announces the running phase through a polite live region", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: progressFixture({ status: "running" }),
    });

    render(<ProgressView />);

    const region = screen.getByRole("status");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveTextContent("Pipeline running.");
  });

  it("announces a completed terminal transition with totals via the live region", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: progressFixture({
        status: "completed",
        source_rows_processed: 3,
        tokens_succeeded: 2,
        tokens_failed: 1,
      }),
    });

    render(<ProgressView />);

    const region = screen.getByRole("status");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveTextContent(/Pipeline completed.*3 rows, 2 succeeded, 1 failed\./);
  });

  it("distinguishes completed-with-failures in the live announcement", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: progressFixture({
        status: "completed_with_failures",
        source_rows_processed: 4,
        tokens_succeeded: 3,
        tokens_failed: 1,
      }),
    });

    render(<ProgressView />);

    expect(screen.getByRole("status")).toHaveTextContent(
      /Pipeline completed with failures.*4 rows, 3 succeeded, 1 failed\./,
    );
  });

  it("announces a failed terminal transition even when recent errors are present", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: progressFixture({
        status: "failed",
        source_rows_processed: 5,
        tokens_succeeded: 1,
        tokens_failed: 4,
        recent_errors: [{ node_id: "rate_colours", message: "HTTP 400", row_id: null }],
      }),
    });

    render(<ProgressView />);

    const region = screen.getByRole("status");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveTextContent(/Pipeline failed.*5 rows, 1 succeeded, 4 failed\./);
  });

  it("announces cancellation through the live region (visible message is visual-only)", () => {
    (useWebSocket as ReturnType<typeof vi.fn>).mockReturnValue({
      activeRunId: "run-1",
      wsDisconnected: false,
      progress: progressFixture({ status: "cancelled" }),
    });

    render(<ProgressView />);

    // The live region is the single announcement source; the visible
    // ``progress-cancelled-msg`` no longer carries a competing role.
    const region = screen.getByRole("status");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toHaveTextContent("Pipeline execution was cancelled.");
    expect(screen.getAllByText("Pipeline execution was cancelled.")).toHaveLength(2);
  });
});
