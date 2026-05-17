import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { InlineRunResults } from "./InlineRunResults";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

vi.mock("@/components/execution/ProgressView", () => ({
  ProgressView: () => <div data-testid="progress-view-stub" />,
}));

vi.mock("@/components/inspector/RunOutputsPanel", () => ({
  RunOutputsPanel: ({ runId }: { runId: string }) => (
    <div data-testid="run-outputs-stub" data-run-id={runId} />
  ),
}));

describe("InlineRunResults", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      runs: [],
      activeRunId: null,
      progress: null,
      diagnosticsByRunId: {},
      diagnosticsLoadingByRunId: {},
      diagnosticsEvaluatingByRunId: {},
      diagnosticsErrorByRunId: {},
      diagnosticsExplanationByRunId: {},
      diagnosticsWorkingViewByRunId: {},
      validationResult: null,
      pendingFanoutGuard: null,
      pendingFanoutSessionId: null,
      isValidating: false,
      isExecuting: false,
      wsDisconnected: false,
      error: null,
    } as never);
    useSessionStore.setState({
      activeSessionId: "sess-1",
    } as never);
  });

  it("renders nothing when there are no runs", () => {
    const { container } = render(<InlineRunResults />);
    expect(container.querySelector("[data-testid='progress-view-stub']")).toBeNull();
    expect(container.querySelector("[data-testid='run-outputs-stub']")).toBeNull();
  });

  it("renders ProgressView for an active running run", () => {
    useExecutionStore.setState({
      activeRunId: "run-A",
      progress: {
        status: "running",
      } as never,
      runs: [{ id: "run-A", status: "running" } as never],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("progress-view-stub")).toBeInTheDocument();
  });

  it("renders RunOutputsPanel for a terminal active run", () => {
    useExecutionStore.setState({
      activeRunId: "run-B",
      progress: {
        status: "completed",
      } as never,
      runs: [{ id: "run-B", status: "completed" } as never],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
      "data-run-id",
      "run-B",
    );
  });

  it("renders RunOutputsPanel for the most recent terminal run when there is no active run", () => {
    useExecutionStore.setState({
      activeRunId: null,
      progress: null,
      runs: [
        { id: "run-latest", status: "completed" } as never,
        { id: "run-older", status: "failed" } as never,
      ],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
      "data-run-id",
      "run-latest",
    );
  });

  it("exposes a 'Past runs' button when historical runs exist", () => {
    useExecutionStore.setState({
      activeRunId: null,
      runs: [
        { id: "run-old-1", status: "completed" } as never,
        { id: "run-old-2", status: "completed" } as never,
      ],
    } as never);
    render(<InlineRunResults />);
    expect(
      screen.getByRole("button", { name: /past runs/i }),
    ).toBeInTheDocument();
  });

  it("hides the 'Past runs' button when no historical runs exist", () => {
    useExecutionStore.setState({
      activeRunId: null,
      runs: [],
    } as never);
    render(<InlineRunResults />);
    expect(
      screen.queryByRole("button", { name: /past runs/i }),
    ).not.toBeInTheDocument();
  });

  it("handles a dangling activeRunId (race during reconnect / session switch)", () => {
    useExecutionStore.setState({
      activeRunId: "run-ghost",
      progress: null,
      runs: [],
    } as never);
    const { container } = render(<InlineRunResults />);
    expect(container.querySelector("[data-testid='progress-view-stub']")).toBeNull();
    expect(container.querySelector("[data-testid='run-outputs-stub']")).toBeNull();
  });
});
