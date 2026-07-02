import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InlineRunResults } from "./InlineRunResults";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
import { _resetNarrativeModeCacheForTesting } from "@/hooks/useNarrativeMode";
import * as apiClient from "@/api/client";

vi.mock("@/components/execution/ProgressView", () => ({
  ProgressView: () => <div data-testid="progress-view-stub" />,
}));

vi.mock("@/components/inspector/RunOutputsPanel", () => ({
  RunOutputsPanel: ({ runId }: { runId: string }) => (
    <div data-testid="run-outputs-stub" data-run-id={runId} />
  ),
}));

vi.mock("@/components/composer/NarrativeResults", () => ({
  NarrativeResults: () => <div data-testid="narrative-results-stub" />,
}));

describe("InlineRunResults", () => {
  beforeEach(() => {
    _resetNarrativeModeCacheForTesting();
    vi.restoreAllMocks();
    // Default: empty catalog so useNarrativeMode resolves to false unless
    // a specific test overrides it. Without these stubs the hook would
    // make a real fetch against the unconfigured test API client.
    vi.spyOn(apiClient, "listTransforms").mockResolvedValue([] as any);
    vi.spyOn(apiClient, "listSources").mockResolvedValue([] as any);
    vi.spyOn(apiClient, "listSinks").mockResolvedValue([] as any);
    useSessionStore.setState({ compositionState: null } as never);
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

  it("warns when an empty run discarded source-validation rows", () => {
    useExecutionStore.setState({
      activeRunId: null,
      progress: null,
      runs: [
        {
          id: "run-empty",
          session_id: "sess-1",
          status: "empty",
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
        } as never,
      ],
    } as never);

    render(<InlineRunResults />);

    const warning = screen.getByRole("alert");
    expect(warning).toHaveTextContent(/2 rows discarded at source validation/i);
    expect(warning).toHaveTextContent(/source_csv_upload/i);
    expect(warning).toHaveTextContent(/run terminated empty/i);
  });

  it("warns with the transform node when transform validation discards rows", () => {
    useExecutionStore.setState({
      activeRunId: null,
      progress: null,
      runs: [
        {
          id: "run-transform-discard",
          session_id: "sess-1",
          status: "completed_with_failures",
          discard_summary: {
            total: 1,
            validation_errors: 0,
            transform_errors: 1,
            sink_discards: 0,
            stages: [
              {
                stage: "transform_validation",
                node_id: "normalize_url",
                count: 1,
              },
            ],
          },
        } as never,
      ],
    } as never);

    render(<InlineRunResults />);

    const warning = screen.getByRole("alert");
    expect(warning).toHaveTextContent(/1 row discarded at transform validation/i);
    expect(warning).toHaveTextContent(/normalize_url/i);
  });

  it("does not warn for an empty run with zero discard rows", () => {
    useExecutionStore.setState({
      activeRunId: null,
      progress: null,
      runs: [
        {
          id: "run-empty-clean",
          session_id: "sess-1",
          status: "empty",
          discard_summary: null,
        } as never,
      ],
    } as never);

    render(<InlineRunResults />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
      "data-run-id",
      "run-empty-clean",
    );
  });

  it("keeps terminal active run status visible", () => {
    useExecutionStore.setState({
      activeRunId: "run-failed",
      progress: {
        status: "failed",
        recent_errors: [{ message: "boom", node_id: "llm", row_id: null }],
      } as never,
      runs: [],
    } as never);

    render(<InlineRunResults />);

    expect(screen.getByTestId("progress-view-stub")).toBeInTheDocument();
    expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
      "data-run-id",
      "run-failed",
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
      screen.getByRole("button", { name: /past runs \(1\)/i }),
    ).toBeInTheDocument();
  });

  it("does not count the current terminal run as a past run", () => {
    useExecutionStore.setState({
      activeRunId: "run-done",
      progress: {
        status: "completed",
      } as never,
      runs: [{ id: "run-done", session_id: "sess-1", status: "completed" } as never],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
      "data-run-id",
      "run-done",
    );
    expect(
      screen.queryByRole("button", { name: /past runs/i }),
    ).not.toBeInTheDocument();
  });

  it("does not count an in-flight run as a past run", () => {
    useExecutionStore.setState({
      activeRunId: "run-A",
      progress: {
        status: "running",
      } as never,
      runs: [{ id: "run-A", session_id: "sess-1", status: "running" } as never],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("progress-view-stub")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /past runs/i }),
    ).not.toBeInTheDocument();
  });

  // elspeth-90db33baac: an unattached live run (no activeRunId/WebSocket in
  // this tab — reload race, or a run started from another tab) must reach the
  // drawer, where the REST-backed Cancel works without in-memory state.
  it("routes an unattached live run into the past-runs drawer so it stays cancellable", async () => {
    useExecutionStore.setState({
      activeRunId: null,
      progress: null,
      runs: [
        { id: "run-orphan", session_id: "sess-1", status: "running" } as never,
      ],
    } as never);
    const user = userEvent.setup();

    render(<InlineRunResults />);
    await user.click(screen.getByRole("button", { name: /past runs \(1\)/i }));

    expect(screen.getByText("run-orphan")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /cancel run run-orphan/i }),
    ).toBeInTheDocument();
  });

  it("shows only terminal runs in the past-runs drawer while another run is active", async () => {
    useExecutionStore.setState({
      activeRunId: "run-active",
      progress: {
        status: "running",
      } as never,
      runs: [
        { id: "run-active", session_id: "sess-1", status: "running" } as never,
        { id: "run-old-1", session_id: "sess-1", status: "completed" } as never,
      ],
    } as never);
    const user = userEvent.setup();

    render(<InlineRunResults />);
    await user.click(screen.getByRole("button", { name: /past runs \(1\)/i }));

    expect(screen.getByText("run-old-1")).toBeInTheDocument();
    expect(screen.queryByText("run-active")).not.toBeInTheDocument();
  });

  it("collapses run results behind a compact retained summary", async () => {
    useExecutionStore.setState({
      activeRunId: null,
      progress: null,
      runs: [
        {
          id: "run-latest",
          session_id: "sess-1",
          status: "completed",
          accounting: {
            source: { rows_processed: 3 },
            sources: { source: { rows_processed: 3 } },
            tokens: {
              emitted: 3,
              terminal: 3,
              succeeded: 3,
              failed: 0,
              structural: 0,
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
        } as never,
      ],
    } as never);
    const user = userEvent.setup();

    render(<InlineRunResults />);
    expect(screen.getByTestId("run-outputs-stub")).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: /hide run results/i }),
    );

    expect(screen.queryByTestId("run-outputs-stub")).not.toBeInTheDocument();
    expect(screen.getByText(/Completed/i)).toBeInTheDocument();
    expect(screen.getByText(/3 rows/i)).toBeInTheDocument();
    expect(screen.getByText(/3 succeeded/i)).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: /show run results/i }),
    );
    expect(screen.getByTestId("run-outputs-stub")).toBeInTheDocument();
  });

  it("opens and closes the Past runs drawer", async () => {
    useExecutionStore.setState({
      activeRunId: null,
      runs: [
        { id: "run-latest", session_id: "sess-1", status: "completed" } as never,
        { id: "run-old-1", session_id: "sess-1", status: "completed" } as never,
      ],
    } as never);
    const user = userEvent.setup();

    render(<InlineRunResults />);
    await user.click(screen.getByRole("button", { name: /past runs/i }));

    expect(screen.getByRole("dialog", { name: /past pipeline runs/i })).toBeInTheDocument();
    expect(screen.getByText("run-old-1")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /close past runs/i }));
    expect(
      screen.queryByRole("dialog", { name: /past pipeline runs/i }),
    ).not.toBeInTheDocument();
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

  it("shows live progress while the active run row is still loading", () => {
    useExecutionStore.setState({
      activeRunId: "run-ghost",
      progress: {
        status: "running",
      } as never,
      runs: [],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("progress-view-stub")).toBeInTheDocument();
  });

  it("loads the active session's runs when mounted", async () => {
    const loadRuns = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ loadRuns } as never);

    render(<InlineRunResults />);

    await waitFor(() => {
      expect(loadRuns).toHaveBeenCalledWith("sess-1");
    });
  });

  // ==========================================================================
  // Phase 6B Task 7 — narrative-mode XOR dispatch (plan 19b:359, 19b:365).
  //
  // When a composition contains a plugin tagged "narrative-summary" in the
  // catalog, the terminal-run output slot must render <NarrativeResults />
  // *rather than* the existing tabular <RunOutputsPanel />. The two are
  // mutually exclusive at the composition level — XOR, not stacked.
  // ==========================================================================

  it("renders NarrativeResults instead of RunOutputsPanel when a composition plugin has the narrative-summary tag (plan 19b:359, 19b:365)", async () => {
    vi.spyOn(apiClient, "listTransforms").mockResolvedValue([
        { name: "batch_classifier_metrics", capability_tags: ["narrative-summary"] } as any,
    ]);
    useSessionStore.setState({
      compositionState: {
        sources: {},
        nodes: [
          {
            id: "n1",
            node_type: "transform",
            plugin: "batch_classifier_metrics",
            input: "src",
            on_success: null,
            on_error: null,
            options: {},
          },
        ],
        edges: [],
        outputs: [],
        metadata: { name: "demo", description: "" },
        version: 1,
      },
    } as never);
    useExecutionStore.setState({
      activeRunId: "run-narr",
      progress: { status: "completed" } as never,
      runs: [{ id: "run-narr", status: "completed" } as never],
    } as never);

    render(<InlineRunResults />);

    // NarrativeResults must render…
    await waitFor(() => {
      expect(screen.getByTestId("narrative-results-stub")).toBeInTheDocument();
    });
    // …and the tabular RunOutputsPanel must NOT (XOR, not stacked).
    expect(screen.queryByTestId("run-outputs-stub")).not.toBeInTheDocument();
  });

  it("renders RunOutputsPanel and not NarrativeResults when no composition plugin carries the narrative-summary tag", async () => {
    vi.spyOn(apiClient, "listTransforms").mockResolvedValue([
        { name: "passthrough", capability_tags: [] } as any,
    ]);
    useSessionStore.setState({
      compositionState: {
        sources: {},
        nodes: [
          {
            id: "n1",
            node_type: "transform",
            plugin: "passthrough",
            input: "src",
            on_success: null,
            on_error: null,
            options: {},
          },
        ],
        edges: [],
        outputs: [],
        metadata: { name: "demo", description: "" },
        version: 1,
      },
    } as never);
    useExecutionStore.setState({
      activeRunId: "run-tabular",
      progress: { status: "completed" } as never,
      runs: [{ id: "run-tabular", status: "completed" } as never],
    } as never);

    render(<InlineRunResults />);

    // Tabular RunOutputsPanel must render…
    await waitFor(() => {
      expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
        "data-run-id",
        "run-tabular",
      );
    });
    // …and NarrativeResults must NOT.
    expect(screen.queryByTestId("narrative-results-stub")).not.toBeInTheDocument();
  });
});
