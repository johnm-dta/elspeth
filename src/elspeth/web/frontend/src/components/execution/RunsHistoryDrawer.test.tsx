import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";
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
    useSessionStore.setState({
      activeSessionId: null,
      sessions: [],
    } as never);
  });

  it("lists every run from the store", () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByText(/r1/)).toBeInTheDocument();
    expect(screen.getByText(/r2/)).toBeInTheDocument();
  });

  // elspeth-e1c5ad0b53: run status renders through ui/StatusBadge so the
  // completed_with_failures / empty distinction carries the ⚠ / ∅ glyphs
  // rather than colour alone, and underscores read as spaces.
  it("renders run status as a StatusBadge with the a11y glyph map", () => {
    useExecutionStore.setState({
      runs: [
        { id: "r1", status: "completed_with_failures" } as never,
        { id: "r2", status: "empty" } as never,
      ],
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);

    const withFailures = screen.getByText("completed with failures");
    expect(withFailures).toHaveClass("status-badge", "status-badge-completed");
    expect(withFailures).toHaveTextContent("⚠");

    const empty = screen.getByText("empty");
    expect(empty).toHaveClass("status-badge", "status-badge-empty");
    expect(empty).toHaveTextContent("∅");
  });

  it("calls onClose when the Close button is clicked", async () => {
    const onClose = vi.fn();
    render(<RunsHistoryDrawer onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: /close runs/i }));
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

  // elspeth-ef8c18a6cb (line-item): the empty state must follow the
  // title-first convention (HeaderSessionSwitcher), never the raw UUID.
  it("names the session by title, not UUID, in the empty state", () => {
    const sessionId = "3f2c9a10-0000-0000-0000-00000000abcd";
    useExecutionStore.setState({ runs: [] } as never);
    useSessionStore.setState({
      activeSessionId: sessionId,
      sessions: [
        {
          id: sessionId,
          title: "Colour survey",
          created_at: "",
          updated_at: "",
        },
      ],
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);

    expect(screen.getByText(/no prior runs for "Colour survey"/i)).toBeInTheDocument();
    expect(screen.queryByText(new RegExp(sessionId))).not.toBeInTheDocument();
  });

  it("falls back to 'this session' in the empty state when no title is known", () => {
    useExecutionStore.setState({ runs: [] } as never);
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByText(/no prior runs for this session/i)).toBeInTheDocument();
  });

  // ── REST-backed cancel (elspeth-90db33baac) ────────────────────────────────
  //
  // ProgressView's Cancel needs the in-memory activeRunId + WebSocket; after
  // a reload those are gone. The drawer offers cancel on live rows via the
  // REST endpoint, gated by the same ConfirmDialog pattern.

  it("offers Cancel only on non-terminal runs", () => {
    useExecutionStore.setState({
      runs: [
        { id: "r1", status: "completed" } as never,
        { id: "r2", status: "running" } as never,
        { id: "r3", status: "pending" } as never,
      ],
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);

    expect(
      screen.queryByRole("button", { name: /cancel run r1/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /cancel run r2/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /cancel run r3/i }),
    ).toBeInTheDocument();
  });

  it("cancels a running run through confirm", async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({
      runs: [{ id: "r2", status: "running" } as never],
      cancel,
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /cancel run r2/i }));

    // Confirm gates the REST call.
    expect(cancel).not.toHaveBeenCalled();
    await userEvent.click(
      screen.getByRole("button", { name: /^cancel pipeline$/i }),
    );
    expect(cancel).toHaveBeenCalledWith("r2");
  });

  it("does not cancel when the confirm dialog is dismissed", async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({
      runs: [{ id: "r2", status: "running" } as never],
      cancel,
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /cancel run r2/i }));
    await userEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(cancel).not.toHaveBeenCalled();
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("disables the row Cancel while cancellation is draining", () => {
    useExecutionStore.setState({
      runs: [
        { id: "r2", status: "running", cancel_requested: true } as never,
      ],
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);

    const button = screen.getByRole("button", { name: /cancel run r2/i });
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent(/cancelling/i);
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

  it("shows the stored run failure cause immediately before diagnostics load", async () => {
    const loadRunDiagnostics = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({
      runs: [
        {
          id: "r2",
          status: "failed",
          error: "HTTP 400: max_output_tokens below minimum value",
        } as never,
      ],
      loadRunDiagnostics,
      diagnosticsByRunId: {},
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /show detail for r2/i }));

    expect(screen.getByTestId("run-stored-failure-detail")).toHaveTextContent(
      "max_output_tokens below minimum value",
    );
  });

  it("keeps the stored run failure cause visible when diagnostics have no failure_detail", async () => {
    useExecutionStore.setState({
      runs: [
        {
          id: "r2",
          status: "failed",
          error: "Pipeline aborted before runtime diagnostics were written.",
        } as never,
      ],
      diagnosticsByRunId: { r2: makeDiagnostics({ failure_detail: null }) },
    } as never);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /show detail for r2/i }));

    expect(screen.queryByTestId("run-failure-detail")).not.toBeInTheDocument();
    expect(screen.getByTestId("run-stored-failure-detail")).toHaveTextContent(
      "Pipeline aborted before runtime diagnostics were written.",
    );
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

  // M08 (WCAG 2.4.3): the drawer is aria-modal with no backdrop/inerting, so
  // focus can land on a control behind it (a click or a global shortcut). Tab
  // must then pull focus back into the drawer rather than walk the page behind.
  it("recaptures focus that has escaped the drawer on Tab", () => {
    const outside = document.createElement("button");
    outside.textContent = "underlying control";
    document.body.appendChild(outside);

    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    outside.focus();
    expect(outside).toHaveFocus();

    fireEvent.keyDown(document, { key: "Tab" });

    expect(screen.getByRole("button", { name: /close/i })).toHaveFocus();

    outside.remove();
  });

  // M08 (WCAG 2.4.3): closing the drawer must return focus to the control that
  // opened it, so keyboard users are not dumped at the top of the document.
  it("restores focus to the opener when the drawer unmounts", () => {
    const opener = document.createElement("button");
    opener.textContent = "Open past runs";
    document.body.appendChild(opener);
    opener.focus();
    expect(opener).toHaveFocus();

    const { unmount } = render(<RunsHistoryDrawer onClose={vi.fn()} />);
    // Focus moved into the drawer (the Close button) while open.
    expect(screen.getByRole("button", { name: /close/i })).toHaveFocus();

    unmount();
    expect(opener).toHaveFocus();

    opener.remove();
  });
});
