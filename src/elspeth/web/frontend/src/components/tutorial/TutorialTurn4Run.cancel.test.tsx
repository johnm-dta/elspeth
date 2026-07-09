import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/api/client";
import { TutorialTurn4Run } from "./TutorialTurn4Run";

vi.mock("@/api/client", () => ({
  runTutorialPipeline: vi.fn(),
  cancelTutorialRun: vi.fn(),
}));

function noop(): void {}

// Distinct session ids per test: the StrictMode dedupe cache is module-level
// and keyed by sessionId, so a reused id would replay a previous test's run.

describe("TutorialTurn4Run — honest cancel + rerun", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("cancel aborts the fetch, fires the server-side cancel, and dispatches onCancelled", async () => {
    vi.useFakeTimers();
    // A run that never settles: the cancel path must not depend on the
    // response coming back.
    vi.mocked(api.runTutorialPipeline).mockImplementation(
      () => new Promise(() => {}),
    );
    vi.mocked(api.cancelTutorialRun).mockResolvedValue({ cancelled: true });
    const onCancelled = vi.fn();

    render(
      <TutorialTurn4Run
        sessionId="sess-cancel-click"
        onCompleted={noop}
        onCancelled={onCancelled}
      />,
    );

    // The cancel affordance appears after the show-cancel delay.
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    fireEvent.click(screen.getByRole("button", { name: /cancel run/i }));

    // Client-side abort AND server-side cancel: aborting only the fetch left
    // the backend run executing (the dishonest-cancel bug).
    const [, signal] = vi.mocked(api.runTutorialPipeline).mock.calls[0];
    expect((signal as AbortSignal).aborted).toBe(true);
    expect(api.cancelTutorialRun).toHaveBeenCalledWith("sess-cancel-click");
    expect(onCancelled).toHaveBeenCalledTimes(1);
  });

  it("409 tutorial_run_cancelled lands on the cancelled state instead of a raw error", async () => {
    vi.mocked(api.runTutorialPipeline).mockRejectedValue({
      status: 409,
      detail: "The tutorial run was cancelled.",
      error_type: "tutorial_run_cancelled",
    });
    const onCancelled = vi.fn();

    render(
      <TutorialTurn4Run
        sessionId="sess-409-cancelled"
        onCompleted={noop}
        onCancelled={onCancelled}
      />,
    );

    await waitFor(() => expect(onCancelled).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("run_already_active-style 409 renders the friendly still-finishing copy with Retry", async () => {
    vi.mocked(api.runTutorialPipeline).mockRejectedValue({
      status: 409,
      detail: "run already active for session sess-409-active",
      error_type: "run_already_active",
    });
    const onCancelled = vi.fn();

    render(
      <TutorialTurn4Run
        sessionId="sess-409-active"
        onCompleted={noop}
        onCancelled={onCancelled}
      />,
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/still finishing/i);
    expect(alert).not.toHaveTextContent(/run already active for session/i);
    expect(onCancelled).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /retry/i }),
    ).toBeInTheDocument();
  });

  it("non-409 failures still surface the backend detail", async () => {
    vi.mocked(api.runTutorialPipeline).mockRejectedValue({
      status: 500,
      detail: "pipeline exploded",
    });

    render(
      <TutorialTurn4Run
        sessionId="sess-500"
        onCompleted={noop}
        onCancelled={noop}
      />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "pipeline exploded",
    );
  });
});
