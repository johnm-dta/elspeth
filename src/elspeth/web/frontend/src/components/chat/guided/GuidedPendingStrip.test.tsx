import { render, screen } from "@testing-library/react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  GuidedPendingStrip,
  STOP_ARMING_DELAY_MS,
} from "./GuidedPendingStrip";
import type { ComposerProgressSnapshot } from "@/types/api";

function snapshot(
  overrides: Partial<ComposerProgressSnapshot>,
): ComposerProgressSnapshot {
  return {
    session_id: "sess-1",
    request_id: "req-1",
    phase: "calling_model",
    headline: "Reading your three project pages",
    evidence: ["Fetched page 1 of 3"],
    likely_next: "Extract the useful fields",
    reason: null,
    updated_at: "2026-07-03T00:00:00Z",
    ...overrides,
  };
}

describe("GuidedPendingStrip", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the backend headline while a compose phase is live", () => {
    render(<GuidedPendingStrip composerProgress={snapshot({})} />);
    expect(
      screen.getByText("Reading your three project pages"),
    ).toBeInTheDocument();
  });

  it("falls back to the generic headline when no progress snapshot exists", () => {
    render(<GuidedPendingStrip composerProgress={null} />);
    expect(screen.getByText("Working on it...")).toBeInTheDocument();
  });

  it("treats idle and terminal snapshots as stale (previous turn's carry-over)", () => {
    // The progress poller lags the pending flags: a snapshot left in
    // "complete"/"cancelled"/"failed"/"idle" describes the PREVIOUS compose,
    // and echoing its headline would misattribute it to the new request.
    for (const phase of ["idle", "complete", "failed", "cancelled"] as const) {
      const { unmount } = render(
        <GuidedPendingStrip composerProgress={snapshot({ phase })} />,
      );
      expect(screen.getByText("Working on it...")).toBeInTheDocument();
      expect(
        screen.queryByText("Reading your three project pages"),
      ).not.toBeInTheDocument();
      unmount();
    }
  });

  it("renders Stop with the app's existing abort name only when onStop is given", () => {
    const onStop = vi.fn();
    const { unmount } = render(
      <GuidedPendingStrip composerProgress={null} onStop={onStop} />,
    );
    const stop = screen.getByRole("button", { name: "Stop composing" });
    expect(stop).toHaveTextContent("Stop");
    unmount();

    // No abortable fetch → no dead interrupt (elspeth-fb4464cdf0).
    render(<GuidedPendingStrip composerProgress={null} />);
    expect(
      screen.queryByRole("button", { name: "Stop composing" }),
    ).not.toBeInTheDocument();
  });

  it("keeps Stop OUTSIDE the role=status live region", () => {
    // A control inside a live region gets re-announced on unrelated content
    // mutations; the status region wraps only the announcement content.
    render(<GuidedPendingStrip composerProgress={null} onStop={vi.fn()} />);
    const status = screen.getByRole("status");
    const stop = screen.getByRole("button", { name: "Stop composing" });
    expect(status.contains(stop)).toBe(false);
  });

  it("hides the ticking elapsed readout from AT and keeps the pulse decorative", () => {
    const { container } = render(
      <GuidedPendingStrip composerProgress={null} />,
    );
    const elapsed = container.querySelector(".composing-elapsed");
    expect(elapsed).not.toBeNull();
    expect(elapsed!.getAttribute("aria-hidden")).toBe("true");
    const pulse = container.querySelector(".composing-pulse");
    expect(pulse!.getAttribute("aria-hidden")).toBe("true");
  });

  it("is a programmatic-only focus target (tabIndex=-1 wrapper)", () => {
    const { container } = render(
      <GuidedPendingStrip composerProgress={null} />,
    );
    const strip = container.querySelector(".guided-pending-strip");
    expect(strip).not.toBeNull();
    expect(strip!.getAttribute("tabindex")).toBe("-1");
  });

  it("arms Stop against pointer double-clicks: pointer-suppressed on mount, live after the delay", () => {
    render(<GuidedPendingStrip composerProgress={null} onStop={vi.fn()} />);
    const stop = screen.getByRole("button", { name: "Stop composing" });
    // Mounted where Send just sat — the second click of a muscle-memory
    // double-click must not abort the request the user just started.
    expect(stop.className).toContain("guided-pending-strip-stop--arming");
    act(() => {
      vi.advanceTimersByTime(STOP_ARMING_DELAY_MS);
    });
    expect(stop.className).not.toContain("guided-pending-strip-stop--arming");
  });

  it("shows optional substep progress with the current item marked", () => {
    render(
      <GuidedPendingStrip
        composerProgress={snapshot({})}
        substeps={["Read output request", "Choose sink shape", "Prepare JSON file"]}
        activeSubstepIndex={1}
      />,
    );

    const list = screen.getByRole("list", { name: "Tutorial step progress" });
    expect(list).toHaveTextContent("Read output request");
    expect(list).toHaveTextContent("Choose sink shape");
    expect(list).toHaveTextContent("Prepare JSON file");
    expect(screen.getByText("Choose sink shape")).toHaveAttribute(
      "aria-current",
      "step",
    );
  });

  it("omits substep progress when no substeps are supplied", () => {
    render(<GuidedPendingStrip composerProgress={snapshot({})} />);

    expect(
      screen.queryByRole("list", { name: "Tutorial step progress" }),
    ).not.toBeInTheDocument();
  });
});
