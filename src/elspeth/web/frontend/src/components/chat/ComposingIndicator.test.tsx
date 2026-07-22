import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ComposingIndicator, formatElapsed } from "./ComposingIndicator";
import type { ComposerProgressSnapshot, CompositionState } from "@/types/api";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

function makeState(overrides: Partial<CompositionState> = {}): CompositionState {
  return {
    id: "state-1",
    ...compositionStateAuthorityFields,
    version: 1,
    sources: {},
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
    ...overrides,
  };
}

describe("ComposingIndicator", () => {
  it("renders backend composer progress when available", () => {
    const progress: ComposerProgressSnapshot = {
      session_id: "session-1",
      request_id: "message-1",
      phase: "using_tools",
      headline: "The model requested plugin schemas.",
      evidence: ["Checking available source, transform, and sink tools."],
      likely_next: "ELSPETH will use the schemas to choose a pipeline shape.",
      reason: null,
      updated_at: "2026-04-26T10:00:00Z",
    };

    render(
      <ComposingIndicator
        latestRequest="Exploit this HTML into JSON"
        compositionState={makeState()}
        composerProgress={progress}
      />,
    );

    expect(screen.getByText("Working on...")).toBeInTheDocument();
    expect(screen.getByText("The model requested plugin schemas.")).toBeInTheDocument();
    expect(screen.queryByText("What ELSPETH can see")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByText("What ELSPETH can see")).toBeInTheDocument();
    expect(screen.getByText("Checking available source, transform, and sink tools.")).toBeInTheDocument();
    expect(screen.getByText("Likely next")).toBeInTheDocument();
    expect(screen.getByText("ELSPETH will use the schemas to choose a pipeline shape.")).toBeInTheDocument();
    expect(screen.queryByText("Working on: convert HTML into JSON")).not.toBeInTheDocument();
    // Backend-evidenced views must NOT carry the estimated marker
    // (elspeth-b189b5b3b8 part c).
    expect(screen.queryByText("(estimated)")).not.toBeInTheDocument();
    expect(screen.queryByText("Best guess from your request")).not.toBeInTheDocument();
  });

  it("shows a broad-strokes read of an HTML to JSON request", () => {
    render(
      <ComposingIndicator
        latestRequest="Exploit this HTML into JSON"
        compositionState={makeState()}
      />,
    );

    expect(screen.getByText("Working on...")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Working on: convert HTML into JSON");
    expect(
      screen.queryByText("Request focus: turn HTML content into structured JSON."),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByText("Request focus: turn HTML content into structured JSON.")).toBeInTheDocument();
    expect(screen.getByText("Current setup: no input yet, no processing steps, no outputs.")).toBeInTheDocument();
    expect(
      screen.getByText("Likely next move: choose an input, extract the useful fields, then save structured JSON."),
    ).toBeInTheDocument();
  });

  it("marks keyword-guessed working views as estimated, distinct from backend evidence", () => {
    // elspeth-b189b5b3b8 part c: with no backend progress snapshot the view is
    // keyword-guessed from the user's message and must not read as if ELSPETH
    // reported it — visible "(estimated)" marker + the estimated section label
    // + an italicising modifier class the CSS hangs off.
    const { container } = render(
      <ComposingIndicator
        latestRequest="Exploit this HTML into JSON"
        compositionState={makeState()}
      />,
    );

    expect(screen.getByText("(estimated)")).toBeInTheDocument();
    expect(screen.queryByText("Best guess from your request")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByText("Best guess from your request")).toBeInTheDocument();
    expect(screen.queryByText("What ELSPETH can see")).not.toBeInTheDocument();
    expect(
      container.querySelector(".composing-working-view--estimated"),
    ).not.toBeNull();
  });

  it("summarizes existing pipeline shape without plugin jargon", () => {
    render(
      <ComposingIndicator
        latestRequest="Add an output file"
        compositionState={makeState({
          sources: {
            source: {
              plugin: "csv",
              options: {},
              on_success: "extract",
              on_validation_failure: "discard",
            },
          },
          nodes: [
            {
              id: "extract",
              node_type: "transform",
              plugin: "field_mapper",
              input: "source",
              on_success: null,
              on_error: null,
              options: {},
            },
          ],
          outputs: [{ name: "json_out", plugin: "json", options: {} }],
        })}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByText("Current setup: input configured, 1 processing step, 1 output.")).toBeInTheDocument();
    expect(screen.getByText("Request focus: produce or update saved output.")).toBeInTheDocument();
  });

  it("keeps long-running details collapsed until requested", () => {
    render(
      <ComposingIndicator
        latestRequest="Save the rows to a JSON artifact"
        compositionState={makeState()}
      />,
    );

    expect(screen.getByText("Working on: saved output")).toBeInTheDocument();
    expect(screen.queryByText("Likely next")).not.toBeInTheDocument();

    const toggle = screen.getByRole("button", { name: "Show details" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);

    expect(screen.getByText("Likely next")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Hide details" }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("keeps requested details open when the backend snapshot arrives for the same request", () => {
    const { rerender } = render(
      <ComposingIndicator
        latestRequest="Save the rows to a JSON artifact"
        compositionState={makeState()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByText("Likely next")).toBeInTheDocument();

    const progress: ComposerProgressSnapshot = {
      session_id: "session-1",
      request_id: "message-1",
      phase: "using_tools",
      headline: "Saving the JSON artifact.",
      evidence: ["Choosing the output sink."],
      likely_next: "ELSPETH will save the file.",
      reason: null,
      updated_at: "2026-04-26T10:00:00Z",
    };
    rerender(
      <ComposingIndicator
        latestRequest="Save the rows to a JSON artifact"
        compositionState={makeState()}
        composerProgress={progress}
      />,
    );

    expect(screen.getByText("What ELSPETH can see")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Hide details" }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("renders terminal progress as a retained last update", () => {
    const progress: ComposerProgressSnapshot = {
      session_id: "session-1",
      request_id: "message-1",
      phase: "cancelled",
      headline: "Composition stopped before saving.",
      evidence: ["The browser stopped the compose request."],
      likely_next: "Revise the request and send it again.",
      reason: "client_cancelled",
      updated_at: "2026-04-26T10:00:00Z",
    };

    render(<ComposingIndicator composerProgress={progress} />);

    expect(screen.getByText("Last composer update")).toBeInTheDocument();
    expect(screen.getByText("Stopped")).toBeInTheDocument();
    expect(screen.getByText("Composition stopped before saving.")).toBeInTheDocument();
    expect(screen.getByRole("status")).not.toHaveTextContent("Working on...");
    expect(screen.getByRole("status")).not.toHaveTextContent(/\bok\b/i);
  });
});

describe("ComposingIndicator elapsed-time readout", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("counts up in mm:ss while composing and is hidden from assistive tech", () => {
    // elspeth-b189b5b3b8 part a: slow must not read identically to stalled.
    const { container } = render(<ComposingIndicator latestRequest="hello" />);

    const readout = container.querySelector(".composing-elapsed");
    expect(readout).not.toBeNull();
    expect(readout?.textContent).toBe("00:00");
    // The once-per-second tick must not spam the role="status" live region.
    expect(readout?.getAttribute("aria-hidden")).toBe("true");

    act(() => {
      vi.advanceTimersByTime(65_000);
    });
    expect(container.querySelector(".composing-elapsed")?.textContent).toBe("01:05");
  });

  it("drops the readout once a terminal phase lands", () => {
    const progress: ComposerProgressSnapshot = {
      session_id: "session-1",
      request_id: "message-1",
      phase: "complete",
      headline: "Pipeline saved.",
      evidence: ["Saved version 3."],
      likely_next: null,
      reason: null,
      updated_at: "2026-04-26T10:00:00Z",
    };

    const { container } = render(<ComposingIndicator composerProgress={progress} />);
    expect(container.querySelector(".composing-elapsed")).toBeNull();
  });
});

describe("formatElapsed", () => {
  it("formats seconds as zero-padded mm:ss", () => {
    expect(formatElapsed(0)).toBe("00:00");
    expect(formatElapsed(9)).toBe("00:09");
    expect(formatElapsed(65)).toBe("01:05");
    expect(formatElapsed(600)).toBe("10:00");
  });

  it("clamps negative input to zero rather than rendering nonsense", () => {
    expect(formatElapsed(-5)).toBe("00:00");
  });
});

describe("ComposingIndicator live region scope", () => {
  it("keeps role=status on a non-interactive summary subregion", () => {
    // The indicator is mounted OUTSIDE ChatPanel's role="log" container
    // (elspeth-76a0cc485e) so its implicit role="status" politeness is the
    // single live region announcing compose progress. ChatPanel.test.tsx pins
    // the outside-the-log placement; this pins the region's own attributes.
    const { container } = render(<ComposingIndicator />);
    const root = container.firstChild as HTMLElement;
    const status = screen.getByRole("status");
    expect(root.getAttribute("role")).toBeNull();
    expect(status.getAttribute("aria-live")).toBeNull();
    expect(status.querySelector("button")).toBeNull();
  });
});
