import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComposingIndicator } from "./ComposingIndicator";
import type { ComposerProgressSnapshot, CompositionState } from "@/types/api";

function makeState(overrides: Partial<CompositionState> = {}): CompositionState {
  return {
    id: "state-1",
    version: 1,
    source: null,
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
    expect(screen.getByText("What ELSPETH can see")).toBeInTheDocument();
    expect(screen.getByText("Checking available source, transform, and sink tools.")).toBeInTheDocument();
    expect(screen.getByText("Likely next")).toBeInTheDocument();
    expect(screen.getByText("ELSPETH will use the schemas to choose a pipeline shape.")).toBeInTheDocument();
    expect(screen.queryByText("Working on: convert HTML into JSON")).not.toBeInTheDocument();
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
    expect(screen.getByText("Request focus: turn HTML content into structured JSON.")).toBeInTheDocument();
    expect(screen.getByText("Current setup: no input yet, no processing steps, no outputs.")).toBeInTheDocument();
    expect(
      screen.getByText("Likely next move: choose an input, extract the useful fields, then save structured JSON."),
    ).toBeInTheDocument();
  });

  it("summarizes existing pipeline shape without plugin jargon", () => {
    render(
      <ComposingIndicator
        latestRequest="Add an output file"
        compositionState={makeState({
          source: {
            plugin: "csv",
            options: {},
            on_success: "extract",
            on_validation_failure: "discard",
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

    expect(screen.getByText("Current setup: input configured, 1 processing step, 1 output.")).toBeInTheDocument();
    expect(screen.getByText("Request focus: produce or update saved output.")).toBeInTheDocument();
  });
});
