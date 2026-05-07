import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { SpecView } from "./SpecView";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionState } from "@/types/index";

const DUMMY_NODE = {
  id: "t1",
  node_type: "transform" as const,
  plugin: "uppercase",
  input: "source_out",
  on_success: "main",
  on_error: null,
  options: {},
};

function makeState(
  overrides: Partial<CompositionState> = {},
): CompositionState {
  return {
    id: "test-session",
    version: 1,
    source: null,
    nodes: [DUMMY_NODE],
    edges: [],
    outputs: [],
    metadata: { name: "test", description: "" },
    ...overrides,
  };
}

describe("SpecView validation banners", () => {
  beforeEach(() => {
    useSessionStore.setState({
      compositionState: null,
    });
  });

  it("renders error banner with 'Errors' label", () => {
    useSessionStore.setState({
      compositionState: makeState({
        validation_errors: ["No source configured."],
      }),
    });
    render(<SpecView />);
    expect(screen.getByText("Errors")).toBeInTheDocument();
    expect(screen.getByText("No source configured.")).toBeInTheDocument();
  });

  it("renders warning banner", () => {
    useSessionStore.setState({
      compositionState: makeState({
        validation_warnings: [
          {
            component: "output.orphan",
            message: "Output 'orphan' has no incoming edge — it will never receive data.",
            severity: "medium",
          },
        ],
      }),
    });
    render(<SpecView />);
    expect(screen.getByText("Warnings")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Output 'orphan' has no incoming edge — it will never receive data.",
      ),
    ).toBeInTheDocument();
  });

  it("renders suggestion banner", () => {
    useSessionStore.setState({
      compositionState: makeState({
        validation_suggestions: [
          {
            component: "pipeline",
            message: "Consider adding error routing — rows that fail transforms currently have no explicit destination.",
            severity: "low",
          },
        ],
      }),
    });
    render(<SpecView />);
    expect(screen.getByText(/Suggestions/)).toBeInTheDocument();
  });

  it("hides banners when no errors, warnings, or suggestions", () => {
    useSessionStore.setState({
      compositionState: makeState({
        validation_errors: [],
        validation_warnings: [],
        validation_suggestions: [],
      }),
    });
    render(<SpecView />);
    expect(screen.queryByText("Errors")).not.toBeInTheDocument();
    expect(screen.queryByText("Warnings")).not.toBeInTheDocument();
    expect(screen.queryByText(/Suggestions/)).not.toBeInTheDocument();
  });
});
