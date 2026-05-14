import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { SpecView } from "./SpecView";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionProposal, CompositionState } from "@/types/index";

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

function makeProposal(
  overrides: Partial<CompositionProposal> = {},
): CompositionProposal {
  return {
    id: "proposal-1",
    session_id: "session-1",
    tool_call_id: "call-1",
    tool_name: "set_pipeline",
    status: "pending",
    summary: "Replace the pipeline.",
    rationale: "Requested by the current composer turn.",
    affects: ["graph", "validation", "yaml"],
    arguments_redacted_json: {},
    base_state_id: null,
    committed_state_id: null,
    audit_event_id: "event-1",
    created_at: "2026-05-14T00:00:00Z",
    updated_at: "2026-05-14T00:00:00Z",
    ...overrides,
  };
}

describe("SpecView validation banners", () => {
  beforeEach(() => {
    useSessionStore.setState({
      compositionState: null,
      compositionProposals: [],
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

  it("renders pending proposal rows when proposals affect graph", () => {
    useSessionStore.setState({
      compositionState: makeState(),
      compositionProposals: [makeProposal()],
    });

    render(<SpecView />);

    expect(screen.getByText("Pending proposal #1")).toBeInTheDocument();
    expect(screen.getByText("Replace the pipeline.")).toBeInTheDocument();
  });
});
