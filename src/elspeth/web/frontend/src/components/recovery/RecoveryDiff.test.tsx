import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { RecoveryDiff } from "./RecoveryDiff";
import type {
  CompositionState,
  EdgeSpec,
  NodeSpec,
} from "@/types/api";
import type { SourceSpec } from "@/types/index";

function makeSource(overrides: Partial<SourceSpec> = {}): SourceSpec {
  return {
    plugin: "csv",
    options: { path: "input.csv" },
    on_success: "clean",
    ...overrides,
  };
}

function makeNode(id: string, overrides: Partial<NodeSpec> = {}): NodeSpec {
  return {
    id,
    node_type: "transform",
    plugin: "passthrough",
    input: "source",
    on_success: null,
    on_error: null,
    options: {},
    ...overrides,
  };
}

function makeEdge(
  id: string,
  fromNode: string,
  toNode: string,
  overrides: Partial<EdgeSpec> = {},
): EdgeSpec {
  return {
    id,
    from_node: fromNode,
    to_node: toNode,
    edge_type: "on_success",
    label: null,
    ...overrides,
  };
}

function makeState(overrides: Partial<CompositionState> = {}): CompositionState {
  return {
    id: "state-1",
    version: 1,
    sources: { source: makeSource() },
    nodes: [makeNode("clean")],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
    ...overrides,
  };
}

describe("RecoveryDiff", () => {
  it("renders added removed and changed source entries", () => {
    render(
      <RecoveryDiff
        currentState={makeState({
          sources: { source: makeSource({ plugin: "csv", options: { path: "old.csv" } }) },
        })}
        recoveredState={makeState({
          sources: { source: makeSource({ plugin: "json", options: { path: "new.json" } }) },
        })}
      />,
    );

    expect(screen.getByText("Changed source")).toBeInTheDocument();
    expect(screen.getByText("source")).toBeInTheDocument();
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.getByText("json")).toBeInTheDocument();
  });

  it("handles null source collections", () => {
    render(
      <RecoveryDiff
        currentState={makeState({ sources: {} })}
        recoveredState={makeState({ sources: { source: makeSource() } })}
      />,
    );

    expect(screen.getByText("Added source")).toBeInTheDocument();
    expect(screen.getByText("csv")).toBeInTheDocument();
  });

  it("renders added removed and changed nodes by id", () => {
    render(
      <RecoveryDiff
        currentState={makeState({
          nodes: [
            makeNode("clean", { plugin: "trim" }),
            makeNode("removed", { plugin: "drop" }),
          ],
        })}
        recoveredState={makeState({
          nodes: [
            makeNode("clean", { plugin: "normalize" }),
            makeNode("added", { plugin: "enrich" }),
          ],
        })}
      />,
    );

    expect(screen.getByText("Changed node")).toBeInTheDocument();
    expect(screen.getByText("clean")).toBeInTheDocument();
    expect(screen.getByText("Removed node")).toBeInTheDocument();
    expect(screen.getByText("removed")).toBeInTheDocument();
    expect(screen.getByText("Added node")).toBeInTheDocument();
    expect(screen.getByText("added")).toBeInTheDocument();
  });

  it("renders added removed and changed edges by stable edge identity", () => {
    render(
      <RecoveryDiff
        currentState={makeState({
          edges: [
            makeEdge("clean->old:on_success", "clean", "old"),
            makeEdge("clean->changed:on_success", "clean", "changed"),
          ],
        })}
        recoveredState={makeState({
          edges: [
            makeEdge("clean->changed:on_success", "clean", "changed", {
              label: "next",
            }),
            makeEdge("clean->new:on_success", "clean", "new"),
          ],
        })}
      />,
    );

    expect(screen.getByText("Changed edge")).toBeInTheDocument();
    expect(screen.getByText("clean->changed:on_success")).toBeInTheDocument();
    expect(screen.getByText("Removed edge")).toBeInTheDocument();
    expect(screen.getByText("clean->old:on_success")).toBeInTheDocument();
    expect(screen.getByText("Added edge")).toBeInTheDocument();
    expect(screen.getByText("clean->new:on_success")).toBeInTheDocument();
  });

  it("shows an empty state when current and partial match", () => {
    const state = makeState();
    render(<RecoveryDiff currentState={state} recoveredState={state} />);

    expect(screen.getByText("No pipeline changes to apply.")).toBeInTheDocument();
  });

  it("keeps large diffs compact until details are opened", async () => {
    const user = userEvent.setup();
    const currentState = makeState({ nodes: [] });
    const recoveredState = makeState({
      nodes: Array.from({ length: 1200 }, (_, index) =>
        makeNode(`node-${index}`, { plugin: `plugin-${index}` }),
      ),
    });

    render(
      <RecoveryDiff
        currentState={currentState}
        recoveredState={recoveredState}
      />,
    );

    expect(screen.getAllByText("1200 additions")).not.toHaveLength(0);
    expect(screen.queryByText(/plugin-1199/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show 1200 additions" }));
    expect(screen.getByText(/plugin-1199/)).toBeInTheDocument();
  });
});
