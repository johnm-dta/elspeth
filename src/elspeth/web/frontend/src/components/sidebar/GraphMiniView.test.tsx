import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { GraphMiniView } from "./GraphMiniView";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionState } from "@/types/index";

vi.mock("@/components/inspector/GraphView", () => ({
  GraphView: () => <div data-testid="graph-view-stub" />,
}));

describe("GraphMiniView", () => {
  beforeEach(() => {
    useSessionStore.setState({
      compositionState: {
        version: 1,
        source: { plugin: "csv", options: {} } as never,
        nodes: [
          { id: "tx-1", node_type: "transform", plugin: "field_mapper", options: {} } as never,
        ],
        edges: [],
        outputs: [{ name: "out-1", plugin: "stdout", options: {} } as never],
      } as never,
      selectedNodeId: null,
    } as never);
  });

  it("renders an aria-labelled mini graph", () => {
    render(<GraphMiniView />);
    expect(
      screen.getByRole("button", { name: /pipeline graph/i }),
    ).toBeInTheDocument();
  });

  it("renders an empty state when no composition exists", () => {
    useSessionStore.setState({ compositionState: null } as never);
    render(<GraphMiniView />);
    expect(screen.getByText(/no pipeline yet/i)).toBeInTheDocument();
  });

  it("dispatches OPEN_GRAPH_MODAL_EVENT when clicked", () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    render(<GraphMiniView />);

    fireEvent.click(screen.getByRole("button", { name: /pipeline graph/i }));

    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  });

  it("renders the override composition instead of the store when compositionStateOverride is supplied", () => {
    // Store is set to an empty composition (would render the empty
    // state if it were used). Override supplies a populated one — the
    // mini graph must render that instead.
    useSessionStore.setState({ compositionState: null } as never);
    const override: CompositionState = {
      version: 99,
      source: { plugin: "csv", options: {} } as never,
      nodes: [
        { id: "tx-1", node_type: "transform", plugin: "field_mapper", options: {} } as never,
        { id: "tx-2", node_type: "transform", plugin: "select_columns", options: {} } as never,
      ],
      edges: [],
      outputs: [
        { name: "out-1", plugin: "stdout", options: {} } as never,
        { name: "out-2", plugin: "csv_sink", options: {} } as never,
      ],
    } as never;
    render(<GraphMiniView compositionStateOverride={override} />);
    // The empty state must NOT be rendered — override wins.
    expect(screen.queryByText(/no pipeline yet/i)).not.toBeInTheDocument();
    // The aria-labelled clickable wrapper is present.
    expect(
      screen.getByRole("button", { name: /pipeline graph/i }),
    ).toBeInTheDocument();
  });
});
