import { describe, it, expect, beforeEach, vi } from "vitest";
import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { GraphView } from "./GraphView";
import { useSessionStore } from "@/stores/sessionStore";
import type { CompositionState, NodeSpec, EdgeSpec } from "@/types/index";

// Mock @xyflow/react — jsdom cannot do DOM measurements required by React Flow.
// Render nodes and edges as simple divs so we can assert on their presence.
vi.mock("@xyflow/react", () => ({
  ReactFlow: ({ nodes, edges, children, colorMode }: any) => (
    <div data-testid="react-flow" data-color-mode={colorMode}>
      {nodes?.map((n: any) => (
        <div key={n.id} data-testid={`node-${n.id}`} style={n.style}>
          {typeof n.data?.label === "string" ? n.data.label : n.data?.label}
        </div>
      ))}
      {edges?.map((e: any) => (
        <div key={e.id} data-testid={`edge-${e.id}`}>
          {e.label}
        </div>
      ))}
      {children}
    </div>
  ),
  Background: ({ color, gap, size }: any) => (
    <div
      data-testid="react-flow-background"
      data-color={color}
      data-gap={gap}
      data-size={size}
    />
  ),
  Controls: ({ showInteractive }: any) => (
    <div
      data-testid="react-flow-controls"
      data-show-interactive={String(showInteractive)}
    />
  ),
  MiniMap: ({ nodeColor, nodeStrokeColor, bgColor, nodeStrokeWidth }: any) => (
    <div
      data-testid="minimap"
      data-bg-color={bgColor}
      data-node-stroke-width={nodeStrokeWidth}
      data-source-color={nodeColor?.({ id: "source" })}
      data-gate-color={nodeColor?.({ id: "quality_gate" })}
      data-sink-color={nodeColor?.({ id: "results" })}
      data-unknown-color={nodeColor?.({ id: "unknown" })}
      data-stroke-color={nodeStrokeColor?.({ id: "source" })}
    />
  ),
}));

vi.mock("@/hooks/useTheme", () => ({
  useTheme: () => ({
    theme: "system",
    resolvedTheme: "light",
    setTheme: vi.fn(),
    toggleTheme: vi.fn(),
  }),
}));

// Mock @dagrejs/dagre — layout is not needed in tests.
vi.mock("@dagrejs/dagre", () => ({
  default: {
    graphlib: {
      Graph: class {
        setDefaultEdgeLabel() {}
        setGraph() {}
        setNode() {}
        setEdge() {}
        node(_id: string) { return { x: 0, y: 0 }; }
      },
    },
    layout() {},
  },
}));

// Mock React Flow CSS to avoid import errors in jsdom.
vi.mock("@xyflow/react/dist/style.css", () => ({}));

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeNode(overrides: Partial<NodeSpec> = {}): NodeSpec {
  return {
    id: "n1",
    node_type: "transform",
    plugin: "llm_transform",
    input: "source_out",
    on_success: "main",
    on_error: null,
    options: {},
    ...overrides,
  };
}

function makeEdge(overrides: Partial<EdgeSpec> = {}): EdgeSpec {
  return {
    id: "e1",
    from_node: "n1",
    to_node: "n2",
    edge_type: "on_success",
    label: null,
    ...overrides,
  };
}

function makeState(overrides: Partial<CompositionState> = {}): CompositionState {
  return {
    id: "test-session",
    version: 1,
    source: null,
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: "test", description: "" },
    ...overrides,
  };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("GraphView", () => {
  beforeEach(() => {
    useSessionStore.setState({ compositionState: null });
    document.documentElement.removeAttribute("style");
  });

  it("renders nodes with type badge and plugin name", () => {
    useSessionStore.setState({
      compositionState: makeState({
        nodes: [makeNode({ id: "classify", node_type: "transform", plugin: "llm_transform" })],
      }),
    });
    render(<GraphView />);
    // The badge renders node.node_type
    expect(screen.getByText("transform")).toBeInTheDocument();
    // The node ID as display name
    expect(screen.getByText("classify")).toBeInTheDocument();
    // The plugin name
    expect(screen.getByText("llm_transform")).toBeInTheDocument();
  });

  it("renders edge labels for on_success", () => {
    useSessionStore.setState({
      compositionState: makeState({
        nodes: [
          makeNode({ id: "n1", node_type: "transform", plugin: "p" }),
          makeNode({ id: "n2", node_type: "transform", plugin: "q" }),
        ],
        edges: [makeEdge({ id: "e1", from_node: "n1", to_node: "n2", edge_type: "on_success" })],
      }),
    });
    render(<GraphView />);
    // EDGE_LABEL_MAP maps on_success -> "success"
    expect(screen.getByText("success")).toBeInTheDocument();
  });

  it("renders edge labels for on_error", () => {
    useSessionStore.setState({
      compositionState: makeState({
        nodes: [
          makeNode({ id: "n1", node_type: "transform", plugin: "p" }),
          makeNode({ id: "n2", node_type: "transform", plugin: "q" }),
        ],
        edges: [makeEdge({ id: "e1", from_node: "n1", to_node: "n2", edge_type: "on_error" })],
      }),
    });
    render(<GraphView />);
    // EDGE_LABEL_MAP maps on_error -> "error"
    expect(screen.getByText("error")).toBeInTheDocument();
  });

  it("shows minimap for >5 nodes", () => {
    const nodes = Array.from({ length: 6 }, (_, i) =>
      makeNode({ id: `n${i}`, node_type: "transform", plugin: "p" }),
    );
    useSessionStore.setState({
      compositionState: makeState({ nodes }),
    });
    render(<GraphView />);
    expect(screen.getByTestId("minimap")).toBeInTheDocument();
  });

  it("hides minimap for <=5 nodes", () => {
    const nodes = Array.from({ length: 3 }, (_, i) =>
      makeNode({ id: `n${i}`, node_type: "transform", plugin: "p" }),
    );
    useSessionStore.setState({
      compositionState: makeState({ nodes }),
    });
    render(<GraphView />);
    expect(screen.queryByTestId("minimap")).not.toBeInTheDocument();
  });

  it("passes resolved theme and token-backed colours into React Flow controls", () => {
    document.documentElement.style.setProperty("--color-badge-source", "#4db89a");
    document.documentElement.style.setProperty("--color-badge-gate", "#c390f9");
    document.documentElement.style.setProperty("--color-badge-sink", "#e07040");
    document.documentElement.style.setProperty("--color-border-strong", "rgba(1, 2, 3, 0.4)");
    document.documentElement.style.setProperty("--color-text-muted", "#7a9a9a");

    const nodes = Array.from({ length: 5 }, (_, i) =>
      makeNode({ id: `n${i}`, node_type: "transform", plugin: "p" }),
    );
    useSessionStore.setState({
      compositionState: makeState({
        source: {
          plugin: "csv",
          options: {},
          on_success: "gate_in",
        },
        nodes: [
          {
            id: "quality_gate",
            node_type: "gate" as const,
            plugin: null,
            input: "gate_in",
            on_success: null,
            on_error: null,
            options: {},
            condition: "row['score'] >= 0.8",
            routes: null,
          },
          ...nodes,
        ],
        outputs: [{ name: "results", plugin: "csv", options: {} }],
      }),
    });

    render(<GraphView />);

    expect(screen.getByTestId("react-flow")).toHaveAttribute("data-color-mode", "light");
    expect(screen.getByTestId("react-flow-background")).toHaveAttribute("data-color", "var(--color-border)");
    expect(screen.getByTestId("react-flow-background")).toHaveAttribute("data-gap", "16");
    expect(screen.getByTestId("react-flow-background")).toHaveAttribute("data-size", "1");

    const minimap = screen.getByTestId("minimap");
    expect(minimap).toHaveAttribute("data-bg-color", "var(--color-surface)");
    expect(minimap).toHaveAttribute("data-source-color", "#4db89a");
    expect(minimap).toHaveAttribute("data-gate-color", "#c390f9");
    expect(minimap).toHaveAttribute("data-sink-color", "#e07040");
    expect(minimap).toHaveAttribute("data-unknown-color", "#7a9a9a");
    expect(minimap).toHaveAttribute("data-stroke-color", "rgba(1, 2, 3, 0.4)");
  });

  it("bridges React Flow CSS variables to the Elspeth theme tokens", () => {
    const appCss = readFileSync("src/App.css", "utf8");

    expect(appCss).toContain("--xy-background-color-default: var(--color-bg);");
    expect(appCss).toContain("--xy-controls-button-background-color-default: var(--color-surface-elevated);");
    expect(appCss).toContain("--xy-controls-button-background-color-hover-default: var(--color-surface-raised);");
    expect(appCss).toContain("--xy-controls-button-color-default: var(--color-text);");
    expect(appCss).toContain("--xy-minimap-background-color-default: var(--color-surface);");
    expect(appCss).toContain("--xy-minimap-mask-stroke-color-default: var(--color-border-strong);");
    expect(appCss).toContain("--xy-edge-stroke-selected-default: var(--color-focus-ring);");
    expect(appCss).toContain(":root .react-flow.react-flow");
    expect(appCss).toMatch(/\[data-theme="light"\]\s+\.react-flow\.react-flow\s*\{[\s\S]*--xy-minimap-mask-background-color-default:\s*rgba\(15, 45, 53, 0\.12\);/);
    expect(appCss).toContain(".react-flow__controls-button:focus-visible");
    expect(appCss).toContain("outline: 2px solid var(--color-focus-ring);");
  });

  // Edge inference tests — verify connection point matching
  describe("edge inference via connection points", () => {
    it("infers source→transform edge when node.input matches source.on_success", () => {
      // This is the ELSPETH connection model: source.on_success is a connection point
      // name that must match node.input for data to flow.
      useSessionStore.setState({
        compositionState: makeState({
          source: {
            plugin: "text",
            options: {},
            on_success: "transform_in",  // Connection point name
          },
          nodes: [
            makeNode({
              id: "my_transform",
              input: "transform_in",  // Matches source.on_success
              on_success: "results",
            }),
          ],
          outputs: [{ name: "results", plugin: "csv", options: {} }],
          edges: [],  // No explicit edges — should be inferred
        }),
      });
      render(<GraphView />);
      // Should infer edge from source to my_transform
      expect(screen.getByTestId("edge-inferred-conn-source-my_transform")).toBeInTheDocument();
    });

    it("infers transform→transform edge when inputs match on_success values", () => {
      useSessionStore.setState({
        compositionState: makeState({
          source: {
            plugin: "csv",
            options: {},
            on_success: "step1_in",
          },
          nodes: [
            makeNode({
              id: "transform1",
              input: "step1_in",
              on_success: "step2_in",
            }),
            makeNode({
              id: "transform2",
              input: "step2_in",
              on_success: "results",
            }),
          ],
          outputs: [{ name: "results", plugin: "csv", options: {} }],
          edges: [],
        }),
      });
      render(<GraphView />);
      // Should infer: source → transform1 → transform2
      expect(screen.getByTestId("edge-inferred-conn-source-transform1")).toBeInTheDocument();
      expect(screen.getByTestId("edge-inferred-conn-transform1-transform2")).toBeInTheDocument();
    });

    it("infers error routing via connection points", () => {
      // Error handler receives rows via on_error connection point matching
      useSessionStore.setState({
        compositionState: makeState({
          source: {
            plugin: "csv",
            options: {},
            on_success: "process_in",
          },
          nodes: [
            makeNode({
              id: "processor",
              input: "process_in",
              on_success: "results",
              on_error: "error_handler_in",  // Connection point for error routing
            }),
            makeNode({
              id: "error_handler",
              input: "error_handler_in",  // Receives errors from processor
              on_success: "errors",
            }),
          ],
          outputs: [
            { name: "results", plugin: "csv", options: {} },
            { name: "errors", plugin: "json", options: {} },
          ],
          edges: [],
        }),
      });
      render(<GraphView />);
      // Error edge should be inferred with error styling
      expect(screen.getByTestId("edge-inferred-conn-processor-error_handler")).toBeInTheDocument();
      // Label should be "error"
      expect(screen.getByText("error")).toBeInTheDocument();
    });

    it("infers gate routes via connection points", () => {
      // Gate routes to different nodes via connection point matching
      useSessionStore.setState({
        compositionState: makeState({
          source: {
            plugin: "csv",
            options: {},
            on_success: "gate_in",
          },
          nodes: [
            {
              id: "quality_gate",
              node_type: "gate" as const,
              plugin: null,
              input: "gate_in",
              on_success: null,
              on_error: null,
              options: {},
              condition: "row['score'] >= 0.8",
              routes: { "true": "high_quality_in", "false": "low_quality_in" },
            },
            makeNode({
              id: "high_quality_handler",
              input: "high_quality_in",
              on_success: "good_output",
            }),
            makeNode({
              id: "low_quality_handler",
              input: "low_quality_in",
              on_success: "review_output",
            }),
          ],
          outputs: [
            { name: "good_output", plugin: "csv", options: {} },
            { name: "review_output", plugin: "csv", options: {} },
          ],
          edges: [],
        }),
      });
      render(<GraphView />);
      // Gate → handlers via route connection matching
      expect(screen.getByTestId("edge-inferred-conn-quality_gate-high_quality_handler")).toBeInTheDocument();
      expect(screen.getByTestId("edge-inferred-conn-quality_gate-low_quality_handler")).toBeInTheDocument();
      // Route labels should be present
      expect(screen.getByText("true")).toBeInTheDocument();
      expect(screen.getByText("false")).toBeInTheDocument();
    });

    it("merges inferred edges with partial explicit edges", () => {
      // When some edges are explicit and others need inference
      useSessionStore.setState({
        compositionState: makeState({
          source: {
            plugin: "csv",
            options: {},
            on_success: "step1_in",
          },
          nodes: [
            makeNode({
              id: "transform1",
              input: "step1_in",
              on_success: "step2_in",
            }),
            makeNode({
              id: "transform2",
              input: "step2_in",
              on_success: "results",
            }),
          ],
          outputs: [{ name: "results", plugin: "csv", options: {} }],
          // Only one explicit edge — the other should be inferred
          edges: [makeEdge({ id: "e1", from_node: "source", to_node: "transform1" })],
        }),
      });
      render(<GraphView />);
      // Explicit edge exists
      expect(screen.getByTestId("edge-e-source-transform1-0")).toBeInTheDocument();
      // Second edge should be inferred (not blocked by explicit edge existing)
      expect(screen.getByTestId("edge-inferred-conn-transform1-transform2")).toBeInTheDocument();
    });

    it("infers transform→sink edges via direct sink references", () => {
      // When on_success points directly to a sink name (not a connection point)
      useSessionStore.setState({
        compositionState: makeState({
          source: {
            plugin: "csv",
            options: {},
            on_success: "process_in",
          },
          nodes: [
            makeNode({
              id: "processor",
              input: "process_in",
              on_success: "results",  // Direct sink reference
              on_error: "errors",     // Direct sink reference
            }),
          ],
          outputs: [
            { name: "results", plugin: "csv", options: {} },
            { name: "errors", plugin: "json", options: {} },
          ],
          edges: [],
        }),
      });
      render(<GraphView />);
      // Sink edges should be inferred
      expect(screen.getByTestId("edge-inferred-sink-processor-results")).toBeInTheDocument();
      expect(screen.getByTestId("edge-inferred-sink-processor-errors-error")).toBeInTheDocument();
    });
  });
});
