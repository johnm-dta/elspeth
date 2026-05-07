// ============================================================================
// GraphView
//
// React Flow (@xyflow/react) DAG visualisation of the current CompositionState.
// Converts nodes and edges to React Flow format with colour-coded node types.
// Read-only (no drag-to-connect, no node deletion). Pan and zoom enabled.
// Auto-layout using dagre (@dagrejs/dagre) for hierarchical top-to-bottom.
//
// ARIA: container has aria-label describing the pipeline structure.
// "Pipeline graph with N components. Use the Spec tab for keyboard-accessible
// detail."
//
// Empty state when no nodes.
// ============================================================================

import { useMemo, useCallback } from "react";
import {
  ReactFlow,
  type Node,
  type Edge,
  type NodeMouseHandler,
  Background,
  Controls,
  MiniMap,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useTheme } from "@/hooks/useTheme";
import { BADGE_COLORS, BADGE_BACKGROUNDS, EDGE_COLORS, EDGE_LABEL_COLOR, VALIDATION_COLORS } from "@/styles/tokens";
import type { CompositionState } from "@/types/index";

const NODE_WIDTH = 260;
const NODE_HEIGHT = 80;
const FALLBACK_MINIMAP_NODE_COLOR_VAR = "--color-text-muted";
const MINIMAP_NODE_STROKE_COLOR_VAR = "--color-border-strong";

const EDGE_LABEL_MAP: Record<string, string> = {
  on_success: "success",
  on_error: "error",
  route_true: "true",
  route_false: "false",
  fork: "fork",
};

type MiniMapNodeKind = keyof typeof BADGE_COLORS;

function readThemeColor(cssVariableName: string, fallbackVariableName: string): string {
  if (typeof document === "undefined" || typeof getComputedStyle !== "function") {
    return `var(${fallbackVariableName})`;
  }

  const rootStyles = getComputedStyle(document.documentElement);
  return (
    rootStyles.getPropertyValue(cssVariableName).trim() ||
    rootStyles.getPropertyValue(fallbackVariableName).trim() ||
    `var(${fallbackVariableName})`
  );
}

function buildMiniMapNodeKindById(
  compositionState: CompositionState | null,
): Map<string, MiniMapNodeKind> {
  const kindById = new Map<string, MiniMapNodeKind>();
  if (!compositionState) {
    return kindById;
  }

  if (compositionState.source) {
    kindById.set("source", "source");
  }
  for (const node of compositionState.nodes) {
    kindById.set(node.id, node.node_type);
  }
  for (const output of compositionState.outputs) {
    kindById.set(output.name, "sink");
  }

  return kindById;
}

// ── Dagre layout ─────────────────────────────────────────────────────────────

/**
 * Apply dagre layout to nodes and edges, returning positioned React Flow
 * nodes. Layout is top-to-bottom (TB) with reasonable spacing.
 */
function layoutGraph(
  rfNodes: Node[],
  rfEdges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 100 });

  for (const node of rfNodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of rfEdges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  const positionedNodes = rfNodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    };
  });

  return { nodes: positionedNodes, edges: rfEdges };
}

// ── GraphView component ──────────────────────────────────────────────────────

export function GraphView() {
  const compositionState = useSessionStore((s) => s.compositionState);
  const selectedNodeId = useSessionStore((s) => s.selectedNodeId);
  const selectNode = useSessionStore((s) => s.selectNode);
  const { resolvedTheme } = useTheme();

  const validationResult = useExecutionStore((s) => s.validationResult);

  const miniMapNodeKindById = useMemo(
    () => buildMiniMapNodeKindById(compositionState),
    [compositionState],
  );

  const getMiniMapNodeColor = useCallback(
    (node: Node) => {
      const nodeKind = miniMapNodeKindById.get(node.id);
      return readThemeColor(
        nodeKind ? `--color-badge-${nodeKind}` : FALLBACK_MINIMAP_NODE_COLOR_VAR,
        FALLBACK_MINIMAP_NODE_COLOR_VAR,
      );
    },
    [miniMapNodeKindById, resolvedTheme],
  );

  const getMiniMapNodeStrokeColor = useCallback(
    () => readThemeColor(MINIMAP_NODE_STROKE_COLOR_VAR, FALLBACK_MINIMAP_NODE_COLOR_VAR),
    [resolvedTheme],
  );

  // Node click handler — toggle selection
  const onNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      selectNode(selectedNodeId === node.id ? null : node.id);
    },
    [selectedNodeId, selectNode],
  );

  // Pane click handler — deselect when clicking background
  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  // Build a map of component_id → validation severity for border coloring
  const nodeValidationMap = useMemo(() => {
    const map: Record<string, "valid" | "warning" | "error"> = {};
    if (!validationResult) return map;

    // All nodes with errors
    for (const err of validationResult.errors) {
      if (err.component_id) {
        map[err.component_id] = "error";
      }
    }

    // All nodes with warnings (only if not already error)
    if (validationResult.warnings) {
      for (const warn of validationResult.warnings) {
        if (warn.component_id && !map[warn.component_id]) {
          map[warn.component_id] = "warning";
        }
      }
    }

    return map;
  }, [validationResult]);

  // Build a map of component_id → tooltip string with error/warning messages
  const nodeMessageMap = useMemo(() => {
    const map: Record<string, string> = {};
    if (!validationResult) return map;

    // Collect messages per component_id
    const errors: Record<string, string[]> = {};
    const warnings: Record<string, string[]> = {};

    for (const err of validationResult.errors) {
      if (err.component_id) {
        (errors[err.component_id] ??= []).push(err.message);
      }
    }

    if (validationResult.warnings) {
      for (const warn of validationResult.warnings) {
        if (warn.component_id) {
          (warnings[warn.component_id] ??= []).push(warn.message);
        }
      }
    }

    // Merge into tooltip strings
    const allIds = new Set([...Object.keys(errors), ...Object.keys(warnings)]);
    for (const id of allIds) {
      const parts: string[] = [];
      if (errors[id]) {
        parts.push(`Errors:\n${errors[id].map((m) => `- ${m}`).join("\n")}`);
      }
      if (warnings[id]) {
        parts.push(`Warnings:\n${warnings[id].map((m) => `- ${m}`).join("\n")}`);
      }
      map[id] = parts.join("\n\n");
    }

    return map;
  }, [validationResult]);

  const { nodes, edges } = useMemo(() => {
    const hasContent =
      compositionState &&
      (compositionState.source !== null ||
        compositionState.nodes.length > 0 ||
        compositionState.outputs.length > 0);
    if (!hasContent) {
      return { nodes: [] as Node[], edges: [] as Edge[] };
    }

    function makeRfNode(
      id: string,
      typeLabel: string,
      subtitle: string | null,
      badgeBg: string,
      badgeColor: string,
      validationStatus?: "valid" | "warning" | "error",
      validationTooltip?: string,
      isSelected?: boolean,
    ): Node {
      // Selection ring takes priority over validation border
      const borderStyle = isSelected
        ? "2px solid var(--color-selected-ring)"
        : validationStatus === "error"
          ? `2px solid ${VALIDATION_COLORS.invalid}`
          : validationStatus === "warning"
            ? `2px solid ${VALIDATION_COLORS.warning}`
            : validationStatus === "valid"
              ? `2px solid ${VALIDATION_COLORS.valid}`
              : "1px solid var(--color-border-strong)";

      return {
        id,
        data: {
          label: (
            <div
              className="graph-node-content"
              title={validationTooltip ?? (validationStatus === "valid" ? "Valid" : undefined)}
            >
              <div className="graph-node-header">
                <span
                  className="graph-node-badge"
                  style={{ backgroundColor: badgeBg, color: badgeColor }}
                >
                  {typeLabel}
                </span>
                <span className="graph-node-label">
                  {id}
                </span>
                {validationStatus && (
                  <span
                    className="graph-validation-dot"
                    style={{
                      backgroundColor:
                        validationStatus === "error"
                          ? VALIDATION_COLORS.invalid
                          : validationStatus === "warning"
                            ? VALIDATION_COLORS.warning
                            : VALIDATION_COLORS.valid,
                    }}
                    title={
                      validationTooltip
                        ? validationTooltip
                        : validationStatus === "error"
                          ? "Has validation errors"
                          : validationStatus === "warning"
                            ? "Has warnings"
                            : "Valid"
                    }
                  />
                )}
              </div>
              {subtitle && (
                <div className="graph-node-subtitle">
                  {subtitle}
                </div>
              )}
            </div>
          ),
        },
        position: { x: 0, y: 0 },
        style: {
          backgroundColor: "var(--color-surface-elevated)",
          border: borderStyle,
          borderRadius: 8,
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
          padding: 0,
          // Selection box-shadow for extra emphasis
          boxShadow: isSelected ? "0 0 0 3px var(--color-selected-ring)" : undefined,
          cursor: "pointer",
        },
      };
    }

    const rfNodes: Node[] = [];

    // Source node (synthetic — "source" is used as from_node in edges)
    if (compositionState.source) {
      rfNodes.push(
        makeRfNode(
          "source",
          "source",
          compositionState.source.plugin,
          BADGE_BACKGROUNDS.source,
          BADGE_COLORS.source,
          nodeValidationMap["source"],
          nodeMessageMap["source"],
          selectedNodeId === "source",
        ),
      );
    }

    // Pipeline nodes (transforms, gates, aggregations, coalesces)
    for (const node of compositionState.nodes) {
      rfNodes.push(
        makeRfNode(
          node.id,
          node.node_type,
          node.plugin,
          BADGE_BACKGROUNDS[node.node_type],
          BADGE_COLORS[node.node_type],
          nodeValidationMap[node.id],
          nodeMessageMap[node.id],
          selectedNodeId === node.id,
        ),
      );
    }

    // Output/sink nodes (synthetic — output names are used as to_node in edges)
    for (const output of compositionState.outputs) {
      rfNodes.push(
        makeRfNode(
          output.name,
          "sink",
          output.plugin,
          BADGE_BACKGROUNDS.sink,
          BADGE_COLORS.sink,
          nodeValidationMap[output.name],
          nodeMessageMap[output.name],
          selectedNodeId === output.name,
        ),
      );
    }

    // Build edges: start with explicit edges from compositionState
    const rfEdges: Edge[] = compositionState.edges.map((edge, i) => ({
      id: `e-${edge.from_node}-${edge.to_node}-${i}`,
      source: edge.from_node,
      target: edge.to_node,
      label: EDGE_LABEL_MAP[edge.edge_type] ?? edge.edge_type,
      animated: edge.edge_type === "on_error",
      style: {
        stroke: edge.edge_type === "on_error" ? EDGE_COLORS.error : EDGE_COLORS.normal,
        strokeWidth: 1.5,
      },
      labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
    }));

    // Build a set of existing edge connections to avoid duplicates
    const existingConnections = new Set(
      rfEdges.map(e => `${e.source}->${e.target}`)
    );
    const nodeIds = new Set(rfNodes.map(n => n.id));

    // Always infer missing edges from connection properties.
    // ELSPETH uses a NAMED CONNECTION POINT model:
    // - source.on_success is a connection point name (e.g., "transform_in")
    // - node.input is the connection point name it reads from
    // - node.on_success/on_error/routes are connection points or sink names
    //
    // Edge inference strategy:
    // 1. Build a producer registry: connection_point → { nodeId, edgeType, label }
    // 2. For each node, look up node.input in the registry to find upstream producer
    // 3. For direct sink references (on_success/on_error/routes pointing to sink names),
    //    create edges directly since sinks are in nodeIds

    // Producer registry: connection_point_name → { nodeId, edgeType, label }
    // This tracks all producers (source and nodes) and their output connection types
    type ProducerInfo = { nodeId: string; edgeType: "success" | "error"; label: string };
    const connectionProducers: Record<string, ProducerInfo> = {};

    // Source produces on its on_success connection
    if (compositionState.source?.on_success) {
      connectionProducers[compositionState.source.on_success] = {
        nodeId: "source",
        edgeType: "success",
        label: "success",
      };
    }

    // Each node can produce on on_success, on_error, or routes
    for (const node of compositionState.nodes) {
      if (node.on_success) {
        connectionProducers[node.on_success] = {
          nodeId: node.id,
          edgeType: "success",
          label: "success",
        };
      }
      if (node.on_error) {
        connectionProducers[node.on_error] = {
          nodeId: node.id,
          edgeType: "error",
          label: "error",
        };
      }
      if (node.routes) {
        for (const [routeLabel, targetConn] of Object.entries(node.routes)) {
          connectionProducers[targetConn] = {
            nodeId: node.id,
            edgeType: "success",
            label: routeLabel,
          };
        }
      }
    }

    // Infer edges by matching node.input to upstream connection producers
    for (const node of compositionState.nodes) {
      if (node.input) {
        const producer = connectionProducers[node.input];
        if (producer && !existingConnections.has(`${producer.nodeId}->${node.id}`)) {
          const isError = producer.edgeType === "error";
          rfEdges.push({
            id: `inferred-conn-${producer.nodeId}-${node.id}`,
            source: producer.nodeId,
            target: node.id,
            label: producer.label,
            animated: isError,
            style: {
              stroke: isError ? EDGE_COLORS.error : EDGE_COLORS.normal,
              strokeWidth: 1.5,
            },
            labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
          });
          existingConnections.add(`${producer.nodeId}->${node.id}`);
        }
      }
    }

    // Handle source → sink direct edge (no transforms in between)
    if (
      compositionState.source?.on_success &&
      nodeIds.has(compositionState.source.on_success) &&
      !existingConnections.has(`source->${compositionState.source.on_success}`)
    ) {
      rfEdges.push({
        id: `inferred-sink-source-${compositionState.source.on_success}`,
        source: "source",
        target: compositionState.source.on_success,
        label: "success",
        style: { stroke: EDGE_COLORS.normal, strokeWidth: 1.5 },
        labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
      });
      existingConnections.add(`source->${compositionState.source.on_success}`);
    }

    // Handle direct sink references (on_success/on_error/routes pointing to sink names)
    // Sinks are in nodeIds, so we can create edges directly to them
    for (const node of compositionState.nodes) {
      // on_success → sink (only if target is a sink, not a connection point)
      if (
        node.on_success &&
        nodeIds.has(node.on_success) &&
        !existingConnections.has(`${node.id}->${node.on_success}`)
      ) {
        rfEdges.push({
          id: `inferred-sink-${node.id}-${node.on_success}`,
          source: node.id,
          target: node.on_success,
          label: "success",
          style: { stroke: EDGE_COLORS.normal, strokeWidth: 1.5 },
          labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
        });
        existingConnections.add(`${node.id}->${node.on_success}`);
      }

      // on_error → sink
      if (
        node.on_error &&
        nodeIds.has(node.on_error) &&
        !existingConnections.has(`${node.id}->${node.on_error}`)
      ) {
        rfEdges.push({
          id: `inferred-sink-${node.id}-${node.on_error}-error`,
          source: node.id,
          target: node.on_error,
          label: "error",
          animated: true,
          style: { stroke: EDGE_COLORS.error, strokeWidth: 1.5 },
          labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
        });
        existingConnections.add(`${node.id}->${node.on_error}`);
      }

      // Gate routes → sink
      if (node.routes) {
        for (const [routeLabel, targetId] of Object.entries(node.routes)) {
          if (nodeIds.has(targetId) && !existingConnections.has(`${node.id}->${targetId}`)) {
            rfEdges.push({
              id: `inferred-sink-${node.id}-${targetId}-${routeLabel}`,
              source: node.id,
              target: targetId,
              label: routeLabel,
              style: { stroke: EDGE_COLORS.normal, strokeWidth: 1.5 },
              labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR },
            });
            existingConnections.add(`${node.id}->${targetId}`);
          }
        }
      }
    }

    return layoutGraph(rfNodes, rfEdges);
  }, [compositionState, nodeValidationMap, nodeMessageMap, selectedNodeId]);

  // Empty state — must match the hasContent check above so that a
  // source-to-sink pipeline (zero transform nodes) still renders.
  if (nodes.length === 0) {
    return (
      <div
        className="empty-state"
      >
        No pipeline to visualise. Start a conversation to build one.
      </div>
    );
  }

  const nodeCount = nodes.length;
  const ariaLabel = `Pipeline graph with ${nodeCount} component${nodeCount !== 1 ? "s" : ""} (source, transforms, sinks). Use the Spec tab for keyboard-accessible detail.`;

  return (
    <div
      style={{ width: "100%", height: "100%" }}
      aria-label={ariaLabel}
      aria-roledescription="Pipeline DAG diagram"
      role="img"
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        colorMode={resolvedTheme}
        fitView
        fitViewOptions={{ padding: 0.15, maxZoom: 1.5, minZoom: 0.3 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} size={1} color="var(--color-border)" />
        <Controls showInteractive={false} />
        {nodes.length > 5 && (
          <MiniMap
            bgColor="var(--color-surface)"
            nodeColor={getMiniMapNodeColor}
            nodeStrokeColor={getMiniMapNodeStrokeColor}
            nodeStrokeWidth={3}
            zoomable
            pannable
            style={{ bottom: 8, right: 8, width: 120, height: 80 }}
          />
        )}
      </ReactFlow>
    </div>
  );
}
