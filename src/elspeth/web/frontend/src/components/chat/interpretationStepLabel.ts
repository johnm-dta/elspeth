// ============================================================================
// interpretationStepLabel.ts — humanise an interpretation event's
// affected_node_id into an operator-facing step label.
//
// Internal node ids (e.g. `guided_xform_1`) must not leak into user-facing
// copy.  We resolve the affected node's plugin from the CURRENT composition
// state and map it to a humanised label ("Summarise", "Fetch", "Output", …),
// falling back to a humanised plugin name for any other plugin, and to the
// raw id when the node is absent from the composition.
//
// Presentational only — reads existing store state, never mutates.
// ============================================================================

import type { CompositionState } from "@/types/index";

/**
 * Well-known plugin → step-label map.  Other plugins present in a composition
 * are humanised from the plugin name (see `humanisePlugin`).
 */
const PLUGIN_STEP_LABELS: Record<string, string> = {
  llm: "Summarise",
  web_scrape: "Fetch",
  field_mapper: "Output",
};

/** Title-case a snake/space-delimited plugin name ("field_mapper" → "Field Mapper"). */
function humanisePlugin(plugin: string): string {
  return plugin
    .split(/[_\s]+/)
    .filter((part) => part.length > 0)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/**
 * Humanised step label for a bare plugin name — the SAME mapping the
 * acknowledgement cards use ("Summarise step · prompt"), exposed for surfaces
 * that hold a plugin directly (e.g. the wire-stage topology) rather than a
 * composition node id. Keeping every surface on this one mapping means the
 * wiring list, the problems strip, and the acknowledge cards all name a step
 * identically.
 */
export function stepLabelForPlugin(plugin: string): string {
  return PLUGIN_STEP_LABELS[plugin] ?? humanisePlugin(plugin);
}

/**
 * Resolve the plugin backing an affected_node_id by searching the
 * composition's nodes, then sources, then outputs.  Returns null when the
 * id is absent (or the composition is unavailable).
 */
export function resolveNodePlugin(
  state: CompositionState | null,
  nodeId: string | null,
): string | null {
  if (state === null || nodeId === null) return null;
  const node = state.nodes.find((candidate) => candidate.id === nodeId);
  if (node && node.plugin) return node.plugin;
  const source = state.sources[nodeId];
  if (source) return source.plugin;
  const output = state.outputs.find((candidate) => candidate.name === nodeId);
  if (output) return output.plugin;
  return null;
}

/**
 * Humanised step label for an affected_node_id.  Falls back to the raw id
 * when the node is absent, and to a generic phrase when there is no id at all.
 */
export function humaniseStepLabel(
  state: CompositionState | null,
  nodeId: string | null,
): string {
  const plugin = resolveNodePlugin(state, nodeId);
  if (plugin !== null) {
    return stepLabelForPlugin(plugin);
  }
  return nodeId ?? "this step";
}

/**
 * Build a stable pipeline-step ordering index over the composition:
 * sources (object order) → nodes (array order) → outputs.  Used to order the
 * acknowledgement cards by pipeline step before created_at.
 */
export function buildStepOrder(
  state: CompositionState | null,
): Map<string, number> {
  const order = new Map<string, number>();
  if (state === null) return order;
  let index = 0;
  for (const key of Object.keys(state.sources)) order.set(key, index++);
  for (const node of state.nodes) order.set(node.id, index++);
  for (const output of state.outputs) order.set(output.name, index++);
  return order;
}
