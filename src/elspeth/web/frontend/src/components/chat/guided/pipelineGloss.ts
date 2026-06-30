// ============================================================================
// pipelineGloss.ts — plain-language derivations for the guided verification panel
//
// Two pure helpers, used by the in-column verification surface (Slice C):
//
//   * pipelineGloss(state)       — one human sentence describing the whole
//                                  pipeline ("This pipeline will read your
//                                  data, rate each row, and write a CSV.").
//   * buildPlainPhraseMap(state) — Map<component_id, plain phrase>, keyed by the
//                                  SAME component_id scheme GraphView uses
//                                  (source → sourceComponentId(name), node →
//                                  node.id, output → output.name). Reused by
//                                  PipelineValidationSummary so a finding's raw
//                                  component_id renders as "rate each row"
//                                  rather than "rater".
//
// These deliberately live in the guided wrapper, NOT in the shared GraphView —
// humanising labels there would bleed into the live composer (see Slice C
// plan §C2). Per-node plain labels in the graph itself are a separate
// follow-up.
// ============================================================================

import type {
  CompositionState,
  NodeSpec,
  OutputSpec,
  SourceSpec,
} from "@/types/index";
import {
  hasCompositionContent,
  sortedSourceEntries,
  sourceComponentId,
} from "@/utils/compositionState";

/** Safe gloss when the composition has no content yet. */
export const GLOSS_FALLBACK = "Your pipeline is taking shape…";

/**
 * Generic phrase used when a validation finding's component_id has no match in
 * the current composition (e.g. a finding for a component that was just
 * removed). Never crash on an unmappable id — fall back to this.
 */
export const UNKNOWN_COMPONENT_PHRASE = "this step";

function sourcePhrase(source: SourceSpec): string {
  const plugin = (source.plugin ?? "").toLowerCase();
  if (plugin.includes("csv")) return "read your CSV";
  if (/api|http|url|web|scrape/.test(plugin)) return "read from an API";
  return "read your data";
}

function transformPhrase(node: NodeSpec): string {
  // Structural node types read off the shape, not the plugin.
  if (node.node_type === "gate") return "filter the rows";
  if (node.node_type === "aggregation") return "summarise the rows";
  if (node.node_type === "coalesce") return "merge the branches";
  const plugin = (node.plugin ?? "").toLowerCase();
  if (/llm|rate|score|classif|grade/.test(plugin)) return "rate each row";
  if (/scrape|fetch|http|web/.test(plugin)) return "scrape each page";
  if (/map|reshape|field/.test(plugin)) return "reshape each row";
  if (/select|column|project/.test(plugin)) return "pick the columns you need";
  return "process each row";
}

function outputPhrase(output: OutputSpec): string {
  const plugin = (output.plugin ?? "").toLowerCase();
  if (plugin.includes("csv")) return "write a CSV";
  if (plugin.includes("json")) return "write a JSON file";
  return "write the results";
}

/** Join with an Oxford comma: [a] → "a"; [a,b] → "a and b"; [a,b,c] → "a, b, and c". */
function oxfordJoin(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

/**
 * Derive a one-sentence, plain-language description of what the pipeline does,
 * in source → transforms → sinks order. Returns GLOSS_FALLBACK for an empty,
 * null, or undefined composition.
 */
export function pipelineGloss(
  state: CompositionState | null | undefined,
): string {
  if (!hasCompositionContent(state)) return GLOSS_FALLBACK;
  const phrases: string[] = [];
  for (const [, source] of sortedSourceEntries(state)) {
    phrases.push(sourcePhrase(source));
  }
  for (const node of state.nodes) {
    phrases.push(transformPhrase(node));
  }
  for (const output of state.outputs) {
    phrases.push(outputPhrase(output));
  }
  if (phrases.length === 0) return GLOSS_FALLBACK;
  return `This pipeline will ${oxfordJoin(phrases)}.`;
}

/**
 * Build a map from component_id → plain phrase, keyed by the SAME scheme
 * GraphView uses for its nodes so validation findings (which carry only
 * component_id) can be rendered as plain node names. Returns an empty map for a
 * null/undefined composition.
 */
export function buildPlainPhraseMap(
  state: CompositionState | null | undefined,
): Map<string, string> {
  const map = new Map<string, string>();
  if (state === null || state === undefined) return map;
  for (const [name, source] of sortedSourceEntries(state)) {
    map.set(sourceComponentId(name), sourcePhrase(source));
  }
  for (const node of state.nodes) {
    map.set(node.id, transformPhrase(node));
  }
  for (const output of state.outputs) {
    map.set(output.name, outputPhrase(output));
  }
  return map;
}
