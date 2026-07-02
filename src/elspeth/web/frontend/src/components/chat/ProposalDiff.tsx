// src/components/chat/ProposalDiff.tsx
//
// Fragment-level before/after projection for mutating composer proposals
// (elspeth-10f76f9250). Mutating tool calls used to render only a raw
// JSON.stringify of their arguments while RecoveryDiff sat one directory over
// with a full added/changed/removed diff UI. This module reuses that diff
// rendering (DiffEntryRow + the recovery-diff row styles, loaded globally via
// styles/index.css) on the standard proposal-approval surface.
//
// Honesty contract: this is a DISPLAY projection, not a client-side replay of
// server mutation semantics.
// - The "after" side of every row is literally what the proposal's arguments
//   say (identity + summary derived from the args), never a client-side
//   simulation of the committed result.
// - The "before" side is the matching fragment of the CURRENT composition
//   state. Callers must only render this for pending, non-stale proposals —
//   for stale or already-resolved proposals the current state is no longer
//   the state the proposal targets, and ToolCallCard falls back to the
//   structured argument-field rendering instead.
// - Tools whose arguments do not map onto state fragments (session tools,
//   blob tools, unknown names) return null: "no projection", not "no change".

import type { CompositionState, EdgeSpec, NodeSpec } from "@/types/api";
import type { OutputSpec, SourceSpec } from "@/types/index";
import {
  DiffEntryRow,
  edgeSummary,
  nodeSummary,
  outputSummary,
  sourceEntrySummary,
  stableStringify,
  type DiffEntry,
  type DiffSection,
} from "@/components/recovery/RecoveryDiff";

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

/** Bounded single-line rendering of an option/metadata value for row summaries. */
function valueSummary(value: unknown): string {
  const rendered = JSON.stringify(value);
  if (rendered === undefined) return "(not set)";
  return rendered.length > 60 ? `${rendered.slice(0, 57)}...` : rendered;
}

function sourceSummaryFromArgs(name: string, args: Record<string, unknown>): string | null {
  const plugin = asString(args.plugin);
  if (plugin === null) return null;
  return name === "source" ? plugin : `${name} (${plugin})`;
}

function nodeSummaryFromArgs(args: Record<string, unknown>): string | null {
  const nodeType = asString(args.node_type);
  if (nodeType === null) return null;
  return [nodeType, asString(args.plugin) ?? "no plugin"].join(" ");
}

function edgeSummaryFromArgs(args: Record<string, unknown>): string | null {
  const fromNode = asString(args.from_node);
  const toNode = asString(args.to_node);
  const edgeType = asString(args.edge_type);
  if (fromNode === null || toNode === null || edgeType === null) return null;
  return `${fromNode} -> ${toNode} (${edgeType})`;
}

function outputSummaryFromArgs(args: Record<string, unknown>): string | null {
  const name = asString(args.sink_name);
  const plugin = asString(args.plugin);
  if (name === null || plugin === null) return null;
  return `${name} (${plugin})`;
}

function upsertEntry(
  section: DiffSection,
  identity: string,
  before: unknown,
  beforeSummary: string | null,
  afterSummary: string,
  after: unknown,
): DiffEntry {
  if (before === undefined) {
    return {
      kind: "added",
      section,
      identity,
      before: undefined,
      after,
      beforeSummary: "",
      afterSummary,
    };
  }
  return {
    kind: "changed",
    section,
    identity,
    before,
    after,
    beforeSummary: beforeSummary ?? "",
    afterSummary,
  };
}

function removeEntry(
  section: DiffSection,
  identity: string,
  before: unknown,
  beforeSummary: string,
): DiffEntry {
  return {
    kind: "removed",
    section,
    identity,
    before,
    after: undefined,
    beforeSummary,
    afterSummary: "",
  };
}

/**
 * Per-key option rows for the patch_*_options tools. Patch semantics are the
 * documented tool contract (shallow merge; null deletes; missing unchanged) —
 * per-key old/new pairs are directly derivable without simulating a merge.
 */
function optionPatchEntries(
  prefix: string,
  currentOptions: Record<string, unknown>,
  patch: Record<string, unknown>,
): DiffEntry[] {
  const entries: DiffEntry[] = [];
  for (const [key, value] of Object.entries(patch)) {
    const identity = `${prefix}.${key}`;
    const hasBefore = key in currentOptions;
    const before = currentOptions[key];
    if (value === null) {
      if (hasBefore) {
        entries.push(removeEntry("option", identity, before, valueSummary(before)));
      }
      // Deleting an option that is not set is a no-op — no row.
      continue;
    }
    if (!hasBefore) {
      entries.push(upsertEntry("option", identity, undefined, null, valueSummary(value), value));
      continue;
    }
    if (stableStringify(before) !== stableStringify(value)) {
      entries.push({
        kind: "changed",
        section: "option",
        identity,
        before,
        after: value,
        beforeSummary: valueSummary(before),
        afterSummary: valueSummary(value),
      });
    }
  }
  return entries;
}

function metadataPatchEntries(
  current: CompositionState,
  patch: Record<string, unknown>,
): DiffEntry[] {
  const entries: DiffEntry[] = [];
  for (const key of ["name", "description"] as const) {
    if (!(key in patch)) continue;
    const before = current.metadata[key];
    const after = patch[key];
    if (stableStringify(before ?? null) === stableStringify(after ?? null)) continue;
    entries.push({
      kind: before === null || before === undefined ? "added" : "changed",
      section: "metadata",
      identity: key,
      before,
      after,
      beforeSummary: before === null || before === undefined ? "" : valueSummary(before),
      afterSummary: valueSummary(after),
    });
  }
  return entries;
}

/**
 * set_pipeline replaces the whole pipeline; project the args' collections
 * against the current state by identity. For identities present on both
 * sides, only the keys the args actually provide are compared (an omitted
 * arg key is "unknown", not "unchanged" — it is simply not compared), so a
 * "Changed" row always reflects an explicitly proposed difference.
 */
function setPipelineEntries(
  current: CompositionState,
  args: Record<string, unknown>,
): DiffEntry[] {
  const entries: DiffEntry[] = [];

  // Sources: named map (args.sources) or the legacy single source (args.source).
  const proposedSources = new Map<string, Record<string, unknown>>();
  const namedSources = asRecord(args.sources);
  if (namedSources !== null) {
    for (const [name, spec] of Object.entries(namedSources)) {
      const record = asRecord(spec);
      if (record !== null) proposedSources.set(name, record);
    }
  }
  const legacySource = asRecord(args.source);
  if (legacySource !== null && !proposedSources.has("source")) {
    proposedSources.set("source", legacySource);
  }

  const currentSources = current.sources ?? {};
  const sourceNames = Array.from(
    new Set([...Object.keys(currentSources), ...proposedSources.keys()]),
  ).sort((left, right) => left.localeCompare(right));
  for (const name of sourceNames) {
    const before: SourceSpec | undefined = currentSources[name];
    const after = proposedSources.get(name);
    if (after === undefined) {
      if (before !== undefined) {
        entries.push(removeEntry("source", name, before, sourceEntrySummary([name, before])));
      }
      continue;
    }
    const afterSummary = sourceSummaryFromArgs(name, after) ?? name;
    if (before === undefined) {
      entries.push(upsertEntry("source", name, undefined, null, afterSummary, after));
    } else if (providedKeysDiffer(before as unknown as Record<string, unknown>, after)) {
      entries.push(upsertEntry("source", name, before, sourceEntrySummary([name, before]), afterSummary, after));
    }
  }

  entries.push(
    ...replaceCollectionEntries<NodeSpec>(
      "node",
      current.nodes,
      (node) => node.id,
      nodeSummary,
      args.nodes,
      (item) => asString(item.id),
      (item) => nodeSummaryFromArgs(item) ?? "node",
      new Map([["id", "id"]]),
    ),
    ...replaceCollectionEntries<EdgeSpec>(
      "edge",
      current.edges,
      (edge) => edge.id,
      edgeSummary,
      args.edges,
      (item) => asString(item.id),
      (item) => edgeSummaryFromArgs(item) ?? "edge",
      new Map([["id", "id"]]),
    ),
    ...replaceCollectionEntries<OutputSpec>(
      "output",
      current.outputs,
      (output) => output.name,
      outputSummary,
      args.outputs,
      (item) => asString(item.sink_name),
      (item) => outputSummaryFromArgs(item) ?? "output",
      // set_pipeline output args key their identity as sink_name; the state
      // spec calls the same field name.
      new Map([["sink_name", "name"]]),
    ),
  );
  return entries;
}

/**
 * Compare only the keys the proposal actually provides against the current
 * fragment. Keys the fragment does not carry at all (e.g. blob_id /
 * inline_blob on source args) are skipped — we cannot honestly call them a
 * change to state the state model does not hold.
 */
function providedKeysDiffer(
  before: Record<string, unknown>,
  provided: Record<string, unknown>,
  keyAliases: Map<string, string> = new Map(),
): boolean {
  for (const [key, value] of Object.entries(provided)) {
    const beforeKey = keyAliases.get(key) ?? key;
    if (!(beforeKey in before)) continue;
    if (stableStringify(before[beforeKey]) !== stableStringify(value)) {
      return true;
    }
  }
  return false;
}

function replaceCollectionEntries<T>(
  section: DiffSection,
  currentItems: T[],
  identityOf: (item: T) => string,
  summarize: (item: T) => string,
  proposedRaw: unknown,
  proposedIdentityOf: (item: Record<string, unknown>) => string | null,
  proposedSummarize: (item: Record<string, unknown>) => string,
  keyAliases: Map<string, string>,
): DiffEntry[] {
  const entries: DiffEntry[] = [];
  const proposedById = new Map<string, Record<string, unknown>>();
  if (Array.isArray(proposedRaw)) {
    for (const raw of proposedRaw) {
      const record = asRecord(raw);
      if (record === null) continue;
      const identity = proposedIdentityOf(record);
      if (identity !== null) proposedById.set(identity, record);
    }
  }
  const currentById = new Map(currentItems.map((item) => [identityOf(item), item]));
  const identities = Array.from(
    new Set([...currentById.keys(), ...proposedById.keys()]),
  ).sort((left, right) => left.localeCompare(right));

  for (const identity of identities) {
    const before = currentById.get(identity);
    const after = proposedById.get(identity);
    if (after === undefined) {
      if (before !== undefined) {
        entries.push(removeEntry(section, identity, before, summarize(before)));
      }
      continue;
    }
    if (before === undefined) {
      entries.push(upsertEntry(section, identity, undefined, null, proposedSummarize(after), after));
      continue;
    }
    if (providedKeysDiffer(before as Record<string, unknown>, after, keyAliases)) {
      entries.push(
        upsertEntry(section, identity, before, summarize(before), proposedSummarize(after), after),
      );
    }
  }
  return entries;
}

/**
 * Project a mutating proposal's arguments onto before/after diff entries
 * against the current composition state.
 *
 * Returns null when no structured projection exists — unknown tool, malformed
 * arguments, or no current state to diff against. Callers fall back to the
 * structured argument-field rendering. Returns [] when a projection exists
 * but finds nothing to report (e.g. a patch whose keys are all no-ops).
 */
export function buildProposalDiff(
  toolName: string,
  args: Record<string, unknown>,
  currentState: CompositionState | null,
): DiffEntry[] | null {
  if (currentState === null) return null;

  switch (toolName) {
    case "set_source": {
      const name = asString(args.source_name) ?? "source";
      const afterSummary = sourceSummaryFromArgs(name, args);
      if (afterSummary === null) return null;
      const before = currentState.sources?.[name];
      return [
        upsertEntry(
          "source",
          name,
          before,
          before === undefined ? null : sourceEntrySummary([name, before]),
          afterSummary,
          args,
        ),
      ];
    }
    case "clear_source": {
      const name = asString(args.source_name) ?? "source";
      const before = currentState.sources?.[name];
      if (before === undefined) return [];
      return [removeEntry("source", name, before, sourceEntrySummary([name, before]))];
    }
    case "upsert_node": {
      const id = asString(args.id);
      const afterSummary = nodeSummaryFromArgs(args);
      if (id === null || afterSummary === null) return null;
      const before = currentState.nodes.find((node) => node.id === id);
      return [
        upsertEntry(
          "node",
          id,
          before,
          before === undefined ? null : nodeSummary(before),
          afterSummary,
          args,
        ),
      ];
    }
    case "remove_node": {
      const id = asString(args.id);
      if (id === null) return null;
      const before = currentState.nodes.find((node) => node.id === id);
      if (before === undefined) return [];
      return [removeEntry("node", id, before, nodeSummary(before))];
    }
    case "upsert_edge": {
      const id = asString(args.id);
      const afterSummary = edgeSummaryFromArgs(args);
      if (id === null || afterSummary === null) return null;
      const before = currentState.edges.find((edge) => edge.id === id);
      return [
        upsertEntry(
          "edge",
          id,
          before,
          before === undefined ? null : edgeSummary(before),
          afterSummary,
          args,
        ),
      ];
    }
    case "remove_edge": {
      const id = asString(args.id);
      if (id === null) return null;
      const before = currentState.edges.find((edge) => edge.id === id);
      if (before === undefined) return [];
      return [removeEntry("edge", id, before, edgeSummary(before))];
    }
    case "set_output": {
      const name = asString(args.sink_name);
      const afterSummary = outputSummaryFromArgs(args);
      if (name === null || afterSummary === null) return null;
      const before = currentState.outputs.find((output) => output.name === name);
      return [
        upsertEntry(
          "output",
          name,
          before,
          before === undefined ? null : outputSummary(before),
          afterSummary,
          args,
        ),
      ];
    }
    case "remove_output": {
      const name = asString(args.sink_name);
      if (name === null) return null;
      const before = currentState.outputs.find((output) => output.name === name);
      if (before === undefined) return [];
      return [removeEntry("output", name, before, outputSummary(before))];
    }
    case "set_metadata": {
      const patch = asRecord(args.patch);
      if (patch === null) return null;
      return metadataPatchEntries(currentState, patch);
    }
    case "patch_source_options": {
      const name = asString(args.source_name) ?? "source";
      const patch = asRecord(args.patch);
      const fragment = currentState.sources?.[name];
      if (patch === null || fragment === undefined) return null;
      return optionPatchEntries(name, fragment.options, patch);
    }
    case "patch_node_options": {
      const nodeId = asString(args.node_id);
      const patch = asRecord(args.patch);
      if (nodeId === null || patch === null) return null;
      const fragment = currentState.nodes.find((node) => node.id === nodeId);
      if (fragment === undefined) return null;
      return optionPatchEntries(nodeId, fragment.options, patch);
    }
    case "patch_output_options": {
      const name = asString(args.sink_name);
      const patch = asRecord(args.patch);
      if (name === null || patch === null) return null;
      const fragment = currentState.outputs.find((output) => output.name === name);
      if (fragment === undefined) return null;
      return optionPatchEntries(name, fragment.options, patch);
    }
    case "set_pipeline": {
      return setPipelineEntries(currentState, args);
    }
    default:
      return null;
  }
}

interface ProposalChangesProps {
  entries: DiffEntry[];
}

/**
 * Renders projected proposal diff entries with the shared recovery-diff row
 * styling. The caller (ToolCallCard) owns the derivability/staleness gate and
 * passes only entries it already computed.
 */
export function ProposalChanges({ entries }: ProposalChangesProps) {
  return (
    <div className="proposal-diff" data-testid="proposal-diff">
      <div className="proposal-diff-heading">Proposed changes</div>
      {entries.length === 0 ? (
        <p className="proposal-diff-empty">
          No difference from the current pipeline.
        </p>
      ) : (
        <ul className="recovery-diff-list proposal-diff-list">
          {entries.map((entry) => (
            <DiffEntryRow
              entry={entry}
              key={`${entry.kind}:${entry.section}:${entry.identity}`}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

interface ArgumentFieldsProps {
  args: Record<string, unknown>;
}

/**
 * Structured field-level rendering of a proposal's (redacted) arguments —
 * the fallback surface when no before/after projection is derivable (stale
 * or resolved proposals, unknown tools, missing state). One row per
 * top-level argument; nested objects render as bounded, formatted JSON.
 */
export function ArgumentFields({ args }: ArgumentFieldsProps) {
  const fields = Object.entries(args);
  if (fields.length === 0) {
    return (
      <p className="tool-call-arg-empty" data-testid="proposal-arg-fields">
        This tool call takes no arguments.
      </p>
    );
  }
  return (
    <dl className="tool-call-arg-fields" data-testid="proposal-arg-fields">
      {fields.map(([key, value]) => (
        <div className="tool-call-arg-field" key={key}>
          <dt>
            <code>{key}</code>
          </dt>
          <dd>
            {typeof value === "object" && value !== null ? (
              <pre className="tool-call-arg-nested">
                {JSON.stringify(value, null, 2)}
              </pre>
            ) : (
              <code>{valueSummary(value)}</code>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}
