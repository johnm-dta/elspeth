import { useMemo, useState } from "react";
import type {
  CompositionState,
  EdgeSpec,
  NodeSpec,
} from "@/types/api";
import type { OutputSpec, SourceSpec } from "@/types/index";
import { sortedSourceEntries } from "@/utils/compositionState";

type DiffKind = "added" | "removed" | "changed";
type DiffSection = "source" | "node" | "edge" | "output";

interface DiffEntry {
  kind: DiffKind;
  section: DiffSection;
  identity: string;
  before: unknown;
  after: unknown;
  beforeSummary: string;
  afterSummary: string;
}

interface DiffGroup {
  kind: DiffKind;
  label: string;
  entries: DiffEntry[];
}

interface RecoveryDiffProps {
  currentState: CompositionState | null;
  recoveredState: CompositionState;
}

const LARGE_DIFF_ROW_THRESHOLD = 50;

function stableStringify(value: unknown): string {
  return JSON.stringify(sortJson(value));
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortJson);
  }
  if (typeof value !== "object" || value === null) {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => [key, sortJson(nested)]),
  );
}

function sourceEntrySummary(entry: [string, SourceSpec]): string {
  const [sourceName, source] = entry;
  return sourceName === "source" ? source.plugin : `${sourceName} (${source.plugin})`;
}

function nodeSummary(node: NodeSpec): string {
  return [node.node_type, node.plugin ?? "no plugin"].join(" ");
}

function edgeIdentity(edge: EdgeSpec): string {
  return edge.id || `${edge.from_node}->${edge.to_node}:${edge.edge_type}`;
}

function edgeSummary(edge: EdgeSpec): string {
  return `${edge.from_node} -> ${edge.to_node} (${edge.edge_type})`;
}

function outputIdentity(output: OutputSpec): string {
  return output.name;
}

function outputSummary(output: OutputSpec): string {
  return `${output.name} (${output.plugin})`;
}

function labelForEntry(entry: DiffEntry): string {
  const action =
    entry.kind === "added"
      ? "Added"
      : entry.kind === "removed"
        ? "Removed"
        : "Changed";
  return `${action} ${entry.section}`;
}

function pluralize(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function addCollectionDiff<T>(
  entries: DiffEntry[],
  section: DiffSection,
  currentItems: T[],
  recoveredItems: T[],
  identityFor: (item: T) => string,
  summarize: (item: T) => string,
): void {
  const currentById = new Map(currentItems.map((item) => [identityFor(item), item]));
  const recoveredById = new Map(
    recoveredItems.map((item) => [identityFor(item), item]),
  );
  const identities = Array.from(
    new Set([...currentById.keys(), ...recoveredById.keys()]),
  ).sort((left, right) => left.localeCompare(right));

  for (const identity of identities) {
    const before = currentById.get(identity);
    const after = recoveredById.get(identity);
    if (before === undefined && after !== undefined) {
      entries.push({
        kind: "added",
        section,
        identity,
        before,
        after,
        beforeSummary: "",
        afterSummary: summarize(after),
      });
      continue;
    }
    if (before !== undefined && after === undefined) {
      entries.push({
        kind: "removed",
        section,
        identity,
        before,
        after,
        beforeSummary: summarize(before),
        afterSummary: "",
      });
      continue;
    }
    if (
      before !== undefined &&
      after !== undefined &&
      stableStringify(before) !== stableStringify(after)
    ) {
      entries.push({
        kind: "changed",
        section,
        identity,
        before,
        after,
        beforeSummary: summarize(before),
        afterSummary: summarize(after),
      });
    }
  }
}

function buildDiff(
  currentState: CompositionState | null,
  recoveredState: CompositionState,
): DiffGroup[] {
  const entries: DiffEntry[] = [];
  addCollectionDiff(
    entries,
    "source",
    sortedSourceEntries(currentState),
    sortedSourceEntries(recoveredState),
    ([sourceName]) => sourceName,
    sourceEntrySummary,
  );
  addCollectionDiff(
    entries,
    "node",
    currentState?.nodes ?? [],
    recoveredState.nodes,
    (node) => node.id,
    nodeSummary,
  );
  addCollectionDiff(
    entries,
    "edge",
    currentState?.edges ?? [],
    recoveredState.edges,
    edgeIdentity,
    edgeSummary,
  );
  addCollectionDiff(
    entries,
    "output",
    currentState?.outputs ?? [],
    recoveredState.outputs,
    outputIdentity,
    outputSummary,
  );

  const groups: DiffGroup[] = [
    {
      kind: "added",
      label: pluralize(
        entries.filter((entry) => entry.kind === "added").length,
        "addition",
      ),
      entries: entries.filter((entry) => entry.kind === "added"),
    },
    {
      kind: "changed",
      label: pluralize(
        entries.filter((entry) => entry.kind === "changed").length,
        "change",
      ),
      entries: entries.filter((entry) => entry.kind === "changed"),
    },
    {
      kind: "removed",
      label: pluralize(
        entries.filter((entry) => entry.kind === "removed").length,
        "removal",
      ),
      entries: entries.filter((entry) => entry.kind === "removed"),
    },
  ];
  return groups.filter((group) => group.entries.length > 0);
}

function DiffEntryRow({ entry }: { entry: DiffEntry }) {
  return (
    <li className={`recovery-diff-row recovery-diff-row--${entry.kind}`}>
      <div className="recovery-diff-row-title">
        <span>{labelForEntry(entry)}</span>
        <code>{entry.identity}</code>
      </div>
      {entry.kind === "changed" ? (
        <div className="recovery-diff-row-change">
          <span>{entry.beforeSummary}</span>
          <span aria-hidden="true">{" -> "}</span>
          <span>{entry.afterSummary}</span>
        </div>
      ) : (
        <div className="recovery-diff-row-change">
          {entry.kind === "added" ? entry.afterSummary : entry.beforeSummary}
        </div>
      )}
    </li>
  );
}

export function RecoveryDiff({
  currentState,
  recoveredState,
}: RecoveryDiffProps) {
  const [expandedGroups, setExpandedGroups] = useState<Set<DiffKind>>(
    () => new Set(),
  );
  const groups = useMemo(
    () => buildDiff(currentState, recoveredState),
    [currentState, recoveredState],
  );

  if (groups.length === 0) {
    return (
      <section className="recovery-diff" aria-labelledby="recovery-diff-title">
        <h3 id="recovery-diff-title">Pipeline changes</h3>
        <p>No pipeline changes to apply.</p>
      </section>
    );
  }

  return (
    <section className="recovery-diff" aria-labelledby="recovery-diff-title">
      <h3 id="recovery-diff-title">Pipeline changes</h3>
      {/* role="group" so the aria-label is exposed — aria-label on a
          role-less div is ignored by AT (WCAG 1.3.1, elspeth-37293a3b7c). */}
      <div
        className="recovery-diff-summary"
        role="group"
        aria-label="Recovery diff summary"
      >
        {groups.map((group) => (
          <span key={group.kind}>{group.label}</span>
        ))}
      </div>
      {groups.map((group) => {
        const isLarge = group.entries.length > LARGE_DIFF_ROW_THRESHOLD;
        const isExpanded = expandedGroups.has(group.kind);
        const showRows = !isLarge || isExpanded;
        return (
          <div
            className={`recovery-diff-group recovery-diff-group--${group.kind}`}
            key={group.kind}
          >
            <div className="recovery-diff-group-header">
              <h4>{group.label}</h4>
              {isLarge ? (
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() =>
                    setExpandedGroups((previous) => {
                      const next = new Set(previous);
                      if (next.has(group.kind)) {
                        next.delete(group.kind);
                      } else {
                        next.add(group.kind);
                      }
                      return next;
                    })
                  }
                >
                  {isExpanded ? `Hide ${group.label}` : `Show ${group.label}`}
                </button>
              ) : null}
            </div>
            {showRows ? (
              <ul className="recovery-diff-list">
                {group.entries.map((entry) => (
                  <DiffEntryRow
                    entry={entry}
                    key={`${entry.kind}:${entry.section}:${entry.identity}`}
                  />
                ))}
              </ul>
            ) : (
              <p className="recovery-diff-compact">
                Details are collapsed to keep this recovery review responsive.
              </p>
            )}
          </div>
        );
      })}
    </section>
  );
}
