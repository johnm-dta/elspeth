import type { CompositionState, SourceSpec } from "@/types/index";

type CompositionContentShape = Pick<
  CompositionState,
  "sources" | "nodes" | "outputs"
>;

export function sortedSourceEntries(
  state: Pick<CompositionState, "sources"> | null | undefined,
): Array<[string, SourceSpec]> {
  if (state === null || state === undefined) return [];
  return Object.entries(state.sources).sort(([left], [right]) =>
    left.localeCompare(right),
  );
}

export function hasSources(
  state: Pick<CompositionState, "sources"> | null | undefined,
): boolean {
  return sortedSourceEntries(state).length > 0;
}

export function sourceComponentId(sourceName: string): string {
  return sourceName === "source" ? "source" : `source:${sourceName}`;
}

export function hasCompositionContent<T extends CompositionContentShape>(
  state: T | null | undefined,
): state is T {
  return (
    state !== null &&
    state !== undefined &&
    (hasSources(state) || state.nodes.length > 0 || state.outputs.length > 0)
  );
}
