import type { CompositionState } from "@/types/index";

type CompositionContentShape = Pick<
  CompositionState,
  "source" | "nodes" | "outputs"
>;

export function hasCompositionContent(
  state: CompositionContentShape | null | undefined,
): boolean {
  return (
    state !== null &&
    state !== undefined &&
    (state.source !== null || state.nodes.length > 0 || state.outputs.length > 0)
  );
}
