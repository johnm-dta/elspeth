import type { CompositionState } from "../types/api";

/**
 * Canonical test fixture for CompositionState.
 *
 * NodeSpec arity (frontend `types/index.ts`):
 *   Required (7): id, node_type, plugin, input, on_success, on_error, options
 *   Optional (6): condition, routes, fork_to, branches, policy, merge
 * This is the frontend contract the fixture mirrors. The Python backend has 13
 * fields but the TypeScript interface marks 6 of them as optional — no `as never`
 * cast is needed once all required fields are supplied.
 *
 * Import from here in all test files that need CompositionState scaffolding.
 * Do NOT duplicate this fixture in individual test files.
 */
export function makeComposition(
  version: number,
  overrides?: Partial<CompositionState>,
): CompositionState {
  return {
    id: "comp-1",
    version,
    source: { plugin: "csv_file", options: { path: "x.csv" } },
    nodes: [
      {
        id: "select_columns",
        node_type: "transform",
        plugin: "select_columns",
        input: "source",
        on_success: null,
        on_error: null,
        options: {},
      },
    ],
    edges: [],
    outputs: [],
    metadata: { name: "demo", description: "" },
    ...overrides,
  };
}
