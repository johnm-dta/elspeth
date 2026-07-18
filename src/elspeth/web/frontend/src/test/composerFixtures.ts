import type { CompositionState } from "../types/api";

export const compositionStateAuthorityFields = {
  session_id: "session-1",
  is_valid: true,
  validation_errors: null,
  validation_warnings: null,
  validation_suggestions: null,
  derived_from_state_id: null,
  created_at: "2026-07-19T00:00:00Z",
  composer_meta: null,
  plugin_policy_findings: [],
} satisfies Pick<
  CompositionState,
  | "session_id"
  | "is_valid"
  | "validation_errors"
  | "validation_warnings"
  | "validation_suggestions"
  | "derived_from_state_id"
  | "created_at"
  | "composer_meta"
  | "plugin_policy_findings"
>;

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
    ...compositionStateAuthorityFields,
    version,
    sources: { source: { plugin: "csv_file", options: { path: "x.csv" } } },
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

/**
 * Returns a Promise that resolves to `value` after `delay` ms — UNLESS the
 * provided AbortSignal aborts first, in which case it rejects with a
 * synthetic AbortError matching the shape the production store's catch arm
 * checks (`err.name === "AbortError"`).
 *
 * Use this in tests that exercise the store's abort-stale-in-flight or
 * clearSession-aborts-controllers contracts; do NOT hand-roll a Promise
 * that ignores the signal — that forces components to paper over the gap
 * with synchronous setState (see elspeth-f018ea84c6).
 */
export function makeAbortablePromise<T>(
  value: T,
  options?: { delay?: number; signal?: AbortSignal },
): Promise<T> {
  const { delay = 0, signal } = options ?? {};
  return new Promise<T>((resolve, reject) => {
    const reject_with_abort = () => {
      const err = new Error("Aborted");
      err.name = "AbortError";
      reject(err);
    };
    if (signal?.aborted) {
      reject_with_abort();
      return;
    }
    const timer = setTimeout(() => resolve(value), delay);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject_with_abort();
    });
  });
}
