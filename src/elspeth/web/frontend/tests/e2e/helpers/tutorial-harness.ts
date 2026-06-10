import { type APIRequestContext } from "@playwright/test";
import { authedContext, tokenFromStorageState } from "./api";
import { readFileSync } from "node:fs";

const STORAGE = "tests/e2e/.auth/staging-user.json";

export async function harnessCtx(): Promise<APIRequestContext> {
  const token = tokenFromStorageState(JSON.parse(readFileSync(STORAGE, "utf-8")));
  return authedContext(token);
}

export async function resetToFirstRun(ctx: APIRequestContext): Promise<void> {
  const r = await ctx.patch("/api/composer-preferences", {
    data: { tutorial_completed_at: null, default_mode: "guided" },
  });
  if (!r.ok()) throw new Error(`reset prefs failed ${r.status()}: ${await r.text()}`);
}

export async function cleanSessions(ctx: APIRequestContext): Promise<void> {
  await ctx.post("/api/tutorial/abandon"); // best-effort; 204 or no-op
  await ctx.delete("/api/tutorial/orphans"); // best-effort
}

export interface InterpEvent { kind: string | null; user_term: string | null; composer_skill_hash: string | null; model_identifier: string | null; }
export async function fetchInterpretationEvents(ctx: APIRequestContext, sid: string): Promise<InterpEvent[]> {
  // Real route is GET /api/sessions/{sid}/interpretations (envelope {events:[...]}),
  // NOT the plan's /interpretation-events. Confirmed in
  // src/elspeth/web/sessions/routes/interpretation.py:195 (list_interpretations).
  const r = await ctx.get(`/api/sessions/${sid}/interpretations`);
  if (!r.ok()) throw new Error(`interp events failed ${r.status()}`);
  return ((await r.json()).events ?? []) as InterpEvent[];
}

// A composition node as returned in CompositionStateResponse.nodes (top-level
// list; each carries id/node_type/plugin). Confirmed shape:
// src/elspeth/web/sessions/schemas.py:224 (nodes: CompositionObjectList) and the
// mocked tutorial.spec.ts compositionState ({id, node_type, plugin, ...}).
export interface CompositionNode { id: string; node_type?: string | null; plugin?: string | null; }
export async function fetchComposition(ctx: APIRequestContext, sid: string): Promise<{
  composer_meta: Record<string, unknown> | null;
  nodes: CompositionNode[];
  sourceInputKeys: string[];
  raw: unknown;
}> {
  // Real route is GET /api/sessions/{sid}/state (CompositionStateResponse, top-level
  // composer_meta + nodes + source), NOT the plan's /composition. Confirmed in
  // src/elspeth/web/sessions/routes/composer.py:1101 (get_current_state). The route
  // returns CompositionStateResponse | None, so body itself may be null.
  const r = await ctx.get(`/api/sessions/${sid}/state`);
  if (!r.ok()) throw new Error(`composition failed ${r.status()}`);
  const body = await r.json();
  const nodes = (Array.isArray(body?.nodes) ? body.nodes : []) as CompositionNode[];
  // Input columns = the keys of the source's first seeded row. These are the
  // user/source-supplied fields (e.g. {url}); the LLM extraction writes NEW keys.
  // Used by the dim-(d) substance check to target the extracted attribute rather
  // than scanning input columns. Grounded in compositionState.source.options.rows.
  const srcRows = (body?.source as { options?: { rows?: unknown } } | null)?.options?.rows;
  const firstRow = Array.isArray(srcRows) && srcRows.length > 0 ? srcRows[0] : null;
  const sourceInputKeys =
    firstRow && typeof firstRow === "object" ? Object.keys(firstRow as Record<string, unknown>) : [];
  return {
    composer_meta: (body?.composer_meta as Record<string, unknown> | null) ?? null,
    nodes,
    sourceInputKeys,
    raw: body,
  };
}

// The id of the web-scrape transform node in the composed DAG. The scrape is a
// TRANSFORM-node call, not an operation (operations are only source_load /
// sink_write / runtime_preflight — core/landscape/schema.py:356). We match on the
// plugin name; fall back to an id/plugin substring for older composer output.
export function scrapeNodeId(nodes: CompositionNode[]): string | null {
  const byPlugin = nodes.find((n) => (n.plugin ?? "").toLowerCase() === "web_scrape");
  if (byPlugin) return byPlugin.id;
  const bySubstr = nodes.find(
    (n) => /scrape|fetch/i.test(n.plugin ?? "") || /scrape|fetch/i.test(n.id ?? ""),
  );
  return bySubstr?.id ?? null;
}

export async function startRealRun(ctx: APIRequestContext, sid: string): Promise<string> {
  const r = await ctx.post(`/api/sessions/${sid}/execute`);
  if (!r.ok()) throw new Error(`execute failed ${r.status()}: ${await r.text()}`);
  return (await r.json()).run_id as string;
}

const TERMINAL = new Set(["completed", "completed_with_failures", "failed", "empty", "cancelled"]);
export async function pollRunTerminal(ctx: APIRequestContext, rid: string, timeoutMs = 240_000): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const r = await ctx.get(`/api/runs/${rid}`);
    if (!r.ok()) throw new Error(`status failed ${r.status()}`);
    const status = (await r.json()).status as string;
    if (TERMINAL.has(status)) return status;
    if (Date.now() > deadline) throw new Error(`run ${rid} did not reach terminal in ${timeoutMs}ms`);
    await new Promise((res) => setTimeout(res, 3000));
  }
}

// One per-token node-state in the diagnostics projection. Confirmed shape
// (src/elspeth/web/execution/schemas.py:573 RunDiagnosticNodeState and a real
// Landscape run): {state_id, token_id, node_id, status, error, attempt, ...}.
// status vocabulary is open|pending|completed|failed (Landscape node_states).
export interface DiagNodeState { token_id: string; node_id: string; status: string; error: unknown | null; attempt?: number; }
export interface DiagToken { token_id: string; row_id: string; states: DiagNodeState[]; }
export interface DiagOperation { node_id: string; operation_type: string; status: string; error_message: string | null; }
export async function fetchDiagnostics(ctx: APIRequestContext, rid: string): Promise<{
  operations: DiagOperation[];
  tokens: DiagToken[];
  failureDetail: { operation_type: string; error_message: string } | null;
}> {
  const r = await ctx.get(`/api/runs/${rid}/diagnostics`);
  if (!r.ok()) throw new Error(`diagnostics failed ${r.status()}`);
  const body = await r.json();
  return {
    operations: body.operations ?? [],
    tokens: body.tokens ?? [],
    failureDetail: body.failure_detail
      ? {
          operation_type: body.failure_detail.operation_type ?? "",
          error_message: body.failure_detail.error_message ?? "",
        }
      : null,
  };
}

// Reachability (spec §6, mechanical). The web scrape is a transform-node CALL,
// NOT an operation, so it is recorded in tokens[].states[] under the scrape
// node id — NOT in the operations table (which only ever holds source_load /
// sink_write / runtime_preflight). We therefore count DISTINCT rows whose scrape
// node-state succeeded, scoped to the composed DAG's scrape node id.
//
// Negative success test: a state counts as a successful scrape unless its status
// is "failed" or it carries an error. This is robust to an unexpected-but-
// successful status string (a positive status === "completed" test would read
// any such state as a failure and recreate the always-0 defect). Dedupe by
// token (a row can have multiple scrape states across retry attempts).
export function reachableSourceCount(tokens: DiagToken[], scrapeNodeId: string | null): number {
  // Match scrape states by exact composition node id when we have it. INSURANCE:
  // the composition node `.id` (from /state) and the runtime `node_id` (in
  // node_states) are expected to be the same string (real ids are descriptive,
  // hash-suffixed, e.g. "transform_scrape_pages_<hash>" — confirmed on a live
  // run), but if the exact match finds NO scrape state at all we fall back to a
  // /scrape|fetch/i substring on node_id. Without this fallback an id-format
  // mismatch would silently return 0 for every run and recreate the always-0
  // defect that compile/--list cannot catch.
  const matchesExact = (nid: string) => scrapeNodeId !== null && nid === scrapeNodeId;
  const matchesSubstr = (nid: string) => /scrape|fetch/i.test(nid);
  const anyExact = tokens.some((t) => (t.states ?? []).some((s) => matchesExact(s.node_id)));
  const match = anyExact ? matchesExact : matchesSubstr;

  const reachableTokens = new Set<string>();
  for (const tok of tokens) {
    for (const st of tok.states ?? []) {
      if (!match(st.node_id)) continue;
      const failed = st.status === "failed" || (st.error !== null && st.error !== undefined);
      if (!failed) reachableTokens.add(tok.token_id);
    }
  }
  return reachableTokens.size;
}
