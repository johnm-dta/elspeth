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

export async function fetchComposition(ctx: APIRequestContext, sid: string): Promise<{ composer_meta: Record<string, unknown> | null; raw: unknown }> {
  // Real route is GET /api/sessions/{sid}/state (CompositionStateResponse, top-level
  // composer_meta), NOT the plan's /composition. Confirmed in
  // src/elspeth/web/sessions/routes/composer.py:1101 (get_current_state). The route
  // returns CompositionStateResponse | None, so body itself may be null.
  const r = await ctx.get(`/api/sessions/${sid}/state`);
  if (!r.ok()) throw new Error(`composition failed ${r.status()}`);
  const body = await r.json();
  return { composer_meta: (body?.composer_meta as Record<string, unknown> | null) ?? null, raw: body };
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

export async function fetchDiagnostics(ctx: APIRequestContext, rid: string): Promise<{ operations: Array<{ node_id: string; operation_type: string; status: string; error_message: string | null }> }> {
  const r = await ctx.get(`/api/runs/${rid}/diagnostics`);
  if (!r.ok()) throw new Error(`diagnostics failed ${r.status()}`);
  const body = await r.json();
  return { operations: body.operations ?? [] };
}

// Reachability: count distinct scrape operations that completed without error.
export function reachableSourceCount(ops: Array<{ operation_type: string; status: string; error_message: string | null }>): number {
  return ops.filter((o) => /scrape|fetch|http/i.test(o.operation_type) && o.status === "completed" && !o.error_message).length;
}
