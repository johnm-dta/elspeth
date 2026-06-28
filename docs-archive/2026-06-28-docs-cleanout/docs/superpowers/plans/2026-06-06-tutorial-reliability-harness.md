# Tutorial Reliability Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repeatable, version-tagged battery that runs the first-run composer tutorial end-to-end against staging (reset between each run), grades every run on four dimensions, and reports how many worked + classified failure states.

**Architecture:** A non-mocked Playwright spec drives the real React tutorial against `elspeth.foundryside.dev` and writes one JSON record per run. Helpers (extending `tests/e2e/helpers/api.ts`) hit the live HTTP API for reset, composition snapshot, interpretation events, the real-system re-run, and run results/diagnostics. A Node aggregator computes headline rates + a failure table and appends a version-stamped row to a trend log. Dimension (d) — creative solution quality — is judged by an agent reading recorded output rows (full LLM-API automation deferred per spec §9).

**Tech Stack:** Playwright (`@playwright/test`, already installed), TypeScript/Node ESM, the live ELSPETH web HTTP API. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-06-tutorial-reliability-harness-design.md` (four-dimension pass criterion: a tutorial-completes / b real-system-re-run / c assumption-handling / d creative-solution-quality).

**TDD adaptation note:** Unit-TDD (write failing test → minimal impl) does not fit browser automation against a live remote service. This plan instead uses **skeleton-first milestones**: pure-code files (prompt/rubric/types, helpers, aggregator) are written and verified with a tiny runnable check; live-staging tasks prove one run before scaling. Every step still has an exact command and expected output, and commits are frequent.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts` | The coupled triple: the fixed non-canonical prompt, the dimension-(c) assumption rubric, the dimension-(d) judge rubric/thresholds, the harness-version constant. Pure data. |
| `src/elspeth/web/frontend/tests/e2e/harness/types.ts` | TS interfaces for the per-run record and batch summary. |
| `src/elspeth/web/frontend/tests/e2e/helpers/tutorial-harness.ts` | Live-API helpers: reset-to-first-run, clean sessions, fetch interpretation events, snapshot composition (+ normalization flag), trigger real-system re-run, poll run terminal, fetch results + diagnostics. Extends the `api.ts` pattern. |
| `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts` | The battery: N independent tests, each resets then drives turns 1→7, captures a/b/c + (d) raw material, writes one per-run JSON record. |
| `src/elspeth/web/frontend/tests/e2e/harness/aggregate.mjs` | Node script: read per-run records → headline rates + failure table → append version-stamped trend row → write markdown report. |
| `src/elspeth/web/frontend/tests/e2e/harness/README.md` | How to run the battery + the dimension-(d) judging step. |
| `notes/tutorial-reliability/` (created) | Committed batch reports + `trend.jsonl`. |
| `.gitignore` (modify) | Ignore `tests/e2e/.harness-results/` (raw per-run records + Playwright artifacts). |

Run records (raw, gitignored): `tests/e2e/.harness-results/<batch_id>/run-NN.json`.

---

## Conventions used by every live task

Invocation environment (from `playwright.staging.config.ts` header):

```bash
cd src/elspeth/web/frontend
export STAGING_BASE_URL=https://elspeth.foundryside.dev
export STAGING_USERNAME=<staging-username>
export STAGING_PASSWORD=<staging-password>
export PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev
```

Load `STAGING_USERNAME` and `STAGING_PASSWORD` from the operator's out-of-repo secret channel. Do not commit live staging credentials to this plan or to harness fixtures. The previously exposed shared staging credential must be rotated before relying on this harness for staging isolation.

The auth token is written to `tests/e2e/.auth/staging-user.json` by `tests/e2e/setup/staging-global-setup.ts` (already exists). Helpers read it with `tokenFromStorageState` from `helpers/api.ts`.

---

## Task 1: Fixed prompt + rubric + harness version (pure data)

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts`

- [ ] **Step 1: Write the file**

```typescript
// The coupled triple: prompt + assumption rubric (dim c) + judge rubric (dim d).
// If the prompt changes, BOTH rubrics must change in lockstep (spec §11).

export const HARNESS_VERSION = "1.0.0";

// Semantically equal to CANONICAL_TUTORIAL_PROMPT but a DIFFERENT string, so
// SHA-256(prompt+model_id) misses the tutorial cache (is_canonical_prompt=false
// in tutorial_service.py) → fresh live composition every run.
export const FIXED_PROMPT =
  "Build a source containing the URLs of five Australian government agency " +
  "websites of your choosing. Set the abuse contact to noreply@dta.gov.au and " +
  "the scraping reason to 'DTA technical demonstration'. For each page, fetch " +
  "the HTML and use an LLM to determine that agency's primary brand colours. " +
  "Then drop the HTML field and write the remaining results to a JSON file.";

// Dimension (c): which interpretation kinds the composer SHOULD raise vs NOT.
// Graded on kind, not exact wording. InterpretationKind values come from the
// backend enum (kind field on InterpretationEventResponse).
export const ASSUMPTION_RUBRIC = {
  // Composer invented the 5 URLs and the subjective colour criterion → verify.
  expectVerify: ["invented_source", "vague_term"] as const,
  // The composer MAY also stage a prompt-template review for the LLM node; this
  // is acceptable-but-not-required, so it is neither under- nor over-flagging.
  allowOptional: ["llm_prompt_template"] as const,
  // Explicit values the user stated → must NOT be raised for review.
  expectWaive: ["abuse_contact", "scraping_reason"] as const,
};

// Dimension (d): the judge rubric the agent applies to recorded output rows.
export const JUDGE_RUBRIC = {
  // Mechanical (computed in the spec/aggregator, NOT by the judge):
  minReachableSources: 5, // all 5 invented URLs must have fetched (scrape op status ok)
  maxDiscardedRows: 0, // discarded_row_count from TutorialRunOutput
  minSubstantiveRows: 4, // >= 4 of 5 rows carry a meaningful colour value
  // Judge-scored (0..1), pass threshold:
  judgePassThreshold: 0.7,
  // The judge question (structured), applied per batch over recorded rows:
  judgePrompt:
    "Given the task (extract each agency's primary brand colours from its " +
    "homepage) and these output rows, score 0..1: are the colour values real, " +
    "specific, and plausibly correct (not nulls, not 'cannot determine', not " +
    "fabricated when the page shows no clear palette)? Penalise degenerate or " +
    "hallucinated answers.",
};
```

- [ ] **Step 2: Verify it parses**

Run: `cd src/elspeth/web/frontend && npx tsc --noEmit tests/e2e/harness/prompt-and-rubric.ts`
Expected: no output (clean compile), exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts
git commit -m "feat(harness): fixed tutorial prompt + c/d rubrics"
```

---

## Task 2: Per-run record types

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/harness/types.ts`

- [ ] **Step 1: Write the types** (mirror spec §8 schema)

```typescript
export type Outcome = "pass" | "tutorial_fault" | "infra_fault";

export type FaultSubclass =
  | "composer-skill-prompt"
  | "specific-tool"
  | "plugin"
  | "frontend-state-machine"
  | "normalization-gap"
  | "assumption-under-flag"
  | "assumption-over-flag"
  | "degenerate-output"
  | "weak-extraction"
  | "invented-source-unreachable"
  | "wrong-dag-shape"
  | "llm-5xx-or-ratelimit"
  | "scrape-target-down-or-throttled"
  | "staging-hiccup"
  | "timeout"
  | null;

export interface RunRecord {
  batch_id: string;
  run_index: number;
  outcome: Outcome;
  fault_subclass: FaultSubclass;
  fix_target: string | null;
  turn_reached: number; // 1..7
  tutorial_run_id: string | null;
  realsystem_run_id: string | null;
  seeded_from_cache: boolean; // MUST be false for a fresh run
  dim_a_tutorial_completed: boolean;
  dim_b_realsystem_passed: boolean;
  dim_c_assumptions_ok: boolean;
  dim_d_solution_quality: {
    status: "pending_judge" | "pass" | "fail";
    judge_score: number | null;
    source_reachable: string; // e.g. "5/5"
    discarded_row_count: number;
    substantive_rows: string; // e.g. "4/5"
  };
  assumptions: {
    raised: Array<{ kind: string | null; term: string | null }>;
    under_flagged: string[];
    over_flagged: string[];
  };
  output_rows: Array<Record<string, unknown>>; // raw material for the judge
  landscape: {
    tutorial_failure: string | null;
    realsystem_failure: string | null;
    normalization_fired: boolean;
  };
  stamp: {
    composer_skill_hash: string | null;
    model_identifier: string | null;
  };
  timing_s: Record<string, number>;
  error: string | null; // exception text on hard fault
}
```

- [ ] **Step 2: Verify compile**

Run: `cd src/elspeth/web/frontend && npx tsc --noEmit tests/e2e/harness/types.ts`
Expected: exit 0, no output.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/harness/types.ts
git commit -m "feat(harness): per-run record types"
```

---

## Task 3: Live-API helpers

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/helpers/tutorial-harness.ts`
- Read first (mirror exact fields): `src/elspeth/web/sessions/schemas.py` (`CompositionStateResponse` for the `composer_meta` location), `src/elspeth/web/execution/schemas.py` (`RunStatusResponse`, `RunResultsResponse`, `RunDiagnosticsResponse`).

Pinned endpoint contracts (from recon):
- Reset: `PATCH /api/composer-preferences` body `{ "tutorial_completed_at": null, "default_mode": "guided" }`.
- Clean: `POST /api/tutorial/abandon` then `DELETE /api/tutorial/orphans`.
- Interpretation events: `GET /api/sessions/{sid}/interpretation-events` → `{ events: InterpretationEventResponse[] }` (fields incl. `kind`, `user_term`, `composer_skill_hash`, `model_identifier`).
- Composition: `GET /api/sessions/{sid}/composition` (carries `composer_meta`; the normalization flag is `composer_meta.tutorial_runtime_normalized === true`).
- Real re-run: `POST /api/sessions/{sid}/execute` → `{ run_id }`.
- Status: `GET /api/runs/{rid}` → `RunStatusResponse` (`status` ∈ pending|running|completed|completed_with_failures|failed|empty|cancelled).
- Results: `GET /api/runs/{rid}/results` → `RunResultsResponse`.
- Diagnostics: `GET /api/runs/{rid}/diagnostics` → `RunDiagnosticsResponse` (`operations[].{node_id,operation_type,status,error_message}`).

- [ ] **Step 1: Write the helpers**

```typescript
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
  const r = await ctx.get(`/api/sessions/${sid}/interpretation-events`);
  if (!r.ok()) throw new Error(`interp events failed ${r.status()}`);
  return ((await r.json()).events ?? []) as InterpEvent[];
}

export async function fetchComposition(ctx: APIRequestContext, sid: string): Promise<{ composer_meta: Record<string, unknown> | null; raw: unknown }> {
  const r = await ctx.get(`/api/sessions/${sid}/composition`);
  if (!r.ok()) throw new Error(`composition failed ${r.status()}`);
  const body = await r.json();
  return { composer_meta: body.composer_meta ?? null, raw: body };
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
```

- [ ] **Step 2: Smoke the helpers against staging** (reset + clean only; safe, idempotent)

Create a throwaway `tests/e2e/.harness-results/smoke.mjs`:

```javascript
import { harnessCtx, resetToFirstRun, cleanSessions } from "../helpers/tutorial-harness.ts";
const ctx = await harnessCtx();
await resetToFirstRun(ctx);
await cleanSessions(ctx);
console.log("smoke ok");
```

Run (after the env exports + one `npx playwright test --config=playwright.staging.config.ts --grep __nonexistent__` to force globalSetup to write the auth file, or run `node --import tsx tests/e2e/setup/staging-global-setup.ts` equivalent):
`cd src/elspeth/web/frontend && npx tsx tests/e2e/.harness-results/smoke.mjs`
Expected: `smoke ok`. (If `tsx` is unavailable, run the equivalent inside a one-off Playwright `test()` — see Task 4's pattern.)

- [ ] **Step 3: Delete the throwaway smoke + commit**

```bash
rm src/elspeth/web/frontend/tests/e2e/.harness-results/smoke.mjs
git add src/elspeth/web/frontend/tests/e2e/helpers/tutorial-harness.ts
git commit -m "feat(harness): live-API helpers (reset, snapshot, run, diagnostics)"
```

---

## Task 4: Walking skeleton — ONE full UI run (dimension a only)

This is the critical de-risk milestone (spec §3). Prove turns 1→7 drive reliably against the live backend with real (multi-minute) composition before adding b/c/d or scaling.

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`
- Modify: `.gitignore` (add `tests/e2e/.harness-results/`)

Selectors (verbatim from the working mocked `tutorial.spec.ts` + `copy.ts`):
- tutorial visible: `getByRole("main", { name: /first-run tutorial/i })`
- Turn 1: `getByRole("button", { name: "Let's go" })`
- Turn 2: `getByLabel("Pipeline description")` (fill), `getByRole("button", { name: "Build it" })`
- Turn 2b: `getByText(/Here is what the composer drafted/i)`; accept each assumption (buttons named `/Accept .../i`); continue `getByRole("button", { name: "Looks good" })` (enabled only after all accepted)
- Turn 3: `getByRole("button", { name: "Looks good, run it" })`
- Turn 4: wait for completion, then `getByRole("button", { name: "Continue" })`
- Turn 5: `getByText(/This is the audit story/i)`, `getByRole("button", { name: "Continue" })`
- Turn 6: `getByRole("radio", { name: /Guided/i })`, `getByRole("button", { name: "Save and go" })`
- Turn 7: `getByRole("heading", { name: "You're ready to use the composer." })`, `getByRole("button", { name: "Take me to the composer" })`

- [ ] **Step 1: Add the gitignore entry**

```bash
echo "tests/e2e/.harness-results/" >> src/elspeth/web/frontend/.gitignore
```

- [ ] **Step 2: Write the skeleton spec** (N=1, accepts ALL assumptions generically)

```typescript
import { test, expect, type Page } from "@playwright/test";
import { FIXED_PROMPT } from "./harness/prompt-and-rubric";
import { harnessCtx, resetToFirstRun, cleanSessions } from "./helpers/tutorial-harness";

// Generic: click every "Accept ..." button rendered in the assumptions panel,
// then wait for the continue button to enable. Returns the count accepted.
async function acceptAllAssumptions(page: Page): Promise<number> {
  const buttons = page.getByRole("button", { name: /^Accept /i });
  const count = await buttons.count();
  for (let i = 0; i < count; i++) await buttons.nth(0).click(); // list shrinks as accepted
  return count;
}

test.describe("tutorial reliability skeleton", () => {
  test.beforeEach(async () => {
    const ctx = await harnessCtx();
    await cleanSessions(ctx);
    await resetToFirstRun(ctx);
    await ctx.dispose();
  });

  test("one full tutorial run reaches graduation", async ({ page }) => {
    test.setTimeout(360_000); // real compose + run can take minutes
    await page.goto("/");
    await expect(page.getByRole("main", { name: /first-run tutorial/i })).toBeVisible();
    await page.getByRole("button", { name: "Let's go" }).click();
    await page.getByLabel("Pipeline description").fill(FIXED_PROMPT);
    await page.getByRole("button", { name: "Build it" }).click();
    await expect(page.getByText(/Here is what the composer drafted/i)).toBeVisible({ timeout: 180_000 });
    await acceptAllAssumptions(page);
    await page.getByRole("button", { name: "Looks good" }).click();
    await page.getByRole("button", { name: "Looks good, run it" }).click();
    await expect(page.getByRole("button", { name: "Continue" })).toBeVisible({ timeout: 240_000 });
    await page.getByRole("button", { name: "Continue" }).click(); // turn 4 → 5
    await expect(page.getByText(/This is the audit story/i)).toBeVisible();
    await page.getByRole("button", { name: "Continue" }).click(); // turn 5 → 6
    await page.getByRole("radio", { name: /Guided/i }).click();
    await page.getByRole("button", { name: "Save and go" }).click();
    await expect(page.getByRole("heading", { name: "You're ready to use the composer." })).toBeVisible();
  });
});
```

- [ ] **Step 3: Run the skeleton against staging**

Run:
```bash
cd src/elspeth/web/frontend
STAGING_BASE_URL=https://elspeth.foundryside.dev STAGING_USERNAME=<staging-username> STAGING_PASSWORD=<staging-password> \
PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
npx playwright test --config=playwright.staging.config.ts tutorial-reliability.staging.spec.ts
```
Expected: 1 passed. If it fails, the trace/video/screenshot under `playwright-report/` localises the broken turn/selector/timeout — fix selectors or timeouts inline until one run is green. **Do not proceed until this is reliably green twice in a row.**

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts src/elspeth/web/frontend/.gitignore
git commit -m "feat(harness): walking-skeleton tutorial spec (dim a)"
```

---

## Task 5: Capture session id + tutorial run id + cache assertion

The spec must learn the session id and tutorial run_id to drive b/c/d. Capture them from network responses.

**Files:**
- Modify: `tutorial-reliability.staging.spec.ts`

- [ ] **Step 1: Add response capture** (inside the test, before `goto`)

```typescript
let sessionId: string | null = null;
let tutorialRunId: string | null = null;
let seededFromCache = false;
page.on("response", async (resp) => {
  const url = resp.url();
  if (url.includes("/api/tutorial/run") && resp.request().method() === "POST" && resp.ok()) {
    const body = await resp.json().catch(() => null);
    if (body) { tutorialRunId = body.run_id ?? null; seededFromCache = body.seeded_from_cache ?? false; }
  }
  const m = url.match(/\/api\/sessions\/([0-9a-f-]{36})\//i);
  if (m && !sessionId) sessionId = m[1];
});
```

- [ ] **Step 2: Assert fresh composition** (after graduation visible)

```typescript
expect(seededFromCache, "fixed prompt must bypass the cache").toBe(false);
expect(sessionId, "session id captured").not.toBeNull();
expect(tutorialRunId, "tutorial run id captured").not.toBeNull();
```

- [ ] **Step 3: Run + verify**

Run: the Task 4 command. Expected: 1 passed (now also asserting fresh + ids captured).

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts
git commit -m "feat(harness): capture session/run ids + assert cache bypass"
```

---

## Task 6: Grade dimensions a/b/c + capture (d) raw material; write per-run record

**Files:**
- Modify: `tutorial-reliability.staging.spec.ts`
- Use: `helpers/tutorial-harness.ts`, `harness/prompt-and-rubric.ts`, `harness/types.ts`

Grading rules:
- **(a)** = graduation heading reached AND no thrown error → `dim_a_tutorial_completed`.
- **(b)** = `composer_meta.tutorial_runtime_normalized !== true` AND the tutorial run reached terminal `completed` with rows. If normalization fired → `dim_b=false`, `fault_subclass="normalization-gap"`, `fix_target="engine: tutorial normalization parity"`.
- **(c)** = for each kind in `ASSUMPTION_RUBRIC.expectVerify`, an event with that kind exists (else `under_flagged`); any raised kind in `expectWaive` → `over_flagged`. `dim_c_assumptions_ok` = no under_flagged AND no over_flagged. Under-flag ⇒ `assumption-under-flag` (serious); else over-flag ⇒ `assumption-over-flag`. `fix_target="composer-skill-prompt: pipeline_composer.md review rules"`.
- **(d) raw material only**: from the tutorial run results, record `output_rows`, `discarded_row_count`; from diagnostics, `reachableSourceCount`. Set `dim_d.status="pending_judge"`. Compute mechanical fails now: reachable < 5 ⇒ `invented-source-unreachable`; discarded > 0 or substantive rows < 4 ⇒ `degenerate-output`.

- [ ] **Step 1: Add a `finally` block that builds + writes the RunRecord**

```typescript
import { mkdirSync, writeFileSync } from "node:fs";
import { ASSUMPTION_RUBRIC } from "./harness/prompt-and-rubric";
import type { RunRecord, FaultSubclass } from "./harness/types";
import { harnessCtx, fetchInterpretationEvents, fetchComposition, fetchDiagnostics, reachableSourceCount } from "./helpers/tutorial-harness";

// ... inside the test, wrap the UI flow in try/finally:
const BATCH_ID = process.env.HARNESS_BATCH_ID ?? "skeleton";
const RUN_INDEX = Number(process.env.HARNESS_RUN_INDEX ?? "1");
let turnReached = 0; // increment after each turn's success
let graduated = false;
let hardError: string | null = null;
try {
  // ... existing turn flow; set turnReached = 1..7 and graduated = true at the end ...
} catch (e) {
  hardError = e instanceof Error ? e.message : String(e);
  throw e; // rethrow so Playwright captures trace/video for this failed run
} finally {
  const ctx = await harnessCtx();
  const events = sessionId ? await fetchInterpretationEvents(ctx, sessionId).catch(() => []) : [];
  const comp = sessionId ? await fetchComposition(ctx, sessionId).catch(() => ({ composer_meta: null })) : { composer_meta: null };
  const diag = tutorialRunId ? await fetchDiagnostics(ctx, tutorialRunId).catch(() => ({ operations: [] })) : { operations: [] };

  const raisedKinds = events.map((e) => e.kind);
  const underFlagged = ASSUMPTION_RUBRIC.expectVerify.filter((k) => !raisedKinds.includes(k));
  const overFlagged = (ASSUMPTION_RUBRIC.expectWaive as readonly string[]).filter((k) => raisedKinds.includes(k));
  const normalized = (comp.composer_meta as Record<string, unknown> | null)?.tutorial_runtime_normalized === true;
  const reachable = reachableSourceCount(diag.operations);

  let outcome: RunRecord["outcome"] = "pass";
  let sub: FaultSubclass = null;
  let fix: string | null = null;
  if (hardError) { outcome = "tutorial_fault"; sub = "frontend-state-machine"; fix = "frontend / timing"; }
  else if (normalized) { outcome = "tutorial_fault"; sub = "normalization-gap"; fix = "engine: tutorial normalization parity"; }
  else if (underFlagged.length) { outcome = "tutorial_fault"; sub = "assumption-under-flag"; fix = "composer-skill-prompt: pipeline_composer.md review rules"; }
  else if (overFlagged.length) { outcome = "tutorial_fault"; sub = "assumption-over-flag"; fix = "composer-skill-prompt: pipeline_composer.md review rules"; }
  else if (reachable < 5) { outcome = "tutorial_fault"; sub = "invented-source-unreachable"; fix = "composer-skill-prompt: generated-source discipline"; }

  const record: RunRecord = {
    batch_id: BATCH_ID, run_index: RUN_INDEX, outcome, fault_subclass: sub, fix_target: fix,
    turn_reached: turnReached, tutorial_run_id: tutorialRunId, realsystem_run_id: null,
    seeded_from_cache: seededFromCache,
    dim_a_tutorial_completed: graduated, dim_b_realsystem_passed: !normalized && graduated,
    dim_c_assumptions_ok: underFlagged.length === 0 && overFlagged.length === 0,
    dim_d_solution_quality: { status: "pending_judge", judge_score: null, source_reachable: `${reachable}/5`, discarded_row_count: 0, substantive_rows: "?/5" },
    assumptions: { raised: events.map((e) => ({ kind: e.kind, term: e.user_term })), under_flagged: [...underFlagged], over_flagged: [...overFlagged] },
    output_rows: [], // filled in Step 2
    landscape: { tutorial_failure: null, realsystem_failure: null, normalization_fired: normalized },
    stamp: { composer_skill_hash: events[0]?.composer_skill_hash ?? null, model_identifier: events[0]?.model_identifier ?? null },
    timing_s: {}, error: hardError,
  };
  const dir = `tests/e2e/.harness-results/${BATCH_ID}`;
  mkdirSync(dir, { recursive: true });
  writeFileSync(`${dir}/run-${String(RUN_INDEX).padStart(2, "0")}.json`, JSON.stringify(record, null, 2));
  await ctx.dispose();
}
```

- [ ] **Step 2: Fill `output_rows` + `discarded_row_count` from tutorial run results**

Capture from the `/api/tutorial/run` response body already intercepted in Task 5 (it returns `output.rows` and `output.discarded_row_count`). Store them in outer `let` vars and set them on the record; compute `substantive_rows` as the count of rows whose colour field is a non-empty string not matching `/cannot|unknown|n\/a|none/i`.

- [ ] **Step 3: Run + inspect the record**

Run: the Task 4 command, then `cat tests/e2e/.harness-results/skeleton/run-01.json`.
Expected: a well-formed record with `dim_a=true`, `dim_b`, `dim_c`, `assumptions.raised` populated, `stamp.composer_skill_hash` non-null.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts
git commit -m "feat(harness): grade dims a/b/c + capture d material, write run record"
```

---

## Task 7: Dimension (b) real-system re-run

Make (b) authoritative: actually re-run via the normal path and record its terminal status. (The normalization flag remains the trigger/annotation.)

**Files:**
- Modify: `tutorial-reliability.staging.spec.ts`

- [ ] **Step 1: After graduation, before the record is built, do the real re-run**

```typescript
import { startRealRun, pollRunTerminal, fetchDiagnostics } from "./helpers/tutorial-harness";
// ... after graduation:
let realStatus: string | null = null;
if (sessionId) {
  const ctx2 = await harnessCtx();
  try {
    const rid = await startRealRun(ctx2, sessionId);
    realsystemRunId = rid; // outer let
    realStatus = await pollRunTerminal(ctx2, rid);
  } catch (e) { realStatus = `error:${e instanceof Error ? e.message : e}`; }
  finally { await ctx2.dispose(); }
}
```

- [ ] **Step 2: Fold real-run status into (b)**

`dim_b_realsystem_passed = realStatus === "completed"`. If `realStatus !== "completed"`, set `outcome="tutorial_fault"`, `fault_subclass="normalization-gap"` (passed a, failed b), `landscape.realsystem_failure = realStatus`.

Note the §10 finding-vs-fix discipline: do NOT change the normalization code; just record the divergence.

- [ ] **Step 3: Run + verify** the record now has `realsystem_run_id` and a real `dim_b`.

Run: Task 4 command; `cat` the record. Expected: `realsystem_run_id` non-null, `dim_b_realsystem_passed` reflects the real run.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts
git commit -m "feat(harness): dimension b real-system re-run"
```

---

## Task 8: Scale to N independent runs with reset between

**Files:**
- Modify: `tutorial-reliability.staging.spec.ts`

- [ ] **Step 1: Wrap the test in a count loop** (independent tests so one failure doesn't block the rest)

```typescript
const BATCH_SIZE = Number(process.env.HARNESS_BATCH_SIZE ?? "1");
for (let i = 1; i <= BATCH_SIZE; i++) {
  test(`tutorial run ${i}/${BATCH_SIZE}`, async ({ page }) => {
    process.env.HARNESS_RUN_INDEX = String(i);
    // ... the full single-run body (beforeEach already resets) ...
  });
}
```

(`playwright.staging.config.ts` already sets `fullyParallel:false, workers:1, retries:0` — runs are sequential and all execute even if some fail.)

- [ ] **Step 2: Run a 2-run batch to verify isolation**

Run:
```bash
cd src/elspeth/web/frontend
HARNESS_BATCH_ID=verify-2 HARNESS_BATCH_SIZE=2 \
STAGING_BASE_URL=https://elspeth.foundryside.dev STAGING_USERNAME=<staging-username> STAGING_PASSWORD=<staging-password> \
PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
npx playwright test --config=playwright.staging.config.ts tutorial-reliability.staging.spec.ts
ls tests/e2e/.harness-results/verify-2/
```
Expected: two records `run-01.json`, `run-02.json`; each independent (run 2 not contaminated by run 1 — fresh session ids).

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts
git commit -m "feat(harness): parameterized N-run battery with reset between"
```

---

## Task 9: Aggregator + version-stamped trend log + report

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/harness/aggregate.mjs`
- Create (dir): `notes/tutorial-reliability/`

- [ ] **Step 1: Write the aggregator**

```javascript
#!/usr/bin/env node
import { readFileSync, readdirSync, writeFileSync, appendFileSync, mkdirSync } from "node:fs";
import { execSync } from "node:child_process";

const batchId = process.argv[2];
if (!batchId) { console.error("usage: aggregate.mjs <batch_id>"); process.exit(1); }
const dir = `tests/e2e/.harness-results/${batchId}`;
const records = readdirSync(dir).filter((f) => /^run-\d+\.json$/.test(f)).map((f) => JSON.parse(readFileSync(`${dir}/${f}`, "utf-8")));

const n = records.length;
const passes = records.filter((r) => r.outcome === "pass");
const infra = records.filter((r) => r.outcome === "infra_fault");
const tutorialDenom = n - infra.length; // exclude infra noise from the rate we drive to 100%
const tutorialPassRate = tutorialDenom ? passes.length / tutorialDenom : 0;
const infraRate = n ? infra.length / n : 0;

const gitSha = execSync("git rev-parse --short HEAD").toString().trim();
const harnessVersion = "1.0.0";
const skillHash = records.find((r) => r.stamp.composer_skill_hash)?.stamp.composer_skill_hash ?? "unknown";
const modelId = records.find((r) => r.stamp.model_identifier)?.stamp.model_identifier ?? "unknown";

// failure table
const fails = records.filter((r) => r.outcome !== "pass");
const tableRows = fails.map((r) => `| ${r.run_index} | ${r.outcome} | ${r.fault_subclass ?? ""} | ${r.fix_target ?? ""} | turn ${r.turn_reached} | ${r.landscape.realsystem_failure ?? r.error ?? ""} |`).join("\n");

const report = `# Tutorial reliability batch \`${batchId}\`

- git: \`${gitSha}\` · harness: \`${harnessVersion}\` · model: \`${modelId}\` · skill_hash: \`${skillHash.slice(0,12)}\`
- runs: ${n} · **tutorial-pass-rate: ${passes.length}/${tutorialDenom} (${(tutorialPassRate*100).toFixed(0)}%)** · infra-noise: ${infra.length}/${n} (${(infraRate*100).toFixed(0)}%)
- dim pass counts — a:${records.filter(r=>r.dim_a_tutorial_completed).length} b:${records.filter(r=>r.dim_b_realsystem_passed).length} c:${records.filter(r=>r.dim_c_assumptions_ok).length} d:${records.filter(r=>r.dim_d_solution_quality.status==="pass").length}(judged)

## Failures
| run | outcome | subclass | fix-target | reached | detail |
|-----|---------|----------|-----------|---------|--------|
${tableRows || "_none_"}
`;

mkdirSync("../../../../notes/tutorial-reliability", { recursive: true });
writeFileSync(`../../../../notes/tutorial-reliability/${batchId}.md`, report);
const trend = { batch_id: batchId, git: gitSha, harness: harnessVersion, model: modelId, skill_hash: skillHash, n, tutorial_pass: passes.length, tutorial_denom: tutorialDenom, infra: infra.length };
appendFileSync("../../../../notes/tutorial-reliability/trend.jsonl", JSON.stringify(trend) + "\n");
console.log(report);
```

(Path `../../../../notes/...` resolves from `tests/e2e/harness/` to the repo root `notes/`.)

- [ ] **Step 2: Run it over the verify-2 batch**

Run: `cd src/elspeth/web/frontend && node tests/e2e/harness/aggregate.mjs verify-2`
Expected: a printed report; `notes/tutorial-reliability/verify-2.md` + a `trend.jsonl` line.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/harness/aggregate.mjs notes/tutorial-reliability/
git commit -m "feat(harness): batch aggregator + version-stamped trend log"
```

---

## Task 10: Dimension (d) judging step + README

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/harness/README.md`

- [ ] **Step 1: Write the README** documenting:
  - the env-var invocation (batch id, batch size);
  - the per-run record location and schema (point at `types.ts`);
  - the **dimension (d) judging loop**: after a batch, the operator/agent reads each record's `output_rows`, applies `JUDGE_RUBRIC.judgePrompt`, and sets `dim_d_solution_quality.status` to `pass`/`fail` + `judge_score`; re-run `aggregate.mjs` to finalize. (Full LLM-API automation is deferred per spec §9.)
  - the mechanical-vs-judge split (reachability/discard computed in-spec; quality judged).

- [ ] **Step 2: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/harness/README.md
git commit -m "docs(harness): README + dimension-d judging loop"
```

---

## Task 11: Run the 10-run battery (the deliverable)

- [ ] **Step 1: Run the battery**

Run:
```bash
cd src/elspeth/web/frontend
HARNESS_BATCH_ID=batch-2026-06-06 HARNESS_BATCH_SIZE=10 \
STAGING_BASE_URL=https://elspeth.foundryside.dev STAGING_USERNAME=<staging-username> STAGING_PASSWORD=<staging-password> \
PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
npx playwright test --config=playwright.staging.config.ts tutorial-reliability.staging.spec.ts
```
Expected: 10 records written (some tests may show red — that is data, not a stop).

- [ ] **Step 2: Judge dimension (d)** per the README loop over the 10 records.

- [ ] **Step 3: Aggregate + read the report**

Run: `node tests/e2e/harness/aggregate.mjs batch-2026-06-06`
Expected: `notes/tutorial-reliability/batch-2026-06-06.md` with tutorial-pass-rate, infra-noise rate, dim counts, and the failure table.

- [ ] **Step 4: Commit the report**

```bash
git add notes/tutorial-reliability/
git commit -m "report(harness): tutorial reliability batch 2026-06-06 (10 runs)"
```

---

## Task 12: File the two findings (scope discipline §10)

- [ ] **Step 1:** File a Filigree bug for the **normalization fidelity gap** (`_normalise_current_tutorial_state_for_execution` repairs the pipeline only on the tutorial path) — attach any `normalization-gap` run records as evidence.
- [ ] **Step 2:** File a Filigree bug/observation for the **cache-as-fakery** risk (canonical cached path can green a broken composer). Reference spec §9/§10.
- [ ] **Step 3:** No commit (tracker writes).

---

## Self-Review

**Spec coverage:**
- §2 four dimensions → (a) Task 4/6, (b) Task 7, (c) Task 6, (d) Task 6 material + Task 10 judging. ✓
- §3 single-run sequence + skeleton-first → Task 4. ✓
- §4 fixed cache-bypass prompt → Task 1 + Task 5 assertion. ✓
- §5 assumption rubric (deterministic) → Task 1 + Task 6. ✓
- §6 solution-quality (judge + mechanical) → Task 1 + Task 6 + Task 10. ✓
- §7 failure classification → `types.ts` (Task 2) + Task 6/7 logic. ✓
- §8 batch record + version tagging → Task 6 record + Task 9 stamp. ✓
- §9 scaling deferral → noted; not built. ✓
- §10 findings not fixes → Task 7 note + Task 12. ✓
- §11 risks (single account/sequential, timing) → Task 8 (workers:1), Task 4 (timeouts). ✓

**Placeholder scan:** no TBD/TODO; every code step has real content. The only deliberately-deferred automation (dim-d LLM-API judge) is explicitly scoped out per §9 and replaced with a concrete agent-judging loop (Task 10). ✓

**Type consistency:** `RunRecord`/`FaultSubclass` (Task 2) used identically in Task 6/7/9; helper names (`harnessCtx`, `resetToFirstRun`, `fetchInterpretationEvents`, `fetchComposition`, `startRealRun`, `pollRunTerminal`, `fetchDiagnostics`, `reachableSourceCount`) defined in Task 3 and used as-is downstream. ✓

**Known empirical risk:** exact selector text/timeouts and the `/api/sessions/{sid}/composition` `composer_meta` field path are confirmed during Task 4/Task 3 against the live app; the plan cites the source models to mirror. This is intrinsic to live-E2E and is why the skeleton task gates the rest.
