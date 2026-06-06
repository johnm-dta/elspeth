// Tutorial reliability battery (non-mocked staging spec).
//
// Drives the REAL first-run composer tutorial against elspeth.foundryside.dev,
// resets between every run, grades each run on four dimensions (a/b/c/d), does a
// real-system re-run for dim (b), and writes one JSON RunRecord per run under
// tests/e2e/.harness-results/<batch_id>/run-NN.json (gitignored).
//
// Integrates plan Tasks 4 (skeleton flow + selectors), 5 (id/cache capture),
// 6 (grade a/b/c + dim-d raw material + write record), 7 (real-system re-run),
// and 8 (parameterized N-run battery with reset between).
//
// Scope discipline (spec §10): this harness only OBSERVES the normalization gap
// and cache-as-fakery; it does not fix them.

import { mkdirSync, writeFileSync } from "node:fs";

import { test, expect, type Page } from "@playwright/test";

import {
  ASSUMPTION_RUBRIC,
  FIXED_PROMPT,
  JUDGE_RUBRIC,
} from "./harness/prompt-and-rubric";
import type { FaultSubclass, RunRecord } from "./harness/types";
import {
  fetchComposition,
  fetchDiagnostics,
  fetchInterpretationEvents,
  harnessCtx,
  pollRunTerminal,
  reachableSourceCount,
  resetToFirstRun,
  cleanSessions,
  startRealRun,
} from "./helpers/tutorial-harness";

const BATCH_ID = process.env.HARNESS_BATCH_ID ?? "skeleton";
const BATCH_SIZE = Number(process.env.HARNESS_BATCH_SIZE ?? "1");

// Generic: click every "Accept ..." button rendered in the assumptions panel,
// then wait for the continue button to enable. Returns the count accepted.
//
// NOTE (live-run caveat, not a compile concern): the LLM-prompt-template review
// requires the region to be scrolled before its Accept enables (see the mocked
// tutorial.spec.ts). A later live task may need to scroll that region.
async function acceptAllAssumptions(page: Page): Promise<number> {
  const buttons = page.getByRole("button", { name: /^Accept /i });
  const count = await buttons.count();
  for (let i = 0; i < count; i++) await buttons.nth(0).click(); // list shrinks as accepted
  return count;
}

// Count rows whose any value is a meaningful, non-degenerate string. We do not
// hard-code the colour field key (it varies with the composed pipeline), so we
// scan the row's values defensively.
function substantiveRowCount(rows: Array<Record<string, unknown>>): number {
  const degenerate = /cannot|unknown|n\/a|none/i;
  return rows.filter((row) =>
    Object.values(row).some(
      (v) => typeof v === "string" && v.trim().length > 0 && !degenerate.test(v),
    ),
  ).length;
}

async function runOnce(page: Page, runIndex: number): Promise<void> {
  test.setTimeout(420_000); // real compose + tutorial run + real-system re-run can take minutes

  // --- per-run state (Task 5 capture targets; all consumed in the record) ---
  let sessionId: string | null = null;
  let tutorialRunId: string | null = null;
  let seededFromCache = false;
  let outputRows: Array<Record<string, unknown>> = [];
  let discardedRowCount = 0;

  let realsystemRunId: string | null = null;
  let realStatus: string | null = null;

  let turnReached = 0; // increment after each turn's success
  let graduated = false;
  let hardError: string | null = null;

  // Capture session id, tutorial run id, cache flag, and output rows from the
  // network (Task 5 + Task 6 Step 2).
  page.on("response", async (resp) => {
    const url = resp.url();
    if (
      url.includes("/api/tutorial/run") &&
      resp.request().method() === "POST" &&
      resp.ok()
    ) {
      const body = await resp.json().catch(() => null);
      if (body) {
        tutorialRunId = body.run_id ?? null;
        seededFromCache = body.seeded_from_cache ?? false;
        const output = body.output ?? {};
        outputRows = Array.isArray(output.rows) ? output.rows : [];
        discardedRowCount = output.discarded_row_count ?? 0;
      }
    }
    const m = url.match(/\/api\/sessions\/([0-9a-f-]{36})\//i);
    if (m && !sessionId) sessionId = m[1];
  });

  try {
    await page.goto("/");
    await expect(
      page.getByRole("main", { name: /first-run tutorial/i }),
    ).toBeVisible();

    // Turn 1: welcome.
    await page.getByRole("button", { name: "Let's go" }).click();
    turnReached = 1;

    // Turn 2: describe + build.
    await page.getByLabel("Pipeline description").fill(FIXED_PROMPT);
    await page.getByRole("button", { name: "Build it" }).click();
    turnReached = 2;

    // Turn 2b: review assumptions, accept all, continue.
    await expect(
      page.getByText(/Here is what the composer drafted/i),
    ).toBeVisible({ timeout: 180_000 });
    await acceptAllAssumptions(page);
    await page.getByRole("button", { name: "Looks good" }).click();

    // Turn 3: run it.
    await page.getByRole("button", { name: "Looks good, run it" }).click();
    turnReached = 3;

    // Turn 4: wait for completion, continue to audit story.
    await expect(page.getByRole("button", { name: "Continue" })).toBeVisible({
      timeout: 240_000,
    });
    await page.getByRole("button", { name: "Continue" }).click();
    turnReached = 4;

    // Turn 5: audit story, continue.
    await expect(page.getByText(/This is the audit story/i)).toBeVisible();
    await page.getByRole("button", { name: "Continue" }).click();
    turnReached = 5;

    // Turn 6: choose default mode, save.
    await page.getByRole("radio", { name: /Guided/i }).click();
    await page.getByRole("button", { name: "Save and go" }).click();
    turnReached = 6;

    // Turn 7: graduation.
    await expect(
      page.getByRole("heading", {
        name: "You're ready to use the composer.",
      }),
    ).toBeVisible();
    turnReached = 7;
    graduated = true;

    // Cache-bypass + id-capture assertions (Task 5 Step 2).
    expect(seededFromCache, "fixed prompt must bypass the cache").toBe(false);
    expect(sessionId, "session id captured").not.toBeNull();
    expect(tutorialRunId, "tutorial run id captured").not.toBeNull();

    // Dimension (b): the real-system re-run via the normal execute path (Task 7).
    if (sessionId) {
      const ctx2 = await harnessCtx();
      try {
        const rid = await startRealRun(ctx2, sessionId);
        realsystemRunId = rid;
        realStatus = await pollRunTerminal(ctx2, rid);
      } catch (e) {
        realStatus = `error:${e instanceof Error ? e.message : String(e)}`;
      } finally {
        await ctx2.dispose();
      }
    }
  } catch (e) {
    hardError = e instanceof Error ? e.message : String(e);
    throw e; // rethrow so Playwright captures trace/video for this failed run
  } finally {
    // --- build + write the per-run RunRecord (Task 6 + Task 7) ---
    const ctx = await harnessCtx();
    const events = sessionId
      ? await fetchInterpretationEvents(ctx, sessionId).catch(() => [])
      : [];
    const comp = sessionId
      ? await fetchComposition(ctx, sessionId).catch(() => ({
          composer_meta: null,
        }))
      : { composer_meta: null };
    const diag = tutorialRunId
      ? await fetchDiagnostics(ctx, tutorialRunId).catch(() => ({
          operations: [],
        }))
      : { operations: [] };

    const raisedKinds = events.map((e) => e.kind);
    const underFlagged = ASSUMPTION_RUBRIC.expectVerify.filter(
      (k) => !raisedKinds.includes(k),
    );
    const overFlagged = (
      ASSUMPTION_RUBRIC.expectWaive as readonly string[]
    ).filter((k) => raisedKinds.includes(k));
    const normalized =
      (comp.composer_meta as Record<string, unknown> | null)
        ?.tutorial_runtime_normalized === true;
    const reachable = reachableSourceCount(diag.operations);
    const substantive = substantiveRowCount(outputRows);

    // Dimension (b) is authoritative on the real-system re-run terminal status;
    // the normalization flag remains the annotation/trigger.
    const realsystemPassed = realStatus === "completed";

    // Classify (spec §7). Precedence: hard frontend fault → normalization/real-run
    // divergence (b) → assumption faults (c) → mechanical (d) faults.
    let outcome: RunRecord["outcome"] = "pass";
    let sub: FaultSubclass = null;
    let fix: string | null = null;
    if (hardError) {
      outcome = "tutorial_fault";
      sub = "frontend-state-machine";
      fix = "frontend / timing";
    } else if (normalized || !realsystemPassed) {
      // passed (a) but failed (b): the "tutorial lies" class.
      outcome = "tutorial_fault";
      sub = "normalization-gap";
      fix = "engine: tutorial normalization parity";
    } else if (underFlagged.length) {
      outcome = "tutorial_fault";
      sub = "assumption-under-flag";
      fix = "composer-skill-prompt: pipeline_composer.md review rules";
    } else if (overFlagged.length) {
      outcome = "tutorial_fault";
      sub = "assumption-over-flag";
      fix = "composer-skill-prompt: pipeline_composer.md review rules";
    } else if (reachable < JUDGE_RUBRIC.minReachableSources) {
      outcome = "tutorial_fault";
      sub = "invented-source-unreachable";
      fix = "composer-skill-prompt: generated-source discipline";
    } else if (
      discardedRowCount > JUDGE_RUBRIC.maxDiscardedRows ||
      substantive < JUDGE_RUBRIC.minSubstantiveRows
    ) {
      outcome = "tutorial_fault";
      sub = "degenerate-output";
      fix = "composer-skill-prompt: extraction discipline";
    }

    const record: RunRecord = {
      batch_id: BATCH_ID,
      run_index: runIndex,
      outcome,
      fault_subclass: sub,
      fix_target: fix,
      turn_reached: turnReached,
      tutorial_run_id: tutorialRunId,
      realsystem_run_id: realsystemRunId,
      seeded_from_cache: seededFromCache,
      dim_a_tutorial_completed: graduated,
      dim_b_realsystem_passed: realsystemPassed,
      dim_c_assumptions_ok:
        underFlagged.length === 0 && overFlagged.length === 0,
      dim_d_solution_quality: {
        status: "pending_judge",
        judge_score: null,
        source_reachable: `${reachable}/${JUDGE_RUBRIC.minReachableSources}`,
        discarded_row_count: discardedRowCount,
        substantive_rows: `${substantive}/${outputRows.length}`,
      },
      assumptions: {
        raised: events.map((e) => ({ kind: e.kind, term: e.user_term })),
        under_flagged: [...underFlagged],
        over_flagged: [...overFlagged],
      },
      output_rows: outputRows,
      landscape: {
        tutorial_failure: hardError,
        realsystem_failure:
          realStatus && realStatus !== "completed" ? realStatus : null,
        normalization_fired: normalized,
      },
      stamp: {
        composer_skill_hash: events[0]?.composer_skill_hash ?? null,
        model_identifier: events[0]?.model_identifier ?? null,
      },
      timing_s: {},
      error: hardError,
    };

    const dir = `tests/e2e/.harness-results/${BATCH_ID}`;
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      `${dir}/run-${String(runIndex).padStart(2, "0")}.json`,
      JSON.stringify(record, null, 2),
    );
    await ctx.dispose();
  }
}

test.describe("tutorial reliability battery", () => {
  test.beforeEach(async () => {
    const ctx = await harnessCtx();
    await cleanSessions(ctx);
    await resetToFirstRun(ctx);
    await ctx.dispose();
  });

  for (let i = 1; i <= BATCH_SIZE; i++) {
    // Independent tests so one failure does not block the rest; reset runs in
    // beforeEach between every one (config is workers:1, retries:0, sequential).
    test(`tutorial run ${i}/${BATCH_SIZE}`, async ({ page }) => {
      await runOnce(page, i);
    });
  }
});
