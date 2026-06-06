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
  reachableSourceCount,
  resetToFirstRun,
  cleanSessions,
  scrapeNodeId,
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

// Dimension (d) output-substance (spec §6): count rows that carry a meaningful,
// non-degenerate value in the LLM-EXTRACTED attribute — NOT in an input column.
//
// We do NOT hard-code the colour field key (it varies per composed pipeline).
// Instead we derive the input columns from the composition source (the keys of
// the first seeded source row, e.g. {url}) and treat every OTHER key in an
// output row as a candidate extraction field. We also exclude an obvious `html`
// key because the prompt strips HTML before the sink. A row is substantive iff
// at least one such extraction field holds a non-empty, non-degenerate string.
//
// The earlier version scanned EVERY value in the row, so any non-degenerate
// string anywhere (the URL, the agency name) made the row count — it measured
// "did we get a row" not "did the row carry a real extracted value", and the
// minSubstantiveRows check never bit. This targets the extraction output.
const DEGENERATE_VALUE = /cannot|unknown|n\/a|none|no clear|not (?:found|available|determined)/i;
const KNOWN_INPUT_KEYS = /^(?:url|source|agency|abuse_contact|scraping_reason|html|html_content|raw_html)$/i;
function substantiveRowCount(
  rows: Array<Record<string, unknown>>,
  sourceInputKeys: string[],
): number {
  const inputKeys = new Set(sourceInputKeys.map((k) => k.toLowerCase()));
  return rows.filter((row) =>
    Object.entries(row).some(([key, v]) => {
      const k = key.toLowerCase();
      if (inputKeys.has(k) || KNOWN_INPUT_KEYS.test(k)) return false; // not an extraction field
      return typeof v === "string" && v.trim().length > 0 && !DEGENERATE_VALUE.test(v);
    }),
  ).length;
}

// Classify a transport/error string as infra noise vs a composition fault.
// 5xx / 429 / rate-limit / connection / timeout / DNS / target-down → infra.
const INFRA_ERROR = /\b5\d\d\b|429|rate.?limit|throttl|timed? ?out|timeout|did not reach terminal|econn|socket hang|network|temporarily unavailable|bad gateway|service unavailable|gateway timeout/i;

async function runOnce(page: Page, runIndex: number): Promise<void> {
  test.setTimeout(420_000); // real compose + tutorial run + real-system re-run can take minutes

  // --- per-run state (Task 5 capture targets; all consumed in the record) ---
  let sessionId: string | null = null;
  let tutorialRunId: string | null = null;
  let seededFromCache = false;
  let outputRows: Array<Record<string, unknown>> = [];
  let discardedRowCount = 0;

  // No separate real-system re-run: when the tutorial applies NO normalization it
  // already executed through the normal ExecutionService→Orchestrator path (re-
  // running the same session collides on output artifacts → FileExistsError), so
  // dim (b) derives from the tutorial run's own success + the normalization flag.
  // realsystemRunId stays null by construction (kept for the record schema).
  const realsystemRunId: string | null = null;

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

    // Dimension (b): NO separate re-run (it would collide on output artifacts).
    // When normalization did not fire, the tutorial run IS the real-system path;
    // dim (b) is derived in the finally block from graduated + rows + the
    // normalization flag (parity principle: a fired normalization = fault).
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
          nodes: [],
          sourceInputKeys: [],
          raw: null,
        }))
      : { composer_meta: null, nodes: [], sourceInputKeys: [], raw: null };
    const scrapeNode = scrapeNodeId(comp.nodes);
    const diag = tutorialRunId
      ? await fetchDiagnostics(ctx, tutorialRunId).catch(() => ({
          operations: [],
          tokens: [],
          failureDetail: null,
        }))
      : { operations: [], tokens: [], failureDetail: null };
    const raisedKinds = events.map((e) => e.kind);
    const underFlagged = ASSUMPTION_RUBRIC.expectVerify.filter(
      (k) => !raisedKinds.includes(k),
    );
    // Over-flagging (spec §5): the composer raised an interpretation review whose
    // semantic TARGET is a value the user stated explicitly (abuse contact /
    // scraping reason). We match the review's `user_term` against the rubric's
    // term patterns — NOT the event `kind` (abuse_contact/scraping_reason are
    // implicit-decision field paths, never InterpretationKind values, so a
    // kind-comparison can never fire). See prompt-and-rubric.ts.
    const overFlagged = ASSUMPTION_RUBRIC.overFlagTerms.filter((_label, i) => {
      const pattern = ASSUMPTION_RUBRIC.overFlagTermPatterns[i];
      return events.some((e) => typeof e.user_term === "string" && pattern.test(e.user_term));
    });
    const normalized =
      (comp.composer_meta as Record<string, unknown> | null)
        ?.tutorial_runtime_normalized === true;
    const reachable = reachableSourceCount(diag.tokens, scrapeNode);
    const substantive = substantiveRowCount(outputRows, comp.sourceInputKeys);

    // Dimension (b): tutorial-backend PARITY. When normalization did NOT fire, the
    // tutorial run already executed through the normal ExecutionService→Orchestrator
    // path, so a graduated run with output rows IS a passing real-system run. A
    // fired normalization means the tutorial was treated differently from a regular
    // run (it repaired the composed pipeline) → dim (b) FAILS. No separate re-run is
    // issued (it would collide on the first run's output artifacts → FileExistsError).
    const dimBPassed = !normalized && graduated && outputRows.length > 0;

    // Classify (spec §7). Two headline numbers depend on this: tutorial-pass-rate
    // (driven to 100%) and infra-noise rate (held separate so prompt changes are
    // not confounded).
    //
    // Precedence:
    //   1. infra noise (timeout / 5xx / 429) thrown during the UI flow — excluded
    //      from the tutorial denominator entirely.
    //   2. hard frontend fault (ambiguous UI error / never-advanced turn).
    //   3. parity break (b): normalization fired → tutorial ran a different pipeline
    //      than a regular run would → normalization-gap.
    //   4. assumption faults (c).
    //   5. mechanical solution-quality (d) — reachability / discard / substantive.
    const hardErrorIsInfra = hardError !== null && INFRA_ERROR.test(hardError);

    let outcome: RunRecord["outcome"] = "pass";
    let sub: FaultSubclass = null;
    let fix: string | null = null;
    if (hardErrorIsInfra) {
      // A transport timeout / 5xx / 429 thrown during the UI flow.
      outcome = "infra_fault";
      sub = /did not reach terminal|timed? ?out|timeout/i.test(hardError ?? "")
        ? "timeout"
        : /\b5\d\d\b|bad gateway|service unavailable|gateway timeout/i.test(hardError ?? "")
          ? "llm-5xx-or-ratelimit"
          : "staging-hiccup";
      fix = null;
    } else if (hardError) {
      // An ambiguous UI error / a turn that never advanced — a genuine frontend
      // fault, which is exactly what this harness exists to catch. Default here
      // (do NOT route ambiguous UI errors into infra, which would deflate signal).
      outcome = "tutorial_fault";
      sub = "frontend-state-machine";
      fix = "frontend / timing";
    } else if (graduated && normalized) {
      // Parity break (backend-parity principle + spec §10): the tutorial REPAIRED
      // the composed pipeline before running; a regular /execute applies no such
      // repair, so the composer's actual output is not proven to run as a regular
      // run. Fix at source (composer emits correct templates / engine handles both)
      // — never a tutorial-only band-aid.
      outcome = "tutorial_fault";
      sub = "normalization-gap";
      fix = "remove tutorial-only normalization: make tutorial == regular run";
    } else if (underFlagged.length) {
      outcome = "tutorial_fault";
      sub = "assumption-under-flag";
      fix = "composer-skill-prompt: pipeline_composer.md review rules";
    } else if (overFlagged.length) {
      outcome = "tutorial_fault";
      sub = "assumption-over-flag";
      fix = "composer-skill-prompt: pipeline_composer.md review rules";
    } else if (reachable < JUDGE_RUBRIC.minReachableSources) {
      // d-mechanical: a scrape node-state failed for one of the invented URLs but
      // the run still completed (e.g. the row diverted). On a real network this is
      // the composer picking a bad URL; transient fetch failures surface as infra
      // above via the real-run why-signal.
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
      dim_b_realsystem_passed: dimBPassed,
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
        realsystem_failure: normalized
          ? "tutorial normalization repaired the pipeline; not run as a regular run"
          : null,
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
