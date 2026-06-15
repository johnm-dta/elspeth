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
import { classifyOutcome, type StepSignal } from "./harness/classify";
import type { RunRecord } from "./harness/types";
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

// Accept every assumption card, then wait for the continue button to enable.
// Robust to any assumption-set the (non-deterministic) composer produces and to
// the LLM-prompt-template REVIEW GATE: that card's Accept button is `disabled`
// until its prompt region (role="region" name="Prompt template review") is
// scrolled to the end (InterpretationReviewTurn: requiresPromptTemplateScroll →
// hasScrolledToEnd). The earlier helper clicked nth(0) blindly and hung on that
// disabled button — the dominant false "infra timeout" in batch-2026-06-06.
//
// Loop: ungate every prompt-template region by scrolling it to the bottom (which
// fires its onScroll handler), then click the first ENABLED Accept button, until
// the continue button ("Looks good") un-disables (pendingCount === 0).
async function acceptAllAssumptions(page: Page): Promise<number> {
  const continueBtn = page.getByRole("button", { name: "Looks good" });
  const acceptButtons = page.getByRole("button", { name: /^Accept /i });
  const promptRegions = page.getByRole("region", {
    name: "Prompt template review",
  });
  let accepted = 0;
  const deadline = Date.now() + 90_000;
  while (await continueBtn.isDisabled().catch(() => true)) {
    if (Date.now() > deadline) {
      throw new Error(
        "assumptions never all became acceptable (continue stayed disabled)",
      );
    }
    // Ungate prompt-template reviews: scroll each region to its end and fire the
    // scroll event the gate listens for.
    const regionCount = await promptRegions.count().catch(() => 0);
    for (let i = 0; i < regionCount; i++) {
      await promptRegions
        .nth(i)
        .evaluate((el) => {
          el.scrollTop = el.scrollHeight;
          el.dispatchEvent(new Event("scroll"));
        })
        .catch(() => {});
    }
    // Click the first currently-enabled Accept button, then re-evaluate (the
    // accepted card unmounts, shrinking the list).
    const total = await acceptButtons.count().catch(() => 0);
    let clicked = false;
    for (let i = 0; i < total; i++) {
      const btn = acceptButtons.nth(i);
      if (await btn.isEnabled().catch(() => false)) {
        await btn.click().catch(() => {});
        accepted += 1;
        clicked = true;
        break;
      }
    }
    if (!clicked) await page.waitForTimeout(300);
  }
  return accepted;
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

async function runOnce(page: Page, runIndex: number): Promise<void> {
  test.setTimeout(900_000); // real compose + tutorial run; headroom for the draft-wait (≤420s) + run-wait (≤360s) below

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

  // --- backend-grounded step signals (de-conflation; see harness/classify.ts) ---
  // Capture the compose (POST /api/sessions/{id}/messages) and run
  // (POST /api/tutorial/run) POSTs REGARDLESS of resp.ok(), plus whether each
  // request even responded before the deadline. Classification keys on these,
  // not on the Playwright timeout string (which always says "Timeout").
  const mkStep = (): StepSignal => ({
    fired: false,
    responded: false,
    status: null,
    bodyText: null,
    elapsedMs: null,
  });
  const step = { compose: mkStep(), run: mkStep() };
  const startMs: { compose: number | null; run: number | null } = { compose: null, run: null };
  const isCompose = (url: string, method: string) =>
    method === "POST" && /\/api\/sessions\/[0-9a-f-]{36}\/messages\b/i.test(url);
  const isRun = (url: string, method: string) =>
    method === "POST" && url.includes("/api/tutorial/run");

  page.on("request", (req) => {
    const url = req.url();
    const method = req.method();
    if (isCompose(url, method)) {
      step.compose.fired = true;
      startMs.compose = Date.now();
    } else if (isRun(url, method)) {
      step.run.fired = true;
      startMs.run = Date.now();
    }
  });
  page.on("requestfailed", (req) => {
    // Transport failure (connection reset / abort) — record the failure text so
    // INFRA_BODY in the classifier can see it. responded stays false.
    const url = req.url();
    const method = req.method();
    const errText = req.failure()?.errorText ?? "connection failed";
    if (isCompose(url, method)) step.compose.bodyText ??= errText;
    else if (isRun(url, method)) step.run.bodyText ??= errText;
  });

  // Capture session id, tutorial run id, cache flag, output rows, AND the
  // compose/run step status+timing+error-body (Task 5 + Task 6 Step 2).
  page.on("response", async (resp) => {
    const url = resp.url();
    const method = resp.request().method();
    const compose = isCompose(url, method);
    const run = isRun(url, method);
    if (compose || run) {
      const tgt = compose ? step.compose : step.run;
      const start = compose ? startMs.compose : startMs.run;
      tgt.responded = true;
      tgt.status = resp.status();
      tgt.elapsedMs = start !== null ? Date.now() - start : null;
      if (!resp.ok()) tgt.bodyText = await resp.text().catch(() => null);
    }
    if (run && resp.ok()) {
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
      // Headroom over the 270s backend composer_timeout + UI settle. The
      // tutorial's heavy 5-source prompt drives many composer turns PLUS the
      // mandatory opus advisor checkpoints (early plan-review + end sign-off,
      // up to 2 passes each), so a healthy-but-slow compose can approach the
      // backend cap before Turn 2b renders. 300s flagged those as failures.
    ).toBeVisible({ timeout: 420_000 });
    await acceptAllAssumptions(page);
    await page.getByRole("button", { name: "Looks good" }).click();

    // Turn 3: run it.
    await page.getByRole("button", { name: "Looks good, run it" }).click();
    turnReached = 3;

    // Turn 4: wait for completion, continue to audit story.
    await expect(page.getByRole("button", { name: "Continue" })).toBeVisible({
      timeout: 360_000, // raised for LLM-provider latency under load
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

    // Classify (spec §7) via the pure, unit-tested classifier (harness/classify.ts).
    // It keys on the BACKEND outcome of the blocking step (compose/run POST status,
    // whether it responded, whether a pipeline was composed) — NOT on the Playwright
    // "Timeout" string, which the old classifier matched and which laundered compose/
    // run validation failures and provider latency into one `infra_fault` bucket
    // (notes/tutorial-harness-infra-timeout-rootcause-2026-06-07.md). The two headline
    // numbers (tutorial-pass-rate vs infra-noise rate) are only meaningful once these
    // are separated.
    const { outcome, sub, fix } = classifyOutcome({
      graduated,
      turnReached,
      compose: step.compose,
      run: step.run,
      composedNodeCount: comp.nodes.length,
      normalized,
      underFlaggedCount: underFlagged.length,
      overFlaggedCount: overFlagged.length,
      reachable,
      minReachable: JUDGE_RUBRIC.minReachableSources,
      discardedRowCount,
      maxDiscarded: JUDGE_RUBRIC.maxDiscardedRows,
      substantive,
      minSubstantive: JUDGE_RUBRIC.minSubstantiveRows,
      outputRowCount: outputRows.length,
      hardError,
    });

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
      timing_s: {
        ...(step.compose.elapsedMs !== null
          ? { compose_s: Math.round(step.compose.elapsedMs / 100) / 10 }
          : {}),
        ...(step.run.elapsedMs !== null
          ? { run_s: Math.round(step.run.elapsedMs / 100) / 10 }
          : {}),
      },
      // Backend step evidence (the de-conflation inputs) — kept in the record so
      // a future batch is diagnosable without re-running: did each POST fire,
      // respond, with what status, in how long.
      steps: {
        compose: {
          fired: step.compose.fired,
          responded: step.compose.responded,
          status: step.compose.status,
          elapsed_s: step.compose.elapsedMs !== null ? Math.round(step.compose.elapsedMs / 100) / 10 : null,
          body: step.compose.bodyText?.slice(0, 500) ?? null,
        },
        run: {
          fired: step.run.fired,
          responded: step.run.responded,
          status: step.run.status,
          elapsed_s: step.run.elapsedMs !== null ? Math.round(step.run.elapsedMs / 100) / 10 : null,
          body: step.run.bodyText?.slice(0, 500) ?? null,
        },
      },
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
  // Inter-run cooldown: spacing the runs reduces the LLM-provider contention that
  // inflated composition/run latency under back-to-back load (batch-2026-06-06:
  // isolated runs ~72s, but 3/10 timed out under rapid succession). Skipped before
  // the first run. Tune via HARNESS_COOLDOWN_MS.
  const COOLDOWN_MS = Number(process.env.HARNESS_COOLDOWN_MS ?? "15000");
  let cooldownNeeded = false;
  test.beforeEach(async () => {
    if (cooldownNeeded) await new Promise((r) => setTimeout(r, COOLDOWN_MS));
    cooldownNeeded = true;
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
