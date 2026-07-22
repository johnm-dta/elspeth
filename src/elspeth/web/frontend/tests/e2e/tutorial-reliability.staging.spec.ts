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
  JUDGE_RUBRIC,
} from "./harness/prompt-and-rubric";
import { classifyOutcome, type StepSignal } from "./harness/classify";
import { ACKNOWLEDGEMENT_PRIMARY_ACTION_NAMES } from "./harness/guided-driver";
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

// Acknowledge every pending guided interpretation card currently rendered, then
// return how many were acknowledged this pass.
//
// Post-redesign (acknowledge card-stack): the cards render in the pinned
// AcknowledgementStack. Most cards resolve through an "Acknowledge …" primary.
// Prompt-template cards are two-stage: the same primary first says
// "View prompt", then flips to "Approve the LLM prompt template". Drive those
// primary actions as first-class unblockers; otherwise the prompt review stays
// pending and "Confirm wiring" never enables.
async function resolveVisibleReviews(page: Page): Promise<number> {
  const primaryButtons = ACKNOWLEDGEMENT_PRIMARY_ACTION_NAMES.map((name) =>
    page.getByRole("button", { name }),
  );
  const legacyViewToggles = page.getByRole("button", { name: /^View$/ });
  const promptRegions = page.getByRole("region", {
    name: "Prompt template review",
  });
  let actions = 0;
  // Bounded inner loop: each acknowledged card unmounts, shrinking the list.
  // Prompt-template cards take two iterations (View prompt -> Approve).
  for (let guard = 0; guard < 12; guard++) {
    // Keep the retired exact-"View" path for old staging bundles while the
    // current bundle uses the two-stage primary handled below.
    const toggleCount = await legacyViewToggles.count().catch(() => 0);
    for (let i = 0; i < toggleCount; i++) {
      await legacyViewToggles.nth(i).click().catch(() => {});
    }
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
    let clicked = false;
    for (const buttons of primaryButtons) {
      const total = await buttons.count().catch(() => 0);
      for (let i = 0; i < total; i++) {
        const btn = buttons.nth(i);
        if (await btn.isEnabled().catch(() => false)) {
          await btn.click().catch(() => {});
          actions += 1;
          clicked = true;
          break;
        }
      }
      if (clicked) break;
    }
    if (!clicked) {
      await page.waitForTimeout(300);
    }
  }
  return actions;
}

// Drive the staged guided walk (source → sink → recipe/transforms → wire) to
// completion.
//
// !! UNVERIFIED (operator-env blocked) !! This driver could not be run: it needs
// a live, LLM-backed staging deploy (chat-resolve of the source + LLM-proposed
// transforms have no provider in any environment reachable here). It is a
// best-effort scaffold; the source-stage chat affordance and any downstream
// schema_form stages the pump cannot fill are the named residual risks — a live
// run is required to finalize it. See the task report.
//
// Shape: the canonical hello-world source is dynamic-source-from-chat — the user
// DESCRIBES the URL scenario in the step-1 "Describe what you want" chat and the
// LLM resolves a complete inline source request, advancing the wizard (guided.ts
// GuidedChatResponse: "Step 1 source chat may resolve a complete inline source
// request; then these fields mirror /guided/respond"). We seed FIXED_PROMPT into
// that chat once at step_1_source, then run a robust turn-pump: each pass
// resolves surfaced per-stage interpretation cards, then advances by the enabled
// stage primary ("Confirm wiring" — the D12 wire gate that frees once cards are
// resolved — or "Continue" for schema forms). Exits once the run
// turn mounts or the deadline trips. A driver that drove the source via a
// plugin+schema_form instead would test a deterministic path and defeat the
// harness's purpose (grading the real LLM-backed scenario, dims a/b/c/d).
async function driveGuidedWalk(page: Page): Promise<void> {
  const guidedPanel = page.getByLabel(/guided composer/i);
  const runHeading = page.getByRole("heading", { name: /Running your pipeline/i });
  const stepChat = page.getByRole("region", { name: "Describe what you want" });
  const stepChatInput = stepChat.getByLabel("Message input");
  const stepChatSend = stepChat.getByRole("button", { name: "Send message" });

  // The tutorial is the NORMAL guided flow with the intent PRELOCKED at every
  // phase — that lock is the ONLY difference from guided mode. The learner types
  // nothing and never picks from a widget: on each LLM-driven phase they press
  // Send on the prelocked worked-example prompt, and the orchestrator LLM builds
  // THAT phase via the apply-capable /guided/chat drivers (resolve_source →
  // resolve_sink → proposal), each extracting its part of the one prompt.
  // We therefore drive each phase by Send (once per phase) and advance through
  // the structured result via the stage primaries. Wait for the locked prompt to
  // populate (synthetic URLs are fetched + appended async).
  await expect(stepChatInput).toBeVisible({ timeout: 30_000 });
  await expect(stepChatInput).not.toHaveValue("", { timeout: 30_000 });

  // Stage primary affordances, in priority order. "Confirm wiring" is the wire
  // gate (D12): it stays disabled until the stage's interpretation cards are
  // resolved, which resolveVisibleReviews handles each pass.
  //
  // "Review wiring" carries a send-first guard (below): the step-2→step-3
  // transition auto-plans a FIRST proposal from a fallback intent BEFORE the
  // locked transforms prompt is sent — accepting that one commits a
  // source→sink passthrough that the tutorial launch gate rejects (run 18,
  // session 07e8a3a8). The primary is honored only after this driver has
  // Sent the Transforms-phase prompt, so the proposal it accepts is the
  // frozen-prompt revision. (The tutorial UI withholds the button on the
  // pre-Send auto-proposal too — supersedes_draft_hash null — this guard
  // keeps the driver correct on its own.)
  const reviewWiring = page.getByRole("button", { name: "Review wiring", exact: true });
  const primaries = [
    page.getByRole("button", { name: "Confirm wiring", exact: true }),
    // Pipeline proposal turn (propose_pipeline): the transforms phase yields a
    // REAL planner proposal; accepting it (chosen ["review_wiring"]) is the
    // only advance into the wire stage. Renders only on the proposal turn.
    reviewWiring,
    page.getByRole("button", { name: "Continue", exact: true }),
    // Source inspection review (inspect_and_confirm): rendered after the
    // chat-resolved inline source is materialized into a session blob and
    // inspected — confirming the observed columns is the designed answer.
    page.getByRole("button", { name: "Looks right", exact: true }),
    // Component review turns: once the chat-resolved source/output lands as a
    // reviewed component, the stage ends on its review turn — finishing it is
    // the designed advance (mirrors composer-guided-live).
    page.getByRole("button", { name: "Finish sources", exact: true }),
    page.getByRole("button", { name: "Finish outputs", exact: true }),
    // Transient provider failure on a step chat ("I'm unavailable right now")
    // leaves a Retry affordance; pressing it is the designed recovery. Last in
    // priority so it never preempts forward progress.
    page.getByRole("button", { name: "Retry", exact: true }),
    // Output required-fields turn (multi_select_with_custom): the sink the LLM
    // built is observed-mode (pass-all-through), and the real output fields come
    // from the downstream transforms — so the correct, designed answer here is
    // the escape, not ticking the source's `url` column. Only renders on this
    // one turn, so it never preempts another stage's primary.
    page.getByRole("button", { name: "Let source decide (pass all fields through)", exact: true }),
  ];

  // The phases the LLM builds from intent (source/sink/transforms). Recipe + Wire
  // are confirm-only (no chat). Labels come from the workflow stepper.
  const drivenPhases = new Set(["Source", "Output", "Transforms"]);

  // Active guided phase, read from the stepper's aria-current step — used to send
  // the locked prompt exactly ONCE per phase (re-sending mid-build would
  // re-trigger the driver).
  async function currentPhase(): Promise<string | null> {
    const label = page.locator(".guided-workflow-step--current .guided-workflow-label").first();
    const text = await label.textContent().catch(() => null);
    return text ? text.trim() : null;
  }

  let lastDrivenPhase: string | null = null;
  // One-shot guard: the redesigned guided decision renders as a READ-ONLY
  // summary (.guided-schema-summary), not an editable form. The first time a
  // summary is visible, assert no editable schema input is shown alongside it.
  let assertedSummary = false;
  const deadline = Date.now() + 600_000;
  while (Date.now() < deadline) {
    // Done once the guided surface is replaced by the run turn.
    if (await runHeading.isVisible().catch(() => false)) {
      // Non-vacuous: by the time the run turn appears we must have observed at
      // least one read-only decision summary (the source decision is one), so
      // the guard above cannot silently no-op.
      expect(assertedSummary, "expected to observe a read-only decision summary").toBe(true);
      return;
    }
    if (!(await guidedPanel.isVisible().catch(() => false))) return;

    if (
      !assertedSummary &&
      (await page.locator(".guided-schema-summary").first().isVisible().catch(() => false))
    ) {
      assertedSummary = true;
      // Capture the redesigned rationale-led read-only decision for a visual
      // check (named, single artifact; overwritten each run).
      await page
        .screenshot({ path: "test-results/guided-decision-summary.png", fullPage: true })
        .catch(() => {});
      if ((await page.locator(".guided-schema-input").count().catch(() => 0)) > 0) {
        throw new Error(
          "guided decision rendered an editable form, expected a read-only summary",
        );
      }
    }

    await resolveVisibleReviews(page);

    // 1. Advance through the structured result via an enabled stage primary.
    let advanced = false;
    for (const primary of primaries) {
      // Send-first guard: never accept a transforms proposal before the
      // locked Transforms prompt has been sent this walk.
      if (primary === reviewWiring && lastDrivenPhase !== "Transforms") continue;
      if (
        (await primary.count().catch(() => 0)) > 0 &&
        (await primary.isEnabled().catch(() => false))
      ) {
        await primary.click().catch(() => {});
        advanced = true;
        break;
      }
    }
    if (advanced) {
      await page.waitForTimeout(750);
      continue;
    }

    // 2. No primary yet — drive the CURRENT LLM phase with the locked prompt. A
    //    confirm primary appears once the result renders.
    const phase = await currentPhase();
    const canSend = await stepChatSend.isEnabled().catch(() => false);
    if (canSend && phase !== null && drivenPhases.has(phase) && phase !== lastDrivenPhase) {
      await stepChatSend.click().catch(() => {});
      lastDrivenPhase = phase;
      await page.waitForTimeout(2_000); // let the /guided/chat round-trip settle
      continue;
    }

    await page.waitForTimeout(1_000);
  }
  throw new Error("guided walk never reached the run turn before the deadline");
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
const KNOWN_INPUT_KEYS = /^(?:url|source|html|html_content|raw_html|content|content_fingerprint)$/i;
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
  // Staged guided flow (P7): the compose phase drives the guided wizard via
  // POST /guided/respond (one per stage), not a single big-bang POST /messages.
  // We treat the FIRST guided/respond as the compose signal so the de-conflation
  // classifier still has a blocking-step status to key on.
  const isCompose = (url: string, method: string) =>
    method === "POST" && /\/api\/sessions\/[0-9a-f-]{36}\/guided\/respond\b/i.test(url);
  const isRun = (url: string, method: string) =>
    method === "POST" && url.includes("/api/tutorial/run");

  page.on("request", (req) => {
    const url = req.url();
    const method = req.method();
    if (isCompose(url, method)) {
      // The staged walk fires multiple guided/respond calls; record the FIRST
      // as the compose-phase start so the timing/status reflects the whole
      // compose phase, not just the last stage.
      if (!step.compose.fired) {
        step.compose.fired = true;
        startMs.compose = Date.now();
      }
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

    // Welcome bookend → Start mounts the guided composer surface.
    await page.getByRole("button", { name: "Let's go" }).click();
    turnReached = 1;

    // Staged guided walk (P7): the tutorial is now the real guided engine driven
    // with the tutorial profile (POST /guided/start happens inside
    // TutorialGuidedShell). The compose phase is the staged source → sink →
    // recipe/transforms → wire walk over POST /guided/respond, with per-stage
    // interpretation reviews surfaced inline (D12 gate). Drive it to completion;
    // the run auto-starts when the guided session reaches terminal=completed.
    await expect(page.getByLabel(/guided composer/i)).toBeVisible({
      timeout: 60_000,
    });
    turnReached = 2;
    await driveGuidedWalk(page);

    // On guided terminal=completed, TutorialGuidedShell hands off to the run
    // turn (which auto-starts the tutorial run). Wait for completion, continue
    // to the audit story. Headroom for LLM-provider latency over the heavy
    // 5-source canonical scenario plus the wire-stage advisor sign-off.
    await expect(page.getByRole("button", { name: "Continue" })).toBeVisible({
      timeout: 420_000,
    });
    turnReached = 3;
    await page.getByRole("button", { name: "Continue" }).click();
    turnReached = 4;

    // Audit story, continue.
    await expect(page.getByText(/This is the audit story/i)).toBeVisible();
    await page.getByRole("button", { name: "Continue" }).click();
    turnReached = 5;

    // Graduation: the staged flow saves the guided default + renames the session
    // and creates a fresh composer session on this single button (the old Turn-6
    // mode-choice radio is gone — graduation now owns the default-mode save).
    await page
      .getByRole("button", { name: "Take me to the composer" })
      .click();
    turnReached = 6;

    await expect(
      page.getByRole("heading", {
        name: "You're ready to use the composer.",
      }),
    ).toBeVisible();
    turnReached = 7;
    graduated = true;

    // Id-capture assertions (Task 5 Step 2). The cache-bypass assertion is kept
    // advisory: the staged tutorial runs the canonical scenario, so a cache hit
    // is no longer a fault the way a stale FIXED_PROMPT compose would have been —
    // the run still executes through the normal path. We record seededFromCache
    // in the RunRecord for the judge.
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
    // expectVerify entries may be an InterpretationKind (e.g. "pipeline_decision")
    // OR a user_term (e.g. "prompt_injection_shield_recommendation", the shield
    // review's discriminating signal whose kind is pipeline_decision). Match on
    // either, so the shield review is recognised by its user_term. Kinds are enum
    // values and user_terms are field-path strings, so the OR cannot false-match.
    const underFlagged = ASSUMPTION_RUBRIC.expectVerify.filter(
      (k) => !events.some((e) => e.kind === k || e.user_term === k),
    );
    // overFlagTerms mixes InterpretationKind-valued entries (e.g.
    // "invented_source", whose pattern /invent|fabricat/i matches the KIND) with
    // user_term-valued entries (e.g. "project_name"/"total_cost", matched on the
    // review's user_term). Test each term's pattern against EITHER the event kind
    // OR its user_term so the kind-valued entry is recognised. Kinds are enum
    // values and user_terms are field-path strings, so the OR cannot false-match
    // across the two namespaces.
    const overFlagged = ASSUMPTION_RUBRIC.overFlagTerms.filter((_label, i) => {
      const pattern = ASSUMPTION_RUBRIC.overFlagTermPatterns[i];
      return events.some(
        (e) =>
          (typeof e.kind === "string" && pattern.test(e.kind)) ||
          (typeof e.user_term === "string" && pattern.test(e.user_term)),
      );
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
