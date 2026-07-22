// ============================================================================
// composer-guided-ab-live.staging.spec.ts — THE TWO-LLM A/B TEST, guided
// surface (real LLM planner + real LLM pipeline run, live deployment).
//
// The guided-wizard twin of composer-ab-live.staging.spec.ts (the freeform A/B
// test). Same A/B contract: one colours CSV goes in, each row forks into TWO
// separate LLM assessments with different prompts (variant A: emotional tone,
// variant B: design usage context), and the variants are reconciled into a
// single row per colour — color_name, hex, and both variant outputs — before
// landing in a JSON sink.
//
// INTENT-UP-FRONT (the shipped 7.1 model — see below). The whole A/B pipeline
// intent is carried on POST /guided/start (profile "live"); the wizard then
// REVIEWS the source (CSV) and output (JSON sink) stages exactly like the
// pass-through colour test, and at the Output→Transforms transition the planner
// auto-fires FROM the root intent and stages the "Review pipeline proposal".
// step_3 is therefore review-only.
//
// WHY NOT drive the transforms via the step_3 docked composer: the frontend
// `chatGuided` routes on `compositionState === null` (sessionStore.ts) — first
// message → POST /guided/start, later messages → POST /guided/chat. By step_3
// the wizard walk has settled durable composition states, so a docked-composer
// Send there routes to /guided/chat and the backend rejects it with a 409
// ("Schema-8 CHAT is not available for step_3_transforms", guided_chat_atomic.py
// preflight, empty deferred_intents). The intent MUST ride on /guided/start
// instead — and /guided/start idempotent-no-ops once a checkpoint exists, so it
// is issued as the FIRST guided operation, before the wizard creates one. This
// spec issues it via the authed REST context (like the freeform twin's direct
// ctx calls); the UI then hydrates the guided wizard from the persisted
// checkpoint on load (selectSession → GET /guided), so no "Switch to guided".
//
// Requires the deployment to have an LLM operator profile configured
// (ELSPETH_WEB__LLM_PROFILES) — without one the composer must honestly decline,
// which composer-decline UX coverage owns, not this spec.
//
// Same gating as the sibling colour specs: skipped entirely unless
// ELSPETH_RUN_COMPOSER_LIVE=1, plus STAGING_* creds and ELSPETH_LIVE_OUTPUTS_DIR
// for the sink-file assertion. The run spends real provider tokens on both the
// planner turns and the two-LLM-per-row pipeline run — that is the point.
//
// Divergence from the freeform A/B's interpretation handling (deliberate, see
// resolveOneAcknowledgement below): the two authored LLM prompts surface as
// guided acknowledgement cards that GATE wizard advancement (ChatPanel's
// "approve-before-advance": GuidedTurn is disabled while there are pending
// interpretations). They are resolved through the guided-NATIVE Accept path
// (View prompt → Approve / Acknowledge), not the freeform spec's out-of-band
// `/interpretations/{id}/resolve` API — the guided run gate reads the client
// `pendingBySession` store, which only the native resolve mutates.
//
// Invocation (dev/staging server):
//   STAGING_BASE_URL=https://elspeth.foundryside.dev \
//   PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
//   STAGING_USERNAME=dta_user STAGING_PASSWORD=dta_pass \
//   ELSPETH_RUN_COMPOSER_LIVE=1 \
//   ELSPETH_LIVE_OUTPUTS_DIR=<server data_dir>/outputs \
//   npx playwright test --config=playwright.staging.config.ts composer-guided-ab-live --retries=0
//
// Optional visual evidence: set GUIDED_AB_SHOT_DIR to a writable directory to
// capture a full-page NN-<phase>.png after every wizard transition (default:
// screenshots are skipped).
//
// On failure the guided session is deliberately KEPT (not deleted): its
// sessions.db chat_messages transcript (writer=compose_loop) and Landscape rows
// are the diagnostic record. The failing session id is printed in the test
// output as "GUIDED-AB-LIVE SESSION <id>".
// ============================================================================

import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Locator, type Page } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
  uploadBlob,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../../../../../..");
// Reuse the pinned parity colour corpus shared with the freeform A/B spec:
// 10 rows of color_name,hex.
const FIXTURE_CSV = resolve(REPO_ROOT, "evals", "composer-parity", "fixtures", "two_llm_colour.csv");
const BLOB_FILENAME = "colours.csv";

const LIVE = process.env.ELSPETH_RUN_COMPOSER_LIVE === "1";
// Where the deployed server writes sink outputs (its data_dir/outputs). The
// dev server shares this filesystem; remote deployments can point this at a
// mount or skip the file assertion by running against the dev box instead.
const OUTPUTS_DIR = process.env.ELSPETH_LIVE_OUTPUTS_DIR;

// Optional visual evidence. When unset, screenshots are skipped entirely
// (mirrors tutorial-probe's SHOT_DIR pattern, but env-gated to a no-op default).
const SHOT_DIR = process.env.GUIDED_AB_SHOT_DIR;
let shotN = 0;
async function shot(page: Page, label: string): Promise<void> {
  if (!SHOT_DIR) return;
  shotN += 1;
  const name = `${String(shotN).padStart(2, "0")}-${label}`;
  await page.screenshot({ path: `${SHOT_DIR}/${name}.png`, fullPage: true }).catch(() => {});
  console.log(`[shot] ${name}`);
}

// Visible buttons (+ enabled state), headings, and any alert/error text — a
// cheap on-timeout inventory that localizes a live divergence instead of
// leaving a blind timeout (lifted from tutorial-probe.staging.spec.ts).
async function affordances(page: Page): Promise<string> {
  return page.evaluate(() => {
    const out: string[] = [];
    const vis = (el: Element) => {
      const r = (el as HTMLElement).getBoundingClientRect();
      const s = getComputedStyle(el as HTMLElement);
      return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none";
    };
    document.querySelectorAll("button").forEach((b) => {
      if (!vis(b)) return;
      out.push(
        `btn[${(b as HTMLButtonElement).disabled ? "OFF" : "on "}]: "${(b.textContent || "").trim().slice(0, 60)}"`,
      );
    });
    document.querySelectorAll("h1,h2,h3").forEach((h) => {
      if (vis(h)) out.push(`hdg: "${(h.textContent || "").trim().slice(0, 90)}"`);
    });
    document
      .querySelectorAll('[role="alert"], .error, [class*="error" i], [class*="Error"]')
      .forEach((e) => {
        if (vis(e)) {
          const t = (e.textContent || "").trim();
          if (t) out.push(`ALERT: "${t.slice(0, 160)}"`);
        }
      });
    return out.join("\n");
  });
}

// Every colour row the fixture contains must appear in the reconciled output.
interface ColourRow {
  color_name: string;
  hex: string;
}
function fixtureRows(): ColourRow[] {
  const lines = readFileSync(FIXTURE_CSV, "utf-8").trim().split("\n").slice(1);
  return lines.map((line) => {
    const [color_name, hex] = line.split(",");
    return { color_name, hex };
  });
}

// Resolve exactly ONE pending interpretation acknowledgement card, if any is
// present and actionable. Returns true when it acted.
//
// The card's primary button (AcknowledgementCard) is single OR two-stage by
// kind: an LLM-prompt card is two-stage — click 1 ("View prompt") reveals the
// prompt and flips the SAME button to "Approve", click 2 approves it; other
// kinds (pipeline decision / model choice / invented source / vague term) are
// single-stage ("Acknowledge"). One role locator matches the primary in every
// state (its accessible name is "View prompt" | "Approve …" | "Acknowledge …")
// while "Change…"/amend does not.
//
// DOCUMENTED ASSUMPTION: the cards carry data-testid="acknowledgement-card" and
// render above the intent box while advancement is gated. If the planner ever
// stops emitting per-prompt reviews, this is simply a no-op and advancement is
// ungated — the caller (advanceWhenReady) degrades gracefully to "already
// enabled".
async function resolveOneAcknowledgement(page: Page): Promise<boolean> {
  const cards = page.getByTestId("acknowledgement-card");
  if ((await cards.count().catch(() => 0)) === 0) return false;
  const card = cards.first();
  const primary = card.getByRole("button", { name: /view prompt|approve|acknowledge/i }).first();
  if (!(await primary.isEnabled().catch(() => false))) return false;
  const label = ((await primary.textContent().catch(() => "")) ?? "").trim();
  await primary.click().catch(() => {});
  if (/view prompt/i.test(label)) {
    // Two-stage prompt card: wait for the reveal to flip the button to Approve,
    // then approve.
    await expect(primary).toHaveText(/approve/i, { timeout: 10_000 });
    await primary.click().catch(() => {});
  }
  return true;
}

// Resolve pending acknowledgements until `primary` re-enables (or a bounded
// deadline). Pending interpretation reviews block the guided GuidedTurn widget
// ("approve-before-advance"), so the step-primary (Review wiring / Confirm
// wiring) stays disabled until every card is cleared. If nothing gates it, the
// loop returns on the first pass.
async function advanceWhenReady(page: Page, primary: Locator, label: string): Promise<void> {
  const deadline = Date.now() + 4 * 60_000;
  while (Date.now() < deadline) {
    if (await primary.isEnabled().catch(() => false)) return;
    const acted = await resolveOneAcknowledgement(page);
    if (!acted) await page.waitForTimeout(1_000);
  }
  await expect(
    primary,
    `${label} never enabled within the deadline — unresolved interpretation acknowledgements?`,
  ).toBeEnabled();
}

test.describe("composer guided live — the two-LLM A/B test (staging)", () => {
  test.skip(
    !LIVE,
    "set ELSPETH_RUN_COMPOSER_LIVE=1 to run the live guided A/B test (real planner + pipeline LLM spend)",
  );

  // Authoring alone is budgeted at up to 600s wall clock; the run adds two LLM
  // calls per row. Give the whole guided journey 25 minutes — matches the
  // freeform A/B twin.
  test.setTimeout(25 * 60_000);

  test("guided wizard authors and runs the fork/reconcile A/B assessment", async ({ page }) => {
    if (SHOT_DIR) mkdirSync(SHOT_DIR, { recursive: true });

    const csv = readFileSync(FIXTURE_CSV, "utf-8");
    const expectedRows = fixtureRows();
    // Unique per invocation so consecutive proof runs never collide on the sink
    // path and each output file is attributable to its run (mirrors the ab spec).
    const sinkFilename = `colour_guided_ab_${Date.now()}.json`;

    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    let sessionId: string | undefined;
    let passed = false;
    try {
      // ── Out-of-band setup: session + colour CSV blob ─────────────────────
      // Upload FIRST so the wizard's CSV source lands on the upload fast-path
      // summary (blob-bound path + observed schema) rather than a SCHEMA_FORM
      // dead end (the documented trap).
      const session = await createSession(ctx, "guided-ab-live-two-llm-test");
      sessionId = session.id;
      console.log(`GUIDED-AB-LIVE SESSION ${sessionId}`);
      await uploadBlob(ctx, sessionId, BLOB_FILENAME, csv);

      // ── Intent-up-front: seed the guided session via POST /guided/start ──
      // The WHOLE A/B pipeline intent (adapted verbatim from composer-ab-live's
      // request, field keys pinned to `tone`/`usage`) is the root intent. This
      // is the FIRST guided operation for the session, before any turn creates a
      // checkpoint (/guided/start idempotent-no-ops once one exists). The wizard
      // then reviews source/output and the planner auto-authors the fork →
      // two-LLM → reconcile pipeline at Output→Transforms.
      const abIntent =
        `I uploaded ${BLOB_FILENAME} (color_name,hex — ${expectedRows.length} rows). ` +
        `Build an A/B assessment pipeline: read the CSV, then fork each row into TWO ` +
        `separate LLM assessments with different prompts — variant A: describe the ` +
        `emotional tone of the colour in one short phrase; variant B: suggest one design ` +
        `usage context for the colour. Then reconcile the two variants back into a single ` +
        `row per colour holding color_name, hex, tone (from variant A) and usage (from ` +
        `variant B), and write all ${expectedRows.length} reconciled rows to a JSON file ` +
        `named ${sinkFilename}.`;
      const startResp = await ctx.post(`/api/sessions/${sessionId}/guided/start`, {
        data: { profile: "live", intent: abIntent, operation_id: randomUUID() },
      });
      expect(
        startResp.ok(),
        `POST /guided/start failed (${startResp.status()}): ${(await startResp.text()).slice(0, 500)}`,
      ).toBe(true);

      // ── Load the guided wizard (hydrated from the persisted checkpoint) ──
      // selectSession GETs /guided and renders the wizard from guidedSession —
      // no "Switch to guided" (the session is already guided). waitForChatReady
      // catches the brief freeform frame (store inits guidedSession=null before
      // the fetch resolves); the guided composer then replaces it.
      const composer = new ComposerPage(page);
      await composer.goto(sessionId);
      await composer.waitForChatReady();
      await expect(page.getByLabel(/guided composer/i)).toBeVisible({ timeout: 30_000 });
      await shot(page, "guided-shell");

      // ── Step 1 source: SINGLE_SELECT — CSV binds the uploaded blob ───────
      await page.getByRole("button", { name: "CSV", exact: true }).click();

      // ── Step 1 source: SCHEMA_FORM summary — upload fast-path auto-config
      // Choosing CSV with the blob already uploaded lands on a summary card
      // (observed schema + blob-bound path) with Edit/Continue. The observed
      // auto-config is exactly what we want — accept it.
      await expect(page.getByRole("button", { name: "Continue", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Continue", exact: true }).click();

      // ── Step 1 source: confirm columns, then finish sources ──────────────
      // The turn order within a stage is server-authored: an INSPECT_AND_CONFIRM
      // ("Looks right") may or may not be interposed before the sources
      // COMPONENT_REVIEW. Handle both explicitly.
      const looksRight = page.getByRole("button", { name: "Looks right" });
      const finishSources = page.getByRole("button", { name: "Finish sources" });
      await expect(looksRight.or(finishSources)).toBeVisible();
      if (await looksRight.isVisible().catch(() => false)) {
        await looksRight.click();
      }
      await expect(finishSources).toBeEnabled();
      await finishSources.click();
      await shot(page, "sources-finished");

      // ── Step 2 sink: SINGLE_SELECT — JSON output ─────────────────────────
      await page.getByRole("button", { name: "JSON", exact: true }).click();

      // ── Step 2 sink: SCHEMA_FORM — path, format, and the explicit
      // write-safety choices the runnable file-sink contract requires
      // (mode + collision_policy are operator decisions the engine refuses to
      // default; the form must collect them or planning wedges — L4).
      await page.getByRole("button", { name: "Edit", exact: true }).click();
      await page.getByLabel(/^path/i).fill(sinkFilename);
      await page.getByLabel(/^format$/i).selectOption("json");
      await page.getByLabel(/^mode$/i).selectOption("write");
      await page.getByLabel(/collision.?policy/i).selectOption("auto_increment");
      await expect(page.getByRole("button", { name: "Continue", exact: true })).toBeEnabled();
      await page.getByRole("button", { name: "Continue", exact: true }).click();

      // ── Step 2 sink: MULTI_SELECT_WITH_CUSTOM — observed pass-through ─────
      // The reconciled A/B rows carry fields (tone, usage) that don't exist yet
      // at sink-config time; "let source decide" makes the sink observed-mode so
      // it writes whatever the reconcile transform produces downstream.
      await page
        .getByRole("button", { name: "Let source decide (pass all fields through)" })
        .click();

      // ── Step 2 sink: COMPONENT_REVIEW — finish outputs ───────────────────
      await expect(page.getByRole("button", { name: "Finish outputs" })).toBeEnabled();
      await page.getByRole("button", { name: "Finish outputs" }).click();
      await shot(page, "outputs-finished");

      // ── THE FRONTIER: Output→Transforms AUTO-FIRES the real planner ──────
      // With the whole A/B intent set at /guided/start, the planner authors the
      // fork → two-LLM → reconcile pipeline at this transition and STAGES the
      // "Review pipeline proposal" — step_3 is review-only (no step_3 chat; that
      // would 409 as Schema-8 CHAT-not-available). Race the committed proposal
      // against any compose failure surface so a failed planner turn fails the
      // spec with the server's own words rather than after a blind timeout; on
      // timeout, dump the visible-affordance inventory to localize the divergence.
      const proposal = page.getByRole("article", { name: "Review pipeline proposal" });
      const failureSurface = page
        .getByText(
          /operation failed|integrity check|terminal failure|does not satisfy the current turn contract|encountered an error|unusable pipeline plan|composer_planner_failure|could not build|planner failure|unavailable|timed out/i,
        )
        .first();
      try {
        await expect
          .poll(
            async () => {
              if (await proposal.isVisible().catch(() => false)) return "proposal";
              if (await failureSurface.isVisible().catch(() => false)) return "failure";
              return "pending";
            },
            {
              timeout: 11 * 60_000,
              intervals: [2_000],
              message: "waiting for the guided A/B planner to auto-stage a pipeline proposal",
            },
          )
          .not.toBe("pending");
      } catch (error) {
        console.log("[affordances @ transforms-frontier timeout]\n" + (await affordances(page)));
        throw error;
      }
      await expect(
        failureSurface,
        "guided planner surfaced a terminal failure while authoring the A/B pipeline — " +
          `inspect sessions.db chat_messages and the server log for session ${sessionId}`,
      ).toHaveCount(0);
      await expect(proposal).toBeVisible();
      await shot(page, "transforms-proposal");

      // ── Step 3 transforms: resolve the prompts' reviews, then advance ────
      // The two authored LLM prompts surface as acknowledgement cards that gate
      // the proposal's "Review wiring" button. Clear them via the guided-native
      // Accept path until the button re-enables (see resolveOneAcknowledgement).
      const reviewButton = proposal.getByRole("button", { name: "Review wiring" });
      await advanceWhenReady(page, reviewButton, "Review wiring");
      await reviewButton.click();
      await shot(page, "wiring-review");

      // ── Step 4 wire: confirm (resolve any acknowledgements surfaced here) ─
      const confirmWiring = page.getByRole("button", { name: "Confirm wiring", exact: true });
      await advanceWhenReady(page, confirmWiring, "Confirm wiring");
      await confirmWiring.click();
      await shot(page, "wiring-confirmed");

      // ── Ready → run ──────────────────────────────────────────────────────
      // Confirm commits the composition; the session lands on "Pipeline ready"
      // with the guided run affordance.
      await expect(page.getByRole("heading", { name: "Pipeline ready" })).toBeVisible({
        timeout: 60_000,
      });
      await shot(page, "pipeline-ready");
      const runButton = page.getByRole("button", { name: "Run pipeline" }).first();
      await expect(runButton).toBeEnabled();
      await runButton.click();

      // Credential-egress confirmation: the run leaves the composer and uses
      // stored credentials, so an alertdialog interposes before execution.
      const runDialog = page.getByRole("alertdialog", { name: "Run pipeline?" });
      await expect(runDialog).toBeVisible();
      await runDialog.getByRole("button", { name: "Run pipeline" }).click();
      await shot(page, "run-started");

      // ── The A/B contract: every row reconciled with both variants ────────
      // The written sink file is the ground truth the goal names; poll for it
      // rather than guessing at the run-results surface.
      if (!OUTPUTS_DIR) {
        throw new Error(
          "ELSPETH_LIVE_OUTPUTS_DIR is not set — cannot verify the JSON output. " +
            "Point it at the deployed server's data_dir/outputs.",
        );
      }
      const outputPath = resolve(OUTPUTS_DIR, sinkFilename);
      await expect
        .poll(() => existsSync(outputPath), {
          timeout: 10 * 60_000,
          message: `waiting for the guided A/B pipeline run to write ${outputPath}`,
        })
        .toBe(true);
      await expect(
        page.getByText(/run failed|execution failed|pipeline failed/i).first(),
      ).toHaveCount(0);
      await shot(page, "run-finished");

      const output = JSON.parse(readFileSync(outputPath, "utf-8")) as Record<string, unknown>[];
      expect(output, `output ${outputPath} must hold all ${expectedRows.length} rows`).toHaveLength(
        expectedRows.length,
      );
      for (const row of expectedRows) {
        const match = output.find(
          (candidate) => candidate.color_name === row.color_name && candidate.hex === row.hex,
        );
        expect(match, `output is missing colour row ${row.color_name} (${row.hex})`).toBeTruthy();
        const reconciled = match as Record<string, unknown>;
        for (const variantField of ["tone", "usage"]) {
          const value = reconciled[variantField];
          expect(
            typeof value === "string" && value.trim().length > 0,
            `row ${row.color_name} must carry a non-empty '${variantField}' from its LLM variant, got: ${JSON.stringify(value)}`,
          ).toBe(true);
        }
      }
      passed = true;
    } finally {
      // Keep failed sessions: their transcript IS the diagnostic artifact. On
      // success the run may still be finalizing server-side (the output file
      // lands before the run settles), so tolerate the active-run 409 briefly
      // and otherwise leave the session rather than fail a green run.
      if (passed && sessionId !== undefined) {
        for (let attempt = 0; attempt < 6; attempt += 1) {
          try {
            await deleteSession(ctx, sessionId);
            break;
          } catch (error) {
            if (attempt === 5 || !String(error).includes("409")) {
              console.log(`GUIDED-AB-LIVE cleanup skipped (run still settling): ${sessionId}`);
              break;
            }
            await new Promise((resolve_) => setTimeout(resolve_, 5_000));
          }
        }
      }
      await ctx.dispose();
    }
  });
});
