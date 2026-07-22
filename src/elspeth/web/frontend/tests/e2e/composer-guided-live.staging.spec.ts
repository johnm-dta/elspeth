// ============================================================================
// composer-guided-live.staging.spec.ts — THE COLOUR TEST (guided wizard,
// REAL LLM planner, live deployment).
//
// This is the codified form of the hand-driven guided e2e that exposed the
// 7.1 authoring bug chain (elspeth-859e2702dd): backend units were green
// because they drove payload shapes the real UI never sends and mocked the
// planner. This spec is the missing gate — real UI payloads meeting the real
// planner on a deployed server. It is expected to FAIL at the current chain
// frontier until the chain is fixed; that is its job (TDD red).
//
// Scenario: upload a 10-row colours CSV → guided wizard (CSV source →
// column confirm → finish sources → JSON output → field review pass-through →
// finish outputs) → planner authors the pipeline → transforms → wiring →
// execute → the JSON output contains all ten colour rows.
//
// Gating (mirrors tests/integration/web/composer/test_bedrock_live_smoke.py):
// skipped entirely unless ELSPETH_RUN_COMPOSER_LIVE=1. No local LLM key is
// needed — the deployed backend holds the planner credential; the run spends
// real provider tokens, which is the point.
//
// Invocation (dev server):
//   STAGING_BASE_URL=https://elspeth.foundryside.dev \
//   PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
//   STAGING_USERNAME=... STAGING_PASSWORD=... \
//   ELSPETH_RUN_COMPOSER_LIVE=1 \
//   ELSPETH_LIVE_OUTPUTS_DIR=<server data_dir>/outputs \
//   npx playwright test --config=playwright.staging.config.ts composer-guided-live --retries=0
//
// On failure the guided session is deliberately KEPT (not deleted): its
// sessions.db chat_messages transcript (writer=compose_loop) and Landscape
// rows are the diagnostic record. The failing session id is printed in the
// test output as "GUIDED-LIVE SESSION <id>".
// ============================================================================

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

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
// Reuse the pinned parity colour corpus: 10 rows of color_name,hex.
const FIXTURE_CSV = resolve(REPO_ROOT, "evals", "composer-parity", "fixtures", "two_llm_colour.csv");
const BLOB_FILENAME = "colours.csv";

const LIVE = process.env.ELSPETH_RUN_COMPOSER_LIVE === "1";
// Where the deployed server writes sink outputs (its data_dir/outputs). The
// dev server shares this filesystem; remote deployments can point this at a
// mount or skip the file assertion by running against the dev box instead.
const OUTPUTS_DIR = process.env.ELSPETH_LIVE_OUTPUTS_DIR;

// Every colour row the fixture contains must appear in the output.
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

test.describe("composer guided live — the colour test (staging)", () => {
  test.skip(
    !LIVE,
    "set ELSPETH_RUN_COMPOSER_LIVE=1 to run the live guided colour test (real planner spend)",
  );

  // Planner turns are real LLM completions; the run itself is deterministic
  // pass-through. Generous but bounded.
  test.setTimeout(8 * 60_000);

  test("guided wizard authors and runs the colour pass-through", async ({ page }) => {
    const csv = readFileSync(FIXTURE_CSV, "utf-8");
    const expectedRows = fixtureRows();
    // Unique per invocation so consecutive proof runs never collide on the
    // sink path and each output file is attributable to its run.
    const sinkFilename = `colour_output_${Date.now()}.json`;

    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    let sessionId: string | undefined;
    let passed = false;
    try {
      // ── Out-of-band setup: session + colour CSV blob ─────────────────────
      // Upload FIRST, then pick CSV in the wizard: the upload fast-path binds
      // the blob and auto-configures the observed schema. (Choosing CSV before
      // uploading routes into a SCHEMA_FORM dead end — the documented trap.)
      const session = await createSession(ctx, "guided-live-colour-test");
      sessionId = session.id;
      console.log(`GUIDED-LIVE SESSION ${sessionId}`);
      await uploadBlob(ctx, sessionId, BLOB_FILENAME, csv);

      const composer = new ComposerPage(page);
      await composer.goto(sessionId);
      await composer.waitForChatReady();
      await page.getByRole("button", { name: "Switch to guided" }).click();
      await expect(page.getByLabel(/guided composer/i)).toBeVisible();

      // ── Step 1 source: SINGLE_SELECT — CSV binds the uploaded blob ───────
      await page.getByRole("button", { name: "CSV", exact: true }).click();

      // ── Step 1 source: SCHEMA_FORM summary — upload fast-path auto-config
      // Choosing CSV with the blob already uploaded lands on a summary card
      // (observed schema + blob-bound path) with Edit/Continue. The observed
      // auto-config is exactly what we want — accept it.
      await expect(page.getByRole("button", { name: "Continue", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Continue", exact: true }).click();

      // ── Step 1 source: confirm columns, then finish sources ──────────────
      // The turn order within a stage is server-authored: an
      // INSPECT_AND_CONFIRM ("Looks right") may or may not be interposed
      // before the sources COMPONENT_REVIEW. Handle both explicitly.
      const looksRight = page.getByRole("button", { name: "Looks right" });
      const finishSources = page.getByRole("button", { name: "Finish sources" });
      await expect(looksRight.or(finishSources)).toBeVisible();
      if (await looksRight.isVisible().catch(() => false)) {
        await looksRight.click();
      }
      await expect(finishSources).toBeEnabled();
      await finishSources.click();

      // ── Step 2 sink: SINGLE_SELECT — JSON output ─────────────────────────
      await page.getByRole("button", { name: "JSON", exact: true }).click();

      // ── Step 2 sink: SCHEMA_FORM — path, format, and the explicit
      // write-safety choices the runnable file-sink contract requires
      // (mode + collision_policy are operator decisions the engine refuses
      // to default; the form must collect them or planning wedges — L4).
      await page.getByRole("button", { name: "Edit", exact: true }).click();
      await page.getByLabel(/^path/i).fill(sinkFilename);
      await page.getByLabel(/^format$/i).selectOption("json");
      await page.getByLabel(/^mode$/i).selectOption("write");
      await page.getByLabel(/collision.?policy/i).selectOption("auto_increment");
      await expect(page.getByRole("button", { name: "Continue", exact: true })).toBeEnabled();
      await page.getByRole("button", { name: "Continue", exact: true }).click();

      // ── Step 2 sink: MULTI_SELECT_WITH_CUSTOM — pass all fields through ──
      await page
        .getByRole("button", { name: "Let source decide (pass all fields through)" })
        .click();

      // ── Step 2 sink: COMPONENT_REVIEW — finish outputs ───────────────────
      await expect(page.getByRole("button", { name: "Finish outputs" })).toBeEnabled();
      await page.getByRole("button", { name: "Finish outputs" }).click();

      // ── THE FRONTIER: Output→Transforms fires the real planner ───────────
      // bind_guided_reviewed_components + plan_guided_pipeline live here; the
      // whole 7.1 bug chain has surfaced at this transition. Race the healthy
      // outcome against the failure surfaces so a terminal 500 fails the spec
      // in seconds with the server's own words, not after a blind timeout.
      const transformsHeading = page.getByRole("heading", {
        name: "Review the transform stages that turn source data into the output.",
      });
      const failureSurface = page
        .getByText(
          /operation failed|integrity check|terminal failure|does not satisfy the current turn contract|encountered an error/i,
        )
        .first();
      await expect(transformsHeading.or(failureSurface)).toBeVisible({ timeout: 5 * 60_000 });
      await expect(
        failureSurface,
        "guided planner surfaced a terminal failure at Output→Transforms " +
          "(the bug-chain frontier) — inspect sessions.db chat_messages and " +
          `the server log for session ${sessionId}`,
      ).toHaveCount(0);
      await expect(transformsHeading).toBeVisible();

      // ── Step 3 transforms: the planner proposes the whole pipeline ───────
      // Pass-through needs no transform chain: the proposal (Components +
      // Routes, source→output with discard routes) is presented directly.
      // Accept it into wiring review.
      const proposal = page.getByRole("article", { name: "Review pipeline proposal" });
      await expect(proposal).toBeVisible({ timeout: 60_000 });
      const reviewButton = proposal.getByRole("button", { name: "Review wiring" });
      await expect(reviewButton).toBeEnabled();
      await reviewButton.click();

      // ── Step 4 wire: confirm ─────────────────────────────────────────────
      const confirmWiring = page.getByRole("button", { name: "Confirm wiring", exact: true });
      await expect(confirmWiring).toBeEnabled();
      await confirmWiring.click();

      // ── Run ──────────────────────────────────────────────────────────────
      // Confirm commits the composition; the session lands on "Pipeline
      // ready" with the guided run affordance.
      await expect(page.getByRole("heading", { name: "Pipeline ready" })).toBeVisible({
        timeout: 60_000,
      });
      const runButton = page.getByRole("button", { name: "Run pipeline" }).first();
      await expect(runButton).toBeEnabled();
      await runButton.click();

      // Credential-egress confirmation: the run leaves the composer and uses
      // stored credentials, so an alertdialog interposes before execution.
      const runDialog = page.getByRole("alertdialog", { name: "Run pipeline?" });
      await expect(runDialog).toBeVisible();
      await runDialog.getByRole("button", { name: "Run pipeline" }).click();

      // ── The colour contract: output holds every fixture row ──────────────
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
          timeout: 4 * 60_000,
          message: `waiting for the pipeline run to write ${outputPath}`,
        })
        .toBe(true);
      await expect(
        page.getByText(/run failed|execution failed|pipeline failed/i).first(),
      ).toHaveCount(0);
      const output = JSON.parse(readFileSync(outputPath, "utf-8")) as Record<string, unknown>[];
      expect(output, `output ${outputPath} must hold all ${expectedRows.length} rows`).toHaveLength(
        expectedRows.length,
      );
      for (const row of expectedRows) {
        expect(
          output.some((o) => o.color_name === row.color_name && o.hex === row.hex),
          `output is missing colour row ${row.color_name} (${row.hex})`,
        ).toBe(true);
      }
      passed = true;
    } finally {
      // Keep failed sessions: their transcript IS the diagnostic artifact.
      // On success the run may still be finalizing server-side (the output
      // file lands before the run settles), so tolerate the active-run 409
      // briefly and otherwise leave the session rather than fail a green run.
      if (passed && sessionId !== undefined) {
        for (let attempt = 0; attempt < 6; attempt += 1) {
          try {
            await deleteSession(ctx, sessionId);
            break;
          } catch (error) {
            if (attempt === 5 || !String(error).includes("409")) {
              console.log(`GUIDED-LIVE cleanup skipped (run still settling): ${sessionId}`);
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
