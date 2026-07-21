// ============================================================================
// composer-freeform-live.staging.spec.ts — THE COLOUR TEST, freeform surface
// (real LLM planner, live deployment).
//
// Freeform sibling of composer-guided-live.staging.spec.ts: the user states
// the outcome in chat, the real planner authors and commits the pipeline
// through the freeform tool loop, and the run writes every colour row to the
// JSON output. Same gating (ELSPETH_RUN_COMPOSER_LIVE=1 + STAGING_* creds +
// ELSPETH_LIVE_OUTPUTS_DIR), same fixture, same ground-truth assertion.
//
// Failed sessions are KEPT: sessions.db chat_messages (writer=compose_loop)
// and Landscape rows are the diagnostic record. The session id is printed as
// "FREEFORM-LIVE SESSION <id>".
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
const FIXTURE_CSV = resolve(REPO_ROOT, "evals", "composer-parity", "fixtures", "two_llm_colour.csv");
const BLOB_FILENAME = "colours.csv";

const LIVE = process.env.ELSPETH_RUN_COMPOSER_LIVE === "1";
const OUTPUTS_DIR = process.env.ELSPETH_LIVE_OUTPUTS_DIR;

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

test.describe("composer freeform live — the colour test (staging)", () => {
  test.skip(
    !LIVE,
    "set ELSPETH_RUN_COMPOSER_LIVE=1 to run the live freeform colour test (real planner spend)",
  );

  test.setTimeout(8 * 60_000);

  test("freeform chat authors and runs the colour pass-through", async ({ page }) => {
    const csv = readFileSync(FIXTURE_CSV, "utf-8");
    const expectedRows = fixtureRows();
    const sinkFilename = `colour_freeform_${Date.now()}.json`;

    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    let sessionId: string | undefined;
    let passed = false;
    try {
      const session = await createSession(ctx, "freeform-live-colour-test");
      sessionId = session.id;
      console.log(`FREEFORM-LIVE SESSION ${sessionId}`);
      await uploadBlob(ctx, sessionId, BLOB_FILENAME, csv);

      const composer = new ComposerPage(page);
      await composer.goto(sessionId);
      await composer.waitForChatReady();

      // ── One outcome-stated request; the planner owns the rest ────────────
      const request =
        `I uploaded ${BLOB_FILENAME} (color_name,hex — ${expectedRows.length} rows). ` +
        `Build a runnable pipeline that reads it and writes every row unchanged to a ` +
        `JSON file named ${sinkFilename}. No transforms needed.`;
      const chat = page.getByRole("textbox", { name: "Message input" });
      await expect(chat).toBeEnabled();
      await chat.fill(request);
      await page.getByRole("button", { name: "Send message" }).click();

      // The freeform tool loop discovers, composes, validates, and commits;
      // the run affordance enables once the committed pipeline is valid.
      const runButton = page.getByRole("button", { name: "Run pipeline" }).first();
      await expect(runButton).toBeEnabled({ timeout: 5 * 60_000 });
      await runButton.click();

      // Credential-egress confirmation dialog (same surface as guided).
      const runDialog = page.getByRole("alertdialog", { name: "Run pipeline?" });
      await expect(runDialog).toBeVisible();
      await runDialog.getByRole("button", { name: "Run pipeline" }).click();

      // ── The colour contract: output holds every fixture row ──────────────
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
      // Keep failed sessions; tolerate the run-still-settling 409 on success.
      if (passed && sessionId !== undefined) {
        for (let attempt = 0; attempt < 6; attempt += 1) {
          try {
            await deleteSession(ctx, sessionId);
            break;
          } catch (error) {
            if (attempt === 5 || !String(error).includes("409")) {
              console.log(`FREEFORM-LIVE cleanup skipped (run still settling): ${sessionId}`);
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
