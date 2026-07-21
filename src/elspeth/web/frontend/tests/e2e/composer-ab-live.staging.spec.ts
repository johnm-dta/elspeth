// ============================================================================
// composer-ab-live.staging.spec.ts — THE TWO-LLM A/B TEST, freeform surface
// (real LLM planner + real LLM pipeline run, live deployment).
//
// The full A/B contract: one colours CSV goes in, each row forks into TWO
// separate LLM assessments with different prompts (variant A: emotional tone,
// variant B: design usage), and the variants are reconciled into a single row
// per colour before landing in a JSON sink. This exercises the composer's
// hardest authoring shape (fork → two llm nodes → reconcile → sink) and the
// deployment's operator LLM profile end-to-end.
//
// Requires the deployment to have an LLM operator profile configured
// (ELSPETH_WEB__LLM_PROFILES) — without one the composer must honestly
// decline, which composer-decline UX coverage owns, not this spec.
//
// Same gating as the sibling colour specs: ELSPETH_RUN_COMPOSER_LIVE=1 +
// STAGING_* creds + ELSPETH_LIVE_OUTPUTS_DIR. Failed sessions are KEPT for
// forensics; the session id is printed as "AB-LIVE SESSION <id>".
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

test.describe("composer freeform live — the two-LLM A/B test (staging)", () => {
  test.skip(
    !LIVE,
    "set ELSPETH_RUN_COMPOSER_LIVE=1 to run the live A/B test (real planner + pipeline LLM spend)",
  );

  // Authoring alone is budgeted at up to 600s wall clock; the run adds two
  // LLM calls per row. Give the whole journey 25 minutes.
  test.setTimeout(25 * 60_000);

  test("freeform chat authors and runs the fork/reconcile A/B assessment", async ({ page }) => {
    const csv = readFileSync(FIXTURE_CSV, "utf-8");
    const expectedRows = fixtureRows();
    const sinkFilename = `colour_ab_${Date.now()}.json`;

    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    let sessionId: string | undefined;
    let passed = false;
    try {
      const session = await createSession(ctx, "ab-live-two-llm-test");
      sessionId = session.id;
      console.log(`AB-LIVE SESSION ${sessionId}`);
      await uploadBlob(ctx, sessionId, BLOB_FILENAME, csv);

      const composer = new ComposerPage(page);
      await composer.goto(sessionId);
      await composer.waitForChatReady();

      // ── One outcome-stated request; the planner owns the design ──────────
      const request =
        `I uploaded ${BLOB_FILENAME} (color_name,hex — ${expectedRows.length} rows). ` +
        `Build an A/B assessment pipeline: read the CSV, then fork each row into TWO separate ` +
        `LLM assessments with different prompts — variant A: describe the emotional tone of the ` +
        `colour in one short phrase; variant B: suggest one design usage context for the colour. ` +
        `Then reconcile the two variants back into a single row per colour holding color_name, ` +
        `hex, tone (from A) and usage (from B), and write all ${expectedRows.length} reconciled ` +
        `rows to a JSON file named ${sinkFilename}.`;
      const chat = page.getByRole("textbox", { name: "Message input" });
      await expect(chat).toBeEnabled();
      await chat.fill(request);
      await page.getByRole("button", { name: "Send message" }).click();

      // ── UX contract while the planner works ──────────────────────────────
      // Race the committed-proposal success signal against any compose failure
      // surface so a failed compose fails the test in seconds, not after the
      // full 11-minute success timeout. Either a specific error banner or the
      // Send affordance re-enabling (compose returned without a proposal) ends
      // the wait early.
      const successMsg = page.getByText(/prepared and validated the requested pipeline/i).first();
      const failureBanner = page
        .getByText(/unusable pipeline plan|composer_planner_failure|could not build|timed out|unavailable|run failed/i)
        .first();
      await expect
        .poll(
          async () => {
            if (await successMsg.isVisible().catch(() => false)) return "success";
            if (await failureBanner.isVisible().catch(() => false)) return "failure";
            return "pending";
          },
          { timeout: 11 * 60_000, intervals: [2_000] },
        )
        .not.toBe("pending");
      await expect(failureBanner, "compose surfaced a failure banner").toHaveCount(0);
      await expect(successMsg).toBeVisible();

      // ── Resolve surfaced interpretation reviews ──────────────────────────
      // Settlement surfaces one pending review event per authored LLM prompt
      // (and any other review sites). The run gate fails closed until every
      // event resolves — accept each as drafted via the same endpoint the UI
      // review card drives.
      const pendingResp = await ctx.get(`/api/sessions/${sessionId}/interpretations?status=pending`);
      const pending = (await pendingResp.json()) as { events: { event_id?: string; id?: string }[] };
      for (const event of pending.events) {
        const eventId = event.event_id ?? event.id;
        const resolveResp = await ctx.post(
          `/api/sessions/${sessionId}/interpretations/${eventId}/resolve`,
          { data: { choice: "accepted_as_drafted" } },
        );
        expect(resolveResp.ok(), `resolve ${eventId} failed: ${await resolveResp.text()}`).toBe(true);
      }

      const runButton = page.getByRole("button", { name: "Run pipeline" }).first();
      await expect(runButton).toBeEnabled({ timeout: 60_000 });

      await runButton.click();
      const runDialog = page.getByRole("alertdialog", { name: "Run pipeline?" });
      await expect(runDialog).toBeVisible();
      await runDialog.getByRole("button", { name: "Run pipeline" }).click();

      // ── The A/B contract: every row reconciled with both variants ────────
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
          message: `waiting for the A/B pipeline run to write ${outputPath}`,
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
      // Keep failed sessions; tolerate the run-still-settling 409 on success.
      if (passed && sessionId !== undefined) {
        for (let attempt = 0; attempt < 6; attempt += 1) {
          try {
            await deleteSession(ctx, sessionId);
            break;
          } catch (error) {
            if (attempt === 5 || !String(error).includes("409")) {
              console.log(`AB-LIVE cleanup skipped (run still settling): ${sessionId}`);
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
