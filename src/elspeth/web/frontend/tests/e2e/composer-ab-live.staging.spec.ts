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

import { expect, test, type APIRequestContext } from "@playwright/test";

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

// Minimal committed-state node shape for the topology assertion (mirrors
// NodeSpec in src/types/index.ts — only the discriminating fields).
interface CommittedNode {
  id: string;
  node_type: string;
  plugin: string | null;
  fork_to?: string[] | null;
}

// OPERATOR RULING (2026-07-22): the two-LLM A/B test exists to exercise
// fork/coalesce. The committed composition MUST hold a real fork (>=1 gate
// whose fork_to names 2 branches), EXACTLY 2 separate llm transform nodes
// ("llm" is a plugin on transform nodes — node_type is never "llm"), and
// exactly 1 coalesce merging the branches back to one row per colour. A
// single llm node with a queries map is the forbidden shortcut (live run 15
// took it and was ruled invalid) — it must fail HERE, before pipeline spend,
// with the actual node inventory in the failure message.
async function assertAbForkCoalesceTopology(
  ctx: APIRequestContext,
  sessionId: string,
): Promise<void> {
  const resp = await ctx.get(`/api/sessions/${sessionId}/state`);
  expect(
    resp.ok(),
    `GET /api/sessions/${sessionId}/state failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
  ).toBe(true);
  const state = (await resp.json()) as { nodes: CommittedNode[] };
  const inventory =
    state.nodes
      .map(
        (n) =>
          `${n.id}(node_type=${n.node_type}, plugin=${n.plugin ?? "null"}${
            n.fork_to ? `, fork_to=[${n.fork_to.join(", ")}]` : ""
          })`,
      )
      .join("; ") || "<no nodes>";
  const forkGates = state.nodes.filter(
    (n) => n.node_type === "gate" && (n.fork_to?.length ?? 0) === 2,
  );
  const llmNodes = state.nodes.filter((n) => n.plugin === "llm");
  const coalesceNodes = state.nodes.filter((n) => n.node_type === "coalesce");
  expect(
    forkGates.length,
    `A/B topology requires >=1 gate node with a 2-branch fork_to (the fork); committed nodes: ${inventory}`,
  ).toBeGreaterThanOrEqual(1);
  expect(
    llmNodes.length,
    `A/B topology requires EXACTLY 2 separate llm transform nodes (a single llm node with a queries map is the forbidden shortcut); committed nodes: ${inventory}`,
  ).toBe(2);
  expect(
    coalesceNodes.length,
    `A/B topology requires exactly 1 coalesce node merging the branches; committed nodes: ${inventory}`,
  ).toBe(1);
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

      // Hydration race: an API-created session sometimes lands on the guided
      // surface (observed runs 5/12/16 — the freeform transcript then never
      // renders and the success poll starves while the backend composes
      // fine). Normalize deterministically via the user-visible escape.
      const exitToFreeform = page.getByRole("button", { name: "Exit to freeform" });
      if (await exitToFreeform.isVisible({ timeout: 3_000 }).catch(() => false)) {
        await exitToFreeform.click();
        await composer.waitForChatReady();
      }

      // ── One outcome-stated request; the planner owns the design ──────────
      const request =
        `I uploaded ${BLOB_FILENAME} (color_name,hex — ${expectedRows.length} rows). ` +
        `Build an A/B assessment pipeline: read the CSV, then fork each row into TWO ` +
        `separate LLM transform nodes with different prompts — variant A: describe the ` +
        `emotional tone of the colour in one short phrase; variant B: suggest one design ` +
        `usage context for the colour. Join the branches with a coalesce that merges both ` +
        `branches back into a single row per colour holding color_name, hex, tone (from A) ` +
        `and usage (from B). Do not use a single LLM node with a queries map — the two ` +
        `assessments must be two separate LLM nodes. Write all ${expectedRows.length} ` +
        `merged rows to a JSON file named ${sinkFilename}.`;
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

      // ── Resolve surfaced interpretation reviews (native cards) ───────────
      // Settlement surfaces one pending review event per authored LLM prompt.
      // The run gate reads the CLIENT pending-review store, which only the
      // native Accept cards clear — an out-of-band API resolve leaves the
      // Run button gated (the server resolves, the SPA never refreshes).
      // Drive the visible cards exactly as a user would: prompt cards are
      // two-stage (View prompt reveals, the same button then approves).
      const gatedRunButton = page.getByRole("button", { name: "Run pipeline" }).first();
      const ackDeadline = Date.now() + 4 * 60_000;
      while (Date.now() < ackDeadline && !(await gatedRunButton.isEnabled().catch(() => false))) {
        // Prompt cards are two-stage: click 1 is "View prompt", after which the
        // same control's accessible name becomes "Approve the LLM prompt
        // template" — unanchored matching covers both stages, and the loop's
        // next pass performs the second click. Non-prompt cards are a single
        // "Acknowledge".
        const cardButton = page
          .getByRole("button", { name: /view prompt|approve|acknowledge/i })
          .first();
        if (await cardButton.isVisible().catch(() => false)) {
          await cardButton.click().catch(() => {});
          await page.waitForTimeout(250);
        } else {
          await page.waitForTimeout(1_000);
        }
      }

      const runButton = page.getByRole("button", { name: "Run pipeline" }).first();
      // The SPA's run gate can lag the server after card resolutions (the
      // resolved state is durable — interpretation_resolve versions — but the
      // client store does not refresh it). A reload is the user-plausible
      // recovery and re-hydrates everything server-authoritative.
      if (!(await runButton.isEnabled().catch(() => false))) {
        await page.reload();
        await page.waitForLoadState("networkidle");
        // Fresh hydration can land an API-created session on the guided
        // surface; the user-visible escape restores the freeform composer
        // with the committed pipeline intact.
        const exitToFreeform = page.getByRole("button", { name: "Exit to freeform" });
        if (await exitToFreeform.isVisible().catch(() => false)) {
          await exitToFreeform.click();
        }
      }
      await expect(runButton).toBeEnabled({ timeout: 120_000 });

      // ── OPERATOR RULING: committed topology must be fork → 2 llm → coalesce
      // Fail fast on the forbidden single-llm queries-map shortcut BEFORE
      // spending pipeline tokens on a run that would be ruled invalid anyway.
      await assertAbForkCoalesceTopology(ctx, sessionId);

      await runButton.click();
      const runDialog = page.getByRole("alertdialog", { name: "Run pipeline?" });
      await expect(runDialog).toBeVisible();
      await runDialog.getByRole("button", { name: "Run pipeline" }).click();

      // Fan-out cost guard (428 ExecutionFanoutGuardRequired): a forking
      // pipeline multiplies per-row LLM calls, so a second confirm dialog
      // ("Review LLM provider calls") interposes with an ack token. Confirm
      // it when it appears — acknowledging the spend is this spec's point.
      const fanoutDialog = page.getByRole("alertdialog", { name: "Review LLM provider calls" });
      if (await fanoutDialog.isVisible({ timeout: 15_000 }).catch(() => false)) {
        await fanoutDialog.getByRole("button", { name: "Execute" }).click();
      }

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
