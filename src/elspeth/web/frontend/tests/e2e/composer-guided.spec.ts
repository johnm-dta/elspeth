// E2E spec: guided-mode wizard recipe-match happy path.
//
// Targets the demo SLA defined in §10.3 of the implementation plan
// (2026-05-11-composer-guided-mode.md):
//   ≤9 user clicks to reach CompletionSummary via the recipe-match path.
//   <30s wall-clock (recipe match is deterministic — zero LLM calls).
//
// ── Gap 6 RESOLVED in this dispatch (Task 10.0) ──────────────────────────────
//
// RecipeOfferPayload now carries `unsatisfied_slots` — the schema for each
// required slot the resolver could not pre-fill (slot_type, description,
// required).  RecipeOfferTurn renders an inline editable form for these
// entries; the Apply button is disabled until every required slot has a
// non-empty value, and the typed values are merged into edited_values.slots
// before submission.  Filling those inputs is typing, not clicks — the SLA
// click budget is unchanged.
//
// For prior gap history (Gap 1 startGuided wiring, Gap 2 S2 path allowlist,
// Gap 3 on_validation_failure, Gap 4 collision_policy, Gap 5 blob_ref
// resolver) see git log on this file plus elspeth-obs-d3d0d7fa70 /
// elspeth-obs-a8a9bc010a / elspeth-obs-f626607b13.

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
  uploadBlob,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

const BLOB_FILENAME = "playwright-orders.csv";

// Minimal CSV content: header + one data row.
const SAMPLE_CSV = "id,name,value\n1,widget,42\n";

// Frontend root: playwright.config.ts passes an absolute .e2e-data path
// anchored to the frontend directory, and the backend stores uploaded blobs
// relative to that data_dir.
const FRONTEND_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const E2E_DATA_DIR = process.env.PLAYWRIGHT_E2E_DATA_DIR
  ? resolve(process.env.PLAYWRIGHT_E2E_DATA_DIR)
  : resolve(FRONTEND_ROOT, ".e2e-data");

// Construct the blob storage path.
// service.py:207-211: data_dir/blobs/{session_id}/{blob_id}_{filename}
function blobStoragePath(sessionId: string, blobId: string): string {
  return resolve(E2E_DATA_DIR, "blobs", sessionId, `${blobId}_${BLOB_FILENAME}`);
}

// Sink output path — must be under data_dir/outputs/ (paths.py:44).
const SINK_OUTPUT_PATH = resolve(E2E_DATA_DIR, "outputs", "playwright-guided-output.json");

async function isolateAuditReadinessSideRail(
  page: Page,
  sessionId: string,
): Promise<void> {
  await page.route(`**/api/sessions/${sessionId}/validate`, async (route) => {
    await route.fulfill({
      json: {
        is_valid: true,
        summary: "Guided demo pipeline validates.",
        checks: [],
        errors: [],
        warnings: [],
        semantic_contracts: [],
      },
    });
  });

  await page.route(`**/api/sessions/${sessionId}/audit-readiness`, async (route) => {
    await route.fulfill({
      json: {
        session_id: sessionId,
        composition_version: 1,
        checked_at: "2026-05-19T12:00:00Z",
        rows: [
          {
            id: "validation",
            label: "Validation",
            status: "ok",
            summary: "Guided demo pipeline validates.",
            detail: null,
            component_ids: [],
          },
          {
            id: "plugin_trust",
            label: "Plugin trust",
            status: "ok",
            summary: "Guided demo plugins are trusted.",
            detail: null,
            component_ids: [],
          },
          {
            id: "provenance",
            label: "Provenance",
            status: "not_applicable",
            summary: "No run provenance yet.",
            detail: null,
            component_ids: [],
          },
          {
            id: "retention",
            label: "Retention",
            status: "not_applicable",
            summary: "No run retention yet.",
            detail: null,
            component_ids: [],
          },
          {
            id: "llm_interpretations",
            label: "LLM interpretations",
            status: "not_applicable",
            summary: "No interpretation events.",
            detail: null,
            component_ids: [],
          },
          {
            id: "secrets",
            label: "Secrets",
            status: "not_applicable",
            summary: "No secret checks in the guided demo.",
            detail: null,
            component_ids: [],
          },
        ],
        validation_result: {
          is_valid: true,
          summary: "Guided demo pipeline validates.",
          checks: [],
          errors: [],
          warnings: [],
          semantic_contracts: [],
        },
      },
    });
  });
}

test.describe("composer-guided — recipe-match happy path", () => {
  test(
    "guided demo path: CSV → classify-rows-llm-jsonl (≤9 clicks, <30s)",
    async ({ page }) => {
      const start = Date.now();
      let clicks = 0;

      // ── Out-of-band setup ──────────────────────────────────────────────────
      // Create session + upload CSV blob via REST before navigating the SPA.
      // These REST calls are NOT counted as user clicks.
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);

      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "playwright-guided-demo");
        sessionId = session.id;
        await isolateAuditReadinessSideRail(page, sessionId);

        // Upload the seed CSV. We keep blob.id to construct the storage path
        // used in the source SchemaForm (Gap 2: S2 path allowlist).
        const blob = await uploadBlob(ctx, sessionId, BLOB_FILENAME, SAMPLE_CSV);

        // ── Navigate to the session ──────────────────────────────────────────
        // The default-freeform contract renders the chat composer after
        // selectSession(); enter guided mode explicitly so this test is not
        // affected by account-level default-mode preference changes in other
        // Playwright specs.
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();
        await page.getByRole("button", { name: "Switch to guided" }).click();
        clicks++;
        await expect(page.getByLabel(/guided composer/i)).toBeVisible();

        // ── Step 1 source: SINGLE_SELECT — pick "csv" ──────────────────────
        // Chip label = plugin.name verbatim (emitters.py:127).
        await expect(
          page.getByRole("button", { name: "csv", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "csv", exact: true }).click();
        clicks++;

        // ── Step 1 source: SCHEMA_FORM — fill required fields, Continue ──
        // CSV source has THREE required fields (config_base.py):
        //   schema:                json-fallback (prefilled {"mode":"observed"})
        //   path:                  text, no default (PathConfig:319)
        //   on_validation_failure: text, no default (SourceDataConfig:358-361) [Gap 3]
        //
        // Path must be under data_dir/blobs/ (S2, tools.py:2086-2108) [Gap 2].
        // We construct the blob storage path from session_id + blob.id.
        const sourcePath = blobStoragePath(sessionId, blob.id);
        await expect(page.getByLabel(/^schema$/i)).toBeVisible();
        await page.getByLabel(/^schema$/i).fill('{"mode":"observed"}');
        await expect(page.getByLabel(/^path$/i)).toBeVisible();
        await page.getByLabel(/^path$/i).fill(sourcePath);
        await expect(page.getByLabel(/on\s+validation\s+failure/i)).toBeVisible();
        await page.getByLabel(/on\s+validation\s+failure/i).fill("discard");

        await expect(
          page.getByRole("button", { name: "Continue", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Continue", exact: true }).click();
        clicks++;

        // ── Step 2 sink: SINGLE_SELECT — pick "json" ───────────────────────
        // _classify_predicate requires sink.outputs[0].plugin == "json"
        // (recipe_match.py:88-89); "jsonl" would fail the recipe match.
        await expect(
          page.getByRole("button", { name: "json", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "json", exact: true }).click();
        clicks++;

        // ── Step 2 sink: SCHEMA_FORM — fill path + collision_policy, Continue
        // JSON sink: REQUIRED path, OPTIONAL collision_policy.
        // collision_policy is REQUIRED in composer mode [Gap 4].
        await expect(page.getByLabel(/^schema$/i)).toBeVisible();
        await page.getByLabel(/^schema$/i).fill('{"mode":"observed"}');
        await expect(page.getByLabel(/^path$/i)).toBeVisible();
        await page.getByLabel(/^path$/i).fill(SINK_OUTPUT_PATH);
        await page.getByLabel(/collision.?policy/i).selectOption("auto_increment");
        await page.getByLabel(/^format$/i).selectOption("json");
        await expect(
          page.getByRole("button", { name: "Continue", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Continue", exact: true }).click();
        clicks++;

        // ── Step 2 required fields: MULTI_SELECT_WITH_CUSTOM ──────────────
        // Add "category" as a custom required field — satisfies
        // _classify_predicate (recipe_match.py:100, _CLASSIFY_KEYWORDS).
        const customInput = page.getByLabel("Custom field", { exact: true });
        await expect(customInput).toBeVisible();
        await customInput.fill("category");
        await expect(
          page.getByRole("button", { name: "Add", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Add", exact: true }).click();
        clicks++;
        await expect(page.getByText("category")).toBeVisible();
        await page.getByRole("button", { name: "Continue", exact: true }).click();
        clicks++;

        // ── Step 2.5 RECIPE_OFFER — fill unsatisfied slots, Apply recipe ──
        // The classify-rows-llm-jsonl recipe declares three required slots
        // that the resolver cannot pre-fill: classifier_template, model,
        // api_key_secret.  RecipeOfferTurn (Task 10.0 / Gap 6) renders an
        // inline editable form for these; Apply is disabled until every
        // required value is filled.  Typing into inputs is NOT counted as
        // clicks in the SLA budget.
        await expect(
          page.getByRole("button", { name: "Apply recipe", exact: true }),
        ).toBeVisible();
        await expect(page.getByLabel(/classifier_template/i)).toBeVisible();
        await page
          .getByLabel(/classifier_template/i)
          .fill("Classify {{ row['name'] }} as widget or gadget.");
        await page.getByLabel(/^model\b/i).fill("anthropic/claude-sonnet-4.6");
        // api_key_secret carries the NAME of an inventory secret_ref, not a
        // raw credential.  The input is type="text" (not password) by design.
        await page.getByLabel(/api_key_secret/i).fill("openrouter-api-key");
        await expect(
          page.getByRole("button", { name: "Apply recipe", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Apply recipe", exact: true }).click();
        clicks++;

        // ── CompletionSummary terminal ─────────────────────────────────────
        await expect(
          page.getByRole("button", { name: "Open freeform editor", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "Open freeform editor", exact: true }).click();
        clicks++;

        // ── Demo SLA assertions ────────────────────────────────────────────
        // ≤9 clicks (9 in this revised path due to Gap 4 adding Show advanced).
        expect(clicks).toBeLessThanOrEqual(9);
        expect(Date.now() - start).toBeLessThan(30_000);
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    },
  );
});
