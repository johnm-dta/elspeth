// E2E spec: guided-mode wizard source/output walk against the live local
// backend. The local Playwright backend intentionally has no LLM provider, so
// this stops at the transform step before any provider-dependent guided chat.
// Wire-stage behavior is covered by tutorial.spec.ts with a deterministic
// guided protocol fixture.

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

// Minimal CSV content: header + one data row. The "category" column lets us
// satisfy the classify recipe's classifier-keyword required-field predicate.
const SAMPLE_CSV = "id,name,category\n1,widget,a\n";

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
const SINK_OUTPUT_PATH = resolve(E2E_DATA_DIR, "outputs", "playwright-guided-output.jsonl");

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

test.describe("composer-guided — source/output live walk", () => {
  test(
    "guided demo: CSV source → JSONL output → transform chat step",
    async ({ page }) => {
      // ── Out-of-band setup ──────────────────────────────────────────────────
      // Create session + upload CSV blob via REST before navigating the SPA.
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);

      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "playwright-guided-demo");
        sessionId = session.id;
        await isolateAuditReadinessSideRail(page, sessionId);

        const blob = await uploadBlob(ctx, sessionId, BLOB_FILENAME, SAMPLE_CSV);

        // ── Navigate + enter guided mode ─────────────────────────────────────
        // "Switch to guided" resolves to the live/empty profile
        // (advisor_checkpoints=false) via GET /guided — the D13 opt-out path.
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();
        await page.getByRole("button", { name: "Switch to guided" }).click();
        await expect(page.getByLabel(/guided composer/i)).toBeVisible();

        // ── Step 1 source: SINGLE_SELECT — pick "csv" ──────────────────────
        await expect(
          page.getByRole("button", { name: "CSV", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "CSV", exact: true }).click();

        // ── Step 1 source: SCHEMA_FORM — schema, path, on_validation_failure
        const sourcePath = blobStoragePath(sessionId, blob.id);
        await page.getByRole("button", { name: "Edit", exact: true }).click();
        await expect(page.getByLabel(/^schema/i)).toBeVisible();
        await page.getByLabel(/^schema/i).fill('{"mode":"observed"}');
        await page.getByLabel(/^path/i).fill(sourcePath);
        await page.getByLabel(/on\s+validation\s+failure/i).fill("discard");
        await expect(
          page.getByRole("button", { name: "Continue", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Continue", exact: true }).click();

        // ── Step 2 sink: SINGLE_SELECT — pick "json" ───────────────────────
        await expect(
          page.getByRole("button", { name: "JSON", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "JSON", exact: true }).click();

        // ── Step 2 sink: SCHEMA_FORM — path + collision_policy + format + mode
        // The file sink requires `mode` set explicitly (write|append).
        await page.getByRole("button", { name: "Edit", exact: true }).click();
        await expect(page.getByLabel(/^schema/i)).toBeVisible();
        await page.getByLabel(/^schema/i).fill('{"mode":"observed"}');
        await page.getByLabel(/^path/i).fill(SINK_OUTPUT_PATH);
        await page.getByLabel(/collision.?policy/i).selectOption("auto_increment");
        await page.getByLabel(/^format$/i).selectOption("jsonl");
        await page.getByLabel(/^mode$/i).selectOption("write");
        await expect(
          page.getByRole("button", { name: "Continue", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Continue", exact: true }).click();

        // ── Step 2 required fields: MULTI_SELECT_WITH_CUSTOM ──────────────
        // "category" is already selected by default — enough to satisfy
        // _classify_predicate (recipe_match.py).
        await expect(page.getByText("category")).toBeVisible();
        await page.getByRole("button", { name: "Continue", exact: true }).click();

        // ── Step 3 transform chat: no provider call yet, controls visible ──
        await expect(
          page.getByRole("heading", {
            name: "Review the transform chain that turns source data into the output.",
          }),
        ).toBeVisible();
        await expect(page.getByRole("textbox", { name: "Message input" })).toBeEnabled();
        await expect(
          page.getByRole("button", { name: "Exit to freeform", exact: true }),
        ).toBeVisible();
        await expect(
          page.getByText("This pipeline will read your CSV and write a JSON file."),
        ).toBeVisible();
        await expect(
          page.getByText("Required fields: id, name, category"),
        ).toBeVisible();
        await expect(
          page.getByText(/Source commit failed|Chat panel encountered an error/i),
        ).toHaveCount(0);
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    },
  );
});
