// E2E spec: guided-mode wizard recipe-match happy path.
//
// Targets the demo SLA defined in §10.3 of the implementation plan
// (2026-05-11-composer-guided-mode.md:4619-4621, corrected at df5306cf):
//   ≤9 user clicks to reach CompletionSummary via the recipe-match path.
//   <30s wall-clock (recipe match is deterministic — zero LLM calls).
//
// ── FIXME — BLOCKED (Phase 9, 2026-05-12) ───────────────────────────────────
//
// This spec was run against the live backend during Phase 9 Task 9.1 and
// verified step-by-step using Playwright traces and server logs. The wizard
// UI advances correctly through csv→SCHEMA_FORM→json→SCHEMA_FORM→
// MULTI_SELECT_WITH_CUSTOM, but the RECIPE_OFFER "Apply recipe" step returns
// HTTP 400.
//
// Root cause (Gap 5 below): the guided wizard's Step 1 SchemaForm path writes
// the source via _execute_set_source (handle_step_1_source in steps.py:83),
// which stores options={path, schema, on_validation_failure} with NO blob_id
// key.  The recipe slot resolver _classify_slot_resolver (recipe_match.py:133)
// reads `source.options.get("blob_id", "")` → returns "".  The recipe builder
// _build_classify_recipe (recipes.py:250) puts `"blob_id": ""` into the source
// args.  _execute_set_pipeline then calls _resolve_source_blob(blob_id=""),
// which fails: "Blob '' not found." → HTTP 400.
//
// The recipe-match path is structurally unreachable when the source is set via
// SchemaForm.  Fixing it requires one of:
//   (a) _classify_slot_resolver also checks source.options["blob_ref"] (set by
//       _execute_set_source_from_blob but not by _execute_set_source);
//   (b) handle_step_1_source detects blob-storage paths and writes blob_ref
//       alongside path;
//   (c) _build_classify_recipe treats blob_id="" as None and falls back to the
//       already-committed source path in state.source.options.
// All three options require backend design review. Filed as observation for
// triage (see below).
//
// Gaps discovered (in order encountered during the Phase 9 run):
//
//   Gap 1 — startGuided not wired to any UI entry point:
//     sessionStore.createSession and selectSession both set guidedSession: null.
//     startGuided() is implemented in the store but never called from any
//     component. ChatPanel's guided-mode discriminator never fires. Fix applied
//     in sessionStore.ts (both createSession and selectSession now call
//     void get().startGuided(id)) — this is the minimum needed for the wizard
//     to render. See observation filed as elspeth-obs-PLACEHOLDER-gap1.
//
//   Gap 2 — S2 path allowlist blocks raw /tmp paths:
//     _validate_source_path (tools.py:2086) rejects paths outside data_dir/blobs/.
//     Test must use the blob's storage path (service.py:207-211 format:
//     data_dir/blobs/{session_id}/{blob_id}_{filename}).  Worked around in spec
//     by constructing the path from known session_id and blob.id.
//
//   Gap 3 — on_validation_failure is required in source SchemaForm:
//     SourceDataConfig.on_validation_failure (config_base.py:358) has no default
//     and appears in the REQUIRED section of the SchemaForm.  The test must fill
//     it with "discard" to enable Continue.
//
//   Gap 4 — collision_policy required in composer mode but hidden behind
//     Show advanced:
//     validate_composer_file_sink_collision_policy (tools.py:2317) rejects null
//     collision_policy when data_dir is not None.  The field is optional in the
//     JSON schema (default null) and rendered behind "Show advanced".  The test
//     must expand the section and set "auto_increment".  Adds 1 to click budget.
//
//   Gap 5 — recipe-match structurally unreachable (blocks Apply recipe → 400):
//     See root cause above.  THIS IS THE HARD BLOCKER.  No in-test workaround
//     exists without modifying the backend.
//
// ── Observations filed ───────────────────────────────────────────────────────
//
// Observations will be filed by the Phase 9 controller after reviewing this
// spec.  Placeholder IDs below; update when observation IDs are assigned.
//
//   elspeth-obs-PLACEHOLDER-gap1: startGuided not wired to UI entry point
//   elspeth-obs-PLACEHOLDER-gap5: recipe-match unreachable via SchemaForm path
//     (source_blob_id always "" for SCHEMA_FORM-set sources)
//
// ── What the spec verifies today ─────────────────────────────────────────────
//
// The smoke spec (smoke.spec.ts) already verifies session CRUD and auth.
// The guided spec WOULD verify the full recipe-match flow once Gap 5 is fixed.
//
// The partial steps 1-4 (csv chip → source schema → json chip → sink schema)
// were verified to work correctly during Phase 9 testing:
//   POST /guided/respond 200 OK ×4 (one per step)
//   POST /guided/respond 400 Bad Request (Apply recipe fails, Gap 5)
// The test is written as fixme so that the step-by-step logic and commentary
// are preserved for the developer who fixes Gap 5.

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

const BLOB_FILENAME = "playwright-orders.csv";

// Minimal CSV content: header + one data row.
const SAMPLE_CSV = "id,name,value\n1,widget,42\n";

// Worktree root: playwright.config.ts starts uvicorn from cwd="../../../.."
// relative to the frontend dir (= worktree root, 4 levels up from frontend/).
// This spec lives at tests/e2e/ (2 levels inside frontend/).
// Total: 6 levels up from this file's __dirname.
const WORKTREE_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../../../../",
);
const E2E_DATA_DIR = resolve(WORKTREE_ROOT, ".e2e-data");

// Construct the blob storage path.
// service.py:207-211: data_dir/blobs/{session_id}/{blob_id}_{filename}
function blobStoragePath(sessionId: string, blobId: string): string {
  return resolve(E2E_DATA_DIR, "blobs", sessionId, `${blobId}_${BLOB_FILENAME}`);
}

// Sink output path — must be under data_dir/outputs/ (paths.py:44).
const SINK_OUTPUT_PATH = resolve(E2E_DATA_DIR, "outputs", "playwright-guided-output.json");

test.describe("composer-guided — recipe-match happy path", () => {
  test.fixme(
    true,
    "BLOCKED: recipe-match path unreachable (Gap 5) — source set via SchemaForm " +
    "has no blob_id in options; _classify_slot_resolver returns source_blob_id=''; " +
    "_resolve_source_blob fails 'Blob not found' → HTTP 400 on Apply recipe. " +
    "Backend fix required (steps.py / recipe_match.py / recipes.py). " +
    "Full blocker analysis in spec file header.",
  );

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

        // Upload the seed CSV. We keep blob.id to construct the storage path
        // used in the source SchemaForm (Gap 2: S2 path allowlist).
        const blob = await uploadBlob(ctx, sessionId, BLOB_FILENAME, SAMPLE_CSV);

        // ── Navigate to the session ──────────────────────────────────────────
        // Auto-start fix (Gap 1): selectSession now calls startGuided() in
        // sessionStore.ts; the wizard renders on navigation.
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

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
        // collision_policy is REQUIRED in composer mode [Gap 4], hidden behind
        // "Show advanced".  We expand and set "auto_increment".
        await expect(page.getByLabel(/^path$/i)).toBeVisible();
        await page.getByLabel(/^path$/i).fill(SINK_OUTPUT_PATH);
        await page.getByRole("button", { name: /show advanced/i }).click();
        clicks++;
        await expect(page.getByLabel(/collision.?policy/i)).toBeVisible();
        await page.getByLabel(/collision.?policy/i).fill('"auto_increment"');
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

        // ── Step 2.5 RECIPE_OFFER — Apply recipe ──────────────────────────
        // Blocked by Gap 5: Apply recipe → HTTP 400 (source_blob_id = "").
        // This assertion is expected to fail until Gap 5 is fixed.
        await expect(
          page.getByRole("button", { name: "Apply recipe", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "Apply recipe", exact: true }).click();
        clicks++;

        // ── CompletionSummary terminal ─────────────────────────────────────
        await expect(
          page.getByRole("button", { name: "Save and exit", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "Save and exit", exact: true }).click();
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
