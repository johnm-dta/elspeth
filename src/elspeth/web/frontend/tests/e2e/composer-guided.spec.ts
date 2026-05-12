// E2E spec: guided-mode wizard recipe-match happy path.
//
// Targets the demo SLA defined in §10.3 of the implementation plan
// (2026-05-11-composer-guided-mode.md:4619-4621, corrected at df5306cf):
//   ≤9 user clicks to reach CompletionSummary via the recipe-match path.
//   <30s wall-clock (recipe match is deterministic — zero LLM calls).
//
// ── Gap 5 fixed (2026-05-12, commit 74ea68eb) ────────────────────────────────
//
// This spec was previously blocked by Gap 5 (recipe-match path unreachable via
// SchemaForm).  The fix uses a combined approach (options a + b):
//
//   (a) recipe_match.py: _classify_slot_resolver / _split_threshold_slot_resolver
//       now read source.options["blob_ref"] (composer-canonical) instead of
//       source.options.get("blob_id", ""). Missing blob_ref → offensive crash
//       (state-machine invariant: resolver is only reached for blob-backed sources).
//
//   (b) steps.py / tools.py: handle_step_1_source now accepts session_engine +
//       session_id, looks up the blob by storage_path after _execute_set_source
//       commits, and injects blob_ref into SourceResolved.options when found.
//       This lets the guided SchemaForm path (where the user types the blob's
//       file path) populate blob_ref without switching to set_source_from_blob.
//
// Gap 5 VERIFIED FIXED: Playwright page snapshot after clicking "Apply recipe"
// shows source_blob_id: f2852ea6-9e19-4db1-a03d-4dba2b92b9da (real UUID) in
// the recipe_offer card. Steps 1–5 now return HTTP 200. The 400 is downstream.
//
// ── Gap 6 — recipe_offer missing unsatisfied-slot UX (blocks Apply recipe) ───
//
// The recipe_offer step still returns HTTP 400 at Apply recipe time.  This is a
// NEW blocker revealed after Gap 5 was fixed.
//
// Root cause: RecipeOfferPayload.slots carries only the pre-filled slots
// {source_blob_id, output_path, label_field}.  The RecipeOfferTurn widget echoes
// payload.slots unchanged in edited_values when "Apply recipe" is clicked.  But
// classify-rows-llm-jsonl declares three required slots without defaults:
//   - classifier_template (str)
//   - model (str)
//   - api_key_secret (str)
// validate_slots raises RecipeValidationError for each missing required slot →
// _execute_apply_pipeline_recipe returns failure → HTTP 400.
//
// The RecipeOfferPayload type (guided.ts:240-244) carries no slot-schema, so the
// frontend cannot know which slots need user input.  Both layers need changes:
//   (a) backend: include the recipe's slot specs (type, required, description) in
//       RecipeOfferPayload so the frontend can distinguish pre-filled from empty.
//   (b) frontend: RecipeOfferTurn renders editable form fields for unsatisfied
//       required slots (using payload.slot_specs[name].required + value absent).
//
// Gaps discovered during Phase 9 Task 9.1 (in order encountered):
//
//   Gap 1 — startGuided not wired to any UI entry point:
//     sessionStore.createSession and selectSession both set guidedSession: null.
//     startGuided() is implemented in the store but never called from any
//     component. ChatPanel's guided-mode discriminator never fires. Fix applied
//     in sessionStore.ts (both createSession and selectSession now call
//     void get().startGuided(id)) — this is the minimum needed for the wizard
//     to render. See observation filed as elspeth-obs-d3d0d7fa70.
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
//     FIXED in commit 74ea68eb (Phase 9 dispatch).
//
//   Gap 6 — recipe_offer missing unsatisfied-slot UX (blocks Apply recipe → 400):
//     See root cause above.  THIS IS THE CURRENT HARD BLOCKER.
//     Filed as elspeth-obs-<gap6-id> (see Observations section).
//
// ── Observations filed ───────────────────────────────────────────────────────
//
//   elspeth-obs-d3d0d7fa70: startGuided not wired to UI entry point (Gap 1)
//     Fixed in sessionStore.ts (createSession + selectSession now call startGuided).
//   elspeth-obs-a8a9bc010a: recipe-match unreachable via SchemaForm path (Gap 5)
//     Fixed in 74ea68eb (this Phase 9 dispatch).
//   elspeth-obs-f626607b13: RecipeOfferTurn missing editable form for
//     unsatisfied required slots; RecipeOfferPayload missing slot-schema (Gap 6).
//
// ── What the spec verifies today ─────────────────────────────────────────────
//
// Steps 1–5 (csv chip → source schema → json chip → sink schema → required
// fields → recipe_offer render) now return HTTP 200 (verified in this dispatch).
// The test is marked fixme because Apply recipe still returns HTTP 400 (Gap 6).
// All step-by-step logic is preserved so whoever fixes Gap 6 can un-fixme in
// one step without reconstructing the test.

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
    "BLOCKED: Apply recipe → HTTP 400 (Gap 6) — recipe_offer payload contains " +
    "only the pre-filled slots {source_blob_id, output_path, label_field}; the " +
    "RecipeOfferTurn widget echoes payload.slots unchanged; missing required slots " +
    "classifier_template / model / api_key_secret cause RecipeValidationError at " +
    "apply time.  Gap 5 (blob_ref lookup) is VERIFIED FIXED (page snapshot shows " +
    "source_blob_id: f2852ea6-...) — the recipe_offer now renders correctly. " +
    "Fix requires: (a) backend to include slot-schema in RecipeOfferPayload so the " +
    "frontend knows which slots need user input, and (b) RecipeOfferTurn editable " +
    "form fields for unsatisfied required slots. " +
    "Full analysis in spec file header."
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
        // Gap 5 FIXED: recipe_offer now renders with correct source_blob_id.
        // Apply recipe still returns HTTP 400 (Gap 6): required slots
        // classifier_template / model / api_key_secret are not pre-filled and
        // the RecipeOfferTurn widget has no editable form to supply them.
        // This assertion is expected to fail until Gap 6 is fixed.
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
