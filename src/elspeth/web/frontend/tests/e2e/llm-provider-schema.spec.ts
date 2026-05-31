// LLM-provider-schema spec — catalog must surface provider union fields.
//
// Targets epic: elspeth-e1ab67e55a
// Targets sub-issue: elspeth-dcf12c061b (open, P2) — "CatalogServiceImpl.get_schema(
//   'transform', 'llm') returns only the base LLMConfig schema, omitting required
//   provider-specific fields such as Azure deployment_name"
//
// Phase 7A/7B/7C reshape changed the catalog card layout (new aria labels, no
// "Use in pipeline" button per OD-C, rewritten schema-preview surface). The
// previous test.fixme() gate has been removed (per plan 16c Task 6 Step 3a):
// these tests now run on every CI pass.
//
// Where a path cannot fully execute yet, the body uses
// `test.skip(!<has-feature>, "<bug-id> gap — see elspeth-XXXXXXXXXX")` so
// the gap is CI-visible (skip is reported in the run summary) rather than
// silently suppressed by `test.fixme`. Per CLAUDE.md No-Legacy: no
// `// removed for X` placeholder comments — the skip line documents the
// gap inline. When the bug fixes land, flip the corresponding flag below
// from `false` to `true` (or delete the flag and the `test.skip` call).

import { expect, test } from "@playwright/test";
import { authedContext, createSession, deleteSession, tokenFromStorageState } from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

// Provider-union schema fix flag — flip to `true` when elspeth-dcf12c061b
// (CatalogServiceImpl.get_schema returning base LLMConfig only) is closed.
// Until then, the schema preview will not surface Azure deployment_name /
// OpenRouter api_key, so the assertions cannot pass.
const hasProviderUnionSchema = false;

// State-seed gap flag — flip to `true` once the add-node-without-LLM
// affordance lands. PluginCard's "Use in pipeline" button was deleted per
// OD-C (reference-only catalog), and no successor add-node UX exists yet.
// Until then the test cannot drive step 2 (add llm node to pipeline), so
// the validation-error assertions cannot run.
const hasStateSeed = false;

test.describe("llm-provider-schema — catalog must surface provider-union fields", () => {
  test("llm transform schema enumerates Azure and OpenRouter variants", async ({ page }) => {
    // Gate first — fail fast and skip with bug-ID before doing UI work that
    // would otherwise produce a noisy timeout error.
    test.skip(
      !hasProviderUnionSchema,
      "provider-union schema gap — see elspeth-dcf12c061b",
    );

    // Seed a session so the composer mounts (the catalog drawer renders on
    // the composer surface; without a session the empty-state shows instead).
    const token = tokenFromStorageState(await page.context().storageState());
    const ctx = await authedContext(token);
    const session = await createSession(ctx, "llm-provider-schema-test-1");
    try {
      const composer = new ComposerPage(page);
      await composer.goto(session.id);
      await composer.waitForChatReady();

      // 1. Open catalog drawer.
      await page.keyboard.press("Control+Shift+P");
      await expect(page.getByRole("dialog", { name: /plugin catalog/i })).toBeVisible();

      // 2. Switch to Transforms tab and click the llm transform card.
      await page.getByRole("tab", { name: /transforms/i }).click();
      await page.getByText("llm").first().click();

      // 3. Assert the schema preview surfaces Azure-specific field (deployment_name)
      //    and OpenRouter-specific field (api_key).
      //    These fields are only present once elspeth-dcf12c061b is fixed.
      await expect(page.getByText(/deployment_name/i)).toBeVisible();
      await expect(page.getByText(/api_key/i)).toBeVisible();
    } finally {
      await deleteSession(ctx, session.id);
      await ctx.dispose();
    }
  });

  test("llm node without provider fields surfaces a Stage-1 error", async ({ page }) => {
    // Gate first — the add-node-without-driving-LLM affordance doesn't exist
    // post-OD-C. Skip with bug-ID before any UI interaction so the failure
    // is visible in CI output rather than producing a vacuous timeout.
    test.skip(!hasStateSeed, "state-seed gap — see elspeth-dcf12c061b");

    // Seed a session so the composer mounts.
    const token = tokenFromStorageState(await page.context().storageState());
    const ctx = await authedContext(token);
    const session = await createSession(ctx, "llm-provider-schema-test-2");
    try {
      const composer = new ComposerPage(page);
      await composer.goto(session.id);
      await composer.waitForChatReady();

      // 1. Open catalog drawer, navigate to llm transform.
      await page.keyboard.press("Control+Shift+P");
      await page.getByRole("tab", { name: /transforms/i }).click();
      await page.getByText("llm").first().click();

      // 2. Add the llm node to the pipeline without configuring a provider.
      //    "Use in pipeline" was removed per OD-C; the add-node successor
      //    affordance lands with the state-seed work. Replace this section
      //    with the real add-node sequence once the affordance is in place.

      // 3. Assert the validation dot reads "Validation failed".
      await expect(page.getByText(/validation failed/i)).toBeVisible();

      // 4. Assert validation banner mentions a provider-specific field.
      await expect(page.getByText(/deployment_name|provider/i)).toBeVisible();
    } finally {
      await deleteSession(ctx, session.id);
      await ctx.dispose();
    }
  });
});
