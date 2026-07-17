// LLM-provider-schema spec — catalog must surface the operator-profiled
// public contract without exposing private provider configuration.
//
// Targets epic: elspeth-e1ab67e55a
// Targets sub-issue: elspeth-dcf12c061b (open, P2).
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
// gap inline. When the bug fixes land, delete the corresponding flag and
// the `test.skip` call.

import { expect, test } from "@playwright/test";
import { authedContext, createSession, deleteSession, tokenFromStorageState } from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

// State-seed gap flag — flip to `true` once the add-node-without-LLM
// affordance lands. PluginCard's "Use in pipeline" button was deleted per
// OD-C (reference-only catalog), and no successor add-node UX exists yet.
// Until then the test cannot drive step 2 (add llm node to pipeline), so
// the validation-error assertions cannot run.
const hasStateSeed = false;

test.describe("llm-provider-schema — catalog must enforce the operator-profile contract", () => {
  test("llm transform schema exposes an operator profile without private provider fields", async ({ page }) => {
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
      const catalog = page.getByRole("dialog", { name: /plugin catalog/i });
      await expect(catalog).toBeVisible();

      // 2. Switch to Transforms tab and expand the llm transform schema.
      await catalog.getByRole("tab", { name: /transforms/i }).click();
      const schemaResponsePromise = page.waitForResponse(
        (response) =>
          response.url().endsWith("/api/catalog/transforms/llm/schema") &&
          response.ok(),
      );
      await catalog.getByRole("button", { name: /^schema for llm$/i }).click();
      const schemaResponse = await schemaResponsePromise;
      const schema = (await schemaResponse.json()) as {
        json_schema: {
          properties: { profile: { enum: string[] } };
        };
      };

      // 3. The public schema exposes only the operator-owned alias and safe
      //    authoring fields. Provider bindings and credentials stay private.
      expect(schema.json_schema.properties.profile.enum).toEqual(["e2e-bedrock"]);
      await expect(catalog.getByText("profile", { exact: true })).toBeVisible();
      await expect(catalog.getByText("Operator-approved LLM profile alias", { exact: true })).toBeVisible();
      for (const privateField of ["provider", "model", "deployment_name", "api_key"]) {
        await expect(catalog.getByText(privateField, { exact: true })).toHaveCount(0);
      }
    } finally {
      await deleteSession(ctx, session.id);
      await ctx.dispose();
    }
  });

  test("llm node without a profile surfaces a Stage-1 error", async ({ page }) => {
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

      // 2. Add the llm node to the pipeline without selecting a profile.
      //    "Use in pipeline" was removed per OD-C; the add-node successor
      //    affordance lands with the state-seed work. Replace this section
      //    with the real add-node sequence once the affordance is in place.

      // 3. Assert the validation dot reads "Validation failed".
      await expect(page.getByText(/validation failed/i)).toBeVisible();

      // 4. Assert validation banner names the missing public profile field.
      await expect(page.getByText(/profile/i)).toBeVisible();
    } finally {
      await deleteSession(ctx, session.id);
      await ctx.dispose();
    }
  });
});
