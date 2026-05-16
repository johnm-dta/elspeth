// composer-preferences.spec.ts — Phase 1B Task 10.
//
// Exercises the four user journeys from plan 13 Task 9 Step 3 against the
// local Playwright webServer. Each test resets the account-level
// preferences row via REST before driving the UI, so the tests are
// order-independent.
//
// Mirrors the project's existing E2E discipline (see smoke.spec.ts):
//   - REST helpers for state setup (no "Testing Through the UI" antipattern).
//   - Page object for UI navigation.
//   - assumes globalSetup has populated storageState with a valid
//     auth_token, so we don't drive the login form here.
//
// Running against staging is the operator's Task 9 step. For staging
// the BACKEND_BASE_URL in helpers/api.ts would need to point at
// elspeth.foundryside.dev (and the storageState would need to carry a
// staging-issued token). This spec ships green against the local
// playwright environment.

import { expect, test, type APIRequestContext } from "@playwright/test";

import {
  authedContext,
  tokenFromStorageState,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

type ComposerMode = "guided" | "freeform";

interface PrefsPayload {
  default_mode: ComposerMode;
  banner_dismissed_at: string | null;
}

async function patchPreferences(
  ctx: APIRequestContext,
  body: Partial<PrefsPayload>,
): Promise<PrefsPayload> {
  const resp = await ctx.patch("/api/composer-preferences", { data: body });
  if (!resp.ok()) {
    throw new Error(
      `PATCH /api/composer-preferences failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  return (await resp.json()) as PrefsPayload;
}

async function getPreferences(ctx: APIRequestContext): Promise<PrefsPayload> {
  const resp = await ctx.get("/api/composer-preferences");
  if (!resp.ok()) {
    throw new Error(
      `GET /api/composer-preferences failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  return (await resp.json()) as PrefsPayload;
}

// Serial execution: every test in this file mutates the SAME
// account-level preferences row (the storageState carries a single
// e2e-tester user across all tests). Under the project default
// (fullyParallel: true), Test 3's settings-opt-out flip would race
// Test 1's reset-to-guided beforeEach, leading to a non-deterministic
// page state. Serial mode serialises the file's tests; other spec
// files remain parallel.
test.describe.configure({ mode: "serial" });

test.describe("composer preferences — default mode + opt-out journeys", () => {
  // Each test seeds preferences via REST then loads the SPA. Page reload is
  // necessary because the preferencesStore caches the bootstrap result.
  test.beforeEach(async ({ page }) => {
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);
    try {
      // Reset to a known baseline: guided default, banner not dismissed.
      await patchPreferences(ctx, {
        default_mode: "guided",
        banner_dismissed_at: null,
      });
    } finally {
      await ctx.dispose();
    }
  });

  test("Journey 1: new-user defaults to guided mode, no banner", async ({
    page,
  }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    // ComposerPage.createSession() waits for aria-label="Chat panel", which
    // only matches the freeform/empty body. In guided mode the body's
    // aria-label is "Guided composer" — so click + wait inline here.
    await page.getByRole("button", { name: "Create new session" }).click();
    await page
      .getByLabel(/guided composer/i)
      .waitFor({ state: "visible" });

    // Guided mode reveals the workflow stepper.
    await expect(page.getByLabel(/guided composer/i)).toBeVisible();

    // The default-changed banner is gated on freeform default; guided users
    // must not see it.
    await expect(
      page.getByRole("status").filter({ hasText: /freeform/i }),
    ).toHaveCount(0);
  });

  test("Journey 2: opted-out user lands in freeform with the banner visible", async ({
    page,
  }) => {
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);
    try {
      await patchPreferences(ctx, {
        default_mode: "freeform",
        banner_dismissed_at: null,
      });
    } finally {
      await ctx.dispose();
    }

    const composer = new ComposerPage(page);
    await composer.goto();
    await composer.createSession("journey-2");

    // Freeform chrome — guided body's aria-label must NOT be present.
    await expect(page.getByLabel(/guided composer/i)).toHaveCount(0);

    // Banner visible.
    const banner = page.getByRole("status").filter({ hasText: /freeform/i });
    await expect(banner).toBeVisible();

    // Dismiss persists. Synchronise on the PATCH response — the optimistic
    // UI flip fires before the network call completes, so an immediate GET
    // after click would race the in-flight PATCH and read stale state.
    const dismissPatchDone = page.waitForResponse(
      (response) =>
        response.url().includes("/api/composer-preferences") &&
        response.request().method() === "PATCH" &&
        response.status() === 200,
    );
    await banner.getByRole("button", { name: /got it|dismiss/i }).click();
    await dismissPatchDone;
    await expect(banner).toHaveCount(0);

    const ctx2 = await authedContext(token);
    try {
      const prefs = await getPreferences(ctx2);
      expect(prefs.banner_dismissed_at).not.toBeNull();
    } finally {
      await ctx2.dispose();
    }
  });

  test("Journey 3: settings opt-out flips default to freeform", async ({
    page,
  }) => {
    const composer = new ComposerPage(page);
    await composer.goto();

    await page.getByRole("button", { name: /account/i }).click();
    // UserMenu items are now plain <button> elements (the menu role
    // contract was dropped in the Phase 1B panel round-2 fix because we
    // don't implement the WAI-ARIA arrow-key contract). The item label
    // also changed from "Settings" to "Composer preferences" since the
    // pane only contains composer prefs today.
    await page
      .getByRole("button", { name: /composer preferences/i })
      .click();

    await expect(
      page.getByRole("dialog", { name: /composer preferences/i }),
    ).toBeVisible();
    // Same PATCH-race concern as Journey 2: synchronise on the response.
    const settingsPatchDone = page.waitForResponse(
      (response) =>
        response.url().includes("/api/composer-preferences") &&
        response.request().method() === "PATCH" &&
        response.status() === 200,
    );
    await page.getByLabel(/freeform/i).click();
    await settingsPatchDone;

    // Close the modal (Escape, which the panel listens for).
    await page.keyboard.press("Escape");

    // Verify the new default landed on the backend.
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);
    try {
      const prefs = await getPreferences(ctx);
      expect(prefs.default_mode).toBe("freeform");
    } finally {
      await ctx.dispose();
    }
  });

  test("Journey 4: inline opt-out from guided chrome flips default to freeform", async ({
    page,
  }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.getByRole("button", { name: "Create new session" }).click();
    await page
      .getByLabel(/guided composer/i)
      .waitFor({ state: "visible" });

    // Confirm we landed in guided mode.
    await expect(page.getByLabel(/guided composer/i)).toBeVisible();

    // The inline checkbox lives in the guided chrome below the wizard.
    const checkbox = page.getByLabel(
      /always start new sessions in freeform mode/i,
    );
    await expect(checkbox).not.toBeChecked();
    const inlinePatchDone = page.waitForResponse(
      (response) =>
        response.url().includes("/api/composer-preferences") &&
        response.request().method() === "PATCH" &&
        response.status() === 200,
    );
    await checkbox.check();
    await inlinePatchDone;

    // Verify the new default landed on the backend.
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);
    try {
      const prefs = await getPreferences(ctx);
      expect(prefs.default_mode).toBe("freeform");
    } finally {
      await ctx.dispose();
    }
  });
});
