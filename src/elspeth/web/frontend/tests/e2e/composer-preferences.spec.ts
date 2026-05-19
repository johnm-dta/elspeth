// composer-preferences.spec.ts — Phase 1B Task 10.
//
// Exercises the four user journeys from plan 13 Task 9 Step 3 against the
// local Playwright webServer. Each test resets the account-level
// preferences row before driving the UI, so the tests are order-independent.
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

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

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

interface CurrentUserPayload {
  user_id: string;
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

async function getCurrentUserId(ctx: APIRequestContext): Promise<string> {
  const resp = await ctx.get("/api/auth/me");
  if (!resp.ok()) {
    throw new Error(
      `GET /api/auth/me failed (${resp.status()}): ${(await resp.text()).slice(0, 500)}`,
    );
  }
  return ((await resp.json()) as CurrentUserPayload).user_id;
}

function sessionDbPathForDirectCleanup(): string | null {
  const configured = process.env.PLAYWRIGHT_SESSION_DB_PATH;
  if (configured) return resolve(configured);

  const configuredDataDir = process.env.PLAYWRIGHT_E2E_DATA_DIR;
  if (configuredDataDir) {
    return resolve(configuredDataDir, "sessions.db");
  }

  // Local Playwright config starts the backend from the repo root, but passes
  // an absolute ELSPETH_WEB__data_dir anchored to the frontend directory.
  // Source-checkout staging on this host uses /home/john/elspeth/data/sessions.db;
  // other staging targets should set PLAYWRIGHT_SESSION_DB_PATH explicitly when
  // a reused account needs reset.
  if (process.env.PLAYWRIGHT_BACKEND_BASE_URL) {
    const sourceCheckoutStagingDb = "/home/john/elspeth/data/sessions.db";
    const stagingBase =
      process.env.STAGING_BASE_URL ??
      process.env.PLAYWRIGHT_BACKEND_BASE_URL;
    return stagingBase.includes("elspeth.foundryside.dev") &&
      existsSync(sourceCheckoutStagingDb)
      ? sourceCheckoutStagingDb
      : null;
  }
  return resolve(process.cwd(), ".e2e-data", "sessions.db");
}

async function clearBannerDismissalDirectly(
  ctx: APIRequestContext,
): Promise<boolean> {
  const dbPath = sessionDbPathForDirectCleanup();
  if (dbPath === null || !existsSync(dbPath)) return false;

  const userId = await getCurrentUserId(ctx);
  execFileSync(
    "python3",
    [
      "-c",
      [
        "import sqlite3, sys",
        "conn = sqlite3.connect(sys.argv[1])",
        "sql = 'UPDATE user_preferences SET banner_dismissed_at = NULL WHERE user_id = ?'",
        "conn.execute(sql, (sys.argv[2],))",
        "conn.commit()",
      ].join("; "),
      dbPath,
      userId,
    ],
    { stdio: "pipe" },
  );
  return true;
}

async function resetPreferences(
  ctx: APIRequestContext,
  defaultMode: ComposerMode,
): Promise<void> {
  // The backend PATCH contract intentionally treats JSON null like an absent
  // banner_dismissed_at field, so REST alone cannot un-dismiss the banner.
  await patchPreferences(ctx, { default_mode: defaultMode });
  const afterPatch = await getPreferences(ctx);
  if (afterPatch.banner_dismissed_at === null) return;

  if (!(await clearBannerDismissalDirectly(ctx))) {
    throw new Error(
      "composer-preferences E2E setup could not clear banner_dismissed_at. " +
        "Use a fresh test account or set PLAYWRIGHT_SESSION_DB_PATH to the sessions.db file.",
    );
  }

  const afterCleanup = await getPreferences(ctx);
  if (afterCleanup.banner_dismissed_at !== null) {
    throw new Error(
      "composer-preferences E2E setup ran direct cleanup, but banner_dismissed_at is still set.",
    );
  }
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
      await resetPreferences(ctx, "guided");
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
    await page.getByRole("button", { name: /session switcher/i }).click();
    await page.getByRole("menuitem", { name: "+ New session" }).click();
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
      await resetPreferences(ctx, "freeform");
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
    await page.getByRole("button", { name: /session switcher/i }).click();
    await page.getByRole("menuitem", { name: "+ New session" }).click();
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
