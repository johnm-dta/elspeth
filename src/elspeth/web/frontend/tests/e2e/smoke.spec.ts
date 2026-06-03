// Smoke spec — end-to-end proof of life for the Playwright harness.
//
// Verifies the full boot path:
//   1. globalSetup ran and wrote storageState with a valid auth_token
//      (otherwise the app redirects to /login and these tests fail).
//   2. Both webServer instances came up healthy on the Playwright-assigned
//      backend and frontend ports.
//   3. The frontend SPA loads, restores auth from localStorage, and renders
//      the composer empty state.
//   4. The backend's /api/sessions endpoint accepts the bearer token and
//      can create + delete a session.
//
// What this spec deliberately does NOT do:
//   - Drive the LLM compose loop (would cost money / be nondeterministic).
//   - Mutate composition state (composer tools are LLM-facing, not REST).
//   - Assert on visual layout or styling.
//
// The richer composer-correctness specs that target epic elspeth-e1ab67e55a
// live alongside this file as test.fixme() stubs (see ./topology.spec.ts
// etc.) and will be unblocked by either an LLM stub server or a direct
// state-mutation REST endpoint.

import { expect, test } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

test.describe("smoke — boot + auth + empty composer", () => {
  test("frontend boots and chat panel renders", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();

    // Empty-state copy from ChatPanel.tsx when there is no active session.
    await expect(
      page.getByText(
        /Use the session switcher to select a session or create a new one\./i,
      ),
    ).toBeVisible();
  });

  test("backend accepts authed token and round-trips a session", async ({
    page,
  }) => {
    // Read the same storageState the browser context loaded, then issue
    // an out-of-band REST request to confirm the bearer token works.
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);

    const ctx = await authedContext(token);
    try {
      const session = await createSession(ctx, "playwright-smoke");
      expect(session.id).toMatch(/^[a-zA-Z0-9_-]+$/);
      expect(session.title).toBe("playwright-smoke");
      await deleteSession(ctx, session.id);
    } finally {
      await ctx.dispose();
    }
  });

  test("composer URL with a session id navigates without error", async ({
    page,
  }) => {
    // Seed a session via API, then navigate the SPA to its hash route.
    // This proves the hash router resolves a session id without depending
    // on the header session-switcher UI.
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);

    const ctx = await authedContext(token);
    try {
      const session = await createSession(ctx, "playwright-smoke-hash");
      const composer = new ComposerPage(page);
      await composer.goto(session.id);
      await composer.waitForChatReady();
      // The chat panel renders when an active session is present —
      // (when no session, the empty-state copy from test 1 shows instead).
      await expect(page.getByLabel("Chat panel")).toBeVisible();
      await deleteSession(ctx, session.id);
    } finally {
      await ctx.dispose();
    }
  });
});
