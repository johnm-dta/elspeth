// Modal flow spec — Graph modal, YAML modal, Catalog modal.
//
// Covers the riskiest modal surface: modal open/close via three entry
// points (SideRail click, Escape key, keyboard shortcut) plus hash-routed
// deep links for graph and yaml.
//
// Existing modal stubs (yaml-export-roundtrip.spec.ts, topology.spec.ts) are
// both marked test.fixme — they are blocked on a direct state-seed endpoint or
// LLM stub server and test a different surface (YAML round-trip correctness and
// topology validation parity). Neither exercises the modal open/close or
// keyboard affordances this spec covers; there is no overlap.
//
// Pattern: REST session creation via helpers/api.ts (same as smoke.spec.ts),
// then UI interaction. No LLM calls.

import { expect, test } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

// ── Shared afterEach cleanup ──────────────────────────────────────────────────
// Each test creates its own session; cleanup is out-of-band so a failing test
// does not accumulate orphaned sessions. The deleteSession helper tolerates
// 404 (session already gone).

test.describe("modal flows — Graph, YAML, Catalog", () => {
  // ── Graph modal ──────────────────────────────────────────────────────────

  test.describe("Graph modal", () => {
    test("Ctrl+Shift+G keyboard shortcut opens the Graph modal; Escape closes it", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-graph-open-close");
        sessionId = session.id;
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        // The GraphMiniView renders "No pipeline yet" when there is no
        // composition state. Click it expecting the OPEN_GRAPH_MODAL_EVENT
        // dispatch. The aria-label on GraphMiniView's empty-state div is not
        // a button so the modal won't open from the empty state — use the
        // keyboard shortcut path instead, which is always available.
        // (See GraphMiniView.tsx: the button is only rendered when state exists.)
        //
        // Use Ctrl+Shift+G as the trigger: it is always wired regardless of
        // composition state (App.tsx:154-163).
        await page.keyboard.press("Control+Shift+G");

        const dialog = page.getByRole("dialog", { name: /pipeline graph/i });
        await expect(dialog).toBeVisible();

        await page.keyboard.press("Escape");
        await expect(dialog).not.toBeVisible();
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });

    test("deep link /#/{id}/graph opens the Graph modal and rewrites hash to canonical", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-graph-deeplink");
        sessionId = session.id;

        // Navigate directly to the verb URL. useHashRouter dispatches
        // OPEN_GRAPH_MODAL_EVENT via queueMicrotask and then rewrites the hash.
        await page.goto(`/#/${sessionId}/graph`);
        await new ComposerPage(page).waitForChatReady();

        const dialog = page.getByRole("dialog", { name: /pipeline graph/i });
        await expect(dialog).toBeVisible();

        // Hash must be rewritten to canonical (verb fragment stripped).
        await expect(page).toHaveURL(new RegExp(`#/${sessionId}$`));
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });

    test("Ctrl+Shift+G keyboard shortcut opens the Graph modal", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-graph-shortcut");
        sessionId = session.id;
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        await page.keyboard.press("Control+Shift+G");

        await expect(
          page.getByRole("dialog", { name: /pipeline graph/i }),
        ).toBeVisible();
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });
  });

  // ── YAML modal ───────────────────────────────────────────────────────────

  test.describe("YAML modal", () => {
    test("SideRail Export YAML button opens the YAML modal; Escape closes it", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-yaml-open-close");
        sessionId = session.id;
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        // ExportYamlButton only renders when an activeSessionId is present
        // (ExportYamlButton.tsx:7). The session navigation sets that in the
        // store, so the button should be visible after waitForChatReady.
        const exportYamlBtn = page.getByRole("button", {
          name: /export yaml/i,
        });
        await expect(exportYamlBtn).toBeVisible();
        await exportYamlBtn.click();

        const dialog = page.getByRole("dialog", { name: /export yaml/i });
        await expect(dialog).toBeVisible();

        await page.keyboard.press("Escape");
        await expect(dialog).not.toBeVisible();
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });

    test("deep link /#/{id}/yaml opens the YAML modal and rewrites hash to canonical", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-yaml-deeplink");
        sessionId = session.id;

        await page.goto(`/#/${sessionId}/yaml`);
        await new ComposerPage(page).waitForChatReady();

        const dialog = page.getByRole("dialog", { name: /export yaml/i });
        await expect(dialog).toBeVisible();

        await expect(page).toHaveURL(new RegExp(`#/${sessionId}$`));
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });

    test("Ctrl+Shift+Y keyboard shortcut opens the YAML modal", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-yaml-shortcut");
        sessionId = session.id;
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        await page.keyboard.press("Control+Shift+Y");

        await expect(
          page.getByRole("dialog", { name: /export yaml/i }),
        ).toBeVisible();
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });
  });

  // ── Catalog modal ─────────────────────────────────────────────────────────
  // CatalogDrawer renders as a drawer (role="dialog" with name "Plugin Catalog")
  // opened by the OPEN_CATALOG_EVENT. No hash-routed deep link exists for
  // catalog (useHashRouter.ts ACTION_VERBS only contains "graph" and "yaml").
  // Deep-link assertion is skipped per the task brief.

  test.describe("Catalog modal", () => {
    test("Catalog (reference) button opens the Catalog drawer; Escape closes it", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-catalog-open-close");
        sessionId = session.id;
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        const catalogBtn = page.getByRole("button", {
          name: /catalog \(reference\)/i,
        });
        await expect(catalogBtn).toBeVisible();
        await catalogBtn.click();

        const drawer = page.getByRole("dialog", { name: /plugin catalog/i });
        await expect(drawer).toBeVisible();

        await page.keyboard.press("Escape");
        await expect(drawer).not.toBeVisible();
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });

    test("Ctrl+Shift+P keyboard shortcut opens the Catalog drawer", async ({
      page,
    }) => {
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);
      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "pw-3b-catalog-shortcut");
        sessionId = session.id;
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        await page.keyboard.press("Control+Shift+P");

        await expect(
          page.getByRole("dialog", { name: /plugin catalog/i }),
        ).toBeVisible();
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    });
  });
});
