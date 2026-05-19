// catalog-reshape.spec.ts — E2E demo-path gate for Phase 7A/7B/7C.
//
// Covers the catalog drawer as reshaped by the Phase 7 work:
//   - InlineChatSourceEntry always visible on Sources tab
//   - FilterChipStrip narrows the plugin list
//   - PluginCard renders reference content (no toolkit affordances)
//   - Alt+1-3 tab switching
//   - Clear-filters restores full list
//
// This spec is the PRIMARY gate for the catalog reshape demo path.
// Task 5 manual smoke is a secondary gate. If this spec is fixme'd in
// CI (e.g., during a flake spike), that is a demo-day risk: escalate
// rather than silently deferring.
//
// Playwright invocation:
//   cd src/elspeth/web/frontend && npx playwright test tests/e2e/catalog-reshape.spec.ts

import { expect, test } from "@playwright/test";
import { authedContext, createSession, deleteSession, tokenFromStorageState } from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

test.describe("catalog-reshape — Phase 7 demo path", () => {
  test("1: Open catalog drawer via Ctrl+Shift+P", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByRole("dialog", { name: /plugin catalog/i })).toBeVisible();
  });

  test("2: Drawer opens with Sources tab active and InlineChatSourceEntry visible", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    // Sources tab active by default.
    await expect(page.getByRole("tab", { name: /sources/i })).toHaveAttribute("aria-selected", "true");
    // Synthetic entry is a pinned affordance — always visible on Sources.
    await expect(page.getByText(/inline data from chat/i)).toBeVisible();
  });

  test("3: Click a FilterChipStrip chip — plugin list narrows", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    // Wait for plugins to load.
    await expect(page.getByRole("button", { name: /^csv$/i })).toBeVisible();
    const initialCount = await page.locator(".plugin-card").count();
    await page.getByRole("button", { name: /^csv$/i }).click();
    const filteredCount = await page.locator(".plugin-card").count();
    expect(filteredCount).toBeLessThan(initialCount);
  });

  test("4: Click InlineChatSourceEntry 'Try it' — drawer closes and chat is prefilled", async ({ page }) => {
    // A session must exist before the chat ChatInput is mounted — the PREFILL
    // event has no receiver on the empty-session landing screen. Diagnose-and-
    // fix per plan Step 2: seed a session via the REST API (matches the pattern
    // used by smoke.spec.ts and schema-preview-parity.spec.ts), navigate the
    // composer at that session, then exercise the catalog flow.
    const token = tokenFromStorageState(await page.context().storageState());
    const ctx = await authedContext(token);
    const session = await createSession(ctx, "catalog-reshape-test-4");
    try {
      const composer = new ComposerPage(page);
      await composer.goto(session.id);
      await composer.waitForChatReady();
      await page.keyboard.press("Control+Shift+P");
      await expect(page.getByText(/inline data from chat/i)).toBeVisible();
      await page.getByRole("button", { name: /try it/i }).first().click();
      // Drawer closes.
      await expect(page.getByRole("dialog", { name: /plugin catalog/i })).not.toBeVisible();
      // Chat input is prefilled with a non-empty string.
      // The textarea exposes aria-label="Message input" (ChatInput.tsx) — the
      // plan's draft used /chat input/ which doesn't match the live component.
      // Aligned to production aria-label so the demo-path gate runs green.
      const chatInput = page.getByRole("textbox", { name: /message input/i });
      const value = await chatInput.inputValue();
      expect(value.trim().length).toBeGreaterThan(0);
    } finally {
      await deleteSession(ctx, session.id);
      await ctx.dispose();
    }
  });

  test("5: Switch to Transforms tab via Alt+2", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByRole("dialog", { name: /plugin catalog/i })).toBeVisible();
    await page.keyboard.press("Alt+2");
    await expect(page.getByRole("tab", { name: /transforms/i })).toHaveAttribute("aria-selected", "true");
  });

  test("6: Clear filters button restores full plugin list", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByRole("button", { name: /^csv$/i })).toBeVisible();
    await page.getByRole("button", { name: /^csv$/i }).click();
    const filteredCount = await page.locator(".plugin-card").count();
    await page.getByRole("button", { name: /clear filters/i }).click();
    const restoredCount = await page.locator(".plugin-card").count();
    expect(restoredCount).toBeGreaterThan(filteredCount);
  });

  test("7: PluginCard shows reference content — no toolkit affordance (OD-C regression gate)", async ({ page }) => {
    // PluginCard is reference-only. The "Use in pipeline" button is removed.
    // "When you'd use this" prose must be present on the CSV source card
    // (which has authored prose from Phase 7A).
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await page.getByRole("button", { name: /reference details for csv/i }).click();
    // Reference prose is available on demand, but no longer floods the
    // top-level catalog list.
    await expect(page.getByText(/use when/i)).toBeVisible();
    // "Use in pipeline" button must NOT be present.
    await expect(page.getByRole("button", { name: /use in pipeline/i })).not.toBeVisible();
  });
});
