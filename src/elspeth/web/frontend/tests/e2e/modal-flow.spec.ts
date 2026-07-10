// Modal flow spec — Graph modal, YAML modal, Catalog modal.
//
// Covers the riskiest modal surface: modal open/close via three entry
// points (SideRail click, Escape key, keyboard shortcut) plus hash-routed
// deep links for graph and yaml.
//
// Existing modal-adjacent stubs (yaml-export-roundtrip.spec.ts,
// topology.spec.ts) are both tracked test.skip (elspeth-7cf763da7c) — they
// still need seeded spec implementations and test a different surface (YAML
// round-trip correctness and topology validation parity). Neither exercises the
// modal open/close or keyboard affordances this spec covers; there is no overlap.
//
// Pattern: REST session creation via helpers/api.ts (same as smoke.spec.ts),
// then UI interaction. No LLM calls.

import { join } from "node:path";

import { expect, test, type APIRequestContext } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  seedCompositionState,
  tokenFromStorageState,
  uploadBlob,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

const SEEDED_SOURCE_FILENAME = "modal-flow-input.csv";
const SEEDED_SOURCE_CONTENT = "id\n1\n";

function storagePathForUploadedBlob(sessionId: string, blobId: string): string {
  const dataDir = process.env.PLAYWRIGHT_E2E_DATA_DIR;
  if (!dataDir) {
    throw new Error("PLAYWRIGHT_E2E_DATA_DIR is required for seeded blob-backed state");
  }
  return join(dataDir, "blobs", sessionId, `${blobId}_${SEEDED_SOURCE_FILENAME}`);
}

function exportableCompositionState(sessionId: string, blobId: string) {
  return {
    version: 1,
    metadata: { name: "E2E exportable pipeline", description: "" },
    sources: {
      source: {
        plugin: "csv",
        on_success: "results",
        options: {
          path: storagePathForUploadedBlob(sessionId, blobId),
          blob_ref: blobId,
          schema: { mode: "observed" },
        },
        on_validation_failure: "discard",
      },
    },
    nodes: [],
    edges: [],
    outputs: [
      {
        name: "results",
        plugin: "csv",
        options: { path: "outputs/output.csv", schema: { mode: "observed" } },
        on_write_failure: "discard",
      },
    ],
  };
}

async function seedExportableCompositionState(
  ctx: APIRequestContext,
  sessionId: string,
): Promise<void> {
  const blob = await uploadBlob(
    ctx,
    sessionId,
    SEEDED_SOURCE_FILENAME,
    SEEDED_SOURCE_CONTENT,
  );
  await seedCompositionState(ctx, sessionId, exportableCompositionState(sessionId, blob.id));
}

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
        await seedExportableCompositionState(ctx, sessionId);
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();

        // ExportYamlButton is content-gated. Seed an exportable state above so
        // this test covers the open/close path instead of the empty-pipeline
        // disabled affordance.
        const exportYamlBtn = page.getByRole("button", {
          name: /export yaml/i,
        });
        await expect(exportYamlBtn).toBeVisible();
        await expect(exportYamlBtn).toBeEnabled();
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
        await seedExportableCompositionState(ctx, sessionId);

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
        await seedExportableCompositionState(ctx, sessionId);
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
