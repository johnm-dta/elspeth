// Release smoke test (non-mocked, runs against a freshly built dist).
//
// Regression guard for the 0.6.0 pre-merge crash (Issue 1): a frontend bundle
// built before `ebbf90fcd` (the ADR-025 plural-`sources` migration) read the
// dropped singular `state.source` and threw
//   TypeError: Cannot read properties of undefined (reading 'options')
// inside ChatPanel (readSourceBlobRef / compositionHasSource), tripping the
// <ErrorBoundary label="Chat panel"> (Layout.tsx) and blanking the composer on
// guided source-select AND on reloading any composed session. The deployed
// dist predated the fix; nothing caught it.
//
// Why a *staging* spec, and why not just reuse composer-guided.spec.ts:
//   - composer-guided.spec.ts MOCKS POST /validate and /audit-readiness and
//     runs under the LOCAL config, which serves the Vite DEV server (source) —
//     so it never renders the built bundle against real plural-`sources` state.
//   - This spec talks to the REAL backend and is served by the actual built
//     dist (playwright.staging.config.ts has no webServer; the deployment is
//     up), so a stale or shape-skewed bundle fails it.
//
// It makes NO LLM calls: it stops at the deterministic source SchemaForm,
// before the recipe-offer / apply (LLM) step.
//
// Wiring: `*.staging.spec.ts` is ignored by the local/CI config and run only by
// playwright.staging.config.ts via `npm run test:e2e:staging:smoke`.

import { expect, test } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
  uploadBlob,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

const BLOB_FILENAME = "release-smoke-orders.csv";
// Minimal CSV: header + one data row. Enough to register a source blob.
const SAMPLE_CSV = "id,name,value\n1,widget,42\n";

test.describe("release smoke — built bundle renders against real backend", () => {
  test("guided source-select renders the source SchemaForm (no chat-panel crash)", async ({
    page,
  }) => {
    // Out-of-band setup via REST (not counted as user interaction): create a
    // session and upload a seed CSV so the csv source chip is selectable.
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    let sessionId: string | undefined;
    try {
      const session = await createSession(ctx, "release-smoke");
      sessionId = session.id;
      await uploadBlob(ctx, sessionId, BLOB_FILENAME, SAMPLE_CSV);

      const composer = new ComposerPage(page);
      await composer.goto(sessionId);
      await composer.waitForChatReady();

      // Enter guided mode and select the csv source. On a pre-migration bundle
      // this re-renders ChatPanel against plural-`sources` backend state, reads
      // the dropped singular `state.source`, and throws — so the SchemaForm
      // below never appears and the ErrorBoundary fallback shows instead.
      await page.getByRole("button", { name: "Switch to guided" }).click();
      await expect(page.getByLabel(/guided composer/i)).toBeVisible();

      await page.getByRole("button", { name: "CSV", exact: true }).click();

      // POSITIVE: the source SchemaForm renders. CSV source has three required
      // fields (config_base.py): schema, path, on_validation_failure.
      await expect(page.getByLabel(/^schema$/i)).toBeVisible();
      await expect(page.getByLabel(/^path$/i)).toBeVisible();
      await expect(page.getByLabel(/on\s+validation\s+failure/i)).toBeVisible();

      // NEGATIVE: the chat-panel ErrorBoundary fallback (ErrorBoundary.tsx:
      // `{label} encountered an error`) is NOT showing.
      await expect(
        page.getByText(/Chat panel encountered an error/i),
      ).toHaveCount(0);
    } finally {
      if (sessionId !== undefined) {
        await deleteSession(ctx, sessionId);
      }
      await ctx.dispose();
    }
  });
});
