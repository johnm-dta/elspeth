import { expect, test } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

test.describe("empty run discard visibility", () => {
  test("run summary surfaces source-validation discards for an empty fixture run", async ({
    page,
  }) => {
    const storageState = await page.context().storageState();
    const token = tokenFromStorageState(storageState);
    const ctx = await authedContext(token);

    try {
      const session = await createSession(ctx, "discard-warning-fixture");
      try {
        const runId = "run-discard-all-source-validation";
        const runFixture = {
          id: runId,
          session_id: session.id,
          status: "empty",
          accounting: {
            source: { rows_processed: 0 },
            tokens: {
              emitted: 0,
              terminal: 0,
              succeeded: 0,
              failed: 0,
              structural: 0,
              pending: 0,
            },
            routing: {
              routed_success: 0,
              routed_failure: 0,
              quarantined: 0,
              discarded: 0,
            },
            integrity: {
              closure: "closed",
              missing_terminal_outcomes: 0,
              duplicate_terminal_outcomes: 0,
            },
          },
          error: null,
          started_at: "2026-05-24T08:00:00.000Z",
          finished_at: "2026-05-24T08:00:01.000Z",
          composition_version: 1,
          discard_summary: {
            total: 2,
            validation_errors: 2,
            transform_errors: 0,
            sink_discards: 0,
            stages: [
              {
                stage: "source_validation",
                node_id: "source_csv_upload",
                count: 2,
              },
            ],
          },
        };

        await page.route(`**/api/sessions/${session.id}/runs`, async (route) => {
          await route.fulfill({ json: [runFixture] });
        });
        await page.route(`**/api/runs/${runId}/outputs`, async (route) => {
          await route.fulfill({
            json: {
              run_id: runId,
              landscape_run_id: "landscape-discard-fixture",
              artifacts: [],
            },
          });
        });

        const composer = new ComposerPage(page);
        await composer.goto(session.id);
        await composer.waitForChatReady();

        const warning = page.getByRole("alert").filter({
          hasText: /2 rows discarded at source validation/i,
        });
        await expect(warning).toBeVisible();
        await expect(warning).toContainText("source_csv_upload");
        await expect(warning).toContainText("Run terminated empty");
      } finally {
        await deleteSession(ctx, session.id);
      }
    } finally {
      await ctx.dispose();
    }
  });
});
