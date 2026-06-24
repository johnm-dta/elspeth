// E2E spec: guided-mode wizard recipe-match → wire stage → completion.
//
// Targets the demo SLA defined in §10.3 of the implementation plan
// (2026-05-11-composer-guided-mode.md): the recipe-match path is
// deterministic (zero LLM calls) and reaches a completed pipeline fast.
//
// ── P7.7: extended to the wire stage + completion (resolves the D12 gate) ────
//
// The pre-P3 spec drove csv → classify recipe → wire and then clicked
// "Confirm wiring" WITHOUT resolving the surfaced interpretation cards. P3's
// D12 gate disables the wire-confirm button while any pending guided
// interpretation review remains (ChatPanel: guidedResponsePending ||
// hasPendingGuidedInterpretations), so that click landed on a disabled button
// and the test could never complete — the known pre-existing failure this task
// fixes. The staged rewrite drives the gate correctly: it APPLIES the classify
// recipe (which jumps Step 2.5 → Step 4 wire and deterministically seeds two
// interpretation reviews — llm_prompt_template + llm_model_choice — from the
// recipe slot values, with NO LLM call), RESOLVES both cards (accept-as-drafted),
// which enables "Confirm wiring", then confirms → terminal=completed.
//
// D13 (live-profile advisor opt-out): the empty/live-guided profile
// (advisor_checkpoints=false) auto-completes a valid wire confirm with NO
// advisor provider call. The local e2e backend has NO LLM provider configured
// (composer_boot_probe_transient_failure on boot), so reaching
// terminal=completed is itself proof the path made no advisor/LLM call.
//
// M1 (from/to edge naming): the wire overlay renders each edge as a listitem
// whose accessible name is "{from} to {to}" — the post-M1 field names — never
// "from_id"/"to_id".
//
// B6 (field_mapper / schema-relax reconciliation re-render): NOT asserted here.
// The in-flow reconciliation the brief describes does not exist in the guided
// flow — the sole call site of rebuild_wire_turn_after_reconciliation passes a
// no-op resurface, guided has no graph-editing respond at the wire stage, and
// the only graph-mutating composer tools (upsert_node/set_pipeline) are MCP-only
// (no REST surface reachable from a Playwright auth context). Surfaced to the
// operator as a spec-vs-reality gap (task report) rather than faked.
//
// For prior gap history (Gap 1 startGuided wiring, Gap 2 S2 path allowlist,
// Gap 3 on_validation_failure, Gap 4 collision_policy, Gap 5 blob_ref
// resolver, Gap 6 unsatisfied_slots) see git log on this file.

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

import {
  authedContext,
  createSession,
  deleteSession,
  tokenFromStorageState,
  uploadBlob,
} from "./helpers/api";
import { ComposerPage } from "./page-objects/composer-page";

const BLOB_FILENAME = "playwright-orders.csv";

// Minimal CSV content: header + one data row. The "category" column lets us
// satisfy the classify recipe's classifier-keyword required-field predicate.
const SAMPLE_CSV = "id,name,category\n1,widget,a\n";

// Frontend root: playwright.config.ts passes an absolute .e2e-data path
// anchored to the frontend directory, and the backend stores uploaded blobs
// relative to that data_dir.
const FRONTEND_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const E2E_DATA_DIR = process.env.PLAYWRIGHT_E2E_DATA_DIR
  ? resolve(process.env.PLAYWRIGHT_E2E_DATA_DIR)
  : resolve(FRONTEND_ROOT, ".e2e-data");

// Construct the blob storage path.
// service.py:207-211: data_dir/blobs/{session_id}/{blob_id}_{filename}
function blobStoragePath(sessionId: string, blobId: string): string {
  return resolve(E2E_DATA_DIR, "blobs", sessionId, `${blobId}_${BLOB_FILENAME}`);
}

// Sink output path — must be under data_dir/outputs/ (paths.py:44).
const SINK_OUTPUT_PATH = resolve(E2E_DATA_DIR, "outputs", "playwright-guided-output.json");

async function isolateAuditReadinessSideRail(
  page: Page,
  sessionId: string,
): Promise<void> {
  await page.route(`**/api/sessions/${sessionId}/validate`, async (route) => {
    await route.fulfill({
      json: {
        is_valid: true,
        summary: "Guided demo pipeline validates.",
        checks: [],
        errors: [],
        warnings: [],
        semantic_contracts: [],
      },
    });
  });

  await page.route(`**/api/sessions/${sessionId}/audit-readiness`, async (route) => {
    await route.fulfill({
      json: {
        session_id: sessionId,
        composition_version: 1,
        checked_at: "2026-05-19T12:00:00Z",
        rows: [
          {
            id: "validation",
            label: "Validation",
            status: "ok",
            summary: "Guided demo pipeline validates.",
            detail: null,
            component_ids: [],
          },
          {
            id: "plugin_trust",
            label: "Plugin trust",
            status: "ok",
            summary: "Guided demo plugins are trusted.",
            detail: null,
            component_ids: [],
          },
          {
            id: "provenance",
            label: "Provenance",
            status: "not_applicable",
            summary: "No run provenance yet.",
            detail: null,
            component_ids: [],
          },
          {
            id: "retention",
            label: "Retention",
            status: "not_applicable",
            summary: "No run retention yet.",
            detail: null,
            component_ids: [],
          },
          {
            id: "llm_interpretations",
            label: "LLM interpretations",
            status: "not_applicable",
            summary: "No interpretation events.",
            detail: null,
            component_ids: [],
          },
          {
            id: "secrets",
            label: "Secrets",
            status: "not_applicable",
            summary: "No secret checks in the guided demo.",
            detail: null,
            component_ids: [],
          },
        ],
        validation_result: {
          is_valid: true,
          summary: "Guided demo pipeline validates.",
          checks: [],
          errors: [],
          warnings: [],
          semantic_contracts: [],
        },
      },
    });
  });
}

// Resolve every surfaced guided interpretation card (accept-as-drafted). The
// llm_prompt_template card is gated behind a scroll-to-end requirement on its
// "Prompt template review" region (InterpretationReviewTurn:
// requiresPromptTemplateScroll → hasScrolledToEnd), so we scroll each such
// region to its end and fire the scroll event the gate listens for, then click
// the first ENABLED accept button. Loop until the wire-confirm button enables
// (the D12 predicate: no pending reviews remain).
async function resolveGuidedReviews(page: Page): Promise<number> {
  const confirmBtn = page.getByRole("button", { name: "Confirm wiring", exact: true });
  const acceptButtons = page.getByRole("button", { name: /^Accept /i });
  const promptRegions = page.getByRole("region", {
    name: "Prompt template review",
  });
  let resolved = 0;
  const deadline = Date.now() + 30_000;
  while (await confirmBtn.isDisabled().catch(() => true)) {
    if (Date.now() > deadline) {
      throw new Error(
        "guided reviews never all resolved (Confirm wiring stayed disabled)",
      );
    }
    const regionCount = await promptRegions.count().catch(() => 0);
    for (let i = 0; i < regionCount; i++) {
      await promptRegions
        .nth(i)
        .evaluate((el) => {
          el.scrollTop = el.scrollHeight;
          el.dispatchEvent(new Event("scroll"));
        })
        .catch(() => {});
    }
    const total = await acceptButtons.count().catch(() => 0);
    let clicked = false;
    for (let i = 0; i < total; i++) {
      const btn = acceptButtons.nth(i);
      if (await btn.isEnabled().catch(() => false)) {
        await btn.click().catch(() => {});
        resolved += 1;
        clicked = true;
        break;
      }
    }
    if (!clicked) await page.waitForTimeout(200);
  }
  return resolved;
}

test.describe("composer-guided — recipe-match wire stage + completion", () => {
  test(
    "guided demo: CSV → classify recipe → wire (from/to overlay) → resolve cards → completed (live profile, no advisor)",
    async ({ page }) => {
      // ── Out-of-band setup ──────────────────────────────────────────────────
      // Create session + upload CSV blob via REST before navigating the SPA.
      const storageState = await page.context().storageState();
      const token = tokenFromStorageState(storageState);
      const ctx = await authedContext(token);

      let sessionId: string | undefined;
      try {
        const session = await createSession(ctx, "playwright-guided-demo");
        sessionId = session.id;
        await isolateAuditReadinessSideRail(page, sessionId);

        const blob = await uploadBlob(ctx, sessionId, BLOB_FILENAME, SAMPLE_CSV);

        // ── Navigate + enter guided mode ─────────────────────────────────────
        // "Switch to guided" resolves to the live/empty profile
        // (advisor_checkpoints=false) via GET /guided — the D13 opt-out path.
        const composer = new ComposerPage(page);
        await composer.goto(sessionId);
        await composer.waitForChatReady();
        await page.getByRole("button", { name: "Switch to guided" }).click();
        await expect(page.getByLabel(/guided composer/i)).toBeVisible();

        // ── Step 1 source: SINGLE_SELECT — pick "csv" ──────────────────────
        await expect(
          page.getByRole("button", { name: "csv", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "csv", exact: true }).click();

        // ── Step 1 source: SCHEMA_FORM — schema, path, on_validation_failure
        const sourcePath = blobStoragePath(sessionId, blob.id);
        await expect(page.getByLabel(/^schema$/i)).toBeVisible();
        await page.getByLabel(/^schema$/i).fill('{"mode":"observed"}');
        await page.getByLabel(/^path$/i).fill(sourcePath);
        await page.getByLabel(/on\s+validation\s+failure/i).fill("discard");
        await expect(
          page.getByRole("button", { name: "Continue", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Continue", exact: true }).click();

        // ── Step 2 sink: SINGLE_SELECT — pick "json" ───────────────────────
        await expect(
          page.getByRole("button", { name: "json", exact: true }),
        ).toBeVisible();
        await page.getByRole("button", { name: "json", exact: true }).click();

        // ── Step 2 sink: SCHEMA_FORM — path + collision_policy + format + mode
        // The file sink requires `mode` set explicitly (write|append).
        await expect(page.getByLabel(/^schema$/i)).toBeVisible();
        await page.getByLabel(/^schema$/i).fill('{"mode":"observed"}');
        await page.getByLabel(/^path$/i).fill(SINK_OUTPUT_PATH);
        await page.getByLabel(/collision.?policy/i).selectOption("auto_increment");
        await page.getByLabel(/^format$/i).selectOption("json");
        await page.getByLabel(/^mode$/i).selectOption("write");
        await expect(
          page.getByRole("button", { name: "Continue", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Continue", exact: true }).click();

        // ── Step 2 required fields: MULTI_SELECT_WITH_CUSTOM ──────────────
        // Add "category" — satisfies _classify_predicate (recipe_match.py).
        const customInput = page.getByLabel("Custom field", { exact: true });
        await expect(customInput).toBeVisible();
        await customInput.fill("category");
        await page.getByRole("button", { name: "Add", exact: true }).click();
        await expect(page.getByText("category")).toBeVisible();
        await page.getByRole("button", { name: "Continue", exact: true }).click();

        // ── Step 2.5 RECIPE_OFFER — fill slots, Apply recipe ──────────────
        // classify-rows-llm-jsonl declares three user-fillable slots. Apply
        // jumps Step 2.5 → Step 4 wire directly (recipe accept is the only
        // LLM-free route to the wire stage).
        await expect(
          page.getByRole("button", { name: "Apply recipe", exact: true }),
        ).toBeVisible();
        await page
          .getByLabel(/classifier_template/i)
          .fill("Classify {{ row['name'] }} as widget or gadget.");
        await page.getByLabel(/^model\b/i).fill("anthropic/claude-sonnet-4.6");
        await page.getByLabel(/api_key_secret/i).fill("openrouter-api-key");
        await expect(
          page.getByRole("button", { name: "Apply recipe", exact: true }),
        ).toBeEnabled();
        await page.getByRole("button", { name: "Apply recipe", exact: true }).click();

        // ── Step 4 wire stage: topology + edge-contract overlay ────────────
        // Both blobs are present: the topology produces edges and the overlay
        // renders each edge as a "{from} to {to}" listitem (M1 — NOT
        // from_id/to_id). The classify pipeline wires source → classifier →
        // output:labelled, so the overlay shows those two edges.
        await expect(page.getByRole("heading", { name: "Review wiring" })).toBeVisible();
        await expect(
          page.getByRole("listitem", { name: "source to classifier" }),
        ).toBeVisible();
        await expect(
          page.getByRole("listitem", { name: "classifier to output:labelled" }),
        ).toBeVisible();
        // M1 guard: the old from_id/to_id naming must not appear in the overlay.
        await expect(
          page.getByRole("listitem", { name: /from_id|to_id/ }),
        ).toHaveCount(0);

        // ── D12 gate: Confirm wiring is disabled until the recipe-seeded
        // interpretation cards (llm_prompt_template + llm_model_choice) are
        // resolved. This is the exact failure the pre-P3 spec hit by clicking
        // straight through.
        await expect(
          page.getByText(/assumptions? to review/i),
        ).toBeVisible();
        await expect(
          page.getByRole("button", { name: "Confirm wiring", exact: true }),
        ).toBeDisabled();

        const resolvedCount = await resolveGuidedReviews(page);
        expect(resolvedCount).toBeGreaterThanOrEqual(2);

        // Gate released → Confirm wiring enabled.
        const confirm = page.getByRole("button", { name: "Confirm wiring", exact: true });
        await expect(confirm).toBeEnabled();
        await confirm.click();

        // ── Completion (D13): valid wire confirm on the live profile reaches
        // terminal=completed with NO advisor provider call. The local backend
        // has no LLM provider, so completion is itself the proof. The completed
        // surface (CompletionSummary) renders "Pipeline ready".
        await expect(
          page.getByRole("heading", { name: "Pipeline ready" }),
        ).toBeVisible();
        await expect(
          page.getByRole("button", { name: "Open freeform editor", exact: true }),
        ).toBeVisible();
        // The wire turn is gone — we are on the completed surface, not re-emitted.
        await expect(
          page.getByRole("button", { name: "Confirm wiring", exact: true }),
        ).toHaveCount(0);
      } finally {
        if (sessionId !== undefined) {
          await deleteSession(ctx, sessionId);
        }
        await ctx.dispose();
      }
    },
  );
});
