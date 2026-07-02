// Schema-preview-parity spec — preview_pipeline schema must match runtime.
//
// Targets epic: elspeth-e1ab67e55a
// Targets sub-issue: elspeth-87f6d5dea5 (open, P2) — "Composer schema-contract
//   preview diverges from runtime — alias handling, nested aggregation
//   requirements, and fork-to-sink checks"
//
// EXPECTED FAILURE: this spec is designed to fail until elspeth-87f6d5dea5
// is fixed. Once the underlying bug is fixed AND the direct-state-seed
// blocker (elspeth-3a7df642c5) is resolved, remove the test.skip() and the
// spec becomes a regression guard.
//
// What this spec WILL assert (once unblocked):
//   Seed source with aliased schema_config; call composer tool
//   preview_pipeline via UI button; assert returned schema matches runtime
//   /api/sessions/{id}/state schema for that node.

import { test } from "@playwright/test";

test.describe("schema-preview-parity — alias handling, nested agg, fork-to-sink", () => {
  test.skip(
    true,
    "blocked: needs direct-state-seed endpoint / LLM stub server — tracked as elspeth-3a7df642c5. " +
      "Also expected to fail on bug elspeth-87f6d5dea5 — that is intentional.",
  );

  test("aliased source schema matches runtime preview", async () => {
    // 1. createSession via API helper.
    // 2. Seed source with schema_config containing aliases.
    // 3. Trigger preview_pipeline.
    // 4. Fetch /api/sessions/{id}/state.
    // 5. Assert the previewed schema for the source equals the runtime schema.
  });
});
