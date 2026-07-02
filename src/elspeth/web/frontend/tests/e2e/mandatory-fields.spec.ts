// Mandatory-fields spec — transform nodes must declare plugin/on_success/on_error.
//
// Targets epic: elspeth-e1ab67e55a
// Targets sub-issue: elspeth-39089c98ee (closed) — "CompositionState.validate()
//   does not enforce mandatory transform fields"
//
// What this spec WILL assert (once unblocked):
//   1. Seed transform node missing plugin. Stage-1 banner reads invalid;
//      validation entry contains "plugin required".
//   2. Patch via API to add plugin; banner clears.
//
// Why skipped:
//   See topology.spec.ts — same blocker (no direct state-mutation REST
//   endpoint; would otherwise need an LLM stub). Tracked as elspeth-3a7df642c5.

import { test } from "@playwright/test";

test.describe("mandatory-fields — transform nodes must declare plugin", () => {
  test.skip(
    true,
    "blocked: needs direct-state-seed endpoint / LLM stub server — tracked as elspeth-3a7df642c5",
  );

  test("transform without plugin is reported invalid", async () => {
    // 1. createSession via API helper.
    // 2. Seed transform node with plugin=null.
    // 3. Click Validate.
    // 4. Assert validation dot reads "Validation failed".
    // 5. Assert one of the validation entries mentions "plugin".
  });

  test("transform with plugin populated is reported valid", async () => {
    // 1. createSession via API helper.
    // 2. Seed transform node with a real plugin id (e.g. "echo").
    // 3. Click Validate.
    // 4. Assert validation dot reads "Validation passed".
  });
});
