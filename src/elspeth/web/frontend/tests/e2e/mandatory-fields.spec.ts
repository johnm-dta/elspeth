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
//   State seeding now exists via seedCompositionState(), but this spec body
//   still needs to be implemented from the outline below. Tracked as
//   elspeth-7cf763da7c.

import { test } from "@playwright/test";

test.describe("mandatory-fields — transform nodes must declare plugin", () => {
  test.skip(
    true,
    "pending seeded spec implementation — tracked as elspeth-7cf763da7c",
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
