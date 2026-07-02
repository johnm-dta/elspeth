// YAML export round-trip spec — Stage-1-valid state always yields settings-valid YAML.
//
// Targets epic: elspeth-e1ab67e55a (parent acceptance criterion: "generate_yaml
// never produces settings-invalid output from a Stage 1-valid state").
//
// What this spec WILL assert (once unblocked):
//   Seed valid pipeline. Click YAML export; copy YAML; load it via
//   /api/composer/import (or a fresh-session reload). Assert the resulting
//   state is also Stage-1 valid.

import { test } from "@playwright/test";

test.describe("yaml-export-roundtrip — generate_yaml output re-validates", () => {
  test.skip(
    true,
    "blocked: needs direct-state-seed endpoint / LLM stub server — tracked as elspeth-3a7df642c5",
  );

  test("valid state → exported YAML → re-imported state is also valid", async () => {
    // 1. createSession via API helper.
    // 2. Seed a known-valid pipeline (source + 1 transform + sink, fully wired).
    // 3. Open YAML tab; click export; capture YAML text.
    // 4. POST the YAML to /api/composer/import (or load via new session).
    // 5. Validate the new session.
    // 6. Assert validation dot reads "Validation passed".
  });
});
