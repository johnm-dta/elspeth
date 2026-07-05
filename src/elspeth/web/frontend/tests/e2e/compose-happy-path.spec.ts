// Compose happy-path spec — through-UI compose loop with stubbed LLM.
//
// Targets epic: elspeth-e1ab67e55a
// Adjacent epic: elspeth-528bde62bb — "Composer LLM evaluation remediation".
//
// This is the ONE spec in the suite that drives through the chat UI rather
// than seeding via API. It is the load-bearing exception to the "Testing
// Through the UI" anti-pattern (e2e-testing-strategies.md:64-69) because
// the compose flow itself IS the system under test here.
//
// What this spec WILL assert (once unblocked):
//   1. Type a minimal prompt in the chat input.
//   2. Composer-progress events stream; tool call preview_pipeline observed.
//   3. Composition state mutates and pipeline becomes Stage-1 valid.
//
// Why skipped:
//   Driving the compose loop requires either a real LLM (cost +
//   nondeterminism — out of scope for E2E) or an LLM stub. The pytest
//   suite uses ChaosLLM for this purpose; porting that to a JS-side stub
//   the webServer can dial is its own work item. Tracked as elspeth-617e1ca703.

import { test } from "@playwright/test";

test.describe("compose-happy-path — through-UI compose loop with stubbed LLM", () => {
  test.skip(
    true,
    "blocked: needs Playwright LLM stub server — tracked as elspeth-617e1ca703",
  );

  test("user prompt → composer tools → valid pipeline state", async () => {
    // 1. Open composer for a fresh session.
    // 2. Focus chat input, type a minimal prompt.
    // 3. Press Enter.
    // 4. Wait for composer-progress SSE/WS event with
    //    tool_call: preview_pipeline (and/or upsert_node, generate_yaml).
    // 5. Assert validation dot transitions to "Validation passed".
  });
});
