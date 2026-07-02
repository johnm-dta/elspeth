// Topology spec — runtime-connection vs UI-edge validation parity.
//
// Targets epic: elspeth-e1ab67e55a
// Targets sub-issue: elspeth-3724f02de9 (closed) — "CompositionState.validate()
//   uses UI edges to decide runtime reachability — rejects valid
//   connection-wired pipelines and blesses invalid edge-only pipelines"
//
// What this spec WILL assert (once unblocked):
//   1. Seed via API: pipeline with source.on_success → node.input
//      connections but no UI edges. Open #/{sessionId}/spec.
//      Validation banner reads "valid"; generate_yaml succeeds.
//   2. Re-seed UI-edge-only chain (no runtime connections); banner
//      reads "invalid".
//
// Why skipped:
//   The composer tools (upsert_node, upsert_edge, etc.) are LLM-facing
//   and not exposed as REST endpoints we can call directly. To seed
//   composition state without driving a real LLM, we need either:
//     (a) a direct state-mutation REST endpoint (e.g.,
//         POST /api/sessions/{id}/state with the canonical state JSON),
//         which doesn't exist today; or
//     (b) an LLM stub server (the existing pytest ChaosLLM pattern,
//         ported to a TypeScript stub the webServer can dial via
//         ELSPETH_WEB__composer_model env override).
//
// Tracked as elspeth-3a7df642c5. Until then this spec is a tracked skip —
// but the test body is sketched so the next contributor does not need
// to re-derive intent from the issue tracker.

import { test } from "@playwright/test";

test.describe("topology — runtime-connection vs UI-edge validation parity", () => {
  test.skip(
    true,
    "blocked: needs direct-state-seed endpoint / LLM stub server — tracked as elspeth-3a7df642c5",
  );

  test("connection-wired chain (no UI edges) validates as valid", async () => {
    // 1. createSession via API helper.
    // 2. Seed compositionState: source with on_success="t1",
    //    node t1 with on_success="sink-out", output sink-out.
    //    Critically: edges = [] (no UI edges).
    // 3. Navigate to #/{sessionId}/spec.
    // 4. Click Validate.
    // 5. Assert validation dot reads "Validation passed".
    // 6. Open YAML tab, click export, parse YAML, assert it contains
    //    the expected source/transform/sink.
  });

  test("UI-edge-only chain (no runtime connections) validates as invalid", async () => {
    // 1. createSession via API helper.
    // 2. Seed compositionState: source with on_success=null,
    //    node t1 with on_success=null, output sink-out (also unwired).
    //    Edges = [{from_node: "source", to_node: "t1"}, {from_node: "t1", to_node: "sink-out"}].
    // 3. Navigate to #/{sessionId}/spec.
    // 4. Click Validate.
    // 5. Assert validation dot reads "Validation failed".
    // 6. Assert the validation banner enumerates an unreachability error.
  });
});
