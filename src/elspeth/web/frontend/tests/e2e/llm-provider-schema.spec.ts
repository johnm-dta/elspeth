// LLM-provider-schema spec — catalog must surface provider union fields.
//
// Targets epic: elspeth-e1ab67e55a
// Targets sub-issue: elspeth-dcf12c061b (open, P2) — "CatalogServiceImpl.get_schema(
//   'transform', 'llm') returns only the base LLMConfig schema, omitting required
//   provider-specific fields such as Azure deployment_name"
//
// EXPECTED FAILURE: this spec is designed to fail until elspeth-dcf12c061b is
// fixed. Once the underlying bug is fixed, remove the test.fixme() — the
// spec becomes a regression guard for the catalog union surface.
//
// What this spec WILL assert (once unblocked):
//   1. Open catalog drawer.
//   2. Click llm transform.
//   3. Provider-union fields render (Azure deployment_name, OpenRouter api_key).
//   4. Adding an llm node without provider fields surfaces a Stage-1 error.

import { test } from "@playwright/test";

test.describe("llm-provider-schema — catalog must surface provider-union fields", () => {
  test.fixme(
    true,
    "Expected to fail on open bug elspeth-dcf12c061b until that is fixed. " +
      "Also blocked on add-node-without-driving-LLM (state-seed gap).",
  );

  test("llm transform schema enumerates Azure and OpenRouter variants", async () => {
    // 1. Open catalog drawer.
    // 2. Find llm transform entry.
    // 3. Assert the rendered schema preview includes Azure-specific
    //    fields (deployment_name) and OpenRouter-specific fields (api_key).
  });

  test("llm node without provider fields surfaces a Stage-1 error", async () => {
    // 1. Seed an llm node with provider unset.
    // 2. Click Validate.
    // 3. Assert validation dot reads "Validation failed".
    // 4. Assert validation banner mentions a provider-specific field.
  });
});
