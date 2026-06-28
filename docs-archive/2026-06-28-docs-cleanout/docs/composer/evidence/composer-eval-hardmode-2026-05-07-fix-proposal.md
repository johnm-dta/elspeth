# Composer Hard-Mode Eval Fix Proposal - 2026-05-07

Source data: `evals/composer-harness/runs/2026-05-07-hardmode/` and `docs/composer/evidence/composer-eval-hardmode-2026-05-07.md`.

Operator correction: actual provider spend was about **$3**. The wall-second estimate in the original report was a bad heuristic, not a real budget overrun.

## Executive Read

The suite does not point to one broad "model bad" failure. It separates into four fixable classes:

1. The harness finalizer is brittle and loses ledgers when expected failure endpoints return non-2xx.
2. The composer often narrates a conceptual build after no state mutation, especially for uploaded-file workflows.
3. The composer can still commit or attempt graphs that violate plugin contracts before runtime.
4. Some runtime/plugin contracts are enforced only when rows execute, so `/validate` says yes to pipelines that are mechanically doomed.

The limit probes remain a relative strength: SharePoint/Outlook, PDF, webhook, and runtime-inspection boundaries were mostly held honestly. The weakest path is not refusal; it is turning a supported CSV-backed request into a committed, executable graph.

## P0: Fix Harness Ledger Reliability

**Evidence**

- 11/15 scenarios needed synthesized `ledger.json`.
- `evals/composer-harness/hardmode/finalize_scenario.sh` calls `evals_get_yaml ... || echo '{}'`, but `evals_get_yaml` delegates to `_evals_http_get`, whose non-2xx path calls `evals_die` and exits the process. The `||` cannot recover from a helper that exits.
- Common failure examples: invalid no-state sessions returned `/state/yaml` 404; invalid preflight states returned 409; P4 happy completed successfully but post-run YAML export returned 409 due output-path collision.

**Fix**

- Add a nonfatal GET helper in `evals/lib/common.sh`, for example `evals_try_get <url> <out> <code_out>`, that returns nonzero instead of calling `evals_die`.
- Use it for final YAML, messages, diagnostics, and any other best-effort finalizer collection.
- Write `ledger.json` in all validate-fail, execute-fail, run-fail, run-timeout, and post-run-artifact-fail paths.
- Include `artifact_collection_errors` in the ledger so harness failures remain visible without blocking aggregation.

**Acceptance**

- Re-run `finalize_scenario.sh` against a no-state session: exit 0, `validate.json`, `final_yaml.json` as `{}`, `messages.json`, and `ledger.json` all exist.
- Re-run P4 happy with existing output collision: ledger preserves `run_status=completed`, rows `4/4`, and records post-run YAML failure as artifact metadata rather than harness failure.

## P0: Fix Uploaded-Blob Custody in Composer Builds

**Evidence**

- P3 happy uploaded `contact_form_submissions.csv` as blob `98b1357d-...`, 1488 bytes, 8 data rows.
- The committed pipeline used a different blob/path: `5a1a81c6-..._hubspot_export.csv`, 51 bytes, header only.
- The run therefore validated and executed, but source emitted zero rows and status was `empty`.

**Fix**

- Make existing uploaded blobs the default source path for file-backed user requests. The composer should call `list_blobs` and bind the matching ready blob via `set_source_from_blob`, not create a header-only inline blob from inferred schema.
- Add an atomic `set_pipeline` path for **existing** blobs, not only `source.inline_blob`. Suggested shape: `source.blob_id` or `source.existing_blob_id`; the tool resolves path/blob_ref authoritatively exactly like `set_source_from_blob`.
- Reject or warn on header-only inline CSV creation when a ready uploaded CSV exists in the same session and the user asked to process that file.
- Preserve blob identity in `final_yaml.json` or ledger artifact metadata, even if runtime YAML strips web-only `blob_ref`.

**Acceptance**

- P3-T1 rerun binds the scenario-uploaded blob id from `blob.json`, not a new header-only blob.
- Source accounting reports 8 rows processed before LLM/routing.
- A regression test creates two blobs with the same headers, one header-only and one with data, then verifies composer-selected source matches the uploaded file named in the request.

## P0: Prevent "Conceptual Build" Replies After No Mutation

**Evidence**

- P1 happy, P3 stress, and P4 edge repeatedly returned prose like "I tried / I can continue" while `state.after.tN.json` remained `null`.
- P3 stress consumed four turns and never mutated state.

**Fix**

- Add a server-side compose-loop guard: when the user intent is build/modify/run and no successful mutation occurred in the turn, the final assistant response must include the concrete blocking tool error, not a generic conceptual plan.
- Better: before final response, force a tool/action decision if state is empty and the response says the workflow is set up, ready, or attempted.
- Add a `state_exists` progress marker to the prompt-visible composer progress so the model can see "still empty" as a hard blocker.
- Strengthen the skill: "Do not say you set up or tried a build unless at least one mutation tool returned `success: true`; if state is empty, call the source/blob setup tool or ask for the missing file."

**Acceptance**

- A no-state build attempt cannot produce a final "I set up conceptually" response without either a mutation or a specific tool failure.
- P3-T4 stress either commits a partial valid source or stops with one actionable reason tied to a failed tool result.

## P1: Make Secret Wiring a Tool-Time Contract

**Evidence**

- P2 happy failed validation with `secret_refs`: `code_themes:api_key` contained a literal value.
- Validation correctly caught the issue, but only after invalid state existed.

**Fix**

- Reuse the credential-field detection used by `/validate` inside composer mutation tools (`set_pipeline`, upsert/patch node, patch source/output options).
- If a credential-bearing field contains a literal, return `ToolResult(success=false)` and do not mutate state.
- Tool response should name the field and point to `list_secret_refs -> validate_secret_ref -> wire_secret_ref`; never echo the value.

**Acceptance**

- A `set_pipeline` call with `api_key: "..."` leaves state unchanged and returns a structured repair instruction.
- A pipeline using `wire_secret_ref` validates and executes through the resolver path.

## P1: Close Static Contract Gaps Before Runtime

### JSON explode after LLM string

**Evidence:** P2 stress validated, executed, then failed because `json_explode.array_field=llm_response` received a string from the LLM transform. This is mechanically doomed: single-query LLM emits a string response field; `json_explode` requires list/tuple.

**Fix:** Express `json_explode.array_field` as a semantic type requirement (`list`) and express single-query LLM `response_field` as `str`. `/validate` should reject `LLM string -> json_explode array_field` before `/execute`. Do not coerce strings in `json_explode`; per tier rules this is upstream graph/config error.

**Acceptance:** P2-T4 state fails `/validate` with a targeted message: "json_explode requires list field; LLM response_field is str. Use structured-output parsing or a transform that parses JSON object/string first."

### Numeric aggregation over categorical strings

**Evidence:** P2 edge reached runtime and failed because `batch_distribution_profile.value_field=financial_barrier` received a string. That plugin is numeric-statistics only; categorical distributions should use `batch_top_k` or a categorical-count aggregation.

**Fix:** Add plugin assistance and validation guidance:

- `batch_distribution_profile.value_field` is numeric-only.
- categorical "distribution", "barrier counts", "theme frequency" maps to `batch_top_k`, not `batch_distribution_profile`.
- When upstream schema is fixed and declares a nonnumeric type, fail `/validate`.
- When upstream schema is observed/unknown, add a data-sample preflight for uploaded CSVs before execute, or at least a high-severity composer warning.

**Acceptance:** P2-T2 rerun uses `batch_top_k` grouped by community/wave for categorical barriers, or `/validate` blocks the numeric plugin before runtime.

### Batch replicate placement

**Evidence:** P1 edge put `batch_replicate` under `transforms` with no trigger. Runtime failed in `transform_fanout_regions...` with `'str' object has no attribute 'contract'`.

**Fix:** Treat batch-aware plugins as requiring the aggregation/batch path unless a plugin explicitly supports row mode. Composer prevalidation should reject `batch_replicate` as a plain transform without `trigger` and `output_mode: transform`.

**Acceptance:** P1-T2 invalid placement fails tool-time or `/validate`; corrected placement either uses the supported batch deaggregation shape or a simpler row-level split plugin.

## P1: Clarify Gate `discard` Semantics

**Evidence**

- P3 edge partial state routed a gate branch to `discard`; validation rejected it as unresolved.
- Validation suggestion says "Use 'discard' to drop rows without routing," which is confusing if gate route maps cannot target `discard`.

**Fix Decision Needed**

Pick one contract and make it mechanical:

- Preferred: support `discard` as a virtual gate route target, with audited terminal discard outcome.
- Alternative: keep gate routes sink-only, but change validation messages and composer skill text to say "gate routes must target a named sink; create a `dropped` sink if you want to retain or drop nonmatches."

**Acceptance**

- The same gate route behavior is accepted/rejected consistently by composer validation, engine assembly, YAML import/export, and docs.

## P2: Improve Recovery From Duplicate Consumers

**Evidence**

- P1 stress failed graph validation: duplicate consumers for `classified_rows`; validation correctly said "Use a gate for fan-out."
- The model still failed to repair after multiple turns.

**Fix**

- Add a targeted repair helper or composer skill example: duplicate consumers -> insert gate/fork node or coalesce pattern, with before/after JSON.
- Consider a `suggest_graph_repair` tool response for duplicate consumer errors that returns an exact mutation skeleton.

**Acceptance**

- Given duplicate consumers, composer can produce a corrected gate/fan-out graph in one follow-up turn.

## P2: Calibrate Cost Tracking

**Evidence**

- Actual run cost was about $3.
- Wall-time heuristic estimated $60.77, which was wrong for this provider/model path.

**Fix**

- Capture provider usage/cost from response metadata where available.
- Aggregate `prompt_tokens`, `cached_prompt_tokens`, `completion_tokens`, and provider/model per scenario.
- Use wall time only as a performance metric, not a budget proxy.

**Acceptance**

- `aggregate.json` and `aggregate_summary.json` report provider token metadata and `cost.source = "not_available"` unless actual provider billing data is integrated; they never fabricate wall-second dollars.

## Suggested Implementation Order

1. Harness finalizer nonfatal artifact collection.
2. Uploaded-blob custody and no-conceptual-build guard.
3. Tool-time secret literal rejection.
4. Static contract blockers for `json_explode`, numeric aggregations, and batch-aware plugin placement.
5. Gate `discard` contract decision and implementation.
6. Composer repair templates for duplicate consumers and categorical aggregation.
7. Cost telemetry calibration.

## Minimal Rerun Set After Fixes

- `p3_t1_happy` - proves uploaded blob custody and non-empty source.
- `p2_t1_happy` - proves secret wiring repair.
- `p2_t4_stress` - proves LLM string -> `json_explode` is blocked before runtime or correctly parsed.
- `p1_t2_edge` - proves batch replicate/fan-out placement no longer runtime-crashes.
- `p4_t1_happy` - guards the already-working path.
- One no-state limit scenario - proves finalizer always writes ledger.
