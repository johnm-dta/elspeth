# Batch LLM Invariant Follow-on Design — Azure Batch and OpenRouter Batch

**Status:** Superseded by ADR-020 (`docs/architecture/adr/020-retire-batch-llm-transforms.md`) — 2026-05-06. The fix path described below was not pursued; the targets were retired instead.
**Date:** 2026-04-21
**Relates to:** `elspeth-be398f0bcb`
**Scope:** Deferred batch-LLM tranche after the remaining transform invariant migration

---

## Summary

`AzureBatchLLMTransform` and `OpenRouterBatchLLMTransform` are not ready for
truthful `passes_through_input = True` annotation yet.

The current blocker is not constructor support or probe plumbing. The blocker is
contract truthfulness:

1. Both batch transforms currently advertise the standard LLM guaranteed-field
   contract via `get_llm_guaranteed_fields(response_field)`.
2. That contract includes:
   `response_field`, `response_field_usage`, and `response_field_model`.
3. Both transforms populate those fields on fully successful rows.
4. Both transforms also emit mixed-success batches where some rows carry
   `response_field=None` plus `response_field_error`, but omit
   `response_field_usage` and `response_field_model`.

That means the declared output contract and the runtime row shape disagree
exactly on the rows most likely to matter operationally.

The recommended direction is:

- keep the current guaranteed-field contract,
- populate `*_usage` and `*_model` on every error-bearing row with truthful
  sentinel values (`None`),
- characterize that behavior first in unit tests,
- then add no-network probe seams that exercise the real batch assembly paths,
  not helpers that fabricate success-shaped rows.

This is the smallest truthful fix and preserves existing downstream contract
expectations.

---

## Current Contract Surface

### Shared declaration source

`src/elspeth/plugins/transforms/llm/__init__.py`

`get_llm_guaranteed_fields(response_field)` currently declares:

- `<response_field>`
- `<response_field>_usage`
- `<response_field>_model`

Both batch transforms use that helper for:

- `declared_output_fields`
- `_output_schema_config.guaranteed_fields`

So downstream DAG validation, composer previews, and schema consumers are
currently entitled to treat all three fields as contract-stable.

### Matrix

| Surface | Azure batch | OpenRouter batch |
|---|---|---|
| Declaration source | `get_llm_guaranteed_fields()` in `__init__` | `get_llm_guaranteed_fields()` in `__init__` |
| Advertised guaranteed fields | `llm_response`, `llm_response_usage`, `llm_response_model` | `llm_response`, `llm_response_usage`, `llm_response_model` |
| Fully successful row path | `_download_results()` success branch calls `populate_llm_operational_fields()` | `_process_single_row()` success branch calls `populate_llm_operational_fields()` |
| Mixed-success / error row path | Template/API/validation/content-filter rows emit `llm_response=None` and `llm_response_error` only | Template/API/transport/validation/content-filter rows emit `llm_response=None` and `llm_response_error` only |
| Fields missing on error-bearing rows today | `llm_response_usage`, `llm_response_model` | `llm_response_usage`, `llm_response_model` |
| Batch result shape | `TransformResult.success_multi(...)` with mixed success/error rows | `TransformResult.success_multi(...)` with mixed success/error rows |

### Concrete runtime row shapes

#### Azure batch

`src/elspeth/plugins/transforms/llm/azure_batch.py`

Fully successful rows currently include:

- original input fields
- `llm_response`
- `llm_response_usage`
- `llm_response_model`

Error-bearing rows currently include:

- original input fields
- `llm_response = None`
- `llm_response_error`

They currently omit:

- `llm_response_usage`
- `llm_response_model`

This is true for at least these branches in `_download_results()`:

- template rendering failures
- missing result / lost error detail rows
- per-row API errors
- invalid response structure rows
- content-filtered rows

Azure already carries `quarantined_indices` metadata in `success_reason`, which
helps the engine distinguish mixed-success outcomes, but that does not repair
the row-level guaranteed-field mismatch.

#### OpenRouter batch

`src/elspeth/plugins/transforms/llm/openrouter_batch.py`

Fully successful rows currently include:

- original input fields
- `llm_response`
- `llm_response_usage`
- `llm_response_model`

Error-bearing rows currently include:

- original input fields
- `llm_response = None`
- `llm_response_error`

They currently omit:

- `llm_response_usage`
- `llm_response_model`

This is true for at least these branches:

- template rendering failures
- HTTP status failures
- request/transport failures
- malformed JSON / malformed response structure
- null content
- non-`stop` finish reasons

Unlike Azure batch, I did not find `quarantined_indices` metadata in the
OpenRouter batch success path. That is an inference from the current source,
not a separately verified engine-level bug report, but it should be confirmed in
the future tranche because mixed-success semantics matter here too.

---

## Evidence From Existing Tests

### Azure batch

`tests/unit/plugins/llm/test_azure_batch.py` already proves:

- successful rows include operational fields such as `llm_response_usage`
- content-filtered rows return `llm_response=None` plus `llm_response_error`
- mixed-success batches return `success_multi(...)`
- mixed outcomes are surfaced through `quarantined_indices`

What it does not currently pin down is:

- error-bearing rows must still contain `llm_response_usage`
- error-bearing rows must still contain `llm_response_model`

### OpenRouter batch

`tests/unit/plugins/llm/test_openrouter_batch.py` already proves:

- successful rows include `llm_response`, `llm_response_usage`,
  and `llm_response_model`
- template/API/content-filter/truncation failures become per-row error rows
- the observed output contract includes the `_error` field when mixed rows exist

What it does not currently pin down is:

- error-bearing rows must still include `llm_response_usage`
- error-bearing rows must still include `llm_response_model`

That missing characterization is exactly why a fabricated probe helper would be
misleading evidence. The real batch assembly path still disagrees with the
declared contract.

---

## Exact Mismatch

The mismatch is not about whether successful rows preserve input fields. They
do.

The mismatch is:

- declaration says usage/model are guaranteed output fields
- mixed-success runtime says usage/model are absent on error-bearing rows

So a future `passes_through_input = True` annotation would currently be saying:

- "this transform preserves all input fields on successful output rows"
- while also silently inheriting a stronger output-schema promise that is
  presently false for error-bearing rows inside those same successful batches

That is why the correct next step is not "add a probe hook." The next step is
"repair or redesign the batch output contract, then add the probe hook."

---

## Option Evaluation

### Option 1: Keep the current guaranteed-field contract and populate `*_usage` / `*_model` on error rows with `None`

**Shape**

- keep `get_llm_guaranteed_fields()` unchanged
- keep `declared_output_fields` unchanged
- on every error-bearing row, write:
  - `llm_response = None`
  - `llm_response_usage = None`
  - `llm_response_model = None`
  - `llm_response_error = {...}`

**Why `None` is the truthful sentinel**

- `usage` is genuinely unavailable on many error paths
- `model` is semantically "the model that actually responded"
- template-rendering failures happen before any provider response exists
- some API failures happen before a usable model identifier is available

So `None` is the safest cross-path value. Reusing the configured request model
for all failure modes would blur "targeted model" and "model that actually
responded."

**Trade-offs**

- downstream DAG/schema validation:
  stable, because the current declared contract remains true after repair
- composer/output-schema previews:
  stable, because no contract shrink is introduced
- audit semantics and row-level explainability:
  improved, because rows still carry a stable field set while error detail stays
  in `*_error`
- test-path integrity:
  strong, because existing unit suites can characterize the real assembly path
  before annotation
- migration risk:
  lowest; existing downstream users depending on `*_usage` / `*_model` continue
  to see those keys

**Cost**

- touches many error-row construction sites in both batch implementations
- requires careful consistency across template failures, transport failures,
  provider failures, and partial result-download failures

### Option 2: Narrow the guaranteed-field contract for batch LLM outputs

**Shape**

- stop advertising `*_usage` / `*_model` as guaranteed for batch transforms
- leave success-path rows as they are
- allow error-bearing rows to omit those fields

**Trade-offs**

- downstream DAG/schema validation:
  weaker; downstream consumers can no longer rely on usage/model for batch LLMs
- composer/output-schema previews:
  would change what previews show as stable output for batch transforms
- audit semantics and row-level explainability:
  avoids sentinels, but at the cost of less stable operational metadata
- test-path integrity:
  still viable, but now the future tranche must also prove the contract split is
  intentional and not drift
- migration risk:
  moderate to high; this is a contract change, not a local bug fix

**Additional cost**

- `get_llm_guaranteed_fields()` is shared across the LLM family, so this option
  likely implies batch-specific field declaration logic rather than a small
  shared helper edit

### Option 3: Redesign the batch result contract so mixed-success batches are not represented as successful pass-through outputs

**Shape**

- change the runtime semantics of mixed-success batch processing
- examples:
  - fail the whole batch,
  - split success/error lanes structurally,
  - or introduce a different batch result category

**Trade-offs**

- downstream DAG/schema validation:
  potentially cleaner long-term, but requires executor semantics work
- composer/output-schema previews:
  largest blast radius because preview assumptions are currently row-shape based
- audit semantics and row-level explainability:
  could become more explicit, but only with substantial engine work
- test-path integrity:
  strongest long-term if done well, but easiest to get wrong because the change
  crosses transform, engine, and possibly UI assumptions
- migration risk:
  highest; this is behavior redesign, not contract repair

**Conclusion**

This is a valid long-term architecture direction, but it is too large and risky
for the immediate invariant tranche unless there is a separate product-level
decision to change batch failure semantics.

---

## Recommendation

Recommend **Option 1**.

Reasons:

1. It makes the existing declared contract true with the smallest behavioral
   change.
2. It keeps single-query and batch LLM output contracts aligned.
3. It avoids changing composer and downstream schema expectations mid-migration.
4. It works with existing helper semantics:
   `populate_llm_operational_fields(..., usage=None, model=None)` already
   expresses "field exists, value unavailable."
5. It lets the future invariant work focus on a real offline batch seam rather
   than combining contract redesign and probe design in one tranche.

Secondary recommendation:

- explicitly characterize whether OpenRouter batch also needs
  `quarantined_indices`-style metadata for engine correctness

That is adjacent to, but separable from, the guaranteed-field repair.

---

## Minimum Truthful No-Network Probe Seams

### OpenRouter batch

**Required runtime path**

The future probe seam must still drive:

- `process(rows, ctx)` or `_process_batch(rows, ctx)`
- per-row `_process_single_row(...)`
- final mixed-row assembly into `output_rows`
- final `TransformResult.success_multi(...)`

**Truthful seam**

Do not add a helper that directly returns a prebuilt output row.

The seam should temporarily replace the state-scoped HTTP client path, most
likely by overriding `_get_http_client(...)` on the instance and returning a
local fake client whose `post(...)` method yields deterministic offline
responses.

**Likely test entry points**

- `tests/unit/plugins/llm/test_openrouter_batch.py`
- future invariant smoke coverage in
  `tests/unit/plugins/transforms/test_forward_invariant_probes.py`
- future pass-through governance in
  `tests/invariants/test_pass_through_invariants.py`

**Characterization tests that must exist before annotation**

- mixed batch with one success row and one error row still emits
  `llm_response_usage` and `llm_response_model` on both rows
- no-network forward probe drives the real batch assembly path and preserves
  baseline input fields on the successful probe row

### Azure batch

**Required runtime path**

The future probe seam must still drive:

- `process(rows, ctx)`
- checkpoint-aware resume logic
- `_check_batch_status(...)`
- `_download_results(...)`
- per-row result assembly from synthetic Azure output/error payloads

**Truthful seam**

Do not add a helper that returns a fabricated success row.

The most credible offline path is the **resume/completed** branch, not the
submit branch:

- seed `ctx` with a synthetic `BatchCheckpointState`
- inject a local fake `_client`
- make `batches.retrieve(...)` return a completed batch
- make `files.content(...)` return synthetic JSONL output/error content
- call `process(rows, ctx)` so the real checkpoint/resume/result-assembly code
  runs unchanged

**Likely test entry points**

- `tests/unit/plugins/llm/test_azure_batch.py`
- future invariant smoke coverage in
  `tests/unit/plugins/transforms/test_forward_invariant_probes.py`
- future pass-through governance in
  `tests/invariants/test_pass_through_invariants.py`

**Characterization tests that must exist before annotation**

- mixed checkpoint-resume batch with one success row and one error row still
  emits `llm_response_usage` and `llm_response_model` on both rows
- template-error rows also carry the full guaranteed field set
- no-network forward probe uses the real completed-batch resume path and
  preserves baseline input fields on the successful probe row

**Additional note**

The submit path still matters. A separate characterization test should continue
to prove that `_submit_batch()` checkpoints batch state before raising
`BatchPendingError`, but that does not need to be the invariant probe seam.

---

## Future Tranche Checklist

### Smallest expected file set

Primary code:

- `src/elspeth/plugins/transforms/llm/azure_batch.py`
- `src/elspeth/plugins/transforms/llm/openrouter_batch.py`

Shared helpers only if needed:

- `src/elspeth/plugins/transforms/llm/__init__.py`
  only if the recommendation changes away from Option 1

Required tests:

- `tests/unit/plugins/llm/test_azure_batch.py`
- `tests/unit/plugins/llm/test_openrouter_batch.py`
- `tests/unit/plugins/transforms/test_forward_invariant_probes.py`
- `tests/invariants/test_pass_through_invariants.py`
- possibly `tests/invariants/test_transform_probe_coverage.py`
  once the classes actually join the in-scope forward set

### Characterization tests to write first

1. Azure batch:
   mixed-success rows include `llm_response_usage=None` and
   `llm_response_model=None` on error-bearing rows.
2. Azure batch:
   template-rendering failures also include the same guaranteed keys.
3. OpenRouter batch:
   mixed-success rows include `llm_response_usage=None` and
   `llm_response_model=None` on error-bearing rows.
4. OpenRouter batch:
   transport/API/content-filter failure rows also include the same guaranteed
   keys.
5. Azure batch:
   offline resume/completed seam drives `_check_batch_status()` and
   `_download_results()` without network.
6. OpenRouter batch:
   offline seam drives `_process_batch()` using a fake client, not a fabricated
   output row helper.

---

## Non-Go Conditions

Do **not** annotate either batch transform as `passes_through_input = True` if
any of the following remain true:

- mixed-success rows still omit `*_usage` or `*_model`
- the future probe helper constructs a synthetic success row directly instead of
  driving `_process_batch()` / `_download_results()`
- the offline seam still requires live credentials, DNS, vendor access, or
  optional extras not present in the base install
- Azure batch coverage bypasses checkpoint/resume/result assembly
- OpenRouter batch coverage bypasses the real per-row failure/success assembly
- composer or DAG previews no longer agree with the actual batch row contract
- engine semantics for mixed-success rows remain ambiguous after the contract
  repair, especially if OpenRouter batch still lacks a reliable quarantine
  signal

---

## Final Recommendation

1. Repair the current row contract first by writing `*_usage=None` and
   `*_model=None` onto every error-bearing batch row.
2. Add characterization tests for that repair in both batch test suites.
3. Add truthful no-network probe seams that exercise the real batch assembly
   paths.
4. Only after those two steps pass should either batch transform be considered
   for `passes_through_input = True`.

Until then, the current deferral remains correct.
