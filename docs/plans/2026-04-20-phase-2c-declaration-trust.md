# Phase 2C Declaration-Trust Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land the paired Phase 2C boundary adopters, `SourceGuaranteedFieldsContract` and `SinkRequiredFieldsContract`, on the ADR-010 framework without losing row-level attribution, without regressing source/sink audit semantics, and without collapsing the existing sink transactional backstop into the new pre-write contract layer.

**Architecture:** Phase 2C should use the existing `boundary_check` dispatch surface, but the current placeholder boundary bundle is not yet sufficient for row-attributed production violations. The implementation should refine the boundary inputs around real source/sink execution seams, run boundary checks per row rather than per batch, preserve the sink’s two-layer architecture, and land both source and sink adopters together in one PR as required by ADR-010’s paired-landing rule.

**Tech Stack:** Python, pytest, Hypothesis, mypy, ruff, Filigree, ADR-010 declaration-contract registry/dispatcher, Landscape integration tests.

**Prerequisites:**
- Work on a branch/worktree that already contains the H2/N1/N3 boundary-dispatch framework landing in code.
- Prefer the explicit declaration-contract bootstrap import surface from the Phase 2B plan; if it has not landed yet, include that prerequisite in the 2C branch before adding the first boundary adopter.
- Treat ADR-010, the current source/sink executors, and the current Filigree tickets as potentially divergent sources; verify repo reality before writing code.
- If running inside the Codex sandbox and `uv run` cannot write to the default cache, prefix test/lint/typecheck commands with `UV_CACHE_DIR=.uv-cache`.

---

## Reality Anchors

These are the current repo facts the plan assumes:

1. The `boundary_check` dispatch site exists, but it is not wired into production source or sink execution yet.
   - `run_boundary_checks(...)` is present in [declaration_dispatch.py](/home/john/elspeth/src/elspeth/engine/executors/declaration_dispatch.py:226).
   - No production source or sink path currently calls it.

2. The current placeholder `BoundaryInputs` shape is good enough for H2 wiring but not yet good enough for production row-attributed contracts.
   - It currently carries `plugin`, `node_id`, `run_id`, `static_contract`, and `rows`, but no `row_id`, no `token_id`, and no contract context ([declaration_contracts.py](/home/john/elspeth/src/elspeth/contracts/declaration_contracts.py:297)).
   - `DeclarationContractViolation` requires `row_id` and `token_id` at construction time ([declaration_contracts.py](/home/john/elspeth/src/elspeth/contracts/declaration_contracts.py:587)).

3. Source-side boundary validation cannot run before identity exists if we want normal declaration-contract audit records.
   - Valid source rows exist first as `SourceRow` values in the orchestrator loop.
   - `RowProcessor.process_row()` creates the initial token, and `_record_source_and_start_traversal()` records the source node state afterward ([processor.py](/home/john/elspeth/src/elspeth/engine/processor.py:1496), [processor.py](/home/john/elspeth/src/elspeth/engine/processor.py:1439)).

4. Sink-side boundary validation must run before schema validation if we want correct attribution.
   - `SinkExecutor.write()` currently calls `_validate_sink_input(...)` before `sink.write(...)` ([sink.py](/home/john/elspeth/src/elspeth/engine/executors/sink.py:397)).
   - `_validate_sink_input(...)` does schema validation first, then the inline transactional backstop. If the 2C layer 1 contract is called after that, missing required fields will still be attributed as a generic `PluginContractViolation`.

5. The sink’s inline Layer 2 transactional backstop already exists in code and must remain distinct from the new 2C contract.
   - `SinkTransactionalInvariantError` is already defined ([errors.py](/home/john/elspeth/src/elspeth/contracts/errors.py:856)).
   - `_validate_sink_input(...)` already documents the two-layer architecture ([sink.py](/home/john/elspeth/src/elspeth/engine/executors/sink.py:172)).

6. Source guaranteed fields are not normalized onto a runtime source attribute today.
   - The DAG builder reads producer guarantees from raw schema config.
   - Sources hold schema state via `_schema_config` / `get_schema_contract()`, but `BaseSource` and `SourceProtocol` do not currently expose a runtime `declared_guaranteed_fields`-style attribute.

7. Boundary adopters must land together.
   - ADR-010’s paired-landing rule explicitly says `SourceGuaranteedFieldsContract` and `SinkRequiredFieldsContract` land in one commit/PR, not staggered boundary adopters ([adr/010-declaration-trust-framework.md](/home/john/elspeth/docs/architecture/adr/010-declaration-trust-framework.md:257)).

8. `BoundaryInputs` already has direct property-test coverage, so a signature change is not isolated to the dispatcher unit test file.
   - `tests/property/engine/test_declaration_dispatch_properties.py` builds `BoundaryInputs` directly as a Hypothesis strategy surface.
   - If Task 1 changes the dataclass shape without updating that strategy, collection or property execution will fail before any 2C adopter logic runs.

9. Pytest bootstrap still maintains its own declaration-contract import surface instead of reusing the authoritative production bootstrap module.
   - `tests/conftest.py` currently imports each contract-defining executor module individually.
   - If 2C adds new registered contract modules and this test bootstrap surface is not updated in the same PR, registry-dependent tests can fail under pytest workers even when production bootstrap is correct.

10. Existing manifest tests still hard-code the current five-contract Phase 2B world.
    - `tests/unit/engine/test_declared_output_fields_contract.py` asserts `len(EXPECTED_CONTRACT_SITES) == 5`.
    - `tests/unit/engine/test_orchestrator_registry_bootstrap.py` asserts the same exact manifest size and name set.

11. Existing failsink fixtures assume today’s pre-2C world where no boundary contract reads sink declaration attributes on those doubles.
    - `tests/unit/engine/test_sink_executor_diversion.py` creates failsink `MagicMock`s without setting `declared_required_fields`.
    - `tests/property/engine/test_sink_executor_diversion_properties.py` does the same for Hypothesis-generated failsink scenarios.
    - Once `SinkRequiredFieldsContract.applies_to()` reads sink declarations directly, those helpers must set `declared_required_fields = frozenset()` explicitly or they will become truthy mock hazards.

12. Source protocol/docs/tests currently expose schema-contract access, but not a runtime guaranteed-fields declaration surface.
    - `SourceProtocol`, `BaseSource`, `docs/contracts/plugin-protocol.md`, and the source protocol/base tests all reflect the current schema-contract-only surface.
    - Adding `declared_guaranteed_fields` is therefore a protocol/doc/test change, not just a concrete-source constructor change.

## Cold-Start Implementor Briefing

This section is here because the implementor should assume zero historical context. Do not rely on prior discussion threads, ticket comments, or implied intent; use the following as the working brief unless repo reality changes.

**Read this code in this order before editing anything:**

1. [ADR-010](/home/john/elspeth/docs/architecture/adr/010-declaration-trust-framework.md:1) for the paired-landing rule and the four dispatch sites.
2. [declaration_contracts.py](/home/john/elspeth/src/elspeth/contracts/declaration_contracts.py:297) for `BoundaryInputs`, `BoundaryOutputs`, `DeclarationContractViolation`, and `EXPECTED_CONTRACT_SITES`.
3. [declaration_dispatch.py](/home/john/elspeth/src/elspeth/engine/executors/declaration_dispatch.py:226) for the current `run_boundary_checks(...)` wrapper and `_dispatch(...)` behavior.
4. [processor.py](/home/john/elspeth/src/elspeth/engine/processor.py:1496) and [processor.py](/home/john/elspeth/src/elspeth/engine/processor.py:1543) for the `process_row()` vs `process_existing_row()` split.
5. [sink.py](/home/john/elspeth/src/elspeth/engine/executors/sink.py:153) and [sink.py](/home/john/elspeth/src/elspeth/engine/executors/sink.py:556) for primary-sink and failsink call posture.
6. [declaration_contract_bootstrap.py](/home/john/elspeth/src/elspeth/engine/executors/declaration_contract_bootstrap.py:1) and [tests/conftest.py](/home/john/elspeth/tests/conftest.py:33) for the production-vs-pytest registry bootstrap surfaces.
7. [plugin_protocols.py](/home/john/elspeth/src/elspeth/contracts/plugin_protocols.py:56), [base.py](/home/john/elspeth/src/elspeth/plugins/infrastructure/base.py:1), and [text_source.py](/home/john/elspeth/src/elspeth/plugins/sources/text_source.py:65) for the source runtime declaration surface and the observed-text rewrite edge case.

**Baseline already verified on April 20, 2026:**

- The dispatcher/registry/property baseline was green:
  - `tests/unit/engine/test_declaration_dispatch.py`
  - `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
  - `tests/property/engine/test_declaration_dispatch_properties.py`
- The likely fallout suites were also green:
  - `tests/unit/engine/test_executors.py -k "missing_required_field or failsink_validation_does_not_emit_coalesce_merge_annotation"`
  - `tests/unit/engine/test_sink_executor_diversion.py`
  - `tests/property/engine/test_sink_executor_diversion_properties.py`
  - `tests/unit/plugins/test_protocols.py`
  - `tests/unit/plugins/test_base_source_contract.py`
  - `tests/unit/contracts/source_contracts/test_source_protocol.py`
  - `tests/unit/engine/test_processor_pipeline_row.py`
  - `tests/unit/engine/test_processor.py -k "process_existing_row or process_row"`
  - `tests/integration/pipeline/orchestrator/test_resume_guardrails.py`
- In total, 105 targeted tests passed during the investigation that produced this plan. Treat that as a starting baseline, not as permission to skip rerunning them.

**Decisions already made; do not re-litigate these unless code reality changed:**

- Keep one public `boundary_check` dispatcher surface unless the refined bundle becomes provably unworkable.
- Source boundary validation must use real `row_id` and `token_id`, which means it cannot happen before token creation.
- Resume processing must not re-run source boundary validation.
- Sink Layer 1 attribution must happen before `_validate_sink_input(...)` and before external sink I/O.
- Sink Layer 2 (`SinkTransactionalInvariantError`) stays in place even after the new Layer 1 contract lands.
- Source runtime guarantees should be surfaced as an attribute, not by reaching into private `_schema_config` inside the contract body.
- No new row-level `logger`/`structlog` emissions should be added for these contract outcomes; the audit trail is the source of truth.

## Acceptance Matrix

This is the behavior matrix the implementation should make true. If code or tests disagree with this matrix, update the implementation or the plan before merging.

| Scenario | Expected signal | Notes |
| ---- | ---- | ---- |
| Valid source row omits a declared guaranteed field | `SourceGuaranteedFieldsViolation` | Tier 1; row-attributed; source node state should end as FAILED, not disappear. |
| Valid source row includes the payload key but the emitted row contract omits it | `SourceGuaranteedFieldsViolation` | Runtime observation is contract intersection, not payload membership alone. |
| Quarantined source row | No source boundary check | Quarantined rows are already on the source validation path; 2C does not double-penalize them. |
| Resume path (`process_existing_row`) | No source boundary check | Resume reuses existing source-row provenance; it should not mint new source-boundary failures. |
| Primary sink row misses a required field | `SinkRequiredFieldsViolation` before schema validation | Attribution belongs to Layer 1, not to generic schema validation. |
| Failsink enriched row misses a failsink-required field | `SinkRequiredFieldsViolation` | Failsinks are still sink boundaries. |
| Row passes Layer 1 but required field is absent at commit boundary | `SinkTransactionalInvariantError` | This is the Layer 2 divergence/backstop case. |
| Non-boundary transforms / non-opted-in sinks | No behavior change | The 2C landing should not perturb existing non-boundary sites. |

## Recommended Boundary Shape Decision

Before implementing either contract, make one narrow framework refinement:

**Recommendation:** keep a single `boundary_check` dispatch surface, but refine the boundary input shape to support row-level identity and contract context. Do not keep the current rows-only placeholder if it forces synthetic row identities or non-standard violation classes.

The minimum useful payload for a production boundary check is:

```python
@dataclass(frozen=True, slots=True)
class BoundaryInputs:
    plugin: Any
    node_id: str
    run_id: str
    row_id: str
    token_id: str
    static_contract: frozenset[str]
    row_data: Any
    row_contract: Any | None = None
```

This plan assumes `run_boundary_checks(...)` will be invoked once per row, not once per batch, on both source and sink paths. That keeps the standard `DeclarationContractViolation` shape honest and avoids inventing aggregate row identities.

If the implementation team concludes the single refined bundle is still too awkward, split it into `SourceBoundaryInputs` and `SinkBoundaryInputs` in the same PR. Do not leave the placeholder shape half-refined.

## Guardrails and Non-Goals

- Do not weaken the current closed-set manifest discipline around `EXPECTED_CONTRACT_SITES`; update the manifest, bootstrap, and drift tests in the same change.
- Do not add row-level logging for source or sink declaration failures. Per the logging policy, those outcomes belong in Landscape-backed audit artifacts, not in transient logs.
- Do not use `hasattr(...)` to probe optional contract helper methods or plugin declaration surfaces. This codebase bans `hasattr` in these paths because it can mask descriptor failures.
- Do not bypass production code paths in new integration tests. Use existing orchestrator/bootstrap paths instead of manually assembling impossible registry or DAG state.
- Do not merge the new sink Layer 1 contract into `_validate_sink_input(...)`; Layer 2 must remain a separate transactional backstop.
- Do not generalize beyond 2C. This plan is not permission to redesign all dispatch bundles or revisit unrelated transform contracts in the same PR.

## Task 0: Verify Prerequisites and Reconcile Board vs Workspace

**Files:**
- Read: `docs/architecture/adr/010-declaration-trust-framework.md`
- Read: `src/elspeth/contracts/declaration_contracts.py`
- Read: `src/elspeth/engine/executors/declaration_dispatch.py`
- Read: `src/elspeth/engine/processor.py`
- Read: `src/elspeth/engine/executors/sink.py`
- Read: `tests/conftest.py`
- Test: `tests/unit/engine/test_declaration_dispatch.py`
- Test: `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
- Test: `tests/property/engine/test_declaration_dispatch_properties.py`
- Test: `tests/unit/engine/test_executors.py`
- Test: `tests/unit/engine/test_sink_executor_diversion.py`
- Test: `tests/property/engine/test_sink_executor_diversion_properties.py`
- Test: `tests/unit/plugins/test_protocols.py`
- Test: `tests/unit/plugins/test_base_source_contract.py`
- Test: `tests/unit/contracts/source_contracts/test_source_protocol.py`
- Test: `tests/unit/engine/test_processor_pipeline_row.py`
- Test: `tests/unit/engine/test_processor.py`
- Test: `tests/integration/pipeline/orchestrator/test_resume_guardrails.py`

**Step 1: Run framework smoke tests first**

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_declaration_dispatch.py \
  tests/unit/engine/test_orchestrator_registry_bootstrap.py \
  tests/property/engine/test_declaration_dispatch_properties.py
```

**Expected result:** all tests pass before any 2C work starts.

**Step 2: Snapshot the adjacent fallout suites before changing code**

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_executors.py -k "missing_required_field or failsink_validation_does_not_emit_coalesce_merge_annotation" \
  tests/unit/engine/test_sink_executor_diversion.py \
  tests/property/engine/test_sink_executor_diversion_properties.py \
  tests/unit/plugins/test_protocols.py \
  tests/unit/plugins/test_base_source_contract.py \
  tests/unit/contracts/source_contracts/test_source_protocol.py \
  tests/unit/engine/test_processor_pipeline_row.py \
  tests/unit/engine/test_processor.py -k "process_existing_row or process_row" \
  tests/integration/pipeline/orchestrator/test_resume_guardrails.py
```

**Expected result:** all tests pass before any 2C work starts, giving a known-good pre-2C baseline for the exact suites most likely to move.

**Step 3: Reconcile tracked siblings with current code**

- Confirm whether `elspeth-5fc876138d` should already be closed, since the `SinkTransactionalInvariantError` class and sink-layer comments are present in code.
- Confirm whether the explicit contract bootstrap from the 2B plan has landed. If not, make it a prerequisite task in this branch.
- Confirm whether pytest should switch from `tests/conftest.py`'s enumerated contract imports to `import elspeth.engine.executors.declaration_contract_bootstrap` in this same PR. Prefer the authoritative bootstrap surface unless a concrete pytest-only reason blocks it.
- Confirm the current state of `elspeth-425047a599` and `elspeth-60890a7388`; if the code already satisfies their acceptance criteria, update Filigree before starting the 2C paired adopter PR.

**Definition of Done:**
- [ ] The branch proves boundary dispatch infrastructure is present and green.
- [ ] Filigree drift that would confuse 2C implementation is identified up front.
- [ ] The paired landing will not be blocked by already-landed-but-open prerequisites.
- [ ] The implementor has a fresh local baseline for every suite this plan expects to move.

## Task 1: Refine Boundary Inputs Around Row-Level Identity

This task should land before either adopter contract body is written.

**Files:**
- Modify: `src/elspeth/contracts/declaration_contracts.py`
- Modify: `src/elspeth/engine/executors/declaration_dispatch.py`
- Modify: `tests/unit/engine/test_declaration_dispatch.py`
- Modify: `tests/property/engine/test_declaration_dispatch_properties.py`
- Create: `tests/unit/engine/test_boundary_dispatch_inputs.py`

**Implementation goals:**
- Refine the boundary bundle to carry real row identity.
- Keep `run_boundary_checks(...)` as the public wrapper.
- Preserve the shared `_dispatch(...)` helper and per-site registry behavior.

**Recommended implementation:**
- Add `row_id` and `token_id` to `BoundaryInputs`.
- Replace the plural placeholder with either `row_data` + `row_contract`, or keep `rows` only if it is always a one-element tuple and still carries separate identity fields.
- Update the boundary bundle docstrings so they describe the actual 2C call posture, not the H2 placeholder.

**Concrete file-local notes:**
- Update the example helpers that currently construct boundary bundles directly, especially the `_example_boundary()` fixture path in `tests/property/engine/test_declaration_dispatch_properties.py`.
- Keep the public dispatcher API boring: `run_boundary_checks(inputs, outputs)` should still just delegate into `_dispatch(...)`.
- If the bundle shape changes, update both the validation semantics and the error messages so failures stay obvious when Hypothesis hits them.

**Tests to write first:**
- Boundary bundle rejects missing identity fields.
- `run_boundary_checks(...)` dispatches only contracts marked for `boundary_check`.
- Existing non-boundary contracts remain unaffected.

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_declaration_dispatch.py \
  tests/unit/engine/test_boundary_dispatch_inputs.py \
  tests/property/engine/test_declaration_dispatch_properties.py
```

**Definition of Done:**
- [ ] The boundary input shape can support normal `DeclarationContractViolation` construction.
- [ ] The boundary site remains a single dispatcher surface.
- [ ] The refined shape is documented as production truth, not placeholder speculation.

## Task 2: Normalize Source Guarantees Onto a Runtime Attribute

Phase 2C should not make `SourceGuaranteedFieldsContract` reach into private source internals or raw config dicts.

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/base.py`
- Modify: `src/elspeth/contracts/plugin_protocols.py`
- Modify: `src/elspeth/plugins/sources/csv_source.py`
- Modify: `src/elspeth/plugins/sources/json_source.py`
- Modify: `src/elspeth/plugins/sources/dataverse.py`
- Modify: `src/elspeth/plugins/sources/azure_blob_source.py`
- Modify: `src/elspeth/plugins/sources/text_source.py`
- Modify: `docs/contracts/plugin-protocol.md`
- Modify: `tests/unit/plugins/test_protocols.py`
- Modify: `tests/unit/plugins/test_base_source_contract.py`
- Modify: `tests/unit/contracts/source_contracts/test_source_protocol.py`
- Create: `tests/unit/plugins/sources/test_declared_guaranteed_fields.py`

**Recommendation:**

Add a runtime source attribute parallel to sink required fields:

```python
declared_guaranteed_fields: frozenset[str] = frozenset()
```

Populate it at source construction time from the source’s effective schema-config guarantees, after any source-specific schema rewrites such as the TextSource observed-mode heuristic.

Prefer a small `BaseSource` helper that records the runtime declaration from a resolved `SchemaConfig`, then call that helper from each concrete source that owns `_schema_config`. `NullSource` should continue to expose the base-class empty frozenset with no special-case override.

**Concrete constructor notes:**
- `TextSource` is the one source with a known pre-existing schema rewrite. Derive `declared_guaranteed_fields` from the post-rewrite `schema_config`, not the raw input dict.
- `csv_source.py`, `json_source.py`, `dataverse.py`, and `azure_blob_source.py` all currently assign `_schema_config` during construction; wire the helper there so the runtime attribute is normalized once.
- Keep the DAG builder and runtime surface aligned, but do not make the runtime contract read raw config dicts just because the DAG builder still does.

**Why this matters:**
- The DAG builder can keep using raw config.
- The runtime contract gets a stable, attribute-based surface.
- The source contract body stays simple and does not need plugin-specific knowledge.

**Tests to write first:**
- Fixed/flexible/observed sources populate the attribute from effective guarantees.
- TextSource’s heuristic rewrite is reflected in the runtime attribute.
- Sources with no explicit guarantees expose an empty set.
- Protocol/example source stubs satisfy the updated `SourceProtocol` surface.
- BaseSource exposes a stable empty default before any concrete source populates the declaration.

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/plugins/sources/test_declared_guaranteed_fields.py \
  tests/unit/plugins/test_protocols.py \
  tests/unit/plugins/test_base_source_contract.py \
  tests/unit/contracts/source_contracts/test_source_protocol.py
```

**Definition of Done:**
- [ ] `BaseSource` / `SourceProtocol` expose a runtime guarantee attribute.
- [ ] Source runtime declarations are derived after any source-specific config rewrite.
- [ ] Future source boundary contracts no longer need to read private `_schema_config` state.

## Task 3: ADR and Implementation for `SourceGuaranteedFieldsContract`

This contract should land in the same PR as Task 4, but its ADR and tests can be developed in parallel once Task 1 and Task 2 are done.

**Files:**
- Create: `docs/architecture/adr/016-source-guaranteed-fields-contract.md`
- Modify: `src/elspeth/contracts/declaration_contracts.py`
- Create: `src/elspeth/engine/executors/source_guaranteed_fields.py`
- Modify: `src/elspeth/contracts/errors.py`
- Modify: `src/elspeth/engine/processor.py`
- Modify: `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
- Modify: `tests/invariants/test_contract_negative_examples_fire.py`
- Modify: `tests/unit/engine/test_processor.py`
- Modify: `tests/unit/engine/test_processor_pipeline_row.py`
- Modify: `tests/integration/pipeline/orchestrator/test_resume_guardrails.py`
- Create: `tests/unit/engine/test_source_guaranteed_fields_contract.py`
- Create: `tests/integration/audit/test_source_guaranteed_fields_roundtrip.py`

**ADR decisions to record:**
- `SourceGuaranteedFieldsViolation` should be Tier 1. A source that lies about guaranteed fields poisons downstream propagation and audit lineage.
- The runtime observation should be the intersection of emitted row payload keys and the emitted row contract fields, not payload keys alone.
- The contract should apply only to valid source rows, never quarantined rows.
- Resume runs should not re-run source boundary validation, because the source boundary was already crossed in the original run.
- Source-boundary failures that occur after token creation must still record a terminal `FAILED` token outcome. A failed source `node_state` alone is not sufficient evidence because `token_outcomes` remains the authoritative terminal-state record.

**Suggested runtime invariant:**

```python
runtime_contract_fields = frozenset(fc.normalized_name for fc in inputs.row_contract.fields)
runtime_payload_fields = frozenset(inputs.row_data.keys())
runtime_observed = runtime_contract_fields & runtime_payload_fields
missing = inputs.static_contract - runtime_observed
```

**Call-site recommendation:**
- Do not run this in the orchestrator loop before token creation.
- Create the token first in `RowProcessor.process_row()`, then run the boundary check using real `row_id` and `token_id`. If it passes, record the source node state as `COMPLETED` and continue traversal normally. If it fires, record a terminal `FAILED` token outcome plus a `FAILED` source node state before stopping traversal.
- Refactor `_record_source_and_start_traversal()` or add a guarded helper so source-boundary failures still produce a source node state rather than disappearing before audit recording.

**Implementation notes:**
- Update `EXPECTED_CONTRACT_SITES` in `declaration_contracts.py` for both new boundary adopters under `boundary_check` in the same commit as the paired adopter landing. Do not stage one boundary manifest entry without the other.
- `process_row()` and `process_existing_row()` currently share `_record_source_and_start_traversal()`. Do not insert source-boundary validation unconditionally inside that shared helper unless the helper grows an explicit resume-safe guard; otherwise resume will start re-validating source boundaries.
- A safe shape is either:
  - a new helper used only by `process_row()` for `create token -> run source boundary -> record source node state -> start traversal`, or
  - a shared helper with an explicit `run_source_boundary: bool` parameter and the resume path passing `False`.
- The source boundary path should use the row contract already attached to `SourceRow.valid(...)`; do not try to reconstruct source contracts from payload keys.
- When the source boundary contract fires, record a terminal `FAILED` token outcome with `error_hash` before re-raising. Do not rely on source `node_states` alone; the token-outcome contract still requires exactly one terminal outcome per token.
- Row-level failure evidence should come from the violation and node-state recording. Do not add logger-based shadow reporting.

**Red tests:**
- Valid source row payload lacks a declared guaranteed field -> `SourceGuaranteedFieldsViolation`.
- Source row payload contains the key but the emitted contract omits it -> violation.
- Quarantined source rows do not invoke the contract.
- Resume path does not invoke the contract.
- `process_row()` records a FAILED terminal token outcome and FAILED source node state when the source boundary contract fires after token creation.
- `process_existing_row()` / resume path does not re-run source boundary validation.

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_source_guaranteed_fields_contract.py \
  tests/integration/audit/test_source_guaranteed_fields_roundtrip.py \
  tests/unit/engine/test_processor_pipeline_row.py \
  tests/unit/engine/test_processor.py -k "process_existing_row or process_row" \
  tests/integration/pipeline/orchestrator/test_resume_guardrails.py
```

**Definition of Done:**
- [ ] ADR-016 lands and states the violation class, payload schema, and source-side call posture.
- [ ] Source boundary violations are row-attributed and source-node-attributed in Landscape.
- [ ] Source boundary violations leave no token without a terminal outcome.
- [ ] Valid source rows proceed unchanged when the guarantee contract passes.

## Task 4: ADR and Implementation for `SinkRequiredFieldsContract`

This contract must be implemented together with Task 3 in the same PR.

**Files:**
- Create: `docs/architecture/adr/017-sink-required-fields-contract.md`
- Create: `src/elspeth/engine/executors/sink_required_fields.py`
- Modify: `src/elspeth/contracts/errors.py`
- Modify: `src/elspeth/engine/executors/sink.py`
- Modify: `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
- Modify: `tests/invariants/test_contract_negative_examples_fire.py`
- Modify: `tests/unit/engine/test_executors.py`
- Modify: `tests/unit/engine/test_sink_executor_diversion.py`
- Modify: `tests/property/engine/test_sink_executor_diversion_properties.py`
- Create: `tests/unit/engine/test_sink_required_fields_contract.py`
- Create: `tests/integration/audit/test_sink_required_fields_roundtrip.py`

**ADR decisions to record:**
- `SinkRequiredFieldsViolation` should be Tier 1, as the framework-owned Layer 1 intent validation.
- Layer 1 and Layer 2 must stay distinct:
  - Layer 1: `SinkRequiredFieldsViolation` via boundary dispatch.
  - Layer 2: `SinkTransactionalInvariantError` via the existing inline backstop.
- The contract should run for both primary sink writes and failsink writes, because failsinks are still sink boundaries and the current Layer 2 backstop already covers both paths.

**Recommended call posture:**
- Run `run_boundary_checks(...)` per token/row before `_validate_sink_input(...)` in both:
  - the primary sink path in `SinkExecutor.write()`
  - the failsink path before `failsink.write(...)`
- Keep `_validate_sink_input(...)` in place as the Layer 2 backstop.
- Do not delete the inline transactional assertion.

**Implementation notes:**
- Add the sink-side payload TypedDict and `@tier_1_error` violation class in `src/elspeth/contracts/errors.py`; do not introduce `SinkRequiredFieldsViolation` ad hoc inside the executor module.
- In the primary sink path, build boundary inputs from each original token (`token_id`, `row_id`, `token.row_data.to_dict()`, `token.row_data.contract`) rather than from the merged batch contract stored on `ctx.contract`.
- In the failsink path, boundary validation should run against the enriched row that will actually be written to the failsink, not the original primary-sink row.
- Existing test doubles are asymmetric today: the primary sink doubles already tend to set `declared_required_fields = frozenset()`, but the failsink doubles do not. Patch the failsink doubles explicitly rather than assuming the whole test surface is already protected.
- Keep contract-attribution ordering strict: boundary dispatch first, then `_validate_sink_input(...)`, then external `write()` / `flush()`.

**Why call it before `_validate_sink_input(...)`:**
- If the row is missing a sink-required field, `input_schema.model_validate(...)` can otherwise raise first and mask the declaration-contract attribution.
- Layer 1 must own missing-required-field attribution; Layer 2 remains the “state diverged before commit” backstop.

**Suggested contract surface:**
- Use the sink’s existing `declared_required_fields`.
- Prefer row payload membership as the primary predicate, with optional row-contract context only for richer error messaging.
- Any sink-like test double used on the write path must set `declared_required_fields` explicitly. Do not rely on missing `MagicMock` attributes evaluating false.

**Red tests:**
- Primary sink path: missing required field -> `SinkRequiredFieldsViolation` before schema validation.
- Failsink path: enriched row missing a failsink-required field -> `SinkRequiredFieldsViolation`.
- Divergence case after Layer 1 passes still reaches `SinkTransactionalInvariantError` from the inline backstop.
- Existing executor tests that currently expect `PluginContractViolation` on missing required fields are updated to assert the new Layer 1 attribution where appropriate.
- Existing failsink helper fixtures keep non-opting-in sinks/failsinks inert by setting `declared_required_fields = frozenset()` explicitly.

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_sink_required_fields_contract.py \
  tests/integration/audit/test_sink_required_fields_roundtrip.py \
  tests/unit/engine/test_executors.py -k "missing_required_field or failsink_validation_does_not_emit_coalesce_merge_annotation" \
  tests/unit/engine/test_sink_executor_diversion.py \
  tests/property/engine/test_sink_executor_diversion_properties.py
```

**Definition of Done:**
- [ ] ADR-017 lands and states the two-layer sink architecture explicitly.
- [ ] Layer 1 is invoked before schema validation and before external sink I/O.
- [ ] Primary sink and failsink paths are both covered.
- [ ] Layer 2 remains intact and distinguishable in audit records.

## Task 5: Paired-Landing Verification and Filigree Cleanup

**Files:**
- Modify: `tests/invariants/test_contract_negative_examples_fire.py`
- Modify: `tests/conftest.py`
- Modify: `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py`
- Modify: `tests/unit/engine/test_declared_output_fields_contract.py`
- Modify: `tests/unit/engine/test_declaration_contract_bootstrap_drift.py` only if its assumptions about the bootstrap import surface need updating
- Modify: `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
- Modify: `tests/unit/scripts/cicd/test_enforce_contract_manifest.py` only if scanner fixtures or expected manifest examples need updating
- Modify: `tests/performance/benchmarks/test_cross_check_overhead.py`
- Modify: `docs/architecture/adr/010-declaration-trust-framework.md` only if adoption-state text needs factual updates
- Modify: relevant Filigree issues with close notes and dependency cleanup

**Verification checklist:**
- Both 2C contracts are in `EXPECTED_CONTRACT_SITES` under `boundary_check`.
- Both contracts have `negative_example()` and `positive_example_does_not_apply()`.
- Both contracts have end-to-end Landscape round-trip coverage.
- Source boundary checks are skipped on resume.
- Sink boundary checks run on primary and failsink paths.
- Pytest registry bootstrap is aligned with the authoritative production bootstrap surface.
- Manifest-count assertions that were Phase 2B-specific now reflect the seven-contract Phase 2C world.
- Bootstrap-drift and manifest-scanner tests still pass after adding the two new modules/sites.
- The benchmark still passes with the two new boundary adopters registered.

Run:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/invariants/test_contract_negative_examples_fire.py \
  tests/invariants/test_contract_non_fire.py \
  tests/unit/engine/test_declared_output_fields_contract.py \
  tests/unit/engine/test_declaration_contract_bootstrap_drift.py \
  tests/unit/engine/test_orchestrator_registry_bootstrap.py \
  tests/integration/audit/test_declaration_contract_landscape_roundtrip.py \
  tests/unit/scripts/cicd/test_enforce_contract_manifest.py \
  tests/performance/benchmarks/test_cross_check_overhead.py
```

Then run:

```bash
uv run ruff check src tests
uv run mypy src
```

**Definition of Done:**
- [ ] Both 2C adopters land together and are mechanically registered.
- [ ] The invariant harness and Landscape round-trip tests both cover them.
- [ ] Filigree reflects the paired landing and no longer treats the source/sink contracts as separate pending boundary subtypes.
- [ ] All baseline suites from the April 20, 2026 investigation have been rerun after the final implementation, not just the brand-new contract tests.

## Recommended Landing Order

1. Task 0: verify prerequisites and reconcile board/workspace drift.
2. Task 1: refine boundary inputs around row-level identity.
3. Task 2: normalize source guarantees onto a runtime attribute.
4. Task 3 and Task 4: land `SourceGuaranteedFieldsContract` and `SinkRequiredFieldsContract` together in one PR.
5. Task 5: cross-cutting verification, adoption-state updates, and Filigree cleanup.

## Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Boundary bundle stays rows-only and cannot support standard declaration-violation identity | Refine boundary inputs before writing adopter bodies; do not fake row IDs. |
| Source boundary validation runs before token creation and loses audit attribution | Run source boundary checks after token creation and integrate them with source node-state recording. |
| Sink required-field failures are still attributed as schema validation bugs | Invoke Layer 1 boundary checks before `_validate_sink_input(...)`. |
| Layer 1 and Layer 2 sink checks collapse back into one signal | Keep `SinkRequiredFieldsViolation` and `SinkTransactionalInvariantError` distinct and test both paths. |
| Failsink path drifts from primary sink path | Run boundary checks in both primary and failsink write flows. |
| Existing failsink/property helpers accidentally opt into Layer 1 because `MagicMock` attributes are truthy | Set `declared_required_fields = frozenset()` explicitly on sink/failsink doubles used by write-path tests. |
| Source runtime guarantees drift from DAG-view guarantees | Normalize source guarantees into a runtime attribute after any source-specific schema rewrite. |
| Source protocol/docs drift from runtime code | Update `SourceProtocol`, `BaseSource`, protocol docs, and protocol/base tests in the same task as the runtime attribute. |
| Pytest bootstrap drifts from production bootstrap and misses newly registered contracts | Prefer importing `elspeth.engine.executors.declaration_contract_bootstrap` from `tests/conftest.py`, or update both surfaces in the same commit with drift tests green. |
| Manifest guards fail late because Phase 2B tests still assert five contracts | Update the hard-coded manifest-count tests as part of the same PR that adds the two 2C contracts. |
| Resume path accidentally re-validates source boundary conditions | Add explicit resume-path regression tests and document that source boundary validation is single-run only. |

## Execution Notes

- Land both 2C adopters in one PR. The paired-landing rule is architectural, not cosmetic.
- Keep each adopter self-contained inside that PR: ADR, violation class, contract module, call-site wiring, manifest update, negative-example coverage, round-trip test, and Filigree note.
- Do not treat “new contract tests pass” as sufficient. The exit gate for this plan includes adjacent fallout suites: boundary property tests, source protocol/base tests, processor/resume tests, sink executor tests, failsink diversion tests, bootstrap-drift tests, and manifest-count tests.
- Prefer collapsing test bootstrap onto the authoritative production bootstrap surface rather than maintaining a second hand-written import list in `tests/conftest.py`.
- If boundary-shape refinement grows beyond the “identity + contract context” change, pause and amend ADR-010 explicitly rather than letting the production code become the de facto spec.
- If the implementation is done inside Codex, keep the command transcripts in the PR description or close-out note. The next agent will not inherit shell history, and the specific green suites matter for 2C signoff.
