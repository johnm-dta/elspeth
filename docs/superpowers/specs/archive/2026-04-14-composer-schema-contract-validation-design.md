# Composer Schema Contract Validation

**Date:** 2026-04-14
**Status:** Draft
**Triggered by:** False-positive pipeline completion — assistant reported a pipeline as complete when the runtime validator later rejected it due to unsatisfied schema contracts.

## Problem

The composer's `preview_pipeline` reports `is_valid: true` for pipelines that fail at runtime with `GraphValidationError: Schema contract violation`. The gap: `CompositionState.validate()` checks topology and structural completeness but never checks schema contracts (producer guaranteed fields vs consumer required fields).

A simple text source + value_transform pipeline demonstrated the failure: the transform declared `required_input_fields: ["text"]`, the source used `observed` schema with no guaranteed fields, the composer said "valid", and the runtime rejected it. Under the proposed fix, that exact text-source shape becomes a narrow accepted case because the shared observed-text rule makes the guarantee explicit on both sides instead of leaving the two validators to drift.

The bug has two halves:
1. **Tooling**: The composer doesn't validate schema contracts.
2. **Skill**: The assistant has no rule requiring field-contract verification before declaring completion.

There is also no mechanical export backstop today. The LLM-facing YAML export surfaces (`composer_mcp`'s `generate_yaml` tool and `GET /api/sessions/{id}/state/yaml`) currently call the pure serializer directly, so they will happily emit YAML for an invalid composition state unless the caller obeys the soft preview rule.

## Scope

### In scope
- Schema contract validation in `CompositionState.validate()`
- Edge contract data exposed in `preview_pipeline` response
- Shared observed-text guarantee rule for deterministic output
- Skill patch for `pipeline_composer.md`

### Out of scope
- Type compatibility checking (requires Pydantic schema instantiation — stays in runtime)
- Transform output field tracking (composer doesn't know what fields a transform adds)
- Full DAG construction in the composer (heavyweight, breaks the pure-function design)

## Design

### 1. Schema Contract Validation in `state.validate()`

**File:** `src/elspeth/web/composer/state.py`

Add validation pass 9 (error-level) to `CompositionState.validate()`.

#### Connection chain resolution

Build a producer map from the connection-field chain:

```
source.on_success = "step_a"  →  producer_map["step_a"] = ProducerEntry(...)
node(input="step_a").on_success = "step_b"  →  producer_map["step_b"] = ProducerEntry(...)
```

For each consumer node, look up the producer for `node.input`. Gates with routes produce data for multiple downstream connection points — each route target gets the same guaranteed fields as the gate's producer (gates are schema-preserving, they don't add or remove fields).

Use a small `ProducerEntry` `NamedTuple` (for example `producer_id`, `plugin_name`, `options`) as the `producer_map` value type. Do not use anonymous positional tuples here; `_walk_to_real_producer(...)` and the downstream contract checks need named fields to stay readable and auditable.

Build `node_by_id = {node.id: node for node in nodes}` once at the top of the pass and use a single `_walk_to_real_producer(...)` helper for both the node loop and the sink loop. The walk-back policy is part of the contract design now, not incidental control flow: both loops must share the same gate traversal, coalesce skip warning, and termination behavior.

Before the composer uses that producer map for any contract walk, it must validate the connection namespace and fail closed on ambiguity. The runtime builder already rejects invalid declarative wiring before graph construction, and the contract pass must not paper over the same class of errors with dict overwrite semantics. At minimum, the composer-side contract pass must emit blocking errors and suppress `edge_contracts` when any of these are present:

- duplicate producers for the same connection name
- duplicate consumers for the same connection name (fan-out without an explicit gate)
- connection names that overlap sink names

This is not optional hygiene. If `producer_map["x"] = ...` silently overwrites an earlier producer, the LLM sees false contract telemetry on an invalid draft, which is worse than no telemetry at all.

#### Contract check per edge

For each producer→consumer pair:

1. **Producer guaranteed fields**: Use a shared raw-config contract helper from `elspeth.contracts.schema` to parse `producer.options["schema"]` and compute effective guarantees. The composer must not carry a separate copy of the runtime's guarantee semantics.

2. **Consumer required fields**:
   - For nodes/transforms/aggregations, use the shared helper that mirrors runtime `required_input_fields` / explicit `schema.required_fields` semantics: top-level `required_input_fields` first, then aggregation-nested `options["required_input_fields"]` for wrapped runtime configs, then explicit `schema.required_fields`.
   - For sinks, use the shared helper that mirrors sink initialisation by applying `SchemaConfig.get_effective_required_fields()` to the sink schema. This is intentionally stricter than reading only explicit `schema.required_fields`.

3. **Compare**: `missing = required - guaranteed`. If non-empty, emit error.

#### Shared contract semantics

The composer and runtime are not intended to evolve separate interpretations of schema contracts. Before adding the new composer pass:

1. Extract raw-config contract helpers into `elspeth.contracts.schema` (or an adjacent L0 contracts module if that proves cleaner).
2. Update `ExecutionGraph` methods that read raw config to delegate to those helpers.
3. Have `CompositionState.validate()` call the same helpers.

At minimum the shared layer must define:

- producer guarantee extraction from raw plugin config
- node consumer requirement extraction from raw plugin config, including the runtime aggregation wrapper shape
- sink consumer requirement extraction from raw sink config using `SchemaConfig.get_effective_required_fields()`
- the closed-list observed-text deterministic guarantee rule

`state.py` remains responsible for connection-chain walking and `EdgeContract` reporting. The shared helper layer owns the field-contract semantics.

#### Guarantee propagation through transforms

A transform node's guaranteed fields are derived from its own `options["schema"]` if present. If a transform has no schema config, it makes no guarantee — downstream consumers see an empty guarantee set for that producer. This is conservative: the composer doesn't know what fields a transform adds or removes, so it only trusts explicit declarations.

In practice this means the contract check is most valuable for the source→first-consumer edge, where the source's schema is the authoritative declaration. For deeper edges, the runtime validator fills the gap.

#### Text source heuristic (closed list — text only)

When producer is a source with `plugin: "text"` and `options["column"]` is set, and the parsed schema is `mode == "observed"` with no explicit `guaranteed_fields`, infer `{column}` as a guaranteed field for contract comparison. If the schema is `fixed` or `flexible`, or if `guaranteed_fields` is already set explicitly, use the normal `SchemaConfig` semantics with no heuristic override or supplementation.

Rationale: The text source always yields `{column: value}` (text_source.py:110). The output is fully deterministic — one field, always present.

**Enforcement link:** Add a comment in `text_source.py` near line 110 referencing the shared observed-text rule (`# Composer/runtime contract helper depends on this: observed text sources infer {column} as guaranteed field`). This ensures a developer changing the text source's output key discovers the dependency. A cross-module test verifies both sides agree.

**Runtime path requirement:** The runtime side must make this guarantee visible to DAG construction, not just to the plugin's private constructor state. In the current codebase, `build_execution_graph()` materializes source schema contracts from `source.config["schema"]`, so mutating only `TextSource._schema_config` is insufficient. The implementation must keep the builder-visible raw config dict synchronized with the normalized schema for the observed-text special case. Because `SchemaConfig` is already a frozen typed dataclass, the implementation should update it mechanically with `dataclasses.replace(...)`, not by round-tripping through `to_dict()` / `from_dict()` or by using `.get()` on serialized Tier 1 data. Use `to_dict()` only for the final sync into `self.config["schema"]`.

**This heuristic list is intentionally closed.** Only `text` qualifies because its output shape is fully determined by config (`column`). Other sources (csv, json, dataverse) have variable schemas depending on input data and do not qualify for heuristic inference. If a new source has fully deterministic output, the decision to add it here requires explicit design review, not incremental expansion. This constraint prevents the composer from gradually rebuilding the runtime validator via accumulated heuristics (the "Shifting the Burden" archetype).

#### Schema parse failure handling

If `SchemaConfig.from_dict()` raises `ValueError` on a producer or consumer's schema dict, emit an **error** (not warning) and set `is_valid` to `false`. Schema config is Tier 2 data (pipeline configuration we control). A malformed schema dict is a plugin configuration bug — the project's plugin ownership model requires these to crash, not degrade gracefully. Treating parse failures as warnings would allow `is_valid: true` for pipelines that will fail at runtime, which is the exact bug this spec fixes.

Consumer parse failures must be handled the same way as producer parse failures. In particular, a malformed consumer schema must not silently degrade to `frozenset()` requirements, because that converts a configuration bug into a false success.

The error message should identify which node's schema failed to parse and include the `ValueError` message for debuggability.

#### Import placement

The shared raw-config contract helpers are imported via deferred import inside the schema-validation pass function, following the existing pattern established by `_validate_gate_expression()` (which defers `from elspeth.core.expression_parser import ...`). This preserves `state.py`'s module-level import constraint ("Imports from L0 contracts.freeze only") while still using the real contract logic at call time. Those helpers, in turn, use `SchemaConfig` inside the L0 contracts layer.

#### Error message format

```
Schema contract violation: 'source' -> 'add_world'.
  Consumer (value_transform) requires fields: ['text']
  Producer (text) guarantees: (none - observed schema)
  Missing fields: ['text']
  Fix: Add missing fields to the source schema (use mode 'fixed' or 'flexible'),
  set explicit guaranteed_fields, or remove from required_input_fields if optional.
```

### 2. Edge Contract Data in Preview Response

**Files:** `src/elspeth/web/composer/state.py`, `src/elspeth/web/composer/tools.py`

#### New dataclass

```python
@dataclass(frozen=True, slots=True)
class EdgeContract:
    """Schema contract check result for a single producer->consumer edge."""
    from_id: str
    to_id: str
    producer_guarantees: tuple[str, ...]
    consumer_requires: tuple[str, ...]
    satisfied: bool
```

#### ValidationSummary extension

```python
@dataclass(frozen=True, slots=True)
class ValidationSummary:
    is_valid: bool
    errors: tuple[ValidationEntry, ...]
    warnings: tuple[ValidationEntry, ...] = ()
    suggestions: tuple[ValidationEntry, ...] = ()
    edge_contracts: tuple[EdgeContract, ...] = ()  # NEW
```

#### Preview response

`_execute_preview_pipeline()` serializes `edge_contracts` into the response:

```json
{
  "edge_contracts": [
    {
      "from": "source",
      "to": "add_world",
      "producer_guarantees": ["text"],
      "consumer_requires": ["text"],
      "missing_fields": [],
      "satisfied": true
    }
  ]
}
```

The `edge_contracts` field appears even when all contracts are satisfied, providing positive confirmation of correct wiring.

This pass intentionally does not add `contracts_all_satisfied` or `has_unverified_edges` as separate stored booleans. Those would duplicate information already present in `edge_contracts` and `warnings`, creating a second truth source. If convenience fields are ever added later, they must be derived mechanically at serialization time rather than stored independently on `ValidationSummary`.

#### YAML export backstop

The pure serializer `elspeth.web.composer.yaml_generator.generate_yaml()` stays validation-blind. It is also used by execution-time validation and run creation, so moving workflow policy into that function would tangle serialization with unrelated call sites.

Instead, the LLM-facing export surfaces must validate immediately before returning YAML:

- `src/elspeth/composer_mcp/server.py` `generate_yaml` tool
- `src/elspeth/web/sessions/routes.py` `GET /api/sessions/{id}/state/yaml`

Those callers must run `state.validate()` and refuse export when `validation.is_valid` is `false`. This is the mechanical backstop for the skill rule. It relies on unsatisfied contracts being error-level; if a future contract path is intentionally warning-only, this export gate must be revisited explicitly rather than assumed.

### 3. Skill Patch for `pipeline_composer.md`

**File:** `src/elspeth/web/composer/skills/pipeline_composer.md`

Three targeted additions:

#### 3a. Completion Criteria — field contract gate

Add item 4 to the existing completion criteria list:

> 4. **All edge contracts are satisfied** — every downstream step's `required_input_fields` must be guaranteed by its upstream producer, and sink schemas may impose their own required fields. Check `edge_contracts` in the preview response. If any edge shows `"satisfied": false`, the pipeline is not complete.
>
> - **Empty `edge_contracts` is not success** — `edge_contracts: []` means no explicit field contracts were declared. It does not mean every contract is satisfied.
> - **Skipped checks are unresolved** — if preview warnings say a contract check was skipped (for example because the producer is a coalesce node), treat that as unresolved rather than satisfied and surface the warning to the user.
>
> Pipelines without `required_input_fields` declarations are not verified by the composer's contract check; the runtime validator remains the final authority.
>
> `generate_yaml` is an export step, not the primary validator. After this fix it should refuse invalid states, but the agent still uses `preview_pipeline` first to diagnose and fix the actual contract failure.

#### 3b. Text Source Safety Rule

Add to the text source gotchas in the Source Semantics Guide:

> **Schema rule for text sources:** Prefer an explicit `fixed` or `flexible` schema when you know the text column shape; it gives the strongest contract and clearer types. Narrow exception: a `text` source with `{"schema": {"mode": "observed"}}` and a non-empty `column` is still treated as guaranteeing `{column}` by the shared composer/runtime contract helper when `guaranteed_fields` is not explicitly set. Do not generalize this exception to other observed sources.

#### 3c. Schema Contract Fix Flow (worked example)

New subsection under Validation, replacing vague "disagreement handling" with a concrete action sequence:

> #### Fixing Schema Contract Violations
>
> When `preview_pipeline` returns an unsatisfied edge contract, follow this sequence:
>
> 1. **Read the violation** — identify which edge failed, what fields are missing, and which node is the producer.
> 2. **Patch the producer contract** — usually by fixing the actual producer shape first, then making the schema explicit. For most sources this means `patch_source_options` to change from `observed` to `fixed`/`flexible` with the required fields declared:
>    ```json
>    patch_source_options({
>      "patch": {"schema": {"mode": "fixed", "fields": ["text: str"]}}
>    })
>    ```
> 3. **Re-preview** — call `preview_pipeline` and verify the edge now shows `"satisfied": true`.
> 4. **Only then call `generate_yaml` or report success.** If `generate_yaml` still refuses the export, treat that as confirmation the pipeline remains unresolved and go back to preview/repair instead of bypassing the gate.
>
> **Example — csv source + value_transform:**
> - `preview_pipeline` returns: `edge_contracts: [{"from": "source", "to": "add_world", "satisfied": false, "consumer_requires": ["text"], "producer_guarantees": []}]`
> - Fix: `patch_source_options({"patch": {"schema": {"mode": "fixed", "fields": ["text: str"]}}})`
> - Re-preview confirms: `"satisfied": true`
>
> **Text-source note:** if the source plugin is `text` and the consumer requires the configured `column`, observed mode is already valid via the shared observed-text rule. If the required field and `column` do not match, fix the `column` or downstream field reference; do not invent a `fixed` schema that claims a different key than the plugin emits.
>
> If `get_pipeline_state` and `preview_pipeline` disagree (e.g., state shows a field but preview shows an unsatisfied contract), treat this as unresolved. Do not report success. Re-run both tools, fix the discrepancy, and confirm before responding.
>
> If `preview_pipeline` still shows `"satisfied": false` after **2** producer-schema patch attempts for the same edge, stop patching and explain the limitation to the user. After 2 attempts, treat the issue as structural rather than a missing field declaration.
>
> If the same producer feeds multiple consumers with conflicting truthful requirements, do not loop trying to force one schema to satisfy all of them. Surface the conflict explicitly and ask whether to split the path, add a branch-local transform/aggregation with an explicit schema, or relax/correct one of the downstream requirements.

#### 3d. Error Explanation Tooling

The composer already exposes `explain_validation_error` via `_VALIDATION_ERROR_PATTERNS` in `src/elspeth/web/composer/tools.py`. The contract pass introduces a new error family, so the catalogue should gain a dedicated pattern for `Schema contract violation:` errors that routes the agent to:

- inspect `preview_pipeline.edge_contracts`
- patch the producer schema with `patch_source_options` or `patch_node_options`
- re-preview until the edge is satisfied or the 2-attempt stop condition is reached

## What This Doesn't Catch

| Gap | Why | Mitigation |
|-----|-----|------------|
| Transform output fields | Composer doesn't instantiate plugins, can't know what fields a transform adds | Runtime DAG validator catches this |
| Type mismatches | Requires Pydantic schema construction | Runtime type validation catches this |
| Coalesce merge semantics | Post-merge guarantees depend on branch policies and merge strategy, not a single static producer schema | Composer emits a skip warning; runtime validator is authoritative |
| Dynamic schemas with no declarations | If neither side declares contracts, nothing to check | S3 suggestion already flags missing schemas |
| Plugin-computed guaranteed fields beyond the closed list | The shared observed-text rule is modeled explicitly, but other plugin-specific guarantee computations are still runtime-only | Runtime validator is authoritative |

The composer's contract check is a fast, early-feedback mechanism. The runtime validator remains the final authority.

## Test Strategy

### Unit tests for `state.validate()`

**Positive cases:**
1. Source with `fixed` schema guaranteeing `text`, consumer with `required_input_fields: ["text"]` — no error.
2. Observed-text rule: `plugin: "text"`, `column: "text"`, `observed` schema, no explicit `guaranteed_fields`, consumer requiring `text` — no error (shared helper infers guarantee).
3. Text source with explicit `guaranteed_fields: ["text"]` and `column: "text"` — the observed-text rule does not interfere; explicit value is used.
4. Consumer with no `required_input_fields` — contract check skipped, no error regardless of producer schema.
5. Consumer with `required_input_fields: []` (empty list) and no `schema.required_fields` — treated as no requirements, skipped.
6. Consumer with no `required_input_fields` but explicit `schema.required_fields: ["text"]` — helper falls back to `schema.required_fields` and satisfied edge contract is recorded.
7. Consumer with `required_input_fields: []` and explicit `schema.required_fields: ["text"]` — empty list does not suppress the fallback; the helper still enforces `schema.required_fields`.

Add direct sink-loop coverage in unit tests as well:
- source-only pipeline (`source.on_success -> sink name`, no transforms) records a single satisfied `output:main` `EdgeContract`, proving the sink loop works independently of the node loop
- sink with explicit `required_fields: ["text"]`, upstream source guarantees `text` — no error and `output:main` edge contract satisfied
- sink with explicit `required_fields: ["text"]`, upstream source guarantees only `line` — error and `output:main` edge contract violated

The agreement suite separately covers the stricter typed-sink parity case (`SchemaConfig.get_effective_required_fields()` from fixed/flexible sink fields).

**Negative cases:**
8. Non-text source with `observed` schema (no guarantees), consumer requiring `text` — error emitted, `is_valid: false`.
9. Partial match: producer guarantees `["text"]`, consumer requires `["text", "score"]` — error for missing `score`.
10. Optional field trap: producer declares `text: str?` (optional), consumer requires `text` — error (optional fields are not guaranteed).
11. No schema config at all on source — guaranteed = empty, consumer requirements flagged.
12. Malformed producer schema dict (e.g., `{"mode": "invalid"}`) — error emitted, `is_valid: false`.
13. Malformed consumer schema dict — blocking error emitted on that node and no fake `EdgeContract` is recorded.
14. Consumer with `required_input_fields: []` and `schema.required_fields: ["text"]` while producer guarantees only `line` — contract violation proves empty-list fallback mirrors runtime semantics rather than silently suppressing requirements.

**Topology cases:**
15. Gate-only pipeline: source → gate → sinks. Gate route targets inherit source's guaranteed fields. Consumer on route target with requirements satisfied by source — no error.
16. Coalesce producer upstream of a consumer — warning emitted and contract check skipped for that edge because guarantees are runtime-computed from branch policies and merge strategy.
17. Multi-hop: source → transform A (no requirements, no schema) → transform B (requires `text`). Use a connection name distinct from the transform ID in the test fixture so the snippet does not conflate routing names with node IDs. Transform A guarantees nothing — error for transform B.
18. Multi-sink with gate routing: source → gate → sink_a, sink_b. Contract check applies per route, not globally.
19. Aggregation consumer with `required_input_fields` — composer rejects when upstream guarantees are missing, proving aggregation nodes participate in the same consumer-contract framework.

The coalesce warning path must be pinned by a concrete unit test, not just described in prose. The test should assert both that the warning is emitted and that no fake `EdgeContract` is recorded for the skipped consumer edge.

**Data integrity:**
15. `ValidationSummary.edge_contracts` contains correct `EdgeContract` entries with proper `from_id`, `to_id`, `producer_guarantees`, `consumer_requires`, `satisfied` for all edges.
16. `EdgeContract.to_dict()` serializes `from_id` as `"from"` key (Python keyword avoidance).

### Integration test for preview response

20. Build a pipeline via composer tools, call `preview_pipeline`, verify `edge_contracts` appears in response with correct fields and is consistent with `errors`.

### Regression test for the reported text-source scenario

21. Build: text source (column=text, observed schema) + value_transform (required_input_fields=["text"]) + csv output. Assert `is_valid` is `true` and the corresponding `edge_contract` is satisfied. The general false-positive regression remains covered by negative case 8.

### Composer/runtime agreement test

22. Build pipelines that the composer validates across both pass and fail cases: wrong field reject, observed-text accept, strict sink typed-schema reject, and aggregation nested-options reject. Generate YAML, instantiate plugins, build `ExecutionGraph`, call `validate_edge_compatibility()`. Assert both validators agree on pass/fail for the same pipeline configuration.

Scope this suite narrowly: it covers only configurations where shared-contract parity is intended. If the composer later adds intentionally stricter preflight checks that runtime does not mirror, those belong in separate asymmetry tests with explicit documentation rather than in this agreement suite.

### Observed-text enforcement test

23. Import `TextSource`, construct with `column: "text"` and minimal valid config, call `load()` on a one-line temp file, verify the yielded row contains key `"text"`. Add deliberate white-box assertions against both the plugin's normalized internal schema config and the builder-visible `source.config["schema"]` entry for the auto-declared observed-text guarantee, because there is no public accessor for that constructor-time state and DAG construction consumes the raw config path. Also pin the fixed-schema control case: `declares_guaranteed_fields` must remain `False` there because it tracks explicit `guaranteed_fields` declarations only, while `get_effective_guaranteed_fields()` still includes required typed fields. This ties the shared observed-text rule to the plugin's actual behavior — if either side changes, this test fails.
