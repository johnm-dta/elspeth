# Output Schema Contract Enforcement

**Date:** 2026-03-20
**Status:** Implemented
**Scope:** BaseTransform, DAG builder, all field-adding transforms

## Problem Statement

ELSPETH's DAG builder validates edges between transforms by checking that a producer's `guaranteed_fields` satisfy a consumer's `required_input_fields`. This validation depends on transforms setting `_output_schema_config` — a `SchemaConfig` instance that declares which fields the transform guarantees in its output.

This contract is currently **optional and unenforced**. The DAG builder reads it via `getattr(transform, "_output_schema_config", None)` and silently skips validation when the attribute is absent. Six transforms declare output fields via `declared_output_fields` but never set `_output_schema_config`, creating a gap where DAG validation cannot verify their output contracts.

### Affected Transforms

| Transform | Has `declared_output_fields` | Has `_output_schema_config` | Gap |
|-----------|------------------------------|----------------------------|-----|
| `rag/transform.py` | Yes (4 fields) | Yes (manual) | Fixed in prior commit, but manual — should use helper |
| `web_scrape.py` | Yes (8 fields) | No | DAG invisible |
| `json_explode.py` | Yes (config-driven) | No | DAG invisible |
| `batch_replicate.py` | Yes (conditional `copy_index`) | No | DAG invisible |
| `field_mapper.py` | **No** (empty default) | No | DAG invisible AND undeclared |
| `batch_stats.py` | **No** (empty default) | No | DAG invisible AND undeclared |

Transforms that are **not affected** (already correct):
- `llm/transform.py` — manually builds `_output_schema_config` with complex multi-query logic
- `llm/openrouter_batch.py` — same
- `llm/azure_batch.py` — same

### Consequences

1. Downstream transforms cannot use `required_input_fields` to declare dependencies on fields produced by these transforms — the DAG builder rejects the edge as "missing field."
2. Pipeline authors must use `required_input_fields: []` (opt-out) to bypass validation, defeating the purpose of static field checking.
3. The gap is silent — no error, no warning. Authors discover it only when a downstream transform fails at runtime or when DAG validation rejects a valid pipeline.

### Root Cause: Shifting the Burden

The `_create_schemas` helper handles runtime Pydantic schemas but not build-time DAG contracts. Authors call `_create_schemas(..., adds_fields=True)` and feel done — the helper's name and behavior signal "schema setup is complete." The actual contract (`_output_schema_config`) requires a separate, non-obvious manual step. Five transforms have already settled into the incomplete pattern.

## Design

### Invariants This Design Enforces

1. **If a transform declares output fields, it MUST provide a DAG contract.** A transform with non-empty `declared_output_fields` and `_output_schema_config is None` is a `FrameworkBugError` at graph-build time. No silent skip.

2. **`_output_schema_config.guaranteed_fields` MUST be a superset of `declared_output_fields`.** The helper method enforces this by construction — it merges `declared_output_fields` into `guaranteed_fields`. Transforms that build `_output_schema_config` manually (LLM transforms) are responsible for maintaining this invariant themselves. A comment in the LLM transform code should reference this spec and the invariant.

3. **Every transform that uses `adds_fields=True` MUST set `declared_output_fields` before calling `_build_output_schema_config`.** The method reads `self.declared_output_fields` — if it's the default empty `frozenset()`, the resulting `guaranteed_fields` contains no transform-specific fields. The DAG builder check (invariant 1) catches this if the transform later sets `declared_output_fields` but forgot the helper call. **Residual gap:** if a developer calls `_build_output_schema_config` *before* populating `declared_output_fields`, the result is a valid but empty contract — the enforcement check does not catch "helper called too early." This is addressable only by code review.

4. **`_create_schemas` remains a pure schema factory.** It does not set `_output_schema_config`. The two concerns (runtime Pydantic schemas and build-time DAG contracts) are handled by separate methods with no hidden coupling.

### Part 1: Base Class Changes (`BaseTransform`)

#### Declare `_output_schema_config` as a class attribute

```python
class BaseTransform:
    _output_schema_config: SchemaConfig | None = None
```

This eliminates the `getattr` pattern in the DAG builder. Direct attribute access with `None` check. Add `SchemaConfig` to the existing `if TYPE_CHECKING:` import block in `base.py` (alongside the existing `LifecycleContext`, `TransformContext`, etc. imports).

#### Add `_build_output_schema_config` helper method

```python
def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
    """Build output schema config for DAG contract propagation.

    Merges the transform's declared_output_fields into guaranteed_fields
    so the DAG builder can validate downstream field requirements.

    The returned SchemaConfig is for DAG contract propagation only. For
    fixed/flexible input schemas, guaranteed_fields may reference field
    names not present in the schema's fields tuple — this is intentional
    and correct for propagation purposes (the output adds fields beyond
    the input schema's declared set).

    Args:
        schema_config: The transform's input schema config (base fields).

    Returns:
        SchemaConfig with guaranteed_fields including declared output fields.
    """
    from elspeth.contracts.schema import SchemaConfig

    base_guaranteed = schema_config.guaranteed_fields or ()
    return SchemaConfig(
        mode=schema_config.mode,
        fields=schema_config.fields,
        guaranteed_fields=tuple(set(base_guaranteed) | self.declared_output_fields),
        audit_fields=schema_config.audit_fields,
        required_fields=schema_config.required_fields,
    )
```

Design decisions:
- **Returns** the `SchemaConfig` — does not set it. The caller assigns explicitly: `self._output_schema_config = self._build_output_schema_config(cfg)`. No hidden side effects.
- **`_create_schemas` stays `@staticmethod`, unchanged.** Runtime Pydantic schemas and build-time DAG contracts are separate concerns handled by separate methods.
- **LLM transforms are unaffected.** They don't use `_create_schemas` or `_build_output_schema_config` — their multi-query field logic requires manual construction, which is correct for their complexity level.

#### Fix type annotation on `_create_schemas`

Change `schema_config: Any` to `schema_config: SchemaConfig` under `TYPE_CHECKING`. The `SchemaConfig` import added to the `TYPE_CHECKING` block (above) serves both the class attribute annotation and this parameter annotation.

### Part 2: DAG Builder Validation (`dag/builder.py`)

At both `getattr` sites (line 223 for transforms, line 251 for aggregations), replace `getattr` with direct attribute access and add the offensive check. Remove the now-stale `getattr` justification comments. Extract the check into a standalone `_validate_output_schema_contract` function for testability and DRY.

```python
def _validate_output_schema_contract(transform: Any) -> None:
    """Validate that transforms declaring output fields provide a DAG contract."""
    if transform.declared_output_fields and transform._output_schema_config is None:
        raise FrameworkBugError(
            f"Transform {transform.name!r} declares output fields "
            f"{sorted(transform.declared_output_fields)} but provides no "
            f"_output_schema_config for DAG contract validation. "
            f"Call self._output_schema_config = self._build_output_schema_config(schema_config) "
            f"in __init__ after setting declared_output_fields."
        )
```

The error message includes the fix instruction — offensive programming with actionable diagnostics.

**Implementation ordering:** All affected transforms MUST be fixed (Part 3) BEFORE this enforcement check is added. Otherwise, unfixed transforms will crash at graph-build time. The implementation plan reverses Parts 2 and 3 for this reason.

### Part 3: Fix All Affected Transforms

All transforms call `_build_output_schema_config` **unconditionally** after setting `declared_output_fields`. When `declared_output_fields` is empty (e.g. `batch_replicate` with `include_copy_index=False`), the helper returns base `guaranteed_fields` only — still useful for DAG propagation.

#### Transforms that already have `declared_output_fields` (add helper call only)

**`rag/transform.py`**: Remove the manual `_output_schema_config` construction (lines 89–99). Replace with:
```python
self._output_schema_config = self._build_output_schema_config(self._rag_config.schema_config)
```
Note: `declared_output_fields` is set at lines 73–79 with prefix-interpolated names (e.g. `{prefix}__rag_context`), before `_create_schemas` at line 82. The ordering is already correct — the helper call goes after both.

**`web_scrape.py`**: Add after `declared_output_fields` is set:
```python
self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
```
Note: `web_scrape` doesn't use `_create_schemas` — it builds schemas manually. The helper call is independent.

**`json_explode.py`**: Add after `declared_output_fields` is set:
```python
self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
```

**`batch_replicate.py`**: Add after `declared_output_fields` is set (unconditionally — see note above):
```python
self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
```

#### Transforms that need `declared_output_fields` populated first

**`field_mapper.py`**: The mapping target names are the output fields. Before `_create_schemas`:
```python
# Mapping targets are the fields this transform adds/renames to in the output.
# Note: for rename-only mappings where the target already exists in the input,
# TransformExecutor collision detection may need review — but declared_output_fields
# is correct for DAG contract purposes (these fields are guaranteed in output).
self.declared_output_fields = frozenset(cfg.mapping.values())
```
Then add the helper call after `_create_schemas`.

**Implementation note:** `field_mapper` with `select_only=False` and rename-only mappings may have the target field already present in the input row. Verify that `TransformExecutor` collision detection handles this case correctly (collision check is against *other transforms'* declared fields, not input row keys). If collision detection checks against input keys, the rename case needs a carve-out. Document findings during implementation.

**`batch_stats.py`**: The output fields are computed from config. Before `_create_schemas`:
```python
stat_fields: set[str] = {"count", "sum", "batch_size"}
if cfg.compute_mean:
    stat_fields.add("mean")
self.declared_output_fields = frozenset(stat_fields)
```
Then add the helper call after `_create_schemas`.

**Design notes on `batch_stats` fields:**
- `group_by` is intentionally **not** in `declared_output_fields` — it is a passthrough field that already exists in the input row. Including it would trigger false collision detection in `TransformExecutor` (which checks `declared_output_fields` against input keys). The transform carries it through to output, but doesn't "add" it.
- `skipped_non_finite` and `skipped_non_finite_indices` are intentionally **not** declared — they are data-dependent (only emitted when non-finite values are encountered), not config-guaranteed.

### Part 4: Testing Strategy

#### Unit tests: `_build_output_schema_config` helper

| Test | Assertion |
|------|-----------|
| Merges base guaranteed + declared output fields | `guaranteed_fields` is the union (exact `frozenset` equality) |
| Empty `declared_output_fields` | Returns base guaranteed only |
| Empty base `guaranteed_fields` | Returns declared output fields only |
| Preserves other config fields | `mode`, `fields`, `audit_fields`, `required_fields` pass through unchanged |
| Non-`None` `audit_fields` passes through | Regression guard if helper is later modified |

#### Unit tests: DAG builder validation

Use a minimal hand-crafted stub implementing `TransformProtocol` with configurable `declared_output_fields` and `_output_schema_config`. Call the builder entry point directly. Do not use a real transform that already has a valid `_output_schema_config`.

| Test | Assertion |
|------|-----------|
| Non-empty `declared_output_fields`, `_output_schema_config = None` | Raises `FrameworkBugError` |
| Non-empty `declared_output_fields`, valid `_output_schema_config` | Passes |
| Empty `declared_output_fields`, `_output_schema_config = None` | Passes (shape-preserving transform) |

#### Per-transform unit tests (pin `guaranteed_fields` content)

Each affected transform gets a test that constructs an instance with a representative config and asserts `_output_schema_config.guaranteed_fields` contains the expected field names. Use **exact `frozenset` equality** (not `issubset`) to catch both missing and spurious fields. These are separate from the existing `test_declared_output_fields` tests — they test different attributes (`declared_output_fields` for collision detection vs `guaranteed_fields` for DAG contracts).

| Transform | Config | Expected `guaranteed_fields` (exact) |
|-----------|--------|--------------------------------------|
| `rag` | `output_prefix="sci"`, `schema: {mode: observed}` | `{"sci__rag_context", "sci__rag_score", "sci__rag_count", "sci__rag_sources"}` |
| `web_scrape` | `content_field="page_content"`, `fingerprint_field="page_hash"` | `{"page_content", "page_hash", "fetch_status", "fetch_url_final", "fetch_url_final_ip", "fetch_request_hash", "fetch_response_raw_hash", "fetch_response_processed_hash"}` |
| `json_explode` | `array_field="items"`, `output_field="item"`, `include_index=True` | `{"item", "item_index"}` |
| `json_explode` | `array_field="items"`, `output_field="item"`, `include_index=False` | `{"item"}` |
| `batch_replicate` | `include_copy_index=True` | `{"copy_index"}` |
| `batch_replicate` | `include_copy_index=False` | `frozenset()` (empty — verify against source that field is truly conditional) |
| `field_mapper` | `mapping={"old_name": "new_name", "source": "target"}` | `{"new_name", "target"}` |
| `batch_stats` | `value_field="amount"`, `compute_mean=True`, `group_by="category"` | `{"count", "sum", "batch_size", "mean"}` (group_by excluded — passthrough field) |
| `batch_stats` | `value_field="amount"`, `compute_mean=False`, `group_by=None` | `{"count", "sum", "batch_size"}` |

#### Integration test: Contract enforcement with real transforms

Test the full contract chain using real transform instances and the builder's validation function directly:

1. **Invariant 2 across all transforms** — parametric test that every field-adding transform's `guaranteed_fields` is a superset of its `declared_output_fields`.
2. **Enforcement passes** — correctly-configured transforms pass `_validate_output_schema_contract`.
3. **Enforcement fires** — a real transform with `_output_schema_config` cleared to `None` raises `FrameworkBugError`.
4. **Exact field pinning** — at least one transform (RAG) verifies `guaranteed_fields` content exactly.

Note: `ExecutionGraph.from_plugin_instances()` requires `WiredTransform`, `SourceProtocol`, `SinkProtocol`, and `GateSettings` — full pipeline wiring that no existing test exercises. Testing the extracted `_validate_output_schema_contract` function with real transform instances covers the contract surface without that overhead. A full-pipeline integration test through `from_plugin_instances()` is a worthwhile future enhancement but is out of scope here.

### Relationship to Existing Systems

| System | When | Purpose | Changed? |
|--------|------|---------|----------|
| `_output_schema_config` / `guaranteed_fields` | Graph build | DAG edge validation — "does this field exist?" | **Yes** — enforced, helper added |
| `_create_schemas` / `adds_fields` | Plugin init | Output Pydantic schema — "accept any fields at runtime?" | **No** — unchanged |
| `propagate_contract` / `transform_adds_fields` | Row processing | Contract propagation — "infer types for new fields?" | **No** — unchanged |
| `declared_output_fields` | Plugin init + executor | Collision detection in TransformExecutor | **Extended** — also feeds `_build_output_schema_config` |

**Layer dependency:** `SchemaConfig` lives at L0 (`contracts/schema.py`). `BaseTransform` lives at L3 (`plugins/infrastructure/base.py`). L3 importing L0 is a valid downward dependency. The `TYPE_CHECKING` import follows the existing pattern in `base.py`.

### Files Changed

| File | Change |
|------|--------|
| `src/elspeth/plugins/infrastructure/base.py` | Add `_output_schema_config` class attr, add `_build_output_schema_config` method, fix `_create_schemas` type annotation, add `SchemaConfig` to `TYPE_CHECKING` imports |
| `src/elspeth/core/dag/builder.py` | Replace `getattr` with direct access + offensive check (2 sites), remove stale `getattr` justification comments |
| `src/elspeth/plugins/transforms/rag/transform.py` | Replace manual construction with helper call, remove `SchemaConfig` import (now in base) |
| `src/elspeth/plugins/transforms/web_scrape.py` | Add helper call |
| `src/elspeth/plugins/transforms/json_explode.py` | Add helper call |
| `src/elspeth/plugins/transforms/batch_replicate.py` | Add helper call |
| `src/elspeth/plugins/transforms/field_mapper.py` | Populate `declared_output_fields`, add helper call, verify collision detection compatibility |
| `src/elspeth/plugins/transforms/batch_stats.py` | Populate `declared_output_fields` (including conditional `group_by`), add helper call |
| `src/elspeth/plugins/transforms/llm/transform.py` | Add comment referencing this spec and Invariant 2 (manual `_output_schema_config` responsibility) |
| `config/cicd/enforce_tier_model/plugins.yaml` | Update allowlist fingerprints if any defensive patterns change |
| Tests (new/modified) | Unit tests for helper, DAG check, per-transform pinning, integration test |

### Implementation Notes for Future Sessions

These notes capture context that may not be obvious from reading the code alone.

#### Key File Locations

- `BaseTransform` class: `src/elspeth/plugins/infrastructure/base.py` ~line 140
- `_create_schemas` static method: same file ~line 188 (has local runtime import of `SchemaConfig` — the new helper must follow the same pattern)
- DAG builder `getattr` sites: `src/elspeth/core/dag/builder.py` lines 223 and 251 — both have multi-line comments justifying `getattr` that must be removed
- `FrameworkBugError`: `src/elspeth/contracts/errors.py` — must be imported in `builder.py` (not currently imported)
- `SchemaConfig`: `src/elspeth/contracts/schema.py` — frozen dataclass with `guaranteed_fields: tuple[str, ...] | None`, `audit_fields: tuple[str, ...] | None`, `required_fields: tuple[str, ...] | None`
- `TYPE_CHECKING` block in `base.py`: starts ~line 30, imports from `contracts.contexts`, `contracts.header_modes`, etc. — add `SchemaConfig` here
- `base.py` has `from __future__ import annotations` (line 34) — this makes `TYPE_CHECKING` annotations safe for the class attribute

#### Existing Patterns to Follow

- The `_create_schemas` method uses a **local runtime import** inside its body: `from elspeth.contracts.schema import SchemaConfig`. The new `_build_output_schema_config` must do the same (not rely on `TYPE_CHECKING` imports for runtime construction).
- LLM transform's `_output_schema_config` construction: `src/elspeth/plugins/transforms/llm/transform.py` lines 981-987 (multi-query) and 1021-1027 (single-query). These set `_output_schema_config` manually. Do NOT modify these — only add a comment referencing this spec.
- `BaseSink` has an analogous pattern: `_output_contract: SchemaContract | None = None` declared as a class attribute (~line 442 of `base.py`). The `BaseTransform` class attribute follows the same convention.

#### Transform-Specific Details

- **`web_scrape.py`**: Does NOT use `_create_schemas`. It builds schemas manually at ~line 243. The `_build_output_schema_config` call is independent of schema creation. The config object is `WebScrapeHTTPConfig` (accessed as `cfg`), and `schema_config` is at `cfg.schema_config`.
- **`field_mapper.py`**: `FieldMapperConfig` at ~line 24 has `mapping: dict[str, str] = Field(default_factory=dict)`. An empty mapping produces `declared_output_fields = frozenset()`, which is correct (no-op mapper is shape-preserving). The `cfg` variable is `FieldMapperConfig.from_dict(config)` at ~line 87.
- **`batch_stats.py`**: `BatchStatsConfig` at ~line 22 has `value_field: str`, `group_by: str | None = None`, `compute_mean: bool = True`. The `cfg` variable is `BatchStatsConfig.from_dict(config)` at ~line 81.
- **`json_explode.py`**: `JSONExplodeConfig` has `output_field: str` (singular, NOT plural), `include_index: bool = True`. Field names are derived from config at ~lines 125-128. The `cfg` variable is `JSONExplodeConfig.from_dict(config)` at ~line 119.
- **`batch_replicate.py`**: `declared_output_fields` is already set conditionally at ~line 108. Only the helper call needs to be added.

#### What NOT to Change

- `_create_schemas` — stays `@staticmethod`, unchanged
- `propagate_contract` / `transform_adds_fields` — runtime concern, unchanged
- LLM transform `_output_schema_config` construction — too complex for the generic helper
- `TransformExecutor` collision detection logic — out of scope (verify it handles `field_mapper` renames correctly, but don't modify it)

#### Verification Checklist

After implementation, verify:
1. `python -m pytest tests/unit/plugins/transforms/ -x` — all transform unit tests pass
2. `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` — tier model passes (update fingerprints if needed)
3. `python -m mypy src/elspeth/plugins/infrastructure/base.py src/elspeth/core/dag/builder.py` — type checks pass
4. `python -m ruff check src/elspeth/` — lint passes
5. The two example pipelines still work: `./examples/chroma_rag/run.sh` and `./examples/chroma_rag_qa/run.sh`
