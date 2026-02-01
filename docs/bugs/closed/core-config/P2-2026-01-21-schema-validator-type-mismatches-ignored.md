# Bug Report: schema_validator ignores type mismatches between producer and consumer schemas

## Summary

- `validate_pipeline_schemas` only checks for missing required field names.
- It does not detect incompatible field types (e.g., producer `value: str`, consumer `value: int`).
- This allows pipelines with incompatible types to pass validation and fail later at runtime.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any pipeline with mismatched field types

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Define a producer schema with `value: str`.
2. Define a consumer schema with `value: int` (required).
3. Call `validate_pipeline_schemas(...)` with those schemas (or run a pipeline).

## Expected Behavior

- Schema validation rejects incompatible field types with a clear error message.

## Actual Behavior

- Validation passes because only field names are checked.

## Evidence

- `_get_missing_required_fields` checks only field names: `src/elspeth/engine/schema_validator.py:80-96`
- Type compatibility logic exists but is unused: `src/elspeth/contracts/data.py:131-205`

## Impact

- User-facing impact: pipelines can start with incompatible types and fail mid-run in transforms/sinks.
- Data integrity / security impact: violates Tier 2 rule (wrong types are upstream bugs) by letting incompatible schemas through.
- Performance or cost impact: reruns and debugging time.

## Root Cause Hypothesis

- Schema validator was implemented as a missing-field check and never upgraded to use the richer compatibility logic in `contracts.data.check_compatibility`.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/schema_validator.py`: use `check_compatibility` (or similar) to detect type mismatches in addition to missing fields.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test in `tests/engine/test_schema_validator.py` asserting a type mismatch is reported.
- Risks or migration steps:
  - Pipelines that previously passed validation may now be rejected; document as correctness fix.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 2 rules (wrong types are upstream bugs)
- Observed divergence: type incompatibilities are not validated at construction time.
- Reason (if known): schema validator uses name-only checks.
- Alignment plan or decision needed: decide whether to enforce full type compatibility at build time.

## Acceptance Criteria

- Type mismatches between producer and consumer schemas are detected and reported.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 1

**Current Code Analysis:**

The bug report's core claim remains valid, though the architecture has evolved significantly since the original report (commit `ae2c0e6f`).

**What Changed (2026-01-24 Schema Validation Refactor):**

Between the bug report date (2026-01-21) and verification (2026-01-25), a major architectural refactor occurred:

1. **schema_validator.py DELETED** - The module mentioned in the bug no longer exists
2. **Two-phase validation implemented** (commits df43269, 430307d, 8809bd1):
   - PHASE 1: Plugin self-validation during `__init__()` (via `validate_output_schema()`)
   - PHASE 2: Edge compatibility during graph construction (via `ExecutionGraph.validate_edge_compatibility()`)
3. **DAG validation simplified** - Now only checks structural issues (cycles, connectivity), not schemas

**Current Implementation (src/elspeth/core/dag.py:794-822):**

The `_get_missing_required_fields()` method still exists but ONLY checks for missing field names:

```python
def _get_missing_required_fields(
    self,
    producer_schema: type[PluginSchema] | None,
    consumer_schema: type[PluginSchema] | None,
) -> list[str]:
    """Get required fields that producer doesn't provide."""
    if producer_schema is None or consumer_schema is None:
        return []  # Dynamic schema

    # Check if either schema is dynamic
    producer_is_dynamic = len(producer_schema.model_fields) == 0 and producer_schema.model_config.get("extra") == "allow"
    consumer_is_dynamic = len(consumer_schema.model_fields) == 0 and consumer_schema.model_config.get("extra") == "allow"

    if producer_is_dynamic or consumer_is_dynamic:
        return []  # Dynamic schema - compatible with anything

    producer_fields = set(producer_schema.model_fields.keys())
    consumer_required = {name for name, field in consumer_schema.model_fields.items() if field.is_required()}

    return sorted(consumer_required - producer_fields)
```

**The Bug Still Exists:**

The validation logic checks ONLY for missing required field **names** (line 820-822). It does NOT check if field **types** are compatible.

**Type Compatibility Logic Exists But Is Unused:**

The bug report correctly identified that `src/elspeth/contracts/data.py:131-205` contains `check_compatibility()` which DOES check type mismatches via `_types_compatible()`. However:

- Grep search confirms `check_compatibility()` is **never called** from `src/elspeth/core/dag.py`
- It's only used in **tests** (`tests/plugins/test_schemas.py`, `tests/contracts/test_plugin_schema.py`)
- Edge validation uses the simpler `_get_missing_required_fields()` instead

**Test Coverage Confirms the Gap:**

Review of `tests/core/test_edge_validation.py` (273 lines, added in commit 8809bd1) shows:
- ✅ Tests for missing required fields
- ✅ Tests for dynamic schema compatibility
- ✅ Tests for gate pass-through validation
- ✅ Tests for coalesce branch compatibility
- ❌ **NO tests for type mismatches** (e.g., producer `value: str`, consumer `value: int`)

Searched test file for type mismatch coverage:
```bash
grep -E "int.*str|type.*compatibility|_types_compatible" tests/core/test_edge_validation.py
# No matches found
```

**Git History:**

The schema validation plan (`docs/plans/2026-01-24-fix-schema-validation-properly.md`) does not mention type checking. The plan focused on:
1. Moving validation from DAG layer to plugin construction (architectural fix)
2. Implementing two-phase validation (self vs compatibility)
3. Enforcing validation cannot be skipped

Type compatibility checking was **not in scope** for this refactor.

**Root Cause Confirmed:**

YES. The bug persists in the new architecture:

1. `ExecutionGraph.validate_edge_compatibility()` calls `_validate_single_edge()`
2. `_validate_single_edge()` calls `_get_missing_required_fields()` (line 704)
3. `_get_missing_required_fields()` only checks field name presence, not types
4. The richer `check_compatibility()` function with type checking exists but is unused

**Example That Would Pass But Shouldn't:**

```python
class ProducerSchema(PluginSchema):
    value: str  # String type

class ConsumerSchema(PluginSchema):
    value: int  # Integer type - INCOMPATIBLE!

graph = ExecutionGraph()
graph.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)
graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=ConsumerSchema)
graph.add_edge("source", "sink", label="continue")

# This SHOULD fail but currently PASSES validation
graph.validate_edge_compatibility()  # No error raised! Field 'value' exists in both schemas
```

**Recommendation:**

**Keep open** - Priority P2 is appropriate. This is a correctness issue but not critical severity:

- **Impact:** Type mismatches go undetected at graph construction time, causing runtime failures during execution
- **Workaround:** Runtime validation at plugin boundaries will catch type errors, just later than ideal
- **Fix:** Replace `_get_missing_required_fields()` call in `_validate_single_edge()` with `check_compatibility()` from contracts.data
- **Test Gap:** Add test case in `tests/core/test_edge_validation.py` for type mismatch detection

**Suggested Fix (one-line change):**

In `src/elspeth/core/dag.py:_validate_single_edge()`, replace:
```python
missing_fields = self._get_missing_required_fields(producer_schema, consumer_schema)
if missing_fields:
    raise ValueError(f"...missing required fields...{missing_fields}")
```

With:
```python
from elspeth.contracts.data import check_compatibility

result = check_compatibility(producer_schema, consumer_schema)
if not result.compatible:
    raise ValueError(
        f"Edge from '{from_node_id}' to '{to_node_id}' invalid: "
        f"missing fields: {result.missing_fields}, "
        f"type mismatches: {result.type_mismatches}"
    )
```
