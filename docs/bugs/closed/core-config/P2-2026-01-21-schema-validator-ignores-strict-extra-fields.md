# Bug Report: schema_validator ignores strict-schema extra-field constraints

## Summary

- Strict schemas (extra="forbid") reject unknown fields, but the schema validator only checks that required fields exist.
- A producer schema that includes additional fields can pass validation even when a strict consumer forbids extras.
- This creates false positives where the pipeline validates but strict sinks/transforms reject rows at runtime.

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
- Data set or fixture: any strict consumer schema with a producer that declares extra fields

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a strict consumer schema (extra="forbid") with fields `id, name`.
2. Create a producer schema that includes `id, name, extra_field`.
3. Call `validate_pipeline_schemas(...)` with these schemas.

## Expected Behavior

- Validation fails because strict consumers do not accept extra fields declared by the producer.

## Actual Behavior

- Validation passes because only missing required fields are checked.

## Evidence

- Strict schemas set `extra="forbid"` in schema factory: `src/elspeth/plugins/schema_factory.py:74-116`
- Schema validator only checks required field names: `src/elspeth/engine/schema_validator.py:80-96`

## Impact

- User-facing impact: pipelines pass validation but strict sinks/transforms may reject rows at runtime.
- Data integrity / security impact: schema compatibility is overstated; audit metadata implies compatibility that does not hold for strict consumers.
- Performance or cost impact: reruns and manual debugging.

## Root Cause Hypothesis

- Validator does not consider `extra="forbid"` semantics when comparing producer and consumer schemas.

## Proposed Fix

- Code changes (modules/files):
  - Extend schema validation to check extra-field compatibility when consumer is strict.
  - If consumer forbids extras and producer declares additional fields, report incompatibility.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that strict consumer rejects producer schemas with extra fields.
- Risks or migration steps:
  - Existing pipelines with strict sinks may now fail validation; document as correctness fix.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (engine validates schema compatibility)
- Observed divergence: strict consumer constraints are ignored in compatibility checks.
- Reason (if known): validator only checks for missing required fields.
- Alignment plan or decision needed: define compatibility rules for strict schemas.

## Acceptance Criteria

- Strict consumers (extra="forbid") reject producer schemas that declare additional fields.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/plugins/schema_factory.py`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 1

**Current Code Analysis:**

The bug is confirmed to still exist after the schema validation refactor (completed 2026-01-24). The architecture has changed significantly, but the core issue remains:

**Old Architecture (at bug report time):**
- `src/elspeth/engine/schema_validator.py` contained validation logic
- Location: Lines 80-96 checked only required fields

**New Architecture (current):**
- Schema validation moved to plugin construction (2-phase model per `docs/plans/2026-01-24-fix-schema-validation-properly.md`)
- PHASE 1: Plugins self-validate during `__init__()` via `validate_output_schema()` protocol method
- PHASE 2: Edge compatibility validated in `ExecutionGraph.validate_edge_compatibility()` during graph construction
- Located at: `src/elspeth/core/dag.py:794-822` in `_get_missing_required_fields()`

**Current Validation Logic:**
```python
# src/elspeth/core/dag.py:794-822
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

    return sorted(consumer_required - producer_fields)  # Only checks MISSING fields
```

**The Bug:**
The validation ONLY checks if the producer is missing required fields from the consumer (`consumer_required - producer_fields`). It does NOT check if the consumer has `extra="forbid"` and would reject additional fields from the producer.

**Concrete Example:**
```python
# Producer schema: {id: int, name: str, extra_field: str}
# Consumer schema: {id: int, name: str} with extra="forbid"

# Current behavior: PASSES validation
# - consumer_required = {id, name}
# - producer_fields = {id, name, extra_field}
# - consumer_required - producer_fields = {} (empty, so no error)

# Expected behavior: SHOULD FAIL
# - Consumer forbids extras but producer declares extra_field
# - At runtime, Pydantic will reject rows with extra_field
```

**Schema Factory Confirmation:**
`src/elspeth/plugins/schema_factory.py:118` correctly sets `extra="forbid"` for strict mode:
```python
extra_mode: ExtraMode = "allow" if config.mode == "free" else "forbid"
```

**Git History:**

No commits attempted to fix this issue. Search for related commits:
```bash
git log --all --grep="extra.*field" --grep="strict.*schema" --grep="forbid" -i
```
Found commits about schema factories and dynamic schema detection, but none addressing the extra-field validation gap.

Recent schema validation refactor commits (2026-01-24):
- `0e2f6da` - fix: add validation to remaining 5 plugins
- `7ee7c51` - feat: add self-validation to all builtin plugins
- `8809bd1` - feat: add edge compatibility validation to ExecutionGraph
- `df43269` - refactor: remove schema validation from DAG layer
- `430307d` - feat: add schema validation to plugin protocols

None of these addressed the `extra="forbid"` constraint checking.

**Root Cause Confirmed:**

YES - The bug persists in the new architecture. The validation logic migrated from the old `schema_validator.py` to `dag.py:_get_missing_required_fields()` but retained the same incomplete check.

The method checks:
✅ Does producer have all required fields consumer needs?
❌ Does producer have extra fields that strict consumer forbids?

**Recommendation:**

**Keep open** - This is a valid P2 bug that needs fixing.

**Fix Location:** `src/elspeth/core/dag.py:_get_missing_required_fields()` should be enhanced to:
1. Check if consumer has `extra="forbid"` in `model_config`
2. If so, verify `producer_fields ⊆ consumer_fields` (producer fields must be subset of consumer fields)
3. Return error if producer declares fields not in consumer's field list

**Test Gap:** No tests currently validate strict consumer + producer-with-extras scenario. The fix should include:
- Test in `tests/core/test_edge_validation.py` (if it exists)
- Test strict consumer rejecting producer with extra fields
- Test free consumer (`extra="allow"`) accepting producer with extra fields

**Impact:** Pipeline passes validation at construction time but fails at runtime when strict consumer receives data with extra fields, breaking the "validation at construction" guarantee of the 2-phase model.
