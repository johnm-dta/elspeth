# Bug Report: Coalesce schema compatibility uses class identity instead of structural compatibility

## Summary

- Coalesce validation rejects branches with equivalent schema definitions when their Pydantic schema classes are distinct objects (e.g., per-instance dynamic/LLM schemas), causing false “incompatible schema” failures.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-30
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Synthetic pipeline with multiple fork gates feeding a single coalesce; transforms built from identical SchemaConfig but instantiated separately

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/core/dag.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Create two separate schema classes with identical fields (e.g., call `create_schema_from_config` twice with the same config).
2. Build an `ExecutionGraph` where two different fork gates (or transforms feeding coalesce) produce branches that merge at the same coalesce node.
3. Call `graph.validate_edge_compatibility()`.

## Expected Behavior

- Coalesce validation should accept compatible schemas (same fields/types) even if they are different class objects, and only reject truly incompatible schemas.

## Actual Behavior

- Validation raises `ValueError` for “incompatible schemas” because it compares schema classes by identity (`!=`) instead of structural compatibility.

## Evidence

- `src/elspeth/core/dag.py:1039-1049` compares `first_schema != other_schema` inside `_get_effective_producer_schema` for multi-input pass-through nodes.
- `src/elspeth/core/dag.py:1071-1083` repeats the identity comparison in `_validate_coalesce_compatibility`.
- `src/elspeth/plugins/schema_factory.py:41-91` shows `create_schema_from_config()` uses `create_model()` per call, producing distinct class objects even for identical configs.

## Impact

- User-facing impact: Valid fork/coalesce configurations can be rejected during validation when branches originate from different transform instances with equivalent schemas.
- Data integrity / security impact: None directly; validation is overly strict, not permissive.
- Performance or cost impact: Increased configuration failures and manual debugging time; no runtime cost.

## Root Cause Hypothesis

- Coalesce compatibility checks rely on class identity (`==`) rather than structural schema compatibility, which fails when equivalent schemas are created as distinct Pydantic classes (common for runtime-generated schemas like LLM transforms).

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/dag.py`: Replace identity checks in `_get_effective_producer_schema` (multi-input pass-through) and `_validate_coalesce_compatibility` with structural compatibility checks (e.g., use `check_compatibility` in both directions, and treat dynamic schemas as compatible).
- Config or schema changes: Unknown
- Tests to add/update:
  - Add a unit test that creates two equivalent schemas via `create_schema_from_config` and verifies coalesce validation passes when branches merge.
  - Ensure existing incompatible-branch tests still fail.
- Risks or migration steps:
  - Risk: Loosening checks could allow subtly incompatible schemas if compatibility logic is too permissive; mitigate by requiring bidirectional compatibility and keeping dynamic detection consistent with `_validate_single_edge`.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Coalesce validation uses class identity rather than schema compatibility, which is stricter than compatibility checks used on standard edges.
- Reason (if known): Unknown
- Alignment plan or decision needed: Align coalesce compatibility logic with `check_compatibility` and dynamic-schema handling used in `_validate_single_edge`.

## Acceptance Criteria

- Coalesce validation accepts structurally compatible schemas even when classes differ.
- Coalesce validation continues to reject genuinely incompatible schemas.
- Existing edge compatibility behavior is unchanged outside coalesce.

## Verification (2026-02-01)

**Status: STILL VALID**

- Coalesce compatibility still uses class identity comparisons, and schema factory still generates per-call Pydantic classes. (`src/elspeth/core/dag.py:1039-1083`, `src/elspeth/plugins/schema_factory.py:41-91`)

## Resolution (2026-02-02)

**Status: FIXED**

### Changes Made

1. **Added `_schemas_structurally_compatible()` helper method** (`src/elspeth/core/dag.py:1057-1106`):
   - Handles dynamic schemas (None or no fields + extra="allow") as compatible with anything
   - Uses bidirectional `check_compatibility()` for explicit schemas
   - Returns `(is_compatible, error_message)` tuple for detailed error reporting

2. **Updated `_get_effective_producer_schema()`** (`src/elspeth/core/dag.py:1039-1054`):
   - Replaced `if first_schema != other_schema:` with structural compatibility check
   - Now uses `_schemas_structurally_compatible()` for multi-input pass-through validation

3. **Updated `_validate_coalesce_compatibility()`** (`src/elspeth/core/dag.py:1129-1141`):
   - Replaced `if first_schema != other_schema:` with structural compatibility check
   - Now uses `_schemas_structurally_compatible()` for coalesce validation

### Tests Added

- `test_coalesce_accepts_structurally_identical_schemas`: Verifies structurally identical schemas (different class objects) pass validation
- `test_coalesce_accepts_dynamic_schemas_from_different_instances`: Verifies dynamic schemas from different `create_schema_from_config()` calls pass validation
- Existing `test_coalesce_rejects_incompatible_branch_schemas` continues to pass (truly incompatible schemas still rejected)

### Verification

All 20 coalesce-related tests pass. All 7 schema validation tests pass.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/test_dag.py -k coalesce`
- New tests required: yes, add coalesce compatibility test for equivalent-but-distinct schema classes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: Unknown
