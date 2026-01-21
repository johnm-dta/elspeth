# Bug Report: Non-finite floats (`NaN`/`Infinity`) can pass source coercion but later crash canonical hashing (should be quarantined at the boundary)

## Summary

- Source schemas use `allow_coercion=True` and rely on Pydantic coercion for numeric fields (`float`, etc.).
- Pydantic will accept inputs like `"nan"`/`"inf"` (or actual `float("nan")`) for `float` fields, producing non-finite floats in pipeline data.
- Canonical hashing (`stable_hash`) explicitly rejects non-finite floats and raises `ValueError`, which can crash the pipeline downstream (e.g., when executors compute input/output hashes).
- These values should be treated as invalid external data and quarantined at the source boundary, not crash the run later.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any source input containing `nan`/`inf` for float fields

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Configure a source schema that includes a `float` field (explicit schema).
2. Provide an external input row where that field is `"nan"` or `"inf"`.
3. Run the pipeline.
4. Observe that the run can crash later when hashing/canonicalizing row data (rather than quarantining at the source).

## Expected Behavior

- Rows containing non-finite floats should fail source validation and be quarantined (or discarded) as invalid external data.

## Actual Behavior

- Non-finite floats can be accepted during source coercion and only crash later during hashing/canonicalization.

## Evidence

- Schema factory maps `"float"` to Python `float` and relies on coercion when `allow_coercion=True`: `src/elspeth/plugins/schema_factory.py:23-126`
- Canonical hashing rejects NaN/Infinity and raises: `src/elspeth/core/canonical.py:47-56`

## Impact

- User-facing impact: pipelines crash on specific data values that should be quarantined; violates "process the other 10,000 rows".
- Data integrity / security impact: audit trail cannot represent NaN/Inf and thus cannot preserve complete provenance unless quarantined at boundary.
- Performance or cost impact: reruns and manual data cleaning.

## Root Cause Hypothesis

- Source schema validation treats "non-finite float" as valid numeric data, but canonicalization treats it as invalid/unrepresentable in the audit model.

## Proposed Fix

- Code changes (modules/files):
  - Enforce `allow_inf_nan=False` for float fields at the source boundary:
    - use constrained float type (e.g., `pydantic.confloat(allow_inf_nan=False)`) or a post-parse validator that rejects `math.isnan` / `math.isinf`.
  - Ensure the error path is quarantinable (record as validation error, yield `SourceRow.quarantined` when configured).
- Config or schema changes:
  - Document non-finite float policy as part of the schema contract.
- Tests to add/update:
  - Add a source plugin test that provides `"nan"`/`"inf"` for a float field and asserts the row is quarantined (not accepted as valid).
- Risks or migration steps:
  - Some users may have relied on NaN values; treat as invalid per canonical policy and require explicit normalization upstream.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 3 external data handling + `src/elspeth/core/canonical.py` strict NaN policy.
- Observed divergence: values that are unhashable/unrepresentable in audit trail can still enter pipeline data.
- Reason (if known): schema factory uses plain `float` type without finiteness constraints.
- Alignment plan or decision needed: define canonical policy for NaN/Inf at ingestion (recommended: reject/quarantine).

## Acceptance Criteria

- Non-finite floats are rejected during source validation and handled via quarantine/discard policy.
- No downstream hashing/canonicalization crash occurs due to NaN/Inf entering pipeline data.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/`
- New tests required: yes

## Notes / Links

- Related code: `src/elspeth/core/canonical.py` (audit integrity policy)

---

## Resolution

**Status:** CLOSED
**Resolved by:** Claude
**Date:** 2026-01-21
**Commit:** (pending)

### Root Cause

The `TYPE_MAP` in `schema_factory.py` mapped `"float"` to Python's plain `float` type, which accepts `NaN` and `Infinity` values. This violated the Three-Tier Trust Model: external data containing non-finite floats passed source validation but then crashed during canonical hashing (which correctly rejects them per RFC 8785).

### Fix Applied

Created a `FiniteFloat` type using Pydantic's `Annotated` with `Field(allow_inf_nan=False)`:

```python
# In src/elspeth/plugins/schema_factory.py
from typing import Annotated
from pydantic import Field

# Finite float type that rejects NaN and Infinity at the source boundary.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": FiniteFloat,  # Rejects NaN/Infinity
    "bool": bool,
    "any": Any,
}
```

### Tests Added

Added `TestNonFiniteFloatRejection` class in `tests/plugins/test_schema_factory.py` with 7 test cases:
- `test_nan_string_rejected_in_float_field` - Rejects `"nan"` string
- `test_infinity_string_rejected_in_float_field` - Rejects `"inf"` string
- `test_negative_infinity_rejected_in_float_field` - Rejects `"-inf"` string
- `test_actual_nan_float_rejected` - Rejects `float("nan")`
- `test_actual_infinity_float_rejected` - Rejects `float("inf")`
- `test_optional_float_still_rejects_nan` - Optional floats also reject NaN
- `test_finite_floats_still_accepted` - Normal finite floats work correctly

### Verification

- All 28 schema factory tests pass
- All 2889 project tests pass (no regressions)
- mypy and ruff checks pass

### Architectural Alignment

This fix aligns with CLAUDE.md Three-Tier Trust Model:
- **Tier 3 (External Data)**: NaN/Infinity now rejected at source boundary via schema validation
- **Tier 1 (Audit DB)**: No change needed; canonical.py correctly rejects NaN/Infinity
- **Defense in depth**: Two layers of protection (schema + canonical) for audit integrity
