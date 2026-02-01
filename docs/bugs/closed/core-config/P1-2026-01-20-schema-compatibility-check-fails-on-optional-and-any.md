# Bug Report: Schema compatibility check rejects valid Optional/Any cases (false negatives)

## Summary

- `check_compatibility()` is intended to validate producer/consumer schema alignment at config time, allowing coercible cases like `int -> float`.
- The current `_types_compatible()` implementation fails when the consumer expects a Union that *includes* a coercible member (e.g., `Optional[float]`) and when the consumer expects `Any`, causing false incompatibility results.
- Error messages are also misleading because `_type_name()` collapses `Optional[float]` to `Optional`, hiding the underlying type.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into System 2 (Contracts) and write bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: local code inspection + minimal Python repro

## Steps To Reproduce

### Case A: Optional[float] consumer incorrectly rejected

```python
from typing import Optional
from elspeth.contracts.data import PluginSchema, check_compatibility

class Producer(PluginSchema):
    x: int

class Consumer(PluginSchema):
    x: Optional[float]

print(check_compatibility(Producer, Consumer))
```

### Case B: Any consumer incorrectly rejected

```python
from typing import Any
from elspeth.contracts.data import PluginSchema, check_compatibility

class Producer(PluginSchema):
    x: int

class Consumer(PluginSchema):
    x: Any

print(check_compatibility(Producer, Consumer))
```

## Expected Behavior

- Case A: Compatible (`int` is acceptable for `Optional[float]` given the documented numeric coercion rule).
- Case B: Compatible (consumer `Any` should accept any producer type).

## Actual Behavior

- Case A: Returns incompatible with a mismatch like `('x', 'Optional', 'int')`.
- Case B: Returns incompatible with a mismatch like `('x', 'Any', 'int')`.

## Evidence

- `check_compatibility()` calls `_types_compatible(producer_field.annotation, consumer_field.annotation)`:
  - `src/elspeth/contracts/data.py:134-189`
- `_types_compatible()` only applies numeric compatibility when `expected is float`, not when `expected` is a Union containing `float`:
  - `src/elspeth/contracts/data.py:205-233`
- `_type_name()` uses `__name__` for typing constructs, collapsing `Optional[float]` to `Optional`:
  - `src/elspeth/contracts/data.py:192-197`

## Impact

- User-facing impact: pipelines can be rejected during config validation even though runtime validation/coercion would accept them (false negatives).
- Data integrity / security impact: low (validation-time), but it blocks valid pipelines and undermines trust in schema validation.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `_types_compatible()` treats Union compatibility as exact-member matching only (`actual in expected_args`) and does not evaluate coercible compatibility against Union members.
- `Any` is treated as a concrete type rather than a top type.
- `_type_name()` is overly lossy for typing constructs.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/data.py`:
    - Treat `expected is Any` as compatible with any `actual`.
    - For Union `expected`, return true if `actual` is compatible with *any* member (including numeric coercion). For Union `actual`, ensure each member is compatible with at least one expected member (subset semantics with coercion).
    - Improve `_type_name()` to preserve detail for `typing` constructs (prefer `str(t)` for `typing.*` objects; optionally special-case Union/Optional for readability).
- Tests to add/update:
  - Add unit tests for:
    - `int -> Optional[float]` compatibility
    - `int | None -> Optional[float]` compatibility (coercion within unions)
    - `int -> Any` compatibility
    - error messages include full types (e.g., `Optional[float]`, not just `Optional`)

## Architectural Deviations

- Spec or doc reference: `src/elspeth/contracts/data.py` docstring claims Union/optional handling and coercion support
- Observed divergence: current compatibility logic rejects common optional/Any cases
- Alignment plan or decision needed: confirm intended semantics for Any and coercion-inside-unions (current docs imply “yes”)

## Acceptance Criteria

- `check_compatibility(Producer(int), Consumer(Optional[float]))` returns compatible.
- `check_compatibility(Producer(int), Consumer(Any))` returns compatible.
- Compatibility error messages preserve meaningful type detail (not collapsed to `Optional`).

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/ -k compatibility`
- New tests required: yes

## Notes / Links

- Related modules: `src/elspeth/plugins/schema_factory.py` (creates optional field unions via `base_type | None`)
