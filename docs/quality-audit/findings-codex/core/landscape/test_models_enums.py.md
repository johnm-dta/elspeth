# Test Defect Report

## Summary

- Assertions are tautological; they only compare the field to the same enum instance passed in, so they do not validate any runtime typing or conversion behavior.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_models_enums.py:20` sets `export_status=ExportStatus.PENDING`, and `tests/core/landscape/test_models_enums.py:29` immediately asserts the same value without checking type or serialization behavior.
```
run = Run(
    ...,
    export_status=ExportStatus.PENDING,
)
assert run.export_status == ExportStatus.PENDING
```
- The same tautological pattern exists for `node_type`, `determinism`, and `default_mode` (`tests/core/landscape/test_models_enums.py:33`, `tests/core/landscape/test_models_enums.py:44`, `tests/core/landscape/test_models_enums.py:48`, `tests/core/landscape/test_models_enums.py:59`, `tests/core/landscape/test_models_enums.py:63`, `tests/core/landscape/test_models_enums.py:72`).

## Impact

- Regressions that convert enum fields to strings or otherwise mutate types would still pass, creating false confidence around enum typing in audit models.

## Root Cause Hypothesis

- Tests were written as smoke checks for constructor usage rather than behavioral checks for type enforcement.

## Recommended Fix

- Strengthen assertions to validate runtime type and enum value rather than identity-only checks; keep in `tests/core/landscape/test_models_enums.py`.
```
assert isinstance(run.export_status, ExportStatus)
assert run.export_status.value == "pending"
```
---
# Test Defect Report

## Summary

- No Tier 1 corruption tests exist for invalid enum values; tests only cover valid enums and never assert that bad inputs crash.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- All tests use valid enums and no `pytest.raises` checks appear, e.g. `tests/core/landscape/test_models_enums.py:33` sets `node_type=NodeType.TRANSFORM` and `tests/core/landscape/test_models_enums.py:44` asserts it; there is no test for invalid strings or wrong enums.
```
node = Node(
    ...,
    node_type=NodeType.TRANSFORM,
    determinism=Determinism.DETERMINISTIC,
    ...
)
assert node.node_type == NodeType.TRANSFORM
```
- The same pattern repeats for `determinism` and `default_mode` without any invalid-input coverage (`tests/core/landscape/test_models_enums.py:48`, `tests/core/landscape/test_models_enums.py:59`, `tests/core/landscape/test_models_enums.py:63`, `tests/core/landscape/test_models_enums.py:72`).

## Impact

- Invalid enum values in Tier 1 audit data could be accepted silently, violating the “crash on anomaly” requirement and weakening audit integrity; tests would not catch such regressions.

## Root Cause Hypothesis

- Focus on happy-path enum acceptance without encoding the Tier 1 “invalid enum must crash” requirement in tests.

## Recommended Fix

- Add explicit negative tests in `tests/core/landscape/test_models_enums.py` that pass invalid strings or wrong enum types for non-optional fields and assert a crash via `pytest.raises` (TypeError/ValueError per desired validation).
```
import pytest

@pytest.mark.parametrize("bad_value", ["transform", 123, None])
def test_node_type_rejects_invalid_enum(bad_value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="test",
            node_type=bad_value,  # type: ignore[arg-type]
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="abc",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
```
