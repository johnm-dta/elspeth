# Test Defect Report

## Summary

- Repeated fixture definitions across multiple classes create duplication and drift risk in the recovery mutation-gap tests.

## Severity

- Severity: minor
- Priority: P2

## Category

- Fixture Duplication

## Evidence

- `tests/core/checkpoint/test_recovery_mutation_gaps.py:153` and `tests/core/checkpoint/test_recovery_mutation_gaps.py:281` both define `in_memory_db`, `checkpoint_manager`, `recovery_manager`, and `recorder` fixtures.
- `tests/core/checkpoint/test_recovery_mutation_gaps.py:179` and `tests/core/checkpoint/test_recovery_mutation_gaps.py:308` (also `tests/core/checkpoint/test_recovery_mutation_gaps.py:423` and `tests/core/checkpoint/test_recovery_mutation_gaps.py:629`) reintroduce near-identical `mock_graph` fixtures.

```python
@pytest.fixture
def in_memory_db(self) -> LandscapeDB:
    from elspeth.core.landscape import LandscapeDB
    return LandscapeDB.in_memory()

@pytest.fixture
def checkpoint_manager(self, in_memory_db: LandscapeDB) -> CheckpointManager:
    from elspeth.core.checkpoint import CheckpointManager
    return CheckpointManager(in_memory_db)
```

## Impact

- Increases maintenance cost and risk of inconsistent setup when fixture behavior changes.
- Makes mutation-gap tests brittle because setup updates must be duplicated across classes, increasing drift risk.

## Root Cause Hypothesis

- Tests were added via copy/paste while targeting mutation survivors without consolidating shared setup.

## Recommended Fix

- Move shared fixtures to module scope in `tests/core/checkpoint/test_recovery_mutation_gaps.py` or to `tests/conftest.py`.
- Use parametrized fixtures or a small fixture factory for graph variants to avoid multiple near-identical `mock_graph` definitions.
- This low-risk refactor reduces drift and keeps setup consistent across all mutation-gap tests.
---
# Test Defect Report

## Summary

- Error-message assertions rely on substring matches, which can let incorrect failure reasons or wrong branches pass.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/checkpoint/test_recovery_mutation_gaps.py:186`, `tests/core/checkpoint/test_recovery_mutation_gaps.py:236`, `tests/core/checkpoint/test_recovery_mutation_gaps.py:589` only check substrings like `"not found"`, `"checkpoint"`, or `"purged"` instead of validating the full reason or key identifiers.

```python
assert "not found" in result.reason.lower()
assert "checkpoint" in result.reason.lower()
assert "purged" in str(exc_info.value).lower()
```

## Impact

- A regression that returns the wrong reason (or wrong branch) can still satisfy the substring, masking failures.
- Reduces mutation-test strength, leaving surviving mutants that alter or degrade error reporting.

## Root Cause Hypothesis

- Assertions were kept loose to avoid brittle string comparisons while targeting mutation survivors.

## Recommended Fix

- Assert full error messages where deterministic (include run_id/row_id in expected strings).
- If message text must be flexible, use anchored regex patterns that validate structure and required identifiers rather than substring checks.
- Keep the stronger checks in this file to maximize mutation-test sensitivity.
