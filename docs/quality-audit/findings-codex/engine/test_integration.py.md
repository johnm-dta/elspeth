# Test Defect Report

## Summary

- Audit trail verification test only checks run/nodes/rows/artifacts counts and omits required node_states, token_outcomes, and hash integrity assertions, so audit regressions can pass unnoticed.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_integration.py:255` shows the “audit trail verification” test stops at run/nodes/rows/artifacts and never inspects node_states/token_outcomes/hashes.
```python
# Verify audit trail
from elspeth.contracts import RunStatus

recorder = LandscapeRecorder(db)
run = recorder.get_run(result.run_id)
assert run is not None
assert run.status == RunStatus.COMPLETED

nodes = recorder.get_nodes(result.run_id)
assert len(nodes) == 3

rows = recorder.get_rows(result.run_id)
assert len(rows) == 3

artifacts = recorder.get_artifacts(result.run_id)
assert len(artifacts) == 1
assert artifacts[0].content_hash == "abc123"
```
- `src/elspeth/core/landscape/schema.py:169` and `src/elspeth/core/landscape/schema.py:119` show audit-critical fields (e.g., `input_hash`, `output_hash`, `error_json`, `token_outcomes.outcome`, `is_terminal`) that the test never asserts.
```python
# node_states_table
Column("input_hash", String(64), nullable=False),
Column("output_hash", String(64)),
Column("error_json", Text),
```
```python
# token_outcomes_table
Column("outcome", String(32), nullable=False),
Column("is_terminal", Integer, nullable=False),
```

## Impact

- Audit integrity regressions (missing input/output hashes, wrong terminal states, or absent error recording) would not be detected.
- Creates false confidence that the audit trail is complete when only counts are verified.

## Root Cause Hypothesis

- Focus on happy‑path pipeline success without a reusable audit verification helper, leading to count‑only checks in “audit” tests.

## Recommended Fix

- Extend `tests/engine/test_integration.py:163` to assert node_states and token_outcomes details:
  - Query node_states for each token and verify `status`, `input_hash`, `output_hash`, and `error_json` fields.
  - Query token_outcomes to assert exactly one terminal outcome per token and expected terminal state.
  - Validate artifacts include `artifact_type`, `content_hash`, and expected `produced_by_state_id`.
- Example pattern (adapt to existing recorder helpers):
```python
states = recorder.get_node_states_for_token(token.token_id)
assert all(s.input_hash for s in states)
assert all(s.output_hash for s in states)
assert all(s.error_json is None for s in states)
outcomes = recorder.get_token_outcomes(token.token_id)
assert len([o for o in outcomes if o.is_terminal]) == 1
```
---
# Test Defect Report

## Summary

- Extensive duplication of inline `ListSource`/`CollectSink` class setups across many tests creates high maintenance overhead and inconsistent fixtures.

## Severity

- Severity: minor
- Priority: P2

## Category

- Fixture Duplication

## Evidence

- `tests/engine/test_integration.py:180` defines `ListSource`/`CollectSink` inline for one test.
```python
class ListSource(_TestSourceBase):
    name = "test_source"
    output_schema = ValueSchema
    ...
class CollectSink(_TestSinkBase):
    name = "output_sink"
    ...
```
- `tests/engine/test_integration.py:565` repeats the same structure with minimal variation; the file contains >10 such blocks (see line list from `rg`: 180, 296, 429, 565, 665, 748, 912, 2223, 2465, 2625, 3266, 3386).
```python
class ListSource(_TestSourceBase):
    name = "source"
    output_schema = RowSchema
    ...
class CollectSink(_TestSinkBase):
    name = "default_sink"
    ...
```

## Impact

- Increases maintenance cost when shared behavior changes (e.g., SourceRow handling or sink write semantics).
- Encourages subtle divergence between tests and raises risk of inconsistent fixtures masking regressions.

## Root Cause Hypothesis

- Lack of shared pytest fixtures or helper factories for common test sources/sinks.

## Recommended Fix

- Introduce module‑level helper classes or pytest fixtures in `tests/conftest.py` (or at top of this file) to centralize `ListSource`/`CollectSink` behavior, and parametrize data per test.
- Example approach:
```python
@pytest.fixture
def list_source():
    def _make(data, schema):
        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = schema
            ...
        return ListSource(data)
    return _make
```
