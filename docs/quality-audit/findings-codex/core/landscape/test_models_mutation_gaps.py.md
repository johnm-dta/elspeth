# Test Defect Report

## Summary

- NodeStatePending is part of the NodeState contract but is untested in this mutation-gap suite; union coverage stops at open/completed/failed.

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/core/landscape/test_models_mutation_gaps.py:426` shows union tests for Open/Completed/Failed only, with no Pending coverage.
```python
class TestNodeStateUnion:
    def test_node_state_open_is_node_state(self) -> None:
        ...
    def test_node_state_completed_is_node_state(self) -> None:
        ...
    def test_node_state_failed_is_node_state(self) -> None:
        ...
```
- `src/elspeth/core/landscape/models.py:141` defines `NodeStatePending`, and `src/elspeth/core/landscape/models.py:228` includes it in the union.
```python
@dataclass(frozen=True)
class NodeStatePending:
    ...
NodeState = NodeStateOpen | NodeStatePending | NodeStateCompleted | NodeStateFailed
```

## Impact

- Mutations to NodeStatePending defaults or required fields (e.g., making `completed_at` optional or changing status literal) would not be caught.
- Async/batch flows that rely on Pending state could regress without any test signal, weakening audit trail guarantees for deferred outputs.

## Root Cause Hypothesis

- NodeStatePending was added or modified after the mutation-gap suite was assembled, and the tests were not updated to include the new union member.

## Recommended Fix

- Add a `TestNodeStatePendingDataclass` fixture and assertions mirroring Completed/Failed: verify `completed_at` and `duration_ms` are required and that `context_before_json`/`context_after_json` default to None.
- Extend `TestNodeStateUnion` to include a `NodeStatePending` instance or assert `typing.get_args(NodeState)` contains `NodeStatePending`.
- Priority justification: Pending is a core audit state for async operations; gaps here can hide regressions in the audit trail contract.
---
# Test Defect Report

## Summary

- Checkpoint tests only cover `aggregation_state_json` defaults and never assert that key required fields (`created_at`, `upstream_topology_hash`, `checkpoint_node_config_hash`) are required.

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- `tests/core/landscape/test_models_mutation_gaps.py:659` defines `minimal_checkpoint` and only tests the optional `aggregation_state_json` field, with no missing-field assertions.
```python
class TestCheckpointDataclass:
    def minimal_checkpoint(self) -> Checkpoint:
        return Checkpoint(
            ...,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
        )

    def test_aggregation_state_json_defaults_to_none(...):
        ...
```
- `src/elspeth/core/landscape/models.py:334` shows these fields are required by the dataclass contract.
```python
checkpoint_id: str
run_id: str
token_id: str
node_id: str
sequence_number: int
created_at: datetime | None
upstream_topology_hash: str
checkpoint_node_config_hash: str
```

## Impact

- A mutation that makes these fields optional or defaulted would slip through, undermining checkpoint validation and recovery integrity.

## Root Cause Hypothesis

- The test file was scoped narrowly to a specific mutation survivor (`aggregation_state_json`) and missed required-field coverage for the checkpoint validation fields added in the Bug #7 fix.

## Recommended Fix

- Add tests that omit `created_at`, `upstream_topology_hash`, and `checkpoint_node_config_hash` and assert `TypeError` on construction.
- Optionally add a test that `created_at=None` is accepted when explicitly supplied, if that is a valid contract.
- Priority justification: these fields are documented as required for checkpoint validation and should be protected against default-removal mutations.
