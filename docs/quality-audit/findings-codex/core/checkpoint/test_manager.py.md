Using using-quality-engineering (test-maintenance-patterns) to guide the test quality audit.

# Test Defect Report

## Summary

- Checkpoint creation/retrieval tests only assert a subset of fields, leaving token_id, node_id, created_at, and topology hash integrity unverified on critical paths.

## Severity

- Severity: major
- Priority: P1

## Category

- [Weak Assertions]

## Evidence

- `tests/core/checkpoint/test_manager.py:92` only asserts checkpoint_id/run_id/sequence_number after creation.
```python
# tests/core/checkpoint/test_manager.py:92
checkpoint = manager.create_checkpoint(
    run_id="run-001",
    token_id="tok-001",
    node_id="node-001",
    sequence_number=1,
    graph=mock_graph,
)

assert checkpoint.checkpoint_id is not None
assert checkpoint.run_id == "run-001"
assert checkpoint.sequence_number == 1
```
- `tests/core/checkpoint/test_manager.py:106` only checks the latest sequence number, not the rest of the record integrity.
```python
# tests/core/checkpoint/test_manager.py:106
latest = manager.get_latest_checkpoint("run-001")

assert latest is not None
assert latest.sequence_number == 3
```
- `src/elspeth/core/checkpoint/manager.py:86` computes required topology hashes that are never asserted in tests.
```python
# src/elspeth/core/checkpoint/manager.py:86
upstream_topology_hash = compute_upstream_topology_hash(graph, node_id)
checkpoint_node_config_hash = stable_hash(node_info.config)
```

## Impact

- Critical checkpoint metadata could be wrong (wrong node_id/token_id or hash mismatch) without tests failing.
- Resume compatibility and audit lineage correctness could silently regress.
- Creates false confidence that checkpoint integrity is verified when only sequencing is checked.

## Root Cause Hypothesis

- Tests were written as smoke checks for create/get flows and never expanded into field-level integrity checks.
- No deterministic fixture or helper exists to assert expected hash values, so assertions stayed minimal.

## Recommended Fix

- Expand assertions in `test_create_checkpoint`, `test_get_latest_checkpoint`, and `test_get_checkpoints_ordered` to validate token_id, node_id, created_at, and topology hashes.
- Use a deterministic graph and compute expected hashes for the assertion (verifies correct node/config selection even if the hash function is reused).
```python
# tests/core/checkpoint/test_manager.py (add assertions)
from elspeth.core.canonical import compute_upstream_topology_hash, stable_hash

expected_upstream = compute_upstream_topology_hash(mock_graph, "node-001")
expected_config = stable_hash(mock_graph.get_node_info("node-001").config)

assert checkpoint.token_id == "tok-001"
assert checkpoint.node_id == "node-001"
assert checkpoint.upstream_topology_hash == expected_upstream
assert checkpoint.checkpoint_node_config_hash == expected_config
assert checkpoint.created_at.tzinfo is not None
```
- Priority justification: these fields are required for resume safety and audit integrity, so weak assertions here mask critical regressions.
---
# Test Defect Report

## Summary

- No tests validate Tier 1 corruption handling for checkpoints; all retrievals assume valid audit data and only test the cutoff rule.

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Tier 1 Corruption Tests]

## Evidence

- `tests/core/checkpoint/test_manager.py:288` inserts only valid checkpoint rows when testing incompatibility, without any corruption scenarios.
```python
# tests/core/checkpoint/test_manager.py:288
checkpoints_table.insert().values(
    checkpoint_id=checkpoint_id,
    run_id=run_id,
    token_id="tok-old",
    node_id="node-old",
    sequence_number=1,
    aggregation_state_json=None,
    upstream_topology_hash="old-upstream-hash",
    checkpoint_node_config_hash="old-node-config-hash",
    created_at=old_date,
)
```
- `tests/core/checkpoint/test_manager.py:327` similarly inserts valid rows for acceptance tests; there is no test that tampers with required fields or invalid values.
```python
# tests/core/checkpoint/test_manager.py:327
checkpoints_table.insert().values(
    checkpoint_id=checkpoint_id,
    run_id="run-001",
    token_id="tok-001",
    node_id="node-001",
    sequence_number=1,
    aggregation_state_json=None,
    upstream_topology_hash="new-upstream-hash",
    checkpoint_node_config_hash="new-node-config-hash",
    created_at=new_date,
)
```

## Impact

- Corrupted audit data (empty/invalid hashes, malformed sequence_number, missing created_at) could be accepted or fail unpredictably without detection.
- Violates the Tier 1 requirement that audit DB anomalies must crash loudly and deterministically.
- Reduces confidence in recovery and audit integrity under data corruption scenarios.

## Root Cause Hypothesis

- Test focus is on happy-path checkpointing and the historical cutoff regression, with no dedicated corruption test harness.
- Lack of a standard corruption-test fixture leads to omission of negative coverage.

## Recommended Fix

- Add explicit corruption tests that tamper checkpoint rows and assert `get_latest_checkpoint` raises a clear error.
- Use raw SQL updates to introduce invalid-but-constraint-safe values (e.g., empty hash strings, negative sequence_number), and assert a specific exception/message.
```python
# tests/core/checkpoint/test_manager.py (new corruption test)
with manager._db.engine.connect() as conn:
    conn.exec_driver_sql(
        "update checkpoints set upstream_topology_hash = '' where checkpoint_id = :cid",
        {"cid": checkpoint_id},
    )
    conn.commit()

with pytest.raises(ValueError, match="upstream_topology_hash"):
    manager.get_latest_checkpoint("run-001")
```
- Priority justification: Tier 1 audit data corruption is a high-risk failure mode; tests should enforce crash behavior.
