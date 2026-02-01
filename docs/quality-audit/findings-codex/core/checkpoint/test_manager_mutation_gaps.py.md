# Test Defect Report

## Summary

- Tests reach into the private `manager._db` attribute for setup, coupling the tests to internal implementation details.

## Severity

- Severity: minor
- Priority: P2

## Category

- Infrastructure Gaps

## Evidence

- `tests/core/checkpoint/test_manager_mutation_gaps.py:54`
- `tests/core/checkpoint/test_manager_mutation_gaps.py:175`
- Code snippet:
```python
db = manager._db
with db.engine.connect() as conn:
    conn.execute(
        runs_table.insert().values(
            run_id="run-001",
            ...
        )
    )
```

## Impact

- Tests become brittle if `CheckpointManager` encapsulates or renames `_db`.
- Setup bypasses any future public API or fixtures intended to manage DB lifecycle, reducing maintainability and clarity.

## Root Cause Hypothesis

- Fixtures were copied for speed and reused without a shared `db` fixture, leading to direct access of internal state.

## Recommended Fix

- Introduce an explicit `db` fixture and use it in `manager` and `setup_run` to avoid private attribute access. Example:
```python
@pytest.fixture
def db(tmp_path: Path) -> LandscapeDB:
    return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

@pytest.fixture
def manager(db: LandscapeDB) -> CheckpointManager:
    return CheckpointManager(db)

@pytest.fixture
def setup_run(db: LandscapeDB) -> str:
    with db.engine.connect() as conn:
        ...
```
- This keeps test setup stable if `CheckpointManager` internals change.
---
# Test Defect Report

## Summary

- The ordering tests only use a single run (`run-001`), so they do not verify that `get_latest_checkpoint` filters by `run_id`; a mutation removing the run filter would still pass these tests.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Edge Cases

## Evidence

- `tests/core/checkpoint/test_manager_mutation_gaps.py:178-186` (only `run-001` is inserted)
- `tests/core/checkpoint/test_manager_mutation_gaps.py:231-236` (only `run-001` checkpoints are created and queried)
- Code snippet:
```python
conn.execute(
    runs_table.insert().values(
        run_id="run-001",
        ...
    )
)
...
manager.create_checkpoint("run-001", "tok-001", "node-001", 1, mock_graph)
latest = manager.get_latest_checkpoint("run-001")
```

## Impact

- A regression that drops or alters the `run_id` filter could return a checkpoint from a different run, yet these tests would still pass.
- This risks cross-run checkpoint contamination during recovery and false confidence in ordering behavior.

## Root Cause Hypothesis

- The tests were narrowly scoped to ordering mutants and did not include multi-run scenarios.

## Recommended Fix

- Add a multi-run test that creates checkpoints for two runs and asserts per-run filtering:
```python
def test_get_latest_filters_by_run_id(self, manager, mock_graph) -> None:
    run_a = _insert_run_and_token(manager, "run-001", "node-001", "row-001", "tok-001")
    run_b = _insert_run_and_token(manager, "run-002", "node-002", "row-002", "tok-002")

    manager.create_checkpoint("run-001", "tok-001", "node-001", 1, mock_graph)
    manager.create_checkpoint("run-002", "tok-002", "node-002", 99, mock_graph)

    assert manager.get_latest_checkpoint("run-001").run_id == "run-001"
    assert manager.get_latest_checkpoint("run-002").run_id == "run-002"
```
- This directly validates the `run_id` filter while keeping the ordering coverage intact.
