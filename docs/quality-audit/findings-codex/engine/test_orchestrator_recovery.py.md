# Test Defect Report

## Summary

- `test_resume_method_exists` relies on `hasattr`/`callable` instead of a behavioral check, which is a bug-hiding defensive pattern and provides no functional coverage of `resume`.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/engine/test_orchestrator_recovery.py:143` defines a method-existence test instead of exercising behavior.
- `tests/engine/test_orchestrator_recovery.py:148` uses `hasattr` on system code.
- Code snippet from `tests/engine/test_orchestrator_recovery.py:143`:
  ```python
  def test_resume_method_exists(
      self,
      orchestrator: Orchestrator,
  ) -> None:
      """Orchestrator has resume() method."""
      assert hasattr(orchestrator, "resume")
      assert callable(orchestrator.resume)
  ```

## Impact

- This test passes even if `resume()` is wired incorrectly or misbehaves.
- It can suppress useful AttributeError details, reducing signal during failures.
- Creates false confidence without validating recovery behavior.

## Root Cause Hypothesis

- Added as a quick existence smoke test; defensive pattern used instead of a behavior-based assertion.

## Recommended Fix

- Replace with a behavior-focused assertion or remove this test (its existence is already exercised by other tests that call `resume()`).
- Example replacement that avoids `hasattr` and validates contract:
  ```python
  resume_point = recovery_manager.get_resume_point(run_id, mock_graph)
  assert resume_point is not None
  with pytest.raises(ValueError, match="payload_store"):
      orchestrator.resume(resume_point, config, graph)
  ```
---
# Test Defect Report

## Summary

- `test_resume_retries_failed_batches` only asserts that a retry batch exists, but does not verify critical invariants (attempt increment, member copying, run completion, checkpoint cleanup), making the assertions too weak for a core recovery path.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_orchestrator_recovery.py:176` only checks original batch status.
- `tests/engine/test_orchestrator_recovery.py:184`–`tests/engine/test_orchestrator_recovery.py:186` only checks `len(retry_batches) >= 1`.
- Code snippet from `tests/engine/test_orchestrator_recovery.py:176`:
  ```python
  original_batch = recorder.get_batch(original_batch_id)
  assert original_batch is not None
  assert original_batch.status == BatchStatus.FAILED

  all_batches = recorder.get_batches(run_id, node_id="agg_node")
  retry_batches = [b for b in all_batches if b.attempt > 0]
  assert len(retry_batches) >= 1
  ```

## Impact

- A regression where `retry_batch()` fails to increment `attempt` correctly or fails to copy batch members would still pass.
- Missing checks for run completion and checkpoint cleanup can hide recovery regressions (e.g., run left in failed status or stale checkpoints).
- False confidence in crash recovery correctness.

## Root Cause Hypothesis

- Focused on proving “a retry exists” rather than asserting the full batch recovery contract and audit trail integrity.

## Recommended Fix

- Strengthen assertions to validate critical invariants for recovery:
  ```python
  retry_batch = retry_batches[0]
  assert len(retry_batches) == 1
  assert retry_batch.attempt == original_batch.attempt + 1
  assert retry_batch.status == BatchStatus.DRAFT

  original_members = recorder.get_batch_members(original_batch_id)
  retry_members = recorder.get_batch_members(retry_batch.batch_id)
  assert [m.token_id for m in retry_members] == [m.token_id for m in original_members]

  run = recorder.get_run(run_id)
  assert run is not None and run.status.value == "completed"
  assert checkpoint_manager.get_checkpoints(run_id) == []
  ```
- Add the necessary fixtures/imports (`checkpoint_manager`, `RunStatus`) to support these checks.
