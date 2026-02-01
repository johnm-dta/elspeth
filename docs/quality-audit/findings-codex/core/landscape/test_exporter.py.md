# Test Defect Report

## Summary

- No test validates export of external call records (`record_type == "call"`), leaving a critical audit trail element unverified

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `src/elspeth/core/landscape/exporter.py:318` emits `record_type == "call"` records, but the test file never asserts on call records (record-type checks in this file stop at routing events such as `tests/core/landscape/test_exporter.py:446`).
- Code snippets:
```python
# src/elspeth/core/landscape/exporter.py
for call in self._recorder.get_calls(state.state_id):
    yield {
        "record_type": "call",
        "call_id": call.call_id,
        "status": call.status,
        "request_hash": call.request_hash,
        "response_hash": call.response_hash,
    }
```
```python
# tests/core/landscape/test_exporter.py
event_records = [r for r in records if r["record_type"] == "routing_event"]
assert len(event_records) == 1
```

## Impact

- Export regressions that drop or mis-serialize external calls would go undetected.
- Breaks auditability requirement to capture full external request/response evidence.

## Root Cause Hypothesis

- Call-record export was added after initial tests; coverage focused on core record types and missed external calls.

## Recommended Fix

- Add `test_exporter_extracts_calls()` that records a call via `LandscapeRecorder.record_call(...)`, exports, and asserts the call record fields (`call_id`, `state_id`, `call_type`, `status`, `request_hash`, `response_hash`, `latency_ms`) plus `record_type == "call"`.
- Priority is P1 because external calls are explicitly part of the audit trail.
---
# Test Defect Report

## Summary

- Manifest hash chain integrity is not validated; tests only check that `final_hash` exists

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_exporter.py:514` only asserts that `final_hash` exists without verifying the hash chain contents.
- `src/elspeth/core/landscape/exporter.py:123` and `src/elspeth/core/landscape/exporter.py:137` show `final_hash` is computed from the ordered signature chain.
- Code snippets:
```python
# tests/core/landscape/test_exporter.py
manifest = manifest_records[0]
assert "record_count" in manifest
assert "final_hash" in manifest
```
```python
# src/elspeth/core/landscape/exporter.py
running_hash.update(record["signature"].encode())
...
"final_hash": running_hash.hexdigest(),
```

## Impact

- A broken hash chain (wrong order, missing records, wrong algorithm) could ship without detection.
- Undermines the tamper-evidence mechanism of signed exports.

## Root Cause Hypothesis

- Tests focus on schema presence rather than semantic correctness of cryptographic integrity checks.

## Recommended Fix

- Recompute expected `final_hash` in the test by hashing the signatures of all non-manifest records in export order and assert equality with `manifest["final_hash"]`.
- Keep the existing field-presence checks, but add a deterministic validation of the hash chain.
---
# Test Defect Report

## Summary

- Node state export tests only cover completed states; open/pending/failed variants are untested

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- `tests/core/landscape/test_exporter.py:250` uses `status="completed"` and `tests/core/landscape/test_exporter.py:265` only asserts `"completed"`.
- `src/elspeth/core/landscape/exporter.py:252` and `src/elspeth/core/landscape/exporter.py:284` show distinct branches for pending/open/failed states.
- Code snippets:
```python
# tests/core/landscape/test_exporter.py
recorder.complete_node_state(
    state_id=state.state_id,
    status="completed",
    output_data={"x": 1},
    duration_ms=10.0,
)
...
assert state_records[0]["status"] == "completed"
```
```python
# src/elspeth/core/landscape/exporter.py
elif isinstance(state, NodeStatePending):
    yield { ... "status": state.status.value, "output_hash": None, ... }
...
else:  # NodeStateFailed
    yield { ... "status": state.status.value, "output_hash": state.output_hash, ... }
```

## Impact

- Export regressions specific to pending/failed/open states (e.g., missing `completed_at`, wrong `output_hash` handling) would not be detected.
- Audit trail for error paths could be silently incorrect.

## Root Cause Hypothesis

- Happy-path bias in tests; error and in-progress states not modeled.

## Recommended Fix

- Add tests for each NodeState variant:
  - Open: create `begin_node_state`, export before completion, assert `completed_at` is None and `output_hash` is None.
  - Pending: `complete_node_state(status="pending", duration_ms=...)`, assert status and timing fields.
  - Failed: `complete_node_state(status="failed", error=...)`, assert status and completed timing fields.
- Keep completed-state test, but expand assertions for status-specific fields.
---
# Test Defect Report

## Summary

- No Tier 1 corruption tests verify the exporter crashes on invalid audit DB data

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- Only error-path test is for a missing run, not corrupt data: `tests/core/landscape/test_exporter.py:140`.
- Tier 1 policy requires crash on corrupted audit data: `CLAUDE.md:40`.
- Code snippets:
```python
# tests/core/landscape/test_exporter.py
with pytest.raises(ValueError, match="Run not found"):
    list(exporter.export_run("nonexistent_run_id"))
```
```markdown
# CLAUDE.md
- Bad data in the audit trail = **crash immediately**
```

## Impact

- If audit data is corrupted (NULLs, invalid enums, broken FK), the exporter could silently coerce or skip data without any test failing.
- Violates the Three-Tier Trust Model and weakens legal-grade auditability.

## Root Cause Hypothesis

- Negative testing limited to "not found" cases; corruption scenarios not modeled.

## Recommended Fix

- Add targeted corruption tests that insert invalid rows directly (bypassing recorder validation) and assert exporter crashes:
  - NULL in required columns,
  - invalid enum values,
  - broken FK references.
- Use the in-memory DB connection to insert corrupt data and confirm `export_run` raises.
