# Bug Report: Gate/Sink Executions Don't Initialize PluginContext state_id

## Summary

- PluginContext for gates and sinks missing `state_id`, preventing external call recording for these plugin types.

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: Branch Bug Scan
- Date: 2026-01-25
- Related run/issue ID: BUG-RECORDER-01

## Evidence

- `src/elspeth/core/landscape/recorder.py` - PluginContext created without state_id for gates/sinks
- External API calls from gates/sinks cannot be recorded

## Impact

- Audit completeness: Cannot trace external calls made by gates or sinks

## Proposed Fix

```python
# Pass state_id through PluginContext for all plugin types
ctx = PluginContext(state_id=state_id, ...)
```

## Acceptance Criteria

- Gate/sink external calls recordable in call_audit table

## Tests

- New tests required: yes, test gate/sink external call recording

---

## RESOLUTION: 2026-01-29

**Status:** FIXED

**Closed By:** Claude Code bug review

**Note:** This is a duplicate of `P2-2026-01-21-gate-sink-missing-state-id-for-record-call.md`.

**Fix Details:**

The bug was fixed in commit `b5f3f50` ("fix(infra): thread safety, integration tests, and Azure audit trail").

The fix sets `ctx.state_id`, `ctx.node_id`, and `ctx._call_index` in both:
- `GateExecutor.execute_gate()` (executors.py:492-494)
- `SinkExecutor.write()` (executors.py:1655-1657)

Gates and sinks can now call `ctx.record_call()` to record external API calls in the audit trail.
