# Bug Report: Replayer returns empty dict instead of failing when SUCCESS call lacks payload

## Summary

`CallReplayer.replay()` returns `{}` instead of raising `ReplayPayloadMissingError` when a SUCCESS call was recorded without persisting the response payload. The code checks only `response_ref` to determine if a response ever existed, but `response_hash` (or `call.status == SUCCESS`) proves a response DID exist - it just wasn't persisted to the payload store.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Claude Opus 4.5
- Date: 2026-02-02
- Related run/issue ID: N/A

## Environment

- Commit/branch: `RC1-bugs-final` @ `e63cb03c`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Triage code review findings and create bug tickets
- Model/version: Claude Opus 4.5
- Tooling and permissions: workspace-write
- Determinism details: N/A
- Notable tool calls or steps: code inspection

## Steps To Reproduce

1. Run a pipeline WITHOUT a payload store configured (or with payload storage disabled)
2. A transform makes an LLM/HTTP call that succeeds with a response
3. The call is recorded: `response_hash` is set, but `response_ref` is `None` (no payload store)
4. Run replay mode against this recorded run
5. For the same request, `replay()` returns `{}` instead of the original response

## Expected Behavior

- If a call has `response_hash` set (or `status == SUCCESS`), this proves a response existed
- If `response_data` cannot be retrieved, `ReplayPayloadMissingError` should be raised
- Caller should know the replay is unreliable, not receive silent `{}`

## Actual Behavior

- Code checks only `call.response_ref` (line 216)
- If `response_ref is None`, assumes "call never had a response" (line 214 comment)
- Returns `{}` as if the original call returned empty (line 221)
- Replay silently diverges from actual recorded behavior

## Evidence

```python
# src/elspeth/plugins/clients/replayer.py:212-221
# Fail if payload was recorded but is now missing (purged).
# Check response_ref to distinguish between:
# - response_ref=None → call never had a response (OK to use {} for errors)
# - response_ref set but response_data=None → response was purged (FAIL)
if response_data is None and call.response_ref is not None:
    raise ReplayPayloadMissingError(call.call_id, request_hash)

# For calls that never had a response (response_ref is None), use empty dict
if response_data is None:
    response_data = {}
```

Contrast with recorder logic (`recorder.py:1881-1894`):
```python
# Hash response (optional - None for errors without response)
response_hash = stable_hash(response_data) if response_data is not None else None

# Auto-persist response to payload store if available and ref not provided
if response_data is not None and response_ref is None and self._payload_store is not None:
    response_bytes = canonical_json(response_data).encode("utf-8")
    response_ref = self._payload_store.store(response_bytes)
```

Key insight: `response_hash` is set when `response_data is not None`, but `response_ref` is only set if `_payload_store is not None`. A run without payload store will have `response_hash` but not `response_ref`.

## Impact

- User-facing impact: Replay mode silently returns wrong data, causing transform behavior to diverge
- Data integrity / security impact: **HIGH** - replay is a core auditability feature; silent divergence undermines trust
- Performance or cost impact: None

## Root Cause Hypothesis

The code was written assuming `response_ref=None` only occurs for genuine "no response" cases (connection timeouts, DNS failures). It doesn't account for the configuration case where a payload store simply wasn't available during recording.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/clients/replayer.py:212-221`
  - Change the condition to check `response_hash` or `status == SUCCESS`:
    ```python
    # A response existed if response_hash is set OR status is SUCCESS
    response_expected = call.response_hash is not None or call.status == CallStatus.SUCCESS

    if response_data is None and response_expected:
        raise ReplayPayloadMissingError(call.call_id, request_hash)

    # Only use {} for genuine no-response cases (errors without response)
    if response_data is None:
        response_data = {}
    ```
- Config or schema changes: None
- Tests to add/update:
  - Add test: SUCCESS call recorded without payload store → replay should fail
  - Add test: Call with `response_hash` set but no payload → replay should fail
  - Add test: ERROR call with no response_hash → replay returns `{}` (this is correct)
- Risks or migration steps: None - this makes replay stricter, which is safer

## Architectural Deviations

- Spec or doc reference: Auditability Standard in CLAUDE.md
- Observed divergence: Silent data divergence in replay mode
- Reason (if known): Incomplete condition check
- Alignment plan or decision needed: None - fix is clear

## Acceptance Criteria

- [ ] Replay raises `ReplayPayloadMissingError` when `response_hash` is set but payload unavailable
- [ ] Replay raises `ReplayPayloadMissingError` when `status == SUCCESS` but payload unavailable
- [ ] Replay still returns `{}` for genuine no-response error cases
- [ ] Tests cover all three scenarios

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/test_replayer.py`
- New tests required: See above

## Notes / Links

- Related issues/PRs: P2-2026-02-02-verifier-silent-payload-gaps (same root cause)
