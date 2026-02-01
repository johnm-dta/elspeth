# Bug Report: Verifier misses payload gaps for SUCCESS calls without response_ref

## Summary

`CallVerifier.verify()` doesn't increment `missing_payloads` when a SUCCESS call was recorded without persisting the response payload. The code checks only `response_ref` to determine if a response ever existed, but `response_hash` (or `call.status == SUCCESS`) proves a response DID exist - it just wasn't persisted to the payload store.

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
4. Run verify mode against this recorded run
5. Verification report shows `missing_payloads = 0` even though payloads ARE missing
6. Verification result for the call shows `is_match=False` but doesn't flag `payload_missing=True`

## Expected Behavior

- If a call has `response_hash` set (or `status == SUCCESS`), this proves a response existed
- If `recorded_response` cannot be retrieved, increment `missing_payloads` counter
- Set `payload_missing=True` on the `VerificationResult`
- Verification report accurately reflects audit gaps

## Actual Behavior

- Code checks only `call.response_ref` (line 215)
- If `response_ref is None` and `recorded_response is None`, treats as "call never had a response" (line 213-214 comment)
- Creates a `VerificationResult` without `payload_missing=True` (line 230-236)
- Does NOT increment `missing_payloads` counter
- Verification report masks audit gaps

## Evidence

```python
# src/elspeth/plugins/clients/verifier.py:211-237
# Handle missing/purged payload explicitly.
# Distinguish between:
# - response_ref=None → call never had a response (NOT a missing payload)
# - response_ref set but response=None → payload was purged (IS missing payload)
if recorded_response is None and call.response_ref is not None:
    result = VerificationResult(
        request_hash=request_hash,
        live_response=live_response,
        recorded_response=None,
        is_match=False,
        payload_missing=True,
    )
    self._report.missing_payloads += 1
    self._report.results.append(result)
    return result

# Call never had a response (e.g., connection timeout, DNS failure)
# Cannot compare, so not a match, but NOT a missing payload
if recorded_response is None:
    result = VerificationResult(
        request_hash=request_hash,
        live_response=live_response,
        recorded_response=None,
        is_match=False,
    )
    self._report.results.append(result)
    return result
```

The second `if` block (lines 229-237) catches SUCCESS calls where payload wasn't persisted and incorrectly classifies them as "call never had a response" instead of "payload is missing".

## Impact

- User-facing impact: Verification reports undercount audit gaps
- Data integrity / security impact: **HIGH** - verification is a core auditability feature; undercounting gaps gives false confidence
- Performance or cost impact: None

## Root Cause Hypothesis

Same as replayer bug: the code assumes `response_ref=None` only occurs for genuine "no response" cases. It doesn't account for runs without a payload store.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/clients/verifier.py:211-237`
  - Change the condition to check `response_hash` or `status == SUCCESS`:
    ```python
    # A response existed if response_hash is set OR status is SUCCESS
    response_expected = call.response_hash is not None or call.status == CallStatus.SUCCESS

    if recorded_response is None and response_expected:
        result = VerificationResult(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=None,
            is_match=False,
            payload_missing=True,
        )
        self._report.missing_payloads += 1
        self._report.results.append(result)
        return result

    # Genuine no-response case (errors without response)
    if recorded_response is None:
        # ... existing logic ...
    ```
- Config or schema changes: None
- Tests to add/update:
  - Add test: SUCCESS call recorded without payload store → `payload_missing=True`, counter incremented
  - Add test: Call with `response_hash` set but no payload → `payload_missing=True`
  - Add test: ERROR call with no response_hash → NOT counted as missing payload
- Risks or migration steps: None - this makes verification stricter, which is safer

## Architectural Deviations

- Spec or doc reference: Auditability Standard in CLAUDE.md
- Observed divergence: Verification report undercounts audit gaps
- Reason (if known): Incomplete condition check
- Alignment plan or decision needed: None - fix is clear

## Acceptance Criteria

- [ ] Verifier sets `payload_missing=True` when `response_hash` is set but payload unavailable
- [ ] Verifier increments `missing_payloads` when `status == SUCCESS` but payload unavailable
- [ ] Verifier still treats genuine no-response error cases as NOT missing payload
- [ ] Tests cover all three scenarios

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/test_verifier.py`
- New tests required: See above

## Notes / Links

- Related issues/PRs: P2-2026-02-02-replayer-silent-empty-response-for-success-calls (same root cause)
