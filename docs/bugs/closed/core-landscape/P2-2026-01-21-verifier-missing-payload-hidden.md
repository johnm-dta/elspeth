# Bug Report: CallVerifier hides missing payloads by substituting empty dicts

## Summary

- `CallVerifier` treats a missing response payload as `{}` by defaulting `get_call_response_data(...) or {}`. When payloads are purged or the payload store is disabled, verification proceeds against an empty baseline instead of reporting "missing recording" or "payload missing".

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of verifier payload handling

## Steps To Reproduce

1. Run with payload store disabled or purge response payloads.
2. Invoke `CallVerifier.verify` for a call that was recorded as SUCCESS.
3. Observe verification proceeds with `recorded_response={}` and does not mark missing payloads.

## Expected Behavior

- Missing response payloads should be explicitly flagged as "payload missing" (similar to `ReplayPayloadMissingError`) and excluded from match/mismatch counts.

## Actual Behavior

- Missing payloads are silently treated as empty dicts and compared against live responses.

## Evidence

- Missing payloads defaulted to `{}`: `src/elspeth/plugins/clients/verifier.py:182-183`

## Impact

- User-facing impact: verification results are misleading; missing baselines can look like mismatches or false matches.
- Data integrity / security impact: audit grade should degrade to ATTRIBUTABLE_ONLY, but verifier does not surface this.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `CallVerifier` does not distinguish between "payload missing" and "payload is an empty dict".

## Proposed Fix

- Code changes (modules/files):
  - If `get_call_response_data` returns None for a SUCCESS call, mark a new flag (e.g., `payload_missing=True`) or surface a dedicated error.
  - Avoid counting these as matches/mismatches.
- Config or schema changes: none.
- Tests to add/update:
  - Add a verification test where payload store is disabled/purged and assert a missing-payload outcome.
- Risks or migration steps:
  - Update report metrics to include missing-payload counts.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability standard: payload availability distinguishes replayable vs attributable-only)
- Observed divergence: verifier does not surface missing payloads.
- Reason (if known): convenience default.
- Alignment plan or decision needed: define how verify should behave with purged payloads.

## Acceptance Criteria

- Verification explicitly reports missing payloads and excludes them from normal diff comparisons.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k verifier_payload`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Current State Analysis

The bug is confirmed to still exist in the current codebase:

**Code Location:** `src/elspeth/plugins/clients/verifier.py:183`
```python
recorded_response = self._recorder.get_call_response_data(call.call_id) or {}
```

This line explicitly converts `None` to `{}` using the `or {}` pattern, which hides the distinction between:
1. A legitimately empty response payload (`{}`)
2. A missing/purged payload (`None`)

**Test Coverage:** The existing test at `tests/plugins/clients/test_verifier.py:447-467` (`test_verify_with_none_recorded_response`) explicitly tests this behavior and confirms the bug:

```python
def test_verify_with_none_recorded_response(self) -> None:
    """Handles calls where recorded response couldn't be retrieved."""
    # ...
    recorder.get_call_response_data.return_value = None  # Payload purged
    # ...
    # Empty dict vs live response will mismatch
    assert result.is_match is False
    assert result.recorded_response == {}  # ← Bug confirmed in test
```

The test acknowledges that `None` gets coerced to `{}`, which then creates a misleading mismatch when compared against the live response.

**API Contract:** The `LandscapeRecorder.get_call_response_data()` method explicitly documents that it returns `None` in multiple scenarios (see `src/elspeth/core/landscape/recorder.py:2689-2708`):
- Call not found
- No response_ref set on the call
- Payload store not configured
- **Response data has been purged from payload store**

### Impact Validation

The bug has the exact impacts described in the original report:

1. **Misleading verification results:** When a payload is purged, verification compares `{}` against the live response, which will always mismatch (unless the live response is also `{}`)
2. **Missing payload tracking:** The verifier has separate tracking for `recorded_call_missing` but no tracking for `recorded_payload_missing`
3. **Audit grade implications:** As noted in the original report, missing payloads should signal degraded audit capability (ATTRIBUTABLE_ONLY vs full REPLAYABLE), but this is not surfaced

### Git History

No changes have been made to the verifier code since RC1 (commit `c786410`). The file has not been touched during any of the bug fix sessions since 2026-01-21.

### Fix Recommendations

The proposed fix in the original report remains valid:

1. **Add payload missing flag:** The `VerificationResult` should track `payload_missing` separately from `recorded_call_missing`
2. **Update reporting:** `VerificationReport` should track missing payloads as a distinct category (not counted as matches or mismatches)
3. **Remove coercion:** Change line 183 from `or {}` to explicit `None` handling
4. **Update test:** Modify `test_verify_with_none_recorded_response` to assert `payload_missing=True` instead of expecting `recorded_response == {}`

This aligns with the CLAUDE.md principle that "The audit trail is the source of truth" - hiding missing data destroys audit integrity by making it impossible to distinguish between "we recorded an empty response" and "we can't verify because the payload is gone".

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Commit `c5fb53e` - "fix(core): address three audit-related bugs"

**Resolution:**

The fix removed the `or {}` pattern and added explicit handling for missing payloads in `src/elspeth/plugins/clients/verifier.py:206-217`:

```python
# Get recorded response
recorded_response = self._recorder.get_call_response_data(call.call_id)

# Handle missing/purged payload explicitly
if recorded_response is None:
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
```

**Changes Made:**

1. Added `payload_missing: bool = False` field to `VerificationResult`
2. Added `missing_payloads: int = 0` counter to `VerificationReport`
3. Explicit `None` check before comparison instead of silent coercion
4. Missing payloads are now counted separately from matches/mismatches

**Test Coverage Updated:**

The test `test_verify_with_none_recorded_response` now asserts `payload_missing=True` instead of checking for empty dict coercion.

**Verified By:** Claude Opus 4.5 (2026-01-29)
