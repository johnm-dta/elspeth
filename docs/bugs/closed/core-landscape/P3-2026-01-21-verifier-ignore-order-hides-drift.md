# Bug Report: CallVerifier ignores list ordering by default, masking drift

## Summary

- `CallVerifier` hard-codes `ignore_order=True` in DeepDiff comparisons. For responses where list order matters (ranked results, tool calls, top-k outputs), order changes are ignored and drift can be missed.

## Severity

- Severity: minor
- Priority: P3

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
- Notable tool calls or steps: code inspection of verifier comparison defaults

## Steps To Reproduce

1. Record a response containing an ordered list (e.g., ranked results `["a", "b", "c"]`).
2. Verify against a live response with the list reordered (e.g., `["c", "b", "a"]`).
3. Observe verification reports a match because order is ignored.

## Expected Behavior

- Order-sensitive responses should detect reordering as drift by default or allow configuration to enforce ordering.

## Actual Behavior

- Ordering differences are ignored globally for all responses.

## Evidence

- Hard-coded `ignore_order=True`: `src/elspeth/plugins/clients/verifier.py:186-190`

## Impact

- User-facing impact: verification can miss real drift in ordered responses.
- Data integrity / security impact: baseline comparisons are weaker than intended.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- DeepDiff is configured with a global ignore-order setting rather than a per-response or configurable policy.

## Proposed Fix

- Code changes (modules/files):
  - Make `ignore_order` configurable and default to `False`, or allow per-call overrides.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test case for ordered list drift detection.
- Risks or migration steps:
  - Some existing comparisons may become more sensitive; document the change.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: verification ignores order changes for all responses.
- Reason (if known): simplified comparison.
- Alignment plan or decision needed: define verification semantics for ordered data.

## Acceptance Criteria

- Verification reports drift when ordered lists change unless explicitly configured to ignore ordering.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k verifier_order`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Code Analysis

The bug is confirmed present in the current codebase:

**Location:** `/home/john/elspeth-rapid/src/elspeth/plugins/clients/verifier.py:186-191`

```python
# Compare using DeepDiff
diff = DeepDiff(
    recorded_response,
    live_response,
    ignore_order=True,  # <-- Hard-coded to True
    exclude_paths=self._ignore_paths,
)
```

The `ignore_order=True` parameter is hard-coded with no configuration option. This means all list comparisons are order-independent by default.

### Test Coverage

There is explicit test coverage documenting this behavior:

**Test:** `tests/plugins/clients/test_verifier.py::TestCallVerifier::test_verify_order_independent_comparison` (lines 468-489)

```python
def test_verify_order_independent_comparison(self) -> None:
    """Verifier ignores order in list comparisons."""
    recorder = self._create_mock_recorder()
    request_data = {"id": 1}
    request_hash = stable_hash(request_data)

    recorded_response = {"items": ["a", "b", "c"]}
    live_response = {"items": ["c", "a", "b"]}  # Same items, different order

    mock_call = self._create_mock_call(request_hash=request_hash)
    recorder.find_call_by_request_hash.return_value = mock_call
    recorder.get_call_response_data.return_value = recorded_response

    verifier = CallVerifier(recorder, source_run_id="run_abc123")
    result = verifier.verify(
        call_type="llm",
        request_data=request_data,
        live_response=live_response,
    )

    # Should match because ignore_order=True
    assert result.is_match is True
```

This test validates the current behavior but also demonstrates the problem: `["a", "b", "c"]` matches `["c", "a", "b"]`, which would hide drift in order-sensitive responses like:
- Top-k ranked search results
- LLM tool call sequences
- Priority-ordered recommendations

### Git History

No changes to this behavior since initial implementation:
- `9b2aa8e feat(plugins): add CallVerifier for verify mode` - Initial implementation with `ignore_order=True`
- `c786410 ELSPETH - Release Candidate 1` - No changes to order handling

The design document (`docs/plans/completed/2026-01-12-phase6-external-calls.md:1200-1206`) also shows `ignore_order=True` was part of the original specification.

### Impact Assessment

The bug remains as described:

1. **Actual Drift Hidden**: For LLM responses with ranked lists (e.g., top-3 recommendations), if the order changes from `["best", "good", "okay"]` to `["okay", "good", "best"]`, verification will report a match even though semantic meaning changed completely.

2. **No Configuration Path**: The `CallVerifier.__init__()` accepts `ignore_paths` for excluding specific fields but has no parameter for controlling order sensitivity.

3. **Global Application**: The setting applies to all list comparisons in all response types, with no per-call or per-field override mechanism.

### Recommendation

This is a valid P3 bug. The behavior is intentional (as evidenced by tests and design docs) but limits verification effectiveness for order-sensitive responses. The proposed fix (make `ignore_order` configurable with sensible defaults) is appropriate and low-risk.

## Phase 1 Implementation (2026-01-30)

**Status: IMPLEMENTED**

Phase 1 adds the `ignore_order` parameter to `CallVerifier` with default `True` (non-breaking).

### Changes Made

1. Added `ignore_order: bool = True` parameter to `CallVerifier.__init__()`
2. Parameter passes through to DeepDiff comparison
3. Default preserves existing behavior (order-independent)
4. Users can now set `ignore_order=False` for order-sensitive verification

### Test Coverage Added

- `test_verify_order_sensitive_when_configured` - explicit False behavior
- `test_ignore_order_handles_duplicate_elements` - multiset semantics
- `test_ignore_order_applies_recursively_to_nested_lists` - recursive behavior
- `test_ignore_order_does_not_affect_dict_keys` - dict key ordering unaffected
- `test_empty_lists_always_match` - edge case
- `test_order_sensitivity_with_realistic_llm_response` - practical LLM example
- `test_verify_order_independent_with_default_config` - renamed from original

### Next Steps

Phase 2 (P3-2026-01-29-verifier-field-level-order-config.md) will add field-level configuration to allow mixed semantics within a single verification session.

## Closure Report (2026-02-01)

**Status:** CLOSED (IMPLEMENTED)

### Closure Notes

- Moved from `docs/bugs/open/` to `docs/bugs/closed/` because Phase 1 is implemented and test coverage is in place (see "Phase 1 Implementation" above).
- **Priority validation:** P3 is appropriate. The fix improves verification fidelity for order-sensitive data but does not block core auditability when defaults remain unchanged.
- Follow-up remains tracked in `docs/bugs/open/core-landscape/P3-2026-01-29-verifier-field-level-order-config.md`.
