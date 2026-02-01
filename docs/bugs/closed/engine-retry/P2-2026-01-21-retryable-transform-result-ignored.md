# Bug Report: Retryable TransformResult errors are never retried

## Summary

- RowProcessor only retries exceptions, not `TransformResult.error(retryable=True)`. This ignores the retryable flag specified in the plugin protocol and prevents transient errors from being retried when they are reported as error results instead of exceptions.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure retry policy in settings.
2. Implement a transform that catches a transient external error and returns `TransformResult.error(..., retryable=True)`.
3. Run the pipeline on data that triggers the transient error.

## Expected Behavior

- RetryManager should retry the transform according to policy when `retryable=True`.

## Actual Behavior

- No retry occurs; the error is routed or discarded immediately.

## Evidence

- RowProcessor explicitly states TransformResult errors are not retried and only exceptions trigger retries: `src/elspeth/engine/processor.py:436-437`.
- Retryable check only inspects exception types: `src/elspeth/engine/processor.py:472-475`.
- Plugin protocol requires retryable errors to be considered by engine policy: `docs/contracts/plugin-protocol.md:1366-1372`.

## Impact

- User-facing impact: Reduced resilience to transient API errors reported as result errors.
- Data integrity / security impact: Error routing outcomes differ from intended retry semantics.
- Performance or cost impact: Increased manual retries or failed runs.

## Root Cause Hypothesis

- Retry logic only wraps exceptions; result-level retryable flags are ignored.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`, possibly `src/elspeth/engine/retry.py`
- Config or schema changes: None
- Tests to add/update: Add a retry test where TransformResult.error(retryable=True) is retried per policy.
- Risks or migration steps: Clarify whether retryable on results is supported or deprecated; update docs accordingly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md:1366-1372`.
- Observed divergence: Retryable result flag has no effect.
- Reason (if known): RetryManager wired only for exceptions.
- Alignment plan or decision needed: Decide if retryable results should be honored or remove the flag from protocol.

## Acceptance Criteria

- Retryable TransformResult errors are retried according to policy, or docs are updated to explicitly forbid this behavior.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k retryable_result`
- New tests required: Yes (retryable TransformResult).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

## Resolution (2026-01-27)

**Status: FIXED**

Fixed in branch fix/rc1-bug-burndown-session-6.

### Fix Applied: Option C - Re-raise retryable exceptions

**Changes made:**

1. **`src/elspeth/engine/processor.py`:**
   - Added `LLMClientError` import
   - Updated `is_retryable()` to recognize `LLMClientError` with `retryable=True`

2. **`src/elspeth/plugins/llm/azure.py`:**
   - Changed to re-raise retryable `LLMClientError` (including `RateLimitError`) instead of converting to `TransformResult.error()`
   - Non-retryable errors (e.g., `ContentPolicyError`) still return `TransformResult.error(retryable=False)`

3. **`src/elspeth/plugins/llm/base.py`:**
   - Same pattern: re-raise retryable exceptions, return error results for non-retryable

4. **Tests updated:**
   - `tests/plugins/llm/test_azure.py`: `test_rate_limit_error_propagates_for_engine_retry`
   - `tests/plugins/llm/test_base.py`: `test_rate_limit_error_propagates_for_engine_retry`, `test_retryable_llm_error_propagates_as_exception`

**Why Option C:**
- Consistent with existing `PooledExecutor` pattern (catches exceptions, applies AIMD retry)
- `TransformResult.error()` is for **semantic errors** (invalid data, business rule violations)
- **Transient failures** (rate limits, network errors) should be retried transparently via exceptions
- No changes needed to `RetryManager` architecture

---

## Verification (2026-01-25)

**Status: SUPERSEDED BY FIX ABOVE**

Verified against current codebase (commit 7540e57 on branch fix/rc1-bug-burndown-session-4).

### Current State Analysis

The bug remains valid with the following findings:

1. **Engine-level retry (RowProcessor + RetryManager):**
   - `_execute_transform_with_retry()` comment at line 426-427 explicitly states: "TransformResult.error() is NOT retried - that's a processing error, not a transient failure. Only exceptions trigger retry."
   - `is_retryable()` at line 462-465 only checks exception types (ConnectionError, TimeoutError, OSError)
   - When `result.status == "error"` at line 769, the processor immediately routes to quarantine or error sink WITHOUT checking `result.retryable`

2. **Pooled execution retry (PooledExecutor):**
   - Introduced on 2026-01-21 (same day as bug report but earlier: 07:14 vs 20:22)
   - Only retries `CapacityError` exceptions (src/elspeth/plugins/pooling/executor.py:271)
   - Does NOT retry `TransformResult.error(retryable=True)` results

3. **Real-world impact - Azure transforms:**
   - `AzureContentSafety` and `AzurePromptShield` return `TransformResult.error(retryable=True)` for network errors
   - Example: `src/elspeth/plugins/transforms/azure/content_safety.py:273-282` (httpx.RequestError)
   - **Sequential mode (pool_size=1):** Network errors with `retryable=True` are NOT retried
   - **Pooled mode (pool_size>1):** Capacity errors (429/503/529) are converted to CapacityError exceptions and ARE retried, but network errors (httpx.RequestError) with `retryable=True` are NOT retried

### Architecture Inconsistency

The plugin protocol (docs/contracts/plugin-protocol.md:1370) states: "Plugins indicate whether errors are `retryable` in their result objects. The engine decides whether and when to retry based on policy."

However, the engine ignores this flag completely. The `retryable` field exists in `TransformResult` (src/elspeth/contracts/results.py:80) but has no effect on engine behavior.

### Current Workaround Pattern

Azure transforms use different error handling strategies for different error types:
- **Capacity errors (429/503/529):** Raise `CapacityError` exception in pooled mode (works with PooledExecutor retry)
- **Network errors:** Return `TransformResult.error(retryable=True)` (NOT retried in either mode)
- **Non-retryable errors:** Return `TransformResult.error(retryable=False)`

This pattern shows that transforms are forced to use exceptions for retryable errors in pooled mode, while the `retryable` flag on results is effectively vestigial.

### Recommendation

This bug should be promoted to P1 because:
1. It affects real production transforms (Azure Content Safety, Prompt Shield)
2. Network errors (httpx.RequestError) are never retried despite being marked `retryable=True`
3. The architecture claims to support this pattern but doesn't implement it
4. The workaround (raising exceptions vs returning errors) is inconsistent across error types

The fix requires either:
- **Option A:** Implement retry support for `TransformResult.error(retryable=True)` in RowProcessor
- **Option B:** Remove the `retryable` field from `TransformResult` and update plugin protocol to require exceptions for retryable errors
- **Option C:** Update transforms to raise exceptions for all retryable errors (not just capacity errors)
