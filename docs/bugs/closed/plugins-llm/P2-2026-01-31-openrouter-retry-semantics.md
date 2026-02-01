# Bug Report: OpenRouter transient HTTP failures never trigger RetryManager retries

## Summary

- OpenRouter returns `TransformResult.error()` for retryable HTTP errors instead of re-raising, so RetryManager never sees the failure and doesn't retry.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/llm/openrouter.py:294-311` - HTTPStatusError and RequestError return `TransformResult.error()`
- Compare to Azure at lines 319-327 which re-raises retryable `LLMClientError`
- Engine only retries on exceptions, not error results

## Impact

- User-facing impact: Transient failures (429, 503) fail immediately instead of retrying
- Data integrity: None

## Proposed Fix

- Re-raise retryable errors as LLMClientError, matching Azure pattern

## Acceptance Criteria

- Transient HTTP errors trigger RetryManager retries

## Verification (2026-02-01)

**Status: STILL VALID**

- OpenRouter still converts HTTP/transient errors into `TransformResult.error()` instead of raising, so retry manager never sees exceptions. (`src/elspeth/plugins/llm/openrouter.py:306-315`, `src/elspeth/engine/processor.py:1061-1063`)
