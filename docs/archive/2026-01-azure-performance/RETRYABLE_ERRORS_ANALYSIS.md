# Retryable Errors Analysis - LLM API Calls

**Date:** 2026-01-26
**Context:** Determining which errors should trigger automatic retry vs immediate failure

---

## Executive Summary

Currently, only **rate limit errors (429)** trigger automatic retry. This is **too restrictive** for production resilience.

**Recommendation:** Expand retryable errors to include:
- ✅ Network/connection errors (timeouts, connection refused)
- ✅ Server errors (500, 502, 503, 504)
- ✅ Rate limits (429) - already implemented
- ❌ Client errors (400, 401, 403, 404, 422) - permanent failures
- ❌ Content policy violations - permanent failures
- ❌ Context length exceeded - permanent failures

---

## Error Categories

### Category 1: Transient Infrastructure Errors ✅ SHOULD RETRY

These errors are caused by temporary infrastructure issues and **will likely succeed on retry**:

| Error Type | HTTP Code | Cause | Retry? | Rationale |
|------------|-----------|-------|--------|-----------|
| **Connection Timeout** | N/A | Network slow/congested | ✅ Yes | Transient network issue |
| **Read Timeout** | N/A | Server slow to respond | ✅ Yes | Server may be overloaded temporarily |
| **Connection Refused** | N/A | Server restarting | ✅ Yes | Server coming back online |
| **Connection Reset** | N/A | Network interruption | ✅ Yes | Transient network issue |
| **DNS Resolution Failure** | N/A | DNS server issue | ✅ Yes | Usually resolves within seconds |
| **Service Unavailable** | 503 | Server overload | ✅ Yes | Server recovering capacity |
| **Bad Gateway** | 502 | Proxy/gateway issue | ✅ Yes | Infrastructure routing problem |
| **Gateway Timeout** | 504 | Upstream timeout | ✅ Yes | Downstream service slow |
| **Internal Server Error** | 500 | Server bug/overload | ✅ Yes | May be transient (race condition, OOM) |

**Example scenarios:**
- Azure datacenter experiencing temporary network issues
- OpenAI API server restarting
- Load balancer routing to unhealthy instance
- Temporary DNS propagation issue

**Retry behavior:**
- AIMD backoff: 100ms → 200ms → 400ms → 800ms → ...
- Max retry time: 300 seconds (configurable)
- On timeout: return error with `reason: capacity_retry_timeout`

---

### Category 2: Rate Limiting ✅ ALREADY IMPLEMENTED

| Error Type | HTTP Code | Cause | Retry? | Rationale |
|------------|-----------|-------|--------|-----------|
| **Rate Limit Exceeded** | 429 | Too many requests | ✅ Yes | Rate limit will reset |

**Current implementation:** ✅ Already retries with AIMD backoff

**Retry behavior:**
- Start with short delay (~100ms)
- Double on each subsequent rate limit
- Max retry time: 300 seconds
- Works well for transient bursts

---

### Category 3: Permanent Client Errors ❌ DO NOT RETRY

These errors indicate **bugs in our code or configuration** - retrying will NOT help:

| Error Type | HTTP Code | Cause | Retry? | Rationale |
|------------|-----------|-------|--------|-----------|
| **Bad Request** | 400 | Malformed request | ❌ No | Request is invalid, will always fail |
| **Unauthorized** | 401 | Invalid API key | ❌ No | Credentials wrong, won't change |
| **Forbidden** | 403 | Insufficient permissions | ❌ No | Account lacks access |
| **Not Found** | 404 | Wrong endpoint/model | ❌ No | URL or model name wrong |
| **Unprocessable Entity** | 422 | Invalid parameters | ❌ No | Request semantically invalid |

**Example scenarios:**
- API key typo in config → 401 (fix config, not retry)
- Model name typo ("gpt-4o" → "gpt-40") → 404 (fix config)
- Temperature = 5.0 (valid range 0-2) → 400 (fix code)
- Content policy violation → 400 (prompt violates policy)

**Failure behavior:**
- Return `TransformResult.error()` immediately
- Include full error details in `reason`
- Mark as `retryable=False` in audit trail
- Row fails with `_error` marker

---

### Category 4: Permanent LLM-Specific Errors ❌ DO NOT RETRY

These errors indicate **prompt issues** that won't be fixed by retrying:

| Error Type | HTTP Code | Cause | Retry? | Rationale |
|------------|-----------|-------|--------|-----------|
| **Content Policy Violation** | 400 | Prompt violates policy | ❌ No | Prompt content inappropriate |
| **Context Length Exceeded** | 400 | Prompt too long | ❌ No | Need to reduce prompt size |
| **Model Overloaded** | 529 | No capacity | ✅ Maybe | Azure-specific, like 503 |

**Content Policy Example:**
```json
{
  "error": {
    "message": "Your request was rejected as a result of our safety system.",
    "type": "invalid_request_error",
    "code": "content_policy_violation"
  }
}
```

**Context Length Example:**
```json
{
  "error": {
    "message": "This model's maximum context length is 8192 tokens...",
    "type": "invalid_request_error",
    "code": "context_length_exceeded"
  }
}
```

**Failure behavior:**
- Do NOT retry (will always fail with same prompt)
- Return error immediately with diagnostic info
- Include error code and message for debugging

---

## Current Implementation Issues

### Issue 1: Only Rate Limits Are Retried

**Current code (llm.py:207):**
```python
is_rate_limit = "rate" in error_str or "429" in error_str
if is_rate_limit:
    raise RateLimitError(str(e)) from e
raise LLMClientError(str(e), retryable=False) from e  # Everything else fails!
```

**Problem:**
- Network timeout → `LLMClientError(retryable=False)` → immediate failure ❌
- 503 Service Unavailable → `LLMClientError(retryable=False)` → immediate failure ❌
- 502 Bad Gateway → `LLMClientError(retryable=False)` → immediate failure ❌

**Impact:** Production system is **fragile** - any transient issue causes permanent row failures.

### Issue 2: No Error Type Classification

**Current:** Simple string matching on error message
```python
is_rate_limit = "rate" in error_str or "429" in error_str
```

**Problems:**
- Doesn't distinguish 400 (permanent) from 500 (transient)
- Doesn't handle network errors (no HTTP code)
- String matching is fragile (what if message changes?)

---

## Recommended Implementation

### Step 1: Define Retryable Error Categories

```python
# src/elspeth/plugins/clients/llm.py

def _is_retryable_error(exception: Exception) -> bool:
    """Determine if an LLM error is retryable.

    Retryable errors (transient):
    - Rate limits (429)
    - Server errors (500, 502, 503, 504, 529)
    - Network/connection errors (timeout, connection refused, etc.)

    Non-retryable errors (permanent):
    - Client errors (400, 401, 403, 404, 422)
    - Content policy violations
    - Context length exceeded

    Returns:
        True if error is likely transient and should be retried
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__

    # Rate limits - always retryable
    if "rate" in error_str or "429" in error_str:
        return True

    # Server errors (5xx) - usually transient
    server_error_codes = ["500", "502", "503", "504", "529"]
    if any(code in error_str for code in server_error_codes):
        return True

    # Network/connection errors - transient
    network_error_patterns = [
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection aborted",
        "network unreachable",
        "host unreachable",
        "dns",
        "getaddrinfo failed",
    ]
    if any(pattern in error_str for pattern in network_error_patterns):
        return True

    # Client errors (4xx except 429) - permanent
    client_error_codes = ["400", "401", "403", "404", "422"]
    if any(code in error_str for code in client_error_codes):
        return False

    # LLM-specific permanent errors
    permanent_error_patterns = [
        "content_policy_violation",
        "content policy",
        "safety system",
        "context_length_exceeded",
        "context length",
        "maximum context",
    ]
    if any(pattern in error_str for pattern in permanent_error_patterns):
        return False

    # Unknown error - be conservative, do NOT retry
    # This prevents infinite retries on unexpected errors
    return False
```

### Step 2: Create Specific Exception Classes

```python
class NetworkError(LLMClientError):
    """Network/connection error - retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ServerError(LLMClientError):
    """Server error (5xx) - retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ContentPolicyError(LLMClientError):
    """Content policy violation - not retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class ContextLengthError(LLMClientError):
    """Context length exceeded - not retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
```

### Step 3: Update Exception Handling

```python
except Exception as e:
    latency_ms = (time.perf_counter() - start) * 1000
    error_type = type(e).__name__
    error_str = str(e).lower()

    # Classify error
    is_retryable = _is_retryable_error(e)

    self._recorder.record_call(
        state_id=self._state_id,
        call_index=call_index,
        call_type=CallType.LLM,
        status=CallStatus.ERROR,
        request_data=request_data,
        error={
            "type": error_type,
            "message": str(e),
            "retryable": is_retryable,
        },
        latency_ms=latency_ms,
    )

    # Raise appropriate exception
    if "rate" in error_str or "429" in error_str:
        raise RateLimitError(str(e)) from e
    elif "content_policy" in error_str or "safety system" in error_str:
        raise ContentPolicyError(str(e)) from e
    elif "context_length" in error_str or "maximum context" in error_str:
        raise ContextLengthError(str(e)) from e
    elif is_retryable:
        # Server error or network error - retryable
        raise LLMClientError(str(e), retryable=True) from e
    else:
        # Client error or unknown - not retryable
        raise LLMClientError(str(e), retryable=False) from e
```

---

## PooledExecutor Retry Integration

### Current Behavior

**PooledExecutor only retries `CapacityError`** (line 271 in executor.py):

```python
except CapacityError as e:
    # Retry with AIMD backoff
    self._throttle.on_capacity_error()
    time.sleep(retry_delay_ms / 1000)
    # Re-acquire semaphore and retry
```

### Required Change

**Should retry ANY retryable error:**

```python
except Exception as e:
    # Check if error is retryable
    is_retryable = getattr(e, "retryable", False)

    if not is_retryable:
        # Permanent error - fail immediately
        return (buffer_idx, TransformResult.error({
            "reason": "permanent_error",
            "error": str(e),
        }))

    # Check timeout
    if time.monotonic() >= max_time:
        return (buffer_idx, TransformResult.error({
            "reason": "retry_timeout",
            "error": str(e),
            "elapsed_seconds": time.monotonic() - start_time,
        }))

    # Retry with AIMD backoff
    self._throttle.on_capacity_error()
    self._semaphore.release()
    time.sleep(retry_delay_ms / 1000)
    self._semaphore.acquire()
    # Continue loop - retry
```

**This enables retry for:**
- Rate limits (429)
- Server errors (500, 502, 503, 504)
- Network errors (timeout, connection refused)

**And fails immediately for:**
- Client errors (400, 401, 403, 404)
- Content policy violations
- Context length exceeded

---

## Testing Recommendations

### Test 1: Network Timeout Should Retry

```python
def test_network_timeout_triggers_retry():
    """Network timeout should retry until success."""
    call_count = [0]

    def mock_chat_completion(**kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise TimeoutError("Read timed out")
        return success_response

    result = transform.process(row, ctx)

    assert result.status == "success"  # Succeeded after retries
    assert call_count[0] == 3  # Failed twice, succeeded third time
```

### Test 2: Server Error Should Retry

```python
def test_server_error_503_triggers_retry():
    """503 Service Unavailable should retry."""
    call_count = [0]

    def mock_chat_completion(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("503 Service Unavailable")
        return success_response

    result = transform.process(row, ctx)

    assert result.status == "success"
    assert call_count[0] == 2  # Failed once, succeeded second time
```

### Test 3: Content Policy Should NOT Retry

```python
def test_content_policy_violation_fails_immediately():
    """Content policy violation should fail without retry."""
    call_count = [0]

    def mock_chat_completion(**kwargs):
        call_count[0] += 1
        raise Exception("Your request was rejected by our safety system")

    result = transform.process(row, ctx)

    assert result.status == "error"
    assert call_count[0] == 1  # Only called once, no retry
```

### Test 4: Invalid API Key Should NOT Retry

```python
def test_invalid_api_key_401_fails_immediately():
    """401 Unauthorized should fail without retry."""
    call_count = [0]

    def mock_chat_completion(**kwargs):
        call_count[0] += 1
        raise Exception("401 Unauthorized: Invalid API key")

    result = transform.process(row, ctx)

    assert result.status == "error"
    assert call_count[0] == 1  # No retry
    assert "401" in result.reason["error"]
```

---

## Production Impact

### Before (Current)

**Transient errors cause permanent failures:**
- Network timeout → row fails permanently ❌
- 503 Service Unavailable → row fails permanently ❌
- Azure datacenter issue → all rows fail permanently ❌

**Result:** Poor production resilience, high false-failure rate

### After (Recommended)

**Transient errors retry automatically:**
- Network timeout → retry with backoff → eventually succeeds ✅
- 503 Service Unavailable → retry → succeeds when service recovers ✅
- Azure datacenter issue → retries during ~5min window → succeeds when resolved ✅

**Permanent errors fail fast:**
- Invalid API key → fail immediately (don't waste time retrying) ✅
- Content policy violation → fail immediately (prompt won't change) ✅

**Result:** Better production resilience, lower false-failure rate

---

## Recommended Actions

### Immediate (High Priority)

1. ✅ **Expand retryable error detection** - Implement `_is_retryable_error()`
2. ✅ **Update PooledExecutor** - Retry any `retryable=True` exception
3. ✅ **Add exception classes** - NetworkError, ServerError, ContentPolicyError
4. ✅ **Add tests** - Verify retry behavior for each error category

### Short-term

1. Monitor retry metrics in production (capacity_retries count)
2. Track permanent vs transient error ratios
3. Tune `max_capacity_retry_seconds` based on observed recovery times

### Long-term

1. Add structured error codes (not just string matching)
2. Implement error-specific retry strategies (e.g., shorter timeout for 503)
3. Add circuit breaker pattern for persistent failures

---

## Summary

**Current state:**
- Only rate limits (429) trigger retry ❌
- All other errors fail immediately ❌
- Poor resilience to transient infrastructure issues ❌

**Recommended state:**
- Rate limits + server errors + network errors trigger retry ✅
- Client errors + content policy + context length fail immediately ✅
- Better resilience to transient issues ✅
- Faster failure on permanent errors ✅

**Next step:** Implement retryable error classification and update PooledExecutor retry logic.

---

**Generated by:** Claude Code (Sonnet 4.5)
**Date:** 2026-01-26
