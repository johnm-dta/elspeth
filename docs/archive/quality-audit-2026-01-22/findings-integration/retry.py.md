## Summary

RetrySettings exposes `exponential_base`, but RetryConfig/RetryManager drop it, so configured exponential backoff bases are ignored across the core config ↔ engine retry seam.

## Severity

- Severity: minor
- Priority: P2

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [x] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [ ] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** core/config ↔ engine/retry

**Integration Point:** RetrySettings → RetryConfig mapping and tenacity wait parameterization

## Evidence

### Side A: core/config (`src/elspeth/core/config.py:557-565`)

```python
class RetrySettings(BaseModel):
    """Retry behavior configuration."""
    model_config = {"frozen": True}

    max_attempts: int = Field(default=3, gt=0, description="Maximum retry attempts")
    initial_delay_seconds: float = Field(default=1.0, gt=0, description="Initial backoff delay")
    max_delay_seconds: float = Field(default=60.0, gt=0, description="Maximum backoff delay")
    exponential_base: float = Field(default=2.0, gt=1.0, description="Exponential backoff base")
```

### Side B: engine/retry (`src/elspeth/engine/retry.py:86-101`)

```python
@classmethod
def from_settings(cls, settings: "RetrySettings") -> "RetryConfig":
    return cls(
        max_attempts=settings.max_attempts,
        base_delay=settings.initial_delay_seconds,
        max_delay=settings.max_delay_seconds,
        jitter=1.0,  # Fixed jitter, not exposed in settings
    )
```

### Coupling Evidence: wait configuration ignores exponential_base (`src/elspeth/engine/retry.py:152-159`)

```python
for attempt_state in Retrying(
    stop=stop_after_attempt(self._config.max_attempts),
    wait=wait_exponential_jitter(
        initial=self._config.base_delay,
        max=self._config.max_delay,
        jitter=self._config.jitter,
    ),
    retry=retry_if_exception(is_retryable),
):
```

## Root Cause Hypothesis

RetrySettings gained an exponential_base field but RetryConfig/RetryManager were not updated, so the configuration contract drifted from engine behavior.

## Recommended Fix

1. Add `exponential_base` to `RetryConfig`.
2. Map `RetrySettings.exponential_base` in `RetryConfig.from_settings`.
3. Pass `exp_base=self._config.exponential_base` into `wait_exponential_jitter`.
4. Add/adjust tests to assert exponential_base affects backoff timing.
5. Align any RetryPolicy mapping to include exponential_base if intended.

## Impact Assessment

- **Coupling Level:** Medium - config and engine must stay synchronized.
- **Maintainability:** Medium - missing fields silently change runtime behavior.
- **Type Safety:** Medium - typed field exists but is unused at runtime.
- **Breaking Change Risk:** Low - adding mapping should not break callers.

## Related Seams

`src/elspeth/engine/orchestrator.py:783`, `src/elspeth/contracts/engine.py:7`
---
Template Version: 1.0
---
## Summary

RetryManager only retries exceptions, while plugins return `TransformResult.error(retryable=True)` for transient LLM failures; the engine routes these error results without retry, violating the retryable contract across the engine ↔ plugins seam.

## Severity

- Severity: minor
- Priority: P2

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [x] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** engine ↔ plugins

**Integration Point:** retry classification (exception-based retries vs TransformResult.retryable)

## Evidence

### Side A: engine/retry (`src/elspeth/engine/retry.py:128-161`)

```python
def execute_with_retry(
    self,
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    ...
    for attempt_state in Retrying(
        stop=stop_after_attempt(self._config.max_attempts),
        wait=wait_exponential_jitter(...),
        retry=retry_if_exception(is_retryable),
        reraise=False,
    ):
```

### Side B: plugins/llm (`src/elspeth/plugins/llm/base.py:234-254`)

```python
except RateLimitError as e:
    return TransformResult.error(
        {"reason": "rate_limited", "error": str(e)},
        retryable=True,
    )
except LLMClientError as e:
    return TransformResult.error(
        {"reason": "llm_call_failed", "error": str(e)},
        retryable=e.retryable,
    )
```

### Coupling Evidence: retryable results are not retried (`src/elspeth/engine/processor.py:418-470`, `src/elspeth/engine/processor.py:769-789`)

```python
# TransformResult.error() is NOT retried - that's a processing error,
# not a transient failure. Only exceptions trigger retry.
def is_retryable(e: BaseException) -> bool:
    return isinstance(e, ConnectionError | TimeoutError | OSError)

...
if result.status == "error":
    if error_sink == "discard":
        ...
    else:
        # Routed to error sink
        ...
```

## Root Cause Hypothesis

Plugin protocols evolved to express retryability in result objects, but engine retry logic remained exception-only, leaving result-level retry signals unconsumed.

## Recommended Fix

1. Decide on the canonical retry signal (exceptions vs `TransformResult.retryable`).
2. If supporting result-level retries, extend RowProcessor/RetryManager to retry when `result.retryable=True` (e.g., loop at `_execute_transform_with_retry` or convert to a retryable exception).
3. Update plugin protocol docs/tests to match the chosen behavior.
4. Ensure audit records capture each retry attempt consistently for result-level retries.

## Impact Assessment

- **Coupling Level:** Medium - engine behavior depends on plugin error semantics.
- **Maintainability:** Medium - retry semantics are split between exceptions and results.
- **Type Safety:** Low - retryability is enforced only at runtime.
- **Breaking Change Risk:** Medium - altering retry behavior affects pipeline outcomes.

## Related Seams

`src/elspeth/plugins/llm/azure.py:259`, `src/elspeth/plugins/llm/openrouter.py:247`, `src/elspeth/plugins/transforms/azure/content_safety.py:258`, `src/elspeth/plugins/transforms/azure/prompt_shield.py:229`
---
Template Version: 1.0
