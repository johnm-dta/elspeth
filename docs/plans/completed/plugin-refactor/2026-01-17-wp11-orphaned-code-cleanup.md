# WP-11: Orphaned Code Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code that was never integrated, while KEEPING audit-critical infrastructure for future phases.

**Architecture:** During development, some code was written but never called:
- `on_register()` hooks - defined in base classes but never invoked by engine
- Some infrastructure was designed for future phases and should be preserved

**Key Decisions (made 2026-01-17):**
- **RetryManager:** KEEP & INTEGRATE - Retries must be auditable with `(run_id, row_id, transform_seq, attempt)`
- **Call infrastructure:** KEEP for Phase 6 - External calls (LLMs, APIs) are a major audit surface

**Tech Stack:** Python 3.12

---

## Task 1: Remove on_register() from base classes

**Files:**
- Modify: `src/elspeth/plugins/base.py`
- Test: Verify no callers exist

**Background:** `on_register()` is defined in 4 base classes but is never called by the engine. It's dead code that creates false expectations.

**Locations:**
- Line 73: `BaseSource.on_register()`
- Line 144: `BaseTransform.on_register()`
- Line 227: `BaseGate.on_register()`
- Line 306: `BaseAggregation.on_register()`

**Step 1: Verify no callers**

```bash
# Search for any code that calls on_register
grep -r "\.on_register\(" src/elspeth/ --include="*.py" | grep -v "def on_register"
```

Expected: No results (method is never called)

**Step 2: Remove from BaseSource**

In `src/elspeth/plugins/base.py`, delete lines 73-77:

```python
# DELETE THIS:
    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered with the engine.

        Override for one-time setup.
        """
```

**Step 3: Remove from BaseTransform**

Delete lines 144-145:

```python
# DELETE THIS:
    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""
```

**Step 4: Remove from BaseGate**

Delete lines 227-228:

```python
# DELETE THIS:
    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""
```

**Step 5: Remove from BaseAggregation**

Delete lines 306-307:

```python
# DELETE THIS:
    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""
```

**Step 6: Run tests**

```bash
pytest tests/plugins/test_base.py -v
```

**Step 7: Commit**

```
git add -A && git commit -m "refactor(plugins): remove unused on_register() hooks

These methods were defined but never called by the engine.
Removing them eliminates false expectations for plugin authors.

Removed from: BaseSource, BaseTransform, BaseGate, BaseAggregation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Verify RetryManager is ready for integration

**Files:**
- Review: `src/elspeth/engine/retry.py`
- Test: Verify existing tests pass

**Background:** RetryManager was built but not integrated into the engine. Rather than delete it, we keep it for Phase 5 integration. This task verifies it's functional.

**Step 1: Review RetryManager**

Read `src/elspeth/engine/retry.py` lines 77-156 to understand the interface:

```python
class RetryManager:
    def execute_with_retry(
        self,
        operation: Callable[[], T],
        *,
        is_retryable: Callable[[BaseException], bool],
        on_retry: Callable[[int, BaseException], None] | None = None,
    ) -> T:
```

**Step 2: Run existing tests**

```bash
pytest tests/engine/test_retry.py -v
```

**Step 3: Document integration point**

Create a note in the code or tracker about where RetryManager should be integrated:

```python
# In retry.py, add or update docstring:
"""
Integration Point (Phase 5):
    The RowProcessor should use RetryManager.execute_with_retry() around
    transform execution. The on_retry callback should call
    recorder.record_retry_attempt() to audit each attempt.

    Example integration in processor.py:
        result = self._retry_manager.execute_with_retry(
            operation=lambda: executor.transform(token, ctx),
            is_retryable=lambda e: getattr(e, 'retryable', False),
            on_retry=lambda attempt, e: self._recorder.record_attempt(
                run_id, token_id, step, attempt, error=str(e)
            ),
        )
"""
```

**Step 4: Commit (if changes made)**

```
git add -A && git commit -m "docs(retry): document integration point for Phase 5

RetryManager is functional but not yet integrated into the engine.
Added documentation for future integration into RowProcessor.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Verify Call infrastructure is intact

**Files:**
- Review: `src/elspeth/contracts/enums.py` (CallType, CallStatus)
- Review: `src/elspeth/contracts/audit.py` (Call dataclass)
- Review: `src/elspeth/core/landscape/recorder.py` (get_calls)

**Background:** The Call infrastructure is for Phase 6 (External Calls - LLMs, APIs). We verify it exists and is not broken by other changes.

**Step 1: Verify enums exist**

```bash
grep -n "class CallType\|class CallStatus" src/elspeth/contracts/enums.py
```

Expected: Both enums exist

**Step 2: Verify Call dataclass exists**

```bash
grep -n "class Call" src/elspeth/contracts/audit.py
```

Expected: Call dataclass exists

**Step 3: Verify get_calls exists**

```bash
grep -n "def get_calls" src/elspeth/core/landscape/recorder.py
```

Expected: Method exists

**Step 4: Run mypy on contracts**

```bash
mypy src/elspeth/contracts/enums.py src/elspeth/contracts/audit.py --strict
```

Expected: No errors related to Call infrastructure

**Step 5: Document Phase 6 dependency**

Add comment in audit.py if not present:

```python
@dataclass
class Call:
    """External call record for Phase 6 (LLM/API audit).

    NOT YET INTEGRATED - infrastructure prepared for Phase 6.
    See: docs/plans/2026-01-12-phase6-external-calls.md
    """
```

**Step 6: Commit (if changes made)**

```
git add -A && git commit -m "docs(audit): clarify Call infrastructure is for Phase 6

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Run full verification

**Step 1: Run mypy on base.py**

```bash
mypy src/elspeth/plugins/base.py --strict
```

Expected: No errors from removed methods

**Step 2: Run all plugin tests**

```bash
pytest tests/plugins/ -v
```

Expected: All tests pass (on_register was never tested since never called)

**Step 3: Run engine tests**

```bash
pytest tests/engine/ -v
```

Expected: All tests pass

**Step 4: Verify no broken imports**

```bash
python -c "from elspeth.plugins.base import BaseSource, BaseTransform, BaseGate, BaseAggregation; print('OK')"
```

**Step 5: Final commit**

```
git add -A && git commit -m "chore: verify WP-11 orphaned code cleanup complete

- Removed unused on_register() from 4 base classes
- Verified RetryManager is functional (Phase 5 integration pending)
- Verified Call infrastructure is intact (Phase 6)
- All tests pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

- [x] `on_register()` removed from `BaseSource`
- [x] `on_register()` removed from `BaseTransform`
- [x] `on_register()` removed from `BaseGate`
- [x] `on_register()` removed from `BaseAggregation`
- [x] No code calls `on_register()` anywhere
- [x] `RetryManager` tests pass (kept for Phase 5)
- [x] `CallType`, `CallStatus` enums exist (kept for Phase 6)
- [x] `Call` dataclass exists (kept for Phase 6)
- [x] `get_calls()` method exists (kept for Phase 6)
- [x] `mypy --strict` passes on base.py
- [x] All plugin tests pass
- [x] All engine tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/plugins/base.py` | MODIFY | Remove on_register() from 4 classes |
| `src/elspeth/engine/retry.py` | MODIFY (optional) | Add integration docs |
| `src/elspeth/contracts/audit.py` | MODIFY (optional) | Clarify Phase 6 comment |

---

## Dependency Notes

- **Depends on:** Nothing
- **Unlocks:** Nothing (pure cleanup)
- **Risk:** Very Low - removing genuinely dead code

---

## Items NOT in this WP (moved elsewhere)

The following were originally in WP-11 but moved to WP-06:
- `AcceptResult.trigger` field cleanup
- `BaseAggregation.should_trigger()` cleanup
- `BaseAggregation.reset()` cleanup

These are cleaned up when WP-06 makes them obsolete, not before.
