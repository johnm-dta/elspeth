# WP-04a: Delete Remaining *Like Protocol Duplications

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the last `*Like` protocol (`AggregationLike`) from executors.py, refactor batch state to executor-internal storage, and rename the `TransformLike` union alias to `RowPlugin`.

**Architecture:** The `AggregationLike` protocol exists solely because the executor monkey-patches `_batch_id` onto aggregation plugins. We fix this by moving batch state tracking into the executor itself, then delete the redundant protocol. The `TransformLike` union alias in orchestrator.py collides with the old protocol name and should be renamed to `RowPlugin` for clarity.

**Tech Stack:** Python 3.12, typing, Protocol

**Prior Art:** `TransformLike` and `GateLike` protocols were already deleted (commit f08c19a). This WP completes the cleanup.

---

## Pre-Conditions

- [x] WP-04 complete (SinkLike deleted)
- [x] TransformLike protocol deleted (verified)
- [x] GateLike protocol deleted (verified)

---

## Task 1: Refactor AggregationExecutor to Store Batch State Internally

**Goal:** Move `_batch_id` tracking from plugin objects to executor internal state.

**Files:**
- Modify: `src/elspeth/engine/executors.py`

**Step 1: Read current implementation**

Review lines 472-660 of executors.py to understand current `_batch_id` access patterns:
- Line 515: `if aggregation._batch_id is None:`
- Line 520: `aggregation._batch_id = batch.batch_id`
- Line 537: `self._member_counts[aggregation._batch_id]`
- Line 539: `batch_id=aggregation._batch_id`
- Line 543-544: Member count update and result assignment
- Line 550: Include in output dict
- Line 611: `batch_id = aggregation._batch_id`
- Line 637: `aggregation._batch_id = None`

**Step 2: Add internal batch_ids dict to __init__**

In `AggregationExecutor.__init__` (around line 480), add:

```python
def __init__(
    self,
    recorder: LandscapeRecorder,
    span_factory: SpanFactory,
    run_id: str,
) -> None:
    """Initialize executor.

    Args:
        recorder: Landscape recorder for audit trail
        span_factory: Span factory for tracing
        run_id: Run identifier for batch creation
    """
    self._recorder = recorder
    self._spans = span_factory
    self._run_id = run_id
    self._member_counts: dict[str, int] = {}  # batch_id -> count for ordinals
    self._batch_ids: dict[str, str | None] = {}  # node_id -> batch_id (NEW)
```

**Step 3: Update accept() method to use internal state**

Replace `aggregation._batch_id` references in the `accept` method:

```python
def accept(
    self,
    aggregation: AggregationProtocol,  # CHANGE: was AggregationLike
    token: TokenInfo,
    ctx: PluginContext,
    step_in_pipeline: int,
) -> AcceptResult:
    """Accept a row into an aggregation batch.

    Creates batch on first accept (if no batch exists for this aggregation).
    Records batch membership for accepted rows.
    ...
    """
    node_id = aggregation.node_id

    # Create batch on first accept
    if self._batch_ids.get(node_id) is None:  # CHANGE: was aggregation._batch_id
        batch = self._recorder.create_batch(
            run_id=self._run_id,
            aggregation_node_id=node_id,
        )
        self._batch_ids[node_id] = batch.batch_id  # CHANGE: was aggregation._batch_id
        self._member_counts[batch.batch_id] = 0

    batch_id = self._batch_ids[node_id]  # NEW: local var for clarity

    # Begin node state for this accept operation
    state_id = self._recorder.begin_node_state(...)

    start = time.perf_counter()
    try:
        result = aggregation.accept(token.row_data, ctx)
        duration_ms = (time.perf_counter() - start) * 1000

        if result.accepted:
            ordinal = self._member_counts[batch_id]  # CHANGE: was aggregation._batch_id
            self._recorder.add_batch_member(
                batch_id=batch_id,  # CHANGE: was aggregation._batch_id
                token_id=token.token_id,
                ordinal=ordinal,
            )
            self._member_counts[batch_id] = ordinal + 1  # CHANGE
            result.batch_id = batch_id  # CHANGE

            # Output for accepted rows
            accept_output = {
                "row": token.row_data,
                "batch_id": batch_id,  # CHANGE: was aggregation._batch_id
                "ordinal": ordinal,
            }
        # ... rest unchanged
```

**Step 4: Update flush() method to use internal state**

Replace `aggregation._batch_id` references in the `flush` method:

```python
def flush(
    self,
    aggregation: AggregationProtocol,  # CHANGE: was AggregationLike
    trigger_reason: str,
    state_id: str,
    ctx: PluginContext,
) -> list[dict[str, Any]]:
    """Flush an aggregation batch and return output rows.

    Transitions batch through: draft -> executing -> completed/failed.
    Resets batch tracking for this aggregation for next batch.
    ...
    """
    node_id = aggregation.node_id
    batch_id = self._batch_ids.get(node_id)  # CHANGE: was aggregation._batch_id

    if batch_id is None:
        raise ValueError(
            f"No batch to flush for aggregation {node_id}. "
            "Call accept() first to create a batch."
        )

    # ... middle of method unchanged (uses batch_id local var) ...

    try:
        # ... flush logic ...

        # Reset for next batch
        self._batch_ids[node_id] = None  # CHANGE: was aggregation._batch_id = None
        if batch_id in self._member_counts:
            del self._member_counts[batch_id]
```

**Step 5: Add helper method for test access (optional but recommended)**

Add a method to allow tests to verify batch state:

```python
def get_batch_id(self, node_id: str) -> str | None:
    """Get current batch ID for an aggregation node.

    Primarily for testing - production code should use AcceptResult.batch_id.
    """
    return self._batch_ids.get(node_id)
```

**Step 6: Run tests to check for failures**

Run: `pytest tests/engine/test_executors.py -v -k aggregation`

Expected: Some failures due to tests checking `aggregation._batch_id`

**Step 7: Commit refactor**

```bash
git add src/elspeth/engine/executors.py
git commit -m "refactor(executors): move batch state from plugin to executor

Batch IDs are now tracked internally in AggregationExecutor._batch_ids
instead of being monkey-patched onto plugin objects. This removes the
need for the AggregationLike protocol's _batch_id attribute.

BREAKING: aggregation._batch_id no longer exists. Use AcceptResult.batch_id
or executor.get_batch_id(node_id) for testing.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Delete AggregationLike Protocol

**Goal:** Remove the now-unnecessary AggregationLike protocol.

**Files:**
- Modify: `src/elspeth/engine/executors.py`

**Step 1: Delete AggregationLike protocol definition**

Remove lines 428-449 (the entire `AggregationLike` class):

```python
# DELETE THIS ENTIRE BLOCK:
class AggregationLike(Protocol):
    """Protocol for aggregation-like plugins.

    Aggregations collect multiple rows into batches and flush them when
    triggered. The _batch_id is mutable state managed by AggregationExecutor.
    """

    name: str
    node_id: str
    _batch_id: str | None  # Mutable, managed by executor

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
        """Accept a row into the current batch."""
        ...

    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        """Flush the current batch and return output rows."""
        ...
```

**Step 2: Verify AggregationProtocol is imported**

Ensure the import at the top of the file includes `AggregationProtocol`:

```python
from elspeth.plugins.protocols import (
    AggregationProtocol,  # Ensure this is present
    GateProtocol,
    SinkProtocol,
    TransformProtocol,
)
```

**Step 3: Run mypy to verify type consistency**

Run: `mypy src/elspeth/engine/executors.py --strict`

Expected: No errors (AggregationProtocol has same interface minus _batch_id)

**Step 4: Commit deletion**

```bash
git add src/elspeth/engine/executors.py
git commit -m "refactor(executors): delete AggregationLike protocol

AggregationLike was an engine-internal protocol that only existed to
declare the _batch_id attribute. Now that batch state is managed
internally by AggregationExecutor, we use AggregationProtocol directly.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Rename TransformLike Union Alias to RowPlugin

**Goal:** Rename the union alias to avoid confusion with the deleted protocol names.

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`

**Step 1: Update the type alias definition**

Change line 29:

```python
# OLD:
TransformLike = BaseTransform | BaseGate | BaseAggregation

# NEW:
RowPlugin = BaseTransform | BaseGate | BaseAggregation
"""Union of all row-processing plugin types for pipeline transforms list."""
```

**Step 2: Update PipelineConfig dataclass**

Change line 45:

```python
@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline."""

    source: SourceProtocol
    transforms: list[RowPlugin]  # CHANGE: was TransformLike
    sinks: dict[str, SinkProtocol]
```

**Step 3: Update _validate_route_destinations parameter**

Change line 182:

```python
def _validate_route_destinations(
    self,
    route_resolution_map: dict[tuple[str, str], str],
    available_sinks: set[str],
    transform_id_map: dict[int, str],
    transforms: list[RowPlugin],  # CHANGE: was TransformLike
) -> None:
```

**Step 4: Update _assign_plugin_node_ids parameter**

Change line 224:

```python
def _assign_plugin_node_ids(
    self,
    source: SourceProtocol,
    transforms: list[RowPlugin],  # CHANGE: was TransformLike
    sinks: dict[str, SinkProtocol],
    ...
) -> None:
```

**Step 5: Update comment in cli.py (optional)**

If there's a comment referencing TransformLike in cli.py, update it.

**Step 6: Run mypy**

Run: `mypy src/elspeth/engine/orchestrator.py --strict`

Expected: No errors

**Step 7: Commit rename**

```bash
git add src/elspeth/engine/orchestrator.py src/elspeth/cli.py
git commit -m "refactor(orchestrator): rename TransformLike alias to RowPlugin

TransformLike was confusing because it shared a name with the deleted
protocol. RowPlugin better describes what it is: a union of all plugin
types that process rows in the transforms pipeline.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Update Tests

**Goal:** Fix tests that reference `aggregation._batch_id`.

**Files:**
- Modify: `tests/engine/test_executors.py`
- Modify: `tests/engine/test_processor.py`

**Step 1: Update test_executors.py assertions**

Find and update these lines:

**Line 920** (in `test_accept_with_audit`):
```python
# OLD:
assert aggregation._batch_id == result.batch_id

# NEW:
assert result.batch_id is not None
assert executor.get_batch_id(aggregation.node_id) == result.batch_id
```

**Line 1007** (in `test_accept_multiple_rows`):
```python
# OLD:
assert aggregation._batch_id == batch_id

# NEW:
assert executor.get_batch_id(aggregation.node_id) == batch_id
```

**Lines 1102-1103** (in `test_flush_with_audit`):
```python
# OLD:
# Verify aggregation._batch_id is reset
assert aggregation._batch_id is None

# NEW:
# Verify batch tracking is reset
assert executor.get_batch_id(aggregation.node_id) is None
```

**Step 2: Update test mock classes in test_processor.py**

Remove `_batch_id` from mock aggregations:

**Line 565** (in mock class):
```python
# OLD:
class MockCountAggregation(BaseAggregation):
    def __init__(self, node_id: str) -> None:
        super().__init__({})
        self.node_id = node_id
        self._batch_id: str | None = None  # DELETE THIS LINE
        self._count: int = 0

# NEW:
class MockCountAggregation(BaseAggregation):
    def __init__(self, node_id: str) -> None:
        super().__init__({})
        self.node_id = node_id
        self._count: int = 0
```

**Line 632** (similar mock class):
```python
# Remove _batch_id attribute from this mock as well
```

**Step 3: Run all aggregation tests**

Run: `pytest tests/engine/test_executors.py tests/engine/test_processor.py -v -k aggregation`

Expected: All tests pass

**Step 4: Run full test suite**

Run: `pytest tests/engine/ -v`

Expected: All tests pass

**Step 5: Commit test updates**

```bash
git add tests/engine/test_executors.py tests/engine/test_processor.py
git commit -m "test(engine): update tests for internal batch state tracking

Tests now use executor.get_batch_id() instead of aggregation._batch_id.
Mock aggregations no longer define _batch_id attribute.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Final Verification

**Step 1: Verify no *Like protocols remain**

Run: `grep -r "TransformLike\|GateLike\|AggregationLike" src/elspeth/engine/ --include="*.py" | grep -v "RowPlugin"`

Expected: No results (only RowPlugin alias remains)

**Step 2: Verify no _batch_id on plugins**

Run: `grep -r "aggregation\._batch_id" src/ tests/`

Expected: No results

**Step 3: Run mypy on engine module**

Run: `mypy src/elspeth/engine/ --strict`

Expected: No errors

**Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`

Expected: All tests pass

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify WP-04a complete - all *Like protocols deleted

Verification:
- TransformLike protocol: deleted (f08c19a)
- GateLike protocol: deleted (f08c19a)
- AggregationLike protocol: deleted (this WP)
- TransformLike alias: renamed to RowPlugin
- _batch_id: moved to executor internal state
- All tests pass
- mypy clean

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `AggregationExecutor._batch_ids` dict added
- [ ] All `aggregation._batch_id` replaced with `self._batch_ids[node_id]`
- [ ] `AggregationLike` protocol deleted from executors.py
- [ ] `AggregationProtocol` imported and used in type hints
- [ ] `TransformLike` alias renamed to `RowPlugin` in orchestrator.py
- [ ] All 3 usages of alias updated (lines 45, 182, 224)
- [ ] Tests updated to use `executor.get_batch_id()`
- [ ] Mock aggregations no longer have `_batch_id`
- [ ] `mypy --strict` passes on engine module
- [ ] All tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/engine/executors.py` | MODIFY | Add `_batch_ids` dict, delete `AggregationLike`, update methods |
| `src/elspeth/engine/orchestrator.py` | MODIFY | Rename `TransformLike` → `RowPlugin` |
| `tests/engine/test_executors.py` | MODIFY | Update assertions to use `executor.get_batch_id()` |
| `tests/engine/test_processor.py` | MODIFY | Remove `_batch_id` from mock classes |

---

## Dependency Notes

- **Depends on:** WP-04 (SinkLike deleted)
- **Unlocks:** Nothing (pure cleanup, enables cleaner WP-06 aggregation work)
- **Risk:** Low - internal refactoring with no behavior change

---

## Rollback Trigger

If any test fails after Task 1 that isn't fixed by Task 4, halt and reassess. The batch state refactor should be transparent to all callers.
