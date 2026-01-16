# Chunk 1: Quick Wins Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 10 trivial integration issues from the v2 audit - plugin metadata gaps, missing hookimpls, stringly-typed status methods, and missing exports.

**Architecture:** These are all additive changes applying existing patterns. Each task is independent and can be committed separately. Total time ~1 hour.

**Tech Stack:** Python 3.11+, pluggy, dataclasses, Enums

**Dependencies:** None - these are leaf fixes with no cross-dependencies.

---

## Task 1: Add Missing Metadata to BaseSource

**Context:** BaseTransform, BaseGate, BaseSink all have `plugin_version` and `determinism` class attributes. BaseSource was missed during centralization.

**Files:**
- Modify: `src/elspeth/plugins/base.py:322-325`
- Test: `tests/plugins/test_base.py` (verify existing tests still pass)

### Step 1: Write the failing test

Add to `tests/plugins/test_base.py`:

```python
class TestBaseSourceMetadata:
    """Verify BaseSource has required metadata attributes."""

    def test_base_source_has_plugin_version(self) -> None:
        """BaseSource should have plugin_version class attribute."""
        from elspeth.plugins.base import BaseSource

        assert hasattr(BaseSource, "plugin_version")
        assert BaseSource.plugin_version == "0.0.0"

    def test_base_source_has_determinism(self) -> None:
        """BaseSource should have determinism class attribute."""
        from elspeth.plugins.base import BaseSource
        from elspeth.contracts import Determinism

        assert hasattr(BaseSource, "determinism")
        assert BaseSource.determinism == Determinism.IO_READ
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_base.py::TestBaseSourceMetadata -v`
Expected: FAIL with `AssertionError: assert False` (hasattr returns False)

### Step 3: Write minimal implementation

Edit `src/elspeth/plugins/base.py` - add after line 324 (after `node_id`):

```python
class BaseSource(ABC):
    """Base class for source plugins.
    ...
    """

    name: str
    output_schema: type[PluginSchema]
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility (ADD THESE TWO LINES)
    determinism: Determinism = Determinism.IO_READ  # Sources read from external world
    plugin_version: str = "0.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/test_base.py::TestBaseSourceMetadata -v`
Expected: PASS

### Step 5: Run broader tests to ensure no regressions

Run: `pytest tests/plugins/test_base.py -v`
Expected: All tests PASS

### Step 6: Commit

```bash
git add src/elspeth/plugins/base.py tests/plugins/test_base.py
git commit -m "fix(plugins): add plugin_version and determinism to BaseSource

BaseSource was missing these metadata attributes that BaseTransform,
BaseGate, and BaseSink already have. Sources default to IO_READ
determinism since they read from external filesystems/APIs."
```

---

## Task 2: Create Aggregations Hookimpl Directory

**Context:** `elspeth_get_aggregations` hookspec exists but no hookimpl returns plugins for it. Create the directory structure and empty hookimpl.

**Files:**
- Create: `src/elspeth/plugins/aggregations/__init__.py`
- Create: `src/elspeth/plugins/aggregations/hookimpl.py`

### Step 1: Create the __init__.py file

Create `src/elspeth/plugins/aggregations/__init__.py`:

```python
"""Built-in aggregation plugins.

Aggregations collect multiple rows until a trigger condition,
then flush them as a batch. Examples: collect 100 rows, collect
until end-of-day, collect by group key.

Currently no built-in aggregations - this is a placeholder for Phase 3.
"""
```

### Step 2: Create the hookimpl file

Create `src/elspeth/plugins/aggregations/hookimpl.py`:

```python
"""Hook implementation for built-in aggregation plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinAggregations:
    """Hook implementer for built-in aggregation plugins.

    Currently returns empty list - no built-in aggregations yet.
    This hookimpl ensures the hook is registered so external plugins
    can provide aggregations.
    """

    @hookimpl
    def elspeth_get_aggregations(self) -> list:
        """Return built-in aggregation plugin classes.

        Returns:
            Empty list - no built-in aggregations yet.
        """
        return []


# Singleton instance for registration
builtin_aggregations = ElspethBuiltinAggregations()
```

### Step 3: Verify files created correctly

Run: `python -c "from elspeth.plugins.aggregations.hookimpl import builtin_aggregations; print(builtin_aggregations)"`
Expected: `<elspeth.plugins.aggregations.hookimpl.ElspethBuiltinAggregations object at ...>`

### Step 4: Commit

```bash
git add src/elspeth/plugins/aggregations/
git commit -m "feat(plugins): create aggregations hookimpl directory

Adds the directory structure and empty hookimpl for aggregation plugins.
Currently returns empty list - placeholder for Phase 3 built-in aggregations."
```

---

## Task 3: Create Coalesces Hookimpl Directory

**Context:** Same as Task 2, but for coalesce plugins (merge results from parallel DAG paths).

**Files:**
- Create: `src/elspeth/plugins/coalesces/__init__.py`
- Create: `src/elspeth/plugins/coalesces/hookimpl.py`

### Step 1: Create the __init__.py file

Create `src/elspeth/plugins/coalesces/__init__.py`:

```python
"""Built-in coalesce plugins.

Coalesces merge results from parallel DAG paths back into a single
stream. Examples: wait for all paths, first-wins, merge by key.

Currently no built-in coalesces - this is a placeholder for Phase 3.
"""
```

### Step 2: Create the hookimpl file

Create `src/elspeth/plugins/coalesces/hookimpl.py`:

```python
"""Hook implementation for built-in coalesce plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinCoalesces:
    """Hook implementer for built-in coalesce plugins.

    Currently returns empty list - no built-in coalesces yet.
    This hookimpl ensures the hook is registered so external plugins
    can provide coalesces.
    """

    @hookimpl
    def elspeth_get_coalesces(self) -> list:
        """Return built-in coalesce plugin classes.

        Returns:
            Empty list - no built-in coalesces yet.
        """
        return []


# Singleton instance for registration
builtin_coalesces = ElspethBuiltinCoalesces()
```

### Step 3: Verify files created correctly

Run: `python -c "from elspeth.plugins.coalesces.hookimpl import builtin_coalesces; print(builtin_coalesces)"`
Expected: `<elspeth.plugins.coalesces.hookimpl.ElspethBuiltinCoalesces object at ...>`

### Step 4: Commit

```bash
git add src/elspeth/plugins/coalesces/
git commit -m "feat(plugins): create coalesces hookimpl directory

Adds the directory structure and empty hookimpl for coalesce plugins.
Currently returns empty list - placeholder for Phase 3 built-in coalesces."
```

---

## Task 4: Register Aggregation/Coalesce Hookimpls in Manager

**Context:** The new hookimpls need to be registered in `register_builtin_plugins()` so the hooks are active.

**Files:**
- Modify: `src/elspeth/plugins/manager.py:156-169`
- Test: `tests/plugins/test_hookimpl_registration.py`

### Step 1: Write the failing tests

Add to `tests/plugins/test_hookimpl_registration.py`:

```python
    def test_builtin_aggregations_registered(self) -> None:
        """Aggregation hookimpl is registered (returns empty list)."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # Should not raise - hook is registered
        aggregations = manager.get_aggregations()
        assert isinstance(aggregations, list)
        # Currently empty, but hook is active
        assert aggregations == []

    def test_builtin_coalesces_registered(self) -> None:
        """Coalesce hookimpl is registered (returns empty list)."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # Should not raise - hook is registered
        coalesces = manager.get_coalesces()
        assert isinstance(coalesces, list)
        # Currently empty, but hook is active
        assert coalesces == []
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/plugins/test_hookimpl_registration.py::TestBuiltinPluginDiscovery::test_builtin_aggregations_registered -v`
Expected: FAIL with `AttributeError: 'PluginManager' object has no attribute 'get_aggregations'` OR returns incorrect value

### Step 3: Update register_builtin_plugins()

Edit `src/elspeth/plugins/manager.py` - modify `register_builtin_plugins()`:

```python
    def register_builtin_plugins(self) -> None:
        """Register all built-in plugin hook implementers.

        Call this once at startup to make built-in plugins discoverable.
        """
        from elspeth.plugins.aggregations.hookimpl import builtin_aggregations
        from elspeth.plugins.coalesces.hookimpl import builtin_coalesces
        from elspeth.plugins.gates.hookimpl import builtin_gates
        from elspeth.plugins.sinks.hookimpl import builtin_sinks
        from elspeth.plugins.sources.hookimpl import builtin_sources
        from elspeth.plugins.transforms.hookimpl import builtin_transforms

        self.register(builtin_sources)
        self.register(builtin_transforms)
        self.register(builtin_gates)
        self.register(builtin_aggregations)
        self.register(builtin_coalesces)
        self.register(builtin_sinks)
```

### Step 4: Verify get_aggregations and get_coalesces exist

Check if PluginManager has these methods. If not, add them:

```python
    def get_aggregations(self) -> list:
        """Get all registered aggregation plugins."""
        results = self._pm.hook.elspeth_get_aggregations()
        return [plugin for plugins in results for plugin in plugins]

    def get_coalesces(self) -> list:
        """Get all registered coalesce plugins."""
        results = self._pm.hook.elspeth_get_coalesces()
        return [plugin for plugins in results for plugin in plugins]
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/plugins/test_hookimpl_registration.py -v`
Expected: All tests PASS

### Step 6: Commit

```bash
git add src/elspeth/plugins/manager.py tests/plugins/test_hookimpl_registration.py
git commit -m "feat(plugins): register aggregation/coalesce hookimpls in manager

The new hookimpl singletons are now registered in register_builtin_plugins().
Added get_aggregations() and get_coalesces() methods to PluginManager."
```

---

## Task 5: Fix Stringly-Typed export_status

**Context:** `set_export_status()` accepts `status: str` but should use `ExportStatus` enum with `_coerce_enum()` like other methods.

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py:379-416`
- Test: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing test

Add to `tests/core/landscape/test_recorder.py`:

```python
class TestExportStatusEnum:
    """Verify set_export_status uses enum validation."""

    def test_set_export_status_accepts_enum(self) -> None:
        """set_export_status accepts ExportStatus enum."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={})

        # Should accept enum without error
        recorder.set_export_status(run.run_id, ExportStatus.COMPLETED)

        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        # Value stored correctly
        assert updated_run.export_status == "completed" or updated_run.export_status == ExportStatus.COMPLETED

    def test_set_export_status_rejects_invalid_string(self) -> None:
        """set_export_status rejects invalid status strings."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={})

        # Should raise ValueError for invalid status
        with pytest.raises(ValueError, match="Invalid.*ExportStatus"):
            recorder.set_export_status(run.run_id, "invalid_status")
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/core/landscape/test_recorder.py::TestExportStatusEnum -v`
Expected: FAIL - currently accepts any string without validation

### Step 3: Update set_export_status()

Edit `src/elspeth/core/landscape/recorder.py` - modify `set_export_status()`:

```python
    def set_export_status(
        self,
        run_id: str,
        status: ExportStatus | str,
        *,
        error: str | None = None,
        export_format: str | None = None,
        export_sink: str | None = None,
    ) -> None:
        """Set export status for a run.

        This is separate from run status so export failures don't mask
        successful pipeline completion.

        Args:
            run_id: Run to update
            status: Export status (ExportStatus enum or string)
            error: Error message if status is 'failed'
            export_format: Format used (csv, json)
            export_sink: Sink name used for export
        """
        status_enum = _coerce_enum(status, ExportStatus)
        updates: dict[str, Any] = {"export_status": status_enum.value}

        if status_enum == ExportStatus.COMPLETED:
            updates["exported_at"] = _now()
        if error is not None:
            updates["export_error"] = error
        if export_format is not None:
            updates["export_format"] = export_format
        if export_sink is not None:
            updates["export_sink"] = export_sink

        with self._db.connection() as conn:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(**updates)
            )
```

Also add import at top if needed:
```python
from elspeth.contracts import ExportStatus
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestExportStatusEnum -v`
Expected: PASS

### Step 5: Run broader recorder tests

Run: `pytest tests/core/landscape/test_recorder.py -v`
Expected: All tests PASS

### Step 6: Commit

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py
git commit -m "fix(landscape): use ExportStatus enum in set_export_status

Applies _coerce_enum pattern to set_export_status() like other methods.
Rejects invalid status strings with clear error message."
```

---

## Task 6: Fix Stringly-Typed batch_status

**Context:** Same as Task 5, but for `update_batch_status()` using `BatchStatus` enum.

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py:1197-1227`
- Test: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing test

Add to `tests/core/landscape/test_recorder.py`:

```python
class TestBatchStatusEnum:
    """Verify update_batch_status uses enum validation."""

    def test_update_batch_status_accepts_enum(self) -> None:
        """update_batch_status accepts BatchStatus enum."""
        from elspeth.contracts import BatchStatus
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={})

        # Create a batch first
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg-node-1",
        )

        # Should accept enum without error
        recorder.update_batch_status(batch.batch_id, BatchStatus.COMPLETED)

    def test_update_batch_status_rejects_invalid_string(self) -> None:
        """update_batch_status rejects invalid status strings."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={})

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg-node-1",
        )

        # Should raise ValueError for invalid status
        with pytest.raises(ValueError, match="Invalid.*BatchStatus"):
            recorder.update_batch_status(batch.batch_id, "invalid_status")
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/core/landscape/test_recorder.py::TestBatchStatusEnum -v`
Expected: FAIL - currently accepts any string

### Step 3: Update update_batch_status()

Edit `src/elspeth/core/landscape/recorder.py` - modify `update_batch_status()`:

```python
    def update_batch_status(
        self,
        batch_id: str,
        status: BatchStatus | str,
        *,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> None:
        """Update batch status.

        Args:
            batch_id: Batch to update
            status: New status (BatchStatus enum or string)
            trigger_reason: Why the batch was triggered
            state_id: Node state for the flush operation
        """
        status_enum = _coerce_enum(status, BatchStatus)
        updates: dict[str, Any] = {"status": status_enum.value}

        if trigger_reason:
            updates["trigger_reason"] = trigger_reason
        if state_id:
            updates["aggregation_state_id"] = state_id
        if status_enum in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            updates["completed_at"] = _now()

        with self._db.connection() as conn:
            conn.execute(
                batches_table.update()
                .where(batches_table.c.batch_id == batch_id)
                .values(**updates)
            )
```

Also add import at top if needed:
```python
from elspeth.contracts import BatchStatus
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestBatchStatusEnum -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py
git commit -m "fix(landscape): use BatchStatus enum in update_batch_status

Applies _coerce_enum pattern to update_batch_status() like other methods.
Rejects invalid status strings with clear error message."
```

---

## Task 7: Fix String/Enum Mixing in Reproducibility

**Context:** `update_grade_after_purge()` compares database string to `enum.value` instead of converting to enum first.

**Files:**
- Modify: `src/elspeth/core/landscape/reproducibility.py:121-125`
- Test: `tests/core/landscape/test_recorder.py` or create new test file

### Step 1: Write the failing test

Create or add to test file:

```python
class TestReproducibilityGradeComparison:
    """Verify reproducibility uses proper enum comparison."""

    def test_update_grade_after_purge_uses_enum(self) -> None:
        """update_grade_after_purge compares enums, not strings."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            set_run_grade,
            update_grade_after_purge,
        )
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={})
        recorder.complete_run(run.run_id)

        # Set to REPLAY_REPRODUCIBLE
        set_run_grade(db, run.run_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # After purge, should degrade to ATTRIBUTABLE_ONLY
        update_grade_after_purge(db, run.run_id)

        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.reproducibility_grade in (
            ReproducibilityGrade.ATTRIBUTABLE_ONLY.value,
            "attributable_only",
        )
```

### Step 2: Run test to verify current behavior (may already pass)

Run: `pytest tests/core/landscape/test_reproducibility.py -v` (or wherever test is)
Expected: May PASS (functionally works), but code style is inconsistent

### Step 3: Update for consistent enum comparison

Edit `src/elspeth/core/landscape/reproducibility.py` lines 121-132:

```python
        current_grade = row[0]

        # Only REPLAY_REPRODUCIBLE needs to degrade
        # NULL grade in our audit data = corruption, fail fast
        if current_grade is None:
            raise ValueError(f"Run {run_id} has NULL reproducibility_grade - audit data corruption")

        # Convert to enum for comparison - invalid value = corruption, let ValueError propagate
        current_grade_enum = ReproducibilityGrade(current_grade)

        if current_grade_enum == ReproducibilityGrade.REPLAY_REPRODUCIBLE:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(
                    reproducibility_grade=ReproducibilityGrade.ATTRIBUTABLE_ONLY.value
                )
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/ -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/reproducibility.py
git commit -m "fix(landscape): use enum comparison in update_grade_after_purge

Convert database string to enum before comparison instead of
comparing string to enum.value. More consistent with rest of codebase."
```

---

## Task 8: Add Checkpoint Skip Logging

**Context:** `_maybe_checkpoint()` silently returns when checkpoint_manager is None but settings say enabled. Should log a warning.

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:117-120`

### Step 1: Locate the existing code

The current code at lines 117-120:
```python
        if self._checkpoint_manager is None:
            return
```

### Step 2: Add logging

Edit `src/elspeth/engine/orchestrator.py` - modify `_maybe_checkpoint()`:

```python
    def _maybe_checkpoint(self, run_id: str, token_id: str, node_id: str) -> None:
        """Create checkpoint if configured.
        ...
        """
        if not self._checkpoint_settings or not self._checkpoint_settings.enabled:
            return
        if self._checkpoint_manager is None:
            # Log once, not every row
            if not getattr(self, "_checkpoint_warning_logged", False):
                logger.warning(
                    "Checkpoint settings enabled but no checkpoint manager configured. "
                    "Checkpointing will not occur for this run."
                )
                self._checkpoint_warning_logged = True
            return

        self._sequence_number += 1
        # ... rest of method
```

### Step 3: Verify logger is imported

Ensure at top of file:
```python
from elspeth.core.logging import get_logger

logger = get_logger(__name__)
```

### Step 4: Run orchestrator tests

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "fix(engine): log warning when checkpoint manager not configured

When checkpoint settings are enabled but no checkpoint manager is
provided, log a warning (once) instead of silently skipping.
Helps diagnose misconfiguration issues."
```

---

## Task 9: Export resolve_config from core

**Context:** `resolve_config` is defined in `core.config` but not exported from `core.__init__`, creating inconsistent API surface.

**Files:**
- Modify: `src/elspeth/core/__init__.py`

### Step 1: Verify resolve_config exists

Run: `python -c "from elspeth.core.config import resolve_config; print(resolve_config)"`
Expected: `<function resolve_config at ...>`

### Step 2: Verify it's not currently exported from core

Run: `python -c "from elspeth.core import resolve_config"`
Expected: `ImportError: cannot import name 'resolve_config'`

### Step 3: Add to exports

Edit `src/elspeth/core/__init__.py`:

Add to imports (line ~15):
```python
from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    DatabaseSettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeExportSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RateLimitSettings,
    RetrySettings,
    RowPluginSettings,
    ServiceRateLimit,
    SinkSettings,
    load_settings,
    resolve_config,  # ADD THIS
)
```

Add to `__all__` (alphabetically):
```python
__all__ = [
    ...
    "resolve_config",
    ...
]
```

### Step 4: Verify export works

Run: `python -c "from elspeth.core import resolve_config; print(resolve_config)"`
Expected: `<function resolve_config at ...>`

### Step 5: Commit

```bash
git add src/elspeth/core/__init__.py
git commit -m "feat(core): export resolve_config from core module

Makes resolve_config available via 'from elspeth.core import resolve_config'
for consistent API surface."
```

---

## Task 10: Final Verification

**Context:** Run full test suite to ensure all changes work together.

### Step 1: Run all plugin tests

Run: `pytest tests/plugins/ -v`
Expected: All tests PASS

### Step 2: Run all landscape tests

Run: `pytest tests/core/landscape/ -v`
Expected: All tests PASS

### Step 3: Run all engine tests

Run: `pytest tests/engine/ -v`
Expected: All tests PASS

### Step 4: Run full test suite

Run: `pytest tests/ -v`
Expected: All tests PASS

### Step 5: Run type checker

Run: `mypy src/elspeth/plugins/base.py src/elspeth/core/landscape/recorder.py src/elspeth/engine/orchestrator.py`
Expected: No errors

---

## Summary

| Task | Description | Files | Time |
|------|-------------|-------|------|
| 1 | BaseSource metadata | base.py | 5 min |
| 2 | Aggregations hookimpl | aggregations/*.py | 3 min |
| 3 | Coalesces hookimpl | coalesces/*.py | 3 min |
| 4 | Register new hookimpls | manager.py | 5 min |
| 5 | ExportStatus enum | recorder.py | 8 min |
| 6 | BatchStatus enum | recorder.py | 8 min |
| 7 | Reproducibility enum | reproducibility.py | 5 min |
| 8 | Checkpoint logging | orchestrator.py | 5 min |
| 9 | Export resolve_config | __init__.py | 3 min |
| 10 | Final verification | - | 10 min |
| **Total** | | | **~55 min** |
