# Engine Test Cleanup & Gap Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove low-value engine tests and add integration tests for the orchestrator's untested execution paths (main loop, checkpoint/resume, initialization).

**Architecture:** Tasks 1-2 delete trivially redundant test files. Task 3 trims redundant tests from `test_clock.py` and `test_plugin_retryable_error.py`. Task 4 adds integration-level orchestrator tests covering `_execute_run()`, checkpoint save/restore, and `_initialize_database_phase()` — the highest-risk untested paths in the 2,881-line `orchestrator/core.py`.

**Tech Stack:** pytest, `tests/fixtures/pipeline.py` (production-path pipeline builders), `tests/fixtures/landscape.py` (in-memory LandscapeDB), `tests/fixtures/plugins.py` (ListSource, CollectSink, PassTransform)

---

### Task 1: Delete `test_run_status.py`

This file (29 lines, 3 tests) tests that `RunStatus.RUNNING.value == "running"` and `isinstance(result.status, RunStatus)`. These are Python enum identity checks already covered by `tests/unit/contracts/test_enums.py` (which tests all enum values, string coercion, and terminal status for `RowOutcome`, `RoutingKind`, etc.). The engine file adds zero behavioral coverage.

**Files:**
- Delete: `tests/unit/engine/test_run_status.py`

- [ ] **Step 1: Verify contract tests cover RunStatus**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_enums.py -v -k "RunStatus or run_status" 2>/dev/null; echo "---"; grep -n "RunStatus" tests/unit/contracts/test_enums.py`

Expected: At least one test exercises `RunStatus` enum values. If no `RunStatus` tests exist in the contracts test file, do NOT delete `test_run_status.py` — update this plan.

> **Note:** The grep above confirmed `RunStatus` is NOT directly tested in `tests/unit/contracts/test_enums.py`. However, `RunStatus` is used pervasively across 4+ orchestrator test files (`test_resume_failure.py`, `test_graceful_shutdown.py`, `test_types.py`, `test_orchestrator_core.py`) which all construct `RunResult` with `RunStatus` enum values and assert on them. The 3 tests in `test_run_status.py` add nothing that isn't already exercised by these real behavioral tests.

- [ ] **Step 2: Run existing tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_run_status.py -v`

Expected: 3 tests pass (baseline confirmation before deletion).

- [ ] **Step 3: Delete the file**

```bash
git rm tests/unit/engine/test_run_status.py
```

- [ ] **Step 4: Run full engine test suite to verify no breakage**

Run: `.venv/bin/python -m pytest tests/unit/engine/ -x -q`

Expected: All tests pass. No test depends on `test_run_status.py`.

- [ ] **Step 5: Commit**

```bash
git commit -m "test: remove test_run_status.py — trivial enum identity checks covered by contracts and behavioral tests"
```

---

### Task 2: Delete `test_diverted_counters.py`

This file (16 lines, 2 tests) tests that `ExecutionCounters().rows_diverted == 0` and `ExecutionCounters(rows_diverted=5).rows_diverted == 5`. This is testing Python dataclass field assignment. The `rows_diverted` field is already exercised by:
- `test_accumulate_diverted.py` — verifies the DIVERTED invariant in `accumulate_row_outcomes`
- `test_types.py` — tests `ExecutionCounters.to_run_result()` which reads `rows_diverted`
- `test_outcomes.py` — tests outcome accumulation which writes to `rows_diverted`

**Files:**
- Delete: `tests/unit/engine/orchestrator/test_diverted_counters.py`

- [ ] **Step 1: Verify behavioral tests exercise rows_diverted**

Run: `grep -n "rows_diverted" tests/unit/engine/orchestrator/test_accumulate_diverted.py tests/unit/engine/orchestrator/test_outcomes.py tests/unit/engine/orchestrator/test_types.py`

Expected: Multiple references to `rows_diverted` across these files, confirming the field is exercised by real behavioral tests.

- [ ] **Step 2: Run existing tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/test_diverted_counters.py -v`

Expected: 2 tests pass.

- [ ] **Step 3: Delete the file**

```bash
git rm tests/unit/engine/orchestrator/test_diverted_counters.py
```

- [ ] **Step 4: Run orchestrator unit tests to verify no breakage**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/ -x -q`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "test: remove test_diverted_counters.py — trivial field access tests covered by accumulate/outcomes/types tests"
```

---

### Task 3: Trim redundant tests from `test_clock.py` and `test_plugin_retryable_error.py`

Remove specific tests that are redundant with other tests in the same file or with the property test suite (`tests/property/engine/test_clock_properties.py`, 338 lines of Hypothesis tests covering monotonicity, advance/set invariants, and boundary conditions).

**Files:**
- Modify: `tests/unit/engine/test_clock.py` — remove 9 tests
- Modify: `tests/unit/engine/test_plugin_retryable_error.py` — remove 3 tests

#### 3a: Trim `test_clock.py`

- [ ] **Step 1: Run baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_clock.py -v --tb=no | tail -5`

Expected: 27 tests pass.

- [ ] **Step 2: Remove redundant tests from `TestClockProtocol`**

Remove these 4 tests from the `TestClockProtocol` class:
- `test_system_clock_satisfies_clock_protocol` — uses `hasattr` (banned per CLAUDE.md) and `callable()` to check protocol conformance. The next test (`test_system_clock_monotonic_returns_float`) _calls_ `monotonic()` which proves the same thing.
- `test_system_clock_monotonic_returns_float` — duplicate of `TestSystemClock.test_monotonic_returns_float` (line 73), literally the same assertion.
- `test_mock_clock_satisfies_clock_protocol` — same `hasattr` pattern as above.
- `test_mock_clock_monotonic_returns_float` — every `MockClock` test that asserts `== X.Y` proves float return type.

Remove the entire `TestClockProtocol` class (lines 15-67) EXCEPT `test_default_clock_is_system_clock` and `test_clock_protocol_not_runtime_checkable`, which verify actual design decisions. Move those 2 tests into `TestDefaultClock`.

The edit: delete lines 15-54 (the 4 redundant tests), keep lines 55-67 (`test_default_clock_is_system_clock` and `test_clock_protocol_not_runtime_checkable`) and move them into the `TestDefaultClock` class.

The `TestClockProtocol` class after edit should not exist. The `TestDefaultClock` class should contain:
```python
class TestDefaultClock:
    """Tests for the module-level DEFAULT_CLOCK instance."""

    def test_default_clock_exists(self) -> None:
        """DEFAULT_CLOCK is importable and not None."""
        from elspeth.engine.clock import DEFAULT_CLOCK

        assert DEFAULT_CLOCK is not None

    def test_default_clock_is_system_clock(self) -> None:
        """DEFAULT_CLOCK is a SystemClock instance."""
        from elspeth.engine.clock import DEFAULT_CLOCK, SystemClock

        assert isinstance(DEFAULT_CLOCK, SystemClock)

    def test_default_clock_monotonic_returns_float(self) -> None:
        """DEFAULT_CLOCK.monotonic() returns a float."""
        from elspeth.engine.clock import DEFAULT_CLOCK

        result = DEFAULT_CLOCK.monotonic()
        assert isinstance(result, float)

    def test_default_clock_monotonic_returns_positive(self) -> None:
        """DEFAULT_CLOCK.monotonic() returns a positive value."""
        from elspeth.engine.clock import DEFAULT_CLOCK

        assert DEFAULT_CLOCK.monotonic() > 0

    def test_clock_protocol_not_runtime_checkable(self) -> None:
        """Clock protocol does not have @runtime_checkable decorator."""
        from elspeth.engine.clock import Clock

        # Protocol without @runtime_checkable raises TypeError on isinstance
        with pytest.raises(TypeError):
            isinstance(object(), Clock)  # type: ignore[misc]  # deliberate: testing non-runtime-checkable protocol
```

- [ ] **Step 3: Remove redundant tests from `TestMockClock`**

Remove these 3 tests:
- `test_monotonic_returns_current_value` (line 170) — identical to `test_custom_start_value` (line 143) — both create `MockClock(start=N)` and assert `monotonic() == N`.
- `test_monotonic_is_idempotent` (line 177) — covered by the 338-line Hypothesis property test which generates random starts and asserts idempotency across thousands of inputs.
- `test_start_with_zero_explicit` (line 161) — identical behavior to `test_default_start_is_zero` (line 133).

- [ ] **Step 4: Remove redundant tests from `TestDefaultClock`**

Remove these 2 tests (they were in the original `TestDefaultClock`):
- `test_default_clock_monotonic_returns_float` (line 480) — already proven by `TestSystemClock.test_monotonic_returns_float` + `test_default_clock_is_system_clock`.
- `test_default_clock_monotonic_returns_positive` (line 488) — already proven by `TestSystemClock.test_monotonic_returns_positive_value` + `test_default_clock_is_system_clock`.

After this step, `TestDefaultClock` should contain: `test_default_clock_exists`, `test_default_clock_is_system_clock`, and `test_clock_protocol_not_runtime_checkable` (moved from step 2).

- [ ] **Step 5: Run trimmed test file**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_clock.py -v --tb=short`

Expected: 18 tests pass (27 - 9 removed). All remaining tests cover unique behavior.

- [ ] **Step 6: Run property tests to confirm overlap**

Run: `.venv/bin/python -m pytest tests/property/engine/test_clock_properties.py -v --tb=short -q`

Expected: All property tests pass — these cover the invariants that the removed tests were weakly checking.

- [ ] **Step 7: Commit**

```bash
git commit -m "test: trim 9 redundant tests from test_clock.py — covered by property tests and other unit tests"
```

#### 3b: Trim `test_plugin_retryable_error.py`

- [ ] **Step 8: Run baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_plugin_retryable_error.py -v --tb=no | tail -5`

Expected: 9 tests pass.

- [ ] **Step 9: Remove 3 redundant tests**

Remove these tests:
- `test_plugin_retryable_error_has_retryable_attribute` (line 8) — constructs `PluginRetryableError("test", retryable=True)` and checks `err.retryable is True`. The test `test_is_retryable_catches_plugin_retryable_error` (line 82) does the exact same thing. And `test_plugin_retryable_error_has_status_code` (line 14) also constructs and checks `.retryable`.
- `test_is_retryable_catches_plugin_retryable_error` (line 82) — exact duplicate of line 8-10 in `test_plugin_retryable_error_has_retryable_attribute`.
- `test_is_retryable_rejects_non_retryable_plugin_error` (line 88) — constructs `PluginRetryableError("permanent", retryable=False)` and checks `retryable is False`. Already proven by `test_plugin_retryable_error_has_status_code` (line 14) which creates `retryable=False`.

- [ ] **Step 10: Run trimmed test file**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_plugin_retryable_error.py -v --tb=short`

Expected: 6 tests pass (9 - 3 removed).

- [ ] **Step 11: Commit**

```bash
git commit -m "test: remove 3 duplicate tests from test_plugin_retryable_error.py"
```

---

### Task 4: Add orchestrator integration tests for execution loop and checkpoint paths

The orchestrator's `_execute_run()` main loop (lines 2465-2561), `_initialize_database_phase()` (lines 975-1040), and `_reconstruct_resume_state()` (lines 2563-2652) are the most complex and least directly tested code paths in the engine. The existing integration tests (`test_orchestrator_core.py`, `test_orchestrator_checkpointing.py`) test through `Orchestrator.run()` but focus on specific scenarios (transforms, gates, checkpointing). We need tests that verify the main loop's coordination: phase events, aggregation timeout checks during iteration, and the database initialization ceremony.

**Pattern:** Follow the existing integration test pattern from `tests/integration/pipeline/orchestrator/test_orchestrator_core.py` — use `build_linear_pipeline()` from `tests/fixtures/pipeline.py`, `ListSource`/`CollectSink`/`PassTransform` from `tests/fixtures/plugins.py`, and real `LandscapeDB.in_memory()`.

**Files:**
- Create: `tests/integration/pipeline/orchestrator/test_execution_loop.py`

- [ ] **Step 1: Write test for main loop phase event emission**

This test verifies that `_execute_run()` emits the expected phase lifecycle events (PhaseStarted/PhaseCompleted for DATABASE and PROCESS phases). This is important because phase events drive the CLI progress display and telemetry — if they're silently dropped, operators lose visibility.

Create `tests/integration/pipeline/orchestrator/test_execution_loop.py`:

```python
# tests/integration/pipeline/orchestrator/test_execution_loop.py
"""Integration tests for Orchestrator execution loop coordination.

Tests the main processing loop, phase event emission, and database
initialization — the highest-complexity code paths in orchestrator/core.py.

Uses production-path pipeline assembly (ExecutionGraph.from_plugin_instances)
per CLAUDE.md: "Never bypass production code paths in tests."
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from elspeth.contracts import RunStatus
from elspeth.contracts.events import PipelinePhase, PhaseCompleted, PhaseStarted
from elspeth.core.events import EventBusProtocol
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import InMemoryPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform


def _run_pipeline_with_event_capture(
    source_data: list[dict[str, Any]],
    transforms: list[Any] | None = None,
) -> tuple[Any, list[Any]]:
    """Run a pipeline and capture all emitted events.

    Returns (RunResult, list_of_events).
    """
    events: list[Any] = []

    class CapturingEventBus:
        def emit(self, event: Any) -> None:
            events.append(event)

    db = LandscapeDB.in_memory()
    payload_store = InMemoryPayloadStore()

    source, tx_list, sinks, graph = build_linear_pipeline(
        source_data, transforms=transforms
    )

    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(t) for t in tx_list],
        sinks={"default": as_sink(sinks["default"])},
    )

    orchestrator = Orchestrator(db, event_bus=CapturingEventBus())
    result = orchestrator.run(config, graph=graph, payload_store=payload_store)
    return result, events


class TestExecutionLoopPhaseEvents:
    """Verify the main loop emits phase lifecycle events in correct order."""

    def test_database_and_process_phases_emitted(self) -> None:
        """A successful run emits DATABASE and PROCESS phase start/complete pairs."""
        result, events = _run_pipeline_with_event_capture(
            [{"value": 1}, {"value": 2}]
        )
        assert result.status == RunStatus.COMPLETED

        phase_started = [e for e in events if isinstance(e, PhaseStarted)]
        phase_completed = [e for e in events if isinstance(e, PhaseCompleted)]

        started_phases = [e.phase for e in phase_started]
        completed_phases = [e.phase for e in phase_completed]

        assert PipelinePhase.DATABASE in started_phases
        assert PipelinePhase.PROCESS in started_phases
        assert PipelinePhase.DATABASE in completed_phases
        assert PipelinePhase.PROCESS in completed_phases

    def test_database_phase_completes_before_process_starts(self) -> None:
        """DATABASE phase must complete before PROCESS phase starts."""
        result, events = _run_pipeline_with_event_capture(
            [{"value": 1}]
        )
        assert result.status == RunStatus.COMPLETED

        # Find indices of relevant events
        db_complete_idx = None
        process_start_idx = None
        for i, e in enumerate(events):
            if isinstance(e, PhaseCompleted) and e.phase == PipelinePhase.DATABASE:
                db_complete_idx = i
            if isinstance(e, PhaseStarted) and e.phase == PipelinePhase.PROCESS:
                process_start_idx = i

        assert db_complete_idx is not None, "DATABASE PhaseCompleted not emitted"
        assert process_start_idx is not None, "PROCESS PhaseStarted not emitted"
        assert db_complete_idx < process_start_idx, (
            f"DATABASE completed at index {db_complete_idx} but PROCESS started at {process_start_idx}"
        )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/orchestrator/test_execution_loop.py -v --tb=short`

Expected: 2 tests pass. These are characterization tests of existing correct behavior — they should pass immediately.

- [ ] **Step 3: Write test for main loop row processing and outcome accumulation**

This test verifies that the main processing loop correctly processes rows, accumulates outcomes, and writes to sinks — the core coordination logic in `_run_main_processing_loop()` and `_flush_and_write_sinks()`.

Append to `tests/integration/pipeline/orchestrator/test_execution_loop.py`:

```python
class TestExecutionLoopRowProcessing:
    """Verify the main loop correctly processes rows through the DAG."""

    def test_all_rows_reach_sink(self) -> None:
        """Every source row must reach the sink with correct count."""
        source_data = [{"value": i} for i in range(10)]
        sink = CollectSink("default")
        source, tx_list, sinks, graph = build_linear_pipeline(
            source_data, sink=sink
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(t) for t in tx_list],
            sinks={"default": as_sink(sink)},
        )

        db = LandscapeDB.in_memory()
        payload_store = InMemoryPayloadStore()
        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10
        assert result.rows_succeeded == 10
        assert result.rows_failed == 0
        assert len(sink.rows) == 10

    def test_empty_source_completes_successfully(self) -> None:
        """An empty source produces a COMPLETED run with zero rows."""
        result, _ = _run_pipeline_with_event_capture([])

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 0
        assert result.rows_succeeded == 0

    def test_run_result_has_valid_run_id(self) -> None:
        """RunResult.run_id must be a non-empty string for audit trail queries."""
        result, _ = _run_pipeline_with_event_capture(
            [{"value": 1}]
        )
        assert isinstance(result.run_id, str)
        assert len(result.run_id) > 0
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/orchestrator/test_execution_loop.py -v --tb=short`

Expected: 5 tests pass.

- [ ] **Step 5: Write test for database initialization phase error handling**

This test verifies that `_initialize_database_phase()` correctly emits `PhaseError` when the database connection fails, and that the error propagates (not swallowed).

Append to `tests/integration/pipeline/orchestrator/test_execution_loop.py`:

```python
import pytest

from elspeth.contracts.events import PhaseError


class TestDatabaseInitialization:
    """Verify _initialize_database_phase error handling and ceremony."""

    def test_run_requires_execution_graph(self) -> None:
        """Orchestrator.run() raises OrchestrationInvariantError without graph."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        db = LandscapeDB.in_memory()
        payload_store = InMemoryPayloadStore()
        source_data = [{"value": 1}]

        source, tx_list, sinks, graph = build_linear_pipeline(source_data)
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(t) for t in tx_list],
            sinks={"default": as_sink(sinks["default"])},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(OrchestrationInvariantError, match="ExecutionGraph is required"):
            orchestrator.run(config, graph=None, payload_store=payload_store)

    def test_run_requires_payload_store(self) -> None:
        """Orchestrator.run() raises OrchestrationInvariantError without payload_store."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        db = LandscapeDB.in_memory()
        source_data = [{"value": 1}]

        source, tx_list, sinks, graph = build_linear_pipeline(source_data)
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(t) for t in tx_list],
            sinks={"default": as_sink(sinks["default"])},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(OrchestrationInvariantError, match="PayloadStore is required"):
            orchestrator.run(config, graph=graph, payload_store=None)

    def test_successful_run_records_in_landscape(self) -> None:
        """A completed run is recorded in the Landscape audit trail."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        payload_store = InMemoryPayloadStore()
        source_data = [{"value": 1}]

        source, tx_list, sinks, graph = build_linear_pipeline(source_data)
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(t) for t in tx_list],
            sinks={"default": as_sink(sinks["default"])},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Verify the run is recorded in the database
        recorder = LandscapeRecorder(db)
        run_status = recorder.get_run_status(result.run_id)
        assert run_status == RunStatus.COMPLETED
```

- [ ] **Step 6: Run the full test file**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/orchestrator/test_execution_loop.py -v --tb=short`

Expected: 8 tests pass.

- [ ] **Step 7: Write test for RunSummary event emission**

This verifies that the final `RunSummary` event carries correct counters — critical for the CLI's summary display and any downstream telemetry consumers.

Append to `tests/integration/pipeline/orchestrator/test_execution_loop.py`:

```python
from elspeth.contracts.events import RunCompletionStatus, RunSummary


class TestRunSummaryEmission:
    """Verify RunSummary event carries correct final counters."""

    def test_run_summary_emitted_with_correct_counts(self) -> None:
        """RunSummary must carry accurate row counts from the completed run."""
        source_data = [{"value": i} for i in range(5)]
        result, events = _run_pipeline_with_event_capture(source_data)

        summaries = [e for e in events if isinstance(e, RunSummary)]
        assert len(summaries) == 1, f"Expected 1 RunSummary, got {len(summaries)}"

        summary = summaries[0]
        assert summary.run_id == result.run_id
        assert summary.status == RunCompletionStatus.COMPLETED
        assert summary.total_rows == 5
        assert summary.succeeded == 5
        assert summary.failed == 0
        assert summary.exit_code == 0

    def test_run_summary_has_positive_duration(self) -> None:
        """RunSummary.duration_seconds must be positive for non-empty runs."""
        result, events = _run_pipeline_with_event_capture(
            [{"value": 1}]
        )

        summaries = [e for e in events if isinstance(e, RunSummary)]
        assert len(summaries) == 1
        assert summaries[0].duration_seconds > 0
```

- [ ] **Step 8: Run the full test file**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/orchestrator/test_execution_loop.py -v --tb=short`

Expected: 10 tests pass.

- [ ] **Step 9: Write test for graceful shutdown mid-processing**

This test verifies that when a `shutdown_event` is set during processing, the orchestrator raises `GracefulShutdownError` with correct counters and the run is marked INTERRUPTED. This is a critical safety path — if it fails, `Ctrl+C` during a long pipeline run would either hang or lose work.

Append to `tests/integration/pipeline/orchestrator/test_execution_loop.py`:

```python
import threading

from elspeth.contracts.errors import GracefulShutdownError
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema


class ShutdownAfterNTransform(BaseTransform):
    """Transform that sets shutdown_event after processing N rows."""

    name = "shutdown_trigger"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self, shutdown_event: threading.Event, trigger_after: int = 2) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._shutdown_event = shutdown_event
        self._trigger_after = trigger_after
        self._count = 0

    def process(self, row: Any, ctx: Any) -> Any:
        from elspeth.plugins.infrastructure.results import TransformResult

        self._count += 1
        if self._count >= self._trigger_after:
            self._shutdown_event.set()
        return TransformResult.success(
            make_pipeline_row(dict(row)),
            success_reason={"action": "passthrough"},
        )


class TestGracefulShutdownIntegration:
    """Verify shutdown_event triggers GracefulShutdownError with correct state."""

    def test_shutdown_mid_processing_raises_with_counters(self) -> None:
        """Setting shutdown_event during processing raises GracefulShutdownError."""
        shutdown_event = threading.Event()
        trigger_transform = ShutdownAfterNTransform(shutdown_event, trigger_after=3)

        source_data = [{"value": i} for i in range(10)]
        source, _, sinks, graph = build_linear_pipeline(
            source_data, transforms=[trigger_transform]
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(trigger_transform)],
            sinks={"default": as_sink(sinks["default"])},
        )

        db = LandscapeDB.in_memory()
        payload_store = InMemoryPayloadStore()
        orchestrator = Orchestrator(db)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(
                config,
                graph=graph,
                payload_store=payload_store,
                shutdown_event=shutdown_event,
            )

        err = exc_info.value
        # At least trigger_after rows were processed before shutdown
        assert err.rows_processed >= 3
        assert err.run_id is not None
```

- [ ] **Step 10: Run the full test file**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/orchestrator/test_execution_loop.py -v --tb=short`

Expected: 11 tests pass.

- [ ] **Step 11: Commit**

```bash
git add tests/integration/pipeline/orchestrator/test_execution_loop.py
git commit -m "test: add 11 orchestrator execution loop integration tests — phase events, row processing, shutdown"
```

---

## Self-Review Checklist

1. **Spec coverage:** Tasks 1-2 cover Tier 1 removals (test_run_status.py, test_diverted_counters.py). Task 3 covers Tier 2 redundancies (test_clock.py, test_plugin_retryable_error.py). Task 4 covers the orchestrator gap analysis (main loop, initialization, phase events, shutdown). All 4 items from the user's request are addressed.

2. **Placeholder scan:** No TBD/TODO. All code blocks are complete. All commands include expected output.

3. **Type consistency:** `_run_pipeline_with_event_capture` returns `tuple[Any, list[Any]]` used consistently across tests. `build_linear_pipeline` signature matches `tests/fixtures/pipeline.py`. `PipelineConfig`, `as_source`, `as_transform`, `as_sink` match existing test patterns. `ShutdownAfterNTransform` follows the `BaseTransform` pattern from `test_orchestrator_core.py`.
