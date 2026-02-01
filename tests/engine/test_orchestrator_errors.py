# tests/engine/test_orchestrator_errors.py
"""Tests for Orchestrator error handling and quarantine functionality.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import SourceRow
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult


class TestOrchestratorErrorHandling:
    """Test error handling in orchestration."""

    def test_run_marks_failed_on_transform_exception(self, payload_store) -> None:
        """If a transform raises, run status should be failed in Landscape."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class ExplodingTransform(BaseTransform):
            name = "exploding"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                raise RuntimeError("Transform exploded!")

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 42}])
        transform = ExplodingTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError, match="Transform exploded!"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify run was marked as failed in Landscape audit trail
        # Query for all runs and find the one that was created
        from elspeth.contracts import RunStatus

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1, "Expected exactly one run in Landscape"

        failed_run = runs[0]
        assert failed_run.status == RunStatus.FAILED, f"Landscape audit trail must record status=FAILED, got status={failed_run.status!r}"


class TestOrchestratorSourceQuarantineValidation:
    """Test that invalid source quarantine destinations fail at startup.

    Per P2-2026-01-19-source-quarantine-silent-drop:
    Source on_validation_failure destinations should be validated at startup,
    just like gate routes and transform error sinks.
    """

    def test_invalid_source_quarantine_destination_fails_at_init(self, payload_store) -> None:
        """Source quarantine to non-existent sink should fail before processing rows.

        When a source has on_validation_failure set to a sink that doesn't exist,
        the orchestrator should fail at initialization with a clear error message,
        NOT silently drop quarantined rows at runtime.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import (
            Orchestrator,
            PipelineConfig,
            RouteValidationError,
        )

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            id: int
            name: str

        class QuarantiningSource(_TestSourceBase):
            """Source that yields one valid row and one quarantined row."""

            name = "quarantining_source"
            output_schema = RowSchema

            def __init__(self) -> None:
                self.load_called = False
                # Track the quarantine destination for validation
                self._on_validation_failure = "nonexistent_quarantine_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                self.load_called = True
                # Valid row
                yield SourceRow.valid({"id": 1, "name": "alice"})
                # Quarantined row - destination doesn't exist!
                yield SourceRow.quarantined(
                    row={"id": 2, "name": "bob", "bad_field": "invalid"},
                    error="Validation failed",
                    destination="nonexistent_quarantine_sink",
                )

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = QuarantiningSource()
        default_sink = CollectSink()
        # Note: NO 'nonexistent_quarantine_sink' provided!

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink)},  # Only default, no quarantine sink
        )

        orchestrator = Orchestrator(db)

        # Should fail at initialization with clear error message
        with pytest.raises(RouteValidationError) as exc_info:
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify error message contains helpful information
        error_msg = str(exc_info.value)
        assert "nonexistent_quarantine_sink" in error_msg  # Invalid destination
        assert "default" in error_msg  # Available sinks

        # Verify no rows were processed (failed at validation, not runtime)
        assert not source.load_called, "Source.load() should not be called - validation failed first"


class TestOrchestratorQuarantineMetrics:
    """Test that QUARANTINED rows are counted separately from FAILED."""

    def test_orchestrator_counts_quarantined_rows(self, payload_store) -> None:
        """Orchestrator should count QUARANTINED rows separately.

        A transform with _on_error="discard" intentionally quarantines rows
        when it returns TransformResult.error(). These should be counted
        as quarantined, not failed.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int
            quality: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class QualityFilter(BaseTransform):
            """Transform that errors on 'bad' quality rows.

            With _on_error="discard", these become QUARANTINED.
            """

            name = "quality_filter"
            input_schema = ValueSchema
            output_schema = ValueSchema
            _on_error = "discard"  # Intentionally discard errors

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                if row.get("quality") == "bad":
                    return TransformResult.error({"reason": "validation_failed", "error": "bad_quality", "value": row["value"]})
                return TransformResult.success(row, success_reason={"action": "quality_check_passed"})

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # 3 rows: good, bad, good
        source = ListSource(
            [
                {"value": 1, "quality": "good"},
                {"value": 2, "quality": "bad"},  # Will be quarantined
                {"value": 3, "quality": "good"},
            ]
        )
        transform = QualityFilter()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify counts
        assert run_result.status == "completed"
        assert run_result.rows_processed == 3, "All 3 rows should be processed"
        assert run_result.rows_succeeded == 2, "2 good quality rows should succeed"
        assert run_result.rows_quarantined == 1, "1 bad quality row should be quarantined"
        assert run_result.rows_failed == 0, "No rows should fail (quarantine != fail)"

        # Only good rows written to sink
        assert len(sink.results) == 2
        assert all(r["quality"] == "good" for r in sink.results)


class TestSourceQuarantineTokenOutcome:
    """Test that source-level quarantine records QUARANTINED token_outcome.

    BUG: Quarantined source rows lack QUARANTINED token_outcome
    - Source yields SourceRow.quarantined() for invalid rows
    - Orchestrator creates token and routes to quarantine sink
    - BUT: orchestrator never calls recorder.record_token_outcome(QUARANTINED)
    - Result: token_outcomes table has no terminal state for quarantined rows

    This violates the audit trail completeness requirement (CLAUDE.md: every row
    reaches exactly one terminal state).
    """

    def test_source_quarantine_records_quarantined_token_outcome(self, payload_store) -> None:
        """Source-quarantined rows MUST have QUARANTINED outcome in token_outcomes.

        When a source yields SourceRow.quarantined(), the orchestrator should:
        1. Create a token for the quarantined row
        2. Record QUARANTINED token_outcome with error_hash
        3. Route the row to the quarantine sink

        The audit trail must be complete - explain/lineage queries should find
        the terminal state for quarantined rows.
        """
        from collections.abc import Iterator

        from elspeth.contracts import ArtifactDescriptor, PluginSchema, RowOutcome
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            id: int
            name: str

        class QuarantiningSource(_TestSourceBase):
            """Source that yields one valid row and one quarantined row."""

            name = "quarantining_source"
            output_schema = RowSchema

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                # Valid row
                yield SourceRow.valid({"id": 1, "name": "alice"})
                # Quarantined row - simulates validation failure
                yield SourceRow.quarantined(
                    row={"id": 2, "name": "bob", "invalid_field": "bad_data"},
                    error="Schema validation failed: unexpected field 'invalid_field'",
                    destination="quarantine",
                )
                # Another valid row
                yield SourceRow.valid({"id": 3, "name": "charlie"})

        class CollectSink(_TestSinkBase):
            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

        source = QuarantiningSource()
        default_sink = CollectSink()
        quarantine_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(quarantine_sink),
            },
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify run completed successfully
        assert run_result.status == "completed"
        assert run_result.rows_processed == 3
        assert run_result.rows_succeeded == 2  # alice and charlie
        assert run_result.rows_quarantined == 1  # bob

        # Verify sinks received correct rows
        assert len(default_sink.results) == 2
        assert len(quarantine_sink.results) == 1
        assert quarantine_sink.results[0]["id"] == 2  # bob went to quarantine

        # === THE CRITICAL ASSERTION ===
        # Query token_outcomes table to verify QUARANTINED outcome was recorded
        run_id = run_result.run_id

        # Get all token outcomes for this run
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == run_id)).fetchall()

        # Should have 3 terminal outcomes: 2 COMPLETED + 1 QUARANTINED
        assert len(outcomes) == 3, f"Expected 3 token outcomes, got {len(outcomes)}"

        # Find the QUARANTINED outcome specifically
        quarantined_outcomes = [o for o in outcomes if o.outcome == RowOutcome.QUARANTINED.value]
        assert len(quarantined_outcomes) == 1, (
            f"Expected exactly 1 QUARANTINED outcome, got {len(quarantined_outcomes)}. "
            f"All outcomes: {[(o.outcome, o.error_hash) for o in outcomes]}"
        )

        # Verify error_hash was recorded (for audit trail completeness)
        quarantined = quarantined_outcomes[0]
        assert quarantined.error_hash is not None, (
            "QUARANTINED outcome must have error_hash for audit trail. "
            "The error hash allows correlating with the original validation error."
        )

    def test_quarantine_outcome_not_recorded_if_sink_fails(self, payload_store) -> None:
        """QUARANTINED outcome must NOT be recorded if quarantine sink write fails.

        Bug: P1-2026-01-31-quarantine-outcome-before-durability

        The current implementation records QUARANTINED outcome BEFORE the sink
        write completes. If the sink fails, the audit trail shows QUARANTINED
        but no durable data exists - violating the durability invariant.

        Expected behavior:
        - If quarantine sink.write() fails, NO QUARANTINED outcome should exist
        - The row should be in a FAILED state instead
        - The pipeline should fail (sink errors are fatal)
        """
        from collections.abc import Iterator

        from elspeth.contracts import ArtifactDescriptor, PluginSchema, RowOutcome
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            id: int
            name: str

        class QuarantiningSource(_TestSourceBase):
            """Source that yields one quarantined row."""

            name = "quarantining_source"
            output_schema = RowSchema

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.quarantined(
                    row={"id": 1, "name": "bad_row"},
                    error="Validation failed",
                    destination="quarantine",
                )

        class FailingSink(_TestSinkBase):
            """Sink that always fails on write."""

            name = "failing_sink"

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Sink write failed!")

        class CollectSink(_TestSinkBase):
            """Normal sink that collects rows."""

            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

        source = QuarantiningSource()
        default_sink = CollectSink()
        failing_quarantine_sink = FailingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(failing_quarantine_sink),
            },
        )

        orchestrator = Orchestrator(db)

        # Run should fail because quarantine sink fails
        with pytest.raises(RuntimeError, match="Sink write failed!"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Get the run that was created (even though it failed)
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(db)
        runs = recorder.list_runs()
        assert len(runs) == 1, "Expected exactly one run in Landscape"
        run_id = runs[0].run_id

        # THE CRITICAL ASSERTION:
        # There should be NO QUARANTINED outcome because sink write failed
        # (durability was not achieved)
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == run_id)).fetchall()

        quarantined_outcomes = [o for o in outcomes if o.outcome == RowOutcome.QUARANTINED.value]

        # BUG: Currently this FAILS because QUARANTINED is recorded BEFORE sink durability
        assert len(quarantined_outcomes) == 0, (
            "QUARANTINED outcome should NOT exist when quarantine sink fails! "
            f"Found {len(quarantined_outcomes)} QUARANTINED outcomes. "
            "This violates the 'outcome = durable output' invariant. "
            "See bug: P1-2026-01-31-quarantine-outcome-before-durability"
        )


class TestQuarantineDestinationRuntimeValidation:
    """Test that invalid quarantine destinations at runtime crash (not silent skip).

    Per P2-2026-01-31-quarantine-invalid-destination-silent-drop:
    When a source yields a quarantined row with an invalid destination at runtime,
    the orchestrator must crash with a clear error. Silent drops violate the
    "every row reaches exactly one terminal state" contract.

    Init-time validation checks `source._on_validation_failure`, but runtime
    uses `source_item.quarantine_destination`. A plugin bug could make these differ.
    """

    def test_invalid_quarantine_destination_at_runtime_crashes(self, payload_store) -> None:
        """Quarantined row with invalid destination at runtime must crash.

        Scenario:
        - Source has `_on_validation_failure = "quarantine"` (passes init validation)
        - But source yields `SourceRow.quarantined(..., destination="nonexistent")`
        - This is a plugin bug - should crash, not silently drop the row

        Per CLAUDE.md:
        - "Plugin bugs must crash"
        - "Every row reaches exactly one terminal state - no silent drops"
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            id: int
            name: str

        class MisbehavingSource(_TestSourceBase):
            """Source that passes init validation but emits invalid destination at runtime.

            This simulates a plugin bug: _on_validation_failure is validated at init,
            but the source yields a different destination at runtime.
            """

            name = "misbehaving_source"
            output_schema = RowSchema

            def __init__(self) -> None:
                # This passes init validation - "quarantine" sink exists
                self._on_validation_failure = "quarantine"

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                # Valid row - no problem
                yield SourceRow.valid({"id": 1, "name": "alice"})

                # BUG: yields destination that differs from _on_validation_failure
                # "typo_sink" doesn't exist - this is a plugin bug
                yield SourceRow.quarantined(
                    row={"id": 2, "name": "bob"},
                    error="validation_failed",
                    destination="typo_sink",  # <-- Plugin bug! Different from _on_validation_failure
                )

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        class QuarantineSink(_TestSinkBase):
            """Valid quarantine sink - exists for init validation to pass."""

            name = "quarantine"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = MisbehavingSource()
        default_sink = CollectSink()
        quarantine_sink = QuarantineSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(quarantine_sink),  # Init validation passes
            },
        )

        orchestrator = Orchestrator(db)

        # Should CRASH with clear error about invalid destination
        # Currently BUGS: silently skips the row (no crash, no audit record)
        with pytest.raises(Exception) as exc_info:
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Error message should identify the problem clearly
        error_msg = str(exc_info.value)
        assert "typo_sink" in error_msg, f"Error should mention the invalid destination 'typo_sink'. Got: {error_msg}"

    def test_none_quarantine_destination_at_runtime_crashes(self, payload_store) -> None:
        """Quarantined row with None destination at runtime must crash.

        Scenario:
        - Source yields SourceRow with is_quarantined=True but quarantine_destination=None
        - This bypasses the factory method contract
        - Should crash, not silently drop

        This tests the case where SourceRow is constructed directly instead of
        using the SourceRow.quarantined() factory method.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            id: int
            name: str

        class BadSourceRow(_TestSourceBase):
            """Source that yields SourceRow with None destination (bypassing factory)."""

            name = "bad_source"
            output_schema = RowSchema

            def __init__(self) -> None:
                self._on_validation_failure = "quarantine"

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                # Valid row - no problem
                yield SourceRow.valid({"id": 1, "name": "alice"})

                # BUG: Directly construct SourceRow with None destination
                # This bypasses the factory method's required destination parameter
                yield SourceRow(
                    row={"id": 2, "name": "bob"},
                    is_quarantined=True,
                    quarantine_error="validation_failed",
                    quarantine_destination=None,  # <-- Missing destination!
                )

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        class QuarantineSink(_TestSinkBase):
            name = "quarantine"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = BadSourceRow()
        default_sink = CollectSink()
        quarantine_sink = QuarantineSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(quarantine_sink),
            },
        )

        orchestrator = Orchestrator(db)

        # Should CRASH with clear error about missing destination
        # Currently BUGS: silently skips the row
        with pytest.raises(Exception) as exc_info:
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Error message should identify the problem
        error_msg = str(exc_info.value)
        assert "quarantine" in error_msg.lower() or "destination" in error_msg.lower(), (
            f"Error should mention quarantine or destination. Got: {error_msg}"
        )
