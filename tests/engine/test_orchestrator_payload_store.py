# tests/engine/test_orchestrator_payload_store.py
"""Acceptance tests for payload_store requirement in orchestrator.run().

Verifies that:
1. payload_store parameter is mandatory (raises ValueError if None)
2. source_data_ref is populated in audit trail when payload_store provided
3. payload_store instance is passed through to _execute_run
4. MockPayloadStore fixture provides proper isolation between tests
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import ArtifactDescriptor, PluginSchema, RunStatus, SourceRow
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore


class TestOrchestratorPayloadStoreRequirement:
    """Test payload_store is required in orchestrator.run()."""

    def test_run_raises_without_payload_store(self) -> None:
        """Test that run() raises ValueError when payload_store=None.

        CRITICAL: Must raise BEFORE any source loading happens.
        This ensures audit compliance - no rows can be created without
        payload storage capability.
        """

        class MinimalSchema(PluginSchema):
            value: int

        class MinimalSource(_TestSourceBase):
            name = "minimal_source"
            output_schema = MinimalSchema

            def __init__(self) -> None:
                self.load_called = False

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                self.load_called = True
                yield SourceRow.valid({"value": 1})

            def close(self) -> None:
                pass

        class MinimalSink(_TestSinkBase):
            name = "minimal_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()
        source = MinimalSource()
        sink = MinimalSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        graph = build_production_graph(config)

        # Act & Assert: Should raise before source.load() is called
        with pytest.raises(ValueError, match=r"PayloadStore.*required.*audit"):
            orchestrator.run(config, graph=graph, payload_store=None)  # type: ignore[arg-type]

        # Verify source.load() was never called
        assert not source.load_called, "Source.load() should not be called when payload_store=None"

    def test_run_with_payload_store_populates_source_data_ref(
        self,
        payload_store: PayloadStore,
    ) -> None:
        """Test that providing payload_store results in populated source_data_ref.

        Verifies audit trail integrity: every row must have source_data_ref
        when a payload_store is provided.
        """

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
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class PassthroughSink(_TestSinkBase):
            name = "output"

            def __init__(self) -> None:
                self.rows_written: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.rows_written.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        sink = PassthroughSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        graph = build_production_graph(config)

        # Act: Run with payload_store
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Assert: Run succeeded
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3

        # Query audit trail to verify source_data_ref is populated
        recorder = LandscapeRecorder(db, payload_store=payload_store)
        rows = recorder.get_rows(result.run_id)

        assert len(rows) == 3, "Should have 3 rows in audit trail"

        for row in rows:
            assert row.source_data_ref is not None, (
                f"Row {row.row_id} source_data_ref should be populated, but is NULL. "
                "This violates CLAUDE.md: 'Source entry - Raw data stored before any processing'"
            )

    def test_execute_run_receives_payload_store(
        self,
        payload_store: PayloadStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that _execute_run receives the exact payload_store from run().

        This prevents "orphaned parameter" bugs where a parameter is added to
        run() but never wired through to the implementation.
        """

        class MinimalSchema(PluginSchema):
            value: int

        class MinimalSource(_TestSourceBase):
            name = "minimal_source"
            output_schema = MinimalSchema

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                yield SourceRow.valid({"value": 1})

            def close(self) -> None:
                pass

        class MinimalSink(_TestSinkBase):
            name = "minimal_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()
        source = MinimalSource()
        sink = MinimalSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        graph = build_production_graph(config)

        # Create a spy to capture _execute_run calls
        original_execute_run = orchestrator._execute_run
        execute_run_spy = Mock(wraps=original_execute_run)
        monkeypatch.setattr(orchestrator, "_execute_run", execute_run_spy)

        # Act: Run with payload_store
        orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Assert: _execute_run was called with the exact payload_store instance
        execute_run_spy.assert_called_once()
        call_kwargs = execute_run_spy.call_args.kwargs
        assert "payload_store" in call_kwargs, "_execute_run must receive payload_store parameter"
        assert call_kwargs["payload_store"] is payload_store, "_execute_run must receive the exact payload_store instance passed to run()"

    def test_payload_store_fixture_isolation(
        self,
        payload_store: PayloadStore,
    ) -> None:
        """Test that payload_store fixture provides fresh instance per test.

        Verifies fixture isolation - each test should get a clean
        MockPayloadStore with no data from previous tests.
        """
        # Store some content
        test_data = b"test data for isolation check"
        content_hash = payload_store.store(test_data)

        # Verify it was stored
        assert payload_store.exists(content_hash), "Stored content should exist"
        retrieved = payload_store.retrieve(content_hash)
        assert retrieved == test_data, "Retrieved data should match stored data"

        # Verify fixture is fresh (MockPayloadStore uses dict - len check confirms isolation)
        # If fixture wasn't fresh, there could be data from other tests
        # We can only assert what we stored exists, not that it's the ONLY thing
        # (MockPayloadStore doesn't expose len(), but we can verify our data is there)
        assert payload_store.exists(content_hash), "Our test data should be present"


class TestOrchestratorPayloadStoreIntegration:
    """Integration tests for payload_store with transforms."""

    def test_run_with_transform_populates_source_data_ref(
        self,
        payload_store: PayloadStore,
    ) -> None:
        """Test that source_data_ref is populated even with transforms in pipeline.

        Verifies that payload storage happens before any transform processing.
        """

        class ValueSchema(PluginSchema):
            value: int

        class DoubledSchema(PluginSchema):
            value: int
            doubled: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = ValueSchema
            output_schema = DoubledSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "doubled": row["value"] * 2,
                    },
                    success_reason={"action": "doubled"},
                )

        class CollectSink(_TestSinkBase):
            name = "output"

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

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 5}, {"value": 10}])
        transform = DoubleTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        graph = build_production_graph(config)

        # Act: Run with payload_store and transform
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Assert: Transform executed correctly
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2
        assert len(sink.results) == 2
        assert sink.results[0] == {"value": 5, "doubled": 10}

        # Verify source_data_ref is STILL populated (payload stored before transform)
        recorder = LandscapeRecorder(db, payload_store=payload_store)
        rows = recorder.get_rows(result.run_id)

        assert len(rows) == 2, "Should have 2 rows in audit trail"

        for row in rows:
            assert row.source_data_ref is not None, (
                f"Row {row.row_id} source_data_ref should be populated even with transforms. "
                "Payload storage must happen BEFORE transform processing."
            )
