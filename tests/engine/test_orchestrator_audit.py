# tests/engine/test_orchestrator_audit.py
"""Tests for Orchestrator audit trail functionality.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import Determinism, NodeID, NodeStateStatus, NodeType, RoutingMode, RunStatus, SinkName, SourceRow
from elspeth.contracts.audit import NodeStateCompleted
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


class TestOrchestratorAuditTrail:
    """Verify audit trail is recorded correctly."""

    def test_run_records_landscape_entries(self, payload_store) -> None:
        """Verify that run creates proper audit trail."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

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

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Query Landscape to verify audit trail
        from elspeth.core.canonical import stable_hash

        recorder = LandscapeRecorder(db)
        run = recorder.get_run(run_result.run_id)

        assert run is not None
        assert run.status == RunStatus.COMPLETED

        # Verify nodes were registered
        nodes = recorder.get_nodes(run_result.run_id)
        assert len(nodes) == 3  # source, transform, sink

        node_names = [n.plugin_name for n in nodes]
        assert "test_source" in node_names
        assert "identity" in node_names
        assert "test_sink" in node_names

        # P1 Fix: Verify node_states have expected statuses with non-null hashes
        rows = recorder.get_rows(run_result.run_id)
        assert len(rows) == 1  # Single row processed
        row = rows[0]

        # Get tokens for this row
        tokens = recorder.get_tokens(row.row_id)
        assert len(tokens) == 1  # Single token (no forks)
        token = tokens[0]

        # Verify node states have input/output hashes
        node_states = recorder.get_node_states_for_token(token.token_id)
        assert len(node_states) >= 1  # At least one node state (transform)

        for state in node_states:
            # All node states should have input_hash (proves we captured input)
            assert state.input_hash is not None, f"Node state {state.state_id} missing input_hash"
            # Successful states should have output_hash - narrow the type first
            if state.status == NodeStateStatus.COMPLETED:
                assert isinstance(state, NodeStateCompleted)
                assert state.output_hash is not None, f"Completed node state {state.state_id} missing output_hash"

        # Verify token outcomes have correct terminal outcome and sink_name
        token_outcomes = recorder.get_token_outcomes_for_row(run_result.run_id, row.row_id)
        assert len(token_outcomes) == 1  # Single terminal outcome
        outcome = token_outcomes[0]
        assert outcome.is_terminal is True
        assert outcome.sink_name == "default"  # Routed to default sink

        # Verify artifacts have content_hash and correct metadata
        artifacts = recorder.get_artifacts(run_result.run_id)
        assert len(artifacts) >= 1  # At least one artifact from sink
        artifact = artifacts[0]
        assert artifact.content_hash is not None, "Artifact missing content_hash"
        assert artifact.artifact_type is not None, "Artifact missing artifact_type"

        # Verify hash integrity using stable_hash
        # The source data hash should match what we compute
        expected_hash = stable_hash({"value": 42})
        assert row.source_data_hash == expected_hash, f"Source data hash mismatch: expected {expected_hash}, got {row.source_data_hash}"


class TestOrchestratorLandscapeExport:
    """Test landscape export integration."""

    def test_orchestrator_exports_landscape_when_configured(self, plugin_manager, payload_store) -> None:
        """Orchestrator should export audit trail after run completes."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.config import (
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

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

        class CollectSink(_TestSinkBase):
            """Sink that captures written rows."""

            name = "collect"

            def __init__(self) -> None:
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, row: Any, ctx: Any) -> ArtifactDescriptor:
                # Row processing writes batches (lists), export writes single records
                if isinstance(row, list):
                    self.captured_rows.extend(row)
                else:
                    self.captured_rows.append(row)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        # Create in-memory DB
        db = LandscapeDB.in_memory()

        # Create sinks
        output_sink = CollectSink()
        export_sink = CollectSink()

        # Build settings with export enabled
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "audit_export": SinkSettings(plugin="json", options={"path": "audit_export.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                export=LandscapeExportSettings(
                    enabled=True,
                    sink="audit_export",
                    format="json",  # JSON works with mock sinks; CSV requires file path
                ),
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "output": as_sink(output_sink),
                "audit_export": as_sink(export_sink),
            },
        )

        # Build graph from config
        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Run with settings
        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline, graph=graph, settings=settings, payload_store=payload_store)

        # Run should complete
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1

        # Export sink should have received audit records
        assert len(export_sink.captured_rows) > 0
        # Should have at least a "run" record type
        record_types = [r.get("record_type") for r in export_sink.captured_rows]
        assert "run" in record_types, f"Expected 'run' record type, got: {record_types}"

    def test_orchestrator_export_with_signing(self, plugin_manager, payload_store) -> None:
        """Orchestrator should sign records when export.sign is True."""
        import os
        from unittest.mock import patch

        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.config import (
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

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

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, row: Any, ctx: Any) -> ArtifactDescriptor:
                # Row processing writes batches (lists), export writes single records
                if isinstance(row, list):
                    self.captured_rows.extend(row)
                else:
                    self.captured_rows.append(row)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()
        output_sink = CollectSink()
        export_sink = CollectSink()

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "audit_export": SinkSettings(plugin="json", options={"path": "audit_export.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                export=LandscapeExportSettings(
                    enabled=True,
                    sink="audit_export",
                    format="json",  # JSON works with mock sinks; CSV requires file path
                    sign=True,
                ),
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "output": as_sink(output_sink),
                "audit_export": as_sink(export_sink),
            },
        )

        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )
        orchestrator = Orchestrator(db)

        # Set signing key environment variable
        with patch.dict(os.environ, {"ELSPETH_SIGNING_KEY": "test-signing-key-12345"}):
            result = orchestrator.run(pipeline, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        assert len(export_sink.captured_rows) > 0

        # All records should have signatures when signing enabled
        for record in export_sink.captured_rows:
            assert "signature" in record, f"Record missing signature: {record}"

        # Should have a manifest record at the end
        record_types = [r.get("record_type") for r in export_sink.captured_rows]
        assert "manifest" in record_types

    def test_orchestrator_export_requires_signing_key_when_sign_enabled(self, plugin_manager, payload_store) -> None:
        """Should raise error when sign=True but ELSPETH_SIGNING_KEY not set."""
        import os
        from unittest.mock import patch

        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.config import (
            ElspethSettings,
            LandscapeExportSettings,
            LandscapeSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

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

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.captured_rows.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()
        output_sink = CollectSink()
        export_sink = CollectSink()

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "audit_export": SinkSettings(plugin="json", options={"path": "audit_export.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                export=LandscapeExportSettings(
                    enabled=True,
                    sink="audit_export",
                    format="csv",
                    sign=True,
                ),
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "output": as_sink(output_sink),
                "audit_export": as_sink(export_sink),
            },
        )

        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )
        orchestrator = Orchestrator(db)

        # Ensure ELSPETH_SIGNING_KEY is not set
        env_without_key = {k: v for k, v in os.environ.items() if k != "ELSPETH_SIGNING_KEY"}
        with (
            patch.dict(os.environ, env_without_key, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            orchestrator.run(pipeline, graph=graph, settings=settings, payload_store=payload_store)

    def test_orchestrator_no_export_when_disabled(self, plugin_manager, payload_store) -> None:
        """Should not export when export.enabled is False."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.config import (
            ElspethSettings,
            LandscapeSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        class ValueSchema(PluginSchema):
            value: int

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

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.captured_rows: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.captured_rows.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        db = LandscapeDB.in_memory()
        output_sink = CollectSink()
        audit_sink = CollectSink()

        # Export disabled (the default)
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "audit": SinkSettings(plugin="json", options={"path": "audit.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            landscape=LandscapeSettings(
                url="sqlite:///:memory:",
                # export.enabled defaults to False
            ),
        )

        source = ListSource([{"value": 42}])

        pipeline = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "output": as_sink(output_sink),
                "audit": as_sink(audit_sink),
            },
        )

        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )
        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline, graph=graph, settings=settings, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        # Output sink should have the row
        assert len(output_sink.captured_rows) == 1
        # Audit sink should be empty (no export)
        assert len(audit_sink.captured_rows) == 0


class TestOrchestratorConfigRecording:
    """Test that runs record the resolved configuration."""

    def test_run_records_resolved_config(self, payload_store) -> None:
        """Run should record the full resolved configuration in Landscape."""
        import json

        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

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

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
        transform = IdentityTransform()
        sink = CollectSink()

        # Create config WITH resolved configuration dict
        resolved_config = {
            "source": {"plugin": "csv", "options": {"path": "test.csv"}},
            "sinks": {"default": {"plugin": "csv"}},
            "default_sink": "default",
        }

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
            config=resolved_config,  # Pass the resolved config
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Query Landscape to verify config was recorded
        recorder = LandscapeRecorder(db)
        run_record = recorder.get_run(run_result.run_id)

        assert run_record is not None
        # settings_json is stored as a JSON string, parse it
        settings = json.loads(run_record.settings_json)
        assert settings != {}
        assert "source" in settings
        assert settings["source"]["plugin"] == "csv"

    def test_run_with_empty_config_records_empty(self, payload_store) -> None:
        """Run with no config passed should record empty dict (current behavior)."""
        import json

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
        sink = CollectSink()

        # No config passed - should default to empty dict
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
            # config not passed - defaults to {}
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Query Landscape to verify empty config was recorded
        recorder = LandscapeRecorder(db)
        run_record = recorder.get_run(run_result.run_id)

        assert run_record is not None
        # This test documents that empty config is recorded when not provided
        # settings_json is stored as a JSON string
        settings = json.loads(run_record.settings_json)
        assert settings == {}


class TestNodeMetadataFromPlugin:
    """Test that node registration uses actual plugin metadata.

    BUG: All nodes were registered with hardcoded plugin_version="1.0.0"
    instead of reading from the actual plugin class attributes.
    """

    def test_node_metadata_records_plugin_version(self, payload_store) -> None:
        """Node registration should use actual plugin metadata.

        Verifies that the node's plugin_version in Landscape matches
        the plugin class's plugin_version attribute.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "versioned_source"
            output_schema = ValueSchema
            plugin_version = "3.7.2"  # Custom version

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class VersionedTransform(BaseTransform):
            name = "versioned_transform"
            input_schema = ValueSchema
            output_schema = ValueSchema
            plugin_version = "2.5.0"  # Custom version (not 1.0.0)
            determinism = Determinism.EXTERNAL_CALL

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "versioned"})

        class VersionedSink(_TestSinkBase):
            name = "versioned_sink"
            plugin_version = "4.1.0"  # Custom version

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
        transform = VersionedTransform()
        sink = VersionedSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # Build graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="versioned_source", config=schema_config)
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="versioned_transform", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="versioned_sink", config=schema_config)
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {0: NodeID("transform")}
        graph._sink_id_map = {SinkName("default"): NodeID("sink")}
        graph._default_sink = "default"

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Query Landscape to verify node metadata
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)
        assert len(nodes) == 3  # source, transform, sink

        # Create lookup by plugin_name
        nodes_by_name = {n.plugin_name: n for n in nodes}

        # Verify source has correct version
        source_node = nodes_by_name["versioned_source"]
        assert source_node.plugin_version == "3.7.2", f"Source plugin_version should be '3.7.2', got '{source_node.plugin_version}'"

        # Verify transform has correct version
        transform_node = nodes_by_name["versioned_transform"]
        assert transform_node.plugin_version == "2.5.0", (
            f"Transform plugin_version should be '2.5.0', got '{transform_node.plugin_version}'"
        )

        # Verify sink has correct version
        sink_node = nodes_by_name["versioned_sink"]
        assert sink_node.plugin_version == "4.1.0", f"Sink plugin_version should be '4.1.0', got '{sink_node.plugin_version}'"

    def test_node_metadata_records_determinism(self, payload_store) -> None:
        """Node registration should record plugin determinism.

        Verifies that nondeterministic plugins are recorded correctly
        in the Landscape for reproducibility tracking.
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "test_source"
            output_schema = ValueSchema
            plugin_version = "1.0.0"

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class NonDeterministicTransform(BaseTransform):
            name = "nondeterministic_transform"
            input_schema = ValueSchema
            output_schema = ValueSchema
            plugin_version = "1.0.0"
            determinism = Determinism.EXTERNAL_CALL  # Explicit nondeterministic

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "nondeterministic"})

        class CollectSink(_TestSinkBase):
            name = "test_sink"
            plugin_version = "1.0.0"

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
        transform = NonDeterministicTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # Build graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="nondeterministic_transform", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink", config=schema_config)
        graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._transform_id_map = {0: NodeID("transform")}
        graph._sink_id_map = {SinkName("default"): NodeID("sink")}
        graph._default_sink = "default"

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Query Landscape to verify determinism recorded
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)

        # Find the transform node
        transform_node = next(n for n in nodes if n.plugin_name == "nondeterministic_transform")

        # Verify determinism is recorded correctly
        assert transform_node.determinism == Determinism.EXTERNAL_CALL, (
            f"Transform determinism should be 'external_call', got '{transform_node.determinism}'"
        )

    def test_aggregation_node_uses_transform_metadata(self, payload_store) -> None:
        """Aggregation nodes should use metadata from their batch-aware transform.

        BUG FIX: P2-2026-01-21-orchestrator-aggregation-metadata-hardcoded
        Previously, aggregation nodes were always registered with hardcoded
        plugin_version="1.0.0" and determinism=DETERMINISTIC, even when
        the underlying transform was non-deterministic (e.g., LLM batch).

        This test verifies the fix: aggregation nodes now correctly inherit
        metadata from their transform plugin instance.
        """
        from elspeth.contracts import AggregationName, ArtifactDescriptor, PluginSchema
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

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

        class NonDeterministicBatchTransform(BaseTransform):
            """Simulates an LLM batch transform - explicitly non-deterministic."""

            name = "nondeterministic_batch"
            input_schema = ValueSchema
            output_schema = ValueSchema
            plugin_version = "2.3.4"  # Custom version - NOT 1.0.0
            determinism = Determinism.NON_DETERMINISTIC  # LLM-like

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(
                self,
                row: dict[str, Any] | list[dict[str, Any]],
                ctx: Any,
            ) -> TransformResult:
                # Batch-aware: handles list or single row
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    return TransformResult.success({"value": total}, success_reason={"action": "batch_aggregated"})
                return TransformResult.success(row, success_reason={"action": "passthrough"})

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                if isinstance(rows, list):
                    self.results.extend(rows)
                else:
                    self.results.append(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 20}])
        batch_transform = NonDeterministicBatchTransform()
        sink = CollectSink()

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="nondeterministic_batch",
            trigger=TriggerConfig(count=2),  # Trigger after 2 rows
            output_mode="transform",
            options={"schema": {"fields": "dynamic"}},
        )

        # Build graph with aggregation
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[],  # No regular transforms, only aggregation
            sinks={"default": as_sink(sink)},
            aggregations={"test_agg": (batch_transform, agg_settings)},
            gates=[],
            default_sink="default",
        )

        # Get aggregation node ID for later lookup
        agg_id_map = graph.get_aggregation_id_map()
        agg_node_id = agg_id_map[AggregationName("test_agg")]

        # Set node_id on transform (as CLI would do)
        batch_transform.node_id = agg_node_id

        # Build aggregation_settings dict for orchestrator
        aggregation_settings = {agg_node_id: agg_settings}

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(batch_transform)],  # Aggregation transform in list
            sinks={"default": as_sink(sink)},
            aggregation_settings=aggregation_settings,
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Query Landscape to verify aggregation node metadata
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)

        # Find the aggregation node
        agg_node = next((n for n in nodes if n.node_id == agg_node_id), None)
        assert agg_node is not None, f"Aggregation node {agg_node_id} not found in Landscape"

        # CRITICAL: Verify aggregation uses transform's metadata, NOT hardcoded values
        assert agg_node.plugin_version == "2.3.4", (
            f"Aggregation plugin_version should be '2.3.4' from transform, got '{agg_node.plugin_version}' (was hardcoded to '1.0.0')"
        )
        assert agg_node.determinism == Determinism.NON_DETERMINISTIC, (
            f"Aggregation determinism should be 'non_deterministic' from transform, "
            f"got '{agg_node.determinism}' (was hardcoded to 'deterministic')"
        )

    def test_config_gate_node_uses_engine_version(self, payload_store) -> None:
        """Config gate nodes should use engine version, not hardcoded '1.0.0'.

        BUG FIX: P2-2026-01-15-node-metadata-hardcoded
        Config gates are engine-internal nodes (not plugins) that evaluate
        expressions. They should use 'engine:{VERSION}' format to indicate
        they're engine components, not user plugins.
        """
        from elspeth import __version__ as ENGINE_VERSION
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.core.config import GateSettings
        from elspeth.core.dag import ExecutionGraph
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

        class CollectSink(_TestSinkBase):
            name = "test_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                if isinstance(rows, list):
                    self.results.extend(rows)
                else:
                    self.results.append(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # Create a config gate (expression-based, not plugin-based)
        # Use literal "True" - the expression parser allows boolean literals
        config_gate = GateSettings(
            name="value_check",
            condition="True",
            routes={"true": "continue", "false": "continue"},
        )

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        # Build graph with config gate
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
            aggregations={},
            gates=[config_gate],
            default_sink="default",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
            gates=[config_gate],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Query Landscape to verify config gate node metadata
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)

        # Find the config gate node (named "config_gate:value_check")
        config_gate_node = next(
            (n for n in nodes if n.plugin_name.startswith("config_gate:")),
            None,
        )
        assert config_gate_node is not None, "Config gate node not found in Landscape"

        # CRITICAL: Verify config gate uses engine version, NOT hardcoded "1.0.0"
        expected_version = f"engine:{ENGINE_VERSION}"
        assert config_gate_node.plugin_version == expected_version, (
            f"Config gate plugin_version should be '{expected_version}', got '{config_gate_node.plugin_version}' (was hardcoded to '1.0.0')"
        )

        # Config gates are deterministic (expression evaluation is pure)
        assert config_gate_node.determinism == Determinism.DETERMINISTIC, (
            f"Config gate determinism should be DETERMINISTIC, got '{config_gate_node.determinism}'"
        )

    def test_coalesce_node_uses_engine_version(self, payload_store) -> None:
        """Coalesce nodes should use engine version, not hardcoded '1.0.0'.

        BUG FIX: P2-2026-01-15-node-metadata-hardcoded (related)
        Coalesce nodes merge tokens from parallel fork paths. Like config gates,
        they're engine-internal and should use 'engine:{VERSION}' format.
        """
        from elspeth import __version__ as ENGINE_VERSION
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Use settings-based approach which properly wires fork/coalesce
        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),  # No file access needed
            transforms=[
                TransformSettings(
                    plugin="passthrough",
                    name="path_a",
                    options={"schema": {"fields": "dynamic"}},
                ),
                TransformSettings(
                    plugin="passthrough",
                    name="path_b",
                    options={"schema": {"fields": "dynamic"}},
                ),
            ],
            sinks={
                "default": SinkSettings(
                    plugin="json",
                    options={"path": "/tmp/test_coalesce_audit.json", "schema": {"fields": "dynamic"}},
                ),
            },
            default_sink="default",
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_paths",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Query Landscape to verify coalesce node metadata
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_result.run_id)

        # Find the coalesce node (named "coalesce:merge_paths")
        coalesce_node = next(
            (n for n in nodes if n.plugin_name.startswith("coalesce:")),
            None,
        )
        assert coalesce_node is not None, "Coalesce node not found in Landscape"

        # CRITICAL: Verify coalesce uses engine version, NOT hardcoded "1.0.0"
        expected_version = f"engine:{ENGINE_VERSION}"
        assert coalesce_node.plugin_version == expected_version, (
            f"Coalesce plugin_version should be '{expected_version}', got '{coalesce_node.plugin_version}' (was hardcoded to '1.0.0')"
        )

        # Coalesce is deterministic (merging is a pure operation)
        assert coalesce_node.determinism == Determinism.DETERMINISTIC, (
            f"Coalesce determinism should be DETERMINISTIC, got '{coalesce_node.determinism}'"
        )
