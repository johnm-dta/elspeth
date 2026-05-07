# tests/fixtures/test_factories.py
"""Smoke tests for test fixture factories.

Hard gate: these must pass before any P1 migration PR.
Tests verify factory contracts so that ~900 callsites can trust them.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts import NodeType

# =============================================================================
# make_context() — PluginContext factory
# =============================================================================


class TestMakeContext:
    """Verify make_context() wires parameters correctly."""

    def test_default_returns_valid_context(self) -> None:
        from tests.fixtures.factories import make_context

        ctx = make_context()
        assert ctx.run_id == "test-run"
        assert ctx.state_id == "state-123"
        assert ctx.config == {}
        assert ctx.node_id is None

    def test_node_id_passes_through(self) -> None:
        from tests.fixtures.factories import make_context

        ctx = make_context(node_id="source-node-42")
        assert ctx.node_id == "source-node-42"

    def test_explicit_run_id(self) -> None:
        from tests.fixtures.factories import make_context

        ctx = make_context(run_id="my-custom-run")
        assert ctx.run_id == "my-custom-run"

    def test_explicit_state_id(self) -> None:
        from tests.fixtures.factories import make_context

        ctx = make_context(state_id="state-retry-3")
        assert ctx.state_id == "state-retry-3"

    def test_explicit_config(self) -> None:
        from tests.fixtures.factories import make_context

        ctx = make_context(config={"key": "value"})
        assert ctx.config == {"key": "value"}

    def test_mock_landscape_is_specced(self) -> None:
        """Default mock landscape must be specced to PluginAuditWriter."""
        from unittest.mock import Mock

        from elspeth.contracts.audit_protocols import PluginAuditWriter
        from tests.fixtures.factories import make_context

        ctx = make_context()
        assert ctx.landscape is not None
        assert isinstance(ctx.landscape, Mock)
        # spec= ensures only real PluginAuditWriter methods are accessible
        assert isinstance(ctx.landscape, PluginAuditWriter)
        # get_node_state must return a mock with token_id matching the token
        node_state = ctx.landscape.get_node_state("any-state-id")
        assert node_state is not None
        assert ctx.token is not None
        assert node_state.token_id == ctx.token.token_id

    def test_explicit_landscape_replaces_mock(self) -> None:
        from unittest.mock import Mock

        from tests.fixtures.factories import make_context

        custom_landscape = Mock()
        ctx = make_context(landscape=custom_landscape)
        assert ctx.landscape is custom_landscape


# =============================================================================
# make_source_context() — PluginContext with real landscape
# =============================================================================


class TestMakeSourceContext:
    """Verify make_source_context() creates a real landscape chain."""

    def test_default_creates_valid_context(self) -> None:
        from tests.fixtures.factories import make_source_context

        ctx = make_source_context()
        assert ctx.run_id == "test-run"
        assert ctx.node_id == "source"

    def test_custom_plugin_name(self) -> None:
        from tests.fixtures.factories import make_source_context

        ctx = make_source_context(plugin_name="json")
        # Should not raise — plugin registered successfully
        assert ctx.run_id == "test-run"

    def test_landscape_is_real_adapter(self) -> None:
        """Verify delegated factory produces a real PluginAuditWriter, not a Mock."""
        from elspeth.contracts.audit_protocols import PluginAuditWriter
        from tests.fixtures.factories import make_source_context

        ctx = make_source_context()
        assert isinstance(ctx.landscape, PluginAuditWriter)

    def test_create_row_round_trip(self) -> None:
        """Prove FK chain is intact: run -> node -> row."""
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(
            run_id="test-run",
            source_node_id="source",
            source_plugin_name="csv",
        )
        row = setup.data_flow.create_row(
            run_id=setup.run_id,
            source_node_id=setup.source_node_id,
            row_index=0,
            data={"field": "value"},
        )
        assert row.row_id is not None


# =============================================================================
# make_operation_context() — PluginContext with operation records
# =============================================================================


class TestMakeOperationContext:
    """Verify make_operation_context() creates full FK chain."""

    def test_default_creates_operation(self) -> None:
        from tests.fixtures.factories import make_operation_context

        ctx = make_operation_context()
        assert ctx.run_id == "test-run"
        assert ctx.node_id == "source"
        assert ctx.operation_id is not None

    def test_sink_context(self) -> None:
        from tests.fixtures.factories import make_operation_context

        ctx = make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
        )
        assert ctx.node_id == "sink"
        assert ctx.operation_id is not None

    def test_record_call_round_trip(self) -> None:
        """Prove FK chain is intact for SOURCE: run -> node -> operation -> call."""
        from elspeth.contracts.enums import CallStatus, CallType
        from tests.fixtures.factories import make_operation_context

        ctx = make_operation_context()
        # Must not raise — proves operation_id FK chain is valid
        call = ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "https://example.com"},
            response_data={"status": 200},
            latency_ms=50.0,
        )
        assert call is not None

    def test_sink_record_call_round_trip(self) -> None:
        """Prove FK chain is intact for SINK path via register_test_node()."""
        from elspeth.contracts.enums import CallStatus, CallType
        from tests.fixtures.factories import make_operation_context

        ctx = make_operation_context(
            operation_type="sink_write",
            node_id="sink",
            node_type="SINK",
        )
        # Must not raise — proves the SINK node's operation_id FK chain is valid
        call = ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "https://example.com"},
            response_data={"status": 200},
            latency_ms=50.0,
        )
        assert call is not None


# =============================================================================
# RecorderSetup + make_recorder_with_run()
# =============================================================================


class TestMakeRecorderWithRun:
    """Verify make_recorder_with_run() creates complete audit infrastructure."""

    def test_default_returns_valid_setup(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run()
        assert setup.run_id is not None
        assert setup.source_node_id is not None
        assert setup.db is not None
        assert setup.factory is not None

    def test_explicit_run_id(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(run_id="deterministic-run-42")
        assert setup.run_id == "deterministic-run-42"

    def test_explicit_source_node_id(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(source_node_id="my-source-node")
        assert setup.source_node_id == "my-source-node"

    def test_source_node_exists_in_db(self) -> None:
        """Source node must be queryable via data_flow repository."""
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(
            run_id="db-check-run",
            source_node_id="db-check-source",
        )
        # Query node via data_flow repository.
        node = setup.data_flow.get_node(setup.source_node_id, setup.run_id)
        assert node is not None
        assert node.node_id == "db-check-source"
        assert node.plugin_name == "source"

    def test_custom_canonical_version(self) -> None:
        """Tests like test_processor.py use 'sha256-rfc8785-v1'."""
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(canonical_version="sha256-rfc8785-v1")
        # Query run via run_lifecycle repository
        run = setup.run_lifecycle.get_run(setup.run_id)
        assert run is not None
        assert run.canonical_version == "sha256-rfc8785-v1"

    def test_custom_source_plugin_name(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(
            source_plugin_name="csv_source",
            source_node_id="csv-node",
        )
        node = setup.data_flow.get_node(setup.source_node_id, setup.run_id)
        assert node is not None
        assert node.plugin_name == "csv_source"

    def test_each_call_creates_fresh_db(self) -> None:
        """Test isolation: two calls must produce independent databases."""
        from tests.fixtures.landscape import make_recorder_with_run

        setup1 = make_recorder_with_run(run_id="run-1")
        setup2 = make_recorder_with_run(run_id="run-2")

        assert setup1.db is not setup2.db
        assert setup1.run_id != setup2.run_id

        # run-1 must not appear in db2's run_lifecycle
        run_in_db2 = setup2.run_lifecycle.get_run("run-1")
        assert run_in_db2 is None

    def test_register_additional_node_succeeds(self) -> None:
        """RecorderSetup.data_flow supports adding more nodes."""
        from elspeth.contracts.schema import SchemaConfig
        from tests.fixtures.landscape import make_recorder_with_run

        setup = make_recorder_with_run(run_id="multi-node-run")
        # register_node must succeed on the data_flow repository
        node = setup.data_flow.register_node(
            run_id=setup.run_id,
            plugin_name="enricher",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="enricher-node",
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        assert node.node_id == "enricher-node"


# =============================================================================
# register_test_node()
# =============================================================================


class TestRegisterTestNode:
    """Verify register_test_node() adds nodes with correct defaults."""

    def test_default_creates_transform_node(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run, register_test_node

        setup = make_recorder_with_run(run_id="reg-test-run")
        node_id = register_test_node(setup.data_flow, setup.run_id, "transform-1")
        assert node_id == "transform-1"

        node = setup.data_flow.get_node(node_id, setup.run_id)
        assert node is not None
        assert node.plugin_name == "transform"

    def test_custom_node_type_and_plugin(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run, register_test_node

        setup = make_recorder_with_run(run_id="custom-node-run")
        node_id = register_test_node(
            setup.data_flow,
            setup.run_id,
            "my-sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
        )
        assert node_id == "my-sink"
        node = setup.data_flow.get_node(node_id, setup.run_id)
        assert node is not None
        assert node.plugin_name == "csv_sink"

    def test_multiple_nodes_in_same_run(self) -> None:
        from tests.fixtures.landscape import make_recorder_with_run, register_test_node

        setup = make_recorder_with_run(run_id="multi-run")
        register_test_node(setup.data_flow, setup.run_id, "t1")
        register_test_node(setup.data_flow, setup.run_id, "t2")
        register_test_node(
            setup.data_flow,
            setup.run_id,
            "sink-1",
            node_type=NodeType.SINK,
            plugin_name="output",
        )

        # All three must be queryable via data_flow repository
        assert setup.data_flow.get_node("t1", setup.run_id) is not None
        assert setup.data_flow.get_node("t2", setup.run_id) is not None
        assert setup.data_flow.get_node("sink-1", setup.run_id) is not None


# =============================================================================
# FailingSink
# =============================================================================


class TestFailingSink:
    """Verify FailingSink raises on write()."""

    def test_write_raises_runtime_error(self) -> None:
        from tests.fixtures.plugins import FailingSink

        sink = FailingSink()
        with pytest.raises(RuntimeError, match="Sink write failed"):
            sink.write([], None)

    def test_custom_error_message(self) -> None:
        from tests.fixtures.plugins import FailingSink

        sink = FailingSink(error_message="Custom kaboom")
        with pytest.raises(RuntimeError, match="Custom kaboom"):
            sink.write([], None)

    def test_default_name(self) -> None:
        from tests.fixtures.plugins import FailingSink

        sink = FailingSink()
        assert sink.name == "failing_sink"

    def test_custom_name(self) -> None:
        from tests.fixtures.plugins import FailingSink

        sink = FailingSink("my_broken_sink")
        assert sink.name == "my_broken_sink"


# =============================================================================
# FailingSource
# =============================================================================


class TestFailingSource:
    """Verify FailingSource raises on load()."""

    def test_load_raises_runtime_error(self) -> None:
        from tests.fixtures.plugins import FailingSource

        source = FailingSource()
        with pytest.raises(RuntimeError, match="Source failed intentionally"):
            list(source.load(None))

    def test_custom_error_message(self) -> None:
        from tests.fixtures.plugins import FailingSource

        source = FailingSource(error_message="Custom load failure")
        with pytest.raises(RuntimeError, match="Custom load failure"):
            list(source.load(None))

    def test_default_name(self) -> None:
        from tests.fixtures.plugins import FailingSource

        source = FailingSource()
        assert source.name == "failing_source"


# =============================================================================
# run_audit_pipeline()
# =============================================================================


class TestRunAuditPipeline:
    """Verify run_audit_pipeline() executes a full pipeline with audit trail."""

    def test_simple_passthrough(self, tmp_path: Any) -> None:
        from tests.fixtures.pipeline import run_audit_pipeline

        source_data = [{"value": 1}, {"value": 2}, {"value": 3}]
        result = run_audit_pipeline(tmp_path, source_data)

        assert result.run_id is not None
        assert len(result.sink.results) == 3
        assert result.db is not None
        assert result.payload_store is not None

    def test_run_exists_in_db(self, tmp_path: Any) -> None:
        """Run must be queryable via a factory built from the returned DB."""
        from elspeth.core.landscape.factory import RecorderFactory
        from tests.fixtures.pipeline import run_audit_pipeline

        result = run_audit_pipeline(tmp_path, [{"x": 1}])
        # result.db is a LandscapeDB (connection manager), not a factory.
        # Build a factory to query it.
        factory = RecorderFactory(result.db)
        run = factory.run_lifecycle.get_run(result.run_id)
        assert run is not None

    def test_single_row(self, tmp_path: Any) -> None:
        from tests.fixtures.pipeline import run_audit_pipeline

        result = run_audit_pipeline(tmp_path, [{"only": "row"}])
        assert len(result.sink.results) == 1

    def test_audit_pipeline_result_fields(self, tmp_path: Any) -> None:
        """AuditPipelineResult must expose all documented fields."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.payload_store import FilesystemPayloadStore
        from tests.fixtures.pipeline import run_audit_pipeline
        from tests.fixtures.plugins import CollectSink

        result = run_audit_pipeline(tmp_path, [{"v": 1}])
        assert isinstance(result.run_id, str)
        assert isinstance(result.db, LandscapeDB)
        assert isinstance(result.payload_store, FilesystemPayloadStore)
        assert isinstance(result.sink, CollectSink)
