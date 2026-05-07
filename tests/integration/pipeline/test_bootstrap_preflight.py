# tests/integration/pipeline/test_bootstrap_preflight.py
"""Integration tests for bootstrap_and_run() dependency and gate dispatch.

These tests exercise the full CLI→bootstrap→orchestrator wiring path
with extensive patching. Moved from tests/unit/engine/ because they
test integration of multiple subsystems, not isolated unit logic.

The pure resolve_preflight() unit tests remain in
tests/unit/engine/test_bootstrap_preflight.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.cli_helpers import PluginBundle
from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.core.dependency_config import (
    CommencementGateConfig,
    DependencyRunResult,
)


def _make_bootstrap_config() -> MagicMock:
    """Create the minimal config surface bootstrap_and_run reads in these tests."""
    mock_config = MagicMock()
    mock_config.depends_on = []
    mock_config.commencement_gates = []
    mock_config.collection_probes = []
    mock_config.gates = []
    mock_config.coalesce = []
    mock_config.landscape.export.enabled = False
    mock_config.landscape.dump_to_jsonl = False
    mock_config.landscape.dump_to_jsonl_path = None
    mock_config.landscape.dump_to_jsonl_fail_on_error = False
    mock_config.landscape.dump_to_jsonl_include_payloads = False
    mock_config.landscape.dump_to_jsonl_payload_base_path = None
    mock_config.landscape.url = "sqlite:///audit.db"
    mock_config.payload_store.backend = "filesystem"
    mock_config.payload_store.base_path = Path(".elspeth/payloads")
    return mock_config


class TestBootstrapDependencyDispatch:
    """Test that bootstrap_and_run() calls dependency resolution when configured."""

    def test_skips_dependencies_when_not_configured(self) -> None:
        """When depends_on is empty, dependency resolver is never called."""
        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = []
        mock_config.collection_probes = []
        mock_config.gates = []
        mock_config.coalesce = []
        mock_config.landscape.export.enabled = False
        mock_config.payload_store.backend = "filesystem"

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.dependency_resolver.detect_cycles") as mock_detect,
            patch("elspeth.engine.dependency_resolver.resolve_dependencies") as mock_resolve,
        ):
            mock_plugins.return_value = MagicMock(spec=PluginBundle)
            mock_graph = MagicMock()
            mock_graph_cls.from_plugin_instances.return_value = mock_graph

            mock_run_result = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.orchestrator.run.return_value = mock_run_result
            mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            from elspeth.cli_helpers import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        mock_detect.assert_not_called()
        mock_resolve.assert_not_called()


class TestBootstrapProgrammaticExecution:
    """Regression coverage for bootstrap_and_run() programmatic boundaries."""

    def test_successful_run_propagates_db_close_failure(self) -> None:
        """close() failure after successful execution is not a false success."""
        mock_config = _make_bootstrap_config()
        mock_db = MagicMock()
        mock_db.close.side_effect = RuntimeError("close failed")

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config", return_value=MagicMock(spec=PluginBundle)),
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()
            mock_db_cls.from_url.return_value = mock_db
            mock_ctx = MagicMock()
            mock_ctx.orchestrator.run.return_value = MagicMock()
            mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            from elspeth.cli_helpers import bootstrap_and_run

            with pytest.raises(RuntimeError, match="close failed"):
                bootstrap_and_run(Path("/fake/pipeline.yaml"))

    def test_pipeline_failure_is_not_masked_by_db_close_failure(self) -> None:
        """close() failure during exception cleanup preserves the pipeline error."""
        mock_config = _make_bootstrap_config()
        mock_db = MagicMock()
        mock_db.close.side_effect = RuntimeError("close failed")

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config", return_value=MagicMock(spec=PluginBundle)),
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()
            mock_db_cls.from_url.return_value = mock_db
            mock_ctx = MagicMock()
            mock_ctx.orchestrator.run.side_effect = ValueError("pipeline failed")
            mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            from elspeth.cli_helpers import bootstrap_and_run

            with pytest.raises(ValueError, match="pipeline failed"):
                bootstrap_and_run(Path("/fake/pipeline.yaml"))

    def test_dependency_runner_requests_silent_orchestrator_output(self) -> None:
        """bootstrap_and_run is programmatic and must not attach CLI formatters."""
        mock_config = _make_bootstrap_config()
        mock_db = MagicMock()

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config", return_value=MagicMock(spec=PluginBundle)),
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()
            mock_db_cls.from_url.return_value = mock_db
            mock_ctx = MagicMock()
            mock_ctx.orchestrator.run.return_value = MagicMock()
            mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            from elspeth.cli_helpers import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        assert mock_orch_ctx.call_args is not None
        assert mock_orch_ctx.call_args.kwargs["output_format"] == "none"

    def test_orchestrator_context_none_output_uses_null_event_bus(self) -> None:
        """The silent output mode uses NullEventBus, not console/json formatters."""
        from elspeth.cli import _orchestrator_context
        from elspeth.core.events import NullEventBus

        mock_config = _make_bootstrap_config()
        mock_config.checkpoint.enabled = False
        mock_graph = MagicMock()
        mock_graph.get_aggregation_id_map.return_value = {}
        mock_plugins = MagicMock(spec=PluginBundle)
        mock_plugins.transforms = []
        mock_plugins.aggregations = {}
        mock_plugins.source = MagicMock()
        mock_plugins.sinks = {"out": MagicMock()}
        mock_db = MagicMock()

        with (
            patch("elspeth.contracts.config.runtime.RuntimeRateLimitConfig.from_settings", return_value=MagicMock()),
            patch("elspeth.contracts.config.runtime.RuntimeConcurrencyConfig.from_settings", return_value=MagicMock()),
            patch("elspeth.contracts.config.runtime.RuntimeCheckpointConfig.from_settings", return_value=MagicMock(enabled=False)),
            patch("elspeth.contracts.config.runtime.RuntimeTelemetryConfig.from_settings", return_value=MagicMock()),
            patch("elspeth.core.rate_limit.RateLimitRegistry") as mock_rate_limit_registry,
            patch("elspeth.telemetry.create_telemetry_manager", return_value=MagicMock()),
            patch("elspeth.engine.Orchestrator") as mock_orchestrator,
        ):
            mock_rate_limit_registry.return_value = MagicMock()
            with _orchestrator_context(mock_config, mock_graph, mock_plugins, db=mock_db, output_format="none"):
                pass

        assert mock_orchestrator.call_args is not None
        assert isinstance(mock_orchestrator.call_args.kwargs["event_bus"], NullEventBus)


class TestBootstrapDependencyResultsFlow:
    """Test that dependency results flow into gate context (regression for bug #1)."""

    def test_dependency_results_passed_to_gate_context(self) -> None:
        """Dependency run results must be visible in commencement gate expressions."""
        from elspeth.core.dependency_config import DependencyConfig

        mock_config = MagicMock()
        mock_config.depends_on = [DependencyConfig(name="indexer", settings="./index.yaml")]
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]
        mock_config.collection_probes = []
        mock_config.gates = []
        mock_config.coalesce = []
        mock_config.landscape.export.enabled = False
        mock_config.payload_store.backend = "filesystem"

        dep_result = DependencyRunResult(
            name="indexer",
            run_id="dep-run-abc",
            settings_hash="sha256:abc",
            duration_ms=1000,
            indexed_at="2026-03-25T12:00:00Z",
        )

        captured_context = {}

        def capture_gate_context(gates: object, context: dict[str, Any]) -> list[Any]:
            captured_context.update(context)
            return []

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config", return_value=MagicMock(spec=PluginBundle)),
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.dependency_resolver.detect_cycles"),
            patch("elspeth.engine.dependency_resolver.resolve_dependencies", return_value=[dep_result]),
            patch("elspeth.engine.commencement.evaluate_commencement_gates", side_effect=capture_gate_context),
        ):
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.orchestrator.run.return_value = MagicMock()
            mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            from elspeth.cli_helpers import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        # Dependency results must be in the gate context — NOT empty
        assert "dependency_runs" in captured_context
        assert "indexer" in captured_context["dependency_runs"]
        assert captured_context["dependency_runs"]["indexer"]["run_id"] == "dep-run-abc"


class TestBootstrapCommencementGateDispatch:
    """Test that bootstrap_and_run() evaluates gates when configured."""

    def test_skips_gates_when_not_configured(self) -> None:
        """When commencement_gates is empty, gate evaluator is never called."""
        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = []
        mock_config.collection_probes = []
        mock_config.gates = []
        mock_config.coalesce = []
        mock_config.landscape.export.enabled = False
        mock_config.payload_store.backend = "filesystem"

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.commencement.evaluate_commencement_gates") as mock_eval_gates,
        ):
            mock_plugins.return_value = MagicMock(spec=PluginBundle)
            mock_graph = MagicMock()
            mock_graph_cls.from_plugin_instances.return_value = mock_graph

            mock_run_result = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.orchestrator.run.return_value = mock_run_result
            mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
            mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            from elspeth.cli_helpers import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        mock_eval_gates.assert_not_called()

    def test_gate_failure_propagates(self) -> None:
        """When a gate fails, CommencementGateFailedError propagates through bootstrap."""
        mock_config = MagicMock()
        mock_config.depends_on = []
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]  # Non-empty triggers gate eval
        mock_config.collection_probes = []
        mock_config.gates = []
        mock_config.coalesce = []
        mock_config.landscape.export.enabled = False

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.cli_helpers.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.core.dag.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch(
                "elspeth.engine.commencement.evaluate_commencement_gates",
                side_effect=CommencementGateFailedError(
                    gate_name="test_gate",
                    condition="False",
                    reason="test failure",
                    context_snapshot={},
                ),
            ),
        ):
            mock_plugins.return_value = MagicMock(spec=PluginBundle)
            mock_graph = MagicMock()
            mock_graph_cls.from_plugin_instances.return_value = mock_graph

            from elspeth.cli_helpers import bootstrap_and_run

            with pytest.raises(CommencementGateFailedError, match="test_gate"):
                bootstrap_and_run(Path("/fake/pipeline.yaml"))
