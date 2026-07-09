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
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.contracts.preflight import DependencyRunResult
from elspeth.core.dependency_config import (
    CommencementGateConfig,
)
from elspeth.plugins.infrastructure.runtime_factory import PluginBundle


class _GraphStub:
    """Minimal validated graph surface consumed by bootstrap/context wiring tests."""

    def validate(self) -> None:
        return None

    def get_aggregation_id_map(self) -> dict[str, str]:
        return {}


class _CloseableDouble:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class _OrchestratorDouble:
    def __init__(self, *, run_result: object | None = None, run_error: BaseException | None = None) -> None:
        self.run_result = run_result if run_result is not None else object()
        self.run_error = run_error

    def run(self, *args: Any, **kwargs: Any) -> object:
        del args, kwargs
        if self.run_error is not None:
            raise self.run_error
        return self.run_result


class _OrchestratorContextManagerDouble:
    def __init__(self, *, run_result: object | None = None, run_error: BaseException | None = None) -> None:
        self.ctx = SimpleNamespace(
            pipeline_config=object(),
            orchestrator=_OrchestratorDouble(run_result=run_result, run_error=run_error),
        )

    def __enter__(self) -> SimpleNamespace:
        return self.ctx

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False


def _make_plugin_bundle() -> PluginBundle:
    source = object()
    return PluginBundle(
        sources={"source": source},
        source_settings_map={"source": object()},
        transforms=[],
        sinks={},
        aggregations={},
    )


def _make_bootstrap_config() -> SimpleNamespace:
    """Create the minimal config surface bootstrap_and_run reads in these tests."""

    def _model_dump(*, mode: str = "python") -> dict[str, Any]:
        del mode
        return {}

    return SimpleNamespace(
        depends_on=[],
        commencement_gates=[],
        collection_probes=[],
        gates=[],
        coalesce=[],
        queues={},
        checkpoint=SimpleNamespace(enabled=True),
        rate_limit=SimpleNamespace(),
        concurrency=SimpleNamespace(),
        telemetry=SimpleNamespace(),
        landscape=SimpleNamespace(
            backend="sqlite",
            export=SimpleNamespace(enabled=False),
            dump_to_jsonl=False,
            dump_to_jsonl_path=None,
            dump_to_jsonl_fail_on_error=False,
            dump_to_jsonl_include_payloads=False,
            dump_to_jsonl_payload_base_path=None,
            url="sqlite:///audit.db",
        ),
        payload_store=SimpleNamespace(
            backend="filesystem",
            base_path=Path(".elspeth/payloads"),
        ),
        model_dump=_model_dump,
    )


class TestBootstrapDependencyDispatch:
    """Test that bootstrap_and_run() calls dependency resolution when configured."""

    def test_skips_dependencies_when_not_configured(self) -> None:
        """When depends_on is empty, dependency resolver is never called."""
        mock_config = _make_bootstrap_config()

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.dependency_resolver.detect_cycles") as mock_detect,
            patch("elspeth.engine.dependency_resolver.resolve_dependencies") as mock_resolve,
        ):
            mock_plugins.return_value = _make_plugin_bundle()
            mock_graph = _GraphStub()
            mock_graph_cls.from_plugin_instances.return_value = mock_graph

            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble()

            from elspeth.cli import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        mock_detect.assert_not_called()
        mock_resolve.assert_not_called()


class TestBootstrapProgrammaticExecution:
    """Regression coverage for bootstrap_and_run() programmatic boundaries."""

    def test_successful_run_propagates_db_close_failure(self) -> None:
        """close() failure after successful execution is not a false success."""
        mock_config = _make_bootstrap_config()
        mock_db = _CloseableDouble(close_error=RuntimeError("close failed"))

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch(
                "elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config",
                return_value=_make_plugin_bundle(),
            ),
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = _GraphStub()
            mock_db_cls.from_url.return_value = mock_db
            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble()

            from elspeth.cli import bootstrap_and_run

            with pytest.raises(RuntimeError, match="close failed"):
                bootstrap_and_run(Path("/fake/pipeline.yaml"))

    def test_pipeline_failure_is_not_masked_by_db_close_failure(self) -> None:
        """close() failure during exception cleanup preserves the pipeline error."""
        mock_config = _make_bootstrap_config()
        mock_db = _CloseableDouble(close_error=RuntimeError("close failed"))

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch(
                "elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config",
                return_value=_make_plugin_bundle(),
            ),
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = _GraphStub()
            mock_db_cls.from_url.return_value = mock_db
            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble(run_error=ValueError("pipeline failed"))

            from elspeth.cli import bootstrap_and_run

            with pytest.raises(ValueError, match="pipeline failed"):
                bootstrap_and_run(Path("/fake/pipeline.yaml"))

    def test_dependency_runner_requests_silent_orchestrator_output(self) -> None:
        """bootstrap_and_run is programmatic and must not attach CLI formatters."""
        mock_config = _make_bootstrap_config()
        mock_db = _CloseableDouble()

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch(
                "elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config",
                return_value=_make_plugin_bundle(),
            ),
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = _GraphStub()
            mock_db_cls.from_url.return_value = mock_db
            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble()

            from elspeth.cli import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        assert mock_orch_ctx.call_args is not None
        assert mock_orch_ctx.call_args.kwargs["output_format"] == "none"

    def test_bootstrap_passes_declared_queues_to_graph_builder(self) -> None:
        """Dependency/preflight graph construction must honor declared queue fan-in nodes."""
        mock_config = _make_bootstrap_config()
        mock_config.queues = {"inbound": object()}

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch(
                "elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config",
                return_value=_make_plugin_bundle(),
            ),
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=[]),
        ):
            mock_graph_cls.from_plugin_instances.return_value = _GraphStub()
            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble()

            from elspeth.cli import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        assert mock_graph_cls.from_plugin_instances.call_args is not None
        assert mock_graph_cls.from_plugin_instances.call_args.kwargs["queues"] == mock_config.queues

    def test_orchestrator_context_none_output_uses_null_event_bus(self) -> None:
        """The silent output mode uses NullEventBus, not console/json formatters."""
        from elspeth.cli import _orchestrator_context
        from elspeth.core.events import NullEventBus

        mock_config = _make_bootstrap_config()
        mock_config.checkpoint.enabled = False
        source = object()
        mock_graph = _GraphStub()
        mock_plugins = SimpleNamespace(
            transforms=[],
            aggregations={},
            source=source,
            sources={"source": source},
            source_settings_map={},
            sinks={"out": object()},
        )
        mock_db = _CloseableDouble()

        with (
            patch("elspeth.contracts.config.runtime.RuntimeRateLimitConfig.from_settings", return_value=SimpleNamespace()),
            patch("elspeth.contracts.config.runtime.RuntimeConcurrencyConfig.from_settings", return_value=SimpleNamespace()),
            patch(
                "elspeth.contracts.config.runtime.RuntimeCheckpointConfig.from_settings",
                return_value=SimpleNamespace(enabled=False),
            ),
            patch("elspeth.contracts.config.runtime.RuntimeTelemetryConfig.from_settings", return_value=SimpleNamespace()),
            patch("elspeth.core.rate_limit.RateLimitRegistry") as mock_rate_limit_registry,
            patch("elspeth.telemetry.create_telemetry_manager", return_value=_CloseableDouble()),
            patch("elspeth.engine.Orchestrator") as mock_orchestrator,
        ):
            mock_rate_limit_registry.return_value = _CloseableDouble()
            with _orchestrator_context(mock_config, mock_graph, mock_plugins, db=mock_db, output_format="none"):
                pass

        assert mock_orchestrator.call_args is not None
        assert isinstance(mock_orchestrator.call_args.kwargs["event_bus"], NullEventBus)

    def test_orchestrator_context_preserves_named_sources(self) -> None:
        """CLI PipelineConfig assembly carries plural source instances into runtime."""
        from elspeth.cli import _orchestrator_context

        mock_config = _make_bootstrap_config()
        mock_config.checkpoint.enabled = False
        mock_graph = _GraphStub()
        first_source = object()
        second_source = object()
        mock_plugins = SimpleNamespace(
            transforms=[],
            aggregations={},
            sources={"orders": first_source, "refunds": second_source},
            sinks={"out": object()},
        )
        mock_db = _CloseableDouble()

        with (
            patch("elspeth.contracts.config.runtime.RuntimeRateLimitConfig.from_settings", return_value=SimpleNamespace()),
            patch("elspeth.contracts.config.runtime.RuntimeConcurrencyConfig.from_settings", return_value=SimpleNamespace()),
            patch(
                "elspeth.contracts.config.runtime.RuntimeCheckpointConfig.from_settings",
                return_value=SimpleNamespace(enabled=False),
            ),
            patch("elspeth.contracts.config.runtime.RuntimeTelemetryConfig.from_settings", return_value=SimpleNamespace()),
            patch("elspeth.core.rate_limit.RateLimitRegistry", return_value=_CloseableDouble()),
            patch("elspeth.telemetry.create_telemetry_manager", return_value=_CloseableDouble()),
            patch("elspeth.engine.Orchestrator"),
            _orchestrator_context(mock_config, mock_graph, mock_plugins, db=mock_db, output_format="none") as ctx,
        ):
            assert list(ctx.pipeline_config.sources) == ["orders", "refunds"]
            assert ctx.pipeline_config.sources["orders"] is first_source
            assert ctx.pipeline_config.sources["refunds"] is second_source


class TestBootstrapDependencyResultsFlow:
    """Test that dependency results flow into gate context (regression for bug #1)."""

    def test_dependency_results_passed_to_gate_context(self) -> None:
        """Dependency run results must be visible in commencement gate expressions."""
        from elspeth.core.dependency_config import DependencyConfig

        mock_config = _make_bootstrap_config()
        mock_config.depends_on = [DependencyConfig(name="indexer", settings="./index.yaml")]
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]

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
            patch(
                "elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config",
                return_value=_make_plugin_bundle(),
            ),
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.dependency_resolver.detect_cycles"),
            patch("elspeth.engine.dependency_resolver.resolve_dependencies", return_value=[dep_result]),
            patch("elspeth.engine.commencement.evaluate_commencement_gates", side_effect=capture_gate_context),
        ):
            mock_graph_cls.from_plugin_instances.return_value = _GraphStub()
            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble()

            from elspeth.cli import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        # Dependency results must be in the gate context — NOT empty
        assert "dependency_runs" in captured_context
        assert "indexer" in captured_context["dependency_runs"]
        assert captured_context["dependency_runs"]["indexer"]["run_id"] == "dep-run-abc"


class TestBootstrapCommencementGateDispatch:
    """Test that bootstrap_and_run() evaluates gates when configured."""

    def test_skips_gates_when_not_configured(self) -> None:
        """When commencement_gates is empty, gate evaluator is never called."""
        mock_config = _make_bootstrap_config()

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.core.landscape.LandscapeDB"),
            patch("elspeth.core.payload_store.FilesystemPayloadStore"),
            patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.engine.commencement.evaluate_commencement_gates") as mock_eval_gates,
        ):
            mock_plugins.return_value = _make_plugin_bundle()
            mock_graph = _GraphStub()
            mock_graph_cls.from_plugin_instances.return_value = mock_graph

            mock_orch_ctx.return_value = _OrchestratorContextManagerDouble()

            from elspeth.cli import bootstrap_and_run

            bootstrap_and_run(Path("/fake/pipeline.yaml"))

        mock_eval_gates.assert_not_called()

    def test_gate_failure_propagates(self) -> None:
        """When a gate fails, CommencementGateFailedError propagates through bootstrap."""
        mock_config = _make_bootstrap_config()
        mock_config.commencement_gates = [CommencementGateConfig(name="test_gate", condition="True")]  # Non-empty triggers gate eval

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(mock_config, [])),
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
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
            mock_plugins.return_value = _make_plugin_bundle()
            mock_graph = _GraphStub()
            mock_graph_cls.from_plugin_instances.return_value = mock_graph

            from elspeth.cli import bootstrap_and_run

            with pytest.raises(CommencementGateFailedError, match="test_gate"):
                bootstrap_and_run(Path("/fake/pipeline.yaml"))
