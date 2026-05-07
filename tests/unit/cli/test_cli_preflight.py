"""Tests that CLI run --execute calls resolve_preflight for depends_on support.

Regression test for P0 bug: CLI inline bootstrap bypassed dependency resolution
and commencement gates. The fix routes the CLI through resolve_preflight() so
depends_on, commencement_gates, and collection_probes are reachable from
``elspeth run --settings <path> --execute``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.core.dependency_config import DependencyRunResult, PreflightResult

runner = CliRunner()


def _make_minimal_config_yaml(tmp_path: Path, *, with_depends_on: bool = False) -> Path:
    """Write a minimal valid pipeline YAML and return its path."""
    import yaml

    config: dict[str, object] = {
        "source": {"plugin": "csv", "options": {"path": str(tmp_path / "input.csv")}},
        "transforms": [],
        "sinks": {"output": {"plugin": "csv", "on_write_failure": "discard", "options": {"path": str(tmp_path / "output.csv")}}},
    }
    if with_depends_on:
        config["depends_on"] = [{"name": "indexer", "settings": "./index.yaml"}]

    settings_path = tmp_path / "pipeline.yaml"
    settings_path.write_text(yaml.dump(config))
    return settings_path


class TestCLIRunCallsResolvePreflight:
    """Verify the CLI run --execute path invokes resolve_preflight."""

    def test_cli_run_execute_calls_resolve_preflight(self, tmp_path: Path) -> None:
        """The --execute path must call resolve_preflight so depends_on is honoured.

        This is the core regression test. Before the fix, resolve_preflight was
        only reachable via bootstrap_and_run() (programmatic path), not the CLI.
        """
        settings_path = _make_minimal_config_yaml(tmp_path, with_depends_on=True)

        with (
            patch("elspeth.cli._load_settings_with_secrets") as mock_load,
            patch("elspeth.cli_helpers.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.cli_helpers.resolve_audit_passphrase", return_value=None),
            patch("elspeth.engine.bootstrap.resolve_preflight") as mock_preflight,
            patch("elspeth.cli._execute_pipeline_with_instances") as mock_execute,
            patch("elspeth.plugins.infrastructure.probe_factory.build_collection_probes", return_value=[]),
        ):
            mock_config = MagicMock()
            mock_config.depends_on = [MagicMock()]
            mock_config.collection_probes = [MagicMock()]
            mock_config.gates = []
            mock_config.coalesce = []
            mock_config.landscape.export.enabled = False
            mock_load.return_value = (mock_config, [])

            mock_plugins.return_value = MagicMock()
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()

            # resolve_preflight returns a PreflightResult with dependency runs
            dep_result = DependencyRunResult(
                name="indexer",
                run_id="dep-run-abc",
                settings_hash="sha256:abc",
                duration_ms=1000,
                indexed_at="2026-03-25T12:00:00Z",
            )
            preflight = PreflightResult(
                dependency_runs=(dep_result,),
                gate_results=(),
            )
            mock_preflight.return_value = preflight

            mock_execute.return_value = {
                "run_id": "test-run-id",
                "status": "completed",
                "rows_processed": 0,
            }

            result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

            # resolve_preflight MUST have been called with the config
            mock_preflight.assert_called_once()
            call_args = mock_preflight.call_args
            assert call_args[0][0] is mock_config  # first positional = config

            # preflight_results MUST be passed through to _execute_pipeline_with_instances
            mock_execute.assert_called_once()
            execute_kwargs = mock_execute.call_args
            assert execute_kwargs.kwargs.get("preflight_results") is preflight

        assert result.exit_code == 0

    def test_cli_run_execute_passes_none_preflight_when_no_depends_on(self, tmp_path: Path) -> None:
        """When no depends_on is configured, preflight is None but still passed."""
        settings_path = _make_minimal_config_yaml(tmp_path, with_depends_on=False)

        with (
            patch("elspeth.cli._load_settings_with_secrets") as mock_load,
            patch("elspeth.cli_helpers.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.cli_helpers.resolve_audit_passphrase", return_value=None),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=None) as mock_preflight,
            patch("elspeth.cli._execute_pipeline_with_instances") as mock_execute,
            patch("elspeth.plugins.infrastructure.probe_factory.build_collection_probes", return_value=[]),
        ):
            mock_config = MagicMock()
            mock_config.depends_on = []
            mock_config.collection_probes = []
            mock_config.gates = []
            mock_config.coalesce = []
            mock_config.landscape.export.enabled = False
            mock_load.return_value = (mock_config, [])

            mock_plugins.return_value = MagicMock()
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()
            mock_execute.return_value = {
                "run_id": "test-run-id",
                "status": "completed",
                "rows_processed": 0,
            }

            result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

            mock_preflight.assert_called_once()
            mock_execute.assert_called_once()
            assert mock_execute.call_args.kwargs.get("preflight_results") is None

        assert result.exit_code == 0

    def test_cli_run_execute_preflight_error_shows_message(self, tmp_path: Path) -> None:
        """If resolve_preflight raises, the CLI shows a helpful error and exits 1."""
        settings_path = _make_minimal_config_yaml(tmp_path, with_depends_on=True)

        with (
            patch("elspeth.cli._load_settings_with_secrets") as mock_load,
            patch("elspeth.cli_helpers.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.cli_helpers.resolve_audit_passphrase", return_value=None),
            patch(
                "elspeth.engine.bootstrap.resolve_preflight",
                side_effect=ValueError("Circular dependency detected: A -> B -> A"),
            ),
            patch("elspeth.cli._execute_pipeline_with_instances") as mock_execute,
            patch("elspeth.plugins.infrastructure.probe_factory.build_collection_probes", return_value=[]),
        ):
            mock_config = MagicMock()
            mock_config.depends_on = [MagicMock()]
            mock_config.collection_probes = [MagicMock()]
            mock_config.gates = []
            mock_config.coalesce = []
            mock_config.landscape.export.enabled = False
            mock_load.return_value = (mock_config, [])

            mock_plugins.return_value = MagicMock()
            mock_graph_cls.from_plugin_instances.return_value = MagicMock()

            result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

            # Pipeline should NOT have been called
            mock_execute.assert_not_called()

        assert result.exit_code == 1
        assert "pre-flight check failed" in result.output.lower()
        assert "circular dependency" in result.output.lower()
