"""Tests that CLI run --execute calls resolve_preflight for depends_on support.

Regression test for P0 bug: CLI inline bootstrap bypassed dependency resolution
and commencement gates. The fix routes the CLI through resolve_preflight() so
depends_on, commencement_gates, and collection_probes are reachable from
``elspeth run --settings <path> --execute``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from elspeth.cli import _preflight_follower_sink_effects, app
from elspeth.contracts.preflight import DependencyRunResult, PreflightResult
from elspeth.contracts.sink_effects import SINK_EFFECT_PROTOCOL_VERSION, SinkEffectInputKind

runner = CliRunner()


def test_follower_preflight_passes_explicit_pipeline_members_kind() -> None:
    sinks = {"output": object()}
    with patch("elspeth.engine.orchestrator.preflight.validate_pipeline_sink_effect_capabilities") as validate:
        _preflight_follower_sink_effects(sinks, {"output": "write"})  # type: ignore[arg-type]
    validate.assert_called_once_with(
        sinks,
        configured_modes={"output": "write"},
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )


@dataclass(slots=True)
class _FakeGraph:
    node_count: int = 0
    edge_count: int = 0

    def validate(self) -> None:
        """Match the ExecutionGraph API used by the CLI run command."""


class _EffectCapableSink:
    name = "effect-capable"
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    supported_effect_modes = frozenset({"write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})

    def inspect_effect(self, _request: object, _ctx: object) -> None: ...

    def prepare_effect(self, _request: object, _ctx: object) -> None: ...

    def commit_effect(self, _plan: object, _ctx: object) -> None: ...

    def reconcile_effect(self, _plan: object, _ctx: object) -> None: ...


class _LegacyResumeSink:
    name = "legacy"

    def __init__(self) -> None:
        self.configure_for_resume = MagicMock(spec=[])


@dataclass(frozen=True, slots=True)
class _FakePluginBundle:
    sources: dict[str, object] = field(default_factory=lambda: {"primary": object()})
    source_settings_map: dict[str, object] = field(default_factory=lambda: {"primary": object()})
    transforms: tuple[object, ...] = ()
    sinks: dict[str, object] = field(default_factory=lambda: {"output": _EffectCapableSink()})
    aggregations: dict[str, object] = field(default_factory=dict)
    sink_effect_modes: dict[str, str] = field(default_factory=lambda: {"output": "write"})


def _make_minimal_config_yaml(tmp_path: Path, *, with_depends_on: bool = False) -> Path:
    """Write a minimal valid pipeline YAML and return its path."""
    import yaml

    config: dict[str, object] = {
        "sources": {"primary": {"plugin": "csv", "options": {"path": str(tmp_path / "input.csv")}}},
        "transforms": [],
        "sinks": {"output": {"plugin": "csv", "on_write_failure": "discard", "options": {"path": str(tmp_path / "output.csv")}}},
    }
    if with_depends_on:
        config["depends_on"] = [{"name": "indexer", "settings": "./index.yaml"}]

    settings_path = tmp_path / "pipeline.yaml"
    settings_path.write_text(yaml.dump(config))
    return settings_path


def test_cli_run_rejects_raw_legacy_sink_before_key_vault_resolution(tmp_path: Path) -> None:
    settings_path = _make_minimal_config_yaml(tmp_path)

    with patch("elspeth.cli.load_secrets_from_config") as load_secrets:
        result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

    assert result.exit_code == 1
    assert "effect protocol" in result.output.lower()
    load_secrets.assert_not_called()


def test_cli_fresh_run_screens_pipeline_and_export_lanes_before_secret_loading(tmp_path: Path) -> None:
    import yaml

    from elspeth.engine.orchestrator.preflight import SinkEffectCapabilityError, SinkEffectExecutionPurpose

    settings_path = tmp_path / "dual-lane.yaml"
    settings_path.write_text(
        yaml.dump(
            {
                "sources": {"primary": {"plugin": "csv", "options": {"path": "input.csv"}}},
                "sinks": {
                    "pipeline": {"plugin": "csv", "options": {"path": "output.csv"}},
                    "audit": {"plugin": "json", "options": {"path": "audit.jsonl"}},
                },
                "landscape": {"export": {"enabled": True, "sink": "audit"}},
            }
        )
    )
    purposes: list[SinkEffectExecutionPurpose] = []

    def validate(_raw: object, *, purpose: SinkEffectExecutionPurpose) -> dict[str, object]:
        purposes.append(purpose)
        if purpose is SinkEffectExecutionPurpose.AUDIT_EXPORT:
            raise SinkEffectCapabilityError("export lane rejected")
        return {}

    with (
        patch(
            "elspeth.plugins.infrastructure.runtime_factory.validate_sink_effect_eligibility_from_raw_config",
            side_effect=validate,
        ),
        patch("elspeth.cli._load_settings_with_secrets") as load_settings_with_secrets,
    ):
        result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

    assert result.exit_code == 1
    assert purposes == [SinkEffectExecutionPurpose.FRESH, SinkEffectExecutionPurpose.AUDIT_EXPORT]
    load_settings_with_secrets.assert_not_called()


def _make_resume_config_and_database(tmp_path: Path) -> tuple[Path, Path]:
    import yaml

    db_path = tmp_path / "landscape.db"
    config = {
        "sources": {
            "primary": {
                "plugin": "csv",
                "on_success": "default",
                "options": {
                    "path": str(tmp_path / "input.csv"),
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            }
        },
        "transforms": [],
        "sinks": {
            "output": {
                "plugin": "csv",
                "on_write_failure": "discard",
                "options": {"path": str(tmp_path / "output.csv"), "schema": {"mode": "observed"}},
            }
        },
        "landscape": {"url": f"sqlite:///{db_path}"},
        "payload_store": {"backend": "filesystem", "base_path": str(tmp_path / "payloads")},
    }
    settings_path = tmp_path / "resume.yaml"
    settings_path.write_text(yaml.dump(config))

    from elspeth.core.landscape import LandscapeDB

    db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
    db.close()
    return settings_path, db_path


def _fake_config(*, with_depends_on: bool) -> SimpleNamespace:
    return SimpleNamespace(
        depends_on=[SimpleNamespace(name="indexer")] if with_depends_on else [],
        collection_probes=[SimpleNamespace(name="probe")] if with_depends_on else [],
        gates=[],
        coalesce=[],
        queues={},
        landscape=SimpleNamespace(export=SimpleNamespace(enabled=False, sink=None)),
    )


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
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.cli_helpers.resolve_audit_passphrase", return_value=None),
            patch("elspeth.engine.bootstrap.resolve_preflight") as mock_preflight,
            patch("elspeth.cli._execute_pipeline_with_instances") as mock_execute,
            patch("elspeth.plugins.infrastructure.probe_factory.build_collection_probes", return_value=[]),
        ):
            mock_config = _fake_config(with_depends_on=True)
            mock_load.return_value = (mock_config, [])

            mock_plugins.return_value = _FakePluginBundle()
            mock_graph_cls.from_plugin_instances.return_value = _FakeGraph()

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
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config") as mock_plugins,
            patch("elspeth.cli.ExecutionGraph") as mock_graph_cls,
            patch("elspeth.cli._ensure_output_directories", return_value=[]),
            patch("elspeth.cli_helpers.resolve_audit_passphrase", return_value=None),
            patch("elspeth.engine.bootstrap.resolve_preflight", return_value=None) as mock_preflight,
            patch("elspeth.cli._execute_pipeline_with_instances") as mock_execute,
            patch("elspeth.plugins.infrastructure.probe_factory.build_collection_probes", return_value=[]),
        ):
            mock_config = _fake_config(with_depends_on=False)
            mock_load.return_value = (mock_config, [])

            mock_plugins.return_value = _FakePluginBundle()
            mock_graph_cls.from_plugin_instances.return_value = _FakeGraph()
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
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config") as mock_plugins,
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
            mock_config = _fake_config(with_depends_on=True)
            mock_load.return_value = (mock_config, [])

            mock_plugins.return_value = _FakePluginBundle()
            mock_graph_cls.from_plugin_instances.return_value = _FakeGraph()

            result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

            # Pipeline should NOT have been called
            mock_execute.assert_not_called()

        assert result.exit_code == 1
        assert "pre-flight check failed" in result.output.lower()
        assert "circular dependency" in result.output.lower()

    def test_cli_run_rejects_legacy_sink_before_output_directories(self, tmp_path: Path) -> None:
        settings_path = _make_minimal_config_yaml(tmp_path)
        legacy_bundle = _FakePluginBundle(sinks={"output": object()}, sink_effect_modes={})

        with (
            patch("elspeth.cli._load_settings_with_secrets", return_value=(_fake_config(with_depends_on=False), [])),
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config", return_value=legacy_bundle),
            patch("elspeth.cli.ExecutionGraph") as graph_cls,
            patch("elspeth.cli._ensure_output_directories") as ensure_directories,
            patch("elspeth.engine.bootstrap.resolve_preflight") as resolve_preflight,
        ):
            graph_cls.from_plugin_instances.return_value = _FakeGraph()
            result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

        assert result.exit_code == 1
        assert "sink effect preflight failed" in result.output.lower()
        ensure_directories.assert_not_called()
        resolve_preflight.assert_not_called()


def test_cli_resume_rejects_legacy_sink_before_resume_mutation_or_payload_access(tmp_path: Path) -> None:
    from elspeth.contracts.checkpoint import ResumeCheck

    settings_path, _db_path = _make_resume_config_and_database(tmp_path)
    sink = _LegacyResumeSink()
    bundle = _FakePluginBundle(sinks={"output": sink}, sink_effect_modes={})
    resume_point = SimpleNamespace(sequence_number=0, barrier_scalars=None)

    with (
        patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config", return_value=bundle),
        patch("elspeth.cli.ExecutionGraph") as graph_cls,
        patch("elspeth.core.checkpoint.RecoveryManager") as recovery_cls,
        patch("elspeth.cli_helpers.resolve_audit_passphrase") as resolve_passphrase,
        patch("elspeth.core.landscape.LandscapeDB.from_url") as open_database,
    ):
        graph_cls.from_plugin_instances.return_value = _FakeGraph()
        recovery = recovery_cls.return_value
        recovery.can_resume.return_value = ResumeCheck(can_resume=True)
        recovery.get_resume_point.return_value = resume_point
        recovery.get_unprocessed_rows.return_value = []
        recovery.count_blocked_barrier_items.return_value = 0
        result = runner.invoke(app, ["resume", "run-1", "-s", str(settings_path), "--execute"])

    assert result.exit_code == 1
    assert "sink effect preflight failed" in result.output.lower()
    sink.configure_for_resume.assert_not_called()
    resolve_passphrase.assert_not_called()
    open_database.assert_not_called()
    assert not (tmp_path / "payloads").exists()


def test_cli_join_rejects_legacy_sink_before_join_admission(tmp_path: Path) -> None:
    settings_path, db_path = _make_resume_config_and_database(tmp_path)
    bundle = _FakePluginBundle(sinks={"output": _LegacyResumeSink()}, sink_effect_modes={})

    with (
        patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config", return_value=bundle),
        patch("elspeth.engine.Orchestrator") as orchestrator_cls,
        patch("elspeth.cli_helpers.resolve_audit_passphrase") as resolve_passphrase,
        patch("elspeth.core.landscape.LandscapeDB.from_url") as open_database,
    ):
        result = runner.invoke(
            app,
            ["join", "run-1", "-s", str(settings_path), "--database", str(db_path)],
        )

    assert result.exit_code == 1
    assert "sink effect preflight failed" in result.output.lower()
    resolve_passphrase.assert_not_called()
    open_database.assert_not_called()
    orchestrator_cls.return_value.join_run.assert_not_called()
