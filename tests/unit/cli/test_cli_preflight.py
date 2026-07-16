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

import pytest
from typer.testing import CliRunner

from elspeth.cli import _preflight_follower_sink_effects, _preflight_raw_settings_sink_effects, app
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

    @classmethod
    def _resolve_sink_effect_mode(cls, _options: object, *, purpose: object) -> object:
        from elspeth.contracts.sink_effects import ResolvedSinkEffectMode

        return ResolvedSinkEffectMode("write")

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

    @property
    def sink_effect_bindings(self) -> dict[str, object]:
        from elspeth.contracts.hashing import stable_hash
        from elspeth.contracts.sink_effects import (
            ResolvedSinkEffectMode,
            SinkEffectExecutionPurpose,
            SinkEffectRuntimeBinding,
        )

        return {
            sink_name: SinkEffectRuntimeBinding(
                sink_name=sink_name,
                sink=sink,
                sink_type=type(sink),
                config_fingerprint=stable_hash({}),
                purpose=SinkEffectExecutionPurpose.FRESH,
                effect_mode=ResolvedSinkEffectMode(mode) if (mode := self.sink_effect_modes.get(sink_name)) is not None else None,
            )
            for sink_name, sink in self.sinks.items()
        }


def _make_minimal_config_yaml(
    tmp_path: Path,
    *,
    with_depends_on: bool = False,
    sink_plugin: str = "csv",
) -> Path:
    """Write a minimal valid pipeline YAML and return its path."""
    import yaml

    config: dict[str, object] = {
        "sources": {
            "primary": {
                "plugin": "csv",
                "on_success": "output",
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
                "plugin": sink_plugin,
                "on_write_failure": "discard",
                "options": {"path": str(tmp_path / "output.csv"), "schema": {"mode": "observed"}},
            }
        },
    }
    if with_depends_on:
        config["depends_on"] = [{"name": "indexer", "settings": "./index.yaml"}]

    settings_path = tmp_path / "pipeline.yaml"
    settings_path.write_text(yaml.dump(config))
    return settings_path


def _explicit_audit_export_settings(*, enabled: bool | str | int, sink: str = "audit") -> dict[str, object]:
    """Return a fully bounded raw export policy for CLI preflight tests."""
    return {
        "enabled": enabled,
        "sink": sink,
        "total_record_limit": 1_000,
        "total_byte_limit": 1_000_000,
        "chunk_limit": 10,
        "per_chunk_record_limit": 100,
        "per_chunk_byte_limit": 100_000,
        "spool_root": ".elspeth/audit-export-spool/cli-preflight",
        "spool_cleanup_age_seconds": 3_600,
        "spool_cleanup_byte_budget": 1_000_000,
        "spool_cleanup_count_budget": 10,
        "content_store": {
            "content_store_id": "cli-preflight-store-v1",
            "namespace": "audit/export",
            "root": ".elspeth/audit-export-content-store/cli-preflight",
            "policy_version": "v1",
            "retention_days": 30,
            "durability": "fsync",
            "orphan_grace_period_seconds": 3_600,
        },
    }


def test_cli_run_rejects_raw_legacy_sink_before_key_vault_resolution(tmp_path: Path) -> None:
    settings_path = _make_minimal_config_yaml(tmp_path, sink_plugin="legacy")
    manager = SimpleNamespace(get_sink_by_name=lambda _name: _LegacyResumeSink)

    with (
        patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=manager),
        patch("elspeth.cli.load_secrets_from_config") as load_secrets,
    ):
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
                "landscape": {"export": _explicit_audit_export_settings(enabled=True)},
            }
        )
    )
    purposes: list[SinkEffectExecutionPurpose] = []

    def validate(
        _raw: object,
        *,
        purpose: SinkEffectExecutionPurpose,
        expand_env_placeholders: bool = False,
        deferrable_env_vars: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        del expand_env_placeholders, deferrable_env_vars
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


@pytest.mark.parametrize(
    ("raw_enabled", "expected_enabled"),
    [
        (True, True),
        ("true", True),
        ("1", True),
        ("yes", True),
        (1, True),
        (False, False),
        ("false", False),
        ("0", False),
        ("no", False),
        (0, False),
    ],
)
def test_raw_export_lane_classification_matches_pydantic_bool_coercion(
    tmp_path: Path,
    raw_enabled: bool | str | int,
    expected_enabled: bool,
) -> None:
    import yaml

    from elspeth.engine.orchestrator.preflight import SinkEffectExecutionPurpose

    settings_path = tmp_path / "coerced-export.yaml"
    settings_path.write_text(
        yaml.dump(
            {
                "sinks": {
                    "pipeline": {"plugin": "csv", "options": {}},
                    "audit": {"plugin": "json", "options": {}},
                },
                "landscape": {"export": _explicit_audit_export_settings(enabled=raw_enabled)},
            }
        )
    )
    purposes: list[SinkEffectExecutionPurpose] = []
    with patch(
        "elspeth.plugins.infrastructure.runtime_factory.validate_sink_effect_eligibility_from_raw_config",
        side_effect=lambda _raw, *, purpose, **_expansion: purposes.append(purpose) or {},
    ):
        _preflight_raw_settings_sink_effects(settings_path, purpose=SinkEffectExecutionPurpose.FRESH)

    expected = [SinkEffectExecutionPurpose.FRESH]
    if expected_enabled:
        expected.append(SinkEffectExecutionPurpose.AUDIT_EXPORT)
    assert purposes == expected


def _database_sink_settings_yaml(tmp_path: Path, *, secrets: dict[str, object] | None = None) -> Path:
    """Settings YAML using the documented ${DATABASE_URL} sink URL placeholder."""
    import yaml

    config: dict[str, object] = {
        "sources": {"primary": {"plugin": "csv", "options": {"path": str(tmp_path / "input.csv"), "schema": {"mode": "observed"}}}},
        "sinks": {
            "db_out": {
                "plugin": "database",
                "options": {
                    "url": "${DATABASE_URL}",
                    "table": "results",
                    "schema": {"mode": "observed"},
                    "if_exists": "append",
                    "effect_ledger": {
                        "table": "_elspeth_results_ledger",
                        "permissions": ["insert", "select"],
                    },
                },
            }
        },
    }
    if secrets is not None:
        config["secrets"] = secrets
    settings_path = tmp_path / "env-url.yaml"
    settings_path.write_text(yaml.dump(config))
    return settings_path


def test_cli_raw_preflight_expands_env_placeholders_before_mode_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """elspeth-19f2382cf4: raw preflight must expand ${DATABASE_URL} before dialect checks."""
    from elspeth.engine.orchestrator.preflight import SinkEffectExecutionPurpose

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'results.db'}")
    settings_path = _database_sink_settings_yaml(tmp_path)

    _preflight_raw_settings_sink_effects(settings_path, purpose=SinkEffectExecutionPurpose.FRESH)


def test_cli_raw_preflight_defers_keyvault_mapped_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vault-mapped sink URL is unknowable before secret loading; the raw gate defers."""
    from elspeth.engine.orchestrator.preflight import SinkEffectExecutionPurpose

    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings_path = _database_sink_settings_yaml(
        tmp_path,
        secrets={
            "source": "keyvault",
            "vault_url": "https://unit-vault.vault.azure.net",
            "mapping": {"DATABASE_URL": "database-url"},
        },
    )

    _preflight_raw_settings_sink_effects(settings_path, purpose=SinkEffectExecutionPurpose.FRESH)


def test_malformed_raw_export_shape_fails_before_secret_loading(tmp_path: Path) -> None:
    import yaml

    settings_path = tmp_path / "malformed-export.yaml"
    settings_path.write_text(
        yaml.dump(
            {
                "sinks": {"pipeline": {"plugin": "csv", "options": {}}},
                "landscape": {"export": {"enabled": []}},
            }
        )
    )
    with (
        patch(
            "elspeth.plugins.infrastructure.runtime_factory.validate_sink_effect_eligibility_from_raw_config",
            return_value={},
        ),
        patch("elspeth.cli._load_settings_with_secrets") as load_settings_with_secrets,
    ):
        result = runner.invoke(app, ["run", "-s", str(settings_path), "--execute"])

    assert result.exit_code == 1
    assert "enabled" in result.output
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
        sinks={"output": SimpleNamespace(options={})},
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
