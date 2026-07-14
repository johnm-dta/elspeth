"""Unit tests for the AWS ECS doctor report and its safety boundaries."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.core.config import TelemetrySettings
from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import ContractCheck
from elspeth.web.doctor import (
    _aws_operator_telemetry_check,
    _bedrock_guardrail_plugins_check,
    _initialize_database,
    _inspect_database,
    collect_checks,
    database_target_check,
    plugin_and_dependency_checks,
    probe_directory_writable,
    sanitize_error,
    schema_check,
)
from elspeth.web.schema_probe import SchemaInitBusyError, SchemaLockCleanupError, SchemaState
from elspeth.web.sessions.schema import SessionSchemaError


def _settings(tmp_path: Path, **overrides: Any) -> WebSettings:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payloads"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "blobs").mkdir(exist_ok=True)
    payload_dir.mkdir(mode=0o700, exist_ok=True)
    values: dict[str, Any] = {
        "deployment_target": "aws-ecs",
        "operator_telemetry": "aws-otlp",
        "operator_telemetry_environment": "test",
        "operator_telemetry_release": "git-test",
        "operator_telemetry_ecs_cluster": "elspeth-test",
        "operator_telemetry_ecs_service": "elspeth-web",
        "operator_telemetry_task_definition_family": "elspeth-web-task",
        "operator_telemetry_task_definition_revision": "1",
        "host": "0.0.0.0",
        "session_db_url": "postgresql+psycopg://doctor:secret@db/session",
        "landscape_url": "postgresql+psycopg://doctor:secret@db/landscape",
        "data_dir": data_dir,
        "payload_store_path": payload_dir,
        "secret_key": "s" * 40,
        "shareable_link_signing_key": bytes(range(32)),
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
    }
    values.update(overrides)
    return WebSettings(**values)


def _by_name(checks: list[ContractCheck]) -> dict[str, ContractCheck]:
    return {check.name: check for check in checks}


def test_sanitize_error_exposes_only_context_and_exception_class() -> None:
    detail = sanitize_error(
        "payload probe failed",
        RuntimeError("postgresql://doctor:hunter2@secret-host/db /srv/private"),  # secret-scan: allow-this-line
    )

    assert detail == "payload probe failed (RuntimeError)"
    assert "hunter2" not in detail
    assert "secret-host" not in detail
    assert "/srv/private" not in detail


def test_existing_directory_probe_succeeds_without_named_artifact(tmp_path: Path) -> None:
    check = probe_directory_writable("payload", tmp_path)

    assert check == ContractCheck("payload_writable", True, "payload directory is writable")
    assert list(tmp_path.glob(".doctor-probe-*")) == []


@pytest.mark.parametrize("path_kind", ["none", "missing", "file"])
def test_non_directory_probe_fails_without_creating_anything(tmp_path: Path, path_kind: str) -> None:
    path: Path | None
    if path_kind == "none":
        path = None
    elif path_kind == "missing":
        path = tmp_path / "missing" / "nested"
    else:
        path = tmp_path / "not-a-directory"
        path.write_text("sentinel")

    before = set(tmp_path.rglob("*"))
    check = probe_directory_writable("payload", path)

    assert check.ok is False
    assert check.name == "payload_writable"
    assert set(tmp_path.rglob("*")) == before
    assert list(tmp_path.rglob(".doctor-probe-*")) == []


def test_probe_uses_mkstemp_with_unpredictable_prefix_and_target_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    calls: list[tuple[str, Path]] = []
    real_mkstemp = doctor.tempfile.mkstemp

    def spy_mkstemp(*, prefix: str, dir: Path) -> tuple[int, str]:
        calls.append((prefix, dir))
        return real_mkstemp(prefix=prefix, dir=dir)

    monkeypatch.setattr(doctor.tempfile, "mkstemp", spy_mkstemp)

    assert probe_directory_writable("blob", tmp_path).ok is True
    assert calls == [(".doctor-probe-", tmp_path)]
    assert list(tmp_path.glob(".doctor-probe-*")) == []


def test_probe_fsync_failure_is_sanitized_and_cleaned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    def fail_fsync(_fd: int) -> None:
        raise OSError("credential-url postgresql://user:hunter2@private/db")  # secret-scan: allow-this-line

    monkeypatch.setattr(doctor.os, "fsync", fail_fsync)

    check = probe_directory_writable("payload", tmp_path)

    assert check == ContractCheck("payload_writable", False, "payload directory probe failed (OSError)")
    assert "hunter2" not in check.detail
    assert list(tmp_path.glob(".doctor-probe-*")) == []


@pytest.mark.parametrize("failing_operation", ["write", "read"])
def test_probe_write_and_read_failures_are_sanitized_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_operation: str,
) -> None:
    import elspeth.web.doctor as doctor

    real_fdopen = doctor.os.fdopen

    class FailingProbe:
        def __init__(self, fd: int, mode: str) -> None:
            self._file = real_fdopen(fd, mode)

        def __enter__(self) -> FailingProbe:
            return self

        def __exit__(self, *args: object) -> None:
            self._file.close()

        def write(self, value: bytes) -> int:
            if failing_operation == "write":
                raise OSError("postgresql://user:hunter2@private/db")  # secret-scan: allow-this-line
            return self._file.write(value)

        def flush(self) -> None:
            self._file.flush()

        def fileno(self) -> int:
            return self._file.fileno()

        def seek(self, offset: int) -> int:
            return self._file.seek(offset)

        def read(self) -> bytes:
            if failing_operation == "read":
                raise OSError("/private/payload/path")
            return self._file.read()

    monkeypatch.setattr(doctor.os, "fdopen", FailingProbe)

    check = probe_directory_writable("payload", tmp_path)

    assert check == ContractCheck("payload_writable", False, "payload directory probe failed (OSError)")
    assert "hunter2" not in check.detail
    assert "/private/payload/path" not in check.detail
    assert list(tmp_path.glob(".doctor-probe-*")) == []


def test_early_unlink_failure_is_sanitized_and_cleanup_removes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    real_unlink = doctor.os.unlink
    calls = 0

    def fail_first_unlink(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("/secret/path")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(doctor.os, "unlink", fail_first_unlink)

    check = probe_directory_writable("data_dir", tmp_path)

    assert check == ContractCheck("data_dir_writable", False, "data_dir directory probe failed (OSError)")
    assert list(tmp_path.glob(".doctor-probe-*")) == []


def test_cleanup_failure_is_a_sanitized_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    def fail_unlink(_path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: Any, **kwargs: Any) -> None:
        raise OSError("/secret/cleanup")

    monkeypatch.setattr(doctor.os, "unlink", fail_unlink)

    check = probe_directory_writable("blob", tmp_path)

    assert check == ContractCheck("blob_writable", False, "blob directory probe cleanup failed (OSError)")


def test_two_concurrent_probes_do_not_collide(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: probe_directory_writable("data_dir", tmp_path), range(2)))

    assert all(check.ok for check in results)
    assert list(tmp_path.glob(".doctor-probe-*")) == []


def test_payload_symlink_fails_before_active_probe_and_redacts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    target = tmp_path / "private-target"
    target.mkdir()
    symlink = tmp_path / "secret-payload-link"
    symlink.symlink_to(target, target_is_directory=True)
    # WebSettings normally resolves paths during validation. model_copy keeps
    # the raw public field shape here so this unit test exercises doctor's own
    # symlink boundary independently of that earlier normalization layer.
    settings = _settings(tmp_path).model_copy(update={"payload_store_path": symlink})
    probed: list[Path | None] = []
    real_probe = doctor.probe_directory_writable

    def spy_probe(label: str, path: Path | None) -> ContractCheck:
        if label == "payload_store":
            probed.append(path)
        return real_probe(label, path)

    monkeypatch.setattr(doctor, "probe_directory_writable", spy_probe)

    check = _by_name(collect_checks(settings))["payload_store_writable"]

    assert check.ok is False
    assert probed == []
    assert str(symlink) not in check.detail
    assert str(target) not in check.detail


def test_group_or_world_writable_payload_fails_before_active_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    payload = tmp_path / "insecure-payload"
    payload.mkdir(mode=0o700)
    payload.chmod(0o770)
    settings = _settings(tmp_path, payload_store_path=payload)
    probed = False
    real_probe = doctor.probe_directory_writable

    def spy_probe(label: str, path: Path | None) -> ContractCheck:
        nonlocal probed
        if label == "payload_store":
            probed = True
        return real_probe(label, path)

    monkeypatch.setattr(doctor, "probe_directory_writable", spy_probe)

    check = _by_name(collect_checks(settings))["payload_store_writable"]

    assert check.ok is False
    assert probed is False
    assert str(payload) not in check.detail
    assert "group/world-writable" in check.detail


def test_collect_uses_raw_none_payload_path_not_derived_fallback(tmp_path: Path) -> None:
    settings = _settings(tmp_path, payload_store_path=None)
    derived_fallback = settings.get_payload_store_path()
    derived_fallback.mkdir(mode=0o700, exist_ok=True)

    check = _by_name(collect_checks(settings))["payload_store_writable"]

    assert check.ok is False
    assert list(derived_fallback.glob(".doctor-probe-*")) == []


def test_capability_failures_are_isolated_and_preserve_complete_report(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.plugins.infrastructure.manager as manager_module

    def fail_manager() -> object:
        raise RuntimeError("postgresql://user:password@private/db")  # secret-scan: allow-this-line

    monkeypatch.setattr(manager_module, "get_shared_plugin_manager", fail_manager)

    checks = plugin_and_dependency_checks()
    by_name = _by_name(checks)

    assert list(by_name) == [
        "aws_s3_plugin",
        "bedrock_provider",
        "aws_operator_telemetry",
        "bedrock_guardrail_plugins",
        "psycopg_dependency",
        "boto3_dependency",
        "ijson_dependency",
        "jinja2_dependency",
    ]
    assert by_name["aws_s3_plugin"].ok is False
    assert "password" not in by_name["aws_s3_plugin"].detail
    assert all(check.detail for check in checks)


def test_operator_telemetry_check_resolves_actual_effective_policy(tmp_path: Path) -> None:
    check = _aws_operator_telemetry_check(_settings(tmp_path))

    assert check == ContractCheck("aws_operator_telemetry", True, "AWS task-local OTLP operator policy is registered")


def test_operator_telemetry_check_rejects_policy_shape_and_endpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.operator_telemetry as operator_telemetry

    settings = _settings(tmp_path)
    effective = operator_telemetry.build_aws_operator_pipeline_telemetry(settings)
    exporter = effective.exporters[0]
    drifted_options = dict(exporter.options)
    drifted_options["headers"] = {"authorization": "private"}
    drifted_exporter = exporter.model_copy(update={"options": drifted_options})
    drifted_policy = effective.model_copy(update={"exporters": [drifted_exporter]})
    monkeypatch.setattr(
        operator_telemetry,
        "build_aws_operator_pipeline_telemetry",
        lambda _settings: drifted_policy,
    )

    assert _aws_operator_telemetry_check(settings).ok is False

    monkeypatch.setattr(operator_telemetry, "AWS_OTLP_ENDPOINT", "https://remote.invalid:4317")
    monkeypatch.setattr(
        operator_telemetry,
        "build_aws_operator_pipeline_telemetry",
        lambda web_settings: TelemetrySettings(
            enabled=True,
            granularity=web_settings.operator_pipeline_telemetry_granularity,
            exporters=[
                {
                    "name": "otlp",
                    "options": {
                        **exporter.options,
                        "endpoint": operator_telemetry.AWS_OTLP_ENDPOINT,
                    },
                }
            ],
        ),
    )

    assert _aws_operator_telemetry_check(settings).ok is False


def test_guardrail_registration_check_requires_positive_detection_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.plugins.infrastructure.manager as manager_module

    class _Manager:
        @staticmethod
        def get_transforms() -> list[object]:
            return [
                SimpleNamespace(
                    name="aws_bedrock_prompt_shield",
                    policy_capabilities=(
                        SimpleNamespace(
                            capability="prompt_shield",
                            control_role="input",
                            blocks_positive_detection=False,
                        ),
                    ),
                ),
                SimpleNamespace(
                    name="aws_bedrock_content_safety",
                    policy_capabilities=(
                        SimpleNamespace(
                            capability="content_safety",
                            control_role="output",
                            blocks_positive_detection=True,
                        ),
                    ),
                ),
            ]

    monkeypatch.setattr(manager_module, "get_shared_plugin_manager", _Manager)

    assert _bedrock_guardrail_plugins_check().ok is False


@pytest.mark.parametrize(
    ("module_name", "check_name"),
    [
        ("elspeth.plugins.transforms.llm.transform", "bedrock_provider"),
        ("psycopg", "psycopg_dependency"),
        ("boto3", "boto3_dependency"),
        ("ijson", "ijson_dependency"),
        ("jinja2", "jinja2_dependency"),
    ],
)
def test_each_lazy_import_failure_keeps_other_named_checks(
    module_name: str,
    check_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    real_import = doctor.importlib.import_module

    def selective_import(name: str, package: str | None = None) -> object:
        if name == module_name:
            raise RuntimeError("secret import failure /private/path")
        return real_import(name, package)

    monkeypatch.setattr(doctor.importlib, "import_module", selective_import)

    checks = plugin_and_dependency_checks()
    by_name = _by_name(checks)

    assert len(checks) == 8
    assert by_name[check_name].ok is False
    assert "secret import failure" not in by_name[check_name].detail
    assert "/private/path" not in by_name[check_name].detail


def test_collection_never_touches_auth_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    auth_db = tmp_path / "auth.db"
    auth_db.write_bytes(b"immutable-auth-sentinel")
    before_bytes = auth_db.read_bytes()
    before_stat = auth_db.stat()

    checks = collect_checks(_settings(tmp_path))

    after_stat = auth_db.stat()
    assert auth_db.read_bytes() == before_bytes
    assert after_stat == before_stat
    assert all("auth" not in check.name for check in checks)


def test_task1_check_names_are_exact_ordered_and_unique(tmp_path: Path) -> None:
    names = [check.name for check in collect_checks(_settings(tmp_path))]

    assert names == [
        "deployment_target",
        "session_db_url",
        "landscape_url",
        "data_dir",
        "payload_store_path",
        "operator_telemetry",
        "operator_telemetry_environment",
        "operator_telemetry_release",
        "operator_telemetry_ecs_cluster",
        "operator_telemetry_ecs_service",
        "operator_telemetry_task_definition_family",
        "operator_telemetry_task_definition_revision",
        "host",
        "secret_key",
        "shareable_link_signing_key",
        "separate_db_targets",
        "data_dir_writable",
        "payload_store_writable",
        "blob_writable",
        "aws_s3_plugin",
        "bedrock_provider",
        "aws_operator_telemetry",
        "bedrock_guardrail_plugins",
        "psycopg_dependency",
        "boto3_dependency",
        "ijson_dependency",
        "jinja2_dependency",
        "session_schema",
        "landscape_schema",
    ]
    assert len(names) == len(set(names))


def test_payload_contract_rejects_same_mode_bits_as_doctor(tmp_path: Path) -> None:
    from elspeth.core.payload_store import FilesystemPayloadStore

    payload = tmp_path / "payload"
    payload.mkdir(mode=0o700)
    payload.chmod(payload.stat().st_mode | stat.S_IWGRP)

    with pytest.raises(ValueError, match="group/world-writable"):
        FilesystemPayloadStore(payload)
    check = _by_name(collect_checks(_settings(tmp_path, payload_store_path=payload)))["payload_store_writable"]
    assert check.ok is False


@pytest.mark.parametrize(
    ("session_url", "landscape_url"),
    [
        (
            "postgresql+psycopg://gate_user:gate_password@gate-host/gate_database",
            "postgresql+psycopg://gate_user:gate_password@gate-host/gate_database",
        ),
        (
            "postgresql+psycopg://gate_user:gate_password@gate-host/gate_database",
            "postgresql+psycopg://gate_user:gate_password@gate-host/gate_database?options=-cstatement_timeout%3D10",
        ),
        (
            "postgresql+psycopg://gate_user:gate_password@gate-host/gate_database?options=-csearch_path%3Dsame_schema",
            "postgresql+psycopg://gate_user:gate_password@gate-host/gate_database?options=-csearch_path%3Dsame_schema",
        ),
    ],
)
def test_database_target_gate_fails_closed_and_redacts_target_material(session_url: str, landscape_url: str) -> None:
    check = database_target_check(session_url, landscape_url)

    assert check.ok is False
    assert check.name == "separate_db_targets"
    for fragment in ("gate_user", "gate_password", "gate-host", "gate_database", "same_schema", "statement_timeout"):
        assert fragment not in check.detail


def test_database_target_gate_accepts_distinct_explicit_schemas_in_same_database() -> None:
    check = database_target_check(
        "postgresql+psycopg://user:password@host/shared?options=-csearch_path%3Dsessions",
        "postgresql+psycopg://user:password@host/shared?options=-csearch_path%3Dlandscape",
    )

    assert check == ContractCheck("separate_db_targets", True, "session and Landscape targets are provably distinct")


def _patch_auxiliary_checks_green(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    monkeypatch.setattr(
        doctor,
        "probe_directory_writable",
        lambda label, _path: ContractCheck(f"{label}_writable", True, f"{label} ready"),
    )
    monkeypatch.setattr(
        doctor,
        "_probe_payload_store",
        lambda _path: ContractCheck("payload_store_writable", True, "payload_store ready"),
    )
    monkeypatch.setattr(
        doctor,
        "plugin_and_dependency_checks",
        lambda *, settings=None: [
            ContractCheck(name, True, "ready")
            for name in (
                "aws_s3_plugin",
                "bedrock_provider",
                "aws_operator_telemetry",
                "bedrock_guardrail_plugins",
                "psycopg_dependency",
                "boto3_dependency",
                "ijson_dependency",
                "jinja2_dependency",
            )
        ],
    )


def test_failing_target_gate_prevents_engine_probe_and_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    _patch_auxiliary_checks_green(monkeypatch)
    shared = "postgresql+psycopg://user:password@host/shared"
    settings = _settings(tmp_path, session_db_url=shared, landscape_url=shared)
    monkeypatch.setattr(doctor, "_inspect_database", MagicMock(side_effect=AssertionError("must not inspect")))
    monkeypatch.setattr(doctor, "_initialize_database", MagicMock(side_effect=AssertionError("must not initialize")))

    checks = _by_name(collect_checks(settings, init_schema=True))

    assert checks["separate_db_targets"].ok is False
    assert checks["session_schema"].ok is False
    assert checks["landscape_schema"].ok is False
    doctor._inspect_database.assert_not_called()
    doctor._initialize_database.assert_not_called()


def test_bad_url_contract_prevents_target_comparison_and_all_database_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    _patch_auxiliary_checks_green(monkeypatch)
    settings = _settings(tmp_path, session_db_url="sqlite:///forbidden.db")
    monkeypatch.setattr(doctor, "database_target_check", MagicMock(side_effect=AssertionError("must not compare")))
    monkeypatch.setattr(doctor, "_inspect_database", MagicMock(side_effect=AssertionError("must not inspect")))

    checks = _by_name(collect_checks(settings))

    assert checks["session_db_url"].ok is False
    assert checks["separate_db_targets"].ok is False
    doctor.database_target_check.assert_not_called()
    doctor._inspect_database.assert_not_called()


@pytest.mark.parametrize(
    ("label", "state", "ok", "detail"),
    [
        ("session_schema", SchemaState.CURRENT, True, "current"),
        ("session_schema", SchemaState.MISSING, False, "missing; rerun with --init-schema"),
        (
            "session_schema",
            SchemaState.STALE,
            False,
            "stale; drop and recreate the session database, then rerun doctor",
        ),
        ("landscape_schema", SchemaState.CURRENT, True, "current"),
        ("landscape_schema", SchemaState.MISSING, False, "missing; rerun with --init-schema"),
        ("landscape_schema", SchemaState.PARTIAL, False, "partial; rerun with --init-schema"),
        (
            "landscape_schema",
            SchemaState.STALE,
            False,
            "stale; drop and recreate the Landscape database, then rerun doctor",
        ),
    ],
)
def test_schema_state_details_are_static(label: str, state: SchemaState, ok: bool, detail: str) -> None:
    assert schema_check(label, state) == ContractCheck(label, ok, detail)


def _engine_with_connection() -> tuple[MagicMock, MagicMock]:
    engine = MagicMock()
    connection = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = connection
    context.__exit__.return_value = False
    engine.connect.return_value = context
    return engine, connection


@pytest.mark.parametrize("label", ["session_schema", "landscape_schema"])
def test_inspect_database_forwards_pool_and_timeout_and_uses_one_connection(
    label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    engine, connection = _engine_with_connection()
    session_factory = MagicMock(return_value=engine)
    landscape_factory = MagicMock(return_value=engine)
    monkeypatch.setattr(doctor, "create_session_engine", session_factory)
    monkeypatch.setattr(doctor, "create_engine", landscape_factory)
    probe = MagicMock(return_value=SchemaState.CURRENT)
    raw_url = "postgresql+psycopg://user:password@host/database"

    state, check = _inspect_database(label, raw_url, probe)

    assert state is SchemaState.CURRENT
    assert check == ContractCheck(label, True, "current")
    expected_kwargs = {
        "connect_args": {"connect_timeout": 10},
        "pool_size": 5,
        "max_overflow": 5,
        "pool_pre_ping": True,
    }
    expected_factory = session_factory if label == "session_schema" else landscape_factory
    expected_factory.assert_called_once_with(raw_url, **expected_kwargs)
    (landscape_factory if label == "session_schema" else session_factory).assert_not_called()
    probe.assert_called_once_with(connection)
    assert str(connection.execute.call_args.args[0]) == "SELECT 1"
    engine.dispose.assert_called_once_with()


@pytest.mark.parametrize("failure_site", ["connect", "probe"])
def test_inspect_database_disposes_after_connection_and_probe_failures(
    failure_site: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    engine, _connection = _engine_with_connection()
    if failure_site == "connect":
        engine.connect.return_value.__enter__.side_effect = RuntimeError(
            "postgresql://user:secret@private/db"  # secret-scan: allow-this-line
        )
    probe = MagicMock(
        side_effect=(
            RuntimeError("postgresql://user:secret@private/db")  # secret-scan: allow-this-line
            if failure_site == "probe"
            else None
        ),
        return_value=SchemaState.CURRENT,
    )
    monkeypatch.setattr(doctor, "create_session_engine", MagicMock(return_value=engine))

    state, check = _inspect_database(
        "session_schema",
        "postgresql+psycopg://user:password@host/database",
        probe,
    )

    assert state is None
    assert check.ok is False
    assert "secret" not in check.detail
    assert "private" not in check.detail
    engine.dispose.assert_called_once_with()


def test_initialize_database_runs_initializer_then_final_probe_on_new_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    engine, connection = _engine_with_connection()
    monkeypatch.setattr(doctor, "create_session_engine", MagicMock(return_value=engine))
    initializer = MagicMock()
    probe = MagicMock(return_value=SchemaState.CURRENT)

    check = _initialize_database(
        "session_schema",
        "postgresql+psycopg://user:password@host/database",
        probe,
        initializer,
    )

    assert check == ContractCheck(
        "session_schema",
        True,
        "current; initialization completed or was already completed",
    )
    initializer.assert_called_once_with(engine)
    probe.assert_called_once_with(connection)
    assert str(connection.execute.call_args.args[0]) == "SELECT 1"
    engine.dispose.assert_called_once_with()


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (
            SchemaInitBusyError("secret busy cause"),
            "another schema initialization is in progress; wait for it to finish and rerun",
        ),
        (
            SchemaLockCleanupError("secret cleanup cause"),
            "initialization may have completed but lock cleanup was not verified; investigate the database connection and rerun",
        ),
        (
            SessionSchemaError("secret compatibility cause"),
            "stale; drop and recreate the session database, then rerun doctor",
        ),
        (
            SchemaCompatibilityError("secret compatibility cause"),
            "stale; drop and recreate the session database, then rerun doctor",
        ),
        (RuntimeError("secret operational cause"), "session schema initialization failed (RuntimeError)"),
    ],
)
def test_initialize_database_catches_named_failures_redacts_and_disposes(
    error: Exception,
    detail: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    engine, _connection = _engine_with_connection()
    monkeypatch.setattr(doctor, "create_session_engine", MagicMock(return_value=engine))

    check = _initialize_database(
        "session_schema",
        "postgresql+psycopg://user:password@host/database",
        MagicMock(return_value=SchemaState.CURRENT),
        MagicMock(side_effect=error),
    )

    assert check == ContractCheck("session_schema", False, detail)
    assert "secret" not in check.detail
    engine.dispose.assert_called_once_with()


def test_initialize_database_disposes_when_final_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    engine, _connection = _engine_with_connection()
    monkeypatch.setattr(doctor, "create_session_engine", MagicMock(return_value=engine))

    check = _initialize_database(
        "session_schema",
        "postgresql+psycopg://user:password@host/database",
        MagicMock(side_effect=RuntimeError("private URL and credentials")),
        MagicMock(),
    )

    assert check == ContractCheck("session_schema", False, "session schema initialization failed (RuntimeError)")
    engine.dispose.assert_called_once_with()


def _patch_database_states(
    monkeypatch: pytest.MonkeyPatch,
    session: SchemaState | None,
    landscape: SchemaState | None,
    events: list[str],
) -> None:
    import elspeth.web.doctor as doctor

    def inspect(label: str, _url: str, _probe: object) -> tuple[SchemaState | None, ContractCheck]:
        events.append(f"inspect:{label}")
        state = session if label == "session_schema" else landscape
        if state is None:
            return None, ContractCheck(label, False, f"{label} connection failed (RuntimeError)")
        return state, schema_check(label, state)

    monkeypatch.setattr(doctor, "_inspect_database", inspect)


def test_init_mode_inspects_both_before_initializing_both_repairable_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    events: list[str] = []
    _patch_auxiliary_checks_green(monkeypatch)
    _patch_database_states(monkeypatch, SchemaState.MISSING, SchemaState.PARTIAL, events)

    def initialize(label: str, _url: str, _probe: object, _init: object) -> ContractCheck:
        events.append(f"init:{label}")
        return ContractCheck(label, True, "current; initialization completed or was already completed")

    monkeypatch.setattr(doctor, "_initialize_database", initialize)

    checks = _by_name(collect_checks(_settings(tmp_path), init_schema=True))

    assert checks["session_schema"].ok is True
    assert checks["landscape_schema"].ok is True
    assert events == [
        "inspect:session_schema",
        "inspect:landscape_schema",
        "init:session_schema",
        "init:landscape_schema",
    ]


@pytest.mark.parametrize(
    ("session", "landscape"),
    [
        (SchemaState.MISSING, SchemaState.STALE),
        (SchemaState.MISSING, None),
    ],
)
def test_stale_or_connection_failure_on_one_target_initializes_neither(
    session: SchemaState,
    landscape: SchemaState | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    events: list[str] = []
    _patch_auxiliary_checks_green(monkeypatch)
    _patch_database_states(monkeypatch, session, landscape, events)
    initializer = MagicMock(side_effect=AssertionError("must not initialize"))
    monkeypatch.setattr(doctor, "_initialize_database", initializer)

    checks = _by_name(collect_checks(_settings(tmp_path), init_schema=True))

    assert checks["session_schema"].ok is False
    assert "complete preflight failed" in checks["session_schema"].detail
    initializer.assert_not_called()
    assert events == ["inspect:session_schema", "inspect:landscape_schema"]


@pytest.mark.parametrize("failed_kind", ["filesystem", "dependency", "plugin"])
def test_any_auxiliary_preflight_failure_blocks_all_initializers(
    failed_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.doctor as doctor

    events: list[str] = []
    _patch_auxiliary_checks_green(monkeypatch)
    _patch_database_states(monkeypatch, SchemaState.MISSING, SchemaState.PARTIAL, events)
    if failed_kind == "filesystem":
        monkeypatch.setattr(
            doctor,
            "probe_directory_writable",
            lambda label, _path: ContractCheck(f"{label}_writable", label != "data_dir", "static result"),
        )
    else:
        failed_name = "boto3_dependency" if failed_kind == "dependency" else "aws_s3_plugin"
        monkeypatch.setattr(
            doctor,
            "plugin_and_dependency_checks",
            lambda *, settings=None: [
                ContractCheck(name, name != failed_name, "static result")
                for name in (
                    "aws_s3_plugin",
                    "bedrock_provider",
                    "aws_operator_telemetry",
                    "bedrock_guardrail_plugins",
                    "psycopg_dependency",
                    "boto3_dependency",
                    "ijson_dependency",
                    "jinja2_dependency",
                )
            ],
        )
    initializer = MagicMock(side_effect=AssertionError("must not initialize"))
    monkeypatch.setattr(doctor, "_initialize_database", initializer)

    checks = _by_name(collect_checks(_settings(tmp_path), init_schema=True))

    assert checks["session_schema"].ok is False
    assert checks["landscape_schema"].ok is False
    initializer.assert_not_called()


def test_one_current_target_initializes_only_other_repairable_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    events: list[str] = []
    _patch_auxiliary_checks_green(monkeypatch)
    _patch_database_states(monkeypatch, SchemaState.CURRENT, SchemaState.MISSING, events)

    def initialize(label: str, _url: str, _probe: object, _init: object) -> ContractCheck:
        events.append(f"init:{label}")
        return ContractCheck(label, True, "current; initialization completed or was already completed")

    monkeypatch.setattr(doctor, "_initialize_database", initialize)

    checks = _by_name(collect_checks(_settings(tmp_path), init_schema=True))

    assert checks["session_schema"] == ContractCheck("session_schema", True, "current")
    assert checks["landscape_schema"].ok is True
    assert events[-1] == "init:landscape_schema"
    assert "init:session_schema" not in events


def test_first_init_success_and_second_failure_are_reported_truthfully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.web.doctor as doctor

    events: list[str] = []
    _patch_auxiliary_checks_green(monkeypatch)
    _patch_database_states(monkeypatch, SchemaState.MISSING, SchemaState.PARTIAL, events)

    def initialize(label: str, _url: str, _probe: object, _init: object) -> ContractCheck:
        events.append(f"init:{label}")
        if label == "landscape_schema":
            return ContractCheck(label, False, "Landscape schema initialization failed (RuntimeError)")
        return ContractCheck(label, True, "current; initialization completed or was already completed")

    monkeypatch.setattr(doctor, "_initialize_database", initialize)

    checks = _by_name(collect_checks(_settings(tmp_path), init_schema=True))

    assert checks["session_schema"].ok is True
    assert checks["landscape_schema"].ok is False
    assert events[-2:] == ["init:session_schema", "init:landscape_schema"]


def test_task2_order_remains_exact_and_unique_after_database_inspection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    _patch_auxiliary_checks_green(monkeypatch)
    _patch_database_states(monkeypatch, SchemaState.CURRENT, SchemaState.CURRENT, events)

    names = [check.name for check in collect_checks(_settings(tmp_path))]

    assert names == [
        "deployment_target",
        "session_db_url",
        "landscape_url",
        "data_dir",
        "payload_store_path",
        "operator_telemetry",
        "operator_telemetry_environment",
        "operator_telemetry_release",
        "operator_telemetry_ecs_cluster",
        "operator_telemetry_ecs_service",
        "operator_telemetry_task_definition_family",
        "operator_telemetry_task_definition_revision",
        "host",
        "secret_key",
        "shareable_link_signing_key",
        "separate_db_targets",
        "data_dir_writable",
        "payload_store_writable",
        "blob_writable",
        "aws_s3_plugin",
        "bedrock_provider",
        "aws_operator_telemetry",
        "bedrock_guardrail_plugins",
        "psycopg_dependency",
        "boto3_dependency",
        "ijson_dependency",
        "jinja2_dependency",
        "session_schema",
        "landscape_schema",
    ]
    assert len(names) == len(set(names))
