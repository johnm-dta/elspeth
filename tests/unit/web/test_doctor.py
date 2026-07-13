"""Unit tests for the AWS ECS doctor report and its safety boundaries."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import ContractCheck
from elspeth.web.doctor import (
    collect_checks,
    plugin_and_dependency_checks,
    probe_directory_writable,
    sanitize_error,
)


def _settings(tmp_path: Path, **overrides: Any) -> WebSettings:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payloads"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "blobs").mkdir(exist_ok=True)
    payload_dir.mkdir(mode=0o700, exist_ok=True)
    values: dict[str, Any] = {
        "deployment_target": "aws-ecs",
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
