"""Fail-closed AWS ECS web-startup validation tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretBytes
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from structlog.testing import capture_logs

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web import aws_ecs_startup as startup
from elspeth.web.config import WebSettings
from elspeth.web.deployment_contract import ContractCheck
from elspeth.web.schema_probe import DatabaseTargetConflictError, SchemaState
from elspeth.web.sessions.schema import SessionSchemaError

_SENTINEL = "opaque-credential SELECT raw_secret /secret/runtime/path"


def _settings(tmp_path: Path, **overrides: Any) -> WebSettings:
    data_dir = tmp_path / "data"
    payload_dir = tmp_path / "payload"
    blob_dir = data_dir / "blobs"
    for directory in (data_dir, payload_dir, blob_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    values: dict[str, Any] = {
        "deployment_target": "aws-ecs",
        "operator_telemetry": "aws-otlp",
        "operator_telemetry_environment": "test",
        "host": "0.0.0.0",
        "data_dir": data_dir,
        "payload_store_path": payload_dir,
        "session_db_url": "postgresql+psycopg://runtime:session-secret@db/session",
        "landscape_url": "postgresql+psycopg://runtime:landscape-secret@db/landscape",
        "secret_key": "s" * 40,
        "shareable_link_signing_key": SecretBytes(bytes(range(32))),
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
    }
    values.update(overrides)
    return WebSettings(**values)


def _assert_redacted(value: object) -> None:
    rendered = repr(value)
    assert "credential" not in rendered
    assert "raw_secret" not in rendered
    assert "/secret/runtime/path" not in rendered
    assert "session-secret" not in rendered
    assert "landscape-secret" not in rendered


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _Connection:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _AttemptEngine:
    def __init__(
        self, outcomes: list[BaseException | _Connection], *, clock: _Clock | None = None, costs: list[float] | None = None
    ) -> None:
        self._outcomes = iter(outcomes)
        self._clock = clock
        self._costs = iter(costs or [0.0] * len(outcomes))
        self.connect_calls = 0

    def connect(self) -> _Connection:
        self.connect_calls += 1
        if self._clock is not None:
            self._clock.now += next(self._costs)
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _DisposableEngine:
    def __init__(self, name: str = "engine") -> None:
        self.name = name
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1


def _operational_error() -> OperationalError:
    return OperationalError("SELECT raw_secret", {"credential": _SENTINEL}, RuntimeError(_SENTINEL))


@pytest.mark.parametrize("failed_name", ["session_db_url", "landscape_url"])
def test_contract_url_failure_prevents_target_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        startup,
        "validate_aws_ecs_settings",
        lambda _settings: [ContractCheck(failed_name, False, _SENTINEL)],
    )
    called = False

    def compare(_session: str, _landscape: str) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(startup, "require_distinct_postgres_targets", compare)

    with pytest.raises(startup.AwsEcsStartupContractError) as exc_info:
        startup.enforce_aws_ecs_contract(settings)

    assert failed_name in str(exc_info.value)
    assert called is False
    _assert_redacted(exc_info.value)


def test_contract_preserves_ordered_duplicate_failed_check_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        startup,
        "validate_aws_ecs_settings",
        lambda _settings: [
            ContractCheck("session_db_url", False, _SENTINEL),
            ContractCheck("session_db_url", False, _SENTINEL),
            ContractCheck("host", False, _SENTINEL),
        ],
    )

    with pytest.raises(startup.AwsEcsStartupContractError) as exc_info:
        startup.enforce_aws_ecs_contract(settings)

    assert str(exc_info.value).index("session_db_url, session_db_url") < str(exc_info.value).index("host")
    _assert_redacted(exc_info.value)


def test_target_conflict_is_translated_without_cause_or_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)

    def reject(_session: str, _landscape: str) -> None:
        raise DatabaseTargetConflictError(_SENTINEL)

    monkeypatch.setattr(startup, "require_distinct_postgres_targets", reject)
    with capture_logs() as logs, pytest.raises(startup.AwsEcsStartupContractError) as exc_info:
        startup.enforce_aws_ecs_contract(settings)

    assert "database targets" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    _assert_redacted(exc_info.value)
    _assert_redacted(logs)


@pytest.mark.parametrize(
    ("session_url", "landscape_url"),
    [
        ("postgresql://runtime@db/audit", "postgresql://runtime@db/audit"),
        (
            "postgresql://runtime@db/audit",
            "postgresql://runtime@db/audit?options=-csearch_path=landscape",
        ),
        (
            "postgresql://runtime@db/audit?options=-csearch_path=shared",
            "postgresql://runtime@db/audit?options=-csearch_path=shared",
        ),
    ],
)
def test_unproven_same_database_targets_raise_static_contract_error(
    tmp_path: Path,
    session_url: str,
    landscape_url: str,
) -> None:
    with pytest.raises(startup.AwsEcsStartupContractError) as exc_info:
        startup.enforce_aws_ecs_contract(_settings(tmp_path, session_db_url=session_url, landscape_url=landscape_url))

    assert "database targets" in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("session_url", "landscape_url"),
    [
        ("postgresql://db/session", "postgresql://db/landscape"),
        (
            "postgresql://db/audit?options=-csearch_path=sessions",
            "postgresql://db/audit?options=-csearch_path=landscape",
        ),
    ],
)
def test_distinct_database_targets_pass(tmp_path: Path, session_url: str, landscape_url: str) -> None:
    startup.enforce_aws_ecs_contract(_settings(tmp_path, session_db_url=session_url, landscape_url=landscape_url))


@pytest.mark.parametrize(
    ("mutate", "label", "env_var"),
    [
        (
            lambda settings: ((settings.data_dir / "blobs").rmdir(), settings.data_dir.rmdir()),
            "data_dir",
            "ELSPETH_WEB__DATA_DIR",
        ),
        (lambda settings: settings.payload_store_path.rmdir(), "payload_store", "ELSPETH_WEB__PAYLOAD_STORE_PATH"),
        (lambda settings: (settings.data_dir / "blobs").rmdir(), "blob", "ELSPETH_WEB__DATA_DIR"),
    ],
)
def test_each_missing_runtime_directory_fails_without_writing(
    tmp_path: Path,
    mutate: Callable[[WebSettings], None],
    label: str,
    env_var: str,
) -> None:
    settings = _settings(tmp_path)
    mutate(settings)
    auth_db = settings.data_dir / "auth.db"

    with pytest.raises(startup.AwsEcsStartupContractError) as exc_info:
        startup.require_runtime_directories_mounted(settings)

    assert label in str(exc_info.value)
    assert env_var in str(exc_info.value)
    if label == "data_dir":
        assert settings.data_dir.exists() is False
    elif label == "payload_store":
        assert settings.payload_store_path is not None
        assert settings.payload_store_path.exists() is False
    else:
        assert (settings.data_dir / "blobs").exists() is False
    assert auth_db.exists() is False
    assert str(settings.data_dir) not in str(exc_info.value)
    if settings.payload_store_path is not None:
        assert str(settings.payload_store_path) not in str(exc_info.value)
    _assert_redacted(exc_info.value)


def test_raw_payload_path_none_fails_without_using_fallback(tmp_path: Path) -> None:
    settings = _settings(tmp_path, payload_store_path=None)

    with pytest.raises(startup.AwsEcsStartupContractError, match="payload_store"):
        startup.require_runtime_directories_mounted(settings)


def test_payload_symlink_is_rejected_by_startup_and_payload_store(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.payload_store_path is not None
    target = tmp_path / "payload-target"
    settings.payload_store_path.rmdir()
    target.mkdir(mode=0o700)
    settings.payload_store_path.symlink_to(target, target_is_directory=True)

    with pytest.raises(startup.AwsEcsStartupContractError, match="payload_store"):
        startup.require_runtime_directories_mounted(settings)
    with pytest.raises(ValueError, match="symlink"):
        FilesystemPayloadStore(settings.payload_store_path)


def test_preexisting_payload_symlink_is_rejected_by_startup_and_payload_store(tmp_path: Path) -> None:
    target = tmp_path / "preexisting-target"
    target.mkdir(mode=0o700)
    configured_path = tmp_path / "preexisting-payload-link"
    configured_path.symlink_to(target, target_is_directory=True)
    settings = _settings(tmp_path, payload_store_path=configured_path)

    assert settings.payload_store_path == configured_path.absolute()
    with pytest.raises(startup.AwsEcsStartupContractError, match="payload_store"):
        startup.require_runtime_directories_mounted(settings)
    with pytest.raises(ValueError, match="symlink"):
        FilesystemPayloadStore(settings.payload_store_path)


@pytest.mark.parametrize("mode", [0o720, 0o702])
def test_payload_unsafe_mode_is_rejected_by_startup_and_payload_store(tmp_path: Path, mode: int) -> None:
    settings = _settings(tmp_path)
    assert settings.payload_store_path is not None
    settings.payload_store_path.chmod(mode)

    with pytest.raises(startup.AwsEcsStartupContractError, match="payload_store"):
        startup.require_runtime_directories_mounted(settings)
    with pytest.raises(ValueError, match="group/world-writable"):
        FilesystemPayloadStore(settings.payload_store_path)


@pytest.mark.parametrize("operation", ["lstat", "resolve"])
def test_secret_bearing_path_failures_are_static(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    settings = _settings(tmp_path)
    assert settings.payload_store_path is not None
    original = getattr(Path, operation)

    def fail(target: Path, *args: object, **kwargs: object) -> object:
        if target == settings.payload_store_path:
            raise OSError(_SENTINEL)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail)
    with capture_logs() as logs, pytest.raises(startup.AwsEcsStartupContractError) as exc_info:
        startup.require_runtime_directories_mounted(settings)

    _assert_redacted(exc_info.value)
    _assert_redacted(logs)


def test_operational_retries_use_new_connections_and_backoff() -> None:
    clock = _Clock()
    third = _Connection("third")
    engine = _AttemptEngine([_operational_error(), _operational_error(), third], clock=clock)
    probed: list[_Connection] = []

    state = startup._probe_with_connection_budget(
        engine,  # type: ignore[arg-type]
        lambda conn: probed.append(conn) or SchemaState.CURRENT,
        label="session_schema",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert state is SchemaState.CURRENT
    assert engine.connect_calls == 3
    assert probed == [third]
    assert clock.sleeps == [1.0, 2.0]


def test_noncurrent_state_is_returned_without_retry() -> None:
    clock = _Clock()
    engine = _AttemptEngine([_Connection("only")], clock=clock)

    state = startup._probe_with_connection_budget(
        engine,  # type: ignore[arg-type]
        lambda _conn: SchemaState.MISSING,
        label="session_schema",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert state is SchemaState.MISSING
    assert engine.connect_calls == 1
    assert clock.sleeps == []


def test_programming_exception_propagates_without_retry() -> None:
    clock = _Clock()
    engine = _AttemptEngine([_Connection("only")], clock=clock)

    with pytest.raises(TypeError, match="programmer bug"):
        startup._probe_with_connection_budget(
            engine,  # type: ignore[arg-type]
            lambda _conn: (_ for _ in ()).throw(TypeError("programmer bug")),
            label="session_schema",
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert engine.connect_calls == 1
    assert clock.sleeps == []


def test_backoff_caps_at_eight_seconds() -> None:
    clock = _Clock()
    engine = _AttemptEngine(
        [_operational_error(), _operational_error(), _operational_error(), _operational_error(), _operational_error(), _Connection("ok")],
        clock=clock,
    )

    state = startup._probe_with_connection_budget(
        engine,  # type: ignore[arg-type]
        lambda _conn: SchemaState.CURRENT,
        label="landscape_schema",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert state is SchemaState.CURRENT
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_sleep_is_clipped_to_reserve_connection_timeout() -> None:
    clock = _Clock()
    engine = _AttemptEngine(
        [_operational_error(), _operational_error(), _operational_error(), _operational_error(), _operational_error(), _Connection("ok")],
        clock=clock,
        costs=[0, 0, 0, 0, 18, 0],
    )

    state = startup._probe_with_connection_budget(
        engine,  # type: ignore[arg-type]
        lambda _conn: SchemaState.CURRENT,
        label="landscape_schema",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert state is SchemaState.CURRENT
    assert clock.sleeps[-1] == 2.0
    assert clock.now == 35.0


@pytest.mark.parametrize("elapsed", [36.0, 45.0])
def test_no_attempt_starts_without_reserved_timeout_or_at_deadline(elapsed: float) -> None:
    clock = _Clock()
    engine = _AttemptEngine([_operational_error(), _Connection("must-not-connect")], clock=clock, costs=[elapsed, 0])

    with pytest.raises(startup.AwsEcsSchemaNotReadyError) as exc_info:
        startup._probe_with_connection_budget(
            engine,  # type: ignore[arg-type]
            lambda _conn: SchemaState.CURRENT,
            label="session_schema",
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert engine.connect_calls == 1
    assert clock.sleeps == []
    assert exc_info.value.__cause__ is None
    _assert_redacted(exc_info.value)


@pytest.mark.parametrize(
    "error",
    [
        SQLAlchemyError(_SENTINEL),
        SessionSchemaError(_SENTINEL),
        SchemaCompatibilityError(_SENTINEL),
    ],
)
def test_nonoperational_database_and_schema_errors_are_translated(error: BaseException) -> None:
    clock = _Clock()
    engine = _AttemptEngine([_Connection("only")], clock=clock)

    with pytest.raises(startup.AwsEcsSchemaNotReadyError) as exc_info:
        startup._probe_with_connection_budget(
            engine,  # type: ignore[arg-type]
            lambda _conn: (_ for _ in ()).throw(error),
            label="landscape_schema",
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert engine.connect_calls == 1
    assert exc_info.value.__cause__ is None
    _assert_redacted(exc_info.value)


def test_terminal_operational_error_and_retry_logs_are_redacted() -> None:
    clock = _Clock()
    engine = _AttemptEngine([_operational_error()], clock=clock, costs=[45])

    with capture_logs() as logs, pytest.raises(startup.AwsEcsSchemaNotReadyError) as exc_info:
        startup._probe_with_connection_budget(
            engine,  # type: ignore[arg-type]
            lambda _conn: SchemaState.CURRENT,
            label="session_schema",
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert exc_info.value.__cause__ is None
    _assert_redacted(exc_info.value)
    _assert_redacted(logs)
    assert len(logs) == 1
    assert set(logs[0]) == {"event", "log_level", "label", "attempt", "elapsed_seconds", "exc_class"}
    assert logs[0]["event"] == "aws_ecs_schema_probe_retry"
    assert logs[0]["attempt"] == 1
    assert logs[0]["exc_class"] == "OperationalError"


def test_validate_only_probes_session_then_landscape_on_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    session_engine = _DisposableEngine("session")
    landscape_engine = _DisposableEngine("landscape")
    order: list[str] = []
    created: list[tuple[str, dict[str, object]]] = []

    def make_engine(url: str, **kwargs: object) -> _DisposableEngine:
        created.append((url, kwargs))
        return landscape_engine

    def probe(engine: _DisposableEngine, callback: Callable[[object], SchemaState], *, label: str, **_kwargs: object) -> SchemaState:
        order.append(label)
        expected = startup.probe_session_schema if label == "session_schema" else startup.probe_landscape_schema
        assert callback is expected
        assert engine is (session_engine if label == "session_schema" else landscape_engine)
        return SchemaState.CURRENT

    monkeypatch.setattr(startup, "create_engine", make_engine)
    monkeypatch.setattr(startup, "_probe_with_connection_budget", probe)

    startup.validate_only_schema_or_raise(settings, session_engine)  # type: ignore[arg-type]

    assert order == ["session_schema", "landscape_schema"]
    assert created == [
        (
            settings.landscape_url,
            {"connect_args": {"connect_timeout": 10}, "pool_size": 5, "max_overflow": 5, "pool_pre_ping": True},
        )
    ]
    assert landscape_engine.dispose_calls == 1
    assert session_engine.dispose_calls == 0


@pytest.mark.parametrize("state", [SchemaState.MISSING, SchemaState.PARTIAL, SchemaState.STALE])
def test_session_noncurrent_stops_before_landscape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: SchemaState) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(startup, "_probe_with_connection_budget", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(startup, "create_engine", lambda *_args, **_kwargs: pytest.fail("Landscape engine must not be built"))

    with pytest.raises(startup.AwsEcsSchemaNotReadyError, match="session_schema"):
        startup.validate_only_schema_or_raise(settings, _DisposableEngine("session"))  # type: ignore[arg-type]


@pytest.mark.parametrize("state", [SchemaState.MISSING, SchemaState.PARTIAL, SchemaState.STALE])
def test_landscape_noncurrent_fails_and_disposes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: SchemaState) -> None:
    settings = _settings(tmp_path)
    landscape_engine = _DisposableEngine("landscape")
    states = iter([SchemaState.CURRENT, state])
    monkeypatch.setattr(startup, "_probe_with_connection_budget", lambda *_args, **_kwargs: next(states))
    monkeypatch.setattr(startup, "create_engine", lambda *_args, **_kwargs: landscape_engine)

    with pytest.raises(startup.AwsEcsSchemaNotReadyError, match="landscape_schema"):
        startup.validate_only_schema_or_raise(settings, _DisposableEngine("session"))  # type: ignore[arg-type]

    assert landscape_engine.dispose_calls == 1


@pytest.mark.parametrize("error", [startup.AwsEcsSchemaNotReadyError("static"), KeyboardInterrupt()])
def test_landscape_engine_disposed_for_probe_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    settings = _settings(tmp_path)
    landscape_engine = _DisposableEngine("landscape")
    calls = 0

    def probe(*_args: object, **_kwargs: object) -> SchemaState:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SchemaState.CURRENT
        raise error

    monkeypatch.setattr(startup, "_probe_with_connection_budget", probe)
    monkeypatch.setattr(startup, "create_engine", lambda *_args, **_kwargs: landscape_engine)

    with pytest.raises(type(error)):
        startup.validate_only_schema_or_raise(settings, _DisposableEngine("session"))  # type: ignore[arg-type]

    assert landscape_engine.dispose_calls == 1
