"""PostgreSQL 16 proofs for schema classification and locked initialization."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Connection, Engine, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from testcontainers.postgres import PostgresContainer
from tests.unit.core.test_schema_shape import _static_check_issues

from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.schema_shape import _text_builtin_identity_rows_on_connection
from elspeth.web import schema_probe as schema_probe_module
from elspeth.web.schema_probe import (
    SchemaInitBusyError,
    SchemaState,
    _run_locked,
    init_landscape_schema,
    init_session_schema,
    probe_landscape_schema,
    probe_session_schema,
)
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.schema import SessionSchemaError

pytestmark = pytest.mark.testcontainer


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def postgres_engine(postgres_url: str) -> Iterator[Engine]:
    identifier = f"elspeth_schema_{uuid.uuid4().hex}"
    assert re.fullmatch(r"[a-z0-9_]+", identifier)
    admin = create_engine(postgres_url)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(f'CREATE DATABASE "{identifier}"')
    engine = create_engine(make_url(postgres_url).set(database=identifier))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{identifier}" WITH (FORCE)')
        admin.dispose()


@pytest.mark.parametrize("kind", ["session", "landscape"])
def test_fresh_create_reaches_current(postgres_engine: Engine, kind: str) -> None:
    if kind == "session":
        assert probe_session_schema(postgres_engine) is SchemaState.MISSING
        init_session_schema(postgres_engine)
        assert probe_session_schema(postgres_engine) is SchemaState.CURRENT
    else:
        assert probe_landscape_schema(postgres_engine) is SchemaState.MISSING
        init_landscape_schema(postgres_engine)
        assert probe_landscape_schema(postgres_engine) is SchemaState.CURRENT


def test_landscape_server_default_is_false(postgres_engine: Engine) -> None:
    init_landscape_schema(postgres_engine)
    values = {
        "run_id": "server-default-run",
        "config_hash": "0" * 64,
        "settings_json": "{}",
        "canonical_version": "test-v1",
        "status": "running",
        "openrouter_catalog_sha256": "0" * 64,
        "openrouter_catalog_source": "live",
    }
    with postgres_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO runs (
                    run_id, started_at, config_hash, settings_json,
                    canonical_version, status, openrouter_catalog_sha256,
                    openrouter_catalog_source
                ) VALUES (
                    :run_id, CURRENT_TIMESTAMP, :config_hash, :settings_json,
                    :canonical_version, :status, :openrouter_catalog_sha256,
                    :openrouter_catalog_source
                )
                """
            ),
            values,
        )
        assert conn.execute(text("SELECT seeded_from_cache FROM runs WHERE run_id='server-default-run'")).scalar_one() is False


@pytest.mark.parametrize("object_name", ["auth_events", "run_attributions", "ix_tokens_run_id"])
def test_additive_gap_is_repaired(postgres_engine: Engine, object_name: str) -> None:
    init_landscape_schema(postgres_engine)
    with postgres_engine.begin() as conn:
        if object_name.startswith("ix_"):
            conn.exec_driver_sql(f"DROP INDEX {object_name}")
        else:
            conn.exec_driver_sql(f"DROP TABLE {object_name}")
    assert probe_landscape_schema(postgres_engine) is SchemaState.PARTIAL
    init_landscape_schema(postgres_engine)
    assert probe_landscape_schema(postgres_engine) is SchemaState.CURRENT


def test_missing_core_landscape_table_is_stale_and_not_repaired(postgres_engine: Engine) -> None:
    init_landscape_schema(postgres_engine)
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE validation_errors")
    assert probe_landscape_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SchemaCompatibilityError):
        init_landscape_schema(postgres_engine)
    assert "validation_errors" not in inspect(postgres_engine).get_table_names()


@pytest.mark.parametrize(
    "mutation",
    [
        "ALTER TABLE runs ALTER COLUMN config_hash TYPE text",
        "ALTER TABLE runs ALTER COLUMN config_hash DROP NOT NULL",
        "ALTER TABLE runs ALTER COLUMN config_hash SET DEFAULT 'changed'",
        "DROP INDEX ix_tokens_run_id; CREATE INDEX ix_tokens_run_id ON tokens (row_id)",
        "ALTER TABLE runs DROP CONSTRAINT ck_runs_openrouter_catalog_source",
        "DROP INDEX ix_token_outcomes_terminal_unique; CREATE UNIQUE INDEX ix_token_outcomes_terminal_unique ON token_outcomes (token_id) WHERE completed = 0",
    ],
)
def test_landscape_reflection_drift_is_stale_and_not_repaired(postgres_engine: Engine, mutation: str) -> None:
    init_landscape_schema(postgres_engine)
    with postgres_engine.begin() as conn:
        for statement in mutation.split(";"):
            conn.exec_driver_sql(statement)
    assert probe_landscape_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SchemaCompatibilityError):
        init_landscape_schema(postgres_engine)


@pytest.mark.parametrize("constraint_type", ["u", "f"])
def test_landscape_missing_unique_or_fk_is_stale(postgres_engine: Engine, constraint_type: str) -> None:
    init_landscape_schema(postgres_engine)
    table_name = "run_sources" if constraint_type == "u" else "tokens"
    with postgres_engine.begin() as conn:
        constraint_name = conn.execute(
            text(
                """
                SELECT conname FROM pg_constraint
                WHERE conrelid = CAST(:table_name AS regclass) AND contype = :constraint_type
                ORDER BY conname LIMIT 1
                """
            ),
            {"table_name": table_name, "constraint_type": constraint_type},
        ).scalar_one()
        assert re.fullmatch(r"[a-z0-9_]+", constraint_name)
        conn.exec_driver_sql(f'ALTER TABLE "{table_name}" DROP CONSTRAINT "{constraint_name}"')
    assert probe_landscape_schema(postgres_engine) is SchemaState.STALE


@pytest.mark.parametrize("kind", ["session", "landscape"])
def test_foreign_target_is_stale_and_nonmutating(postgres_engine: Engine, kind: str) -> None:
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE unrelated (id integer primary key)")
    if kind == "session":
        assert probe_session_schema(postgres_engine) is SchemaState.STALE
        with pytest.raises(SessionSchemaError):
            init_session_schema(postgres_engine)
    else:
        assert probe_landscape_schema(postgres_engine) is SchemaState.STALE
        with pytest.raises(SchemaCompatibilityError):
            init_landscape_schema(postgres_engine)
    assert inspect(postgres_engine).get_table_names() == ["unrelated"]


def test_session_partial_schema_is_stale_and_nonmutating(postgres_engine: Engine) -> None:
    session_metadata.tables["sessions"].create(postgres_engine)
    before = inspect(postgres_engine).get_table_names()
    assert probe_session_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SessionSchemaError):
        init_session_schema(postgres_engine)
    assert inspect(postgres_engine).get_table_names() == before


def test_session_wrong_shape_is_stale_and_nonmutating(postgres_engine: Engine) -> None:
    init_session_schema(postgres_engine)
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE sessions ALTER COLUMN title TYPE text")
    assert probe_session_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SessionSchemaError):
        init_session_schema(postgres_engine)


@pytest.mark.parametrize("kind", ["session", "landscape"])
def test_concurrent_initializers_serialize_to_one_create_all(
    postgres_engine: Engine,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = session_metadata if kind == "session" else schema_probe_module.landscape_metadata
    initializer = init_session_schema if kind == "session" else init_landscape_schema
    original_create_all = metadata.create_all
    holder_entered = threading.Event()
    release_holder = threading.Event()
    call_count = 0
    count_lock = threading.Lock()

    def held_create_all(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        with count_lock:
            call_count += 1
        holder_entered.set()
        assert release_holder.wait(timeout=10)
        original_create_all(*args, **kwargs)

    monkeypatch.setattr(metadata, "create_all", held_create_all)
    failures: list[BaseException] = []

    def worker() -> None:
        try:
            initializer(postgres_engine)
        except BaseException as exc:
            failures.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker)
        assert holder_entered.wait(timeout=10)
        second = executor.submit(worker)
        deadline = time.monotonic() + 10
        observed_waiter = False
        try:
            while time.monotonic() < deadline:
                with postgres_engine.connect() as observer:
                    rows = (
                        observer.execute(
                            text(
                                """
                            SELECT granted FROM pg_locks
                            WHERE locktype='advisory' AND classid=:classid
                              AND objid = hashtext('elspeth_schema_init')::oid
                            """
                            ),
                            {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
                        )
                        .scalars()
                        .all()
                    )
                if rows.count(True) == 1 and rows.count(False) == 1:
                    observed_waiter = True
                    break
                time.sleep(0.05)
            assert observed_waiter
        finally:
            release_holder.set()
            first.result(timeout=15)
            second.result(timeout=15)
    assert failures == []
    assert call_count == 1
    assert (probe_session_schema if kind == "session" else probe_landscape_schema)(postgres_engine) is SchemaState.CURRENT


def test_lock_timeout_maps_to_redacted_busy_error(
    postgres_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(schema_probe_module, "_LOCK_TIMEOUT", "250ms")
    holder = postgres_engine.connect()
    holder.execute(
        text("SELECT pg_advisory_lock(:classid, hashtext('elspeth_schema_init'))"),
        {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
    )
    try:
        with pytest.raises(SchemaInitBusyError) as exc_info:
            init_session_schema(postgres_engine)
        assert isinstance(exc_info.value.__cause__, OperationalError)
        assert "postgresql" not in str(exc_info.value).lower()
        assert inspect(postgres_engine).get_table_names() == []
    finally:
        holder.execute(
            text("SELECT pg_advisory_unlock(:classid, hashtext('elspeth_schema_init'))"),
            {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
        )
        holder.close()


class _CustomBaseException(BaseException):
    pass


@pytest.mark.parametrize("failure_kind", ["sqlstate_22012", "base_exception"])
def test_lock_is_released_after_body_failure(postgres_engine: Engine, failure_kind: str) -> None:
    def body(conn: Connection) -> None:
        if failure_kind == "sqlstate_22012":
            conn.exec_driver_sql("SELECT 1 / 0")
        raise _CustomBaseException

    expected = SQLAlchemyError if failure_kind == "sqlstate_22012" else _CustomBaseException
    with pytest.raises(expected):
        _run_locked(
            postgres_engine,
            target="elspeth_schema_init",
            body=body,
            verify=lambda _conn: pytest.fail("verify must not run after body failure"),
        )

    with postgres_engine.connect() as observer:
        acquired = observer.execute(
            text("SELECT pg_try_advisory_lock(:classid, hashtext('elspeth_schema_init'))"),
            {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
        ).scalar_one()
        assert acquired is True
        assert (
            observer.execute(
                text("SELECT pg_advisory_unlock(:classid, hashtext('elspeth_schema_init'))"),
                {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
            ).scalar_one()
            is True
        )


@pytest.mark.parametrize("kind", ["session", "landscape"])
def test_size_one_pool_initializer_reaches_current(postgres_url: str, kind: str) -> None:
    identifier = f"elspeth_schema_{uuid.uuid4().hex}"
    assert re.fullmatch(r"[a-z0-9_]+", identifier)
    admin = create_engine(postgres_url)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(f'CREATE DATABASE "{identifier}"')
    engine = create_engine(make_url(postgres_url).set(database=identifier), pool_size=1, max_overflow=0)
    try:
        if kind == "session":
            init_session_schema(engine)
            assert probe_session_schema(engine) is SchemaState.CURRENT
        else:
            init_landscape_schema(engine)
            assert probe_landscape_schema(engine) is SchemaState.CURRENT
    finally:
        engine.dispose()
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{identifier}" WITH (FORCE)')
        admin.dispose()


def test_catalog_proof_resists_shadowed_oid_equality(postgres_engine: Engine) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA equality_shadow"))
        connection.execute(
            text(
                "CREATE FUNCTION equality_shadow.always_equal(left_oid oid, right_oid oid) RETURNS boolean LANGUAGE sql IMMUTABLE AS 'SELECT true'"
            )
        )
        connection.execute(
            text("CREATE OPERATOR equality_shadow.= (FUNCTION = equality_shadow.always_equal, LEFTARG = oid, RIGHTARG = oid)")
        )
        connection.execute(
            text("CREATE FUNCTION equality_shadow.chr(codepoint integer) RETURNS integer LANGUAGE sql IMMUTABLE AS 'SELECT codepoint'")
        )
        connection.execute(text("SET LOCAL search_path = equality_shadow, pg_catalog, public"))
        rows = _text_builtin_identity_rows_on_connection(connection)
        assert rows is not None
        assert ("text_result", "chr", 1, "int4") not in rows
        issues = _static_check_issues(
            "btrim(value_text, chr(49)) IS NOT NULL",
            "btrim(value_text::text, chr(49)) IS NOT NULL",
            builtin_connection=connection,
        )
    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)


def test_catalog_proof_resists_shadowed_text_concatenation(postgres_engine: Engine) -> None:
    with postgres_engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA concat_shadow"))
        connection.execute(
            text(
                "CREATE FUNCTION concat_shadow.text_text(left_value text, right_value text) RETURNS integer LANGUAGE sql IMMUTABLE AS 'SELECT 0'"
            )
        )
        connection.execute(text("CREATE OPERATOR concat_shadow.|| (FUNCTION = concat_shadow.text_text, LEFTARG = text, RIGHTARG = text)"))
        connection.execute(
            text(
                "CREATE FUNCTION concat_shadow.integer_text(left_value integer, right_value text) RETURNS text LANGUAGE sql IMMUTABLE AS 'SELECT ''text,text''::text'"
            )
        )
        connection.execute(
            text("CREATE OPERATOR concat_shadow.|| (FUNCTION = concat_shadow.integer_text, LEFTARG = integer, RIGHTARG = text)")
        )
        connection.execute(
            text(
                "CREATE FUNCTION concat_shadow.text_varchar(left_value text, right_value varchar) RETURNS text LANGUAGE sql IMMUTABLE AS 'SELECT left_value'"
            )
        )
        connection.execute(
            text("CREATE OPERATOR concat_shadow.|| (FUNCTION = concat_shadow.text_varchar, LEFTARG = text, RIGHTARG = varchar)")
        )
        connection.execute(text("SET LOCAL search_path = concat_shadow, pg_catalog, public"))
        rows = _text_builtin_identity_rows_on_connection(connection)
        assert rows is not None
        assert ("operator_text_result", "||", 2, "text,text") not in rows
        issues = _static_check_issues(
            "btrim(value_text, chr(49) || chr(50)) IS NOT NULL",
            "btrim(value_text::text, chr(49) || chr(50)) IS NOT NULL",
            builtin_connection=connection,
        )
    assert any(issue.subject.endswith("CHECK constraint SQL mismatch") for issue in issues)
