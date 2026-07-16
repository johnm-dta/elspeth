"""PostgreSQL 16 proofs for schema classification and locked initialization."""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import structlog
from sqlalchemy import Connection, Engine, create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer
from tests.unit.core.test_schema_shape import _static_check_issues

from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.database import SchemaCompatibilityError
from elspeth.core.schema_shape import _text_builtin_identity_rows_on_connection
from elspeth.web import schema_probe as schema_probe_module
from elspeth.web.preferences.models import UpdateComposerPreferencesRequest
from elspeth.web.preferences.service import PreferencesService
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
from elspeth.web.sessions.models import skill_markdown_history_table
from elspeth.web.sessions.schema import SessionSchemaError, initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

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


def test_postgres_session_init_does_not_poison_later_sqlite_schema(postgres_engine: Engine) -> None:
    init_session_schema(postgres_engine)

    sqlite_engine = create_engine("sqlite:///:memory:")
    initialize_session_schema(sqlite_engine)

    assert inspect(sqlite_engine).get_foreign_keys("chat_messages")


def _seed_postgres_trigger_rows(postgres_engine: Engine, *, session_id: str, include_completion: bool) -> None:
    with postgres_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sessions (
                    id, user_id, auth_provider_type, title, trust_mode,
                    density_default, created_at, updated_at,
                    interpretation_review_disabled
                ) VALUES (
                    :session_id, 'trigger-user', 'local', 'Trigger proof',
                    'auto_commit', 'high', CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, false
                )
                """
            ),
            {"session_id": session_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO composition_states (
                    id, session_id, version, is_valid, created_at, provenance
                ) VALUES (
                    :state_id, :session_id, 1, false, CURRENT_TIMESTAMP,
                    'session_seed'
                )
                """
            ),
            {"state_id": f"{session_id}-state", "session_id": session_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO chat_messages (
                    id, session_id, role, content, sequence_no,
                    writer_principal, created_at
                ) VALUES (
                    :message_id, :session_id, 'user', 'original', 1,
                    'route_user_message', CURRENT_TIMESTAMP
                )
                """
            ),
            {"message_id": f"{session_id}-message", "session_id": session_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO interpretation_events (
                    id, session_id, choice, created_at, resolved_at, actor,
                    interpretation_source
                ) VALUES (
                    :event_id, :session_id, 'opted_out', CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, 'trigger-user',
                    'auto_interpreted_opt_out'
                )
                """
            ),
            {"event_id": f"{session_id}-interpretation", "session_id": session_id},
        )
        if include_completion:
            conn.execute(
                text(
                    """
                    INSERT INTO composer_completion_events (
                        id, session_id, composition_state_id, event_type,
                        actor, created_at
                    ) VALUES (
                        :event_id, :session_id, :state_id, 'export_yaml',
                        'trigger-user', CURRENT_TIMESTAMP
                    )
                    """
                ),
                {
                    "event_id": f"{session_id}-completion",
                    "session_id": session_id,
                    "state_id": f"{session_id}-state",
                },
            )


def test_postgres_session_audit_triggers_are_installed_and_enforced(postgres_engine: Engine) -> None:
    init_session_schema(postgres_engine)

    with postgres_engine.connect() as conn:
        names = {
            str(row[0])
            for row in conn.execute(
                text(
                    """
                    SELECT trigger.tgname
                    FROM pg_catalog.pg_trigger AS trigger
                    JOIN pg_catalog.pg_class AS relation
                      ON relation.oid = trigger.tgrelid
                    JOIN pg_catalog.pg_namespace AS namespace
                      ON namespace.oid = relation.relnamespace
                    WHERE NOT trigger.tgisinternal
                      AND namespace.nspname = current_schema()
                    """
                )
            )
        }
    assert names == {
        "trg_interpretation_events_immutable_resolved",
        "trg_interpretation_events_no_delete_resolved",
        "trg_composer_completion_events_no_update",
        "trg_composer_completion_events_no_delete",
        "trg_chat_messages_immutable_content",
        "trg_chat_messages_no_delete",
    }

    protected_session = "trigger-protected"
    _seed_postgres_trigger_rows(postgres_engine, session_id=protected_session, include_completion=True)

    blocked_mutations = (
        ("UPDATE interpretation_events SET actor = 'attacker' WHERE id = :row_id", f"{protected_session}-interpretation"),
        ("DELETE FROM interpretation_events WHERE id = :row_id", f"{protected_session}-interpretation"),
        ("UPDATE composer_completion_events SET actor = 'attacker' WHERE id = :row_id", f"{protected_session}-completion"),
        ("DELETE FROM composer_completion_events WHERE id = :row_id", f"{protected_session}-completion"),
        ("UPDATE chat_messages SET content = 'tampered' WHERE id = :row_id", f"{protected_session}-message"),
        ("DELETE FROM chat_messages WHERE id = :row_id", f"{protected_session}-message"),
    )
    for statement, row_id in blocked_mutations:
        with pytest.raises(DBAPIError, match=r"append-only|immutable"), postgres_engine.begin() as conn:
            conn.execute(text(statement), {"row_id": row_id})

    with pytest.raises(DBAPIError, match="append-only"), postgres_engine.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE id = :session_id"), {"session_id": protected_session})

    with postgres_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO interpretation_events (
                    id, session_id, composition_state_id, affected_node_id,
                    tool_call_id, user_term, kind, llm_draft, choice,
                    created_at, actor, model_identifier, model_version,
                    provider, composer_skill_hash, interpretation_source
                ) VALUES (
                    :event_id, :session_id, :state_id, 'llm-node',
                    'pending-tool-call', 'term', 'vague_term', 'draft',
                    'pending', CURRENT_TIMESTAMP, 'trigger-user', 'model-id',
                    'model-version', 'provider', :skill_hash, 'user_approved'
                )
                """
            ),
            {
                "event_id": f"{protected_session}-pending",
                "session_id": protected_session,
                "state_id": f"{protected_session}-state",
                "skill_hash": "0" * 64,
            },
        )
        conn.execute(
            text("DELETE FROM interpretation_events WHERE id = :event_id"),
            {"event_id": f"{protected_session}-pending"},
        )

    cascade_session = "trigger-cascade"
    _seed_postgres_trigger_rows(postgres_engine, session_id=cascade_session, include_completion=False)
    with postgres_engine.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE id = :session_id"), {"session_id": cascade_session})
        assert (
            conn.execute(
                text("SELECT count(*) FROM interpretation_events WHERE session_id = :session_id"),
                {"session_id": cascade_session},
            ).scalar_one()
            == 0
        )
        assert (
            conn.execute(
                text("SELECT count(*) FROM chat_messages WHERE session_id = :session_id"),
                {"session_id": cascade_session},
            ).scalar_one()
            == 0
        )


@pytest.mark.parametrize(
    "trigger_mutation",
    [
        "DROP TRIGGER trg_chat_messages_no_delete ON chat_messages",
        "ALTER TABLE chat_messages DISABLE TRIGGER trg_chat_messages_no_delete",
    ],
)
def test_missing_or_disabled_postgres_audit_trigger_marks_session_schema_stale(
    postgres_engine: Engine,
    trigger_mutation: str,
) -> None:
    init_session_schema(postgres_engine)
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql(trigger_mutation)

    assert probe_session_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SessionSchemaError, match="trigger"):
        initialize_session_schema(postgres_engine)


def test_preferences_upsert_round_trips_on_postgres(postgres_engine: Engine) -> None:
    """The account preferences write path must use PostgreSQL's upsert builder."""
    init_session_schema(postgres_engine)
    service = PreferencesService(postgres_engine)

    transition = asyncio.run(
        service.update_composer_preferences(
            "postgres-preferences-user",
            UpdateComposerPreferencesRequest(default_mode="guided", tutorial_completed_at=None),
        )
    )

    assert transition.prior is None
    assert transition.current.default_mode == "guided"
    assert transition.current.tutorial_completed_at is None
    assert asyncio.run(service.get_composer_preferences("postgres-preferences-user")) == transition.current


def test_skill_markdown_history_upsert_round_trips_on_postgres(postgres_engine: Engine, tmp_path: Path) -> None:
    """Composer startup history writes must use PostgreSQL's upsert builder."""
    init_session_schema(postgres_engine)
    service = SessionServiceImpl(
        postgres_engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )

    first_inserted = asyncio.run(
        service.upsert_skill_markdown_history(
            skill_hash="a" * 64,
            filename="pipeline_composer.md",
            content="# Composer skill",
        )
    )
    duplicate_inserted = asyncio.run(
        service.upsert_skill_markdown_history(
            skill_hash="a" * 64,
            filename="pipeline_composer.md",
            content="# Composer skill",
        )
    )

    with postgres_engine.connect() as conn:
        rows = conn.execute(select(skill_markdown_history_table)).fetchall()
    assert first_inserted is True
    assert duplicate_inserted is False
    assert len(rows) == 1


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


def test_run_source_contract_hash_column_fits_runtime_hash(postgres_engine: Engine) -> None:
    init_landscape_schema(postgres_engine)
    runtime_hash = SchemaContract(mode="OBSERVED", fields=(), locked=True).version_hash()

    with postgres_engine.connect() as conn:
        column_width = conn.execute(
            text(
                """
                SELECT character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'run_sources'
                  AND column_name = 'schema_contract_hash'
                """
            )
        ).scalar_one()

    assert column_width == len(runtime_hash)


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


def test_populated_pre_epoch_23_landscape_missing_policy_table_is_stale_without_catalog_mutation(
    postgres_engine: Engine,
) -> None:
    init_landscape_schema(postgres_engine)
    with postgres_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO runs (
                    run_id, started_at, config_hash, settings_json,
                    canonical_version, status, openrouter_catalog_sha256,
                    openrouter_catalog_source
                ) VALUES (
                    'pre-epoch-23', CURRENT_TIMESTAMP, :config_hash, '{}',
                    'v1', 'completed', :catalog_hash, 'bundled'
                )
                """
            ),
            {"config_hash": "a" * 64, "catalog_hash": "b" * 64},
        )
        conn.exec_driver_sql("DROP TABLE run_web_plugin_policy")
    before_tables = inspect(postgres_engine).get_table_names()

    assert probe_landscape_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SchemaCompatibilityError):
        init_landscape_schema(postgres_engine)

    assert inspect(postgres_engine).get_table_names() == before_tables
    with postgres_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM runs WHERE run_id='pre-epoch-23'")).scalar_one() == 1


def test_landscape_runtime_role_has_dml_but_no_ddl(postgres_engine: Engine) -> None:
    init_landscape_schema(postgres_engine)
    role = f"elspeth_runtime_{uuid.uuid4().hex}"
    password = f"runtime-{uuid.uuid4().hex}"
    database_name = postgres_engine.url.database
    assert database_name is not None and re.fullmatch(r"[a-z0-9_]+", database_name)
    assert re.fullmatch(r"[a-z0-9_]+", role)
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
        conn.exec_driver_sql(f'GRANT CONNECT ON DATABASE "{database_name}" TO "{role}"')
        conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA public TO "{role}"')
        conn.exec_driver_sql(f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{role}"')
        conn.exec_driver_sql(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{role}"')

    runtime = create_engine(postgres_engine.url.set(username=role, password=password))
    try:
        with runtime.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM runs")).scalar_one() == 0
        with pytest.raises(SQLAlchemyError), runtime.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE runtime_must_not_create (id integer primary key)")
    finally:
        runtime.dispose()
        with postgres_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP OWNED BY "{role}"')
            conn.exec_driver_sql(f'DROP ROLE "{role}"')


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


def test_postgres_sqlite_sequence_is_foreign_and_nonmutating(postgres_engine: Engine) -> None:
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sqlite_sequence (name text)")
    before = inspect(postgres_engine).get_table_names()

    assert probe_session_schema(postgres_engine) is SchemaState.STALE
    with pytest.raises(SessionSchemaError):
        init_session_schema(postgres_engine)
    assert inspect(postgres_engine).get_table_names() == before


def test_session_partial_schema_is_stale_and_nonmutating(postgres_engine: Engine) -> None:
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sessions (id VARCHAR PRIMARY KEY)")
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


def test_schema_init_lock_functions_cannot_be_shadowed(postgres_engine: Engine) -> None:
    with postgres_engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA lock_shadow")
        conn.exec_driver_sql("CREATE TABLE lock_shadow.calls (name text NOT NULL)")
        conn.exec_driver_sql(
            """
            CREATE FUNCTION lock_shadow.set_config(name text, new_value text, local_value boolean)
            RETURNS text LANGUAGE plpgsql VOLATILE AS $$
            BEGIN
                INSERT INTO lock_shadow.calls VALUES ('set_config');
                RETURN new_value;
            END
            $$
            """
        )
        conn.exec_driver_sql(
            """
            CREATE FUNCTION lock_shadow.hashtext(value text)
            RETURNS integer LANGUAGE plpgsql VOLATILE AS $$
            BEGIN
                INSERT INTO lock_shadow.calls VALUES ('hashtext');
                RETURN 7;
            END
            $$
            """
        )
        conn.exec_driver_sql(
            """
            CREATE FUNCTION lock_shadow.pg_advisory_lock(class_id integer, object_id integer)
            RETURNS void LANGUAGE plpgsql VOLATILE AS $$
            BEGIN
                INSERT INTO lock_shadow.calls VALUES ('lock');
                RETURN;
            END
            $$
            """
        )
        conn.exec_driver_sql(
            """
            CREATE FUNCTION lock_shadow.pg_advisory_unlock(class_id integer, object_id integer)
            RETURNS boolean LANGUAGE plpgsql VOLATILE AS $$
            BEGIN
                INSERT INTO lock_shadow.calls VALUES ('unlock');
                RETURN true;
            END
            $$
            """
        )

    shadowed = create_engine(
        postgres_engine.url,
        connect_args={"options": "-csearch_path=lock_shadow,pg_catalog,public"},
    )
    observed: list[tuple[list[str], int]] = []

    def body(conn: Connection) -> None:
        calls = list(conn.execute(text("SELECT name FROM lock_shadow.calls ORDER BY name")).scalars())
        lock_count = conn.execute(
            text("SELECT count(*) FROM pg_catalog.pg_locks WHERE locktype = 'advisory' AND pid = pg_catalog.pg_backend_pid()")
        ).scalar_one()
        observed.append((calls, lock_count))

    try:
        _run_locked(shadowed, target="shadow-proof", body=body, verify=lambda _conn: None)
    finally:
        shadowed.dispose()

    assert observed == [([], 1)]


class _CustomBaseException(BaseException):
    pass


@pytest.mark.parametrize("outcome", ["success", "sqlstate_22012", "base_exception"])
def test_lock_is_released_after_body_outcome(postgres_engine: Engine, outcome: str) -> None:
    owner_pids: list[int] = []

    def body(conn: Connection) -> None:
        owner_pids.append(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        if outcome == "sqlstate_22012":
            conn.exec_driver_sql("SELECT 1 / 0")
        if outcome == "base_exception":
            raise _CustomBaseException

    observer_engine = create_engine(postgres_engine.url, poolclass=NullPool)
    try:
        with observer_engine.connect() as observer:
            observer_pid = observer.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()
            if outcome == "success":
                _run_locked(
                    postgres_engine,
                    target="elspeth_schema_init",
                    body=body,
                    verify=lambda _conn: None,
                )
            else:
                expected = SQLAlchemyError if outcome == "sqlstate_22012" else _CustomBaseException
                with pytest.raises(expected):
                    _run_locked(
                        postgres_engine,
                        target="elspeth_schema_init",
                        body=body,
                        verify=lambda _conn: pytest.fail("verify must not run after body failure"),
                    )

            assert len(owner_pids) == 1
            assert observer_pid != owner_pids[0]
            acquired = observer.execute(
                text("SELECT pg_try_advisory_lock(:classid, hashtext('elspeth_schema_init'))"),
                {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
            ).scalar_one()
            try:
                assert acquired is True
            finally:
                if acquired:
                    assert (
                        observer.execute(
                            text("SELECT pg_advisory_unlock(:classid, hashtext('elspeth_schema_init'))"),
                            {"classid": schema_probe_module.ELSPETH_SCHEMA_INIT_LOCK_CLASSID},
                        ).scalar_one()
                        is True
                    )
    finally:
        observer_engine.dispose()


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
