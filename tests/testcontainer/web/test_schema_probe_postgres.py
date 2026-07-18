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
from sqlalchemy import Connection, Engine, create_engine, event, inspect, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool
from testcontainers.postgres import PostgresContainer
from tests.unit.core.test_schema_shape import _static_check_issues

from elspeth.contracts import Artifact
from elspeth.contracts.enums import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.database import _REQUIRED_TRIGGERS, LandscapeDB, SchemaCompatibilityError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, audit_export_snapshot_chunks_table
from elspeth.core.landscape.schema import schema_identity_table as landscape_schema_identity_table
from elspeth.core.schema_identity import SCHEMA_IDENTITY_APPLICATION_ID
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
from elspeth.web.sessions.models import SESSION_SCHEMA_EPOCH, skill_markdown_history_table
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.models import schema_identity_table as session_schema_identity_table
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


@pytest.mark.parametrize("kind", ["session", "landscape"])
def test_postgres_fresh_schema_stamps_cross_dialect_identity(postgres_engine: Engine, kind: str) -> None:
    if kind == "session":
        init_session_schema(postgres_engine)
        identity_table = session_schema_identity_table
        expected_epoch = SESSION_SCHEMA_EPOCH
    else:
        init_landscape_schema(postgres_engine)
        identity_table = landscape_schema_identity_table
        expected_epoch = SQLITE_SCHEMA_EPOCH

    with postgres_engine.connect() as conn:
        row = conn.execute(select(identity_table)).one()

    assert row.singleton_id == 1
    assert row.application_id == SCHEMA_IDENTITY_APPLICATION_ID
    assert row.store_kind == kind
    assert row.schema_epoch == expected_epoch


@pytest.mark.parametrize("construction", ["constructor", "from_url"])
def test_landscape_database_construction_stamps_postgres_identity(postgres_engine: Engine, construction: str) -> None:
    url = postgres_engine.url.render_as_string(hide_password=False)
    database = LandscapeDB(url) if construction == "constructor" else LandscapeDB.from_url(url)
    try:
        with database.engine.connect() as conn:
            row = conn.execute(select(landscape_schema_identity_table)).one()
    finally:
        database.close()

    assert row.application_id == SCHEMA_IDENTITY_APPLICATION_ID
    assert row.store_kind == "landscape"
    assert row.schema_epoch == SQLITE_SCHEMA_EPOCH


@pytest.mark.parametrize(
    ("kind", "column_name", "wrong_value"),
    [
        ("session", "application_id", "another-application"),
        ("session", "schema_epoch", SESSION_SCHEMA_EPOCH - 1),
        ("landscape", "store_kind", "session"),
        ("landscape", "schema_epoch", SQLITE_SCHEMA_EPOCH - 1),
    ],
)
def test_postgres_schema_identity_drift_is_stale_and_not_repaired(
    postgres_engine: Engine,
    kind: str,
    column_name: str,
    wrong_value: str | int,
) -> None:
    if kind == "session":
        init_session_schema(postgres_engine)
        identity_table = session_schema_identity_table
        probe = probe_session_schema
        initialize = init_session_schema
        error_type = SessionSchemaError
    else:
        init_landscape_schema(postgres_engine)
        identity_table = landscape_schema_identity_table
        probe = probe_landscape_schema
        initialize = init_landscape_schema
        error_type = SchemaCompatibilityError

    with postgres_engine.begin() as conn:
        conn.execute(update(identity_table).values({column_name: wrong_value}))

    assert probe(postgres_engine) is SchemaState.STALE
    with pytest.raises(error_type):
        initialize(postgres_engine)
    with postgres_engine.connect() as conn:
        assert conn.execute(select(identity_table.c[column_name])).scalar_one() == wrong_value


def test_landscape_tokens_bind_row_id_and_run_id_to_the_same_row(postgres_engine: Engine) -> None:
    """Fresh PostgreSQL schemas carry the epoch-24 ownership constraint."""
    init_landscape_schema(postgres_engine)

    foreign_keys = inspect(postgres_engine).get_foreign_keys("tokens")
    assert any(
        tuple(foreign_key["constrained_columns"]) == ("row_id", "run_id")
        and foreign_key["referred_table"] == "rows"
        and tuple(foreign_key["referred_columns"]) == ("row_id", "run_id")
        for foreign_key in foreign_keys
    )


def test_postgres_landscape_trigger_inventory_validates_and_unsealed_chunk_mutations_apply(postgres_engine: Engine) -> None:
    """Fresh PostgreSQL has the complete trigger contract and preserves row semantics."""
    init_landscape_schema(postgres_engine)

    with postgres_engine.connect() as conn:
        trigger_names = set(
            conn.exec_driver_sql("SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = current_schema()").scalars()
        )
    assert trigger_names >= set(_REQUIRED_TRIGGERS)

    url = postgres_engine.url.render_as_string(hide_password=False)
    validated = LandscapeDB.from_url(url, create_tables=False)
    validated.close()

    content_hash = "a" * 64
    with postgres_engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(
                audit_export_snapshot_chunks_table.insert().values(
                    snapshot_id="unsealed-postgres-snapshot",
                    ordinal=0,
                    content_ref=f"sha256:{content_hash}",
                    content_hash=content_hash,
                    size_bytes=10,
                    record_count=1,
                    predecessor_seal_hash=None,
                    cumulative_records=1,
                    cumulative_bytes=10,
                    chunk_seal_hash="b" * 64,
                )
            )
            updated = conn.execute(
                update(audit_export_snapshot_chunks_table)
                .where(audit_export_snapshot_chunks_table.c.snapshot_id == "unsealed-postgres-snapshot")
                .values(size_bytes=11, cumulative_bytes=11)
            )
            assert updated.rowcount == 1
            assert conn.execute(
                select(
                    audit_export_snapshot_chunks_table.c.size_bytes,
                    audit_export_snapshot_chunks_table.c.cumulative_bytes,
                ).where(audit_export_snapshot_chunks_table.c.snapshot_id == "unsealed-postgres-snapshot")
            ).one() == (11, 11)

            deleted = conn.execute(
                audit_export_snapshot_chunks_table.delete().where(
                    audit_export_snapshot_chunks_table.c.snapshot_id == "unsealed-postgres-snapshot"
                )
            )
            assert deleted.rowcount == 1
            assert (
                conn.execute(
                    select(audit_export_snapshot_chunks_table.c.snapshot_id).where(
                        audit_export_snapshot_chunks_table.c.snapshot_id == "unsealed-postgres-snapshot"
                    )
                ).first()
                is None
            )
        finally:
            transaction.rollback()


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
        conn.execute(
            text(
                """
                INSERT INTO guided_operations (
                    session_id, operation_id, kind, status, request_hash,
                    lease_token, lease_expires_at, attempt, created_at, updated_at
                ) VALUES (
                    :session_id, :operation_id, 'guided_start', 'in_progress',
                    :request_hash, 'lease-token', CURRENT_TIMESTAMP + INTERVAL '1 minute',
                    1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """
            ),
            {"session_id": session_id, "operation_id": f"{session_id}-operation", "request_hash": "a" * 64},
        )
        conn.execute(
            text(
                """
                INSERT INTO guided_operation_events (
                    session_id, operation_id, sequence, event_kind, actor,
                    attempt, request_hash, lease_expires_at, occurred_at
                ) VALUES (
                    :session_id, :operation_id, 1, 'claimed', 'worker-a',
                    1, :request_hash, CURRENT_TIMESTAMP + INTERVAL '1 minute',
                    CURRENT_TIMESTAMP
                )
                """
            ),
            {"session_id": session_id, "operation_id": f"{session_id}-operation", "request_hash": "a" * 64},
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
        "trg_guided_operation_events_no_update",
        "trg_guided_operation_events_no_delete",
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
        ("UPDATE guided_operation_events SET actor = 'attacker' WHERE operation_id = :row_id", f"{protected_session}-operation"),
        ("DELETE FROM guided_operation_events WHERE operation_id = :row_id", f"{protected_session}-operation"),
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
        assert (
            conn.execute(
                text("SELECT count(*) FROM guided_operation_events WHERE session_id = :session_id"),
                {"session_id": cascade_session},
            ).scalar_one()
            == 0
        )


@pytest.mark.parametrize(
    "trigger_mutation",
    [
        "DROP TRIGGER trg_chat_messages_no_delete ON chat_messages",
        "ALTER TABLE chat_messages DISABLE TRIGGER trg_chat_messages_no_delete",
        "DROP TRIGGER trg_guided_operation_events_no_delete ON guided_operation_events",
        "ALTER TABLE guided_operation_events DISABLE TRIGGER trg_guided_operation_events_no_delete",
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


def test_artifact_idempotency_index_and_behavior(postgres_engine: Engine) -> None:
    init_landscape_schema(postgres_engine)
    indexes = {entry["name"]: entry for entry in inspect(postgres_engine).get_indexes("artifacts")}
    idempotency_index = indexes["uq_artifacts_run_idempotency_key"]
    assert idempotency_index["unique"] is True
    assert idempotency_index["column_names"] == ["run_id", "idempotency_key"]
    assert "idempotency_key IS NOT NULL" in str(idempotency_index["dialect_options"]["postgresql_where"])

    db_url = postgres_engine.url.render_as_string(hide_password=False)
    db = LandscapeDB.from_url(db_url, create_tables=False)
    try:
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id="postgres-artifact-idempotency",
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
        schema = SchemaConfig.from_dict({"mode": "observed"})
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="postgres-artifact-source",
            schema_config=schema,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="postgres-artifact-sink",
            schema_config=schema,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="postgres-artifact-source",
            row_index=0,
            data={"value": 1},
            row_id="postgres-artifact-row",
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, token_id="postgres-artifact-token")
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id="postgres-artifact-sink",
            run_id=run.run_id,
            step_index=0,
            input_data={"value": 1},
            state_id="postgres-artifact-state",
        )
        values = {
            "run_id": run.run_id,
            "state_id": state.state_id,
            "sink_node_id": "postgres-artifact-sink",
            "artifact_type": "csv",
            "path": "/output/postgres.csv",
            "content_hash": "sha256:postgres",
            "size_bytes": 128,
            "idempotency_key": "postgres-artifact-row:csv_sink",
        }

        first = factory.execution.register_artifact(**values, artifact_id="postgres-artifact-first")
        retried = factory.execution.register_artifact(**values, artifact_id="postgres-artifact-retry")
        assert retried == first

        with pytest.raises(AuditIntegrityError, match="content_hash"):
            factory.execution.register_artifact(**(values | {"content_hash": "sha256:divergent"}))

        null_first = factory.execution.register_artifact(**(values | {"idempotency_key": None}))
        null_second = factory.execution.register_artifact(**(values | {"idempotency_key": None}))
        assert null_first.artifact_id != null_second.artifact_id
        assert len(factory.execution.get_artifacts(run.run_id)) == 3
    finally:
        db.close()


@pytest.mark.parametrize("divergent", [False, True], ids=["identical", "divergent"])
def test_artifact_idempotency_contenders_use_independent_postgres_connections(
    postgres_engine: Engine,
    divergent: bool,
) -> None:
    init_landscape_schema(postgres_engine)
    db_url = postgres_engine.url.render_as_string(hide_password=False)
    db = LandscapeDB.from_url(db_url, create_tables=False, pool_size=2, max_overflow=0, pool_timeout=5)
    try:
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id="postgres-artifact-contention",
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
        schema = SchemaConfig.from_dict({"mode": "observed"})
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="postgres-artifact-contention-source",
            schema_config=schema,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="postgres-artifact-contention-sink",
            schema_config=schema,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="postgres-artifact-contention-source",
            row_index=0,
            data={"value": 1},
            row_id="postgres-artifact-contention-row",
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, token_id="postgres-artifact-contention-token")
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id="postgres-artifact-contention-sink",
            run_id=run.run_id,
            step_index=0,
            input_data={"value": 1},
            state_id="postgres-artifact-contention-state",
        )
        values = {
            "run_id": run.run_id,
            "state_id": state.state_id,
            "sink_node_id": "postgres-artifact-contention-sink",
            "artifact_type": "csv",
            "path": "/output/postgres-contention.csv",
            "size_bytes": 128,
            "idempotency_key": "postgres-artifact-contention-row:csv_sink",
        }

        transaction_barrier = threading.Barrier(2)
        physical_connections: set[int] = set()
        connection_guard = threading.Lock()

        def _synchronize_contender_transactions(conn: Connection) -> None:
            if not threading.current_thread().name.startswith("postgres-artifact-contender"):
                return
            with connection_guard:
                physical_connections.add(id(conn.connection.driver_connection))
            transaction_barrier.wait(timeout=15)

        event.listen(db.engine, "begin", _synchronize_contender_transactions)

        def _contend(ordinal: int) -> Artifact | AuditIntegrityError:
            content_hash = "sha256:postgres-contention"
            if divergent and ordinal == 1:
                content_hash = "sha256:postgres-divergent"
            contender = RecorderFactory(db)
            try:
                return contender.execution.register_artifact(
                    **values,
                    content_hash=content_hash,
                    artifact_id=f"postgres-artifact-proposal-{ordinal}",
                )
            except AuditIntegrityError as exc:
                return exc

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="postgres-artifact-contender") as executor:
                futures = [executor.submit(_contend, ordinal) for ordinal in range(2)]
                outcomes = [future.result(timeout=30) for future in futures]
        finally:
            event.remove(db.engine, "begin", _synchronize_contender_transactions)

        assert len(physical_connections) == 2
        successful = [outcome for outcome in outcomes if isinstance(outcome, Artifact)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, AuditIntegrityError)]
        if divergent:
            assert len(successful) == 1
            assert len(failures) == 1
            assert "content_hash" in str(failures[0])
        else:
            assert len(successful) == 2
            assert failures == []
            winning_ids = {artifact.artifact_id for artifact in successful}
            assert len(winning_ids) == 1
            assert winning_ids <= {"postgres-artifact-proposal-0", "postgres-artifact-proposal-1"}
            assert len({artifact.created_at for artifact in successful}) == 1

        durable = factory.execution.get_artifacts(run.run_id)
        assert len(durable) == 1
        assert durable[0] == successful[0]
    finally:
        db.close()


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
