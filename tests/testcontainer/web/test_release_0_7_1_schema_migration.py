"""PostgreSQL proof for the versioned release-0.7.1 schema migrator."""

from __future__ import annotations

import io
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
import pytest
from psycopg import sql
from scripts import migrate_release_0_7_1_aws_ecs_schema as migration
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.core.landscape.database import _sqlite_epoch_23_shape_metadata
from elspeth.core.landscape.schema import (
    artifacts_table,
    calls_table,
    node_states_table,
    nodes_table,
    operations_table,
    rows_table,
    run_sources_table,
    runs_table,
    tokens_table,
)
from elspeth.core.schema_identity import SCHEMA_IDENTITY_TABLE_NAME
from elspeth.web.schema_probe import SchemaState, init_landscape_schema, init_session_schema, probe_landscape_schema, probe_session_schema
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import POSTGRESQL_AUDIT_DDL_COHORT
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.models import schema_identity_table as session_schema_identity_table

pytestmark = pytest.mark.testcontainer

_SAFE_IDENTIFIER = re.compile(r"[a-z0-9_]+\Z")


def _identifier(prefix: str) -> str:
    value = f"{prefix}_{uuid.uuid4().hex}"
    assert _SAFE_IDENTIFIER.fullmatch(value)
    return value


def _render_url(base_url: str | URL, *, database: str, username: str | None = None, password: str | None = None) -> str:
    parsed = make_url(base_url).set(database=database)
    if username is not None:
        parsed = parsed.set(username=username, password=password)
    return parsed.render_as_string(hide_password=False)


def _psycopg_connect(url: str) -> psycopg.Connection[object]:
    parsed = make_url(url)
    assert parsed.host is not None
    assert parsed.username is not None
    assert parsed.database is not None
    return psycopg.connect(
        host=parsed.host,
        port=parsed.port or 5432,
        dbname=parsed.database,
        user=parsed.username,
        password=parsed.password,
        autocommit=True,
    )


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@dataclass(frozen=True, slots=True)
class _DatabasePair:
    postgres_url: str
    session_database: str
    landscape_database: str

    @property
    def session_url(self) -> str:
        return _render_url(self.postgres_url, database=self.session_database)

    @property
    def landscape_url(self) -> str:
        return _render_url(self.postgres_url, database=self.landscape_database)


@pytest.fixture
def database_pair(postgres_url: str) -> Iterator[_DatabasePair]:
    pair = _DatabasePair(
        postgres_url=postgres_url,
        session_database=_identifier("release_session"),
        landscape_database=_identifier("release_landscape"),
    )
    with _psycopg_connect(postgres_url) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(pair.session_database)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(pair.landscape_database)))
    try:
        yield pair
    finally:
        with _psycopg_connect(postgres_url) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(pair.session_database)))
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(pair.landscape_database)))


def _create_release_session_shape(engine: Engine) -> None:
    session_metadata.create_all(engine)
    with engine.begin() as conn:
        for entry in POSTGRESQL_AUDIT_DDL_COHORT:
            conn.execute(text(f'DROP TRIGGER "{entry.trigger_name}" ON "{entry.table.name}"'))
            conn.execute(text(f'DROP FUNCTION "{entry.function_name}"()'))
        conn.execute(text(f'DROP TABLE "{SCHEMA_IDENTITY_TABLE_NAME}"'))


def _create_release_landscape_shape(engine: Engine, *, hash_width: int) -> None:
    assert hash_width in {16, 32}
    # Build the genuine release predecessor. Deriving this fixture from current
    # metadata and merely dropping the identity table masked every later epoch
    # as soon as release/0.7.1 advanced beyond epoch 24.
    predecessor = _sqlite_epoch_23_shape_metadata()
    predecessor.create_all(engine)
    with engine.begin() as conn:
        if hash_width == 16:
            conn.execute(text("ALTER TABLE run_sources ALTER COLUMN schema_contract_hash TYPE VARCHAR(16)"))
        tables = set(inspect(conn).get_table_names())
        assert SCHEMA_IDENTITY_TABLE_NAME not in tables
        assert "sink_effects" not in tables
        assert "audit_export_snapshots" not in tables


def _create_release_0_7_0_shapes(pair: _DatabasePair, *, hash_width: int) -> tuple[Engine, Engine]:
    session = create_session_engine(pair.session_url)
    landscape = create_engine(pair.landscape_url)
    _create_release_session_shape(session)
    _create_release_landscape_shape(landscape, hash_width=hash_width)
    with session.connect() as conn:
        assert SCHEMA_IDENTITY_TABLE_NAME not in inspect(conn).get_table_names()
        assert _trigger_rows(conn) == ()
    return session, landscape


def _seed_representative_rows(session: Engine, landscape: Engine, *, hash_width: int) -> None:
    now = datetime.now(UTC)
    with session.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO sessions (
                    id, user_id, auth_provider_type, title, trust_mode,
                    density_default, created_at, updated_at,
                    interpretation_review_disabled
                ) VALUES (
                    'migration-session', 'migration-user', 'local', 'Preserve me',
                    'auto_commit', 'high', :now, :now, false
                )
                """
            ),
            {"now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO composition_states (
                    id, session_id, version, is_valid, created_at, provenance
                ) VALUES (
                    'migration-state', 'migration-session', 1, false, :now,
                    'session_seed'
                )
                """
            ),
            {"now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO chat_messages (
                    id, session_id, role, content, sequence_no,
                    writer_principal, created_at
                ) VALUES (
                    'migration-message', 'migration-session', 'user',
                    'unaltered content', 1, 'route_user_message', :now
                )
                """
            ),
            {"now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO interpretation_events (
                    id, session_id, choice, created_at, resolved_at, actor,
                    interpretation_source
                ) VALUES (
                    'migration-interpretation', 'migration-session', 'opted_out',
                    :now, :now, 'migration-user', 'auto_interpreted_opt_out'
                )
                """
            ),
            {"now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO composer_completion_events (
                    id, session_id, composition_state_id, event_type,
                    actor, created_at
                ) VALUES (
                    'migration-completion', 'migration-session', 'migration-state',
                    'export_yaml', 'migration-user', :now
                )
                """
            ),
            {"now": now},
        )

    contract_hash = "c" * hash_width
    with landscape.begin() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id="migration-run",
                started_at=now,
                config_hash="a" * 64,
                settings_json='{"preserve":true}',
                canonical_version="sha256-rfc8785-v1",
                status="completed",
                openrouter_catalog_sha256="b" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        conn.execute(
            nodes_table.insert().values(
                node_id="migration-source",
                run_id="migration-run",
                plugin_name="csv",
                node_type="SOURCE",
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="d" * 64,
                config_json="{}",
                registered_at=now,
            )
        )
        conn.execute(
            run_sources_table.insert().values(
                run_id="migration-run",
                source_node_id="migration-source",
                source_name="primary",
                plugin_name="csv",
                lifecycle_state="loaded",
                config_hash="d" * 64,
                schema_json="{}",
                schema_contract_json='{"mode":"OBSERVED"}',
                schema_contract_hash=contract_hash,
                field_resolution_json="{}",
                recorded_at=now,
            )
        )
        conn.execute(
            rows_table.insert().values(
                row_id="migration-row",
                run_id="migration-run",
                source_node_id="migration-source",
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="e" * 64,
                source_data_ref="sha256:migration-row",
                created_at=now,
            )
        )
        conn.execute(
            tokens_table.insert().values(
                token_id="migration-token",
                row_id="migration-row",
                run_id="migration-run",
                created_at=now,
            )
        )
        conn.execute(
            node_states_table.insert().values(
                state_id="migration-node-state",
                token_id="migration-token",
                run_id="migration-run",
                node_id="migration-source",
                step_index=0,
                attempt=0,
                status="completed",
                input_hash="f" * 64,
                output_hash="0" * 64,
                started_at=now,
                completed_at=now,
            )
        )
        conn.execute(
            operations_table.insert().values(
                operation_id="migration-operation",
                run_id="migration-run",
                node_id="migration-source",
                operation_type="sink_write",
                started_at=now,
                completed_at=now,
                status="completed",
                input_data_ref="sha256:migration-input",
                input_data_hash="1" * 64,
                output_data_ref="sha256:migration-output",
                output_data_hash="2" * 64,
                duration_ms=12.5,
            )
        )
        conn.execute(
            calls_table.insert().values(
                call_id="migration-call",
                operation_id="migration-operation",
                call_index=0,
                call_type="http",
                status="success",
                request_hash="3" * 64,
                response_hash="4" * 64,
                latency_ms=10.0,
                created_at=now,
            )
        )
        conn.execute(
            artifacts_table.insert().values(
                artifact_id="migration-artifact",
                run_id="migration-run",
                produced_by_state_id="migration-node-state",
                sink_node_id="migration-source",
                artifact_type="csv",
                path_or_uri="s3://migration-bucket/preserved.csv",
                content_hash="5" * 64,
                size_bytes=128,
                idempotency_key="migration-run:migration-artifact",
                created_at=now,
            )
        )


def _row_fingerprints(engine: Engine) -> tuple[tuple[str, int, str], ...]:
    fingerprints: list[tuple[str, int, str]] = []
    with engine.connect() as conn:
        table_names = sorted(inspect(conn).get_table_names())
        for table_name in table_names:
            assert _SAFE_IDENTIFIER.fullmatch(table_name)
            count, digest = conn.execute(
                text(
                    f"SELECT count(*), md5(COALESCE(string_agg(row_to_json(row_value)::text, E'\\n' "
                    f"ORDER BY row_to_json(row_value)::text), '')) FROM \"{table_name}\" AS row_value"
                )
            ).one()
            if int(count) > 0:
                fingerprints.append((table_name, int(count), str(digest)))
    return tuple(fingerprints)


def _catalog_fingerprint(engine: Engine) -> tuple[tuple[object, ...], ...]:
    with engine.connect() as conn:
        return tuple(
            tuple(row)
            for row in conn.execute(
                text(
                    """
                    SELECT object_kind, object_name, definition
                    FROM (
                        SELECT 'table'::text AS object_kind, cls.relname AS object_name,
                               cls.relkind::text AS definition
                        FROM pg_catalog.pg_class AS cls
                        JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
                        WHERE ns.nspname = current_schema() AND cls.relkind IN ('r', 'p')
                        UNION ALL
                        SELECT 'column', cls.relname || '.' || attribute.attname,
                               pg_catalog.format_type(attribute.atttypid, attribute.atttypmod)
                               || ':' || attribute.attnotnull::text
                               || ':' || COALESCE(pg_catalog.pg_get_expr(default_value.adbin, default_value.adrelid), '')
                        FROM pg_catalog.pg_attribute AS attribute
                        JOIN pg_catalog.pg_class AS cls ON cls.oid = attribute.attrelid
                        JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
                        LEFT JOIN pg_catalog.pg_attrdef AS default_value
                          ON default_value.adrelid = attribute.attrelid
                         AND default_value.adnum = attribute.attnum
                        WHERE ns.nspname = current_schema()
                          AND cls.relkind IN ('r', 'p')
                          AND attribute.attnum > 0
                          AND NOT attribute.attisdropped
                        UNION ALL
                        SELECT 'constraint', cls.relname || '.' || constraint_value.conname,
                               pg_catalog.pg_get_constraintdef(constraint_value.oid, true)
                        FROM pg_catalog.pg_constraint AS constraint_value
                        JOIN pg_catalog.pg_class AS cls ON cls.oid = constraint_value.conrelid
                        JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
                        WHERE ns.nspname = current_schema()
                        UNION ALL
                        SELECT 'index', indexes.tablename || '.' || indexes.indexname,
                               indexes.indexdef
                        FROM pg_catalog.pg_indexes AS indexes
                        WHERE indexes.schemaname = current_schema()
                        UNION ALL
                        SELECT 'trigger', trigger.tgname,
                               pg_catalog.pg_get_triggerdef(trigger.oid, true)
                        FROM pg_catalog.pg_trigger AS trigger
                        JOIN pg_catalog.pg_class AS cls ON cls.oid = trigger.tgrelid
                        JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
                        WHERE ns.nspname = current_schema() AND NOT trigger.tgisinternal
                        UNION ALL
                        SELECT 'function', procedure.proname,
                               pg_catalog.pg_get_functiondef(procedure.oid)
                        FROM pg_catalog.pg_proc AS procedure
                        JOIN pg_catalog.pg_namespace AS ns ON ns.oid = procedure.pronamespace
                        WHERE ns.nspname = current_schema()
                    ) AS objects
                    ORDER BY object_kind, object_name, definition
                    """
                )
            )
        )


def _trigger_rows(conn: object) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        tuple(str(value) for value in row)
        for row in conn.execute(  # type: ignore[union-attr]
            text(
                """
                SELECT trigger.tgname, relation.relname, procedure.proname,
                       trigger.tgenabled::text
                FROM pg_catalog.pg_trigger AS trigger
                JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger.tgrelid
                JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                JOIN pg_catalog.pg_proc AS procedure ON procedure.oid = trigger.tgfoid
                WHERE NOT trigger.tgisinternal
                  AND namespace.nspname = current_schema()
                ORDER BY trigger.tgname, relation.relname
                """
            )
        )
    )


def _assert_triggers_enforced(session: Engine) -> None:
    expected = {(entry.trigger_name, entry.table.name, entry.function_name, "O") for entry in POSTGRESQL_AUDIT_DDL_COHORT}
    with session.connect() as conn:
        assert set(_trigger_rows(conn)) == expected

    mutations = (
        "UPDATE interpretation_events SET actor='attacker' WHERE id='migration-interpretation'",
        "DELETE FROM interpretation_events WHERE id='migration-interpretation'",
        "UPDATE composer_completion_events SET actor='attacker' WHERE id='migration-completion'",
        "DELETE FROM composer_completion_events WHERE id='migration-completion'",
        "UPDATE chat_messages SET content='tampered' WHERE id='migration-message'",
        "DELETE FROM chat_messages WHERE id='migration-message'",
    )
    for statement in mutations:
        with pytest.raises(DBAPIError, match=r"append-only|immutable"), session.begin() as conn:
            conn.execute(text(statement))


@pytest.mark.parametrize("hash_width", [16, 32])
def test_exact_pre_state_migrates_without_changing_rows_and_rerun_is_stable(
    database_pair: _DatabasePair,
    hash_width: int,
) -> None:
    session, landscape = _create_release_0_7_0_shapes(database_pair, hash_width=hash_width)
    try:
        _seed_representative_rows(session, landscape, hash_width=hash_width)
        before_session = _row_fingerprints(session)
        before_landscape = _row_fingerprints(landscape)

        result = migration.run_migration(database_pair.session_url, database_pair.landscape_url)

        assert result.code is migration.ResultCode.MIGRATION_APPLIED
        assert result.already_applied is False
        after_session = {name: (count, digest) for name, count, digest in _row_fingerprints(session)}
        after_landscape = {name: (count, digest) for name, count, digest in _row_fingerprints(landscape)}
        assert all(after_session[name] == (count, digest) for name, count, digest in before_session)
        assert all(
            after_landscape[name] == (count, digest) for name, count, digest in before_landscape if name not in {"artifacts", "operations"}
        )
        assert probe_session_schema(session) is SchemaState.CURRENT
        assert probe_landscape_schema(landscape) is SchemaState.CURRENT
        with landscape.connect() as conn:
            width = conn.execute(
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
            assert width == 32
            assert conn.execute(text("SELECT schema_contract_hash FROM run_sources")).scalar_one() == "c" * hash_width
            operation = conn.execute(
                text(
                    """
                    SELECT operation_id, sink_effect_id, input_data_ref,
                           output_data_ref, duration_ms
                    FROM operations WHERE operation_id = 'migration-operation'
                    """
                )
            ).one()
            assert tuple(operation) == (
                "migration-operation",
                None,
                "sha256:migration-input",
                "sha256:migration-output",
                12.5,
            )
            artifact = conn.execute(
                text(
                    """
                    SELECT artifact_id, produced_by_state_id, sink_effect_id,
                           publication_performed, publication_evidence_kind,
                           path_or_uri, size_bytes
                    FROM artifacts WHERE artifact_id = 'migration-artifact'
                    """
                )
            ).one()
            assert tuple(artifact) == (
                "migration-artifact",
                "migration-node-state",
                None,
                True,
                "legacy_returned",
                "s3://migration-bucket/preserved.csv",
                128,
            )
            assert (
                conn.execute(text("SELECT operation_id FROM calls WHERE call_id = 'migration-call'")).scalar_one() == "migration-operation"
            )
        _assert_triggers_enforced(session)

        environment = {
            migration.SESSION_OWNER_URL_ENV: database_pair.session_url,
            migration.LANDSCAPE_OWNER_URL_ENV: database_pair.landscape_url,
        }
        outputs: list[str] = []
        for _ in range(2):
            stdout = io.StringIO()
            assert migration.main(["--apply"], environ=environment, stdout=stdout) == 0
            outputs.append(stdout.getvalue())
        assert outputs[0] == outputs[1]
        assert '"already_applied":true' in outputs[0]
    finally:
        session.dispose()
        landscape.dispose()


@pytest.mark.parametrize(
    "invalid_state",
    [
        "unexpected_width",
        "foreign_table",
        "partial_trigger",
        "disabled_trigger",
        "duplicate_trigger",
        "unexpected_identity",
    ],
)
def test_unrecognized_pre_state_fails_before_any_ddl(
    database_pair: _DatabasePair,
    invalid_state: str,
) -> None:
    session, landscape = _create_release_0_7_0_shapes(database_pair, hash_width=16)
    try:
        _seed_representative_rows(session, landscape, hash_width=16)
        if invalid_state == "unexpected_width":
            with landscape.begin() as conn:
                conn.execute(text("ALTER TABLE run_sources ALTER COLUMN schema_contract_hash TYPE VARCHAR(17)"))
        elif invalid_state == "foreign_table":
            with session.begin() as conn:
                conn.execute(text("CREATE TABLE foreign_release_table (id integer primary key)"))
        elif invalid_state in {"partial_trigger", "disabled_trigger", "duplicate_trigger"}:
            entry = POSTGRESQL_AUDIT_DDL_COHORT[0]
            with session.begin() as conn:
                conn.execute(text(entry.function_sql))
                conn.execute(text(entry.trigger_sql))
                if invalid_state == "disabled_trigger":
                    conn.execute(text(f'ALTER TABLE "{entry.table.name}" DISABLE TRIGGER "{entry.trigger_name}"'))
                elif invalid_state == "duplicate_trigger":
                    conn.execute(
                        text(
                            f'CREATE TRIGGER "{entry.trigger_name}" BEFORE UPDATE ON sessions '
                            f'FOR EACH ROW EXECUTE FUNCTION "{entry.function_name}"()'
                        )
                    )
        else:
            session_schema_identity_table.create(session, checkfirst=False)

        before = (_catalog_fingerprint(session), _catalog_fingerprint(landscape))
        with pytest.raises(migration.MigrationFailure) as exc_info:
            migration.run_migration(database_pair.session_url, database_pair.landscape_url)

        assert exc_info.value.code is migration.ResultCode.PRECONDITION_FAILED
        assert (_catalog_fingerprint(session), _catalog_fingerprint(landscape)) == before
    finally:
        session.dispose()
        landscape.dispose()


@pytest.mark.parametrize("invalid_rows", ["token_run_mismatch", "duplicate_artifact_key"])
def test_invalid_tier1_rows_fail_before_either_database_changes(
    database_pair: _DatabasePair,
    invalid_rows: str,
) -> None:
    session, landscape = _create_release_0_7_0_shapes(database_pair, hash_width=16)
    try:
        _seed_representative_rows(session, landscape, hash_width=16)
        with landscape.begin() as conn:
            if invalid_rows == "token_run_mismatch":
                conn.execute(
                    runs_table.insert().values(
                        run_id="migration-other-run",
                        started_at=datetime.now(UTC),
                        config_hash="6" * 64,
                        settings_json="{}",
                        canonical_version="sha256-rfc8785-v1",
                        status="running",
                        openrouter_catalog_sha256="7" * 64,
                        openrouter_catalog_source="bundled",
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id="migration-forged-token",
                        row_id="migration-row",
                        run_id="migration-other-run",
                        created_at=datetime.now(UTC),
                    )
                )
            else:
                conn.execute(
                    artifacts_table.insert().values(
                        artifact_id="migration-duplicate-artifact",
                        run_id="migration-run",
                        produced_by_state_id="migration-node-state",
                        sink_node_id="migration-source",
                        artifact_type="csv",
                        path_or_uri="s3://migration-bucket/duplicate.csv",
                        content_hash="8" * 64,
                        size_bytes=64,
                        idempotency_key="migration-run:migration-artifact",
                        created_at=datetime.now(UTC),
                    )
                )

        before = (
            _catalog_fingerprint(session),
            _catalog_fingerprint(landscape),
            _row_fingerprints(session),
            _row_fingerprints(landscape),
        )
        with pytest.raises(migration.MigrationFailure) as exc_info:
            migration.run_migration(database_pair.session_url, database_pair.landscape_url)

        assert exc_info.value.code is migration.ResultCode.PRECONDITION_FAILED
        assert (
            _catalog_fingerprint(session),
            _catalog_fingerprint(landscape),
            _row_fingerprints(session),
            _row_fingerprints(landscape),
        ) == before
    finally:
        session.dispose()
        landscape.dispose()


def test_only_session_current_landscape_old_partial_resumes(database_pair: _DatabasePair) -> None:
    session = create_session_engine(database_pair.session_url)
    landscape = create_engine(database_pair.landscape_url)
    try:
        init_session_schema(session)
        _create_release_landscape_shape(landscape, hash_width=16)

        result = migration.run_migration(database_pair.session_url, database_pair.landscape_url)

        assert result.code is migration.ResultCode.MIGRATION_APPLIED
        assert probe_session_schema(session) is SchemaState.CURRENT
        assert probe_landscape_schema(landscape) is SchemaState.CURRENT
    finally:
        session.dispose()
        landscape.dispose()


def test_landscape_rebuild_failure_rolls_back_then_exact_partial_resumes(
    database_pair: _DatabasePair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, landscape = _create_release_0_7_0_shapes(database_pair, hash_width=16)
    try:
        _seed_representative_rows(session, landscape, hash_width=16)
        before_landscape = (_catalog_fingerprint(landscape), _row_fingerprints(landscape))
        monkeypatch.setattr(
            migration,
            "_ARTIFACTS_COPY_SQL",
            "INSERT INTO artifacts_epoch_26 (artifact_id) SELECT NULL FROM artifacts",
        )

        with pytest.raises(migration.MigrationFailure) as exc_info:
            migration.run_migration(database_pair.session_url, database_pair.landscape_url)

        assert exc_info.value.code is migration.ResultCode.DATABASE_ERROR
        assert probe_session_schema(session) is SchemaState.CURRENT
        with landscape.connect() as conn:
            assert migration.classify_landscape_state(conn) is migration.LandscapeState.RELEASE_0_7_0_WIDTH_16
        assert (_catalog_fingerprint(landscape), _row_fingerprints(landscape)) == before_landscape

        monkeypatch.undo()
        result = migration.run_migration(database_pair.session_url, database_pair.landscape_url)
        assert result.code is migration.ResultCode.MIGRATION_APPLIED
        assert probe_landscape_schema(landscape) is SchemaState.CURRENT
    finally:
        session.dispose()
        landscape.dispose()


def test_reverse_partial_fails_before_session_ddl(database_pair: _DatabasePair) -> None:
    session = create_session_engine(database_pair.session_url)
    landscape = create_engine(database_pair.landscape_url)
    try:
        _create_release_session_shape(session)
        init_landscape_schema(landscape)
        before = (_catalog_fingerprint(session), _catalog_fingerprint(landscape))

        with pytest.raises(migration.MigrationFailure) as exc_info:
            migration.run_migration(database_pair.session_url, database_pair.landscape_url)

        assert exc_info.value.code is migration.ResultCode.PRECONDITION_FAILED
        assert (_catalog_fingerprint(session), _catalog_fingerprint(landscape)) == before
    finally:
        session.dispose()
        landscape.dispose()


def test_runtime_role_cannot_execute_migration_ddl(database_pair: _DatabasePair) -> None:
    session, landscape = _create_release_0_7_0_shapes(database_pair, hash_width=16)
    role = _identifier("release_runtime")
    password = f"runtime-{uuid.uuid4().hex}"
    try:
        with _psycopg_connect(database_pair.postgres_url) as admin:
            admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
            for database in (database_pair.session_database, database_pair.landscape_database):
                admin.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(database), sql.Identifier(role)))
        for owner_url in (database_pair.session_url, database_pair.landscape_url):
            with _psycopg_connect(owner_url) as owner:
                owner.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
                owner.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
                owner.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(role)))
        session_runtime = _render_url(database_pair.postgres_url, database=database_pair.session_database, username=role, password=password)
        landscape_runtime = _render_url(
            database_pair.postgres_url,
            database=database_pair.landscape_database,
            username=role,
            password=password,
        )
        before = (_catalog_fingerprint(session), _catalog_fingerprint(landscape))

        with pytest.raises(migration.MigrationFailure) as exc_info:
            migration.run_migration(session_runtime, landscape_runtime)

        assert exc_info.value.code is migration.ResultCode.PERMISSION_DENIED
        assert (_catalog_fingerprint(session), _catalog_fingerprint(landscape)) == before
    finally:
        session.dispose()
        landscape.dispose()
        for owner_url in (database_pair.session_url, database_pair.landscape_url):
            with _psycopg_connect(owner_url) as owner:
                owner.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
        with _psycopg_connect(database_pair.postgres_url) as admin:
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def test_owner_migration_preserves_runtime_dml_grants(database_pair: _DatabasePair) -> None:
    session, landscape = _create_release_0_7_0_shapes(database_pair, hash_width=16)
    role = _identifier("migrated_runtime")
    password = f"runtime-{uuid.uuid4().hex}"
    try:
        with _psycopg_connect(database_pair.postgres_url) as admin:
            admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
            for database in (database_pair.session_database, database_pair.landscape_database):
                admin.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(database), sql.Identifier(role)))
        for owner_url in (database_pair.session_url, database_pair.landscape_url):
            with _psycopg_connect(owner_url) as owner:
                owner.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
                owner.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
                owner.execute(
                    sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(role))
                )

        result = migration.run_migration(database_pair.session_url, database_pair.landscape_url)
        assert result.code is migration.ResultCode.MIGRATION_APPLIED

        session_runtime = create_session_engine(
            _render_url(database_pair.postgres_url, database=database_pair.session_database, username=role, password=password)
        )
        landscape_runtime = create_engine(
            _render_url(database_pair.postgres_url, database=database_pair.landscape_database, username=role, password=password)
        )
        try:
            assert probe_session_schema(session_runtime) is SchemaState.CURRENT
            assert probe_landscape_schema(landscape_runtime) is SchemaState.CURRENT
        finally:
            session_runtime.dispose()
            landscape_runtime.dispose()

        with landscape.connect() as conn:
            grants = {
                (str(table_name), str(privilege))
                for table_name, privilege in conn.execute(
                    text(
                        """
                        SELECT table_name, privilege_type
                        FROM information_schema.role_table_grants
                        WHERE table_schema = current_schema() AND grantee = :role
                        """
                    ),
                    {"role": role},
                )
            }
        for table_name in ("operations", "artifacts", *migration._LANDSCAPE_NEW_TABLE_NAMES):
            assert {(table_name, privilege) for privilege in {"SELECT", "INSERT", "UPDATE", "DELETE"}} <= grants
        assert (SCHEMA_IDENTITY_TABLE_NAME, "SELECT") in grants
        assert (SCHEMA_IDENTITY_TABLE_NAME, "UPDATE") not in grants
    finally:
        session.dispose()
        landscape.dispose()
        for owner_url in (database_pair.session_url, database_pair.landscape_url):
            with _psycopg_connect(owner_url) as owner:
                owner.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
        with _psycopg_connect(database_pair.postgres_url) as admin:
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
