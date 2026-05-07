"""Tests for current-schema session database bootstrap."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, inspect, text
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, metadata, sessions_table
from elspeth.web.sessions.schema import SessionSchemaError, initialize_session_schema


@pytest.fixture
def engine():
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


def test_initialize_session_schema_creates_current_schema_without_alembic_table() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)

    inspector = inspect(eng)
    assert set(inspector.get_table_names()) == set(metadata.tables)
    assert "alembic_version" not in inspector.get_table_names()
    run_columns = {column["name"] for column in inspector.get_columns("runs")}
    assert "rows_routed_success" in run_columns
    assert "rows_routed_failure" in run_columns
    assert "content_hash" in {column["name"] for column in inspector.get_columns("blobs")}
    assert "ck_blobs_ready_hash" in {check["name"] for check in inspector.get_check_constraints("blobs")}


def test_initialize_session_schema_is_idempotent_for_current_schema() -> None:
    eng = create_session_engine("sqlite:///:memory:")

    initialize_session_schema(eng)
    initialize_session_schema(eng)


def test_initialize_session_schema_rejects_legacy_alembic_database() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('007')"))
        conn.execute(text("CREATE TABLE _alembic_tmp_blobs (id VARCHAR PRIMARY KEY)"))

    with pytest.raises(SessionSchemaError, match="current V0 schema"):
        initialize_session_schema(eng)


def test_initialize_session_schema_rejects_partial_stale_schema() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id VARCHAR PRIMARY KEY)"))

    with pytest.raises(SessionSchemaError, match="current V0 schema"):
        initialize_session_schema(eng)


def test_current_schema_enforces_ready_blob_hash_check(engine) -> None:
    session_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                auth_provider_type="local",
                title="Test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

        with pytest.raises(IntegrityError):
            conn.execute(
                insert(blobs_table).values(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    filename="artifact.txt",
                    mime_type="text/plain",
                    size_bytes=4,
                    content_hash=None,
                    storage_path="blobs/artifact.txt",
                    created_at=datetime.now(UTC),
                    created_by="user",
                    status="ready",
                )
            )
