"""Pin the blob_inline_resolutions audit table shape."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import insert, inspect, select, text
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    SESSION_SCHEMA_EPOCH,
    blob_inline_resolutions_table,
    blobs_table,
    composition_states_table,
    runs_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


def test_blob_inline_resolutions_table_exists_with_expected_columns(engine) -> None:
    inspector = inspect(engine)

    assert "blob_inline_resolutions" in inspector.get_table_names()
    assert {column["name"] for column in inspector.get_columns("blob_inline_resolutions")} == {
        "run_id",
        "attempt",
        "field_path",
        "blob_id",
        "content_hash",
        "byte_length",
        "mime_type",
        "encoding",
        "resolved_at",
    }


def test_blob_inline_resolutions_schema_epoch_is_18(engine) -> None:
    assert SESSION_SCHEMA_EPOCH == 18
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA user_version")).scalar_one() == 18


def test_blob_inline_resolutions_blob_id_is_historical_without_live_blob_fk(engine) -> None:
    inspector = inspect(engine)

    foreign_keys = inspector.get_foreign_keys("blob_inline_resolutions")
    constrained_columns = {column for foreign_key in foreign_keys for column in foreign_key["constrained_columns"]}

    assert "blob_id" not in constrained_columns


def test_blob_inline_resolutions_round_trip(engine) -> None:
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with engine.begin() as conn:
        _seed_run_and_blob(conn, run_id=run_id, blob_id=blob_id)
        conn.execute(
            insert(blob_inline_resolutions_table).values(
                run_id=run_id,
                attempt=1,
                field_path="node:classify.options.system_prompt",
                blob_id=blob_id,
                content_hash="a" * 64,
                byte_length=42,
                mime_type="text/plain",
                encoding="utf-8",
                resolved_at=datetime.now(UTC),
            )
        )

    with engine.connect() as conn:
        rows = conn.execute(select(blob_inline_resolutions_table)).fetchall()

    assert len(rows) == 1
    assert rows[0].field_path == "node:classify.options.system_prompt"


def test_blob_inline_resolutions_field_path_check_rejects_positional(engine) -> None:
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with (
        pytest.raises(IntegrityError, match="ck_blob_inline_resolutions_field_path"),
        engine.begin() as conn,
    ):
        _seed_run_and_blob(conn, run_id=run_id, blob_id=blob_id)
        conn.execute(
            insert(blob_inline_resolutions_table).values(
                run_id=run_id,
                attempt=1,
                field_path="transforms[2].options.system_prompt",
                blob_id=blob_id,
                content_hash="a" * 64,
                byte_length=10,
                mime_type="text/plain",
                encoding="utf-8",
                resolved_at=datetime.now(UTC),
            )
        )


def test_blob_inline_resolutions_encoding_check_rejects_unknown(engine) -> None:
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with (
        pytest.raises(IntegrityError, match="ck_blob_inline_resolutions_encoding"),
        engine.begin() as conn,
    ):
        _seed_run_and_blob(conn, run_id=run_id, blob_id=blob_id)
        conn.execute(
            insert(blob_inline_resolutions_table).values(
                run_id=run_id,
                attempt=1,
                field_path="source.options.system_prompt",
                blob_id=blob_id,
                content_hash="a" * 64,
                byte_length=10,
                mime_type="text/plain",
                encoding="ascii",
                resolved_at=datetime.now(UTC),
            )
        )


def test_blob_inline_resolutions_attempt_supports_resume(engine) -> None:
    run_id = str(uuid4())
    blob_id = str(uuid4())
    with engine.begin() as conn:
        _seed_run_and_blob(conn, run_id=run_id, blob_id=blob_id)
        for attempt in (1, 2):
            conn.execute(
                insert(blob_inline_resolutions_table).values(
                    run_id=run_id,
                    attempt=attempt,
                    field_path="source.options.system_prompt",
                    blob_id=blob_id,
                    content_hash="a" * 64,
                    byte_length=10,
                    mime_type="text/plain",
                    encoding="utf-8",
                    resolved_at=datetime.now(UTC),
                )
            )

    with engine.connect() as conn:
        rows = conn.execute(select(blob_inline_resolutions_table)).fetchall()

    assert len(rows) == 2


def _seed_run_and_blob(conn, *, run_id: str, blob_id: str) -> None:
    session_id = str(uuid4())
    state_id = str(uuid4())
    now = datetime.now(UTC)
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id="schema-test-user",
            auth_provider_type="local",
            title="Schema test",
            trust_mode="auto_commit",
            density_default="high",
            created_at=now,
            updated_at=now,
            interpretation_review_disabled=False,
        )
    )
    conn.execute(
        insert(composition_states_table).values(
            id=state_id,
            session_id=session_id,
            version=1,
            source=None,
            nodes=None,
            edges=None,
            outputs=None,
            metadata_=None,
            is_valid=True,
            validation_errors=None,
            composer_meta=None,
            created_at=now,
            derived_from_state_id=None,
            provenance="session_seed",
        )
    )
    conn.execute(
        insert(runs_table).values(
            id=run_id,
            session_id=session_id,
            state_id=state_id,
            status="pending",
            started_at=now,
            finished_at=None,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            error=None,
            landscape_run_id=None,
            pipeline_yaml=None,
        )
    )
    conn.execute(
        insert(blobs_table).values(
            id=blob_id,
            session_id=session_id,
            filename="prompt.txt",
            mime_type="text/plain",
            size_bytes=42,
            content_hash="a" * 64,
            storage_path="/tmp/prompt.txt",
            created_at=now,
            created_by="user",
            source_description=None,
            status="ready",
            creation_modality="verbatim",
        )
    )
