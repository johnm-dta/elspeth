"""Schema tests for the rev-4 audit_access_log table (spec §6.3).

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions import models
from tests.unit.web.conftest import _make_session


def test_table_exists():
    assert "audit_access_log" in models.metadata.tables


def test_writer_principal_check(engine):
    now = datetime.now(UTC)
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Accepted values
        for principal in ("audit_grade_view", "admin_tool"):
            conn.execute(
                insert(models.audit_access_log_table).values(
                    id=f"al_{principal}",
                    timestamp=now,
                    session_id="s1",
                    requesting_principal="user_42",
                    request_path="/api/sessions/s1/messages",
                    query_args={"include_tool_rows": True},
                    ip_address="10.0.0.1",
                    writer_principal=principal,
                )
            )
        # Rejected
        with pytest.raises(IntegrityError, match="ck_audit_access_log_writer_principal"):
            conn.execute(
                insert(models.audit_access_log_table).values(
                    id="al_rogue",
                    timestamp=now,
                    session_id="s1",
                    requesting_principal="user_42",
                    request_path="/api/sessions/s1/messages",
                    query_args={},
                    writer_principal="rogue_view",
                )
            )


def test_session_delete_cascades_audit_access_log(engine):
    """Archive/delete lifecycle guard.

    ``archive_session`` ultimately deletes the parent session row after
    deleting the child tables it already knows about. Phase 3 writes
    ``audit_access_log`` rows, so the FK must cascade or archived sessions
    that have been viewed with ``include_tool_rows`` will fail deletion.
    """
    now = datetime.now(UTC)
    with engine.begin() as conn:
        _make_session(conn, session_id="s_archive")
        conn.execute(
            insert(models.audit_access_log_table).values(
                id="log1",
                timestamp=now,
                session_id="s_archive",
                requesting_principal="alice",
                request_path="/api/sessions/s_archive/messages",
                query_args={"include_tool_rows": True},
                ip_address=None,
                writer_principal="audit_grade_view",
            )
        )
        conn.execute(delete(models.sessions_table).where(models.sessions_table.c.id == "s_archive"))
        remaining = conn.execute(
            select(models.audit_access_log_table.c.id).where(models.audit_access_log_table.c.session_id == "s_archive")
        ).fetchall()
        assert remaining == []
