"""Tests pinning the SessionServiceImpl constructor signature.

Phase 1 extends the constructor with required ``telemetry`` and ``log``
arguments so that ``persist_compose_turn`` and the audit-failure
disposition path can emit OTel counters and (only when the audit
system itself fails) log diagnostics. The signature is part of the
service's public contract; this test prevents accidental drift.
"""

from __future__ import annotations

import pytest
import structlog
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


def test_constructor_accepts_telemetry_and_log(engine, tmp_path):
    telem = build_sessions_telemetry()
    log = structlog.get_logger("test")
    service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=telem,
        log=log,
    )
    assert service._telemetry is telem
    assert service._log is log


def test_constructor_rejects_missing_telemetry(engine, tmp_path):
    with pytest.raises(TypeError, match="telemetry"):
        SessionServiceImpl(engine, data_dir=tmp_path)  # type: ignore[call-arg]


def test_constructor_rejects_missing_log(engine, tmp_path):
    telem = build_sessions_telemetry()
    with pytest.raises(TypeError, match="log"):
        SessionServiceImpl(engine, data_dir=tmp_path, telemetry=telem)  # type: ignore[call-arg]
