"""Property-test fixtures for Phase 3 compose-loop persistence."""

from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def populated_audit_db():
    """Real initialized engine reserved for mixed compose-loop audit traces."""

    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    return engine
