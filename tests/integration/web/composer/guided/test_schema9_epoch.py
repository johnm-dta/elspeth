from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import SESSION_SCHEMA_EPOCH
from elspeth.web.sessions.schema import SessionSchemaError, initialize_session_schema


def test_guided_confirmation_admission_hard_cut_runs_inside_session_epoch_36() -> None:
    assert SESSION_SCHEMA_EPOCH == 36
    assert SQLITE_SCHEMA_EPOCH == 29


def test_epoch_35_session_store_fails_before_schema_use(tmp_path: Path) -> None:
    path = tmp_path / "epoch-35.db"
    engine = create_session_engine(f"sqlite:///{path}")
    initialize_session_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("UPDATE elspeth_schema_identity SET schema_epoch = 35 WHERE store_kind = 'session'"))
        connection.execute(text("PRAGMA user_version = 35"))

    with pytest.raises(SessionSchemaError, match=r"SESSION_SCHEMA_EPOCH=36.*Delete the session DB file and restart"):
        initialize_session_schema(engine)
