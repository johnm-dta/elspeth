"""Cross-dialect schema identity table and bootstrap proofs."""

from sqlalchemy import select, text, update

from elspeth.core.landscape.database import LandscapeDB, LandscapeSchemaShape, probe_schema_shape
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH
from elspeth.core.landscape.schema import schema_identity_table as landscape_identity_table
from elspeth.core.schema_identity import SCHEMA_IDENTITY_APPLICATION_ID
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import SESSION_SCHEMA_EPOCH
from elspeth.web.sessions.models import schema_identity_table as session_identity_table
from elspeth.web.sessions.schema import initialize_session_schema, probe_current_schema


def test_landscape_bootstrap_stamps_cross_dialect_identity() -> None:
    database = LandscapeDB.in_memory()

    with database.engine.connect() as connection:
        row = connection.execute(select(landscape_identity_table)).one()

    assert row.singleton_id == 1
    assert row.application_id == SCHEMA_IDENTITY_APPLICATION_ID
    assert row.store_kind == "landscape"
    assert row.schema_epoch == SQLITE_SCHEMA_EPOCH


def test_session_bootstrap_stamps_cross_dialect_identity() -> None:
    engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)

    with engine.connect() as connection:
        row = connection.execute(select(session_identity_table)).one()

    assert row.singleton_id == 1
    assert row.application_id == SCHEMA_IDENTITY_APPLICATION_ID
    assert row.store_kind == "session"
    assert row.schema_epoch == SESSION_SCHEMA_EPOCH


def test_landscape_probe_rejects_semantic_only_identity_epoch_drift() -> None:
    database = LandscapeDB.in_memory()
    with database.engine.begin() as connection:
        connection.execute(update(landscape_identity_table).values(schema_epoch=SQLITE_SCHEMA_EPOCH - 1))

    assert probe_schema_shape(database.engine) is LandscapeSchemaShape.DIVERGENT


def test_session_probe_rejects_semantic_only_identity_store_drift() -> None:
    engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)
    with engine.begin() as connection:
        connection.execute(update(session_identity_table).values(store_kind="landscape"))

    assert probe_current_schema(engine) is False


def test_landscape_probe_classifies_non_numeric_identity_epoch_as_divergent() -> None:
    database = LandscapeDB.in_memory()
    with database.engine.begin() as connection:
        connection.execute(text("UPDATE elspeth_schema_identity SET schema_epoch = 'not-an-integer'"))

    assert probe_schema_shape(database.engine) is LandscapeSchemaShape.DIVERGENT


def test_session_probe_classifies_non_numeric_identity_epoch_as_stale() -> None:
    engine = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("UPDATE elspeth_schema_identity SET schema_epoch = 'not-an-integer'"))

    assert probe_current_schema(engine) is False
