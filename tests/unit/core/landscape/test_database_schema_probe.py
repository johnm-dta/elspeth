"""Non-mutating Landscape schema-state classification tests."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from elspeth.core.landscape.database import LandscapeSchemaShape, probe_schema_shape
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, metadata, schema_identity_table
from elspeth.core.schema_identity import insert_schema_identity


@pytest.fixture
def engine() -> Engine:
    value = create_engine("sqlite:///:memory:")
    yield value
    value.dispose()


def _create_full(engine: Engine) -> None:
    metadata.create_all(engine)
    with engine.begin() as connection:
        insert_schema_identity(
            connection,
            schema_identity_table,
            store_kind="landscape",
            schema_epoch=SQLITE_SCHEMA_EPOCH,
        )


def test_zero_user_tables_are_empty(engine: Engine) -> None:
    assert probe_schema_shape(engine) is LandscapeSchemaShape.EMPTY


def test_tableless_incompatible_epoch_is_divergent(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(f"PRAGMA user_version = {SQLITE_SCHEMA_EPOCH + 1}")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.DIVERGENT


@pytest.mark.parametrize("with_landscape", [False, True])
def test_unrelated_table_is_foreign(engine: Engine, *, with_landscape: bool) -> None:
    if with_landscape:
        _create_full(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)"))
    assert probe_schema_shape(engine) is LandscapeSchemaShape.FOREIGN


def test_full_metadata_matches(engine: Engine) -> None:
    _create_full(engine)
    assert probe_schema_shape(engine) is LandscapeSchemaShape.MATCHES


@pytest.mark.parametrize("table_name", ["auth_events", "run_attributions"])
def test_missing_additive_table_is_incomplete(engine: Engine, table_name: str) -> None:
    _create_full(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP TABLE {table_name}")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.INCOMPLETE


def test_missing_additive_index_is_incomplete(engine: Engine) -> None:
    _create_full(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_tokens_run_id")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.INCOMPLETE


def test_additive_index_name_on_wrong_table_does_not_satisfy_tokens(engine: Engine) -> None:
    _create_full(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_tokens_run_id")
        conn.exec_driver_sql("CREATE INDEX ix_tokens_run_id ON rows (row_id)")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.INCOMPLETE


def test_bare_run_coordination_event_primary_key_is_divergent(engine: Engine) -> None:
    _create_full(engine)
    with engine.connect() as conn:
        ddl = conn.exec_driver_sql("SELECT sql FROM sqlite_master WHERE type='table' AND name='run_coordination_events'").scalar_one()
        assert "AUTOINCREMENT" in ddl
        conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        conn.exec_driver_sql("DROP TABLE run_coordination_events")
        conn.exec_driver_sql(ddl.replace(" PRIMARY KEY AUTOINCREMENT", " PRIMARY KEY"))
        for index in metadata.tables["run_coordination_events"].indexes:
            index.create(conn)
        conn.commit()

    assert probe_schema_shape(engine) is LandscapeSchemaShape.DIVERGENT


@pytest.mark.parametrize(
    "replacement",
    [
        # Bare keyword hidden in a block comment.
        " PRIMARY KEY /* AUTOINCREMENT */",
        # Full proof phrase hidden in a block comment: the proof regex must
        # not accept comment text as a declaration (elspeth-e2f27fb78e).
        " PRIMARY KEY /* PRIMARY KEY AUTOINCREMENT */",
        # Full proof phrase hidden in a line comment.
        " PRIMARY KEY -- PRIMARY KEY AUTOINCREMENT\n",
    ],
)
def test_autoincrement_in_sql_comment_does_not_prove_table_shape(engine: Engine, replacement: str) -> None:
    _create_full(engine)
    with engine.connect() as conn:
        ddl = conn.exec_driver_sql("SELECT sql FROM sqlite_master WHERE type='table' AND name='run_coordination_events'").scalar_one()
        conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        conn.exec_driver_sql("DROP TABLE run_coordination_events")
        conn.exec_driver_sql(ddl.replace(" PRIMARY KEY AUTOINCREMENT", replacement))
        for index in metadata.tables["run_coordination_events"].indexes:
            index.create(conn)
        conn.commit()

    assert probe_schema_shape(engine) is LandscapeSchemaShape.DIVERGENT


def test_missing_core_table_is_divergent(engine: Engine) -> None:
    _create_full(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE validation_errors")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.DIVERGENT


def test_additive_gap_plus_surviving_shape_error_is_divergent(engine: Engine) -> None:
    _create_full(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE auth_events")
        conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN unexpected TEXT")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.DIVERGENT


@pytest.mark.parametrize("epoch", [1, SQLITE_SCHEMA_EPOCH + 1])
def test_incompatible_nonzero_epoch_is_divergent(engine: Engine, epoch: int) -> None:
    _create_full(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"PRAGMA user_version = {epoch}")
    assert probe_schema_shape(engine) is LandscapeSchemaShape.DIVERGENT


def test_connection_probe_reuses_supplied_connection(engine: Engine) -> None:
    with engine.connect() as conn:
        assert probe_schema_shape(conn) is LandscapeSchemaShape.EMPTY
