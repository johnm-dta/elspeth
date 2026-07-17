"""Run-scoped token ownership regressions for the current schema."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from elspeth.contracts import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import _REQUIRED_COMPOSITE_FOREIGN_KEYS
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, token_parents_table, tokens_table
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


def _forge_token_run(*, setup, token_id: str, forged_run_id: str) -> None:
    """Simulate a legacy bad writer that disabled FK enforcement."""
    raw = setup.db.engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF")
        cursor.execute("UPDATE tokens SET run_id = ? WHERE token_id = ?", (forged_run_id, token_id))
        assert cursor.rowcount == 1
        raw.commit()
        cursor.execute("PRAGMA foreign_keys = ON")
        assert cursor.execute("PRAGMA foreign_keys").fetchone() == (1,)
    finally:
        raw.close()


def test_current_epoch_preserves_token_row_run_ownership_for_sqlite_and_postgres() -> None:
    assert SQLITE_SCHEMA_EPOCH == 29
    assert (
        "tokens",
        ("row_id", "run_id"),
        "rows",
        ("row_id", "run_id"),
    ) in _REQUIRED_COMPOSITE_FOREIGN_KEYS

    postgres_ddl = str(CreateTable(tokens_table).compile(dialect=postgresql.dialect()))
    assert "FOREIGN KEY(row_id, run_id) REFERENCES rows (row_id, run_id)" in postgres_ddl


def test_fresh_sqlite_rejects_cross_run_token_parent() -> None:
    setup = make_recorder_with_run(run_id="parent-child-run-A", source_node_id="source-A")
    child_row = setup.data_flow.create_row(
        "parent-child-run-A",
        "source-A",
        row_index=0,
        data={"side": "child"},
        source_row_index=0,
        ingest_sequence=0,
    )
    child = setup.data_flow.create_token(child_row.row_id)

    setup.factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="parent-child-run-B")
    source_b = register_test_node(
        setup.data_flow,
        "parent-child-run-B",
        "source-B",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    parent_row = setup.data_flow.create_row(
        "parent-child-run-B",
        source_b,
        row_index=0,
        data={"side": "parent"},
        source_row_index=0,
        ingest_sequence=0,
    )
    parent = setup.data_flow.create_token(parent_row.row_id)

    values: dict[str, object] = {
        "token_id": child.token_id,
        "parent_token_id": parent.token_id,
        "ordinal": 0,
    }
    if "run_id" in token_parents_table.c:
        values["run_id"] = "parent-child-run-A"

    with pytest.raises(IntegrityError), setup.db.write_connection() as conn:
        conn.execute(token_parents_table.insert().values(**values))


def test_fresh_sqlite_rejects_cross_run_token_row_pair() -> None:
    setup = make_recorder_with_run(run_id="run-A", source_node_id="source-A")
    row = setup.data_flow.create_row(
        "run-A",
        "source-A",
        row_index=0,
        data={"value": 1},
        row_id="row-A",
        source_row_index=0,
        ingest_sequence=0,
    )
    setup.factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")

    with pytest.raises(IntegrityError), setup.db.engine.begin() as conn:
        conn.execute(
            tokens_table.insert().values(
                token_id="forged-token",
                row_id=row.row_id,
                run_id="run-B",
                created_at=datetime.now(UTC),
            )
        )


def test_read_path_rejects_legacy_forged_token_run_mismatch() -> None:
    setup = make_recorder_with_run(run_id="run-A", source_node_id="source-A")
    row = setup.data_flow.create_row(
        "run-A",
        "source-A",
        row_index=0,
        data={"value": 1},
        row_id="row-A",
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.data_flow.create_token(row.row_id, token_id="token-A")
    setup.factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
    _forge_token_run(setup=setup, token_id=token.token_id, forged_run_id="run-B")

    with pytest.raises(AuditIntegrityError, match=r"token-A.*row-A.*run-A.*run-B"):
        setup.data_flow._resolve_token_ownership(token.token_id)
