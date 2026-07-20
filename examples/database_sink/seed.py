#!/usr/bin/env python
"""Provision the operator-owned target table and exactly-once effect ledger.

The ``database`` sink is a *recoverable, exactly-once* publisher: it appends
rows to a target table you own and records each committed batch in a
target-side ``_elspeth_*`` effect ledger, so a resumed run republishes nothing.
By that contract the runtime **never** creates these tables for you — the
operator provisions them once, up front. This script is that provisioning step
for the example; in production you would run the equivalent DDL against your
own database (SQLite or PostgreSQL) as part of deployment.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine

from elspeth.plugins.sinks.database_sink import database_effect_ledger_table

DB_PATH = Path(__file__).parent / "output" / "deals.db"
TARGET_TABLE = "high_value_deals"
LEDGER_TABLE = "_elspeth_sink_effects"  # must be a namespaced `_elspeth_` identifier


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    metadata = MetaData()

    # Operator-owned target table. The sink's `fixed` schema is matched by column
    # NAME at preflight; declared types are advisory under SQLite's flexible typing.
    Table(
        TARGET_TABLE,
        metadata,
        Column("id", Integer),
        Column("customer", Text),
        Column("amount", Integer),
        Column("category", Text),
        Column("region", Text),
    )

    # Version-1 exactly-once effect ledger (fixed schema owned by the sink runtime).
    database_effect_ledger_table(metadata, LEDGER_TABLE)

    metadata.create_all(engine)
    engine.dispose()
    print(f"Provisioned {TARGET_TABLE!r} + ledger {LEDGER_TABLE!r} in {DB_PATH}")  # noqa: T201


if __name__ == "__main__":
    main()
