"""Cross-dialect database identity and schema-epoch records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import CheckConstraint, Column, Connection, Integer, MetaData, String, Table, insert, select

SCHEMA_IDENTITY_TABLE_NAME = "elspeth_schema_identity"
SCHEMA_IDENTITY_APPLICATION_ID = "elspeth"
SCHEMA_IDENTITY_SINGLETON_ID = 1

SchemaStoreKind = Literal["landscape", "session"]
SchemaIdentityMismatch = Literal["identity_row_count", "application_id", "store_kind", "schema_epoch"]


@dataclass(frozen=True, slots=True)
class SchemaIdentity:
    """One immutable observation from a database identity table."""

    application_id: str
    store_kind: str
    schema_epoch: int


def create_schema_identity_table(metadata: MetaData) -> Table:
    """Attach the shared singleton identity-table shape to ``metadata``."""
    return Table(
        SCHEMA_IDENTITY_TABLE_NAME,
        metadata,
        Column("singleton_id", Integer, primary_key=True),
        Column("application_id", String(64), nullable=False),
        Column("store_kind", String(32), nullable=False),
        Column("schema_epoch", Integer, nullable=False),
        CheckConstraint("singleton_id = 1", name="ck_elspeth_schema_identity_singleton"),
        CheckConstraint("length(application_id) > 0", name="ck_elspeth_schema_identity_application_non_blank"),
        CheckConstraint("store_kind IN ('landscape', 'session')", name="ck_elspeth_schema_identity_store_kind"),
        CheckConstraint("schema_epoch > 0", name="ck_elspeth_schema_identity_epoch_positive"),
    )


def insert_schema_identity(
    connection: Connection,
    table: Table,
    *,
    store_kind: SchemaStoreKind,
    schema_epoch: int,
) -> None:
    """Stamp a freshly created store; duplicate stamps fail closed."""
    connection.execute(
        insert(table).values(
            singleton_id=SCHEMA_IDENTITY_SINGLETON_ID,
            application_id=SCHEMA_IDENTITY_APPLICATION_ID,
            store_kind=store_kind,
            schema_epoch=schema_epoch,
        )
    )


def read_schema_identities(connection: Connection, table: Table) -> tuple[SchemaIdentity, ...]:
    """Read all identity rows so missing or non-singleton state stays visible."""
    rows = connection.execute(
        select(
            table.c.application_id,
            table.c.store_kind,
            table.c.schema_epoch,
        ).order_by(table.c.singleton_id)
    )
    return tuple(
        SchemaIdentity(
            application_id=str(row.application_id),
            store_kind=str(row.store_kind),
            schema_epoch=int(row.schema_epoch),
        )
        for row in rows
    )


def schema_identity_mismatch(
    rows: tuple[SchemaIdentity, ...],
    *,
    store_kind: SchemaStoreKind,
    schema_epoch: int,
) -> SchemaIdentityMismatch | None:
    """Return a static mismatch code, or ``None`` for the exact expected row."""
    if len(rows) != 1:
        return "identity_row_count"
    identity = rows[0]
    if identity.application_id != SCHEMA_IDENTITY_APPLICATION_ID:
        return "application_id"
    if identity.store_kind != store_kind:
        return "store_kind"
    if identity.schema_epoch != schema_epoch:
        return "schema_epoch"
    return None
