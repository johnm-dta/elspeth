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
    """Read all identity rows so missing or malformed state stays visible.

    Do not coerce database values: a drifted TEXT epoch containing ``"24"``
    must not masquerade as the current INTEGER epoch, and arbitrary text must
    classify as a mismatch instead of escaping as a raw conversion error.
    The static empty/zero sentinels are impossible in a valid identity row.
    """
    rows = connection.execute(
        select(
            table.c.application_id,
            table.c.store_kind,
            table.c.schema_epoch,
        ).order_by(table.c.singleton_id)
    )
    return tuple(
        SchemaIdentity(
            application_id=row.application_id if type(row.application_id) is str else "",
            store_kind=row.store_kind if type(row.store_kind) is str else "",
            schema_epoch=row.schema_epoch if type(row.schema_epoch) is int else 0,
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
