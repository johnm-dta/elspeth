"""``NodeStateRepository._acquire_routing_group_authority`` lock SQL discipline.

Routing-group ownership is serialized through the two-argument advisory
lock under the classid reserved in
``src/elspeth/contracts/advisory_locks.py``. Every function in the lock
SQL must be ``pg_catalog``-qualified so a writable schema earlier in
``search_path`` cannot shadow ``pg_advisory_xact_lock``/``hashtext`` and
silently change the locked value (elspeth-eb0fd1543a).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from elspeth.contracts.advisory_locks import ELSPETH_ROUTING_GROUP_LOCK_CLASSID
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.node_states import NodeStateRepository


class _Connection:
    def __init__(self, dialect_name: str) -> None:
        self.dialect = SimpleNamespace(name=dialect_name)
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def exec_driver_sql(self, statement: str, parameters: tuple[object, ...]) -> None:
        self.statements.append((statement, parameters))


def test_postgresql_uses_registered_classid_and_pg_catalog_qualification() -> None:
    conn = _Connection("postgresql")
    NodeStateRepository._acquire_routing_group_authority(conn, routing_group_id="group-1")
    assert conn.statements == [
        (
            "SELECT pg_catalog.pg_advisory_xact_lock(%s, pg_catalog.hashtext(%s))",
            (ELSPETH_ROUTING_GROUP_LOCK_CLASSID, "group-1"),
        )
    ]


def test_sqlite_is_noop() -> None:
    conn = _Connection("sqlite")
    NodeStateRepository._acquire_routing_group_authority(conn, routing_group_id="group-1")
    assert conn.statements == []


def test_unsupported_dialect_raises() -> None:
    conn = _Connection("mysql")
    with pytest.raises(LandscapeRecordError, match="unsupported"):
        NodeStateRepository._acquire_routing_group_authority(conn, routing_group_id="group-1")
    assert conn.statements == []
