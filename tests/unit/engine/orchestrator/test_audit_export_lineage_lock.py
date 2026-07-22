"""``_acquire_signer_lineage_authority`` lock SQL discipline.

The signer-policy recheck and registry CAS insert are serialized per
export lineage. On PostgreSQL the lock MUST use the two-argument
advisory-lock form with the classid reserved in
``src/elspeth/contracts/advisory_locks.py``, and every function must be
``pg_catalog``-qualified so a writable schema earlier in ``search_path``
cannot shadow the lock protocol (elspeth-eb0fd1543a).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from elspeth.contracts.advisory_locks import ELSPETH_AUDIT_EXPORT_LOCK_CLASSID
from elspeth.engine.orchestrator.audit_export_effects import _acquire_signer_lineage_authority


def _key() -> SimpleNamespace:
    return SimpleNamespace(
        source_run_id="run-1",
        exporter_version="1",
        serialization_version="2",
        export_format=SimpleNamespace(value="jsonl"),
    )


class _Connection:
    def __init__(self, dialect_name: str) -> None:
        self.dialect = SimpleNamespace(name=dialect_name)
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def exec_driver_sql(self, statement: str, parameters: tuple[object, ...]) -> None:
        self.statements.append((statement, parameters))


def test_postgresql_uses_registered_classid_and_pg_catalog_qualification() -> None:
    conn = _Connection("postgresql")
    _acquire_signer_lineage_authority(conn, _key())
    assert conn.statements == [
        (
            "SELECT pg_catalog.pg_advisory_xact_lock(%s, pg_catalog.hashtext(%s))",
            (ELSPETH_AUDIT_EXPORT_LOCK_CLASSID, "run-1\x1f1\x1f2\x1fjsonl"),
        )
    ]


def test_sqlite_is_noop() -> None:
    conn = _Connection("sqlite")
    _acquire_signer_lineage_authority(conn, _key())
    assert conn.statements == []


def test_unsupported_dialect_raises() -> None:
    conn = _Connection("mysql")
    with pytest.raises(RuntimeError, match="unsupported Landscape backend"):
        _acquire_signer_lineage_authority(conn, _key())
    assert conn.statements == []
