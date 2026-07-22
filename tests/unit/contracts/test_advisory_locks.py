"""Advisory-lock classid registry invariants (ABI commitments).

Every classid in ``src/elspeth/contracts/advisory_locks.py`` is
on-the-wire ABI shared by every ELSPETH version connected to one
PostgreSQL cluster. These tests pin the exact values so an accidental
edit fails loudly, and enforce the registry's own rules: distinct
values, signed-int4 range.
"""

from __future__ import annotations

from elspeth.contracts import advisory_locks
from elspeth.contracts.advisory_locks import (
    ELSPETH_AUDIT_EXPORT_LOCK_CLASSID,
    ELSPETH_ROUTING_GROUP_LOCK_CLASSID,
    ELSPETH_SCHEMA_INIT_LOCK_CLASSID,
    ELSPETH_SESSIONS_LOCK_CLASSID,
)

_INT4_MIN = -(2**31)
_INT4_MAX = 2**31 - 1


def _registered_classids() -> dict[str, int]:
    return {name: value for name, value in vars(advisory_locks).items() if name.endswith("_LOCK_CLASSID")}


def test_classid_values_are_pinned_abi() -> None:
    """Changing any value here requires an ADR and a coordinated deploy."""
    assert ELSPETH_SESSIONS_LOCK_CLASSID == 0x454C5350
    assert ELSPETH_SCHEMA_INIT_LOCK_CLASSID == 0x53434845
    assert ELSPETH_ROUTING_GROUP_LOCK_CLASSID == 0x524F5554
    assert ELSPETH_AUDIT_EXPORT_LOCK_CLASSID == 0x41455850


def test_all_registry_constants_are_covered_by_the_pin_test() -> None:
    assert set(_registered_classids()) == {
        "ELSPETH_SESSIONS_LOCK_CLASSID",
        "ELSPETH_SCHEMA_INIT_LOCK_CLASSID",
        "ELSPETH_ROUTING_GROUP_LOCK_CLASSID",
        "ELSPETH_AUDIT_EXPORT_LOCK_CLASSID",
    }


def test_classids_are_distinct() -> None:
    classids = _registered_classids()
    assert len(set(classids.values())) == len(classids)


def test_classids_fit_signed_int4() -> None:
    for name, value in _registered_classids().items():
        assert _INT4_MIN <= value <= _INT4_MAX, name
