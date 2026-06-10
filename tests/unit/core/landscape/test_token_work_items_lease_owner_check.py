"""Tier-1 CHECK-constraint regression for token_work_items.lease_owner.

The schema constraint ``ck_token_work_items_lease_owner_required_when_leased``
enforces that any row persisted with ``status='leased'`` carries a non-empty
``lease_owner``. The constraint is the structural complement of the
``recover_expired_leases`` ABA-window fix (elspeth-28aaa36a62) — that
recovery path treats ``lease_owner=NULL`` as a recoverable wedge, so the
CHECK closes the gap by refusing to *create* such a wedge in the first
place.

The constraint's literal MUST match ``TokenWorkStatus.LEASED.value``
*exactly*. ``TokenWorkStatus`` is a ``StrEnum`` whose ``.value`` is
lowercase ``"leased"``. A case-mismatched literal (e.g. ``'LEASED'``)
makes both arms of the CHECK trivially satisfied for every row and
silently nullifies the Tier-1 invariant (elspeth-36d5635402).

These tests assert the constraint actually fires, exercising the schema
through a Tier-1 ``LandscapeDB`` engine. Foreign keys are disabled in the
test connection so the test isolates the CHECK from FK requirements on
``tokens``, ``rows``, and ``nodes`` parent tables; the CHECK runs
independently of ``PRAGMA foreign_keys`` in SQLite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import token_work_items_table


def _base_row(work_item_id: str) -> dict[str, object]:
    """Return the minimal column set for a valid token_work_items insert.

    All NOT-NULL columns populated; lease_owner left to the caller so each
    test can vary the lease-related fields independently.
    """
    now = datetime.now(UTC)
    return {
        "work_item_id": work_item_id,
        "run_id": "run-test",
        "token_id": "tok-test",
        "row_id": "row-test",
        "node_id": "node-test",
        "step_index": 1,
        "ingest_sequence": 1,
        "row_payload_json": "{}",
        "attempt": 1,
        "available_at": now,
        "created_at": now,
        "updated_at": now,
    }


class TestLeaseOwnerCheckConstraint:
    """Constraint must reject LEASED rows with empty/null lease_owner."""

    def test_leased_row_with_null_lease_owner_rejected(self) -> None:
        """A row with status='leased' and lease_owner=NULL must violate the CHECK.

        Pre-fix this row silently committed because the literal 'LEASED'
        never matched the lowercase enum value, so the second arm
        (``status != 'LEASED'``) was always True.
        """
        db = LandscapeDB("sqlite:///:memory:")
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            row = _base_row("wi-leased-null")
            row["status"] = TokenWorkStatus.LEASED.value
            row["lease_owner"] = None

            with pytest.raises(IntegrityError) as excinfo:
                conn.execute(token_work_items_table.insert().values(**row))

            assert "ck_token_work_items_lease_owner_required_when_leased" in str(excinfo.value)

    def test_leased_row_with_empty_lease_owner_rejected(self) -> None:
        """A row with status='leased' and lease_owner='' must violate the CHECK.

        ``length(lease_owner) > 0`` is the second clause; an empty string is
        as wedged as NULL for recovery purposes.
        """
        db = LandscapeDB("sqlite:///:memory:")
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            row = _base_row("wi-leased-empty")
            row["status"] = TokenWorkStatus.LEASED.value
            row["lease_owner"] = ""

            with pytest.raises(IntegrityError) as excinfo:
                conn.execute(token_work_items_table.insert().values(**row))

            assert "ck_token_work_items_lease_owner_required_when_leased" in str(excinfo.value)

    def test_leased_row_with_owner_succeeds(self) -> None:
        """Positive case: status='leased' + non-empty lease_owner must commit.

        Guards against a future overcorrection that breaks the happy path.
        """
        db = LandscapeDB("sqlite:///:memory:")
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            row = _base_row("wi-leased-ok")
            row["status"] = TokenWorkStatus.LEASED.value
            row["lease_owner"] = "worker-1"
            row["lease_expires_at"] = datetime.now(UTC)

            conn.execute(token_work_items_table.insert().values(**row))
            conn.commit()

    def test_non_leased_row_with_null_lease_owner_succeeds(self) -> None:
        """Non-LEASED rows may carry NULL lease_owner — that's the normal
        state for READY / BLOCKED / PENDING_SINK / TERMINAL / FAILED.
        The CHECK must only apply when status='leased'.
        """
        db = LandscapeDB("sqlite:///:memory:")
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            row = _base_row("wi-ready-null")
            row["status"] = TokenWorkStatus.READY.value
            row["lease_owner"] = None

            conn.execute(token_work_items_table.insert().values(**row))
            conn.commit()

    def test_check_literal_matches_enum_value(self) -> None:
        """Static guard: the constraint's literal must equal ``TokenWorkStatus.LEASED.value``.

        Catches a future regression that re-introduces case drift in the
        schema definition — keeps the schema literal honest with the enum.
        """
        check = next(
            c
            for c in token_work_items_table.constraints
            if getattr(c, "name", "") == "ck_token_work_items_lease_owner_required_when_leased"
        )
        sqltext = str(check.sqltext)  # type: ignore[attr-defined]
        assert f"'{TokenWorkStatus.LEASED.value}'" in sqltext, (
            f"CHECK literal must match TokenWorkStatus.LEASED.value={TokenWorkStatus.LEASED.value!r}; got sqltext={sqltext!r}"
        )
        assert "'LEASED'" not in sqltext, "CHECK literal must not use uppercase 'LEASED' (case mismatch silently nullifies the constraint)."
