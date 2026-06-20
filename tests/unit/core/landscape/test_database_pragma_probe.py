"""Tests for the Landscape SQLite PRAGMA startup probe (elspeth-458ef96b78).

The Landscape DB is the legal audit record.  Tier-1 doctrine requires that
the durability and concurrency PRAGMAs the audit subsystem depends on are
verified to have actually taken effect at engine creation — if SQLite
silently refuses any of them, we MUST refuse to open the database rather
than record audit events under weaker-than-contracted guarantees.

This module exercises two contracts:

1. **Positive contract** — under normal conditions, a fresh LandscapeDB
   reports the canonical PRAGMA set (WAL + synchronous=NORMAL +
   foreign_keys=ON + busy_timeout=5000).  This is the happy-path proof
   that the production hook applies what it claims to apply.

2. **Negative contract** — when the configure hook is sabotaged to skip
   ``synchronous=NORMAL``, opening the DB raises
   :class:`AuditIntegrityError`.  This is the fault-injection proof that
   the probe will catch a regression in the hook, a future SQLite version
   that silently rejects a PRAGMA, or a downstream caller that overrides
   ``_configure_sqlite``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB


class TestPragmaProbePositive:
    """Under normal construction, all four invariants are honoured."""

    def test_file_backed_reports_canonical_pragmas(self, tmp_path: Path) -> None:
        """A freshly opened file-backed Landscape DB has WAL + NORMAL + FK + busy_timeout."""
        db_path = tmp_path / "pragma_probe.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        try:
            with db.connection() as conn:
                assert conn.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"
                # synchronous=NORMAL is integer 1 in SQLite's PRAGMA report.
                assert conn.execute(text("PRAGMA synchronous")).scalar_one() == 1
                assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
                assert conn.execute(text("PRAGMA busy_timeout")).scalar_one() == 5000
        finally:
            db.close()

    def test_memory_backed_reports_canonical_pragmas(self) -> None:
        """Memory-backed DBs report journal_mode=memory but otherwise match the file contract."""
        db = LandscapeDB.in_memory()
        try:
            with db.connection() as conn:
                # SQLite has no on-disk journal for :memory:; WAL silently
                # downgrades to 'memory'.  The probe accepts that, which
                # this test pins.
                assert conn.execute(text("PRAGMA journal_mode")).scalar_one() == "memory"
                assert conn.execute(text("PRAGMA synchronous")).scalar_one() == 1
                assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
                assert conn.execute(text("PRAGMA busy_timeout")).scalar_one() == 5000
        finally:
            db.close()


class TestPragmaProbeNegative:
    """If the configure hook fails to apply an invariant, opening MUST fail."""

    def test_missing_synchronous_pragma_raises_audit_integrity_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sabotage the hook to skip synchronous=NORMAL; the probe must catch it.

        This is the regression gate the ticket asks for: the probe exists
        precisely so that a buggy/incomplete hook (or a future SQLite that
        silently rejects a PRAGMA) is converted into an immediate refusal
        rather than a silent downgrade of audit-integrity guarantees.
        """

        def _sabotaged_configure(engine: Engine) -> None:
            """Apply every PRAGMA EXCEPT synchronous=NORMAL.

            SQLite defaults synchronous to FULL (2), so the probe will
            observe 2 where it expects 1 (NORMAL) and must raise.
            """

            @event.listens_for(engine, "connect")
            def _hook(dbapi_connection: object, connection_record: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
                cursor.execute("PRAGMA journal_mode=WAL")
                # Intentionally omitted: PRAGMA synchronous=NORMAL
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

        monkeypatch.setattr(LandscapeDB, "_configure_sqlite", staticmethod(_sabotaged_configure))

        with pytest.raises(AuditIntegrityError) as excinfo:
            LandscapeDB.in_memory()

        # Message must name the offending PRAGMA so an operator can act
        # without re-instrumenting.  SQLite default synchronous is FULL (2).
        msg = str(excinfo.value)
        assert "synchronous" in msg
        assert "'1'" in msg  # expected
        assert "'2'" in msg  # observed (SQLite default = FULL)

    def test_wrong_busy_timeout_raises_audit_integrity_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hook that sets a non-canonical busy_timeout must also be caught.

        Pinning a second negative case proves the probe checks all four
        invariants, not just synchronous — so a future PRAGMA addition
        that we forget to wire into the hook will be flagged.
        """

        def _sabotaged_configure(engine: Engine) -> None:
            @event.listens_for(engine, "connect")
            def _hook(dbapi_connection: object, connection_record: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                # Wrong busy_timeout — exercises the invariant comparison.
                cursor.execute("PRAGMA busy_timeout=42")
                cursor.close()

        monkeypatch.setattr(LandscapeDB, "_configure_sqlite", staticmethod(_sabotaged_configure))

        with pytest.raises(AuditIntegrityError) as excinfo:
            LandscapeDB.in_memory()

        msg = str(excinfo.value)
        assert "busy_timeout" in msg
        assert "'5000'" in msg
        assert "'42'" in msg
