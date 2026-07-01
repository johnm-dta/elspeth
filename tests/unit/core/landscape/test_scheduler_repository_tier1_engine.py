"""Tier-1 engine gate tests for TokenSchedulerRepository (elspeth-34d83daedc).

``TokenSchedulerRepository`` must refuse any engine that bypasses
``LandscapeDB._configure_sqlite``.  The scheduler touches the audit DB —
Tier-1 doctrine: crash on anything that does not meet the PRAGMA invariants
rather than silently operate without referential integrity or WAL journalling.

Two contracts are exercised here:

1. **Negative contract** — constructing with a bare ``create_engine()`` that
   has no ``foreign_keys=ON`` / ``journal_mode=WAL`` must raise
   :class:`AuditIntegrityError` immediately, before any scheduler work begins.

2. **Positive contract** — constructing with ``LandscapeDB.in_memory().engine``
   (a ``Tier1Engine``) must succeed without raising.
"""

from __future__ import annotations

from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import metadata


class _PostgresEngineWithoutPragmas:
    dialect = type("_Dialect", (), {"name": "postgresql"})()

    def connect(self) -> None:
        raise AssertionError("SQLite PRAGMA probe should not run for PostgreSQL engines")


class TestSchedulerRepositoryTier1EngineGate:
    """TokenSchedulerRepository must reject engines that bypass LandscapeDB."""

    def test_bare_engine_raises_audit_integrity_error(self) -> None:
        """A bare create_engine() with no PRAGMA configuration must be rejected.

        SQLite defaults ``foreign_keys=OFF`` and ``journal_mode=delete``, so
        any bare engine misses both invariants the scheduler depends on.  The
        constructor must raise before any scheduler state is initialised.
        """
        engine = create_engine("sqlite:///:memory:", echo=False)
        # Do NOT call LandscapeDB._configure_sqlite — this is the failure case.
        metadata.create_all(engine)

        with pytest.raises(AuditIntegrityError) as excinfo:
            TokenSchedulerRepository(engine)  # type: ignore[arg-type]  # intentional: testing the runtime gate

        msg = str(excinfo.value)
        # Message must name the violated PRAGMA so an operator can diagnose.
        assert "foreign_keys" in msg
        assert "Tier-1" in msg or "audit-integrity" in msg.lower() or "TokenSchedulerRepository" in msg

    def test_landscape_db_engine_succeeds(self) -> None:
        """An engine vended by LandscapeDB must be accepted without raising.

        ``LandscapeDB.in_memory()`` calls both ``_configure_sqlite`` and
        ``_verify_sqlite_pragmas`` before vending its engine.  The engine
        property returns a ``Tier1Engine`` (static NewType brand).  The
        scheduler's constructor must not raise.
        """
        db = LandscapeDB.in_memory()
        try:
            # LandscapeDB.in_memory() already creates all tables — no need to
            # call metadata.create_all(engine) again.
            repo = TokenSchedulerRepository(db.engine)
            # Basic sanity: the repo is usable (no exception on a read).
            assert repo.count_active_work(run_id="nonexistent-run") == 0
        finally:
            db.close()

    def test_postgresql_engine_does_not_run_sqlite_pragma_probe(self) -> None:
        """Non-SQLite engines must not be probed with SQLite-only PRAGMAs."""
        engine = Tier1Engine(cast(Engine, _PostgresEngineWithoutPragmas()))

        repo = TokenSchedulerRepository(engine)

        assert isinstance(repo, TokenSchedulerRepository)
