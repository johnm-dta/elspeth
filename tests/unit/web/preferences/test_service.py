"""Tests for PreferencesService.

Test isolation convention: each test uses a distinct ``user_id`` matching
its scenario name (e.g., ``"alice-get-default"``, ``"alice-update-persist"``).
This eliminates accidental cross-test state sharing without the overhead of
rebuilding the schema per test. The ``engine`` / ``service`` fixtures are
module-scoped; isolation is via disjoint user namespaces, not per-test
teardown.

SQLite engine uses ``StaticPool`` + ``check_same_thread=False`` so the
worker-thread DB calls (via ``run_sync_in_worker``) share the same
in-memory database as the test-thread setup. Same pattern as the
``composer_test_client`` fixture in ``tests/integration/web/conftest.py``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from elspeth.web.preferences.models import UpdateComposerPreferencesRequest
from elspeth.web.preferences.service import PreferencesService
from elspeth.web.sessions.models import metadata


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def service(engine):
    return PreferencesService(engine, now=lambda: datetime(2026, 5, 15, tzinfo=UTC))


def test_get_for_new_user_returns_guided_default(service):
    """A user with no row gets the server-side default 'guided'."""
    prefs = asyncio.run(service.get_composer_preferences("alice-get-default"))
    assert prefs.default_mode == "guided"
    assert prefs.banner_dismissed_at is None


def test_update_persists_and_round_trips(service):
    payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    result = asyncio.run(service.update_composer_preferences("alice-update-persist", payload))
    assert result.default_mode == "freeform"
    prefs = asyncio.run(service.get_composer_preferences("alice-update-persist"))
    assert prefs.default_mode == "freeform"


def test_partial_update_only_touches_provided_fields(service):
    asyncio.run(
        service.update_composer_preferences(
            "alice-partial",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )
    stamp = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-partial",
            UpdateComposerPreferencesRequest(banner_dismissed_at=stamp),
        )
    )
    prefs = asyncio.run(service.get_composer_preferences("alice-partial"))
    assert prefs.default_mode == "freeform"  # not reset by the partial update
    # SQLite strips tzinfo on DateTime(timezone=True) round-trips; compare
    # the naive value. The Pydantic model accepts datetime|None either way.
    assert prefs.banner_dismissed_at is not None
    assert prefs.banner_dismissed_at.replace(tzinfo=None) == stamp.replace(tzinfo=None)


def test_users_are_isolated(service):
    asyncio.run(service.update_composer_preferences("alice-isolated", UpdateComposerPreferencesRequest(default_mode="freeform")))
    bob_prefs = asyncio.run(service.get_composer_preferences("bob-isolated"))
    assert bob_prefs.default_mode == "guided"


def test_row_to_prefs_guards_corrupt_mode(service):
    """Tier-1 read guard: stored value outside the literal raises.

    The DB-level CHECK constraint blocks corrupt writes via SQLAlchemy
    inserts (covered by ``test_default_composer_mode_check_constraint_closes_the_enum``
    in the schema suite). The read guard tested here is the second line
    of defence — it catches CHECK-bypass scenarios (direct SQLite shell
    writes, schema-version drift, on-disk corruption, tampering). We
    exercise the guard directly on the conversion function rather than
    mock-inserting an illegal row, because the CHECK would (correctly)
    reject any such insert.
    """
    bad_row = SimpleNamespace(
        default_composer_mode="kiosk",
        banner_dismissed_at=None,
        updated_at=datetime.now(UTC),
    )
    with pytest.raises(RuntimeError, match="kiosk"):
        service._row_to_prefs(bad_row, "alice-corrupt")


def test_patch_returns_written_values_not_a_reread(service):
    """Finding 7: PATCH must NOT re-read the row after writing.

    If ``update_composer_preferences`` were implemented as
    ``write_row(); return await get_composer_preferences(user_id)``, then
    a pre-existing corrupt row would cause the post-write re-read to
    crash on the Tier-1 guard — even though the PATCH itself succeeded.

    We verify the no-re-read contract by mutating the row's stored mode
    to a value the guard would reject, performing a valid PATCH, and
    confirming the response is the written value (not a guard-crash).

    The CHECK constraint normally forbids writing 'kiosk'; we bypass it
    with PRAGMA ignore_check_constraints to simulate the legitimate
    failure scenario (DB-shell write, schema drift, etc.).
    """
    user = "alice-finding-7"
    # Seed a valid row first via the normal write path.
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))
    # Then mutate the stored mode to a corrupt value (CHECK bypass via
    # PRAGMA — simulates the out-of-band corruption the guard exists to
    # catch).
    from sqlalchemy import text

    with service._engine.begin() as conn:
        conn.execute(text("PRAGMA ignore_check_constraints = ON"))
        conn.execute(text("UPDATE user_preferences SET default_composer_mode = 'kiosk' WHERE user_id = :uid").bindparams(uid=user))
        conn.execute(text("PRAGMA ignore_check_constraints = OFF"))

    # Valid PATCH supplying the new mode: must succeed and return the
    # written value rather than crashing on a re-read of the corrupt row.
    result = asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="guided")))
    assert result.default_mode == "guided"
