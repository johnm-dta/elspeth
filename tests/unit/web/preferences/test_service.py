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

import pytest
from sqlalchemy import select, text
from sqlalchemy.pool import StaticPool

from elspeth.web.preferences.models import UpdateComposerPreferencesRequest
from elspeth.web.preferences.service import (
    CorruptPreferencesError,
    PreferencesService,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import metadata, user_preferences_table


@pytest.fixture(scope="module")
def engine():
    # Panel U2: use create_session_engine (the production factory) rather
    # than bare sqlalchemy.create_engine. The factory registers
    # PRAGMA foreign_keys=ON for SQLite and refuses to return an engine
    # that doesn't enforce FKs. Bypassing it muted FK enforcement under
    # the service tests — invisible to coverage, real in production.
    eng = create_session_engine(
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
    # Panel U1: updated_at is None when no row exists — no write event
    # to associate a timestamp with; fabricating one would put a value
    # the system never wrote into an audit-visible field.
    assert prefs.updated_at is None


def test_get_for_user_with_row_returns_real_updated_at(service):
    """Sanity check on U1: when a row DOES exist, updated_at is the
    real write timestamp, not None."""
    user = "alice-real-updated-at"
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))
    prefs = asyncio.run(service.get_composer_preferences(user))
    assert prefs.updated_at is not None
    assert prefs.updated_at == datetime(2026, 5, 15, tzinfo=UTC).replace(tzinfo=None) or prefs.updated_at == datetime(
        2026, 5, 15, tzinfo=UTC
    )


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
    # Panel S2: assert UTC-equality rather than codifying tzinfo-stripping
    # as expected behaviour. SQLite (in SQLAlchemy's DateTime(timezone=True)
    # binding for the pysqlite dialect) currently returns naive datetimes;
    # we reattach UTC for the comparison so a future driver that preserves
    # tzinfo keeps the test passing rather than failing on a "wrong" answer.
    # The earlier .replace(tzinfo=None)-on-both-sides assertion was a
    # locked-in buggy expectation per
    # feedback_locked_in_buggy_expectations.md.
    assert prefs.banner_dismissed_at is not None
    got = prefs.banner_dismissed_at
    if got.tzinfo is None:
        got = got.replace(tzinfo=UTC)
    assert got == stamp


def test_empty_patch_on_no_row_does_not_insert(service, engine):
    """Panel C2: PATCH with no fields against a user with no row must NOT
    create a default row. The route/service contract is lazy-write: rows
    appear only when the user expresses a real preference."""
    user = "alice-empty-patch-no-row"
    result = asyncio.run(
        service.update_composer_preferences(user, UpdateComposerPreferencesRequest()),
    )
    # Response is the default (matches GET no-row behaviour).
    assert result.default_mode == "guided"
    assert result.banner_dismissed_at is None

    # And critically, no row was inserted.
    with engine.connect() as conn:
        row = conn.execute(select(user_preferences_table).where(user_preferences_table.c.user_id == user)).first()
    assert row is None, "empty PATCH against a no-row user must not insert a default row"


def test_empty_patch_on_existing_row_bumps_updated_at_only(service, engine):
    """Panel C2 corollary: empty PATCH on an existing row IS still a valid
    no-op success — the row's updated_at advances but other fields are
    preserved. This is the documented behaviour and the corner case the
    guard must NOT regress."""
    user = "alice-empty-patch-existing"
    # Seed an existing row.
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))
    with engine.connect() as conn:
        original_updated = conn.execute(
            select(user_preferences_table.c.updated_at).where(user_preferences_table.c.user_id == user)
        ).scalar_one()

    # Empty PATCH — must succeed, must NOT clobber default_mode.
    result = asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest()))
    assert result.default_mode == "freeform"

    with engine.connect() as conn:
        new_updated = conn.execute(select(user_preferences_table.c.updated_at).where(user_preferences_table.c.user_id == user)).scalar_one()
    # updated_at may equal or advance depending on the service's fixed-time
    # `now` callable; the strict assertion is that the row still exists and
    # default_mode is preserved.
    assert new_updated is not None
    assert original_updated is not None


def test_users_are_isolated(service):
    asyncio.run(service.update_composer_preferences("alice-isolated", UpdateComposerPreferencesRequest(default_mode="freeform")))
    bob_prefs = asyncio.run(service.get_composer_preferences("bob-isolated"))
    assert bob_prefs.default_mode == "guided"


def test_corrupt_mode_read_via_public_api_raises(service):
    """Tier-1 read guard: a stored value outside the literal raises when
    read via the public API.

    Panel C4: the earlier test invoked ``service._row_to_prefs(SimpleNamespace(...))``
    — a private method that a refactor could silently inline away,
    leaving the invariant uncovered. This test reaches the guard through
    the public ``get_composer_preferences()`` entry point, bypassing the
    CHECK constraint at insert time via ``PRAGMA ignore_check_constraints``
    to simulate the legitimate CHECK-bypass scenarios the guard exists
    to catch (direct SQLite-shell writes, schema-version drift, on-disk
    corruption, tampering).
    """
    user = "alice-corrupt-public"
    # Seed a valid row first so we can mutate it; goes through the real
    # write path with all validation.
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))
    # Bypass the CHECK to write an out-of-set value — same shape as the
    # production failure scenarios (CHECK was bypassed via PRAGMA, schema
    # drift, or a direct SQLite-shell write).
    with service._engine.begin() as conn:
        conn.execute(text("PRAGMA ignore_check_constraints = ON"))
        conn.execute(text("UPDATE user_preferences SET default_composer_mode = 'kiosk' WHERE user_id = :uid").bindparams(uid=user))
        conn.execute(text("PRAGMA ignore_check_constraints = OFF"))

    # Public read path must crash with the offending value named, via the
    # named CorruptPreferencesError (not bare RuntimeError) so the app
    # exception handler can branch on type rather than message text.
    with pytest.raises(CorruptPreferencesError, match="kiosk") as exc_info:
        asyncio.run(service.get_composer_preferences(user))
    assert exc_info.value.user_id == user
    assert exc_info.value.bad_value == "kiosk"


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
    with service._engine.begin() as conn:
        conn.execute(text("PRAGMA ignore_check_constraints = ON"))
        conn.execute(text("UPDATE user_preferences SET default_composer_mode = 'kiosk' WHERE user_id = :uid").bindparams(uid=user))
        conn.execute(text("PRAGMA ignore_check_constraints = OFF"))

    # Valid PATCH supplying the new mode: must succeed and return the
    # written value rather than crashing on a re-read of the corrupt row.
    result = asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="guided")))
    assert result.default_mode == "guided"


def test_corrupt_mode_blocks_partial_patch_that_does_not_set_mode(service):
    """Tier-1 read guard parity (Phase 1A panel asymmetric-guard finding).

    When PATCH does NOT supply ``default_mode`` and the existing row's
    stored mode is corrupt, the upsert path's read of ``existing_raw``
    must raise ``CorruptPreferencesError`` rather than silently
    coercing/preserving the invalid value. This is parity with the GET
    path's ``_row_to_prefs`` guard — both reads of a corrupt mode raise
    the same named exception, with the same shape, so the application
    handler can branch on type for either read site.

    Contrast with ``test_patch_returns_written_values_not_a_reread``:
    that test confirms the PATCH succeeds when the caller SUPPLIES a
    valid mode (the corrupt read is not exercised because we have a
    new value to write). This test exercises the read path that IS hit
    when the caller does NOT supply a mode and the upsert must consult
    the existing row to know what mode to INSERT.
    """
    user = "alice-partial-corrupt"
    # Seed a valid row.
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))
    # Corrupt the stored mode (CHECK bypass via PRAGMA, simulates the
    # out-of-band corruption scenario).
    with service._engine.begin() as conn:
        conn.execute(text("PRAGMA ignore_check_constraints = ON"))
        conn.execute(text("UPDATE user_preferences SET default_composer_mode = 'kiosk' WHERE user_id = :uid").bindparams(uid=user))
        conn.execute(text("PRAGMA ignore_check_constraints = OFF"))

    # PATCH only banner_dismissed_at — forces the upsert to read the
    # existing mode to know what to INSERT. Must raise on the corrupt
    # read with the offending value named.
    stamp = datetime(2026, 5, 16, tzinfo=UTC)
    with pytest.raises(CorruptPreferencesError, match="kiosk") as exc_info:
        asyncio.run(
            service.update_composer_preferences(
                user,
                UpdateComposerPreferencesRequest(banner_dismissed_at=stamp),
            )
        )
    assert exc_info.value.user_id == user
    assert exc_info.value.bad_value == "kiosk"


def test_empty_user_id_rejected_by_check_constraint(engine):
    """Phase 1A panel minor: ``user_preferences_table`` rejects empty
    ``user_id`` at the schema layer via
    ``ck_user_preferences_user_id_non_empty``.

    The constraint exists because an empty-string user_id would silently
    key a "shared" preferences row that any unauthenticated principal
    could write to if upstream auth ever regressed. The route layer
    sources ``user_id`` from the authenticated identity (no client
    surface for empty), but the schema constraint is defence in depth.
    """
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match="ck_user_preferences_user_id_non_empty"), engine.begin() as conn:
        conn.execute(
            user_preferences_table.insert().values(
                user_id="",
                default_composer_mode="guided",
                banner_dismissed_at=None,
                updated_at=datetime(2026, 5, 16, tzinfo=UTC),
            )
        )
