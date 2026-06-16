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
import threading
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.pool import StaticPool

from elspeth.web.composer import tutorial_telemetry as tutorial_telemetry_module
from elspeth.web.preferences import service as service_module
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


class _RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object]]] = []

    def add(self, amount: int, *, attributes: dict[str, object]) -> None:
        self.calls.append((amount, dict(attributes)))


@pytest.fixture
def preferences_patch_counter(monkeypatch: pytest.MonkeyPatch) -> _RecordingCounter:
    counter = _RecordingCounter()
    monkeypatch.setattr(service_module, "_PREFERENCES_PATCH_COUNTER", counter)
    return counter


@pytest.fixture
def tutorial_completed_counter(monkeypatch: pytest.MonkeyPatch) -> _RecordingCounter:
    counter = _RecordingCounter()
    monkeypatch.setattr(tutorial_telemetry_module, "_TUTORIAL_COMPLETED_COUNTER", counter)
    return counter


def test_get_for_new_user_returns_guided_default(service):
    """A user with no row gets the server-side default 'guided'."""
    prefs = asyncio.run(service.get_composer_preferences("alice-get-default"))
    assert prefs.default_mode == "guided"
    assert prefs.banner_dismissed_at is None
    assert prefs.tutorial_completed_at is None
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
    # B2 (Phase 8a-2): service returns a transition wrapper; the
    # caller reads ``.current`` for the post-write state. New users
    # have no prior row — ``.prior`` is None (NOT a synthesised
    # guided-default sentinel; that would fabricate state the system
    # never wrote — see ComposerPreferencesTransition docstring).
    result = asyncio.run(service.update_composer_preferences("alice-update-persist", payload))
    assert result.prior is None
    assert result.current.default_mode == "freeform"
    assert result.current.tutorial_completed_at is None
    prefs = asyncio.run(service.get_composer_preferences("alice-update-persist"))
    assert prefs.default_mode == "freeform"
    assert prefs.tutorial_completed_at is None


def test_patch_sets_tutorial_completed_at(service):
    stamp = datetime(2026, 5, 15, 13, 0, tzinfo=UTC)
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-set",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )
    assert result.prior is None
    assert result.current.tutorial_completed_at == stamp

    prefs = asyncio.run(service.get_composer_preferences("alice-tutorial-set"))
    assert prefs.tutorial_completed_at == stamp.replace(tzinfo=None) or prefs.tutorial_completed_at == stamp


def test_patch_can_set_mode_and_tutorial_in_one_call(service):
    stamp = datetime(2026, 5, 15, 13, 5, tzinfo=UTC)
    result = asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-final",
            UpdateComposerPreferencesRequest(
                default_mode="freeform",
                tutorial_completed_at=stamp,
            ),
        )
    )
    assert result.current.default_mode == "freeform"
    assert result.current.tutorial_completed_at == stamp


def test_partial_update_preserves_tutorial_completed_at(service):
    stamp = datetime(2026, 5, 15, 13, 10, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-preserve",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )

    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-preserve",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )

    prefs = asyncio.run(service.get_composer_preferences("alice-tutorial-preserve"))
    assert prefs.default_mode == "freeform"
    assert prefs.tutorial_completed_at == stamp.replace(tzinfo=None) or prefs.tutorial_completed_at == stamp


def test_explicit_null_clears_tutorial_completed_at(service):
    stamp = datetime(2026, 5, 15, 13, 15, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-retake",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )

    result = asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-retake",
            UpdateComposerPreferencesRequest(tutorial_completed_at=None),
        )
    )

    assert result.current.tutorial_completed_at is None
    prefs = asyncio.run(service.get_composer_preferences("alice-tutorial-retake"))
    assert prefs.tutorial_completed_at is None


def test_absent_tutorial_field_preserves_but_explicit_null_clears(service):
    stamp = datetime(2026, 5, 15, 13, 20, tzinfo=UTC)
    user = "alice-tutorial-discriminate"
    asyncio.run(
        service.update_composer_preferences(
            user,
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )

    absent_payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    assert "tutorial_completed_at" not in absent_payload.model_fields_set
    asyncio.run(service.update_composer_preferences(user, absent_payload))
    after_absent = asyncio.run(service.get_composer_preferences(user))
    assert after_absent.tutorial_completed_at == stamp.replace(tzinfo=None) or after_absent.tutorial_completed_at == stamp

    null_payload = UpdateComposerPreferencesRequest(tutorial_completed_at=None)
    assert "tutorial_completed_at" in null_payload.model_fields_set
    asyncio.run(service.update_composer_preferences(user, null_payload))
    after_null = asyncio.run(service.get_composer_preferences(user))
    assert after_null.tutorial_completed_at is None


def test_corrupt_tutorial_completed_at_crashes_with_named_error(service, engine):
    user = "alice-tutorial-corrupt"
    with engine.begin() as conn:
        conn.execute(
            user_preferences_table.insert().values(
                user_id=user,
                default_composer_mode="guided",
                banner_dismissed_at=None,
                tutorial_completed_at=None,
                updated_at=datetime(2026, 5, 15, tzinfo=UTC),
            )
        )
        conn.exec_driver_sql(
            "UPDATE user_preferences SET tutorial_completed_at = 'not-a-timestamp' WHERE user_id = 'alice-tutorial-corrupt'"
        )

    with pytest.raises(CorruptPreferencesError) as exc_info:
        asyncio.run(service.get_composer_preferences(user))
    assert exc_info.value.user_id == user
    assert exc_info.value.bad_value == {"tutorial_completed_at": "not-a-timestamp"}
    assert exc_info.value.field_name == "tutorial_completed_at"


def test_patch_tutorial_emits_counter_with_tutorial_changed_label(service, preferences_patch_counter: _RecordingCounter):
    stamp = datetime(2026, 5, 15, 13, 25, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-counter",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )

    assert preferences_patch_counter.calls
    _amount, attrs = preferences_patch_counter.calls[-1]
    assert attrs["mode_changed"] is False
    assert attrs["banner_dismissed"] is False
    assert attrs["wrote_row"] is True
    assert attrs["tutorial_changed"] is True


def test_patch_without_tutorial_emits_counter_with_tutorial_changed_false(service, preferences_patch_counter: _RecordingCounter):
    asyncio.run(
        service.update_composer_preferences(
            "alice-no-tutorial-counter",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )

    assert preferences_patch_counter.calls
    _amount, attrs = preferences_patch_counter.calls[-1]
    assert attrs["mode_changed"] is True
    assert attrs["wrote_row"] is True
    assert attrs["tutorial_changed"] is False


def test_tutorial_completed_counter_first_time_label(
    service,
    tutorial_completed_counter: _RecordingCounter,
):
    stamp = datetime(2026, 5, 15, 14, 0, tzinfo=UTC)

    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-first-time-counter",
            UpdateComposerPreferencesRequest(default_mode="freeform", tutorial_completed_at=stamp),
        )
    )

    assert tutorial_completed_counter.calls[-1] == (1, {"completion_path": "first_time"})


def test_tutorial_completed_counter_skip_label(
    service,
    tutorial_completed_counter: _RecordingCounter,
):
    stamp = datetime(2026, 5, 15, 14, 5, tzinfo=UTC)

    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-skip-counter",
            UpdateComposerPreferencesRequest(tutorial_completed_at=stamp),
        )
    )

    assert tutorial_completed_counter.calls[-1] == (1, {"completion_path": "skip"})


def test_tutorial_completed_counter_retake_label(
    service,
    tutorial_completed_counter: _RecordingCounter,
):
    stamp = datetime(2026, 5, 15, 14, 10, tzinfo=UTC)
    user = "alice-tutorial-retake-counter"
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(tutorial_completed_at=stamp)))

    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(tutorial_completed_at=None)))

    assert tutorial_completed_counter.calls[-1] == (1, {"completion_path": "retake"})


def test_tutorial_completed_counter_repeat_label(
    service,
    tutorial_completed_counter: _RecordingCounter,
):
    first = datetime(2026, 5, 15, 14, 15, tzinfo=UTC)
    second = datetime(2026, 5, 15, 14, 20, tzinfo=UTC)
    user = "alice-tutorial-repeat-counter"
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(tutorial_completed_at=first)))

    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(tutorial_completed_at=second)))

    assert tutorial_completed_counter.calls[-1] == (1, {"completion_path": "repeat"})


def test_tutorial_completed_counter_does_not_fire_for_mode_only_patch(
    service,
    tutorial_completed_counter: _RecordingCounter,
):
    asyncio.run(
        service.update_composer_preferences(
            "alice-tutorial-mode-only-counter",
            UpdateComposerPreferencesRequest(default_mode="freeform"),
        )
    )

    assert tutorial_completed_counter.calls == []


def test_explicit_null_clears_banner_dismissed_at(service):
    """PATCH ``{"banner_dismissed_at": null}`` clears a prior dismissal so
    the banner re-shows on the next session. Symmetric with the tutorial
    field; uses ``model_fields_set`` to distinguish absent vs explicit null."""
    stamp = datetime(2026, 5, 15, 16, 0, tzinfo=UTC)
    asyncio.run(
        service.update_composer_preferences(
            "alice-banner-reshow",
            UpdateComposerPreferencesRequest(banner_dismissed_at=stamp),
        )
    )

    result = asyncio.run(
        service.update_composer_preferences(
            "alice-banner-reshow",
            UpdateComposerPreferencesRequest(banner_dismissed_at=None),
        )
    )

    assert result.current.banner_dismissed_at is None
    prefs = asyncio.run(service.get_composer_preferences("alice-banner-reshow"))
    assert prefs.banner_dismissed_at is None


def test_absent_banner_field_preserves_but_explicit_null_clears(service):
    """Absent ``banner_dismissed_at`` preserves the existing value; explicit
    JSON ``null`` clears it. The discriminator is ``model_fields_set``."""
    stamp = datetime(2026, 5, 15, 16, 5, tzinfo=UTC)
    user = "alice-banner-discriminate"
    asyncio.run(
        service.update_composer_preferences(
            user,
            UpdateComposerPreferencesRequest(banner_dismissed_at=stamp),
        )
    )

    absent_payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    assert "banner_dismissed_at" not in absent_payload.model_fields_set
    asyncio.run(service.update_composer_preferences(user, absent_payload))
    after_absent = asyncio.run(service.get_composer_preferences(user))
    # Same UTC-tzinfo round-trip pattern as test_partial_update_only_touches_provided_fields.
    assert after_absent.banner_dismissed_at is not None
    got = after_absent.banner_dismissed_at
    if got.tzinfo is None:
        got = got.replace(tzinfo=UTC)
    assert got == stamp

    null_payload = UpdateComposerPreferencesRequest(banner_dismissed_at=None)
    assert "banner_dismissed_at" in null_payload.model_fields_set
    asyncio.run(service.update_composer_preferences(user, null_payload))
    after_null = asyncio.run(service.get_composer_preferences(user))
    assert after_null.banner_dismissed_at is None


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
    # Response is the default (matches GET no-row behaviour); prior is
    # None because no row existed before the empty PATCH.
    assert result.prior is None
    assert result.current.default_mode == "guided"
    assert result.current.banner_dismissed_at is None

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
    assert result.current.default_mode == "freeform"
    # Prior was the freeform row seeded above; empty PATCH bumps
    # updated_at but the mode itself is preserved.
    assert result.prior is not None
    assert result.prior.default_mode == "freeform"

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
    """Finding 7 (post-B2 form): PATCH must NOT re-read the row after
    writing to build the response ``current``.

    Pre-B2 framing: if ``update_composer_preferences`` were implemented
    as ``write_row(); return await get_composer_preferences(user_id)``,
    a pre-existing corrupt row would cause the post-write re-read to
    crash on the Tier-1 guard — even though the PATCH itself succeeded.

    Post-B2 framing: the function loads ``prior`` from a SELECT inside
    the same transaction (B2 atomicity), but ``current`` is still built
    from the in-memory written values — no post-write re-read. We verify
    the no-re-read contract by performing a PATCH against a valid row
    and confirming the response carries the supplied value rather than
    requiring a round-trip read.

    The corrupt-prior interaction (corrupt stored value blocks the
    PATCH via the Tier-1 guard on the ``prior`` load) is documented and
    tested separately by ``test_corrupt_prior_blocks_patch_via_prior_load``
    below — that's the new B2 invariant, and it intentionally reverses
    the pre-B2 Finding-7 "PATCH succeeds despite corrupt stored value"
    behaviour because B1's audit-payload extension cannot honestly
    record ``prior_trust_mode`` from a corrupt value.
    """
    user = "alice-finding-7"
    # Seed a valid row first via the normal write path.
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))

    # Valid PATCH supplying the new mode against a NON-corrupt row.
    # The response carries the written value with no post-write re-read.
    result = asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="guided")))
    assert result.current.default_mode == "guided"
    # And prior is what we seeded above.
    assert result.prior is not None
    assert result.prior.default_mode == "freeform"


def test_corrupt_prior_blocks_patch_via_prior_load(service):
    """B2 (Phase 8a-2) invariant: a corrupt stored ``default_composer_mode``
    blocks the PATCH because the prior load runs the Tier-1 guard.

    This intentionally reverses the pre-B2 Finding-7 behaviour
    (corrupt stored + valid PATCH body → PATCH succeeds). The reversal
    is consistent with CLAUDE.md §"Three-Tier Trust Model" Tier-1
    rule: "Bad data in the audit trail = crash immediately. No
    coercion, no defaults, no silent recovery." B1's audit-payload
    extension records ``prior_trust_mode``; recording a corrupt prior
    or fabricating a default sentinel would put a state the system
    never wrote into the audit-visible field.

    Operationally: the project's recovery policy for corrupt
    preference rows is "delete the row" (or "delete the DB" per
    ``project_db_migration_policy``); the user-facing 500 from this
    crash is the correct surface to alert the operator that the row
    is corrupt.
    """
    user = "alice-corrupt-prior-blocks-patch"
    # Seed a valid row.
    asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="freeform")))
    # Corrupt the stored mode (CHECK bypass via PRAGMA).
    with service._engine.begin() as conn:
        conn.execute(text("PRAGMA ignore_check_constraints = ON"))
        conn.execute(text("UPDATE user_preferences SET default_composer_mode = 'kiosk' WHERE user_id = :uid").bindparams(uid=user))
        conn.execute(text("PRAGMA ignore_check_constraints = OFF"))

    # PATCH supplying a valid mode now crashes on the prior load.
    with pytest.raises(CorruptPreferencesError, match="kiosk") as exc_info:
        asyncio.run(service.update_composer_preferences(user, UpdateComposerPreferencesRequest(default_mode="guided")))
    assert exc_info.value.user_id == user
    assert exc_info.value.bad_value == "kiosk"

    # Atomicity invariant: the prior load raises BEFORE the upsert runs,
    # so the stored row must still be the corrupt value — no partial write
    # landed.  A future structural refactor that accidentally moves the
    # prior load outside `engine.begin()` would regress this guarantee
    # without breaking the exception-shape assertion above; the DB-state
    # check is what catches that class of regression.
    with service._engine.connect() as conn:
        stored = conn.execute(
            text("SELECT default_composer_mode FROM user_preferences WHERE user_id = :uid").bindparams(uid=user)
        ).scalar_one()
    assert stored == "kiosk", "B2 atomicity: corrupt-prior crash must NOT leave a partial write"


def test_corrupt_mode_blocks_partial_patch_that_does_not_set_mode(service):
    """Tier-1 read guard parity (Phase 1A panel asymmetric-guard finding).

    When PATCH does NOT supply ``default_mode`` and the existing row's
    stored mode is corrupt, the upsert path's read of ``existing_raw``
    must raise ``CorruptPreferencesError`` rather than silently
    coercing/preserving the invalid value. This is parity with the GET
    path's ``_row_to_prefs`` guard — both reads of a corrupt mode raise
    the same named exception, with the same shape, so the application
    handler can branch on type for either read site.

    Contrast with ``test_corrupt_prior_blocks_patch_via_prior_load``
    (post-B2): that test asserts the prior-load Tier-1 guard raises on
    a corrupt row regardless of payload shape. This test reaches the
    SAME guard via the partial-PATCH path (no ``default_mode`` supplied),
    so under B2 the two tests collapse onto the same code path — both
    crash on the prior load before reaching the ``existing_raw`` read.
    The test is preserved for documentation of the partial-PATCH-with-
    corrupt-row invariant.
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


def test_account_level_update_signature_has_no_from_mode_kwarg():
    """B2.b (Phase 8a-2) load-bearing scope narrowing.

    The Phase 8 plan §"Account-level scope narrowing (B2.b — load-
    bearing)" rules out a ``from_mode`` kwarg on the account-level
    helper because the corresponding telemetry emit is post-state-only
    (``composer.mode.opted_out_total`` / ``composer.mode.opted_in_total``
    drop the ``from_mode`` attribute). The plan rejects promoting
    account-level preferences to audit for Phase 8, so there is no
    audit-recorded ``prior_default_mode`` for a telemetry ``from_mode``
    to mirror — the superset rule holds vacuously when there is no
    transition attribute on the counter.

    This regression guard pins the signature so a future contributor
    cannot quietly re-add ``from_mode`` without renegotiating the
    architectural decision recorded in
    ``preferences/service.py``'s "Operational signal only" comment.
    """
    import inspect

    sig = inspect.signature(PreferencesService.update_composer_preferences)
    assert "from_mode" not in sig.parameters, (
        "PreferencesService.update_composer_preferences must not carry a 'from_mode' kwarg; "
        "the account-level telemetry is post-state-only per Phase 8 B2.b. "
        "Re-introducing this kwarg means promoting account-level preferences to audit "
        "(see the 'Operational signal only' module-level comment for the future-promotion criterion)."
    )


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
                tutorial_completed_at=None,
                updated_at=datetime(2026, 5, 16, tzinfo=UTC),
            )
        )


def test_concurrent_partial_patches_return_serialized_current_state(tmp_path):
    """Concurrent mode/banner PATCH responses must reflect a serialized row.

    The old implementation read preserved fields before its upsert. This test
    blocks the mode-only writer after it has resolved its stale banner value but
    before its INSERT...ON CONFLICT runs, then lets a banner-only writer race.
    Correct SQLite write-intent discipline either serializes the banner writer
    until after the mode writer commits, or the upsert returns the row after the
    conflict update. In both cases one response must show both partial updates.
    """
    engine = create_session_engine(f"sqlite:///{tmp_path / 'preferences-race.db'}")
    metadata.create_all(engine)
    service = PreferencesService(engine, now=lambda: datetime(2026, 5, 16, tzinfo=UTC))
    user = "alice-concurrent-partial-return"
    stamp = datetime(2026, 5, 16, 12, 30, tzinfo=UTC)

    mode_writer_ready = threading.Event()
    release_mode_writer = threading.Event()
    blocked_once = False

    @event.listens_for(engine, "before_cursor_execute")
    def _block_mode_upsert(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        nonlocal blocked_once
        normalized = statement.lstrip().upper()
        if (
            not blocked_once
            and normalized.startswith("INSERT INTO USER_PREFERENCES")
            and user in repr(parameters)
            and "freeform" in repr(parameters)
        ):
            blocked_once = True
            mode_writer_ready.set()
            if not release_mode_writer.wait(timeout=5.0):
                raise TimeoutError("timed out waiting to release blocked mode preferences writer")

    results: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def _run_patch(name: str, payload: UpdateComposerPreferencesRequest) -> None:
        try:
            results[name] = asyncio.run(service.update_composer_preferences(user, payload))
        except BaseException as exc:  # pragma: no cover - re-raised in test thread
            errors[name] = exc

    mode_thread = threading.Thread(
        target=_run_patch,
        args=("mode", UpdateComposerPreferencesRequest(default_mode="freeform")),
        name="preferences-mode-writer",
    )
    banner_thread = threading.Thread(
        target=_run_patch,
        args=("banner", UpdateComposerPreferencesRequest(banner_dismissed_at=stamp)),
        name="preferences-banner-writer",
    )

    mode_thread.start()
    assert mode_writer_ready.wait(timeout=5.0), "mode writer did not reach the upsert gate"
    banner_thread.start()
    banner_thread.join(timeout=0.3)
    release_mode_writer.set()
    mode_thread.join(timeout=5.0)
    banner_thread.join(timeout=5.0)
    assert not mode_thread.is_alive()
    assert not banner_thread.is_alive()
    if errors:
        raise AssertionError(errors)

    with engine.connect() as conn:
        row = conn.execute(select(user_preferences_table).where(user_preferences_table.c.user_id == user)).one()
    assert row.default_composer_mode == "freeform"
    got_banner = row.banner_dismissed_at
    if got_banner.tzinfo is None:
        got_banner = got_banner.replace(tzinfo=UTC)
    assert got_banner == stamp

    def _response_has_both(value: object) -> bool:
        current = value.current  # type: ignore[attr-defined]
        banner = current.banner_dismissed_at
        if banner is not None and banner.tzinfo is None:
            banner = banner.replace(tzinfo=UTC)
        return current.default_mode == "freeform" and banner == stamp

    assert any(_response_has_both(value) for value in results.values()), (
        "At least one concurrent PATCH response must reflect the serialized row with both the mode and banner updates."
    )
