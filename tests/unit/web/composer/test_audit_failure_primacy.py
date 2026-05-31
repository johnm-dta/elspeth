"""Audit-failure primacy disposition (spec §5.2.2 / §5.5 rows 9-10).

Failure injection patches SQLAlchemy's dialect-level ``do_commit``
hook for one COMMIT attempt. This:

1. Complies with spec §8.6 (no mocking of ``persist_compose_turn``'s
   private helpers — ``_acquire_session_advisory_lock``,
   ``_reserve_sequence_range``, ``_insert_chat_message``, and
   ``_insert_composition_state`` exist to be exercised, not mocked).
2. Simulates the **dominant** production trigger named in spec §4.5 —
   COMMIT-time failure (disk full, fsync failure, network partition
   between the last INSERT and COMMIT) — rather than the INSERT-time
   failure the earlier plan draft simulated by mocking
   ``_insert_chat_message``.
3. Exercises the production code's actual ``try: with engine.begin():
   ... except OperationalError: ...`` path end to end. The wrapped
   COMMIT failure surfaces from ``engine.begin().__exit__`` and is
   caught by the outer ``except`` clause in ``persist_compose_turn``.
4. Avoids assigning to ``sqlite3.Connection.commit``. That method is
   read-only on CPython's sqlite3 connection object, so a test that
   patches it fails during setup for the wrong reason.

The earlier draft used ``patch.object(service, "_insert_chat_message",
side_effect=OperationalError(...))``. That violates spec §8.6 (helpers
are mocked) and tests the wrong failure point (INSERT-time).
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

# Shared ``engine`` fixture and ``_make_session`` helper come from
# ``tests/unit/web/conftest.py`` — the parent-package conftest that
# both the sessions suite and this composer suite share. pytest
# auto-loads it for every test under ``tests/unit/web/...``, so the
# ``engine`` fixture is visible here without further wiring; the
# ``_make_session`` helper is imported explicitly via its absolute
# path (a bare ``from .conftest`` would resolve to
# ``tests/unit/web/composer/conftest.py``, which does not exist —
# synthesised review B5).
from tests.unit.web.conftest import _make_session as _make_session_in_conn


@pytest.fixture
def service(engine, tmp_path):
    return SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


def _make_session(service, session_id):
    """Open a transaction on the service's engine and call the conftest
    helper. Wraps the connection-level helper so audit-primacy tests
    can express setup tersely."""
    with service._engine.begin() as conn:
        _make_session_in_conn(conn, session_id=session_id)


@contextlib.contextmanager
def _force_commit_failure(engine: Engine) -> Iterator[None]:
    """Inject an ``OperationalError`` on the next COMMIT.

    Patches ``engine.dialect.do_commit`` for one call. SQLAlchemy calls
    this hook from ``engine.begin().__exit__``; raising a
    ``sqlite3.OperationalError`` here is wrapped by SQLAlchemy as
    ``sqlalchemy.exc.OperationalError`` and reaches
    ``persist_compose_turn``'s outer OperationalError handler. The
    original hook is restored in ``finally`` so cleanup paths (e.g.
    test teardown) can commit normally.

    SQLite-only — the test suite for audit-failure primacy runs
    against the in-memory SQLite engine. The CL-PP-11 testcontainer
    Postgres test exercises a different scenario (advisory-lock
    contention) and does not require commit-failure injection.
    """
    original_do_commit = engine.dialect.do_commit
    fired = False

    def _fail_once(dbapi_conn: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise sqlite3.OperationalError("simulated COMMIT failure (test injection)")
        original_do_commit(dbapi_conn)

    engine.dialect.do_commit = _fail_once
    try:
        yield
    finally:
        engine.dialect.do_commit = original_do_commit


def test_audit_fail_no_plugin_crash_raises_audit_integrity_error(service):
    """Tool succeeded (plugin_crash_pending=False), audit COMMIT failed:
    ``persist_compose_turn`` must increment the Tier-1 counter AND
    raise :class:`AuditIntegrityError` chained from the original
    ``OperationalError``. Returning a flag would be a doctrine
    violation — the caller could ignore the flag and proceed with
    corrupted audit state. Closes synthesised review finding H1."""
    from elspeth.contracts.errors import TIER_1_ERRORS, AuditIntegrityError
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p1")
    starting = observed_value(service._telemetry.tool_row_tier1_violation_total)

    with (
        _force_commit_failure(service._engine),
        pytest.raises(AuditIntegrityError) as exc_info,
    ):
        service.persist_compose_turn(
            session_id="p1",
            assistant_content="hi",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    # The original OperationalError is preserved as the chained cause.
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, OperationalError)

    # Counter increments before the raise — telemetry-after-audit per
    # CLAUDE.md primacy.
    assert observed_value(service._telemetry.tool_row_tier1_violation_total) == starting + 1

    # The exception must be in TIER_1_ERRORS so ``except Exception:``
    # blocks cannot silently swallow it.
    assert isinstance(exc_info.value, TIER_1_ERRORS)


def test_audit_fail_during_plugin_crash_records_unwind_failure(service):
    """Tool failed (plugin_crash_pending=True) AND audit COMMIT failed:
    ``persist_compose_turn`` must increment the unwind-audit-failure
    counter and RETURN an outcome with ``unwind_audit_failed=True``.
    The unwind path returns rather than raises because the caller
    already has a captured plugin-crash exception to raise; surfacing
    a separate audit exception here would mask the original tool
    failure. The audit failure is recorded via counter + slog
    (permitted under CLAUDE.md primacy because the audit system
    itself failed)."""
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p2")
    starting = observed_value(service._telemetry.tool_row_persist_failed_during_unwind_total)

    with _force_commit_failure(service._engine):
        outcome = service.persist_compose_turn(
            session_id="p2",
            assistant_content="hi",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=True,
        )

    assert outcome.assistant_id is None
    assert outcome.unwind_audit_failed is True
    assert observed_value(service._telemetry.tool_row_persist_failed_during_unwind_total) == starting + 1


@contextlib.contextmanager
def _force_non_integrity_non_operational_sqlalchemy_error(engine: Engine) -> Iterator[None]:
    """Inject a non-Integrity, non-Operational ``SQLAlchemyError`` on the
    next COMMIT.

    Mirrors the ``_force_commit_failure`` shape but raises
    :class:`sqlalchemy.exc.DataError` from the dialect-level commit hook.
    SQLAlchemy wraps the underlying DB-API exception class; here we raise
    the SQLAlchemy-side ``DataError`` directly because the dialect hook
    is the rewrap site itself, and we want the
    ``persist_compose_turn`` outer ``except SQLAlchemyError`` branch to
    fire on a class that is neither ``IntegrityError`` nor
    ``OperationalError``.

    ``DataError`` is the spec-§5.5-row-9 representative for "audit
    raised non-Integrity error" — any DBAPIError sibling (DatabaseError,
    InterfaceError, ProgrammingError) would exercise the same code
    path; the test pins one specific class to keep the assertion
    grounded.
    """
    from sqlalchemy.exc import DataError

    original_do_commit = engine.dialect.do_commit
    fired = False

    def _fail_once(dbapi_conn: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise DataError(
                "simulated commit-time DataError (test injection)",
                {},
                Exception("underlying DBAPI cause"),
            )
        original_do_commit(dbapi_conn)

    engine.dialect.do_commit = _fail_once
    try:
        yield
    finally:
        engine.dialect.do_commit = original_do_commit


def test_audit_fail_non_integrity_non_operational_raises_audit_integrity_error(service):
    """Spec §5.2.2 / §5.5 row 9: a non-Integrity, non-Operational
    ``SQLAlchemyError`` on the audit insert path is a Tier-1 audit
    corruption and MUST raise :class:`AuditIntegrityError` chained
    through the original SQLAlchemyError, with the
    ``tool_row_tier1_violation_total`` counter incremented.

    Before the catch was broadened, ``DataError`` /
    ``DatabaseError`` / ``DBAPIError`` siblings propagated uncaught
    past the disposition logic and the Tier-1 counter never fired —
    silently breaking the SLO=0 contract the spec asserts. This
    regression also tests that the disposition is NOT asymmetric on
    ``plugin_crash_pending`` for this branch (unlike OperationalError):
    arbitrary SQLAlchemyError subclasses have no established recovery
    shape and masking the audit failure to "preserve" a primary error
    would silently lose a Tier-1 corruption signal.
    """
    from sqlalchemy.exc import DataError, SQLAlchemyError

    from elspeth.contracts.errors import TIER_1_ERRORS, AuditIntegrityError
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p3")
    starting_tier1 = observed_value(service._telemetry.tool_row_tier1_violation_total)
    starting_unwind = observed_value(service._telemetry.tool_row_persist_failed_during_unwind_total)

    with (
        _force_non_integrity_non_operational_sqlalchemy_error(service._engine),
        pytest.raises(AuditIntegrityError) as exc_info,
    ):
        service.persist_compose_turn(
            session_id="p3",
            assistant_content="hi",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    # Cause is the simulated DataError (a SQLAlchemyError that is
    # neither IntegrityError nor OperationalError).
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, DataError)
    assert isinstance(exc_info.value.__cause__, SQLAlchemyError)

    # Tier-1 counter incremented; unwind counter untouched (no asymmetry
    # for this branch — see test docstring).
    assert observed_value(service._telemetry.tool_row_tier1_violation_total) == starting_tier1 + 1
    assert observed_value(service._telemetry.tool_row_persist_failed_during_unwind_total) == starting_unwind

    # Tier-1 registration prevents accidental swallowing.
    assert isinstance(exc_info.value, TIER_1_ERRORS)


def test_audit_fail_non_integrity_non_operational_raises_even_on_unwind_path(service):
    """Companion to
    ``test_audit_fail_non_integrity_non_operational_raises_audit_integrity_error``:
    on the unwind path (``plugin_crash_pending=True``), arbitrary
    SQLAlchemyError subclasses STILL raise AuditIntegrityError. This
    is the deliberate asymmetry vs. OperationalError documented in
    the broadened catch — there is no established recovery shape for
    DataError / ProgrammingError / DBAPIError siblings, and silently
    swallowing them to preserve a primary plugin error would lose the
    Tier-1 corruption signal the spec § 5.5 row 9 demands.
    """
    from sqlalchemy.exc import DataError

    from elspeth.contracts.errors import AuditIntegrityError

    _make_session(service, "p4")

    with (
        _force_non_integrity_non_operational_sqlalchemy_error(service._engine),
        pytest.raises(AuditIntegrityError) as exc_info,
    ):
        service.persist_compose_turn(
            session_id="p4",
            assistant_content="hi",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=True,
        )

    assert isinstance(exc_info.value.__cause__, DataError)


@pytest.mark.asyncio
async def test_compose_loop_rejects_unwind_audit_failure_without_plugin_crash(
    composer_service_with_real_sessions,
    fake_llm_two_tool_calls,
    result_session_id,
) -> None:
    """AuditOutcome(None, unwind=True) is invalid without an in-flight crash."""

    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.web.sessions._persist_payload import AuditOutcome
    from elspeth.web.sessions.protocol import ComposerSessionPreferencesRecord

    class _ImpossibleOutcomeSessionsService:
        async def get_composer_preferences(self, session_id: Any) -> ComposerSessionPreferencesRecord:
            return ComposerSessionPreferencesRecord(
                session_id=session_id,
                trust_mode="auto_commit",
                density_default="high",
                interpretation_review_disabled=False,
                updated_at=datetime.now(UTC),
            )

        async def persist_compose_turn_async(self, **_kwargs: Any) -> AuditOutcome:
            return AuditOutcome(assistant_id=None, unwind_audit_failed=True)

        async def upsert_skill_markdown_history(self, **_kwargs: Any) -> bool:
            # Phase 5b Task 5 follow-on (F-5c). Mock satisfies the protocol
            # so the compose-loop entry-time upsert is a no-op.
            return False

    composer_service_with_real_sessions._sessions_service = _ImpossibleOutcomeSessionsService()

    with pytest.raises(AuditIntegrityError) as exc_info:
        await composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_two_tool_calls,
            session_id=result_session_id,
        )

    assert exc_info.value.failed_turn is not None
    assert exc_info.value.failed_turn.assistant_message_id is None
    assert exc_info.value.failed_turn.tool_calls_attempted == 2
