"""Service layer for the user_preferences table.

Read path: returns the user's row; falls back to 'guided' when no row exists.
Crashes on Tier-1 read of a corrupt mode value (any stored value outside
{"guided", "freeform"} is a code bug, DB corruption, or tampering — never
a recoverable situation; the DB-level CHECK constraint on
``user_preferences_table.default_composer_mode`` is the first line of
defence, but the read guard here catches the case where the CHECK was
bypassed via direct SQL or schema-version drift).

Write path: upserts the row, touching only fields the caller actually set.
Returns the response model built from the values just written rather than
re-reading — this avoids the corrupt-mode PATCH lockout (Finding 7): a
pre-existing corrupt row would crash a re-read even when the PATCH
write succeeded and supplied a valid new mode.

SQLite-dialect-specific (``ON CONFLICT ... DO UPDATE``) — the deployed
dialect. If a future phase migrates to Postgres, swap the import to
``sqlalchemy.dialects.postgresql.insert``; both expose the same
``on_conflict_do_update`` API.

Async/sync bridge: uses ``run_sync_in_worker`` from
``elspeth.web.async_workers`` (single-worker pool with cancellation
drain). See ``sessions/service.py`` for the canonical usage pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from opentelemetry import metrics
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.preferences.models import (
    ComposerMode,
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)
from elspeth.web.sessions.models import user_preferences_table


class CorruptPreferencesError(RuntimeError):
    """Raised when the sessions DB returns a preferences row that violates
    a closed-list invariant (Tier-1 read guard).

    Carrying a named type rather than bare ``RuntimeError`` lets the
    application's exception handlers (``app.py``) match this specific
    failure mode for incident response without string-grepping the
    message. Subclasses ``RuntimeError`` so existing
    ``except RuntimeError`` callers continue to catch it during the
    transition — there are no such callers today; this is forward-fit
    headroom for the explain/diagnose surface in mcp/.

    Attributes:
      user_id: the row's primary key, included so the operator can
        locate it.
      bad_value: the offending value, named exactly as stored, so the
        operator can confirm the corruption rather than re-derive it.
    """

    def __init__(self, user_id: str, bad_value: object) -> None:
        super().__init__(f"user_preferences row for {user_id!r} has invalid default_composer_mode={bad_value!r}")
        self.user_id = user_id
        self.bad_value = bad_value


# ── Telemetry (Panel S1) ───────────────────────────────────────────────────
# Operational signal only — preferences are user state, not a pipeline
# decision boundary, so NO Landscape emit (see CLAUDE.md primacy rule:
# audit/telemetry/log in order; preferences don't gate any audit-visible
# pipeline behaviour today). The counter exists so the no-Landscape
# decision is *acknowledged in code* rather than implicit silence;
# CLAUDE.md "every emission point must send or explicitly acknowledge
# nothing to send" is satisfied by the explicit no-op shape.
#
# If a future phase wires a preference into an execution boundary
# (e.g. trust_mode gating auto-commit), promote this to a Landscape emit
# at that moment — the counter is the seam.
_meter = metrics.get_meter(__name__)
_PREFERENCES_PATCH_COUNTER = _meter.create_counter(
    "composer.preferences.patch_total",
    description=("Composer-preferences PATCH operations. Attributes: mode_changed (bool), banner_dismissed (bool), wrote_row (bool)."),
)

_DEFAULT_MODE: ComposerMode = "guided"
_VALID_MODES: frozenset[ComposerMode] = frozenset({"guided", "freeform"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PreferencesService:
    """Reads and writes per-user composer preferences."""

    def __init__(self, engine: Engine, *, now: Callable[[], datetime] = _utcnow) -> None:
        self._engine = engine
        self._now = now

    async def get_composer_preferences(self, user_id: str) -> ComposerPreferences:
        """Return the user's preferences, falling back to 'guided' if no row exists.

        Default policy:
          - No row => 'guided' (new-user default; the existing-user
            session-count heuristic was retired under
            ``project_db_migration_policy`` — see plan 12 Task 5).
          - Row exists => use stored value; crash if stored value is
            corrupt.
        """

        def _sync() -> ComposerPreferences:
            with self._engine.connect() as conn:
                row = conn.execute(select(user_preferences_table).where(user_preferences_table.c.user_id == user_id)).first()
                if row is not None:
                    return self._row_to_prefs(row, user_id)

            # No row: return the new-user guided default. We do not write
            # a row here (lazy — avoid write traffic for users who never
            # touch preferences). Panel U1: updated_at=None because no
            # write event exists to associate a timestamp with;
            # fabricating self._now() would put a value the system never
            # actually wrote into an audit-visible field.
            return ComposerPreferences(
                default_mode=_DEFAULT_MODE,
                banner_dismissed_at=None,
                updated_at=None,
            )

        return await run_sync_in_worker(_sync)

    def _row_to_prefs(self, row: Any, user_id: str) -> ComposerPreferences:
        """Convert a DB row to the response model with a Tier-1 read guard.

        A stored mode outside the validated set is a fault we caused
        (bug, tampering, or DB corruption). Crash with the offending
        value named so the operator can diagnose.

        ``row: Any`` matches the established sessions/service.py
        convention (see lines 326, 346, 1945, 2836) and avoids
        ``type: ignore[attr-defined]`` noise on every column access.
        SQLAlchemy ``Row`` objects don't have a useful static type for
        the column attributes the engine exposes via dot access.
        """
        mode = row.default_composer_mode
        if mode not in _VALID_MODES:
            raise CorruptPreferencesError(user_id, mode)
        return ComposerPreferences(
            default_mode=mode,
            banner_dismissed_at=row.banner_dismissed_at,
            updated_at=row.updated_at,
        )

    async def update_composer_preferences(self, user_id: str, payload: UpdateComposerPreferencesRequest) -> ComposerPreferences:
        """Upsert the preferences row, touching only fields in ``payload``.

        Empty payloads are accepted as no-ops (the request succeeds; if a
        row already exists, only ``updated_at`` advances).

        Returns the response model built directly from the written values
        — no round-trip read. This prevents the corrupt-mode PATCH
        lockout: a pre-existing corrupt ``default_mode`` row would crash
        a re-read even when the PATCH write succeeded and supplied a
        valid new mode. Since the values just written are validated
        (Tier-3 boundary already ran on ``payload``) and trusted, the
        Tier-1 guard is not re-run.

        telemetry: increments ``composer.preferences.patch_total`` with
        attributes describing whether the mode or banner field was touched
        and whether a row was actually written (the empty-PATCH-no-row
        guard returns ``wrote_row=False``). Operational signal only — no
        Landscape emit; see module-level comment for the no-Landscape
        rationale.

        Empty-payload contract (Panel C2): a PATCH with no fields set
        against a user with no existing row is a no-op success — the
        response is the default-construct payload and NO row is inserted.
        The earlier behaviour (insert a default row on empty PATCH)
        contradicted the documented lazy-write contract on the GET side.
        """
        now = self._now()
        payload_is_empty = payload.default_mode is None and payload.banner_dismissed_at is None

        def _sync() -> tuple[ComposerMode, datetime | None, bool]:
            """Returns (resolved_mode, resolved_banner_dismissed_at, wrote).

            Race window — TRACKED, not fixed in this commit. The
            read-then-decide-to-upsert structure has a TOCTOU window:
            between the empty-PATCH existence check (and the read of
            existing_raw/resolved_banner below) and the INSERT...ON
            CONFLICT, another writer on a concurrent connection could
            interpose. The race outcomes are *benign at the DB layer*
            because the ON CONFLICT update_clause only includes fields
            the caller actually set, so concurrent writes don't clobber
            each other's mode/banner values. The only observable effect
            is that an HTTP response may carry momentarily-stale
            resolved_banner from another writer's interleaved write —
            the next GET returns the correct row.

            The robust fix (single-statement upsert with COALESCE +
            BEGIN IMMEDIATE for the no-row existence check) is
            explicitly coupled with the deferred sessions/engine.py
            PRAGMA work (journal_mode=WAL, busy_timeout,
            synchronous=NORMAL) per the Phase 1A panel's MAJOR 4
            finding — both land together when PRAGMA discipline is
            scheduled. Filing a tracked filigree issue under
            sessions/engine.py rather than landing a half-fix here
            (SQLAlchemy's SQLite dialect rejects isolation_level=
            "IMMEDIATE" via execution_options; the per-engine
            ``do_begin`` event listener is the correct mechanism but
            affects every write transaction on the sessions DB, not
            just preferences — that warrants its own commit and
            review).
            """
            with self._engine.begin() as conn:
                # Panel C2 guard: empty PATCH against a no-row user is a
                # no-write no-op. Check existence BEFORE the upsert so we
                # can skip the INSERT entirely. Benign race documented
                # above: a concurrent writer interposing between this
                # check and a downstream PATCH affects only response
                # staleness, not DB integrity.
                if payload_is_empty:
                    exists = conn.execute(
                        select(user_preferences_table.c.user_id).where(user_preferences_table.c.user_id == user_id)
                    ).first()
                    if exists is None:
                        return _DEFAULT_MODE, None, False

                # Determine the mode to insert (NOT NULL column). On
                # conflict, only fields the caller set are updated.
                insert_mode: ComposerMode
                if payload.default_mode is not None:
                    insert_mode = payload.default_mode
                else:
                    existing_raw = conn.execute(
                        select(user_preferences_table.c.default_composer_mode).where(user_preferences_table.c.user_id == user_id)
                    ).scalar_one_or_none()
                    if existing_raw is None:
                        insert_mode = _DEFAULT_MODE
                    elif existing_raw in _VALID_MODES:
                        # Panel S3: mypy does NOT narrow `Any in frozenset[T]`
                        # (no TypeGuard available for `in`-membership); the
                        # runtime conditional is the actual guard, and the
                        # explicit cast makes the type contract visible
                        # instead of relying on Any → ComposerMode being
                        # silently accepted. The earlier "mypy narrows…"
                        # comment was inaccurate cargo-cult risk.
                        insert_mode = cast(ComposerMode, existing_raw)
                    else:
                        # Tier-1 read guard parity with _row_to_prefs:
                        # both read paths raise the same named exception
                        # on corruption. The earlier bare RuntimeError
                        # was asymmetric (the GET path raised; this PATCH
                        # read path raised a less-informative type).
                        raise CorruptPreferencesError(user_id, existing_raw)

                # If the caller did not set banner_dismissed_at, preserve
                # the existing value (read in the same transaction so the
                # full post-write state is returned without a second
                # round-trip).
                if payload.banner_dismissed_at is not None:
                    resolved_banner: datetime | None = payload.banner_dismissed_at
                else:
                    resolved_banner = conn.execute(
                        select(user_preferences_table.c.banner_dismissed_at).where(user_preferences_table.c.user_id == user_id)
                    ).scalar_one_or_none()

                values: dict[str, object] = {
                    "user_id": user_id,
                    "default_composer_mode": insert_mode,
                    "banner_dismissed_at": payload.banner_dismissed_at,
                    "updated_at": now,
                }
                stmt = sqlite_insert(user_preferences_table).values(**values)
                update_clause: dict[str, object] = {"updated_at": now}
                if payload.default_mode is not None:
                    update_clause["default_composer_mode"] = payload.default_mode
                if payload.banner_dismissed_at is not None:
                    update_clause["banner_dismissed_at"] = payload.banner_dismissed_at
                stmt = stmt.on_conflict_do_update(index_elements=["user_id"], set_=update_clause)
                conn.execute(stmt)

            return insert_mode, resolved_banner, True

        written_mode, written_banner, wrote = await run_sync_in_worker(_sync)
        # Panel S1: operational telemetry only — no Landscape (user state,
        # not pipeline decision boundary). See module-level comment for
        # the no-Landscape rationale and the future-promote criterion.
        _PREFERENCES_PATCH_COUNTER.add(
            1,
            attributes={
                "mode_changed": payload.default_mode is not None,
                "banner_dismissed": payload.banner_dismissed_at is not None,
                "wrote_row": wrote,
            },
        )
        return ComposerPreferences(
            default_mode=written_mode,
            banner_dismissed_at=written_banner,
            # Panel U1 corollary: when the empty-PATCH guard short-circuits
            # (no row exists, no fields supplied) no `updated_at` was
            # written or read; return None to match the no-row GET semantic
            # rather than a fabricated `now`.
            updated_at=now if wrote else None,
        )
