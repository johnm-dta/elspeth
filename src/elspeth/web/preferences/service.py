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
            # touch preferences).
            return ComposerPreferences(
                default_mode=_DEFAULT_MODE,
                banner_dismissed_at=None,
                updated_at=self._now(),
            )

        return await run_sync_in_worker(_sync)

    def _row_to_prefs(self, row: object, user_id: str) -> ComposerPreferences:
        """Convert a DB row to the response model with a Tier-1 read guard.

        A stored mode outside the validated set is a fault we caused
        (bug, tampering, or DB corruption). Crash with the offending
        value named so the operator can diagnose.
        """
        mode = row.default_composer_mode  # type: ignore[attr-defined]
        if mode not in _VALID_MODES:
            raise RuntimeError(f"user_preferences row for {user_id!r} has invalid default_composer_mode={mode!r}")
        return ComposerPreferences(
            default_mode=mode,
            banner_dismissed_at=row.banner_dismissed_at,  # type: ignore[attr-defined]
            updated_at=row.updated_at,  # type: ignore[attr-defined]
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

        telemetry: deferred to Phase 8 polish — preference-change event.
        A PATCH here is a user-preference-change event that belongs in
        operational telemetry. Wiring deferred; no ``telemetry.emit()``
        call is made here. Phase 8 will add the emit once the telemetry
        helper is stable.
        """
        now = self._now()

        def _sync() -> tuple[ComposerMode, datetime | None]:
            """Returns (resolved_mode, resolved_banner_dismissed_at) after write."""
            with self._engine.begin() as conn:
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
                        # Tier-1 narrowing: mypy narrows `existing_raw` to
                        # ComposerMode via the `in _VALID_MODES` guard.
                        insert_mode = existing_raw
                    else:
                        raise RuntimeError(f"user_preferences row for {user_id!r} has invalid default_composer_mode={existing_raw!r}")

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

            return insert_mode, resolved_banner

        written_mode, written_banner = await run_sync_in_worker(_sync)
        return ComposerPreferences(
            default_mode=written_mode,
            banner_dismissed_at=written_banner,
            updated_at=now,
        )
