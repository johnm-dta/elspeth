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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opentelemetry import metrics
from sqlalchemy import String, select
from sqlalchemy import cast as sql_cast
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.composer.tutorial_telemetry import record_tutorial_completed_path
from elspeth.web.preferences.models import (
    ComposerMode,
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)
from elspeth.web.sessions.models import user_preferences_table


@dataclass(frozen=True, slots=True)
class ComposerPreferencesTransition:
    """Result of an account-level composer-preferences PATCH.

    Carries the prior state (or ``None`` if no row existed before the
    PATCH) and the current state. The prior load happens **inside the
    same transaction** as the write, so there is no TOCTOU window — the
    Phase 8 plan §"Service signature precondition (B2 — load-bearing)"
    explicitly rejects the route-handler read-before-write alternative.

    ``prior`` is ``None`` when no row existed before this PATCH.
    Synthesising a ``ComposerPreferences(default_mode="guided", ...)``
    sentinel here would fabricate a state the system never wrote, which
    contradicts CLAUDE.md §"Three-Tier Trust Model" fabrication test
    ("if the external system's behaviour changes and the field starts
    appearing with a different value than what we inferred, will the
    audit trail silently contain two contradictory sources of truth?").
    Callers are expected to handle ``prior is None`` explicitly.

    Per B2.b (load-bearing): the account-level Phase 8 telemetry
    consumer reads only ``current.default_mode``. The symmetric
    ``(prior, current)`` shape is preserved for code-shape consistency
    with the per-session function and to leave the seam open if a
    future phase wires account-level preferences into an execution
    boundary (the future-promotion criterion documented in the
    module-level "Operational signal only" comment).

    Both fields hold immutable Pydantic models or ``None``; no
    container fields, so no ``__post_init__`` deep-freeze guard is
    required (CLAUDE.md §"Frozen Dataclass Immutability"; scalar /
    model wrappers do not need guards).
    """

    prior: ComposerPreferences | None
    current: ComposerPreferences


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
      field_name: the closed/typed field that failed its Tier-1 guard.
      bad_value: the offending value, named exactly as stored, so the
        operator can confirm the corruption rather than re-derive it.
    """

    def __init__(self, user_id: str, bad_value: object, *, field_name: str = "default_composer_mode") -> None:
        super().__init__(f"user_preferences row for {user_id!r} has invalid {field_name}={bad_value!r}")
        self.user_id = user_id
        self.field_name = field_name
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
    description=(
        "Composer-preferences PATCH operations. Attributes: mode_changed (bool), "
        "banner_dismissed (bool), tutorial_changed (bool), wrote_row (bool)."
    ),
)

_DEFAULT_MODE: ComposerMode = "guided"
_VALID_MODES: frozenset[ComposerMode] = frozenset({"guided", "freeform"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _select_preferences_for_user(user_id: str) -> Any:
    """Select preferences with tutorial timestamp kept as raw text.

    SQLAlchemy's SQLite DateTime result processor raises before the
    service can name the corrupt field if a direct SQL write stores a
    non-datetime string. The tutorial column is new and has an explicit
    Tier-1 guard, so select it as text and parse it in `_row_to_prefs`.
    """
    return select(
        user_preferences_table.c.default_composer_mode,
        user_preferences_table.c.banner_dismissed_at,
        sql_cast(user_preferences_table.c.tutorial_completed_at, String).label("tutorial_completed_at"),
        user_preferences_table.c.updated_at,
    ).where(user_preferences_table.c.user_id == user_id)


def _decode_tutorial_completed_at(user_id: str, raw_value: object) -> datetime | None:
    if raw_value is None:
        return None
    if type(raw_value) is datetime:
        return raw_value
    if type(raw_value) is str:
        try:
            return datetime.fromisoformat(raw_value.removesuffix("Z") + "+00:00" if raw_value.endswith("Z") else raw_value)
        except ValueError as exc:
            raise CorruptPreferencesError(
                user_id,
                {"tutorial_completed_at": raw_value},
                field_name="tutorial_completed_at",
            ) from exc
    raise CorruptPreferencesError(
        user_id,
        {"tutorial_completed_at": raw_value},
        field_name="tutorial_completed_at",
    )


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
                row = conn.execute(_select_preferences_for_user(user_id)).first()
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
                tutorial_completed_at=None,
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
        tutorial_completed_at = _decode_tutorial_completed_at(user_id, row.tutorial_completed_at)
        return ComposerPreferences(
            default_mode=mode,
            banner_dismissed_at=row.banner_dismissed_at,
            tutorial_completed_at=tutorial_completed_at,
            updated_at=row.updated_at,
        )

    async def update_composer_preferences(self, user_id: str, payload: UpdateComposerPreferencesRequest) -> ComposerPreferencesTransition:
        """Upsert the preferences row, touching only fields in ``payload``.

        Empty payloads are accepted as no-ops (the request succeeds; if a
        row already exists, only ``updated_at`` advances).

        Returns a ``ComposerPreferencesTransition`` carrying both the
        prior row state (or ``None`` when no row existed before the
        PATCH) and the post-write state built directly from the written
        values — no round-trip read for ``current``. This prevents the
        corrupt-mode PATCH lockout: a pre-existing corrupt
        ``default_mode`` row would crash a re-read even when the PATCH
        write succeeded and supplied a valid new mode. Since the values
        just written are validated (Tier-3 boundary already ran on
        ``payload``) and trusted, the Tier-1 guard is not re-run on
        ``current``.

        ``transition.prior`` is loaded inside the same transaction as
        the write (B2 — Phase 8 plan §"Service signature precondition")
        — no TOCTOU window between read and write. ``prior is None``
        when no row existed before this PATCH; the no-row-but-non-empty
        PATCH path produces ``prior=None`` AND a new ``current`` row.

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
        tutorial_in_payload = "tutorial_completed_at" in payload.model_fields_set
        banner_in_payload = "banner_dismissed_at" in payload.model_fields_set
        payload_is_empty = payload.default_mode is None and not banner_in_payload and not tutorial_in_payload

        def _sync() -> tuple[ComposerPreferences, bool, ComposerPreferences | None]:
            """Returns (current_prefs, wrote, prior_prefs)."""
            with self._engine.begin() as conn:
                # B2 (load-bearing): load the prior row inside the same
                # transaction as the upsert. The result is `None` if no
                # row exists for this user — synthesising a default
                # sentinel here would fabricate state the system never
                # wrote (see ComposerPreferencesTransition docstring and
                # CLAUDE.md §"Three-Tier Trust Model" fabrication test).
                prior_row = conn.execute(_select_preferences_for_user(user_id)).first()
                prior_prefs: ComposerPreferences | None
                if prior_row is None:
                    prior_prefs = None
                else:
                    prior_prefs = self._row_to_prefs(prior_row, user_id)

                # Panel C2 guard: empty PATCH against a no-row user is a
                # no-write no-op. Check existence BEFORE the upsert so we
                # can skip the INSERT entirely. The B2 prior-load above
                # already SELECTed the row under the sessions engine's
                # BEGIN IMMEDIATE write transaction; no concurrent writer can
                # interpose between the existence check and the downstream
                # upsert.
                if payload_is_empty and prior_row is None:
                    return (
                        ComposerPreferences(
                            default_mode=_DEFAULT_MODE,
                            banner_dismissed_at=None,
                            tutorial_completed_at=None,
                            updated_at=None,
                        ),
                        False,
                        prior_prefs,
                    )

                # Determine the mode to insert (NOT NULL column). On
                # conflict, only fields the caller set are updated.
                insert_mode: ComposerMode
                if payload.default_mode is not None:
                    insert_mode = payload.default_mode
                elif prior_prefs is not None:
                    insert_mode = prior_prefs.default_mode
                else:
                    insert_mode = _DEFAULT_MODE

                # banner_dismissed_at uses `model_fields_set` to distinguish
                # "absent from JSON" (preserve existing) from "explicit null"
                # (clear the dismissal — re-show the banner on next session).
                # Symmetric with tutorial_completed_at; see models.py docstring.
                if banner_in_payload:
                    resolved_banner: datetime | None = payload.banner_dismissed_at
                elif prior_prefs is not None:
                    resolved_banner = prior_prefs.banner_dismissed_at
                else:
                    resolved_banner = None

                if tutorial_in_payload:
                    resolved_tutorial: datetime | None = payload.tutorial_completed_at
                elif prior_prefs is not None:
                    resolved_tutorial = prior_prefs.tutorial_completed_at
                else:
                    resolved_tutorial = None

                values: dict[str, object] = {
                    "user_id": user_id,
                    "default_composer_mode": insert_mode,
                    "banner_dismissed_at": resolved_banner,
                    "tutorial_completed_at": resolved_tutorial,
                    "updated_at": now,
                }
                stmt = sqlite_insert(user_preferences_table).values(**values)
                update_clause: dict[str, object] = {"updated_at": now}
                if payload.default_mode is not None:
                    update_clause["default_composer_mode"] = payload.default_mode
                if banner_in_payload:
                    update_clause["banner_dismissed_at"] = payload.banner_dismissed_at
                if tutorial_in_payload:
                    update_clause["tutorial_completed_at"] = payload.tutorial_completed_at
                stmt = stmt.on_conflict_do_update(index_elements=["user_id"], set_=update_clause)
                row = conn.execute(
                    stmt.returning(
                        user_preferences_table.c.default_composer_mode,
                        user_preferences_table.c.banner_dismissed_at,
                        sql_cast(user_preferences_table.c.tutorial_completed_at, String).label("tutorial_completed_at"),
                        user_preferences_table.c.updated_at,
                    )
                ).one()

            returned = self._row_to_prefs(row, user_id)
            current = ComposerPreferences(
                default_mode=payload.default_mode if payload.default_mode is not None else returned.default_mode,
                banner_dismissed_at=payload.banner_dismissed_at if banner_in_payload else returned.banner_dismissed_at,
                tutorial_completed_at=payload.tutorial_completed_at if tutorial_in_payload else returned.tutorial_completed_at,
                updated_at=now,
            )
            return current, True, prior_prefs

        current, wrote, prior_prefs = await run_sync_in_worker(_sync)
        # Panel S1: operational telemetry only — no Landscape (user state,
        # not pipeline decision boundary). See module-level comment for
        # the no-Landscape rationale and the future-promote criterion.
        _PREFERENCES_PATCH_COUNTER.add(
            1,
            attributes={
                "mode_changed": payload.default_mode is not None,
                "banner_dismissed": payload.banner_dismissed_at is not None,
                "tutorial_changed": tutorial_in_payload,
                "wrote_row": wrote,
            },
        )
        if tutorial_in_payload:
            prior_tutorial = prior_prefs.tutorial_completed_at if prior_prefs is not None else None
            addressed_mode = "default_mode" in payload.model_fields_set
            if prior_tutorial is None and payload.tutorial_completed_at is not None and addressed_mode:
                record_tutorial_completed_path("first_time")
            elif prior_tutorial is None and payload.tutorial_completed_at is not None and not addressed_mode:
                record_tutorial_completed_path("skip")
            elif prior_tutorial is not None and payload.tutorial_completed_at is None:
                record_tutorial_completed_path("retake")
            elif prior_tutorial is not None and payload.tutorial_completed_at is not None:
                record_tutorial_completed_path("repeat")
        return ComposerPreferencesTransition(prior=prior_prefs, current=current)
