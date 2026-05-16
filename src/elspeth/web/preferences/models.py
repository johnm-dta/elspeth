"""Pydantic models for the composer-preferences API.

``ComposerPreferences`` is the full response payload for GET; it is also the
response for PATCH (the service returns the post-write state). The PATCH
request body is ``UpdateComposerPreferencesRequest``, which is a partial
form where each field is independently optional.

``ComposerPreferences`` (response, server-built) uses
``ConfigDict(strict=True, extra="forbid")``. ``strict=True`` is safe on
the server side because the datetime fields are always real ``datetime``
instances at construction.

``UpdateComposerPreferencesRequest`` (request, JSON-bound) uses
``ConfigDict(extra="forbid")`` only — same pattern as
``blobs/schemas.py::CreateInlineBlobRequest``. ``strict=True`` on a
request body with a ``datetime`` field would reject the standard JSON
ISO-8601 string representation (Pydantic v2 strict mode rejects
string→datetime coercion entirely), which is too aggressive for a
Tier-3 boundary whose contract is "validate, coerce where the standard
wire format permits, never fabricate".

The Literal ``ComposerMode`` still rejects ``"kiosk"`` and any other
out-of-set value on both models, and ``extra="forbid"`` rejects typos.

The Literal ``ComposerMode`` is the single source of truth for the
permitted-values set. It is paired with:
  - the DB-level CHECK constraint on ``user_preferences_table``
  - the Tier-1 read guard in ``PreferencesService._row_to_prefs``

Extending the set requires updating all three call sites in lockstep —
the Literal here, the CHECK in ``sessions/models.py``, and the service
read guard.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

ComposerMode = Literal["guided", "freeform"]


class ComposerPreferences(BaseModel):
    """The full preferences payload returned by GET and PATCH.

    ``updated_at`` is nullable (Panel U1): when no DB row exists for the
    user, the response payload represents the in-server *default* — there
    has been no write event to associate a timestamp with, and fabricating
    ``self._now()`` here would put a value in the audit-visible field that
    the system never actually wrote (CLAUDE.md fabrication test). The
    no-row GET path and the empty-PATCH-on-no-row path both return
    ``updated_at=None``; every other response returns the real write time.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    default_mode: ComposerMode
    banner_dismissed_at: datetime | None
    updated_at: datetime | None


class UpdateComposerPreferencesRequest(BaseModel):
    """Partial-update payload for PATCH.

    Every field is independently optional; the service writes only the
    fields the caller actually set. An empty PATCH is a no-op (the
    request succeeds; ``updated_at`` is bumped if any row already
    exists; if no row exists, none is created — see PreferencesService
    Panel C2 guard for the no-insert contract).

    ``banner_dismissed_at`` semantics — IMPORTANT for callers:

      - Field absent from JSON OR JSON ``null`` → "not set in this
        PATCH" — the field is unchanged.
      - ISO-8601 datetime string → set to that value (one-way
        dismissal; see spec 12 §"Banner lifecycle").

    Both "absent" and "explicit null" collapse to the same wire shape
    because Pydantic v2 cannot distinguish JSON-missing from JSON-null
    without a sentinel. The user-visible spec is one-way dismissal —
    there is no "un-dismiss" affordance — so the no-clear contract
    matches user-visible behaviour. If a future spec adds an
    un-dismiss surface, the model must grow a sentinel (e.g.
    ``Annotated[datetime | None | UnsetType]``) and the service must
    branch on it; the current code does not branch on null.
    """

    model_config = ConfigDict(extra="forbid")

    default_mode: ComposerMode | None = None
    banner_dismissed_at: datetime | None = None
