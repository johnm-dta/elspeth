"""Guided-mode exception taxonomy.

Two semantically distinct exception categories arise in guided mode:

``ValueError``
    **Client fault** — the external caller (HTTP request, wire-shape) provided
    data that violates the contract.  The route handler should catch these and
    return HTTP 400.  Examples: ``edited_values`` is ``None`` when required,
    ``chosen`` carries an unexpected value for the current turn type.

``InvariantError``
    **Server bug** — a code-internal invariant has been violated.  This always
    indicates a defect in the server code, not a bad client request.  The route
    handler returns HTTP 500 with a **static** "Server invariant violated"
    message — ``str(exc)`` is NOT interpolated into the response body because
    ``from_dict`` raise sites embed ``{d!r}`` of the corrupted Tier-1 record,
    which can carry Tier-3 ``sample_rows`` content (PR #37 review finding B1).
    Diagnostic detail is preserved via ``slog`` with ``exc_class`` + bounded
    frames only, under the CLAUDE.md audit-system-failure logging exemption.
    Examples: a ``from_dict`` method read a malformed Tier-1 record, a staging
    field that should have been set before the current code path was reached
    is ``None``, or a current guided-state discriminator names an unsupported
    state.

Both exception types extend ``Exception`` directly (not ``AssertionError``).
``AssertionError`` is elided by ``python -O`` and therefore cannot be relied
upon as a crash gate for server-bug detection.
"""

from __future__ import annotations


class InvariantError(Exception):
    """Raised when a server-side invariant is violated.

    This is always a bug in the server code, never a client error.
    The HTTP dispatcher catches this and returns HTTP 500 with a **static**
    "Server invariant violated" detail message — ``str(exc)`` is not echoed
    to clients because some raise sites (``GuidedSession.from_dict`` et al.)
    embed ``{d!r}`` of the corrupted record, which can carry Tier-3 source
    data (see module docstring for the B1 review-finding context).

    Tier classification of ``from_dict`` failures — why a malformed *persisted*
    record is a 500 (Tier-1 crash class), not a 4xx / quarantine (Tier-3 class):

    The persisted ``guided_session`` blob is a **Tier-1 checkpoint**. It is
    written *only* by ``GuidedSession.to_dict()`` and read back by our own
    ``from_dict()`` — an unbroken chain of custody (our statement → our DB →
    our read). It is **not** an operator-authored config surface like
    ``source.options`` or composer-authored pipeline YAML: no client- or
    operator-facing path can author the ``composer_meta["guided_session"]``
    key. ``set_metadata`` is ``name``/``description``-only (its patch model is
    ``extra="forbid"`` and ``with_metadata`` writes ``state.metadata``, never
    ``composer_meta``); ``composer_meta`` appears on the *response* schema, not
    on any request body; and session fork copies our own prior ``to_dict()``
    output under Tier-1 ``AuditIntegrityError`` guards. A failure in strict
    checkpoint decoding therefore means our serialization contract was violated, the DB is corrupt,
    or the record was tampered with — all Tier-1 anomalies. The web-appropriate
    expression of "crash on a Tier-1 anomaly" (``raise`` ≠ process-kill) is
    exactly this sanitized 500: the server stays up, there is no silent recovery
    that would mask the corruption, and no Tier-3 string leaks. A 4xx would
    falsely blame the caller's *current* request (the fault is in previously
    persisted state); "quarantine" has no meaning for a single-record session
    resume (there is no batch to continue past). The ``schema_version`` mismatch
    branch also raises here, so an old-but-uncorrupted session 500s too — that
    is consistent with this project's "delete the DB on migration" policy for
    the composer state blob (no Alembic path for ``guided_session``).
    """


class WireConfirmRejectedError(Exception):
    """A STEP_4_WIRE confirm was attempted against a pipeline that cannot
    complete (``state.validate().is_valid`` is False).

    Raised by the wire-stage dispatch branch so the route handler can return
    a **structured HTTP 409 rejection** naming what is invalid — instead of
    the pre-fix behaviour, where the failed confirm returned HTTP 200,
    re-emitted the wire turn, and *persisted a new composition-state version
    per click* (15 clicks = 15 minted versions with zero user feedback;
    ux-review elspeth-3b35abf148 variant 3).

    Not a ``ValueError`` (the request body is well-formed — the *state* is
    what blocks completion) and not an :class:`InvariantError` (nothing is
    corrupted server-side). The route handler must NOT run the generic
    dispatcher-HTTPException persistence path for this class: a rejected
    confirm mints no new composition version. The ``guided_turn_answered``
    audit event still drains via the route's finally block, so the rejected
    attempt remains on the audit record.

    ``issues`` carries ``ValidationEntry.to_dict()`` payloads
    (``component`` / ``message`` / ``severity``) — the same strings already
    egressed to the browser via the wire-turn payload and the persisted
    ``validation_errors`` column, so no new egress surface is opened.
    """

    def __init__(self, *, step: str, issues: tuple[dict[str, str], ...]) -> None:
        self.step = step
        self.issues = issues
        super().__init__(f"wire confirm rejected at {step}: {len(issues)} validation issue(s)")


class GuidedSolverResponseShapeError(Exception):
    """The LLM produced a guided tool call with an invalid response shape.

    Distinct from :class:`InvariantError` (server-side bug) and ``ValueError``
    (client-payload bug): this exception means an *external system* (the LLM)
    produced malformed discovery-tool arguments or another unexpected guided
    response. Current step-chat callers convert it to their closed synthetic
    failure response.

    NOT a subclass of ``ValueError`` because the auto-drop wrapper docstring
    explicitly excludes ``ValueError`` to preserve client-payload-bug routing.
    NOT a subclass of ``InvariantError`` because that class is documented as
    "server invariant violated" -- the wrong category for "external LLM
    misbehaved."

    It is not a ``ValueError`` because client payloads are a separate trust
    boundary, and not an :class:`InvariantError` because malformed model output
    is not server corruption.
    """
