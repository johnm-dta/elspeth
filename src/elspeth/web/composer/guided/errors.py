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
    is ``None``, or the recipe-predicate registry references a recipe that
    isn't registered.

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
    output under Tier-1 ``AuditIntegrityError`` guards. The blob *carries*
    Tier-3 content (the LLM-authored ``ChainProposal``), but the courier is
    ours — per the CLAUDE.md container-vs-values rule, trust attaches to the
    value's *author*, not its carrier, and the envelope that ``from_dict``
    validates is the part we authored. ``ChainProposal.from_dict`` is permissive
    on the LLM content itself (its shape was already checked in-flight at
    ``chain_solver``); it only re-checks the envelope we wrote. A failure there
    therefore means our serialization contract was violated, the DB is corrupt,
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
