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
    """
