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
    handler (or FastAPI's default unhandled-exception path) should return HTTP
    500 with a "Server invariant violated" prefix.  Examples: a ``from_dict``
    method read a malformed Tier-1 record, a staging field that should have
    been set before the current code path was reached is ``None``, or the
    recipe-predicate registry references a recipe that isn't registered.

Both exception types extend ``Exception`` directly (not ``AssertionError``).
``AssertionError`` is elided by ``python -O`` and therefore cannot be relied
upon as a crash gate for server-bug detection.
"""

from __future__ import annotations


class InvariantError(Exception):
    """Raised when a server-side invariant is violated.

    This is always a bug in the server code, never a client error.
    The HTTP dispatcher catches this and returns HTTP 500 with a clear
    "Server invariant violated" prefix in the detail string.
    """
