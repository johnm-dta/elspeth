"""Request-id middleware — correlation across response, slog, and (future) telemetry.

Every inbound HTTP request receives a correlation id stored on
``request.state.request_id`` and echoed back as the ``X-Request-ID``
response header.  Exception handlers, structured log events, and audit
records reference the same id so operators can pair a user-reported
error to its slog event and the pending (elspeth-149856079f) Landscape
event with a single lookup.

**Security invariants**

The middleware accepts a caller-supplied ``X-Request-ID`` so that a
front-end or reverse proxy can thread an existing trace id through the
stack — useful for cross-service correlation.  A naive echo would let
an attacker poison the log correlation field: a request with
``X-Request-ID: payload-to-inject\\r\\nevent="fake_admin_login"`` could
smuggle structured content through log pipelines that do not escape it,
or blow up slog-event sizes.  This middleware therefore rejects supplied
ids that are:

* longer than :data:`MAX_REQUEST_ID_LENGTH` characters,
* empty,
* or contain anything outside a conservative printable-ASCII allowlist.

Violations are silently replaced with a freshly generated UUID4.  No
error response is produced — a broken correlation id is operational
noise, not a client-visible failure.

Layer: L3 (application middleware).  Uses Starlette ``BaseHTTPMiddleware``
for compatibility with the existing FastAPI wiring pattern in
``web/app.py``.
"""

from __future__ import annotations

import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"
"""Header name used for inbound and outbound correlation ids."""

MAX_REQUEST_ID_LENGTH = 64
"""Upper bound on accepted inbound request-id length.

Sized to comfortably accommodate UUID4 hex (32) and common ``req-<uuid>``
or ``<service>-<uuid>`` prefixes while rejecting attacker payloads
measured in kilobytes.  The exact value is test-pinned in
``test_request_id.TestMaxLengthConstant`` — any change requires
updating that bound check.
"""

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
"""Accepted character set for inbound ids.

Deliberately narrower than ``ASCII printable``: excludes whitespace,
control characters, quotes, brackets, and backslashes.  This blocks
log-injection vectors (CRLF, tabs, pipe-smuggling) without disrupting
the common id conventions (UUIDs, hyphen-separated words,
alphanumerics).
"""


def _is_safe_request_id(value: str) -> bool:
    """Is the supplied id safe to echo into slog events and response headers?"""
    if not value or len(value) > MAX_REQUEST_ID_LENGTH:
        return False
    return bool(_SAFE_REQUEST_ID.fullmatch(value))


def _generate_request_id() -> str:
    """Fresh UUID4 id for requests without a safe supplied id."""
    return str(uuid.uuid4())


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to every request and response.

    The assigned id is stored on ``request.state.request_id`` so that
    downstream dependencies (including the app-level exception handlers
    in ``web/app.py``) can read it without re-parsing the headers, and
    echoed back in the ``X-Request-ID`` response header for client
    correlation.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied if _is_safe_request_id(supplied) else _generate_request_id()
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
