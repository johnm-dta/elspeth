"""Shared HMAC header fingerprinting for audit recording.

Extracted from AuditedHTTPClient to be reusable by any client that records
HTTP calls to the audit trail (e.g., DataverseClient callers). Both
AuditedHTTPClient and direct-httpx callers import from this module.

The fingerprinting logic ensures:
1. Raw secrets (bearer tokens, API keys) are NEVER stored in the audit trail
2. Different credentials produce different fingerprints (HMAC-SHA256)
3. Replay/verify can distinguish requests by credential identity
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from elspeth.contracts.errors import FrameworkBugError

# Well-known sensitive headers (exact match, case-insensitive).
# Checked first for O(1) lookup before falling back to word matching.
SENSITIVE_HEADERS_EXACT = frozenset(
    {
        # Request headers
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "x-access-token",
        "x-csrf-token",
        "x-xsrf-token",
        "ocp-apim-subscription-key",
        # Response headers
        "set-cookie",
        "www-authenticate",
        "proxy-authenticate",
    }
)

# Words that indicate sensitive content when they appear as complete
# delimiter-separated segments in header names.
# e.g. "X-Auth-Token" splits to {"x","auth","token"} → matches "auth","token"
# but "X-Author" splits to {"x","author"} → no match (avoids false positives)
SENSITIVE_HEADER_WORDS = frozenset(
    {
        "auth",
        "authkey",
        "authtoken",
        "accesstoken",
        "apikey",
        "authorization",
        "key",
        "secret",
        "token",
        "password",
        "credential",
    }
)

SENSITIVE_QUERY_PARAM_WORDS = SENSITIVE_HEADER_WORDS | frozenset({"sig", "signature", "pwd", "pass", "sid"})

# Query redaction is an audit-safety boundary, not an unbounded parser. These
# budgets prevent attacker-controlled URLs and redirect Location values from
# forcing one expensive fingerprint per query field before the HTTP call is
# recorded.
MAX_AUDIT_QUERY_FIELDS = 256
MAX_AUDIT_QUERY_CHARS = 4096
MAX_AUDIT_QUERY_FINGERPRINTS = 64
_QUERY_REDACTION_SENTINEL_KEY = "__elspeth_query_redacted"
_QUERY_REDACTION_SENTINEL_VALUE = "too_many_fields"


def _query_redaction_sentinel() -> dict[str, str | int | float]:
    return {_QUERY_REDACTION_SENTINEL_KEY: _QUERY_REDACTION_SENTINEL_VALUE}


def _query_redaction_sentinel_string() -> str:
    return urlencode(_query_redaction_sentinel())


def _too_many_sensitive_query_params(names: list[str]) -> bool:
    sensitive_count = 0
    for name in names:
        if is_sensitive_query_param(name):
            sensitive_count += 1
            if sensitive_count > MAX_AUDIT_QUERY_FINGERPRINTS:
                return True
    return False


def _param_payload_too_large(params: Mapping[str, str | int | float]) -> bool:
    return any(len(key) + len(str(value)) > MAX_AUDIT_QUERY_CHARS for key, value in params.items())


def is_sensitive_header(header_name: str) -> bool:
    """Check if a header name indicates sensitive content.

    Uses delimiter-separated word matching to avoid false positives from
    broad substring matching (e.g. "key" in "monkey", "auth" in "author"),
    while still catching common compact forms like ``apikey`` and
    ``authkey``.

    Args:
        header_name: Header name to check

    Returns:
        True if header likely contains secrets
    """
    lower_name = header_name.lower()
    if lower_name in SENSITIVE_HEADERS_EXACT:
        return True
    segments = [seg for seg in re.split(r"[^a-z0-9]+", lower_name) if seg]
    if any(seg in SENSITIVE_HEADER_WORDS for seg in segments):
        return True
    return lower_name.startswith("x") and lower_name[1:] in SENSITIVE_HEADER_WORDS


def fingerprint_headers(headers: dict[str, str]) -> dict[str, str]:
    """Fingerprint sensitive request headers for audit recording.

    Sensitive headers (auth, api keys, tokens) are replaced with HMAC
    fingerprints so that:
    1. Raw secrets are NEVER stored in the audit trail
    2. Different credentials produce different fingerprints
    3. Replay/verify can distinguish requests by credential identity

    In dev mode (ELSPETH_ALLOW_RAW_SECRETS=true), sensitive headers are
    removed entirely (no fingerprint key required).

    Args:
        headers: Full headers dict

    Returns:
        Headers dict with sensitive values fingerprinted (or removed in dev mode)

    Raises:
        FrameworkBugError: If sensitive header exists but no fingerprint key
            is configured and dev mode is not enabled.
    """
    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    # Check if fingerprint key is available
    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    result: dict[str, str] = {}

    for k, v in headers.items():
        if is_sensitive_header(k):
            if allow_raw:
                # Dev mode: remove header (don't store secrets, don't require key)
                pass
            elif have_key:
                # Fingerprint the sensitive value
                fp = secret_fingerprint(v)
                result[k] = f"<fingerprint:{fp}>"
            else:
                # No key and not dev mode — config error that prevents auditable operation.
                raise FrameworkBugError(
                    f"Sensitive header '{k}' cannot be fingerprinted: "
                    f"ELSPETH_FINGERPRINT_KEY is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'. "
                    f"Authenticated HTTP calls require a fingerprint key for audit integrity. "
                    f"Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true for dev mode."
                )
        else:
            # Non-sensitive header: include as-is
            result[k] = v

    return result


def filter_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter out sensitive response headers from audit recording.

    Response headers that may contain secrets (cookies, auth challenges)
    are not recorded. Uses the same word-boundary matching as request
    header filtering.

    Args:
        headers: Full headers dict

    Returns:
        Headers dict with sensitive headers removed
    """
    return {k: v for k, v in headers.items() if not is_sensitive_header(k)}


def is_sensitive_query_param(param_name: str) -> bool:
    """Check if a query parameter name indicates sensitive content."""
    lower_name = param_name.lower()
    if lower_name in SENSITIVE_HEADERS_EXACT:
        return True
    segments = [seg for seg in re.split(r"[^a-z0-9]+", lower_name) if seg]
    if any(seg in SENSITIVE_QUERY_PARAM_WORDS for seg in segments):
        return True
    return lower_name.startswith("x") and lower_name[1:] in SENSITIVE_QUERY_PARAM_WORDS


def fingerprint_params(
    params: Mapping[str, str | int | float] | None,
) -> dict[str, str | int | float] | None:
    """Fingerprint sensitive query parameters for audit recording.

    Values are scalar query-parameter values (the call sites in ``http.py``
    pass ``dict[str, str | int | float]``). Sensitive values are replaced with
    a ``<fingerprint:...>`` string; non-sensitive values pass through
    unchanged. The value type is therefore statically ``str | int | float`` —
    not ``Any`` — so this builds deterministic, precisely-typed audit hash
    material rather than an untyped Tier-3 boundary.
    """
    if params is None:
        return None
    if len(params) > MAX_AUDIT_QUERY_FIELDS or _param_payload_too_large(params):
        return _query_redaction_sentinel()
    if _too_many_sensitive_query_params(list(params.keys())):
        return _query_redaction_sentinel()

    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    result: dict[str, str | int | float] = {}

    for k, v in params.items():
        if is_sensitive_query_param(k):
            if allow_raw:
                continue
            if have_key:
                fp = secret_fingerprint(str(v))
                result[k] = f"<fingerprint:{fp}>"
                continue
            raise FrameworkBugError(
                f"Sensitive query parameter '{k}' cannot be fingerprinted: "
                f"ELSPETH_FINGERPRINT_KEY is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'. "
                f"Authenticated HTTP calls require a fingerprint key for audit integrity. "
                f"Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true for dev mode."
            )
        result[k] = v

    return result


def fingerprint_url(url: str) -> str:
    """Fingerprint sensitive query parameters inside a URL for audit recording."""
    parsed = urlsplit(url)
    if not parsed.query:
        return url
    if len(parsed.query) > MAX_AUDIT_QUERY_CHARS:
        return urlunsplit(parsed._replace(query=_query_redaction_sentinel_string()))

    try:
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=MAX_AUDIT_QUERY_FIELDS)
    except ValueError:
        return urlunsplit(parsed._replace(query=_query_redaction_sentinel_string()))
    if _too_many_sensitive_query_params([name for name, _value in query_pairs]):
        return urlunsplit(parsed._replace(query=_query_redaction_sentinel_string()))

    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    fingerprinted_query: list[tuple[str, str]] = []
    for k, v in query_pairs:
        if is_sensitive_query_param(k):
            if allow_raw:
                continue
            if have_key:
                fp = secret_fingerprint(v)
                fingerprinted_query.append((k, f"<fingerprint:{fp}>"))
                continue
            raise FrameworkBugError(
                f"Sensitive URL query parameter '{k}' cannot be fingerprinted: "
                f"ELSPETH_FINGERPRINT_KEY is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'. "
                f"Authenticated HTTP calls require a fingerprint key for audit integrity. "
                f"Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true for dev mode."
            )
        fingerprinted_query.append((k, v))

    return urlunsplit(parsed._replace(query=urlencode(fingerprinted_query)))
