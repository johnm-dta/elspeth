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
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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
    """Check if a query parameter name indicates sensitive content.

    Checks exact match lists and word boundary patterns similar to headers.
    """
    lower_name = param_name.lower()
    if lower_name in SENSITIVE_HEADERS_EXACT:
        return True
    segments = [seg for seg in re.split(r"[^a-z0-9]+", lower_name) if seg]
    if any(seg in SENSITIVE_HEADER_WORDS for seg in segments):
        return True
    # Support common abbreviations in query parameters
    additional_sensitive_words = {"sig", "signature", "pwd", "pass", "sid"}
    return any(seg in additional_sensitive_words for seg in segments)


def fingerprint_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Fingerprint sensitive query parameters for audit recording.

    Sensitive parameters (auth, api keys, tokens, signatures) are replaced
    with HMAC fingerprints so that:
    1. Raw secrets are NEVER stored in the audit trail
    2. Different credentials produce different fingerprints
    3. Replay/verify can distinguish requests by credential identity

    In dev mode (ELSPETH_ALLOW_RAW_SECRETS=true), sensitive parameters are
    removed entirely (no fingerprint key required).
    """
    if params is None:
        return None

    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    result: dict[str, Any] = {}

    for k, v in params.items():
        if is_sensitive_query_param(k):
            if allow_raw:
                # Dev mode: remove query parameter
                pass
            elif have_key:
                # Fingerprint the sensitive value (convert value to str for hashing)
                fp = secret_fingerprint(str(v))
                result[k] = f"<fingerprint:{fp}>"
            else:
                raise FrameworkBugError(
                    f"Sensitive query parameter '{k}' cannot be fingerprinted: "
                    f"ELSPETH_FINGERPRINT_KEY is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'. "
                    f"Authenticated HTTP calls require a fingerprint key for audit integrity. "
                    f"Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true for dev mode."
                )
        else:
            result[k] = v

    return result


def fingerprint_url(url: str) -> str:
    """Fingerprint sensitive query parameters inside a URL string for audit recording."""
    if not url:
        return url

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    if not parsed.query:
        return url

    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    fingerprinted_qsl = []

    for k, v in query_params:
        if is_sensitive_query_param(k):
            if allow_raw:
                # Dev mode: remove parameter
                continue
            elif have_key:
                fp = secret_fingerprint(v)
                fingerprinted_qsl.append((k, f"<fingerprint:{fp}>"))
            else:
                raise FrameworkBugError(
                    f"Sensitive URL query parameter '{k}' cannot be fingerprinted: "
                    f"ELSPETH_FINGERPRINT_KEY is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'. "
                    f"Authenticated HTTP calls require a fingerprint key for audit integrity. "
                    f"Set ELSPETH_FINGERPRINT_KEY or ELSPETH_ALLOW_RAW_SECRETS=true for dev mode."
                )
        else:
            fingerprinted_qsl.append((k, v))

    new_query = urlencode(fingerprinted_qsl)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)
