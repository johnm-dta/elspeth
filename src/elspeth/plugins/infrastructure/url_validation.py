"""Shared URL validators for credential-bearing plugin configuration."""

from __future__ import annotations

import re
from urllib.parse import urlparse

_ASCII_IPV4_LABEL_RE = re.compile(r"^[0-9]+$")


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.casefold()
    if normalized == "localhost":
        return True
    if normalized in {"::1", "0:0:0:0:0:0:0:1"}:
        return True
    parts = normalized.split(".")
    if len(parts) != 4 or any(_ASCII_IPV4_LABEL_RE.fullmatch(part) is None for part in parts):
        return False
    octets = tuple(int(part) for part in parts)
    return octets[0] == 127 and all(0 <= octet <= 255 for octet in octets)


def validate_credential_safe_https_url(value: str, *, field_name: str, allow_http_loopback: bool = False) -> str:
    """Return a stripped credential-safe URL.

    HTTPS is required for credential-bearing URLs, except when a caller
    explicitly permits HTTP loopback endpoints for local compatible test
    servers or local sidecar gateways.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")

    parsed = urlparse(stripped)
    if not parsed.hostname:
        raise ValueError(f"{field_name} must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain embedded credentials")
    if parsed.scheme == "https":
        return stripped
    if allow_http_loopback and parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return stripped
    if allow_http_loopback:
        raise ValueError(f"{field_name} must use HTTPS unless targeting an HTTP loopback host")
    raise ValueError(f"{field_name} must use HTTPS scheme, got {parsed.scheme!r}")
