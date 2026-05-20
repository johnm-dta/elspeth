"""Shared URL validators for credential-bearing plugin configuration."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


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
