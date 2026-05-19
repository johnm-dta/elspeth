"""Shared URL validators for credential-bearing plugin configuration."""

from __future__ import annotations

from urllib.parse import urlparse


def validate_credential_safe_https_url(value: str, *, field_name: str) -> str:
    """Return a stripped HTTPS URL with hostname and no embedded credentials."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")

    parsed = urlparse(stripped)
    if parsed.scheme != "https":
        raise ValueError(f"{field_name} must use HTTPS scheme, got {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError(f"{field_name} must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain embedded credentials")
    return stripped
