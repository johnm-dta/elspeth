"""URL validation helpers for browser-facing OIDC auth flows."""

from __future__ import annotations

from urllib.parse import SplitResult, urlsplit

from elspeth.core.security import SSRFBlockedError, validate_literal_ip_for_ssrf

_HTTPS_DEFAULT_PORT = 443


def _effective_https_port(parsed: SplitResult, *, field_name: str) -> int:
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid URL") from exc
    if port is None:
        return _HTTPS_DEFAULT_PORT
    return port


def _parse_https_url(raw_value: str, *, field_name: str) -> SplitResult:
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")

    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"{field_name} must be an HTTPS URL")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError(f"{field_name} must be an absolute URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not include embedded credentials")
    _effective_https_port(parsed, field_name=field_name)
    return parsed


def _origin(parsed: SplitResult, *, field_name: str) -> tuple[str, str, int]:
    host = parsed.hostname
    if host is None:
        raise ValueError(f"{field_name} must be an absolute URL")
    return ("https", host.lower(), _effective_https_port(parsed, field_name=field_name))


def validate_oidc_authorization_endpoint(endpoint: str, *, issuer: str) -> str:
    """Return a browser-safe OIDC authorization endpoint for the given issuer."""
    endpoint_value = endpoint.strip()
    issuer_value = issuer.strip().rstrip("/")

    endpoint_url = _parse_https_url(endpoint_value, field_name="authorization_endpoint")
    issuer_url = _parse_https_url(issuer_value, field_name="issuer")

    if _origin(endpoint_url, field_name="authorization_endpoint") != _origin(issuer_url, field_name="issuer"):
        raise ValueError("authorization_endpoint must use the same origin as issuer")
    return endpoint_value


def validate_oidc_issuer(issuer: str) -> str:
    """Return an HTTPS OIDC issuer URL after syntax and literal-IP SSRF checks."""
    issuer_value = issuer.strip().rstrip("/")
    issuer_url = _parse_https_url(issuer_value, field_name="issuer")

    if issuer_url.query or issuer_url.fragment:
        raise ValueError("issuer must not include a query string or fragment")
    host = issuer_url.hostname
    if host is None:
        raise ValueError("issuer must be an absolute URL")
    try:
        validate_literal_ip_for_ssrf(host)
    except SSRFBlockedError as exc:
        raise ValueError("issuer host is blocked by SSRF policy") from exc
    return issuer_value
