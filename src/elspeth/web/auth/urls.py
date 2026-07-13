"""URL validation helpers for browser-facing OIDC auth flows."""

from __future__ import annotations

import ipaddress
import re
from typing import NamedTuple
from urllib.parse import SplitResult, urlsplit

from elspeth.core.security import SSRFBlockedError, validate_literal_ip_for_ssrf

_HTTPS_DEFAULT_PORT = 443
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z", re.ASCII)
_PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")
_ENCODED_URL_SEPARATOR = re.compile(r"%(?:2f|5c|3f|23|40)", re.IGNORECASE)


class _Origin(NamedTuple):
    scheme: str
    host: str
    port: int


def _static_error(field_name: str, check: str) -> ValueError:
    return ValueError(f"{field_name} failed {check} check")


def _effective_https_port(parsed: SplitResult, *, field_name: str) -> int:
    try:
        port = parsed.port
    except ValueError:
        raise _static_error(field_name, "valid-port") from None
    if port is None:
        return _HTTPS_DEFAULT_PORT
    if port == 0:
        raise _static_error(field_name, "valid-port")
    return port


def _validate_percent_encoding(value: str, *, field_name: str) -> None:
    index = 0
    while True:
        index = value.find("%", index)
        if index < 0:
            break
        if _PERCENT_ESCAPE.match(value, index) is None:
            raise _static_error(field_name, "percent-encoding")
        index += 3
    if _ENCODED_URL_SEPARATOR.search(value):
        raise _static_error(field_name, "encoded-separator")


def _canonical_host(parsed: SplitResult, *, field_name: str) -> str:
    host = parsed.hostname
    if host is None:
        raise _static_error(field_name, "absolute-URL")
    if not host.isascii():
        raise _static_error(field_name, "ASCII-host")
    host = host.lower()
    if not host or host.endswith(".") or "*" in host or "%" in host:
        raise _static_error(field_name, "canonical-host")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Browsers accept several legacy numeric IPv4 forms that Python URL
        # parsers treat as DNS names. Reject the whole ambiguity class.
        labels = host.split(".")
        numeric_like = all(label.isdigit() for label in labels) or any(
            label.startswith("0x") or (len(label) > 1 and label.startswith("0") and label.isdigit()) for label in labels
        )
        if numeric_like:
            raise _static_error(field_name, "browser-host-equivalence") from None
        if len(host) > 253 or any(_DNS_LABEL.fullmatch(label) is None for label in labels):
            raise _static_error(field_name, "DNS-host") from None
        return host

    if address.version == 6 and "%" in host:
        raise _static_error(field_name, "IPv6-zone")
    try:
        validate_literal_ip_for_ssrf(str(address))
    except SSRFBlockedError:
        raise _static_error(field_name, "public-literal-IP") from None
    return address.compressed


def _parse_https_url(raw_value: str, *, field_name: str) -> tuple[str, SplitResult, _Origin]:
    if not isinstance(raw_value, str):
        raise _static_error(field_name, "string")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw_value):
        raise _static_error(field_name, "control-character")
    value = raw_value.strip()
    if not value:
        raise _static_error(field_name, "nonblank")
    if "\\" in value:
        raise _static_error(field_name, "browser-parser-equivalence")
    _validate_percent_encoding(value, field_name=field_name)

    try:
        parsed = urlsplit(value)
    except ValueError:
        raise _static_error(field_name, "valid-URL") from None
    if parsed.scheme.lower() != "https":
        raise _static_error(field_name, "HTTPS")
    if not parsed.netloc or parsed.hostname is None:
        raise _static_error(field_name, "absolute-URL")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise _static_error(field_name, "no-credentials")
    host = _canonical_host(parsed, field_name=field_name)
    port = _effective_https_port(parsed, field_name=field_name)
    return value, parsed, _Origin("https", host, port)


def _parse_bare_origin(raw_value: str, *, field_name: str) -> tuple[str, _Origin]:
    _value, parsed, origin = _parse_https_url(raw_value, field_name=field_name)
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise _static_error(field_name, "bare-origin")
    host = f"[{origin.host}]" if ":" in origin.host else origin.host
    port = "" if origin.port == _HTTPS_DEFAULT_PORT else f":{origin.port}"
    return f"https://{host}{port}", origin


def _parse_browser_endpoint(raw_value: str, *, field_name: str) -> tuple[str, _Origin]:
    value, parsed, origin = _parse_https_url(raw_value, field_name=field_name)
    if not parsed.path or parsed.path == "/":
        raise _static_error(field_name, "non-root-path")
    if parsed.query or parsed.fragment:
        raise _static_error(field_name, "no-query-or-fragment")
    return value, origin


def validate_oidc_browser_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and canonicalize a closed set of exact HTTPS origins."""
    normalized: list[str] = []
    seen: set[_Origin] = set()
    for raw_origin in origins:
        value, origin = _parse_bare_origin(raw_origin, field_name="OIDC browser allowed origin")
        if origin in seen:
            raise ValueError("OIDC browser allowed origin failed duplicate check")
        seen.add(origin)
        normalized.append(value)
    return tuple(normalized)


def oidc_browser_endpoint_origin(endpoint: str) -> str:
    """Return the canonical origin of an already validated browser endpoint."""
    _value, origin = _parse_browser_endpoint(endpoint, field_name="browser_endpoint")
    host = f"[{origin.host}]" if ":" in origin.host else origin.host
    port = "" if origin.port == _HTTPS_DEFAULT_PORT else f":{origin.port}"
    return f"https://{host}{port}"


def validate_oidc_browser_endpoints(
    authorization_endpoint: str,
    token_endpoint: str,
    *,
    issuer: str,
    allowed_origins: tuple[str, ...] = (),
) -> tuple[str, str]:
    """Return a validated authorization/token pair bound to one exact origin."""
    authorization_value, authorization_origin = _parse_browser_endpoint(
        authorization_endpoint,
        field_name="authorization_endpoint",
    )
    token_value, token_origin = _parse_browser_endpoint(token_endpoint, field_name="token_endpoint")
    _issuer_value, _issuer_parsed, issuer_origin = _parse_https_url(issuer, field_name="issuer")

    normalized_allowed = validate_oidc_browser_origins(allowed_origins)
    allowed = {_parse_bare_origin(value, field_name="OIDC browser allowed origin")[1] for value in normalized_allowed}

    if authorization_origin != token_origin:
        raise ValueError("authorization_endpoint and token_endpoint must use the same origin")
    if authorization_origin != issuer_origin and authorization_origin not in allowed:
        raise ValueError("browser endpoint origin is not allowed")
    return authorization_value, token_value


def validate_oidc_authorization_endpoint(endpoint: str, *, issuer: str) -> str:
    """Compatibility validator for the historical same-origin endpoint API."""
    endpoint_value, endpoint_origin = _parse_browser_endpoint(endpoint, field_name="authorization_endpoint")
    _issuer_value, _issuer_parsed, issuer_origin = _parse_https_url(issuer, field_name="issuer")
    if endpoint_origin != issuer_origin:
        raise ValueError("authorization_endpoint must use the same origin as issuer")
    return endpoint_value


def validate_oidc_issuer(issuer: str) -> str:
    """Return an HTTPS OIDC issuer URL after syntax and literal-IP SSRF checks."""
    issuer_value, issuer_url, _origin = _parse_https_url(issuer.strip().rstrip("/"), field_name="issuer")
    if issuer_url.query or issuer_url.fragment:
        raise ValueError("issuer failed no-query-or-fragment check")
    return issuer_value
