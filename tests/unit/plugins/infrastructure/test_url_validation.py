"""Tests for credential-bearing plugin URL validation."""

from __future__ import annotations

import pytest

from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8199/v1",
        "http://127.0.0.1:8199/v1",
        "http://127.0.0.42:8199/v1",
        "http://[::1]:8199/v1",
        "http://[0:0:0:0:0:0:0:1]:8199/v1",
    ],
)
def test_credential_url_allows_explicit_http_loopback_when_opted_in(url: str) -> None:
    assert validate_credential_safe_https_url(url, field_name="base_url", allow_http_loopback=True) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "http://127.0.0.999:8199/v1",
        "http://127.example.com/v1",
    ],
)
def test_credential_url_rejects_non_loopback_http_when_opted_in(url: str) -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_credential_safe_https_url(url, field_name="base_url", allow_http_loopback=True)
