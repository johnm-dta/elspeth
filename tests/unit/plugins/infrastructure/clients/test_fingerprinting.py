"""Tests for shared HMAC header fingerprinting utilities.

Covers all branches exhaustively per the spec:
- fingerprint_headers: no key + non-dev → FrameworkBugError, no key + dev → removed,
  key present → HMAC fingerprint
- is_sensitive_header: exact match, word match, x-prefix match, non-sensitive
- filter_response_headers: sensitive removed, non-sensitive kept
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from elspeth.contracts.errors import FrameworkBugError
from elspeth.plugins.infrastructure.clients.fingerprinting import (
    SENSITIVE_HEADER_WORDS,
    SENSITIVE_HEADERS_EXACT,
    filter_response_headers,
    fingerprint_headers,
    fingerprint_params,
    fingerprint_url,
    is_sensitive_header,
)


class TestIsSensitiveHeader:
    """Branch coverage for is_sensitive_header()."""

    @pytest.mark.parametrize(
        "header",
        [
            "authorization",
            "Authorization",
            "AUTHORIZATION",
            "proxy-authorization",
            "cookie",
            "x-api-key",
            "api-key",
            "x-auth-token",
            "x-access-token",
            "x-csrf-token",
            "x-xsrf-token",
            "ocp-apim-subscription-key",
            "set-cookie",
            "www-authenticate",
            "proxy-authenticate",
        ],
    )
    def test_exact_match_sensitive(self, header: str) -> None:
        """Headers in SENSITIVE_HEADERS_EXACT are detected (case-insensitive)."""
        assert is_sensitive_header(header) is True

    @pytest.mark.parametrize(
        "header",
        [
            "X-Custom-Auth-Header",  # "auth" as word segment
            "X-Secret-Value",  # "secret" as word segment
            "My-Token-Header",  # "token" as word segment
            "X-Password-Reset",  # "password" as word segment
            "Custom-Credential-Store",  # "credential" as word segment
        ],
    )
    def test_word_match_sensitive(self, header: str) -> None:
        """Headers with sensitive words as delimiter-separated segments are detected."""
        assert is_sensitive_header(header) is True

    @pytest.mark.parametrize(
        "header",
        [
            "xtoken",  # x-prefix + "token" word → sensitive
            "xsecret",  # x-prefix + "secret" word → sensitive
            "xauth",  # x-prefix + "auth" word → sensitive
        ],
    )
    def test_x_prefix_match_sensitive(self, header: str) -> None:
        """Headers starting with 'x' followed by a sensitive word are detected."""
        assert is_sensitive_header(header) is True

    @pytest.mark.parametrize(
        "header",
        [
            "Content-Type",
            "Accept",
            "User-Agent",
            "Cache-Control",
            "X-Request-Id",
            "X-Correlation-Id",
            "X-Author",  # "author" is NOT "auth" — no false positive
            "Monkey-Business",  # "monkey" contains "key" but is not a word match
            "Content-Length",
        ],
    )
    def test_non_sensitive_passthrough(self, header: str) -> None:
        """Non-sensitive headers are correctly identified."""
        assert is_sensitive_header(header) is False

    def test_constants_are_frozensets(self) -> None:
        """Constants are immutable frozensets."""
        assert isinstance(SENSITIVE_HEADERS_EXACT, frozenset)
        assert isinstance(SENSITIVE_HEADER_WORDS, frozenset)
        assert len(SENSITIVE_HEADERS_EXACT) > 0
        assert len(SENSITIVE_HEADER_WORDS) > 0


class TestFingerprintHeaders:
    """Branch coverage for fingerprint_headers()."""

    def test_no_key_non_dev_mode_raises(self) -> None:
        """No fingerprint key + non-dev mode → FrameworkBugError."""
        env = {k: v for k, v in os.environ.items() if k not in ("ELSPETH_FINGERPRINT_KEY", "ELSPETH_ALLOW_RAW_SECRETS")}
        with patch.dict(os.environ, env, clear=True), pytest.raises(FrameworkBugError, match="cannot be fingerprinted"):
            fingerprint_headers({"Authorization": "Bearer token123"})

    def test_no_key_dev_mode_removes_header(self) -> None:
        """No fingerprint key + dev mode → sensitive headers removed entirely."""
        env = {k: v for k, v in os.environ.items() if k != "ELSPETH_FINGERPRINT_KEY"}
        env["ELSPETH_ALLOW_RAW_SECRETS"] = "true"
        with patch.dict(os.environ, env, clear=True):
            result = fingerprint_headers(
                {
                    "Authorization": "Bearer token123",
                    "Content-Type": "application/json",
                }
            )
            assert "Authorization" not in result
            assert result["Content-Type"] == "application/json"

    def test_key_present_fingerprints(self) -> None:
        """Fingerprint key present → HMAC fingerprint applied."""
        env = dict(os.environ)
        env["ELSPETH_FINGERPRINT_KEY"] = "test-key-for-fingerprinting"
        env.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
        with patch.dict(os.environ, env, clear=True):
            result = fingerprint_headers(
                {
                    "Authorization": "Bearer token123",
                    "Content-Type": "application/json",
                }
            )
            assert result["Authorization"].startswith("<fingerprint:")
            assert result["Authorization"].endswith(">")
            assert result["Content-Type"] == "application/json"

    def test_key_present_different_tokens_different_fingerprints(self) -> None:
        """Different credentials produce different fingerprints."""
        env = dict(os.environ)
        env["ELSPETH_FINGERPRINT_KEY"] = "test-key-for-fingerprinting"
        env.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
        with patch.dict(os.environ, env, clear=True):
            result_a = fingerprint_headers({"Authorization": "Bearer token-A"})
            result_b = fingerprint_headers({"Authorization": "Bearer token-B"})
            assert result_a["Authorization"] != result_b["Authorization"]

    def test_no_sensitive_headers_no_key_needed(self) -> None:
        """When no sensitive headers are present, fingerprint key is not required."""
        env = {k: v for k, v in os.environ.items() if k not in ("ELSPETH_FINGERPRINT_KEY", "ELSPETH_ALLOW_RAW_SECRETS")}
        with patch.dict(os.environ, env, clear=True):
            result = fingerprint_headers(
                {
                    "Content-Type": "application/json",
                    "Accept": "text/html",
                }
            )
            assert result == {"Content-Type": "application/json", "Accept": "text/html"}

    def test_empty_headers(self) -> None:
        """Empty headers dict returns empty dict."""
        result = fingerprint_headers({})
        assert result == {}


class TestFilterResponseHeaders:
    """Branch coverage for filter_response_headers()."""

    def test_removes_sensitive_headers(self) -> None:
        """Sensitive response headers (cookies, auth challenges) are removed."""
        headers = {
            "Content-Type": "application/json",
            "Set-Cookie": "session=abc123",
            "WWW-Authenticate": "Bearer",
            "X-Request-Id": "req-456",
        }
        result = filter_response_headers(headers)
        assert "Set-Cookie" not in result
        assert "WWW-Authenticate" not in result
        assert result["Content-Type"] == "application/json"
        assert result["X-Request-Id"] == "req-456"

    def test_preserves_all_non_sensitive(self) -> None:
        """Non-sensitive headers pass through unchanged."""
        headers = {
            "Content-Type": "text/html",
            "Content-Length": "1024",
            "Cache-Control": "no-cache",
        }
        result = filter_response_headers(headers)
        assert result == headers


class TestFingerprintQueryBounds:
    """Bounds for query redaction at external URL/params boundaries."""

    def test_fingerprint_url_redacts_oversized_query_without_per_value_fingerprints(self) -> None:
        """Oversized URL queries are redacted wholesale before expensive fingerprinting."""
        env = dict(os.environ)
        env["ELSPETH_FINGERPRINT_KEY"] = "test-key-for-fingerprinting"
        env.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
        query = "&".join(f"token=SECRET-{idx}" for idx in range(600))

        with (
            patch.dict(os.environ, env, clear=True),
            patch("elspeth.core.security.secret_fingerprint", side_effect=AssertionError("per-value fingerprint should be skipped")) as fp,
        ):
            result = fingerprint_url(f"https://api.example.com/search?{query}")

        assert result == "https://api.example.com/search?__elspeth_query_redacted=too_many_fields"
        fp.assert_not_called()
        assert "SECRET" not in result
        assert "token" not in result

    def test_fingerprint_params_redacts_oversized_mapping_without_per_value_fingerprints(self) -> None:
        """Oversized explicit query params are redacted wholesale before expensive fingerprinting."""
        env = dict(os.environ)
        env["ELSPETH_FINGERPRINT_KEY"] = "test-key-for-fingerprinting"
        env.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
        params = {f"token-{idx}": f"SECRET-{idx}" for idx in range(600)}

        with (
            patch.dict(os.environ, env, clear=True),
            patch("elspeth.core.security.secret_fingerprint", side_effect=AssertionError("per-value fingerprint should be skipped")) as fp,
        ):
            result = fingerprint_params(params)

        assert result == {"__elspeth_query_redacted": "too_many_fields"}
        fp.assert_not_called()

    def test_fingerprint_url_redacts_overlong_query_value_without_fingerprinting(self) -> None:
        """One giant sensitive query value is still over the audit-redaction budget."""
        env = dict(os.environ)
        env["ELSPETH_FINGERPRINT_KEY"] = "test-key-for-fingerprinting"
        env.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
        query = "token=" + ("S" * 10_000)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("elspeth.core.security.secret_fingerprint", side_effect=AssertionError("overlong value should be skipped")) as fp,
        ):
            result = fingerprint_url(f"https://api.example.com/search?{query}")

        assert result == "https://api.example.com/search?__elspeth_query_redacted=too_many_fields"
        fp.assert_not_called()
        assert "S" * 128 not in result
