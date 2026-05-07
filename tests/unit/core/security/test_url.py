# tests/core/security/test_url.py
"""Tests for URL sanitization types.

Tests for:
- SanitizedDatabaseUrl sanitizes credentials from database URLs
- SanitizedWebhookUrl sanitizes tokens from webhook URLs
- Fingerprints are computed when key is available
- SecretFingerprintError is raised when key is missing in production mode
"""

import pytest

from elspeth.contracts.url import (
    SENSITIVE_PARAMS,
    SanitizedDatabaseUrl,
    SanitizedWebhookUrl,
)
from elspeth.core.config import SecretFingerprintError


class TestSanitizedDatabaseUrl:
    """Tests for SanitizedDatabaseUrl."""

    def test_url_without_password_unchanged(self) -> None:
        """URL without password is returned unchanged."""
        url = "postgresql://user@host/db"
        result = SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=False)

        assert result.sanitized_url == url
        assert result.fingerprint is None

    def test_password_removed_from_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Password is removed from sanitized URL."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:secret@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert "user@host/db" in result.sanitized_url

    def test_fingerprint_computed_when_password_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fingerprint is computed for the password."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:secret@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        assert result.fingerprint is not None
        assert len(result.fingerprint) == 64  # SHA256 hex

    def test_same_password_same_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same password produces same fingerprint."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url1 = "postgresql://user:secret@host1/db"
        url2 = "postgresql://user:secret@host2/db"

        result1 = SanitizedDatabaseUrl.from_raw_url(url1)
        result2 = SanitizedDatabaseUrl.from_raw_url(url2)

        assert result1.fingerprint == result2.fingerprint

    def test_different_passwords_different_fingerprints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Different passwords produce different fingerprints."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url1 = "postgresql://user:secret1@host/db"
        url2 = "postgresql://user:secret2@host/db"

        result1 = SanitizedDatabaseUrl.from_raw_url(url1)
        result2 = SanitizedDatabaseUrl.from_raw_url(url2)

        assert result1.fingerprint != result2.fingerprint

    def test_raises_when_password_present_no_key_production_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises SecretFingerprintError when password present but no key in production mode."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "postgresql://user:secret@host/db"

        with pytest.raises(SecretFingerprintError, match="ELSPETH_FINGERPRINT_KEY"):
            SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=True)

    def test_dev_mode_sanitizes_without_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dev mode (fail_if_no_key=False) sanitizes but returns None fingerprint."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "postgresql://user:secret@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=False)

        assert "secret" not in result.sanitized_url
        assert result.fingerprint is None

    def test_sqlite_url_unchanged(self) -> None:
        """SQLite URLs without credentials are unchanged."""
        url = "sqlite:///./audit.db"
        result = SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=False)

        assert result.sanitized_url == url
        assert result.fingerprint is None

    def test_empty_password_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty password (user:@host) is still sanitized from the URL.

        Regression: urlparse().password returns "" for user:@host, which is falsy.
        A `not parsed.password` check would skip sanitization, leaving the
        credential-bearing colon in the URL.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        # Colon must be stripped — sanitized URL should not contain ":@"
        assert ":@" not in result.sanitized_url
        assert "user@host/db" in result.sanitized_url

    def test_empty_password_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty password produces a fingerprint (of the empty string)."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        # Empty string still gets fingerprinted for audit completeness
        assert result.fingerprint is not None
        assert len(result.fingerprint) == 64

    def test_unix_socket_dsn_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unix-socket DSNs (no hostname) are sanitized without corruption.

        Regression: urlparse("postgresql://user:pass@/dbname").hostname is None.
        Naive f-string interpolation produces "user@None" instead of "user@".
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:pass@/dbname?host=/var/run/postgresql"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        assert "pass" not in result.sanitized_url
        assert "None" not in result.sanitized_url
        # Should preserve path and query
        assert "/dbname" in result.sanitized_url
        assert "host=/var/run/postgresql" in result.sanitized_url
        assert result.sanitized_url.startswith("postgresql://user@/dbname")
        assert result.fingerprint is not None

    def test_no_username_no_hostname_dsn(self) -> None:
        """DSN with only password (no user, no host) sanitizes cleanly."""
        url = "postgresql://:pass@/dbname"

        result = SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=False)

        assert "pass" not in result.sanitized_url
        assert "None" not in result.sanitized_url
        assert "/dbname" in result.sanitized_url

    def test_templated_port_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-numeric templated port (e.g., ${PORT}) is preserved without crash.

        Regression: urlparse().port raises ValueError on non-numeric ports.
        DSNs with template placeholders must sanitize the password while
        preserving the raw port string for downstream resolution.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:secret@host:${PORT}/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert "${PORT}" in result.sanitized_url
        assert "user@host:${PORT}/db" in result.sanitized_url
        assert result.fingerprint is not None

    def test_malformed_port_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed non-numeric port string is preserved without crash."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:secret@host:abc/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert ":abc" in result.sanitized_url

    def test_numeric_port_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Normal numeric port still sanitizes correctly."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:secret@host:5432/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert ":5432" in result.sanitized_url
        assert "user@host:5432/db" in result.sanitized_url

    def test_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SanitizedDatabaseUrl is frozen (immutable)."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        result = SanitizedDatabaseUrl.from_raw_url("postgresql://user:pass@host/db")

        with pytest.raises(AttributeError):
            result.sanitized_url = "hacked"  # type: ignore[misc]

    def test_direct_construction_rejects_empty_password(self) -> None:
        """Direct construction with user:@host (empty password) is rejected.

        Bug: elspeth-ec29cad8c2. urlparse().password returns "" for user:@host,
        which is falsy. The old `if parsed.password:` guard missed it, allowing
        credential-bearing URLs to bypass the sanitization invariant.
        """
        with pytest.raises(ValueError, match="cannot contain a password"):
            SanitizedDatabaseUrl(
                sanitized_url="postgresql://user:@host/db",
                fingerprint=None,
            )

    def test_direct_construction_rejects_full_password(self) -> None:
        """Direct construction with user:pass@host is rejected."""
        with pytest.raises(ValueError, match="cannot contain a password"):
            SanitizedDatabaseUrl(
                sanitized_url="postgresql://user:secret@host/db",
                fingerprint=None,
            )

    def test_direct_construction_accepts_no_password(self) -> None:
        """Direct construction without password is accepted."""
        result = SanitizedDatabaseUrl(
            sanitized_url="postgresql://user@host/db",
            fingerprint=None,
        )
        assert result.sanitized_url == "postgresql://user@host/db"


class TestSanitizedWebhookUrl:
    """Tests for SanitizedWebhookUrl."""

    def test_url_without_tokens_unchanged(self) -> None:
        """URL without sensitive params is returned unchanged."""
        url = "https://api.example.com/webhook?format=json"
        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert result.sanitized_url == url
        assert result.fingerprint is None

    def test_token_param_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token query parameter is removed."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com/webhook?token=sk-abc123"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "sk-abc123" not in result.sanitized_url
        assert "token" not in result.sanitized_url

    def test_api_key_param_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api_key query parameter is removed."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com/hook?api_key=secret123&format=json"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret123" not in result.sanitized_url
        assert "api_key" not in result.sanitized_url
        assert "format=json" in result.sanitized_url

    def test_multiple_sensitive_params_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple sensitive params are all removed."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com?token=abc&secret=xyz&keep=me"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "abc" not in result.sanitized_url
        assert "xyz" not in result.sanitized_url
        assert "keep=me" in result.sanitized_url

    def test_basic_auth_password_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Basic Auth password in URL is removed."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://user:password@api.example.com/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "password" not in result.sanitized_url
        assert "user" not in result.sanitized_url
        assert "@" not in result.sanitized_url.split("//")[1].split("/")[0]  # No @ in netloc

    def test_basic_auth_username_only_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Basic Auth username without password is removed (token in username field)."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://token@api.example.com/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # SECURITY: Username can contain bearer tokens - must be stripped
        assert "token" not in result.sanitized_url
        assert "@" not in result.sanitized_url.split("//")[1].split("/")[0]  # No @ in netloc
        assert result.sanitized_url == "https://api.example.com/webhook"
        assert result.fingerprint is not None

    def test_basic_auth_username_empty_password_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Basic Auth with username and empty password removed (token:@host pattern)."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://token:@api.example.com/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # SECURITY: Username can contain bearer tokens - must be stripped
        assert "token" not in result.sanitized_url
        assert "@" not in result.sanitized_url.split("//")[1].split("/")[0]  # No @ in netloc
        assert result.sanitized_url == "https://api.example.com/webhook"
        assert result.fingerprint is not None

    def test_basic_auth_username_and_password_both_fingerprinted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both username and password are included in fingerprint.

        Verifies that:
        1. Fingerprints are present (not None)
        2. Both credentials contribute to the fingerprint
        3. The fingerprint matches the expected HMAC
        """
        from elspeth.core.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url_userpass = "https://user:pass@api.example.com/webhook"
        url_user_only = "https://user@api.example.com/webhook"

        result_userpass = SanitizedWebhookUrl.from_raw_url(url_userpass)
        result_user_only = SanitizedWebhookUrl.from_raw_url(url_user_only)

        # Both must have fingerprints (not None)
        assert result_userpass.fingerprint is not None
        assert result_user_only.fingerprint is not None

        # Verify exact fingerprints match expected HMAC of sorted credentials
        # For user:pass -> sorted(["user", "pass"]) = ["pass", "user"] -> JSON array
        import json as json_module

        expected_userpass = secret_fingerprint(json_module.dumps(sorted(["pass", "user"]), separators=(",", ":")))
        assert result_userpass.fingerprint == expected_userpass

        # For user only -> JSON array with single element
        expected_user_only = secret_fingerprint(json_module.dumps(["user"], separators=(",", ":")))
        assert result_user_only.fingerprint == expected_user_only

        # Different fingerprints because one has password, one doesn't
        assert result_userpass.fingerprint != result_user_only.fingerprint

    def test_fingerprint_computed_for_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fingerprint is computed from token values only."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com?token=secret123"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert result.fingerprint is not None
        assert len(result.fingerprint) == 64

    def test_fingerprint_of_tokens_only_not_full_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fingerprint is computed from tokens only, not full URL.

        Same token in different URLs should have same fingerprint.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url1 = "https://api1.example.com/path1?token=same-token"
        url2 = "https://api2.example.com/path2?token=same-token"

        result1 = SanitizedWebhookUrl.from_raw_url(url1)
        result2 = SanitizedWebhookUrl.from_raw_url(url2)

        # Same token = same fingerprint (despite different URLs)
        assert result1.fingerprint == result2.fingerprint

    def test_raises_when_token_present_no_key_production_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises SecretFingerprintError when token present but no key in production mode."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://api.example.com?token=secret"

        with pytest.raises(SecretFingerprintError, match="ELSPETH_FINGERPRINT_KEY"):
            SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=True)

    def test_dev_mode_sanitizes_without_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dev mode (fail_if_no_key=False) sanitizes but returns None fingerprint."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://api.example.com?token=secret"

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert "secret" not in result.sanitized_url
        assert result.fingerprint is None

    def test_case_insensitive_param_matching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sensitive param matching is case-insensitive."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com?TOKEN=secret&API_KEY=key123"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert "key123" not in result.sanitized_url

    def test_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SanitizedWebhookUrl is frozen (immutable)."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        result = SanitizedWebhookUrl.from_raw_url("https://api.example.com?token=x")

        with pytest.raises(AttributeError):
            result.sanitized_url = "hacked"  # type: ignore[misc]

    def test_ipv6_url_without_auth_unchanged(self) -> None:
        """IPv6 URL without auth preserves bracket notation."""
        url = "https://[::1]:8443/webhook"
        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert result.sanitized_url == url
        assert "[::1]" in result.sanitized_url
        assert result.fingerprint is None

    def test_ipv6_with_username_only_preserves_brackets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IPv6 URL with username-only auth preserves bracket notation after stripping."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://token@[::1]:8443/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Regression test for bug: must preserve IPv6 brackets when rebuilding netloc
        assert result.sanitized_url == "https://[::1]:8443/webhook"
        assert "token" not in result.sanitized_url
        assert "[::1]" in result.sanitized_url
        assert result.fingerprint is not None

    def test_ipv6_with_username_and_password_preserves_brackets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IPv6 URL with username:password preserves bracket notation after stripping."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://user:pass@[::1]:8443/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert result.sanitized_url == "https://[::1]:8443/webhook"
        assert "user" not in result.sanitized_url
        assert "pass" not in result.sanitized_url
        assert "[::1]" in result.sanitized_url
        assert result.fingerprint is not None

    def test_ipv6_full_address_with_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IPv6 full address with auth preserves bracket notation."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://token@[2001:db8::1]:443/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert result.sanitized_url == "https://[2001:db8::1]:443/webhook"
        assert "token" not in result.sanitized_url
        assert "[2001:db8::1]" in result.sanitized_url
        assert result.fingerprint is not None

    def test_ipv6_no_port_with_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IPv6 without port preserves bracket notation when stripping auth."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://token@[::1]/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert result.sanitized_url == "https://[::1]/webhook"
        assert "token" not in result.sanitized_url
        assert "[::1]" in result.sanitized_url
        assert result.fingerprint is not None

    def test_templated_port_with_auth_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-numeric templated port preserved when stripping auth.

        Regression: urlparse().port raises ValueError on non-numeric ports.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://user:pass@api.example.com:${PORT}/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "user" not in result.sanitized_url
        assert "pass" not in result.sanitized_url
        assert "${PORT}" in result.sanitized_url
        assert result.sanitized_url == "https://api.example.com:${PORT}/webhook"
        assert result.fingerprint is not None


class TestSanitizedWebhookUrlPostInit:
    """Tests for SanitizedWebhookUrl.__post_init__ constructor guards."""

    def test_rejects_username_only_auth(self) -> None:
        """Direct construction with username-only auth is rejected.

        Bug fix: elspeth-80090f1076. A bearer token in the username field
        (e.g., https://token@host) bypassed the __post_init__ guard because
        it only checked `parsed.password`, not `parsed.username`.
        """
        with pytest.raises(ValueError, match=r"cannot contain.*userinfo"):
            SanitizedWebhookUrl(
                sanitized_url="https://bearer-token@api.example.com/webhook",
                fingerprint=None,
            )

    def test_rejects_empty_password_auth(self) -> None:
        """Direct construction with user:@host (empty password) is rejected.

        Bug fix: elspeth-80090f1076. Empty-password Basic Auth like
        https://token:@host has parsed.password="" (falsy), so the old
        `if parsed.password:` guard missed it.
        """
        with pytest.raises(ValueError, match=r"cannot contain.*userinfo"):
            SanitizedWebhookUrl(
                sanitized_url="https://token:@api.example.com/webhook",
                fingerprint=None,
            )

    def test_rejects_full_basic_auth(self) -> None:
        """Direct construction with user:pass@host is still rejected."""
        with pytest.raises(ValueError, match="cannot contain"):
            SanitizedWebhookUrl(
                sanitized_url="https://user:secret@api.example.com/webhook",
                fingerprint=None,
            )

    def test_accepts_clean_url(self) -> None:
        """Direct construction with no userinfo is accepted."""
        result = SanitizedWebhookUrl(
            sanitized_url="https://api.example.com/webhook",
            fingerprint=None,
        )
        assert result.sanitized_url == "https://api.example.com/webhook"


class TestExtractRawPort:
    """Tests for _extract_raw_port helper."""

    def test_simple_port(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("host:5432") == ":5432"

    def test_no_port(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("host") == ""

    def test_templated_port(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("host:${PORT}") == ":${PORT}"

    def test_with_userinfo(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("user:pass@host:5432") == ":5432"

    def test_ipv6_with_port(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("[::1]:8443") == ":8443"

    def test_ipv6_without_port(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("[::1]") == ""

    def test_ipv6_with_userinfo_and_port(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("user:pass@[::1]:8443") == ":8443"

    def test_empty_netloc(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("") == ""

    def test_templated_port_with_userinfo(self) -> None:
        from elspeth.contracts.url import _extract_raw_port

        assert _extract_raw_port("user:pass@host:${PORT}") == ":${PORT}"


class TestSensitiveParams:
    """Tests for SENSITIVE_PARAMS coverage."""

    @pytest.mark.parametrize(
        "param",
        [
            "token",
            "api_key",
            "apikey",
            "secret",
            "key",
            "password",
            "auth",
            "access_token",
            "client_secret",
            "api_secret",
            "bearer",
            "signature",
            "sig",
            "authorization",
            "x-api-key",
            "credential",
            "credentials",
        ],
    )
    def test_sensitive_param_in_set(self, param: str) -> None:
        """All expected sensitive params are in the set."""
        assert param in SENSITIVE_PARAMS

    @pytest.mark.parametrize(
        "param",
        [
            "token",
            "api_key",
            "secret",
            "access_token",
            "signature",
        ],
    )
    def test_sensitive_param_removed_from_url(self, param: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each sensitive param type is removed from URLs."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = f"https://api.example.com?{param}=secret_value&keep=me"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret_value" not in result.sanitized_url
        assert param not in result.sanitized_url.split("?")[-1]
        assert "keep=me" in result.sanitized_url

    def test_empty_token_value_still_strips_param_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty token value still removes param name from URL.

        Regression test for: URLs like ?token= should strip the 'token' key
        even though the value is empty, to prevent leaking parameter names.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://api.example.com/hook?token="

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        # SECURITY: Parameter name 'token' must not appear in sanitized URL
        assert "token" not in result.sanitized_url
        assert result.sanitized_url == "https://api.example.com/hook"
        assert result.fingerprint is None  # No non-empty values to fingerprint

    def test_empty_api_key_value_still_strips_param_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty api_key value still removes param name from URL."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://api.example.com/hook?api_key=&format=json"

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        # SECURITY: Parameter name 'api_key' must not appear
        assert "api_key" not in result.sanitized_url
        assert "format=json" in result.sanitized_url
        assert result.fingerprint is None

    def test_empty_token_does_not_trigger_fingerprint_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty token value should not trigger SecretFingerprintError.

        No non-empty secrets = no fingerprint needed = no error in production mode.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://api.example.com?token="

        # Should NOT raise - no actual secret values to fingerprint
        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=True)

        assert "token" not in result.sanitized_url
        assert result.fingerprint is None

    def test_mixed_empty_and_nonempty_sensitive_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mix of empty and non-empty sensitive params strips all keys, fingerprints non-empty."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com?token=&api_key=real-secret&format=json"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Both keys stripped
        assert "token" not in result.sanitized_url
        assert "api_key" not in result.sanitized_url
        assert "real-secret" not in result.sanitized_url
        # Non-sensitive param preserved
        assert "format=json" in result.sanitized_url
        # Fingerprint computed from non-empty value only
        assert result.fingerprint is not None


class TestBracketDotParamSanitization:
    """Tests for bracket/dot notation param matching.

    SECURITY: parse_qs preserves bracket notation in keys, so api_key[0]=secret
    produces key "api_key[0]" which must still match "api_key" in SENSITIVE_PARAMS.
    """

    def test_bracket_notation_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api_key[0]=secret must be stripped as sensitive."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com?api_key[0]=secret&format=json"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert "api_key" not in result.sanitized_url
        assert "format=json" in result.sanitized_url
        assert result.fingerprint is not None

    def test_dot_notation_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """token.value=secret must be stripped as sensitive."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com?token.value=secret&keep=me"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert "token" not in result.sanitized_url
        assert "keep=me" in result.sanitized_url
        assert result.fingerprint is not None

    def test_bracket_notation_rejected_by_post_init(self) -> None:
        """__post_init__ must reject bracket-notated sensitive params."""
        with pytest.raises(ValueError, match="sensitive query parameters"):
            SanitizedWebhookUrl(
                sanitized_url="https://api.example.com?api_key[0]=secret",
                fingerprint=None,
            )

    def test_dot_notation_rejected_by_post_init(self) -> None:
        """__post_init__ must reject dot-notated sensitive params."""
        with pytest.raises(ValueError, match="sensitive query parameters"):
            SanitizedWebhookUrl(
                sanitized_url="https://api.example.com?token.value=secret",
                fingerprint=None,
            )


class TestFragmentTokenSanitization:
    """Tests for URL fragment token sanitization.

    SECURITY: OAuth implicit flow and some APIs put tokens in URL fragments
    (e.g., #access_token=xxx). These must be sanitized like query params.
    """

    def test_fragment_access_token_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fragment access_token is removed from sanitized URL."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://example.com/callback#access_token=sk-abc123"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "sk-abc123" not in result.sanitized_url
        assert "access_token" not in result.sanitized_url
        assert result.sanitized_url == "https://example.com/callback"
        assert result.fingerprint is not None

    def test_fragment_token_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fragment token parameter is removed."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://api.example.com/auth#token=secret-value"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret-value" not in result.sanitized_url
        assert "token" not in result.sanitized_url
        assert result.fingerprint is not None

    def test_fragment_with_multiple_params_strips_only_sensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fragment with mixed params keeps non-sensitive, strips sensitive."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://example.com/callback#access_token=secret&state=abc123&token_type=bearer"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Sensitive param stripped
        assert "access_token" not in result.sanitized_url
        assert "secret" not in result.sanitized_url
        # Non-sensitive params preserved
        assert "state=abc123" in result.sanitized_url
        assert "token_type=bearer" in result.sanitized_url
        assert result.fingerprint is not None

    def test_fragment_and_query_both_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both query params and fragment params are sanitized."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://example.com/callback?api_key=query-secret#access_token=fragment-secret"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Both secrets removed
        assert "query-secret" not in result.sanitized_url
        assert "fragment-secret" not in result.sanitized_url
        assert "api_key" not in result.sanitized_url
        assert "access_token" not in result.sanitized_url
        # URL structure preserved but empty
        assert result.sanitized_url == "https://example.com/callback"
        assert result.fingerprint is not None

    def test_fragment_fingerprint_matches_expected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fragment token fingerprint matches expected HMAC value."""
        from elspeth.core.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://example.com/callback#access_token=my-token"

        result = SanitizedWebhookUrl.from_raw_url(url)

        import json as json_module

        expected = secret_fingerprint(json_module.dumps(["my-token"], separators=(",", ":")))
        assert result.fingerprint == expected

    def test_fragment_and_query_fingerprint_combined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fingerprint combines secrets from both query and fragment."""
        import json as json_module

        from elspeth.core.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://example.com/callback?api_key=key1#access_token=key2"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Fingerprint is canonical JSON array of sorted values
        expected = secret_fingerprint(json_module.dumps(sorted(["key1", "key2"]), separators=(",", ":")))
        assert result.fingerprint == expected

    def test_fragment_raises_when_no_key_production_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises SecretFingerprintError when fragment token present but no key."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://example.com/callback#access_token=secret"

        with pytest.raises(SecretFingerprintError, match="ELSPETH_FINGERPRINT_KEY"):
            SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=True)

    def test_fragment_dev_mode_sanitizes_without_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dev mode sanitizes fragment without fingerprint."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://example.com/callback#access_token=secret"

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert "secret" not in result.sanitized_url
        assert "access_token" not in result.sanitized_url
        assert result.fingerprint is None

    def test_fragment_empty_value_strips_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty fragment token value still strips the key name."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://example.com/callback#access_token="

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert "access_token" not in result.sanitized_url
        assert result.sanitized_url == "https://example.com/callback"
        assert result.fingerprint is None  # No non-empty value to fingerprint

    def test_fragment_empty_does_not_trigger_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty fragment token value doesn't trigger SecretFingerprintError."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        url = "https://example.com/callback#access_token="

        # Should NOT raise - no actual secret value
        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=True)

        assert "access_token" not in result.sanitized_url
        assert result.fingerprint is None

    def test_fragment_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fragment param matching is case-insensitive."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "https://example.com/callback#ACCESS_TOKEN=secret"

        result = SanitizedWebhookUrl.from_raw_url(url)

        assert "secret" not in result.sanitized_url
        assert "ACCESS_TOKEN" not in result.sanitized_url

    def test_non_sensitive_fragment_unchanged(self) -> None:
        """URL with only non-sensitive fragment is returned unchanged."""
        url = "https://example.com/page#section=intro&highlight=true"

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert result.sanitized_url == url
        assert result.fingerprint is None

    def test_plain_fragment_without_params_unchanged(self) -> None:
        """URL with plain fragment (no key=value) is unchanged."""
        url = "https://example.com/page#section-header"

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        assert result.sanitized_url == url
        assert result.fingerprint is None

    def test_oauth_implicit_flow_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Real-world OAuth implicit flow URL is properly sanitized."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        # Typical OAuth implicit flow response URL
        url = "https://myapp.com/callback#access_token=ya29.abc123&token_type=Bearer&expires_in=3600&state=xyz"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Secret stripped
        assert "ya29.abc123" not in result.sanitized_url
        assert "access_token" not in result.sanitized_url
        # Non-sensitive params preserved
        assert "token_type=Bearer" in result.sanitized_url
        assert "expires_in=3600" in result.sanitized_url
        assert "state=xyz" in result.sanitized_url
        assert result.fingerprint is not None


class TestPercentEncodedPasswordFingerprint:
    """Tests that fingerprinting uses decoded passwords, not percent-encoded text.

    urlparse().password preserves percent-encoding (e.g., p%40ss for p@ss).
    Fingerprinting the encoded form means the fingerprint represents the URL
    encoding rather than the actual secret value.
    """

    def test_database_url_fingerprints_decoded_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fingerprint of percent-encoded password matches fingerprint of decoded password."""
        from elspeth.contracts.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        # Password "p@ss" must be percent-encoded in URL as "p%40ss"
        url = "postgresql://user:p%40ss@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        # Fingerprint should be of the actual password "p@ss", not "p%40ss"
        expected = secret_fingerprint("p@ss")
        assert result.fingerprint == expected

    def test_database_url_percent_encoded_colon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fingerprint handles percent-encoded colon in password."""
        from elspeth.contracts.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        # Password "p:ss" encoded as "p%3Ass"
        url = "postgresql://user:p%3Ass@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        expected = secret_fingerprint("p:ss")
        assert result.fingerprint == expected

    def test_database_url_unencoded_password_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Passwords without percent-encoding are unaffected by decoding."""
        from elspeth.contracts.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url = "postgresql://user:plainpassword@host/db"

        result = SanitizedDatabaseUrl.from_raw_url(url)

        expected = secret_fingerprint("plainpassword")
        assert result.fingerprint == expected

    def test_webhook_basic_auth_fingerprints_decoded_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Webhook Basic Auth fingerprints use decoded username and password."""
        import json as json_module

        from elspeth.contracts.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        # Username "us@r" encoded as "us%40r", password "p@ss" as "p%40ss"
        url = "https://us%40r:p%40ss@api.example.com/webhook"

        result = SanitizedWebhookUrl.from_raw_url(url)

        # Fingerprint should use decoded values
        expected = secret_fingerprint(json_module.dumps(sorted(["us@r", "p@ss"]), separators=(",", ":")))
        assert result.fingerprint == expected


class TestIntegrationWithArtifactDescriptor:
    """Integration tests with ArtifactDescriptor."""

    def test_database_artifact_uses_sanitized_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ArtifactDescriptor.for_database accepts SanitizedDatabaseUrl."""
        from elspeth.contracts.results import ArtifactDescriptor

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        sanitized = SanitizedDatabaseUrl.from_raw_url("postgresql://user:secret@host/db")

        descriptor = ArtifactDescriptor.for_database(
            url=sanitized,
            table="results",
            content_hash="abc123",
            payload_size=100,
            row_count=10,
        )

        # URL should not contain secret
        assert "secret" not in descriptor.path_or_uri
        assert descriptor.metadata is not None
        assert "url_fingerprint" in descriptor.metadata

    def test_webhook_artifact_uses_sanitized_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ArtifactDescriptor.for_webhook accepts SanitizedWebhookUrl."""
        from elspeth.contracts.results import ArtifactDescriptor

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        sanitized = SanitizedWebhookUrl.from_raw_url("https://api.example.com?token=secret")

        descriptor = ArtifactDescriptor.for_webhook(
            url=sanitized,
            content_hash="def456",
            request_size=50,
            response_code=200,
        )

        # URL should not contain secret
        assert "secret" not in descriptor.path_or_uri
        assert descriptor.metadata is not None
        assert "url_fingerprint" in descriptor.metadata
