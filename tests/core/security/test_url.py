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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
        url = "postgresql://user:secret@host/db"

        with pytest.raises(SecretFingerprintError, match="ELSPETH_FINGERPRINT_KEY"):
            SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=True)

    def test_dev_mode_sanitizes_without_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dev mode (fail_if_no_key=False) sanitizes but returns None fingerprint."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
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

    def test_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SanitizedDatabaseUrl is frozen (immutable)."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        result = SanitizedDatabaseUrl.from_raw_url("postgresql://user:pass@host/db")

        with pytest.raises(AttributeError):
            result.sanitized_url = "hacked"  # type: ignore[misc]


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
        from elspeth.core.security.fingerprint import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        url_userpass = "https://user:pass@api.example.com/webhook"
        url_user_only = "https://user@api.example.com/webhook"

        result_userpass = SanitizedWebhookUrl.from_raw_url(url_userpass)
        result_user_only = SanitizedWebhookUrl.from_raw_url(url_user_only)

        # Both must have fingerprints (not None)
        assert result_userpass.fingerprint is not None
        assert result_user_only.fingerprint is not None

        # Verify exact fingerprints match expected HMAC of sorted credentials
        # For user:pass -> sorted(["user", "pass"]) = ["pass", "user"] -> "pass|user"
        expected_userpass = secret_fingerprint("pass|user")
        assert result_userpass.fingerprint == expected_userpass

        # For user only -> "user"
        expected_user_only = secret_fingerprint("user")
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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
        url = "https://api.example.com?token=secret"

        with pytest.raises(SecretFingerprintError, match="ELSPETH_FINGERPRINT_KEY"):
            SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=True)

    def test_dev_mode_sanitizes_without_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dev mode (fail_if_no_key=False) sanitizes but returns None fingerprint."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
        url = "https://api.example.com/hook?token="

        result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

        # SECURITY: Parameter name 'token' must not appear in sanitized URL
        assert "token" not in result.sanitized_url
        assert result.sanitized_url == "https://api.example.com/hook"
        assert result.fingerprint is None  # No non-empty values to fingerprint

    def test_empty_api_key_value_still_strips_param_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty api_key value still removes param name from URL."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
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
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)
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
