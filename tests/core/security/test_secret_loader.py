# tests/core/security/test_secret_loader.py
"""Tests for the SecretLoader abstraction.

This module tests:
1. SecretRef data class for audit-safe secret references
2. EnvSecretLoader for environment variable secrets
3. KeyVaultSecretLoader for Azure Key Vault secrets
4. CompositeSecretLoader for fallback chains
5. Caching behavior (P2 fix: Key Vault called at most once per secret per process)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


class TestSecretRef:
    """Test SecretRef data class."""

    def test_secret_ref_contains_no_secret_value(self) -> None:
        """SecretRef should never contain the actual secret value."""
        from elspeth.core.security.secret_loader import SecretRef

        ref = SecretRef(name="my-api-key", fingerprint="abc123", source="env")

        # Verify no way to access secret value
        assert not hasattr(ref, "value")
        assert not hasattr(ref, "secret")
        assert "sk-" not in str(ref)  # No secret patterns in repr

    def test_secret_ref_is_immutable(self) -> None:
        """SecretRef should be immutable (frozen dataclass)."""
        from elspeth.core.security.secret_loader import SecretRef

        ref = SecretRef(name="test", fingerprint="fp", source="env")

        with pytest.raises(AttributeError):
            ref.name = "changed"  # type: ignore[misc]


class TestEnvSecretLoader:
    """Test environment variable secret loading."""

    def test_loads_secret_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvSecretLoader retrieves secret from environment variable."""
        from elspeth.core.security.secret_loader import EnvSecretLoader

        monkeypatch.setenv("MY_SECRET", "secret-value-123")

        loader = EnvSecretLoader()
        value, ref = loader.get_secret("MY_SECRET")

        assert value == "secret-value-123"
        assert ref.name == "MY_SECRET"
        assert ref.source == "env"

    def test_raises_when_env_var_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvSecretLoader raises SecretNotFoundError when env var missing."""
        from elspeth.core.security.secret_loader import EnvSecretLoader, SecretNotFoundError

        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        loader = EnvSecretLoader()

        with pytest.raises(SecretNotFoundError, match="NONEXISTENT_VAR"):
            loader.get_secret("NONEXISTENT_VAR")

    def test_raises_when_env_var_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EnvSecretLoader raises SecretNotFoundError when env var is empty string."""
        from elspeth.core.security.secret_loader import EnvSecretLoader, SecretNotFoundError

        monkeypatch.setenv("EMPTY_VAR", "")

        loader = EnvSecretLoader()

        with pytest.raises(SecretNotFoundError, match="EMPTY_VAR"):
            loader.get_secret("EMPTY_VAR")

    def test_whitespace_only_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whitespace-only env var is considered valid (user's responsibility)."""
        from elspeth.core.security.secret_loader import EnvSecretLoader

        monkeypatch.setenv("WHITESPACE_VAR", "   ")

        loader = EnvSecretLoader()
        value, _ = loader.get_secret("WHITESPACE_VAR")

        assert value == "   "


class TestKeyVaultSecretLoader:
    """Test Azure Key Vault secret loading."""

    def test_loads_secret_from_keyvault(self) -> None:
        """KeyVaultSecretLoader retrieves secret from Azure Key Vault."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        mock_secret = MagicMock()
        mock_secret.value = "keyvault-secret-value"

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")
            value, ref = loader.get_secret("my-secret-name")

        assert value == "keyvault-secret-value"
        assert ref.name == "my-secret-name"
        assert ref.source == "keyvault"
        mock_client.get_secret.assert_called_once_with("my-secret-name")

    def test_raises_when_secret_not_found(self) -> None:
        """KeyVaultSecretLoader raises SecretNotFoundError when secret doesn't exist."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader, SecretNotFoundError

        mock_client = MagicMock()

        # Use the actual Azure exception type - generic Exception no longer triggers fallback
        from azure.core.exceptions import ResourceNotFoundError as AzureResourceNotFoundError

        mock_client.get_secret.side_effect = AzureResourceNotFoundError("SecretNotFound")

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            with pytest.raises(SecretNotFoundError):
                loader.get_secret("nonexistent-secret")

    def test_raises_when_secret_value_is_none(self) -> None:
        """KeyVaultSecretLoader raises SecretNotFoundError when secret value is None."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader, SecretNotFoundError

        mock_secret = MagicMock()
        mock_secret.value = None

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            with pytest.raises(SecretNotFoundError, match="has no value"):
                loader.get_secret("secret-with-null-value")

    def test_raises_when_resource_not_found(self) -> None:
        """KeyVaultSecretLoader raises SecretNotFoundError for ResourceNotFoundError (HTTP 404)."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader, SecretNotFoundError

        mock_client = MagicMock()

        # Simulate Azure SDK ResourceNotFoundError
        from azure.core.exceptions import ResourceNotFoundError as AzureResourceNotFoundError

        mock_client.get_secret.side_effect = AzureResourceNotFoundError("Secret not found")

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            with pytest.raises(SecretNotFoundError, match="nonexistent-secret"):
                loader.get_secret("nonexistent-secret")

    def test_propagates_auth_errors_not_converts_to_not_found(self) -> None:
        """KeyVaultSecretLoader must propagate auth errors, NOT convert to SecretNotFoundError.

        P2 Bug: Converting ClientAuthenticationError to SecretNotFoundError causes
        CompositeSecretLoader to silently fall back to lower-priority backends
        (e.g., env vars), potentially using the wrong credential in production.
        """
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        mock_client = MagicMock()

        # Simulate Azure SDK auth failure
        from azure.core.exceptions import ClientAuthenticationError

        mock_client.get_secret.side_effect = ClientAuthenticationError("Authentication failed")

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            # Should propagate ClientAuthenticationError, NOT convert to SecretNotFoundError
            with pytest.raises(ClientAuthenticationError):
                loader.get_secret("my-secret")

    def test_propagates_http_errors_not_converts_to_not_found(self) -> None:
        """KeyVaultSecretLoader must propagate HTTP errors (rate limiting, server errors).

        P2 Bug: Converting HttpResponseError (429 rate limit, 5xx server errors) to
        SecretNotFoundError causes silent fallback to wrong credentials.
        """
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        mock_client = MagicMock()

        # Simulate Azure SDK rate limit (HTTP 429)
        from azure.core.exceptions import HttpResponseError

        rate_limit_error = HttpResponseError("Rate limit exceeded")
        rate_limit_error.status_code = 429
        mock_client.get_secret.side_effect = rate_limit_error

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            # Should propagate HttpResponseError, NOT convert to SecretNotFoundError
            with pytest.raises(HttpResponseError):
                loader.get_secret("my-secret")

    def test_propagates_network_errors_not_converts_to_not_found(self) -> None:
        """KeyVaultSecretLoader must propagate network errors, NOT convert to SecretNotFoundError.

        P2 Bug: Converting ServiceRequestError (network issues) to SecretNotFoundError
        causes silent fallback when Key Vault is temporarily unreachable.
        """
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        mock_client = MagicMock()

        # Simulate Azure SDK network failure
        from azure.core.exceptions import ServiceRequestError

        mock_client.get_secret.side_effect = ServiceRequestError("Network unreachable")

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            # Should propagate ServiceRequestError, NOT convert to SecretNotFoundError
            with pytest.raises(ServiceRequestError):
                loader.get_secret("my-secret")


class TestKeyVaultSecretLoaderCaching:
    """Test Key Vault caching behavior (P2 fix).

    CRITICAL: Key Vault should be called AT MOST ONCE per secret per process.
    This prevents rate limiting and reduces costs.
    """

    def test_keyvault_called_only_once_per_secret(self) -> None:
        """Key Vault API should be called only once per secret, not on every get_secret()."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        mock_secret = MagicMock()
        mock_secret.value = "cached-value"

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            # Call get_secret multiple times for the same secret
            value1, _ = loader.get_secret("my-secret")
            value2, _ = loader.get_secret("my-secret")
            value3, _ = loader.get_secret("my-secret")

        # Key Vault should only be called ONCE
        assert mock_client.get_secret.call_count == 1
        assert value1 == value2 == value3 == "cached-value"

    def test_different_secrets_are_cached_independently(self) -> None:
        """Each secret should be cached independently."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        def mock_get_secret(name: str) -> MagicMock:
            secret = MagicMock()
            secret.value = f"value-for-{name}"
            return secret

        mock_client = MagicMock()
        mock_client.get_secret.side_effect = mock_get_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            # Get different secrets
            value_a, _ = loader.get_secret("secret-a")
            value_b, _ = loader.get_secret("secret-b")

            # Get them again (should be cached)
            value_a2, _ = loader.get_secret("secret-a")
            value_b2, _ = loader.get_secret("secret-b")

        # Each secret fetched exactly once
        assert mock_client.get_secret.call_count == 2
        calls = mock_client.get_secret.call_args_list
        assert call("secret-a") in calls
        assert call("secret-b") in calls

        # Values cached correctly
        assert value_a == value_a2 == "value-for-secret-a"
        assert value_b == value_b2 == "value-for-secret-b"

    def test_cache_clear_forces_refetch(self) -> None:
        """Clearing cache should force a refetch from Key Vault."""
        from elspeth.core.security.secret_loader import KeyVaultSecretLoader

        mock_secret = MagicMock()
        mock_secret.value = "original-value"

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            loader = KeyVaultSecretLoader(vault_url="https://test-vault.vault.azure.net")

            # First fetch
            loader.get_secret("my-secret")
            assert mock_client.get_secret.call_count == 1

            # Clear cache
            loader.clear_cache()

            # Fetch again - should hit Key Vault
            loader.get_secret("my-secret")
            assert mock_client.get_secret.call_count == 2


class TestCompositeSecretLoader:
    """Test composite secret loader with fallback chain."""

    def test_tries_backends_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CompositeSecretLoader tries backends in order until one succeeds."""
        from elspeth.core.security.secret_loader import (
            CompositeSecretLoader,
            EnvSecretLoader,
            KeyVaultSecretLoader,
        )

        # Set up env var
        monkeypatch.setenv("MY_SECRET", "env-value")

        mock_client = MagicMock()
        mock_secret = MagicMock()
        mock_secret.value = "keyvault-value"
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            env_loader = EnvSecretLoader()
            kv_loader = KeyVaultSecretLoader(vault_url="https://test.vault.azure.net")

            # Env first, then Key Vault
            composite = CompositeSecretLoader(backends=[env_loader, kv_loader])
            value, ref = composite.get_secret("MY_SECRET")

        # Should get from env (first in chain)
        assert value == "env-value"
        assert ref.source == "env"
        # Key Vault should NOT be called
        mock_client.get_secret.assert_not_called()

    def test_falls_back_to_next_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CompositeSecretLoader falls back to next backend on SecretNotFoundError."""
        from elspeth.core.security.secret_loader import (
            CompositeSecretLoader,
            EnvSecretLoader,
            KeyVaultSecretLoader,
        )

        # Env var NOT set
        monkeypatch.delenv("KV_ONLY_SECRET", raising=False)

        mock_client = MagicMock()
        mock_secret = MagicMock()
        mock_secret.value = "keyvault-value"
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            env_loader = EnvSecretLoader()
            kv_loader = KeyVaultSecretLoader(vault_url="https://test.vault.azure.net")

            composite = CompositeSecretLoader(backends=[env_loader, kv_loader])
            value, ref = composite.get_secret("KV_ONLY_SECRET")

        # Should fall back to Key Vault
        assert value == "keyvault-value"
        assert ref.source == "keyvault"
        mock_client.get_secret.assert_called_once_with("KV_ONLY_SECRET")

    def test_raises_when_all_backends_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CompositeSecretLoader raises SecretNotFoundError when all backends fail."""
        from elspeth.core.security.secret_loader import (
            CompositeSecretLoader,
            EnvSecretLoader,
            KeyVaultSecretLoader,
            SecretNotFoundError,
        )

        monkeypatch.delenv("NOWHERE_SECRET", raising=False)

        mock_client = MagicMock()

        # Use the actual Azure exception for "not found"
        from azure.core.exceptions import ResourceNotFoundError as AzureResourceNotFoundError

        mock_client.get_secret.side_effect = AzureResourceNotFoundError("Not found in Key Vault")

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            env_loader = EnvSecretLoader()
            kv_loader = KeyVaultSecretLoader(vault_url="https://test.vault.azure.net")

            composite = CompositeSecretLoader(backends=[env_loader, kv_loader])

            with pytest.raises(SecretNotFoundError, match=r"NOWHERE_SECRET.*not found in any backend"):
                composite.get_secret("NOWHERE_SECRET")

    def test_composite_does_not_fallback_on_auth_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CompositeSecretLoader must NOT fall back when Key Vault has auth errors.

        P2 Bug Scenario: Key Vault is configured (env var pointing to it) but auth fails.
        Current broken behavior: Auth error → SecretNotFoundError → falls back to env var.
        Correct behavior: Auth error propagates, pipeline fails fast.

        This is a security regression - using wrong credential silently is worse than failing.
        """
        from elspeth.core.security.secret_loader import (
            CompositeSecretLoader,
            EnvSecretLoader,
            KeyVaultSecretLoader,
        )

        # Set up an env var that could be used as fallback (e.g., dev credential)
        monkeypatch.setenv("API_KEY", "dev-key-from-env")

        mock_client = MagicMock()

        # Simulate auth failure in Key Vault
        from azure.core.exceptions import ClientAuthenticationError

        mock_client.get_secret.side_effect = ClientAuthenticationError("Token expired")

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            env_loader = EnvSecretLoader()
            kv_loader = KeyVaultSecretLoader(vault_url="https://prod-vault.vault.azure.net")

            # Key Vault first, env fallback (typical production config)
            composite = CompositeSecretLoader(backends=[kv_loader, env_loader])

            # Should propagate auth error, NOT silently use dev env var
            with pytest.raises(ClientAuthenticationError):
                composite.get_secret("API_KEY")


class TestCachedSecretLoader:
    """Test the caching wrapper for any secret loader."""

    def test_caches_successful_lookups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CachedSecretLoader caches successful secret lookups."""
        from elspeth.core.security.secret_loader import CachedSecretLoader, EnvSecretLoader

        monkeypatch.setenv("CACHED_SECRET", "the-value")

        inner = EnvSecretLoader()
        cached = CachedSecretLoader(inner)

        # Multiple calls
        v1, _ = cached.get_secret("CACHED_SECRET")
        v2, _ = cached.get_secret("CACHED_SECRET")
        v3, _ = cached.get_secret("CACHED_SECRET")

        assert v1 == v2 == v3 == "the-value"

    def test_does_not_cache_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CachedSecretLoader does not cache SecretNotFoundError."""
        from elspeth.core.security.secret_loader import (
            CachedSecretLoader,
            EnvSecretLoader,
            SecretNotFoundError,
        )

        monkeypatch.delenv("MISSING_SECRET", raising=False)

        inner = EnvSecretLoader()
        cached = CachedSecretLoader(inner)

        # First call - fails
        with pytest.raises(SecretNotFoundError):
            cached.get_secret("MISSING_SECRET")

        # Now set the env var
        monkeypatch.setenv("MISSING_SECRET", "now-exists")

        # Should try again (not return cached failure)
        value, _ = cached.get_secret("MISSING_SECRET")
        assert value == "now-exists"


class TestBackwardCompatibility:
    """Test backward compatibility with existing get_fingerprint_key() interface."""

    def test_get_fingerprint_key_uses_secret_loader_with_caching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_fingerprint_key() should use the new SecretLoader with caching."""
        from elspeth.core.security.fingerprint import (
            clear_fingerprint_key_cache,
            get_fingerprint_key,
        )

        # Clear any cached values from previous tests
        clear_fingerprint_key_cache()

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://test-vault.vault.azure.net")

        mock_secret = MagicMock()
        mock_secret.value = "cached-fingerprint-key"

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            # Call multiple times
            key1 = get_fingerprint_key()
            key2 = get_fingerprint_key()
            key3 = get_fingerprint_key()

            # All should return same value
            assert key1 == key2 == key3 == b"cached-fingerprint-key"

            # Key Vault should only be called ONCE (caching works)
            assert mock_client.get_secret.call_count == 1

        # Clean up
        clear_fingerprint_key_cache()

    def test_env_var_still_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ELSPETH_FINGERPRINT_KEY env var should still take precedence."""
        from elspeth.core.security.fingerprint import get_fingerprint_key

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-key")
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://test-vault.vault.azure.net")

        key = get_fingerprint_key()

        assert key == b"env-key"
