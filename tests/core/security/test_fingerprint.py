# tests/core/security/test_fingerprint.py
"""Tests for secret fingerprinting."""

import pytest

from elspeth.core.security.fingerprint import get_fingerprint_key, secret_fingerprint


class TestSecretFingerprint:
    """Test secret fingerprinting utility."""

    def test_fingerprint_returns_hex_string(self) -> None:
        """Fingerprint should be a hex string."""
        result = secret_fingerprint("my-api-key", key=b"test-key")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_fingerprint_is_deterministic(self) -> None:
        """Same secret + same key = same fingerprint."""
        key = b"test-key"
        fp1 = secret_fingerprint("my-secret", key=key)
        fp2 = secret_fingerprint("my-secret", key=key)
        assert fp1 == fp2

    def test_different_secrets_have_different_fingerprints(self) -> None:
        """Different secrets should produce different fingerprints."""
        key = b"test-key"
        fp1 = secret_fingerprint("secret-a", key=key)
        fp2 = secret_fingerprint("secret-b", key=key)
        assert fp1 != fp2

    def test_different_keys_produce_different_fingerprints(self) -> None:
        """Same secret with different keys should differ."""
        fp1 = secret_fingerprint("my-secret", key=b"key-1")
        fp2 = secret_fingerprint("my-secret", key=b"key-2")
        assert fp1 != fp2

    def test_fingerprint_length_is_64_chars(self) -> None:
        """SHA256 hex digest is 64 characters."""
        result = secret_fingerprint("test", key=b"key")
        assert len(result) == 64

    def test_fingerprint_golden_vector(self) -> None:
        """Verify HMAC-SHA256 algorithm with known test vector.

        This locks the algorithm to HMAC-SHA256. If the implementation
        changes to plain SHA256 or another algorithm, this test will fail.
        """
        result = secret_fingerprint("my-secret", key=b"test-key")

        # Precomputed: hmac.new(b"test-key", b"my-secret", sha256).hexdigest()
        expected = "2294b9e7a6dcb8be10f155c556b2ca74f419c7bd2ce6e1beec723751498f73c2"
        assert result == expected

    def test_fingerprint_without_key_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When key not provided, uses ELSPETH_FINGERPRINT_KEY env var."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-key-value")

        result = secret_fingerprint("my-secret")

        # Verify the env key is actually used by checking against expected HMAC
        # Precomputed: hmac.new(b"env-key-value", b"my-secret", sha256).hexdigest()
        expected = "9bbccfbb68be10d7a8b2649a63b421167e1c05cd78e52fe2761f1743691c5630"
        assert result == expected

    def test_fingerprint_without_key_raises_if_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError if no key provided and env var missing."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)

        with pytest.raises(ValueError, match="ELSPETH_FINGERPRINT_KEY"):
            secret_fingerprint("my-secret")


class TestGetFingerprintKey:
    """Test fingerprint key retrieval."""

    def test_get_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_fingerprint_key() reads from environment."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "my-secret-key")

        key = get_fingerprint_key()

        assert key == b"my-secret-key"

    def test_get_key_raises_if_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError if env var not set."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)

        with pytest.raises(ValueError):
            get_fingerprint_key()


class TestKeyVaultIntegration:
    """Test Azure Key Vault integration for fingerprint key."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear fingerprint key cache before each test."""
        from elspeth.core.security.fingerprint import clear_fingerprint_key_cache

        clear_fingerprint_key_cache()
        yield
        clear_fingerprint_key_cache()

    def test_keyvault_used_when_env_var_missing_and_vault_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ELSPETH_FINGERPRINT_KEY is missing but ELSPETH_KEYVAULT_URL is set, uses Key Vault."""
        from unittest.mock import MagicMock, patch

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        # Mock the SecretClient to avoid real Azure calls
        mock_secret = MagicMock()
        mock_secret.value = "keyvault-secret-value"
        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            key = get_fingerprint_key()

        assert key == b"keyvault-secret-value"
        mock_client.get_secret.assert_called_once_with("elspeth-fingerprint-key")

    def test_keyvault_uses_custom_secret_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ELSPETH_KEYVAULT_SECRET_NAME overrides the default secret name."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")
        monkeypatch.setenv("ELSPETH_KEYVAULT_SECRET_NAME", "custom-key-name")

        mock_secret = type("MockSecret", (), {"value": "custom-secret"})()

        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.secret_loader._get_keyvault_client",
            return_value=mock_client,
        ):
            key = get_fingerprint_key()

        mock_client.get_secret.assert_called_once_with("custom-key-name")
        assert key == b"custom-secret"

    def test_env_var_takes_precedence_over_keyvault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ELSPETH_FINGERPRINT_KEY env var takes precedence over Key Vault."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-var-key")
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        key = get_fingerprint_key()

        # Should use env var, not Key Vault
        assert key == b"env-var-key"

    def test_raises_when_neither_env_nor_keyvault_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when neither env var nor Key Vault is configured."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)

        with pytest.raises(ValueError, match=r"ELSPETH_FINGERPRINT_KEY.*ELSPETH_KEYVAULT_URL"):
            get_fingerprint_key()

    def test_keyvault_error_includes_helpful_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Key Vault errors are wrapped with helpful context."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.get_secret.side_effect = Exception("Azure auth failed")

        with (
            patch(
                "elspeth.core.security.secret_loader._get_keyvault_client",
                return_value=mock_client,
            ),
            pytest.raises(ValueError, match="Failed to retrieve fingerprint key from Key Vault"),
        ):
            get_fingerprint_key()

    def test_keyvault_raises_when_secret_value_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when Key Vault secret has no value."""
        from unittest.mock import MagicMock, patch

        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        mock_secret = MagicMock()
        mock_secret.value = None
        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with (
            patch("elspeth.core.security.secret_loader._get_keyvault_client", return_value=mock_client),
            pytest.raises(ValueError, match="has no value"),
        ):
            get_fingerprint_key()
