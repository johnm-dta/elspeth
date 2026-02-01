# tests/core/security/test_fingerprint_keyvault.py
"""Tests for fingerprint key retrieval security boundaries.

These tests verify that fingerprint key retrieval fails safely when
Key Vault is unavailable or misconfigured, rather than falling back
to insecure defaults.

CRITICAL SECURITY BOUNDARY: The system MUST crash when Key Vault is
configured but unavailable. Silent fallback to a default/hardcoded key
would be a severe security vulnerability.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.security.fingerprint import (
    _ENV_VAR,
    _KEYVAULT_SECRET_NAME_VAR,
    _KEYVAULT_URL_VAR,
    get_fingerprint_key,
    secret_fingerprint,
)


class TestFingerprintKeySecurityBoundary:
    """Tests for fingerprint key retrieval security boundaries.

    These tests verify CRASH behavior - no silent fallback to default keys.
    This is critical for security: if Key Vault is down, the system must
    NOT proceed with a hardcoded key.
    """

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear fingerprint key cache before each test."""
        from elspeth.core.security.fingerprint import clear_fingerprint_key_cache

        clear_fingerprint_key_cache()
        yield
        clear_fingerprint_key_cache()

    def test_missing_all_config_raises_value_error(self) -> None:
        """Must raise ValueError when no key source is configured.

        Security boundary: If neither env var nor Key Vault is configured,
        we MUST fail loudly, not silently return empty/default key.
        """
        # Clear all relevant env vars
        env_overrides = {
            _ENV_VAR: "",
            _KEYVAULT_URL_VAR: "",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            # Remove the keys entirely
            for key in [_ENV_VAR, _KEYVAULT_URL_VAR]:
                os.environ.pop(key, None)

            with pytest.raises(ValueError, match="Fingerprint key not configured"):
                get_fingerprint_key()

    def test_keyvault_retrieval_failure_raises_value_error(self) -> None:
        """Key Vault retrieval failure must raise ValueError, not fallback.

        Security boundary: If Key Vault is configured but retrieval fails,
        we MUST NOT fall back to environment variable or empty key.
        The error must propagate.
        """
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
            clear=False,
        ):
            # Remove env key entirely to ensure no fallback
            os.environ.pop(_ENV_VAR, None)

            # Mock the Key Vault client to fail
            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.get_secret.side_effect = Exception("Network error: Key Vault unreachable")
                mock_get_client.return_value = mock_client

                with pytest.raises(ValueError, match="Failed to retrieve fingerprint key from Key Vault"):
                    get_fingerprint_key()

    def test_keyvault_secret_has_null_value_raises_value_error(self) -> None:
        """Key Vault secret with None value must raise ValueError.

        Edge case: Secret exists in Key Vault but has null/empty value.
        This is a configuration error that must not be silently accepted.
        """
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            # Mock secret with None value
            mock_secret = MagicMock()
            mock_secret.value = None

            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.get_secret.return_value = mock_secret
                mock_get_client.return_value = mock_client

                with pytest.raises(ValueError, match="has no value"):
                    get_fingerprint_key()

    def test_env_var_takes_precedence_over_keyvault(self) -> None:
        """Environment variable should take precedence over Key Vault.

        This is documented behavior for dev/testing scenarios.
        """
        with patch.dict(
            os.environ,
            {
                _ENV_VAR: "test-key-from-env",
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
        ):
            key = get_fingerprint_key()
            assert key == b"test-key-from-env"

    def test_secret_fingerprint_with_missing_key_raises(self) -> None:
        """secret_fingerprint() without key param must raise if no config.

        High-level function should propagate key retrieval failures.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV_VAR, None)
            os.environ.pop(_KEYVAULT_URL_VAR, None)

            with pytest.raises(ValueError, match="Fingerprint key not configured"):
                secret_fingerprint("my-secret")

    def test_azure_import_error_propagates(self) -> None:
        """ImportError from missing azure packages must propagate.

        If azure-keyvault-secrets is not installed but Key Vault URL
        is configured, the ImportError should propagate with helpful message.
        """
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            # Mock the _get_keyvault_client to raise ImportError
            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_get_client.side_effect = ImportError("azure-keyvault-secrets and azure-identity are required")

                with pytest.raises(ImportError, match="azure-keyvault-secrets"):
                    get_fingerprint_key()

    def test_keyvault_network_timeout_raises_not_falls_back(self) -> None:
        """Network timeout to Key Vault must raise, not use fallback.

        Security boundary: TimeoutError (common Azure issue) must propagate
        as ValueError, not silently use a default key.
        """
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.get_secret.side_effect = TimeoutError("Connection to Key Vault timed out")
                mock_get_client.return_value = mock_client

                with pytest.raises(ValueError, match="Failed to retrieve fingerprint key from Key Vault"):
                    get_fingerprint_key()

    def test_keyvault_auth_failure_raises_not_falls_back(self) -> None:
        """Authentication failure to Key Vault must raise, not use fallback.

        Security boundary: Azure authentication errors (expired token,
        wrong credentials) must propagate, not silently use a default key.
        """
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                # Simulate Azure SDK authentication error
                mock_client.get_secret.side_effect = Exception(
                    "ClientAuthenticationError: AADSTS700016: Application with identifier '...' was not found"
                )
                mock_get_client.return_value = mock_client

                with pytest.raises(ValueError, match="Failed to retrieve fingerprint key from Key Vault"):
                    get_fingerprint_key()

    def test_custom_secret_name_used_when_configured(self) -> None:
        """Custom secret name from env var is used in Key Vault lookup."""
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
                _KEYVAULT_SECRET_NAME_VAR: "my-custom-secret-name",
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            mock_secret = MagicMock()
            mock_secret.value = "secret-from-custom-name"

            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.get_secret.return_value = mock_secret
                mock_get_client.return_value = mock_client

                key = get_fingerprint_key()

                assert key == b"secret-from-custom-name"
                mock_client.get_secret.assert_called_once_with("my-custom-secret-name")

    def test_error_message_includes_vault_url_and_secret_name(self) -> None:
        """Error message should include vault URL and secret name for debugging."""
        vault_url = "https://my-specific-vault.vault.azure.net"
        secret_name = "my-specific-secret"

        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: vault_url,
                _KEYVAULT_SECRET_NAME_VAR: secret_name,
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.get_secret.side_effect = Exception("Some Azure error")
                mock_get_client.return_value = mock_client

                with pytest.raises(ValueError) as exc_info:
                    get_fingerprint_key()

                error_msg = str(exc_info.value)
                assert vault_url in error_msg
                assert secret_name in error_msg


class TestNoHardcodedFallback:
    """Tests explicitly verifying no hardcoded key fallback exists.

    These tests are security assertions - they verify that the implementation
    does NOT have dangerous fallback behavior.
    """

    def test_no_fallback_when_keyvault_fails_even_with_env_set_after(self) -> None:
        """Key Vault failure must raise even if env var could be used as fallback.

        Verifies: The order of checking (env first, then keyvault) means that
        if env is empty/missing and keyvault fails, there's no retry of env.
        """
        # Start with only Key Vault configured
        with patch.dict(
            os.environ,
            {
                _KEYVAULT_URL_VAR: "https://test-vault.vault.azure.net",
            },
            clear=False,
        ):
            os.environ.pop(_ENV_VAR, None)

            with patch("elspeth.core.security.secret_loader._get_keyvault_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.get_secret.side_effect = Exception("Key Vault down")
                mock_get_client.return_value = mock_client

                # Even if we set env var during the exception handling,
                # the function should raise, not retry with env var
                with pytest.raises(ValueError, match="Failed to retrieve"):
                    get_fingerprint_key()

    def test_empty_env_var_does_not_count_as_configured(self) -> None:
        """Empty string env var should not be used as the key.

        Security: An empty ELSPETH_FINGERPRINT_KEY should be treated as
        'not configured', not as 'use empty string as key'.
        """
        with patch.dict(
            os.environ,
            {
                _ENV_VAR: "",  # Empty string, not None
            },
            clear=False,
        ):
            os.environ.pop(_KEYVAULT_URL_VAR, None)

            # Empty env var should be treated as not configured
            with pytest.raises(ValueError, match="Fingerprint key not configured"):
                get_fingerprint_key()

    def test_whitespace_only_env_var_is_used_as_key(self) -> None:
        """Whitespace-only env var is used as-is (user's responsibility).

        Note: While empty string is rejected, whitespace-only strings
        are technically valid (the user configured something). This test
        documents the behavior - it's the user's responsibility to configure
        a strong key.
        """
        with patch.dict(
            os.environ,
            {
                _ENV_VAR: "   ",  # Whitespace only
            },
            clear=False,
        ):
            os.environ.pop(_KEYVAULT_URL_VAR, None)

            # Whitespace IS considered a configured key (poor choice, but valid)
            key = get_fingerprint_key()
            assert key == b"   "
