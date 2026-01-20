# tests/integration/test_keyvault_fingerprint.py
"""Integration tests for Azure Key Vault fingerprint key retrieval.

These tests require real Azure credentials and a Key Vault with a secret.
Skip by default; run with: pytest -m integration tests/integration/test_keyvault_fingerprint.py

Setup:
1. Create Azure Key Vault
2. Create secret named 'elspeth-fingerprint-key' with a random value
3. Set ELSPETH_KEYVAULT_URL environment variable
4. Authenticate via Azure CLI: az login
"""

import os

import pytest

from elspeth.core.security import get_fingerprint_key


@pytest.mark.integration
class TestKeyVaultIntegration:
    """Integration tests requiring real Azure Key Vault."""

    @pytest.fixture(autouse=True)
    def require_keyvault_url(self) -> None:
        """Skip if ELSPETH_KEYVAULT_URL not configured."""
        if not os.environ.get("ELSPETH_KEYVAULT_URL"):
            pytest.skip("ELSPETH_KEYVAULT_URL not set - skipping Key Vault integration test")

    def test_retrieves_secret_from_real_keyvault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Can retrieve fingerprint key from real Azure Key Vault."""
        # Remove env var to force Key Vault path
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        key = get_fingerprint_key()

        assert isinstance(key, bytes)
        assert len(key) > 0
