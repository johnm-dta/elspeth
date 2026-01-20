# Azure Key Vault Support for Fingerprint Key

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable ELSPETH to retrieve the HMAC fingerprint key from Azure Key Vault instead of only environment variables.

**Architecture:** Extend `get_fingerprint_key()` to check environment variable first (for dev/testing), then fall back to Azure Key Vault if `ELSPETH_KEYVAULT_URL` is set. Uses `DefaultAzureCredential` for flexible auth (managed identity, service principal, Azure CLI, etc.).

**Tech Stack:** azure-keyvault-secrets, azure-identity (already in azure deps)

---

## Task 1: Add azure-keyvault-secrets Dependency

**Files:**
- Modify: `pyproject.toml:93-98`

**Step 1: Add the dependency**

In `pyproject.toml`, update the `azure` optional dependencies:

```toml
azure = [
    # Phase 7: Azure plugin pack
    "azure-storage-blob>=12.19",
    "azure-identity>=1.15",
    "azure-keyvault-secrets>=4.7",  # Key Vault support for fingerprint key
    "jinja2>=3.1",  # Used by AzureBlobSink for path templating
]
```

**Step 2: Install updated dependencies**

Run: `uv pip install -e ".[azure]"`
Expected: Successfully installs azure-keyvault-secrets

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
deps: add azure-keyvault-secrets for fingerprint key storage

Enables retrieving the HMAC fingerprint key from Azure Key Vault
instead of only environment variables.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Write Failing Tests for Key Vault Integration

**Files:**
- Modify: `tests/core/security/test_fingerprint.py`

**Step 1: Write the failing tests**

Add a new test class at the end of `test_fingerprint.py`:

```python
class TestKeyVaultIntegration:
    """Test Azure Key Vault integration for fingerprint key."""

    def test_keyvault_used_when_env_var_missing_and_vault_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ELSPETH_FINGERPRINT_KEY is missing but ELSPETH_KEYVAULT_URL is set, uses Key Vault."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        # Mock the SecretClient to avoid real Azure calls
        mock_secret = type("MockSecret", (), {"value": "keyvault-secret-value"})()

        with pytest.MonkeyPatch.context() as m:
            # Import here to allow mocking
            from unittest.mock import MagicMock, patch

            mock_client = MagicMock()
            mock_client.get_secret.return_value = mock_secret

            with patch(
                "elspeth.core.security.fingerprint._get_keyvault_client",
                return_value=mock_client,
            ):
                key = get_fingerprint_key()

            assert key == b"keyvault-secret-value"
            mock_client.get_secret.assert_called_once_with("elspeth-fingerprint-key")

    def test_keyvault_uses_custom_secret_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ELSPETH_KEYVAULT_SECRET_NAME overrides the default secret name."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")
        monkeypatch.setenv("ELSPETH_KEYVAULT_SECRET_NAME", "custom-key-name")

        mock_secret = type("MockSecret", (), {"value": "custom-secret"})()

        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        with patch(
            "elspeth.core.security.fingerprint._get_keyvault_client",
            return_value=mock_client,
        ):
            key = get_fingerprint_key()

        mock_client.get_secret.assert_called_once_with("custom-key-name")
        assert key == b"custom-secret"

    def test_env_var_takes_precedence_over_keyvault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ELSPETH_FINGERPRINT_KEY env var takes precedence over Key Vault."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-var-key")
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        key = get_fingerprint_key()

        # Should use env var, not Key Vault
        assert key == b"env-var-key"

    def test_raises_when_neither_env_nor_keyvault_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises ValueError when neither env var nor Key Vault is configured."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_KEYVAULT_URL", raising=False)

        with pytest.raises(ValueError, match="ELSPETH_FINGERPRINT_KEY.*ELSPETH_KEYVAULT_URL"):
            get_fingerprint_key()

    def test_keyvault_error_includes_helpful_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Key Vault errors are wrapped with helpful context."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_KEYVAULT_URL", "https://my-vault.vault.azure.net")

        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.get_secret.side_effect = Exception("Azure auth failed")

        with patch(
            "elspeth.core.security.fingerprint._get_keyvault_client",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="Failed to retrieve fingerprint key from Key Vault"):
                get_fingerprint_key()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/core/security/test_fingerprint.py::TestKeyVaultIntegration -v`
Expected: FAIL - `_get_keyvault_client` doesn't exist

**Step 3: Commit failing tests**

```bash
git add tests/core/security/test_fingerprint.py
git commit -m "$(cat <<'EOF'
test: add failing tests for Key Vault fingerprint key integration

Tests Key Vault as fallback when env var not set, custom secret names,
env var precedence, and error handling.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement Key Vault Support in fingerprint.py

**Files:**
- Modify: `src/elspeth/core/security/fingerprint.py`

**Step 1: Add imports and constants**

Add after existing imports (line 22):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from azure.keyvault.secrets import SecretClient

_ENV_VAR = "ELSPETH_FINGERPRINT_KEY"
_KEYVAULT_URL_VAR = "ELSPETH_KEYVAULT_URL"
_KEYVAULT_SECRET_NAME_VAR = "ELSPETH_KEYVAULT_SECRET_NAME"
_DEFAULT_SECRET_NAME = "elspeth-fingerprint-key"
```

**Step 2: Add Key Vault client helper**

Add after the constants:

```python
def _get_keyvault_client(vault_url: str) -> "SecretClient":
    """Create a Key Vault SecretClient using DefaultAzureCredential.

    Args:
        vault_url: The Key Vault URL (e.g., https://my-vault.vault.azure.net)

    Returns:
        SecretClient configured with DefaultAzureCredential

    Raises:
        ImportError: If azure-keyvault-secrets or azure-identity not installed
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
    except ImportError as e:
        raise ImportError(
            "azure-keyvault-secrets and azure-identity are required for Key Vault support. "
            "Install with: uv pip install 'elspeth[azure]'"
        ) from e

    credential = DefaultAzureCredential()
    return SecretClient(vault_url=vault_url, credential=credential)
```

**Step 3: Update get_fingerprint_key function**

Replace the existing `get_fingerprint_key` function:

```python
def get_fingerprint_key() -> bytes:
    """Get the fingerprint key from environment or Azure Key Vault.

    Resolution order:
    1. ELSPETH_FINGERPRINT_KEY environment variable (immediate, for dev/testing)
    2. Azure Key Vault (if ELSPETH_KEYVAULT_URL is set)

    Environment variables:
    - ELSPETH_FINGERPRINT_KEY: Direct key value (takes precedence)
    - ELSPETH_KEYVAULT_URL: Key Vault URL (e.g., https://my-vault.vault.azure.net)
    - ELSPETH_KEYVAULT_SECRET_NAME: Secret name in Key Vault (default: elspeth-fingerprint-key)

    Returns:
        The fingerprint key as bytes

    Raises:
        ValueError: If neither env var nor Key Vault is configured, or Key Vault retrieval fails
    """
    # Priority 1: Environment variable (fast path for dev/testing)
    env_key = os.environ.get(_ENV_VAR)
    if env_key:
        return env_key.encode("utf-8")

    # Priority 2: Azure Key Vault
    vault_url = os.environ.get(_KEYVAULT_URL_VAR)
    if vault_url:
        secret_name = os.environ.get(_KEYVAULT_SECRET_NAME_VAR, _DEFAULT_SECRET_NAME)
        try:
            client = _get_keyvault_client(vault_url)
            secret = client.get_secret(secret_name)
            return secret.value.encode("utf-8")
        except ImportError:
            raise  # Re-raise ImportError as-is
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve fingerprint key from Key Vault "
                f"(url={vault_url}, secret={secret_name}): {e}"
            ) from e

    # Neither configured
    raise ValueError(
        f"Fingerprint key not configured. Set either:\n"
        f"  - {_ENV_VAR} environment variable (for dev/testing), or\n"
        f"  - {_KEYVAULT_URL_VAR} environment variable (for production with Azure Key Vault)"
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/core/security/test_fingerprint.py -v`
Expected: All tests PASS

**Step 5: Run type checker**

Run: `mypy src/elspeth/core/security/fingerprint.py`
Expected: Success (no errors)

**Step 6: Commit implementation**

```bash
git add src/elspeth/core/security/fingerprint.py
git commit -m "$(cat <<'EOF'
feat(security): add Azure Key Vault support for fingerprint key

The HMAC fingerprint key can now be stored in Azure Key Vault for
production deployments. Resolution order:
1. ELSPETH_FINGERPRINT_KEY env var (dev/testing)
2. Azure Key Vault via ELSPETH_KEYVAULT_URL (production)

Uses DefaultAzureCredential for flexible auth (managed identity,
service principal, Azure CLI, etc.).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update Security Module Exports

**Files:**
- Modify: `src/elspeth/core/security/__init__.py`

**Step 1: Add new export (optional, for testing)**

The `_get_keyvault_client` is internal, so no changes needed to `__init__.py`.
Verify the module still works:

Run: `python -c "from elspeth.core.security import get_fingerprint_key, secret_fingerprint; print('OK')"`
Expected: `OK`

**Step 2: Commit (skip if no changes)**

No changes needed - the public API is unchanged.

---

## Task 5: Add Integration Test (Optional, Skipped by Default)

**Files:**
- Create: `tests/integration/test_keyvault_fingerprint.py`

**Step 1: Write integration test**

```python
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

    def test_retrieves_secret_from_real_keyvault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Can retrieve fingerprint key from real Azure Key Vault."""
        # Remove env var to force Key Vault path
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        key = get_fingerprint_key()

        assert isinstance(key, bytes)
        assert len(key) > 0
```

**Step 2: Verify test is skipped without credentials**

Run: `pytest tests/integration/test_keyvault_fingerprint.py -v`
Expected: SKIPPED (ELSPETH_KEYVAULT_URL not set)

**Step 3: Commit**

```bash
git add tests/integration/test_keyvault_fingerprint.py
git commit -m "$(cat <<'EOF'
test: add integration test for Key Vault fingerprint retrieval

Requires real Azure credentials; skipped by default.
Run with: pytest -m integration tests/integration/

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `examples/azure_blob_sentiment/.env.example`
- Modify: `examples/azure_blob_sentiment/README.md`

**Step 1: Update .env.example**

Add Key Vault config options at the end of the file:

```bash
# Azure Key Vault (production alternative to ELSPETH_FINGERPRINT_KEY)
# ELSPETH_KEYVAULT_URL=https://your-vault.vault.azure.net
# ELSPETH_KEYVAULT_SECRET_NAME=elspeth-fingerprint-key
```

**Step 2: Add Key Vault section to README.md**

Add after the "Authentication Options" section:

```markdown
## Secret Management with Azure Key Vault

For production deployments, store the ELSPETH fingerprint key in Azure Key Vault instead of environment variables.

### Setup

1. Create a Key Vault and add a secret:
```bash
# Create Key Vault (if needed)
az keyvault create --name my-elspeth-vault --resource-group my-rg --location eastus

# Add the fingerprint key secret
az keyvault secret set \
  --vault-name my-elspeth-vault \
  --name elspeth-fingerprint-key \
  --value "$(openssl rand -base64 32)"
```

2. Grant access to your workload identity:
```bash
# For Managed Identity (recommended for Azure-hosted workloads)
az keyvault set-policy --name my-elspeth-vault \
  --object-id <managed-identity-object-id> \
  --secret-permissions get

# For Service Principal
az keyvault set-policy --name my-elspeth-vault \
  --spn <service-principal-app-id> \
  --secret-permissions get
```

3. Configure ELSPETH:
```bash
export ELSPETH_KEYVAULT_URL="https://my-elspeth-vault.vault.azure.net"
# Optional: custom secret name (default: elspeth-fingerprint-key)
# export ELSPETH_KEYVAULT_SECRET_NAME="my-custom-secret-name"
```

### Resolution Order

ELSPETH checks for the fingerprint key in this order:
1. `ELSPETH_FINGERPRINT_KEY` environment variable (for dev/testing)
2. Azure Key Vault (if `ELSPETH_KEYVAULT_URL` is set)

This allows local development with env vars while production uses Key Vault.
```

**Step 3: Commit documentation**

```bash
git add examples/azure_blob_sentiment/.env.example examples/azure_blob_sentiment/README.md
git commit -m "$(cat <<'EOF'
docs: add Azure Key Vault configuration for fingerprint key

Documents how to store the HMAC fingerprint key in Key Vault for
production deployments, including setup commands and resolution order.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Run Full Test Suite and Lint

**Files:** None (verification only)

**Step 1: Run linter**

Run: `ruff check src/elspeth/core/security/ tests/core/security/`
Expected: No errors

**Step 2: Run type checker**

Run: `mypy src/elspeth/core/security/`
Expected: Success

**Step 3: Run all security tests**

Run: `pytest tests/core/security/ -v`
Expected: All tests pass

**Step 4: Run full test suite**

Run: `pytest tests/ -x --ignore=tests/integration`
Expected: All tests pass

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add azure-keyvault-secrets dependency | `pyproject.toml` |
| 2 | Write failing tests for Key Vault integration | `tests/core/security/test_fingerprint.py` |
| 3 | Implement Key Vault support | `src/elspeth/core/security/fingerprint.py` |
| 4 | Verify module exports | `src/elspeth/core/security/__init__.py` |
| 5 | Add integration test (optional) | `tests/integration/test_keyvault_fingerprint.py` |
| 6 | Update documentation | `.env.example`, `README.md` |
| 7 | Final verification | (none) |

**Environment Variables:**
- `ELSPETH_FINGERPRINT_KEY` - Direct key value (dev/testing, takes precedence)
- `ELSPETH_KEYVAULT_URL` - Key Vault URL (production)
- `ELSPETH_KEYVAULT_SECRET_NAME` - Secret name (default: `elspeth-fingerprint-key`)
