# src/elspeth/core/security/fingerprint.py
"""Secret fingerprinting using HMAC-SHA256.

Secrets (API keys, tokens, passwords) should never appear in the audit trail.
Instead, we store a fingerprint that can verify "same secret was used"
without revealing the actual secret value.

Usage:
    from elspeth.core.security import secret_fingerprint

    # With explicit key
    fp = secret_fingerprint(api_key, key=signing_key)

    # With environment variable (ELSPETH_FINGERPRINT_KEY)
    fp = secret_fingerprint(api_key)
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from azure.keyvault.secrets import SecretClient

_ENV_VAR = "ELSPETH_FINGERPRINT_KEY"
_KEYVAULT_URL_VAR = "ELSPETH_KEYVAULT_URL"
_KEYVAULT_SECRET_NAME_VAR = "ELSPETH_KEYVAULT_SECRET_NAME"
_DEFAULT_SECRET_NAME = "elspeth-fingerprint-key"


def _get_keyvault_client(vault_url: str) -> SecretClient:
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
            "azure-keyvault-secrets and azure-identity are required for Key Vault support. Install with: uv pip install 'elspeth[azure]'"
        ) from e

    credential = DefaultAzureCredential()
    return SecretClient(vault_url=vault_url, credential=credential)


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
            if secret.value is None:
                raise ValueError(f"Secret '{secret_name}' has no value")
            return secret.value.encode("utf-8")
        except ImportError:
            raise  # Re-raise ImportError as-is
        except Exception as e:
            raise ValueError(f"Failed to retrieve fingerprint key from Key Vault (url={vault_url}, secret={secret_name}): {e}") from e

    # Neither configured
    raise ValueError(
        f"Fingerprint key not configured. Set {_ENV_VAR} (dev/testing) or {_KEYVAULT_URL_VAR} (production with Azure Key Vault)."
    )


def secret_fingerprint(secret: str, *, key: bytes | None = None) -> str:
    """Compute HMAC-SHA256 fingerprint of a secret.

    The fingerprint can be stored in the audit trail to verify that
    the same secret was used across runs, without exposing the secret.

    Args:
        secret: The secret value to fingerprint (API key, token, etc.)
        key: HMAC key. If not provided, reads from ELSPETH_FINGERPRINT_KEY env var.

    Returns:
        64-character hex string (SHA256 digest)

    Raises:
        ValueError: If key is None and ELSPETH_FINGERPRINT_KEY not set

    Example:
        >>> fp = secret_fingerprint("sk-abc123", key=b"my-signing-key")
        >>> len(fp)
        64
        >>> fp == secret_fingerprint("sk-abc123", key=b"my-signing-key")
        True
    """
    if key is None:
        key = get_fingerprint_key()

    digest = hmac.new(
        key=key,
        msg=secret.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    return digest
