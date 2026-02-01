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

from elspeth.core.security.secret_loader import (
    KeyVaultSecretLoader,
    SecretNotFoundError,
)

if TYPE_CHECKING:
    from azure.keyvault.secrets import SecretClient

_ENV_VAR = "ELSPETH_FINGERPRINT_KEY"
_KEYVAULT_URL_VAR = "ELSPETH_KEYVAULT_URL"
_KEYVAULT_SECRET_NAME_VAR = "ELSPETH_KEYVAULT_SECRET_NAME"
_DEFAULT_SECRET_NAME = "elspeth-fingerprint-key"

# Module-level cache for the fingerprint key (P2 fix)
# This ensures Key Vault is called at most once per process
_cached_fingerprint_key: bytes | None = None
_cached_keyvault_loader: KeyVaultSecretLoader | None = None


def _get_keyvault_client(vault_url: str) -> SecretClient:
    """Create a Key Vault SecretClient using DefaultAzureCredential.

    Args:
        vault_url: The Key Vault URL (e.g., https://my-vault.vault.azure.net)

    Returns:
        SecretClient configured with DefaultAzureCredential

    Raises:
        ImportError: If azure-keyvault-secrets or azure-identity not installed
    """
    # Delegate to secret_loader's implementation
    from elspeth.core.security.secret_loader import _get_keyvault_client as _loader_get_client

    return _loader_get_client(vault_url)


def _get_cached_keyvault_loader(vault_url: str) -> KeyVaultSecretLoader:
    """Get or create a cached KeyVaultSecretLoader instance.

    This ensures we reuse the same loader (and its cache) across calls.
    """
    global _cached_keyvault_loader

    if _cached_keyvault_loader is None or _cached_keyvault_loader._vault_url != vault_url:
        _cached_keyvault_loader = KeyVaultSecretLoader(vault_url=vault_url)

    return _cached_keyvault_loader


def get_fingerprint_key() -> bytes:
    """Get the fingerprint key from environment or Azure Key Vault.

    Resolution order:
    1. ELSPETH_FINGERPRINT_KEY environment variable (immediate, for dev/testing)
    2. Azure Key Vault (if ELSPETH_KEYVAULT_URL is set) - CACHED after first fetch

    Environment variables:
    - ELSPETH_FINGERPRINT_KEY: Direct key value (takes precedence)
    - ELSPETH_KEYVAULT_URL: Key Vault URL (e.g., https://my-vault.vault.azure.net)
    - ELSPETH_KEYVAULT_SECRET_NAME: Secret name in Key Vault (default: elspeth-fingerprint-key)

    Returns:
        The fingerprint key as bytes

    Raises:
        ValueError: If neither env var nor Key Vault is configured, or Key Vault retrieval fails

    Note:
        Key Vault lookups are cached - the API is called at most once per secret
        per process lifetime. This prevents rate limiting and reduces costs.
    """
    global _cached_fingerprint_key

    # Priority 1: Environment variable (fast path for dev/testing)
    # Always check env var first - it may change at runtime
    env_key = os.environ.get(_ENV_VAR)
    if env_key:
        return env_key.encode("utf-8")

    # Priority 2: Check module-level cache (P2 fix: avoid repeated Key Vault calls)
    if _cached_fingerprint_key is not None:
        return _cached_fingerprint_key

    # Priority 3: Azure Key Vault (with caching via KeyVaultSecretLoader)
    vault_url = os.environ.get(_KEYVAULT_URL_VAR)
    if vault_url:
        secret_name = os.environ.get(_KEYVAULT_SECRET_NAME_VAR, _DEFAULT_SECRET_NAME)
        try:
            loader = _get_cached_keyvault_loader(vault_url)
            secret_value, _ = loader.get_secret(secret_name)
            # Cache at module level as well (belt and suspenders)
            _cached_fingerprint_key = secret_value.encode("utf-8")
            return _cached_fingerprint_key
        except ImportError:
            raise  # Re-raise ImportError as-is
        except SecretNotFoundError as e:
            raise ValueError(f"Failed to retrieve fingerprint key from Key Vault (url={vault_url}, secret={secret_name}): {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to retrieve fingerprint key from Key Vault (url={vault_url}, secret={secret_name}): {e}") from e

    # Neither configured
    raise ValueError(
        f"Fingerprint key not configured. Set {_ENV_VAR} (dev/testing) or {_KEYVAULT_URL_VAR} (production with Azure Key Vault)."
    )


def clear_fingerprint_key_cache() -> None:
    """Clear the cached fingerprint key, forcing refetch on next access.

    This is primarily useful for testing. In production, the cache persists
    for the process lifetime.
    """
    global _cached_fingerprint_key, _cached_keyvault_loader

    _cached_fingerprint_key = None
    if _cached_keyvault_loader is not None:
        _cached_keyvault_loader.clear_cache()
        _cached_keyvault_loader = None


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
