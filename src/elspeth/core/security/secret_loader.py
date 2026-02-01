# src/elspeth/core/security/secret_loader.py
"""General-purpose secret loading abstraction.

This module provides a unified interface for loading secrets from multiple backends
(environment variables, Azure Key Vault, etc.) with built-in caching to prevent
repeated API calls.

Usage:
    from elspeth.core.security.secret_loader import (
        CompositeSecretLoader,
        EnvSecretLoader,
        KeyVaultSecretLoader,
    )

    # Create a loader chain (env first, then Key Vault)
    loader = CompositeSecretLoader(backends=[
        EnvSecretLoader(),
        KeyVaultSecretLoader(vault_url="https://my-vault.vault.azure.net"),
    ])

    # Get a secret (cached automatically)
    value, ref = loader.get_secret("OPENAI_API_KEY")
    # ref contains fingerprint and source for audit trail
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from azure.keyvault.secrets import SecretClient


class SecretNotFoundError(Exception):
    """Raised when a secret cannot be found in any backend."""

    pass


@dataclass(frozen=True, slots=True)
class SecretRef:
    """Reference to a secret for audit trail (never contains actual value).

    This is what gets recorded in the audit trail - the fingerprint allows
    verification that the same secret was used, without exposing the value.

    Attributes:
        name: The secret name/identifier
        fingerprint: HMAC-SHA256 fingerprint of the secret value (if available)
        source: Where the secret was loaded from ("env", "keyvault", etc.)
    """

    name: str
    fingerprint: str
    source: str


class SecretLoader(Protocol):
    """Protocol for secret loading backends.

    Each backend implements this protocol to provide a consistent interface
    for loading secrets from different sources.
    """

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Load a secret by name.

        Args:
            name: The secret identifier (env var name, Key Vault secret name, etc.)

        Returns:
            Tuple of (secret_value, SecretRef for audit)

        Raises:
            SecretNotFoundError: If the secret doesn't exist in this backend
        """
        ...


def _get_keyvault_client(vault_url: str) -> SecretClient:
    """Create a Key Vault SecretClient using DefaultAzureCredential.

    This is a shared helper used by KeyVaultSecretLoader.

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


class EnvSecretLoader:
    """Load secrets from environment variables.

    This is the fast path for development and testing. Environment variables
    take precedence over other backends when configured.
    """

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Load a secret from an environment variable.

        Args:
            name: The environment variable name

        Returns:
            Tuple of (secret_value, SecretRef)

        Raises:
            SecretNotFoundError: If the env var is not set or is empty
        """
        value = os.environ.get(name)

        if value is None or value == "":
            raise SecretNotFoundError(f"Environment variable '{name}' not set or empty")

        # No fingerprint computed here - that's the caller's responsibility
        # (they have the fingerprint key, we don't)
        ref = SecretRef(name=name, fingerprint="", source="env")
        return value, ref


class KeyVaultSecretLoader:
    """Load secrets from Azure Key Vault with built-in caching.

    This loader caches secrets to prevent repeated Key Vault API calls.
    Each secret is fetched at most once per KeyVaultSecretLoader instance.

    Caching behavior:
    - Successful lookups are cached forever (until clear_cache() or process restart)
    - Failed lookups (SecretNotFoundError) are NOT cached (allows retry after fix)
    """

    def __init__(self, vault_url: str) -> None:
        """Initialize the Key Vault loader.

        Args:
            vault_url: The Key Vault URL (e.g., https://my-vault.vault.azure.net)
        """
        self._vault_url = vault_url
        self._client: SecretClient | None = None
        self._cache: dict[str, str] = {}

    def _get_client(self) -> SecretClient:
        """Get or create the Key Vault client (lazy initialization)."""
        if self._client is None:
            self._client = _get_keyvault_client(self._vault_url)
        return self._client

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Load a secret from Azure Key Vault (with caching).

        Args:
            name: The secret name in Key Vault

        Returns:
            Tuple of (secret_value, SecretRef)

        Raises:
            SecretNotFoundError: If the secret doesn't exist or has no value
            azure.core.exceptions.ClientAuthenticationError: If authentication fails
            azure.core.exceptions.HttpResponseError: For HTTP errors (rate limiting, server errors)
            azure.core.exceptions.ServiceRequestError: For network/connectivity issues
        """
        # Check cache first
        if name in self._cache:
            ref = SecretRef(name=name, fingerprint="", source="keyvault")
            return self._cache[name], ref

        # Import Azure exceptions for proper error handling
        # These are only used when azure-keyvault-secrets is available
        try:
            from azure.core.exceptions import ResourceNotFoundError as AzureResourceNotFoundError
        except ImportError:
            # If azure.core isn't available, we'll raise ImportError from _get_client() anyway
            AzureResourceNotFoundError = Exception  # type: ignore[misc, assignment]

        # Fetch from Key Vault
        try:
            client = self._get_client()
            secret = client.get_secret(name)
            value: str | None = secret.value

            if value is None:
                raise SecretNotFoundError(f"Key Vault secret '{name}' has no value")

            # Cache the result
            self._cache[name] = value

            ref = SecretRef(name=name, fingerprint="", source="keyvault")
            return value, ref

        except SecretNotFoundError:
            # Re-raise our own SecretNotFoundError (from value is None check)
            raise
        except ImportError:
            # Re-raise ImportError as-is - missing package is different from missing secret
            raise
        except AzureResourceNotFoundError as e:
            # HTTP 404 - secret genuinely doesn't exist in Key Vault
            # This is the ONLY Azure exception that should trigger fallback
            raise SecretNotFoundError(f"Secret '{name}' not found in Key Vault ({self._vault_url})") from e
        # All other Azure exceptions (auth errors, rate limits, network issues) propagate
        # Do NOT catch Exception - operational failures must fail fast, not silently fall back

    def clear_cache(self) -> None:
        """Clear the secret cache, forcing refetch on next access."""
        self._cache.clear()


class CachedSecretLoader:
    """Caching wrapper for any SecretLoader.

    Wraps another loader and caches successful lookups. Failed lookups
    (SecretNotFoundError) are NOT cached to allow retry after configuration fix.
    """

    def __init__(self, inner: SecretLoader) -> None:
        """Initialize the caching wrapper.

        Args:
            inner: The underlying loader to cache
        """
        self._inner = inner
        self._cache: dict[str, tuple[str, SecretRef]] = {}

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Load a secret (from cache if available).

        Args:
            name: The secret identifier

        Returns:
            Tuple of (secret_value, SecretRef)

        Raises:
            SecretNotFoundError: If the secret doesn't exist
        """
        if name in self._cache:
            return self._cache[name]

        result = self._inner.get_secret(name)
        self._cache[name] = result
        return result

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class CompositeSecretLoader:
    """Load secrets from multiple backends in priority order.

    Tries each backend in order until one succeeds. This allows configuration
    like "use env var if set, otherwise fall back to Key Vault".
    """

    def __init__(self, backends: list[SecretLoader]) -> None:
        """Initialize with a list of backends to try in order.

        Args:
            backends: List of SecretLoader instances, tried in order
        """
        if not backends:
            raise ValueError("CompositeSecretLoader requires at least one backend")
        self._backends = backends

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Load a secret, trying backends in order.

        Args:
            name: The secret identifier

        Returns:
            Tuple of (secret_value, SecretRef from successful backend)

        Raises:
            SecretNotFoundError: If no backend has the secret
        """
        for backend in self._backends:
            try:
                return backend.get_secret(name)
            except SecretNotFoundError:
                continue

        raise SecretNotFoundError(f"Secret '{name}' not found in any backend")
