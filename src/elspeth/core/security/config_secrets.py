"""Config-based secret loading from Azure Key Vault.

This module loads secrets specified in pipeline configuration and injects
them into environment variables before config resolution.

IMPORTANT: This module reuses the existing KeyVaultSecretLoader from
secret_loader.py to avoid code duplication and maintain consistent caching.

Resolution records are returned with HMAC fingerprints (not plaintext values)
for deferred audit recording. Fingerprinting happens immediately after each
secret is loaded — plaintext values never leave this module.

Usage:
    from elspeth.core.config import SecretsConfig
    from elspeth.core.security.config_secrets import load_secrets_from_config

    config = SecretsConfig(
        source="keyvault",
        vault_url="https://my-vault.vault.azure.net",
        mapping={"AZURE_OPENAI_KEY": "azure-openai-key"},
    )
    resolutions = load_secrets_from_config(config)
    # Now os.environ["AZURE_OPENAI_KEY"] contains the secret value
    # resolutions contain fingerprints (not values) for audit recording
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from elspeth.contracts import SecretResolutionInput

if TYPE_CHECKING:
    from elspeth.core.config import SecretsConfig


class SecretLoadError(Exception):
    """Raised when secret loading fails.

    This error indicates a configuration or infrastructure problem that
    prevents the pipeline from starting. The error message includes
    debugging information (vault URL, secret name, env var name).
    """

    pass


def load_secrets_from_config(config: SecretsConfig) -> list[SecretResolutionInput]:
    """Load secrets from configured source and inject into environment.

    When source is 'env', this function does nothing (secrets come from
    environment variables as usual).

    When source is 'keyvault', all mapped secrets are loaded from Azure
    Key Vault and injected into os.environ, overriding any existing values.
    Each secret is fingerprinted immediately after loading — plaintext values
    never leave this function.

    Args:
        config: SecretsConfig specifying source and mapping

    Returns:
        List of resolution records for deferred audit recording.
        Each record contains: env_var_name, source, vault_url, secret_name,
        timestamp, latency_ms, fingerprint (HMAC-SHA256 hex digest).

        Plaintext secret values are NOT included in the returned records.

    Raises:
        SecretLoadError: If any secret cannot be loaded (fail fast)
    """
    if config.source == "env":
        # Nothing to do - secrets are already in environment
        return []

    # source == "keyvault"
    # Preflight check for fingerprint key before any Key Vault calls.
    # Audit recording requires ELSPETH_FINGERPRINT_KEY to compute secret fingerprints.
    # Without it, secrets would be fetched but audit recording would fail later,
    # leaving secret resolution events unrecorded (violates auditability standard).
    fingerprint_key_available = os.environ.get("ELSPETH_FINGERPRINT_KEY") or "ELSPETH_FINGERPRINT_KEY" in config.mapping
    if not fingerprint_key_available:
        raise SecretLoadError(
            "ELSPETH_FINGERPRINT_KEY is required when loading secrets from Key Vault.\n"
            "The fingerprint key is used to compute HMAC fingerprints of secrets for the audit trail.\n"
            "Fix by either:\n"
            "  1. Set ELSPETH_FINGERPRINT_KEY environment variable, or\n"
            "  2. Add ELSPETH_FINGERPRINT_KEY to your secrets mapping to load from Key Vault:\n"
            "     secrets:\n"
            "       source: keyvault\n"
            "       vault_url: https://my-vault.vault.azure.net\n"
            "       mapping:\n"
            "         ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key\n"
            "         # ... other secrets"
        )

    try:
        from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ServiceRequestError

        from elspeth.core.security.secret_loader import (
            KeyVaultSecretLoader,
            SecretNotFoundError,
        )
    except ImportError as e:
        raise SecretLoadError("Azure Key Vault packages not installed. Install with: uv pip install 'elspeth[azure]'") from e

    # Create loader (has built-in caching).
    # Note: KeyVaultSecretLoader uses lazy client initialization — the constructor
    # does no network I/O. Azure exceptions (auth, HTTP, network) are only raised
    # during get_secret() calls, where they're caught in the loop below.
    assert config.vault_url is not None, "vault_url required when source=keyvault"
    loader = KeyVaultSecretLoader(vault_url=config.vault_url)

    # Load each mapped secret, fingerprint immediately, collect resolution records.
    # Plaintext values are fingerprinted and discarded within this loop iteration —
    # they never accumulate in the resolutions list.
    from elspeth.contracts.security import get_fingerprint_key, secret_fingerprint

    resolutions: list[SecretResolutionInput] = []

    # Fingerprint key is available after loading ELSPETH_FINGERPRINT_KEY
    # (either from env or from Key Vault earlier in this loop).
    # Defer key acquisition until the first non-fingerprint-key secret.
    fingerprint_key: bytes | None = None

    # Ensure ELSPETH_FINGERPRINT_KEY is loaded first when present in mapping.
    # Without this, a mapping where ELSPETH_FINGERPRINT_KEY appears after other
    # secrets would fail: get_fingerprint_key() reads os.environ, but the Key Vault
    # secret hasn't been injected yet. User YAML ordering must not cause failures.
    _FP_KEY = "ELSPETH_FINGERPRINT_KEY"
    ordered_mapping: list[tuple[str, str]] = []
    if _FP_KEY in config.mapping:
        ordered_mapping.append((_FP_KEY, config.mapping[_FP_KEY]))
    for env_var_name, keyvault_secret_name in config.mapping.items():
        if env_var_name != _FP_KEY:
            ordered_mapping.append((env_var_name, keyvault_secret_name))

    # Phase 1: Fetch all secrets and compute fingerprints WITHOUT mutating os.environ.
    # This ensures that a failure partway through does not leave partial state.
    pending_env: list[tuple[str, str]] = []  # (env_var_name, secret_value)

    for env_var_name, keyvault_secret_name in ordered_mapping:
        timestamp = time.time()  # Wall-clock for audit record
        start_perf = time.perf_counter()  # Monotonic for latency measurement
        try:
            secret_value, _ref = loader.get_secret(keyvault_secret_name)
            latency_ms = (time.perf_counter() - start_perf) * 1000

            # Stage the env var for atomic application later
            pending_env.append((env_var_name, str(secret_value)))

            # Compute fingerprint immediately — plaintext never leaves this function.
            # Acquire fingerprint key lazily (it may have just been loaded above).
            # For ELSPETH_FINGERPRINT_KEY itself, we need to use the just-fetched value
            # since it's not in os.environ yet.
            if fingerprint_key is None:
                if env_var_name == _FP_KEY:
                    fingerprint_key = str(secret_value).encode()
                else:
                    fingerprint_key = get_fingerprint_key()
            fp = secret_fingerprint(str(secret_value), key=fingerprint_key)

            resolutions.append(
                SecretResolutionInput(
                    env_var_name=env_var_name,
                    source="keyvault",
                    vault_url=config.vault_url,
                    secret_name=keyvault_secret_name,
                    timestamp=timestamp,
                    resolution_latency_ms=latency_ms,
                    fingerprint=fp,
                )
            )

        except SecretNotFoundError as e:
            # P0-2: Catch specific exception for missing secrets
            raise SecretLoadError(
                f"Secret '{keyvault_secret_name}' not found in Key Vault ({config.vault_url})\n"
                f"Mapped from: {env_var_name}\n"
                f"Verify the secret exists: az keyvault secret show --vault-name <vault> --name {keyvault_secret_name}"
            ) from e
        except ImportError as e:
            # Azure SDK not installed
            raise SecretLoadError("Azure Key Vault packages not installed. Install with: uv pip install 'elspeth[azure]'") from e
        except ClientAuthenticationError as e:
            raise SecretLoadError(
                f"Failed to authenticate to Key Vault ({config.vault_url})\n"
                f"DefaultAzureCredential could not find valid credentials.\n"
                f"Ensure Managed Identity, Azure CLI login, or service principal env vars are configured.\n"
                f"Error: {e}"
            ) from e
        except (HttpResponseError, ServiceRequestError) as e:
            raise SecretLoadError(
                f"Failed to load secret '{keyvault_secret_name}' from Key Vault ({config.vault_url})\n"
                f"Mapped from: {env_var_name}\n"
                f"Error: {e}"
            ) from e
        except ValueError as e:
            raise SecretLoadError(
                f"Fingerprint computation failed for secret '{keyvault_secret_name}' "
                f"(env var: {env_var_name})\n"
                f"Check ELSPETH_FINGERPRINT_KEY is set and non-empty.\n"
                f"Error: {e}"
            ) from e

    # Phase 2: All secrets fetched successfully — apply to os.environ atomically.
    for env_var_name, secret_value in pending_env:
        os.environ[env_var_name] = secret_value

    return resolutions
