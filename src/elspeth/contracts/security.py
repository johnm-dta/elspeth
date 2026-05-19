"""Secret fingerprinting primitives.

These are stdlib-only (hashlib, os) and belong in the contracts layer
because they define the fingerprinting contract used across all layers.

Secrets (API keys, tokens, passwords) should never appear in the audit trail.
Instead, we store a fingerprint that can verify "same secret was used"
without revealing the actual secret value.

Usage:
    from elspeth.contracts.security import secret_fingerprint

    # With explicit key
    fp = secret_fingerprint(api_key, key=signing_key)

    # With environment variable (ELSPETH_FINGERPRINT_KEY)
    fp = secret_fingerprint(api_key)

Note: ELSPETH_FINGERPRINT_KEY can be set directly or loaded from Azure Key Vault
via the secrets configuration in your pipeline YAML.
"""

from __future__ import annotations

import hashlib
import os

_ENV_VAR = "ELSPETH_FINGERPRINT_KEY"
_BLAKE2B_KEY_SIZE = 64
_FINGERPRINT_PERSON = b"elspeth-sec-fp"
_FINGERPRINT_KEY_PERSON = b"elspeth-fp-key"


class SecretFingerprintError(Exception):
    """Raised when secret fingerprinting fails.

    This occurs when:
    - Secret-like field names are found in config but ELSPETH_FINGERPRINT_KEY
      is not set and ELSPETH_ALLOW_RAW_SECRETS is not 'true'
    - A config dict contains both a secret field (e.g., 'api_key') and the
      corresponding fingerprint field ('api_key_fingerprint'), which would
      allow the pre-existing value to overwrite the computed keyed fingerprint
    """

    pass


def get_fingerprint_key() -> bytes:
    """Get the fingerprint key from environment.

    The fingerprint key should be set via:
    1. Direct environment variable: ELSPETH_FINGERPRINT_KEY=your-key
    2. Pipeline secrets config loading from Azure Key Vault

    Returns:
        The fingerprint key as bytes

    Raises:
        ValueError: If ELSPETH_FINGERPRINT_KEY is not set
    """
    env_key = os.environ.get(_ENV_VAR)
    if not env_key:
        raise ValueError(
            f"Fingerprint key not configured. Set {_ENV_VAR} environment variable "
            f"or configure it in your pipeline's secrets section to load from Azure Key Vault."
        )
    return env_key.encode("utf-8")


def secret_fingerprint(secret: str, *, key: bytes | None = None) -> str:
    """Compute a keyed BLAKE2b fingerprint of a secret.

    The fingerprint can be stored in the audit trail to verify that
    the same secret was used across runs, without exposing the secret.

    Args:
        secret: The secret value to fingerprint (API key, token, etc.)
        key: Fingerprint key. If not provided, reads from ELSPETH_FINGERPRINT_KEY env var.

    Returns:
        64-character hex string (32-byte digest)

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

    if len(key) == 0:
        raise ValueError("Fingerprint key must not be empty — an empty keyed hash key produces meaningless fingerprints")

    blake2_key = key
    if len(blake2_key) > _BLAKE2B_KEY_SIZE:
        blake2_key = hashlib.blake2b(blake2_key, digest_size=_BLAKE2B_KEY_SIZE, person=_FINGERPRINT_KEY_PERSON).digest()

    digest = hashlib.blake2b(
        secret.encode("utf-8"),
        digest_size=32,
        key=blake2_key,
        person=_FINGERPRINT_PERSON,
    ).hexdigest()

    return digest
