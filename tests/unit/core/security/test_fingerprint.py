# tests/core/security/test_fingerprint.py
"""Tests for secret fingerprinting."""

import pytest

from elspeth.contracts.security import get_fingerprint_key, secret_fingerprint


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
        """The keyed digest is 64 hex characters."""
        result = secret_fingerprint("test", key=b"key")
        assert len(result) == 64

    def test_fingerprint_golden_vector(self) -> None:
        """Verify the PBKDF2-HMAC-SHA256 algorithm with a known test vector.

        This locks the algorithm to PBKDF2-HMAC-SHA256. If the implementation
        changes to an unkeyed hash or another algorithm, this test will fail.
        """
        result = secret_fingerprint("my-secret", key=b"test-key")

        # Precomputed with PBKDF2-HMAC-SHA256, 210000 iterations, dklen=32.
        expected = "e50795d435c7dc95af053d8112410b16ccc7f2b82df4ede8022fc47d5f618d5e"
        assert result == expected

    def test_fingerprint_without_key_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When key not provided, uses ELSPETH_FINGERPRINT_KEY env var."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-key-value")

        result = secret_fingerprint("my-secret")

        # Verify the env key is actually used by checking against expected PBKDF2.
        expected = "ff4d30a6dfcf2c96442d555f620dcb7a57fc5f508672b02c11098d467a43c4cf"
        assert result == expected

    def test_fingerprint_without_key_raises_if_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError if no key provided and env var missing."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

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

        with pytest.raises(ValueError):
            get_fingerprint_key()
