# tests/property/core/test_fingerprint_properties.py
"""Property-based tests for secret fingerprinting (HMAC-SHA256).

These tests verify the fundamental cryptographic properties of ELSPETH's
secret fingerprinting system:

Cryptographic Properties:
- Determinism: Same inputs always produce same output
- Fixed output length: Always 64 hex characters (SHA256)
- Collision resistance: Different inputs produce different outputs
- Key sensitivity: Different keys produce different outputs

Format Properties:
- Output is valid hexadecimal
- Output is lowercase

These invariants are critical for audit integrity - fingerprints prove
"the same secret was used" without revealing the secret itself.
"""

from __future__ import annotations

import string

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.security.fingerprint import secret_fingerprint

# =============================================================================
# Strategies for generating secrets and keys
# =============================================================================

# Secret strings - can be anything (API keys, tokens, passwords)
secrets = st.text(min_size=0, max_size=200)

# Keys - non-empty byte sequences
keys = st.binary(min_size=1, max_size=64)

# Non-empty secrets for collision tests
non_empty_secrets = st.text(min_size=1, max_size=200)

# Printable ASCII secrets (common for API keys)
ascii_secrets = st.text(
    min_size=1,
    max_size=100,
    alphabet=string.ascii_letters + string.digits + "-_.",
)


# =============================================================================
# Determinism Property Tests
# =============================================================================


class TestFingerprintDeterminismProperties:
    """Property tests for HMAC determinism."""

    @given(secret=secrets, key=keys)
    @settings(max_examples=200)
    def test_same_inputs_same_output(self, secret: str, key: bytes) -> None:
        """Property: secret_fingerprint(s, k) == secret_fingerprint(s, k) always.

        HMAC is deterministic - this is the fundamental property that allows
        fingerprints to verify "same secret was used" across runs.
        """
        fp1 = secret_fingerprint(secret, key=key)
        fp2 = secret_fingerprint(secret, key=key)

        assert fp1 == fp2, f"Fingerprint not deterministic: '{fp1}' != '{fp2}'"

    @given(secret=secrets, key=keys)
    @settings(max_examples=100)
    def test_repeated_calls_are_idempotent(self, secret: str, key: bytes) -> None:
        """Property: Multiple calls with same inputs produce identical results.

        This verifies there's no internal state that affects output.
        """
        results = [secret_fingerprint(secret, key=key) for _ in range(5)]
        assert all(r == results[0] for r in results), f"Results varied: {results}"

    @given(secret=non_empty_secrets, key=keys)
    @settings(max_examples=100)
    def test_fingerprint_independent_of_call_order(self, secret: str, key: bytes) -> None:
        """Property: Fingerprinting A then B gives same result as B then A.

        This verifies no cross-contamination between fingerprint operations.
        """
        other_secret = secret + "_other"

        # Compute in one order
        fp_a1 = secret_fingerprint(secret, key=key)
        _ = secret_fingerprint(other_secret, key=key)
        fp_a2 = secret_fingerprint(secret, key=key)

        assert fp_a1 == fp_a2, "Fingerprint affected by intervening calls"


# =============================================================================
# Output Format Property Tests
# =============================================================================


class TestFingerprintFormatProperties:
    """Property tests for output format invariants."""

    @given(secret=secrets, key=keys)
    @settings(max_examples=200)
    def test_output_length_is_64_characters(self, secret: str, key: bytes) -> None:
        """Property: Output is always exactly 64 characters (SHA256 hex).

        SHA256 produces 256 bits = 32 bytes = 64 hex characters.
        This is a cryptographic constant that never varies.
        """
        fp = secret_fingerprint(secret, key=key)
        assert len(fp) == 64, f"Expected 64 chars, got {len(fp)}: '{fp}'"

    @given(secret=secrets, key=keys)
    @settings(max_examples=200)
    def test_output_is_valid_hex(self, secret: str, key: bytes) -> None:
        """Property: Output contains only hexadecimal characters [0-9a-f].

        This ensures the fingerprint can be safely stored in any database
        column type (varchar, text) without encoding issues.
        """
        fp = secret_fingerprint(secret, key=key)
        valid_hex_chars = set("0123456789abcdef")

        invalid_chars = set(fp) - valid_hex_chars
        assert not invalid_chars, f"Invalid hex chars in fingerprint: {invalid_chars}"

    @given(secret=secrets, key=keys)
    @settings(max_examples=200)
    def test_output_is_lowercase(self, secret: str, key: bytes) -> None:
        """Property: Output is lowercase hex (no uppercase A-F).

        ELSPETH convention: all hex values are lowercase for consistent
        string comparison and database indexing.
        """
        fp = secret_fingerprint(secret, key=key)
        assert fp == fp.lower(), f"Fingerprint contains uppercase: '{fp}'"

    @given(secret=secrets, key=keys)
    @settings(max_examples=100)
    def test_output_is_string_type(self, secret: str, key: bytes) -> None:
        """Property: Output is always a str, not bytes.

        This ensures consistent handling in JSON serialization and database storage.
        """
        fp = secret_fingerprint(secret, key=key)
        assert isinstance(fp, str), f"Expected str, got {type(fp)}"


# =============================================================================
# Collision Resistance Property Tests
# =============================================================================


class TestFingerprintCollisionResistanceProperties:
    """Property tests for collision resistance."""

    @given(secret1=non_empty_secrets, secret2=non_empty_secrets, key=keys)
    @settings(max_examples=200)
    def test_different_secrets_different_fingerprints(self, secret1: str, secret2: str, key: bytes) -> None:
        """Property: Different secrets produce different fingerprints (same key).

        This is the core security property - an attacker cannot find two
        secrets that produce the same fingerprint (collision resistance).

        Note: We use assume() to skip when secrets happen to be equal,
        since that's a different property (determinism).
        """
        assume(secret1 != secret2)

        fp1 = secret_fingerprint(secret1, key=key)
        fp2 = secret_fingerprint(secret2, key=key)

        assert fp1 != fp2, f"Collision found! '{secret1}' and '{secret2}' both produce '{fp1}'"

    @given(secret=non_empty_secrets, key1=keys, key2=keys)
    @settings(max_examples=200)
    def test_different_keys_different_fingerprints(self, secret: str, key1: bytes, key2: bytes) -> None:
        """Property: Same secret with different keys produces different fingerprints.

        This is the key sensitivity property - changing the HMAC key
        completely changes the output. Critical for key rotation scenarios.
        """
        assume(key1 != key2)

        fp1 = secret_fingerprint(secret, key=key1)
        fp2 = secret_fingerprint(secret, key=key2)

        assert fp1 != fp2, f"Key insensitivity! Secret '{secret}' produces same fingerprint '{fp1}' with different keys"


# =============================================================================
# Edge Case Property Tests
# =============================================================================


class TestFingerprintEdgeCaseProperties:
    """Property tests for edge cases and boundary conditions."""

    @given(key=keys)
    @settings(max_examples=50)
    def test_empty_secret_produces_valid_fingerprint(self, key: bytes) -> None:
        """Property: Empty string secret produces valid 64-char hex fingerprint.

        Empty secrets are valid inputs (though unusual). The fingerprint
        should still be deterministic and properly formatted.
        """
        fp = secret_fingerprint("", key=key)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    @given(key=keys)
    @settings(max_examples=50)
    def test_empty_secret_is_deterministic(self, key: bytes) -> None:
        """Property: Empty string produces same fingerprint on repeated calls."""
        fp1 = secret_fingerprint("", key=key)
        fp2 = secret_fingerprint("", key=key)

        assert fp1 == fp2

    @given(secret=secrets)
    @settings(max_examples=50)
    def test_single_byte_key_works(self, secret: str) -> None:
        """Property: Minimum-length key (1 byte) produces valid fingerprint.

        HMAC handles short keys by zero-padding to block size internally.
        """
        key = b"x"
        fp = secret_fingerprint(secret, key=key)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    @given(secret=secrets)
    @settings(max_examples=50)
    def test_long_key_works(self, secret: str) -> None:
        """Property: Long keys (> SHA256 block size) produce valid fingerprints.

        HMAC handles long keys by hashing them first. 64 bytes is the
        SHA256 block size, so 128 bytes triggers the long-key path.
        """
        key = b"x" * 128  # > 64 byte block size
        fp = secret_fingerprint(secret, key=key)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# =============================================================================
# Unicode Handling Property Tests
# =============================================================================


class TestFingerprintUnicodeProperties:
    """Property tests for Unicode secret handling."""

    @given(
        secret=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
        ),
        key=keys,
    )
    @settings(max_examples=100)
    def test_unicode_secrets_produce_deterministic_fingerprints(self, secret: str, key: bytes) -> None:
        """Property: Unicode secrets (emoji, CJK, etc.) fingerprint deterministically.

        Secrets may contain international characters. The fingerprint must
        be deterministic regardless of Unicode normalization form.
        """
        fp1 = secret_fingerprint(secret, key=key)
        fp2 = secret_fingerprint(secret, key=key)

        assert fp1 == fp2
        assert len(fp1) == 64

    @given(key=keys)
    @settings(max_examples=20)
    def test_emoji_secrets_work(self, key: bytes) -> None:
        """Property: Emoji in secrets produce valid fingerprints.

        Real-world edge case: some systems use emoji in tokens.
        """
        secret = "api-key-🔑-secret-🔒"
        fp = secret_fingerprint(secret, key=key)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    @given(key=keys)
    @settings(max_examples=20)
    def test_cjk_secrets_work(self, key: bytes) -> None:
        """Property: CJK characters in secrets produce valid fingerprints."""
        secret = "密码-パスワード-비밀번호"
        fp = secret_fingerprint(secret, key=key)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)
