"""Tests for ``shareable_link_signing_key`` and ``shareable_link_lifetime_seconds`` on WebSettings.

Phase 6A Task 2 (UX redesign 2026-05). See plan
``docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md`` §"Task 2".

The signing key is a required, no-default Pydantic field. The web service
refuses to start without it, mirroring the operator gate at the top of the
plan ("Staging config ``shareable_link_signing_key`` (B9)"). Tests pass the
four other required composer fields explicitly so the only failure mode
under test is the new key/lifetime fields.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretBytes, ValidationError

from elspeth.web.config import WebSettings

_VALID_32_BYTE_KEY = b"0" * 32

# A 32-byte high-entropy key. Distinct from ``_VALID_32_BYTE_KEY`` because the
# weak-key validator (see ``WebSettings._reject_known_weak_signing_key``) treats
# uniform-byte placeholders like ``b"0" * 32`` as known-weak on non-loopback
# hosts. Tests that exercise non-loopback hosts MUST use this value.
_VALID_NON_WEAK_KEY = b"\xab\xcd" * 16


def _other_required_kwargs() -> dict[str, Any]:
    """Required-but-unrelated composer fields. Mirrors tests/unit/web/test_config.py."""
    return {
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
    }


def test_signing_key_minimum_length() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        WebSettings(
            shareable_link_signing_key=b"too-short",
            **_other_required_kwargs(),
        )


def test_signing_key_accepts_32_bytes() -> None:
    settings = WebSettings(
        shareable_link_signing_key=_VALID_32_BYTE_KEY,
        **_other_required_kwargs(),
    )
    # SecretBytes wraps the raw key; ``.get_secret_value()`` unwraps for comparison.
    assert settings.shareable_link_signing_key.get_secret_value() == _VALID_32_BYTE_KEY


def test_signing_key_accepts_longer_keys() -> None:
    """Pydantic bytes coercion: longer keys (e.g. 44-char base64 output) accepted."""
    key = b"x" * 64
    settings = WebSettings(
        shareable_link_signing_key=key,
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_signing_key.get_secret_value() == key


def test_signing_key_required() -> None:
    """Field(...) — no default. Even when other required fields are present,
    a WebSettings construction without ``shareable_link_signing_key`` raises."""
    with pytest.raises(ValidationError) as exc_info:
        WebSettings(**_other_required_kwargs())
    # The validation error explicitly identifies the missing field by name.
    assert "shareable_link_signing_key" in str(exc_info.value)


def test_signing_key_lifetime_default_30_days() -> None:
    settings = WebSettings(
        shareable_link_signing_key=_VALID_32_BYTE_KEY,
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_lifetime_seconds == 30 * 24 * 3600


def test_signing_key_lifetime_override() -> None:
    settings = WebSettings(
        shareable_link_signing_key=_VALID_32_BYTE_KEY,
        shareable_link_lifetime_seconds=3600,
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_lifetime_seconds == 3600


def test_signing_key_lifetime_rejects_zero() -> None:
    """Lifetime must be > 0 (gt=0). Zero or negative makes no sense."""
    with pytest.raises(ValidationError):
        WebSettings(
            shareable_link_signing_key=_VALID_32_BYTE_KEY,
            shareable_link_lifetime_seconds=0,
            **_other_required_kwargs(),
        )


def test_signing_key_lifetime_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        WebSettings(
            shareable_link_signing_key=_VALID_32_BYTE_KEY,
            shareable_link_lifetime_seconds=-1,
            **_other_required_kwargs(),
        )


# --- DC-2 FIX-L: SecretBytes + strict + weak-key validator -----------------
#
# Phase 6A gap-analysis (independently re-tested by a fresh reviewer) flagged
# three exfiltration / weakening paths around the signing key:
#   1. [HIGH]   Pydantic v2 default repr leaks the raw bytes value.
#   2. [MEDIUM] Lax mode coerces ``str → bytes`` via utf-8, breaking the
#               32-character-equivalent entropy floor for multibyte inputs
#               (e.g. ``"a" * 30 + "ñ"`` → 32 utf-8 bytes from 31 characters).
#   3. [LOW]    No CI/staging guard against uniform-byte placeholder keys
#               (b'\x00' * 32, b'0' * 32, b'1' * 32) on non-loopback hosts.


def test_signing_key_not_leaked_via_repr() -> None:
    """DC-2 [HIGH] — ``repr(WebSettings(...))`` must NOT contain raw key bytes.

    Pydantic v2's default repr lists every field value. Wrapping the field in
    ``SecretBytes`` masks the value (``b'**********'``) so tracebacks, debug
    logs, and REPL inspection do not exfiltrate the HMAC key.
    """
    distinctive_key = b"\xde\xad\xbe\xef" * 8  # 32 bytes, recognisable pattern
    settings = WebSettings(
        shareable_link_signing_key=distinctive_key,
        **_other_required_kwargs(),
    )
    rendered = repr(settings)
    # The raw bytes pattern must not appear.
    assert b"\xde\xad\xbe\xef".hex() not in rendered.lower()
    assert "deadbeef" not in rendered.lower()
    # And neither should the literal repr of the bytes object.
    assert repr(distinctive_key) not in rendered
    # The mask should appear in its place.
    assert "**********" in rendered


def test_signing_key_rejects_utf8_multibyte_coercion() -> None:
    """DC-2 [MEDIUM] — multibyte-utf-8 ambiguity is foreclosed at the boundary.

    Without strict mode + explicit base64 decoding, Pydantic would coerce
    ``str → bytes`` via utf-8. A 31-character string containing a 2-byte
    codepoint (``'a' * 30 + 'ñ'``) encodes to 32 utf-8 bytes and slips past
    the byte-length floor with only 31 characters of entropy.

    Our resolution: str inputs go through explicit base64 decoding (the same
    encoding used by the documented ``openssl rand -base64 32`` recipe).
    Non-ASCII strings are rejected by ``base64.b64decode(validate=True)``.
    """
    with pytest.raises(ValidationError, match="base64"):
        WebSettings(
            shareable_link_signing_key="a" * 30 + "ñ",  # type: ignore[arg-type]
            **_other_required_kwargs(),
        )


def test_signing_key_accepts_base64_encoded_string_from_env_var() -> None:
    """Env-var ingestion path: ``openssl rand -base64 32`` output passes.

    The 44-char base64 string is the documented operator recipe (see
    ``docs/guides/sharing-pipelines.md``). It MUST decode to 32 raw bytes
    via explicit base64 decoding — no fall-through to utf-8 coercion.
    """
    import base64

    raw_key = b"\xab\xcd" * 16  # 32 random-ish bytes
    base64_str = base64.b64encode(raw_key).decode("ascii")  # 44 chars
    assert len(base64_str) == 44
    settings = WebSettings(
        shareable_link_signing_key=base64_str,  # type: ignore[arg-type]
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_signing_key.get_secret_value() == raw_key


def test_signing_key_rejects_short_base64_decoded_value() -> None:
    """A base64-decodable str whose decoded bytes are < 32 still fails the floor.

    Belt-and-braces: even if the operator base64-encodes 16 random bytes
    instead of 32, the min-length validator catches it.
    """
    import base64

    short_key = b"x" * 16  # 16 bytes — below the floor
    base64_str = base64.b64encode(short_key).decode("ascii")
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        WebSettings(
            shareable_link_signing_key=base64_str,  # type: ignore[arg-type]
            **_other_required_kwargs(),
        )


def test_signing_key_validator_rejects_known_weak_key_on_external_host() -> None:
    """DC-2 [LOW] — known-weak placeholder keys are rejected on non-loopback hosts.

    Mirrors ``_enforce_secret_key_in_production`` (config.py:388). Test
    fixtures use ``b'\\x00' * 32`` / ``b'0' * 32`` as convenient placeholders;
    those are operationally indistinguishable from "the operator forgot to
    generate a real key." On a non-loopback host the model_validator refuses.
    """
    for weak_key in (b"\x00" * 32, b"0" * 32, b"1" * 32):
        with pytest.raises(ValidationError, match="known-weak"):
            WebSettings(
                host="external.example.com",
                shareable_link_signing_key=weak_key,
                secret_key="non-default-secret-for-non-loopback-test",
                **_other_required_kwargs(),
            )


def test_signing_key_validator_allows_weak_key_on_loopback() -> None:
    """The validator only fires on non-loopback hosts — local dev unaffected."""
    # Default host=127.0.0.1; ``b'\\x00' * 32`` accepted (matches fixture usage).
    settings = WebSettings(
        shareable_link_signing_key=b"\x00" * 32,
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_signing_key.get_secret_value() == b"\x00" * 32


def test_signing_key_validator_allows_high_entropy_key_on_external_host() -> None:
    """A high-entropy key (non-uniform bytes) is accepted on non-loopback hosts."""
    settings = WebSettings(
        host="external.example.com",
        shareable_link_signing_key=_VALID_NON_WEAK_KEY,
        secret_key="non-default-secret-for-non-loopback-test",
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_signing_key.get_secret_value() == _VALID_NON_WEAK_KEY


def test_signing_key_field_type_is_secret_bytes() -> None:
    """Defense-in-depth: confirm the field type is ``SecretBytes``, not raw bytes.

    Type assertion guards against accidental rollback to plaintext bytes in
    future refactors. The ``WebSettings`` field annotation IS the contract.
    """
    settings = WebSettings(
        shareable_link_signing_key=_VALID_32_BYTE_KEY,
        **_other_required_kwargs(),
    )
    assert isinstance(settings.shareable_link_signing_key, SecretBytes)
