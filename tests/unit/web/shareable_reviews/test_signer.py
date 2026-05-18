"""Tests for ``ShareTokenSigner`` — sign/verify round-trip + tamper detection.

Phase 6A Task 3 (UX redesign 2026-05). The signer wraps the standard-library
HMAC primitive (``hmac.new(..., hashlib.sha256)``) — the same primitive backing
``core/landscape/exporter.py::_sign_record`` — but NOT the exporter's record-
signing API. The payload shape is a URL-safe self-contained token; verify uses
``hmac.compare_digest`` for constant-time comparison at an attacker-controlled
boundary (mirrors ``core/payload_store.py:111,163``, ``web/blobs/service.py:759``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from elspeth.web.shareable_reviews.signer import (
    InvalidToken,
    ShareTokenPayload,
    ShareTokenSigner,
)


def _make_payload() -> ShareTokenPayload:
    now = datetime.now(UTC)
    return ShareTokenPayload(
        version=1,
        session_id=uuid4(),
        state_id=uuid4(),
        created_at=now,
        expires_at=now + timedelta(days=7),
        nonce_hex="deadbeef" * 4,
        payload_digest="sha256:" + ("ab" * 32),
        created_by_user_id="user-1",
    )


def test_round_trip() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    payload = _make_payload()
    token = signer.sign(payload)
    decoded = signer.verify(token)
    assert decoded == payload


def test_tampered_signature_rejected() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    token = signer.sign(_make_payload())
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    with pytest.raises(InvalidToken):
        signer.verify(tampered)


def test_wrong_key_rejected() -> None:
    signer_a = ShareTokenSigner(b"a" * 32)
    signer_b = ShareTokenSigner(b"b" * 32)
    token = signer_a.sign(_make_payload())
    with pytest.raises(InvalidToken):
        signer_b.verify(token)


def test_expired_token_rejected() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    now = datetime.now(UTC)
    expired = ShareTokenPayload(
        version=1,
        session_id=uuid4(),
        state_id=uuid4(),
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(seconds=1),
        nonce_hex="ff" * 16,
        payload_digest="sha256:" + ("ab" * 32),
        created_by_user_id="user-1",
    )
    token = signer.sign(expired)
    with pytest.raises(InvalidToken, match="expired"):
        signer.verify(token)


def test_url_safe_token() -> None:
    signer = ShareTokenSigner(b"k" * 32)
    token = signer.sign(_make_payload())
    assert "+" not in token
    assert "/" not in token
    # urlsafe_b64encode uses '-' and '_' in place of '+' and '/'; '=' is the
    # padding byte. Everything else is alphanumeric.
    assert all(c.isalnum() or c in "-_=" for c in token)


def test_compare_digest_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify uses ``hmac.compare_digest`` — not ``==``.

    Constant-time comparison defeats timing-side-channel attacks on signature
    verifiers. The signer MUST call through ``hmac.compare_digest`` (not a
    locally-imported alias) so this monkeypatch sees the call. If a future
    refactor switches to ``from hmac import compare_digest`` and calls the
    unqualified name, this test fails — which is exactly the desired protection.
    """
    import hmac as hmac_mod

    calls: list[tuple[bytes, bytes]] = []
    real = hmac_mod.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac_mod, "compare_digest", spy)
    signer = ShareTokenSigner(b"k" * 32)
    token = signer.sign(_make_payload())
    signer.verify(token)
    assert calls, "ShareTokenSigner.verify must use hmac.compare_digest"


def test_verify_rejects_single_byte_tamper() -> None:
    """Behavioural test: token differing by one character is rejected."""
    signer = ShareTokenSigner(b"x" * 32)
    valid_token = signer.sign(_make_payload())
    tampered = valid_token[:-1] + ("a" if valid_token[-1] != "a" else "b")
    with pytest.raises(InvalidToken):
        signer.verify(tampered)


def test_payload_digest_in_signed_envelope() -> None:
    """payload_digest is signed — swapping it after-the-fact must reject."""
    signer = ShareTokenSigner(b"k" * 32)
    p1 = _make_payload()
    token = signer.sign(p1)
    decoded = signer.verify(token)
    assert decoded.payload_digest == p1.payload_digest


def test_signing_key_below_minimum_length_rejected() -> None:
    """Defense in depth: the signer itself rejects sub-32-byte keys.

    The WebSettings field_validator also enforces the floor, but a direct
    construction (e.g. in a unit test) must not silently accept a weak key.
    """
    with pytest.raises(ValueError, match="at least 32 bytes"):
        ShareTokenSigner(b"too-short")


def test_truncated_token_rejected() -> None:
    """Truncating a token below header + signature size is detected."""
    signer = ShareTokenSigner(b"k" * 32)
    with pytest.raises(InvalidToken):
        signer.verify("a")


def test_malformed_base64_rejected() -> None:
    """A non-base64 string fails decode cleanly."""
    signer = ShareTokenSigner(b"k" * 32)
    with pytest.raises(InvalidToken):
        signer.verify("!!!not-base64!!!")


def test_payload_with_unknown_version_rejected() -> None:
    """A signed payload with a future version must reject (defense against
    forward-compat bypass attempts)."""
    signer = ShareTokenSigner(b"k" * 32)
    now = datetime.now(UTC)
    p = ShareTokenPayload(
        version=99,
        session_id=uuid4(),
        state_id=uuid4(),
        created_at=now,
        expires_at=now + timedelta(days=7),
        nonce_hex="ff" * 16,
        payload_digest="sha256:" + ("ab" * 32),
        created_by_user_id="user-1",
    )
    token = signer.sign(p)
    with pytest.raises(InvalidToken, match="unsupported version"):
        signer.verify(token)
