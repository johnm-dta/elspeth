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
from pydantic import ValidationError

from elspeth.web.config import WebSettings

_VALID_32_BYTE_KEY = b"0" * 32


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
    assert settings.shareable_link_signing_key == _VALID_32_BYTE_KEY


def test_signing_key_accepts_longer_keys() -> None:
    """Pydantic bytes coercion: longer keys (e.g. 44-char base64 output) accepted."""
    key = b"x" * 64
    settings = WebSettings(
        shareable_link_signing_key=key,
        **_other_required_kwargs(),
    )
    assert settings.shareable_link_signing_key == key


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
