"""Tests for shared audit hash assertion helpers."""

from __future__ import annotations

import hashlib
import hmac

import pytest
from tests.fixtures.audit_hashing import (
    assert_hmac_sha256_hex,
    assert_prefixed_canonical_sha256,
    assert_stable_hash,
    canonical_sha256_hex,
)

from elspeth.core.canonical import canonical_json, stable_hash


def test_assert_stable_hash_binds_to_exact_payload() -> None:
    payload = {"b": 2, "a": 1}

    assert_stable_hash(stable_hash(payload), payload)

    with pytest.raises(AssertionError):
        assert_stable_hash(stable_hash({"a": 1}), payload)


def test_assert_prefixed_canonical_sha256_binds_to_canonical_json() -> None:
    payload = {"b": 2, "a": 1}
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    assert canonical_sha256_hex(payload) == expected
    assert_prefixed_canonical_sha256(f"sha256:{expected}", payload)

    with pytest.raises(AssertionError):
        assert_prefixed_canonical_sha256(f"sha256:{'0' * 64}", payload)


def test_assert_hmac_sha256_hex_binds_to_canonical_json() -> None:
    payload = {"record": {"b": 2, "a": 1}}
    key = b"test-signing-key"
    expected = hmac.new(key, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()

    assert_hmac_sha256_hex(expected, key, payload)

    with pytest.raises(AssertionError):
        assert_hmac_sha256_hex("0" * 64, key, payload)
