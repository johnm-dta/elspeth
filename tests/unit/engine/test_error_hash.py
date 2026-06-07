"""Tests for the empty-safe audit error_hash helper (elspeth-501c14847b)."""

from __future__ import annotations

import hashlib
import re

from elspeth.engine._error_hash import compute_error_hash

_EMPTY_SHA = hashlib.sha256(b"").hexdigest()[:16]  # e3b0c44298fc1c14


def test_empty_message_does_not_collide_with_constant_and_is_type_distinguished() -> None:
    """The real trigger: a message-less exception yields str(e) == '' (e.g.
    ValueError()), which previously hashed to the constant sha256('') prefix for
    every such error. The helper must avoid that constant and distinguish errors
    by exception type."""
    empty = str(ValueError())
    assert empty == ""  # the real-world empty-message source

    h_value_err = compute_error_hash(empty, exception_type="ValueError")
    h_key_err = compute_error_hash(empty, exception_type="KeyError")

    assert h_value_err != _EMPTY_SHA  # no longer collides into the constant
    assert h_key_err != _EMPTY_SHA
    assert h_value_err != h_key_err  # distinguished by exception type
    assert re.fullmatch(r"[0-9a-f]{16}", h_value_err)


def test_empty_message_without_type_is_stable_but_non_constant() -> None:
    h = compute_error_hash("")
    assert h != _EMPTY_SHA
    assert h == compute_error_hash("")  # deterministic
    assert re.fullmatch(r"[0-9a-f]{16}", h)


def test_non_empty_message_is_byte_identical_to_inline_hash() -> None:
    """Critical no-regression invariant: every non-empty hash is unchanged, so no
    existing audit hash moves and no fingerprint-baseline reconciliation is needed."""
    for msg in ["TransformError: boom", "late_arrival_after_merge", "quarantined_in_batch:b1:0"]:
        inline = hashlib.sha256(msg.encode()).hexdigest()[:16]
        assert compute_error_hash(msg) == inline
        # exception_type is ignored for non-empty messages (does not perturb the hash).
        assert compute_error_hash(msg, exception_type="X") == inline
