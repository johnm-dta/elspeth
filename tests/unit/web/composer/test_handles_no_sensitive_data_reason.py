"""Tests for HandlesNoSensitiveDataReason construction-time validators (spec §4.2.3).

Six test cases covering the validators added in Task 5:
  1. Empty sensitive_data_locations raises ValueError with explicit guidance.
  2. why_arguments_safe shorter than 32 chars raises ValueError with guidance.
  3. why_responses_safe shorter than 32 chars raises ValueError with guidance.
  4. Whitespace-only why_arguments_safe (len >= 32, stripped len == 0) is rejected.
  5. Frozen container: sensitive_data_locations is tuple after construction.
  6. Identity-preserving idempotency: second __post_init__ call is a no-op.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.redaction import HandlesNoSensitiveDataReason

# ---------------------------------------------------------------------------
# Helper — a fully valid instance (used in tests 5 and 6).
# ---------------------------------------------------------------------------

_VALID_LOCATIONS = ("no LLM-supplied inputs",)
_VALID_ARGS_REASON = "A" * 32
_VALID_RESP_REASON = "B" * 32


def _ok() -> HandlesNoSensitiveDataReason:
    """Construct a valid HandlesNoSensitiveDataReason for structural tests."""
    return HandlesNoSensitiveDataReason(
        sensitive_data_locations=_VALID_LOCATIONS,
        why_arguments_safe=_VALID_ARGS_REASON,
        why_responses_safe=_VALID_RESP_REASON,
    )


# ---------------------------------------------------------------------------
# Test 1: empty sensitive_data_locations
# ---------------------------------------------------------------------------


def test_empty_sensitive_data_locations_raises_value_error() -> None:
    """Empty tuple is rejected; error message names the field and explains the fix."""
    with pytest.raises(ValueError, match="sensitive_data_locations"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=(),
            why_arguments_safe=_VALID_ARGS_REASON,
            why_responses_safe=_VALID_RESP_REASON,
        )


def test_empty_sensitive_data_locations_error_contains_guidance() -> None:
    """Error message must direct operator to a remediation (not just say 'invalid')."""
    with pytest.raises(ValueError, match="at least one"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=(),
            why_arguments_safe=_VALID_ARGS_REASON,
            why_responses_safe=_VALID_RESP_REASON,
        )


# ---------------------------------------------------------------------------
# Test 2: why_arguments_safe shorter than 32 chars
# ---------------------------------------------------------------------------


def test_why_arguments_safe_too_short_raises_value_error() -> None:
    """31 non-whitespace characters in why_arguments_safe is rejected."""
    with pytest.raises(ValueError, match="why_arguments_safe"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=_VALID_LOCATIONS,
            why_arguments_safe="A" * 31,
            why_responses_safe=_VALID_RESP_REASON,
        )


def test_why_arguments_safe_exactly_32_chars_is_accepted() -> None:
    """32 characters is the minimum; the boundary value must pass (< 32, not <= 32)."""
    instance = HandlesNoSensitiveDataReason(
        sensitive_data_locations=_VALID_LOCATIONS,
        why_arguments_safe="A" * 32,
        why_responses_safe=_VALID_RESP_REASON,
    )
    assert len(instance.why_arguments_safe) == 32


# ---------------------------------------------------------------------------
# Test 3: why_responses_safe shorter than 32 chars
# ---------------------------------------------------------------------------


def test_why_responses_safe_too_short_raises_value_error() -> None:
    """31 non-whitespace characters in why_responses_safe is rejected."""
    with pytest.raises(ValueError, match="why_responses_safe"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=_VALID_LOCATIONS,
            why_arguments_safe=_VALID_ARGS_REASON,
            why_responses_safe="B" * 31,
        )


def test_why_responses_safe_exactly_32_chars_is_accepted() -> None:
    """32 characters is the minimum; the boundary value must pass (< 32, not <= 32)."""
    instance = HandlesNoSensitiveDataReason(
        sensitive_data_locations=_VALID_LOCATIONS,
        why_arguments_safe=_VALID_ARGS_REASON,
        why_responses_safe="B" * 32,
    )
    assert len(instance.why_responses_safe) == 32


# ---------------------------------------------------------------------------
# Test 4: whitespace-only string >= 32 chars is rejected (.strip() semantics)
# ---------------------------------------------------------------------------


def test_whitespace_only_why_arguments_safe_is_rejected() -> None:
    """A string of 33 spaces strips to empty (len 0), which is < 32; must be rejected."""
    whitespace_33 = " " * 33
    assert len(whitespace_33) >= 32, "precondition: raw length is >= 32"
    assert len(whitespace_33.strip()) == 0, "precondition: stripped length is 0"
    with pytest.raises(ValueError, match="why_arguments_safe"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=_VALID_LOCATIONS,
            why_arguments_safe=whitespace_33,
            why_responses_safe=_VALID_RESP_REASON,
        )


def test_whitespace_only_why_responses_safe_is_rejected() -> None:
    """A string of 33 spaces strips to empty (len 0), which is < 32; must be rejected."""
    whitespace_33 = " " * 33
    with pytest.raises(ValueError, match="why_responses_safe"):
        HandlesNoSensitiveDataReason(
            sensitive_data_locations=_VALID_LOCATIONS,
            why_arguments_safe=_VALID_ARGS_REASON,
            why_responses_safe=whitespace_33,
        )


# ---------------------------------------------------------------------------
# Test 5: frozen container — sensitive_data_locations is tuple after construction
# ---------------------------------------------------------------------------


def test_sensitive_data_locations_is_tuple_after_construction() -> None:
    """freeze_fields converts the container; post-construction type must be tuple."""
    instance = _ok()
    assert isinstance(instance.sensitive_data_locations, tuple)


def test_sensitive_data_locations_values_are_preserved() -> None:
    """Freeze-guard must not alter the contents of sensitive_data_locations."""
    instance = HandlesNoSensitiveDataReason(
        sensitive_data_locations=("response.body", "request.headers"),
        why_arguments_safe=_VALID_ARGS_REASON,
        why_responses_safe=_VALID_RESP_REASON,
    )
    assert instance.sensitive_data_locations == ("response.body", "request.headers")


# ---------------------------------------------------------------------------
# Test 6: identity-preserving idempotency — second __post_init__ call is a no-op
# ---------------------------------------------------------------------------


def test_post_init_idempotency_no_exception() -> None:
    """Calling __post_init__ a second time on an already-frozen instance must not raise."""
    instance = _ok()
    # freeze_fields is identity-preserving idempotent: it skips object.__setattr__
    # when the field is already frozen (already a tuple). The validators re-run
    # harmlessly because field values have not changed.
    instance.__post_init__()  # must not raise


def test_post_init_idempotency_field_identity_preserved() -> None:
    """After a second __post_init__, field values must be unchanged (identity stable)."""
    instance = _ok()
    locations_before = instance.sensitive_data_locations
    args_reason_before = instance.why_arguments_safe
    resp_reason_before = instance.why_responses_safe

    instance.__post_init__()

    assert instance.sensitive_data_locations is locations_before
    assert instance.why_arguments_safe is args_reason_before
    assert instance.why_responses_safe is resp_reason_before
