# tests/property/contracts/test_validation_rejection_properties.py
"""Property-based tests for CONTRACT boundary validation rejection.

This module focuses on validation rejection as CONTRACT boundaries:

1. **Decimal rejection** - Non-finite Decimal values (NaN, Infinity, sNaN)
2. **Enum rejection** - Invalid enum values for audit trail integrity
3. **Config rejection** - Invalid configuration at load time (Tier 3 boundary)

These tests verify that ELSPETH correctly REJECTS invalid inputs at boundaries
where data enters the system. This is critical for audit integrity.

NOTE: Float and NumPy non-finite rejection is comprehensively tested in:
    tests/property/canonical/test_nan_rejection.py

This file complements that coverage with enum and config contract validation,
and extends Decimal coverage to include signaling NaN (sNaN) and nested cases.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.canonical import canonical_json, stable_hash

# =============================================================================
# Strategies for Non-Finite Decimal Values
# =============================================================================

# Non-finite Decimal values (unique to this file - not in test_nan_rejection.py)
non_finite_decimals = st.sampled_from(
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("sNaN"),  # Signaling NaN
    ]
)


# =============================================================================
# Decimal Rejection Property Tests
# =============================================================================


class TestDecimalRejectionProperties:
    """Property tests verifying non-finite Decimal values are rejected.

    The Decimal type has special non-finite values that must be caught:
    - Decimal("NaN") - quiet NaN
    - Decimal("sNaN") - signaling NaN
    - Decimal("Infinity") - positive infinity
    - Decimal("-Infinity") - negative infinity

    All of these would corrupt audit trail integrity if allowed through.

    NOTE: This extends float/numpy coverage in test_nan_rejection.py and
    explicitly includes Decimal("sNaN") edge cases.
    """

    @given(value=non_finite_decimals)
    @settings(max_examples=20)
    def test_canonical_json_rejects_non_finite_decimals(self, value: Decimal) -> None:
        """Property: canonical_json() rejects non-finite Decimal values."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"value": value})

    @given(value=non_finite_decimals)
    @settings(max_examples=20)
    def test_stable_hash_rejects_non_finite_decimals(self, value: Decimal) -> None:
        """Property: stable_hash() rejects non-finite Decimal values."""
        with pytest.raises(ValueError, match="non-finite"):
            stable_hash({"value": value})

    @given(value=non_finite_decimals)
    @settings(max_examples=20)
    def test_decimal_rejection_at_top_level(self, value: Decimal) -> None:
        """Property: Non-finite Decimals rejected even at top level."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(value)

    @given(value=non_finite_decimals)
    @settings(max_examples=20)
    def test_decimal_in_list_rejected(self, value: Decimal) -> None:
        """Property: Non-finite Decimals in lists are rejected."""
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json([Decimal("1.5"), value, Decimal("2.5")])

    @given(value=non_finite_decimals, depth=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_decimal_rejection_in_nested_structures(self, value: Decimal, depth: int) -> None:
        """Property: Non-finite Decimal nested deep in dicts is rejected."""
        from typing import Any

        data: dict[str, Any] = {"level": value}
        for _ in range(depth):
            data = {"nested": data}
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json(data)

    def test_finite_decimals_accepted(self) -> None:
        """Verify finite Decimal values are accepted (positive control)."""
        finite_decimals = [
            Decimal("0"),
            Decimal("1.5"),
            Decimal("-100.25"),
            Decimal("1E-10"),
            Decimal("9.999999999999999999E+100"),
        ]
        for d in finite_decimals:
            result = canonical_json({"value": d})
            assert isinstance(result, str)


# =============================================================================
# Enum Rejection Property Tests (UNIQUE - contract boundary tests)
# =============================================================================


class TestEnumRejection:
    """Property tests for invalid enum value rejection.

    Enums stored in the audit trail MUST reject invalid values. These tests
    verify that arbitrary strings that are not valid enum members raise ValueError.

    This is critical for audit integrity: if we accept "invalid_status" as a
    RowOutcome, the audit trail would contain garbage that can't be interpreted.
    """

    @given(
        invalid_value=st.text(min_size=1, max_size=20).filter(
            lambda s: s
            not in (
                "completed",
                "routed",
                "forked",
                "failed",
                "quarantined",
                "consumed_in_batch",
                "coalesced",
                "expanded",
                "buffered",
            )
        )
    )
    @settings(max_examples=50)
    def test_invalid_row_outcome_rejected(self, invalid_value: str) -> None:
        """Property: Invalid RowOutcome values raise ValueError."""
        from elspeth.contracts.enums import RowOutcome

        with pytest.raises(ValueError):
            RowOutcome(invalid_value)

    @given(invalid_value=st.text(min_size=1, max_size=20).filter(lambda s: s not in ("continue", "route", "fork_to_paths")))
    @settings(max_examples=50)
    def test_invalid_routing_kind_rejected(self, invalid_value: str) -> None:
        """Property: Invalid RoutingKind values raise ValueError."""
        from elspeth.contracts.enums import RoutingKind

        with pytest.raises(ValueError):
            RoutingKind(invalid_value)

    @given(invalid_value=st.text(min_size=1, max_size=20).filter(lambda s: s not in ("running", "completed", "failed")))
    @settings(max_examples=50)
    def test_invalid_run_status_rejected(self, invalid_value: str) -> None:
        """Property: Invalid RunStatus values raise ValueError."""
        from elspeth.contracts.enums import RunStatus

        with pytest.raises(ValueError):
            RunStatus(invalid_value)

    @given(invalid_value=st.text(min_size=1, max_size=20).filter(lambda s: s not in ("open", "pending", "completed", "failed")))
    @settings(max_examples=50)
    def test_invalid_node_state_status_rejected(self, invalid_value: str) -> None:
        """Property: Invalid NodeStateStatus values raise ValueError."""
        from elspeth.contracts.enums import NodeStateStatus

        with pytest.raises(ValueError):
            NodeStateStatus(invalid_value)


# =============================================================================
# Configuration Rejection Property Tests (UNIQUE - Tier 3 boundary tests)
# =============================================================================


class TestConfigRejection:
    """Property tests for invalid configuration rejection.

    Configuration validation is a trust boundary (Tier 3 - external data).
    Invalid config values MUST be rejected at load time, not silently coerced.
    """

    @given(max_attempts=st.integers(max_value=0))
    @settings(max_examples=30)
    def test_invalid_retry_max_attempts_rejected(self, max_attempts: int) -> None:
        """Property: RetryConfig rejects max_attempts < 1."""
        from elspeth.contracts.config import RuntimeRetryConfig

        with pytest.raises(ValueError, match="max_attempts"):
            RuntimeRetryConfig(max_attempts=max_attempts, base_delay=1.0, max_delay=60.0, jitter=1.0, exponential_base=2.0)

    @given(count=st.integers(max_value=0))
    @settings(max_examples=30)
    def test_invalid_trigger_count_rejected(self, count: int) -> None:
        """Property: TriggerConfig rejects count <= 0."""
        from pydantic import ValidationError

        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError):
            TriggerConfig(count=count)

    @given(timeout=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30)
    def test_invalid_trigger_timeout_rejected(self, timeout: float) -> None:
        """Property: TriggerConfig rejects timeout_seconds <= 0."""
        from pydantic import ValidationError

        from elspeth.core.config import TriggerConfig

        with pytest.raises(ValidationError):
            TriggerConfig(timeout_seconds=timeout)
