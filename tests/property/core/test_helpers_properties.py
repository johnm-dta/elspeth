# tests/property/core/test_helpers_properties.py
"""Property-based tests for landscape helper functions.

These tests verify the invariants of ELSPETH's core helper functions:

ID Generation Properties:
- generate_id() produces valid UUID4 hex strings
- IDs are unique (no collisions in reasonable sample)
- Format is consistent (32 lowercase hex chars)

Enum Coercion Properties (Tier 1 Trust):
- Valid enum values coerce correctly
- Enum instances pass through unchanged
- Invalid strings CRASH (ValueError) - no silent coercion

Timestamp Properties:
- now() returns UTC timestamps
- Timestamps are timezone-aware
- Sequential calls are monotonic (or equal)
"""

from __future__ import annotations

from datetime import UTC, timedelta
from enum import Enum

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.landscape._helpers import coerce_enum, generate_id, now

# =============================================================================
# Strategies for generating test data
# =============================================================================

# Invalid enum value strings (not valid for any enum we'll test)
invalid_enum_values = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "a", "b", "c")
)


# Sample enum for coercion tests (avoid Test* prefix to prevent pytest collection)
class SampleStatus(str, Enum):
    """Sample enum for property tests."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# All valid sample enum values
sample_status_values = st.sampled_from(list(SampleStatus))


# =============================================================================
# generate_id() Property Tests
# =============================================================================


class TestGenerateIdProperties:
    """Property tests for generate_id() function."""

    @given(st.data())
    @settings(max_examples=100)
    def test_generates_32_char_hex_string(self, data: st.DataObject) -> None:
        """Property: generate_id() produces 32-character hex strings."""
        id_ = generate_id()

        assert isinstance(id_, str)
        assert len(id_) == 32
        assert all(c in "0123456789abcdef" for c in id_)

    @given(st.data())
    @settings(max_examples=100)
    def test_generates_lowercase_hex(self, data: st.DataObject) -> None:
        """Property: generate_id() produces lowercase hex (no uppercase)."""
        id_ = generate_id()

        # No uppercase letters
        assert id_ == id_.lower()

    @given(st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_ids_are_unique(self, count: int) -> None:
        """Property: Multiple calls produce unique IDs.

        While UUID4 collisions are theoretically possible, they're
        astronomically unlikely. This test catches broken RNG or
        implementation bugs.
        """
        ids = [generate_id() for _ in range(count)]

        # All IDs should be unique
        assert len(set(ids)) == count

    def test_format_matches_uuid4_hex(self) -> None:
        """Property: Output format matches uuid.uuid4().hex convention."""
        import uuid

        id_ = generate_id()

        # Should be parseable as a UUID (proves correct format)
        parsed = uuid.UUID(id_)
        assert parsed.version == 4  # UUID version 4

    @given(st.data())
    @settings(max_examples=50)
    def test_no_hyphens_or_braces(self, data: st.DataObject) -> None:
        """Property: Output is bare hex (no UUID formatting characters)."""
        id_ = generate_id()

        assert "-" not in id_
        assert "{" not in id_
        assert "}" not in id_


# =============================================================================
# coerce_enum() Property Tests - Tier 1 Trust Model
# =============================================================================


class TestCoerceEnumProperties:
    """Property tests for coerce_enum() function.

    Per Data Manifesto: This handles Tier 1 data (audit database).
    Invalid values must CRASH immediately, not silently coerce.
    """

    @given(status=sample_status_values)
    @settings(max_examples=50)
    def test_enum_instance_passthrough(self, status: SampleStatus) -> None:
        """Property: Enum instances pass through unchanged."""
        result = coerce_enum(status, SampleStatus)

        assert result is status
        assert isinstance(result, SampleStatus)

    @given(status=sample_status_values)
    @settings(max_examples=50)
    def test_valid_string_coerces_correctly(self, status: SampleStatus) -> None:
        """Property: Valid string values coerce to correct enum."""
        string_value = status.value
        result = coerce_enum(string_value, SampleStatus)

        assert result == status
        assert isinstance(result, SampleStatus)

    @given(invalid=invalid_enum_values)
    @settings(max_examples=100)
    def test_invalid_string_crashes(self, invalid: str) -> None:
        """Property: Invalid strings raise ValueError (Tier 1 crash semantics).

        This is the core Tier 1 Trust invariant - bad data in our audit
        trail is evidence of corruption, so we crash immediately rather
        than silently coercing to some default.
        """
        with pytest.raises(ValueError):
            coerce_enum(invalid, SampleStatus)

    def test_empty_string_crashes(self) -> None:
        """Property: Empty string is invalid and crashes."""
        with pytest.raises(ValueError):
            coerce_enum("", SampleStatus)

    @given(status=sample_status_values)
    @settings(max_examples=50)
    def test_coercion_is_idempotent(self, status: SampleStatus) -> None:
        """Property: Coercing an already-coerced value is idempotent."""
        once = coerce_enum(status.value, SampleStatus)
        twice = coerce_enum(once, SampleStatus)
        thrice = coerce_enum(twice, SampleStatus)

        assert once == twice == thrice
        assert once is twice is thrice  # Same object (enum identity)

    def test_wrong_enum_type_crashes(self) -> None:
        """Property: Valid value for wrong enum type crashes.

        Even if 'pending' is a valid SampleStatus value, trying to
        coerce it as a different enum type should fail.
        """

        class OtherEnum(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        with pytest.raises(ValueError):
            coerce_enum("pending", OtherEnum)

    @given(status=sample_status_values)
    @settings(max_examples=20)
    def test_coercion_preserves_enum_semantics(self, status: SampleStatus) -> None:
        """Property: Coerced values behave like native enum values."""
        coerced = coerce_enum(status.value, SampleStatus)

        # All enum properties work correctly
        assert coerced.name == status.name
        assert coerced.value == status.value
        assert coerced == status


# =============================================================================
# now() Property Tests
# =============================================================================


class TestNowProperties:
    """Property tests for now() timestamp function."""

    def test_returns_utc_timezone(self) -> None:
        """Property: now() returns UTC timezone-aware datetime."""
        ts = now()

        assert ts.tzinfo is not None
        assert ts.tzinfo == UTC

    def test_is_timezone_aware(self) -> None:
        """Property: Returned datetime is timezone-aware (not naive)."""
        ts = now()

        # Naive datetimes have tzinfo=None
        assert ts.tzinfo is not None

    @given(st.data())
    @settings(max_examples=50)
    def test_sequential_calls_are_close(self, data: st.DataObject) -> None:
        """Property: Sequential calls happen close in time."""
        ts1 = now()
        ts2 = now()
        ts3 = now()

        span = max(ts1, ts2, ts3) - min(ts1, ts2, ts3)
        assert span < timedelta(seconds=2)

    def test_utc_offset_is_zero(self) -> None:
        """Property: UTC offset is +00:00 (not some other timezone)."""
        ts = now()

        # UTC has zero offset
        offset = ts.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0

    def test_can_compare_with_other_utc_timestamps(self) -> None:
        """Property: Timestamps can be compared with other UTC times."""
        from datetime import datetime

        ts = now()
        other = datetime.now(UTC)

        # Should be able to compare without TypeError
        _ = ts < other or ts == other or ts > other

    def test_not_naive_datetime(self) -> None:
        """Property: Result is never a naive datetime.

        Naive datetimes cause subtle bugs in timestamp comparisons
        and database storage. ELSPETH requires all timestamps be
        timezone-aware.
        """
        from datetime import datetime

        ts = now()

        # Can't compare naive and aware datetimes
        # Create naive datetime (no tzinfo) using datetime() constructor

        naive = datetime(2020, 1, 1, 12, 0, 0)  # noqa: DTZ001 - intentionally naive

        with pytest.raises(TypeError):
            _ = ts < naive  # type: ignore[operator]


# =============================================================================
# Combined Property Tests
# =============================================================================


class TestHelperInteractionProperties:
    """Property tests for helper function interactions."""

    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=20)
    def test_ids_and_timestamps_independent(self, count: int) -> None:
        """Property: ID generation doesn't affect timestamp generation."""
        # Generate interleaved IDs and timestamps
        pairs = [(generate_id(), now()) for _ in range(count)]

        # IDs should still be unique
        ids = [p[0] for p in pairs]
        assert len(set(ids)) == count

        # Timestamps should still be close in time
        timestamps = [p[1] for p in pairs]
        span = max(timestamps) - min(timestamps)
        assert span < timedelta(seconds=2)
