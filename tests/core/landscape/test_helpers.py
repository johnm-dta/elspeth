"""Tests for landscape helper functions."""

from datetime import UTC, datetime
from enum import Enum

import pytest

from elspeth.core.landscape._helpers import coerce_enum, generate_id, now


class TestNow:
    """Tests for now() helper."""

    def test_returns_utc_datetime(self) -> None:
        """now() returns UTC datetime."""
        result = now()
        assert isinstance(result, datetime)
        assert result.tzinfo == UTC

    def test_returns_current_time(self) -> None:
        """now() returns approximately current time."""
        before = datetime.now(UTC)
        result = now()
        after = datetime.now(UTC)
        assert before <= result <= after


class SampleEnum(Enum):
    """Sample enum for testing."""

    VALUE_A = "value_a"
    VALUE_B = "value_b"


class TestGenerateId:
    """Tests for generate_id() helper."""

    def test_returns_hex_string(self) -> None:
        """generate_id() returns hex string."""
        result = generate_id()
        assert isinstance(result, str)
        # UUID4 hex is 32 characters
        assert len(result) == 32
        # All characters are hex
        assert all(c in "0123456789abcdef" for c in result)

    def test_returns_unique_ids(self) -> None:
        """generate_id() returns unique IDs each call."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestCoerceEnum:
    """Tests for coerce_enum() helper."""

    def test_returns_enum_unchanged(self) -> None:
        """coerce_enum passes through enum values."""
        result = coerce_enum(SampleEnum.VALUE_A, SampleEnum)
        assert result is SampleEnum.VALUE_A

    def test_converts_string_to_enum(self) -> None:
        """coerce_enum converts valid string to enum."""
        result = coerce_enum("value_a", SampleEnum)
        assert result == SampleEnum.VALUE_A

    def test_crashes_on_invalid_string(self) -> None:
        """coerce_enum crashes on invalid string (Tier 1 trust)."""
        with pytest.raises(ValueError):
            coerce_enum("invalid_value", SampleEnum)
