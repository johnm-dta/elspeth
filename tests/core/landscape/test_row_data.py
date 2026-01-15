# tests/core/landscape/test_row_data.py
"""Tests for RowDataState enum and RowDataResult type.

These types replace the ambiguous dict | None return from get_row_data()
with explicit state discrimination.
"""

import pytest

from elspeth.core.landscape.row_data import RowDataResult, RowDataState


class TestRowDataState:
    """Test that all expected states are defined."""

    def test_all_states_defined(self) -> None:
        assert RowDataState.AVAILABLE.value == "available"
        assert RowDataState.PURGED.value == "purged"
        assert RowDataState.NEVER_STORED.value == "never_stored"
        assert RowDataState.STORE_NOT_CONFIGURED.value == "store_not_configured"
        assert RowDataState.ROW_NOT_FOUND.value == "row_not_found"


class TestRowDataResult:
    """Test RowDataResult invariants."""

    def test_available_with_data(self) -> None:
        result = RowDataResult(state=RowDataState.AVAILABLE, data={"key": "value"})
        assert result.data == {"key": "value"}

    def test_available_without_data_raises(self) -> None:
        with pytest.raises(ValueError, match="AVAILABLE state requires non-None data"):
            RowDataResult(state=RowDataState.AVAILABLE, data=None)

    def test_non_available_with_data_raises(self) -> None:
        with pytest.raises(ValueError, match="state requires None data"):
            RowDataResult(state=RowDataState.PURGED, data={"unexpected": "data"})

    def test_purged_state(self) -> None:
        result = RowDataResult(state=RowDataState.PURGED, data=None)
        assert result.state == RowDataState.PURGED
        assert result.data is None

    def test_never_stored_state(self) -> None:
        result = RowDataResult(state=RowDataState.NEVER_STORED, data=None)
        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None

    def test_store_not_configured_state(self) -> None:
        result = RowDataResult(state=RowDataState.STORE_NOT_CONFIGURED, data=None)
        assert result.state == RowDataState.STORE_NOT_CONFIGURED
        assert result.data is None

    def test_row_not_found_state(self) -> None:
        result = RowDataResult(state=RowDataState.ROW_NOT_FOUND, data=None)
        assert result.state == RowDataState.ROW_NOT_FOUND
        assert result.data is None

    def test_frozen_immutability(self) -> None:
        from dataclasses import FrozenInstanceError

        result = RowDataResult(state=RowDataState.AVAILABLE, data={"key": "value"})
        with pytest.raises(FrozenInstanceError):
            result.state = RowDataState.PURGED  # type: ignore[misc]
