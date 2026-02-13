"""Unit tests for RowDataState and RowDataResult invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from elspeth.core.landscape.row_data import RowDataResult, RowDataState


def test_row_data_state_values() -> None:
    assert RowDataState.AVAILABLE.value == "available"
    assert RowDataState.PURGED.value == "purged"
    assert RowDataState.NEVER_STORED.value == "never_stored"
    assert RowDataState.STORE_NOT_CONFIGURED.value == "store_not_configured"
    assert RowDataState.ROW_NOT_FOUND.value == "row_not_found"


def test_row_data_result_available_requires_data() -> None:
    with pytest.raises(ValueError, match="AVAILABLE state requires non-None data"):
        RowDataResult(state=RowDataState.AVAILABLE, data=None)


def test_row_data_result_non_available_requires_none_data() -> None:
    with pytest.raises(ValueError, match="state requires None data"):
        RowDataResult(state=RowDataState.PURGED, data={"unexpected": "payload"})


def test_row_data_result_allows_available_with_data() -> None:
    result = RowDataResult(state=RowDataState.AVAILABLE, data={"id": 1})
    assert result.state == RowDataState.AVAILABLE
    assert result.data == {"id": 1}


@pytest.mark.parametrize("non_dict_data", [[1, 2, 3], "payload", 42, 3.14, True])
def test_row_data_result_available_rejects_non_dict_data(non_dict_data: Any) -> None:
    with pytest.raises(TypeError, match="AVAILABLE state requires dict data"):
        RowDataResult(state=RowDataState.AVAILABLE, data=non_dict_data)  # type: ignore[arg-type]


def test_row_data_result_allows_non_available_with_none() -> None:
    result = RowDataResult(state=RowDataState.ROW_NOT_FOUND, data=None)
    assert result.state == RowDataState.ROW_NOT_FOUND
    assert result.data is None


def test_row_data_result_is_frozen() -> None:
    result = RowDataResult(state=RowDataState.AVAILABLE, data={"id": 1})
    with pytest.raises(FrozenInstanceError):
        result.state = RowDataState.PURGED  # type: ignore[misc]
