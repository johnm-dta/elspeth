# src/elspeth/core/landscape/row_data.py
"""Row data retrieval types with explicit state discrimination.

Replaces the ambiguous `dict | None` return from get_row_data() with
explicit state handling. Callers can now match on state instead of
guessing why data is None.

States:
    AVAILABLE: Data was found and returned
    PURGED: Data existed but was deleted (retention policy)
    NEVER_STORED: Row exists but source_data_ref was never set
    STORE_NOT_CONFIGURED: No payload store configured
    ROW_NOT_FOUND: Row ID doesn't exist in the database
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RowDataState(str, Enum):
    """Discriminator for row data retrieval results.

    Inherits from str for JSON serialization compatibility.
    """

    AVAILABLE = "available"
    PURGED = "purged"
    NEVER_STORED = "never_stored"
    STORE_NOT_CONFIGURED = "store_not_configured"
    ROW_NOT_FOUND = "row_not_found"


@dataclass(frozen=True)
class RowDataResult:
    """Result of a row data retrieval with explicit state.

    Invariants:
        - AVAILABLE state requires non-None data
        - All other states require None data

    Example:
        result = recorder.get_row_data(row_id)
        match result.state:
            case RowDataState.AVAILABLE:
                process(result.data)  # Safe - guaranteed non-None
            case RowDataState.PURGED:
                log.info("Data was purged per retention policy")
            case RowDataState.ROW_NOT_FOUND:
                raise KeyError(f"Unknown row: {row_id}")
    """

    state: RowDataState
    data: dict[str, Any] | None

    def __post_init__(self) -> None:
        if self.state == RowDataState.AVAILABLE and self.data is None:
            raise ValueError("AVAILABLE state requires non-None data")
        if self.state != RowDataState.AVAILABLE and self.data is not None:
            raise ValueError(f"{self.state} state requires None data")
