# src/elspeth/tui/types.py
"""Type definitions for TUI data contracts.

These TypedDicts define the exact shape of data passed between
TUI components. Using direct field access (data["field"]) instead
of .get() ensures missing fields fail loudly.
"""

from typing import TypedDict


class NodeInfo(TypedDict):
    """Information about a single pipeline node."""

    name: str
    node_id: str | None


class SourceInfo(TypedDict):
    """Information about the pipeline source."""

    name: str
    node_id: str | None


class TokenDisplayInfo(TypedDict):
    """Token information formatted for TUI display.

    Note: This is a DISPLAY type, not the canonical TokenInfo from contracts.
    It contains presentation-specific fields like 'path' for breadcrumb display.
    """

    token_id: str
    row_id: str
    path: list[str]


class LineageData(TypedDict):
    """Data contract for lineage tree display.

    All fields are required. If data is unavailable, the caller
    must handle that BEFORE constructing LineageData - not inside
    the widget via .get() defaults.
    """

    run_id: str
    source: SourceInfo
    transforms: list[NodeInfo]
    sinks: list[NodeInfo]
    tokens: list[TokenDisplayInfo]
