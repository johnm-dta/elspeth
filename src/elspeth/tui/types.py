# src/elspeth/tui/types.py
"""Type definitions for TUI data contracts.

These TypedDicts define the exact shape of data passed between
TUI components. Using direct field access (data["field"]) instead
of .get() ensures missing fields fail loudly.
"""

from typing import Any, TypedDict


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


class NodeStateInfo(TypedDict, total=False):
    """Node state information for detail panel display.

    This TypedDict uses total=False to indicate that fields are
    conditionally present based on context:

    Required fields (always present from _load_node_state):
    - node_id: Unique identifier for the node
    - plugin_name: Name of the plugin (e.g., "csv_source", "filter")
    - node_type: Type of node ("source", "transform", "sink", etc.)

    Optional fields (present only after execution has occurred):
    - state_id: Unique identifier for this execution state
    - token_id: Token that visited this node
    - status: Execution status ("open", "completed", "failed")
    - started_at: When execution began (ISO timestamp string)
    - completed_at: When execution finished (ISO timestamp string)
    - duration_ms: Execution duration in milliseconds
    - input_hash: Hash of input data
    - output_hash: Hash of output data (if completed)
    - error_json: JSON string of error details (if failed)
    - artifact: Artifact info dict (for sink nodes)
    """

    # Required - always present from node registration
    node_id: str
    plugin_name: str
    node_type: str

    # Optional - present after execution
    state_id: str
    token_id: str
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    input_hash: str
    output_hash: str
    error_json: str
    artifact: dict[str, Any]
