"""Type definitions for TUI data contracts.

These TypedDicts define the exact shape of data passed between
TUI components. Using direct field access (data["field"]) instead
of .get() ensures missing fields fail loudly.

Error and artifact display types follow the Tier 1 trust model:
- After JSON parsing, data is validated and converted to typed structures
- Display code accesses fields directly (crash on missing = bug)
"""

from typing import Any, Required, TypedDict


class NodeInfo(TypedDict):
    """Information about a single pipeline node."""

    name: str
    node_id: str | None
    node_type: str


class SourceInfo(TypedDict):
    """Information about the pipeline source."""

    name: str | None
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


class TreeNodeDict(TypedDict):
    """Flattened tree node for rendering.

    Output of LineageTree.get_tree_nodes(). All fields are required
    since they're generated internally by _flatten_tree().
    """

    label: str
    node_id: str | None
    node_type: str
    depth: int
    has_children: bool
    expanded: bool


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
    node_id: Required[str]
    plugin_name: Required[str]
    node_type: Required[str]

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


class ExecutionErrorDisplay(TypedDict, total=False):
    """Parsed ExecutionError for display.

    From contracts.errors.ExecutionError - used by executors for exceptions.
    exception and type fields are REQUIRED after validation.
    """

    exception: Required[str]  # Exception message (REQUIRED)
    type: Required[str]  # Exception class name (e.g., "ValueError") (REQUIRED)
    traceback: str  # Optional full traceback
    phase: str  # Optional phase (e.g., "flush")


class TransformErrorDisplay(TypedDict, total=False):
    """Parsed TransformErrorReason for display.

    From contracts.errors.TransformErrorReason - used by transforms.
    The 'reason' field is REQUIRED and identifies the error category.
    """

    reason: Required[str]  # Error category (e.g., "api_error", "missing_field") (REQUIRED)
    error: str  # Exception message or description
    message: str  # Human-readable error message
    error_type: str  # Sub-category
    field: str  # Field name for field-related errors


class CoalesceErrorDisplay(TypedDict, total=False):
    """Parsed CoalesceFailureReason for display.

    From contracts.errors.CoalesceFailureReason - used by CoalesceExecutor.
    The 'failure_reason' field is REQUIRED and identifies the failure type.
    """

    failure_reason: Required[str]  # Why coalesce failed (REQUIRED)
    expected_branches: Required[list[str]]  # Branches expected (REQUIRED)
    branches_arrived: Required[list[str]]  # Branches that arrived (REQUIRED)
    merge_policy: Required[str]  # Merge policy in effect (REQUIRED)
    timeout_ms: int  # Timeout in ms (if timeout failure)
    select_branch: str  # Target branch for select policy


class ArtifactDisplay(TypedDict, total=False):
    """Parsed Artifact for display.

    From contracts.audit.Artifact - produced by sinks.
    All fields are required and match the dataclass contract.
    """

    artifact_id: Required[str]
    path_or_uri: Required[str]
    content_hash: Required[str]
    size_bytes: Required[int]
