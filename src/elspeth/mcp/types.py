"""TypedDict definitions for MCP server return types.

These TypedDicts give static structure to the dicts returned by
LandscapeAnalyzer methods. At runtime they are plain dicts and
serialize identically via json.dumps() -- the MCP wire format is
unchanged. mypy uses them to verify that return structures match.

Follows the naming convention from tui/types.py:
  - {Noun}Record   -- items in a list (SQL row projections)
  - {Noun}Detail   -- dataclass_to_dict conversions (audit entities)
  - {Noun}Report   -- aggregate / analysis single-dict returns

Design decisions:
  - ``total=False`` + ``Required[]`` when some keys are conditionally present
  - ``dict[str, Any]`` for genuinely dynamic sub-structures
    (outcome distributions, per-plugin stats, dataclass-converted nested objects)
  - Functional TypedDict form for ``DAGEdge`` (``"from"`` is a Python keyword)
"""

from typing import Any, Literal, Required, TypedDict

RunStatusValue = Literal["running", "completed", "completed_with_failures", "failed", "empty", "interrupted"]
ExportStatusValue = Literal["pending", "completed", "failed"]
OperationTypeValue = Literal["source_load", "sink_write", "runtime_preflight"]
OPERATION_TYPE_VALUES: tuple[OperationTypeValue, ...] = ("source_load", "sink_write", "runtime_preflight")
OperationStatusValue = Literal["open", "completed", "failed", "pending"]
OPERATION_STATUS_VALUES: tuple[OperationStatusValue, ...] = ("open", "completed", "failed", "pending")
CallTypeValue = Literal["llm", "http", "http_redirect", "sql", "vector", "filesystem"]
CallStatusValue = Literal["success", "error"]
NodeStateStatusValue = Literal["open", "pending", "completed", "failed"]
NodeTypeValue = Literal["source", "transform", "gate", "aggregation", "coalesce", "queue", "sink"]
RoutingModeValue = Literal["move", "copy", "divert"]
DAGFlowTypeValue = Literal["normal", "divert"]
SchemaModeValue = Literal["fixed", "flexible", "observed", "parse"]
ContractModeValue = Literal["FIXED", "FLEXIBLE", "OBSERVED"]
ContractViolationTypeValue = Literal["type_mismatch", "missing_field", "extra_field"]
DiagnosticSeverityValue = Literal["CRITICAL", "WARNING", "INFO"]
DiagnosticStatusValue = Literal["OK", "WARNING", "CRITICAL"]
DivertTypeValue = Literal["quarantine", "error"]


class RunRecord(TypedDict):
    """A run record as returned by ``list_runs``."""

    run_id: str
    status: RunStatusValue
    started_at: str | None
    completed_at: str | None
    config_hash: str
    export_status: ExportStatusValue | None


class RowRecord(TypedDict):
    """A source row record as returned by ``list_rows``."""

    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_row_index: int
    ingest_sequence: int
    source_data_hash: str
    source_data_ref: str | None
    created_at: str | None


class TokenRecord(TypedDict):
    """A token record as returned by ``list_tokens``."""

    token_id: str
    row_id: str
    branch_name: str | None
    fork_group_id: str | None
    join_group_id: str | None
    step_in_pipeline: int | None
    expand_group_id: str | None
    created_at: str | None


class TokenChildRecord(TypedDict):
    """Forward lineage record: what child tokens were created from a parent.

    Used by ``get_token_children`` to trace what a COALESCED parent merged into.
    The child_token_id is the merged token; ordinal is this parent's position
    in the merge (0-indexed).
    """

    child_token_id: str
    """The token created from this parent (e.g., the coalesced result)."""

    parent_token_id: str
    """The parent token being queried."""

    ordinal: int
    """This parent's position in the child's parent list (0-indexed)."""


class OperationRecord(TypedDict):
    """A source/sink operation record as returned by ``list_operations``."""

    operation_id: str
    run_id: str
    node_id: str
    plugin_name: str
    operation_type: OperationTypeValue
    status: OperationStatusValue
    started_at: str | None
    completed_at: str | None
    duration_ms: float | None
    error_message: str | None


class OperationCallRecord(TypedDict):
    """A call made during a source/sink operation (``get_operation_calls``)."""

    call_id: str
    operation_id: str
    call_index: int
    call_type: CallTypeValue
    status: CallStatusValue
    latency_ms: float | None
    request_hash: str
    response_hash: str | None
    created_at: str | None


class NodeStateRecord(TypedDict, total=False):
    """A node state record as returned by ``get_node_states``.

    Core fields are always present. Context fields (context_after, error,
    success_reason) are included only when ``include_context=True``.
    """

    # Always present
    state_id: Required[str]
    token_id: Required[str]
    node_id: Required[str]
    step_index: Required[int]
    attempt: Required[int]
    status: Required[NodeStateStatusValue]
    input_hash: Required[str]
    output_hash: str | None
    duration_ms: float | None
    started_at: str | None
    completed_at: str | None

    # Optional context fields (when include_context=True)
    context_after: dict[str, Any] | None
    error: dict[str, Any] | None
    success_reason: dict[str, Any] | None


class CollisionFieldRecord(TypedDict):
    """Details of a single field collision during coalesce.

    When the coalesce node status is FAILED (e.g., union_collision_policy='fail'),
    ``winner_branch`` and ``winner_value`` are None because no winner was selected.
    """

    field: str
    """Field name where collision occurred."""

    winner_branch: str | None
    """Branch whose value was kept (last_wins or first_wins), or None if merge failed."""

    winner_value: Any
    """The value that won, or None if merge failed."""

    competing_values: list[tuple[str, Any]]
    """List of (branch_name, value) entries in merge order."""


class CollisionRecord(TypedDict):
    """A collision event from a coalesce merge operation.

    Returned by ``list_collisions`` for quick identification of merge conflicts.
    When ``status`` is FAILED, the merge aborted and no winner was selected.
    """

    run_id: str
    token_id: str
    node_id: str
    plugin_name: str
    status: NodeStateStatusValue
    """Node state status: COMPLETED if merge succeeded, FAILED if it aborted."""
    completed_at: str | None
    collision_fields: list[CollisionFieldRecord]
    union_field_origins: dict[str, str]
    """Field name → originating branch for all union-merged fields."""


# Dataclass mirror types: dict[str, Any] aliases for dataclass_to_dict()
# conversions where the exact shape depends on the dataclass variant.

RunDetail = dict[str, Any]
"""Run detail dict from ``dataclass_to_dict(Run)``."""

NodeDetail = dict[str, Any]
"""Node detail dict from ``dataclass_to_dict(Node)``."""

CallDetail = dict[str, Any]
"""Call detail dict from ``dataclass_to_dict(Call)``."""


class RunSummaryCounts(TypedDict):
    """Count sub-dict inside ``RunSummaryReport``."""

    rows: int
    tokens: int
    nodes: int
    node_states: int
    operations: int
    source_loads: int
    sink_writes: int


class RunSummaryErrors(TypedDict):
    """Error count sub-dict inside ``RunSummaryReport``."""

    validation: int
    transform: int
    total: int


class OutcomeDistributionEntry(TypedDict):
    """Single entry in outcome distribution (ADR-019 two-axis terminal model)."""

    outcome: str | None
    path: str
    completed: bool
    count: int


class RunSummaryReport(TypedDict):
    """Return type for ``get_run_summary``."""

    run_id: str
    status: RunStatusValue
    started_at: str | None
    completed_at: str | None
    run_duration_seconds: float | None
    counts: RunSummaryCounts
    errors: RunSummaryErrors
    outcome_distribution: list[OutcomeDistributionEntry]
    avg_state_duration_ms: float | None


class DAGNode(TypedDict):
    """A node in the DAG structure."""

    node_id: str
    plugin_name: str
    node_type: NodeTypeValue
    sequence: int | None


# Functional form because "from" is a Python keyword
DAGEdge = TypedDict(
    "DAGEdge",
    {
        "from": str,
        "to": str,
        "label": str,
        "mode": RoutingModeValue,
        "flow_type": DAGFlowTypeValue,
    },
)


class DAGStructureReport(TypedDict):
    """Return type for ``get_dag_structure``."""

    run_id: str
    nodes: list[DAGNode]
    edges: list[DAGEdge]
    node_count: int
    edge_count: int
    terminal_sink_map: dict[str, str]
    mermaid: str


class NodePerformance(TypedDict):
    """Per-node performance statistics."""

    node_id: str
    plugin: str
    type: NodeTypeValue
    executions: int
    avg_ms: float | None
    min_ms: float | None
    max_ms: float | None
    total_ms: float | None
    pct_of_total: float
    failures: int


class PerformanceReport(TypedDict):
    """Return type for ``get_performance_report``."""

    run_id: str
    total_processing_time_ms: float
    node_count: int
    bottlenecks: list[NodePerformance]
    high_variance_nodes: list[NodePerformance]
    node_performance: list[NodePerformance]


class ValidationErrorGroup(TypedDict):
    """Validation error group by source plugin."""

    source_plugin: str
    schema_mode: SchemaModeValue
    count: int


class TransformErrorGroup(TypedDict):
    """Transform error group by transform plugin."""

    transform_plugin: str
    count: int


class ValidationErrorSummary(TypedDict):
    """Validation errors sub-dict in ``ErrorAnalysisReport``."""

    total: int
    by_source: list[ValidationErrorGroup]
    sample_data: list[dict[str, Any] | None]


class TransformErrorSummary(TypedDict):
    """Transform errors sub-dict in ``ErrorAnalysisReport``."""

    total: int
    by_transform: list[TransformErrorGroup]
    sample_details: list[dict[str, Any] | None]


class ErrorAnalysisReport(TypedDict):
    """Return type for ``get_error_analysis``."""

    run_id: str
    validation_errors: ValidationErrorSummary
    transform_errors: TransformErrorSummary


class LLMSummary(TypedDict):
    """LLM call summary totals."""

    total_calls: int
    total_latency_ms: float
    avg_latency_ms: float | None


class LLMPluginStats(TypedDict):
    """Per-plugin LLM statistics."""

    total_calls: int
    successful: int
    failed: int
    avg_latency_ms: float
    total_latency_ms: float


class LLMUsageReport(TypedDict, total=False):
    """Return type for ``get_llm_usage_report``.

    Uses ``total=False`` because the empty-case returns ``message``
    instead of ``llm_summary`` / ``by_plugin``.
    """

    run_id: Required[str]
    # Present when there are LLM calls
    call_types: dict[str, int]  # dynamic call type names
    llm_summary: LLMSummary
    by_plugin: dict[str, LLMPluginStats]  # dynamic plugin names
    # Present when there are NO LLM calls
    message: str


class OutcomeSummary(TypedDict):
    """Outcome summary counts."""

    terminal_tokens: int
    non_terminal_tokens: int
    fork_operations: int
    join_operations: int


class OutcomeAnalysisReport(TypedDict):
    """Return type for ``get_outcome_analysis``."""

    run_id: str
    summary: OutcomeSummary
    outcome_distribution: list[OutcomeDistributionEntry]
    sink_distribution: dict[str, int]  # dynamic sink names


class ColumnInfo(TypedDict):
    """Column metadata from SQLAlchemy inspector."""

    name: str
    type: str
    nullable: bool


class ForeignKeyInfo(TypedDict):
    """Foreign key metadata."""

    columns: list[str]
    references: str


class TableInfo(TypedDict):
    """Table metadata from SQLAlchemy inspector."""

    columns: list[ColumnInfo]
    primary_key: list[str]
    foreign_keys: list[ForeignKeyInfo]


class SchemaDescription(TypedDict):
    """Return type for ``describe_schema``."""

    tables: dict[str, TableInfo]  # dynamic table names
    table_count: int
    hint: str


class ValidationErrorDetail(TypedDict):
    """A single validation error in ``get_errors``."""

    error_id: str
    node_id: str | None
    row_hash: str
    row_data: dict[str, Any] | None
    schema_mode: SchemaModeValue
    created_at: str | None


class TransformErrorDetail(TypedDict):
    """A single transform error in ``get_errors``."""

    error_id: str
    token_id: str
    transform_id: str
    row_data: dict[str, Any] | None
    error_details: dict[str, Any] | None
    created_at: str | None


class ErrorsReport(TypedDict, total=False):
    """Return type for ``get_errors``.

    Uses ``total=False`` because validation_errors and transform_errors
    are conditionally present depending on the ``error_type`` parameter.
    """

    run_id: Required[str]
    validation_errors: list[ValidationErrorDetail]
    transform_errors: list[TransformErrorDetail]


class DiagnosticProblem(TypedDict, total=False):
    """A single problem in ``diagnose``.

    Uses ``total=False`` because different problem types include
    different fields (``run_ids``, ``runs``, ``operations``, ``count``).
    """

    severity: Required[DiagnosticSeverityValue]
    type: Required[str]
    message: Required[str]
    count: int
    run_ids: list[str]
    runs: list[dict[str, Any]]
    operations: list[dict[str, Any]]


class RecentRunSummary(TypedDict):
    """A recent run in the diagnose summary."""

    run_id: str
    status: RunStatusValue
    started: str | None


class DiagnosticReport(TypedDict):
    """Return type for ``diagnose``."""

    status: DiagnosticStatusValue
    problems: list[DiagnosticProblem]
    recent_runs: list[RecentRunSummary]
    recommendations: list[str]
    next_steps: list[str]


class FailedNodeState(TypedDict):
    """A failed node state in failure context."""

    state_id: str
    token_id: str
    plugin: str
    type: NodeTypeValue
    step: int
    attempt: int
    started: str | None


class FailureTransformError(TypedDict):
    """A transform error in failure context."""

    token_id: str
    plugin: str
    details: dict[str, Any] | None


class FailureValidationError(TypedDict):
    """A validation error in failure context."""

    plugin: str | None
    row_hash: str | None
    sample_data: dict[str, Any] | None


class FailurePatterns(TypedDict):
    """Patterns identified in failure analysis."""

    plugins_failing: list[str]
    has_retries: bool
    failure_count: int
    transform_error_count: int
    validation_error_count: int


class FailureContextReport(TypedDict):
    """Return type for ``get_failure_context``."""

    run_id: str
    run_status: RunStatusValue
    failed_node_states: list[FailedNodeState]
    transform_errors: list[FailureTransformError]
    validation_errors: list[FailureValidationError]
    patterns: FailurePatterns
    next_steps: list[str]


class RecentRunDetail(TypedDict):
    """A run in the recent activity timeline."""

    run_id: str
    full_run_id: str
    status: RunStatusValue
    started: str | None
    duration_seconds: float | None
    rows_processed: int
    node_executions: int


class RecentActivityReport(TypedDict):
    """Return type for ``get_recent_activity``."""

    time_window_minutes: int
    total_runs: int
    status_summary: dict[str, int]  # dynamic status names
    runs: list[RecentRunDetail]


class ContractField(TypedDict):
    """A field in the schema contract."""

    normalized_name: str
    original_name: str
    python_type: str
    required: bool
    nullable: bool
    source: str


class RunContractReport(TypedDict):
    """Return type for ``get_run_contract``."""

    run_id: str
    mode: ContractModeValue
    locked: bool
    fields: list[ContractField]
    field_count: int
    version_hash: str


class FieldExplanation(TypedDict):
    """Return type for ``explain_field``."""

    run_id: str
    normalized_name: str
    original_name: str
    python_type: str
    required: bool
    nullable: bool
    source: str
    contract_mode: ContractModeValue


class ContractViolationRecord(TypedDict):
    """A single contract violation."""

    error_id: str
    violation_type: ContractViolationTypeValue
    normalized_field_name: str | None
    original_field_name: str | None
    expected_type: str | None
    actual_type: str | None
    error: str
    schema_mode: SchemaModeValue
    destination: str
    created_at: str


class ContractViolationsReport(TypedDict):
    """Return type for ``list_contract_violations``."""

    run_id: str
    total_violations: int
    violations: list[ContractViolationRecord]
    limit: int


class DivertSummary(TypedDict):
    """Summary of a divert (quarantine/error routing)."""

    diverted: bool
    divert_type: DivertTypeValue
    from_node: str
    to_sink: str
    edge_label: str
    reason_hash: str | None


class ExplainTokenResult(TypedDict):
    """Return type for ``explain_token``.

    Top-level keys are typed but deeply nested sub-structures
    (routing_events items, node_states items) use ``dict[str, Any]``
    because they come from ``dataclass_to_dict()`` on complex nested
    dataclasses -- typing them fully would duplicate the entire audit
    dataclass hierarchy.
    """

    token: dict[str, Any]
    source_row: dict[str, Any]
    node_states: list[dict[str, Any]]
    routing_events: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    parent_tokens: list[dict[str, Any]]
    validation_errors: list[dict[str, Any]]
    transform_errors: list[dict[str, Any]]
    outcome: dict[str, Any] | None
    divert_summary: DivertSummary | None


class ErrorResult(TypedDict):
    """Returned for early-exit ``{"error": "..."}`` paths."""

    error: str


class FieldNotFoundError(TypedDict):
    """Returned when a field is not found in ``explain_field``."""

    error: str
    available_fields: list[str]
