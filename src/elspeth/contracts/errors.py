"""Error and reason schema contracts.

TypedDict schemas for structured error payloads in the audit trail.
These provide consistent shapes for executor error recording.
"""

from typing import Any, Literal, NotRequired, TypedDict


class ExecutionError(TypedDict):
    """Schema for execution error payloads.

    Used by executors when recording node state failures.
    """

    exception: str  # String representation of the exception
    type: str  # Exception class name (e.g., "ValueError")
    traceback: NotRequired[str]  # Optional full traceback
    phase: NotRequired[str]  # Optional phase indicator (e.g., "flush" for sink flush errors)


class CoalesceFailureReason(TypedDict, total=False):
    """Schema for coalesce/barrier failure payloads.

    Used by CoalesceExecutor when recording fork-join barrier failures.
    These are internal engine errors, not transform or plugin errors.
    """

    failure_reason: str  # Why coalesce failed (e.g., "late_arrival_after_merge")
    waiting_tokens: list[str]  # Token IDs still waiting at barrier
    barrier: str  # Barrier identifier
    expected_branches: list[str]  # Branches expected to arrive
    actual_branches: list[str]  # Branches that actually arrived
    branches_arrived: list[str]  # Alias for actual_branches (backwards compat)
    merge_policy: str  # Merge policy in effect
    timeout_ms: int  # Timeout that triggered failure
    select_branch: str | None  # Target branch for select merge policy


class ConfigGateReason(TypedDict):
    """Reason from config-driven gate (expression evaluation).

    Used by gates defined via GateSettings with condition expressions.
    The executor auto-generates this reason structure at executors.py:739.

    Fields:
        condition: The expression that was evaluated (e.g., "row['score'] > 100")
        result: The route label that matched (e.g., "true", "false")
    """

    condition: str
    result: str


class PluginGateReason(TypedDict):
    """Reason from plugin-based gate.

    Used by custom gate plugins implementing GateProtocol.
    Enforces minimum auditability: every routing decision MUST have
    a rule description and the value that triggered it.

    Required fields:
        rule: Human-readable description of what logic fired
        matched_value: The value that triggered the routing decision

    Optional fields (for threshold-style gates):
        threshold: The threshold value compared against
        field: The field name that was compared
        comparison: The comparison operator used (">", "<", ">=", etc.)
    """

    rule: str
    matched_value: Any
    threshold: NotRequired[float]
    field: NotRequired[str]
    comparison: NotRequired[str]


# Discriminated union - field presence distinguishes variants:
# - ConfigGateReason has "condition" and "result"
# - PluginGateReason has "rule" and "matched_value"
RoutingReason = ConfigGateReason | PluginGateReason


# Literal type for common transform actions (extensible - str also accepted)
TransformActionCategory = Literal[
    # Processing actions
    "processed",  # Generic successful processing
    "mapped",  # Field mapping completed
    "validated",  # Validation passed
    "enriched",  # Data enrichment from external source
    "transformed",  # Data transformation applied
    "normalized",  # Data normalization applied
    "filtered",  # Row passed filter criteria
    "classified",  # Classification assigned
    # Skip/passthrough actions
    "passthrough",  # No changes made (intentional)
    "skipped",  # Processing skipped (e.g., data already present)
    "cached",  # Result retrieved from cache
]


class TransformSuccessReason(TypedDict):
    """Metadata for successful transform operations.

    Provides structured audit information about what a transform did,
    beyond just the input/output data. This enables:
    - Efficient audit queries (fields_modified without diffing)
    - Data quality monitoring (validation_warnings for non-blocking warnings)
    - Conditional path tracking (action distinguishes code paths)

    Used when transforms return TransformResult.success() with optional
    success_reason parameter.

    Required field:
        action: What the transform did. Use TransformActionCategory values
                for common actions, or custom strings for plugin-specific actions.

    Optional fields:
        fields_modified: List of field names that were changed
        fields_added: List of field names that were added
        fields_removed: List of field names that were removed
        validation_warnings: Non-blocking validation issues (data quality flags)
        metadata: Additional plugin-specific context

    Example usage:
        # Simple action tracking
        TransformResult.success(row, success_reason={"action": "enriched"})

        # Field change tracking
        TransformResult.success(row, success_reason={
            "action": "mapped",
            "fields_modified": ["customer_id", "amount"],
            "fields_added": ["currency_code"],
        })

        # Data quality warning
        TransformResult.success(row, success_reason={
            "action": "validated",
            "validation_warnings": ["amount near threshold (995 of 1000 limit)"],
        })
    """

    action: str  # Use TransformActionCategory or custom string

    # Field tracking
    fields_modified: NotRequired[list[str]]
    fields_added: NotRequired[list[str]]
    fields_removed: NotRequired[list[str]]

    # Data quality
    validation_warnings: NotRequired[list[str]]

    # Extensibility
    metadata: NotRequired[dict[str, Any]]


# =============================================================================
# Transform Error Reason Types
# =============================================================================


class TemplateErrorEntry(TypedDict):
    """Entry in template_errors list for batch processing failures."""

    row_index: int
    error: str


class RowErrorEntry(TypedDict):
    """Entry in row_errors list for batch processing failures."""

    row_index: int
    reason: str
    error: NotRequired[str]


class UsageStats(TypedDict, total=False):
    """LLM token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QueryFailureDetail(TypedDict):
    """Detailed information about a failed query in multi-query transforms.

    Used when transforms need to report more than just the query name.
    """

    query: str
    error: NotRequired[str]
    error_type: NotRequired[str]
    status_code: NotRequired[int]


class ErrorDetail(TypedDict):
    """Detailed information about an error in batch processing.

    Used when more context is needed than a simple error message string.
    """

    message: str
    error_type: NotRequired[str]
    row_index: NotRequired[int]
    details: NotRequired[str]


# Literal type for compile-time validation of error categories
TransformErrorCategory = Literal[
    # API/Network errors
    "api_error",
    "api_call_failed",
    "llm_call_failed",
    "llm_retryable_error_no_retry",  # LLM error that would be retried but retry disabled
    "network_error",
    "permanent_error",
    "retry_timeout",
    "transient_error_no_retry",  # Transient error (connection/timeout) but retry disabled
    # Field/validation errors
    "missing_field",
    "type_mismatch",
    "validation_failed",
    "invalid_input",
    # Template errors
    "template_rendering_failed",
    "all_templates_failed",
    # JSON/response parsing errors
    "json_parse_failed",
    "invalid_json",  # Generic JSON parse failure
    "invalid_json_response",
    "invalid_json_type",
    "empty_choices",
    "malformed_response",
    "missing_output_field",
    "response_truncated",
    # Batch processing errors
    "batch_error",
    "batch_create_failed",
    "batch_failed",
    "batch_cancelled",
    "batch_expired",
    "batch_timeout",
    "batch_retrieve_failed",
    "file_upload_failed",
    "file_download_failed",
    "all_output_lines_malformed",
    "all_rows_failed",
    "result_not_found",
    "query_failed",
    "rate_limited",
    # Content filtering
    "blocked_content",
    "content_filtered",
    "content_safety_violation",
    "prompt_injection_detected",
    # Generic (for tests and edge cases)
    "test_error",
    "property_test_error",
    "simulated_failure",
    "deliberate_failure",
    "intentional_failure",
]


class TransformErrorReason(TypedDict):
    """Reason for transform processing error.

    Used when transforms return TransformResult.error().
    The reason field describes what category of error occurred.
    Additional fields provide context specific to the error type.

    Recorded in the audit trail (Tier 1 - full trust) for legal traceability.
    Every transform error must be attributable to its cause.

    Required field:
        reason: Error category from TransformErrorCategory literal type.
                Compile-time validated to prevent typos.

    Common context fields:
        error: Exception message or detailed error description
        field: Field name for field-related errors
        error_type: Sub-category (e.g., "http_error", "network_error")
        message: Human-readable error message (alternative to error)

    Multi-query/template context:
        query: Which query in multi-query failed
        template_hash: Template version for debugging
        template_errors: List of per-row template failures (batch processing)

    LLM response context:
        max_tokens: Configured max tokens limit
        completion_tokens: Actual tokens used in response
        prompt_tokens: Tokens used in prompt
        raw_response: Truncated raw LLM response content
        raw_response_preview: Alternative name for truncated preview
        content_after_fence_strip: Content after markdown fence removal
        usage: Token usage stats from LLM response
        response: Full response object for debugging
        response_keys: Keys present in response dict
        body_preview: HTTP body preview for errors
        content_type: Content-Type header value

    Type validation context:
        expected: Expected type or value
        actual: Actual type or value received
        value: The actual value (truncated for audit)

    Rate limiting/timeout context:
        elapsed_seconds: Time elapsed before timeout
        max_seconds: Maximum allowed time
        status_code: HTTP status code

    Content filtering context:
        matched_pattern: Regex pattern that matched
        match_context: Context around the match
        categories: Content safety violation categories

    Batch processing context:
        batch_id: Azure/OpenRouter batch job ID
        queries_completed: Number of queries completed before failure
        row_errors: List of per-row error entries

    Example usage:
        # API error with exception details
        TransformResult.error({
            "reason": "api_error",
            "error": str(e),
            "error_type": "http_error",
        })

        # Field-related error
        TransformResult.error({
            "reason": "missing_field",
            "field": "customer_id",
        })

        # LLM response truncation
        TransformResult.error({
            "reason": "response_truncated",
            "error": "Response was truncated at 1000 tokens",
            "query": "sentiment",
            "max_tokens": 1000,
            "completion_tokens": 1000,
        })
    """

    # REQUIRED - error category (Literal-typed for compile-time validation)
    reason: TransformErrorCategory

    # Common context
    error: NotRequired[str]
    field: NotRequired[str]
    error_type: NotRequired[str]
    message: NotRequired[str]

    # Multi-query/template context
    query: NotRequired[str]
    template_hash: NotRequired[str]
    template_file_path: NotRequired[str]  # Path to template file; absent = inline template
    template_errors: NotRequired[list[TemplateErrorEntry]]
    failed_queries: NotRequired[list[str | QueryFailureDetail]]  # Query names or detailed failures
    succeeded_count: NotRequired[int]  # Number of successful queries
    total_count: NotRequired[int]  # Total number of queries attempted

    # LLM response context
    max_tokens: NotRequired[int]
    completion_tokens: NotRequired[int]
    prompt_tokens: NotRequired[int]
    raw_response: NotRequired[str]  # LLM response text; absent = empty/unavailable
    raw_response_preview: NotRequired[str]  # Truncated preview; absent = empty/unavailable
    content_after_fence_strip: NotRequired[str]
    usage: NotRequired[UsageStats | dict[str, int]]
    response: NotRequired[dict[str, Any]]
    response_keys: NotRequired[list[str] | None]
    body_preview: NotRequired[str]  # HTTP body preview; absent = empty/unavailable
    content_type: NotRequired[str]

    # Type validation context
    expected: NotRequired[str]
    actual: NotRequired[str]
    value: NotRequired[str]

    # Rate limiting/timeout context
    elapsed_seconds: NotRequired[float]
    max_seconds: NotRequired[float]
    elapsed_hours: NotRequired[float]  # Batch timeout (hours scale)
    max_wait_hours: NotRequired[float]  # Batch max wait time (hours)
    status_code: NotRequired[int]

    # Content filtering context
    matched_pattern: NotRequired[str]
    match_context: NotRequired[str]
    categories: NotRequired[list[str] | dict[str, dict[str, Any]]]  # List of names OR detailed severity/threshold map
    attacks: NotRequired[dict[str, bool]]  # Prompt shield attack flags (user_prompt_attack, document_attack)

    # Batch processing context
    batch_id: NotRequired[str]
    queries_completed: NotRequired[int]
    row_errors: NotRequired[list[RowErrorEntry]]
    output_file_id: NotRequired[str]  # Batch output file reference
    malformed_count: NotRequired[int]  # Count of malformed batch lines
    errors: NotRequired[list[str | ErrorDetail]]  # Error messages or structured errors


# =============================================================================
# Control Flow Exceptions
# =============================================================================


class BatchPendingError(Exception):
    """Raised when batch is submitted but not yet complete.

    This is NOT an error condition - it's a control flow signal
    telling the engine to schedule a retry check later.

    The exception carries the checkpoint state so the caller can persist
    it and restore it when scheduling a retry. This enables crash recovery
    and correct resume behavior.

    Attributes:
        batch_id: Azure batch job ID
        status: Current batch status (e.g., "submitted", "in_progress")
        check_after_seconds: When to check again (default 300s = 5 min)
        checkpoint: Checkpoint data to persist for retry (batch_id, row_mapping, etc.)
        node_id: Transform node ID that raised this (for checkpoint keying)

    Example:
        # Phase 1: Submit batch
        batch_id = client.batches.create(...)
        checkpoint_data = {"batch_id": batch_id, "row_mapping": {...}}
        ctx.update_checkpoint(checkpoint_data)
        raise BatchPendingError(
            batch_id, "submitted",
            check_after_seconds=300,
            checkpoint=checkpoint_data,
            node_id=self.node_id,
        )

        # Caller catches, persists checkpoint, schedules retry

        # Phase 2: Resume and check (caller passes checkpoint back via orchestrator)
        checkpoint = ctx.get_checkpoint()
        if checkpoint.get("batch_id"):
            status = client.batches.retrieve(batch_id).status
            if status == "in_progress":
                raise BatchPendingError(batch_id, "in_progress", checkpoint=checkpoint)
            elif status == "completed":
                # Download results and return
    """

    def __init__(
        self,
        batch_id: str,
        status: str,
        *,
        check_after_seconds: int = 300,
        checkpoint: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> None:
        """Initialize BatchPendingError.

        Args:
            batch_id: Azure batch job ID
            status: Current batch status
            check_after_seconds: Seconds until next check (default 300)
            checkpoint: Checkpoint data for retry (caller should persist this)
            node_id: Transform node ID (for checkpoint keying)
        """
        self.batch_id = batch_id
        self.status = status
        self.check_after_seconds = check_after_seconds
        self.checkpoint = checkpoint
        self.node_id = node_id
        super().__init__(f"Batch {batch_id} is {status}, check after {check_after_seconds}s")


class AuditIntegrityError(Exception):
    """Raised when audit database operations fail unexpectedly.

    This indicates catastrophic failure: database corruption, transaction
    failure, or bug in Landscape recording code. Per Tier 1 trust model,
    ELSPETH cannot continue with compromised audit integrity.

    Examples of conditions that trigger this:
    - Run record not found after INSERT/UPDATE
    - NodeState not found after completion update
    - Batch record not found after update
    - State marked terminal but still shows OPEN status

    Recovery: These errors are NOT recoverable. The pipeline must stop
    and the database must be investigated for corruption or tampering.
    """

    pass


class OrchestrationInvariantError(Exception):
    """Raised when orchestration invariants are violated.

    This indicates a bug in the engine: node_id not set before execution,
    routing to non-existent edges, etc. These are programming errors
    that must crash immediately.

    Examples of conditions that trigger this:
    - Transform/Gate/Sink executed without node_id assigned
    - Edge resolution fails due to missing node_id
    - Last transform in pipeline has no node_id

    Recovery: These errors indicate bugs in orchestration code that must
    be fixed. They should never occur in correct operation.
    """

    pass


class FrameworkBugError(Exception):
    """Raised when the framework encounters an internal inconsistency.

    This indicates a bug in ELSPETH itself, not user error or external failure.
    Unlike OrchestrationInvariantError (specific to orchestration flow), this
    is a general-purpose exception for any framework-level bug.

    Examples of conditions that trigger this:
    - Double-completing an operation (already completed, trying to complete again)
    - Missing required context (record_call with neither state_id nor operation_id)
    - Completing a non-existent operation

    Recovery: These errors indicate bugs in framework code that must be fixed.
    They should never occur in correct operation.
    """

    pass


class PluginContractViolation(RuntimeError):
    """Raised when a plugin violates its contract with the framework.

    This indicates a bug in a plugin (Source, Transform, Gate, Sink) that must
    be fixed. Unlike user data errors (which are quarantined), plugin bugs
    MUST crash the pipeline per CLAUDE.md's "plugin bugs must crash" rule.

    Examples of conditions that trigger this:
    - Transform emits non-canonical data (NaN, Infinity, non-serializable types)
    - Plugin returns wrong type from method
    - Plugin violates interface contract

    Recovery: Fix the plugin. These errors indicate bugs in plugin code.
    """

    pass
