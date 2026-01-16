"""Error and reason schema contracts.

TypedDict schemas for structured error payloads in the audit trail.
These provide consistent shapes for executor error recording.
"""

from typing import Any, NotRequired, TypedDict


class ExecutionError(TypedDict):
    """Schema for execution error payloads.

    Used by executors when recording node state failures.
    """

    exception: str  # String representation of the exception
    type: str  # Exception class name (e.g., "ValueError")
    traceback: NotRequired[str]  # Optional full traceback


class RoutingReason(TypedDict):
    """Schema for gate routing reason payloads.

    Used by gates to explain routing decisions in audit trail.
    """

    rule: str  # Human-readable rule description
    matched_value: Any  # The value that triggered the route
    threshold: NotRequired[float]  # Threshold value if applicable
    field: NotRequired[str]  # Field name if applicable
    comparison: NotRequired[str]  # Comparison operator used


class TransformReason(TypedDict):
    """Schema for transform reason payloads.

    Used by transforms to explain processing decisions.
    """

    action: str  # What the transform did
    fields_modified: NotRequired[list[str]]  # Fields that were changed
    validation_errors: NotRequired[list[str]]  # Any validation issues
