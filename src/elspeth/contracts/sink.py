"""Sink-specific contracts for cross-boundary data types.

This module defines contracts for sink validation and output target compatibility.
"""

from dataclasses import dataclass, field

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    """Result of sink output target validation.

    Used to report whether an existing output target (file, table, etc.)
    is compatible with the configured schema for append/resume operations.

    This is a value object - immutable once created. Use factory methods
    `success()` and `failure()` for clean construction.

    Attributes:
        valid: True if output target matches schema (or no validation needed)
        target_fields: Fields found in existing output target
        schema_fields: Fields defined in schema configuration
        missing_fields: Schema fields not present in target
        extra_fields: Target fields not present in schema (strict mode)
        order_mismatch: True if fields match but order differs (CSV strict mode)
        error_message: Human-readable error description for failures
    """

    valid: bool
    target_fields: tuple[str, ...] = field(default_factory=tuple)
    schema_fields: tuple[str, ...] = field(default_factory=tuple)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    extra_fields: tuple[str, ...] = field(default_factory=tuple)
    order_mismatch: bool = False
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate consistency between valid flag and diagnostic fields."""
        if not self.valid and not self.error_message:
            raise ValueError("OutputValidationResult with valid=False must have error_message")
        if self.valid and self.error_message is not None:
            raise ValueError(f"OutputValidationResult with valid=True must not have error_message, got: {self.error_message!r}")
        if self.valid and self.missing_fields:
            raise ValueError(f"OutputValidationResult with valid=True must not have missing_fields, got: {self.missing_fields!r}")
        if self.valid and self.extra_fields:
            raise ValueError(f"OutputValidationResult with valid=True must not have extra_fields, got: {self.extra_fields!r}")
        if self.valid and self.order_mismatch:
            raise ValueError("OutputValidationResult with valid=True must not have order_mismatch=True")
        freeze_fields(self, "target_fields", "schema_fields", "missing_fields", "extra_fields")

    @classmethod
    def success(cls, target_fields: list[str] | None = None) -> "OutputValidationResult":
        """Create a successful validation result.

        Args:
            target_fields: Fields found in existing output target (if any)

        Returns:
            OutputValidationResult with valid=True
        """
        return cls(
            valid=True,
            target_fields=tuple(target_fields) if target_fields else (),
        )

    @classmethod
    def failure(
        cls,
        message: str,
        *,
        target_fields: list[str] | None = None,
        schema_fields: list[str] | None = None,
        missing_fields: list[str] | None = None,
        extra_fields: list[str] | None = None,
        order_mismatch: bool = False,
    ) -> "OutputValidationResult":
        """Create a failed validation result with diagnostic details.

        Args:
            message: Human-readable error description
            target_fields: Fields found in existing output target
            schema_fields: Fields defined in schema configuration
            missing_fields: Schema fields not present in target
            extra_fields: Target fields not present in schema
            order_mismatch: True if fields match but order differs

        Returns:
            OutputValidationResult with valid=False and diagnostic info
        """
        return cls(
            valid=False,
            target_fields=tuple(target_fields) if target_fields else (),
            schema_fields=tuple(schema_fields) if schema_fields else (),
            missing_fields=tuple(missing_fields) if missing_fields else (),
            extra_fields=tuple(extra_fields) if extra_fields else (),
            order_mismatch=order_mismatch,
            error_message=message,
        )
