"""Sink-specific contracts for cross-boundary data types.

This module defines contracts for sink validation, output target compatibility,
and public sink capability policy shared by runtime and composer layers.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class SinkCapabilities:
    """Public policy metadata for a sink plugin family.

    Plugin discovery remains owned by the plugin manager. This registry captures
    cross-layer policy that must stay consistent before plugin instances exist.
    """

    requires_path_option: bool = False
    eligible_as_failsink: bool = False
    local_recovery_file: bool = False
    default_file_extension: str | None = None


SINK_CAPABILITIES_BY_PLUGIN: Final[Mapping[str, SinkCapabilities]] = MappingProxyType(
    {
        "csv": SinkCapabilities(
            requires_path_option=True,
            eligible_as_failsink=True,
            local_recovery_file=True,
            default_file_extension="csv",
        ),
        "json": SinkCapabilities(
            requires_path_option=True,
            eligible_as_failsink=True,
            local_recovery_file=True,
            default_file_extension="json",
        ),
        "text": SinkCapabilities(
            requires_path_option=True,
            eligible_as_failsink=False,
            local_recovery_file=False,
            default_file_extension="txt",
        ),
    }
)


def _plugins_requiring_path() -> frozenset[str]:
    return frozenset(plugin_name for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items() if capabilities.requires_path_option)


def _failsink_eligible_plugins() -> frozenset[str]:
    return frozenset(plugin_name for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items() if capabilities.eligible_as_failsink)


def _local_recovery_plugins() -> frozenset[str]:
    return frozenset(plugin_name for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items() if capabilities.local_recovery_file)


def _file_sink_repair_extensions() -> dict[str, str]:
    extensions: dict[str, str] = {}
    for plugin_name, capabilities in SINK_CAPABILITIES_BY_PLUGIN.items():
        extension = capabilities.default_file_extension
        if extension is not None:
            extensions[plugin_name] = extension
    return extensions


FILE_SINK_PLUGINS: Final[frozenset[str]] = _plugins_requiring_path()
FAILSINK_ELIGIBLE_SINK_PLUGINS: Final[frozenset[str]] = _failsink_eligible_plugins()
LOCAL_RECOVERY_SINK_PLUGINS: Final[frozenset[str]] = _local_recovery_plugins()
FILE_SINK_REPAIR_EXTENSIONS: Final[Mapping[str, str]] = MappingProxyType(_file_sink_repair_extensions())


def format_sink_plugin_names(plugin_names: Collection[str]) -> str:
    """Format a plugin-name set for validation and composer messages."""

    ordered = tuple(sorted(plugin_names))
    if not ordered:
        return "no plugins"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} or {ordered[1]}"
    return f"{', '.join(ordered[:-1])}, or {ordered[-1]}"


FAILSINK_ELIGIBLE_PLUGIN_TEXT: Final[str] = format_sink_plugin_names(FAILSINK_ELIGIBLE_SINK_PLUGINS)
FILE_SINK_PLUGIN_TEXT: Final[str] = format_sink_plugin_names(FILE_SINK_PLUGINS)
FILE_SINK_PLUGIN_SLASH_TEXT: Final[str] = "/".join(sorted(FILE_SINK_PLUGINS))


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
