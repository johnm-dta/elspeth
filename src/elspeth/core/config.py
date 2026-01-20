# src/elspeth/core/config.py
"""
Configuration schema and loading for Elspeth pipelines.

Uses Pydantic for validation and Dynaconf for multi-source loading.
Settings are frozen (immutable) after construction.
"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts.enums import RunMode

# Compiled regex for validating route destination identifiers
_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Reserved edge labels that cannot be used as route labels or fork branch names.
# "continue" is used by the DAG builder for edges between sequential nodes.
# Using these as user-defined labels would cause edge_map collisions in the orchestrator,
# leading to routing events recorded against wrong edges (audit corruption).
_RESERVED_EDGE_LABELS = frozenset({"continue"})


class SecretFingerprintError(Exception):
    """Raised when secrets are found but cannot be fingerprinted.

    This occurs when:
    - Secret-like field names are found in config
    - ELSPETH_FINGERPRINT_KEY is not set
    - ELSPETH_ALLOW_RAW_SECRETS is not set to 'true'
    """

    pass


class TriggerConfig(BaseModel):
    """Trigger configuration for aggregation batches.

    Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
    The engine evaluates all configured triggers after each accept and fires when
    ANY condition is met.

    Trigger types:
    - count: Fire after N rows accumulated
    - timeout: Fire after N seconds since first accept
    - condition: Fire when expression evaluates to true

    Note: end_of_source is IMPLICIT - always checked at source exhaustion.
    It is not configured here because it always applies.

    Example YAML (combined triggers):
        trigger:
          count: 1000           # Fire after 1000 rows
          timeout_seconds: 3600         # Or after 1 hour
          condition: "row['type'] == 'flush_signal'"  # Or on special row
    """

    model_config = {"frozen": True}

    count: int | None = Field(
        default=None,
        gt=0,
        description="Fire after N rows accumulated",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Fire after N seconds since first accept",
    )
    condition: str | None = Field(
        default=None,
        description="Fire when expression evaluates to true",
    )

    @field_validator("condition")
    @classmethod
    def validate_condition_expression(cls, v: str | None) -> str | None:
        """Validate condition is a valid expression at config time."""
        if v is None:
            return v

        from elspeth.engine.expression_parser import (
            ExpressionParser,
            ExpressionSecurityError,
            ExpressionSyntaxError,
        )

        try:
            ExpressionParser(v)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Forbidden construct in condition: {e}") from e
        return v

    @model_validator(mode="after")
    def validate_at_least_one_trigger(self) -> "TriggerConfig":
        """At least one trigger must be configured."""
        if self.count is None and self.timeout_seconds is None and self.condition is None:
            raise ValueError("at least one trigger must be configured (count, timeout_seconds, or condition)")
        return self

    @property
    def has_count(self) -> bool:
        """Whether count trigger is configured."""
        return self.count is not None

    @property
    def has_timeout(self) -> bool:
        """Whether timeout trigger is configured."""
        return self.timeout_seconds is not None

    @property
    def has_condition(self) -> bool:
        """Whether condition trigger is configured."""
        return self.condition is not None


class AggregationSettings(BaseModel):
    """Aggregation configuration for batching rows.

    Aggregations collect rows until a trigger fires, then process the batch.
    The engine evaluates trigger conditions - plugins only accept/reject rows.

    Output modes:
    - single: Batch produces one aggregated result row
    - passthrough: Batch releases all accepted rows unchanged
    - transform: Batch applies a transform function to produce results

    Example YAML:
        aggregations:
          - name: batch_stats
            plugin: stats_aggregation
            trigger:
              count: 100
            output_mode: single
            options:
              fields: ["value"]
              compute_mean: true
    """

    model_config = {"frozen": True}

    name: str = Field(description="Aggregation identifier (unique within pipeline)")
    plugin: str = Field(description="Plugin name to instantiate")
    trigger: TriggerConfig = Field(description="When to flush the batch")
    output_mode: Literal["single", "passthrough", "transform"] = Field(
        default="single",
        description="How batch produces output rows",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class GateSettings(BaseModel):
    """Gate configuration for config-driven routing.

    Gates are defined in YAML and evaluated by the engine using ExpressionParser.
    The condition expression determines routing; route labels map to destinations.

    Example YAML:
        gates:
          - name: quality_check
            condition: "row['confidence'] >= 0.85"
            routes:
              high: continue
              low: review_sink
          - name: parallel_analysis
            condition: "True"
            routes:
              all: fork
            fork_to:
              - path_a
              - path_b
    """

    model_config = {"frozen": True}

    name: str = Field(description="Gate identifier (unique within pipeline)")
    condition: str = Field(description="Expression to evaluate (validated by ExpressionParser)")
    routes: dict[str, str] = Field(description="Maps route labels to destinations ('continue' or sink name)")
    fork_to: list[str] | None = Field(
        default=None,
        description="List of paths for fork operations",
    )

    @field_validator("condition")
    @classmethod
    def validate_condition_expression(cls, v: str) -> str:
        """Validate that condition is a valid expression at config time."""
        from elspeth.engine.expression_parser import (
            ExpressionParser,
            ExpressionSecurityError,
            ExpressionSyntaxError,
        )

        try:
            ExpressionParser(v)
        except ExpressionSyntaxError as e:
            raise ValueError(f"Invalid condition syntax: {e}") from e
        except ExpressionSecurityError as e:
            raise ValueError(f"Forbidden construct in condition: {e}") from e
        return v

    @field_validator("routes")
    @classmethod
    def validate_routes(cls, v: dict[str, str]) -> dict[str, str]:
        """Routes must have at least one entry with valid destinations.

        Also validates that route labels don't use reserved edge labels,
        which would cause edge_map collisions in the orchestrator.
        """
        if not v:
            raise ValueError("routes must have at least one entry")

        for label, destination in v.items():
            # Check route label is not reserved
            if label in _RESERVED_EDGE_LABELS:
                raise ValueError(f"Route label '{label}' is reserved and cannot be used. Reserved labels: {sorted(_RESERVED_EDGE_LABELS)}")

            if destination in ("continue", "fork"):
                continue
            if not _IDENTIFIER_PATTERN.match(destination):
                raise ValueError(f"Route destination '{destination}' for label '{label}' must be 'continue', 'fork', or a valid identifier")
        return v

    @field_validator("fork_to")
    @classmethod
    def validate_fork_to_labels(cls, v: list[str] | None) -> list[str] | None:
        """Validate fork branch names don't use reserved edge labels.

        Fork branches become edge labels in the DAG, so they must not collide
        with reserved labels like 'continue'.
        """
        if v is None:
            return v

        for branch in v:
            if branch in _RESERVED_EDGE_LABELS:
                raise ValueError(f"Fork branch '{branch}' is reserved and cannot be used. Reserved labels: {sorted(_RESERVED_EDGE_LABELS)}")
        return v

    @model_validator(mode="after")
    def validate_fork_consistency(self) -> "GateSettings":
        """Ensure fork_to is provided when routes use 'fork' destination."""
        has_fork_route = any(dest == "fork" for dest in self.routes.values())
        if has_fork_route and not self.fork_to:
            raise ValueError("fork_to is required when any route destination is 'fork'")
        if self.fork_to and not has_fork_route:
            raise ValueError("fork_to is only valid when a route destination is 'fork'")
        return self

    @model_validator(mode="after")
    def validate_boolean_routes(self) -> "GateSettings":
        """Validate route labels match the condition's return type.

        Boolean expressions (comparisons, and/or, not) must use "true"/"false"
        as route labels. Using labels like "above"/"below" for a condition like
        `row['amount'] > 1000` is a config error - the expression evaluates to
        True/False, not "above"/"below".
        """
        from elspeth.engine.expression_parser import ExpressionParser

        parser = ExpressionParser(self.condition)
        if parser.is_boolean_expression():
            route_labels = set(self.routes.keys())
            expected_labels = {"true", "false"}

            # Check for common mistakes
            if route_labels != expected_labels:
                missing = expected_labels - route_labels
                extra = route_labels - expected_labels

                # Build helpful error message
                msg_parts = [f"Gate '{self.name}' has a boolean condition ({self.condition!r}) but route labels don't match."]

                if extra:
                    msg_parts.append(f"Found labels {sorted(extra)!r} but boolean expressions evaluate to True/False, not these values.")
                if missing:
                    msg_parts.append(f"Missing required labels: {sorted(missing)!r}.")
                msg_parts.append('Use routes: {"true": <destination>, "false": <destination>}')

                raise ValueError(" ".join(msg_parts))

        return self


class CoalesceSettings(BaseModel):
    """Configuration for coalesce (token merging) operations.

    Coalesce merges tokens from parallel fork paths back into a single token.
    Tokens are correlated by row_id (same source row that was forked).

    Example YAML:
        coalesce:
          - name: merge_analysis
            branches:
              - sentiment_path
              - entity_path
            policy: require_all
            merge: union

          - name: quorum_merge
            branches:
              - fast_model
              - slow_model
              - fallback_model
            policy: quorum
            quorum_count: 2
            merge: nested
            timeout_seconds: 30
    """

    model_config = {"frozen": True}

    name: str = Field(description="Unique identifier for this coalesce point")
    branches: list[str] = Field(
        min_length=2,
        description="Branch names to wait for (from fork_to paths)",
    )
    policy: Literal["require_all", "quorum", "best_effort", "first"] = Field(
        default="require_all",
        description="How to handle partial arrivals",
    )
    merge: Literal["union", "nested", "select"] = Field(
        default="union",
        description="How to combine row data from branches",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Max wait time (required for best_effort, optional for quorum)",
    )
    quorum_count: int | None = Field(
        default=None,
        gt=0,
        description="Minimum branches required (required for quorum policy)",
    )
    select_branch: str | None = Field(
        default=None,
        description="Which branch to take for 'select' merge strategy",
    )

    @model_validator(mode="after")
    def validate_policy_requirements(self) -> "CoalesceSettings":
        """Validate policy-specific requirements."""
        if self.policy == "quorum" and self.quorum_count is None:
            raise ValueError(f"Coalesce '{self.name}': quorum policy requires quorum_count")
        if self.policy == "quorum" and self.quorum_count is not None and self.quorum_count > len(self.branches):
            raise ValueError(
                f"Coalesce '{self.name}': quorum_count ({self.quorum_count}) cannot exceed number of branches ({len(self.branches)})"
            )
        if self.policy == "best_effort" and self.timeout_seconds is None:
            raise ValueError(f"Coalesce '{self.name}': best_effort policy requires timeout_seconds")
        return self

    @model_validator(mode="after")
    def validate_merge_requirements(self) -> "CoalesceSettings":
        """Validate merge strategy requirements."""
        if self.merge == "select" and self.select_branch is None:
            raise ValueError(f"Coalesce '{self.name}': select merge strategy requires select_branch")
        if self.select_branch is not None and self.select_branch not in self.branches:
            raise ValueError(
                f"Coalesce '{self.name}': select_branch '{self.select_branch}' must be one of the expected branches: {self.branches}"
            )
        return self


class DatasourceSettings(BaseModel):
    """Source plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv_local, json, http_poll, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class RowPluginSettings(BaseModel):
    """Transform plugin configuration per architecture.

    Note: Gate routing is now config-driven only (see GateSettings).
    Plugin-based gates were removed - use the gates: section instead.
    """

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class LandscapeExportSettings(BaseModel):
    """Landscape export configuration for audit compliance.

    Exports audit trail to a configured sink after run completes.
    Optional cryptographic signing for legal-grade integrity.
    """

    model_config = {"frozen": True}

    enabled: bool = Field(
        default=False,
        description="Enable audit trail export after run completes",
    )
    sink: str | None = Field(
        default=None,
        description="Sink name to export to (must be defined in sinks)",
    )
    format: Literal["csv", "json"] = Field(
        default="csv",
        description="Export format: csv (human-readable) or json (machine)",
    )
    sign: bool = Field(
        default=False,
        description="HMAC sign each record for integrity verification",
    )


class LandscapeSettings(BaseModel):
    """Landscape audit system configuration per architecture."""

    model_config = {"frozen": True}

    enabled: bool = Field(default=True, description="Enable audit trail recording")
    backend: Literal["sqlite", "postgresql"] = Field(
        default="sqlite",
        description="Database backend type",
    )
    # NOTE: Using str instead of Path - Path mangles PostgreSQL DSNs like
    # "postgresql://user:pass@host/db" (pathlib interprets // as UNC path)
    url: str = Field(
        default="sqlite:///./runs/audit.db",
        description="Full SQLAlchemy database URL",
    )
    export: LandscapeExportSettings = Field(
        default_factory=LandscapeExportSettings,
        description="Post-run audit export configuration",
    )


class ConcurrencySettings(BaseModel):
    """Parallel processing configuration per architecture."""

    model_config = {"frozen": True}

    max_workers: int = Field(
        default=4,
        gt=0,
        description="Maximum parallel workers (default 4, production typically 16)",
    )


class DatabaseSettings(BaseModel):
    """Database connection configuration."""

    model_config = {"frozen": True}

    url: str = Field(description="SQLAlchemy database URL")
    pool_size: int = Field(default=5, gt=0, description="Connection pool size")
    echo: bool = Field(default=False, description="Echo SQL statements")


class ServiceRateLimit(BaseModel):
    """Rate limit configuration for a specific service."""

    model_config = {"frozen": True}

    requests_per_second: int = Field(gt=0, description="Maximum requests per second")
    requests_per_minute: int | None = Field(default=None, gt=0, description="Maximum requests per minute")


class RateLimitSettings(BaseModel):
    """Configuration for rate limiting external calls.

    Example YAML:
        rate_limit:
          enabled: true
          default_requests_per_second: 10
          persistence_path: ./rate_limits.db
          services:
            openai:
              requests_per_second: 5
              requests_per_minute: 100
            weather_api:
              requests_per_second: 20
    """

    model_config = {"frozen": True}

    enabled: bool = Field(default=True, description="Enable rate limiting for external calls")
    default_requests_per_second: int = Field(default=10, gt=0, description="Default rate limit for unconfigured services")
    default_requests_per_minute: int | None = Field(default=None, gt=0, description="Optional per-minute rate limit")
    persistence_path: str | None = Field(default=None, description="SQLite path for cross-process limits")
    services: dict[str, ServiceRateLimit] = Field(default_factory=dict, description="Per-service rate limit configurations")

    def get_service_config(self, service_name: str) -> ServiceRateLimit:
        """Get rate limit config for a service, with fallback to defaults."""
        if service_name in self.services:
            return self.services[service_name]
        return ServiceRateLimit(
            requests_per_second=self.default_requests_per_second,
            requests_per_minute=self.default_requests_per_minute,
        )


class CheckpointSettings(BaseModel):
    """Configuration for crash recovery checkpointing.

    Checkpoint frequency trade-offs:
    - every_row: Safest, can resume from any row. Higher I/O overhead.
    - every_n: Balance safety and performance. Lose up to N-1 rows on crash.
    - aggregation_only: Fastest, checkpoint only at aggregation flushes.
    """

    model_config = {"frozen": True}

    enabled: bool = True
    frequency: Literal["every_row", "every_n", "aggregation_only"] = "every_row"
    checkpoint_interval: int | None = Field(default=None, gt=0)  # Required if frequency == "every_n"
    aggregation_boundaries: bool = True  # Always checkpoint at aggregation flush

    @model_validator(mode="after")
    def validate_interval(self) -> "CheckpointSettings":
        if self.frequency == "every_n" and self.checkpoint_interval is None:
            raise ValueError("checkpoint_interval required when frequency='every_n'")
        return self


class RetrySettings(BaseModel):
    """Retry behavior configuration."""

    model_config = {"frozen": True}

    max_attempts: int = Field(default=3, gt=0, description="Maximum retry attempts")
    initial_delay_seconds: float = Field(default=1.0, gt=0, description="Initial backoff delay")
    max_delay_seconds: float = Field(default=60.0, gt=0, description="Maximum backoff delay")
    exponential_base: float = Field(default=2.0, gt=1.0, description="Exponential backoff base")


class PayloadStoreSettings(BaseModel):
    """Payload store configuration."""

    model_config = {"frozen": True}

    backend: str = Field(default="filesystem", description="Storage backend type")
    base_path: Path = Field(
        default=Path(".elspeth/payloads"),
        description="Base path for filesystem backend",
    )
    retention_days: int = Field(default=90, gt=0, description="Payload retention in days")


class ElspethSettings(BaseModel):
    """Top-level Elspeth configuration matching architecture specification.

    This is the single source of truth for pipeline configuration.
    All settings are validated and frozen after construction.
    """

    model_config = {"frozen": True}

    # Required - core pipeline definition
    datasource: DatasourceSettings = Field(
        description="Source plugin configuration (exactly one per run)",
    )
    sinks: dict[str, SinkSettings] = Field(
        description="Named sink configurations (one or more required)",
    )
    output_sink: str = Field(
        description="Default sink for rows that complete the pipeline",
    )

    # Run mode configuration
    run_mode: RunMode = Field(
        default=RunMode.LIVE,
        description="Execution mode: live (real calls), replay (use recorded), verify (compare)",
    )
    replay_source_run_id: str | None = Field(
        default=None,
        description="Run ID to replay/verify against (required for replay/verify modes)",
    )

    # Optional - transform chain
    row_plugins: list[RowPluginSettings] = Field(
        default_factory=list,
        description="Ordered list of transforms/gates to apply",
    )

    # Optional - engine-level gates (config-driven routing)
    gates: list[GateSettings] = Field(
        default_factory=list,
        description="Engine-level gates for config-driven routing (evaluated by ExpressionParser)",
    )

    # Optional - coalesce configuration (for merging fork paths)
    coalesce: list[CoalesceSettings] = Field(
        default_factory=list,
        description="Coalesce configurations for merging forked paths",
    )

    # Optional - aggregations (config-driven batching)
    aggregations: list[AggregationSettings] = Field(
        default_factory=list,
        description="Aggregation configurations for batching rows",
    )

    # Optional - subsystem configuration with defaults
    landscape: LandscapeSettings = Field(
        default_factory=LandscapeSettings,
        description="Audit trail configuration",
    )
    concurrency: ConcurrencySettings = Field(
        default_factory=ConcurrencySettings,
        description="Parallel processing configuration",
    )
    retry: RetrySettings = Field(
        default_factory=RetrySettings,
        description="Retry behavior configuration",
    )
    payload_store: PayloadStoreSettings = Field(
        default_factory=PayloadStoreSettings,
        description="Large payload storage configuration",
    )
    checkpoint: CheckpointSettings = Field(
        default_factory=CheckpointSettings,
        description="Crash recovery checkpoint configuration",
    )
    rate_limit: RateLimitSettings = Field(
        default_factory=RateLimitSettings,
        description="Rate limiting configuration",
    )

    @model_validator(mode="after")
    def validate_output_sink_exists(self) -> "ElspethSettings":
        """Ensure output_sink references a defined sink."""
        if self.output_sink not in self.sinks:
            raise ValueError(f"output_sink '{self.output_sink}' not found in sinks. Available sinks: {list(self.sinks.keys())}")
        return self

    @model_validator(mode="after")
    def validate_export_sink_exists(self) -> "ElspethSettings":
        """Ensure export.sink references a defined sink when enabled."""
        if self.landscape.export.enabled:
            if self.landscape.export.sink is None:
                raise ValueError("landscape.export.sink is required when export is enabled")
            if self.landscape.export.sink not in self.sinks:
                raise ValueError(
                    f"landscape.export.sink '{self.landscape.export.sink}' not found in sinks. Available sinks: {list(self.sinks.keys())}"
                )
        return self

    @model_validator(mode="after")
    def validate_unique_aggregation_names(self) -> "ElspethSettings":
        """Ensure aggregation names are unique."""
        names = [agg.name for agg in self.aggregations]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate aggregation name(s): {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_replay_source_run_id(self) -> "ElspethSettings":
        """Ensure replay_source_run_id is set when mode requires it.

        Replay and verify modes need a source run ID to replay/compare against.
        Live mode does not require (and ignores) replay_source_run_id.
        """
        if self.run_mode in (RunMode.REPLAY, RunMode.VERIFY) and not self.replay_source_run_id:
            raise ValueError(f"replay_source_run_id is required when run_mode is '{self.run_mode.value}'")
        return self

    @field_validator("sinks")
    @classmethod
    def validate_sinks_not_empty(cls, v: dict[str, SinkSettings]) -> dict[str, SinkSettings]:
        """At least one sink is required."""
        if not v:
            raise ValueError("At least one sink is required")
        return v


# Regex pattern for ${VAR} or ${VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand ${VAR} and ${VAR:-default} patterns in config values.

    Args:
        config: Configuration dict (may contain nested structures)

    Returns:
        New dict with environment variables expanded
    """
    import os

    def _expand_string(value: str) -> str:
        """Expand ${VAR} patterns in a string."""

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # No env var and no default - keep original (will likely cause error)
            return match.group(0)

        return _ENV_VAR_PATTERN.sub(replacer, value)

    def _expand_value(value: Any) -> Any:
        """Expand env vars in a single value."""
        if isinstance(value, str):
            return _expand_string(value)
        elif isinstance(value, dict):
            return {k: _expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand_value(item) for item in value]
        else:
            return value

    return {k: _expand_value(v) for k, v in config.items()}


# Secret field names that should be fingerprinted (exact matches)
_SECRET_FIELD_NAMES = frozenset({"api_key", "token", "password", "secret", "credential"})

# Secret field suffixes that should be fingerprinted
_SECRET_FIELD_SUFFIXES = ("_secret", "_key", "_token", "_password", "_credential")


def _is_secret_field(field_name: str) -> bool:
    """Check if a field name represents a secret that should be fingerprinted."""
    return field_name in _SECRET_FIELD_NAMES or field_name.endswith(_SECRET_FIELD_SUFFIXES)


def _fingerprint_secrets(
    options: dict[str, Any],
    *,
    fail_if_no_key: bool = True,
) -> dict[str, Any]:
    """Recursively replace secret fields with their fingerprints.

    Walks nested dicts and lists to find and fingerprint all secret fields,
    not just top-level ones.

    Args:
        options: Plugin options dict (may contain nested structures)
        fail_if_no_key: If True, raise if ELSPETH_FINGERPRINT_KEY not set
                        and secrets are found. If False, redact secrets
                        without fingerprinting (for dev mode).

    Returns:
        New dict with secrets replaced by fingerprints (or redacted)

    Raises:
        SecretFingerprintError: If secrets found but no fingerprint key available
                                and fail_if_no_key is True
    """
    from elspeth.core.security import get_fingerprint_key, secret_fingerprint

    # Check if we have a fingerprint key available
    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    def _process_value(key: str, value: Any) -> tuple[str, Any, bool]:
        """Process a single value, returning (new_key, new_value, was_secret)."""
        if isinstance(value, dict):
            return key, _recurse(value), False
        elif isinstance(value, list):
            return key, [_process_value("", item)[1] for item in value], False
        elif isinstance(value, str) and _is_secret_field(key):
            # This is a secret field
            if have_key:
                fp = secret_fingerprint(value)
                return f"{key}_fingerprint", fp, True
            elif fail_if_no_key:
                raise SecretFingerprintError(
                    f"Secret field '{key}' found but ELSPETH_FINGERPRINT_KEY "
                    "is not set. Either set the environment variable or use "
                    "ELSPETH_ALLOW_RAW_SECRETS=true for development "
                    "(not recommended for production)."
                )
            else:
                # Dev mode: keep original value (user explicitly opted in)
                return key, value, False
        else:
            return key, value, False

    def _recurse(d: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in d.items():
            new_key, new_value, _was_secret = _process_value(key, value)
            result[new_key] = new_value
        return result

    return _recurse(options)


def _sanitize_dsn(
    url: str,
    *,
    fail_if_no_key: bool = True,
) -> tuple[str, str | None, bool]:
    """Sanitize a database connection URL by removing/fingerprinting the password.

    Args:
        url: Database connection URL (SQLAlchemy format)
        fail_if_no_key: If True, raise if password found but no fingerprint key.
                        If False (dev mode), just remove password without fingerprint.

    Returns:
        Tuple of (sanitized_url, password_fingerprint or None, had_password)
        The third element indicates whether the original URL had a password.

    Raises:
        SecretFingerprintError: If password found, no key available,
                                and fail_if_no_key=True

    Example:
        >>> _sanitize_dsn("postgresql://user:secret@host/db")
        ("postgresql://user@host/db", "abc123...", True)
    """
    from sqlalchemy.engine import URL
    from sqlalchemy.engine.url import make_url
    from sqlalchemy.exc import ArgumentError

    try:
        parsed = make_url(url)
    except ArgumentError:
        # Not a valid SQLAlchemy URL - return as-is (might be a path or other format)
        return url, None, False

    if parsed.password is None:
        # No password in URL
        return url, None, False

    # Check if we have a fingerprint key
    from elspeth.core.security import get_fingerprint_key

    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    # Compute fingerprint if we have a key
    password_fingerprint = None
    if have_key:
        from elspeth.core.security import secret_fingerprint

        password_fingerprint = secret_fingerprint(parsed.password)
    elif fail_if_no_key:
        raise SecretFingerprintError(
            "Database URL contains a password but ELSPETH_FINGERPRINT_KEY "
            "is not set. Either set the environment variable or use "
            "ELSPETH_ALLOW_RAW_SECRETS=true for development "
            "(not recommended for production)."
        )
    # else: dev mode - just remove password without fingerprint

    # Reconstruct URL without password using URL.create()
    # NOTE: Do NOT use parsed.set(password=None) - it replaces with '***' not removal
    sanitized = URL.create(
        drivername=parsed.drivername,
        username=parsed.username,
        password=None,
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
        query=parsed.query,
    )

    return str(sanitized), password_fingerprint, True


def _expand_config_templates(
    raw_config: dict[str, Any],
    settings_path: Path | None = None,
) -> dict[str, Any]:
    """Expand template_file and lookup_file references in config.

    This function is called at load time to expand file references into
    their contents. Secrets are NOT fingerprinted here - that happens
    in resolve_config() when creating the audit copy.

    Args:
        raw_config: Raw config dict from Dynaconf
        settings_path: Path to settings file for resolving relative template paths

    Returns:
        Config with template files expanded (secrets still present)

    Raises:
        TemplateFileError: If template/lookup files not found or invalid
    """
    if settings_path is None:
        return raw_config

    config = dict(raw_config)

    # === Row plugin options - expand template files ===
    if "row_plugins" in config and isinstance(config["row_plugins"], list):
        plugins = []
        for plugin_config in config["row_plugins"]:
            if isinstance(plugin_config, dict):
                plugin = dict(plugin_config)
                if "options" in plugin and isinstance(plugin["options"], dict):
                    plugin["options"] = _expand_template_files(plugin["options"], settings_path)
                plugins.append(plugin)
            else:
                plugins.append(plugin_config)
        config["row_plugins"] = plugins

    return config


def _fingerprint_config_for_audit(
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint secrets in config for audit storage.

    Called by resolve_config() to create a copy safe for audit storage.
    The original config (with secrets) is untouched.

    Processes:
    - datasource.options
    - sinks.*.options
    - row_plugins[*].options
    - aggregations[*].options
    - landscape.url (DSN password)

    Args:
        config_dict: Config dict to fingerprint (will be copied)

    Returns:
        Deep copy with secrets fingerprinted

    Raises:
        SecretFingerprintError: If secrets found but no fingerprint key
                                and ELSPETH_ALLOW_RAW_SECRETS is not set
    """
    import copy
    import os

    # Check dev mode override
    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"
    fail_if_no_key = not allow_raw

    # Deep copy to avoid mutating the original
    config = copy.deepcopy(config_dict)

    # === Landscape URL (DSN password) ===
    if "landscape" in config and isinstance(config["landscape"], dict):
        landscape = config["landscape"]
        if "url" in landscape and isinstance(landscape["url"], str):
            # _sanitize_dsn returns (sanitized_url, fingerprint, had_password)
            sanitized_url, password_fp, had_password = _sanitize_dsn(
                landscape["url"],
                fail_if_no_key=fail_if_no_key,
            )
            landscape["url"] = sanitized_url
            if password_fp:
                landscape["url_password_fingerprint"] = password_fp
            elif had_password and not fail_if_no_key:
                # Dev mode: password was removed but not fingerprinted
                landscape["url_password_redacted"] = True

    # === Datasource options ===
    if "datasource" in config and isinstance(config["datasource"], dict):
        ds = config["datasource"]
        if "options" in ds and isinstance(ds["options"], dict):
            ds["options"] = _fingerprint_secrets(ds["options"], fail_if_no_key=fail_if_no_key)

    # === Sink options ===
    if "sinks" in config and isinstance(config["sinks"], dict):
        for sink in config["sinks"].values():
            if isinstance(sink, dict) and "options" in sink and isinstance(sink["options"], dict):
                sink["options"] = _fingerprint_secrets(sink["options"], fail_if_no_key=fail_if_no_key)

    # === Row plugin options ===
    if "row_plugins" in config and isinstance(config["row_plugins"], list):
        for plugin in config["row_plugins"]:
            if isinstance(plugin, dict) and "options" in plugin and isinstance(plugin["options"], dict):
                plugin["options"] = _fingerprint_secrets(plugin["options"], fail_if_no_key=fail_if_no_key)

    # === Aggregation options ===
    if "aggregations" in config and isinstance(config["aggregations"], list):
        for agg in config["aggregations"]:
            if isinstance(agg, dict) and "options" in agg and isinstance(agg["options"], dict):
                agg["options"] = _fingerprint_secrets(agg["options"], fail_if_no_key=fail_if_no_key)

    return config


class TemplateFileError(Exception):
    """Error loading template or lookup file."""


def _expand_template_files(
    options: dict[str, Any],
    settings_path: Path,
) -> dict[str, Any]:
    """Expand template_file and lookup_file to loaded content.

    Args:
        options: Plugin options dict
        settings_path: Path to settings file for resolving relative paths

    Returns:
        New dict with files loaded and paths recorded

    Raises:
        TemplateFileError: If files not found or invalid
    """
    result = dict(options)

    # Handle template_file
    if "template_file" in result:
        if "template" in result:
            raise TemplateFileError("Cannot specify both 'template' and 'template_file'")
        template_file = result.pop("template_file")
        template_path = Path(template_file)
        if not template_path.is_absolute():
            template_path = (settings_path.parent / template_path).resolve()

        if not template_path.exists():
            raise TemplateFileError(f"Template file not found: {template_path}")

        result["template"] = template_path.read_text(encoding="utf-8")
        result["template_source"] = template_file

    # Handle lookup_file
    if "lookup_file" in result:
        if "lookup" in result:
            raise TemplateFileError("Cannot specify both 'lookup' and 'lookup_file'")
        lookup_file = result.pop("lookup_file")
        lookup_path = Path(lookup_file)
        if not lookup_path.is_absolute():
            lookup_path = (settings_path.parent / lookup_path).resolve()

        if not lookup_path.exists():
            raise TemplateFileError(f"Lookup file not found: {lookup_path}")

        try:
            loaded = yaml.safe_load(lookup_path.read_text(encoding="utf-8"))
            # Coerce None (empty file) to {} so it gets a distinct hash from "no lookup"
            # This ensures empty lookup files are auditable as "intentionally empty"
            result["lookup"] = loaded if loaded is not None else {}
        except yaml.YAMLError as e:
            raise TemplateFileError(f"Invalid YAML in lookup file: {e}") from e

        result["lookup_source"] = lookup_file

    return result


def load_settings(config_path: Path) -> ElspethSettings:
    """Load settings from YAML file with environment variable overrides.

    Uses Dynaconf for multi-source loading with precedence:
    1. Environment variables (ELSPETH_*) - highest priority
    2. Config file (settings.yaml)
    3. Defaults from Pydantic schema - lowest priority

    Environment variable format: ELSPETH_DATABASE__URL for nested keys.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ElspethSettings instance

    Raises:
        ValidationError: If configuration fails Pydantic validation
        FileNotFoundError: If config file doesn't exist
    """
    from dynaconf import Dynaconf

    # Explicit check for file existence (Dynaconf silently accepts missing files)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load from file + environment
    dynaconf_settings = Dynaconf(
        envvar_prefix="ELSPETH",
        settings_files=[str(config_path)],
        environments=False,  # No [default]/[production] sections
        load_dotenv=False,  # Don't auto-load .env
        merge_enabled=True,  # Deep merge nested dicts
    )

    # Dynaconf returns uppercase keys; convert to lowercase for Pydantic
    # Also filter out internal Dynaconf settings
    internal_keys = {"LOAD_DOTENV", "ENVIRONMENTS", "SETTINGS_FILES"}
    raw_config = {k.lower(): v for k, v in dynaconf_settings.as_dict().items() if k not in internal_keys}

    # Expand ${VAR} and ${VAR:-default} patterns in config values
    raw_config = _expand_env_vars(raw_config)

    # Expand template files in plugin options before validation
    # NOTE: Secrets are NOT fingerprinted here - they stay available for runtime.
    # Fingerprinting happens in resolve_config() when creating the audit copy.
    raw_config = _expand_config_templates(raw_config, settings_path=config_path)

    return ElspethSettings(**raw_config)


def resolve_config(settings: ElspethSettings) -> dict[str, Any]:
    """Convert validated settings to a dict for audit storage.

    This is the resolved configuration that gets stored in Landscape
    for reproducibility. It includes all settings (explicit + defaults).

    IMPORTANT: This function fingerprints secrets before returning.
    The returned dict is safe for audit storage but should NOT be used
    for runtime operations that need actual secret values.

    Args:
        settings: Validated ElspethSettings instance

    Returns:
        Dict representation suitable for JSON serialization (secrets fingerprinted)
    """
    config_dict = settings.model_dump(mode="json")
    # Fingerprint secrets for audit storage
    return _fingerprint_config_for_audit(config_dict)
