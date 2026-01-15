# src/elspeth/core/config.py
"""
Configuration schema and loading for Elspeth pipelines.

Uses Pydantic for validation and Dynaconf for multi-source loading.
Settings are frozen (immutable) after construction.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DatasourceSettings(BaseModel):
    """Source plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv_local, json, http_poll, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class RowPluginSettings(BaseModel):
    """Transform or gate plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name")
    type: Literal["transform", "gate"] = Field(
        default="transform",
        description="Plugin type: transform (pass-through) or gate (routing)",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    routes: dict[str, str] | None = Field(
        default=None,
        description="Gate routing map: result -> sink_name or 'continue'",
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
    requests_per_minute: int | None = Field(
        default=None, gt=0, description="Maximum requests per minute"
    )


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

    enabled: bool = Field(
        default=True, description="Enable rate limiting for external calls"
    )
    default_requests_per_second: int = Field(
        default=10, gt=0, description="Default rate limit for unconfigured services"
    )
    default_requests_per_minute: int | None = Field(
        default=None, gt=0, description="Optional per-minute rate limit"
    )
    persistence_path: str | None = Field(
        default=None, description="SQLite path for cross-process limits"
    )
    services: dict[str, ServiceRateLimit] = Field(
        default_factory=dict, description="Per-service rate limit configurations"
    )

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
    checkpoint_interval: int | None = Field(
        default=None, gt=0
    )  # Required if frequency == "every_n"
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
    initial_delay_seconds: float = Field(
        default=1.0, gt=0, description="Initial backoff delay"
    )
    max_delay_seconds: float = Field(
        default=60.0, gt=0, description="Maximum backoff delay"
    )
    exponential_base: float = Field(
        default=2.0, gt=1.0, description="Exponential backoff base"
    )


class PayloadStoreSettings(BaseModel):
    """Payload store configuration."""

    model_config = {"frozen": True}

    backend: str = Field(default="filesystem", description="Storage backend type")
    base_path: Path = Field(
        default=Path(".elspeth/payloads"),
        description="Base path for filesystem backend",
    )
    retention_days: int = Field(
        default=90, gt=0, description="Payload retention in days"
    )


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

    # Optional - transform chain
    row_plugins: list[RowPluginSettings] = Field(
        default_factory=list,
        description="Ordered list of transforms/gates to apply",
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
            raise ValueError(
                f"output_sink '{self.output_sink}' not found in sinks. "
                f"Available sinks: {list(self.sinks.keys())}"
            )
        return self

    @model_validator(mode="after")
    def validate_export_sink_exists(self) -> "ElspethSettings":
        """Ensure export.sink references a defined sink when enabled."""
        if self.landscape.export.enabled:
            if self.landscape.export.sink is None:
                raise ValueError(
                    "landscape.export.sink is required when export is enabled"
                )
            if self.landscape.export.sink not in self.sinks:
                raise ValueError(
                    f"landscape.export.sink '{self.landscape.export.sink}' not found in sinks. "
                    f"Available sinks: {list(self.sinks.keys())}"
                )
        return self

    @field_validator("sinks")
    @classmethod
    def validate_sinks_not_empty(
        cls, v: dict[str, SinkSettings]
    ) -> dict[str, SinkSettings]:
        """At least one sink is required."""
        if not v:
            raise ValueError("At least one sink is required")
        return v


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
    raw_config = {
        k.lower(): v
        for k, v in dynaconf_settings.as_dict().items()
        if k not in internal_keys
    }
    return ElspethSettings(**raw_config)


def resolve_config(settings: ElspethSettings) -> dict[str, Any]:
    """Convert validated settings to a dict for audit storage.

    This is the resolved configuration that gets stored in Landscape
    for reproducibility. It includes all settings (explicit + defaults).

    Args:
        settings: Validated ElspethSettings instance

    Returns:
        Dict representation suitable for JSON serialization
    """
    return settings.model_dump(mode="json")
