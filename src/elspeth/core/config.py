# src/elspeth/core/config.py
"""
Configuration schema and loading for Elspeth pipelines.

Uses Pydantic for validation and Dynaconf for multi-source loading.
Settings are frozen (immutable) after construction.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DatabaseSettings(BaseModel):
    """Database connection configuration."""

    model_config = {"frozen": True}

    url: str = Field(description="SQLAlchemy database URL")
    pool_size: int = Field(default=5, gt=0, description="Connection pool size")
    echo: bool = Field(default=False, description="Echo SQL statements")


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
        default=Path(".elspeth/payloads"), description="Base path for filesystem backend"
    )
    retention_days: int = Field(
        default=90, gt=0, description="Payload retention in days"
    )


class ElspethSettings(BaseModel):
    """Top-level Elspeth configuration."""

    model_config = {"frozen": True}

    database: DatabaseSettings
    retry: RetrySettings = Field(default_factory=RetrySettings)
    payload_store: PayloadStoreSettings = Field(default_factory=PayloadStoreSettings)
    run_id_prefix: str = Field(default="run", description="Prefix for run IDs")

    @field_validator("run_id_prefix")
    @classmethod
    def validate_run_id_prefix(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError("run_id_prefix must be a valid identifier")
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
        load_dotenv=False,   # Don't auto-load .env
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

