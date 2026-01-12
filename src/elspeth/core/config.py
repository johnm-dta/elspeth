# src/elspeth/core/config.py
"""
Configuration schema and loading for Elspeth pipelines.

Uses Pydantic for validation and Dynaconf for multi-source loading.
Settings are frozen (immutable) after construction.
"""

from pathlib import Path
from typing import Any

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
