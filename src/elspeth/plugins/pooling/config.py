# src/elspeth/plugins/pooling/config.py
"""Pool configuration for concurrent API transforms."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from elspeth.plugins.pooling.throttle import ThrottleConfig


class PoolConfig(BaseModel):
    """Pool configuration for concurrent API requests.

    Attributes:
        pool_size: Number of concurrent requests (must be >= 1)
        min_dispatch_delay_ms: Floor for delay between dispatches
        max_dispatch_delay_ms: Ceiling for delay
        backoff_multiplier: Multiply delay on capacity error (must be > 1)
        recovery_step_ms: Subtract from delay on success
        max_capacity_retry_seconds: Max time to retry capacity errors per row
    """

    model_config = {"extra": "forbid"}

    pool_size: int = Field(1, ge=1, description="Number of concurrent requests")
    min_dispatch_delay_ms: int = Field(0, ge=0, description="Minimum dispatch delay in milliseconds")
    max_dispatch_delay_ms: int = Field(5000, ge=0, description="Maximum dispatch delay in milliseconds")
    backoff_multiplier: float = Field(2.0, gt=1.0, description="Backoff multiplier on capacity error")
    recovery_step_ms: int = Field(50, ge=0, description="Recovery step in milliseconds")
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")

    @model_validator(mode="after")
    def _validate_delay_invariants(self) -> Self:
        """Validate min_dispatch_delay_ms <= max_dispatch_delay_ms."""
        if self.min_dispatch_delay_ms > self.max_dispatch_delay_ms:
            raise ValueError(
                f"min_dispatch_delay_ms ({self.min_dispatch_delay_ms}) cannot exceed max_dispatch_delay_ms ({self.max_dispatch_delay_ms})"
            )
        return self

    def to_throttle_config(self) -> ThrottleConfig:
        """Convert to ThrottleConfig for runtime use."""
        return ThrottleConfig(
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
        )
