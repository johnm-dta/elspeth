# src/elspeth/plugins/pooling/__init__.py
"""Shared pooling infrastructure for parallel API transforms."""

from elspeth.plugins.pooling.config import PoolConfig
from elspeth.plugins.pooling.errors import CapacityError, is_capacity_error
from elspeth.plugins.pooling.executor import PooledExecutor, RowContext
from elspeth.plugins.pooling.throttle import AIMDThrottle, ThrottleConfig

__all__ = [
    "AIMDThrottle",
    "CapacityError",
    "PoolConfig",
    "PooledExecutor",
    "RowContext",
    "ThrottleConfig",
    "is_capacity_error",
]
