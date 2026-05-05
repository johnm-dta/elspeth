"""Shared pooling infrastructure for parallel API transforms."""

from elspeth.contracts.engine import BufferEntry
from elspeth.contracts.errors import CapacityError, is_capacity_error
from elspeth.plugins.infrastructure.pooling.config import PoolConfig
from elspeth.plugins.infrastructure.pooling.executor import PooledExecutor, RowContext
from elspeth.plugins.infrastructure.pooling.throttle import AIMDThrottle, ThrottleConfig

__all__ = [
    "AIMDThrottle",
    "BufferEntry",
    "CapacityError",
    "PoolConfig",
    "PooledExecutor",
    "RowContext",
    "ThrottleConfig",
    "is_capacity_error",
]
