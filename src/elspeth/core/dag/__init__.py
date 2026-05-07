"""DAG (Directed Acyclic Graph) operations for execution planning."""

from elspeth.core.dag.coalesce_merge import (
    merge_guaranteed_fields,
    merge_union_fields,
)
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.core.dag.models import (
    GraphValidationError,
    GraphValidationWarning,
    NodeConfig,
    NodeInfo,
    WiredTransform,
)

__all__ = [
    "ExecutionGraph",
    "GraphValidationError",
    "GraphValidationWarning",
    "NodeConfig",
    "NodeInfo",
    "WiredTransform",
    "merge_guaranteed_fields",
    "merge_union_fields",
]
