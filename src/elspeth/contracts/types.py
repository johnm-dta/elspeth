"""Semantic type aliases for compile-time type safety.

NewType creates distinct types that mypy treats as incompatible,
preventing accidental misuse of semantically different string values.
"""

from collections.abc import Callable
from typing import NewType

NODE_ID_MAX_LENGTH = 64
"""Maximum supported node identifier length across DAG and audit storage."""

# Node identifiers - deterministic hash-based IDs for graph nodes
NodeID = NewType("NodeID", str)
"""Unique node identifier in execution graph (e.g., 'coalesce_merge_abc123')"""

# Semantic names for pipeline components
CoalesceName = NewType("CoalesceName", str)
"""User-defined name for coalesce point (e.g., 'merge_results')"""

BranchName = NewType("BranchName", str)
"""Branch name from gate fork (e.g., 'path_a', 'analysis_path')"""

SinkName = NewType("SinkName", str)
"""User-defined sink name (e.g., 'output', 'errors')"""

GateName = NewType("GateName", str)
"""User-defined gate name (e.g., 'classifier', 'router')"""

AggregationName = NewType("AggregationName", str)
"""User-defined aggregation name (e.g., 'batch_processor')"""

StepResolver = Callable[[NodeID], int]
"""Resolves a NodeID to its 1-indexed audit step position in the DAG.

Injected into executors, token managers, and coalesce executors at construction
time so they can resolve step_in_pipeline internally instead of receiving it
as a threaded parameter through every method call.

The canonical implementation is RowProcessor._resolve_audit_step_for_node.
"""
