"""Semantic type aliases for compile-time type safety.

NewType creates distinct types that mypy treats as incompatible,
preventing accidental misuse of semantically different string values.
"""

from typing import NewType

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
