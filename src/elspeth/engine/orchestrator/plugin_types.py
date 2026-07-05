"""Plugin type aliases used by orchestrator and engine traversal code."""

from __future__ import annotations

from elspeth.contracts import TransformProtocol

# Type alias for row-processing plugins in the transforms pipeline.
# NOTE: BaseAggregation was DELETED - aggregation is now handled by
# batch-aware transforms (is_batch_aware=True on TransformProtocol)
RowPlugin = TransformProtocol
"""Row-processing plugin type for pipeline transforms list."""
