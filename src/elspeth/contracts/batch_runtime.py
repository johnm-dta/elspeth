"""Runtime protocol for row-pipelined batch transforms.

This contract is intentionally lower-level than ``BatchTransformProtocol``:
``BatchTransformProtocol`` models aggregation-style batch plugins, while this
protocol models transforms that accept one row, do concurrent internal work,
and emit the row result through an output adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.contracts.contexts import TransformContext
    from elspeth.contracts.schema_contract import PipelineRow


@runtime_checkable
class BatchTransformRuntimeProtocol(Protocol):
    """Protocol consumed by the engine's single-row transform executor."""

    node_id: str | None

    @property
    def batch_runtime_enabled(self) -> bool:
        """Whether this transform participates in row-pipelined batch runtime."""
        ...

    @property
    def batch_pool_size(self) -> int:
        """Preferred number of pending row submissions."""
        ...

    @property
    def batch_wait_timeout(self) -> float:
        """Maximum seconds the executor should wait for one row result."""
        ...

    def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
        """Submit one row to the transform's internal batch runtime."""
        ...

    def connect_output(self, output: Any, max_pending: int = 30) -> None:
        """Connect the engine-owned output adapter."""
        ...

    def evict_submission(self, token_id: str, state_id: str) -> bool:
        """Evict a timed-out token/state submission from the internal buffer."""
        ...
