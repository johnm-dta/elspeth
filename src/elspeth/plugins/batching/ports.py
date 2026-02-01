# src/elspeth/plugins/batching/ports.py
"""Port abstractions for pipeline stages.

Every pipeline stage has input and output ports. Data flows through ports,
not through return values. This enables:
- Decoupled stages (transform doesn't know if downstream is sink or transform)
- Streaming (results flow as they complete, not batched at end)
- Backpressure (downstream can signal "slow down" via blocking)

Retry Safety:
    The emit() method includes a state_id parameter to ensure that results
    from different attempts (e.g., after a timeout and retry) are routed to
    the correct waiter. This prevents stale results from being delivered to
    retry attempts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.contracts import ExceptionResult, TransformResult
    from elspeth.contracts.identity import TokenInfo


@runtime_checkable
class OutputPort(Protocol):
    """Abstract output interface for pipeline stages.

    Any stage that can receive results implements this protocol.
    This includes:
    - Sinks (write results to external storage)
    - Transforms (accept results as input for further processing)
    - Buffers (accumulate results for batching)

    The emitting stage doesn't know or care what's downstream.
    It just calls emit() and the result flows through.
    """

    def emit(self, token: TokenInfo, result: TransformResult | ExceptionResult, state_id: str | None) -> None:
        """Accept a result from upstream.

        May block if downstream is applying backpressure (e.g., buffer full).

        Args:
            token: Token identifying this row's lineage
            result: The transform result to pass downstream, or ExceptionResult for plugin bugs
            state_id: State ID for the attempt that produced this result (for retry safety).
                     May be None when state tracking is not used.

        Raises:
            RuntimeError: If the port is closed/shutdown
        """
        ...


class NullOutputPort:
    """Output port that discards all results.

    Useful for testing or when results should be dropped.
    """

    def emit(self, token: TokenInfo, result: TransformResult | ExceptionResult, state_id: str | None) -> None:
        """Discard the result."""
        pass


class CollectorOutputPort:
    """Output port that collects results into a list.

    Useful for testing - captures all emitted results for verification.
    """

    def __init__(self) -> None:
        self.results: list[tuple[TokenInfo, TransformResult | ExceptionResult, str | None]] = []

    def emit(self, token: TokenInfo, result: TransformResult | ExceptionResult, state_id: str | None) -> None:
        """Collect the result."""
        self.results.append((token, result, state_id))

    def clear(self) -> None:
        """Clear collected results."""
        self.results.clear()
