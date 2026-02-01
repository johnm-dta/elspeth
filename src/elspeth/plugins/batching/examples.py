# src/elspeth/plugins/batching/examples.py
"""Examples showing how to use the batching infrastructure.

This file demonstrates:
1. Converting an existing transform to use batch processing
2. Making a sink implement OutputPort
3. Wiring up the orchestrator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# These would be your actual imports
if TYPE_CHECKING:
    from elspeth.contracts import TransformResult
    from elspeth.contracts.identity import TokenInfo


# =============================================================================
# Example 1: Making a Sink implement OutputPort
# =============================================================================


class OutputPortSinkAdapter:
    """Adapter that makes any sink implement OutputPort.

    Wraps an existing sink to receive streamed results via emit().
    Results are buffered and written in batches for efficiency.
    """

    def __init__(
        self,
        sink: Any,  # Your actual sink type
        batch_size: int = 100,
    ) -> None:
        self._sink = sink
        self._batch_size = batch_size
        self._buffer: list[tuple[TokenInfo, TransformResult, str | None]] = []

    def emit(self, token: TokenInfo, result: TransformResult, state_id: str | None) -> None:
        """Accept a result from upstream.

        Buffers results and writes in batches for efficiency.
        """
        self._buffer.append((token, result, state_id))

        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        """Write buffered results to sink."""
        if not self._buffer:
            return

        # Convert to the format your sink expects
        _tokens = [token for token, _, _ in self._buffer]
        # Your sink.write() call here
        # self._sink.write(tokens=_tokens, ...)

        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffer and close."""
        self._flush()


# =============================================================================
# Example 2: Converting an LLM Transform to use batching
# =============================================================================

"""
Before (synchronous):

    class AzureLLMTransform(BaseTransform):
        def process(self, row: dict, ctx: PluginContext) -> TransformResult:
            # Called once per row, blocks until complete
            response = self._call_llm(row)
            return TransformResult.success(
                {"response": response},
                success_reason={"action": "processed"},
            )


After (batched with FIFO ordering):

    class AzureLLMTransform(BaseTransform, BatchTransformMixin):
        def __init__(self, config: dict, output: OutputPort):
            super().__init__(config)
            self.init_batch_processing(
                max_pending=30,  # Match pool size
                output=output,
                name="azure-llm",
            )

        def accept(self, row: dict, ctx: PluginContext) -> None:
            # Called once per row, returns quickly (may block on backpressure)
            self.accept_row(row, ctx, self._do_processing)

        def _do_processing(self, row: dict, ctx: PluginContext) -> TransformResult:
            # Runs in worker thread, can take as long as needed
            response = self._call_llm(row)
            return TransformResult.success(
                {"response": response},
                success_reason={"action": "processed"},
            )

        def close(self) -> None:
            self.shutdown_batch_processing()
            super().close()
"""


# =============================================================================
# Example 3: Orchestrator Integration
# =============================================================================

"""
Before (orchestrator pulls results):

    for row_index, source_item in enumerate(source_iterator):
        results = processor.process_row(row_index, row_data, transforms, ctx)
        for result in results:
            pending_tokens[sink_name].append(result.token)

    # At end: write all accumulated tokens
    for sink_name, tokens in pending_tokens.items():
        sink.write(tokens=tokens)


After (orchestrator pushes, results flow through ports):

    # Setup: wire transform output to sink
    sink_port = OutputPortSinkAdapter(sink)
    transform.init_batch_processing(max_pending=30, output=sink_port)

    # Run: just feed the transform
    for row_index, source_item in enumerate(source_iterator):
        row_data = source_item.row
        ctx = make_context(row_index, row_data)
        transform.accept(row_data, ctx)  # Returns quickly, may block on backpressure

    # End: flush and close
    transform.flush_batch_processing()  # Wait for all rows to complete
    sink_port.close()  # Write any remaining buffered results
"""


# =============================================================================
# Example 4: Chaining Transforms (Transform → Transform → Sink)
# =============================================================================

"""
Multiple transforms can be chained via output ports:

    # Setup: wire chain backwards (sink first, then transforms)
    sink_port = OutputPortSinkAdapter(sink)
    transform_b.init_batch_processing(max_pending=20, output=sink_port)
    transform_a.init_batch_processing(max_pending=30, output=TransformOutputAdapter(transform_b))

    # Run: feed first transform
    for row_data, ctx in source:
        transform_a.accept(row_data, ctx)

    # End: flush in forward order
    transform_a.flush_batch_processing()
    transform_b.flush_batch_processing()
    sink_port.close()


Where TransformOutputAdapter makes a transform act as an output port:

    class TransformOutputAdapter:
        def __init__(self, transform: BatchTransformMixin):
            self._transform = transform

        def emit(self, token: TokenInfo, result: TransformResult) -> None:
            # Create context from token
            ctx = PluginContext(token=token, ...)
            # Feed the next transform
            self._transform.accept(result.row, ctx)
"""


# =============================================================================
# Example 5: What the throughput improvement looks like
# =============================================================================

"""
Before (sequential, pool underutilized):

    Row 1: [== 10 queries ==]......................
    Row 2:                    [== 10 queries ==]......................
    Row 3:                                        [== 10 queries ==]

    Pool utilization: 10/30 = 33%
    Total time: 3x row_time


After (pipelined, pool fully utilized):

    Row 1: [== 10 queries ==]─────────────────────► emit
    Row 2: [== 10 queries ==]─────────────────────► emit (waits for row 1)
    Row 3: [== 10 queries ==]─────────────────────► emit (waits for row 2)

    All three rows' queries share the pool concurrently!

    Pool utilization: 30/30 = 100%
    Total time: ~1x row_time (limited by slowest row + FIFO wait)

    3x throughput improvement!
"""
