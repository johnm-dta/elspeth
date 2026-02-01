# src/elspeth/plugins/batching/__init__.py
"""Plugin-level pipelining for concurrent row processing with FIFO ordering.

This module enables transforms to process multiple rows concurrently while
guaranteeing strict FIFO output ordering. The orchestrator sees synchronous
behavior; concurrency is hidden inside the plugin boundary.

Architecture:
    Every pipeline stage has:
    - Input port: accept() - takes work, may block on backpressure
    - Output port: emit() - sends results to next stage

    The transform doesn't know if downstream is another transform or a sink.
    It just emits to its output port.

Example:
    class MyLLMTransform(BaseTransform, BatchTransformMixin):
        def __init__(self, config, output: OutputPort):
            super().__init__(config)
            self.init_batch_processing(max_pending=30, output=output)

        def accept(self, row: dict, ctx: PluginContext) -> None:
            self.accept_row(row, ctx, self._do_llm_processing)

        def _do_llm_processing(self, row: dict, ctx: PluginContext) -> TransformResult:
            # Actual LLM work here
            return TransformResult.success(row, success_reason={"action": "processed"})
"""

from elspeth.plugins.batching.mixin import BatchTransformMixin
from elspeth.plugins.batching.ports import OutputPort
from elspeth.plugins.batching.row_reorder_buffer import RowReorderBuffer

__all__ = [
    "BatchTransformMixin",
    "OutputPort",
    "RowReorderBuffer",
]
