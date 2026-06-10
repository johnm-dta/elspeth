"""Plugin executors that wrap plugin calls with audit recording.

Each executor handles a specific plugin type:
- TransformExecutor: Row transforms
- GateExecutor: Routing gates
- AggregationExecutor: Stateful aggregations
- SinkExecutor: Output sinks
"""

from elspeth.engine.executors.aggregation import AggregationExecutor
from elspeth.engine.executors.gate import GateExecutor
from elspeth.engine.executors.sink import DiversionCounts, SinkExecutor
from elspeth.engine.executors.state_guard import NodeStateGuard
from elspeth.engine.executors.transform import TransformExecutor
from elspeth.engine.executors.types import GateOutcome, MissingEdgeError

__all__ = [
    "AggregationExecutor",
    "DiversionCounts",
    "GateExecutor",
    "GateOutcome",
    "MissingEdgeError",
    "NodeStateGuard",
    "SinkExecutor",
    "TransformExecutor",
]
