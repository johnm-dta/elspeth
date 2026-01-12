"""Built-in gate plugins for ELSPETH.

Gates evaluate rows and decide routing. Each gate returns a GateResult
with a RoutingAction indicating where the row should go.
"""

from elspeth.plugins.gates.field_match_gate import FieldMatchGate
from elspeth.plugins.gates.threshold_gate import ThresholdGate

__all__ = ["FieldMatchGate", "ThresholdGate"]
