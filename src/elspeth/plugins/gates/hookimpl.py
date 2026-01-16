"""Hook implementation for built-in gate plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinGates:
    """Hook implementer for built-in gate plugins."""

    @hookimpl
    def elspeth_get_gates(self) -> list[type[Any]]:
        """Return built-in gate plugin classes."""
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate
        from elspeth.plugins.gates.filter_gate import FilterGate
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        return [ThresholdGate, FieldMatchGate, FilterGate]


# Singleton instance for registration
builtin_gates = ElspethBuiltinGates()
