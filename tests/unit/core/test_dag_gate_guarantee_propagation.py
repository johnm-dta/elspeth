# tests/unit/core/test_dag_gate_guarantee_propagation.py
"""Gate nodes are pure routing: guarantee propagation must walk through them.

Sibling of the coalesce pass-through guarantee fix (6b431fd03,
elspeth-0b14977817). The builder assigns each gate the RAW
``output_schema_config`` of its upstream producer (``_assign_schema(gate_id,
_best_schema_config(producer_id))``). When that producer is a pass-through
transform, the raw config names only the transform's OWN declared fields —
the source columns it forwards are missing. ``walk_effective_guarantee_vote``
then stopped at the gate (gates are not ``passes_through_input`` transforms),
so ``validate_edge_schemas`` compared branch consumers against the
under-computed raw set and falsely rejected runnable
source → pass-through → gate → branch pipelines.

The reference-correct algorithm lives in the composer preview
(``web/composer/state.py::_connection_propagation_vote``), which recurses
straight through gates. These tests pin the engine walker to the same
result: a gate's effective guarantee is composed from its predecessors,
exactly like a pass-through transform with no declared fields of its own.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.config import (
    CoalesceSettings,
    GateSettings,
    SourceSettings,
    TransformSettings,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.guarantees import get_effective_guaranteed_fields
from elspeth.core.dag.wiring import WiredTransform


class _SourceWithGuarantees:
    """Mock source declaring guaranteed_fields via its config schema."""

    name = "mock_source_guaranteeing"
    output_schema = None
    _on_validation_failure = "discard"
    on_success = "output"

    def __init__(self, guaranteed: tuple[str, ...]) -> None:
        self.config = {"schema": {"mode": "observed", "guaranteed_fields": list(guaranteed)}}


class _PassThroughTransform:
    """Mock pass-through transform that ADDS one field on top of its input.

    Models a real LLM node (``passes_through_input=True``): forwards the
    source columns and appends a response field. Its raw declared guarantee
    names ONLY the added field.
    """

    input_schema = None
    output_schema = None
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = True

    def __init__(self, name: str, added_field: str) -> None:
        self.name = name
        self.config = {"schema": {"mode": "observed", "guaranteed_fields": [added_field]}}
        self._output_schema_config = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=(added_field,),
        )


class _RequiringTransform:
    """Mock transform with explicit required_input_fields in its config."""

    input_schema = None
    output_schema = None
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = False

    def __init__(self, name: str, required: tuple[str, ...]) -> None:
        self.name = name
        self.config = {
            "schema": {"mode": "observed"},
            "required_input_fields": list(required),
        }
        self._output_schema_config = SchemaConfig(mode="observed", fields=None)


class _BuilderMockSink:
    name = "mock_sink"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {}
    _on_write_failure: str = "discard"
    declared_required_fields: ClassVar[frozenset[str]] = frozenset()

    def _reset_diversion_log(self) -> None:
        pass


def _build_prep_then_fork_graph(
    *,
    branch_required: tuple[str, ...],
) -> ExecutionGraph:
    """source(color_name,hex) → pass-through 'prep'(+clean) → gate fork → branches → coalesce → sink."""
    source = _SourceWithGuarantees(("color_name", "hex"))
    prep = _PassThroughTransform("prep_cleanup", "clean")
    branch_a = _RequiringTransform("branch_requiring_a", branch_required)
    branch_b = _RequiringTransform("branch_requiring_b", branch_required)
    wired = [
        WiredTransform(
            plugin=prep,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="prep", plugin=prep.name, input="source_out", on_success="prepped", on_error="discard", options={}
            ),
        ),
        WiredTransform(
            plugin=branch_a,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="b_a", plugin=branch_a.name, input="branch_a", on_success="b_a_out", on_error="discard", options={}
            ),
        ),
        WiredTransform(
            plugin=branch_b,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="b_b", plugin=branch_b.name, input="branch_b", on_success="b_b_out", on_error="discard", options={}
            ),
        ),
    ]
    fork_gate = GateSettings(
        name="splitter",
        input="prepped",
        condition="True",
        routes={"true": "fork", "false": "fork"},
        fork_to=["branch_a", "branch_b"],
    )
    coalesce = CoalesceSettings(
        name="reconcile",
        branches={"branch_a": "b_a_out", "branch_b": "b_b_out"},
        policy="require_all",
        merge="union",
        on_success="output",
    )
    return ExecutionGraph.from_plugin_instances(
        sources={"primary": source},  # type: ignore[arg-type]
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
        transforms=wired,
        sinks={"output": _BuilderMockSink()},  # type: ignore[dict-item]
        aggregations={},
        gates=[fork_gate],
        coalesce_settings=[coalesce],
    )


class TestGateGuaranteeTransparency:
    def test_fork_branches_inherit_upstream_effective_guarantee(self) -> None:
        """Branch consumers requiring source columns must validate through the gate.

        The pass-through 'prep' transform effectively guarantees
        {color_name, hex, clean}; the gate is pure routing, so each fork
        branch sees the same set. Before the fix the gate's raw-inherited
        schema named only {clean} and validate_edge_schemas rejected the
        gate → branch edges with "Missing fields: ['color_name', 'hex']".
        """
        graph = _build_prep_then_fork_graph(branch_required=("color_name", "hex"))

        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1
        assert get_effective_guaranteed_fields(graph, gate_nodes[0].node_id) == frozenset({"color_name", "hex", "clean"})

    def test_gate_directly_after_source_still_guarantees_source_fields(self) -> None:
        """Control: direct source → gate was already correct (raw == effective)."""
        source = _SourceWithGuarantees(("color_name", "hex"))
        branch_a = _RequiringTransform("branch_requiring_a", ("color_name", "hex"))
        branch_b = _RequiringTransform("branch_requiring_b", ("color_name", "hex"))
        wired = [
            WiredTransform(
                plugin=branch_a,  # type: ignore[arg-type]
                settings=TransformSettings(
                    name="b_a", plugin=branch_a.name, input="branch_a", on_success="b_a_out", on_error="discard", options={}
                ),
            ),
            WiredTransform(
                plugin=branch_b,  # type: ignore[arg-type]
                settings=TransformSettings(
                    name="b_b", plugin=branch_b.name, input="branch_b", on_success="b_b_out", on_error="discard", options={}
                ),
            ),
        ]
        fork_gate = GateSettings(
            name="splitter",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "fork"},
            fork_to=["branch_a", "branch_b"],
        )
        coalesce = CoalesceSettings(
            name="reconcile",
            branches={"branch_a": "b_a_out", "branch_b": "b_b_out"},
            policy="require_all",
            merge="union",
            on_success="output",
        )
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},  # type: ignore[arg-type]
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=wired,
            sinks={"output": _BuilderMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )
        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1
        assert get_effective_guaranteed_fields(graph, gate_nodes[0].node_id) >= frozenset({"color_name", "hex"})

    def test_prep_then_fork_builds_without_contract_rejection(self) -> None:
        """End-to-end: the full build (including validate_edge_compatibility) succeeds.

        This is the build that raised GraphValidationError
        ("Schema contract violation: edge 'gate' → 'branch'") before the fix.
        _build_prep_then_fork_graph would raise on construction if the edge
        check rejected, so reaching an assertion at all IS the fix.
        """
        graph = _build_prep_then_fork_graph(branch_required=("color_name", "hex"))
        assert graph is not None

    def test_gate_guarantee_never_shrinks_below_raw_inheritance(self) -> None:
        """Monotonicity: composing predecessors can only grow the gate's set.

        The gate's raw-inherited schema (from the pass-through 'prep') names
        {clean}; the composed effective set must be a superset — the fix may
        turn false rejections into passes, never a pass into a rejection.
        """
        graph = _build_prep_then_fork_graph(branch_required=())
        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1
        gate = gate_nodes[0]
        raw = gate.output_schema_config.get_effective_guaranteed_fields() if gate.output_schema_config is not None else frozenset()
        assert get_effective_guaranteed_fields(graph, gate.node_id) >= raw


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
