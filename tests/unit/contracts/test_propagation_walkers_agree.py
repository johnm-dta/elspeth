"""Walker-agreement property test — ADR-009 §Clause 1 governance gate.

The ADR-009 design states that the runtime DAG walker
(``core/dag/graph.py::_walk_effective_guaranteed_fields``) and the composer
preview walker (``web/composer/state.py::_effective_producer_guarantees``)
share an aggregation rule (``compose_propagation``) and a participation
predicate (``SchemaConfig.participates_in_propagation``). Both are consulted
directly by each walker, so drift in the primitives themselves is
structurally impossible.

What this test pins: the *traversal composition* — the two walkers compute
predecessor field sets differently because they walk different graph views
(runtime DAG vs. composer producer-graph), but they must produce equivalent
effective-guarantee answers on topologies both can see.

Strategy: generate random chains of ``passthrough`` nodes over a ``text``
source with varied declared guarantees, build both views from the same
spec, and assert the walker outputs match for every intermediate node.
Shrinking produces minimal repros when drift is detected.

Scope note: the strategy is pass-through-chain only for v1. Chains are
where pass-through propagation is load-bearing — one mis-annotated upstream
poisons all downstream inheritance. Coalesce/fork topologies and
non-pass-through breakpoints are covered by ``test_compose_propagation``
unit tests and the concrete integration scenarios in
``test_composer_runtime_agreement.py``. Track 2 may extend this harness
when the propagation rule generalizes to other declarations.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import (
    ElspethSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)

# Small field alphabet so shrinking converges to meaningful minimal
# topologies. The point is to test propagation through pass-through
# chains, not to explore field-name edge cases.
_FIELD_POOL = ("alpha", "beta", "gamma")


@st.composite
def pass_through_chain_spec(draw: st.DrawFn) -> dict[str, object]:
    """Generate a linear pass-through chain spec.

    Shape: source -> t1 (passthrough) -> ... -> tN (passthrough) -> sink.

    Dimensions exercised:

    - chain length (2..4 transforms)
    - source declared ``guaranteed_fields`` (non-empty subset of
      ``_FIELD_POOL``). Required non-empty so required_input_fields on
      transforms and sink resolves, which forces ``edge_contracts`` to
      record producer guarantees at every hop (otherwise Stage 1 skips
      the edge and the test loses visibility into the composer walker).

    Uses the first declared field as the source column. The text source
    emits a row with that column name, inheriting it through the chain.
    """
    chain_length = draw(st.integers(min_value=2, max_value=4))
    declared = draw(
        st.lists(
            st.sampled_from(_FIELD_POOL),
            min_size=1,
            max_size=len(_FIELD_POOL),
            unique=True,
        )
    )
    declared.sort()

    return {
        "chain_length": chain_length,
        "declared_fields": tuple(declared),
        "column": declared[0],
    }


def _build_composer_state(
    spec: dict[str, object],
    *,
    text_path: Path,
    output_path: Path,
) -> CompositionState:
    chain_length = int(spec["chain_length"])
    declared_fields = list(spec["declared_fields"])  # type: ignore[arg-type]
    column = str(spec["column"])

    source_spec = SourceSpec(
        plugin="text",
        on_success="t1",
        options={
            "path": str(text_path),
            "column": column,
            "schema_config": {
                "mode": "observed",
                "guaranteed_fields": declared_fields,
            },
        },
        on_validation_failure="quarantine",
    )

    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []
    for i in range(chain_length):
        node_id = f"t{i + 1}"
        next_target = "main" if i == chain_length - 1 else f"t{i + 2}"
        nodes.append(
            NodeSpec(
                id=node_id,
                node_type="transform",
                plugin="passthrough",
                input=node_id,
                on_success=next_target,
                on_error="discard",
                options={
                    # required_input_fields forces Stage 1 to record an
                    # edge_contract for the incoming edge — this is the only
                    # path to observe the composer walker's output.
                    "required_input_fields": declared_fields,
                    # Declared guaranteed_fields at every hop ensures each
                    # transform's own schema "participates_in_propagation"
                    # per ADR-009 §Clause 1. Without explicit declaration, a
                    # shape-preserving pass-through abstains (its schema has
                    # no declared fields), which is the legitimate traversal
                    # divergence ADR-009 acknowledges — runtime stops the
                    # chain at the first non-declaring hop, composer walks
                    # back unconditionally. The agreement property holds
                    # only when every hop declares, so the generator forces
                    # that shape.
                    "schema": {
                        "mode": "observed",
                        "guaranteed_fields": list(declared_fields),
                    },
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        from_node = "source" if i == 0 else f"t{i}"
        edges.append(
            EdgeSpec(
                id=f"e{i + 1}",
                from_node=from_node,
                to_node=node_id,
                edge_type="on_success",
                label=None,
            )
        )

    outputs = (
        OutputSpec(
            name="main",
            plugin="csv",
            options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
            on_write_failure="discard",
        ),
    )

    return CompositionState(
        source=source_spec,
        nodes=tuple(nodes),
        edges=tuple(edges),
        outputs=outputs,
        metadata=PipelineMetadata(),
        version=1,
    )


def _build_runtime_graph(
    spec: dict[str, object],
    *,
    text_path: Path,
    output_path: Path,
) -> ExecutionGraph:
    """Build an ExecutionGraph via the production assembly path."""
    chain_length = int(spec["chain_length"])
    declared_fields = list(spec["declared_fields"])  # type: ignore[arg-type]
    column = str(spec["column"])

    transforms: list[TransformSettings] = []
    for i in range(chain_length):
        node_id = f"t{i + 1}"
        next_target = "main" if i == chain_length - 1 else f"t{i + 2}"
        transforms.append(
            TransformSettings(
                name=node_id,
                plugin="passthrough",
                input=node_id,
                on_success=next_target,
                on_error="discard",
                options={
                    "required_input_fields": declared_fields,
                    # Declared guaranteed_fields makes each hop participate
                    # in ADR-007 propagation — same shape as composer side.
                    "schema": {
                        "mode": "observed",
                        "guaranteed_fields": declared_fields,
                    },
                },
            )
        )

    config = ElspethSettings(
        source=SourceSettings(
            plugin="text",
            on_success="t1",
            options={
                "path": str(text_path),
                "column": column,
                "schema_config": {
                    "mode": "observed",
                    "guaranteed_fields": declared_fields,
                },
                "on_validation_failure": "discard",
            },
        ),
        transforms=transforms,
        sinks={
            "main": SinkSettings(
                plugin="csv",
                on_write_failure="discard",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
        },
    )
    instances = instantiate_plugins_from_config(config)
    return ExecutionGraph.from_plugin_instances(
        source=instances.source,
        source_settings=instances.source_settings,
        transforms=instances.transforms,
        sinks=instances.sinks,
        aggregations=instances.aggregations,
        gates=list(config.gates),
        coalesce_settings=list(config.coalesce) if config.coalesce else None,
    )


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(spec=pass_through_chain_spec())
def test_walkers_agree_on_chain_topologies(
    spec: dict[str, object],
    tmp_path_factory: object,
) -> None:
    """Composer and runtime walkers must produce the same effective guarantees.

    For every transform in the generated chain, compare:

    - Composer: ``producer_guarantees`` recorded on the outbound edge in
      ``ValidationSummary.edge_contracts`` (populated by
      ``_effective_producer_guarantees`` — the composer preview walker).
    - Runtime: ``ExecutionGraph.get_effective_guaranteed_fields(node_id)``
      — the runtime DAG walker.

    Disagreement means the two walkers' traversals or the inputs they
    compute before calling ``compose_propagation`` have drifted. The
    shrunk failing spec tells you which chain length + declared-fields
    combination triggers the drift.
    """
    # Hypothesis does not compose with function-scoped fixtures; use the
    # session-scoped tmp_path_factory and mint a new directory per example.
    tmp_dir = tmp_path_factory.mktemp("walker_agree", numbered=True)  # type: ignore[attr-defined]
    text_path = tmp_dir / "input.txt"
    text_path.write_text("hello\n", encoding="utf-8")
    output_path = tmp_dir / "out.csv"

    state = _build_composer_state(
        spec,
        text_path=text_path,
        output_path=output_path,
    )
    composer_summary = state.validate()
    assert composer_summary.is_valid, (
        f"Generated spec must be valid on the composer side. "
        f"Errors: {[(e.component, e.message) for e in composer_summary.errors]!r}. "
        f"Spec: {spec!r}"
    )

    graph = _build_runtime_graph(
        spec,
        text_path=text_path,
        output_path=output_path,
    )

    # Map composer node ID (t1/t2/...) to runtime NodeID via the builder's
    # sequence map — transforms are built in chain order.
    runtime_transform_ids = graph.get_transform_id_map()
    # Sequence key is 0-indexed; composer uses 1-indexed names.
    composer_to_runtime_id: dict[str, str] = {f"t{seq + 1}": node_id for seq, node_id in runtime_transform_ids.items()}

    # Composer edge contracts keyed by producer (``from_id``). Every edge
    # whose consumer declares required_input_fields produces a contract.
    composer_producer_map: dict[str, frozenset[str]] = {
        edge.from_id: frozenset(edge.producer_guarantees) for edge in composer_summary.edge_contracts
    }

    chain_length = int(spec["chain_length"])
    # Build the full list of producer composer-ids: every transform in the
    # chain. We expect the composer to have recorded producer_guarantees for
    # each, since every consumer (next transform or sink) declares required
    # fields.
    checked = 0
    for i in range(chain_length):
        composer_id = f"t{i + 1}"
        if composer_id not in composer_producer_map:
            continue  # no edge_contract recorded for this producer
        if composer_id not in composer_to_runtime_id:
            continue  # no runtime mapping (should not happen in chain)
        runtime_id = composer_to_runtime_id[composer_id]
        runtime_guarantees = graph.get_effective_guaranteed_fields(runtime_id)
        composer_guarantees = composer_producer_map[composer_id]
        assert composer_guarantees == runtime_guarantees, (
            f"Walker disagreement at node {composer_id!r} "
            f"(runtime id {runtime_id!r}): "
            f"composer says {sorted(composer_guarantees)!r}, "
            f"runtime says {sorted(runtime_guarantees)!r}. "
            f"Spec: {spec!r}"
        )
        checked += 1

    # Sanity: the test must actually verify at least one node; otherwise
    # edge_contracts scoping swallowed all the comparisons and the test is
    # vacuous. Every chain has at least one intermediate edge with recorded
    # contract (required_input_fields on transforms + sink guarantees this).
    assert checked >= 1, (
        f"Test made no walker comparisons for spec {spec!r}. edge_contracts: {[e.to_dict() for e in composer_summary.edge_contracts]!r}"
    )
