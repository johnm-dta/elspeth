# tests/property/engine/test_sink_routing_invariant.py
"""Property-based tests for sink routing invariants.

Every row that reaches a sink must be routed to a sink that was configured
in the pipeline. This is a core audit integrity invariant: if a row appears
in a sink_name that doesn't exist in the pipeline configuration, the audit
trail is corrupted.

Properties tested:
- All sink_name values in token_outcomes are in the configured sinks set
- Linear pipelines route all rows to the configured sink
- Gate pipelines route rows only to configured sinks
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.core.config import GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform
from tests.fixtures.stores import MockPayloadStore

# =============================================================================
# Helpers
# =============================================================================


def _build_production_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build graph using production code path (from_plugin_instances).

    Uses explicit source settings + wired transforms.
    """
    from elspeth.contracts import TransformProtocol

    row_transforms: list[TransformProtocol] = []

    for transform in config.transforms:
        if isinstance(transform, TransformProtocol):
            row_transforms.append(transform)

    default_sink = next(iter(config.sinks))
    source_on_success = "source_out" if row_transforms else default_sink
    final_destination = config.gates[0].input if config.gates else default_sink

    if not row_transforms and config.gates:
        source_on_success = config.gates[0].input

    config.sources["primary"].on_success = source_on_success
    source_settings = SourceSettings(plugin=config.sources["primary"].name, on_success=source_on_success, options={})
    wired_transforms = wire_transforms(
        row_transforms,
        source_connection=source_on_success,
        final_sink=final_destination,
    )

    return ExecutionGraph.from_plugin_instances(
        sources={"primary": config.sources["primary"]},
        source_settings_map={"primary": source_settings},
        transforms=wired_transforms,
        sinks=config.sinks,
        aggregations={},
        gates=list(config.gates),
        coalesce_settings=list(config.coalesce_settings) if config.coalesce_settings else None,
    )


def get_all_sink_names_from_outcomes(db: LandscapeDB, run_id: str) -> set[str]:
    """Query all non-NULL sink_name values from token_outcomes for a run."""
    with db.connection() as conn:
        results = conn.execute(
            text("""
                SELECT DISTINCT o.sink_name
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                  AND o.sink_name IS NOT NULL
            """),
            {"run_id": run_id},
        ).fetchall()
        return {r[0] for r in results}


# =============================================================================
# Properties
# =============================================================================


class TestSinkRoutingInvariant:
    """Property: Every RowResult.sink_name must be in the configured sinks set."""

    @given(num_rows=st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_linear_pipeline_sink_names_in_configured_sinks(self, num_rows: int) -> None:
        """Linear pipeline: all sink_names must be in configured sinks."""
        with make_landscape_db() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i, "value": f"row_{i}"} for i in range(num_rows)]

            source = ListSource(rows)
            transform = PassTransform()
            sink = CollectSink("default")

            configured_sinks = {"default": as_sink(sink)}

            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[as_transform(transform)],
                sinks=configured_sinks,
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(
                config,
                graph=_build_production_graph(config),
                payload_store=payload_store,
            )

            # All sink_name values in outcomes must be in configured sinks
            actual_sink_names = get_all_sink_names_from_outcomes(db, run.run_id)
            assert actual_sink_names <= set(configured_sinks.keys()), (
                f"SINK ROUTING VIOLATION: outcomes reference sinks {actual_sink_names - set(configured_sinks.keys())} "
                f"that are not in configured sinks {set(configured_sinks.keys())}"
            )

            # Also verify via RunResult.routed_destinations
            for dest_sink in run.routed_destinations:
                assert dest_sink in configured_sinks, (
                    f"RunResult.routed_destinations contains '{dest_sink}' which is not in configured sinks {set(configured_sinks.keys())}"
                )

    @given(num_rows=st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_multi_sink_gate_pipeline_sink_names_in_configured_sinks(self, num_rows: int) -> None:
        """Gate pipeline with multiple sinks: all sink_names must be configured.

        Uses a fork gate that routes rows to two sinks. Verifies that
        all recorded sink_names are in the configured sinks set.
        """
        from elspeth.core.config import ElspethSettings

        with make_landscape_db() as db:
            payload_store = MockPayloadStore()
            rows = [{"value": i} for i in range(num_rows)]

            source = ListSource(rows, on_success="route_in")
            sink_a = CollectSink("sink_a")
            sink_b = CollectSink("sink_b")

            configured_sinks = {
                "sink_a": as_sink(sink_a),
                "sink_b": as_sink(sink_b),
            }

            gate = GateSettings(
                name="fork_gate",
                input="route_in",
                condition="True",
                routes={"true": "fork", "false": "sink_b"},
                fork_to=["sink_a", "sink_b"],
            )

            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[],
                sinks=configured_sinks,
                gates=[gate],
            )

            graph = ExecutionGraph.from_plugin_instances(
                sources={"primary": as_source(source)},
                source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="route_in", options={})},
                transforms=[],
                sinks=configured_sinks,
                gates=[gate],
                aggregations={},
                coalesce_settings=[],
            )

            elspeth_settings = ElspethSettings(
                sources={"primary": {"plugin": "test", "on_success": "route_in", "options": {}}},
                sinks={
                    "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                    "sink_b": {"plugin": "test", "on_write_failure": "discard"},
                },
                gates=[gate],
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(
                config,
                graph=graph,
                settings=elspeth_settings,
                payload_store=payload_store,
            )

            # All sink_name values in outcomes must be in configured sinks
            actual_sink_names = get_all_sink_names_from_outcomes(db, run.run_id)
            assert actual_sink_names <= set(configured_sinks.keys()), (
                f"SINK ROUTING VIOLATION: outcomes reference sinks {actual_sink_names - set(configured_sinks.keys())} "
                f"that are not in configured sinks {set(configured_sinks.keys())}"
            )

    @given(
        num_rows=st.integers(min_value=1, max_value=5),
        num_transforms=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=20, deadline=None)
    def test_multi_transform_pipeline_sink_names_in_configured_sinks(self, num_rows: int, num_transforms: int) -> None:
        """Multi-transform pipeline: all sink_names must be in configured sinks."""
        with make_landscape_db() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i} for i in range(num_rows)]

            source = ListSource(rows)
            transforms = [PassTransform() for _ in range(num_transforms)]
            sink = CollectSink("default")

            configured_sinks = {"default": as_sink(sink)}

            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[as_transform(t) for t in transforms],
                sinks=configured_sinks,
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(
                config,
                graph=_build_production_graph(config),
                payload_store=payload_store,
            )

            actual_sink_names = get_all_sink_names_from_outcomes(db, run.run_id)
            assert actual_sink_names <= set(configured_sinks.keys()), (
                f"SINK ROUTING VIOLATION: outcomes reference sinks {actual_sink_names - set(configured_sinks.keys())} "
                f"that are not in configured sinks {set(configured_sinks.keys())}"
            )
