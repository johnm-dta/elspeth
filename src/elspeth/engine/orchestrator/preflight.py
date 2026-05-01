"""Single-source-of-truth pipeline assembly + route-target preflight.

Both the composer ``/validate`` endpoint and the execution service's run path
must assemble :class:`PipelineConfig` identically and run the same four
route-target validators that the orchestrator runs at run-init. This module
owns that contract.

**Layer note.** This module sits in ``engine/`` (L2). It cannot import
``cli_helpers`` (L3) and therefore takes primitives instead of ``PluginBundle``.
Both call sites unpack their bundle locally before calling.

**Mutation note.** Aggregation transforms have ``node_id`` assigned in this
helper (mirrors ``service.py`` runtime path). The mutation is intentional —
the orchestrator depends on ``node_id`` being set before run, and folding
aggregations into the transforms list is the canonical wiring step. Reorder
with care.

**Idempotency note.** The four validators are pure (no I/O, no mutation of
their inputs). The orchestrator runs them again at run-init
(``core.py:1746-1777`` and the resume site at ``:1940-1967``). Calling them
here in the composer/service surfaces failures earlier with a cleaner error
surface; the second call at run-init either passes again or raises the same
error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING

from elspeth.contracts.types import AggregationName
from elspeth.engine.orchestrator.types import PipelineConfig
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
    from elspeth.core.config import AggregationSettings, ElspethSettings
    from elspeth.core.dag import WiredTransform
    from elspeth.core.dag.graph import ExecutionGraph


def assemble_and_validate_pipeline_config(
    *,
    source: SourceProtocol,
    transforms: Sequence[WiredTransform],
    sinks: Mapping[str, SinkProtocol],
    aggregations: Mapping[str, tuple[TransformProtocol, AggregationSettings]],
    settings: ElspethSettings,
    graph: ExecutionGraph,
) -> PipelineConfig:
    """Fold aggregations into transforms, build :class:`PipelineConfig`, and
    run the four orchestrator route-target validators.

    Mirrors the assembly logic in ``src/elspeth/web/execution/service.py`` and
    the four-validator pre-init call site in
    ``src/elspeth/engine/orchestrator/core.py``.

    Args:
        source: Instantiated source plugin.
        transforms: Wired transforms from ``PluginBundle.transforms``.
        sinks: Sink instances keyed by name from ``PluginBundle.sinks``.
        aggregations: Aggregation transforms + settings keyed by aggregation
            name (from ``PluginBundle.aggregations``).
        settings: Full ``ElspethSettings`` (provides ``gates`` and
            ``coalesce`` settings).
        graph: ``ExecutionGraph`` (provides ``aggregation_id_map`` for
            ``node_id`` assignment).

    Returns:
        Assembled ``PipelineConfig`` with aggregations folded into transforms.

    Raises:
        RouteValidationError: A route target references a non-existent sink,
            or a sink failsink violates one of the failsink rules.
        OrchestrationInvariantError: A framework invariant has been broken
            (e.g. a transform ``on_error`` is ``None`` when ``TransformSettings``
            requires it). This is a programmer-bug class — callers in the
            composer/service path should let it propagate to a 500 rather than
            translating it to a per-pipeline validation failure.
    """
    all_transforms: list[TransformProtocol] = [t.plugin for t in transforms]
    aggregation_settings: dict[str, AggregationSettings] = {}
    agg_id_map = graph.get_aggregation_id_map()

    for agg_name, (transform, agg_config) in aggregations.items():
        node_id = agg_id_map[AggregationName(agg_name)]
        aggregation_settings[node_id] = agg_config
        # Intentional mutation: aggregation transforms need node_id set so
        # the orchestrator can index aggregation_settings by node_id at run
        # time. Mirrors service.py:663-667.
        transform.node_id = node_id
        all_transforms.append(transform)

    pipeline_config = PipelineConfig(
        source=source,
        transforms=all_transforms,
        sinks=sinks,
        gates=list(settings.gates),
        aggregation_settings=aggregation_settings,
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else []),
    )

    available_sinks = set(pipeline_config.sinks.keys())

    validate_route_destinations(
        route_resolution_map=graph.get_route_resolution_map(),
        available_sinks=available_sinks,
        transform_id_map=graph.get_transform_id_map(),
        transforms=pipeline_config.transforms,
        config_gate_id_map=graph.get_config_gate_id_map(),
        config_gates=pipeline_config.gates,
    )

    validate_transform_error_sinks(
        transforms=pipeline_config.transforms,
        available_sinks=available_sinks,
    )

    validate_source_quarantine_destination(
        source=pipeline_config.source,
        available_sinks=available_sinks,
    )

    sink_validation_stubs = {name: SimpleNamespace(on_write_failure=sink._on_write_failure) for name, sink in pipeline_config.sinks.items()}
    sink_plugins = {name: sink.name for name, sink in pipeline_config.sinks.items()}
    validate_sink_failsink_destinations(
        sink_configs=sink_validation_stubs,
        available_sinks=available_sinks,
        sink_plugins=sink_plugins,
    )

    return pipeline_config
