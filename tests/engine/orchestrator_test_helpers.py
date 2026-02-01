# tests/engine/orchestrator_test_helpers.py
"""Shared helpers for orchestrator tests.

Extracted from test_orchestrator.py to support split test modules.

This module provides graph construction using the production code path
(ExecutionGraph.from_plugin_instances), ensuring tests exercise the same
code that runs in production. This catches bugs that would otherwise hide
in manually constructed test graphs (see BUG-LINEAGE-01 for historical context).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


def build_production_graph(
    config: PipelineConfig,
    default_sink: str | None = None,
) -> ExecutionGraph:
    """Build graph using production code path (from_plugin_instances).

    Uses the same code path as production, ensuring tests catch bugs that
    would otherwise hide in manually constructed test graphs.

    Args:
        config: PipelineConfig with source, transforms, sinks, gates, etc.
        default_sink: Output sink name. If None, uses "default" or first sink.

    Returns:
        ExecutionGraph built via production factory method.

    Raises:
        GraphValidationError: If graph construction fails validation.

    Example:
        >>> config = PipelineConfig(
        ...     source=my_source,
        ...     transforms=[transform_a, transform_b],
        ...     sinks={"default": my_sink},
        ... )
        >>> graph = build_production_graph(config)
    """
    from elspeth.core.config import AggregationSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.protocols import TransformProtocol

    # Determine default sink
    if default_sink is None:
        if "default" in config.sinks:
            default_sink = "default"
        elif config.sinks:
            default_sink = next(iter(config.sinks))
        else:
            default_sink = ""

    # Separate transforms from gates (gates are handled via config.gates)
    # Only TransformProtocol instances go into the transforms list
    row_transforms: list[TransformProtocol] = []
    aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]] = {}

    for transform in config.transforms:
        # config.transforms is list[RowPlugin] = list[TransformProtocol | GateProtocol]
        # Only TransformProtocol instances should be passed to from_plugin_instances
        if isinstance(transform, TransformProtocol):
            row_transforms.append(transform)

    # Build aggregations dict from config.aggregation_settings
    # The settings dict maps agg_name -> AggregationSettings
    # We need to find/create the transform instance for each
    for agg_name, agg_settings in config.aggregation_settings.items():
        # For test purposes, create a minimal transform if not already present
        # In production, the transform would be instantiated from the plugin name
        # Here we create a passthrough for testing
        from tests.conftest import _TestTransformBase

        class _AggTransform(_TestTransformBase):
            name = agg_settings.plugin

            def process(self, row: dict[str, Any], ctx: Any) -> Any:
                from elspeth.plugins.results import TransformResult

                return TransformResult.success(row, success_reason={"action": "test"})

        aggregations[agg_name] = (_AggTransform(), agg_settings)

    return ExecutionGraph.from_plugin_instances(
        source=config.source,
        transforms=row_transforms,
        sinks=config.sinks,
        aggregations=aggregations,
        gates=list(config.gates),
        default_sink=default_sink,
        coalesce_settings=list(config.coalesce_settings) if config.coalesce_settings else None,
    )
