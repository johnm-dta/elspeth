"""Runtime plugin construction helpers.

This module is L3-neutral plugin infrastructure: CLI and web entry points can
use it to build plugin instances, while L2 engine code continues to consume
already-instantiated primitives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.contracts.freeze import freeze_fields

if TYPE_CHECKING:
    from elspeth.core.config import AggregationSettings, ElspethSettings, SourceSettings, TransformSettings
    from elspeth.core.dag.wiring import WiredTransform


@dataclass(frozen=True, slots=True)
class PluginBundle:
    """Pre-instantiated plugin instances from configuration.

    Typed fields enable mypy checking and IDE autocomplete on all access sites.

    Per ADR-025 §1, the source surface is plural by contract. Callers iterate
    ``sources`` / ``source_settings_map`` keyed by source name; PluginBundle no
    longer exposes a singular ``source`` shim.
    """

    sources: Mapping[str, SourceProtocol]
    source_settings_map: Mapping[str, SourceSettings]
    transforms: Sequence[WiredTransform]
    sinks: Mapping[str, SinkProtocol]
    aggregations: Mapping[str, tuple[TransformProtocol, AggregationSettings]]

    def __post_init__(self) -> None:
        from elspeth.contracts.errors import OrchestrationInvariantError

        if not self.sources:
            raise OrchestrationInvariantError("PluginBundle requires at least one source")
        if set(self.sources) != set(self.source_settings_map):
            raise OrchestrationInvariantError(
                f"PluginBundle sources and source_settings_map keys must match. "
                f"sources={sorted(self.sources)}, source_settings_map={sorted(self.source_settings_map)}"
            )
        freeze_fields(self, "sources", "source_settings_map", "transforms", "sinks", "aggregations")


def instantiate_plugins_from_config(
    config: ElspethSettings,
    *,
    preflight_mode: bool = False,
) -> PluginBundle:
    """Instantiate all plugins from configuration.

    Creates plugin instances before graph construction, enabling schema
    extraction from instance attributes.

    When preflight_mode=True, plugin constructors observe
    plugin_preflight_mode_enabled() and may defer side-effectful client setup
    to lifecycle/operation methods.
    """
    from elspeth.core.dag.wiring import WiredTransform
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode

    manager = get_shared_plugin_manager()

    with plugin_preflight_mode(preflight_mode):
        source_settings_by_name = config.sources

        sources = {}
        for source_name, source_config in source_settings_by_name.items():
            source_cls = manager.get_source_by_name(source_config.plugin)
            source_instance = source_cls(dict(source_config.options))
            source_instance.on_success = source_config.on_success
            sources[source_name] = source_instance

        transforms: list[WiredTransform] = []
        for plugin_config in config.transforms:
            transform_cls = manager.get_transform_by_name(plugin_config.plugin)
            transform = transform_cls(dict(plugin_config.options))
            transform.on_success = plugin_config.on_success
            transform.on_error = plugin_config.on_error
            transforms.append(WiredTransform(plugin=transform, settings=plugin_config))

        aggregations = {}
        for agg_config in config.aggregations:
            transform_cls = manager.get_transform_by_name(agg_config.plugin)
            transform = transform_cls(dict(agg_config.options))
            transform.on_success = agg_config.on_success
            transform.on_error = agg_config.on_error

            # Aggregations require transforms that can consume multiple rows.
            # A non-batch-aware transform would silently ignore the trigger.
            if not transform.is_batch_aware:
                raise ValueError(
                    f"Aggregation '{agg_config.name}' uses transform '{agg_config.plugin}' "
                    f"which has is_batch_aware=False. Aggregations require batch-aware "
                    f"transforms that can process multiple rows at once. "
                    f"Use a batch-aware transform like 'batch_stats' or 'batch_replicate', "
                    f"or set is_batch_aware=True on your custom transform."
                )

            aggregations[agg_config.name] = (transform, agg_config)

        sinks = {}
        for sink_name, sink_config in config.sinks.items():
            sink_cls = manager.get_sink_by_name(sink_config.plugin)
            sinks[sink_name] = sink_cls(dict(sink_config.options))
            sinks[sink_name]._on_write_failure = sink_config.on_write_failure

        bundle = PluginBundle(
            sources=sources,
            source_settings_map=source_settings_by_name,
            transforms=transforms,
            sinks=sinks,
            aggregations=aggregations,
        )

    # Value-source compliance check. Single source of truth: every entry point
    # that builds a runtime bundle passes through this function, so hand-authored
    # YAML with hallucinated values is rejected at construction time.
    from elspeth.engine.orchestrator.preflight import validate_value_source_compliance

    validate_value_source_compliance(_value_source_wired_transforms(bundle))
    return bundle


def _value_source_wired_transforms(bundle: PluginBundle) -> tuple[WiredTransform, ...]:
    """Return ordinary and aggregation-backed transforms for value-source checks."""
    from elspeth.core.dag.wiring import WiredTransform

    aggregation_transforms = tuple(
        WiredTransform(plugin=transform, settings=cast("TransformSettings", agg_settings))
        for transform, agg_settings in bundle.aggregations.values()
    )
    return (*bundle.transforms, *aggregation_transforms)


def make_sink_factory(config: ElspethSettings) -> Callable[[str], SinkProtocol]:
    """Create a factory that produces fresh sink instances from config.

    Used by the export phase, which runs after the pipeline's sinks have
    already been closed. The factory creates a new, unstarted instance each time
    it is called. Delayed export sinks are constructed in preflight mode so
    compliant constructors defer credential and client initialization until
    after the export lifecycle capability gate.
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode

    def factory(sink_name: str) -> SinkProtocol:
        if sink_name not in config.sinks:
            raise ValueError(f"Export sink '{sink_name}' not found in sink configuration")
        sink_config = config.sinks[sink_name]
        manager = get_shared_plugin_manager()
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        with plugin_preflight_mode(True):
            sink = sink_cls(dict(sink_config.options))
        sink._on_write_failure = sink_config.on_write_failure
        return sink

    return factory
