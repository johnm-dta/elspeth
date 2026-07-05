"""Pipeline configuration validation functions.

These functions validate route configurations at pipeline initialization,
BEFORE any rows are processed. This catches config errors early instead
of failing mid-run with cryptic errors.

Validations performed:
- Gate route destinations reference existing sinks
- Transform on_error destinations reference existing sinks
- Source quarantine destinations reference existing sinks

IMPORTANT: Import Cycle Prevention
----------------------------------
This module imports RouteValidationError from types.py (a leaf module).
It imports protocols at runtime because isinstance() checks require them.
Other imports use TYPE_CHECKING to avoid cycles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

# Import GateName at runtime - used in function body, not just type hints
from elspeth.contracts import RouteDestination, RouteDestinationKind
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import GateName
from elspeth.engine.orchestrator.types import RouteValidationError

if TYPE_CHECKING:
    from elspeth.contracts import SourceProtocol
    from elspeth.contracts.types import NodeID
    from elspeth.core.config import GateSettings
    from elspeth.engine.orchestrator.plugin_types import RowPlugin
    from elspeth.engine.orchestrator.types import PipelineConfig


def validate_route_destinations(
    route_resolution_map: Mapping[tuple[NodeID, str], RouteDestination],
    available_sinks: set[str],
    transform_id_map: Mapping[int, NodeID],
    transforms: Sequence[RowPlugin],
    config_gate_id_map: Mapping[GateName, NodeID] | None = None,
    config_gates: Sequence[GateSettings] | None = None,
) -> None:
    """Validate all route destinations reference existing sinks.

    Called at pipeline initialization, BEFORE any rows are processed.
    This catches config errors early instead of failing mid-run.

    Args:
        route_resolution_map: Maps (gate_node_id, route_label) -> resolved destination
        available_sinks: Set of sink names from PipelineConfig
        transform_id_map: Maps transform sequence -> node_id
        transforms: List of transform plugins
        config_gate_id_map: Maps config gate name -> node_id
        config_gates: List of config gate settings

    Raises:
        RouteValidationError: If any route references a non-existent sink
    """
    # Build reverse lookup: node_id -> gate name
    # All gates in config_gates MUST have entries in their ID maps
    # (graph construction bug if missing)
    node_id_to_gate_name: dict[str, str] = {}

    # Add config gates to the lookup
    if config_gate_id_map and config_gates:
        for gate_config in config_gates:
            # Graph must have ID for every config gate - crash if missing
            node_id = config_gate_id_map[GateName(gate_config.name)]
            node_id_to_gate_name[node_id] = gate_config.name

    # Check each route destination
    for (gate_node_id, route_label), destination in route_resolution_map.items():
        if destination.kind in (
            RouteDestinationKind.CONTINUE,
            RouteDestinationKind.FORK,
            RouteDestinationKind.PROCESSING_NODE,
            RouteDestinationKind.DISCARD,
        ):
            continue

        if destination.sink_name is None:
            raise OrchestrationInvariantError(
                f"Route destination for gate_node_id={gate_node_id!r}, route_label={route_label!r} has kind='sink' but sink_name is None"
            )

        # destination should be a sink name
        if destination.sink_name not in available_sinks:
            # Every gate in route_resolution_map MUST have a name mapping
            gate_name = node_id_to_gate_name[gate_node_id]
            raise RouteValidationError(
                f"Gate '{gate_name}' can route to '{destination.sink_name}' "
                f"(via route label '{route_label}') but no sink named "
                f"'{destination.sink_name}' exists. Available sinks: {sorted(available_sinks)}"
            )


def validate_pipeline_route_targets(
    *,
    config: PipelineConfig,
    route_resolution_map: Mapping[tuple[NodeID, str], RouteDestination],
    transform_id_map: Mapping[int, NodeID],
    config_gate_id_map: Mapping[GateName, NodeID],
) -> None:
    """Run the full route-target preflight bundle for a pipeline config."""

    available_sinks = set(config.sinks.keys())
    validate_route_destinations(
        route_resolution_map=route_resolution_map,
        available_sinks=available_sinks,
        transform_id_map=transform_id_map,
        transforms=config.transforms,
        config_gate_id_map=config_gate_id_map,
        config_gates=config.gates,
    )
    validate_transform_error_sinks(
        transforms=config.transforms,
        available_sinks=available_sinks,
    )
    for source in config.sources.values():
        validate_source_quarantine_destination(
            source=source,
            available_sinks=available_sinks,
        )
    sink_validation_stubs = {name: SimpleNamespace(on_write_failure=sink._on_write_failure) for name, sink in config.sinks.items()}
    sink_plugins = {name: sink.name for name, sink in config.sinks.items()}
    validate_sink_failsink_destinations(
        sink_configs=sink_validation_stubs,
        available_sinks=available_sinks,
        sink_plugins=sink_plugins,
    )


def validate_transform_error_sinks(
    transforms: Sequence[RowPlugin],
    available_sinks: set[str],
) -> None:
    """Validate all transform on_error destinations reference existing sinks.

    Called at pipeline initialization, BEFORE any rows are processed.
    This catches config errors early instead of failing mid-run with KeyError.

    Args:
        transforms: List of transform plugins
        available_sinks: Set of sink names from PipelineConfig

    Raises:
        RouteValidationError: If any transform on_error references a non-existent sink
    """
    for transform in transforms:
        on_error = transform.on_error
        # on_error is always set (required by TransformSettings) — Tier 1 invariant
        if on_error is None:
            raise OrchestrationInvariantError(
                f"Transform '{transform.name}' has on_error=None — this should be impossible since TransformSettings requires on_error"
            )

        if on_error == "discard":
            # "discard" is a special value, not a sink name
            continue

        # on_error should reference an existing sink
        if on_error not in available_sinks:
            raise RouteValidationError(
                f"Transform '{transform.name}' has on_error='{on_error}' "
                f"but no sink named '{on_error}' exists. "
                f"Available sinks: {sorted(available_sinks)}. "
                f"Use 'discard' to drop error rows without routing."
            )


def validate_source_quarantine_destination(
    source: SourceProtocol,
    available_sinks: set[str],
) -> None:
    """Validate source quarantine destination references an existing sink.

    Called at pipeline initialization, BEFORE any rows are processed.
    This catches config errors early instead of silently dropping quarantined
    rows at runtime.

    Args:
        source: Source plugin instance
        available_sinks: Set of sink names from PipelineConfig

    Raises:
        RouteValidationError: If source on_validation_failure references
            a non-existent sink
    """
    # _on_validation_failure is required by SourceProtocol
    on_validation_failure = source._on_validation_failure

    if on_validation_failure == "discard":
        # "discard" is a special value, not a sink name
        return

    # on_validation_failure should reference an existing sink
    if on_validation_failure not in available_sinks:
        raise RouteValidationError(
            f"Source '{source.name}' has on_validation_failure='{on_validation_failure}' "
            f"but no sink named '{on_validation_failure}' exists. "
            f"Available sinks: {sorted(available_sinks)}. "
            f"Use 'discard' to drop invalid rows without routing."
        )


# Sink plugin names eligible as failsinks (Rule 4 of validate_sink_failsink_destinations).
# MUST be a subset of the runtime sink registry — drift here means engine
# pre-validation accepts a plugin that does not exist, deferring a guaranteed
# runtime crash (PluginNotFoundError in get_sink_by_name) past validation.
# Enforced by tests/unit/web/composer/test_skill_drift.py::TestEngineValidatorPluginDrift::test_allowed_failsink_plugins_subset_of_registered_sinks.
_ALLOWED_FAILSINK_PLUGINS: frozenset[str] = frozenset({"csv", "json"})


def validate_sink_failsink_destinations(
    sink_configs: Mapping[str, Any],
    available_sinks: set[str],
    sink_plugins: Mapping[str, str],
    allowed_failsink_plugins: frozenset[str] = _ALLOWED_FAILSINK_PLUGINS,
) -> None:
    """Validate all sink on_write_failure destinations.

    Called at pipeline initialization, before any rows are processed.
    Parallel to validate_transform_error_sinks() for transform on_error.

    Rules:
    1. 'discard' is always valid
    2. Sink name must exist in available_sinks
    3. Sink cannot reference itself
    4. Target sink must use csv or json plugin type
    5. Target sink must have on_write_failure='discard' (no chains)

    Args:
        sink_configs: Dict of sink_name -> config object with on_write_failure attr.
        available_sinks: Set of all sink names in the pipeline.
        sink_plugins: Dict of sink_name -> plugin type name (e.g., "csv", "chroma_sink").
        allowed_failsink_plugins: Set of plugin types allowed as failsinks.

    Raises:
        RouteValidationError: If any sink's on_write_failure is invalid.
    """
    for sink_name, config in sink_configs.items():
        dest = config.on_write_failure
        if dest is None:
            raise OrchestrationInvariantError(
                f"Sink '{sink_name}' has _on_write_failure=None — injection was skipped. This is a framework bug."
            )
        if dest == "discard":
            continue

        # Rule 2: must exist
        if dest not in available_sinks:
            raise RouteValidationError(
                f"Sink '{sink_name}' on_write_failure references unknown sink '{dest}'. Available sinks: {sorted(available_sinks)}."
            )

        # Rule 3: no self-reference
        if dest == sink_name:
            raise RouteValidationError(f"Sink '{sink_name}' on_write_failure references itself. A sink cannot be its own failsink.")

        # Rule 4: must be a file sink.
        # Direct access — Rule 2 guarantees dest exists in available_sinks,
        # so it must also exist in sink_plugins. If the maps are inconsistent,
        # the KeyError crashes through as a framework bug.
        plugin_type = sink_plugins[dest]
        if plugin_type not in allowed_failsink_plugins:
            raise RouteValidationError(
                f"Sink '{sink_name}' on_write_failure references '{dest}' "
                f"(plugin='{plugin_type}'), but failsinks must use csv or json plugins."
            )

        # Rule 5: no chains — target must use 'discard'
        if dest in sink_configs:
            target_dest = sink_configs[dest].on_write_failure
            if target_dest != "discard":
                raise RouteValidationError(
                    f"Sink '{sink_name}' on_write_failure references '{dest}', "
                    f"but '{dest}' has on_write_failure='{target_dest}'. "
                    f"Failsink targets must have on_write_failure='discard' (no chains)."
                )
