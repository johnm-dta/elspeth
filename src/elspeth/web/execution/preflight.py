"""Shared runtime preflight helpers for web validation and execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from elspeth.contracts import SinkProtocol
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.core.config import ElspethSettings, load_bounded_pipeline_yaml, resolve_config
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.plugins.infrastructure.runtime_factory import (
    PluginBundle,
    instantiate_plugins_from_config,
    make_sink_factory,
)
from elspeth.web.execution import schemas as execution_schemas
from elspeth.web.execution.protocol import ValidationSettings
from elspeth.web.paths import (
    NESTED_LOCAL_PATH_OPTION_KEYS,
    SINK_LOCAL_PATH_OPTION_KEYS,
    SOURCE_LOCAL_PATH_OPTION_KEYS,
    resolve_data_path,
)
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId

RUNTIME_CHECK_PLUGIN_INSTANTIATION = execution_schemas.RUNTIME_CHECK_PLUGIN_INSTANTIATION
RUNTIME_CHECK_GRAPH_STRUCTURE = execution_schemas.RUNTIME_CHECK_GRAPH_STRUCTURE
RUNTIME_CHECK_SCHEMA_COMPATIBILITY = execution_schemas.RUNTIME_CHECK_SCHEMA_COMPATIBILITY

RUNTIME_GRAPH_VALIDATION_CHECKS: tuple[str, str, str] = (
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
)
assert RUNTIME_GRAPH_VALIDATION_CHECKS == (
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
)


@dataclass(slots=True)
class RuntimeGraphBundle:
    """Transient runtime setup result.

    Not frozen: ExecutionGraph is mutable runtime state. This object is not
    persisted and should not cross request boundaries.
    """

    plugin_bundle: PluginBundle
    graph: ExecutionGraph


@trust_boundary(
    tier=3,
    source=(
        "operator/composer-authored pipeline YAML produced by YamlGenerator and re-parsed "
        "here; carries Tier-3 source/sink/transform path options (e.g. persist_directory)"
    ),
    source_param="pipeline_yaml",
    suppresses=("R1",),
    invariant=(
        "raises TypeError on structurally malformed generator output (non-dict top-level, "
        "non-dict source/options/sinks, non-list transforms); optional path keys are read "
        "with .get and confined to data_dir, never coerced or defaulted"
    ),
    test_ref="tests/unit/web/execution/test_service.py::TestResolveYamlPaths::test_non_dict_yaml_raises_type_error",
    test_fingerprint="0b6962a40eb0f2ab584fb2c1de368235d046257d81da266750c8f8011e651c36",
)
def resolve_runtime_yaml_paths(pipeline_yaml: str, data_dir: str) -> str:
    """Rewrite relative source/sink paths in pipeline YAML to absolute paths.

    Plugins call PathConfig.resolved_path() with no base_dir, so relative
    paths resolve against CWD. The validation path-allowlist check approves
    paths relative to data_dir. This function closes that gap by making
    every source/sink path absolute before the YAML reaches the plugin
    layer, so what is allowlisted is what is actually loaded.
    """
    if not isinstance(pipeline_yaml, str):
        raise TypeError(f"YamlGenerator.generate_yaml() must return str; got {type(pipeline_yaml).__name__}")

    loaded_config = load_bounded_pipeline_yaml(pipeline_yaml)
    if type(loaded_config) is not dict:
        raise TypeError(f"YAML generator produced non-dict top-level value (got {type(loaded_config).__name__})")
    config = cast(dict[str, Any], loaded_config)

    def _rewrite_component_path_options(
        component: dict[str, Any],
        component_path: str,
        path_option_keys: tuple[str, ...],
        *,
        require_options: bool = False,
    ) -> None:
        if "options" not in component:
            if require_options:
                raise TypeError(f"YAML generator produced '{component_path}' without required 'options' key")
            return
        opts = component["options"]
        if type(opts) is not dict:
            raise TypeError(f"YAML generator produced non-dict '{component_path}.options' value (got {type(opts).__name__})")
        path_options = cast(dict[str, Any], opts)
        for key in path_option_keys:
            if key in path_options and not Path(str(path_options[key])).is_absolute():
                path_options[key] = str(resolve_data_path(str(path_options[key]), data_dir))

    if "source" in config:
        source = config["source"]
        if source is not None and type(source) is not dict:
            raise TypeError(f"YAML generator produced non-dict 'source' value (got {type(source).__name__})")
        if source is not None:
            _rewrite_component_path_options(
                cast(dict[str, Any], source),
                "source",
                SOURCE_LOCAL_PATH_OPTION_KEYS,
                require_options=True,
            )

    if "sources" in config:
        sources = config["sources"]
        if sources is not None and type(sources) is not dict:
            raise TypeError(f"YAML generator produced non-dict 'sources' value (got {type(sources).__name__})")
        if sources is not None:
            source_map = cast(dict[str, Any], sources)
            for source_name, source_cfg in source_map.items():
                if type(source_cfg) is not dict:
                    raise TypeError(
                        f"YAML generator produced non-dict source 'sources.{source_name}' value (got {type(source_cfg).__name__})"
                    )
                _rewrite_component_path_options(
                    cast(dict[str, Any], source_cfg),
                    f"sources.{source_name}",
                    SOURCE_LOCAL_PATH_OPTION_KEYS,
                    require_options=True,
                )

    if "sinks" in config:
        sinks = config["sinks"]
        if sinks is not None and type(sinks) is not dict:
            raise TypeError(f"YAML generator produced non-dict 'sinks' value (got {type(sinks).__name__})")
        if sinks is not None:
            sink_map = cast(dict[str, Any], sinks)
            for sink_name, sink_cfg in sink_map.items():
                if sink_cfg is not None:
                    if type(sink_cfg) is not dict:
                        raise TypeError(f"YAML generator produced non-dict sink '{sink_name}' value (got {type(sink_cfg).__name__})")
                    _rewrite_component_path_options(
                        cast(dict[str, Any], sink_cfg),
                        f"sinks.{sink_name}",
                        SINK_LOCAL_PATH_OPTION_KEYS,
                    )

    # Nested transform provider_config paths (RAG retrieval transforms carry a
    # local Chroma persist_directory under options.provider_config). Confine
    # the same way as sink paths: rewrite relative values to absolute under
    # data_dir so the allowlist approves what the plugin actually reads/writes.
    transforms = config.get("transforms")
    if transforms is not None:
        if not isinstance(transforms, list):
            raise TypeError(f"YAML generator produced non-list 'transforms' value (got {type(transforms).__name__})")
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            opts = transform.get("options")
            if not isinstance(opts, dict):
                continue
            provider_config = opts.get("provider_config")
            if not isinstance(provider_config, dict):
                continue
            for key in NESTED_LOCAL_PATH_OPTION_KEYS:
                if key in provider_config and not Path(str(provider_config[key])).is_absolute():
                    provider_config[key] = str(resolve_data_path(str(provider_config[key]), data_dir))

    return yaml.dump(config, default_flow_style=False)


def runtime_preflight_settings_hash(settings: ValidationSettings) -> str:
    """Return a non-secret hash of settings that affect runtime preflight.

    Current ValidationSettings exposes only data_dir. If new settings affect
    validation later, add them here deliberately and keep secret-bearing fields
    out of the payload.
    """
    payload = {
        "data_dir": str(Path(settings.data_dir).expanduser().resolve()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configured_plugin_ids(settings: ElspethSettings) -> tuple[PluginId, ...]:
    return (
        *(PluginId("source", source.plugin) for source in settings.sources.values()),
        *(PluginId("transform", transform.plugin) for transform in settings.transforms),
        *(PluginId("transform", aggregation.plugin) for aggregation in settings.aggregations),
        *(PluginId("sink", sink.plugin) for sink in settings.sinks.values()),
    )


def require_settings_plugins_available(
    settings: ElspethSettings,
    plugin_snapshot: PluginAvailabilitySnapshot,
) -> None:
    """Fail before construction when settings exceed one frozen approval."""
    unavailable = tuple(plugin_id for plugin_id in _configured_plugin_ids(settings) if plugin_id not in plugin_snapshot.available)
    if unavailable:
        raise ValueError("Configured plugin is not available in the frozen plugin snapshot.")


def require_settings_sink_available(
    settings: ElspethSettings,
    plugin_snapshot: PluginAvailabilitySnapshot,
    sink_name: str,
) -> None:
    """Check one delayed sink against the same snapshot that approved the run."""
    if sink_name not in settings.sinks:
        raise ValueError(f"Export sink '{sink_name}' not found in sink configuration")
    plugin_id = PluginId("sink", settings.sinks[sink_name].plugin)
    if plugin_id not in plugin_snapshot.available:
        raise ValueError("Configured sink is not available in the frozen plugin snapshot.")


def instantiate_runtime_plugins(
    settings: ElspethSettings,
    *,
    plugin_snapshot: PluginAvailabilitySnapshot,
) -> PluginBundle:
    """Instantiate web runtime plugins only after frozen-snapshot approval."""
    require_settings_plugins_available(settings, plugin_snapshot)
    return instantiate_plugins_from_config(settings, preflight_mode=True)


def make_policy_bound_sink_factory(
    settings: ElspethSettings,
    *,
    plugin_snapshot: PluginAvailabilitySnapshot,
) -> Callable[[str], SinkProtocol]:
    """Bind delayed sink construction to the run's frozen approval."""
    factory = make_sink_factory(settings)

    def policy_bound_factory(sink_name: str) -> SinkProtocol:
        require_settings_sink_available(settings, plugin_snapshot, sink_name)
        return factory(sink_name)

    return policy_bound_factory


def _profiled_plugin_ids(plugin_snapshot: PluginAvailabilitySnapshot) -> frozenset[PluginId]:
    return frozenset(plugin_id for plugin_id, _aliases in plugin_snapshot.usable_profile_aliases)


def _authored_sources(config: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = config.get("sources")
    if isinstance(sources, Mapping):
        return sources
    source = config.get("source")
    if isinstance(source, Mapping):
        return {"source": source}
    return {}


def _authored_named_components(config: Mapping[str, Any], key: str) -> dict[str, Mapping[str, Any]]:
    raw = config.get(key)
    if not isinstance(raw, (list, tuple)):
        return {}
    return {
        str(component["name"]): component for component in raw if isinstance(component, Mapping) and isinstance(component.get("name"), str)
    }


def _authored_options(component: object) -> dict[str, Any] | None:
    if not isinstance(component, Mapping):
        return None
    options = component.get("options")
    if not isinstance(options, Mapping):
        return None
    return cast(dict[str, Any], deep_thaw(options))


@contextmanager
def _audit_safe_plugin_configs(
    bundle: PluginBundle,
    *,
    audit_safe_settings: Mapping[str, Any],
    plugin_snapshot: PluginAvailabilitySnapshot,
) -> Iterator[None]:
    """Expose authored configs while the graph snapshots node audit data."""
    profiled = _profiled_plugin_ids(plugin_snapshot)
    if not profiled:
        yield
        return

    authored_sources = _authored_sources(audit_safe_settings)
    authored_transforms = _authored_named_components(audit_safe_settings, "transforms")
    authored_aggregations = _authored_named_components(audit_safe_settings, "aggregations")
    authored_sinks = audit_safe_settings.get("sinks")
    sink_map = authored_sinks if isinstance(authored_sinks, Mapping) else {}
    restored: list[tuple[Any, Any]] = []

    def substitute(plugin: Any, component: object, plugin_id: PluginId) -> None:
        if plugin_id not in profiled:
            return
        options = _authored_options(component)
        if options is None:
            raise RuntimeError("Audit-safe authored settings are missing profiled plugin options.")
        restored.append((plugin, plugin.config))
        plugin.config = options

    try:
        for source_name, source in bundle.sources.items():
            substitute(source, authored_sources.get(source_name), PluginId("source", source.name))
        for wired in bundle.transforms:
            substitute(
                wired.plugin,
                authored_transforms.get(wired.settings.name),
                PluginId("transform", wired.settings.plugin),
            )
        for aggregation_name, (plugin, aggregation_settings) in bundle.aggregations.items():
            substitute(
                plugin,
                authored_aggregations.get(aggregation_name),
                PluginId("transform", aggregation_settings.plugin),
            )
        for sink_name, sink in bundle.sinks.items():
            substitute(sink, sink_map.get(sink_name), PluginId("sink", sink.name))
        yield
    finally:
        for plugin, executable_config in restored:
            plugin.config = executable_config


def audit_safe_resolved_config(
    settings: ElspethSettings,
    *,
    audit_safe_settings: Mapping[str, Any],
    plugin_snapshot: PluginAvailabilitySnapshot,
) -> dict[str, Any]:
    """Resolve defaults/secrets, then restore authored options for profiled plugins."""
    resolved = resolve_config(settings)
    profiled = _profiled_plugin_ids(plugin_snapshot)
    if not profiled:
        return resolved

    authored_sources = _authored_sources(audit_safe_settings)
    authored_transforms = _authored_named_components(audit_safe_settings, "transforms")
    authored_aggregations = _authored_named_components(audit_safe_settings, "aggregations")
    authored_sinks = audit_safe_settings.get("sinks")
    sink_map = authored_sinks if isinstance(authored_sinks, Mapping) else {}

    def restore_options(component: dict[str, Any], authored: object, kind: str) -> None:
        plugin_name = component.get("plugin")
        if not isinstance(plugin_name, str) or PluginId(cast(Any, kind), plugin_name) not in profiled:
            return
        options = _authored_options(authored)
        if options is None:
            raise RuntimeError("Audit-safe authored settings are missing profiled plugin options.")
        component["options"] = options

    resolved_sources = resolved.get("sources")
    if isinstance(resolved_sources, dict):
        for source_name, component in resolved_sources.items():
            if isinstance(component, dict):
                restore_options(component, authored_sources.get(source_name), "source")
    resolved_transforms = resolved.get("transforms")
    if isinstance(resolved_transforms, list):
        for component in resolved_transforms:
            if isinstance(component, dict):
                restore_options(component, authored_transforms.get(str(component.get("name"))), "transform")
    resolved_aggregations = resolved.get("aggregations")
    if isinstance(resolved_aggregations, list):
        for component in resolved_aggregations:
            if isinstance(component, dict):
                restore_options(component, authored_aggregations.get(str(component.get("name"))), "transform")
    resolved_sinks = resolved.get("sinks")
    if isinstance(resolved_sinks, dict):
        for sink_name, component in resolved_sinks.items():
            if isinstance(component, dict):
                restore_options(component, sink_map.get(sink_name), "sink")
    return resolved


def build_runtime_graph(settings: ElspethSettings, bundle: PluginBundle) -> ExecutionGraph:
    """Build an ExecutionGraph through the production graph factory."""
    from elspeth.engine.orchestrator.preflight import execution_sinks_for_runtime

    return ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=execution_sinks_for_runtime(settings, bundle.sinks),
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
        queues=settings.queues,
    )


def build_validated_runtime_graph(
    settings: ElspethSettings,
    *,
    plugin_snapshot: PluginAvailabilitySnapshot,
    audit_safe_settings: Mapping[str, Any] | None = None,
) -> RuntimeGraphBundle:
    """Instantiate runtime plugins, build the graph, and run both runtime graph checks.

    The web wrapper always constructs under preflight mode. Lifecycle methods
    still run normally once the approved bundle reaches the orchestrator.
    """
    bundle = instantiate_runtime_plugins(settings, plugin_snapshot=plugin_snapshot)
    if audit_safe_settings is None:
        graph = build_runtime_graph(settings, bundle)
    else:
        with _audit_safe_plugin_configs(
            bundle,
            audit_safe_settings=audit_safe_settings,
            plugin_snapshot=plugin_snapshot,
        ):
            graph = build_runtime_graph(settings, bundle)
    graph.validate()
    graph.validate_edge_compatibility()
    return RuntimeGraphBundle(plugin_bundle=bundle, graph=graph)
