"""Shared runtime preflight helpers for web validation and execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from elspeth.core.dag.graph import ExecutionGraph
from elspeth.plugins.infrastructure.runtime_factory import PluginBundle, instantiate_plugins_from_config
from elspeth.web.execution.protocol import ValidationSettings
from elspeth.web.paths import resolve_data_path

RUNTIME_CHECK_PLUGIN_INSTANTIATION = "plugin_instantiation"
RUNTIME_CHECK_GRAPH_STRUCTURE = "graph_structure"
RUNTIME_CHECK_SCHEMA_COMPATIBILITY = "schema_compatibility"

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

    loaded_config = yaml.safe_load(pipeline_yaml)
    if type(loaded_config) is not dict:
        raise TypeError(f"YAML generator produced non-dict top-level value (got {type(loaded_config).__name__})")
    config = cast(dict[str, Any], loaded_config)

    def _rewrite_component_path_options(
        component: dict[str, Any],
        component_path: str,
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
        for key in ("path", "file"):
            if key in path_options and not Path(str(path_options[key])).is_absolute():
                path_options[key] = str(resolve_data_path(str(path_options[key]), data_dir))

    if "source" in config:
        source = config["source"]
        if source is not None and type(source) is not dict:
            raise TypeError(f"YAML generator produced non-dict 'source' value (got {type(source).__name__})")
        if source is not None:
            _rewrite_component_path_options(cast(dict[str, Any], source), "source", require_options=True)

    if "sources" in config:
        sources = config["sources"]
        if sources is not None and type(sources) is not dict:
            raise TypeError(f"YAML generator produced non-dict 'sources' value (got {type(sources).__name__})")
        if sources is not None:
            source_map = cast(dict[str, Any], sources)
            for source_name, source_cfg in source_map.items():
                if type(source_cfg) is not dict:
                    raise TypeError(f"YAML generator produced non-dict source 'sources.{source_name}' value (got {type(source_cfg).__name__})")
                _rewrite_component_path_options(cast(dict[str, Any], source_cfg), f"sources.{source_name}", require_options=True)

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
                    _rewrite_component_path_options(cast(dict[str, Any], sink_cfg), f"sinks.{sink_name}")

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


def instantiate_runtime_plugins(settings: Any, *, preflight_mode: bool = False) -> PluginBundle:
    """Instantiate configured plugins through the production helper."""
    return instantiate_plugins_from_config(settings, preflight_mode=preflight_mode)


def build_runtime_graph(settings: Any, bundle: PluginBundle) -> ExecutionGraph:
    """Build an ExecutionGraph through the production graph factory."""
    return ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
        queues=settings.queues,
    )


def build_validated_runtime_graph(settings: Any) -> RuntimeGraphBundle:
    """Instantiate runtime plugins, build the graph, and run both runtime graph checks.

    Used by execution before running the pipeline, so this must use normal
    runtime mode. Composer/web validation calls instantiate_runtime_plugins()
    directly with preflight_mode=True instead.
    """
    bundle = instantiate_runtime_plugins(settings, preflight_mode=False)
    graph = build_runtime_graph(settings, bundle)
    graph.validate()
    graph.validate_edge_compatibility()
    return RuntimeGraphBundle(plugin_bundle=bundle, graph=graph)
