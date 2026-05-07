"""Shared runtime preflight helpers for web validation and execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elspeth.cli_helpers import PluginBundle, instantiate_plugins_from_config
from elspeth.core.dag.graph import ExecutionGraph
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

    config = yaml.safe_load(pipeline_yaml)
    if not isinstance(config, dict):
        raise TypeError(f"YAML generator produced non-dict top-level value (got {type(config).__name__})")

    source = config.get("source")
    if source is not None:
        if not isinstance(source, dict):
            raise TypeError(f"YAML generator produced non-dict 'source' value (got {type(source).__name__})")
        opts = source.get("options")
        if opts is not None:
            if not isinstance(opts, dict):
                raise TypeError(f"YAML generator produced non-dict 'source.options' value (got {type(opts).__name__})")
            for key in ("path", "file"):
                if key in opts and not Path(str(opts[key])).is_absolute():
                    opts[key] = str(resolve_data_path(str(opts[key]), data_dir))

    sinks = config.get("sinks")
    if sinks is not None:
        if not isinstance(sinks, dict):
            raise TypeError(f"YAML generator produced non-dict 'sinks' value (got {type(sinks).__name__})")
        for sink_name, sink_cfg in sinks.items():
            if sink_cfg is not None:
                if not isinstance(sink_cfg, dict):
                    raise TypeError(f"YAML generator produced non-dict sink '{sink_name}' value (got {type(sink_cfg).__name__})")
                opts = sink_cfg.get("options")
                if opts is not None:
                    if not isinstance(opts, dict):
                        raise TypeError(f"YAML generator produced non-dict 'sinks.{sink_name}.options' value (got {type(opts).__name__})")
                    for key in ("path", "file"):
                        if key in opts and not Path(str(opts[key])).is_absolute():
                            opts[key] = str(resolve_data_path(str(opts[key]), data_dir))

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
        source=bundle.source,
        source_settings=bundle.source_settings,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
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
