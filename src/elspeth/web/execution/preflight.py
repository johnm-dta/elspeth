"""Shared runtime preflight helpers for web validation and execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.plugins.infrastructure.runtime_factory import PluginBundle, instantiate_plugins_from_config
from elspeth.web.execution.protocol import ValidationSettings
from elspeth.web.paths import (
    NESTED_LOCAL_PATH_OPTION_KEYS,
    SINK_LOCAL_PATH_OPTION_KEYS,
    SOURCE_LOCAL_PATH_OPTION_KEYS,
    resolve_data_path,
)

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

    loaded_config = yaml.safe_load(pipeline_yaml)
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
