"""Shared plugin construction contracts for schema-aware data plugins."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import PluginSchema
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.transforms.passthrough import PassThrough

PluginFactory = Callable[[dict[str, Any]], object]
ConfigFactory = Callable[[Path, dict[str, Any]], dict[str, Any]]
PluginVerifier = Callable[[object], None]


def _source_config(tmp_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(tmp_path / "input.csv"),
        "schema": schema,
        "on_validation_failure": "discard",
    }


def _transform_config(tmp_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    _ = tmp_path
    return {"schema": schema}


def _sink_output_path(tmp_path: Path) -> Path:
    return tmp_path / "output.csv"


def _sink_config(tmp_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(_sink_output_path(tmp_path)),
        "schema": schema,
    }


def _verify_source(plugin: object) -> None:
    assert isinstance(plugin, CSVSource)
    assert issubclass(plugin.output_schema, PluginSchema)
    assert isinstance(plugin.declared_guaranteed_fields, frozenset)


def _verify_transform(plugin: object) -> None:
    assert isinstance(plugin, PassThrough)
    assert issubclass(plugin.input_schema, PluginSchema)
    assert issubclass(plugin.output_schema, PluginSchema)
    assert isinstance(plugin.declared_input_fields, frozenset)


def _verify_sink(plugin: object) -> None:
    assert isinstance(plugin, CSVSink)
    assert issubclass(plugin.input_schema, PluginSchema)
    assert isinstance(plugin.declared_required_fields, frozenset)


PLUGIN_CASES = (
    pytest.param("source", CSVSource, _source_config, _verify_source, id="source"),
    pytest.param("transform", PassThrough, _transform_config, _verify_transform, id="transform"),
    pytest.param("sink", CSVSink, _sink_config, _verify_sink, id="sink"),
)

VALID_SCHEMAS = (
    pytest.param({"mode": "observed"}, id="observed"),
    pytest.param({"mode": "observed", "guaranteed_fields": ["id"]}, id="observed-guarantees"),
    pytest.param({"mode": "fixed", "fields": ["id: int", "name: str"]}, id="fixed"),
    pytest.param({"mode": "flexible", "fields": ["id: int"]}, id="flexible"),
)

INVALID_SCHEMAS = (
    pytest.param({"mode": "fixed", "fields": ["invalid syntax"]}, "Invalid field spec", id="invalid-field-spec"),
    pytest.param({"mode": "fixed", "fields": ["id: nonsense"]}, "Unknown type", id="unknown-field-type"),
    pytest.param({"mode": "fixed"}, "'fields' key is required", id="fixed-missing-fields"),
    pytest.param({"mode": "observed", "fields": ["id: int"]}, "Observed schemas", id="observed-with-fields"),
    pytest.param({"mode": "bogus"}, "Invalid schema mode", id="invalid-mode"),
)


@pytest.mark.parametrize(("role", "plugin_factory", "config_factory", "verify_plugin"), PLUGIN_CASES)
@pytest.mark.parametrize("schema", VALID_SCHEMAS)
def test_data_plugins_accept_valid_schema_on_init(
    role: str,
    plugin_factory: PluginFactory,
    config_factory: ConfigFactory,
    verify_plugin: PluginVerifier,
    schema: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Source, transform, and sink roles must all parse supported schema modes at construction."""
    plugin = plugin_factory(config_factory(tmp_path, deepcopy(schema)))

    assert role in {"source", "transform", "sink"}
    verify_plugin(plugin)


@pytest.mark.parametrize(("role", "plugin_factory", "config_factory", "verify_plugin"), PLUGIN_CASES)
@pytest.mark.parametrize(("schema", "error_match"), INVALID_SCHEMAS)
def test_data_plugins_reject_invalid_schema_on_init(
    role: str,
    plugin_factory: PluginFactory,
    config_factory: ConfigFactory,
    verify_plugin: PluginVerifier,
    schema: dict[str, Any],
    error_match: str,
    tmp_path: Path,
) -> None:
    """Malformed schema declarations must fail before plugin operation begins."""
    _ = role, verify_plugin

    with pytest.raises(PluginConfigError, match=error_match):
        plugin_factory(config_factory(tmp_path, deepcopy(schema)))

    assert not _sink_output_path(tmp_path).exists()
