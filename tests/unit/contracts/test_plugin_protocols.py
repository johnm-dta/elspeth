"""Shared plugin construction contracts for schema-aware data plugins."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import PluginSchema
from elspeth.contracts.plugin_protocols import SourceProtocol, TransformProtocol
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.transforms.passthrough import PassThrough

PluginFactory = Callable[[dict[str, Any]], object]
ConfigFactory = Callable[[Path, dict[str, Any]], dict[str, Any]]
PluginVerifier = Callable[[object, "SchemaExpectation"], None]


@dataclass(frozen=True, slots=True)
class SchemaExpectation:
    schema: dict[str, Any]
    source_guarantees: frozenset[str]
    transform_static_contract: frozenset[str]
    sink_requirements: frozenset[str]


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


def _verify_source(plugin: object, expectation: SchemaExpectation) -> None:
    assert isinstance(plugin, CSVSource)
    assert issubclass(plugin.output_schema, PluginSchema)
    assert plugin.declared_guaranteed_fields == expectation.source_guarantees


def _verify_transform(plugin: object, expectation: SchemaExpectation) -> None:
    assert isinstance(plugin, PassThrough)
    assert issubclass(plugin.input_schema, PluginSchema)
    assert issubclass(plugin.output_schema, PluginSchema)
    assert plugin.declared_input_fields == frozenset()
    assert plugin.effective_static_contract() == expectation.transform_static_contract


def _verify_sink(plugin: object, expectation: SchemaExpectation) -> None:
    assert isinstance(plugin, CSVSink)
    assert issubclass(plugin.input_schema, PluginSchema)
    assert plugin.declared_required_fields == expectation.sink_requirements


PLUGIN_CASES = (
    pytest.param(CSVSource, _source_config, _verify_source, id="source"),
    pytest.param(PassThrough, _transform_config, _verify_transform, id="transform"),
    pytest.param(CSVSink, _sink_config, _verify_sink, id="sink"),
)

VALID_SCHEMA_EXPECTATIONS = (
    pytest.param(
        SchemaExpectation(
            schema={"mode": "observed"},
            source_guarantees=frozenset(),
            transform_static_contract=frozenset(),
            sink_requirements=frozenset(),
        ),
        id="observed",
    ),
    pytest.param(
        SchemaExpectation(
            schema={"mode": "observed", "guaranteed_fields": ["id"]},
            source_guarantees=frozenset({"id"}),
            transform_static_contract=frozenset({"id"}),
            sink_requirements=frozenset(),
        ),
        id="observed-guarantees",
    ),
    pytest.param(
        SchemaExpectation(
            schema={"mode": "observed", "required_fields": ["id"]},
            source_guarantees=frozenset(),
            transform_static_contract=frozenset(),
            sink_requirements=frozenset({"id"}),
        ),
        id="observed-requirements",
    ),
    pytest.param(
        SchemaExpectation(
            schema={"mode": "fixed", "fields": ["id: int", "name: str"]},
            source_guarantees=frozenset({"id", "name"}),
            transform_static_contract=frozenset({"id", "name"}),
            sink_requirements=frozenset({"id", "name"}),
        ),
        id="fixed",
    ),
    pytest.param(
        SchemaExpectation(
            schema={"mode": "flexible", "fields": ["id: int"]},
            source_guarantees=frozenset({"id"}),
            transform_static_contract=frozenset({"id"}),
            sink_requirements=frozenset({"id"}),
        ),
        id="flexible",
    ),
)

INVALID_SCHEMAS = (
    pytest.param({"mode": "fixed", "fields": ["invalid syntax"]}, "Invalid field spec", id="invalid-field-spec"),
    pytest.param({"mode": "fixed", "fields": ["id: nonsense"]}, "Unknown type", id="unknown-field-type"),
    pytest.param({"mode": "fixed"}, "'fields' key is required", id="fixed-missing-fields"),
    pytest.param({"mode": "observed", "fields": ["id: int"]}, "Observed schemas", id="observed-with-fields"),
    pytest.param({"mode": "bogus"}, "Invalid schema mode", id="invalid-mode"),
)


@pytest.mark.parametrize(("plugin_factory", "config_factory", "verify_plugin"), PLUGIN_CASES)
@pytest.mark.parametrize("expectation", VALID_SCHEMA_EXPECTATIONS)
def test_data_plugins_accept_valid_schema_on_init(
    plugin_factory: PluginFactory,
    config_factory: ConfigFactory,
    verify_plugin: PluginVerifier,
    expectation: SchemaExpectation,
    tmp_path: Path,
) -> None:
    """Supported schema modes must produce the role's runtime declaration surface."""
    plugin = plugin_factory(config_factory(tmp_path, deepcopy(expectation.schema)))

    verify_plugin(plugin, expectation)


@pytest.mark.parametrize(("plugin_factory", "config_factory", "verify_plugin"), PLUGIN_CASES)
@pytest.mark.parametrize(("schema", "error_match"), INVALID_SCHEMAS)
def test_data_plugins_reject_invalid_schema_on_init(
    plugin_factory: PluginFactory,
    config_factory: ConfigFactory,
    verify_plugin: PluginVerifier,
    schema: dict[str, Any],
    error_match: str,
    tmp_path: Path,
) -> None:
    """Malformed schema declarations must fail before plugin operation begins."""
    _ = verify_plugin

    with pytest.raises(PluginConfigError, match=error_match):
        plugin_factory(config_factory(tmp_path, deepcopy(schema)))

    assert not _sink_output_path(tmp_path).exists()


def test_source_on_success_matches_transform_post_construction_injection_contract() -> None:
    """Source and transform success routing are both injected after construction."""
    assert SourceProtocol.__annotations__["on_success"] == str | None
    assert TransformProtocol.__annotations__["on_success"] == str | None
