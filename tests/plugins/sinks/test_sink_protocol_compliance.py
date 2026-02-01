# tests/plugins/sinks/conftest.py
"""Shared fixtures and parametrized tests for sink plugins.

All sink plugins must implement SinkProtocol and have required attributes.
This conftest provides parametrized tests to avoid duplication across
individual sink test files.
"""

from typing import Any

import pytest

from elspeth.plugins.protocols import SinkProtocol

# Schema configs for tests
# CSV and Database sinks require fixed columns (strict mode)
# JSON sink accepts dynamic schemas
STRICT_SCHEMA = {"mode": "strict", "fields": ["id: int"]}
DYNAMIC_SCHEMA = {"fields": "dynamic"}

# Sink configurations for parametrized testing
SINK_CONFIGS = [
    pytest.param(
        "elspeth.plugins.sinks.csv_sink.CSVSink",
        {"path": "/tmp/test.csv", "schema": STRICT_SCHEMA},  # CSV requires strict
        "csv",
        id="csv",
    ),
    pytest.param(
        "elspeth.plugins.sinks.json_sink.JSONSink",
        {"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA},  # JSON accepts dynamic
        "json",
        id="json",
    ),
    pytest.param(
        "elspeth.plugins.sinks.database_sink.DatabaseSink",
        {"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA},  # Database requires strict
        "database",
        id="database",
    ),
]


def _import_sink_class(class_path: str) -> type:
    """Import a sink class from its fully qualified path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class TestSinkProtocolCompliance:
    """Parametrized protocol compliance tests for all sink plugins."""

    @pytest.mark.parametrize("class_path,config,expected_name", SINK_CONFIGS)
    def test_implements_protocol(self, class_path: str, config: dict[str, Any], expected_name: str) -> None:
        """All sinks must implement SinkProtocol."""
        sink_class = _import_sink_class(class_path)
        sink = sink_class(config)
        assert isinstance(sink, SinkProtocol), f"{sink_class.__name__} must implement SinkProtocol"

    @pytest.mark.parametrize("class_path,config,expected_name", SINK_CONFIGS)
    def test_has_required_attributes(self, class_path: str, config: dict[str, Any], expected_name: str) -> None:
        """All sinks must have name and input_schema attributes."""
        sink_class = _import_sink_class(class_path)
        assert sink_class.name == expected_name
        sink = sink_class(config)
        assert hasattr(sink, "input_schema"), f"{sink_class.__name__} must have input_schema attribute"
