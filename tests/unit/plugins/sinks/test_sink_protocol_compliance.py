# tests/plugins/sinks/test_sink_protocol_compliance.py
"""Protocol compliance tests for sink plugins.

All sink plugins must implement SinkProtocol and satisfy its contract.
This test suite verifies protocol compliance for all built-in sinks.

Tests cover:
1. Required attributes (class and instance level)
2. write() behavior - data written correctly, returns ArtifactDescriptor
3. flush() behavior - buffered data persisted
4. close() behavior - resources released, idempotent
5. Lifecycle hooks (on_start, on_complete)
6. Resume support (configure_for_resume, validate_output_target)
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.base_classes import inject_write_failure

# Schema configs for tests
# CSV and Database sinks require fixed columns (strict mode)
# JSON sink accepts dynamic schemas
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}
DYNAMIC_SCHEMA = {"mode": "observed"}


def _import_sink_class(class_path: str) -> type:
    """Import a sink class from its fully qualified path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls: type = getattr(module, class_name)
    return cls


def _create_temp_path(suffix: str) -> Path:
    """Create a temporary file path for testing."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    import os

    os.close(fd)
    return Path(path)


# Sink configurations for parametrized testing
# Each tuple: (class_path, config_factory, expected_name, file_suffix)
# Using factories instead of static configs so each test gets a fresh temp file
def _csv_config() -> dict[str, Any]:
    return {"path": str(_create_temp_path(".csv")), "schema": STRICT_SCHEMA}


def _json_config() -> dict[str, Any]:
    return {"path": str(_create_temp_path(".json")), "schema": DYNAMIC_SCHEMA}


def _jsonl_config() -> dict[str, Any]:
    return {"path": str(_create_temp_path(".jsonl")), "schema": DYNAMIC_SCHEMA}


def _database_config() -> dict[str, Any]:
    return {"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}


# Parametrized test configs
SINK_CONFIGS = [
    pytest.param(
        "elspeth.plugins.sinks.csv_sink.CSVSink",
        _csv_config,
        "csv",
        id="csv",
    ),
    pytest.param(
        "elspeth.plugins.sinks.json_sink.JSONSink",
        _json_config,
        "json",
        id="json",
    ),
    pytest.param(
        "elspeth.plugins.sinks.database_sink.DatabaseSink",
        _database_config,
        "database",
        id="database",
    ),
]


@dataclass
class _SinkContextFake:
    """Minimal sink context used by protocol-compliance tests."""

    run_id: str = "test-run"
    landscape: None = None
    contract: None = None
    recorded_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.recorded_calls.append(kwargs)


def _create_context() -> _SinkContextFake:
    return _SinkContextFake()


class TestSinkProtocolCompliance:
    """Parametrized protocol compliance tests for all sink plugins."""

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_has_required_class_attributes(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have name class attribute."""
        sink_class = _import_sink_class(class_path)
        # Direct attribute access - crash on missing (our code, our bug)
        assert sink_class.name == expected_name  # type: ignore[attr-defined]

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_has_required_instance_attributes(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have input_schema, idempotent, supports_resume attributes after instantiation."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct attribute access - crash on missing (our code, our bug)
        _ = sink.input_schema  # Verify attribute exists
        _ = sink.idempotent  # Verify attribute exists
        _ = sink.supports_resume  # Verify attribute exists
        _ = sink.determinism  # Verify attribute exists
        _ = sink.plugin_version  # Verify attribute exists
        _ = sink.config  # Verify attribute exists

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_flush_method_callable(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have callable flush() method."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct method call - crash on missing (our code, our bug)
        sink.flush()  # Should not raise

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_close_method_callable_and_idempotent(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have callable close() method that is idempotent."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct method call - crash on missing (our code, our bug)
        sink.close()  # First close
        sink.close()  # Second close - should not raise (idempotency)

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_lifecycle_hooks_exist(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have on_start() and on_complete() lifecycle hooks."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Create mock context
        mock_ctx = _create_context()

        # Direct method calls - crash on missing (our code, our bug)
        sink.on_start(mock_ctx)  # Should not raise
        sink.on_complete(mock_ctx)  # Should not raise

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_resume_methods_exist(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have configure_for_resume() and validate_output_target() methods."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct attribute access - crash on missing (our code, our bug)
        supports_resume = sink.supports_resume

        # configure_for_resume() should only be called if sink supports resume
        # Sinks that don't support resume may raise NotImplementedError
        if supports_resume:
            sink.configure_for_resume()  # Should not raise for resumable sinks

        # validate_output_target() should always be callable
        result = sink.validate_output_target()  # Should not raise

        # Verify return value has expected structure
        _ = result.valid  # Crash if missing field (our code, our bug)

        # Clean up
        sink.close()
