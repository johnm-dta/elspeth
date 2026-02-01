"""Tests for CLI helper functions."""

from pathlib import Path

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform


def test_instantiate_plugins_from_config(tmp_path: Path):
    """Verify helper instantiates all plugins from config."""
    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "data: str"

default_sink: output
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)
    plugins = instantiate_plugins_from_config(config)

    # Verify structure
    assert "source" in plugins
    assert "transforms" in plugins
    assert "sinks" in plugins
    assert "aggregations" in plugins

    # Verify types
    assert isinstance(plugins["source"], BaseSource)
    assert len(plugins["transforms"]) == 1
    assert isinstance(plugins["transforms"][0], BaseTransform)
    assert "output" in plugins["sinks"]
    assert isinstance(plugins["sinks"]["output"], BaseSink)

    # CRITICAL: Verify schemas NOT None
    assert plugins["source"].output_schema is not None
    assert plugins["transforms"][0].input_schema is not None

    # Verify plugin identity (not just type) - plugins must have correct name
    assert plugins["source"].name == "csv", f"Expected source plugin 'csv', got '{plugins['source'].name}'"
    assert plugins["transforms"][0].name == "passthrough", f"Expected transform plugin 'passthrough', got '{plugins['transforms'][0].name}'"
    assert plugins["sinks"]["output"].name == "csv", f"Expected sink plugin 'csv', got '{plugins['sinks']['output'].name}'"

    # Verify config propagation - options must be preserved in plugin.config
    assert plugins["source"].config["path"] == "test.csv", f"Source config path not propagated: {plugins['source'].config}"
    assert plugins["source"].config["on_validation_failure"] == "discard", (
        f"Source config on_validation_failure not propagated: {plugins['source'].config}"
    )
    assert plugins["sinks"]["output"].config["path"] == "output.csv", (
        f"Sink config path not propagated: {plugins['sinks']['output'].config}"
    )


def test_instantiate_plugins_raises_on_invalid_plugin():
    """Verify helper raises clear error for unknown plugin."""
    from pydantic import TypeAdapter

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import ElspethSettings

    config_dict = {
        "source": {"plugin": "nonexistent", "options": {}},
        "sinks": {"out": {"plugin": "csv", "options": {"path": "o.csv"}}},
        "default_sink": "out",
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    with pytest.raises(ValueError, match="nonexistent"):
        instantiate_plugins_from_config(config)


def test_aggregation_rejects_non_batch_aware_transform(tmp_path: Path):
    """Aggregations must use batch-aware transforms (is_batch_aware=True).

    Non-batch-aware transforms process rows individually, which means aggregation
    triggers are silently ignored - a dangerous misconfiguration. This test verifies
    that instantiate_plugins_from_config rejects such configurations early.
    """
    # Config with aggregation using 'passthrough' - a non-batch-aware transform
    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

aggregations:
  - name: my_batch
    plugin: passthrough
    options:
      schema:
        fields: dynamic
    trigger:
      count: 10

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "data: str"

default_sink: output
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)

    with pytest.raises(ValueError) as exc_info:
        instantiate_plugins_from_config(config)

    # Verify error message is helpful
    error_msg = str(exc_info.value)
    assert "my_batch" in error_msg, "Error should mention aggregation name"
    assert "passthrough" in error_msg, "Error should mention plugin name"
    assert "is_batch_aware" in error_msg, "Error should explain the requirement"


def test_aggregation_accepts_batch_aware_transform(tmp_path: Path):
    """Aggregations accept batch-aware transforms (is_batch_aware=True).

    This is the positive case - batch_stats is a batch-aware transform
    that should be accepted for aggregation use.
    """
    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic
    on_validation_failure: discard

aggregations:
  - name: stats_batch
    plugin: batch_stats
    options:
      schema:
        fields: dynamic
      value_field: amount
    trigger:
      count: 10

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "data: str"

default_sink: output
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)

    # Should NOT raise - batch_stats has is_batch_aware=True
    plugins = instantiate_plugins_from_config(config)

    assert "aggregations" in plugins
    assert "stats_batch" in plugins["aggregations"]
    transform, _ = plugins["aggregations"]["stats_batch"]
    assert transform.is_batch_aware is True


def test_aggregation_rejects_transform_without_is_batch_aware_attribute():
    """Transforms missing is_batch_aware attribute should be rejected.

    The validation uses getattr(transform, 'is_batch_aware', False) which
    should default to False for transforms that don't define the attribute.
    This tests the fallback behavior.
    """
    from unittest.mock import MagicMock, patch

    from pydantic import TypeAdapter

    from elspeth.core.config import ElspethSettings

    # Create a mock transform class that doesn't have is_batch_aware
    mock_transform = MagicMock()
    del mock_transform.is_batch_aware  # Ensure attribute doesn't exist
    mock_transform.name = "mock_transform"

    # Mock the plugin manager to return our broken transform
    mock_manager = MagicMock()
    mock_manager.get_source_by_name.return_value = MagicMock(return_value=MagicMock())
    mock_manager.get_transform_by_name.return_value = MagicMock(return_value=mock_transform)
    mock_manager.get_sink_by_name.return_value = MagicMock(return_value=MagicMock())

    config_dict = {
        "source": {"plugin": "csv", "options": {"path": "t.csv"}},
        "aggregations": [{"name": "broken_agg", "plugin": "mock", "options": {}, "trigger": {"count": 5}}],
        "sinks": {"out": {"plugin": "csv", "options": {"path": "o.csv"}}},
        "default_sink": "out",
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    with patch("elspeth.cli._get_plugin_manager", return_value=mock_manager), pytest.raises(ValueError) as exc_info:
        instantiate_plugins_from_config(config)

    # Verify the error mentions is_batch_aware requirement
    assert "is_batch_aware" in str(exc_info.value)
