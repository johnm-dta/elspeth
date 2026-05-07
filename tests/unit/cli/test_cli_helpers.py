"""Tests for CLI helper functions."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from elspeth.cli_helpers import PluginBundle, instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.core.dag import WiredTransform
from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform


def test_instantiate_returns_plugin_bundle(tmp_path: Path):
    """instantiate_plugins_from_config returns a PluginBundle dataclass, not a dict."""
    config_yaml = """
source:
  plugin: csv
  on_success: pass1
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

transforms:
  - name: pass1
    plugin: passthrough
    input: pass1
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "data: str"

"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)
    bundle = instantiate_plugins_from_config(config)

    # Must be a PluginBundle, not a dict
    assert isinstance(bundle, PluginBundle)


def test_plugin_bundle_is_frozen(tmp_path: Path):
    """PluginBundle must be immutable (frozen dataclass)."""
    config_yaml = """
source:
  plugin: csv
  on_success: pass1
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

transforms:
  - name: pass1
    plugin: passthrough
    input: pass1
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "data: str"

"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)
    bundle = instantiate_plugins_from_config(config)

    with pytest.raises(AttributeError, match="cannot assign to field"):
        bundle.source = None  # type: ignore[assignment,misc]


def test_plugin_bundle_attribute_access(tmp_path: Path):
    """PluginBundle fields are accessible as typed attributes."""
    config_yaml = """
source:
  plugin: csv
  on_success: pass1
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

transforms:
  - name: pass1
    plugin: passthrough
    input: pass1
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "data: str"

"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)
    bundle = instantiate_plugins_from_config(config)

    # Verify typed attribute access (not dict["key"])
    assert isinstance(bundle.source, BaseSource)
    assert bundle.source.name == "csv"
    assert bundle.source.config["path"] == "test.csv"
    assert bundle.source.config["on_validation_failure"] == "discard"
    assert bundle.source.output_schema is not None

    assert len(bundle.transforms) == 1
    wired = bundle.transforms[0]
    assert isinstance(wired, WiredTransform)
    assert isinstance(wired.plugin, BaseTransform)
    assert wired.plugin.name == "passthrough"
    assert wired.plugin.input_schema is not None

    assert "output" in bundle.sinks
    assert isinstance(bundle.sinks["output"], BaseSink)
    assert bundle.sinks["output"].name == "csv"
    assert bundle.sinks["output"].config["path"] == "output.csv"

    assert isinstance(bundle.aggregations, Mapping)
    assert len(bundle.aggregations) == 0

    # source_settings is the SourceSettings config object
    assert bundle.source_settings is config.source


def test_plugin_bundle_supports_dataclasses_replace(tmp_path: Path):
    """PluginBundle must support dataclasses.replace() for the resume path.

    The resume path in cli.py uses dataclasses.replace() to swap the source
    and sinks while preserving everything else. This test verifies the frozen
    dataclass supports this operation correctly.
    """
    from dataclasses import replace

    from elspeth.plugins.sources.null_source import NullSource

    config_yaml = """
source:
  plugin: csv
  on_success: pass1
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

transforms:
  - name: pass1
    plugin: passthrough
    input: pass1
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "data: str"

"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)
    bundle = instantiate_plugins_from_config(config)

    # Replace source (as resume path does)
    null_source = NullSource({})
    null_source.on_success = bundle.source.on_success
    replaced = replace(bundle, source=null_source)

    # Replaced field changed
    assert isinstance(replaced, PluginBundle)
    assert replaced.source is null_source

    # Unchanged fields preserved by value (deep_freeze detaches MappingProxyType
    # inputs, so identity is not preserved across __post_init__ re-freeze).
    # transforms is a tuple (identity-preserved), sinks/aggregations are proxies (detached).
    assert replaced.transforms is bundle.transforms
    assert replaced.sinks == bundle.sinks
    assert replaced.aggregations == bundle.aggregations
    assert replaced.source_settings is bundle.source_settings


def test_instantiate_plugins_raises_on_invalid_plugin():
    """Verify helper raises clear error for unknown plugin."""
    from pydantic import TypeAdapter

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import ElspethSettings

    config_dict = {
        "source": {"plugin": "nonexistent", "on_success": "out", "options": {}},
        "sinks": {"out": {"plugin": "csv", "on_write_failure": "discard", "options": {"path": "o.csv"}}},
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
  on_success: my_batch
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

aggregations:
  - name: my_batch
    plugin: passthrough
    input: my_batch
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed
    trigger:
      count: 10

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "data: str"

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
  on_success: stats_batch
  options:
    path: test.csv
    schema:
      mode: observed
    on_validation_failure: discard

aggregations:
  - name: stats_batch
    plugin: batch_stats
    input: stats_batch
    on_success: output
    on_error: discard
    options:
      schema:
        mode: observed
      value_field: amount
    trigger:
      count: 10

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "data: str"

"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    config = load_settings(config_file)

    # Should NOT raise - batch_stats has is_batch_aware=True
    bundle = instantiate_plugins_from_config(config)

    assert "stats_batch" in bundle.aggregations
    transform, _ = bundle.aggregations["stats_batch"]
    assert transform.is_batch_aware is True


def test_aggregation_rejects_transform_without_is_batch_aware_attribute():
    """Transforms with is_batch_aware=False should be rejected for aggregation.

    The validation checks transform.is_batch_aware directly (plugins are
    system-owned code, so the attribute always exists). This tests the
    rejection of non-batch-aware transforms via the mock path.
    """
    from unittest.mock import MagicMock, patch

    from pydantic import TypeAdapter

    from elspeth.core.config import ElspethSettings

    # Create a mock transform that explicitly declares is_batch_aware=False
    mock_transform = MagicMock()
    mock_transform.is_batch_aware = False
    mock_transform.name = "mock_transform"

    # Mock the plugin manager to return our broken transform
    mock_manager = MagicMock()
    mock_manager.get_source_by_name.return_value = MagicMock(return_value=MagicMock())
    mock_manager.get_transform_by_name.return_value = MagicMock(return_value=mock_transform)
    mock_manager.get_sink_by_name.return_value = MagicMock(return_value=MagicMock())

    config_dict = {
        "source": {"plugin": "csv", "on_success": "broken_agg", "options": {"path": "t.csv", "on_validation_failure": "discard"}},
        "aggregations": [
            {
                "name": "broken_agg",
                "plugin": "mock",
                "input": "broken_agg",
                "on_success": "out",
                "on_error": "discard",
                "options": {},
                "trigger": {"count": 5},
            }
        ],
        "sinks": {"out": {"plugin": "csv", "on_write_failure": "discard", "options": {"path": "o.csv"}}},
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    with (
        patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=mock_manager),
        pytest.raises(ValueError) as exc_info,
    ):
        instantiate_plugins_from_config(config)

    # Verify the error mentions is_batch_aware requirement
    assert "is_batch_aware" in str(exc_info.value)
