"""Tests for plugin instantiation error handling."""

import tempfile
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import ElspethSettings


def test_unknown_source_plugin_error():
    """Verify clear error for unknown source plugin."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: nonexistent_source  # Unknown plugin
  options:
    path: test.csv

sinks:
  output:
    plugin: csv
    options:
      path: out.csv
      schema:
        mode: strict
        fields:
          - "data: str"

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        # Should show plugin name and available plugins
        assert "nonexistent_source" in result.output.lower()
        assert "available" in result.output.lower()

    finally:
        config_file.unlink()


def test_unknown_transform_plugin_error():
    """Verify clear error for unknown transform plugin."""
    config_dict = {
        "source": {
            "plugin": "csv",
            "options": {"path": "test.csv", "schema": {"fields": "dynamic"}, "on_validation_failure": "discard"},
        },
        "transforms": [{"plugin": "nonexistent_transform", "options": {}}],
        "sinks": {"out": {"plugin": "csv", "options": {"path": "out.csv", "schema": {"mode": "strict", "fields": ["data: str"]}}}},
        "default_sink": "out",
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    with pytest.raises(ValueError) as exc_info:
        instantiate_plugins_from_config(config)

    assert "nonexistent_transform" in str(exc_info.value).lower()
    assert "available" in str(exc_info.value).lower()


def test_plugin_initialization_error():
    """Verify plugin __init__() errors are surfaced clearly."""
    # This test requires a plugin that can fail during __init__()
    # For example, if a plugin expects a required option that's missing
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    # Missing required 'path' option
    schema: {fields: dynamic}

sinks:
  output:
    plugin: csv
    options:
      path: out.csv
      schema:
        mode: strict
        fields:
          - "data: str"

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        # Should show clear error from plugin instantiation
        output_lower = result.output.lower()
        assert "error" in output_lower, f"Expected 'error' in output, got: {result.output}"
        # Error should mention the plugin type or config issue
        # (missing path and on_validation_failure for csv source)
        assert "path" in output_lower or "csv" in output_lower or "source" in output_lower, (
            f"Expected error to mention plugin/config issue, got: {result.output}"
        )
        # Ensure it's NOT reporting as valid
        assert "pipeline configuration valid" not in output_lower, "Should not report as valid when config has missing required fields"

    finally:
        config_file.unlink()


def test_schema_extraction_from_instance():
    """Verify schemas are NOT None after instantiation."""
    config_dict = {
        "source": {
            "plugin": "csv",
            "options": {"path": "test.csv", "schema": {"mode": "strict", "fields": ["value: float"]}, "on_validation_failure": "discard"},
        },
        "sinks": {"out": {"plugin": "csv", "options": {"path": "out.csv", "schema": {"mode": "strict", "fields": ["value: float"]}}}},
        "default_sink": "out",
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    plugins = instantiate_plugins_from_config(config)

    # CRITICAL: Schemas must NOT be None
    assert plugins["source"].output_schema is not None, "Source schema should be populated"
    assert plugins["sinks"]["out"].input_schema is not None, "Sink schema should be populated"


def test_fork_join_validation():
    """Test schema validation across fork/join patterns with coalesce.

    CRITICAL GAP from review: Missing fork/join topology tests.
    """
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      mode: strict
      fields:
        - "value: float"
    on_validation_failure: discard

gates:
  - name: split
    condition: "row['value'] > 50"
    routes:
      "true": fork
      "false": low_values
    fork_to:
      - branch_high
      - branch_low

transforms:
  - plugin: passthrough
    options:
      schema:
        mode: strict
        fields:
          - "value: float"

coalesce:
  - name: merge
    branches:
      - branch_high
      - branch_low
    strategy: first_complete

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "value: float"
  low_values:
    plugin: csv
    options:
      path: low.csv
      schema:
        mode: strict
        fields:
          - "value: float"

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should pass validation - fork/join pattern with compatible schemas
        assert result.exit_code == 0
        # Use exact phrase to avoid matching "invalid"
        assert "pipeline configuration valid" in result.output.lower(), (
            f"Expected 'Pipeline configuration valid' in output, got: {result.output}"
        )

    finally:
        config_file.unlink()


def test_fork_to_separate_sinks_without_coalesce():
    """Test fork pattern where branches go to separate sinks (no coalesce).

    CRITICAL GAP from review: Missing test for fork without coalesce.
    This tests the case where a fork creates copies that independently
    route to different sinks in each branch.
    """
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      mode: strict
      fields:
        - "value: float"
    on_validation_failure: discard

gates:
  - name: split
    condition: "row['value'] > 50"
    routes:
      "true": fork
      "false": continue
    fork_to:
      - high_values
      - low_values

sinks:
  high_values:
    plugin: csv
    options:
      path: high.csv
      schema:
        mode: strict
        fields:
          - "value: float"
  low_values:
    plugin: csv
    options:
      path: low.csv
      schema:
        mode: strict
        fields:
          - "value: float"
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "value: float"

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should pass validation - fork branches to separate sinks with compatible schemas
        assert result.exit_code == 0
        # Use exact phrase to avoid matching "invalid"
        assert "pipeline configuration valid" in result.output.lower(), (
            f"Expected 'Pipeline configuration valid' in output, got: {result.output}"
        )

        # NOTE: This test validates that the configuration is accepted.
        # Verifying the actual DAG structure (edge existence) would require
        # exposing the ExecutionGraph from the CLI, which is not currently
        # available. The validation logic itself is tested at the unit level.

    finally:
        config_file.unlink()


def test_coalesce_compatible_branch_schemas():
    """Test coalesce validation allows compatible schemas from multiple branches.

    LIMITATION: This test uses identical transforms on both branches, so schemas
    are always compatible. True incompatibility testing requires per-branch
    transform configuration (not yet supported in the pipeline config schema).

    This test validates:
    - Fork → coalesce topology is recognized by validation
    - Coalesce nodes are created correctly
    - Compatible schemas pass validation at merge points

    TODO: Add true incompatibility test when per-branch transforms are implemented.
    See Issue: Per-branch transform configuration support
    """
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      mode: strict
      fields:
        - "value: float"
    on_validation_failure: discard

gates:
  - name: split
    condition: "row['value'] > 50"
    routes:
      "true": fork
      "false": continue
    fork_to:
      - branch_high
      - branch_low

transforms:
  - plugin: passthrough
    options:
      schema:
        mode: strict
        fields:
          - "value: float"

# Simulate branches producing different schemas
# (In reality this would require different transform chains per branch)
# This test validates the CONCEPT that coalesce must check schema compatibility

coalesce:
  - name: merge
    branches:
      - branch_high
      - branch_low
    strategy: first_complete

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "value: float"

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should pass - both branches have compatible schemas (same passthrough transform)
        # NOTE: True incompatibility test requires per-branch transform chains (future enhancement)
        assert result.exit_code == 0

    finally:
        config_file.unlink()


def test_dynamic_schema_to_specific_schema_validation():
    """Test validation behavior when dynamic schema feeds specific schema.

    CRITICAL GAP from review: Undefined behavior for dynamic → specific.

    Expected behavior (documented): Dynamic schemas skip validation (pass-through).

    BUG: After refactoring to from_plugin_instances(), schemas are instantiated
    so they're never None. Need to update validation logic to detect dynamic schemas
    by checking model_fields == {} and model_config['extra'] == 'allow'.
    """
    runner = CliRunner()

    # Case 1: Dynamic source → Specific sink (should PASS - validation skipped)
    config_yaml_dynamic_to_specific = """
source:
  plugin: csv
  options:
    path: test.csv
    schema: {fields: dynamic}  # Dynamic schema
    on_validation_failure: discard

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "field_a: str"  # Specific schema

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml_dynamic_to_specific)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should PASS - dynamic schemas skip validation
        assert result.exit_code == 0
        assert "pipeline configuration valid" in result.output.lower()

    finally:
        config_file.unlink()

    # Case 2: Specific source → Dynamic transform → Specific sink (should PASS)
    config_yaml_mixed = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      mode: strict
      fields:
        - "value: float"  # Specific
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema: {fields: dynamic}  # Dynamic transform

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        mode: strict
        fields:
          - "value: float"  # Specific

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml_mixed)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should PASS - dynamic schemas in chain skip validation
        assert result.exit_code == 0
        assert "pipeline configuration valid" in result.output.lower()

    finally:
        config_file.unlink()
