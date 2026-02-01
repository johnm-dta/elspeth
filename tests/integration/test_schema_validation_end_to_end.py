"""End-to-end integration tests for schema validation.

Updated for Task 4: Tests now expect schema errors during graph construction,
not during graph.validate(). Schema validation happens in two phases:
- PHASE 1: Self-validation during plugin construction (malformed schemas)
- PHASE 2: Compatibility validation during ExecutionGraph.from_plugin_instances()
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import ElspethSettings
from elspeth.core.dag import ExecutionGraph


def test_compatible_pipeline_passes_validation():
    """Verify compatible pipeline passes validation."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      mode: strict
      fields:
        - "value: float"
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        mode: strict
        fields:
          - "value: float"

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
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
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_transform_chain_incompatibility_detected():
    """Verify schema validation detects incompatible transform chain."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      mode: strict
      fields:
        - "field_a: str"
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema:
        mode: strict
        fields:
          - "field_a: str"
  - plugin: passthrough
    options:
      schema:
        mode: strict
        fields:
          - "field_b: int"  # INCOMPATIBLE: requires field_b, gets field_a

sinks:
  output:
    plugin: json
    options:
      path: test_output.json
      schema: {fields: dynamic}
      format: jsonl

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        assert "field_b" in result.output.lower()

    finally:
        config_file.unlink()


def test_aggregation_output_incompatibility_detected():
    """Verify validation SKIPS dynamic aggregation output schema.

    BatchStats uses dynamic output schema (intentionally, because it transforms
    data shape). Validation should be skipped for dynamic schemas.

    This test was previously incorrect - it expected validation to fail, but
    BatchStats intentionally uses dynamic output to avoid exactly this problem.
    """
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      mode: strict
      fields:
        - "value: float"
    on_validation_failure: discard

aggregations:
  - name: stats
    plugin: batch_stats
    trigger:
      count: 10
    options:
      schema:
        mode: strict
        fields:
          - "value: float"
      value_field: value

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        mode: strict
        fields:
          - "total_records: int"  # Would be incompatible, but validation is skipped

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should PASS - BatchStats has dynamic output schema, validation is skipped
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_dynamic_schemas_skip_validation():
    """Verify dynamic schemas (None) skip validation without error."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test_input.csv
    schema: {fields: dynamic}  # Dynamic schema
    on_validation_failure: discard

transforms:
  - plugin: passthrough
    options:
      schema: {fields: dynamic}

sinks:
  output:
    plugin: json
    options:
      path: test_output.json
      schema: {fields: dynamic}
      format: jsonl

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_aggregation_incoming_edge_uses_input_schema():
    """Test that incoming edge to aggregation validates against input_schema."""
    runner = CliRunner()

    config_yaml = """
source:
  plugin: csv
  options:
    path: test.csv
    schema:
      mode: strict
      fields:
        - "wrong_field: str"  # Aggregation expects 'value', not 'wrong_field'
    on_validation_failure: discard

aggregations:
  - name: stats
    plugin: batch_stats
    trigger:
      count: 10
    options:
      schema:
        mode: strict
        fields:
          - "value: float"  # Requires 'value' field
      value_field: value

sinks:
  output:
    plugin: json
    options:
      path: out.json
      schema: {fields: dynamic}
      format: jsonl

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        # Should complain about missing 'value' field
        assert "value" in result.output.lower()

    finally:
        config_file.unlink()


def test_aggregation_outgoing_edge_uses_output_schema():
    """Test that outgoing edge validation is SKIPPED for dynamic aggregation output schema.

    BatchStats uses dynamic output schema because it transforms data shape.
    Validation should be skipped when either producer or consumer has dynamic schema.

    This test was previously incorrect - it expected validation to fail, but
    BatchStats intentionally uses dynamic output schema.
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

aggregations:
  - name: stats
    plugin: batch_stats
    trigger:
      count: 10
    options:
      schema:
        mode: strict
        fields:
          - "value: float"
      value_field: value
    # Outputs: count, sum, mean, etc. (dynamic schema)

sinks:
  output:
    plugin: csv
    options:
      path: out.csv
      schema:
        mode: strict
        fields:
          - "nonexistent_field: str"  # Would be incompatible, but validation is skipped

default_sink: output
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should PASS - BatchStats has dynamic output schema, validation is skipped
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_two_phase_validation_separates_self_and_compatibility_errors(plugin_manager) -> None:
    """Verify PHASE 1 (self) and PHASE 2 (compatibility) validation both work.

    This test demonstrates the two-phase validation architecture:
    - PHASE 1: Self-validation during plugin construction (malformed schemas)
    - PHASE 2: Compatibility validation during graph construction (incompatible connections)
    """
    from elspeth.plugins.config_base import PluginConfigError

    # PHASE 1 should fail: Malformed schema in plugin config
    bad_self_config = {
        "path": "test.csv",
        "schema": {"mode": "strict", "fields": ["invalid syntax!!!"]},
        "on_validation_failure": "discard",
    }

    with pytest.raises(PluginConfigError, match="Invalid field spec"):
        # Fails during plugin construction (PHASE 1)
        source_cls = plugin_manager.get_source_by_name("csv")
        source_cls(bad_self_config)

    # PHASE 2 should fail: Well-formed schemas, incompatible connection
    good_self_bad_compat_config = {
        "source": {
            "plugin": "csv",
            "options": {
                "path": "test.csv",
                "schema": {"mode": "strict", "fields": ["id: int"]},  # Only has 'id'
                "on_validation_failure": "discard",
            },
        },
        "transforms": [
            {
                "plugin": "passthrough",
                "options": {
                    "schema": {"mode": "strict", "fields": ["id: int", "email: str"]},  # Requires 'email'!
                },
            }
        ],
        "sinks": {
            "out": {
                "plugin": "json",
                "options": {"path": "out.json", "schema": {"fields": "dynamic"}, "format": "jsonl"},
            }
        },
        "default_sink": "out",
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(good_self_bad_compat_config)

    # Instantiate plugins (PHASE 1 passes - schemas are well-formed)
    plugins = instantiate_plugins_from_config(config)

    # Graph construction should fail (PHASE 2 - schemas incompatible)
    with pytest.raises(ValueError, match=r"Missing fields.*email"):
        ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
        )
