# tests/integration/config/test_schema_validation_regression.py
"""Regression test proving P0 bug is fixed.

This test explicitly verifies that schema validation now works
when it was previously non-functional.
"""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from elspeth.cli import app


def test_schema_validation_actually_works() -> None:
    """REGRESSION TEST: Prove schema validation detects incompatibilities.

    Before fix: Validation passed even with incompatible schemas
    After fix: Validation fails correctly

    This is the canonical test proving P0-2026-01-24 is resolved.
    """
    runner = CliRunner()

    # This exact config would PASS validation before fix (bug)
    # Should FAIL validation after fix (correct)
    config_yaml = """
sources:
  primary:
    plugin: csv
    on_success: source_out
    options:
      path: input.csv
      schema:
        mode: fixed
        fields:
          - "field_a: str"
          - "field_b: int"
      on_validation_failure: discard

transforms:
  - name: passthrough_0
    input: source_out
    on_success: output
    on_error: discard
    plugin: passthrough
    options:
      schema:
        mode: fixed
        fields:
          - "field_a: str"
          - "field_b: int"

sinks:
  output:
    plugin: csv
    on_write_failure: discard
    options:
      path: output.csv
      schema:
        mode: fixed
        fields:
          - "field_c: float"  # INCOMPATIBLE: requires field_c, gets field_a/field_b
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        # CRITICAL ASSERTION: Must fail validation
        assert result.exit_code != 0, "Validation should detect incompatibility"

        # CRITICAL ASSERTION: Must mention the missing field
        assert "field_c" in result.output.lower(), "Error should mention missing field"

        # OPTIONAL: Could also check for "schema" keyword
        assert "schema" in result.output.lower() or "missing" in result.output.lower()

    finally:
        config_file.unlink()


def test_compatible_schemas_still_pass() -> None:
    """REGRESSION TEST: Ensure compatible pipelines still work.

    Verify fix doesn't over-restrict - compatible schemas should still pass.
    """
    runner = CliRunner()

    config_yaml = """
sources:
  primary:
    plugin: csv
    on_success: source_out
    options:
      path: input.csv
      schema:
        mode: fixed
        fields:
          - "field_a: str"
          - "field_b: int"
      on_validation_failure: discard

transforms:
  - name: passthrough_0
    input: source_out
    on_success: output
    on_error: discard
    plugin: passthrough
    options:
      schema:
        mode: fixed
        fields:
          - "field_a: str"
          - "field_b: int"

sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
      schema:
        mode: flexible
        fields:
          - "field_a: str"  # Compatible: subset of producer schema
      format: jsonl
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        # Should pass validation
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_from_plugin_instances_exists() -> None:
    """REGRESSION TEST: Verify new from_plugin_instances() API exists.

    After fix, from_plugin_instances() should be the primary API.
    Note: from_config() will be deleted in Plan 4 Task 11 cleanup.
    """
    from elspeth.core.dag import ExecutionGraph

    # from_plugin_instances should exist
    assert hasattr(ExecutionGraph, "from_plugin_instances"), "from_plugin_instances() is the new API"

    # Verify it's callable
    assert callable(ExecutionGraph.from_plugin_instances), "from_plugin_instances should be a callable method"
