# Schema Validation Refactor - Testing Tasks (8-10)

> **Previous:** `02-cli-refactor.md` | **Next:** `04-cleanup.md`

This file contains comprehensive testing tasks addressing gaps identified in multi-agent review.

---

## Task 8: Comprehensive Integration Tests + Error Handling

**Files:**
- Create: `tests/integration/test_schema_validation_end_to_end.py`
- Create: `tests/cli/test_plugin_errors.py`

**Purpose:** Add comprehensive test coverage including error handling tests identified as missing in review.

### Step 1: Write end-to-end integration tests

**File:** `tests/integration/test_schema_validation_end_to_end.py`

```python
"""End-to-end integration tests for schema validation."""

import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner
from elspeth.cli import app


def test_compatible_pipeline_passes_validation():
    """Verify compatible pipeline passes validation."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields:
        value: {type: float}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          value: {type: float}

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields:
        field_a: {type: str}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}
  - plugin: passthrough
    options:
      schema:
        fields:
          field_b: {type: int}  # INCOMPATIBLE: requires field_b, gets field_a

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        assert "field_b" in result.output.lower()

    finally:
        config_file.unlink()


def test_aggregation_output_incompatibility_detected():
    """Verify validation detects aggregation output incompatible with sink."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields:
        value: {type: float}

aggregations:
  - name: stats
    plugin: batch_stats
    trigger:
      count: 10
    options:
      schema:
        fields:
          value: {type: float}
      value_field: value

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          total_records: {type: int}  # INCOMPATIBLE: agg outputs count/sum/mean

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        assert "total_records" in result.output.lower() or "schema" in result.output.lower()

    finally:
        config_file.unlink()


def test_dynamic_schemas_skip_validation():
    """Verify dynamic schemas (None) skip validation without error."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields: dynamic  # Dynamic schema

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        wrong_field: {type: str}  # Aggregation expects 'value', not 'wrong_field'

aggregations:
  - name: stats
    plugin: batch_stats
    trigger:
      count: 10
    options:
      schema:
        fields:
          value: {type: float}  # Requires 'value' field
      value_field: value

sinks:
  output:
    plugin: csv
    options:
      path: out.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
    """Test that outgoing edge from aggregation validates against output_schema."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

aggregations:
  - name: stats
    plugin: batch_stats
    trigger:
      count: 10
    options:
      schema:
        fields:
          value: {type: float}
      value_field: value
    # Outputs: count, sum, mean, etc.

sinks:
  output:
    plugin: csv
    options:
      path: out.csv
      schema:
        fields:
          nonexistent_field: {type: str}  # INCOMPATIBLE with aggregation output

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        # Should complain about missing field from aggregation output
        assert "nonexistent_field" in result.output.lower() or "schema" in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 2: Write plugin error handling tests

**File:** `tests/cli/test_plugin_errors.py`

```python
"""Tests for plugin instantiation error handling."""

import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner
from elspeth.cli import app
from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import ElspethSettings
from pydantic import TypeAdapter


def test_unknown_source_plugin_error():
    """Verify clear error for unknown source plugin."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: nonexistent_source  # Unknown plugin
  options:
    path: test.csv

sinks:
  output:
    plugin: csv
    options:
      path: out.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
        "datasource": {"plugin": "csv", "options": {"path": "test.csv"}},
        "row_plugins": [{"plugin": "nonexistent_transform", "options": {}}],
        "sinks": {"out": {"plugin": "csv", "options": {"path": "out.csv"}}},
        "output_sink": "out"
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    with pytest.raises(ValueError, match="nonexistent_transform"):
        with pytest.raises(ValueError, match="Available"):
            instantiate_plugins_from_config(config)


def test_plugin_initialization_error():
    """Verify plugin __init__() errors are surfaced clearly."""
    # This test requires a plugin that can fail during __init__()
    # For example, if a plugin expects a required option that's missing
    runner = CliRunner()  # FIX: Add missing runner initialization

    config_yaml = """
datasource:
  plugin: csv
  options:
    # Missing required 'path' option
    schema:
      fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: out.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        assert result.exit_code != 0
        # Should show clear error from plugin instantiation
        assert "error" in result.output.lower()

    finally:
        config_file.unlink()


def test_schema_extraction_from_instance():
    """Verify schemas are NOT None after instantiation."""
    config_dict = {
        "datasource": {
            "plugin": "csv",
            "options": {
                "path": "test.csv",
                "schema": {"fields": {"value": {"type": "float"}}}
            }
        },
        "sinks": {
            "out": {
                "plugin": "csv",
                "options": {
                    "path": "out.csv",
                    "schema": {"fields": {"value": {"type": "float"}}}
                }
            }
        },
        "output_sink": "out"
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
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

gates:
  - name: split
    condition: "row['value'] > 50"
    routes:
      true: continue
      false: low_values
    fork_to:
      - branch_high
      - branch_low

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}

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
        fields:
          value: {type: float}
  low_values:
    plugin: csv
    options:
      path: low.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should pass validation - fork/join pattern with compatible schemas
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_fork_to_separate_sinks_without_coalesce():
    """Test fork pattern where branches go to separate sinks (no coalesce).

    CRITICAL GAP from review: Missing test for fork without coalesce.
    """
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

gates:
  - name: split
    condition: "row['value'] > 50"
    routes:
      true: high_values
      false: low_values
    fork_to:
      - branch_high
      - branch_low

sinks:
  high_values:
    plugin: csv
    options:
      path: high.csv
      schema:
        fields:
          value: {type: float}
  low_values:
    plugin: csv
    options:
      path: low.csv
      schema:
        fields:
          value: {type: float}
  output:
    plugin: csv
    options:
      path: output.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should pass validation - fork branches to separate sinks with compatible schemas
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

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
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

gates:
  - name: split
    condition: "row['value'] > 50"
    routes:
      true: continue
      false: continue
    fork_to:
      - branch_high
      - branch_low

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}

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
        fields:
          value: {type: float}

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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
    """
    runner = CliRunner()

    # Case 1: Dynamic source → Specific sink (should PASS - validation skipped)
    config_yaml_dynamic_to_specific = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic  # Dynamic schema

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        fields:
          field_a: {type: str}  # Specific schema

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml_dynamic_to_specific)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should PASS - dynamic schemas skip validation
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()

    # Case 2: Specific source → Dynamic transform → Specific sink (should PASS)
    config_yaml_mixed = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}  # Specific

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic  # Dynamic transform

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        fields:
          value: {type: float}  # Specific

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml_mixed)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])
        # Should PASS - dynamic schemas in chain skip validation
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 3: Run all tests

Run: `pytest tests/integration/test_schema_validation_end_to_end.py -v`
Run: `pytest tests/cli/test_plugin_errors.py -v`

Expected: All PASS

### Step 4: Commit

```bash
git add tests/integration/test_schema_validation_end_to_end.py tests/cli/test_plugin_errors.py
git commit -m "test: add comprehensive schema validation tests

- Compatible pipeline passes validation
- Transform chain incompatibility detected
- Aggregation output incompatibility detected
- Dynamic schemas skip validation correctly
- Aggregation incoming edge uses input_schema
- Aggregation outgoing edge uses output_schema
- Plugin error handling tests (unknown plugins, init errors)
- Schema extraction verification

Addresses test coverage gaps from multi-agent review

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 8.5: Critical Unit Tests for Schema Validation Gaps

**Files:**
- Test: `tests/core/test_dag.py`

**Purpose:** Close critical test coverage gaps identified in Round 3 QA review.

**CRITICAL CONTEXT:** These tests address P0 blockers:
1. Coalesce incompatible schema behavior was UNDEFINED
2. Aggregation schema transition not tested in single topology
3. Error message diagnostic quality not verified

### Step 1: Write test for coalesce incompatible branch schemas

**File:** `tests/core/test_dag.py`

```python
def test_coalesce_rejects_incompatible_branch_schemas():
    """Coalesce with branches producing different schemas should fail validation.

    CRITICAL GAP: This tests the code added in Task 3 to validate coalesce
    schema compatibility. Without this validation, incompatible schemas would
    merge silently, causing data corruption in the audit trail.

    NOTE: This test manually constructs the graph because the config schema
    doesn't yet support per-branch transform chains. The validation logic
    itself must work regardless of how the graph is constructed.
    """
    from elspeth.contracts import PluginSchema
    from elspeth.core.dag import ExecutionGraph, GraphValidationError, RoutingMode
    from pydantic import BaseModel

    # Define two INCOMPATIBLE schemas
    class SchemaA(BaseModel):
        field_x: int
        common: str

    class SchemaB(BaseModel):
        field_y: float  # Different field!
        common: str

    graph = ExecutionGraph()

    # Build fork/join topology with incompatible branches
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=SchemaA)
    graph.add_node("fork_gate", node_type="gate", plugin_name="fork")

    # Branch A keeps SchemaA
    graph.add_node("transform_a", node_type="transform", plugin_name="passthrough_a",
                   input_schema=SchemaA, output_schema=SchemaA)

    # Branch B transforms to incompatible SchemaB
    graph.add_node("transform_b", node_type="transform", plugin_name="transform_b",
                   input_schema=SchemaA, output_schema=SchemaB)

    graph.add_node("coalesce", node_type="coalesce", plugin_name="merge")
    graph.add_node("sink", node_type="sink", plugin_name="csv")

    # Connect fork
    graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("fork_gate", "transform_a", label="branch_a", mode=RoutingMode.COPY)
    graph.add_edge("fork_gate", "transform_b", label="branch_b", mode=RoutingMode.COPY)

    # Connect join
    graph.add_edge("transform_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
    graph.add_edge("transform_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)
    graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

    # Should crash with clear error about incompatible schemas
    with pytest.raises(GraphValidationError) as exc_info:
        graph.validate()

    error_msg = str(exc_info.value)
    # Verify error message quality
    assert "coalesce" in error_msg.lower()
    assert "incompatible" in error_msg.lower()
    # Should identify both schemas
    assert "SchemaA" in error_msg or "SchemaB" in error_msg
```

### Step 2: Write test for aggregation schema transition

**File:** `tests/core/test_dag.py`

```python
def test_aggregation_schema_transition_in_topology():
    """Aggregation input_schema and output_schema validated on respective edges.

    CRITICAL GAP: Aggregations have dual schemas but existing tests only check
    incoming or outgoing edges separately. This test verifies BOTH in a single
    topology, ensuring the schema transition is validated correctly.
    """
    from elspeth.contracts import PluginSchema
    from elspeth.core.dag import ExecutionGraph, RoutingMode
    from pydantic import BaseModel

    # Input: individual rows with single value
    class RowSchema(BaseModel):
        value: float
        timestamp: str

    # Output: aggregated batch statistics
    class BatchSchema(BaseModel):
        count: int
        sum: float
        mean: float
        min_value: float
        max_value: float

    graph = ExecutionGraph()

    # Source produces rows
    graph.add_node("source", node_type="source", plugin_name="csv",
                   output_schema=RowSchema)

    # Aggregation consumes RowSchema, produces BatchSchema
    graph.add_node("stats_agg", node_type="aggregation", plugin_name="statistics",
                   input_schema=RowSchema,    # Incoming rows
                   output_schema=BatchSchema)  # Outgoing batches

    # Sink consumes BatchSchema
    graph.add_node("sink", node_type="sink", plugin_name="csv",
                   input_schema=BatchSchema)

    # Connect edges
    graph.add_edge("source", "stats_agg", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("stats_agg", "sink", label="continue", mode=RoutingMode.MOVE)

    # Should pass - both edges validated against correct schemas
    graph.validate()  # Should not raise

    # Verify the validation actually checked both edges
    # (If this passes, it means source→agg used input_schema and agg→sink used output_schema)


def test_aggregation_schema_transition_incompatible_output():
    """Aggregation with incompatible output_schema should fail validation.

    Tests that validation uses aggregation's OUTPUT schema for outgoing edges.
    """
    from elspeth.contracts import PluginSchema
    from elspeth.core.dag import ExecutionGraph, GraphValidationError, RoutingMode
    from pydantic import BaseModel

    class RowSchema(BaseModel):
        value: float

    class BatchSchema(BaseModel):
        count: int
        sum: float

    class IncompatibleSinkSchema(BaseModel):
        result: str  # Completely different!

    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=RowSchema)
    graph.add_node("agg", node_type="aggregation", plugin_name="stats",
                   input_schema=RowSchema, output_schema=BatchSchema)
    graph.add_node("sink", node_type="sink", plugin_name="csv",
                   input_schema=IncompatibleSinkSchema)  # Incompatible!

    graph.add_edge("source", "agg", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("agg", "sink", label="continue", mode=RoutingMode.MOVE)

    # Should crash - aggregation output_schema incompatible with sink input_schema
    with pytest.raises(GraphValidationError):
        graph.validate()
```

### Step 3: Write test for error message diagnostic quality

**File:** `tests/core/test_dag.py`

```python
def test_schema_validation_error_includes_diagnostic_details():
    """Schema validation errors should include field names, types, and node names.

    CRITICAL GAP: Existing test only checks that field name appears in error.
    This test verifies the FULL diagnostic quality - all information needed to
    debug the config issue must be present.
    """
    from elspeth.contracts import PluginSchema
    from elspeth.core.dag import ExecutionGraph, GraphValidationError, RoutingMode
    from pydantic import BaseModel

    class ProducerSchema(BaseModel):
        field_a: str
        field_b: int

    class ConsumerSchema(BaseModel):
        field_a: str
        field_c: float  # Requires field_c, not provided by producer

    graph = ExecutionGraph()
    graph.add_node("producer", node_type="transform", plugin_name="prod_plugin",
                   output_schema=ProducerSchema)
    graph.add_node("consumer", node_type="sink", plugin_name="cons_plugin",
                   input_schema=ConsumerSchema)

    graph.add_edge("producer", "consumer", label="continue", mode=RoutingMode.MOVE)

    with pytest.raises(GraphValidationError) as exc_info:
        graph.validate()

    error_msg = str(exc_info.value)

    # MINIMUM requirements for diagnostic quality:
    # 1. Missing field name
    assert "field_c" in error_msg, "Error must identify missing field"

    # 2. Producer node identification
    assert "producer" in error_msg or "prod_plugin" in error_msg, \
        "Error must identify which node is producing incompatible schema"

    # 3. Consumer node identification
    assert "consumer" in error_msg or "cons_plugin" in error_msg, \
        "Error must identify which node requires the missing field"

    # STRETCH GOAL (nice to have, not required for approval):
    # - Field type information ("field_c: float")
    # - List of available fields from producer
    # - Suggestion for fix
```

### Step 4: Run the new unit tests

Run: `pytest tests/core/test_dag.py::test_coalesce_rejects_incompatible_branch_schemas -v`
Expected: PASS (with Task 3 coalesce validation implemented)

Run: `pytest tests/core/test_dag.py::test_aggregation_schema_transition_in_topology -v`
Expected: PASS (existing validation should handle this)

Run: `pytest tests/core/test_dag.py::test_aggregation_schema_transition_incompatible_output -v`
Expected: PASS

Run: `pytest tests/core/test_dag.py::test_schema_validation_error_includes_diagnostic_details -v`
Expected: PASS (may require error message enhancement)

### Step 5: Commit

```bash
git add tests/core/test_dag.py
git commit -m "test(dag): add critical unit tests for schema validation gaps

CRITICAL FIXES from Round 3 QA Review:

1. test_coalesce_rejects_incompatible_branch_schemas
   - Addresses P0 blocker: coalesce incompatible schema behavior was UNDEFINED
   - Tests validation logic added in Task 3
   - Manual graph construction bypasses config schema limitation

2. test_aggregation_schema_transition_in_topology
   - Tests input_schema and output_schema in single topology
   - Verifies both edges validated against correct schemas

3. test_aggregation_schema_transition_incompatible_output
   - Ensures aggregation output_schema validated on outgoing edges

4. test_schema_validation_error_includes_diagnostic_details
   - Verifies error messages include: field name, producer, consumer
   - Ensures errors are actionable for operators

These tests close the 3 critical gaps identified in multi-agent review
that would have left undefined behavior in production.

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 9: Regression Prevention Test

**Files:**
- Create: `tests/integration/test_schema_validation_regression.py`

**Purpose:** Explicitly test that the old bug is fixed - prove validation now detects errors it previously missed.

### Step 1: Write regression test

**File:** `tests/integration/test_schema_validation_regression.py`

```python
"""Regression test proving P0 bug is fixed.

This test explicitly verifies that schema validation now works
when it was previously non-functional.
"""

import tempfile
from pathlib import Path
from typer.testing import CliRunner
from elspeth.cli import app


def test_schema_validation_actually_works():
    """REGRESSION TEST: Prove schema validation detects incompatibilities.

    Before fix: Validation passed even with incompatible schemas
    After fix: Validation fails correctly

    This is the canonical test proving P0-2026-01-24 is resolved.
    """
    runner = CliRunner()

    # This exact config would PASS validation before fix (bug)
    # Should FAIL validation after fix (correct)
    config_yaml = """
datasource:
  plugin: csv
  options:
    path: input.csv
    schema:
      fields:
        field_a: {type: str}
        field_b: {type: int}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}
          field_b: {type: int}

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        fields:
          field_c: {type: float}  # INCOMPATIBLE: requires field_c, gets field_a/field_b

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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


def test_compatible_schemas_still_pass():
    """REGRESSION TEST: Ensure compatible pipelines still work.

    Verify fix doesn't over-restrict - compatible schemas should still pass.
    """
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: input.csv
    schema:
      fields:
        field_a: {type: str}
        field_b: {type: int}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}
          field_b: {type: int}

sinks:
  output:
    plugin: csv
    options:
      path: output.csv
      schema:
        fields:
          field_a: {type: str}  # Compatible: subset of producer schema

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        # Should pass validation
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    finally:
        config_file.unlink()


def test_from_config_no_longer_exists():
    """REGRESSION TEST: Verify deprecated from_config() is deleted.

    After fix, from_config() should not exist (per CLAUDE.md no-legacy policy).
    """
    from elspeth.core.dag import ExecutionGraph

    # from_config should not exist as attribute
    assert not hasattr(ExecutionGraph, "from_config"), \
        "from_config() should be deleted per no-legacy policy"

    # from_plugin_instances should exist
    assert hasattr(ExecutionGraph, "from_plugin_instances"), \
        "from_plugin_instances() is the new API"
```

### Step 2: Run test to verify it passes

Run: `pytest tests/integration/test_schema_validation_regression.py::test_schema_validation_actually_works -v`

Expected: PASS (proving validation now works)

### Step 3: Commit

```bash
git add tests/integration/test_schema_validation_regression.py
git commit -m "test: add regression test proving P0 bug is fixed

- Verify validation detects incompatibilities (previously missed)
- Verify compatible schemas still pass
- Verify from_config() deleted (no legacy code)
- Canonical test proving P0-2026-01-24 resolved

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 10: Run Full Test Suite and Fix Regressions

**Files:**
- Various (fix any test failures)

**Purpose:** Ensure all existing tests still pass with new architecture.

### Step 1: Run full test suite

Run: `pytest tests/ -v`

Expected: Some failures possible (tests using old from_config() API)

### Step 2: Fix any test regressions

For each failing test:
1. Identify if it uses `from_config()`
2. Update to use `from_plugin_instances()` via helper
3. Ensure schemas are available from plugin instances

**Example fix pattern:**

```python
# OLD (broken):
graph = ExecutionGraph.from_config(config, manager)

# NEW (fixed):
plugins = instantiate_plugins_from_config(config)
graph = ExecutionGraph.from_plugin_instances(
    source=plugins["source"],
    transforms=plugins["transforms"],
    sinks=plugins["sinks"],
    aggregations=plugins["aggregations"],
    gates=list(config.gates),
    output_sink=config.output_sink,
)
```

### Step 3: Run type checking

Run: `mypy src/elspeth`

Expected: No new type errors

### Step 4: Run linting

Run: `ruff check src/elspeth`

Expected: No new lint errors

### Step 5: Commit fixes

```bash
git add tests/
git commit -m "fix: update tests to use from_plugin_instances API

- Migrate tests from from_config() to from_plugin_instances()
- Fix type errors and lint issues
- Ensure full test suite passes

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

**Testing Complete! Next:** `04-cleanup.md` for documentation and final cleanup
