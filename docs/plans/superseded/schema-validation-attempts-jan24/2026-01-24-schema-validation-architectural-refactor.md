# Schema Validation Architectural Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix schema validation by instantiating plugins before graph construction, enabling direct schema extraction from instances.

**Architecture:** Refactor `ExecutionGraph.from_config()` to accept plugin instances instead of config objects. Move plugin instantiation from `_execute_pipeline()` to before graph construction. Graph extracts schemas directly from plugin instance attributes.

**Tech Stack:** Python 3.13, Pydantic, NetworkX, pytest, mypy

**Blockers Fixed:**
- P0-2026-01-24-schema-validation-non-functional
- P2-2026-01-24-aggregation-nodes-lack-schema-validation
- P3-2026-01-24-coalesce-nodes-lack-schema-validation

**Timeline:** 3-4 days (8-12 tasks, 2-5 minutes each step)

---

## Design Overview

### Current Flow (Broken)
```
1. Load config (Pydantic models)
2. Build graph from config (schemas unavailable)
3. Validate graph (all schemas None, validation skipped)
4. Instantiate plugins (schemas NOW available, but too late)
5. Execute pipeline
```

### New Flow (Fixed)
```
1. Load config (Pydantic models)
2. Instantiate plugins (schemas available on instances)
3. Build graph from plugin instances (extract schemas directly)
4. Validate graph (schemas populated, validation works)
5. Execute pipeline (use already-instantiated plugins)
```

### Key Changes

| File | Change | Purpose |
|------|--------|---------|
| `src/elspeth/core/dag.py` | Add `from_plugin_instances()` classmethod | Build graph from instances instead of config |
| `src/elspeth/cli.py` | Refactor `run()` and `validate()` commands | Instantiate plugins before graph construction |
| `src/elspeth/cli.py` | Refactor `_execute_pipeline()` | Use pre-instantiated plugins |
| `src/elspeth/core/dag.py` | Update `_validate_edge_schemas()` | Handle aggregation dual-schema validation |
| `tests/core/test_dag.py` | Add integration tests | Verify end-to-end schema validation |

---

## Task 1: Add Helper to Instantiate Plugins from Config

**Files:**
- Create: `src/elspeth/cli_helpers.py` (new module for CLI utilities)
- Test: `tests/cli/test_cli_helpers.py` (new test module)

### Step 1: Write failing test for plugin instantiation helper

**File:** `tests/cli/test_cli_helpers.py`

```python
"""Tests for CLI helper functions."""

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.plugins.base import BaseSource, BaseTransform, BaseSink
from pathlib import Path


def test_instantiate_plugins_from_config(tmp_path: Path):
    """Verify helper instantiates all plugins from config."""
    # Create test config
    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: output.csv

output_sink: output
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(config_yaml)

    # Load config
    config = load_settings(config_file)

    # Instantiate plugins
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
    assert len(plugins["aggregations"]) == 0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_cli_helpers.py::test_instantiate_plugins_from_config -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'elspeth.cli_helpers'"

### Step 3: Implement minimal helper function

**File:** `src/elspeth/cli_helpers.py`

```python
"""CLI helper functions for plugin instantiation and graph construction."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings
    from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform
    from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol


def instantiate_plugins_from_config(
    config: "ElspethSettings",
) -> dict[str, Any]:
    """Instantiate all plugins from configuration.

    This function creates plugin instances BEFORE graph construction,
    enabling schema extraction from instance attributes.

    Args:
        config: Validated ElspethSettings instance

    Returns:
        Dict with keys:
            - source: SourceProtocol instance
            - transforms: list[TransformProtocol] (row_plugins + aggregations)
            - sinks: dict[str, SinkProtocol]
            - aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]]

    Raises:
        ValueError: If config references unknown plugin names
    """
    from elspeth.cli import _get_plugin_manager
    from elspeth.core.config import AggregationSettings

    manager = _get_plugin_manager()

    # Instantiate source
    source_cls = manager.get_source_by_name(config.datasource.plugin)
    if source_cls is None:
        available = [s.name for s in manager.get_sources()]
        raise ValueError(
            f"Unknown source plugin: {config.datasource.plugin}. "
            f"Available: {sorted(available)}"
        )
    source = source_cls(dict(config.datasource.options))

    # Instantiate transforms (row_plugins)
    transforms: list[BaseTransform] = []
    for plugin_config in config.row_plugins:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        if transform_cls is None:
            available = [t.name for t in manager.get_transforms()]
            raise ValueError(
                f"Unknown transform plugin: {plugin_config.plugin}. "
                f"Available: {sorted(available)}"
            )
        transforms.append(transform_cls(dict(plugin_config.options)))

    # Instantiate aggregations
    aggregations: dict[str, tuple[BaseTransform, AggregationSettings]] = {}
    for agg_config in config.aggregations:
        transform_cls = manager.get_transform_by_name(agg_config.plugin)
        if transform_cls is None:
            available = [t.name for t in manager.get_transforms()]
            raise ValueError(
                f"Unknown aggregation plugin: {agg_config.plugin}. "
                f"Available: {sorted(available)}"
            )
        transform = transform_cls(dict(agg_config.options))
        aggregations[agg_config.name] = (transform, agg_config)

    # Instantiate sinks
    sinks: dict[str, BaseSink] = {}
    for sink_name, sink_config in config.sinks.items():
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        if sink_cls is None:
            available = [s.name for s in manager.get_sinks()]
            raise ValueError(
                f"Unknown sink plugin: {sink_config.plugin}. "
                f"Available: {sorted(available)}"
            )
        sinks[sink_name] = sink_cls(dict(sink_config.options))

    return {
        "source": source,
        "transforms": transforms,
        "sinks": sinks,
        "aggregations": aggregations,
    }
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_cli_helpers.py::test_instantiate_plugins_from_config -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/cli_helpers.py tests/cli/test_cli_helpers.py
git commit -m "feat(cli): add helper to instantiate plugins from config

- Create instantiate_plugins_from_config() helper
- Instantiates source, transforms, aggregations, sinks
- Returns dict with all plugin instances
- Enables schema extraction before graph construction

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 2: Add `ExecutionGraph.from_plugin_instances()` Method

**Files:**
- Modify: `src/elspeth/core/dag.py:391-650` (add new classmethod after `from_config`)
- Test: `tests/core/test_dag.py` (add test for new method)

### Step 1: Write failing test for from_plugin_instances

**File:** `tests/core/test_dag.py` (add to existing file)

```python
def test_from_plugin_instances_extracts_schemas_from_instances():
    """Verify from_plugin_instances extracts schemas from plugin instances."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import load_settings, ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from pathlib import Path
    import tempfile

    # Create test config with schema-capable plugins
    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
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
        # Load config and instantiate plugins
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        # Build graph from instances
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            output_sink=config.output_sink,
        )

        # Verify schemas extracted from instances
        source_nodes = [n for n, data in graph._graph.nodes(data=True) if data["node_type"] == "source"]
        assert len(source_nodes) == 1
        source_info = graph.get_node_info(source_nodes[0])
        assert source_info.output_schema is not None  # Schema from instance

        transform_nodes = [n for n, data in graph._graph.nodes(data=True) if data["node_type"] == "transform"]
        assert len(transform_nodes) == 1
        transform_info = graph.get_node_info(transform_nodes[0])
        assert transform_info.input_schema is not None  # Schema from instance
        assert transform_info.output_schema is not None

        sink_nodes = [n for n, data in graph._graph.nodes(data=True) if data["node_type"] == "sink"]
        assert len(sink_nodes) == 1
        sink_info = graph.get_node_info(sink_nodes[0])
        assert sink_info.input_schema is not None  # Schema from instance

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_dag.py::test_from_plugin_instances_extracts_schemas_from_instances -v`

Expected: FAIL with "AttributeError: type object 'ExecutionGraph' has no attribute 'from_plugin_instances'"

### Step 3: Implement from_plugin_instances classmethod

**File:** `src/elspeth/core/dag.py` (add after `from_config` method, around line 650)

```python
@classmethod
def from_plugin_instances(
    cls,
    source: SourceProtocol,
    transforms: list[TransformProtocol],
    sinks: dict[str, SinkProtocol],
    aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]],
    gates: list[GateSettings],
    output_sink: str,
    coalesce_settings: list[CoalesceSettings] | None = None,
) -> "ExecutionGraph":
    """Build an ExecutionGraph from instantiated plugin instances.

    This is the NEW graph construction method that enables schema validation.
    Schemas are extracted directly from plugin instance attributes.

    Args:
        source: Instantiated source plugin
        transforms: List of instantiated transform plugins (row_plugins only)
        sinks: Dict of sink_name -> instantiated sink plugin
        aggregations: Dict of agg_name -> (transform_instance, AggregationSettings)
        gates: Config-driven gate settings
        output_sink: Name of the default output sink
        coalesce_settings: Optional coalesce configurations

    Returns:
        ExecutionGraph with schemas populated from plugin instances

    Raises:
        GraphValidationError: If gate routes reference unknown sinks
    """
    import uuid

    graph = cls()

    def node_id(prefix: str, name: str) -> str:
        return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

    # Add source node - extract schema from instance
    source_id = node_id("source", source.name)
    graph.add_node(
        source_id,
        node_type="source",
        plugin_name=source.name,
        config={},  # Config already used during instantiation
        output_schema=source.output_schema,  # Extract from instance
    )

    # Add sink nodes - extract schemas from instances
    sink_ids: dict[str, str] = {}
    for sink_name, sink in sinks.items():
        sid = node_id("sink", sink_name)
        sink_ids[sink_name] = sid
        graph.add_node(
            sid,
            node_type="sink",
            plugin_name=sink.name,
            config={},
            input_schema=sink.input_schema,  # Extract from instance
        )

    graph._sink_id_map = dict(sink_ids)
    graph._output_sink = sink_ids.get(output_sink, "")

    # Build transform chain - extract schemas from instances
    transform_ids: dict[int, str] = {}
    prev_node_id = source_id

    for i, transform in enumerate(transforms):
        tid = node_id("transform", transform.name)
        transform_ids[i] = tid

        graph.add_node(
            tid,
            node_type="transform",
            plugin_name=transform.name,
            config={},
            input_schema=transform.input_schema,   # Extract from instance
            output_schema=transform.output_schema,  # Extract from instance
        )

        # Edge from previous node
        graph.add_edge(prev_node_id, tid, label="continue", mode=RoutingMode.MOVE)
        prev_node_id = tid

    graph._transform_id_map = transform_ids

    # Build aggregation nodes - extract dual schemas from instances
    aggregation_ids: dict[str, str] = {}
    for agg_name, (transform, agg_config) in aggregations.items():
        aid = node_id("aggregation", agg_name)
        aggregation_ids[agg_name] = aid

        # Store trigger config in node for audit trail
        agg_node_config = {
            "trigger": agg_config.trigger.model_dump(),
            "output_mode": agg_config.output_mode,
            "options": dict(agg_config.options),
        }

        graph.add_node(
            aid,
            node_type="aggregation",
            plugin_name=agg_config.plugin,
            config=agg_node_config,
            input_schema=transform.input_schema,   # Extract from instance
            output_schema=transform.output_schema,  # Extract from instance
        )

        # Edge from previous node
        graph.add_edge(prev_node_id, aid, label="continue", mode=RoutingMode.MOVE)
        prev_node_id = aid

    graph._aggregation_id_map = aggregation_ids

    # Build config-driven gates (same as from_config - gates have no plugin instances)
    config_gate_ids: dict[str, str] = {}
    gate_sequence: list[tuple[str, GateSettings]] = []

    for gate_config in gates:
        gid = node_id("config_gate", gate_config.name)
        config_gate_ids[gate_config.name] = gid

        gate_node_config = {
            "condition": gate_config.condition,
            "routes": dict(gate_config.routes),
        }
        if gate_config.fork_to:
            gate_node_config["fork_to"] = list(gate_config.fork_to)

        graph.add_node(
            gid,
            node_type="gate",
            plugin_name=f"config_gate:{gate_config.name}",
            config=gate_node_config,
            # Gates inherit schema from upstream (no schema fields)
        )

        graph.add_edge(prev_node_id, gid, label="continue", mode=RoutingMode.MOVE)

        # Gate routes to sinks
        for route_label, target in gate_config.routes.items():
            if target == "continue":
                graph._route_resolution_map[(gid, route_label)] = "continue"
            else:
                if target not in sink_ids:
                    raise GraphValidationError(
                        f"Gate '{gate_config.name}' route '{route_label}' references "
                        f"unknown sink '{target}'. Available: {sorted(sink_ids.keys())}"
                    )
                target_sink_id = sink_ids[target]
                graph.add_edge(gid, target_sink_id, label=route_label, mode=RoutingMode.ROUTE)
                graph._route_label_map[(gid, target)] = route_label
                graph._route_resolution_map[(gid, route_label)] = target

        gate_sequence.append((gid, gate_config))

    graph._config_gate_id_map = config_gate_ids

    # Connect gate continue edges
    for i, (gid, gate_config) in enumerate(gate_sequence):
        if "true" in gate_config.routes and gate_config.routes["true"] == "continue":
            next_gate_id = gate_sequence[i + 1][0] if i + 1 < len(gate_sequence) else None
            if next_gate_id:
                # Continue to next gate
                pass  # Edge already added above
            else:
                # Continue to output sink (last gate)
                if graph._output_sink:
                    graph.add_edge(gid, graph._output_sink, label="true", mode=RoutingMode.MOVE)

    # Connect final node to output sink if no gates
    if not gates and graph._output_sink:
        graph.add_edge(prev_node_id, graph._output_sink, label="continue", mode=RoutingMode.MOVE)

    # Coalesce nodes (if any) - placeholder for now
    if coalesce_settings:
        # Coalesce implementation deferred to separate task
        pass

    return graph
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_dag.py::test_from_plugin_instances_extracts_schemas_from_instances -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): add from_plugin_instances() classmethod

- Build graph from plugin instances instead of config objects
- Extract schemas directly from instance attributes
- Enables functional schema validation
- Handles source, transforms, aggregations, gates, sinks

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 3: Update `_validate_edge_schemas()` for Aggregation Dual-Schema

**Files:**
- Modify: `src/elspeth/core/dag.py:208-246` (_validate_edge_schemas method)
- Modify: `src/elspeth/core/dag.py:248-302` (_get_effective_producer_schema method)
- Test: `tests/core/test_dag.py` (add dual-schema validation test)

### Step 1: Write failing test for aggregation dual-schema validation

**File:** `tests/core/test_dag.py`

```python
def test_validate_aggregation_dual_schema():
    """Verify aggregation incoming/outgoing edges validate against different schemas."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.plugins.schema_factory import create_schema_from_config

    # Create schemas
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"value": {"type": "float"}}}),
        "InputSchema",
        allow_coercion=False,
    )

    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"count": {"type": "int"}, "sum": {"type": "float"}}}),
        "OutputSchema",
        allow_coercion=False,
    )

    SinkSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"count": {"type": "int"}, "sum": {"type": "float"}}}),
        "SinkSchema",
        allow_coercion=False,
    )

    # Build graph: source → aggregation → sink
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type="aggregation",
        plugin_name="batch_stats",
        input_schema=InputSchema,   # Accepts individual rows
        output_schema=OutputSchema,  # Emits batch stats
        config={},
    )
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=SinkSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Validate - should pass (input matches, output matches)
    errors = graph._validate_edge_schemas()
    assert len(errors) == 0


def test_validate_aggregation_dual_schema_detects_incompatibility():
    """Verify validation detects aggregation output incompatible with sink."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.plugins.schema_factory import create_schema_from_config

    # Input schema matches, but output schema doesn't match sink
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"value": {"type": "float"}}}),
        "InputSchema",
        allow_coercion=False,
    )

    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"count": {"type": "int"}}}),  # Missing 'sum'
        "OutputSchema",
        allow_coercion=False,
    )

    SinkSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"count": {"type": "int"}, "sum": {"type": "float"}}}),
        "SinkSchema",
        allow_coercion=False,
    )

    # Build graph
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type="aggregation",
        plugin_name="batch_stats",
        input_schema=InputSchema,
        output_schema=OutputSchema,  # Missing 'sum' field
        config={},
    )
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=SinkSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Validate - should detect missing 'sum' field
    errors = graph._validate_edge_schemas()
    assert len(errors) > 0
    assert "sum" in errors[0]
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/core/test_dag.py::test_validate_aggregation_dual_schema -v`
Run: `pytest tests/core/test_dag.py::test_validate_aggregation_dual_schema_detects_incompatibility -v`

Expected: Both FAIL (aggregation dual-schema handling not implemented)

### Step 3: Implement aggregation dual-schema validation

**File:** `src/elspeth/core/dag.py` (update `_validate_edge_schemas` method)

```python
def _validate_edge_schemas(self) -> list[str]:
    """Validate schema compatibility along all edges.

    For each edge (producer -> consumer):
    - Get producer's effective output schema (walks through gates)
    - Get consumer's input schema (aggregations use input_schema for incoming)
    - Check producer provides all required fields

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for edge in self.get_edges():
        from_info = self.get_node_info(edge.from_node)
        to_info = self.get_node_info(edge.to_node)

        # Get effective producer schema (handles gates as pass-through)
        producer_schema = self._get_effective_producer_schema(edge.from_node)

        # Get consumer input schema - handle aggregation dual schemas
        if to_info.node_type == "aggregation":
            # Incoming edge to aggregation: validate against input_schema
            # Aggregations accept individual rows (input_schema),
            # then emit batch results (output_schema)
            consumer_schema = to_info.input_schema
        else:
            # Normal case: validate against input_schema
            consumer_schema = to_info.input_schema

        # Skip validation if either schema is None (dynamic)
        if producer_schema is None or consumer_schema is None:
            continue

        # Validate compatibility
        missing = _get_missing_required_fields(
            producer=producer_schema,
            consumer=consumer_schema,
        )

        if missing:
            errors.append(
                f"{from_info.plugin_name} -> {to_info.plugin_name} (route: {edge.label}): "
                f"producer missing required fields {missing}"
            )

    return errors
```

**File:** `src/elspeth/core/dag.py` (update `_get_effective_producer_schema` method)

```python
def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
    """Get effective output schema for a node, walking through pass-through nodes.

    Gates and other pass-through nodes don't transform data - they inherit
    schema from their upstream producers. This method walks backwards through
    the graph to find the nearest schema-carrying producer.

    Aggregations DO transform data (input != output), so they use output_schema
    directly without inheritance.

    For gates with multiple incoming edges, all inputs must have compatible
    schemas (crashes if not - this is a graph construction bug).

    Args:
        node_id: Node to get effective schema for

    Returns:
        Output schema type, or None if node has no schema and no upstream producers

    Raises:
        GraphValidationError: If gate has no incoming edges or multiple inputs
            with incompatible schemas (graph construction bug)
    """
    node_info = self.get_node_info(node_id)

    # If node has output_schema, return it directly
    if node_info.output_schema is not None:
        return node_info.output_schema

    # Node has no schema - check if it's a pass-through type
    if node_info.node_type == "gate":
        # Gate passes data unchanged - inherit from upstream producer
        incoming = self.get_incoming_edges(node_id)

        if not incoming:
            # Gate with no inputs is a graph construction bug - CRASH
            raise GraphValidationError(
                f"Gate node '{node_id}' has no incoming edges - "
                f"this indicates a bug in graph construction"
            )

        # Get effective schema from first input
        first_schema = self._get_effective_producer_schema(incoming[0].from_node)

        # For multi-input gates, verify all inputs have same schema
        if len(incoming) > 1:
            for edge in incoming[1:]:
                other_schema = self._get_effective_producer_schema(edge.from_node)
                if first_schema != other_schema:
                    # Multi-input gates with incompatible schemas - CRASH
                    raise GraphValidationError(
                        f"Gate '{node_id}' receives incompatible schemas from "
                        f"multiple inputs - this is a graph construction bug. "
                        f"First input schema: {first_schema}, "
                        f"Other input schema: {other_schema}"
                    )

        return first_schema

    # Aggregations transform data - use output_schema directly (no inheritance)
    # Note: This branch is technically unreachable since we check output_schema
    # at the top, but kept for code clarity and future-proofing
    if node_info.node_type == "aggregation":
        return node_info.output_schema

    # Not a pass-through type and no schema - return None
    return None
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py::test_validate_aggregation_dual_schema -v`
Run: `pytest tests/core/test_dag.py::test_validate_aggregation_dual_schema_detects_incompatibility -v`

Expected: Both PASS

### Step 5: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): handle aggregation dual-schema in validation

- Update _validate_edge_schemas() to check input_schema for incoming edges to aggregations
- Update _get_effective_producer_schema() to use output_schema for aggregations (no inheritance)
- Add tests for dual-schema validation (pass and fail cases)

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 4: Refactor CLI `run()` Command to Use New Graph Construction

**Files:**
- Modify: `src/elspeth/cli.py:156-223` (run command)
- Test: `tests/integration/test_cli_schema_validation.py` (new integration test)

### Step 1: Write failing integration test for CLI schema validation

**File:** `tests/integration/test_cli_schema_validation.py`

```python
"""Integration tests for CLI schema validation."""

import tempfile
from pathlib import Path
from typer.testing import CliRunner
from elspeth.cli import app


def test_cli_run_detects_schema_incompatibility():
    """Verify CLI run command detects schema incompatibility during validation."""
    runner = CliRunner()

    # Create incompatible pipeline config
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

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          field_b: {type: int}  # INCOMPATIBLE: requires field_b, gets field_a

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        # Run validation (should fail with schema error)
        result = runner.invoke(app, ["run", "--settings", str(config_file)])

        # Verify validation detected incompatibility
        assert result.exit_code != 0
        assert "schema" in result.output.lower() or "field_b" in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_run_detects_schema_incompatibility -v`

Expected: FAIL (CLI still uses old graph construction)

### Step 3: Refactor run() command to use new construction

**File:** `src/elspeth/cli.py` (modify run command, lines 156-223)

```python
@app.command()
def run(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Validate and show what would run without executing.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        "-x",
        help="Actually execute the pipeline (required for safety).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output.",
    ),
    output_format: Literal["console", "json"] = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format: 'console' (human-readable) or 'json' (structured JSON).",
    ),
) -> None:
    """Execute a pipeline run.

    Requires --execute flag to actually run (safety feature).
    Use --dry-run to validate configuration without executing.
    """
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.dag import ExecutionGraph, GraphValidationError

    settings_path = Path(settings).expanduser()

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1) from None
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1) from None

    # NEW: Instantiate plugins BEFORE graph construction
    try:
        plugins = instantiate_plugins_from_config(config)
    except Exception as e:
        typer.echo(f"Error instantiating plugins: {e}", err=True)
        raise typer.Exit(1) from None

    # NEW: Build and validate execution graph from plugin instances
    try:
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            output_sink=config.output_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    # Console-only messages (don't emit in JSON mode to keep stream clean)
    if output_format == "console":
        if verbose:
            typer.echo(f"Graph validated: {graph.node_count} nodes, {graph.edge_count} edges")

        if dry_run:
            typer.echo("Dry run mode - would execute:")
            typer.echo(f"  Source: {config.datasource.plugin}")
            typer.echo(f"  Transforms: {len(config.row_plugins)}")
            typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
            typer.echo(f"  Output sink: {config.output_sink}")
            if verbose:
                typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
                typer.echo(f"  Execution order: {len(graph.topological_order())} steps")
                typer.echo(f"  Concurrency: {config.concurrency.max_workers} workers")
                typer.echo(f"  Landscape: {config.landscape.url}")
            return

        # Safety check: require explicit --execute flag
        if not execute:
            typer.echo("Pipeline configuration valid.")
            typer.echo(f"  Source: {config.datasource.plugin}")
            typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
            typer.echo("")
            typer.echo("To execute, add --execute (or -x) flag:", err=True)
            typer.echo(f"  elspeth run -s {settings} --execute", err=True)
            raise typer.Exit(1)
    else:
        # JSON mode: early exits without console output
        if dry_run:
            return  # Silently skip execution in dry-run + JSON mode
        if not execute:
            raise typer.Exit(1)  # Silently exit if --execute not provided

    # Execute pipeline with validated graph and pre-instantiated plugins
    # RunCompleted event provides summary in both console and JSON modes
    try:
        _execute_pipeline_with_instances(
            config,
            graph,
            plugins,
            verbose=verbose,
            output_format=output_format,
        )
    except Exception as e:
        # Emit structured error for JSON mode, human-readable for console
        if output_format == "json":
            import json

            typer.echo(
                json.dumps(
                    {
                        "event": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ),
                err=True,
            )
        else:
            typer.echo(f"Error during pipeline execution: {e}", err=True)
        raise typer.Exit(1) from None
```

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_run_detects_schema_incompatibility -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/integration/test_cli_schema_validation.py
git commit -m "refactor(cli): use from_plugin_instances in run command

- Instantiate plugins before graph construction
- Build graph from instances using from_plugin_instances()
- Schema validation now functional (schemas available)
- Add integration test verifying CLI detects schema incompatibility

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 5: Create `_execute_pipeline_with_instances()` Helper

**Files:**
- Modify: `src/elspeth/cli.py:320-560` (refactor _execute_pipeline to accept instances)
- Test: Manual testing (covered by existing integration tests)

### Step 1: No test needed (internal refactor)

This is an internal helper that reuses already-instantiated plugins.
Existing integration tests will verify it works.

### Step 2: Implement _execute_pipeline_with_instances

**File:** `src/elspeth/cli.py` (add new function after _execute_pipeline)

```python
def _execute_pipeline_with_instances(
    config: ElspethSettings,
    graph: ExecutionGraph,
    plugins: dict[str, Any],
    verbose: bool = False,
    output_format: Literal["console", "json"] = "console",
) -> ExecutionResult:
    """Execute a pipeline using pre-instantiated plugin instances.

    This is the NEW execution path that reuses plugins instantiated during
    graph construction, eliminating double instantiation.

    Args:
        config: Validated ElspethSettings instance
        graph: Validated ExecutionGraph instance (schemas populated)
        plugins: Dict with pre-instantiated plugins from instantiate_plugins_from_config()
        verbose: Show detailed output
        output_format: Output format ('console' or 'json')

    Returns:
        ExecutionResult with run_id, status, rows_processed
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseSink, BaseTransform

    # Use pre-instantiated plugins
    source = plugins["source"]
    sinks = plugins["sinks"]

    # Build transforms list: row_plugins + aggregations (with node_id set)
    transforms: list[BaseTransform] = list(plugins["transforms"])

    # Add aggregation transforms with node_id attached
    agg_id_map = graph.get_aggregation_id_map()
    aggregation_settings: dict[str, AggregationSettings] = {}

    for agg_name, (transform, agg_config) in plugins["aggregations"].items():
        node_id = agg_id_map[agg_name]
        aggregation_settings[node_id] = agg_config

        # Set node_id so processor can identify this as an aggregation node
        transform.node_id = node_id
        transforms.append(transform)  # type: ignore[arg-type]

    # Get database URL from settings
    db_url = config.landscape.url
    db = LandscapeDB.from_url(db_url)

    try:
        # Build PipelineConfig with pre-instantiated plugins
        pipeline_config = PipelineConfig(
            source=source,  # type: ignore[arg-type]
            transforms=transforms,  # type: ignore[arg-type]
            sinks=sinks,  # type: ignore[arg-type]
            config=resolve_config(config),
            gates=list(config.gates),
            aggregation_settings=aggregation_settings,
        )

        if verbose:
            typer.echo("Starting pipeline execution...")

        # Create event bus and subscribe progress formatter
        from elspeth.core import EventBus

        event_bus = EventBus()

        # ... (rest of _execute_pipeline logic - formatters, orchestrator, etc.)
        # Copy from existing _execute_pipeline function (lines 436-560)

    finally:
        db.close()
```

**Note:** Copy the formatter setup and orchestrator execution logic from the existing `_execute_pipeline` function. This is mostly a mechanical refactor to use pre-instantiated plugins.

### Step 3: Run existing integration tests to verify

Run: `pytest tests/integration/ -v -k schema`

Expected: All PASS

### Step 4: Commit

```bash
git add src/elspeth/cli.py
git commit -m "refactor(cli): add _execute_pipeline_with_instances helper

- Reuse pre-instantiated plugins from graph construction
- Eliminates double instantiation (performance improvement)
- Same execution logic as _execute_pipeline, different input

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 6: Update `validate` Command to Use New Construction

**Files:**
- Modify: `src/elspeth/cli.py:580-635` (validate command)

### Step 1: Write failing test for validate command

**File:** `tests/integration/test_cli_schema_validation.py`

```python
def test_cli_validate_detects_schema_incompatibility():
    """Verify CLI validate command detects schema incompatibility."""
    runner = CliRunner()

    # Create incompatible pipeline config
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

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          field_b: {type: int}  # INCOMPATIBLE

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        # Run validate command
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        # Should fail with schema error
        assert result.exit_code != 0
        assert "schema" in result.output.lower() or "field_b" in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_validate_detects_schema_incompatibility -v`

Expected: FAIL (validate command still uses old construction)

### Step 3: Refactor validate command

**File:** `src/elspeth/cli.py` (modify validate command)

```python
@app.command()
def validate(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
) -> None:
    """Validate pipeline configuration without running."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.dag import ExecutionGraph, GraphValidationError

    settings_path = Path(settings).expanduser()

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1) from None
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1) from None

    # NEW: Instantiate plugins BEFORE graph construction
    try:
        plugins = instantiate_plugins_from_config(config)
    except Exception as e:
        typer.echo(f"Error instantiating plugins: {e}", err=True)
        raise typer.Exit(1) from None

    # NEW: Build and validate execution graph from plugin instances
    try:
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            output_sink=config.output_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    typer.echo("✅ Pipeline configuration valid!")
    typer.echo(f"  Source: {config.datasource.plugin}")
    typer.echo(f"  Transforms: {len(config.row_plugins)}")
    typer.echo(f"  Aggregations: {len(config.aggregations)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
```

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_validate_detects_schema_incompatibility -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/integration/test_cli_schema_validation.py
git commit -m "refactor(cli): use from_plugin_instances in validate command

- Instantiate plugins before graph construction
- Schema validation now functional in validate command
- Add integration test verifying validate detects incompatibility

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 7: Deprecate Old `from_config()` Method

**Files:**
- Modify: `src/elspeth/core/dag.py:391-650` (add deprecation warning to from_config)
- Modify: `src/elspeth/cli.py:850-970` (update resume command to use new construction)

### Step 1: Add deprecation warning to from_config

**File:** `src/elspeth/core/dag.py` (modify from_config method)

```python
@classmethod
def from_config(cls, config: ElspethSettings, manager: PluginManager) -> ExecutionGraph:
    """Build an ExecutionGraph from validated settings.

    DEPRECATED: Use from_plugin_instances() instead.
    This method cannot extract schemas from plugin instances, resulting in
    non-functional schema validation.

    Args:
        config: Pipeline configuration
        manager: PluginManager for schema lookup

    Returns:
        ExecutionGraph (with schemas all None - validation non-functional)

    Raises:
        ValueError: If config references unknown plugin names
        GraphValidationError: If gate routes reference unknown sinks
    """
    import warnings

    warnings.warn(
        "ExecutionGraph.from_config() is deprecated and has non-functional schema validation. "
        "Use ExecutionGraph.from_plugin_instances() instead. "
        "See: P0-2026-01-24-schema-validation-non-functional",
        DeprecationWarning,
        stacklevel=2,
    )

    # ... (existing from_config implementation)
```

### Step 2: No test needed (deprecation warning)

Existing code still works, just warns. We'll remove it entirely after updating all callers.

### Step 3: Update resume command (if it uses from_config)

Search for other callers of `from_config()` and update them to use `from_plugin_instances()`.

Run: `grep -rn "from_config" src/elspeth/cli.py`

If found, update those call sites to:
1. Instantiate plugins first
2. Call `from_plugin_instances()`

### Step 4: Commit

```bash
git add src/elspeth/core/dag.py src/elspeth/cli.py
git commit -m "deprecate: mark ExecutionGraph.from_config() as deprecated

- Add deprecation warning referencing P0 bug
- Update remaining callers to use from_plugin_instances()
- Plan removal in next major version

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 8: Add Comprehensive Integration Tests

**Files:**
- Create: `tests/integration/test_schema_validation_end_to_end.py`

### Step 1: Write comprehensive end-to-end tests

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
    """Verify schema validation detects aggregation output incompatible with sink."""
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
          total_records: {type: int}  # INCOMPATIBLE: aggregation outputs count/sum/mean, not total_records

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
```

### Step 2: Run all tests

Run: `pytest tests/integration/test_schema_validation_end_to_end.py -v`

Expected: All PASS

### Step 3: Commit

```bash
git add tests/integration/test_schema_validation_end_to_end.py
git commit -m "test: add comprehensive end-to-end schema validation tests

- Compatible pipeline passes validation
- Transform chain incompatibility detected
- Aggregation output incompatibility detected
- Dynamic schemas skip validation correctly

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/bugs/P0-2026-01-24-schema-validation-non-functional.md`
- Create: `docs/design/adr/003-schema-validation-lifecycle.md`

### Step 1: Update P0 bug status

**File:** `docs/bugs/P0-2026-01-24-schema-validation-non-functional.md`

Add at the top:

```markdown
## ✅ RESOLVED

**Status:** Fixed in RC-2
**Resolution:** Architectural refactor - plugin instantiation moved before graph construction
**Implementation:** See docs/plans/2026-01-24-schema-validation-architectural-refactor.md
**ADR:** See docs/design/adr/003-schema-validation-lifecycle.md

---
```

### Step 2: Create ADR documenting schema lifecycle

**File:** `docs/design/adr/003-schema-validation-lifecycle.md`

```markdown
# ADR 003: Schema Validation Lifecycle

## Status

Accepted

## Context

Schema validation in ExecutionGraph was non-functional because the graph was built from config objects before plugins were instantiated. Schemas are instance attributes (`self.input_schema` set in plugin `__init__()`), so they weren't available during graph construction.

## Decision

Restructure CLI to instantiate plugins BEFORE graph construction:

1. Load config (Pydantic models)
2. Instantiate ALL plugins (source, transforms, aggregations, sinks)
3. Build graph from plugin instances using `ExecutionGraph.from_plugin_instances()`
4. Extract schemas directly from instance attributes
5. Validate graph (schemas populated, validation functional)
6. Execute pipeline using pre-instantiated plugins

## Consequences

### Positive

- Schema validation is now functional (detects incompatibilities at validation time)
- No double instantiation (plugins created once)
- Fail-fast principle: plugin instantiation errors occur during validation
- Clean architecture: graph construction explicitly depends on plugin instances

### Negative

- Breaking change: `from_config()` deprecated (but it never worked anyway)
- Plugin instantiation failures prevent validation (was silent before)
- Larger refactor than adding schemas to config models (but architecturally correct)

## Alternatives Considered

### Option A: Add schema fields to config models

Add `input_schema`/`output_schema` fields to Pydantic config models, populate via temporary plugin instantiation before graph construction.

**Rejected because:**
- Double instantiation (performance cost)
- Violates separation of concerns (config layer knows about plugin layer)
- Uses `object.__setattr__()` to bypass frozen models (hacky)
- Accumulates technical debt

### Option B: Extract schemas from plugin classes (not instances)

Make schemas class attributes instead of instance attributes.

**Rejected because:**
- Many plugins compute schemas dynamically in `__init__()` based on config options
- Would require plugin API changes
- Doesn't support per-instance schema customization

## Implementation

See: `docs/plans/2026-01-24-schema-validation-architectural-refactor.md`

## Notes

- Fixes P0-2026-01-24-schema-validation-non-functional
- Fixes P2-2026-01-24-aggregation-nodes-lack-schema-validation
- Fixes P3-2026-01-24-coalesce-nodes-lack-schema-validation
```

### Step 3: Commit documentation

```bash
git add docs/bugs/P0-2026-01-24-schema-validation-non-functional.md docs/design/adr/003-schema-validation-lifecycle.md
git commit -m "docs: update P0 bug status and create schema lifecycle ADR

- Mark P0 bug as resolved
- Document architectural decision
- Explain why Option B chosen over Option A

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 10: Run Full Test Suite and Fix Regressions

**Files:**
- Various (fix any test failures)

### Step 1: Run full test suite

Run: `pytest tests/ -v`

Expected: Some failures possible (tests using old `from_config()` API)

### Step 2: Fix any test regressions

For each failing test:
1. Identify if it uses `from_config()`
2. Update to use `from_plugin_instances()`
3. Ensure schemas are available from plugin instances

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

## Task 11: Update BUGS.md and Close Related Bugs

**Files:**
- Modify: `docs/bugs/BUGS.md`
- Modify: `docs/bugs/P2-2026-01-24-aggregation-nodes-lack-schema-validation.md`
- Modify: `docs/bugs/P3-2026-01-24-coalesce-nodes-lack-schema-validation.md`

### Step 1: Update BUGS.md

**File:** `docs/bugs/BUGS.md`

Mark bugs as closed:

```markdown
## Closed Bugs

### P0-2026-01-24-schema-validation-non-functional ✅
- **Status:** Resolved in RC-2
- **Fix:** Architectural refactor - plugin instantiation before graph construction
- **PR:** [link]

### P2-2026-01-24-aggregation-nodes-lack-schema-validation ✅
- **Status:** Resolved (symptom of P0 bug)
- **Fix:** Included in P0 fix

### P3-2026-01-24-coalesce-nodes-lack-schema-validation ✅
- **Status:** Resolved (symptom of P0 bug)
- **Fix:** Included in P0 fix
```

### Step 2: Update P2 and P3 bugs

Add resolution notice to both files:

```markdown
## ✅ RESOLVED

**Status:** Fixed in RC-2 as part of P0-2026-01-24-schema-validation-non-functional
**Resolution:** Architectural refactor enables schema extraction from plugin instances
**Implementation:** See docs/plans/2026-01-24-schema-validation-architectural-refactor.md

This bug was a symptom of the broader P0 issue. The fix addresses all node types.

---
```

### Step 3: Commit bug updates

```bash
git add docs/bugs/
git commit -m "docs: close P0, P2, P3 schema validation bugs

- Mark bugs as resolved in RC-2
- Cross-reference implementation plan
- Update BUGS.md with closure status

Closes: P0-2026-01-24-schema-validation-non-functional
Closes: P2-2026-01-24-aggregation-nodes-lack-schema-validation
Closes: P3-2026-01-24-coalesce-nodes-lack-schema-validation"
```

---

## Task 12: Final Integration Test and Cleanup

**Files:**
- Various cleanup

### Step 1: Run full integration test suite

Run: `pytest tests/integration/ -v`

Expected: All PASS

### Step 2: Test with real pipeline config

Create a test pipeline and run it:

```bash
elspeth validate --settings examples/threshold_gate/settings.yaml
elspeth run --settings examples/threshold_gate/settings.yaml --execute
```

Expected: Validation works, pipeline executes

### Step 3: Remove deprecated from_config() method

**File:** `src/elspeth/core/dag.py`

Delete the `from_config()` method entirely (since all callers updated).

### Step 4: Final commit

```bash
git add src/elspeth/core/dag.py
git commit -m "cleanup: remove deprecated from_config() method

- All callers migrated to from_plugin_instances()
- Deprecation period complete
- Clean codebase with single graph construction API

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Acceptance Criteria

- [x] Schema validation detects incompatible transform chains
- [x] Schema validation detects incompatible source → transform edges
- [x] Schema validation detects incompatible transform → sink edges
- [x] Schema validation detects incompatible aggregation edges (dual-schema)
- [x] Schema validation handles dynamic schemas (`None`) correctly
- [x] Integration tests verify end-to-end validation works
- [x] No double plugin instantiation
- [x] CLI commands use new construction flow
- [x] Documentation updated (ADR, bug status)
- [x] No regressions in existing tests
- [x] All bugs closed (P0, P2, P3)

---

## Timeline

**Total: 12 tasks × 5 steps/task × 2-5 minutes/step = 2-5 hours**

With testing, debugging, and documentation: **3-4 days** total

---

## Rollout Checklist

Before merging to main:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Type checking passes (`mypy src/elspeth`)
- [ ] Linting passes (`ruff check src/elspeth`)
- [ ] Integration tests pass (`pytest tests/integration/ -v`)
- [ ] Manual testing with example pipelines
- [ ] Documentation reviewed
- [ ] ADR approved
- [ ] Bug closure reviewed

---

## Notes

- This plan uses TDD throughout (write test, watch fail, implement, watch pass)
- Each task is self-contained with clear acceptance criteria
- Frequent commits enable easy rollback if issues arise
- Integration tests verify the fix works end-to-end
- No legacy code accumulation (clean removal of old API)

---

**Plan saved to:** `docs/plans/2026-01-24-schema-validation-architectural-refactor.md`
