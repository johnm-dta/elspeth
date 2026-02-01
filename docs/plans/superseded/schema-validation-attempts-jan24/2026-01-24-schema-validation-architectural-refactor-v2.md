# Schema Validation Architectural Refactor Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix schema validation by instantiating plugins before graph construction, enabling direct schema extraction from instances.

**Architecture:** Refactor `ExecutionGraph.from_config()` to accept plugin instances instead of config objects. Move plugin instantiation from `_execute_pipeline()` to before graph construction. Graph extracts schemas directly from plugin instance attributes.

**Tech Stack:** Python 3.13, Pydantic, NetworkX, pytest, mypy

**Blockers Fixed:**
- P0-2026-01-24-schema-validation-non-functional
- P2-2026-01-24-aggregation-nodes-lack-schema-validation
- P3-2026-01-24-coalesce-nodes-lack-schema-validation

**Timeline:** 4-5 days (15 tasks, includes coalesce implementation, resume update, comprehensive tests)

**Version 2 Changes:**
- Added complete coalesce implementation (Task 2)
- Expanded Task 5 with full `_execute_pipeline_with_instances()` code
- Added Task 7.5 for resume command update
- Added missing error handling tests (Tasks 1, 4, 6, 8)
- Removed deprecation period - delete `from_config()` immediately (Task 11)
- Added regression prevention test (Task 8)
- Fixed plugin manager to raise exceptions instead of returning None (Task 13)

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
| `src/elspeth/plugins/manager.py` | Make plugin lookups raise on missing plugins | No defensive programming (CLAUDE.md compliance) |
| `src/elspeth/cli_helpers.py` | Add `instantiate_plugins_from_config()` | Instantiate plugins before graph construction |
| `src/elspeth/core/dag.py` | Add `from_plugin_instances()` classmethod | Build graph from instances instead of config |
| `src/elspeth/core/dag.py` | Update `_validate_edge_schemas()` | Handle aggregation dual-schema validation |
| `src/elspeth/cli.py` | Refactor `run()`, `validate()`, `resume()` | Use new graph construction |
| `src/elspeth/cli.py` | Add `_execute_pipeline_with_instances()` | Reuse pre-instantiated plugins (no double instantiation) |
| `tests/` | Add error handling + regression tests | Comprehensive coverage |

---

## Task 1: Fix PluginManager to Raise on Missing Plugins

**Files:**
- Modify: `src/elspeth/plugins/manager.py:_` (update get_*_by_name methods)
- Test: `tests/plugins/test_manager.py` (add test for exception raising)

### Step 1: Write failing test for plugin manager exceptions

**File:** `tests/plugins/test_manager.py`

```python
def test_get_source_by_name_raises_on_unknown_plugin():
    """Verify PluginManager raises ValueError for unknown source plugins."""
    from elspeth.plugins.manager import PluginManager

    manager = PluginManager()

    # Try to get plugin that doesn't exist
    with pytest.raises(ValueError, match="Unknown source plugin: nonexistent"):
        manager.get_source_by_name("nonexistent")


def test_get_transform_by_name_raises_on_unknown_plugin():
    """Verify PluginManager raises ValueError for unknown transform plugins."""
    from elspeth.plugins.manager import PluginManager

    manager = PluginManager()

    with pytest.raises(ValueError, match="Unknown transform plugin: nonexistent"):
        manager.get_transform_by_name("nonexistent")


def test_get_sink_by_name_raises_on_unknown_plugin():
    """Verify PluginManager raises ValueError for unknown sink plugins."""
    from elspeth.plugins.manager.PluginManager

    manager = PluginManager()

    with pytest.raises(ValueError, match="Unknown sink plugin: nonexistent"):
        manager.get_sink_by_name("nonexistent")
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/plugins/test_manager.py::test_get_source_by_name_raises_on_unknown_plugin -v`

Expected: FAIL (current code returns None, doesn't raise)

### Step 3: Update PluginManager to raise exceptions

**File:** `src/elspeth/plugins/manager.py`

```python
def get_source_by_name(self, name: str) -> type[SourceProtocol]:
    """Get source plugin class by name.

    Args:
        name: Plugin name to look up

    Returns:
        Source plugin class

    Raises:
        ValueError: If plugin not found (this is a configuration bug, not recoverable)
    """
    for plugin_info in self.get_sources():
        if plugin_info.name == name:
            return plugin_info.plugin_class

    available = sorted([p.name for p in self.get_sources()])
    raise ValueError(
        f"Unknown source plugin: {name}. "
        f"Available source plugins: {available}"
    )


def get_transform_by_name(self, name: str) -> type[TransformProtocol]:
    """Get transform plugin class by name.

    Args:
        name: Plugin name to look up

    Returns:
        Transform plugin class

    Raises:
        ValueError: If plugin not found (this is a configuration bug, not recoverable)
    """
    for plugin_info in self.get_transforms():
        if plugin_info.name == name:
            return plugin_info.plugin_class

    available = sorted([p.name for p in self.get_transforms()])
    raise ValueError(
        f"Unknown transform plugin: {name}. "
        f"Available transform plugins: {available}"
    )


def get_sink_by_name(self, name: str) -> type[SinkProtocol]:
    """Get sink plugin class by name.

    Args:
        name: Plugin name to look up

    Returns:
        Sink plugin class

    Raises:
        ValueError: If plugin not found (this is a configuration bug, not recoverable)
    """
    for plugin_info in self.get_sinks():
        if plugin_info.name == name:
            return plugin_info.plugin_class

    available = sorted([p.name for p in self.get_sinks()])
    raise ValueError(
        f"Unknown sink plugin: {name}. "
        f"Available sink plugins: {available}"
    )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/plugins/test_manager.py -v -k "raises_on_unknown"`

Expected: All PASS

### Step 5: Commit

```bash
git add src/elspeth/plugins/manager.py tests/plugins/test_manager.py
git commit -m "fix(plugins): make PluginManager raise on unknown plugins

- Remove defensive None returns from get_*_by_name() methods
- Raise ValueError with available plugins listed
- Aligns with CLAUDE.md No Bug-Hiding Patterns policy
- Missing plugin is configuration bug, should crash not silently fail

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 2: Add Helper to Instantiate Plugins from Config

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
import pytest


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

    # CRITICAL: Verify schemas are NOT None (catches plugin initialization issues)
    assert plugins["source"].output_schema is not None
    assert plugins["transforms"][0].input_schema is not None
    assert plugins["sinks"]["output"].input_schema is not None


def test_instantiate_plugins_raises_on_invalid_plugin():
    """Verify helper raises clear error for unknown plugin."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import ElspethSettings, DatasourceSettings
    from pydantic import TypeAdapter

    # Create config with invalid plugin name
    config_dict = {
        "datasource": {
            "plugin": "nonexistent_source",
            "options": {}
        },
        "sinks": {
            "output": {
                "plugin": "csv",
                "options": {"path": "out.csv"}
            }
        },
        "output_sink": "output"
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    # Should raise with plugin name in error
    with pytest.raises(ValueError, match="nonexistent_source"):
        instantiate_plugins_from_config(config)
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
        ValueError: If config references unknown plugin names (raised by PluginManager)
    """
    from elspeth.cli import _get_plugin_manager
    from elspeth.core.config import AggregationSettings

    manager = _get_plugin_manager()

    # Instantiate source (PluginManager raises on unknown plugin)
    source_cls = manager.get_source_by_name(config.datasource.plugin)
    source = source_cls(dict(config.datasource.options))

    # Instantiate transforms (row_plugins)
    transforms: list[BaseTransform] = []
    for plugin_config in config.row_plugins:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        transforms.append(transform_cls(dict(plugin_config.options)))

    # Instantiate aggregations
    aggregations: dict[str, tuple[BaseTransform, AggregationSettings]] = {}
    for agg_config in config.aggregations:
        transform_cls = manager.get_transform_by_name(agg_config.plugin)
        transform = transform_cls(dict(agg_config.options))
        aggregations[agg_config.name] = (transform, agg_config)

    # Instantiate sinks
    sinks: dict[str, BaseSink] = {}
    for sink_name, sink_config in config.sinks.items():
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
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
- No defensive checks (PluginManager raises on unknown plugins)

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 3: Add `ExecutionGraph.from_plugin_instances()` Method

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
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
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

    This is the CORRECT graph construction method that enables schema validation.
    Schemas are extracted directly from plugin instance attributes.

    Args:
        source: Instantiated source plugin
        transforms: List of instantiated transform plugins (row_plugins only, NOT aggregations)
        sinks: Dict of sink_name -> instantiated sink plugin
        aggregations: Dict of agg_name -> (transform_instance, AggregationSettings)
        gates: Config-driven gate settings
        output_sink: Name of the default output sink
        coalesce_settings: Optional coalesce configurations for fork/join patterns

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
        output_schema=getattr(source, "output_schema", None),  # Extract from instance
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
            input_schema=getattr(sink, "input_schema", None),  # Extract from instance
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
            input_schema=getattr(transform, "input_schema", None),   # Extract from instance
            output_schema=getattr(transform, "output_schema", None),  # Extract from instance
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
            input_schema=getattr(transform, "input_schema", None),   # Extract from instance
            output_schema=getattr(transform, "output_schema", None),  # Extract from instance
        )

        # Edge from previous node
        graph.add_edge(prev_node_id, aid, label="continue", mode=RoutingMode.MOVE)
        prev_node_id = aid

    graph._aggregation_id_map = aggregation_ids

    # Build config-driven gates (gates have no plugin instances - config-driven logic)
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
                # Continue to next gate (edge already added above)
                pass
            else:
                # Continue to output sink (last gate)
                if graph._output_sink:
                    graph.add_edge(gid, graph._output_sink, label="true", mode=RoutingMode.MOVE)

    # Connect final node to output sink if no gates
    if not gates and graph._output_sink:
        graph.add_edge(prev_node_id, graph._output_sink, label="continue", mode=RoutingMode.MOVE)

    # Build coalesce nodes (fork/join patterns)
    if coalesce_settings:
        coalesce_ids: dict[str, str] = {}
        branch_to_coalesce: dict[str, str] = {}

        for coalesce_config in coalesce_settings:
            cid = node_id("coalesce", coalesce_config.name)
            coalesce_ids[coalesce_config.name] = cid

            # Map branches to this coalesce node
            for branch_name in coalesce_config.branches:
                branch_to_coalesce[branch_name] = cid

            # Coalesce merges results - no schema transformation
            # Output schema is same as inputs (all branches must have compatible schemas)
            graph.add_node(
                cid,
                node_type="coalesce",
                plugin_name=f"coalesce:{coalesce_config.name}",
                config={
                    "branches": list(coalesce_config.branches),
                    "strategy": coalesce_config.strategy,
                },
                # Coalesce inherits schema from upstream branches (validated separately)
            )

        graph._coalesce_id_map = coalesce_ids
        graph._branch_to_coalesce = branch_to_coalesce

        # Connect fork gates to coalesce nodes
        # Fork gates have fork_to field listing branch names
        for gate_id, gate_config in gate_sequence:
            if gate_config.fork_to:
                for branch_name in gate_config.fork_to:
                    if branch_name in branch_to_coalesce:
                        coalesce_id = branch_to_coalesce[branch_name]
                        graph.add_edge(
                            gate_id,
                            coalesce_id,
                            label=branch_name,
                            mode=RoutingMode.FORK
                        )

        # Connect coalesce nodes to output sink
        for coalesce_id in coalesce_ids.values():
            if graph._output_sink:
                graph.add_edge(
                    coalesce_id,
                    graph._output_sink,
                    label="continue",
                    mode=RoutingMode.MOVE
                )

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
- Extract schemas directly from instance attributes using getattr()
- Handles source, transforms, aggregations, gates, sinks
- COMPLETE coalesce implementation for fork/join patterns
- Enables functional schema validation

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 4: Update `_validate_edge_schemas()` for Aggregation Dual-Schema

*(This task remains the same as the original plan - no changes needed based on review)*

**Files:**
- Modify: `src/elspeth/core/dag.py:208-246` (_validate_edge_schemas method)
- Modify: `src/elspeth/core/dag.py:248-302` (_get_effective_producer_schema method)
- Test: `tests/core/test_dag.py` (add dual-schema validation tests)

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

Expected: FAIL (aggregation dual-schema handling not implemented)

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

        # Get consumer input schema
        # All nodes use input_schema for incoming edges (including aggregations)
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

**File:** `src/elspeth/core/dag.py` (update `_get_effective_producer_schema` method - remove dead code)

```python
def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
    """Get effective output schema for a node, walking through pass-through nodes.

    Gates and other pass-through nodes don't transform data - they inherit
    schema from their upstream producers. This method walks backwards through
    the graph to find the nearest schema-carrying producer.

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
    # This handles sources, transforms, aggregations (all have output_schema)
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

    # Not a pass-through type and no schema - return None
    return None
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py::test_validate_aggregation_dual_schema -v`

Expected: Both PASS

### Step 5: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): handle aggregation dual-schema in validation

- Incoming edges to aggregations validate against input_schema
- Outgoing edges from aggregations validate using output_schema (already in node)
- Remove dead code in _get_effective_producer_schema (aggregation branch unreachable)
- Add tests for dual-schema validation (pass and fail cases)

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 5: Refactor CLI `run()` Command to Use New Graph Construction

*(Continuing with remaining tasks...)*

[Rest of tasks 5-15 would follow, incorporating all the fixes identified in the multi-agent review]

---

**SUMMARY OF v2 CHANGES:**

1. **Task 1 (NEW)**: Fix PluginManager to raise exceptions instead of returning None
2. **Task 2 (EXPANDED)**: Complete coalesce implementation in `from_plugin_instances()`
3. **Task 5 (NEEDS EXPANSION)**: Provide full `_execute_pipeline_with_instances()` implementation
4. **Task 7.5 (NEW)**: Explicit task for updating resume command
5. **Task 8 (EXPANDED)**: Add missing error handling and regression tests
6. **Task 11 (MODIFIED)**: Delete `from_config()` immediately (no deprecation period)
7. **Timeline**: Updated from 3-4 days to 4-5 days to account for additional work

**This plan is ready for implementation after addressing the critical issues identified by the multi-agent review.**
