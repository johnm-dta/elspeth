# Schema Validation Refactor - Foundation Tasks (1-4)

> **Previous:** `00-overview.md` | **Next:** `02-cli-refactor.md`

This file contains the foundation tasks that fix the core architecture.

---

## Task 1: Fix PluginManager to Raise on Missing Plugins

**Files:**
- Modify: `src/elspeth/plugins/manager.py` (update get_*_by_name methods)
- Test: `tests/plugins/test_manager.py` (add test for exception raising)

**Purpose:** Eliminate defensive programming - missing plugins are configuration bugs that should crash, not return None.

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
    from elspeth.plugins.manager import PluginManager

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
        ValueError: If plugin not found (configuration bug, should crash)
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
        ValueError: If plugin not found (configuration bug, should crash)
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
        ValueError: If plugin not found (configuration bug, should crash)
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
- Create: `src/elspeth/cli_helpers.py`
- Test: `tests/cli/test_cli_helpers.py`

**Purpose:** Create helper function to instantiate all plugins before graph construction.

### Step 1: Write failing test

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


def test_instantiate_plugins_raises_on_invalid_plugin():
    """Verify helper raises clear error for unknown plugin."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import ElspethSettings
    from pydantic import TypeAdapter

    config_dict = {
        "datasource": {"plugin": "nonexistent", "options": {}},
        "sinks": {"out": {"plugin": "csv", "options": {"path": "o.csv"}}},
        "output_sink": "out"
    }

    adapter = TypeAdapter(ElspethSettings)
    config = adapter.validate_python(config_dict)

    with pytest.raises(ValueError, match="nonexistent"):
        instantiate_plugins_from_config(config)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_cli_helpers.py::test_instantiate_plugins_from_config -v`

Expected: FAIL (module doesn't exist)

### Step 3: Implement helper

**File:** `src/elspeth/cli_helpers.py`

```python
"""CLI helper functions for plugin instantiation."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings


def instantiate_plugins_from_config(config: "ElspethSettings") -> dict[str, Any]:
    """Instantiate all plugins from configuration.

    Creates plugin instances BEFORE graph construction,
    enabling schema extraction from instance attributes.

    Args:
        config: Validated ElspethSettings instance

    Returns:
        Dict with keys:
            - source: SourceProtocol instance
            - transforms: list[TransformProtocol] (row_plugins only)
            - sinks: dict[str, SinkProtocol]
            - aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]]

    Raises:
        ValueError: If config references unknown plugins (raised by PluginManager)
    """
    from elspeth.cli import _get_plugin_manager
    from elspeth.core.config import AggregationSettings

    manager = _get_plugin_manager()

    # Instantiate source (raises on unknown plugin)
    source_cls = manager.get_source_by_name(config.datasource.plugin)
    source = source_cls(dict(config.datasource.options))

    # Instantiate transforms
    transforms = []
    for plugin_config in config.row_plugins:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        transforms.append(transform_cls(dict(plugin_config.options)))

    # Instantiate aggregations
    aggregations = {}
    for agg_config in config.aggregations:
        transform_cls = manager.get_transform_by_name(agg_config.plugin)
        transform = transform_cls(dict(agg_config.options))
        aggregations[agg_config.name] = (transform, agg_config)

    # Instantiate sinks
    sinks = {}
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
- No defensive checks (PluginManager raises on unknown)
- Enables schema extraction before graph construction

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 3: Add `ExecutionGraph.from_plugin_instances()` Method

**Files:**
- Modify: `src/elspeth/core/dag.py` (add new classmethod)
- Test: `tests/core/test_dag.py`

**Purpose:** Build graph from plugin instances to enable schema extraction. **Includes COMPLETE coalesce implementation** (critical fix from review).

### Step 1: Write failing test

**File:** `tests/core/test_dag.py`

```python
def test_from_plugin_instances_extracts_schemas():
    """Verify from_plugin_instances extracts schemas from instances."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import load_settings
    from elspeth.core.dag import ExecutionGraph
    import tempfile
    from pathlib import Path

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
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            output_sink=config.output_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )

        # Verify schemas extracted
        source_nodes = [n for n, d in graph._graph.nodes(data=True) if d["node_type"] == "source"]
        source_info = graph.get_node_info(source_nodes[0])
        assert source_info.output_schema is not None

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_dag.py::test_from_plugin_instances_extracts_schemas -v`

Expected: FAIL (method doesn't exist)

### Step 3: Implement from_plugin_instances

**File:** `src/elspeth/core/dag.py` (add after `from_config`, ~line 650)

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
    """Build ExecutionGraph from plugin instances.

    CORRECT method for graph construction - enables schema validation.
    Schemas extracted directly from instance attributes.

    Args:
        source: Instantiated source plugin
        transforms: Instantiated transforms (row_plugins only, NOT aggregations)
        sinks: Dict of sink_name -> instantiated sink
        aggregations: Dict of agg_name -> (transform_instance, AggregationSettings)
        gates: Config-driven gate settings
        output_sink: Default output sink name
        coalesce_settings: Coalesce configs for fork/join patterns

    Returns:
        ExecutionGraph with schemas populated

    Raises:
        GraphValidationError: If gate routes reference unknown sinks
    """
    import uuid

    graph = cls()

    def node_id(prefix: str, name: str) -> str:
        return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

    # Add source - extract schema from instance
    source_id = node_id("source", source.name)
    graph.add_node(
        source_id,
        node_type="source",
        plugin_name=source.name,
        config={},
        output_schema=getattr(source, "output_schema", None),
    )

    # Add sinks
    sink_ids: dict[str, str] = {}
    for sink_name, sink in sinks.items():
        sid = node_id("sink", sink_name)
        sink_ids[sink_name] = sid
        graph.add_node(
            sid,
            node_type="sink",
            plugin_name=sink.name,
            config={},
            input_schema=getattr(sink, "input_schema", None),
        )

    graph._sink_id_map = dict(sink_ids)
    graph._output_sink = sink_ids.get(output_sink, "")

    # Build transform chain
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
            input_schema=getattr(transform, "input_schema", None),
            output_schema=getattr(transform, "output_schema", None),
        )

        graph.add_edge(prev_node_id, tid, label="continue", mode=RoutingMode.MOVE)
        prev_node_id = tid

    graph._transform_id_map = transform_ids

    # Build aggregations - dual schemas
    aggregation_ids: dict[str, str] = {}
    for agg_name, (transform, agg_config) in aggregations.items():
        aid = node_id("aggregation", agg_name)
        aggregation_ids[agg_name] = aid

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
            input_schema=getattr(transform, "input_schema", None),
            output_schema=getattr(transform, "output_schema", None),
        )

        graph.add_edge(prev_node_id, aid, label="continue", mode=RoutingMode.MOVE)
        prev_node_id = aid

    graph._aggregation_id_map = aggregation_ids

    # Build gates (config-driven, no instances)
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
        )

        graph.add_edge(prev_node_id, gid, label="continue", mode=RoutingMode.MOVE)

        # Gate routes to sinks
        for route_label, target in gate_config.routes.items():
            if target == "continue":
                graph._route_resolution_map[(gid, route_label)] = "continue"
            else:
                if target not in sink_ids:
                    raise GraphValidationError(
                        f"Gate '{gate_config.name}' route '{route_label}' "
                        f"references unknown sink '{target}'"
                    )
                target_sink_id = sink_ids[target]
                graph.add_edge(gid, target_sink_id, label=route_label, mode=RoutingMode.ROUTE)
                graph._route_label_map[(gid, target)] = route_label
                graph._route_resolution_map[(gid, route_label)] = target

        gate_sequence.append((gid, gate_config))

    graph._config_gate_id_map = config_gate_ids

    # ===== COALESCE IMPLEMENTATION (BUILD NODES AND MAPPINGS FIRST) =====
    # Build coalesce nodes BEFORE connecting gates (needed for branch routing)
    if coalesce_settings:
        coalesce_ids: dict[str, str] = {}
        branch_to_coalesce: dict[str, str] = {}

        for coalesce_config in coalesce_settings:
            cid = node_id("coalesce", coalesce_config.name)
            coalesce_ids[coalesce_config.name] = cid

            # Map branches to this coalesce
            for branch_name in coalesce_config.branches:
                branch_to_coalesce[branch_name] = cid

            # Coalesce merges - no schema transformation
            graph.add_node(
                cid,
                node_type="coalesce",
                plugin_name=f"coalesce:{coalesce_config.name}",
                config={
                    "branches": list(coalesce_config.branches),
                    "strategy": coalesce_config.strategy,
                },
            )

        graph._coalesce_id_map = coalesce_ids
        graph._branch_to_coalesce = branch_to_coalesce
    else:
        branch_to_coalesce = {}

    # ===== CONNECT FORK GATES - EXPLICIT DESTINATIONS ONLY =====
    # CRITICAL: No fallback behavior. All fork branches must have explicit destinations.
    # This prevents silent configuration bugs (typos, missing destinations).
    for gate_id, gate_config in gate_sequence:
        if gate_config.fork_to:
            for branch_name in gate_config.fork_to:
                if branch_name in branch_to_coalesce:
                    # Explicit coalesce destination
                    coalesce_id = branch_to_coalesce[branch_name]
                    graph.add_edge(
                        gate_id,
                        coalesce_id,
                        label=branch_name,
                        mode=RoutingMode.FORK
                    )
                elif branch_name in sink_ids:
                    # Explicit sink destination (branch name matches sink name)
                    graph.add_edge(
                        gate_id,
                        sink_ids[branch_name],
                        label=branch_name,
                        mode=RoutingMode.COPY
                    )
                else:
                    # NO FALLBACK - this is a configuration error
                    raise GraphValidationError(
                        f"Gate '{gate_config.name}' has fork branch '{branch_name}' with no destination.\n"
                        f"Fork branches must either:\n"
                        f"  1. Be listed in a coalesce 'branches' list, or\n"
                        f"  2. Match a sink name exactly\n"
                        f"\n"
                        f"Available coalesce branches: {sorted(branch_to_coalesce.keys())}\n"
                        f"Available sinks: {sorted(sink_ids.keys())}"
                    )

    # ===== CONNECT GATE CONTINUE ROUTES =====
    # CRITICAL FIX: Handle ALL continue routes, not just "true"
    for i, (gid, gate_config) in enumerate(gate_sequence):
        # Check if ANY route resolves to "continue"
        has_continue_route = any(target == "continue" for target in gate_config.routes.values())

        if has_continue_route:
            # Determine next node in chain
            if i + 1 < len(gate_sequence):
                next_node_id = gate_sequence[i + 1][0]
            else:
                next_node_id = graph._output_sink

            # Add continue edge if not already present
            if not next_node_id:
                raise GraphValidationError(
                    f"Gate '{gate_config.name}' has 'continue' route but is the last gate "
                    "and no output_sink is configured. Continue routes must have a target."
                )

            if not graph._graph.has_edge(gid, next_node_id, key="continue"):
                graph.add_edge(gid, next_node_id, label="continue", mode=RoutingMode.MOVE)

    # ===== CONNECT FINAL NODE TO OUTPUT (NO GATES CASE) =====
    if not gates and graph._output_sink:
        graph.add_edge(prev_node_id, graph._output_sink, label="continue", mode=RoutingMode.MOVE)

    # ===== CONNECT COALESCE TO OUTPUT =====
    if coalesce_settings:
        for coalesce_id in coalesce_ids.values():
            if graph._output_sink:
                graph.add_edge(
                    coalesce_id,
                    graph._output_sink,
                    label="continue",
                    mode=RoutingMode.MOVE
                )

    # ===== VALIDATE COALESCE SCHEMA COMPATIBILITY =====
    # CRITICAL: Coalesce merges multiple fork branches - schemas must be compatible
    # Addresses P0 blocker from Round 3 QA review
    if coalesce_settings:
        for coalesce_id in coalesce_ids.values():
            # Get all incoming edges to this coalesce
            incoming_edges = list(graph._graph.in_edges(coalesce_id, data=True, keys=True))

            if len(incoming_edges) < 2:
                # Coalesce with < 2 inputs is degenerate but valid (pass-through)
                continue

            # Extract schemas from all producers
            incoming_schemas = []
            for pred_id, _, _, edge_data in incoming_edges:
                producer_schema = graph._get_effective_producer_schema(pred_id)
                incoming_schemas.append((pred_id, producer_schema))

            # Filter to only specific schemas (dynamic schemas are None)
            specific_schemas = [
                (node_id, schema)
                for node_id, schema in incoming_schemas
                if schema is not None
            ]

            # If we have 2+ specific schemas, they must be compatible
            if len(specific_schemas) >= 2:
                base_node_id, base_schema = specific_schemas[0]

                for node_id, schema in specific_schemas[1:]:
                    # Check schema compatibility
                    if not _schemas_compatible(base_schema, schema):
                        raise GraphValidationError(
                            f"Coalesce node '{coalesce_id}' receives incompatible schemas. "
                            f"Branch from '{base_node_id}' has schema {base_schema.__name__}, "
                            f"but branch from '{node_id}' has schema {schema.__name__}. "
                            f"All branches merging at a coalesce must have compatible schemas."
                        )

    return graph
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_dag.py::test_from_plugin_instances_extracts_schemas -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): add from_plugin_instances() with complete coalesce

- Build graph from plugin instances
- Extract schemas using getattr() (legitimate boundary case)
- COMPLETE coalesce implementation (critical fix from review)
- Handles fork/join patterns with branch mapping
- Enables functional schema validation

Fixes: P3-2026-01-24-coalesce-nodes-lack-schema-validation
Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 4: Update `_validate_edge_schemas()` for Aggregation Dual-Schema

**Files:**
- Modify: `src/elspeth/core/dag.py` (_validate_edge_schemas, _get_effective_producer_schema)
- Test: `tests/core/test_dag.py`

**Purpose:** Handle aggregation dual-schema validation (incoming uses input_schema, outgoing uses output_schema).

### Step 1: Write failing tests

**File:** `tests/core/test_dag.py`

```python
def test_validate_aggregation_dual_schema():
    """Verify aggregation edges validate against correct schemas."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.plugins.schema_factory import create_schema_from_config

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

    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type="aggregation",
        plugin_name="batch_stats",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        config={},
    )
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=OutputSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    errors = graph._validate_edge_schemas()
    assert len(errors) == 0  # Should pass


def test_validate_aggregation_detects_incompatibility():
    """Verify validation detects aggregation output mismatch."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.plugins.schema_factory import create_schema_from_config

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

    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type="aggregation",
        plugin_name="batch_stats",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        config={},
    )
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=SinkSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    errors = graph._validate_edge_schemas()
    assert len(errors) > 0
    assert "sum" in errors[0]
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/core/test_dag.py::test_validate_aggregation_dual_schema -v`

Expected: FAIL (dual-schema handling not implemented)

### Step 3: Implement validation updates

**File:** `src/elspeth/core/dag.py` (_validate_edge_schemas - simplified)

```python
def _validate_edge_schemas(self) -> list[str]:
    """Validate schema compatibility along all edges.

    For each edge (producer -> consumer):
    - Get producer's effective output schema (walks through gates)
    - Get consumer's input schema (all nodes, including aggregations)
    - Check producer provides all required fields

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for edge in self.get_edges():
        from_info = self.get_node_info(edge.from_node)
        to_info = self.get_node_info(edge.to_node)

        # Get effective producer schema
        producer_schema = self._get_effective_producer_schema(edge.from_node)

        # Get consumer input schema
        # All nodes use input_schema for incoming edges
        consumer_schema = to_info.input_schema

        # Skip if either schema is None (dynamic)
        if producer_schema is None or consumer_schema is None:
            continue

        # Validate compatibility
        missing = _get_missing_required_fields(
            producer=producer_schema,
            consumer=consumer_schema,
        )

        if missing:
            errors.append(
                f"{from_info.plugin_name} -> {to_info.plugin_name} "
                f"(route: {edge.label}): producer missing required fields {missing}"
            )

    return errors
```

**File:** `src/elspeth/core/dag.py` (_get_effective_producer_schema - remove dead code)

```python
def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
    """Get effective output schema, walking through pass-through nodes.

    Gates inherit from upstream. Aggregations/transforms use output_schema.

    Args:
        node_id: Node to get schema for

    Returns:
        Output schema type, or None if no schema

    Raises:
        GraphValidationError: If gate has no inputs or incompatible multi-inputs
    """
    node_info = self.get_node_info(node_id)

    # If node has output_schema, return it
    # Handles sources, transforms, aggregations
    if node_info.output_schema is not None:
        return node_info.output_schema

    # Gate - inherit from upstream
    if node_info.node_type == "gate":
        incoming = self.get_incoming_edges(node_id)

        if not incoming:
            raise GraphValidationError(
                f"Gate node '{node_id}' has no incoming edges - graph construction bug"
            )

        # Get schema from first input
        first_schema = self._get_effective_producer_schema(incoming[0].from_node)

        # Verify all inputs have same schema
        if len(incoming) > 1:
            for edge in incoming[1:]:
                other_schema = self._get_effective_producer_schema(edge.from_node)
                if first_schema != other_schema:
                    raise GraphValidationError(
                        f"Gate '{node_id}' receives incompatible schemas - "
                        f"graph construction bug"
                    )

        return first_schema

    # No schema
    return None
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py -k "aggregation_dual" -v`

Expected: Both PASS

### Step 5: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): handle aggregation dual-schema validation

- Incoming edges validate against input_schema
- Outgoing edges validate using output_schema from node
- Removed dead code in _get_effective_producer_schema
- Simplified validation logic

Fixes: P2-2026-01-24-aggregation-nodes-lack-schema-validation
Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

**Foundation Complete! Next:** `02-cli-refactor.md` for Tasks 5-7
