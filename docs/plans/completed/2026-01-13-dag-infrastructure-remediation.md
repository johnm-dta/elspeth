# DAG Infrastructure Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing `ExecutionGraph` class into the execution pipeline so that DAG validation occurs before any execution and the Orchestrator uses the graph for node/edge registration instead of ad-hoc dict construction.

**Architecture:** The codebase has 169 lines of dead code in `dag.py` - a complete `ExecutionGraph` class that nothing uses. The Orchestrator currently builds its own ad-hoc edge map via `dict[tuple[str,str], str]` and iterates transforms with naive `enumerate()`. This plan wires the existing graph into the pipeline: `ElspethSettings` → `ExecutionGraph.from_config()` → `graph.validate()` → `Orchestrator.run(config, graph)`.

**Tech Stack:** NetworkX (already used), Pydantic (config), pytest

---

## Task 1: Add get_node_info() Method to ExecutionGraph

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
class TestExecutionGraphAccessors:
    """Access node info and edges from graph."""

    def test_get_node_info(self) -> None:
        """Get NodeInfo for a node."""
        from elspeth.core.dag import ExecutionGraph, NodeInfo

        graph = ExecutionGraph()
        graph.add_node(
            "node_1",
            node_type="transform",
            plugin_name="my_plugin",
            config={"key": "value"},
        )

        info = graph.get_node_info("node_1")

        assert isinstance(info, NodeInfo)
        assert info.node_id == "node_1"
        assert info.node_type == "transform"
        assert info.plugin_name == "my_plugin"
        assert info.config == {"key": "value"}

    def test_get_node_info_missing(self) -> None:
        """Get NodeInfo for missing node raises."""
        import pytest
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        with pytest.raises(KeyError):
            graph.get_node_info("nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors::test_get_node_info -v`
Expected: FAIL with `'ExecutionGraph' object has no attribute 'get_node_info'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/dag.py` in the `ExecutionGraph` class:

```python
    def get_node_info(self, node_id: str) -> NodeInfo:
        """Get NodeInfo for a node.

        Args:
            node_id: The node ID

        Returns:
            NodeInfo for the node

        Raises:
            KeyError: If node doesn't exist
        """
        if not self._graph.has_node(node_id):
            raise KeyError(f"Node not found: {node_id}")
        return self._graph.nodes[node_id]["info"]
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): add get_node_info() method to ExecutionGraph

Returns NodeInfo for a node by ID, raises KeyError if not found.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add get_edges() Method to ExecutionGraph

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py` in `TestExecutionGraphAccessors`:

```python
    def test_get_edges(self) -> None:
        """Get all edges with data."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="source", plugin_name="src")
        graph.add_node("b", node_type="transform", plugin_name="tf")
        graph.add_node("c", node_type="sink", plugin_name="sink")
        graph.add_edge("a", "b", label="continue", mode="move")
        graph.add_edge("b", "c", label="output", mode="copy")

        edges = list(graph.get_edges())

        assert len(edges) == 2
        # Each edge is (from_id, to_id, data_dict)
        assert ("a", "b", {"label": "continue", "mode": "move"}) in edges
        assert ("b", "c", {"label": "output", "mode": "copy"}) in edges

    def test_get_edges_empty_graph(self) -> None:
        """Empty graph returns empty list."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        edges = list(graph.get_edges())

        assert edges == []
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors::test_get_edges -v`
Expected: FAIL with `'ExecutionGraph' object has no attribute 'get_edges'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/dag.py` in the `ExecutionGraph` class:

```python
    def get_edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all edges with their data.

        Returns:
            List of (from_node, to_node, edge_data) tuples
        """
        return [
            (u, v, dict(data))
            for u, v, data in self._graph.edges(data=True)
        ]
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors::test_get_edges -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): add get_edges() method to ExecutionGraph

Returns list of (from_node, to_node, edge_data) tuples.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add from_config() Factory Method - Minimal Case

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
class TestExecutionGraphFromConfig:
    """Build ExecutionGraph from ElspethSettings."""

    def test_from_config_minimal(self) -> None:
        """Build graph from minimal config (source -> sink only)."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have: source -> output_sink
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.get_source() is not None
        assert len(graph.get_sinks()) == 1

    def test_from_config_is_valid(self) -> None:
        """Graph from valid config passes validation."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should not raise
        graph.validate()
        assert graph.is_acyclic()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig::test_from_config_minimal -v`
Expected: FAIL with `'ExecutionGraph' has no attribute 'from_config'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/dag.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings


class ExecutionGraph:
    # ... existing methods ...

    @classmethod
    def from_config(cls, config: ElspethSettings) -> ExecutionGraph:
        """Build an ExecutionGraph from validated settings.

        Creates nodes for:
        - Source (from config.datasource)
        - Sinks (from config.sinks)

        Creates edges for:
        - Linear flow: source -> output_sink

        Args:
            config: Validated ElspethSettings

        Returns:
            ExecutionGraph ready for validation and execution
        """
        import uuid

        graph = cls()

        # Generate unique node IDs
        def node_id(prefix: str, name: str) -> str:
            return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

        # Add source node
        source_id = node_id("source", config.datasource.plugin)
        graph.add_node(
            source_id,
            node_type="source",
            plugin_name=config.datasource.plugin,
            config=config.datasource.options,
        )

        # Add sink nodes
        sink_ids: dict[str, str] = {}
        for sink_name, sink_config in config.sinks.items():
            sid = node_id("sink", sink_name)
            sink_ids[sink_name] = sid
            graph.add_node(
                sid,
                node_type="sink",
                plugin_name=sink_config.plugin,
                config=sink_config.options,
            )

        # Edge from source to output sink (minimal case - no transforms)
        graph.add_edge(
            source_id,
            sink_ids[config.output_sink],
            label="continue",
            mode="move",
        )

        return graph
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): add from_config() factory method (minimal case)

Builds ExecutionGraph from ElspethSettings with source and sink nodes.
Transforms will be added in the next task.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extend from_config() for Transforms

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py` in `TestExecutionGraphFromConfig`:

```python
    def test_from_config_with_transforms(self) -> None:
        """Build graph with transform chain."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
                RowPluginSettings(plugin="transform_b"),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have: source -> transform_a -> transform_b -> output_sink
        assert graph.node_count == 4
        assert graph.edge_count == 3

        # Topological order should be correct
        order = graph.topological_order()
        assert len(order) == 4
        # Source should be first (has "source" in node_id)
        assert "source" in order[0]
        # Sink should be last (has "sink" in node_id)
        assert "sink" in order[-1]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig::test_from_config_with_transforms -v`
Expected: FAIL (node_count will be 2, not 4)

**Step 3: Update implementation**

Update `from_config()` in `src/elspeth/core/dag.py`:

```python
    @classmethod
    def from_config(cls, config: ElspethSettings) -> ExecutionGraph:
        """Build an ExecutionGraph from validated settings.

        Creates nodes for:
        - Source (from config.datasource)
        - Transforms (from config.row_plugins, in order)
        - Sinks (from config.sinks)

        Creates edges for:
        - Linear flow: source -> transforms -> output_sink
        """
        import uuid

        graph = cls()

        def node_id(prefix: str, name: str) -> str:
            return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

        # Add source node
        source_id = node_id("source", config.datasource.plugin)
        graph.add_node(
            source_id,
            node_type="source",
            plugin_name=config.datasource.plugin,
            config=config.datasource.options,
        )

        # Add sink nodes
        sink_ids: dict[str, str] = {}
        for sink_name, sink_config in config.sinks.items():
            sid = node_id("sink", sink_name)
            sink_ids[sink_name] = sid
            graph.add_node(
                sid,
                node_type="sink",
                plugin_name=sink_config.plugin,
                config=sink_config.options,
            )

        # Build transform chain
        prev_node_id = source_id
        for i, plugin_config in enumerate(config.row_plugins):
            is_gate = plugin_config.type == "gate"
            ntype = "gate" if is_gate else "transform"
            tid = node_id(ntype, plugin_config.plugin)

            graph.add_node(
                tid,
                node_type=ntype,
                plugin_name=plugin_config.plugin,
                config=plugin_config.options,
            )

            # Edge from previous node
            graph.add_edge(prev_node_id, tid, label="continue", mode="move")
            prev_node_id = tid

        # Edge from last transform (or source) to output sink
        graph.add_edge(
            prev_node_id,
            sink_ids[config.output_sink],
            label="continue",
            mode="move",
        )

        return graph
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): extend from_config() to handle transforms

Builds linear transform chain: source -> transform_a -> transform_b -> sink.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extend from_config() for Gate Routes

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py` in `TestExecutionGraphFromConfig`:

```python
    def test_from_config_with_gate_routes(self) -> None:
        """Build graph with gate routing to multiple sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="safety_gate",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have:
        #   source -> safety_gate -> results (via "continue"/"clean")
        #                         -> flagged (via "suspicious")
        assert graph.node_count == 4  # source, gate, results, flagged
        # Edges: source->gate, gate->results (continue), gate->flagged (route)
        assert graph.edge_count == 3

    def test_from_config_validates_route_targets(self) -> None:
        """Gate routes must reference existing sinks."""
        import pytest
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(
                    plugin="gate",
                    type="gate",
                    routes={"bad": "nonexistent_sink"},
                ),
            ],
            output_sink="output",
        )

        with pytest.raises(GraphValidationError) as exc_info:
            ExecutionGraph.from_config(config)

        assert "nonexistent_sink" in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig::test_from_config_with_gate_routes -v`
Expected: FAIL (edge_count will be 2, not 3)

**Step 3: Update implementation**

Update `from_config()` in `src/elspeth/core/dag.py` to handle gate routes:

```python
    @classmethod
    def from_config(cls, config: ElspethSettings) -> ExecutionGraph:
        """Build an ExecutionGraph from validated settings.

        Creates nodes for:
        - Source (from config.datasource)
        - Transforms (from config.row_plugins, in order)
        - Sinks (from config.sinks)

        Creates edges for:
        - Linear flow: source -> transforms -> output_sink
        - Gate routes: gate -> routed_sink

        Raises:
            GraphValidationError: If gate routes reference unknown sinks
        """
        import uuid

        graph = cls()

        def node_id(prefix: str, name: str) -> str:
            return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

        # Add source node
        source_id = node_id("source", config.datasource.plugin)
        graph.add_node(
            source_id,
            node_type="source",
            plugin_name=config.datasource.plugin,
            config=config.datasource.options,
        )

        # Add sink nodes
        sink_ids: dict[str, str] = {}
        for sink_name, sink_config in config.sinks.items():
            sid = node_id("sink", sink_name)
            sink_ids[sink_name] = sid
            graph.add_node(
                sid,
                node_type="sink",
                plugin_name=sink_config.plugin,
                config=sink_config.options,
            )

        # Build transform chain
        prev_node_id = source_id
        for i, plugin_config in enumerate(config.row_plugins):
            is_gate = plugin_config.type == "gate"
            ntype = "gate" if is_gate else "transform"
            tid = node_id(ntype, plugin_config.plugin)

            graph.add_node(
                tid,
                node_type=ntype,
                plugin_name=plugin_config.plugin,
                config=plugin_config.options,
            )

            # Edge from previous node
            graph.add_edge(prev_node_id, tid, label="continue", mode="move")

            # Gate routes to sinks
            # Edge labels ARE route labels (not sink names)
            # Example: route "suspicious" -> sink "flagged"
            # Creates edge: gate_node -> flagged_node with label="suspicious"
            if is_gate and plugin_config.routes:
                for route_label, target in plugin_config.routes.items():
                    if target == "continue":
                        continue  # Not a sink route
                    if target not in sink_ids:
                        raise GraphValidationError(
                            f"Gate '{plugin_config.plugin}' routes '{route_label}' "
                            f"to unknown sink '{target}'. "
                            f"Available sinks: {list(sink_ids.keys())}"
                        )
                    # Edge label = route_label (e.g., "suspicious")
                    graph.add_edge(tid, sink_ids[target], label=route_label, mode="move")

            prev_node_id = tid

        # Edge from last transform (or source) to output sink
        graph.add_edge(
            prev_node_id,
            sink_ids[config.output_sink],
            label="continue",
            mode="move",
        )

        return graph
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): extend from_config() for gate routes

Gates can route to named sinks. Edge labels are route labels,
not sink names. Validates that route targets exist.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add Explicit ID Mapping Methods

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py` in `TestExecutionGraphAccessors`:

```python
    def test_get_sink_id_map(self) -> None:
        """Get explicit sink_name -> node_id mapping."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)
        sink_map = graph.get_sink_id_map()

        # Explicit mapping - no substring matching
        assert "results" in sink_map
        assert "flagged" in sink_map
        assert sink_map["results"] != sink_map["flagged"]

    def test_get_transform_id_map(self) -> None:
        """Get explicit sequence -> node_id mapping for transforms."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
                RowPluginSettings(plugin="transform_b"),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)
        transform_map = graph.get_transform_id_map()

        # Explicit mapping by sequence position
        assert 0 in transform_map  # transform_a
        assert 1 in transform_map  # transform_b
        assert transform_map[0] != transform_map[1]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors::test_get_sink_id_map -v`
Expected: FAIL with `'ExecutionGraph' object has no attribute 'get_sink_id_map'`

**Step 3: Update implementation**

Update `src/elspeth/core/dag.py`:

1. Add instance attributes in `__init__`:

```python
    def __init__(self) -> None:
        self._graph: DiGraph[str] = nx.DiGraph()
        self._sink_id_map: dict[str, str] = {}
        self._transform_id_map: dict[int, str] = {}
        self._output_sink: str = ""
```

2. Update `from_config()` to populate these mappings:

```python
        # Store explicit mapping for get_sink_id_map() - NO substring matching
        graph._sink_id_map = dict(sink_ids)

        # Build transform chain
        transform_ids: dict[int, str] = {}
        prev_node_id = source_id
        for i, plugin_config in enumerate(config.row_plugins):
            # ... existing code ...
            transform_ids[i] = tid  # Track sequence -> node_id
            # ... rest of loop ...

        # Store explicit mapping for get_transform_id_map()
        graph._transform_id_map = transform_ids

        # Store output_sink for reference
        graph._output_sink = config.output_sink
```

3. Add accessor methods:

```python
    def get_sink_id_map(self) -> dict[str, str]:
        """Get explicit sink_name -> node_id mapping.

        Returns:
            Dict mapping each sink's logical name to its graph node ID.
            No substring matching required - use this for direct lookup.
        """
        return dict(self._sink_id_map)

    def get_transform_id_map(self) -> dict[int, str]:
        """Get explicit sequence -> node_id mapping for transforms.

        Returns:
            Dict mapping transform sequence position (0-indexed) to node ID.
        """
        return dict(self._transform_id_map)

    def get_output_sink(self) -> str:
        """Get the output sink name."""
        return self._output_sink
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): add explicit ID mapping methods

Adds get_sink_id_map() and get_transform_id_map() for direct lookup
without substring matching. Eliminates brittle string matching.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add Graph Validation to CLI validate Command

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: `tests/cli/test_validate_command.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_validate_command.py`:

```python
class TestValidateCommandGraphValidation:
    """Validate command validates graph structure."""

    def test_validate_detects_invalid_route(self, tmp_path: Path) -> None:
        """Validate command catches gate routing to nonexistent sink."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: my_gate
    type: gate
    routes:
      bad_route: nonexistent_sink

output_sink: output
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code != 0
        output = result.stdout + (result.stderr or "")
        assert "nonexistent_sink" in output.lower()

    def test_validate_shows_graph_info(self, tmp_path: Path) -> None:
        """Validate command shows graph structure on success."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv
  flagged:
    plugin: csv

row_plugins:
  - plugin: classifier
    type: gate
    routes:
      suspicious: flagged
      clean: continue

output_sink: results
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code == 0
        # Should show graph info
        assert "node" in result.stdout.lower() or "valid" in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_validate_command.py::TestValidateCommandGraphValidation -v`
Expected: FAIL (validate command doesn't build/validate graph)

**Step 3: Update implementation**

Update `validate` command in `src/elspeth/cli.py`:

```python
from elspeth.core.dag import ExecutionGraph, GraphValidationError


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
    settings_path = Path(settings)

    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1)

    # Build and validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration valid: {settings_path.name}")
    typer.echo(f"  Source: {config.datasource.plugin}")
    typer.echo(f"  Transforms: {len(config.row_plugins)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Output: {config.output_sink}")
    typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_validate_command.py::TestValidateCommandGraphValidation -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/cli.py tests/cli/test_validate_command.py
git commit -m "$(cat <<'EOF'
feat(cli): add graph validation to validate command

Builds ExecutionGraph from config and validates it before reporting
success. Shows graph node/edge counts in output.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add Graph Validation to CLI run Command

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: `tests/cli/test_run_command.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_run_command.py`:

```python
class TestRunCommandGraphValidation:
    """Run command validates graph before execution."""

    def test_run_validates_graph_before_execution(self, tmp_path: Path) -> None:
        """Run command validates graph before any execution."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: bad_gate
    type: gate
    routes:
      error: missing_sink

output_sink: output
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])

        # Should fail at validation, not during execution
        assert result.exit_code != 0
        output = result.stdout + (result.stderr or "")
        assert "missing_sink" in output.lower() or "graph" in output.lower()

    def test_dry_run_shows_graph_info(self, tmp_path: Path) -> None:
        """Dry run shows graph structure."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv
  flagged:
    plugin: csv

row_plugins:
  - plugin: classifier
    type: gate
    routes:
      suspicious: flagged
      clean: continue

output_sink: results
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run", "-v"])

        assert result.exit_code == 0
        # Verbose should show graph info
        assert "node" in result.stdout.lower() or "edge" in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_run_command.py::TestRunCommandGraphValidation -v`
Expected: FAIL (run command doesn't build/validate graph)

**Step 3: Update implementation**

Update `run` command in `src/elspeth/cli.py`:

```python
@app.command()
def run(
    settings: str = typer.Option(..., "--settings", "-s", help="Path to settings YAML file."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Validate without executing."),
    execute: bool = typer.Option(False, "--execute", "-x", help="Actually execute the pipeline."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output."),
) -> None:
    """Execute a pipeline run."""
    settings_path = Path(settings)

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1)

    # Build and validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1)

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
        return

    # ... rest of command unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_run_command.py::TestRunCommandGraphValidation -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/cli.py tests/cli/test_run_command.py
git commit -m "$(cat <<'EOF'
feat(cli): add graph validation to run command

Validates ExecutionGraph before any execution attempt. Shows graph
info in verbose/dry-run modes.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update Orchestrator.run() to Accept Graph Parameter

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py`:

```python
class TestOrchestratorAcceptsGraph:
    """Orchestrator accepts ExecutionGraph parameter."""

    def test_orchestrator_run_accepts_graph(self) -> None:
        """Orchestrator.run() accepts graph parameter."""
        from unittest.mock import MagicMock
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.from_url("sqlite:///:memory:")

        # Build a simple graph
        graph = ExecutionGraph()
        graph.add_node("source_1", node_type="source", plugin_name="csv")
        graph.add_node("sink_1", node_type="sink", plugin_name="csv")
        graph.add_edge("source_1", "sink_1", label="continue", mode="move")

        orchestrator = Orchestrator(db)

        # Should accept graph parameter (signature check)
        import inspect
        sig = inspect.signature(orchestrator.run)
        assert "graph" in sig.parameters
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorAcceptsGraph -v`
Expected: FAIL with `'graph' not in sig.parameters`

**Step 3: Update implementation**

Update `Orchestrator.run()` in `src/elspeth/engine/orchestrator.py`:

```python
from elspeth.core.dag import ExecutionGraph


class Orchestrator:
    # ... existing code ...

    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
    ) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration with plugins
            graph: Pre-validated execution graph (required)

        Raises:
            ValueError: If graph is not provided
        """
        if graph is None:
            raise ValueError(
                "ExecutionGraph is required. "
                "Build with ExecutionGraph.from_config(settings)"
            )

        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config, graph)

            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"
            return result

        except Exception:
            recorder.complete_run(run.run_id, status="failed")
            raise
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorAcceptsGraph -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): accept ExecutionGraph parameter in run()

Graph is now required - no legacy fallback per CLAUDE.md policy.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update _execute_run() to Use Graph for Node Registration

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py`:

```python
    def test_orchestrator_uses_graph_node_ids(self) -> None:
        """Orchestrator uses node IDs from graph."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from unittest.mock import MagicMock

        db = LandscapeDB.from_url("sqlite:///:memory:")

        # Build config and graph
        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )
        graph = ExecutionGraph.from_config(config)

        # Create mock source and sink
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])  # Empty source

        mock_sink = MagicMock()
        mock_sink.name = "csv"

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline_config, graph=graph)

        # Source should have node_id set from graph
        assert hasattr(mock_source, "node_id")
        assert mock_source.node_id == graph.get_source()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorAcceptsGraph::test_orchestrator_uses_graph_node_ids -v`
Expected: FAIL (Orchestrator doesn't use graph node IDs)

**Step 3: Update implementation**

Update `_execute_run()` in `src/elspeth/engine/orchestrator.py`:

```python
    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> RunResult:
        """Execute the run using the execution graph."""

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Register nodes with Landscape using graph's node IDs
        for node_id in execution_order:
            node_info = graph.get_node_info(node_id)
            recorder.register_node(
                run_id=run_id,
                node_id=node_id,  # Use graph's ID
                plugin_name=node_info.plugin_name,
                node_type=NodeType(node_info.node_type.upper()),
                plugin_version="1.0.0",
                config=node_info.config,
            )

        # Register edges from graph
        edge_map: dict[tuple[str, str], str] = {}
        for from_id, to_id, edge_data in graph.get_edges():
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=from_id,
                to_node_id=to_id,
                label=edge_data["label"],
                mode=edge_data["mode"],
            )
            edge_map[(from_id, edge_data["label"])] = edge.edge_id

        # Get explicit node ID mappings from graph
        source_id = graph.get_source()
        sink_id_map = graph.get_sink_id_map()
        transform_id_map = graph.get_transform_id_map()

        # Set node_id on source plugin
        config.source.node_id = source_id

        # Set node_id on transforms using graph's transform_id_map
        for seq, transform in enumerate(config.transforms):
            if seq in transform_id_map:
                transform.node_id = transform_id_map[seq]

        # Set node_id on sinks using explicit mapping
        for sink_name, sink in config.sinks.items():
            if sink_name in sink_id_map:
                sink.node_id = sink_id_map[sink_name]
            else:
                raise ValueError(f"Sink '{sink_name}' not found in graph")

        # ... rest of execution logic ...
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorAcceptsGraph -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): use graph node IDs for registration

Registers nodes and edges from ExecutionGraph instead of building
ad-hoc dict. Uses explicit ID mappings - no substring matching.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Wire CLI to Pass Graph to Orchestrator

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: `tests/integration/test_cli_integration.py`

**Step 1: Write the failing test**

Add to `tests/integration/test_cli_integration.py`:

```python
class TestFullPipelineWithGraph:
    """End-to-end pipeline with graph validation."""

    def test_csv_pipeline_uses_graph(self, tmp_path: Path) -> None:
        """CSV pipeline is validated and executed via graph."""
        from typer.testing import CliRunner
        from elspeth.cli import app

        runner = CliRunner()

        input_file = tmp_path / "input.csv"
        input_file.write_text("id,value\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: {input_file}

sinks:
  results:
    plugin: csv
    options:
      path: {output_file}

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///{audit_db}
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0
        assert output_file.exists()
        # Verbose output should mention graph or completion
        assert "node" in result.stdout.lower() or "completed" in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_cli_integration.py::TestFullPipelineWithGraph -v`
Expected: FAIL (_execute_pipeline doesn't pass graph)

**Step 3: Update implementation**

Update `_execute_pipeline()` in `src/elspeth/cli.py` to accept and pass graph:

```python
def _execute_pipeline(
    config: ElspethSettings,
    graph: ExecutionGraph,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a pipeline from validated configuration.

    Args:
        config: Validated ElspethSettings instance
        graph: Validated ExecutionGraph
        verbose: Show detailed output

    Returns:
        Dict with run_id, status, rows_processed.
    """
    # ... setup code ...

    if verbose:
        typer.echo("Starting pipeline execution...")
        typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")

    # Execute via Orchestrator WITH GRAPH
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline_config, graph=graph)

    return {
        "run_id": result.run_id,
        "status": result.status,
        "rows_processed": result.rows_processed,
    }
```

And update the `run` command to pass the graph:

```python
    # Execute WITH GRAPH
    try:
        result = _execute_pipeline(config, graph, verbose=verbose)
        typer.echo(f"\nRun completed: {result['status']}")
        typer.echo(f"  Rows processed: {result['rows_processed']}")
        typer.echo(f"  Run ID: {result['run_id']}")
    except Exception as e:
        typer.echo(f"Error during pipeline execution: {e}", err=True)
        raise typer.Exit(1)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_cli_integration.py::TestFullPipelineWithGraph -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add src/elspeth/cli.py tests/integration/test_cli_integration.py
git commit -m "$(cat <<'EOF'
feat(cli): pass ExecutionGraph to Orchestrator

Completes the integration: config -> graph -> validate -> execute.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Remove Ad-hoc Edge Map Construction from Orchestrator

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`

**Step 1: Identify dead code**

The old `_execute_run()` method built its own edge map:

```python
edge_map: dict[tuple[str, str], str] = {}
# ... manual edge registration ...
```

This is now replaced by:
```python
for from_id, to_id, edge_data in graph.get_edges():
    edge = recorder.register_edge(...)
```

**Step 2: Verify no old code paths remain**

Run: `grep -n "enumerate(config.transforms)" src/elspeth/engine/orchestrator.py`
Expected: No matches (old iteration pattern should be gone)

**Step 3: Run all orchestrator tests**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -v`
Expected: All tests pass

**Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "$(cat <<'EOF'
refactor(orchestrator): remove ad-hoc edge map construction

All node/edge registration now uses ExecutionGraph. The old
enumerate() iteration and manual dict building are removed.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final Verification

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

**Step 2: Verify graph is used end-to-end**

```bash
cat > /tmp/test.yaml << 'EOF'
datasource:
  plugin: csv
  options:
    path: /tmp/input.csv

sinks:
  results:
    plugin: csv
    options:
      path: /tmp/output.csv
  flagged:
    plugin: csv
    options:
      path: /tmp/flagged.csv

row_plugins:
  - plugin: threshold_gate
    type: gate
    options:
      field: score
      threshold: 0.5
    routes:
      below: flagged
      above: continue

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:////tmp/audit.db
EOF

# Create test input
echo "id,score" > /tmp/input.csv
echo "1,0.8" >> /tmp/input.csv
echo "2,0.3" >> /tmp/input.csv

# Validate (should show graph info)
.venv/bin/python -m elspeth validate -s /tmp/test.yaml

# Should show: "Graph: 4 nodes, 3 edges"
```

**Step 3: Verify invalid graph is caught**

```bash
cat > /tmp/bad.yaml << 'EOF'
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: gate
    type: gate
    routes:
      error: nonexistent_sink

output_sink: output
EOF

.venv/bin/python -m elspeth validate -s /tmp/bad.yaml
# Should fail with: "routes 'error' to unknown sink 'nonexistent_sink'"
```

**Step 4: Run type checking**

Run: `.venv/bin/python -m mypy src/elspeth/core/dag.py src/elspeth/engine/orchestrator.py src/elspeth/cli.py`
Expected: Success

**Step 5: No commit needed - verification only**

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Add get_node_info() method | `dag.py`, `test_dag.py` |
| 2 | Add get_edges() method | `dag.py`, `test_dag.py` |
| 3 | Add from_config() - minimal | `dag.py`, `test_dag.py` |
| 4 | Extend from_config() for transforms | `dag.py`, `test_dag.py` |
| 5 | Extend from_config() for gate routes | `dag.py`, `test_dag.py` |
| 6 | Add explicit ID mapping methods | `dag.py`, `test_dag.py` |
| 7 | Add graph validation to validate command | `cli.py`, `test_validate_command.py` |
| 8 | Add graph validation to run command | `cli.py`, `test_run_command.py` |
| 9 | Update Orchestrator.run() signature | `orchestrator.py`, `test_orchestrator.py` |
| 10 | Update _execute_run() to use graph | `orchestrator.py`, `test_orchestrator.py` |
| 11 | Wire CLI to pass graph to Orchestrator | `cli.py`, `test_cli_integration.py` |
| 12 | Remove ad-hoc edge map construction | `orchestrator.py` |
| 13 | Final verification | (verification only) |

**Estimated total:** ~400-500 lines changed across 4 files
