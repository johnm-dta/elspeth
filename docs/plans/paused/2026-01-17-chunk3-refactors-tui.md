# Chunk 3: Refactors & TUI Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish consistent repository pattern usage, create type-safe routing abstractions, add protocol contracts, and wire up the TUI explain functionality end-to-end.

**Architecture:** Refactor recorder to delegate to repositories (DRY), create RoutingMap dataclass for type-safe edge/destination lookup, add ExecutionGraphProtocol and SinkLike contracts, and wire ExplainApp through CLI to database using existing NodeState discriminated unions.

**Tech Stack:** Python dataclasses, Protocol (structural typing), Textual TUI framework, existing contracts

**Risk Level:** Medium-High - larger refactors with coupled TUI changes

---

## ⚠️ IMPLEMENTATION ORDER

Per the audit notes, Tasks 3.5-3.11 are **coupled** (TUI wiring) and should be done together. Task 3.8 (remove defensive .get()) should be done **LAST** after the NodeState pattern matching is proven stable.

**Recommended order:**
1. Task 3.1: Repository pattern (independent, do first - other tasks depend on it)
2. Tasks 3.2-3.4: Independent refactors (can be parallelized)
3. Task 3.7: Use existing NodeState contracts (must come before 3.8)
4. Tasks 3.5-3.6, 3.9-3.11: TUI wiring (sequential, coupled)
5. Task 3.8: Remove .get() calls (LAST - high risk, depends on 3.7)

---

## Task 3.1: Use Repository Pattern Consistently

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/landscape/test_recorder_uses_repositories.py` (CREATE)

### Context

The recorder currently has inline conversion logic in `get_run()`, `get_nodes()`, etc. that duplicates the conversion logic in repositories. The repositories were created but recorder wasn't updated to use them consistently.

**Pattern to apply:**
```python
# FROM (inline conversion in recorder):
def get_run(self, run_id: str) -> Run | None:
    row = ...  # fetch from DB
    return Run(status=RunStatus(row.status), ...)  # Inline conversion

# TO (delegate to repository):
def get_run(self, run_id: str) -> Run | None:
    row = ...  # fetch from DB
    return RunRepository(None).load(row)  # Repository does conversion
```

### Step 1: Write failing test for repository delegation

```python
# tests/core/landscape/test_recorder_uses_repositories.py
"""Tests verifying recorder uses repositories for model loading."""

from unittest.mock import MagicMock, patch

import pytest

from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class TestRecorderUsesRepositories:
    """Verify recorder delegates to repository classes."""

    def test_get_run_uses_run_repository(self) -> None:
        """get_run should delegate to RunRepository.load()."""
        # Create a real DB with in-memory SQLite
        db = LandscapeDB(":memory:")
        db.initialize()
        recorder = LandscapeRecorder(db)

        # Create a run
        run = recorder.begin_run(config={}, canonical_version="v1")

        with patch(
            "elspeth.core.landscape.recorder.RunRepository"
        ) as mock_repo_class:
            # Set up mock to capture the instance
            mock_instance = MagicMock()
            mock_repo_class.return_value = mock_instance
            mock_instance.load.return_value = run

            result = recorder.get_run(run.run_id)

            # Verify repository was instantiated and load() called
            mock_repo_class.assert_called_once_with(None)
            mock_instance.load.assert_called_once()

    def test_get_nodes_uses_node_repository(self) -> None:
        """get_nodes should delegate to NodeRepository.load()."""
        from elspeth.contracts import Determinism, NodeType

        db = LandscapeDB(":memory:")
        db.initialize()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="node-1",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )

        with patch(
            "elspeth.core.landscape.recorder.NodeRepository"
        ) as mock_repo_class:
            mock_instance = MagicMock()
            mock_repo_class.return_value = mock_instance

            # Create a mock node to return
            mock_node = MagicMock()
            mock_instance.load.return_value = mock_node

            nodes = recorder.get_nodes(run.run_id)

            # Verify repository was used
            mock_repo_class.assert_called_with(None)
            assert mock_instance.load.called

    def test_get_row_uses_row_repository(self) -> None:
        """get_row should delegate to RowRepository.load()."""
        from elspeth.contracts import Determinism, NodeType

        db = LandscapeDB(":memory:")
        db.initialize()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source-1",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.IO_READ,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source-1",
            row_index=0,
            data={"test": "data"},  # Note: parameter is 'data', not 'source_data'
        )

        with patch(
            "elspeth.core.landscape.recorder.RowRepository"
        ) as mock_repo_class:
            mock_instance = MagicMock()
            mock_repo_class.return_value = mock_instance
            mock_instance.load.return_value = row

            result = recorder.get_row(row.row_id)

            mock_repo_class.assert_called_once_with(None)
            mock_instance.load.assert_called_once()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder_uses_repositories.py -v`
Expected: FAIL - repository not called

### Step 3: Update recorder to use repositories

In `src/elspeth/core/landscape/recorder.py`, add import:

```python
from elspeth.core.landscape.repositories import (
    BatchRepository,
    CallRepository,
    EdgeRepository,
    NodeRepository,
    RoutingEventRepository,
    RowRepository,
    RunRepository,
    TokenParentRepository,
    TokenRepository,
)
```

Update `get_run()` method (around line 303):

```python
def get_run(self, run_id: str) -> Run | None:
    """Get a run by ID."""
    with self._db.connection() as conn:
        result = conn.execute(
            select(runs_table).where(runs_table.c.run_id == run_id)
        )
        row = result.fetchone()

    if row is None:
        return None

    return RunRepository(None).load(row)
```

Update `get_nodes()` method (around line 552):

```python
def get_nodes(self, run_id: str) -> list[Node]:
    """Get all nodes for a run."""
    with self._db.connection() as conn:
        result = conn.execute(
            select(nodes_table)
            .where(nodes_table.c.run_id == run_id)
            .order_by(nodes_table.c.sequence_in_pipeline.nullslast())
        )
        rows = result.fetchall()

    repo = NodeRepository(None)
    return [repo.load(row) for row in rows]
```

Similarly update:
- `get_rows()` - use RowRepository
- `get_tokens()` - use TokenRepository
- `get_row()` - use RowRepository
- `get_token()` - use TokenRepository
- `get_token_parents()` - use TokenParentRepository

**Note:** Keep `get_node_state()` using the `_row_to_node_state()` helper - it handles the discriminated union conversion which is specialized.

### Step 4: Run test to verify it passes

Run: `pytest tests/core/landscape/test_recorder_uses_repositories.py -v`
Expected: PASS

### Step 5: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

### Step 6: Commit

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder_uses_repositories.py
git commit -m "$(cat <<'EOF'
refactor(recorder): delegate to repositories for model loading

Use RunRepository, NodeRepository, RowRepository, TokenRepository,
TokenParentRepository in recorder's get_* methods instead of
duplicating conversion logic inline. DRY improvement.

get_node_state() retains specialized _row_to_node_state() for
discriminated union handling.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.2: Create RoutingMap Dataclass

**Files:**
- Create: `src/elspeth/engine/routing_map.py`
- Modify: `src/elspeth/engine/orchestrator.py`
- Modify: `src/elspeth/engine/processor.py`
- Modify: `src/elspeth/engine/executors.py`
- Test: `tests/engine/test_routing_map.py` (CREATE)

### Context

Currently `edge_map` and `route_resolution_map` are raw dicts with `tuple[str, str]` keys. This task creates a type-safe wrapper with clear error messages.

### Step 1: Write failing test for RoutingMap

```python
# tests/engine/test_routing_map.py
"""Tests for RoutingMap dataclass."""

import pytest


class TestRoutingMap:
    """Test type-safe routing resolution."""

    def test_get_edge_id_success(self) -> None:
        """Should return edge ID for valid key."""
        from elspeth.engine.routing_map import RoutingMap

        routing_map = RoutingMap(
            edges={("node-1", "continue"): "edge-1"},
            destinations={("node-1", "continue"): "continue"},
        )

        assert routing_map.get_edge_id("node-1", "continue") == "edge-1"

    def test_get_edge_id_missing_raises(self) -> None:
        """Should raise MissingEdgeError with context."""
        from elspeth.engine.routing_map import MissingEdgeError, RoutingMap

        routing_map = RoutingMap(
            edges={("node-1", "continue"): "edge-1"},
            destinations={},
        )

        with pytest.raises(MissingEdgeError) as exc_info:
            routing_map.get_edge_id("node-2", "route_a")

        assert "node-2" in str(exc_info.value)
        assert "route_a" in str(exc_info.value)

    def test_get_destination_success(self) -> None:
        """Should return destination for valid key."""
        from elspeth.engine.routing_map import RoutingMap

        routing_map = RoutingMap(
            edges={},
            destinations={("gate-1", "suspicious"): "quarantine_sink"},
        )

        assert routing_map.get_destination("gate-1", "suspicious") == "quarantine_sink"

    def test_get_destination_missing_raises(self) -> None:
        """Should raise MissingDestinationError with context."""
        from elspeth.engine.routing_map import MissingDestinationError, RoutingMap

        routing_map = RoutingMap(
            edges={},
            destinations={("gate-1", "normal"): "continue"},
        )

        with pytest.raises(MissingDestinationError) as exc_info:
            routing_map.get_destination("gate-1", "suspicious")

        assert "gate-1" in str(exc_info.value)
        assert "suspicious" in str(exc_info.value)

    def test_from_maps_factory(self) -> None:
        """Should construct from edge_map and route_resolution_map."""
        from elspeth.engine.routing_map import RoutingMap

        edge_map = {("n1", "continue"): "e1", ("n2", "route"): "e2"}
        route_map = {("n1", "continue"): "continue", ("n2", "route"): "sink1"}

        routing_map = RoutingMap.from_maps(edge_map, route_map)

        assert routing_map.get_edge_id("n1", "continue") == "e1"
        assert routing_map.get_destination("n2", "route") == "sink1"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_routing_map.py -v`
Expected: FAIL - module doesn't exist

### Step 3: Create routing_map.py

```python
# src/elspeth/engine/routing_map.py
"""Type-safe routing resolution for pipeline execution.

Wraps the raw edge_map and route_resolution_map dicts with proper
error messages and type safety.
"""

from dataclasses import dataclass


class MissingEdgeError(Exception):
    """Raised when edge lookup fails."""

    def __init__(self, node_id: str, label: str, available: list[tuple[str, str]]) -> None:
        self.node_id = node_id
        self.label = label
        self.available = available
        super().__init__(
            f"No edge found for node '{node_id}' with label '{label}'. "
            f"Available edges: {available}"
        )


class MissingDestinationError(Exception):
    """Raised when destination lookup fails."""

    def __init__(self, node_id: str, label: str, available: list[tuple[str, str]]) -> None:
        self.node_id = node_id
        self.label = label
        self.available = available
        super().__init__(
            f"No destination found for node '{node_id}' with label '{label}'. "
            f"Available routes: {available}"
        )


@dataclass(frozen=True)
class RoutingMap:
    """Type-safe routing resolution.

    Provides clear error context when lookups fail, unlike raw dict access.
    """

    edges: dict[tuple[str, str], str]
    destinations: dict[tuple[str, str], str]

    def get_edge_id(self, node_id: str, label: str) -> str:
        """Get edge ID for a node/label pair.

        Args:
            node_id: Source node ID
            label: Edge label (e.g., "continue", "route_to_sink")

        Returns:
            Edge ID

        Raises:
            MissingEdgeError: If no edge exists for this node/label
        """
        key = (node_id, label)
        if key not in self.edges:
            raise MissingEdgeError(
                node_id=node_id,
                label=label,
                available=list(self.edges.keys()),
            )
        return self.edges[key]

    def get_destination(self, node_id: str, label: str) -> str:
        """Get destination for a node/label pair.

        Args:
            node_id: Gate node ID
            label: Route label

        Returns:
            Destination ("continue" or sink name)

        Raises:
            MissingDestinationError: If no destination configured
        """
        key = (node_id, label)
        if key not in self.destinations:
            raise MissingDestinationError(
                node_id=node_id,
                label=label,
                available=list(self.destinations.keys()),
            )
        return self.destinations[key]

    @classmethod
    def from_maps(
        cls,
        edge_map: dict[tuple[str, str], str],
        route_resolution_map: dict[tuple[str, str], str],
    ) -> "RoutingMap":
        """Create RoutingMap from raw dict maps.

        Args:
            edge_map: Maps (node_id, label) -> edge_id
            route_resolution_map: Maps (node_id, label) -> destination

        Returns:
            RoutingMap instance
        """
        return cls(edges=edge_map, destinations=route_resolution_map)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/engine/test_routing_map.py -v`
Expected: PASS

### Step 5: Update orchestrator to use RoutingMap

In `src/elspeth/engine/orchestrator.py`, after building edge_map and route_resolution_map (around line 386):

```python
from elspeth.engine.routing_map import RoutingMap

# After building edge_map and route_resolution_map:
routing_map = RoutingMap.from_maps(edge_map, route_resolution_map)
```

Update RowProcessor instantiation (around line 443):

```python
# Change from:
processor = RowProcessor(
    recorder=recorder,
    span_factory=self._span_factory,
    run_id=run_id,
    source_node_id=source_id,
    edge_map=edge_map,
    route_resolution_map=route_resolution_map,
)

# To:
processor = RowProcessor(
    recorder=recorder,
    span_factory=self._span_factory,
    run_id=run_id,
    source_node_id=source_id,
    routing_map=routing_map,
)
```

### Step 6: Update RowProcessor to accept RoutingMap

In `src/elspeth/engine/processor.py`:

```python
from elspeth.engine.routing_map import RoutingMap

class RowProcessor:
    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        source_node_id: str,
        routing_map: RoutingMap,  # Changed from edge_map + route_resolution_map
    ) -> None:
        self._recorder = recorder
        self._span_factory = span_factory
        self._run_id = run_id
        self._source_node_id = source_node_id
        self._routing_map = routing_map
```

Update usages from `self._edge_map[(node_id, label)]` to `self._routing_map.get_edge_id(node_id, label)`.

### Step 7: Update executors to use RoutingMap

In `src/elspeth/engine/executors.py`, update GateExecutor and any other classes that access edge_map or route_resolution_map to use the RoutingMap methods.

### Step 8: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

### Step 9: Commit

```bash
git add src/elspeth/engine/routing_map.py src/elspeth/engine/orchestrator.py src/elspeth/engine/processor.py src/elspeth/engine/executors.py tests/engine/test_routing_map.py
git commit -m "$(cat <<'EOF'
feat(engine): add RoutingMap dataclass for type-safe routing

Replace raw dict access for edge_map and route_resolution_map with
RoutingMap that provides clear error messages on lookup failures.

Updates orchestrator, RowProcessor, and executors to use the new
type-safe interface.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.3: Create ExecutionGraphProtocol

**Files:**
- Modify: `src/elspeth/contracts/engine.py`
- Test: `tests/contracts/test_execution_graph_protocol.py` (CREATE)

### Step 1: Verify actual ExecutionGraph interface

First, read the actual ExecutionGraph class to ensure the protocol matches:

```bash
grep -n "def " src/elspeth/core/dag.py | head -30
```

### Step 2: Write test for protocol compliance

```python
# tests/contracts/test_execution_graph_protocol.py
"""Tests for ExecutionGraphProtocol."""

import pytest


class TestExecutionGraphProtocol:
    """Verify ExecutionGraph implements the protocol."""

    def test_execution_graph_satisfies_protocol(self) -> None:
        """ExecutionGraph should satisfy ExecutionGraphProtocol."""
        from elspeth.contracts import ExecutionGraphProtocol
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        # Check it has required methods (based on actual ExecutionGraph)
        assert hasattr(graph, "add_node")
        assert hasattr(graph, "add_edge")
        assert hasattr(graph, "get_edges")
        assert hasattr(graph, "topological_order")
        assert hasattr(graph, "get_source")
        assert hasattr(graph, "get_sink_id_map")
        assert hasattr(graph, "get_transform_id_map")
        assert hasattr(graph, "get_route_resolution_map")
        assert hasattr(graph, "get_output_sink")

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should support isinstance checks."""
        from elspeth.contracts import ExecutionGraphProtocol
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        assert isinstance(graph, ExecutionGraphProtocol)
```

### Step 3: Add ExecutionGraphProtocol to contracts/engine.py

**IMPORTANT:** Match the actual ExecutionGraph signatures exactly.

Add to `src/elspeth/contracts/engine.py`:

```python
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.contracts import EdgeInfo, NodeInfo, RoutingMode


@runtime_checkable
class ExecutionGraphProtocol(Protocol):
    """Contract for execution graph implementations.

    Defines the interface that orchestrator expects from graph objects.
    This allows alternative graph implementations (e.g., for testing).

    Note: Signatures must match actual ExecutionGraph in core/dag.py.
    """

    def add_node(
        self,
        node_id: str,
        *,
        node_type: str,
        plugin_name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add a node to the graph."""
        ...

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        label: str,
        mode: "RoutingMode",
    ) -> None:
        """Add an edge between nodes."""
        ...

    def get_edges(self) -> list["EdgeInfo"]:
        """Get all edges in the graph as typed EdgeInfo contracts."""
        ...

    def topological_order(self) -> list[str]:
        """Get nodes in topological order."""
        ...

    def get_source(self) -> str | None:
        """Get the source node ID."""
        ...

    def get_sinks(self) -> list[str]:
        """Get all sink node IDs."""
        ...

    def get_sink_id_map(self) -> dict[str, str]:
        """Get mapping of sink names to node IDs."""
        ...

    def get_transform_id_map(self) -> dict[int, str]:
        """Get mapping of transform sequence to node IDs."""
        ...

    def get_route_resolution_map(self) -> dict[tuple[str, str], str]:
        """Get route resolution mapping."""
        ...

    def get_output_sink(self) -> str:
        """Get the default output sink name."""
        ...

    def get_node_info(self, node_id: str) -> "NodeInfo":
        """Get NodeInfo for a node.

        Raises:
            KeyError: If node doesn't exist
        """
        ...
```

### Step 4: Export from contracts/__init__.py

Add `ExecutionGraphProtocol` to exports in `src/elspeth/contracts/__init__.py`.

### Step 5: Run test to verify it passes

Run: `pytest tests/contracts/test_execution_graph_protocol.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/contracts/engine.py src/elspeth/contracts/__init__.py tests/contracts/test_execution_graph_protocol.py
git commit -m "$(cat <<'EOF'
feat(contracts): add ExecutionGraphProtocol

Define the contract that orchestrator expects from execution graphs.
Protocol signatures match actual ExecutionGraph implementation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.4: Create SinkLike Protocol

**Files:**
- Modify: `src/elspeth/contracts/engine.py`
- Test: `tests/contracts/test_sink_like_protocol.py` (CREATE)

### Step 1: Write test for SinkLike protocol

```python
# tests/contracts/test_sink_like_protocol.py
"""Tests for SinkLike protocol."""

import pytest


class TestSinkLikeProtocol:
    """Verify SinkLike protocol works with adapters."""

    def test_sink_like_importable(self) -> None:
        """SinkLike should be importable from contracts."""
        from elspeth.contracts import SinkLike

        assert SinkLike is not None

    def test_sink_adapter_has_required_interface(self) -> None:
        """SinkAdapter should have the SinkLike interface."""
        from elspeth.engine.adapters import SinkAdapter

        # Check it has the required attributes/methods
        assert hasattr(SinkAdapter, "write")
        assert hasattr(SinkAdapter, "name")
        assert hasattr(SinkAdapter, "node_id")
```

### Step 2: Add SinkLike protocol

Add to `src/elspeth/contracts/engine.py`:

```python
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext


@runtime_checkable
class SinkLike(Protocol):
    """Protocol for bulk sink operations (Phase 3B adapter interface).

    This is the interface that SinkAdapter provides for bulk writes,
    as opposed to SinkProtocol which defines single-row writes.

    Used by orchestrator when writing accumulated tokens to sinks.
    """

    name: str
    node_id: str | None

    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "PluginContext",
    ) -> Any:  # Returns ArtifactDescriptor
        """Write multiple rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with output metadata
        """
        ...

    def flush(self) -> None:
        """Flush any buffered data to the underlying sink."""
        ...

    def close(self) -> None:
        """Close the sink and release resources."""
        ...
```

### Step 3: Export and run test

Run: `pytest tests/contracts/test_sink_like_protocol.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/elspeth/contracts/engine.py tests/contracts/test_sink_like_protocol.py
git commit -m "$(cat <<'EOF'
feat(contracts): add SinkLike protocol for bulk sink operations

Defines the interface for bulk write operations as implemented by
SinkAdapter, distinct from SinkProtocol's single-row interface.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.5: Add Database Parameter to ExplainApp

**Files:**
- Modify: `src/elspeth/tui/explain_app.py`
- Test: `tests/tui/test_explain_app_db.py` (CREATE)

### Step 1: Write test for db parameter

```python
# tests/tui/test_explain_app_db.py
"""Tests for ExplainApp database integration."""

from unittest.mock import MagicMock

import pytest


class TestExplainAppDatabase:
    """Test ExplainApp accepts database connection."""

    def test_explain_app_accepts_db_parameter(self) -> None:
        """ExplainApp should accept db parameter."""
        from elspeth.tui.explain_app import ExplainApp

        mock_db = MagicMock()
        app = ExplainApp(db=mock_db, run_id="run-123")

        assert app._db is mock_db
        assert app.run_id == "run-123"

    def test_explain_app_db_defaults_to_none(self) -> None:
        """ExplainApp should work without db (degraded mode)."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp(run_id="run-123")

        assert app._db is None

    def test_explain_app_creates_screen_with_db(self) -> None:
        """ExplainApp should create ExplainScreen when db provided."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.tui.explain_app import ExplainApp

        db = LandscapeDB(":memory:")
        db.initialize()

        app = ExplainApp(db=db, run_id="run-123")

        # Screen should be created in __init__, not compose()
        assert app._screen is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/tui/test_explain_app_db.py -v`
Expected: FAIL - db parameter not accepted, _screen not created

### Step 3: Update ExplainApp.__init__

Update `src/elspeth/tui/explain_app.py`:

```python
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from elspeth.tui.constants import WidgetIDs
from elspeth.tui.screens.explain_screen import ExplainScreen
from elspeth.tui.widgets.lineage_tree import LineageTree  # Import at module level

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


class ExplainApp(App[None]):
    """Interactive TUI for exploring run lineage.

    Architecture:
    - ExplainApp: Textual App that manages widgets and user interaction
    - ExplainScreen: State manager that handles data loading (created in __init__)
    - Widgets created in compose() based on ExplainScreen state
    """

    TITLE = "ELSPETH Explain"
    CSS = f"""
    Screen {{
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }}

    #{WidgetIDs.LINEAGE_TREE} {{
        height: 100%;
        border: solid green;
    }}

    #{WidgetIDs.DETAIL_PANEL} {{
        height: 100%;
        border: solid blue;
    }}
    """

    BINDINGS = [  # noqa: RUF012 - Textual pattern
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        db: "LandscapeDB | None" = None,
        run_id: str | None = None,
        token_id: str | None = None,
        row_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._db = db
        self.run_id = run_id
        self.token_id = token_id
        self.row_id = row_id

        # Create ExplainScreen in __init__, not compose()
        # ExplainScreen manages state; we query it in compose() for widget data
        self._screen: ExplainScreen | None = None
        if db is not None and run_id is not None:
            self._screen = ExplainScreen(db=db, run_id=run_id)

    def compose(self) -> ComposeResult:
        """Create child widgets based on ExplainScreen state."""
        yield Header()

        if self._screen is not None:
            lineage_data = self._screen.get_lineage_data()

            if lineage_data is not None:
                # LineageTree imported at module level
                yield LineageTree(lineage_data, id=WidgetIDs.LINEAGE_TREE)
                yield Static("Select a node to view details", id=WidgetIDs.DETAIL_PANEL)
            else:
                yield Static("Failed to load lineage data", id=WidgetIDs.LINEAGE_TREE)
                yield Static("No data available", id=WidgetIDs.DETAIL_PANEL)
        else:
            yield Static("Lineage Tree (no database)", id=WidgetIDs.LINEAGE_TREE)
            yield Static("Detail Panel (no database)", id=WidgetIDs.DETAIL_PANEL)

        yield Footer()

    def action_refresh(self) -> None:
        """Refresh lineage data."""
        self.notify("Refreshing...")

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press q to quit, arrow keys to navigate")
```

### Step 4: Run test to verify it passes

Run: `pytest tests/tui/test_explain_app_db.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/tui/explain_app.py tests/tui/test_explain_app_db.py
git commit -m "$(cat <<'EOF'
feat(tui): add database parameter to ExplainApp

ExplainApp now:
- Accepts db parameter for database connection
- Creates ExplainScreen in __init__ (not compose) for state management
- Queries ExplainScreen in compose() to build widgets

This separates state management (ExplainScreen) from widget creation
(compose), following Textual patterns.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.6: Wire ExplainScreen Data Loading

**Files:**
- Modify: `src/elspeth/tui/screens/explain_screen.py` (minor updates)
- Test: Covered by Task 3.5 tests

### Context

ExplainScreen already has good discriminated union state management. Task 3.5 wired it into ExplainApp correctly (created in __init__, queried in compose).

This task ensures the data loading in ExplainScreen properly populates all fields.

### Step 1: Verify ExplainScreen state management is correct

The current ExplainScreen already:
- Has discriminated union states (UninitializedState, LoadingFailedState, LoadedState)
- Creates LineageTree in LoadedState
- Has `get_lineage_data()` method

No changes needed to the architecture - Task 3.5 already handles the integration correctly.

### Step 2: Commit (if any minor fixes needed)

```bash
git add src/elspeth/tui/screens/explain_screen.py
git commit -m "$(cat <<'EOF'
chore(tui): verify ExplainScreen integration with ExplainApp

ExplainScreen state management confirmed working with new
ExplainApp wiring from Task 3.5.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.7: Use Existing NodeState Contracts in TUI

**Files:**
- Modify: `src/elspeth/tui/widgets/node_detail.py`
- Test: `tests/tui/test_node_detail_contracts.py` (CREATE)

### ⚠️ IMPORTANT: Use Existing Contracts, Don't Create New TypedDict

The codebase already has discriminated union NodeState types:
```python
NodeState = NodeStateOpen | NodeStateCompleted | NodeStateFailed
```

These are **"Our Data"** types with guaranteed fields per status. The TUI should use pattern matching on these existing types instead of creating a new TypedDict.

### Step 1: Write test for NodeState pattern matching

```python
# tests/tui/test_node_detail_contracts.py
"""Tests for NodeDetailPanel using NodeState contracts."""

from datetime import datetime, UTC

import pytest

from elspeth.contracts import (
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStateStatus,
)


class TestNodeDetailWithContracts:
    """Test NodeDetailPanel uses NodeState discriminated union."""

    def test_render_open_state(self) -> None:
        """Should render NodeStateOpen correctly."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = NodeStateOpen(
            state_id="state-1",
            node_id="node-1",
            token_id="token-1",
            run_id="run-1",
            step_index=0,
            started_at=datetime.now(UTC),
            input_hash="abc123",
            status=NodeStateStatus.OPEN,
        )

        panel = NodeDetailPanel(state)
        content = panel.render_content()

        assert "state-1" in content
        assert "OPEN" in content or "open" in content.lower()
        # Should NOT have completed_at for open state
        assert "Completed:" not in content or "N/A" in content

    def test_render_completed_state(self) -> None:
        """Should render NodeStateCompleted with all fields."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = NodeStateCompleted(
            state_id="state-2",
            node_id="node-1",
            token_id="token-1",
            run_id="run-1",
            step_index=0,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            input_hash="abc123",
            output_hash="def456",
            status=NodeStateStatus.COMPLETED,
        )

        panel = NodeDetailPanel(state)
        content = panel.render_content()

        assert "state-2" in content
        assert "def456" in content  # output_hash present

    def test_render_failed_state_with_error(self) -> None:
        """Should render NodeStateFailed with error info."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        state = NodeStateFailed(
            state_id="state-3",
            node_id="node-1",
            token_id="token-1",
            run_id="run-1",
            step_index=0,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            input_hash="abc123",
            error_json='{"type": "ValueError", "message": "bad input"}',
            status=NodeStateStatus.FAILED,
        )

        panel = NodeDetailPanel(state)
        content = panel.render_content()

        assert "state-3" in content
        assert "ValueError" in content or "bad input" in content

    def test_render_none_state(self) -> None:
        """Should handle None state gracefully."""
        from elspeth.tui.widgets.node_detail import NodeDetailPanel

        panel = NodeDetailPanel(None)
        content = panel.render_content()

        assert "No node selected" in content
```

### Step 2: Update NodeDetailPanel to accept NodeState union

Update `src/elspeth/tui/widgets/node_detail.py`:

```python
"""Node detail panel widget for displaying node state information."""

import json
from typing import Any

import structlog

from elspeth.contracts import (
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
)

logger = structlog.get_logger(__name__)


class NodeDetailPanel:
    """Panel displaying detailed information about a selected node.

    Uses pattern matching on NodeState discriminated union to access
    fields safely. This is "Our Data" from Landscape - guaranteed fields
    per status type.
    """

    def __init__(self, node_state: NodeState | None) -> None:
        """Initialize with node state data.

        Args:
            node_state: NodeState (Open, Completed, or Failed), or None if nothing selected
        """
        self._state = node_state

    def render_content(self) -> str:
        """Render panel content as formatted string.

        Uses pattern matching on NodeState union to access fields
        appropriate for each status.
        """
        if self._state is None:
            return "No node selected. Select a node from the tree to view details."

        lines: list[str] = []

        # Pattern match on NodeState union
        match self._state:
            case NodeStateOpen() as state:
                lines.extend(self._render_header(state))
                lines.extend(self._render_identity(state))
                lines.extend(self._render_open_status(state))
                lines.extend(self._render_input_hash(state))

            case NodeStateCompleted() as state:
                lines.extend(self._render_header(state))
                lines.extend(self._render_identity(state))
                lines.extend(self._render_completed_status(state))
                lines.extend(self._render_hashes(state))

            case NodeStateFailed() as state:
                lines.extend(self._render_header(state))
                lines.extend(self._render_identity(state))
                lines.extend(self._render_failed_status(state))
                lines.extend(self._render_error(state))

        return "\n".join(lines)

    def _render_header(self, state: NodeState) -> list[str]:
        """Render header section."""
        # node_id is on all NodeState types
        return [
            f"=== Node {state.node_id} ===",
            "",
        ]

    def _render_identity(self, state: NodeState) -> list[str]:
        """Render identity section - fields common to all states."""
        return [
            "Identity:",
            f"  State ID:  {state.state_id}",
            f"  Node ID:   {state.node_id}",
            f"  Token ID:  {state.token_id}",
            "",
        ]

    def _render_open_status(self, state: NodeStateOpen) -> list[str]:
        """Render status for open (in-progress) state."""
        return [
            "Status:",
            f"  Status:     {state.status.value}",
            f"  Started:    {state.started_at}",
            "",
        ]

    def _render_completed_status(self, state: NodeStateCompleted) -> list[str]:
        """Render status for completed state."""
        duration_ms = None
        if state.started_at and state.completed_at:
            delta = state.completed_at - state.started_at
            duration_ms = int(delta.total_seconds() * 1000)

        lines = [
            "Status:",
            f"  Status:     {state.status.value}",
            f"  Started:    {state.started_at}",
            f"  Completed:  {state.completed_at}",
        ]
        if duration_ms is not None:
            lines.append(f"  Duration:   {duration_ms} ms")
        lines.append("")
        return lines

    def _render_failed_status(self, state: NodeStateFailed) -> list[str]:
        """Render status for failed state."""
        return [
            "Status:",
            f"  Status:     {state.status.value}",
            f"  Started:    {state.started_at}",
            f"  Completed:  {state.completed_at}",
            "",
        ]

    def _render_input_hash(self, state: NodeStateOpen) -> list[str]:
        """Render input hash for open state."""
        return [
            "Data Hashes:",
            f"  Input:   {state.input_hash}",
            "",
        ]

    def _render_hashes(self, state: NodeStateCompleted) -> list[str]:
        """Render both hashes for completed state."""
        return [
            "Data Hashes:",
            f"  Input:   {state.input_hash}",
            f"  Output:  {state.output_hash or '(none)'}",
            "",
        ]

    def _render_error(self, state: NodeStateFailed) -> list[str]:
        """Render error info for failed state."""
        lines = ["Error:"]

        error_json = state.error_json
        if error_json:
            # error_json is "Our Data" - we wrote it via our error recording code.
            # If it's not valid JSON, that's a bug in our serialization, not user data.
            # Crash to surface the bug rather than silently hiding it.
            error = json.loads(error_json)  # Will raise JSONDecodeError if malformed
            if isinstance(error, dict):
                lines.append(f"  Type:    {error.get('type', 'unknown')}")
                lines.append(f"  Message: {error.get('message', 'unknown')}")
            else:
                lines.append(f"  {error}")
        else:
            lines.append("  (no error details)")

        lines.append("")
        return lines

    def update_state(self, node_state: NodeState | None) -> None:
        """Update the displayed node state."""
        self._state = node_state
```

### Step 3: Run tests

Run: `pytest tests/tui/test_node_detail_contracts.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/elspeth/tui/widgets/node_detail.py tests/tui/test_node_detail_contracts.py
git commit -m "$(cat <<'EOF'
refactor(tui): use NodeState discriminated union in NodeDetailPanel

Replace dict-based state with proper NodeState contracts
(NodeStateOpen, NodeStateCompleted, NodeStateFailed).

Pattern matching ensures field access is type-safe:
- Open state: has input_hash, no completed_at/output_hash
- Completed state: has all timing and hash fields
- Failed state: has error_json

This is "Our Data" from Landscape - fields are guaranteed per status.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.8: Remove Defensive .get() Calls from NodeDetailPanel

**Files:**
- Already done in Task 3.7!

### Context

Task 3.7 replaced the dict-based approach with pattern matching on NodeState discriminated union. This inherently removes the need for defensive `.get()` calls because:

1. Pattern matching guarantees we only access fields that exist on each type
2. `NodeStateCompleted.completed_at` is guaranteed present - no need for `.get()`
3. `NodeStateOpen` doesn't have `completed_at` - we don't try to access it

### Step 1: Verify no .get() on required fields

Review `node_detail.py` to confirm:
- Direct attribute access on dataclass fields
- Only JSON parsing uses try/except (legitimate for string parsing)
- No `.get()` with defaults that hide missing data

### Step 2: Run full TUI test suite

Run: `pytest tests/tui/ -v --tb=short`
Expected: All pass

### Step 3: Commit (if any cleanup needed)

```bash
git add src/elspeth/tui/widgets/node_detail.py
git commit -m "$(cat <<'EOF'
refactor(tui): confirm no defensive .get() in NodeDetailPanel

Task 3.7's pattern matching approach inherently prevents defensive
.get() usage - fields are accessed only when guaranteed present.

Per CLAUDE.md, NodeState is "Our Data" - crash on anomalies, don't
hide them with defaults.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.9: Implement Token Lineage Path Tracing

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/landscape/test_token_path_tracing.py` (CREATE)

**DEPENDS ON:** Task 3.1 (recorder uses repositories)

### Step 1: Write test for trace_token_path

```python
# tests/core/landscape/test_token_path_tracing.py
"""Tests for token lineage path tracing."""

import pytest

from elspeth.contracts import Determinism, NodeType, RoutingMode
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class TestTokenPathTracing:
    """Test trace_token_path() method."""

    def test_trace_simple_path(self) -> None:
        """Should trace token through source -> transform -> sink."""
        db = LandscapeDB(":memory:")
        db.initialize()
        recorder = LandscapeRecorder(db)

        # Set up run with nodes
        run = recorder.begin_run(config={}, canonical_version="v1")

        recorder.register_node(
            run_id=run.run_id,
            node_id="source-1",
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.IO_READ,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="transform-1",
            plugin_name="enrich",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink-1",
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.IO_WRITE,
        )

        # Register edges
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="source-1",
            to_node_id="transform-1",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="transform-1",
            to_node_id="sink-1",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Create row and token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source-1",
            row_index=0,
            data={"name": "test"},  # Note: parameter is 'data', not 'source_data'
        )
        token = recorder.create_token(row_id=row.row_id)

        # Record node states for the token's path
        # Note: begin_node_state() does NOT take run_id - it's derived from token
        recorder.begin_node_state(
            token_id=token.token_id,
            node_id="source-1",
            step_index=0,
            input_data={"name": "test"},
        )
        recorder.begin_node_state(
            token_id=token.token_id,
            node_id="transform-1",
            step_index=1,
            input_data={"name": "test"},
        )
        recorder.begin_node_state(
            token_id=token.token_id,
            node_id="sink-1",
            step_index=2,
            input_data={"name": "test", "enriched": True},
        )

        # Trace the path
        path = recorder.trace_token_path(token.token_id)

        assert path == ["csv_source", "enrich", "csv_sink"]

    def test_trace_nonexistent_token_returns_empty(self) -> None:
        """Should return empty list for nonexistent token."""
        db = LandscapeDB(":memory:")
        db.initialize()
        recorder = LandscapeRecorder(db)

        path = recorder.trace_token_path("nonexistent-token")

        assert path == []
```

### Step 2: Add trace_token_path method to recorder

Add to `src/elspeth/core/landscape/recorder.py`:

```python
def trace_token_path(self, token_id: str) -> list[str]:
    """Get ordered list of plugin names for a token's DAG path.

    Traces the token through node states to show which plugins
    processed it in order.

    Args:
        token_id: Token ID to trace

    Returns:
        List of plugin names in processing order, empty if token not found
    """
    # Get all node states for this token
    states = self.get_node_states_for_token(token_id)

    if not states:
        return []

    # Get the token to find the row, then the run
    token = self.get_token(token_id)
    if token is None:
        return []

    row = self.get_row(token.row_id)
    if row is None:
        return []

    # Get all nodes for this run and build lookup map
    nodes = self.get_nodes(row.run_id)
    node_map = {n.node_id: n.plugin_name for n in nodes}

    # Build path from states (already ordered by step_index)
    path: list[str] = []
    for state in states:
        # Direct access - if state.node_id isn't in node_map, that's a
        # data integrity bug. This is "Our Data" - crash, don't hide.
        plugin_name = node_map[state.node_id]
        path.append(plugin_name)

    return path
```

### Step 3: Run tests

Run: `pytest tests/core/landscape/test_token_path_tracing.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_token_path_tracing.py
git commit -m "$(cat <<'EOF'
feat(recorder): add trace_token_path() for lineage visualization

Returns ordered list of plugin names a token passed through,
enabling TUI breadcrumb display of token journey.

Uses direct dict access on node_map - if a node_state references
a node_id that doesn't exist, that's a data integrity bug that
should crash (per "Our Data" principle).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.10: Fix Tokens Field in ExplainScreen

**Files:**
- Modify: `src/elspeth/tui/screens/explain_screen.py`
- Test: `tests/tui/test_explain_screen_tokens.py` (CREATE)

### Step 1: Write test for token loading

```python
# tests/tui/test_explain_screen_tokens.py
"""Tests for token loading in ExplainScreen."""

import pytest

from elspeth.contracts import Determinism, NodeType, RoutingMode
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.tui.screens.explain_screen import ExplainScreen, LoadedState


class TestExplainScreenTokens:
    """Test ExplainScreen populates tokens field."""

    def test_loaded_state_has_tokens(self) -> None:
        """LoadedState should have tokens populated."""
        db = LandscapeDB(":memory:")
        db.initialize()
        recorder = LandscapeRecorder(db)

        # Create run with nodes
        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.IO_READ,
        )

        # Create row and token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source-1",
            row_index=0,
            data={"name": "test"},  # Note: parameter is 'data', not 'source_data'
        )
        token = recorder.create_token(row_id=row.row_id)

        # Create screen
        screen = ExplainScreen(db=db, run_id=run.run_id)

        # Check state
        assert isinstance(screen.state, LoadedState)
        lineage_data = screen.get_lineage_data()
        assert lineage_data is not None

        # Tokens should be populated (not hardcoded [])
        assert "tokens" in lineage_data
        assert len(lineage_data["tokens"]) >= 1
        assert lineage_data["tokens"][0]["token_id"] == token.token_id
```

### Step 2: Update ExplainScreen to load tokens

Update `_load_pipeline_structure()` in `src/elspeth/tui/screens/explain_screen.py`:

First, add imports at the top of the file:
```python
from elspeth.tui.types import LineageData, TokenDisplayInfo
```

Then update the method:
```python
def _load_pipeline_structure(
    self, db: LandscapeDB, run_id: str
) -> LoadedState | LoadingFailedState:
    """Load pipeline structure from database."""
    try:
        recorder = LandscapeRecorder(db)
        nodes = recorder.get_nodes(run_id)
        rows = recorder.get_rows(run_id)

        # Organize nodes by type
        source_nodes = [n for n in nodes if n.node_type == NodeType.SOURCE]
        transform_nodes = [n for n in nodes if n.node_type == NodeType.TRANSFORM]
        sink_nodes = [n for n in nodes if n.node_type == NodeType.SINK]

        # Load tokens for all rows (limit for performance)
        tokens_display: list[TokenDisplayInfo] = []
        for row in rows[:100]:
            row_tokens = recorder.get_tokens(row.row_id)
            for token in row_tokens:
                path = recorder.trace_token_path(token.token_id)
                tokens_display.append({
                    "token_id": token.token_id,
                    "row_id": token.row_id,
                    "path": path,
                })

        lineage_data: LineageData = {
            "run_id": run_id,
            "source": {
                "name": source_nodes[0].plugin_name if source_nodes else "unknown",
                "node_id": source_nodes[0].node_id if source_nodes else None,
            }
            if source_nodes
            else {"name": "unknown", "node_id": None},
            "transforms": [
                {"name": n.plugin_name, "node_id": n.node_id}
                for n in transform_nodes
            ],
            "sinks": [
                {"name": n.plugin_name, "node_id": n.node_id} for n in sink_nodes
            ],
            "tokens": tokens_display,  # Now populated!
        }
        tree = LineageTree(lineage_data)
        return LoadedState(
            db=db,
            run_id=run_id,
            lineage_data=lineage_data,
            tree=tree,
        )
    except Exception:
        return LoadingFailedState(db=db, run_id=run_id)
```

### Step 3: Run tests

Run: `pytest tests/tui/test_explain_screen_tokens.py -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/elspeth/tui/screens/explain_screen.py tests/tui/test_explain_screen_tokens.py
git commit -m "$(cat <<'EOF'
fix(tui): populate tokens field in ExplainScreen

Load actual tokens from database instead of hardcoding empty list.
Uses trace_token_path() for path display in token info.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3.11: CLI Passes Database to TUI

**Files:**
- Modify: `src/elspeth/cli.py`
- Modify: `src/elspeth/core/landscape/recorder.py` (add get_latest_run)
- Test: `tests/cli/test_explain_command.py` (CREATE)

### Step 1: Add get_latest_run to recorder

Add to `src/elspeth/core/landscape/recorder.py`:

```python
def get_latest_run(self) -> Run | None:
    """Get the most recently started run.

    Returns:
        Most recent Run or None if no runs exist
    """
    with self._db.connection() as conn:
        result = conn.execute(
            select(runs_table)
            .order_by(runs_table.c.started_at.desc())
            .limit(1)
        )
        row = result.fetchone()

    if row is None:
        return None

    return RunRepository(None).load(row)
```

### Step 2: Write test for explain command

```python
# tests/cli/test_explain_command.py
"""Tests for explain CLI command."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from elspeth.cli import app


class TestExplainCommand:
    """Test explain command passes db to TUI."""

    def test_explain_no_tui_mode(self) -> None:
        """explain --no-tui should use ExplainScreen.render()."""
        runner = CliRunner()

        with patch("elspeth.cli.load_settings") as mock_load:
            with patch("elspeth.cli.LandscapeDB") as mock_db_class:
                with patch("elspeth.cli.LandscapeRecorder") as mock_recorder_class:
                    # Set up mocks
                    mock_settings = MagicMock()
                    mock_settings.landscape.database = ":memory:"
                    mock_load.return_value = mock_settings

                    mock_db = MagicMock()
                    mock_db_class.return_value = mock_db

                    mock_recorder = MagicMock()
                    mock_recorder_class.return_value = mock_recorder

                    # Mock get_latest_run for "latest" resolution
                    mock_run = MagicMock()
                    mock_run.run_id = "run-123"
                    mock_recorder.get_latest_run.return_value = mock_run

                    result = runner.invoke(
                        app,
                        ["explain", "--run", "latest", "--no-tui"],
                    )

                    # Should have tried to load data
                    mock_recorder.get_latest_run.assert_called_once()

    def test_explain_with_specific_run_id(self) -> None:
        """explain with specific run ID should not resolve 'latest'."""
        runner = CliRunner()

        with patch("elspeth.cli.load_settings") as mock_load:
            with patch("elspeth.cli.LandscapeDB") as mock_db_class:
                with patch("elspeth.cli.ExplainScreen") as mock_screen_class:
                    mock_settings = MagicMock()
                    mock_settings.landscape.database = ":memory:"
                    mock_load.return_value = mock_settings

                    mock_db = MagicMock()
                    mock_db_class.return_value = mock_db

                    mock_screen = MagicMock()
                    mock_screen.render.return_value = "test output"
                    mock_screen_class.return_value = mock_screen

                    result = runner.invoke(
                        app,
                        ["explain", "--run", "specific-run-id", "--no-tui"],
                    )

                    # Should create screen with the specific run_id
                    mock_screen_class.assert_called_once()
                    call_kwargs = mock_screen_class.call_args[1]
                    assert call_kwargs["run_id"] == "specific-run-id"
```

### Step 3: Update explain command in cli.py

Update `src/elspeth/cli.py`:

First, add these imports at the top of the file (with other imports):
```python
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.tui.explain_app import ExplainApp
from elspeth.tui.screens.explain_screen import ExplainScreen
```

Then add the explain command:
```python
@app.command()
def explain(
    run_id: str = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run ID to explain (or 'latest').",
    ),
    row: str | None = typer.Option(
        None,
        "--row",
        help="Row ID or index to explain.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="Token ID for precise lineage.",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="Output text instead of interactive TUI.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """Explain lineage for a row or token."""
    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    # Create database connection
    db = LandscapeDB(settings.landscape.database)

    # Resolve "latest" run_id
    actual_run_id = run_id
    if run_id == "latest":
        recorder = LandscapeRecorder(db)
        latest_run = recorder.get_latest_run()
        if latest_run is None:
            typer.echo("No runs found", err=True)
            raise typer.Exit(1)
        actual_run_id = latest_run.run_id

    if no_tui or json_output:
        # Text/JSON output mode - use ExplainScreen directly
        screen = ExplainScreen(db=db, run_id=actual_run_id)

        if json_output:
            import json
            lineage_data = screen.get_lineage_data()
            typer.echo(json.dumps(lineage_data, default=str))
        else:
            typer.echo(screen.render())
    else:
        # Interactive TUI mode
        tui_app = ExplainApp(
            db=db,
            run_id=actual_run_id,
            token_id=token,
            row_id=row,
        )
        tui_app.run()
```

### Step 4: Run tests

Run: `pytest tests/cli/test_explain_command.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/cli.py src/elspeth/core/landscape/recorder.py tests/cli/test_explain_command.py
git commit -m "$(cat <<'EOF'
feat(cli): wire explain command to database and TUI

explain command now:
- Loads settings with load_settings()
- Creates LandscapeDB connection from settings
- Resolves "latest" run_id via get_latest_run()
- Uses ExplainScreen for --no-tui/--json modes
- Uses ExplainApp for interactive TUI mode

Adds get_latest_run() to LandscapeRecorder.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

### Run full test suite

```bash
pytest tests/ -v
```

### Run type checker

```bash
mypy src/elspeth/ --ignore-missing-imports
```

### Run linter

```bash
ruff check src/elspeth/
```

### Manual TUI test

```bash
elspeth explain --run latest
elspeth explain --run latest --no-tui
elspeth explain --run latest --json
```

---

## Summary

| Task | Description | Risk | Key Change |
|------|-------------|------|------------|
| 3.1 | Use Repository pattern | Low | Recorder delegates to repositories |
| 3.2 | Create RoutingMap | Low | Type-safe routing with clear errors |
| 3.3 | ExecutionGraphProtocol | Low | Contract for graph implementations |
| 3.4 | SinkLike Protocol | Low | Contract for bulk sink operations |
| 3.5 | Add db to ExplainApp | Low | ExplainScreen created in __init__ |
| 3.6 | Wire ExplainScreen | Low | Verified existing architecture |
| 3.7 | Use NodeState contracts | Medium | Pattern matching, no new TypedDict |
| 3.8 | Remove .get() calls | **Done in 3.7** | Pattern matching prevents need |
| 3.9 | trace_token_path | Low | Direct dict access, crash on missing |
| 3.10 | Fix tokens field | Medium | Load real tokens from DB |
| 3.11 | CLI wiring | Medium | Full explain command implementation |

### Key Architectural Decisions

1. **Use existing contracts** - NodeState discriminated union instead of new TypedDict
2. **Pattern matching** - Ensures field access matches state type
3. **Direct dict access** - "Our Data" crashes on anomalies, doesn't hide them
4. **ExplainScreen in __init__** - State management separated from widget creation
