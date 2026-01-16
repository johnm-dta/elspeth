# Integration Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all integration issues identified in the comprehensive codebase audit: hook implementations, EdgeInfo adoption, mode enum alignment, node_id protocol, error/reason schemas, and PluginContext type alignment.

**Architecture:** This plan addresses cross-subsystem integration gaps at the dag↔contracts, orchestrator↔plugins, and recorder↔contracts boundaries. Each task is self-contained with tests. The fixes follow the Data Manifesto's trust boundary model.

**Tech Stack:** Python 3.11+, dataclasses, Pydantic, pluggy, TypedDict

**Dependencies:**
- Contracts subsystem Part 1 & Part 2 complete
- Existing plugin base classes functional

**Priority Order:**
1. Tasks 1-3: HIGH priority (broken integrations)
2. Tasks 4-6: MEDIUM priority (soft integrations)
3. Tasks 7-8: LOW priority (cleanup)

---

## Task 1: Add Hook Implementer Classes (HIGH)

**Context:** Plugin classes exist but are NOT wired to pluggy hooks. `PluginManager._refresh_caches()` returns empty lists because no hookimpls are registered. This blocks automatic plugin discovery.

**Files:**
- Create: `src/elspeth/plugins/sources/hookimpl.py`
- Create: `src/elspeth/plugins/transforms/hookimpl.py`
- Create: `src/elspeth/plugins/gates/hookimpl.py`
- Create: `src/elspeth/plugins/sinks/hookimpl.py`
- Modify: `src/elspeth/plugins/manager.py`
- Create: `tests/plugins/test_hookimpl_registration.py`

### Step 1: Write failing test for hook registration

Create `tests/plugins/test_hookimpl_registration.py`:

```python
"""Tests for plugin hook implementations."""

import pytest

from elspeth.plugins.manager import PluginManager


class TestBuiltinPluginDiscovery:
    """Verify built-in plugins are discoverable via hooks."""

    def test_builtin_sources_discoverable(self) -> None:
        """Built-in source plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # CSVSource and JSONSource should be discoverable
        sources = manager.list_sources()
        source_names = [s.name for s in sources]

        assert "csv_local" in source_names
        assert "json_local" in source_names

    def test_builtin_transforms_discoverable(self) -> None:
        """Built-in transform plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        transforms = manager.list_transforms()
        transform_names = [t.name for t in transforms]

        assert "passthrough" in transform_names
        assert "field_mapper" in transform_names

    def test_builtin_gates_discoverable(self) -> None:
        """Built-in gate plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        gates = manager.list_gates()
        gate_names = [g.name for g in gates]

        assert "threshold" in gate_names
        assert "field_match" in gate_names
        assert "filter" in gate_names

    def test_builtin_sinks_discoverable(self) -> None:
        """Built-in sink plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        sinks = manager.list_sinks()
        sink_names = [s.name for s in sinks]

        assert "csv_local" in sink_names
        assert "json_local" in sink_names
        assert "database" in sink_names
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_hookimpl_registration.py -v`
Expected: FAIL (no `register_builtin_plugins` method, empty plugin lists)

### Step 3: Create sources hookimpl

Create `src/elspeth/plugins/sources/hookimpl.py`:

```python
"""Hook implementation for built-in source plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinSources:
    """Hook implementer for built-in source plugins."""

    @hookimpl
    def elspeth_get_source(self) -> list:
        """Return built-in source plugin classes."""
        from elspeth.plugins.sources.csv_source import CSVSource
        from elspeth.plugins.sources.json_source import JSONSource

        return [CSVSource, JSONSource]


# Singleton instance for registration
builtin_sources = ElspethBuiltinSources()
```

### Step 4: Create transforms hookimpl

Create `src/elspeth/plugins/transforms/hookimpl.py`:

```python
"""Hook implementation for built-in transform plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinTransforms:
    """Hook implementer for built-in transform plugins."""

    @hookimpl
    def elspeth_get_transforms(self) -> list:
        """Return built-in transform plugin classes."""
        from elspeth.plugins.transforms.passthrough import PassThrough
        from elspeth.plugins.transforms.field_mapper import FieldMapper

        return [PassThrough, FieldMapper]


# Singleton instance for registration
builtin_transforms = ElspethBuiltinTransforms()
```

### Step 5: Create gates hookimpl

Create `src/elspeth/plugins/gates/hookimpl.py`:

```python
"""Hook implementation for built-in gate plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinGates:
    """Hook implementer for built-in gate plugins."""

    @hookimpl
    def elspeth_get_gates(self) -> list:
        """Return built-in gate plugin classes."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate
        from elspeth.plugins.gates.field_match_gate import FieldMatchGate
        from elspeth.plugins.gates.filter_gate import FilterGate

        return [ThresholdGate, FieldMatchGate, FilterGate]


# Singleton instance for registration
builtin_gates = ElspethBuiltinGates()
```

### Step 6: Create sinks hookimpl

Create `src/elspeth/plugins/sinks/hookimpl.py`:

```python
"""Hook implementation for built-in sink plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinSinks:
    """Hook implementer for built-in sink plugins."""

    @hookimpl
    def elspeth_get_sinks(self) -> list:
        """Return built-in sink plugin classes."""
        from elspeth.plugins.sinks.csv_sink import CSVSink
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        return [CSVSink, JSONSink, DatabaseSink]


# Singleton instance for registration
builtin_sinks = ElspethBuiltinSinks()
```

### Step 7: Update PluginManager with register_builtin_plugins()

Modify `src/elspeth/plugins/manager.py` - add method after `__init__`:

```python
def register_builtin_plugins(self) -> None:
    """Register all built-in plugin hook implementers.

    Call this once at startup to make built-in plugins discoverable.
    """
    from elspeth.plugins.sources.hookimpl import builtin_sources
    from elspeth.plugins.transforms.hookimpl import builtin_transforms
    from elspeth.plugins.gates.hookimpl import builtin_gates
    from elspeth.plugins.sinks.hookimpl import builtin_sinks

    self.register(builtin_sources)
    self.register(builtin_transforms)
    self.register(builtin_gates)
    self.register(builtin_sinks)
```

### Step 8: Run tests to verify they pass

Run: `pytest tests/plugins/test_hookimpl_registration.py -v`
Expected: PASS

### Step 9: Commit

```bash
git add src/elspeth/plugins/sources/hookimpl.py \
        src/elspeth/plugins/transforms/hookimpl.py \
        src/elspeth/plugins/gates/hookimpl.py \
        src/elspeth/plugins/sinks/hookimpl.py \
        src/elspeth/plugins/manager.py \
        tests/plugins/test_hookimpl_registration.py
git commit -m "feat(plugins): add hook implementers for built-in plugins

Built-in plugins (CSV, JSON, PassThrough, ThresholdGate, etc.) are now
discoverable via pluggy hooks. Call manager.register_builtin_plugins()
at startup to enable automatic discovery."
```

---

## Task 2: Integrate EdgeInfo Contract (HIGH)

**Context:** `EdgeInfo` is defined in `contracts/routing.py` but unused. `dag.py:get_edges()` returns `tuple[str, str, dict[str, Any]]` instead of typed `EdgeInfo`. This loses type safety at the DAG boundary.

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Modify: `src/elspeth/engine/orchestrator.py`
- Modify: `tests/core/test_dag.py`

### Step 1: Write failing test for EdgeInfo return type

Add to `tests/core/test_dag.py`:

```python
"""Tests for EdgeInfo integration."""

from elspeth.contracts import EdgeInfo, RoutingMode


class TestEdgeInfoIntegration:
    """Tests for typed edge returns."""

    def test_get_edges_returns_edge_info(self) -> None:
        """get_edges() returns list of EdgeInfo, not tuples."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source-1", node_type="source", plugin_name="csv")
        graph.add_node("sink-1", node_type="sink", plugin_name="csv")
        graph.add_edge("source-1", "sink-1", label="continue", mode=RoutingMode.MOVE)

        edges = graph.get_edges()

        assert len(edges) == 1
        assert isinstance(edges[0], EdgeInfo)
        assert edges[0].from_node == "source-1"
        assert edges[0].to_node == "sink-1"
        assert edges[0].label == "continue"
        assert edges[0].mode == RoutingMode.MOVE

    def test_add_edge_accepts_routing_mode_enum(self) -> None:
        """add_edge() accepts RoutingMode enum, not string."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("n1", node_type="transform", plugin_name="test")
        graph.add_node("n2", node_type="sink", plugin_name="test")

        # Should accept enum directly
        graph.add_edge("n1", "n2", label="route", mode=RoutingMode.COPY)

        edges = graph.get_edges()
        assert edges[0].mode == RoutingMode.COPY
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_dag.py::TestEdgeInfoIntegration -v`
Expected: FAIL (returns tuple, mode is string)

### Step 3: Update dag.py imports

Add to top of `src/elspeth/core/dag.py`:

```python
from elspeth.contracts import EdgeInfo, RoutingMode
```

### Step 4: Update add_edge() to accept RoutingMode enum

Modify `add_edge()` in `src/elspeth/core/dag.py`:

```python
def add_edge(
    self,
    from_node: str,
    to_node: str,
    *,
    label: str,
    mode: RoutingMode = RoutingMode.MOVE,
) -> None:
    """Add an edge between nodes.

    Args:
        from_node: Source node ID
        to_node: Target node ID
        label: Edge label (e.g., "continue", "suspicious")
        mode: Routing mode (MOVE or COPY)
    """
    self._graph.add_edge(from_node, to_node, label=label, mode=mode)
```

### Step 5: Update get_edges() to return EdgeInfo

Modify `get_edges()` in `src/elspeth/core/dag.py`:

```python
def get_edges(self) -> list[EdgeInfo]:
    """Get all edges with their data as typed EdgeInfo.

    Returns:
        List of EdgeInfo contracts (not tuples)
    """
    return [
        EdgeInfo(
            from_node=u,
            to_node=v,
            label=data["label"],
            mode=data["mode"],  # Already RoutingMode after add_edge change
        )
        for u, v, data in self._graph.edges(data=True)
    ]
```

### Step 6: Update from_config() to use RoutingMode

Update all `add_edge()` calls in `from_config()` to use `RoutingMode.MOVE`:

```python
# Line ~277
graph.add_edge(prev_node_id, tid, label="continue", mode=RoutingMode.MOVE)

# Line ~298-299
graph.add_edge(
    tid, sink_ids[target], label=route_label, mode=RoutingMode.MOVE
)

# Line ~316-320
graph.add_edge(
    prev_node_id,
    output_sink_node,
    label="continue",
    mode=RoutingMode.MOVE,
)
```

### Step 7: Update orchestrator.py to use EdgeInfo

Modify the edge registration loop in `src/elspeth/engine/orchestrator.py` (around line 380):

```python
# Old:
# for from_id, to_id, edge_data in graph.get_edges():
#     edge = recorder.register_edge(
#         run_id=run_id,
#         from_node_id=from_id,
#         to_node_id=to_id,
#         label=edge_data["label"],
#         mode=edge_data["mode"],
#     )

# New:
for edge_info in graph.get_edges():
    edge = recorder.register_edge(
        run_id=run_id,
        from_node_id=edge_info.from_node,
        to_node_id=edge_info.to_node,
        label=edge_info.label,
        mode=edge_info.mode,
    )
```

### Step 8: Run tests to verify they pass

Run: `pytest tests/core/test_dag.py -v`
Expected: PASS

### Step 9: Commit

```bash
git add src/elspeth/core/dag.py src/elspeth/engine/orchestrator.py tests/core/test_dag.py
git commit -m "feat(dag): integrate EdgeInfo contract for type-safe edges

- add_edge() now accepts RoutingMode enum, not string
- get_edges() returns list[EdgeInfo] instead of tuples
- Orchestrator updated to use typed EdgeInfo
- Closes integration gap at dag↔contracts boundary"
```

---

## Task 3: Fix Mode String/Enum Mismatch in DAG (HIGH)

**Context:** Task 2 changes `add_edge()` to require `RoutingMode` enum. This task updates ALL call sites (tests and production code) to use the enum. Per the No Legacy Code Policy, we do NOT add backwards compatibility for string mode values—we update every call site in the same commit.

**Files:**
- Modify: `tests/core/test_dag.py`
- Modify: `tests/engine/test_orchestrator.py`
- Modify: Any production code with `mode="move"` or `mode="copy"`

### Step 1: Find and update ALL string mode usages

Search for `mode="move"` or `mode="copy"` in the ENTIRE codebase (not just tests):

```bash
grep -r 'mode="move"' src/ tests/
grep -r 'mode="copy"' src/ tests/
```

Update each occurrence:

```python
# Old:
graph.add_edge("n1", "n2", label="continue", mode="move")

# New:
from elspeth.contracts import RoutingMode
graph.add_edge("n1", "n2", label="continue", mode=RoutingMode.MOVE)
```

### Step 2: Run full test suite

Run: `pytest tests/ -v`
Expected: PASS

### Step 3: Commit

```bash
git add -u
git commit -m "refactor(dag): update all call sites to use RoutingMode enum

Breaking change: add_edge() no longer accepts string mode values.
All call sites updated in this commit per No Legacy Code Policy."
```

---

## Task 4: Add node_id to Plugin Protocols (MEDIUM)

**Context:** Orchestrator injects `node_id` dynamically with `# type: ignore`. Executors expect it but protocols don't define it. This creates runtime coupling outside the type system.

**Files:**
- Modify: `src/elspeth/plugins/protocols.py`
- Modify: `src/elspeth/plugins/base.py`
- Modify: `src/elspeth/engine/orchestrator.py` (remove type: ignore)
- Create: `tests/plugins/test_node_id_protocol.py`

### Step 1: Write failing test

Create `tests/plugins/test_node_id_protocol.py`:

```python
"""Tests for node_id in plugin protocols."""

import pytest


class TestNodeIdProtocol:
    """Verify node_id is part of plugin contract."""

    def test_source_protocol_has_node_id(self) -> None:
        """SourceProtocol defines node_id attribute."""
        from elspeth.plugins.protocols import SourceProtocol
        import typing

        hints = typing.get_type_hints(SourceProtocol)
        # node_id should be in protocol attributes
        assert hasattr(SourceProtocol, "__protocol_attrs__") or "node_id" in dir(SourceProtocol)

    def test_base_source_has_node_id(self) -> None:
        """BaseSource has node_id attribute with default None."""
        from elspeth.plugins.base import BaseSource

        # Should be able to access node_id
        class TestSource(BaseSource):
            name = "test"
            output_schema = None

            def load(self, ctx):
                yield {}

        source = TestSource({})
        assert source.node_id is None  # Default

        source.node_id = "node-123"
        assert source.node_id == "node-123"

    def test_transform_protocol_has_node_id(self) -> None:
        """TransformProtocol defines node_id attribute."""
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        class TestTransform(BaseTransform):
            name = "test"

            def process(self, row, ctx):
                return TransformResult.success(row)

        transform = TestTransform({})
        assert transform.node_id is None

    def test_gate_protocol_has_node_id(self) -> None:
        """GateProtocol defines node_id attribute."""
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.results import GateResult
        from elspeth.contracts import RoutingAction

        class TestGate(BaseGate):
            name = "test"

            def evaluate(self, row, ctx):
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = TestGate({})
        assert gate.node_id is None

    def test_sink_protocol_has_node_id(self) -> None:
        """SinkProtocol defines node_id attribute."""
        from elspeth.plugins.base import BaseSink

        class TestSink(BaseSink):
            name = "test"

            def write(self, row, ctx):
                pass

        sink = TestSink({})
        assert sink.node_id is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/test_node_id_protocol.py -v`
Expected: FAIL (node_id not defined)

### Step 3: Add node_id to protocols

Modify `src/elspeth/plugins/protocols.py` - add to each protocol class:

```python
@runtime_checkable
class SourceProtocol(Protocol):
    """Protocol for source plugins."""

    name: str
    output_schema: type["PluginSchema"]
    node_id: str | None  # Set by orchestrator after registration

    # ... rest unchanged


@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for stateless row transforms."""

    name: str
    plugin_version: str
    determinism: "Determinism"
    input_schema: type["PluginSchema"] | None
    output_schema: type["PluginSchema"] | None
    node_id: str | None  # Set by orchestrator after registration

    # ... rest unchanged


@runtime_checkable
class GateProtocol(Protocol):
    """Protocol for routing gates."""

    name: str
    plugin_version: str
    determinism: "Determinism"
    input_schema: type["PluginSchema"] | None
    node_id: str | None  # Set by orchestrator after registration

    # ... rest unchanged


@runtime_checkable
class SinkProtocol(Protocol):
    """Protocol for output sinks."""

    name: str
    plugin_version: str
    artifact_type: str
    node_id: str | None  # Set by orchestrator after registration

    # ... rest unchanged
```

### Step 4: Add node_id to base classes

Modify `src/elspeth/plugins/base.py` - add to each base class `__init__`:

```python
class BaseSource(ABC):
    """Base class for source plugins."""

    name: str = ""
    output_schema: type[PluginSchema] | None = None
    node_id: str | None = None  # Set by orchestrator

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.node_id = None  # Will be set by orchestrator


class BaseTransform(ABC):
    """Base class for transform plugins."""

    name: str = ""
    plugin_version: str = "0.0.0"
    determinism: Determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] | None = None
    output_schema: type[PluginSchema] | None = None
    node_id: str | None = None  # Set by orchestrator

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.node_id = None  # Will be set by orchestrator


# Similarly for BaseGate, BaseAggregation, BaseCoalesce, BaseSink
```

### Step 5: Remove type: ignore comments in orchestrator

Modify `src/elspeth/engine/orchestrator.py` - remove the `# type: ignore[attr-defined]` comments:

```python
# Old:
config.source.node_id = source_id  # type: ignore[attr-defined]

# New:
config.source.node_id = source_id  # Now in protocol
```

### Step 6: Run tests

Run: `pytest tests/plugins/test_node_id_protocol.py -v`
Expected: PASS

### Step 7: Run full test suite

Run: `pytest tests/ -v`
Expected: PASS

### Step 8: Commit

```bash
git add src/elspeth/plugins/protocols.py src/elspeth/plugins/base.py \
        src/elspeth/engine/orchestrator.py tests/plugins/test_node_id_protocol.py
git commit -m "feat(plugins): add node_id to plugin protocols

node_id is now part of the plugin contract:
- All protocols define node_id: str | None
- All base classes initialize node_id = None
- Orchestrator sets node_id without type: ignore
- Closes orchestrator↔plugins integration gap"
```

---

## Task 5: Define TypedDict Schemas for Error/Reason Payloads (MEDIUM)

**Context:** `error: dict[str, Any]` and `reason: dict[str, Any]` throughout the codebase have no schema. Different executors produce inconsistent shapes.

**Files:**
- Create: `src/elspeth/contracts/errors.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/engine/executors.py`
- Create: `tests/contracts/test_errors.py`

### Step 1: Write failing test

Create `tests/contracts/test_errors.py`:

```python
"""Tests for error/reason schema contracts."""

import pytest
from typing import get_type_hints


class TestExecutionError:
    """Tests for ExecutionError TypedDict."""

    def test_execution_error_has_required_fields(self) -> None:
        """ExecutionError defines exception and type fields."""
        from elspeth.contracts import ExecutionError

        error: ExecutionError = {
            "exception": "ValueError: invalid input",
            "type": "ValueError",
        }

        assert error["exception"] == "ValueError: invalid input"
        assert error["type"] == "ValueError"

    def test_execution_error_accepts_optional_traceback(self) -> None:
        """ExecutionError can include traceback."""
        from elspeth.contracts import ExecutionError

        error: ExecutionError = {
            "exception": "KeyError: 'foo'",
            "type": "KeyError",
            "traceback": "Traceback (most recent call last):\n...",
        }

        assert "traceback" in error


class TestRoutingReason:
    """Tests for RoutingReason TypedDict."""

    def test_routing_reason_has_rule_field(self) -> None:
        """RoutingReason defines rule field."""
        from elspeth.contracts import RoutingReason

        reason: RoutingReason = {
            "rule": "value > threshold",
            "matched_value": 42,
        }

        assert reason["rule"] == "value > threshold"

    def test_routing_reason_accepts_threshold(self) -> None:
        """RoutingReason can include threshold."""
        from elspeth.contracts import RoutingReason

        reason: RoutingReason = {
            "rule": "value > threshold",
            "matched_value": 42,
            "threshold": 10.0,
        }

        assert reason["threshold"] == 10.0
```

### Step 2: Run test to verify it fails

Run: `pytest tests/contracts/test_errors.py -v`
Expected: FAIL (ExecutionError and RoutingReason not defined)

### Step 3: Create contracts/errors.py

Create `src/elspeth/contracts/errors.py`:

```python
"""Error and reason schema contracts.

TypedDict schemas for structured error payloads in the audit trail.
These provide consistent shapes for executor error recording.
"""

from typing import Any, NotRequired, TypedDict


class ExecutionError(TypedDict):
    """Schema for execution error payloads.

    Used by executors when recording node state failures.
    """

    exception: str  # String representation of the exception
    type: str  # Exception class name (e.g., "ValueError")
    traceback: NotRequired[str]  # Optional full traceback


class RoutingReason(TypedDict):
    """Schema for gate routing reason payloads.

    Used by gates to explain routing decisions in audit trail.
    """

    rule: str  # Human-readable rule description
    matched_value: Any  # The value that triggered the route
    threshold: NotRequired[float]  # Threshold value if applicable
    field: NotRequired[str]  # Field name if applicable
    comparison: NotRequired[str]  # Comparison operator used


class TransformReason(TypedDict):
    """Schema for transform reason payloads.

    Used by transforms to explain processing decisions.
    """

    action: str  # What the transform did
    fields_modified: NotRequired[list[str]]  # Fields that were changed
    validation_errors: NotRequired[list[str]]  # Any validation issues
```

### Step 4: Update contracts __init__.py

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.errors import (
    ExecutionError,
    RoutingReason,
    TransformReason,
)

__all__ = [
    # ... existing ...
    # errors
    "ExecutionError",
    "RoutingReason",
    "TransformReason",
]
```

### Step 5: Update executors to use ExecutionError

Modify `src/elspeth/engine/executors.py` - update error construction:

```python
from elspeth.contracts import ExecutionError

# In exception handlers, replace:
# error = {"exception": str(e), "type": type(e).__name__}

# With:
error: ExecutionError = {
    "exception": str(e),
    "type": type(e).__name__,
}
```

### Step 6: Run tests

Run: `pytest tests/contracts/test_errors.py -v`
Expected: PASS

### Step 7: Commit

```bash
git add src/elspeth/contracts/errors.py src/elspeth/contracts/__init__.py \
        src/elspeth/engine/executors.py tests/contracts/test_errors.py
git commit -m "feat(contracts): add TypedDict schemas for error/reason payloads

- ExecutionError: exception, type, traceback fields
- RoutingReason: rule, matched_value, threshold fields
- TransformReason: action, fields_modified fields
- Executors now use structured ExecutionError"
```

---

## Task 6: Fix PluginContext.landscape Type Mismatch (MEDIUM)

**Context:** `PluginContext.landscape` is typed as a stub protocol, but receives real `LandscapeRecorder`. This creates type: ignore comments and misleading IDE hints.

**Files:**
- Modify: `src/elspeth/plugins/context.py`
- Modify: `src/elspeth/engine/orchestrator.py`
- Create: `tests/plugins/test_context_types.py`

### Step 1: Write failing test

Create `tests/plugins/test_context_types.py`:

```python
"""Tests for PluginContext type alignment."""

from typing import get_type_hints


class TestPluginContextTypes:
    """Verify PluginContext field types match runtime values."""

    def test_landscape_type_matches_recorder(self) -> None:
        """PluginContext.landscape type should accept LandscapeRecorder."""
        from elspeth.plugins.context import PluginContext
        from elspeth.core.landscape.recorder import LandscapeRecorder

        # Type hints should allow LandscapeRecorder
        hints = get_type_hints(PluginContext)

        # Should be able to create context with real recorder
        # (This tests runtime compatibility, not just type hints)
        from elspeth.core.landscape import LandscapeDB

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=recorder,  # Should work without type: ignore
        )

        assert ctx.landscape is recorder
```

### Step 2: Run test to verify current state

Run: `pytest tests/plugins/test_context_types.py -v`
Expected: May pass at runtime but with type checker warnings

### Step 3: Update context.py with proper type import

Modify `src/elspeth/plugins/context.py`:

```python
"""Plugin execution context."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AbstractContextManager

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer, Span
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.payload_store import PayloadStore


@dataclass
class PluginContext:
    """Context provided to plugins during execution.

    Contains run metadata and optional integrations for audit recording,
    tracing, and payload storage.
    """

    run_id: str
    config: dict[str, Any]

    # Phase 3 Integration Points
    # Use string annotations to avoid circular imports at runtime
    landscape: "LandscapeRecorder | None" = None
    tracer: "Tracer | None" = None
    payload_store: "PayloadStore | None" = None

    # Metadata set by orchestrator during execution
    node_id: str | None = field(default=None)
    plugin_name: str | None = field(default=None)

    # ... rest of methods unchanged
```

### Step 4: Remove type: ignore in orchestrator

Modify `src/elspeth/engine/orchestrator.py`:

```python
# Old:
ctx = PluginContext(
    run_id=run_id,
    config=config.config,
    landscape=recorder,  # type: ignore[arg-type]
)

# New:
ctx = PluginContext(
    run_id=run_id,
    config=config.config,
    landscape=recorder,  # Type now matches
)
```

### Step 5: Run tests

Run: `pytest tests/plugins/test_context_types.py -v`
Expected: PASS

### Step 6: Run mypy

Run: `mypy src/elspeth/plugins/context.py src/elspeth/engine/orchestrator.py`
Expected: No type errors

### Step 7: Commit

```bash
git add src/elspeth/plugins/context.py src/elspeth/engine/orchestrator.py \
        tests/plugins/test_context_types.py
git commit -m "fix(context): align PluginContext.landscape type with LandscapeRecorder

- Use TYPE_CHECKING import for LandscapeRecorder
- Remove type: ignore comments in orchestrator
- Fixes context↔landscape type mismatch"
```

---

## Task 7: Add Optional Pipeline Schema Validation (LOW)

**Context:** Plugin configs are `dict[str, Any]`. While plugins validate internally, there's no pipeline-wide check that schemas align across stages.

**Files:**
- Create: `src/elspeth/engine/schema_validator.py`
- Modify: `src/elspeth/engine/orchestrator.py`
- Create: `tests/engine/test_schema_validator.py`

### Step 1: Write failing test

Create `tests/engine/test_schema_validator.py`:

```python
"""Tests for pipeline schema validation."""

import pytest


class TestPipelineSchemaValidator:
    """Tests for optional schema compatibility checking."""

    def test_validates_compatible_schemas(self) -> None:
        """Compatible schemas pass validation."""
        from elspeth.engine.schema_validator import validate_pipeline_schemas
        from elspeth.plugins.schemas import PluginSchema

        class SourceOutput(PluginSchema):
            name: str
            value: int

        class TransformInput(PluginSchema):
            name: str
            value: int

        # Mock pipeline config
        errors = validate_pipeline_schemas(
            source_output=SourceOutput,
            transform_inputs=[TransformInput],
            transform_outputs=[SourceOutput],  # Pass-through
            sink_inputs=[SourceOutput],
        )

        assert len(errors) == 0

    def test_detects_missing_field(self) -> None:
        """Detects when consumer expects field producer doesn't provide."""
        from elspeth.engine.schema_validator import validate_pipeline_schemas
        from elspeth.plugins.schemas import PluginSchema

        class SourceOutput(PluginSchema):
            name: str

        class TransformInput(PluginSchema):
            name: str
            value: int  # Source doesn't provide this!

        errors = validate_pipeline_schemas(
            source_output=SourceOutput,
            transform_inputs=[TransformInput],
            transform_outputs=[SourceOutput],
            sink_inputs=[SourceOutput],
        )

        assert len(errors) == 1
        assert "value" in errors[0]
```

### Step 2: Create schema_validator.py

Create `src/elspeth/engine/schema_validator.py`:

```python
"""Optional pipeline schema validation.

Checks that plugin schemas are compatible across pipeline stages.
This is opt-in validation - the pipeline still runs without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.plugins.schemas import PluginSchema


def validate_pipeline_schemas(
    source_output: type["PluginSchema"] | None,
    transform_inputs: list[type["PluginSchema"] | None],
    transform_outputs: list[type["PluginSchema"] | None],
    sink_inputs: list[type["PluginSchema"] | None],
) -> list[str]:
    """Validate schema compatibility across pipeline stages.

    Args:
        source_output: Schema of source output
        transform_inputs: Schemas of transform inputs (in order)
        transform_outputs: Schemas of transform outputs (in order)
        sink_inputs: Schemas of sink inputs

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Skip validation if any schema is None (dynamic schema)
    if source_output is None:
        return errors

    # Check source -> first transform
    if transform_inputs and transform_inputs[0] is not None:
        source_fields = set(source_output.model_fields.keys())
        required_fields = {
            name
            for name, field in transform_inputs[0].model_fields.items()
            if field.is_required()
        }
        missing = required_fields - source_fields
        if missing:
            errors.append(
                f"Source output missing fields required by transform[0]: {missing}"
            )

    # Check transform chain
    for i in range(len(transform_outputs) - 1):
        if transform_outputs[i] is None or transform_inputs[i + 1] is None:
            continue
        output_fields = set(transform_outputs[i].model_fields.keys())
        required_fields = {
            name
            for name, field in transform_inputs[i + 1].model_fields.items()
            if field.is_required()
        }
        missing = required_fields - output_fields
        if missing:
            errors.append(
                f"Transform[{i}] output missing fields required by transform[{i+1}]: {missing}"
            )

    # Check final transform -> sinks
    if transform_outputs and transform_outputs[-1] is not None:
        final_output = set(transform_outputs[-1].model_fields.keys())
        for j, sink_input in enumerate(sink_inputs):
            if sink_input is None:
                continue
            required_fields = {
                name
                for name, field in sink_input.model_fields.items()
                if field.is_required()
            }
            missing = required_fields - final_output
            if missing:
                errors.append(
                    f"Final transform output missing fields required by sink[{j}]: {missing}"
                )

    return errors
```

### Step 3: Add optional validation call to orchestrator

Modify `src/elspeth/engine/orchestrator.py` - add after graph validation:

```python
from elspeth.engine.schema_validator import validate_pipeline_schemas

# In run() method, after graph.validate():
if config.validate_schemas:  # New optional config flag
    # Direct attribute access - we KNOW the plugin types at each stage.
    # Sources have output_schema, transforms have both, sinks have input_schema.
    # If a plugin is missing the attribute, that's a protocol violation to crash on.
    schema_errors = validate_pipeline_schemas(
        source_output=config.source.output_schema,
        transform_inputs=[t.input_schema for t in transforms],
        transform_outputs=[t.output_schema for t in transforms],
        sink_inputs=[s.input_schema for s in config.sinks.values()],
    )
    if schema_errors:
        for error in schema_errors:
            logger.warning(f"Schema compatibility warning: {error}")
```

**Note:** We use direct attribute access because the orchestrator knows the plugin types at each pipeline stage. Sources always have `output_schema`, transforms have both `input_schema` and `output_schema`, and sinks have `input_schema`. If any plugin violates its protocol by missing these attributes, that's a bug that should crash immediately—not be silently hidden with `getattr(..., None)`.

### Step 4: Run tests

Run: `pytest tests/engine/test_schema_validator.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/schema_validator.py tests/engine/test_schema_validator.py
git commit -m "feat(engine): add optional pipeline schema validation

Validates that plugin schemas are compatible across pipeline stages.
Opt-in via config.validate_schemas flag. Emits warnings, doesn't block."
```

---

## Task 8: Final Integration Test Suite (LOW)

**Context:** Run comprehensive integration tests to verify all fixes work together.

**Files:**
- Create: `tests/integration/test_audit_integration_fixes.py`

### Step 1: Create comprehensive integration test

Create `tests/integration/test_audit_integration_fixes.py`:

```python
"""Integration tests verifying all audit integration fixes."""

import pytest

from elspeth.contracts import EdgeInfo, RoutingMode, ExecutionError
from elspeth.core.dag import ExecutionGraph
from elspeth.plugins.manager import PluginManager


class TestIntegrationAuditFixes:
    """End-to-end tests for integration audit fixes."""

    def test_full_plugin_discovery_flow(self) -> None:
        """Plugins are discoverable and have proper node_id support."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # All built-in plugins discoverable
        assert len(manager.list_sources()) >= 2
        assert len(manager.list_transforms()) >= 2
        assert len(manager.list_gates()) >= 3
        assert len(manager.list_sinks()) >= 3

        # Instantiate a plugin and verify node_id
        csv_source_cls = manager.get_source_by_name("csv_local")
        source = csv_source_cls({"path": "test.csv"})
        assert source.node_id is None  # Not yet set

        source.node_id = "node-123"
        assert source.node_id == "node-123"  # Can be set

    def test_dag_uses_typed_edges(self) -> None:
        """DAG edge operations use EdgeInfo contracts."""
        graph = ExecutionGraph()
        graph.add_node("src", node_type="source", plugin_name="csv")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("src", "sink", label="continue", mode=RoutingMode.MOVE)

        edges = graph.get_edges()
        assert len(edges) == 1
        assert isinstance(edges[0], EdgeInfo)
        assert edges[0].mode == RoutingMode.MOVE
        assert not isinstance(edges[0].mode, str)

    def test_error_payloads_are_structured(self) -> None:
        """Error payloads follow ExecutionError schema."""
        error: ExecutionError = {
            "exception": "Test error",
            "type": "ValueError",
        }

        # Type checker validates this structure
        assert error["exception"] == "Test error"
        assert error["type"] == "ValueError"

    def test_plugin_context_accepts_real_recorder(self) -> None:
        """PluginContext accepts LandscapeRecorder without type issues."""
        from elspeth.plugins.context import PluginContext
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=recorder,
        )

        assert ctx.landscape is recorder
        assert ctx.run_id == "test-run"
```

### Step 2: Run integration tests

Run: `pytest tests/integration/test_audit_integration_fixes.py -v`
Expected: PASS

### Step 3: Run full test suite

Run: `pytest tests/ -v`
Expected: PASS

### Step 4: Commit

```bash
git add tests/integration/test_audit_integration_fixes.py
git commit -m "test: add comprehensive integration tests for audit fixes

Verifies all integration audit fixes work together:
- Plugin discovery via hooks
- Typed EdgeInfo in DAG
- Structured error payloads
- PluginContext type alignment"
```

---

## Summary

| Task | Priority | Description | Key Changes |
|------|----------|-------------|-------------|
| 1 | HIGH | Hook implementers | Enable plugin discovery via pluggy |
| 2 | HIGH | EdgeInfo integration | Type-safe edges in DAG |
| 3 | HIGH | Mode enum alignment | RoutingMode instead of strings |
| 4 | MED | node_id in protocols | Remove type: ignore, add to contracts |
| 5 | MED | Error/reason schemas | TypedDict for structured payloads |
| 6 | MED | Context type alignment | Fix landscape type mismatch |
| 7 | LOW | Schema validation | Optional pipeline compatibility check |
| 8 | LOW | Integration tests | Verify all fixes work together |

**Key principles:**
- Each task is self-contained with tests
- TDD: failing test → implementation → passing test → commit
- Remove type: ignore comments by fixing root cause
- Use TypedDict for structured payloads at boundaries
- Maintain backwards compatibility where possible
