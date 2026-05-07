# Web UX Sub-Spec 4: Composer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ComposerService module: frozen immutable data models for pipeline composition, LLM tool-use loop with discovery and mutation tools, Stage 1 validation, deterministic YAML generation, and API wiring. After this plan, `POST /api/sessions/{id}/messages` triggers an LLM-driven pipeline composition loop that mutates a `CompositionState` and returns the result.

**Architecture:** `CompositionState` is a frozen dataclass with `freeze_fields()` on all container fields. Mutation methods return new instances via `dataclasses.replace()` with manual freeze. `to_dict()` recursively unwraps frozen containers (`MappingProxyType` -> `dict`, `tuple` -> `list`) for YAML/JSON serialization. Six discovery tools delegate to `CatalogService`; six mutation tools validate and return `ToolResult`. `ComposerServiceImpl` runs a bounded LiteLLM tool-use loop. `generate_yaml()` calls `state.to_dict()` before `yaml.dump()` to produce deterministic ELSPETH pipeline YAML. Route handlers catch `ComposerConvergenceError` (422) and LLM client errors (502) with structured error bodies.

**Tech Stack:** Python dataclasses, `freeze_fields()`, LiteLLM, PyYAML, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-03-28-web-ux-sub4-composer-design.md`

**Depends on:** Sub-Specs 2 (Auth & Sessions), 3 (Catalog — `CatalogService` protocol).
**Blocks:** Sub-Specs 5 (Execution), 6 (Frontend).

---

### Task 1: Data Models — SourceSpec, NodeSpec, EdgeSpec, OutputSpec, PipelineMetadata

**Files:**
- Create: `src/elspeth/web/composer/__init__.py`
- Create: `src/elspeth/web/composer/state.py`
- Create: `tests/unit/web/composer/__init__.py`
- Create: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write tests for all data model frozen dataclasses**

```python
# tests/unit/web/composer/test_state.py
"""Tests for CompositionState and supporting data models."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)


class TestSourceSpec:
    def test_create(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="transform_1",
            options={"path": "/data/input.csv"},
            on_validation_failure="quarantine",
        )
        assert s.plugin == "csv"
        assert s.on_success == "transform_1"
        assert s.options["path"] == "/data/input.csv"

    def test_frozen(self) -> None:
        s = SourceSpec(
            plugin="csv", on_success="t1", options={}, on_validation_failure="discard"
        )
        with pytest.raises(AttributeError):
            s.plugin = "json"  # type: ignore[misc]

    def test_options_deep_frozen(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"key": "val"}},
            on_validation_failure="discard",
        )
        with pytest.raises(TypeError):
            s.options["new"] = "x"  # type: ignore[index]

    def test_options_nested_frozen(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"key": "val"}},
            on_validation_failure="discard",
        )
        with pytest.raises(TypeError):
            s.options["nested"]["mutate"] = "x"  # type: ignore[index]


class TestNodeSpec:
    def _make_transform(self, **overrides: Any) -> NodeSpec:
        defaults: dict[str, Any] = {
            "id": "transform_1",
            "node_type": "transform",
            "plugin": "uppercase",
            "input": "source_out",
            "on_success": "sink_main",
            "on_error": None,
            "options": {"field": "name"},
            "condition": None,
            "routes": None,
            "fork_to": None,
            "branches": None,
            "policy": None,
            "merge": None,
        }
        defaults.update(overrides)
        return NodeSpec(**defaults)

    def _make_gate(self, **overrides: Any) -> NodeSpec:
        defaults: dict[str, Any] = {
            "id": "gate_1",
            "node_type": "gate",
            "plugin": None,
            "input": "source_out",
            "on_success": None,
            "on_error": None,
            "options": {},
            "condition": "row['score'] >= 0.5",
            "routes": {"high": "sink_good", "low": "sink_bad"},
            "fork_to": None,
            "branches": None,
            "policy": None,
            "merge": None,
        }
        defaults.update(overrides)
        return NodeSpec(**defaults)

    def test_create_transform(self) -> None:
        n = self._make_transform()
        assert n.id == "transform_1"
        assert n.node_type == "transform"
        assert n.plugin == "uppercase"

    def test_create_gate(self) -> None:
        n = self._make_gate()
        assert n.condition == "row['score'] >= 0.5"
        assert n.routes["high"] == "sink_good"

    def test_frozen(self) -> None:
        n = self._make_transform()
        with pytest.raises(AttributeError):
            n.id = "new_id"  # type: ignore[misc]

    def test_options_deep_frozen(self) -> None:
        n = self._make_transform(options={"nested": {"k": "v"}})
        with pytest.raises(TypeError):
            n.options["new"] = 1  # type: ignore[index]

    def test_routes_deep_frozen(self) -> None:
        n = self._make_gate()
        with pytest.raises(TypeError):
            n.routes["extra"] = "val"  # type: ignore[index]

    def test_fork_to_is_tuple(self) -> None:
        n = self._make_gate(fork_to=("path_a", "path_b"))
        assert isinstance(n.fork_to, tuple)
        assert n.fork_to == ("path_a", "path_b")

    def test_branches_is_tuple(self) -> None:
        n = NodeSpec(
            id="coal_1",
            node_type="coalesce",
            plugin=None,
            input="join_point",
            on_success="sink_main",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=("path_a", "path_b"),
            policy="require_all",
            merge="nested",
        )
        assert isinstance(n.branches, tuple)


class TestEdgeSpec:
    def test_create(self) -> None:
        e = EdgeSpec(
            id="e1",
            from_node="source",
            to_node="transform_1",
            edge_type="on_success",
            label=None,
        )
        assert e.from_node == "source"
        assert e.to_node == "transform_1"

    def test_frozen(self) -> None:
        e = EdgeSpec(
            id="e1",
            from_node="source",
            to_node="t1",
            edge_type="on_success",
            label=None,
        )
        with pytest.raises(AttributeError):
            e.id = "e2"  # type: ignore[misc]


class TestOutputSpec:
    def test_create(self) -> None:
        o = OutputSpec(
            name="main_output",
            plugin="csv",
            options={"path": "/out.csv"},
            on_write_failure="quarantine",
        )
        assert o.name == "main_output"
        assert o.plugin == "csv"

    def test_frozen(self) -> None:
        o = OutputSpec(
            name="out", plugin="csv", options={}, on_write_failure="discard"
        )
        with pytest.raises(AttributeError):
            o.name = "new"  # type: ignore[misc]

    def test_options_deep_frozen(self) -> None:
        o = OutputSpec(
            name="out",
            plugin="csv",
            options={"nested": {"k": 1}},
            on_write_failure="discard",
        )
        with pytest.raises(TypeError):
            o.options["new"] = 2  # type: ignore[index]


class TestPipelineMetadata:
    def test_defaults(self) -> None:
        m = PipelineMetadata()
        assert m.name == "Untitled Pipeline"
        assert m.description == ""
    def test_custom(self) -> None:
        m = PipelineMetadata(
            name="My Pipeline",
            description="Does things",
        )
        assert m.name == "My Pipeline"

    def test_frozen(self) -> None:
        m = PipelineMetadata()
        with pytest.raises(AttributeError):
            m.name = "new"  # type: ignore[misc]


class TestValidationSummary:
    def test_valid(self) -> None:
        v = ValidationSummary(is_valid=True, errors=())
        assert v.is_valid is True
        assert v.errors == ()

    def test_with_errors(self) -> None:
        v = ValidationSummary(is_valid=False, errors=("No source configured.",))
        assert v.is_valid is False
        assert len(v.errors) == 1
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: `ModuleNotFoundError: No module named 'elspeth.web.composer'`

- [ ] **Step 3: Implement data models**

```python
# src/elspeth/web/composer/__init__.py
"""Composer module — LLM-driven pipeline composition."""

# src/elspeth/web/composer/state.py
"""CompositionState and supporting data models for pipeline composition.

All dataclasses are frozen with slots. Container fields (options, routes,
fork_to, branches) are deep-frozen via freeze_fields() in __post_init__.
Mutation methods return new instances — they never modify the original.

Layer: L3 (application). Imports from L0 (contracts.freeze) only.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class PipelineMetadata:
    """Pipeline-level metadata.

    All fields are scalars or None. frozen=True is sufficient.
    """

    name: str = "Untitled Pipeline"
    description: str = ""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Pipeline source configuration.

    Attributes:
        plugin: Source plugin name (e.g. "csv", "json", "dataverse").
        on_success: Named connection point for the first downstream node.
        options: Plugin-specific configuration (path, schema, etc.).
        on_validation_failure: How to handle rows that fail schema validation.
    """

    plugin: str
    on_success: str
    options: Mapping[str, Any]
    on_validation_failure: str

    def __post_init__(self) -> None:
        freeze_fields(self, "options")


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Transform, gate, aggregation, or coalesce node.

    Attributes:
        id: Unique node identifier within the pipeline.
        node_type: One of "transform", "gate", "aggregation", "coalesce".
        plugin: Plugin name. None for gates and coalesces.
        input: Named connection point this node reads from.
        on_success: Named connection point for successful output. None for gates.
        on_error: Named connection point for error output. None if not diverted.
        options: Plugin-specific configuration.
        condition: Gate expression. None for non-gates.
        routes: Gate route mapping. None for non-gates.
        fork_to: Fork destinations for fork gates. None for non-fork nodes.
        branches: Branch inputs for coalesce nodes. None for non-coalesce nodes.
        policy: Coalesce policy. None for non-coalesce nodes.
        merge: Coalesce merge strategy. None for non-coalesce nodes.
    """

    id: str
    node_type: str
    plugin: str | None
    input: str
    on_success: str | None
    on_error: str | None
    options: Mapping[str, Any]
    condition: str | None
    routes: Mapping[str, str] | None
    fork_to: tuple[str, ...] | None
    branches: tuple[str, ...] | None
    policy: str | None
    merge: str | None

    def __post_init__(self) -> None:
        freeze_fields(self, "options")
        if self.routes is not None:
            freeze_fields(self, "routes")


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Connection between two nodes.

    Attributes:
        id: Unique edge identifier.
        from_node: Source node ID (or "source" for the pipeline source).
        to_node: Destination node ID or sink name.
        edge_type: One of "on_success", "on_error", "route_true", "route_false", "fork".
        label: Display label (e.g. the route key for gate edges).
    """

    id: str
    from_node: str
    to_node: str
    edge_type: str
    label: str | None


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Sink configuration.

    Attributes:
        name: Sink name (used as connection point in edges and routes).
        plugin: Sink plugin name (e.g. "csv", "json", "database").
        options: Plugin-specific configuration.
        on_write_failure: How to handle write failures ("discard" or "quarantine").
    """

    name: str
    plugin: str
    options: Mapping[str, Any]
    on_write_failure: str

    def __post_init__(self) -> None:
        freeze_fields(self, "options")


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Stage 1 validation result.

    errors is a tuple of human-readable strings. frozen=True is sufficient
    since tuples of strings are immutable.
    """

    is_valid: bool
    errors: tuple[str, ...]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/state.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/__init__.py src/elspeth/web/composer/state.py \
  tests/unit/web/composer/__init__.py tests/unit/web/composer/test_state.py
git commit -m "feat(web/composer): add frozen data models — SourceSpec, NodeSpec, EdgeSpec, OutputSpec, PipelineMetadata"
```

---

### Task 2: CompositionState — Immutability, Versioning, Mutation Methods

**Files:**
- Modify: `src/elspeth/web/composer/state.py`
- Modify: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write CompositionState tests**

```python
# tests/unit/web/composer/test_state.py (append to existing file)


class TestCompositionState:
    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _make_source(self) -> SourceSpec:
        return SourceSpec(
            plugin="csv",
            on_success="transform_1",
            options={"path": "/data/in.csv"},
            on_validation_failure="quarantine",
        )

    def _make_node(self, id: str = "transform_1") -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="transform",
            plugin="uppercase",
            input="source_out",
            on_success="sink_main",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_edge(self, id: str = "e1") -> EdgeSpec:
        return EdgeSpec(
            id=id,
            from_node="source",
            to_node="transform_1",
            edge_type="on_success",
            label=None,
        )

    def _make_output(self, name: str = "main_output") -> OutputSpec:
        return OutputSpec(
            name=name,
            plugin="csv",
            options={"path": "/out.csv"},
            on_write_failure="quarantine",
        )

    # --- Immutability ---

    def test_frozen(self) -> None:
        state = self._empty_state()
        with pytest.raises(AttributeError):
            state.version = 2  # type: ignore[misc]

    def test_nodes_tuple_frozen(self) -> None:
        """nodes is a tuple — cannot append."""
        state = self._empty_state()
        assert isinstance(state.nodes, tuple)

    def test_metadata_frozen(self) -> None:
        """metadata is a frozen dataclass — deep freeze via freeze_fields."""
        state = self._empty_state()
        with pytest.raises(AttributeError):
            state.metadata.name = "mutated"  # type: ignore[misc]

    # --- with_source ---

    def test_with_source_returns_new_instance(self) -> None:
        state = self._empty_state()
        src = self._make_source()
        new_state = state.with_source(src)
        assert new_state is not state
        assert new_state.source is src
        assert state.source is None  # original unchanged

    def test_with_source_increments_version(self) -> None:
        state = self._empty_state()
        new_state = state.with_source(self._make_source())
        assert new_state.version == 2

    # --- with_node ---

    def test_with_node_adds(self) -> None:
        state = self._empty_state()
        node = self._make_node()
        new_state = state.with_node(node)
        assert len(new_state.nodes) == 1
        assert new_state.nodes[0].id == "transform_1"
        assert new_state.version == 2

    def test_with_node_replaces_existing(self) -> None:
        state = self._empty_state()
        node1 = self._make_node("t1")
        node2 = self._make_node("t1")  # same ID
        state2 = state.with_node(node1)
        state3 = state2.with_node(node2)
        assert len(state3.nodes) == 1
        assert state3.version == 3

    def test_with_node_preserves_order(self) -> None:
        state = self._empty_state()
        state = state.with_node(self._make_node("a"))
        state = state.with_node(self._make_node("b"))
        state = state.with_node(self._make_node("c"))
        assert [n.id for n in state.nodes] == ["a", "b", "c"]

    # --- without_node ---

    def test_without_node_removes(self) -> None:
        state = self._empty_state().with_node(self._make_node("t1"))
        new_state = state.without_node("t1")
        assert len(new_state.nodes) == 0
        assert new_state.version == 3

    def test_without_node_nonexistent_returns_none(self) -> None:
        state = self._empty_state()
        result = state.without_node("nonexistent")
        assert result is None

    # --- with_edge ---

    def test_with_edge_adds(self) -> None:
        state = self._empty_state()
        edge = self._make_edge()
        new_state = state.with_edge(edge)
        assert len(new_state.edges) == 1
        assert new_state.version == 2

    def test_with_edge_replaces_by_id(self) -> None:
        state = self._empty_state()
        e1 = EdgeSpec(id="e1", from_node="source", to_node="t1", edge_type="on_success", label=None)
        e1_updated = EdgeSpec(id="e1", from_node="source", to_node="t2", edge_type="on_success", label=None)
        state2 = state.with_edge(e1).with_edge(e1_updated)
        assert len(state2.edges) == 1
        assert state2.edges[0].to_node == "t2"

    # --- without_edge ---

    def test_without_edge_removes(self) -> None:
        state = self._empty_state().with_edge(self._make_edge("e1"))
        new_state = state.without_edge("e1")
        assert len(new_state.edges) == 0

    def test_without_edge_nonexistent_returns_none(self) -> None:
        state = self._empty_state()
        result = state.without_edge("nonexistent")
        assert result is None

    # --- with_output ---

    def test_with_output_adds(self) -> None:
        state = self._empty_state()
        output = self._make_output()
        new_state = state.with_output(output)
        assert len(new_state.outputs) == 1
        assert new_state.version == 2

    def test_with_output_replaces_by_name(self) -> None:
        state = self._empty_state()
        o1 = self._make_output("out")
        o2 = OutputSpec(name="out", plugin="json", options={}, on_write_failure="discard")
        state2 = state.with_output(o1).with_output(o2)
        assert len(state2.outputs) == 1
        assert state2.outputs[0].plugin == "json"

    # --- without_output ---

    def test_without_output_removes(self) -> None:
        state = self._empty_state().with_output(self._make_output("out"))
        new_state = state.without_output("out")
        assert len(new_state.outputs) == 0

    def test_without_output_nonexistent_returns_none(self) -> None:
        result = self._empty_state().without_output("nope")
        assert result is None

    # --- with_metadata ---

    def test_with_metadata_partial_update(self) -> None:
        state = self._empty_state()
        new_state = state.with_metadata({"name": "My Pipeline"})
        assert new_state.metadata.name == "My Pipeline"
        assert new_state.metadata.description == ""  # unchanged
        assert new_state.version == 2

    def test_with_metadata_full_update(self) -> None:
        state = self._empty_state()
        new_state = state.with_metadata({
            "name": "P1",
            "description": "Desc",
        })
        assert new_state.metadata.name == "P1"
        assert new_state.metadata.description == "Desc"

    # --- Freeze integrity after replace ---

    # --- to_dict ---

    def test_to_dict_unwraps_frozen_containers(self) -> None:
        """to_dict() converts MappingProxyType -> dict and tuple -> list."""
        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"k": "v"}},
            on_validation_failure="discard",
        )
        state = state.with_source(src)
        state = state.with_node(self._make_node("t1"))
        state = state.with_output(self._make_output("out"))

        d = state.to_dict()
        # All containers should be plain dicts and lists
        assert isinstance(d, dict)
        assert isinstance(d["nodes"], list)
        assert isinstance(d["source"]["options"], dict)
        assert isinstance(d["source"]["options"]["nested"], dict)
        assert isinstance(d["outputs"], list)

    def test_to_dict_roundtrip_yaml(self) -> None:
        """to_dict() output is yaml.dump()-safe (no MappingProxyType errors)."""
        import yaml

        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"deep": {"k": "v"}}},
            on_validation_failure="quarantine",
        )
        state = state.with_source(src)
        d = state.to_dict()
        # Should not raise RepresenterError
        yaml_str = yaml.dump(d, default_flow_style=False)
        assert "csv" in yaml_str

    def test_mutation_refreezes_containers(self) -> None:
        """Mutation methods must re-freeze since dataclasses.replace() skips __post_init__."""
        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"k": "v"}},
            on_validation_failure="discard",
        )
        new_state = state.with_source(src)
        # The nodes tuple on the new state should be frozen
        assert isinstance(new_state.nodes, tuple)
        # Source options should still be frozen after replace
        with pytest.raises(TypeError):
            new_state.source.options["new"] = "x"  # type: ignore[union-attr, index]
```

- [ ] **Step 2: Implement CompositionState**

Add to `src/elspeth/web/composer/state.py`:

```python
@dataclass(frozen=True, slots=True)
class CompositionState:
    """Immutable, versioned snapshot of a pipeline under construction.

    Every edit produces a new instance with incremented version.
    All container fields are deep-frozen via freeze_fields().

    IMPORTANT: dataclasses.replace() does NOT call __post_init__,
    so mutation methods must manually call _refreeze() after replace.

    Attributes:
        source: The pipeline's single data source. None until set.
        nodes: Ordered tuple of transform, gate, aggregation, coalesce nodes.
        edges: Connections between nodes.
        outputs: Sink configurations.
        metadata: Pipeline name, description, landscape URL.
        version: Monotonically increasing per session, starting at 1.
    """

    source: SourceSpec | None
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    outputs: tuple[OutputSpec, ...]
    metadata: PipelineMetadata
    version: int

    def __post_init__(self) -> None:
        # nodes, edges, outputs are tuples of frozen dataclasses — tuple is
        # already immutable and contents are individually frozen. No freeze
        # guard needed. metadata is a frozen dataclass with scalar-only fields.
        # Source is a frozen dataclass with its own freeze guard.
        # Nothing to freeze here beyond what frozen=True provides.
        pass

    # --- Mutation methods ---

    def with_source(self, source: SourceSpec) -> CompositionState:
        """Return new state with the given source, version incremented."""
        return replace(self, source=source, version=self.version + 1)

    def with_node(self, node: NodeSpec) -> CompositionState:
        """Add or replace a node (matched by id). Version incremented."""
        existing_ids = [n.id for n in self.nodes]
        if node.id in existing_ids:
            # Replace at original position to preserve order
            idx = existing_ids.index(node.id)
            node_list = list(self.nodes)
            node_list[idx] = node
            nodes = tuple(node_list)
        else:
            # Append new node
            nodes = self.nodes + (node,)
        return replace(self, nodes=nodes, version=self.version + 1)

    def without_node(self, node_id: str) -> CompositionState | None:
        """Remove node by id. Returns None if node not found."""
        if not any(n.id == node_id for n in self.nodes):
            return None
        nodes = tuple(n for n in self.nodes if n.id != node_id)
        # Also remove edges referencing this node
        edges = tuple(
            e for e in self.edges
            if e.from_node != node_id and e.to_node != node_id
        )
        return replace(
            self, nodes=nodes, edges=edges, version=self.version + 1
        )

    def with_edge(self, edge: EdgeSpec) -> CompositionState:
        """Add or replace an edge (matched by id). Version incremented."""
        edges = tuple(e for e in self.edges if e.id != edge.id) + (edge,)
        return replace(self, edges=edges, version=self.version + 1)

    def without_edge(self, edge_id: str) -> CompositionState | None:
        """Remove edge by id. Returns None if edge not found."""
        if not any(e.id == edge_id for e in self.edges):
            return None
        edges = tuple(e for e in self.edges if e.id != edge_id)
        return replace(self, edges=edges, version=self.version + 1)

    def with_output(self, output: OutputSpec) -> CompositionState:
        """Add or replace an output (matched by name). Version incremented."""
        outputs = tuple(o for o in self.outputs if o.name != output.name) + (output,)
        return replace(self, outputs=outputs, version=self.version + 1)

    def without_output(self, output_name: str) -> CompositionState | None:
        """Remove output by name. Returns None if output not found."""
        if not any(o.name == output_name for o in self.outputs):
            return None
        outputs = tuple(o for o in self.outputs if o.name != output_name)
        return replace(self, outputs=outputs, version=self.version + 1)

    def with_metadata(self, patch: dict[str, Any]) -> CompositionState:
        """Update metadata fields from partial dict. Version incremented."""
        current = self.metadata
        new_meta = PipelineMetadata(
            name=patch.get("name", current.name),
            description=patch.get("description", current.description),
        )
        return replace(self, metadata=new_meta, version=self.version + 1)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Recursively unwrap frozen containers to plain Python types.

        Converts MappingProxyType -> dict, tuple -> list recursively.
        The result is suitable for yaml.dump() and JSON serialization.
        """
        def _unfreeze(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                return {k: _unfreeze(v) for k, v in obj.items()}
            if isinstance(obj, (tuple, list)):
                return [_unfreeze(item) for item in obj]
            return obj

        result: dict[str, Any] = {
            "version": self.version,
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
            },
            "source": None,
            "nodes": [],
            "edges": [],
            "outputs": [],
        }

        if self.source is not None:
            result["source"] = {
                "plugin": self.source.plugin,
                "on_success": self.source.on_success,
                "options": _unfreeze(self.source.options),
                "on_validation_failure": self.source.on_validation_failure,
            }

        for node in self.nodes:
            node_dict: dict[str, Any] = {
                "id": node.id,
                "node_type": node.node_type,
                "plugin": node.plugin,
                "input": node.input,
                "on_success": node.on_success,
                "on_error": node.on_error,
                "options": _unfreeze(node.options),
            }
            if node.condition is not None:
                node_dict["condition"] = node.condition
            if node.routes is not None:
                node_dict["routes"] = _unfreeze(node.routes)
            if node.fork_to is not None:
                node_dict["fork_to"] = list(node.fork_to)
            if node.branches is not None:
                node_dict["branches"] = list(node.branches)
            if node.policy is not None:
                node_dict["policy"] = node.policy
            if node.merge is not None:
                node_dict["merge"] = node.merge
            result["nodes"].append(node_dict)

        for edge in self.edges:
            result["edges"].append({
                "id": edge.id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "edge_type": edge.edge_type,
                "label": edge.label,
            })

        for output in self.outputs:
            result["outputs"].append({
                "name": output.name,
                "plugin": output.plugin,
                "options": _unfreeze(output.options),
                "on_write_failure": output.on_write_failure,
            })

        return result

    # --- Validation ---

    def validate(self) -> ValidationSummary:
        """Run Stage 1 composition-time validation.

        Pure function of the current state — no catalog or engine consultation.
        Returns ValidationSummary with is_valid and human-readable errors.
        """
        errors: list[str] = []

        # 1. Source exists
        if self.source is None:
            errors.append("No source configured.")

        # 2. At least one output
        if not self.outputs:
            errors.append("No sinks configured.")

        # 3. Edge references valid
        node_ids = {n.id for n in self.nodes}
        output_names = {o.name for o in self.outputs}
        valid_from = node_ids | {"source"}
        valid_to = node_ids | output_names
        for edge in self.edges:
            if edge.from_node not in valid_from:
                errors.append(
                    f"Edge '{edge.id}' references unknown node "
                    f"'{edge.from_node}' as from_node."
                )
            if edge.to_node not in valid_to:
                errors.append(
                    f"Edge '{edge.id}' references unknown node "
                    f"'{edge.to_node}' as to_node."
                )

        # 4. Node IDs unique
        seen_node_ids: set[str] = set()
        for node in self.nodes:
            if node.id in seen_node_ids:
                errors.append(f"Duplicate node ID: '{node.id}'.")
            seen_node_ids.add(node.id)

        # 5. Output names unique
        seen_output_names: set[str] = set()
        for output in self.outputs:
            if output.name in seen_output_names:
                errors.append(f"Duplicate output name: '{output.name}'.")
            seen_output_names.add(output.name)

        # 6. Edge IDs unique
        seen_edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.id in seen_edge_ids:
                errors.append(f"Duplicate edge ID: '{edge.id}'.")
            seen_edge_ids.add(edge.id)

        # 7. Node type field consistency
        for node in self.nodes:
            if node.node_type == "gate":
                if node.condition is None:
                    errors.append(
                        f"Gate '{node.id}' is missing required field 'condition'."
                    )
                if node.routes is None:
                    errors.append(
                        f"Gate '{node.id}' is missing required field 'routes'."
                    )
            elif node.node_type == "transform":
                if node.condition is not None:
                    errors.append(
                        f"Transform '{node.id}' must not have 'condition' field."
                    )
                if node.routes is not None:
                    errors.append(
                        f"Transform '{node.id}' must not have 'routes' field."
                    )
            elif node.node_type == "coalesce":
                if node.branches is None:
                    errors.append(
                        f"Coalesce '{node.id}' is missing required field 'branches'."
                    )
                if node.policy is None:
                    errors.append(
                        f"Coalesce '{node.id}' is missing required field 'policy'."
                    )
            elif node.node_type == "aggregation":
                if node.plugin is None:
                    errors.append(
                        f"Aggregation '{node.id}' is missing required field 'plugin'."
                    )

        # 8. Connection completeness
        edge_destinations = {e.to_node for e in self.edges}
        source_on_success = self.source.on_success if self.source else None
        for node in self.nodes:
            reachable = (
                node.id in edge_destinations
                or node.input == source_on_success
            )
            if not reachable:
                errors.append(
                    f"Node '{node.id}' input '{node.input}' is not reachable "
                    f"from any edge or the source on_success."
                )

        return ValidationSummary(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
        )
```

- [ ] **Step 3: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/state.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/state.py tests/unit/web/composer/test_state.py
git commit -m "feat(web/composer): add CompositionState with immutable mutations and Stage 1 validation"
```

---

### Task 3: Stage 1 Validation Tests

**Files:**
- Modify: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write comprehensive validation tests**

```python
# tests/unit/web/composer/test_state.py (append to existing file)


class TestStage1Validation:
    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None, nodes=(), edges=(), outputs=(),
            metadata=PipelineMetadata(), version=1,
        )

    def _make_source(self, on_success: str = "t1") -> SourceSpec:
        return SourceSpec(
            plugin="csv", on_success=on_success,
            options={}, on_validation_failure="quarantine",
        )

    def _make_transform(self, id: str, input: str, on_success: str) -> NodeSpec:
        return NodeSpec(
            id=id, node_type="transform", plugin="uppercase",
            input=input, on_success=on_success, on_error=None,
            options={}, condition=None, routes=None,
            fork_to=None, branches=None, policy=None, merge=None,
        )

    def _make_output(self, name: str = "main") -> OutputSpec:
        return OutputSpec(
            name=name, plugin="csv", options={}, on_write_failure="discard",
        )

    def _make_edge(
        self, id: str, from_node: str, to_node: str,
        edge_type: str = "on_success",
    ) -> EdgeSpec:
        return EdgeSpec(
            id=id, from_node=from_node, to_node=to_node,
            edge_type=edge_type, label=None,
        )

    def test_empty_state_has_errors(self) -> None:
        result = self._empty_state().validate()
        assert not result.is_valid
        assert "No source configured." in result.errors
        assert "No sinks configured." in result.errors

    def test_minimal_valid_pipeline(self) -> None:
        """source -> transform -> sink, fully connected."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="t1"))
        state = state.with_node(self._make_transform("t1", "source_out", "main"))
        state = state.with_output(self._make_output("main"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "main"))
        result = state.validate()
        # t1's input is "source_out" which matches source.on_success="t1"
        # — wait, source.on_success is "t1", but node.input is "source_out".
        # Connection completeness checks node.input against source.on_success
        # and edge to_node. Edge e1 has to_node="t1" which matches node.id,
        # so it's reachable via edge.
        assert result.is_valid, result.errors

    def test_dangling_edge_from_node(self) -> None:
        state = self._empty_state()
        state = state.with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_edge(
            self._make_edge("e1", "nonexistent", "main")
        )
        result = state.validate()
        assert not result.is_valid
        assert any("nonexistent" in e and "from_node" in e for e in result.errors)

    def test_dangling_edge_to_node(self) -> None:
        state = self._empty_state()
        state = state.with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_edge(
            self._make_edge("e1", "source", "nonexistent")
        )
        result = state.validate()
        assert not result.is_valid
        assert any("nonexistent" in e and "to_node" in e for e in result.errors)

    def test_duplicate_node_ids(self) -> None:
        """Two nodes with same id — caught by validation, not by with_node (which replaces)."""
        # We construct directly to bypass with_node's dedup
        node = self._make_transform("dup", "in", "out")
        state = CompositionState(
            source=self._make_source(),
            nodes=(node, node),
            edges=(),
            outputs=(self._make_output(),),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = state.validate()
        assert not result.is_valid
        assert any("Duplicate node ID" in e for e in result.errors)

    def test_duplicate_output_names(self) -> None:
        out = self._make_output("dup")
        state = CompositionState(
            source=self._make_source(),
            nodes=(),
            edges=(),
            outputs=(out, out),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = state.validate()
        assert not result.is_valid
        assert any("Duplicate output name" in e for e in result.errors)

    def test_duplicate_edge_ids(self) -> None:
        edge = self._make_edge("dup", "source", "main")
        state = CompositionState(
            source=self._make_source(),
            nodes=(),
            edges=(edge, edge),
            outputs=(self._make_output(),),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = state.validate()
        assert not result.is_valid
        assert any("Duplicate edge ID" in e for e in result.errors)

    def test_gate_missing_condition(self) -> None:
        gate = NodeSpec(
            id="g1", node_type="gate", plugin=None, input="in",
            on_success=None, on_error=None, options={},
            condition=None,  # missing!
            routes={"high": "s1"},
            fork_to=None, branches=None, policy=None, merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("condition" in e for e in result.errors)

    def test_gate_missing_routes(self) -> None:
        gate = NodeSpec(
            id="g1", node_type="gate", plugin=None, input="in",
            on_success=None, on_error=None, options={},
            condition="row['x'] > 1",
            routes=None,  # missing!
            fork_to=None, branches=None, policy=None, merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(gate)
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        result = state.validate()
        assert not result.is_valid
        assert any("routes" in e for e in result.errors)

    def test_transform_with_condition_is_error(self) -> None:
        node = NodeSpec(
            id="t1", node_type="transform", plugin="uppercase", input="in",
            on_success="out", on_error=None, options={},
            condition="row['x'] > 1",  # not allowed on transforms
            routes=None, fork_to=None, branches=None, policy=None, merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("condition" in e for e in result.errors)

    def test_coalesce_missing_branches(self) -> None:
        node = NodeSpec(
            id="c1", node_type="coalesce", plugin=None, input="join",
            on_success="out", on_error=None, options={},
            condition=None, routes=None, fork_to=None,
            branches=None,  # missing!
            policy="require_all", merge="nested",
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "c1"))
        result = state.validate()
        assert not result.is_valid
        assert any("branches" in e for e in result.errors)

    def test_aggregation_missing_plugin(self) -> None:
        node = NodeSpec(
            id="a1", node_type="aggregation", plugin=None, input="in",
            on_success="out", on_error=None, options={},
            condition=None, routes=None, fork_to=None,
            branches=None, policy=None, merge=None,
        )
        state = self._empty_state().with_source(self._make_source())
        state = state.with_output(self._make_output())
        state = state.with_node(node)
        state = state.with_edge(self._make_edge("e1", "source", "a1"))
        result = state.validate()
        assert not result.is_valid
        assert any("plugin" in e for e in result.errors)

    def test_unreachable_node(self) -> None:
        """Node exists but no edge points to it and source.on_success doesn't match."""
        state = self._empty_state()
        state = state.with_source(self._make_source(on_success="other"))
        state = state.with_node(self._make_transform("t1", "somewhere", "main"))
        state = state.with_output(self._make_output())
        # No edge to t1, and source.on_success="other" != node.input="somewhere"
        result = state.validate()
        assert not result.is_valid
        assert any("not reachable" in e for e in result.errors)
```

- [ ] **Step 2: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests PASS (validation logic was implemented in Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_state.py
git commit -m "test(web/composer): comprehensive Stage 1 validation tests"
```

---

### Task 4: Composition Tools — ToolResult and Mutation Tools

**Files:**
- Create: `src/elspeth/web/composer/tools.py`
- Create: `tests/unit/web/composer/test_tools.py`

- [ ] **Step 1: Write ToolResult and mutation tool tests**

```python
# tests/unit/web/composer/test_tools.py
"""Tests for composition tools — discovery delegation and mutation + validation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSchemaInfo,
    PluginSummary,
)
from elspeth.web.composer.tools import (
    ToolResult,
    execute_tool,
    get_expression_grammar,
    get_tool_definitions,
)


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None, nodes=(), edges=(), outputs=(),
        metadata=PipelineMetadata(), version=1,
    )


def _mock_catalog() -> MagicMock:
    """Mock CatalogService with real PluginSummary/PluginSchemaInfo instances.

    AC #16: Tests must use real PluginSummary and PluginSchemaInfo instances,
    not plain dicts. Mock return types must match the CatalogService protocol.
    """
    catalog = MagicMock()
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv", description="CSV file source",
            plugin_type="source", config_fields=[
                ConfigFieldSummary(name="path", type="string", required=True, description="File path", default=None),
            ],
        ),
        PluginSummary(
            name="json", description="JSON file source",
            plugin_type="source", config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(
            name="uppercase", description="Uppercase transform",
            plugin_type="transform", config_fields=[],
        ),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name="csv", description="CSV file sink",
            plugin_type="sink", config_fields=[],
        ),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv", plugin_type="source", description="CSV file source",
        json_schema={"title": "CsvSourceConfig", "properties": {"path": {"type": "string"}}},
    )
    return catalog


class TestToolResult:
    def test_frozen(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=False, errors=("err",)),
            affected_nodes=("n1", "n2"),
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_affected_nodes_frozen(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=("n1",),
        )
        assert isinstance(result.affected_nodes, tuple)


class TestSetSource:
    def test_sets_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert result.updated_state.source is not None
        assert result.updated_state.source.plugin == "csv"
        assert result.updated_state.version == 2
        assert "source" in result.affected_nodes

    def test_unknown_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: foobar")
        result = execute_tool(
            "set_source",
            {
                "plugin": "foobar",
                "on_success": "t1",
                "options": {},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert result.updated_state.source is None  # unchanged
        assert result.updated_state.version == 1


class TestUpsertNode:
    def test_adds_new_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "uppercase",
                "input": "source_out",
                "on_success": "main",
                "options": {},
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.nodes) == 1
        assert "t1" in result.affected_nodes

    def test_replaces_existing_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result1 = execute_tool(
            "upsert_node",
            {
                "id": "t1", "node_type": "transform", "plugin": "uppercase",
                "input": "in", "on_success": "out", "options": {},
            },
            state, catalog,
        )
        result2 = execute_tool(
            "upsert_node",
            {
                "id": "t1", "node_type": "transform", "plugin": "uppercase",
                "input": "new_in", "on_success": "out", "options": {"field": "x"},
            },
            result1.updated_state, catalog,
        )
        assert result2.success is True
        assert len(result2.updated_state.nodes) == 1
        assert result2.updated_state.nodes[0].input == "new_in"

    def test_gate_node_no_plugin_validation(self) -> None:
        """Gates don't have plugins — should not validate against catalog."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1", "node_type": "gate", "plugin": None,
                "input": "in", "on_success": None, "options": {},
                "condition": "row['x'] > 0",
                "routes": {"high": "s1", "low": "s2"},
            },
            state, catalog,
        )
        assert result.success is True
        catalog.get_schema.assert_not_called()


class TestUpsertEdge:
    def test_adds_edge(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_edge",
            {
                "id": "e1", "from_node": "source", "to_node": "t1",
                "edge_type": "on_success", "label": None,
            },
            state, catalog,
        )
        assert result.success is True
        assert len(result.updated_state.edges) == 1
        assert "source" in result.affected_nodes
        assert "t1" in result.affected_nodes


class TestRemoveNode:
    def test_removes_node_and_edges(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Add a node and an edge to it
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1", "node_type": "transform", "plugin": "uppercase",
                "input": "in", "on_success": "out", "options": {},
            },
            state, catalog,
        )
        r2 = execute_tool(
            "upsert_edge",
            {
                "id": "e1", "from_node": "source", "to_node": "t1",
                "edge_type": "on_success", "label": None,
            },
            r1.updated_state, catalog,
        )
        # Remove the node — edge should also be removed
        r3 = execute_tool("remove_node", {"id": "t1"}, r2.updated_state, catalog)
        assert r3.success is True
        assert len(r3.updated_state.nodes) == 0
        assert len(r3.updated_state.edges) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_node", {"id": "nope"}, state, catalog)
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged


class TestRemoveEdge:
    def test_removes_edge(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "upsert_edge",
            {
                "id": "e1", "from_node": "source", "to_node": "t1",
                "edge_type": "on_success", "label": None,
            },
            state, catalog,
        )
        r2 = execute_tool("remove_edge", {"id": "e1"}, r1.updated_state, catalog)
        assert r2.success is True
        assert len(r2.updated_state.edges) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_edge", {"id": "nope"}, state, catalog)
        assert result.success is False


class TestSetMetadata:
    def test_partial_update(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_metadata", {"patch": {"name": "My Pipeline"}}, state, catalog,
        )
        assert result.success is True
        assert result.updated_state.metadata.name == "My Pipeline"
        assert result.updated_state.metadata.description == ""  # preserved
        assert result.affected_nodes == ()  # metadata doesn't affect nodes


class TestDiscoveryTools:
    def test_list_sources_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_sources", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_sources.assert_called_once()

    def test_list_transforms_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_transforms", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_transforms.assert_called_once()

    def test_list_sinks_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_sinks", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_sinks.assert_called_once()

    def test_get_plugin_schema_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "csv"},
            _empty_state(), catalog,
        )
        assert result.success is True
        catalog.get_schema.assert_called_once_with("source", "csv")

    def test_get_expression_grammar_is_static(self) -> None:
        grammar = get_expression_grammar()
        assert "row" in grammar
        assert isinstance(grammar, str)

    def test_get_current_state_returns_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("get_current_state", {}, state, catalog)
        assert result.success is True
        # State is unchanged
        assert result.updated_state.version == 1


class TestToolDefinitions:
    def test_has_twelve_tools(self) -> None:
        """6 discovery + 6 mutation = 12 tools."""
        defs = get_tool_definitions()
        assert len(defs) == 12

    def test_all_have_json_schema(self) -> None:
        for defn in get_tool_definitions():
            assert "name" in defn
            assert "description" in defn
            assert "parameters" in defn


class TestToolResultValidation:
    def test_mutation_includes_validation(self) -> None:
        """Every mutation tool result includes validation summary."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv", "on_success": "t1",
                "options": {}, "on_validation_failure": "quarantine",
            },
            state, catalog,
        )
        assert result.validation is not None
        # Source is set but no sinks — should have validation error
        assert not result.validation.is_valid
        assert any("No sinks" in e for e in result.validation.errors)
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ToolResult and tool executor**

```python
# src/elspeth/web/composer/tools.py
"""Composition tools — discovery and mutation tools for the LLM composer.

Discovery tools delegate to CatalogService. Mutation tools modify
CompositionState and return ToolResult with validation.

Layer: L3 (application). Imports from L0 (contracts.freeze) and
L3 (web/composer/state).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from elspeth.contracts.freeze import freeze_fields

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)


class CatalogServiceProtocol(Protocol):
    """Protocol for catalog service dependency.

    Return types match the CatalogService protocol from Sub-Spec 3:
    list methods return list[PluginSummary], get_schema returns
    PluginSchemaInfo. Import these from web.catalog.schemas.
    """

    def list_sources(self) -> list[Any]: ...
    def list_transforms(self) -> list[Any]: ...
    def list_sinks(self) -> list[Any]: ...
    def get_schema(self, plugin_type: str, name: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the operation succeeded.
        updated_state: Full state after mutation (or original if success=False).
        validation: Stage 1 validation result for the updated state.
        affected_nodes: Node IDs changed or with changed edges.
        data: Optional data payload for discovery tools.
    """

    success: bool
    updated_state: CompositionState
    validation: ValidationSummary
    affected_nodes: tuple[str, ...]
    data: Any = None

    def __post_init__(self) -> None:
        freeze_fields(self, "affected_nodes")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for LLM tool response."""
        result: dict[str, Any] = {
            "success": self.success,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": list(self.validation.errors),
            },
            "affected_nodes": list(self.affected_nodes),
            "version": self.updated_state.version,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


# --- Expression Grammar (static) ---

_EXPRESSION_GRAMMAR = """\
Gate Expression Syntax Reference
=================================

Variables:
  row      - The current row as a dict. Access fields via row['field_name'].

Operators:
  ==, !=, <, >, <=, >=   Comparison
  and, or, not            Boolean logic
  in, not in              Membership test
  +, -, *, /, //, %       Arithmetic

Built-in functions:
  len(), str(), int(), float(), bool(), abs(), min(), max(), round()
  isinstance(), type()

Examples:
  row['confidence'] >= 0.85
  row['status'] == 'approved'
  row['category'] in ('A', 'B', 'C')
  len(row.get('tags', [])) > 0
  row['score'] > 0.5 and row['status'] != 'rejected'

Security:
  Expressions are validated by ExpressionParser. Attribute access, imports,
  function calls to non-builtins, and dunder access are forbidden.
"""


def get_expression_grammar() -> str:
    """Return the gate expression syntax reference."""
    return _EXPRESSION_GRAMMAR


# --- Tool Definitions for LLM ---

def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 12 tools: 6 discovery, 6 mutation.
    """
    return [
        # Discovery tools
        {
            "name": "list_sources",
            "description": "List available source plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_transforms",
            "description": "List available transform plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_sinks",
            "description": "List available sink plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_plugin_schema",
            "description": "Get the full configuration schema for a plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_type": {
                        "type": "string",
                        "enum": ["source", "transform", "sink"],
                        "description": "Plugin type.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Plugin name (e.g. 'csv').",
                    },
                },
                "required": ["plugin_type", "name"],
            },
        },
        {
            "name": "get_expression_grammar",
            "description": "Get the gate expression syntax reference.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_current_state",
            "description": "Get the full current pipeline composition state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        # Mutation tools
        {
            "name": "set_source",
            "description": "Set or replace the pipeline source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Source plugin name."},
                    "on_success": {"type": "string", "description": "Connection name for downstream."},
                    "options": {"type": "object", "description": "Plugin-specific config."},
                    "on_validation_failure": {
                        "type": "string",
                        "enum": ["discard", "quarantine"],
                        "description": "How to handle validation failures.",
                    },
                },
                "required": ["plugin", "on_success", "options", "on_validation_failure"],
            },
        },
        {
            "name": "upsert_node",
            "description": "Add or update a pipeline node (transform, gate, aggregation, coalesce).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique node identifier."},
                    "node_type": {
                        "type": "string",
                        "enum": ["transform", "gate", "aggregation", "coalesce"],
                    },
                    "plugin": {"type": ["string", "null"], "description": "Plugin name. Null for gates/coalesces."},
                    "input": {"type": "string", "description": "Input connection name."},
                    "on_success": {"type": ["string", "null"], "description": "Output connection. Null for gates."},
                    "on_error": {"type": ["string", "null"], "description": "Error output connection."},
                    "options": {"type": "object", "description": "Plugin-specific config."},
                    "condition": {"type": ["string", "null"], "description": "Gate expression."},
                    "routes": {"type": ["object", "null"], "description": "Gate route mapping."},
                    "fork_to": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Fork destinations."},
                    "branches": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Coalesce branch inputs."},
                    "policy": {"type": ["string", "null"], "description": "Coalesce policy."},
                    "merge": {"type": ["string", "null"], "description": "Coalesce merge strategy."},
                },
                "required": ["id", "node_type", "input"],
            },
        },
        {
            "name": "upsert_edge",
            "description": "Add or update a connection between nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique edge identifier."},
                    "from_node": {"type": "string", "description": "Source node ID or 'source'."},
                    "to_node": {"type": "string", "description": "Destination node ID or sink name."},
                    "edge_type": {
                        "type": "string",
                        "enum": ["on_success", "on_error", "route_true", "route_false", "fork"],
                    },
                    "label": {"type": ["string", "null"], "description": "Display label."},
                },
                "required": ["id", "from_node", "to_node", "edge_type"],
            },
        },
        {
            "name": "remove_node",
            "description": "Remove a node and all its edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "remove_edge",
            "description": "Remove an edge by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Edge ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "set_metadata",
            "description": "Update pipeline metadata (name and description only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "object",
                        "description": "Partial metadata update. Only included fields are changed.",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required": ["patch"],
            },
        },
    ]


# --- State serialization ---

def _serialize_state(state: CompositionState) -> dict[str, Any]:
    """Serialize CompositionState to a JSON-compatible dict.

    Delegates to state.to_dict() which recursively unwraps frozen
    containers (MappingProxyType -> dict, tuple -> list).
    """
    return state.to_dict()


# --- Tool Executor ---

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> ToolResult:
    """Execute a composition tool by name.

    Discovery tools return data without modifying state.
    Mutation tools return ToolResult with updated state and validation.
    Invalid tool names return a failure result with an error message.
    """
    # Discovery tools
    if tool_name == "list_sources":
        return _discovery_result(state, catalog.list_sources())

    if tool_name == "list_transforms":
        return _discovery_result(state, catalog.list_transforms())

    if tool_name == "list_sinks":
        return _discovery_result(state, catalog.list_sinks())

    if tool_name == "get_plugin_schema":
        try:
            schema = catalog.get_schema(arguments["plugin_type"], arguments["name"])
            return _discovery_result(state, schema)
        except (ValueError, KeyError) as exc:
            return _failure_result(state, str(exc))

    if tool_name == "get_expression_grammar":
        return _discovery_result(state, get_expression_grammar())

    if tool_name == "get_current_state":
        serialized = _serialize_state(state)
        validation = state.validate()
        serialized["validation"] = {
            "is_valid": validation.is_valid,
            "errors": list(validation.errors),
        }
        return _discovery_result(state, serialized)

    # Mutation tools
    if tool_name == "set_source":
        return _execute_set_source(arguments, state, catalog)

    if tool_name == "upsert_node":
        return _execute_upsert_node(arguments, state, catalog)

    if tool_name == "upsert_edge":
        return _execute_upsert_edge(arguments, state)

    if tool_name == "remove_node":
        return _execute_remove_node(arguments, state)

    if tool_name == "remove_edge":
        return _execute_remove_edge(arguments, state)

    if tool_name == "set_metadata":
        return _execute_set_metadata(arguments, state)

    return _failure_result(state, f"Unknown tool: {tool_name}")


def _discovery_result(state: CompositionState, data: Any) -> ToolResult:
    """Build a ToolResult for a discovery (read-only) tool."""
    validation = state.validate()
    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=data,
    )


def _failure_result(
    state: CompositionState,
    error_msg: str,
) -> ToolResult:
    """Build a ToolResult for a failed mutation."""
    validation = state.validate()
    return ToolResult(
        success=False,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data={"error": error_msg},
    )


def _mutation_result(
    new_state: CompositionState,
    affected: tuple[str, ...],
) -> ToolResult:
    """Build a ToolResult for a successful mutation."""
    validation = new_state.validate()
    return ToolResult(
        success=True,
        updated_state=new_state,
        validation=validation,
        affected_nodes=affected,
    )


def _execute_set_source(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> ToolResult:
    """Set or replace the pipeline source."""
    plugin = args["plugin"]
    # Validate plugin exists in catalog
    try:
        catalog.get_schema("source", plugin)
    except (ValueError, KeyError) as exc:
        return _failure_result(state, f"Unknown source plugin '{plugin}': {exc}")

    source = SourceSpec(
        plugin=plugin,
        on_success=args["on_success"],
        options=args.get("options", {}),
        on_validation_failure=args.get("on_validation_failure", "quarantine"),
    )
    new_state = state.with_source(source)
    return _mutation_result(new_state, ("source",))


def _execute_upsert_node(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> ToolResult:
    """Add or update a pipeline node."""
    node_type = args["node_type"]
    plugin = args.get("plugin")

    # Validate plugin for types that require one
    if node_type in ("transform", "aggregation") and plugin is not None:
        try:
            catalog.get_schema("transform", plugin)
        except (ValueError, KeyError) as exc:
            return _failure_result(
                state, f"Unknown {node_type} plugin '{plugin}': {exc}"
            )

    fork_to = args.get("fork_to")
    if fork_to is not None:
        fork_to = tuple(fork_to)

    branches = args.get("branches")
    if branches is not None:
        branches = tuple(branches)

    node = NodeSpec(
        id=args["id"],
        node_type=node_type,
        plugin=plugin,
        input=args["input"],
        on_success=args.get("on_success"),
        on_error=args.get("on_error"),
        options=args.get("options", {}),
        condition=args.get("condition"),
        routes=args.get("routes"),
        fork_to=fork_to,
        branches=branches,
        policy=args.get("policy"),
        merge=args.get("merge"),
    )

    node_id = args["id"]
    new_state = state.with_node(node)

    # Affected: the node itself plus nodes with edges referencing it
    affected = {node_id}
    for edge in new_state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_upsert_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Add or update an edge."""
    edge = EdgeSpec(
        id=args["id"],
        from_node=args["from_node"],
        to_node=args["to_node"],
        edge_type=args["edge_type"],
        label=args.get("label"),
    )
    new_state = state.with_edge(edge)
    return _mutation_result(new_state, (args["from_node"], args["to_node"]))


def _execute_remove_node(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove a node and its edges."""
    node_id = args["id"]

    # Collect affected nodes before removal (edges that reference this node)
    affected = {node_id}
    for edge in state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    new_state = state.without_node(node_id)
    if new_state is None:
        return _failure_result(state, f"Node '{node_id}' not found.")

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_remove_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove an edge."""
    edge_id = args["id"]

    # Find the edge to get affected nodes
    edge = next((e for e in state.edges if e.id == edge_id), None)
    if edge is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    affected = (edge.from_node, edge.to_node)
    new_state = state.without_edge(edge_id)
    if new_state is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    return _mutation_result(new_state, affected)


def _execute_set_metadata(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Update pipeline metadata."""
    patch = args.get("patch", args)
    # If the LLM passes fields directly instead of under "patch"
    if "patch" in args and isinstance(args["patch"], dict):
        patch = args["patch"]

    new_state = state.with_metadata(patch)
    return _mutation_result(new_state, ())
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/tools.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "feat(web/composer): add composition tools — 6 discovery, 6 mutation, ToolResult"
```

---

### Task 5: YAML Generator

**Files:**
- Create: `src/elspeth/web/composer/yaml_generator.py`
- Create: `tests/unit/web/composer/test_yaml_generator.py`

- [ ] **Step 1: Write YAML generator tests**

```python
# tests/unit/web/composer/test_yaml_generator.py
"""Tests for deterministic YAML generation from CompositionState."""
from __future__ import annotations

import yaml

import pytest

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.yaml_generator import generate_yaml


def _make_linear_pipeline() -> CompositionState:
    """Source -> transform -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="transform_1",
            options={"path": "/data/input.csv", "schema": {"fields": ["name", "age"]}},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="transform_1",
                node_type="transform",
                plugin="uppercase",
                input="source_out",
                on_success="main_output",
                on_error=None,
                options={"field": "name"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(
            EdgeSpec(id="e1", from_node="source", to_node="transform_1", edge_type="on_success", label=None),
            EdgeSpec(id="e2", from_node="transform_1", to_node="main_output", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="main_output",
                plugin="csv",
                options={"path": "/data/output.csv"},
                on_write_failure="quarantine",
            ),
        ),
        metadata=PipelineMetadata(name="Linear Pipeline", description="A simple pipeline"),
        version=5,
    )


def _make_gate_pipeline() -> CompositionState:
    """Source -> gate -> two sinks."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="quality_check",
            options={"path": "/data/in.csv"},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="quality_check",
                node_type="gate",
                plugin=None,
                input="source_out",
                on_success=None,
                on_error=None,
                options={},
                condition="row['confidence'] >= 0.85",
                routes={"high": "good_output", "low": "review_output"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="good_output", plugin="csv", options={"path": "/good.csv"}, on_write_failure="quarantine"),
            OutputSpec(name="review_output", plugin="csv", options={"path": "/review.csv"}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(name="Gate Pipeline"),
        version=3,
    )


def _make_aggregation_pipeline() -> CompositionState:
    """Source -> aggregation -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv", on_success="batch_agg",
            options={"path": "/data/in.csv"}, on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="batch_agg", node_type="aggregation", plugin="batch_counter",
                input="source_out", on_success="main_output", on_error=None,
                options={"batch_size": 10}, condition=None, routes=None,
                fork_to=None, branches=None, policy=None, merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="main_output", plugin="csv", options={}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _make_fork_coalesce_pipeline() -> CompositionState:
    """Source -> fork gate -> two paths -> coalesce -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv", on_success="fork_gate",
            options={}, on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="fork_gate", node_type="gate", plugin=None,
                input="source_out", on_success=None, on_error=None,
                options={}, condition="True",
                routes={"all": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None, policy=None, merge=None,
            ),
            NodeSpec(
                id="merge_point", node_type="coalesce", plugin=None,
                input="join", on_success="main_output", on_error=None,
                options={}, condition=None, routes=None, fork_to=None,
                branches=("path_a", "path_b"), policy="require_all", merge="nested",
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="main_output", plugin="csv", options={}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


class TestGenerateYaml:
    def test_linear_pipeline(self) -> None:
        state = _make_linear_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        # Source
        assert parsed["source"]["plugin"] == "csv"
        assert parsed["source"]["on_success"] == "transform_1"
        assert parsed["source"]["options"]["path"] == "/data/input.csv"
        assert parsed["source"]["options"]["on_validation_failure"] == "quarantine"

        # Transform
        assert len(parsed["transforms"]) == 1
        t = parsed["transforms"][0]
        assert t["name"] == "transform_1"
        assert t["plugin"] == "uppercase"
        assert t["input"] == "source_out"
        assert t["on_success"] == "main_output"
        assert t["options"]["field"] == "name"

        # Sink
        assert "main_output" in parsed["sinks"]
        s = parsed["sinks"]["main_output"]
        assert s["plugin"] == "csv"

    def test_gate_pipeline(self) -> None:
        state = _make_gate_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "gates" in parsed
        assert len(parsed["gates"]) == 1
        g = parsed["gates"][0]
        assert g["name"] == "quality_check"
        assert g["condition"] == "row['confidence'] >= 0.85"
        assert g["routes"]["high"] == "good_output"
        assert g["routes"]["low"] == "review_output"

    def test_aggregation_pipeline(self) -> None:
        state = _make_aggregation_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "aggregations" in parsed
        assert len(parsed["aggregations"]) == 1
        a = parsed["aggregations"][0]
        assert a["name"] == "batch_agg"
        assert a["plugin"] == "batch_counter"
        assert a["options"]["batch_size"] == 10

    def test_fork_coalesce_pipeline(self) -> None:
        state = _make_fork_coalesce_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "gates" in parsed
        gate = parsed["gates"][0]
        assert gate["fork_to"] == ["path_a", "path_b"]

        assert "coalesce" in parsed
        coal = parsed["coalesce"][0]
        assert coal["branches"] == ["path_a", "path_b"]
        assert coal["policy"] == "require_all"
        assert coal["merge"] == "nested"

    def test_deterministic(self) -> None:
        """Same state produces byte-identical YAML."""
        state = _make_linear_pipeline()
        yaml1 = generate_yaml(state)
        yaml2 = generate_yaml(state)
        assert yaml1 == yaml2

    def test_landscape_key_never_emitted(self) -> None:
        """landscape key is never emitted — URL comes from WebSettings at execution time (S1 fix)."""
        state = _make_linear_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert "landscape" not in parsed

    def test_on_error_emitted_when_set(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv", on_success="t1",
                options={}, on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="t1", node_type="transform", plugin="uppercase",
                    input="in", on_success="out", on_error="error_sink",
                    options={}, condition=None, routes=None,
                    fork_to=None, branches=None, policy=None, merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),
                OutputSpec(name="error_sink", plugin="csv", options={}, on_write_failure="discard"),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["transforms"][0]["on_error"] == "error_sink"

    def test_frozen_state_serializes_without_error(self) -> None:
        """generate_yaml() handles frozen state objects (MappingProxyType, tuple).

        AC #15: No RepresenterError from PyYAML on frozen containers.
        Verifies that generate_yaml() correctly calls state.to_dict()
        before yaml.dump().
        """
        state = _make_linear_pipeline()
        # State has been through freeze_fields() — options are MappingProxyType
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["source"]["plugin"] == "csv"
        # Nested frozen options must serialize correctly
        assert parsed["source"]["options"]["schema"]["fields"] == ["name", "age"]

    def test_empty_state_minimal_yaml(self) -> None:
        """Empty state produces minimal valid YAML (no source, no sinks)."""
        state = CompositionState(
            source=None, nodes=(), edges=(), outputs=(),
            metadata=PipelineMetadata(), version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed.get("source") is None or "source" not in parsed
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement YAML generator**

```python
# src/elspeth/web/composer/yaml_generator.py
"""Deterministic YAML generator — CompositionState to ELSPETH pipeline YAML.

Pure function. Same CompositionState always produces byte-identical YAML.
Uses yaml.dump() with sort_keys=True for determinism.

Layer: L3 (application). Imports from L3 (web/composer/state) only.
"""
from __future__ import annotations

from typing import Any

import yaml

from elspeth.web.composer.state import CompositionState


def generate_yaml(state: CompositionState) -> str:
    """Convert a CompositionState to ELSPETH pipeline YAML.

    The output is deterministic: same state produces byte-identical YAML.
    Maps CompositionState fields to the YAML structure expected by
    ELSPETH's load_settings() parser.

    Calls state.to_dict() to unwrap all frozen containers
    (MappingProxyType -> dict, tuple -> list) before passing to
    yaml.dump(). This avoids RepresenterError from PyYAML on frozen
    types. See spec R4 and AC #15.

    Args:
        state: The pipeline composition state to serialize.

    Returns:
        YAML string representing the pipeline configuration.
    """
    # Unwrap frozen containers to plain Python types (R4).
    # to_dict() recursively converts MappingProxyType -> dict,
    # tuple -> list. Without this, yaml.dump() raises RepresenterError.
    state_dict = state.to_dict()

    doc: dict[str, Any] = {}

    # Source
    source = state_dict.get("source")
    if source is not None:
        source_options = dict(source["options"])
        source_options["on_validation_failure"] = source["on_validation_failure"]
        doc["source"] = {
            "plugin": source["plugin"],
            "on_success": source["on_success"],
            "options": source_options,
        }

    # Transforms
    transforms = [n for n in state_dict["nodes"] if n["node_type"] == "transform"]
    if transforms:
        doc["transforms"] = []
        for t in transforms:
            entry: dict[str, Any] = {
                "name": t["id"],
                "plugin": t["plugin"],
                "input": t["input"],
                "on_success": t["on_success"],
            }
            if t.get("on_error") is not None:
                entry["on_error"] = t["on_error"]
            if t["options"]:
                entry["options"] = t["options"]
            doc["transforms"].append(entry)

    # Gates
    gates = [n for n in state_dict["nodes"] if n["node_type"] == "gate"]
    if gates:
        doc["gates"] = []
        for g in gates:
            entry = {
                "name": g["id"],
                "input": g["input"],
                "condition": g.get("condition"),
                "routes": g.get("routes", {}),
            }
            if g.get("fork_to") is not None:
                entry["fork_to"] = g["fork_to"]
            doc["gates"].append(entry)

    # Aggregations
    aggregations = [n for n in state_dict["nodes"] if n["node_type"] == "aggregation"]
    if aggregations:
        doc["aggregations"] = []
        for a in aggregations:
            entry = {
                "name": a["id"],
                "plugin": a["plugin"],
                "input": a["input"],
                "on_success": a["on_success"],
            }
            if a.get("on_error") is not None:
                entry["on_error"] = a["on_error"]
            if a["options"]:
                entry["options"] = a["options"]
            doc["aggregations"].append(entry)

    # Coalesce
    coalesces = [n for n in state_dict["nodes"] if n["node_type"] == "coalesce"]
    if coalesces:
        doc["coalesce"] = []
        for c in coalesces:
            entry = {
                "name": c["id"],
                "branches": c.get("branches", []),
                "policy": c.get("policy"),
                "merge": c.get("merge"),
            }
            doc["coalesce"].append(entry)

    # Sinks
    if state_dict["outputs"]:
        doc["sinks"] = {}
        for output in state_dict["outputs"]:
            sink_entry: dict[str, Any] = {
                "plugin": output["plugin"],
                "on_write_failure": output["on_write_failure"],
            }
            if output["options"]:
                sink_entry["options"] = output["options"]
            doc["sinks"][output["name"]] = sink_entry

    # landscape key is intentionally omitted — URL comes from
    # WebSettings.get_landscape_url() at execution time (security fix S1).

    return yaml.dump(doc, default_flow_style=False, sort_keys=True)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/yaml_generator.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/yaml_generator.py tests/unit/web/composer/test_yaml_generator.py
git commit -m "feat(web/composer): add deterministic YAML generator"
```

---

### Task 6: ComposerService Protocol and Prompts

**Files:**
- Create: `src/elspeth/web/composer/protocol.py`
- Create: `src/elspeth/web/composer/prompts.py`

- [ ] **Step 1: Implement ComposerService protocol**

```python
# src/elspeth/web/composer/protocol.py
"""ComposerService protocol and result types.

Layer: L3 (application). Defines the service boundary for LLM-driven
pipeline composition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from elspeth.web.composer.state import CompositionState


@dataclass(frozen=True, slots=True)
class ComposerResult:
    """Result of a compose() call.

    Attributes:
        message: The assistant's text response.
        state: The (possibly updated) CompositionState.
    """

    message: str
    state: CompositionState


class ComposerServiceError(Exception):
    """Base exception for composer service errors."""


class ComposerConvergenceError(ComposerServiceError):
    """Raised when the LLM tool-use loop exceeds max_turns."""

    def __init__(self, max_turns: int) -> None:
        super().__init__(
            f"Composer did not converge within {max_turns} turns. "
            f"The LLM kept making tool calls without producing a final response."
        )
        self.max_turns = max_turns


class ComposerService(Protocol):
    """Protocol for the LLM-driven pipeline composer.

    Accepts a user message, pre-fetched chat history, and current state.
    Runs the LLM tool-use loop. Returns the assistant's response
    and the (possibly updated) state. Does NOT depend on SessionService —
    the route handler mediates (seam contract B).
    """

    async def compose(
        self,
        message: str,
        messages: list[Any],  # list[ChatMessageRecord]
        state: CompositionState,
    ) -> ComposerResult:
        """Run the LLM composition loop.

        Args:
            message: The user's chat message.
            messages: Pre-fetched chat history (list[ChatMessageRecord]).
                The route handler calls session_service.get_messages()
                and passes the result. ComposerService does NOT depend
                on SessionService — the route handler mediates.
            state: The current CompositionState.

        Returns:
            ComposerResult with assistant message and updated state.

        Raises:
            ComposerConvergenceError: If the loop exceeds max_turns.
        """
        ...
```

- [ ] **Step 2: Implement prompts module**

```python
# src/elspeth/web/composer/prompts.py
"""System prompt and message construction for the LLM composer.

_build_messages() returns a NEW list on every call — never a cached
reference. This is critical because the tool-use loop appends to the
list during iteration.

Layer: L3 (application).
"""
from __future__ import annotations

import json
from typing import Any

from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import CatalogServiceProtocol, _serialize_state


SYSTEM_PROMPT = """\
You are an ELSPETH pipeline composer. Your job is to translate the user's \
natural-language description into a valid pipeline configuration using the \
provided tools.

Rules:
1. Always check the current state (get_current_state) before making changes.
2. Always check plugin schemas (get_plugin_schema) before configuring a plugin.
3. Use list_sources/list_transforms/list_sinks to discover available plugins.
4. After making changes, review the validation result in the tool response. \
If there are errors, fix them before responding to the user.
5. When the pipeline is complete and valid, respond with a summary of what \
was built.
6. Do not fabricate plugin names or configuration fields. Only use plugins \
and fields that appear in the catalog.
7. Use get_expression_grammar to understand gate expression syntax before \
writing conditions.
8. Connect nodes with edges using upsert_edge after creating nodes.
9. Every pipeline needs at least: a source, one or more sinks, and edges \
connecting them.
"""


def build_context_message(
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> dict[str, str]:
    """Build the injected context message with current state and plugin summary.

    Args:
        state: Current composition state.
        catalog: For building the plugin summary.

    Returns:
        A dict with "role" and "content" suitable for the LLM message list.
    """
    serialized = _serialize_state(state)
    validation = state.validate()
    serialized["validation"] = {
        "is_valid": validation.is_valid,
        "errors": list(validation.errors),
    }

    # Build lightweight plugin summary (names only).
    # CatalogService returns PluginSummary instances (not dicts) — use .name attribute.
    source_names = [p.name for p in catalog.list_sources()]
    transform_names = [p.name for p in catalog.list_transforms()]
    sink_names = [p.name for p in catalog.list_sinks()]

    context = {
        "current_state": serialized,
        "available_plugins": {
            "sources": source_names,
            "transforms": transform_names,
            "sinks": sink_names,
        },
    }

    return {
        "role": "system",
        "content": f"Current pipeline state and available plugins:\n{json.dumps(context, indent=2)}",
    }


def build_messages(
    chat_history: list[Any],  # list[ChatMessageRecord], pre-fetched by route handler
    state: CompositionState,
    user_message: str,
    catalog: CatalogServiceProtocol,
) -> list[dict[str, Any]]:
    """Build the full message list for the LLM.

    IMPORTANT: Returns a NEW list on every call. Never returns a cached
    or shared reference. The tool-use loop appends to this list during
    iteration; returning a cached reference would cause cross-turn
    contamination.

    Message sequence:
    1. System message (static prompt)
    2. Injected context (current state + plugin summary)
    3. Chat history (previous messages in this session)
    4. Current user message

    Args:
        chat_history: Pre-fetched chat history (list[ChatMessageRecord]
            from session_service.get_messages(), passed by route handler).
        state: Current CompositionState.
        user_message: The user's current message.
        catalog: CatalogService for context injection.

    Returns:
        A new list of message dicts for the LLM.
    """
    messages: list[dict[str, Any]] = []

    # 1. System prompt
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # 2. Injected context
    messages.append(build_context_message(state, catalog))

    # 3. Chat history
    if chat_history:
        messages.extend(chat_history)

    # 4. Current user message
    messages.append({"role": "user", "content": user_message})

    return messages
```

- [ ] **Step 3: Run mypy on both modules**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/protocol.py src/elspeth/web/composer/prompts.py`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/protocol.py src/elspeth/web/composer/prompts.py
git commit -m "feat(web/composer): add ComposerService protocol and prompt construction"
```

---

### Task 7: ComposerServiceImpl — LLM Tool-Use Loop

**Files:**
- Create: `src/elspeth/web/composer/service.py`
- Create: `tests/unit/web/composer/test_service.py`

- [ ] **Step 1: Write composer loop tests with mock LLM**

```python
# tests/unit/web/composer/test_service.py
"""Tests for ComposerServiceImpl — LLM tool-use loop with mock LLM."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.web.catalog.schemas import (
    PluginSchemaInfo,
    PluginSummary,
)
from elspeth.web.composer.protocol import ComposerConvergenceError, ComposerResult
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
)


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None, nodes=(), edges=(), outputs=(),
        metadata=PipelineMetadata(), version=1,
    )


def _mock_catalog() -> MagicMock:
    """Mock CatalogService with real PluginSummary/PluginSchemaInfo instances.

    AC #16: Tests must use real PluginSummary and PluginSchemaInfo instances,
    not plain dicts. Mock return types must match the CatalogService protocol.
    """
    catalog = MagicMock()
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="uppercase", description="Uppercase", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="csv", description="CSV sink", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv", plugin_type="source", description="CSV source",
        json_schema={"title": "Config", "properties": {}},
    )
    return catalog


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock LiteLLM response."""
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()

    message.content = content
    message.tool_calls = None

    if tool_calls:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = json.dumps(tc["arguments"])
            mock_tool_calls.append(mock_tc)
        message.tool_calls = mock_tool_calls

    choice.message = message
    response.choices = [choice]
    return response


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.composer_model = "gpt-4o"
    settings.composer_max_turns = 20
    settings.composer_timeout_seconds = 120.0
    return settings


class TestComposerTextOnlyResponse:
    @pytest.mark.asyncio
    async def test_text_only_returns_immediately(self) -> None:
        """LLM responds with text only — no tool calls, loop terminates."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(
            catalog=catalog, settings=settings,
        )
        state = _empty_state()

        llm_response = _make_llm_response(content="I'll help you build a pipeline!")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            result = await service.compose("Build me a CSV pipeline", [], state)

        assert isinstance(result, ComposerResult)
        assert result.message == "I'll help you build a pipeline!"
        assert result.state.version == 1  # unchanged


class TestComposerSingleToolCall:
    @pytest.mark.asyncio
    async def test_single_tool_call_then_text(self) -> None:
        """LLM makes one tool call, then responds with text."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: tool call to set_source
        tool_response = _make_llm_response(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "name": "set_source",
                "arguments": {
                    "plugin": "csv",
                    "on_success": "t1",
                    "options": {"path": "/data.csv"},
                    "on_validation_failure": "quarantine",
                },
            }],
        )
        # Turn 2: text response
        text_response = _make_llm_response(content="I've set up a CSV source.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [tool_response, text_response]
            result = await service.compose("Use CSV as source", [], state)

        assert result.message == "I've set up a CSV source."
        assert result.state.source is not None
        assert result.state.source.plugin == "csv"
        assert result.state.version == 2


class TestComposerMultiTurnToolCalls:
    @pytest.mark.asyncio
    async def test_multi_turn_state_accumulates(self) -> None:
        """Multiple tool calls across turns — state accumulates."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source
        turn1 = _make_llm_response(
            tool_calls=[{
                "id": "call_1",
                "name": "set_source",
                "arguments": {
                    "plugin": "csv", "on_success": "t1",
                    "options": {}, "on_validation_failure": "quarantine",
                },
            }],
        )
        # Turn 2: set_metadata
        turn2 = _make_llm_response(
            tool_calls=[{
                "id": "call_2",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "My Pipeline"}},
            }],
        )
        # Turn 3: text
        turn3 = _make_llm_response(content="Pipeline configured.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [turn1, turn2, turn3]
            result = await service.compose("Build a pipeline", [], state)

        assert result.state.source is not None
        assert result.state.metadata.name == "My Pipeline"
        assert result.state.version == 3  # two mutations


class TestComposerConvergence:
    @pytest.mark.asyncio
    async def test_max_turns_exceeded_raises(self) -> None:
        """Loop exceeding max_turns raises ComposerConvergenceError."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_turns = 2  # very low limit
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Every turn makes a tool call — never produces text
        tool_response = _make_llm_response(
            tool_calls=[{
                "id": "call_loop",
                "name": "get_current_state",
                "arguments": {},
            }],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = tool_response
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Loop forever", [], state)
            assert exc_info.value.max_turns == 2


class TestComposerErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_to_llm(self) -> None:
        """Unknown tool name returns error message, LLM can retry."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: invalid tool
        bad_call = _make_llm_response(
            tool_calls=[{
                "id": "call_bad",
                "name": "nonexistent_tool",
                "arguments": {},
            }],
        )
        # Turn 2: text response (self-corrected)
        text = _make_llm_response(content="Sorry, let me try again.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Do something", [], state)

        assert result.message == "Sorry, let me try again."
        # State unchanged — the bad tool call didn't modify anything
        assert result.state.version == 1

    @pytest.mark.asyncio
    async def test_malformed_arguments_returns_error(self) -> None:
        """Malformed tool arguments return error, not crash."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_source with missing required field
        bad_call = _make_llm_response(
            tool_calls=[{
                "id": "call_bad",
                "name": "set_source",
                "arguments": {"plugin": "csv"},  # missing on_success
            }],
        )
        # Turn 2: text
        text = _make_llm_response(content="Fixed.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [bad_call, text]
            result = await service.compose("Setup", [], state)

        assert result.message == "Fixed."


class TestBuildMessages:
    @pytest.mark.asyncio
    async def test_build_messages_returns_new_list(self) -> None:
        """_build_messages must return a new list on every call."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        msgs1 = service._build_messages(None, state, "Hello")
        msgs2 = service._build_messages(None, state, "Hello")

        assert msgs1 is not msgs2  # different list objects
        assert msgs1 == msgs2  # same content


class TestComposerMultipleToolCallsPerTurn:
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_turn(self) -> None:
        """LLM returns multiple tool calls in one response — all executed."""
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: two tool calls in one response
        multi_call = _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "set_source",
                    "arguments": {
                        "plugin": "csv", "on_success": "t1",
                        "options": {}, "on_validation_failure": "quarantine",
                    },
                },
                {
                    "id": "call_2",
                    "name": "set_metadata",
                    "arguments": {"patch": {"name": "Dual Call Pipeline"}},
                },
            ],
        )
        # Turn 2: text
        text = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [multi_call, text]
            result = await service.compose("Setup", [], state)

        assert result.state.source is not None
        assert result.state.metadata.name == "Dual Call Pipeline"
        assert result.state.version == 3  # two mutations
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ComposerServiceImpl**

```python
# src/elspeth/web/composer/service.py
"""ComposerServiceImpl — bounded LLM tool-use loop for pipeline composition.

Uses LiteLLM for provider abstraction. Model configured via
WebSettings.composer_model. Tool calls are executed against
CompositionState + CatalogService.

Layer: L3 (application).
"""
from __future__ import annotations

import json
from typing import Any

import litellm

from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerResult,
)
from elspeth.web.composer.prompts import build_messages
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import (
    CatalogServiceProtocol,
    ToolResult,
    execute_tool,
    get_tool_definitions,
)


class ComposerServiceImpl:
    """LLM-driven pipeline composer.

    Runs a bounded tool-use loop: sends messages to the LLM, executes
    any tool calls against the CompositionState, appends results, and
    repeats until the LLM produces a text-only response or max_turns
    is exceeded.

    Args:
        catalog: CatalogService for discovery tool delegation.
        settings: WebSettings with composer_model, composer_max_turns,
            composer_timeout_seconds.
    """

    def __init__(
        self,
        catalog: CatalogServiceProtocol,
        settings: Any,
    ) -> None:
        self._catalog = catalog
        self._model = settings.composer_model
        self._max_turns = settings.composer_max_turns

    async def compose(
        self,
        message: str,
        messages: list[Any],  # list[ChatMessageRecord]
        state: CompositionState,
    ) -> ComposerResult:
        """Run the LLM composition loop.

        Args:
            message: The user's chat message.
            messages: Pre-fetched chat history (from route handler).
                ComposerService does NOT depend on SessionService.
            state: The current CompositionState.

        Returns:
            ComposerResult with assistant message and updated state.

        Raises:
            ComposerConvergenceError: If the loop exceeds max_turns.
        """
        llm_messages = self._build_messages(messages, state, message)
        tools = self._get_litellm_tools()

        for _turn in range(self._max_turns):
            response = await self._call_llm(llm_messages, tools)
            assistant_message = response.choices[0].message

            # If no tool calls, the LLM is done — return text response
            if not assistant_message.tool_calls:
                return ComposerResult(
                    message=assistant_message.content or "",
                    state=state,
                )

            # Append the assistant message (with tool_calls metadata)
            llm_messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            })

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    # Malformed arguments — return error to LLM
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({
                            "error": f"Invalid JSON in arguments: {exc}",
                        }),
                    })
                    continue

                try:
                    result = execute_tool(
                        tool_name, arguments, state, self._catalog
                    )
                    # Update state if mutation succeeded
                    state = result.updated_state
                    # Return tool result to LLM
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result.to_dict()),
                    })
                except Exception as exc:
                    # Unexpected error — return to LLM, don't crash
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({
                            "error": f"Tool execution error: {exc}",
                        }),
                    })

        raise ComposerConvergenceError(self._max_turns)

    def _build_messages(
        self,
        chat_history: list[Any],  # list[ChatMessageRecord]
        state: CompositionState,
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Build the message list. Returns a NEW list on every call.

        This is critical: the tool-use loop appends to this list during
        iteration. Returning a cached reference would cause cross-turn
        contamination.
        """
        return build_messages(
            chat_history=chat_history,
            state=state,
            user_message=user_message,
            catalog=self._catalog,
        )

    def _get_litellm_tools(self) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM function format."""
        definitions = get_tool_definitions()
        return [
            {
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn["description"],
                    "parameters": defn["parameters"],
                },
            }
            for defn in definitions
        ]

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Call the LLM via LiteLLM. Separated for test mocking."""
        return await litellm.acompletion(
            model=self._model,
            messages=messages,
            tools=tools,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/service.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_service.py
git commit -m "feat(web/composer): implement ComposerServiceImpl — bounded LLM tool-use loop"
```

---

### Task 8: Wire POST /api/sessions/{id}/messages to ComposerService

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py`

This task depends on Sub-Spec 2 (sessions module) being implemented first. The wiring connects the existing route handler to the ComposerService.

- [ ] **Step 1: Write route integration test**

```python
# tests/unit/web/composer/test_route_integration.py
"""Tests for POST /api/sessions/{id}/messages wiring to ComposerService."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.web.composer.protocol import ComposerResult
from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None, nodes=(), edges=(), outputs=(),
        metadata=PipelineMetadata(), version=1,
    )


class TestMessageRouteComposerWiring:
    """Tests that the route handler correctly calls ComposerService.compose()."""

    @pytest.mark.asyncio
    async def test_first_message_creates_initial_state(self) -> None:
        """First message in a session should create an empty initial state."""
        state = _empty_state()
        # The route handler should construct this empty state when no
        # existing state is found for the session.
        assert state.source is None
        assert state.nodes == ()
        assert state.version == 1

    @pytest.mark.asyncio
    async def test_composer_result_contains_state(self) -> None:
        """ComposerResult includes both message and state."""
        state = _empty_state()
        result = ComposerResult(
            message="Here's your pipeline.",
            state=state.with_source(
                SourceSpec(
                    plugin="csv", on_success="t1",
                    options={}, on_validation_failure="quarantine",
                )
            ),
        )
        assert result.message == "Here's your pipeline."
        assert result.state.source is not None
        assert result.state.version == 2

    @pytest.mark.asyncio
    async def test_state_version_changes_trigger_persistence(self) -> None:
        """If state version changed, the new state should be persisted."""
        original = _empty_state()
        updated = original.with_source(
            SourceSpec(
                plugin="csv", on_success="t1",
                options={}, on_validation_failure="quarantine",
            )
        )
        # Version changed: 1 -> 2 — route handler should persist
        assert updated.version != original.version

    @pytest.mark.asyncio
    async def test_convergence_error_returns_422(self) -> None:
        """ComposerConvergenceError maps to HTTP 422 with structured body."""
        from elspeth.web.composer.protocol import ComposerConvergenceError

        exc = ComposerConvergenceError(max_turns=20)
        assert exc.max_turns == 20
        # The route handler catches this and returns:
        # {error_type: "convergence", message: "...", turns_used: 20}
        # with HTTP 422

    @pytest.mark.asyncio
    async def test_llm_failure_returns_502(self) -> None:
        """LLM client failures map to HTTP 502 with structured body."""
        # LiteLLM network/rate-limit/auth errors propagate to the route
        # handler, which returns:
        # {error_type: "llm_unavailable", message: "..."} with HTTP 502
        # or {error_type: "llm_auth_error", message: "..."} for auth failures
        pass  # Integration test — depends on route handler wiring

    @pytest.mark.asyncio
    async def test_state_unchanged_skips_persistence(self) -> None:
        """If state version unchanged, no persistence needed."""
        original = _empty_state()
        # ComposerResult with same state = no tool calls were made
        result = ComposerResult(message="Just chatting.", state=original)
        assert result.state.version == original.version
```

- [ ] **Step 2: Implement route wiring**

Add to `src/elspeth/web/sessions/routes.py` (the exact location depends on Sub-Spec 2 implementation). The route handler pattern is:

```python
# In the POST /api/sessions/{id}/messages handler:

async def send_message(
    session_id: str,
    body: MessageRequest,
    session_service: SessionService = Depends(get_session_service),
    composer_service: ComposerService = Depends(get_composer_service),
    current_user: UserIdentity = Depends(get_current_user),
) -> MessageResponse:
    """Handle a user message — trigger the LLM composer.

    1. Load session (verify ownership).
    2. Persist user message.
    3. Load current CompositionState (or create empty initial state).
    4. Call ComposerService.compose().
    5. Persist assistant message.
    6. If state changed, persist new state version.
    7. Return {message, state}.
    """
    # 1. Load session
    session = await session_service.get_session(session_id, current_user.id)

    # 2. Persist user message
    user_msg = await session_service.add_message(
        session_id, role="user", content=body.content
    )

    # 3. Load or create composition state
    state = await session_service.get_latest_state(session_id)
    if state is None:
        state = CompositionState(
            source=None, nodes=(), edges=(), outputs=(),
            metadata=PipelineMetadata(), version=1,
        )

    # 4. Call composer — with structured HTTP error handling (S16)
    try:
        # Pre-fetch chat history — ComposerService does NOT depend on
        # SessionService; the route handler mediates (H1 fix, seam contract B).
        chat_messages = await session_service.get_messages(session_id)
        result = await composer_service.compose(body.content, chat_messages, state)
    except ComposerConvergenceError as exc:
        # M2: 422 for convergence errors — use "detail" not "message"
        raise HTTPException(
            status_code=422,
            detail={
                "error_type": "convergence",
                "detail": str(exc),
                "turns_used": exc.max_turns,
            },
        ) from exc
    except Exception as exc:
        # M2: 502 for LLM client failures — use "detail" not "message"
        error_type = "llm_auth_error" if "auth" in str(exc).lower() else "llm_unavailable"
        raise HTTPException(
            status_code=502,
            detail={
                "error_type": error_type,
                "detail": str(exc),
            },
        ) from exc

    # 5. Persist assistant message
    assistant_msg = await session_service.add_message(
        session_id, role="assistant", content=result.message
    )

    # 6. Validate and persist state if changed (M5 fix — is_valid population)
    if result.state.version != state.version:
        summary = result.state.validate()
        await session_service.save_composition_state(
            session_id,
            state=result.state,
            is_valid=summary.is_valid,
            validation_errors=list(summary.errors) if summary.errors else None,
        )

    # 7. Return response — wire field is "state" (H2 fix)
    return MessageResponse(message=assistant_msg, state=result.state)
```

**S18 — Revert system message injection:** When a user reverts to a prior composition version (via Sub-Spec 2's `set_active_state`), the route handler must inject a system message into the chat history: `"Pipeline reverted to version N."` This gives the LLM context that the state has been rolled back. The injected message uses `role="system"` and is persisted as a `ChatMessage` so it appears in the conversation history on subsequent turns. This is handled by the revert endpoint in Sub-Spec 2 -- when `set_active_state` is called, it persists the system message before returning. The `POST /messages` handler here does not need special revert logic because the system message will already be in the chat history by the time the next compose call runs.

- [ ] **Step 3: Implement GET /api/sessions/{id}/state/yaml endpoint**

Add to `src/elspeth/web/sessions/routes.py`:

```python
# GET /api/sessions/{id}/state/yaml — return generated YAML for current state.

async def get_session_yaml(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    current_user: UserIdentity = Depends(get_current_user),
) -> dict[str, str]:
    """Return the generated YAML for the session's current composition state.

    Response: {yaml: str} — the YAML string ready for display in the
    frontend's YAML tab.

    Returns HTTP 404 if the session has no CompositionState yet.
    Authentication and session ownership checks are identical to the
    messages endpoint.
    """
    # Load session (verify ownership)
    session = await session_service.get_session(session_id, current_user.id)

    # Load current state
    state = await session_service.get_latest_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="No composition state for this session.")

    yaml_str = generate_yaml(state)
    return {"yaml": yaml_str}
```

Add a test for this endpoint:

```python
# tests/unit/web/composer/test_route_integration.py (append)

class TestYamlEndpoint:
    @pytest.mark.asyncio
    async def test_yaml_endpoint_returns_yaml_string(self) -> None:
        """GET /api/sessions/{id}/state/yaml returns generated YAML."""
        from elspeth.web.composer.yaml_generator import generate_yaml

        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv", on_success="t1",
                options={"path": "/data.csv"}, on_validation_failure="quarantine",
            )
        ).with_output(
            OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard")
        )
        yaml_str = generate_yaml(state)
        assert "csv" in yaml_str
        assert isinstance(yaml_str, str)

    @pytest.mark.asyncio
    async def test_yaml_endpoint_404_when_no_state(self) -> None:
        """GET /api/sessions/{id}/state/yaml returns 404 when no state exists."""
        # When session_service.get_latest_state() returns None,
        # the endpoint should return HTTP 404.
        pass  # Integration test — depends on Sub-Spec 2 session service
```

- [ ] **Step 4: Run all composer tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Run mypy on all composer modules**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/routes.py tests/unit/web/composer/test_route_integration.py
git commit -m "feat(web/composer): wire POST /api/sessions/{id}/messages to ComposerService"
```

---

## Verification Checklist

After all tasks, verify:

1. `CompositionState` is frozen. All container fields are deep-frozen via `freeze_fields()`. Mutations return new instances. `pytest tests/unit/web/composer/test_state.py`
2. `CompositionState.to_dict()` recursively unwraps `MappingProxyType` -> `dict` and `tuple` -> `list`. Frozen state objects serialize through `yaml.dump()` without `RepresenterError`.
3. `ToolResult` is frozen with `affected_nodes` deep-frozen. `pytest tests/unit/web/composer/test_tools.py`
4. All 12 tools (6 discovery, 6 mutation) work. Mutations return `ToolResult` with validation. Invalid input returns `success=False`, not exceptions.
5. Stage 1 validation catches all 8 check categories.
6. YAML generator is deterministic. Linear, gate, aggregation, fork/coalesce pipelines produce correct YAML. `generate_yaml()` calls `state.to_dict()` before `yaml.dump()`.
7. `ComposerServiceImpl` runs bounded loop. Mock LLM tests cover: text-only, single tool call, multi-turn, convergence error, unknown tool, malformed arguments, multiple tool calls per turn.
8. `_build_messages()` returns a new list on every call.
9. Model is configured via `WebSettings.composer_model`.
10. `GET /api/sessions/{id}/state/yaml` returns the generated YAML string. Returns 404 when no state exists.
11. Route handler catches `ComposerConvergenceError` -> 422 and LLM client errors -> 502 with structured JSON error bodies.
12. Revert system message ("Pipeline reverted to version N.") is injected into chat history when state is reverted (handled by Sub-Spec 2's `set_active_state`).
13. All mock catalogs use real `PluginSummary` and `PluginSchemaInfo` instances, not plain dicts (AC #16).

```bash
# Full test suite
.venv/bin/python -m pytest tests/unit/web/composer/ -v

# Type checking
.venv/bin/python -m mypy src/elspeth/web/composer/

# Freeze guard CI check
.venv/bin/python scripts/cicd/enforce_freeze_guards.py
```

---

## Round 4 Review Amendments

> **Status: Amendments below have been integrated into the plan body.**

The following changes were identified during the Round 4 expert panel review. Fix IDs reference the review finding codes.

1. **compose() signature change (H1, seam contract B).** The `ComposerService` protocol signature changes from `compose(message, session, state)` to `compose(message, messages, state)` where `messages: list[ChatMessageRecord]` is pre-fetched chat history. The route handler calls `session_service.get_messages(session_id)` and passes the result to `compose()`. `ComposerService` has no dependency on `SessionService`. Affects:
   - Task 5 (ComposerService protocol): update `compose()` signature and docstring.
   - Task 6 (ComposerServiceImpl): accept `messages` parameter, use it instead of fetching history internally.
   - Task 8 (route handler wiring): call `session_service.get_messages(session_id)` before invoking `compose()`, pass result as `messages`.

2. **from_dict() factory methods (C2, seam contract A).** Add `from_dict()` class methods to `CompositionState`, `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, and `PipelineMetadata`. These are the inverse of `to_dict()` and must satisfy the round-trip invariant: `CompositionState.from_dict(s.to_dict()) == s`. Affects:
   - Task 1 (state.py): implement `from_dict()` on all six dataclasses.
   - Task 1 (test_state.py): add round-trip tests for each dataclass — `from_dict(x.to_dict()) == x`.
   - New AC #17: `from_dict()` exists on all six dataclasses.
   - New AC #18: round-trip invariant holds for all six dataclasses.

3. **landscape_url removed from PipelineMetadata (C3/S1).** The `landscape_url` field is removed from `PipelineMetadata` for security reasons (LLM could be tricked into diverting the audit trail). The YAML generator no longer emits a `landscape` key. Affects:
   - Task 1 (state.py): remove `landscape_url` field from `PipelineMetadata`.
   - Task 4 (yaml_generator.py): do not emit a `landscape` key.
   - Task 4 (test_yaml_generator.py): update expected YAML output — no `landscape` key.

4. **Source path allowlist in set_source tool (C3/S2).** The `set_source()` mutation tool must validate that any `path` or `file` key in source options resolves to a path under `{WebSettings.data_dir}/uploads/`. Returns `success=False` with an error message if the resolved path is outside the allowed directory. Affects:
   - Task 3 (tools.py): add path validation to `set_source()`.
   - Task 3 (test_tools.py): add tests for path traversal rejection and valid paths.

5. **Route handler state persistence contract (M5, seam contract B).** After `compose()` returns, the route handler must: (a) call `state.validate()` to obtain a `ValidationSummary`, (b) pass `is_valid` and `errors` to `session_service.save_composition_state()`. Without this, `is_valid` defaults to `False` and the Execute button never enables. Affects:
   - Task 8 (route handler wiring): add `validate()` call and pass results to `save_composition_state()`.

6. **Error envelope standardisation (M2, seam contract G).** Composer error responses must use `detail` (not `message`) as the human-readable field, plus `error_type` for domain errors. Specific shapes:
   - 422 convergence: `{"error_type": "convergence", "detail": "...", "turns_used": int}`
   - 502 LLM errors: `{"error_type": "llm_unavailable", "detail": "..."}`
   - Affects: Task 8 (route handler wiring) — update error response bodies. Verification checklist item 11.

7. **Rate limiting on POST /messages (H8p).** Use `WebSettings.composer_rate_limit_per_minute` (default 10) to enforce per-user rate limiting on the messages endpoint. Return HTTP 429 when the limit is exceeded. Implementation: per-user in-memory counter with sliding window. Affects:
   - Task 8 (route handler wiring): add rate limiting middleware/guard to `POST /api/sessions/{id}/messages`.

---

## Round 5 Review Findings

Three-reviewer panel (Reality, Architecture, Quality) examining L3 task-plans 4A-4D.

### Summary

| Task-Plan | Blockers | Warnings | Status |
|-----------|----------|----------|--------|
| 4A (Data Models) | 0 | 2 | Ready |
| 4B (Composition Tools) | 0 | 3 | Ready (dependency correction: 4B does not block 4C) |
| 4C (YAML Generator) | 0 | 2 | Ready (can run parallel with 4B after 4A) |
| 4D (Composer Service) | 4 | 4 | **Fix blockers before execution** |

### Blocking Issues (all in 4D)

| ID | Issue | Fix |
|----|-------|-----|
| B-4D-1 | `_make_settings()` mock missing `data_dir` — S2 path tests meaningless | Add `settings.data_dir = Path("/data")` |
| B-4D-2 | Route HTTP contract tests (422/502) are `pass` stubs | Implement with TestClient + mocked ComposerService |
| B-4D-3 | Off-by-one in max_turns — LLM can't self-correct on final turn | Add bonus LLM call after final tool execution |
| B-4D-4 | 3 protocol call mismatches in route handler template | Fix method names and argument shapes |

### Spec/Contract Fixes Applied

- **B3 (seam contracts):** Updated `seam-contracts.md` Seam B signature from `list[ChatMessageRecord]` to `list[dict[str, Any]]` with conversion rationale.
- **W8 (spec):** `ToolResult.data` field already present in spec (user-applied).
- **W7 (dependency):** 4B does not block 4C — documented in 4B and 4C findings.

### Execution Order (updated)

```
4A (Data Models)
├── 4B (Tools)  ──┐
├── 4C (YAML)  ───┤  ← 4B and 4C can run in parallel
└──────────────────┘
        └── 4D (Composer Service + Route Wiring)  ← depends on 4A, 4B, 4C
```

4D must not start Task 8 (route wiring) until Sub-2E has merged (`sessions/routes.py` must exist).

Detailed findings: see `## Round 5 Review Findings` in each task-plan (4A, 4B, 4C, 4D).
