# Web UX Task-Plan 4A: Data Models & CompositionState

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement all Spec types, CompositionState with freeze_fields/to_dict/from_dict/mutation methods, and Stage 1 validation
**Parent Plan:** `plans/2026-03-28-web-ux-sub4-composer.md`
**Spec:** `specs/2026-03-28-web-ux-sub4-composer-design.md`
**Depends On:** Sub-Plan 1 (Foundation) — completed
**Blocks:** Task-Plan 4B (Composition Tools)

---

## File Map

| Action | File |
|--------|------|
| Create | `src/elspeth/web/composer/__init__.py` |
| Create | `src/elspeth/web/composer/state.py` |
| Create | `tests/unit/web/composer/__init__.py` |
| Create | `tests/unit/web/composer/test_state.py` |

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

    def test_from_dict_round_trip(self) -> None:
        s = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"key": "val"}},
            on_validation_failure="quarantine",
        )
        restored = SourceSpec.from_dict({
            "plugin": "csv",
            "on_success": "t1",
            "options": {"nested": {"key": "val"}},
            "on_validation_failure": "quarantine",
        })
        assert restored == s


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

    def test_from_dict_with_optional_fields(self) -> None:
        """from_dict reconstructs optional fields; missing ones default to None."""
        d = {
            "id": "g1", "node_type": "gate", "plugin": None,
            "input": "in", "on_success": None, "on_error": None,
            "options": {},
            "condition": "row['x'] > 1",
            "routes": {"high": "s1"},
            "fork_to": ["path_a", "path_b"],
        }
        n = NodeSpec.from_dict(d)
        assert n.condition == "row['x'] > 1"
        assert n.fork_to == ("path_a", "path_b")
        assert n.branches is None
        assert n.policy is None
        assert n.merge is None

    def test_from_dict_converts_list_to_tuple(self) -> None:
        """to_dict() serialises tuples as lists; from_dict() must convert back."""
        d = {
            "id": "c1", "node_type": "coalesce", "plugin": None,
            "input": "join", "on_success": "out", "on_error": None,
            "options": {},
            "branches": ["a", "b"],
            "policy": "require_all", "merge": "nested",
        }
        n = NodeSpec.from_dict(d)
        assert isinstance(n.branches, tuple)
        assert n.branches == ("a", "b")


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

    def test_from_dict_round_trip(self) -> None:
        e = EdgeSpec(
            id="e1", from_node="source", to_node="t1",
            edge_type="on_success", label="main",
        )
        restored = EdgeSpec.from_dict({
            "id": "e1", "from_node": "source", "to_node": "t1",
            "edge_type": "on_success", "label": "main",
        })
        assert restored == e


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

    def test_from_dict_round_trip(self) -> None:
        o = OutputSpec(
            name="out", plugin="csv",
            options={"path": "/out.csv"}, on_write_failure="quarantine",
        )
        restored = OutputSpec.from_dict({
            "name": "out", "plugin": "csv",
            "options": {"path": "/out.csv"}, "on_write_failure": "quarantine",
        })
        assert restored == o


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

    def test_from_dict_round_trip(self) -> None:
        m = PipelineMetadata(name="My Pipeline", description="Desc")
        restored = PipelineMetadata.from_dict({
            "name": "My Pipeline", "description": "Desc",
        })
        assert restored == m

    def test_from_dict_uses_defaults_for_missing_fields(self) -> None:
        """Missing fields fall back to dataclass defaults."""
        restored = PipelineMetadata.from_dict({})
        assert restored.name == "Untitled Pipeline"
        assert restored.description == ""


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
from typing import Any, Self

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class PipelineMetadata:
    """Pipeline-level metadata.

    All fields are scalars or None. frozen=True is sufficient.
    """

    name: str = "Untitled Pipeline"
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            name=d.get("name", "Untitled Pipeline"),
            description=d.get("description", ""),
        )


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            plugin=d["plugin"],
            on_success=d["on_success"],
            options=d["options"],
            on_validation_failure=d["on_validation_failure"],
        )


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation).

        Optional fields (condition, routes, fork_to, branches, policy, merge)
        default to None when absent from the dict. fork_to and branches are
        converted from list to tuple since to_dict() serialises tuples as lists.
        """
        fork_to = d.get("fork_to")
        branches = d.get("branches")
        return cls(
            id=d["id"],
            node_type=d["node_type"],
            plugin=d["plugin"],
            input=d["input"],
            on_success=d["on_success"],
            on_error=d["on_error"],
            options=d["options"],
            condition=d.get("condition"),
            routes=d.get("routes"),
            fork_to=tuple(fork_to) if fork_to is not None else None,
            branches=tuple(branches) if branches is not None else None,
            policy=d.get("policy"),
            merge=d.get("merge"),
        )


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            id=d["id"],
            from_node=d["from_node"],
            to_node=d["to_node"],
            edge_type=d["edge_type"],
            label=d["label"],
        )


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation)."""
        return cls(
            name=d["name"],
            plugin=d["plugin"],
            options=d["options"],
            on_write_failure=d["on_write_failure"],
        )


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

    # --- from_dict round-trip ---

    def test_from_dict_round_trip_empty(self) -> None:
        """Empty state round-trips through to_dict/from_dict."""
        state = self._empty_state()
        restored = CompositionState.from_dict(state.to_dict())
        assert restored == state

    def test_from_dict_round_trip_fully_populated(self) -> None:
        """Fully populated state round-trips through to_dict/from_dict.

        This is the Seam A invariant: state == CompositionState.from_dict(state.to_dict()).
        Covers nested MappingProxyType fields, optional None fields, and tuple fields.
        """
        gate = NodeSpec(
            id="gate_1",
            node_type="gate",
            plugin=None,
            input="source_out",
            on_success=None,
            on_error=None,
            options={},
            condition="row['score'] >= 0.5",
            routes={"high": "sink_good", "low": "sink_bad"},
            fork_to=("path_a", "path_b"),
            branches=None,
            policy=None,
            merge=None,
        )
        coalesce = NodeSpec(
            id="coal_1",
            node_type="coalesce",
            plugin=None,
            input="join_point",
            on_success="main_output",
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=("path_a", "path_b"),
            policy="require_all",
            merge="nested",
        )
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="transform_1",
                options={"path": "/data/in.csv", "nested": {"key": "val"}},
                on_validation_failure="quarantine",
            ),
            nodes=(self._make_node("transform_1"), gate, coalesce),
            edges=(
                self._make_edge("e1"),
                EdgeSpec(
                    id="e2",
                    from_node="gate_1",
                    to_node="sink_good",
                    edge_type="route_true",
                    label="high",
                ),
            ),
            outputs=(
                self._make_output("main_output"),
                OutputSpec(
                    name="sink_good",
                    plugin="json",
                    options={"indent": 2},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(
                name="Test Pipeline",
                description="A fully populated test state",
            ),
            version=42,
        )
        restored = CompositionState.from_dict(state.to_dict())
        assert restored == state

    def test_from_dict_round_trip_none_optional_fields(self) -> None:
        """NodeSpec optional fields omitted by to_dict() reconstruct as None."""
        node = self._make_node("t1")
        state = self._empty_state().with_node(node)
        restored = CompositionState.from_dict(state.to_dict())
        restored_node = restored.nodes[0]
        assert restored_node.condition is None
        assert restored_node.routes is None
        assert restored_node.fork_to is None
        assert restored_node.branches is None
        assert restored_node.policy is None
        assert restored_node.merge is None

    def test_from_dict_containers_are_frozen(self) -> None:
        """from_dict() output has deep-frozen containers (not plain dicts)."""
        state = self._empty_state()
        src = SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"nested": {"k": "v"}},
            on_validation_failure="discard",
        )
        state = state.with_source(src)
        restored = CompositionState.from_dict(state.to_dict())
        with pytest.raises(TypeError):
            restored.source.options["new"] = "x"  # type: ignore[union-attr, index]
        with pytest.raises(TypeError):
            restored.source.options["nested"]["mutate"] = "y"  # type: ignore[union-attr, index]
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from a plain dict (inverse of to_dict serialisation).

        Calls from_dict() on each nested Spec type. This is the only way
        to construct CompositionState from deserialised JSON (Spec AC #18).
        The round-trip invariant holds:
            state == CompositionState.from_dict(state.to_dict())
        """
        source_data = d["source"]
        return cls(
            source=SourceSpec.from_dict(source_data) if source_data is not None else None,
            nodes=tuple(NodeSpec.from_dict(n) for n in d["nodes"]),
            edges=tuple(EdgeSpec.from_dict(e) for e in d["edges"]),
            outputs=tuple(OutputSpec.from_dict(o) for o in d["outputs"]),
            metadata=PipelineMetadata.from_dict(d["metadata"]),
            version=d["version"],
        )

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

## Self-Review Checklist

- [ ] All six frozen dataclasses (`PipelineMetadata`, `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `ValidationSummary`) are implemented with `frozen=True, slots=True`
- [ ] Container fields (`options`, `routes`) use `freeze_fields()` in `__post_init__`
- [ ] `CompositionState` mutation methods (`with_source`, `with_node`, `without_node`, `with_edge`, `without_edge`, `with_output`, `without_output`, `with_metadata`) return new instances via `replace()` and increment version
- [ ] `with_node` checks existence first, then builds tuple once (no redundant filter+append; Amendment R5)
- [ ] `without_node` cascades edge removal for edges referencing the removed node
- [ ] `without_*` methods return `None` when the target does not exist
- [ ] `to_dict()` recursively unwraps `MappingProxyType` to `dict` and `tuple` to `list`
- [ ] `from_dict()` class methods on all `*Spec` types and `CompositionState` reconstruct frozen dataclass instances from plain dicts
- [ ] Round-trip invariant holds: `state == CompositionState.from_dict(state.to_dict())` for fully populated states
- [ ] `to_dict()` output round-trips through `yaml.dump()` without `RepresenterError`
- [ ] `validate()` checks: source exists, outputs exist, edge references valid, unique IDs, node-type field consistency, connection completeness
- [ ] All tests pass: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
- [ ] mypy passes: `.venv/bin/python -m mypy src/elspeth/web/composer/state.py`
- [ ] No defensive programming patterns (no `.get()` on typed fields, no `getattr` with defaults)
- [ ] Layer dependency respected: `state.py` imports from L0 (`contracts.freeze`) only

---

## Round 4 Review Amendments

### Added `from_dict()` factory methods (2026-03-28)

**Reason:** Seam A requires the round-trip invariant `state == CompositionState.from_dict(state.to_dict())`. Sub-2's `from_record()` calls `CompositionState.from_dict()` to reconstruct domain objects from deserialised JSON. Spec AC #17 and #18 mandate `from_dict()` on every `*Spec` type and `CompositionState`.

**Changes:**

- Added `from typing import Self` to the state module imports
- Added `@classmethod from_dict(cls, d: dict[str, Any]) -> Self` to: `PipelineMetadata`, `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `CompositionState`
- `NodeSpec.from_dict()` handles optional fields that `to_dict()` conditionally omits (defaulting to `None`) and converts lists back to tuples for `fork_to` and `branches`
- `PipelineMetadata.from_dict()` uses `d.get()` with dataclass defaults for missing fields (this is the one legitimate use of `.get()` -- the dict comes from `to_dict()` which always includes all fields, but the spec says "missing fields use the dataclass defaults" for forward compatibility)
- `CompositionState.from_dict()` delegates to each nested Spec type's `from_dict()`
- Added per-type `from_dict` tests in `TestSourceSpec`, `TestNodeSpec`, `TestEdgeSpec`, `TestOutputSpec`, `TestPipelineMetadata`
- Added round-trip invariant tests in `TestCompositionState`: empty state, fully populated state (gate + coalesce + nested options), None optional field preservation, and deep-freeze verification on restored instances
- Updated self-review checklist with `from_dict()` and round-trip invariant items

### Fixed `with_node` redundant work on replace path (2026-03-29)

**Reason:** Go/no-go review identified that `with_node` built a filtered+appended
tuple on line 1 that was thrown away when the replace branch fired on line 3-7.
The replace branch reconstructed from `self.nodes` (the original), so the first
line's work was wasted.

**Fix:** Restructured to check existence first, then build the appropriate tuple
once: replace-at-position for existing IDs, append for new IDs. No redundant
intermediate tuple construction.

---

## Round 5 Review Findings

**Warnings (implement during execution):**

- **W-4A-1: `from_dict()` with extra unknown keys.** `NodeSpec.from_dict()` silently ignores extra keys in the input dict. This is correct for forward compatibility (a future schema version might add fields) but means malformed dicts won't crash. Since this is Tier 1 data (from our DB), consider adding a `_KNOWN_FIELDS` frozenset check that raises on unexpected keys -- but only if the round-trip invariant test covers this case. Low priority.

- **W-4A-2: `validate()` after `from_dict()` round-trip not tested.** The round-trip tests compare `from_dict(s.to_dict()) == s` but don't call `validate()` on the reconstructed state. A state that round-trips correctly but fails validation would indicate a serialisation bug. Add `assert reconstructed.validate()[0]` to the round-trip test for valid states.

**No blocking issues for 4A.** The data model plan is clean.
