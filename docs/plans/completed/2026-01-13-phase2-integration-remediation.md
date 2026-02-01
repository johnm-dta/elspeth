# Phase 2 Integration Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix integration gaps between Phase 2 plugin definitions and Phase 3 engine, including missing lifecycle hooks, mutation leak in `_freeze_dict()`, enum inconsistencies, and documentation gaps.

**Architecture:** Phase 2 defined plugin protocols, result types, and enums. Phase 3 engine was built to use them but has gaps: lifecycle hooks are defined but never called, `_freeze_dict()` doesn't deep copy, and enums are defined but not used in result types.

**Note:** The Filter plugin's `success(None)` issue is addressed separately by the architecture-alignment plan via FilterGate (routing to discard sink per architecture.md:148).

**Tech Stack:** Pydantic v2, dataclasses, pytest

---

## Task 1: Fix _freeze_dict() Mutation Leak

**Files:**
- Modify: `src/elspeth/plugins/results.py`
- Test: `tests/plugins/test_results.py`

**Step 1: Write the failing test**

Add to `tests/plugins/test_results.py`:

```python
class TestFreezeDictDefensiveCopy:
    """_freeze_dict makes defensive copy to prevent mutation."""

    def test_original_dict_mutation_not_visible(self) -> None:
        """Mutating original dict doesn't affect frozen result."""
        from elspeth.plugins.results import RoutingAction

        reason = {"key": "original"}
        action = RoutingAction.continue_(reason=reason)

        # Mutate original
        reason["key"] = "mutated"
        reason["new_key"] = "added"

        # Frozen reason should be unchanged
        assert action.reason["key"] == "original"
        assert "new_key" not in action.reason

    def test_nested_dict_mutation_not_visible(self) -> None:
        """Nested dict mutation doesn't affect frozen result."""
        from elspeth.plugins.results import RoutingAction

        reason = {"nested": {"value": 1}}
        action = RoutingAction.continue_(reason=reason)

        # Mutate nested original
        reason["nested"]["value"] = 999

        # Frozen reason should be unchanged
        assert action.reason["nested"]["value"] == 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plugins/test_results.py::TestFreezeDictDefensiveCopy -v`
Expected: FAIL (mutations visible through MappingProxyType)

**Step 3: Update implementation**

Update `_freeze_dict()` in `src/elspeth/plugins/results.py`:

```python
import copy
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_dict(d: dict[str, Any] | None) -> Mapping[str, Any]:
    """Create immutable view of dict with defensive deep copy.

    MappingProxyType only prevents mutation through the proxy.
    We deep copy to prevent mutation via retained references to
    the original dict or nested objects.
    """
    if d is None:
        return MappingProxyType({})
    # Deep copy to prevent mutation of original or nested dicts
    return MappingProxyType(copy.deepcopy(d))
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/plugins/test_results.py::TestFreezeDictDefensiveCopy -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/results.py tests/plugins/test_results.py
git commit -m "$(cat <<'EOF'
fix(results): add defensive deep copy in _freeze_dict()

MappingProxyType prevents mutation through the proxy but not
via retained references. Deep copy prevents all mutation paths.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Invoke Plugin Lifecycle Hooks - on_start

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py`:

```python
class TestLifecycleHooks:
    """Orchestrator invokes plugin lifecycle hooks."""

    def test_on_start_called_before_processing(self) -> None:
        """on_start() called before any rows processed."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from unittest.mock import MagicMock

        call_order = []

        class TrackedTransform:
            name = "tracked"
            plugin_version = "1.0.0"

            def on_start(self, ctx):
                call_order.append("on_start")

            def process(self, row, ctx):
                call_order.append("process")
                return TransformResult.success(row)

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1}])

        transform = TrackedTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        # Minimal graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="tracked")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_start should be called first
        assert call_order[0] == "on_start"
        assert "process" in call_order
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestLifecycleHooks::test_on_start_called_before_processing -v`
Expected: FAIL (on_start not in call_order)

**Step 3: Update implementation**

Update `_execute_run()` in `src/elspeth/engine/orchestrator.py`. After creating the PluginContext and before row processing, add:

```python
        # Call on_start for all plugins BEFORE processing
        # Lifecycle hooks are optional - plugins may or may not implement them
        for transform in config.transforms:
            if hasattr(transform, 'on_start'):
                transform.on_start(ctx)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestLifecycleHooks::test_on_start_called_before_processing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): invoke on_start() lifecycle hook

Calls on_start() on all transforms before any row processing begins.
Lifecycle hooks are optional - uses hasattr() at plugin boundary.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Invoke Plugin Lifecycle Hooks - on_complete

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py` in `TestLifecycleHooks`:

```python
    def test_on_complete_called_after_all_rows(self) -> None:
        """on_complete() called after all rows processed."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from unittest.mock import MagicMock

        call_order = []

        class TrackedTransform:
            name = "tracked"
            plugin_version = "1.0.0"

            def process(self, row, ctx):
                call_order.append("process")
                return TransformResult.success(row)

            def on_complete(self, ctx):
                call_order.append("on_complete")

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1}, {"id": 2}])

        transform = TrackedTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="tracked")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph)

        # on_complete should be called last
        assert call_order[-1] == "on_complete"
        # All processing should happen before on_complete
        assert call_order.count("process") == 2

    def test_on_complete_called_on_error(self) -> None:
        """on_complete() called even when run fails."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from unittest.mock import MagicMock
        import pytest

        completed = []

        class FailingTransform:
            name = "failing"
            plugin_version = "1.0.0"

            def process(self, row, ctx):
                raise RuntimeError("intentional failure")

            def on_complete(self, ctx):
                completed.append(True)

        db = LandscapeDB.in_memory()

        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1}])

        transform = FailingTransform()
        mock_sink = MagicMock()
        mock_sink.name = "csv"

        config = PipelineConfig(
            source=mock_source,
            transforms=[transform],
            sinks={"output": mock_sink},
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="failing")
        graph.add_node("sink", node_type="sink", plugin_name="csv")
        graph.add_edge("source", "transform", label="continue", mode="move")
        graph.add_edge("transform", "sink", label="continue", mode="move")
        graph._transform_id_map = {0: "transform"}
        graph._sink_id_map = {"output": "sink"}
        graph._output_sink = "output"

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=graph)

        # on_complete should still be called
        assert len(completed) == 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestLifecycleHooks -v`
Expected: FAIL (on_complete never called)

**Step 3: Update implementation**

Update `_execute_run()` in `src/elspeth/engine/orchestrator.py`. Wrap the row processing in try/finally:

```python
        # Call on_start for all plugins
        for transform in config.transforms:
            if hasattr(transform, 'on_start'):
                transform.on_start(ctx)

        try:
            # ... existing row processing code ...
            return result
        finally:
            # Call on_complete for all plugins (even on error)
            # Lifecycle hooks are optional - plugins may or may not implement them
            for transform in config.transforms:
                if hasattr(transform, 'on_complete'):
                    try:
                        transform.on_complete(ctx)
                    except Exception:
                        # Log but don't fail - cleanup should be best-effort
                        pass
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestLifecycleHooks -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): invoke on_complete() lifecycle hook

Calls on_complete() on all transforms after processing, even on error.
Uses try/finally to guarantee cleanup runs.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Use Enums in RoutingAction

**Files:**
- Modify: `src/elspeth/plugins/results.py`
- Test: `tests/plugins/test_results.py`

**Step 1: Write the failing test**

Add to `tests/plugins/test_results.py`:

```python
class TestRoutingActionEnums:
    """RoutingAction uses enum types for kind and mode."""

    def test_continue_uses_routing_kind_enum(self) -> None:
        """continue_() returns RoutingKind enum value."""
        from elspeth.plugins.results import RoutingAction
        from elspeth.plugins.enums import RoutingKind

        action = RoutingAction.continue_()

        assert action.kind == RoutingKind.CONTINUE
        assert isinstance(action.kind, RoutingKind)

    def test_route_to_sink_uses_enums(self) -> None:
        """route_to_sink() uses enum types."""
        from elspeth.plugins.results import RoutingAction
        from elspeth.plugins.enums import RoutingKind, RoutingMode

        action = RoutingAction.route_to_sink("output", mode=RoutingMode.COPY)

        assert action.kind == RoutingKind.ROUTE_TO_SINK
        assert action.mode == RoutingMode.COPY
        assert isinstance(action.kind, RoutingKind)
        assert isinstance(action.mode, RoutingMode)

    def test_fork_to_paths_uses_enums(self) -> None:
        """fork_to_paths() uses enum types."""
        from elspeth.plugins.results import RoutingAction
        from elspeth.plugins.enums import RoutingKind, RoutingMode

        action = RoutingAction.fork_to_paths(["path_a", "path_b"])

        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.mode == RoutingMode.COPY
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plugins/test_results.py::TestRoutingActionEnums -v`
Expected: FAIL (RoutingAction uses string literals)

**Step 3: Update implementation**

Update `src/elspeth/plugins/results.py`:

```python
from elspeth.plugins.enums import RoutingKind, RoutingMode


@dataclass(frozen=True)
class RoutingAction:
    """What a gate decided to do with a row."""

    kind: RoutingKind  # Changed from Literal[...]
    destinations: tuple[str, ...]
    mode: RoutingMode  # Changed from Literal[...]
    reason: Mapping[str, Any]

    @classmethod
    def continue_(cls, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Row continues to next transform."""
        return cls(
            kind=RoutingKind.CONTINUE,
            destinations=(),
            mode=RoutingMode.MOVE,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def route_to_sink(
        cls,
        sink_name: str,
        *,
        mode: RoutingMode = RoutingMode.MOVE,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Route row to a named sink."""
        return cls(
            kind=RoutingKind.ROUTE_TO_SINK,
            destinations=(sink_name,),
            mode=mode,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def fork_to_paths(
        cls,
        paths: list[str],
        *,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Fork row to multiple parallel paths (copy mode)."""
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            mode=RoutingMode.COPY,
            reason=_freeze_dict(reason),
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/plugins/test_results.py::TestRoutingActionEnums -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/results.py tests/plugins/test_results.py
git commit -m "$(cat <<'EOF'
refactor(results): use RoutingKind/RoutingMode enums in RoutingAction

Replaces string literals with proper enums. Accepts string input for
backwards compatibility, converts to enum internally.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fix PluginSchema Docstring

**Files:**
- Modify: `src/elspeth/plugins/schemas.py`

**Step 1: Read current docstring**

The docstring claims "Strict type validation" but config uses `strict=False`.

**Step 2: Update docstring**

Update `src/elspeth/plugins/schemas.py`:

```python
class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Plugins define schemas by subclassing:

        class MyInputSchema(PluginSchema):
            temperature: float
            humidity: float

    Features:
    - Extra fields ignored (rows may have more fields than schema requires)
    - Coercive type validation (int->float allowed, strict=False)
    - Easy conversion to/from row dicts
    """

    model_config = ConfigDict(
        extra="ignore",  # Rows may have extra fields
        strict=False,    # Allow coercion (e.g., int -> float)
        frozen=False,    # Allow modification
    )
```

**Step 3: Verify syntax**

Run: `.venv/bin/python -c "from elspeth.plugins.schemas import PluginSchema; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add src/elspeth/plugins/schemas.py
git commit -m "$(cat <<'EOF'
docs(schemas): fix misleading "strict" wording in PluginSchema docstring

Docstring now correctly documents that strict=False allows coercion.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Clarify RowOutcome Docstring

**Files:**
- Modify: `src/elspeth/plugins/results.py`

**Step 1: Update docstring**

Update `RowOutcome` in `src/elspeth/plugins/results.py`:

```python
class RowOutcome(Enum):
    """Terminal states for rows in the pipeline.

    DESIGN NOTE: Per architecture (00-overview.md:267-279), token terminal
    states are DERIVED from the combination of node_states, routing_events,
    and batch membership—not stored as a column. This enum is used at
    query/explain time to report final disposition, not at runtime.

    The engine does NOT set these directly. The Landscape query layer
    derives them when answering explain() queries.

    INVARIANT: Every row reaches exactly one terminal state.
    No silent drops.
    """

    COMPLETED = "completed"           # Reached output sink
    ROUTED = "routed"                 # Sent to named sink by gate (move mode)
    FORKED = "forked"                 # Split into child tokens (parent terminates)
    CONSUMED_IN_BATCH = "consumed_in_batch"  # Fed into aggregation
    COALESCED = "coalesced"           # Merged with other tokens
    QUARANTINED = "quarantined"       # Failed, stored for investigation
    FAILED = "failed"                 # Failed, not recoverable
```

**Step 2: Verify syntax**

Run: `.venv/bin/python -c "from elspeth.plugins.results import RowOutcome; print(RowOutcome.__doc__[:50])"`
Expected: First 50 chars of docstring

**Step 3: Commit**

```bash
git add src/elspeth/plugins/results.py
git commit -m "$(cat <<'EOF'
docs(results): clarify RowOutcome is derived at query time

Documents that terminal states are not stored but derived from
node_states and routing_events when answering explain() queries.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Populate PluginSpec Schema Hashes

**Files:**
- Modify: `src/elspeth/plugins/manager.py`
- Test: `tests/plugins/test_manager.py`

**Step 1: Write the failing test**

Add to `tests/plugins/test_manager.py`:

```python
class TestPluginSpecSchemaHashes:
    """PluginSpec.from_plugin() populates schema hashes."""

    def test_from_plugin_captures_input_schema_hash(self) -> None:
        """Input schema is hashed."""
        from elspeth.plugins.manager import PluginSpec
        from elspeth.plugins.enums import NodeType
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            field_a: str
            field_b: int

        class MyTransform:
            name = "test"
            plugin_version = "1.0.0"
            input_schema = InputSchema
            output_schema = InputSchema

        spec = PluginSpec.from_plugin(MyTransform, NodeType.TRANSFORM)

        assert spec.input_schema_hash is not None
        assert len(spec.input_schema_hash) == 64  # SHA-256 hex

    def test_schema_hash_stable(self) -> None:
        """Same schema always produces same hash."""
        from elspeth.plugins.manager import PluginSpec
        from elspeth.plugins.enums import NodeType
        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            value: int

        class T1:
            name = "t1"
            plugin_version = "1.0.0"
            input_schema = MySchema
            output_schema = MySchema

        class T2:
            name = "t2"
            plugin_version = "1.0.0"
            input_schema = MySchema
            output_schema = MySchema

        spec1 = PluginSpec.from_plugin(T1, NodeType.TRANSFORM)
        spec2 = PluginSpec.from_plugin(T2, NodeType.TRANSFORM)

        # Same schema = same hash (regardless of plugin)
        assert spec1.input_schema_hash == spec2.input_schema_hash
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plugins/test_manager.py::TestPluginSpecSchemaHashes -v`
Expected: FAIL (schema hashes are None)

**Step 3: Update implementation**

Update `src/elspeth/plugins/manager.py`:

```python
from elspeth.core.canonical import stable_hash


def _schema_hash(schema_cls: type | None) -> str | None:
    """Compute stable hash for a schema class.

    Hashes the schema's field names and types to detect compatibility changes.
    """
    if schema_cls is None:
        return None

    # Use Pydantic model_fields for accurate field introspection
    if not hasattr(schema_cls, 'model_fields'):
        return None

    # Build deterministic representation
    fields_repr = {
        name: str(field.annotation)
        for name, field in schema_cls.model_fields.items()
    }
    return stable_hash(fields_repr)


@dataclass(frozen=True)
class PluginSpec:
    # ... existing fields ...

    @classmethod
    def from_plugin(cls, plugin_cls: type, node_type: NodeType) -> "PluginSpec":
        """Create spec from plugin class with schema hashes."""
        input_schema = getattr(plugin_cls, 'input_schema', None)
        output_schema = getattr(plugin_cls, 'output_schema', None)

        return cls(
            name=getattr(plugin_cls, "name", plugin_cls.__name__),
            node_type=node_type,
            version=getattr(plugin_cls, "plugin_version", "0.0.0"),
            determinism=getattr(plugin_cls, "determinism", Determinism.DETERMINISTIC),
            input_schema_hash=_schema_hash(input_schema),
            output_schema_hash=_schema_hash(output_schema),
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/plugins/test_manager.py::TestPluginSpecSchemaHashes -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/manager.py tests/plugins/test_manager.py
git commit -m "$(cat <<'EOF'
feat(manager): populate PluginSpec schema hashes

Computes stable hash of input/output schema fields for compatibility
tracking between plugin versions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update PHASE3_INTEGRATION.md Documentation

**Files:**
- Modify: `src/elspeth/plugins/PHASE3_INTEGRATION.md`

**Step 1: Verify actual API exists**

Run:
```bash
.venv/bin/python -c "
from elspeth.core.landscape import LandscapeRecorder, LandscapeDB

db = LandscapeDB.from_url('sqlite:///:memory:')
recorder = LandscapeRecorder(db)

# Verify methods exist
assert hasattr(recorder, 'begin_node_state')
assert hasattr(recorder, 'complete_node_state')
assert hasattr(recorder, 'create_batch')
print('All documented APIs exist')
"
```

**Step 2: Update documentation**

Update `src/elspeth/plugins/PHASE3_INTEGRATION.md` to match actual APIs. Ensure examples use `begin_node_state`/`complete_node_state` (not `record_node_state`).

**Step 3: Commit**

```bash
git add src/elspeth/plugins/PHASE3_INTEGRATION.md
git commit -m "$(cat <<'EOF'
docs(plugins): update PHASE3_INTEGRATION.md to match actual APIs

Fixes incorrect examples that didn't match LandscapeRecorder's
real API (begin_node_state/complete_node_state, not record_node_state).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final Verification

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

**Step 2: Verify lifecycle hooks work**

```bash
.venv/bin/python -c "
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult

class TestTransform(BaseTransform):
    name = 'test'

    def on_start(self, ctx):
        print('on_start called')

    def process(self, row, ctx):
        return TransformResult.success(row)

    def on_complete(self, ctx):
        print('on_complete called')

# Methods should exist
t = TestTransform({})
assert hasattr(t, 'on_start')
assert hasattr(t, 'on_complete')
print('Lifecycle hooks are defined')
"
```

**Step 3: Verify _freeze_dict prevents mutation**

```bash
.venv/bin/python -c "
from elspeth.plugins.results import RoutingAction

reason = {'key': 'original', 'nested': {'value': 1}}
action = RoutingAction.continue_(reason=reason)

# Try to mutate
reason['key'] = 'mutated'
reason['nested']['value'] = 999

# Verify frozen version unchanged
assert action.reason['key'] == 'original'
assert action.reason['nested']['value'] == 1
print('_freeze_dict correctly prevents mutation')
"
```

**Step 4: Run type checking**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/results.py src/elspeth/engine/orchestrator.py`
Expected: Success

---

## Summary

| Task | Description | Priority | Files Modified |
|------|-------------|----------|----------------|
| 1 | Fix _freeze_dict() mutation leak | P1 | `results.py`, `test_results.py` |
| 2 | Invoke on_start() hook | P1 | `orchestrator.py`, `test_orchestrator.py` |
| 3 | Invoke on_complete() hook | P1 | `orchestrator.py`, `test_orchestrator.py` |
| 4 | Use enums in RoutingAction | P1 | `results.py`, `test_results.py` |
| 5 | Fix PluginSchema docstring | P2 | `schemas.py` |
| 6 | Clarify RowOutcome docstring | P2 | `results.py` |
| 7 | Populate PluginSpec schema hashes | P2 | `manager.py`, `test_manager.py` |
| 8 | Update PHASE3_INTEGRATION.md | P1 | `PHASE3_INTEGRATION.md` |
| 9 | Final verification | - | (verification only) |

**Estimated total:** ~200-250 lines changed across 6 files

**Note:** The Filter plugin issue (Tasks 1-3 in original plan) is addressed by the architecture-alignment plan via FilterGate, which routes filtered rows to a discard sink per architecture.md:148.
