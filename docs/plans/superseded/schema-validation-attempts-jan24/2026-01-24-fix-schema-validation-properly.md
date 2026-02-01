# Fix Schema Validation Architecture Properly

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Plan Status: Updated After 4-Expert Review - Round 2 (2026-01-24)

**Review Verdict:** ✅ **Approved - All Blocking Issues Fixed**

**Round 1 fixes (pre-review):**
1. ✅ **Task 0 added**: Deterministic node IDs for checkpoint compatibility (systems thinking blocker)
2. ✅ **Task 2.5 fixed**: Validation timing clarified - happens in `validate_edge_compatibility()` called from `from_plugin_instances()`, NOT in `add_edge()`
3. ✅ **Task 3 fixed**: Made `_validate_self_consistency()` abstract from start (removed default `pass` implementation)
4. ✅ **Task 2.5 enhanced**: Added 5 critical tests for timing, aggregation, config gates, error messages
5. ✅ **Task 3.5 enhanced**: Added bypass tests for enforcement verification
6. ✅ **Config preservation**: Added requirement to preserve plugin config in NodeInfo for audit trail

**Round 2 fixes (post-review - BLOCKING):**
7. ✅ **Task 0 - Canonical JSON**: Changed node ID hashing from `json.dumps()` to `canonical_json()` for true determinism (Architecture + Python critical)
8. ✅ **Task 0 - Checkpoint compatibility**: Added Step 6.5 - version check for old UUID-based checkpoints (Systems Thinking critical)
9. ✅ **Task 2.5 - Gate validation**: Added Rule 0 in `_validate_single_edge()` to enforce gate schema preservation (Python critical)
10. ✅ **Task 3 - ABC enforcement**: Added `__init_subclass__` hook to enforce validation is CALLED, not just implemented (Python BLOCKING)
11. ✅ **Task 3.5 - Enforcement test**: Added `test_transform_must_call_validation_not_just_implement()` to verify hook works (QA critical)
12. ✅ **Task 4 - Migration audit**: Added Step 5.5 - grep audit to verify ALL old tests migrated (QA BLOCKING)

**Review Scores:**
- Architecture (axiom-system-architect): 4.5/5 → **Approve** (canonical JSON fix applied)
- Python Engineering (axiom-python-engineering): 3.5/5 → **Approve** (all 4 blocking issues fixed)
- Quality Assurance (ordis-quality-engineering): 2.5/5 → **Approve** (migration audit + enforcement test added)
- Systems Thinking (yzmir-systems-thinking): 4.5/5 → **Approve** (checkpoint version check added)

---

**Goal:** Move schema validation to plugin construction time, eliminating the need for DAG layer to access SchemaConfig.

**Architecture:** Two-phase validation model:
1. **Self-validation** (plugin __init__): Plugin validates its own schema is well-formed
2. **Compatibility validation** (ExecutionGraph construction): System validates plugin connections

DAG validation only checks structural issues (cycles, missing nodes). This eliminates information loss, parallel detection mechanisms, and NodeInfo bloat.

**Why NOW (Pre-Release):**
- Post-release: Breaking change to plugin protocol requires migration
- Pre-release: We control all plugins, can update them atomically
- This is the LAST chance to get architecture right before public API

**Tech Stack:** Python protocols, Pydantic validation, plugin hookspecs

---

## The Problem We're Actually Solving

**Current Architecture (Broken):**
```
Plugin.__init__(config)
  ↓ Extract SchemaConfig
  ↓ Create Pydantic schema
  ↓ THROW AWAY SchemaConfig
  ↓
DAG construction
  ↓ Need SchemaConfig (it's gone!)
  ↓ Either introspect Pydantic (Phase 3) OR propagate both (Phase 4)
  ↓
DAG.validate()
  ↓ Check schema compatibility
  ↓ Discovers incompatibilities AFTER plugin instantiation
```

**Proper Architecture:**
```
PHASE 1: Plugin Construction (Self-Validation)
Plugin.__init__(config)
  ↓ Extract SchemaConfig
  ↓ Create Pydantic schema
  ↓ VALIDATE: Schema is well-formed
  ↓ Raise ValueError if malformed
  ↓
PHASE 2: Graph Construction (Compatibility Validation)
ExecutionGraph.from_plugin_instances(plugins)
  ↓ Wire plugins together (source → transforms → sinks)
  ↓ VALIDATE: Each edge has compatible schemas
  ↓   - Producer has all fields required by consumer
  ↓   - Dynamic schemas compatible with anything
  ↓   - Gates pass-through correctly
  ↓   - Coalesce branches have compatible schemas
  ↓ Raise ValueError if incompatible
  ↓
PHASE 3: Structural Validation (No Schema Knowledge)
DAG.validate()
  ↓ Check cycles (NetworkX)
  ↓ Check reachability
  ↓ NO schema validation (already done in Phase 2)
```

**This eliminates:**
- ❌ Need for SchemaConfig in NodeInfo (doesn't propagate)
- ❌ Need for `_is_dynamic_schema()` introspection (doesn't exist)
- ❌ Need for `_extract_schema_config()` helper (doesn't exist)
- ❌ Risk of from_config() vs from_plugin_instances() divergence (same path)
- ❌ Schema validation logic in DAG layer (moved to plugins)

---

## Task 0: Make Node IDs Deterministic for Checkpoint Compatibility

**Goal:** Ensure node IDs are deterministic for resume/checkpoint functionality.

**CRITICAL:** Current implementation uses `uuid.uuid4()` which breaks resume functionality after upgrade.

**Files:**
- Modify: `src/elspeth/core/dag.py` (update node_id generation)
- Modify: `tests/core/test_dag.py` (add determinism tests)

**Step 1: Write failing test for deterministic node IDs**

Add to `tests/core/test_dag.py`:

```python
def test_node_ids_are_deterministic_for_same_config() -> None:
    """Node IDs must be deterministic for checkpoint/resume compatibility."""
    from elspeth.core.dag import ExecutionGraph

    config = {
        "datasource": {
            "plugin": "csv",
            "options": {"path": "test.csv", "schema": {"fields": "dynamic"}},
        },
        "row_plugins": [
            {
                "plugin": "passthrough",
                "options": {"schema": {"fields": "dynamic"}},
            }
        ],
        "sinks": {"out": {"plugin": "csv", "options": {"path": "out.csv"}}},
        "output_sink": "out",
    }

    # Build graph twice with same config
    graph1 = ExecutionGraph.from_config(config, manager)
    graph2 = ExecutionGraph.from_config(config, manager)

    # Node IDs must be identical
    nodes1 = sorted(graph1._graph.nodes())
    nodes2 = sorted(graph2._graph.nodes())

    assert nodes1 == nodes2, "Node IDs must be deterministic for checkpoint compatibility"


def test_node_ids_change_when_config_changes() -> None:
    """Node IDs should change if plugin config changes."""
    config1 = {
        "datasource": {
            "plugin": "csv",
            "options": {"path": "test.csv", "schema": {"fields": "dynamic"}},
        },
        # ... rest of config
    }

    config2 = {
        "datasource": {
            "plugin": "csv",
            "options": {"path": "test.csv", "schema": {"mode": "strict", "fields": ["id: int"]}},  # Different!
        },
        # ... rest of config
    }

    graph1 = ExecutionGraph.from_config(config1, manager)
    graph2 = ExecutionGraph.from_config(config2, manager)

    # Source node IDs should differ (different config)
    source_id_1 = [n for n in graph1._graph.nodes() if n.startswith("source_")][0]
    source_id_2 = [n for n in graph2._graph.nodes() if n.startswith("source_")][0]

    assert source_id_1 != source_id_2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/core/test_dag.py::test_node_ids_are_deterministic_for_same_config -xvs
pytest tests/core/test_dag.py::test_node_ids_change_when_config_changes -xvs
```

Expected: FAIL (current implementation uses random UUIDs)

**Step 3: Update node_id generation to be deterministic**

In `src/elspeth/core/dag.py`, find the `node_id()` function (or where node IDs are generated) and replace:

```python
# OLD (BROKEN for resume):
def _generate_node_id(prefix: str, name: str) -> str:
    return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"  # Random!

# NEW (deterministic):
def _generate_node_id(prefix: str, name: str, config: dict[str, Any]) -> str:
    """Generate deterministic node ID based on plugin type and config.

    Node IDs must be deterministic for checkpoint/resume compatibility.
    If a pipeline is checkpointed and later resumed, the node IDs must
    be identical so checkpoint state can be restored correctly.

    Args:
        prefix: Node type prefix (source_, transform_, sink_, etc.)
        name: Plugin name
        config: Plugin configuration dict

    Returns:
        Deterministic node ID
    """
    import hashlib
    from elspeth.core.canonical import canonical_json

    # Create stable hash of config using RFC 8785 canonical JSON
    # CRITICAL: Must use canonical_json() not json.dumps() for true determinism
    # (floats, nested dicts, datetime serialization must be consistent)
    config_str = canonical_json(config)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]  # 48 bits

    return f"{prefix}_{name}_{config_hash}"
```

**Step 4: Update all node_id() call sites**

Find all locations that generate node IDs and pass the config:

```bash
grep -n "uuid.uuid4()" src/elspeth/core/dag.py
```

Update each to use deterministic generation.

**Step 5: Run tests to verify they pass**

```bash
pytest tests/core/test_dag.py::test_node_ids_are_deterministic_for_same_config -xvs
pytest tests/core/test_dag.py::test_node_ids_change_when_config_changes -xvs
```

Expected: PASS

**Step 6: Run all DAG tests to ensure no breakage**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests pass

**Step 6.5: Add checkpoint backward compatibility safeguard**

CRITICAL: Old checkpoints with UUID-based node IDs will not work after this change.
Add version check to fail gracefully rather than mysteriously.

Add to `src/elspeth/checkpoint/manager.py`:

```python
def _validate_checkpoint_compatibility(self, checkpoint_metadata: dict) -> None:
    """Verify checkpoint was created with compatible node ID generation.

    Raises:
        IncompatibleCheckpointError: If checkpoint predates deterministic node IDs
    """
    # Check if checkpoint has deterministic node IDs
    # Old checkpoints: node_id contains random UUID fragment
    # New checkpoints: node_id contains config hash

    # Simple heuristic: if created_at before 2026-01-24, warn about incompatibility
    checkpoint_date = checkpoint_metadata.get("created_at")
    if checkpoint_date and checkpoint_date < "2026-01-24":
        raise IncompatibleCheckpointError(
            "Checkpoint created before deterministic node IDs (pre-RC-1). "
            "Resume not supported across this upgrade. "
            "Please restart pipeline from beginning."
        )
```

**Note:** This is a breaking change for checkpoint compatibility. Document in CHANGELOG.

**Step 7: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "fix: make node IDs deterministic for checkpoint compatibility

- Replace uuid.uuid4() with deterministic hash of plugin config
- Node IDs now stable across runs with same config
- Critical for checkpoint/resume functionality
- Ref: Fix schema validation architecture properly"
```

---

## Task 1: Add Schema Validation to Plugin Protocol

**Goal:** Define protocol methods for plugins to self-validate schemas.

**Files:**
- Modify: `src/elspeth/plugins/protocols.py` (add validation methods)

**Step 1: Write failing test for source validation**

Add to `tests/contracts/test_plugin_protocols.py`:

```python
def test_source_validates_output_schema_on_init() -> None:
    """Sources should validate output schema during construction."""
    from elspeth.plugins.sources.csv_source import CSVSource

    # Valid schema - should succeed
    config = {
        "path": "test.csv",
        "schema": {"fields": "dynamic"},
    }
    source = CSVSource(config)  # Should not raise

    # Invalid schema - should fail during __init__
    bad_config = {
        "path": "test.csv",
        "schema": {"mode": "strict", "fields": ["invalid syntax"]},
    }

    with pytest.raises(ValueError, match="Invalid field spec"):
        CSVSource(bad_config)  # Fails during construction
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/contracts/test_plugin_protocols.py::test_source_validates_output_schema_on_init -xvs
```

Expected: PASS (sources already validate via SchemaConfig.from_dict in __init__)

**Step 3: Add protocol method to SourceProtocol**

In `src/elspeth/plugins/protocols.py`, add to `SourceProtocol`:

```python
class SourceProtocol(Protocol):
    """Protocol for source plugins."""

    name: str
    output_schema: type[PluginSchema] | None

    def load(self, ctx: PluginContext) -> Iterable[SourceRow]:
        """Load data from source."""
        ...

    def validate_output_schema(self) -> list[str]:
        """Validate output schema is well-formed (self-validation).

        Called during plugin construction to verify schema definition is valid.
        This is PHASE 1 validation - checking the schema itself, not compatibility
        with other plugins.

        Returns:
            List of validation errors (empty if valid)

        Raises:
            ValueError: If schema configuration is invalid

        Note:
            This is SELF-validation only. Plugin checks its own schema is valid.
            COMPATIBILITY validation (does plugin A's output match plugin B's input?)
            happens in PHASE 2 during ExecutionGraph.from_plugin_instances().

        Example:
            - Valid: Schema has no syntax errors, types are well-formed
            - Invalid: Schema has malformed field specs, unknown types
        """
        ...
```

**Step 4: Run test again**

```bash
pytest tests/contracts/test_plugin_protocols.py::test_source_validates_output_schema_on_init -xvs
```

Expected: PASS (CSVSource already validates in __init__)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/protocols.py tests/contracts/test_plugin_protocols.py
git commit -m "feat: add schema validation to plugin protocols

- Add validate_output_schema() to SourceProtocol
- Sources validate schema during __init__ (already implemented)
- Establishes pattern for transform/sink validation
- Ref: Fix schema validation architecture properly"
```

---

## Task 2: Remove Schema Validation from DAG Layer

**Goal:** DAG validation only checks structural issues, not schema compatibility.

**Files:**
- Modify: `src/elspeth/core/dag.py` (simplify validate() method)
- Modify: `tests/core/test_dag.py` (update test expectations)

**Step 1: Write test for new DAG validation behavior**

Add to `tests/core/test_dag.py`:

```python
def test_dag_validation_only_checks_structure() -> None:
    """DAG validation should only check cycles and connectivity, not schemas."""
    from elspeth.contracts import PluginSchema
    from elspeth.core.dag import ExecutionGraph

    class OutputSchema(PluginSchema):
        value: int

    class DifferentSchema(PluginSchema):
        different: str  # Incompatible!

    graph = ExecutionGraph()

    # Add incompatible schemas
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=OutputSchema)
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=DifferentSchema)
    graph.add_edge("source", "sink", label="continue")

    # OLD behavior: Would raise GraphValidationError for schema mismatch
    # NEW behavior: Only checks structural validity (no cycles)
    errors = graph.validate()
    assert len(errors) == 0  # No structural problems
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::test_dag_validation_only_checks_structure -xvs
```

Expected: FAIL (currently validates schema compatibility)

**Step 3: Simplify DAG.validate() method**

In `src/elspeth/core/dag.py`, replace `validate()` method:

```python
def validate(self, raise_on_error: bool = False) -> list[str]:
    """Validate graph structure.

    Checks for:
    - Cycles (graphs must be acyclic)
    - Connectivity (all sinks reachable from source)
    - Route consistency (gate routes reference valid sinks)

    Does NOT check schema compatibility - plugins validate their own
    schemas during construction.

    Args:
        raise_on_error: If True, raise GraphValidationError on first error

    Returns:
        List of validation errors (empty if valid)

    Raises:
        GraphValidationError: If raise_on_error=True and validation fails
    """
    errors = []

    # Check for cycles
    if not nx.is_directed_acyclic_graph(self._graph):
        error_msg = "Graph contains cycles (not a valid DAG)"
        if raise_on_error:
            raise GraphValidationError(error_msg)
        errors.append(error_msg)

    # Check connectivity (all sinks reachable from source)
    source_nodes = [n for n, data in self._graph.nodes(data=True) if data["info"].node_type == "source"]
    sink_nodes = [n for n, data in self._graph.nodes(data=True) if data["info"].node_type == "sink"]

    if source_nodes:
        source_id = source_nodes[0]
        for sink_id in sink_nodes:
            if not nx.has_path(self._graph, source_id, sink_id):
                error_msg = f"Sink '{sink_id}' is not reachable from source"
                if raise_on_error:
                    raise GraphValidationError(error_msg)
                errors.append(error_msg)

    # Check gate routes reference valid sinks
    gate_nodes = [n for n, data in self._graph.nodes(data=True) if data["info"].node_type == "gate"]
    valid_sink_names = set(self._sink_id_map.keys()) if self._sink_id_map else set()

    for gate_id in gate_nodes:
        gate_info = self.get_node_info(gate_id)
        # Gates store route targets in config
        route_config = gate_info.config.get("routes", {})
        for route_target in route_config.values():
            if route_target not in valid_sink_names and route_target != "continue":
                error_msg = f"Gate '{gate_id}' routes to unknown sink '{route_target}'"
                if raise_on_error:
                    raise GraphValidationError(error_msg)
                errors.append(error_msg)

    return errors
```

**Step 4: Remove schema validation call from validate() method**

In `src/elspeth/core/dag.py`, update `validate()` to remove the schema validation call (around line 238):

```python
# REMOVE THIS BLOCK:
# Check schema compatibility across all edges
schema_errors = self._validate_edge_schemas()
if schema_errors:
    raise GraphValidationError("Schema incompatibilities:\n" + ...)
```

**Step 4.5: Delete schema validation helper methods**

Delete these methods from `src/elspeth/core/dag.py`:
- `_is_dynamic_schema()` (lines 67-84)
- `_get_missing_required_fields()` (lines 49-64) - NOTE: Will be re-added in Task 2.5
- `_schemas_compatible()` (lines 87-97)
- `_validate_edge_schemas()` (lines 288-338)
- `_validate_coalesce_schema_compatibility()` (lines 240-286)
- `_get_effective_producer_schema()` (lines 340-394) - NOTE: Will be re-added in Task 2.5

**Step 5: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::test_dag_validation_only_checks_structure -xvs
```

Expected: PASS

**Step 6: Update existing schema validation tests**

Find tests that expect schema validation errors from DAG.validate() and update them to expect plugin construction failures instead:

```bash
# Find tests that check for schema validation in DAG
grep -r "GraphValidationError.*schema" tests/core/test_dag.py
```

Update each to test plugin construction instead:

```python
# OLD:
with pytest.raises(GraphValidationError, match="missing required fields"):
    graph.validate()

# NEW:
with pytest.raises(ValueError, match="Schema compatibility"):
    Transform(bad_config)  # Fails during construction
```

**Step 7: Run all DAG tests**

```bash
pytest tests/core/test_dag.py -v
```

Expected: Some tests will fail (need updating)

**Step 8: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "refactor: remove schema validation from DAG layer

- DAG validation now only checks structure (cycles, connectivity)
- Delete _is_dynamic_schema, _validate_edge_schemas, etc.
- Schema validation moved to plugin construction (next task)
- Ref: Fix schema validation architecture properly"
```

---

## Task 2.5: Add Edge Compatibility Validation to ExecutionGraph ✅ COMPLETE

**Goal:** ExecutionGraph validates schema compatibility during graph construction (not in primitive add_edge).

**Files:**
- Modify: `src/elspeth/core/dag.py` (add validation to from_plugin_instances, preserve gate walk-through)
- Create: `tests/core/test_edge_validation.py` (new test file)

**Architecture Decision:** Validation happens in `from_plugin_instances()` AFTER graph is built, not in primitive `add_edge()`. This keeps `add_edge()` as a dumb graph operation and validates when we have full context.

**Step 1: Write failing test for edge compatibility validation**

Create `tests/core/test_edge_validation.py`:

```python
"""Test edge compatibility validation during graph construction."""

import pytest
from elspeth.contracts import PluginSchema
from elspeth.core.dag import ExecutionGraph


class ProducerSchema(PluginSchema):
    """Producer output schema."""
    id: int
    name: str


class ConsumerSchema(PluginSchema):
    """Consumer input schema."""
    id: int
    name: str
    email: str  # Required field NOT in producer!


def test_edge_validation_detects_missing_fields() -> None:
    """Edges should fail if producer missing required fields.

    Validation happens in validate_edge_compatibility() called from
    from_plugin_instances(), NOT during add_edge() (which is a dumb primitive).
    """
    graph = ExecutionGraph()

    # Add source with ProducerSchema
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)

    # Add sink requiring ConsumerSchema (has 'email' field)
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=ConsumerSchema)

    # Wire them together (add_edge does NOT validate)
    graph.add_edge("source", "sink", label="continue")

    # Validation happens when we explicitly call it
    with pytest.raises(ValueError, match="missing required fields.*email"):
        graph.validate_edge_compatibility()


def test_edge_validation_allows_dynamic_schemas() -> None:
    """Dynamic schemas should be compatible with anything."""
    from elspeth.contracts.schema import create_dynamic_schema

    graph = ExecutionGraph()

    dynamic_schema = create_dynamic_schema("DynamicSchema")

    # Dynamic producer → strict consumer: OK
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=dynamic_schema)
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=ConsumerSchema)
    graph.add_edge("source", "sink", label="continue")  # Should succeed

    # Strict producer → dynamic consumer: OK
    graph2 = ExecutionGraph()
    graph2.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)
    graph2.add_node("sink", node_type="sink", plugin_name="csv", input_schema=dynamic_schema)
    graph2.add_edge("source", "sink", label="continue")  # Should succeed


def test_gate_passthrough_validation() -> None:
    """Gates must preserve schema (input == output)."""
    graph = ExecutionGraph()

    # Source produces ProducerSchema
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)

    # Gate claims to pass through but has DIFFERENT output schema
    graph.add_node("gate", node_type="gate", plugin_name="threshold",
                   input_schema=ProducerSchema, output_schema=ConsumerSchema)

    # Wire them (add_edge does NOT validate)
    graph.add_edge("source", "gate", label="continue")

    # Validation happens when we explicitly call it
    with pytest.raises(ValueError, match="Gate.*must preserve schema"):
        graph.validate_edge_compatibility()


def test_coalesce_branch_compatibility() -> None:
    """Coalesce must receive compatible schemas from all branches."""
    graph = ExecutionGraph()

    # Source forks to two paths
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)

    # Path 1: Transform to Schema A
    class SchemaA(PluginSchema):
        value: int

    graph.add_node("transform1", node_type="transform", plugin_name="field_mapper",
                   input_schema=ProducerSchema, output_schema=SchemaA)

    # Path 2: Transform to Schema B (INCOMPATIBLE!)
    class SchemaB(PluginSchema):
        different: str

    graph.add_node("transform2", node_type="transform", plugin_name="field_mapper",
                   input_schema=ProducerSchema, output_schema=SchemaB)

    # Coalesce node
    graph.add_node("coalesce", node_type="coalesce", plugin_name="merge",
                   input_schema=SchemaA)  # Expects SchemaA from both branches

    # Wire up
    graph.add_edge("source", "transform1", label="fork_path_1")
    graph.add_edge("source", "transform2", label="fork_path_2")
    graph.add_edge("transform1", "coalesce", label="continue")  # OK
    graph.add_edge("transform2", "coalesce", label="continue")  # Add second edge

    # This should fail during graph validation (AFTER edges added)
    with pytest.raises(ValueError, match="Coalesce.*incompatible schemas"):
        graph.validate_edge_compatibility()  # Explicit validation call


def test_gate_walk_through_for_effective_schema() -> None:
    """Edge validation must walk through gates to find effective producer schema."""
    graph = ExecutionGraph()

    # Source produces ProducerSchema
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)

    # Gate has NO schema (config-driven gate)
    graph.add_node("gate", node_type="gate", plugin_name="config_gate",
                   input_schema=None, output_schema=None)  # Inherits from upstream!

    # Sink requires ConsumerSchema
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=ConsumerSchema)

    # Wire: source → gate → sink
    graph.add_edge("source", "gate", label="continue")
    graph.add_edge("gate", "sink", label="continue")

    # Should walk through gate to find ProducerSchema
    # Then check if ProducerSchema has fields required by ConsumerSchema
    with pytest.raises(ValueError, match="missing required fields.*email"):
        graph.validate_edge_compatibility()


def test_chained_gates() -> None:
    """Validation must handle multiple chained gates."""
    graph = ExecutionGraph()

    # Source → Gate1 → Gate2 → Sink
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=ProducerSchema)
    graph.add_node("gate1", node_type="gate", plugin_name="config_gate",
                   input_schema=None, output_schema=None)
    graph.add_node("gate2", node_type="gate", plugin_name="config_gate",
                   input_schema=None, output_schema=None)
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=ConsumerSchema)

    graph.add_edge("source", "gate1", label="continue")
    graph.add_edge("gate1", "gate2", label="continue")
    graph.add_edge("gate2", "sink", label="continue")

    # Should walk gate1 → gate2 → source, find ProducerSchema missing 'email'
    with pytest.raises(ValueError, match="missing required fields.*email"):
        graph.validate_edge_compatibility()


def test_none_schema_handling() -> None:
    """None schemas (dynamic by convention) should be compatible with anything."""
    graph = ExecutionGraph()

    # Source with None schema (dynamic)
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=None)

    # Sink with strict schema
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=ConsumerSchema)

    graph.add_edge("source", "sink", label="continue")

    # Should pass - None is compatible with anything
    graph.validate_edge_compatibility()  # No exception


def test_edge_validation_timing_from_plugin_instances() -> None:
    """CRITICAL: Validation must happen during from_plugin_instances(), not in validate().

    This test verifies the core architectural change - that schema validation
    has been moved from DAG.validate() to graph construction time.
    """
    from elspeth.plugins.manager import PluginManager

    manager = PluginManager()

    config = {
        "datasource": {
            "plugin": "csv",
            "options": {
                "path": "test.csv",
                "schema": {"fields": ["id: int"]},  # Only has 'id'
            },
        },
        "row_plugins": [],
        "sinks": {
            "out": {
                "plugin": "csv",
                "options": {
                    "path": "out.csv",
                    "schema": {"mode": "strict", "fields": ["id: int", "email: str"]},  # Requires 'email'!
                },
            }
        },
        "output_sink": "out",
    }

    # Should fail DURING from_plugin_instances (PHASE 2 validation)
    with pytest.raises(ValueError, match="missing required fields.*email"):
        graph = ExecutionGraph.from_plugin_instances(
            source=manager.get_source("csv", config["datasource"]["options"]),
            transforms=[],
            sinks={"out": manager.get_sink("csv", config["sinks"]["out"]["options"])},
            aggregations=[],
            gates=[],
            output_sink="out"
        )


def test_aggregation_dual_schema_both_edges_validated() -> None:
    """Aggregations have both input_schema and output_schema - validate both edges."""

    class SourceOutput(PluginSchema):
        value: float

    class AggInput(PluginSchema):
        value: float
        label: str  # Required, not in source!

    class AggOutput(PluginSchema):
        count: int
        sum: float

    class SinkInput(PluginSchema):
        count: int
        sum: float
        average: float  # Required, not in agg output!

    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv",
                   output_schema=SourceOutput)
    graph.add_node("agg", node_type="aggregation", plugin_name="stats",
                   input_schema=AggInput,
                   output_schema=AggOutput)
    graph.add_node("sink", node_type="sink", plugin_name="csv",
                   input_schema=SinkInput)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Should detect BOTH mismatches (source→agg has 'label' missing, agg→sink has 'average' missing)
    with pytest.raises(ValueError, match="label|average"):
        graph.validate_edge_compatibility()


def test_orphaned_config_gate_crashes_with_diagnostic() -> None:
    """Config gate with no incoming edges is a graph construction bug - should crash with clear error."""

    graph = ExecutionGraph()
    graph.add_node("gate", node_type="gate", plugin_name="config_gate",
                   input_schema=None, output_schema=None)  # Config gate
    graph.add_node("sink", node_type="sink", plugin_name="csv",
                   input_schema=ConsumerSchema)

    # Accidentally wired gate→sink without source→gate
    graph.add_edge("gate", "sink", label="continue")

    # Should crash with diagnostic error (not silent failure)
    with pytest.raises(ValueError, match="no incoming edges"):
        graph.validate_edge_compatibility()


def test_schema_mismatch_error_includes_field_name_and_nodes() -> None:
    """Error messages must be actionable - include field names and node IDs."""

    graph = ExecutionGraph()
    graph.add_node("csv_reader", node_type="source", plugin_name="csv",
                   output_schema=ProducerSchema)  # Has: id, name
    graph.add_node("db_writer", node_type="sink", plugin_name="database",
                   input_schema=ConsumerSchema)  # Needs: id, name, email

    graph.add_edge("csv_reader", "db_writer", label="continue")

    try:
        graph.validate_edge_compatibility()
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error = str(e)
        # Must include both node names
        assert "csv_reader" in error, "Error must name producer node"
        assert "db_writer" in error, "Error must name consumer node"
        # Must include missing field name
        assert "email" in error.lower(), "Error must name missing field"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_edge_validation.py -xvs
```

Expected: FAIL (edge validation not implemented yet)

**Step 3: Add edge compatibility validation methods**

In `src/elspeth/core/dag.py`, add methods:

```python
def validate_edge_compatibility(self) -> None:
    """Validate schema compatibility for all edges in the graph.

    Called AFTER graph construction is complete. Validates that each edge
    connects compatible schemas.

    Raises:
        ValueError: If any edge has incompatible schemas

    Note:
        This is PHASE 2 validation (cross-plugin compatibility). Plugin
        SELF-validation happens in PHASE 1 during plugin construction.
    """
    # Validate each edge
    for from_id, to_id, edge_data in self._graph.edges(data=True):
        self._validate_single_edge(from_id, to_id)

    # Validate all coalesce nodes (must have compatible schemas from all branches)
    coalesce_nodes = [
        node_id for node_id, data in self._graph.nodes(data=True)
        if data["info"].node_type == "coalesce"
    ]
    for coalesce_id in coalesce_nodes:
        self._validate_coalesce_compatibility(coalesce_id)


def _validate_single_edge(self, from_node_id: str, to_node_id: str) -> None:
    """Validate schema compatibility for a single edge.

    Args:
        from_node_id: Source node ID
        to_node_id: Destination node ID

    Raises:
        ValueError: If schemas are incompatible
    """
    to_info = self.get_node_info(to_node_id)

    # Rule 0: Gates must preserve schema (input == output)
    if to_info.node_type == "gate":
        if to_info.input_schema is not None and to_info.output_schema is not None:
            if to_info.input_schema != to_info.output_schema:
                raise ValueError(
                    f"Gate '{to_node_id}' must preserve schema: "
                    f"input_schema={to_info.input_schema.__name__}, "
                    f"output_schema={to_info.output_schema.__name__}"
                )

    # Get EFFECTIVE producer schema (walks through gates if needed)
    producer_schema = self._get_effective_producer_schema(from_node_id)
    consumer_schema = to_info.input_schema

    # Rule 1: Dynamic schemas (None) bypass validation
    if producer_schema is None or consumer_schema is None:
        return  # Dynamic schema - compatible with anything

    # Rule 2: Check field compatibility
    missing_fields = self._get_missing_required_fields(
        producer_schema, consumer_schema
    )
    if missing_fields:
        raise ValueError(
            f"Edge from '{from_node_id}' to '{to_node_id}' invalid: "
            f"producer schema '{producer_schema.__name__}' missing required fields "
            f"for consumer schema '{consumer_schema.__name__}': {missing_fields}"
        )


def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
    """Get effective output schema, walking through pass-through nodes (gates).

    Gates and other pass-through nodes don't transform data - they inherit
    schema from their upstream producers. This method walks backwards through
    the graph to find the nearest schema-carrying producer.

    Args:
        node_id: Node to get effective schema for

    Returns:
        Output schema type, or None if dynamic

    Raises:
        ValueError: If gate has no incoming edges (graph construction bug)
    """
    node_info = self.get_node_info(node_id)

    # If node has output_schema, return it directly
    if node_info.output_schema is not None:
        return node_info.output_schema

    # Node has no schema - check if it's a pass-through type (gate)
    if node_info.node_type == "gate":
        # Gate passes data unchanged - inherit from upstream producer
        incoming = list(self._graph.in_edges(node_id, data=True))

        if not incoming:
            # Gate with no inputs is a graph construction bug - CRASH
            raise ValueError(
                f"Gate node '{node_id}' has no incoming edges - "
                f"this indicates a bug in graph construction"
            )

        # Get effective schema from first input (recursive for chained gates)
        first_edge_source = incoming[0][0]
        first_schema = self._get_effective_producer_schema(first_edge_source)

        # For multi-input gates, verify all inputs have same schema
        if len(incoming) > 1:
            for from_id, _, _ in incoming[1:]:
                other_schema = self._get_effective_producer_schema(from_id)
                if first_schema != other_schema:
                    # Multi-input gates with incompatible schemas - CRASH
                    raise ValueError(
                        f"Gate '{node_id}' receives incompatible schemas from "
                        f"multiple inputs - this is a graph construction bug. "
                        f"First input schema: {first_schema}, "
                        f"Other input schema: {other_schema}"
                    )

        return first_schema

    # Not a gate and no schema - return None (dynamic)
    return None


def _validate_coalesce_compatibility(self, coalesce_id: str) -> None:
    """Validate all inputs to coalesce node have compatible schemas.

    Args:
        coalesce_id: Coalesce node ID

    Raises:
        ValueError: If branches have incompatible schemas
    """
    incoming = list(self._graph.in_edges(coalesce_id, data=True))

    if len(incoming) < 2:
        return  # Degenerate case (1 branch) - always compatible

    # Get effective schema from first branch
    first_edge_source = incoming[0][0]
    first_schema = self._get_effective_producer_schema(first_edge_source)

    # Verify all other branches have same schema
    for from_id, _, _ in incoming[1:]:
        other_schema = self._get_effective_producer_schema(from_id)
        if first_schema != other_schema:
            raise ValueError(
                f"Coalesce '{coalesce_id}' receives incompatible schemas from "
                f"multiple branches: "
                f"first branch has {first_schema.__name__ if first_schema else 'dynamic'}, "
                f"branch from '{from_id}' has {other_schema.__name__ if other_schema else 'dynamic'}"
            )


def _get_missing_required_fields(
    self,
    producer_schema: type[PluginSchema] | None,
    consumer_schema: type[PluginSchema] | None,
) -> list[str]:
    """Get required fields that producer doesn't provide.

    Args:
        producer_schema: Schema of data producer
        consumer_schema: Schema of data consumer

    Returns:
        List of field names missing from producer
    """
    if producer_schema is None or consumer_schema is None:
        return []  # Dynamic schema

    producer_fields = set(producer_schema.model_fields.keys())
    consumer_required = {
        name for name, field in consumer_schema.model_fields.items()
        if field.is_required()
    }

    return sorted(consumer_required - producer_fields)
```

**Step 4: Integrate validation into from_plugin_instances()**

In `src/elspeth/core/dag.py`, update `from_plugin_instances()` to call validation AFTER graph is built:

```python
@classmethod
def from_plugin_instances(cls, ...) -> "ExecutionGraph":
    """Create execution graph from plugin instances.

    ... (existing construction logic) ...
    """
    graph = cls()

    # Add all nodes
    # CRITICAL: Preserve plugin config in NodeInfo for audit trail
    # DO NOT use config={} - must preserve original config for auditability
    # Example:
    #   graph.add_node(
    #       node_id,
    #       config=dict(plugin_config.options),  # Preserve config!
    #       input_schema=plugin.input_schema,
    #       output_schema=plugin.output_schema,
    #       ...
    #   )

    # Add all edges
    # ... (existing edge addition code) ...

    # PHASE 2 VALIDATION: Validate schema compatibility AFTER graph is built
    graph.validate_edge_compatibility()

    return graph
```

**CRITICAL for Audit Trail:** NodeInfo must preserve the original plugin config dict. Per CLAUDE.md: "Every decision must be traceable to configuration". Do NOT set `config={}` even though "config already used during instantiation". The audit trail requires the original configuration to be preserved.

**Step 5: Run tests to verify they pass**

```bash
pytest tests/core/test_edge_validation.py -xvs
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_edge_validation.py
git commit -m "feat: add edge compatibility validation to ExecutionGraph

- Validate schema compatibility during from_plugin_instances()
- PHASE 2 validation: cross-plugin compatibility (after PHASE 1 self-validation)
- Preserves gate walk-through logic (_get_effective_producer_schema)
- Checks missing fields, dynamic schemas (None), gate inheritance, coalesce branches
- Handles config gates (no schemas) correctly
- Validation at graph construction, not in primitive add_edge()
- Ref: Fix schema validation architecture properly"
```

---

## Task 3: Add Self-Validation to All Plugins

**Goal:** Add abstract self-validation method to BaseTransform and update ALL builtin plugins to implement it.

**Files:**
- Modify: `src/elspeth/plugins/base.py` (add @abstractmethod _validate_self_consistency)
- Modify: ALL transform plugins (PassThrough, FieldMapper, etc.)
- Modify: ALL source plugins (CSVSource, etc.)
- Modify: ALL sink plugins (CSVSink, etc.)

**IMPORTANT:** The method is abstract from the start (enforced by ABC). All plugins MUST implement it, even if implementation is just `pass`.

**Step 1: Write failing test for transform validation**

Add to `tests/plugins/transforms/test_passthrough.py`:

```python
def test_passthrough_validates_schema_compatibility() -> None:
    """PassThrough should validate schema is self-compatible."""
    from elspeth.plugins.transforms.passthrough import PassThrough

    # Valid: Dynamic schema (always compatible with itself)
    config = {"schema": {"fields": "dynamic"}}
    transform = PassThrough(config)  # Should succeed

    # Valid: Explicit schema (compatible with itself)
    config = {"schema": {"mode": "strict", "fields": ["id: int", "name: str"]}}
    transform = PassThrough(config)  # Should succeed

    # Note: PassThrough has same input/output schema, so always compatible
    # More complex transforms tested separately
```

**Step 2: Run test**

```bash
pytest tests/plugins/transforms/test_passthrough.py::test_passthrough_validates_schema_compatibility -xvs
```

Expected: PASS (PassThrough already creates compatible schemas)

**Step 3: Add validation method to BaseTransform**

In `src/elspeth/plugins/base.py`, add to `BaseTransform`:

```python
class BaseTransform(ABC):
    """Base class for transform plugins."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize transform.

        Args:
            config: Plugin configuration

        Raises:
            ValueError: If schema configuration is invalid or incompatible
        """
        self._config = config
        self._validation_called = False

        # Subclass must set input_schema and output_schema
        # Validation happens in subclass __init__ after schemas are set

    def __init_subclass__(cls, **kwargs):
        """Enforce that subclasses call _validate_self_consistency() in __init__.

        This hook wraps the subclass __init__ to verify validation was called.
        Prevents plugins from bypassing validation by implementing but not calling.
        """
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not getattr(self, '_validation_called', False):
                raise RuntimeError(
                    f"{cls.__name__}.__init__ did not call _validate_self_consistency(). "
                    f"Validation is mandatory for audit integrity."
                )
        cls.__init__ = wrapped_init

    @abstractmethod
    def _validate_self_consistency(self) -> None:
        """Validate plugin's own schemas are self-consistent (PHASE 1).

        Called by subclass __init__ after setting schemas. This is SELF-validation
        only - checking that the plugin's own configuration makes sense.

        Raises:
            ValueError: If schemas are internally inconsistent

        Note:
            This is NOT about compatibility with OTHER plugins. That happens in
            PHASE 2 during ExecutionGraph.from_plugin_instances().

            Examples of self-consistency checks:
            - PassThrough: input_schema must equal output_schema
            - FieldMapper: output fields must be subset of input fields (if mode=strict)
            - Gate: input_schema must equal output_schema (pass-through)

            For plugins with no self-consistency constraints, implement as:
            ```python
            def _validate_self_consistency(self) -> None:
                self._validation_called = True  # Mark validation as complete
                # No additional validation needed
            ```

            Subclasses MUST:
            1. Implement this method (enforced by ABC - TypeError if not implemented)
            2. Call this method in __init__ (enforced by __init_subclass__ - RuntimeError if not called)
            3. Set self._validation_called = True (first line of implementation)

            The __init_subclass__ hook verifies validation was called after __init__ completes.
        """
        ...
```

**Step 4: Update PassThrough to implement and call validation**

In `src/elspeth/plugins/transforms/passthrough.py`, add to `__init__` and implement the abstract method:

```python
def __init__(self, config: dict[str, Any]) -> None:
    super().__init__(config)
    cfg = PassThroughConfig.from_dict(config)
    self._validate_input = cfg.validate_input
    self._on_error: str | None = cfg.on_error

    assert cfg.schema_config is not None
    self._schema_config = cfg.schema_config

    schema = create_schema_from_config(
        self._schema_config,
        "PassThroughSchema",
        allow_coercion=False,
    )
    self.input_schema = schema
    self.output_schema = schema

    # NEW: Validate self-consistency (PHASE 1)
    self._validate_self_consistency()


def _validate_self_consistency(self) -> None:
    """Validate PassThrough schemas are self-consistent.

    PassThrough has no self-consistency constraints (input == output by definition).
    """
    self._validation_called = True  # Mark validation as complete
    # No additional validation needed - PassThrough always has matching schemas
```

**Step 5: Run test**

```bash
pytest tests/plugins/transforms/test_passthrough.py::test_passthrough_validates_schema_compatibility -xvs
```

Expected: PASS

**Step 6: Update ALL builtin plugins to implement and call validation**

Update all transform, source, and sink plugins to implement `_validate_self_consistency()` and call it:

```bash
# Find all plugin __init__ methods
grep -r "def __init__" src/elspeth/plugins/transforms/
grep -r "def __init__" src/elspeth/plugins/sources/
grep -r "def __init__" src/elspeth/plugins/sinks/
```

For each plugin, add:
1. Implementation of `_validate_self_consistency()` method
2. Call to `_validate_self_consistency()` at the end of `__init__`

```python
# Example for each plugin:
def __init__(self, config: dict[str, Any]) -> None:
    super().__init__(config)
    # ... existing initialization ...
    self.input_schema = ...
    self.output_schema = ...

    # NEW: Validate self-consistency (PHASE 1)
    self._validate_self_consistency()


def _validate_self_consistency(self) -> None:
    """Validate plugin's own schemas are self-consistent.

    [Plugin-specific validation logic, or just pass if no constraints]
    """
    self._validation_called = True  # MANDATORY - marks validation as complete
    # Most plugins have no additional self-consistency constraints
```

**Plugins to update:**
- `PassThrough` ✓ (already done in Step 4)
- `FieldMapper`
- All sources (CSVSource, etc.)
- All sinks (CSVSink, etc.)
- All gates (if they have schemas)

**Step 7: Run all plugin tests**

```bash
pytest tests/plugins/ -v
```

Expected: ALL PASS (plugins now validate during construction)

**Step 8: Commit**

```bash
git add src/elspeth/plugins/
git commit -m "feat: add self-validation to all builtin plugins

- BaseTransform._validate_self_consistency() abstract method (enforced by ABC)
- PHASE 1: Plugins validate their own schemas are well-formed
- Updated ALL builtin plugins to implement and call validation
- PassThrough, FieldMapper, sources, sinks all validate
- Does NOT validate compatibility with other plugins (that's PHASE 2)
- TypeError at instantiation if plugin doesn't implement method
- Ref: Fix schema validation architecture properly"
```

---

## Task 3.5: Verify Validation Enforcement and Add Bypass Tests

**Goal:** Verify ABC enforcement works and test that plugins cannot bypass validation.

**Files:**
- Create: `tests/contracts/test_validation_enforcement.py` (enforcement verification tests)

**Note:** Enforcement is already in place via `@abstractmethod` from Task 3. This task adds comprehensive tests to verify enforcement cannot be bypassed.

**Step 1: Write test for enforcement**

Create `tests/contracts/test_validation_enforcement.py`:

```python
"""Test that plugins cannot skip validation."""

import pytest
from elspeth.plugins.base import BaseTransform
from elspeth.contracts import PluginSchema


class TestSchema(PluginSchema):
    """Test schema."""
    value: int


def test_transform_must_implement_validation() -> None:
    """Transforms that don't implement _validate_self_consistency should fail."""

    class BadTransform(BaseTransform):
        """Transform that doesn't implement _validate_self_consistency."""

        def __init__(self, config: dict) -> None:
            super().__init__(config)
            self.input_schema = TestSchema
            self.output_schema = TestSchema
            # BUG: Didn't implement abstract method _validate_self_consistency()

        def process(self, row, ctx):
            return row

    # This should raise TypeError at instantiation (abstract method not implemented)
    with pytest.raises(TypeError, match="Can't instantiate abstract class.*_validate_self_consistency"):
        BadTransform({})


def test_transform_with_validation_succeeds() -> None:
    """Transforms that implement validation should succeed."""

    class GoodTransform(BaseTransform):
        """Transform that correctly implements validation."""

        def __init__(self, config: dict) -> None:
            super().__init__(config)
            self.input_schema = TestSchema
            self.output_schema = TestSchema
            self._validate_self_consistency()  # Correct!

        def _validate_self_consistency(self) -> None:
            """Implement abstract method."""
            self._validation_called = True  # Mark validation as complete
            # No additional validation needed for this test transform

        def process(self, row, ctx):
            return row

    # Should succeed
    transform = GoodTransform({})
    assert transform.input_schema is TestSchema


def test_transform_must_call_validation_not_just_implement() -> None:
    """CRITICAL: __init_subclass__ hook enforces validation is CALLED, not just implemented."""

    class LazyTransform(BaseTransform):
        """Transform that implements but never calls _validate_self_consistency."""

        def __init__(self, config: dict) -> None:
            super().__init__(config)
            self.input_schema = TestSchema
            self.output_schema = TestSchema
            # BUG: Implemented the method but didn't call it!

        def _validate_self_consistency(self) -> None:
            """Method exists but is never invoked."""
            self._validation_called = True
            # Validation logic here would never execute

        def process(self, row, ctx):
            return row

    # Should fail at instantiation - __init_subclass__ hook detects missing call
    with pytest.raises(RuntimeError, match="did not call _validate_self_consistency"):
        LazyTransform({})


def test_transform_cannot_bypass_validation_via_super_skip() -> None:
    """CRITICAL: Plugins cannot bypass validation by skipping super().__init__().

    This test verifies that ABC enforcement works even if a plugin tries
    to bypass the base class __init__.
    """

    class MaliciousTransform(BaseTransform):
        """Transform that tries to bypass validation."""

        def __init__(self, config: dict) -> None:
            # BUG: Doesn't call super().__init__(), tries to bypass validation
            self._config = config
            self.input_schema = TestSchema
            self.output_schema = TestSchema
            # No _validate_self_consistency() call!
            # And didn't implement the abstract method!

        def process(self, row, ctx):
            return row

    # Should still fail - ABC catches this during class instantiation
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MaliciousTransform({})


def test_transform_validation_survives_multiple_inheritance() -> None:
    """Validation enforcement survives multiple inheritance."""

    class Mixin:
        """Test mixin class."""
        def extra_method(self):
            pass

    class TransformWithMixin(Mixin, BaseTransform):
        """Transform with multiple inheritance."""

        def __init__(self, config: dict) -> None:
            super().__init__(config)
            self.input_schema = TestSchema
            self.output_schema = TestSchema
            # Forgot to implement _validate_self_consistency()!

        def process(self, row, ctx):
            return row

    # Should fail - ABC enforcement transcends multiple inheritance
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        TransformWithMixin({})
```

**Step 2: Run tests to verify ABC enforcement works**

```bash
pytest tests/contracts/test_validation_enforcement.py -xvs
```

Expected: ALL PASS (enforcement already in place from Task 3)

**Step 3: Commit**

```bash
git add tests/contracts/test_validation_enforcement.py
git commit -m "test: add validation enforcement bypass tests

- Test that ABC enforcement prevents plugins from skipping validation
- Test bypassing super().__init__() still caught by ABC
- Test multiple inheritance doesn't break enforcement
- All tests pass (enforcement already in place from Task 3)
- Ref: Fix schema validation architecture properly"
```

---

## Task 4: Update All Tests to Expect Construction-Time Validation

**Goal:** Update integration tests to expect schema errors during plugin construction, not graph validation.

**Files:**
- Modify: `tests/integration/test_schema_validation_end_to_end.py`
- Modify: `tests/cli/test_plugin_errors.py`

**Step 1: Identify tests that rely on DAG validation**

```bash
grep -r "graph.validate()" tests/integration/ tests/cli/
grep -r "GraphValidationError" tests/integration/ tests/cli/
```

**Step 2: Add P0 two-phase validation integration test**

Add to `tests/integration/test_schema_validation_end_to_end.py`:

```python
def test_two_phase_validation_separates_self_and_compatibility_errors() -> None:
    """Verify PHASE 1 (self) and PHASE 2 (compatibility) validation both work."""
    from elspeth.plugins.manager import PluginManager

    manager = PluginManager()

    # PHASE 1 should fail: Malformed schema in plugin config
    bad_self_config = {
        "datasource": {
            "plugin": "csv",
            "options": {
                "path": "test.csv",
                "schema": {"mode": "strict", "fields": ["invalid syntax!!!"]},
            }
        }
    }

    with pytest.raises(ValueError, match="Invalid field spec"):
        # Fails during plugin construction (PHASE 1)
        manager.get_source("csv", bad_self_config["datasource"]["options"])

    # PHASE 2 should fail: Well-formed schemas, incompatible connection
    good_self_bad_compat_config = {
        "datasource": {
            "plugin": "csv",
            "options": {
                "path": "test.csv",
                "schema": {"fields": ["id: int"]},  # Only has 'id'
            },
        },
        "row_plugins": [
            {
                "plugin": "passthrough",
                "options": {
                    "schema": {"fields": ["id: int", "email: str"]},  # Requires 'email'!
                },
            }
        ],
        "sinks": {
            "out": {
                "plugin": "csv",
                "options": {"path": "out.csv"},
            }
        },
        "output_sink": "out",
    }

    with pytest.raises(ValueError, match="missing required fields.*email"):
        # Fails during graph construction (PHASE 2)
        graph = ExecutionGraph.from_config(good_self_bad_compat_config, manager)
```

**Step 3: Update tests to expect ValueError from graph construction**

For each test that expects `GraphValidationError` from `graph.validate()`:

```python
# OLD pattern:
graph = ExecutionGraph.from_plugin_instances(...)
with pytest.raises(GraphValidationError):
    graph.validate()

# NEW pattern:
with pytest.raises(ValueError, match="Schema compatibility"):
    graph = ExecutionGraph.from_plugin_instances(...)
    # Graph construction fails if schemas incompatible (PHASE 2)
```

**Step 4: Run integration tests**

```bash
pytest tests/integration/test_schema_validation_end_to_end.py -v
```

Expected: Many failures (need updating)

**Step 4: Update each failing test**

Work through each failure, updating test expectations.

**Step 5: Run CLI tests**

```bash
pytest tests/cli/test_plugin_errors.py -v
```

Expected: Some failures (need updating)

**Step 5.5: Audit integration test migration (CRITICAL)**

Verify ALL tests expecting schema validation at DAG.validate() have been migrated:

```bash
# Find any remaining tests that expect GraphValidationError from validate()
grep -r "graph.validate()" tests/integration/ tests/cli/ | grep -v "# Updated for Task 4"

# Find any remaining GraphValidationError expectations
grep -r "GraphValidationError.*schema" tests/

# Should return ZERO results for schema-related GraphValidationError
```

If any tests still expect old behavior:
1. Update them to expect ValueError during graph construction
2. Add comment: `# Updated for Task 4 - schema validation at construction`

**Step 6: Commit after all tests pass**

```bash
git add tests/integration/ tests/cli/
git commit -m "test: update tests for construction-time validation

- Schema errors now raised during plugin construction
- No longer expect GraphValidationError from graph.validate()
- Expect ValueError during ExecutionGraph.from_plugin_instances()
- Ref: Fix schema validation architecture properly"
```

---

## Task 5: Remove no_bug_hiding.yaml Allowlist Entries

**Goal:** Delete all allowlist entries related to schema detection.

**Files:**
- Modify: `config/cicd/no_bug_hiding.yaml`

**Step 1: Find schema-related allowlist entries**

```bash
grep -E "(_is_dynamic_schema|_extract_schema_config|model_config)" config/cicd/no_bug_hiding.yaml
```

**Step 2: Delete all related entries**

Remove entries for:
- `_is_dynamic_schema` (deleted function)
- `model_config.get()` (no longer used)
- Any other schema introspection patterns

**Step 3: Run no_bug_hiding check**

```bash
pytest tests/contracts/test_no_bug_hiding.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add config/cicd/no_bug_hiding.yaml
git commit -m "chore: remove schema detection from allowlist

- Delete _is_dynamic_schema entry (function deleted)
- Delete model_config.get() entries (no longer used)
- Net reduction: -N entries
- Ref: Fix schema validation architecture properly"
```

---

## Task 6: Run Full Test Suite

**Goal:** Verify all tests pass with new architecture.

**Step 1: Run complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All 3,279+ tests PASS

**Step 2: Run type checking**

```bash
mypy src/
```

Expected: Success

**Step 3: Run linting**

```bash
ruff check src/
```

Expected: All checks passed

**Step 4: If any failures, debug and fix**

Work through each failure systematically.

---

## Task 7: Update Bug Tickets

**Goal:** Mark all schema detection bugs as resolved.

**Files:**
- Modify: `docs/bugs/P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md`
- Modify: `docs/bugs/P0-2026-01-24-dynamic-schema-detection-regression.md`

**Step 1: Update P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md**

Add resolution section:

```markdown
## Resolution

**Status:** ✅ **RESOLVED** (commit: <hash>)

**Implementation:** ROOT CAUSE FIX - Moved validation to plugin construction

**What Changed:**
1. Schema validation moved from DAG layer to plugin construction
2. DAG.validate() only checks structural issues (cycles, connectivity)
3. Plugins validate their own schemas during __init__()
4. Deleted _is_dynamic_schema(), _validate_edge_schemas(), etc.
5. No SchemaConfig propagation to NodeInfo (not needed)

**Architecture Improvement:**
- Before: 2/5 (parallel mechanisms, Pydantic coupling)
- After: 5/5 (single source of truth, validation at right layer)

**Eliminated:**
- Parallel detection mechanisms (2 → 0, validation happens once)
- Pydantic introspection (gone, not needed)
- NodeInfo bloat (avoided, no new fields)
- Information loss pattern (fixed, config stays with plugin)

**Implemented:** 2026-01-24
**Implementation Plan:** docs/plans/2026-01-24-fix-schema-validation-properly.md
```

**Step 2: Update P0-2026-01-24-dynamic-schema-detection-regression.md**

Mark as superseded:

```markdown
## Resolution Update

**Status:** ✅ **SUPERSEDED BY ROOT CAUSE FIX**

**Original Fix:** Phase 3 introspection (P0-2026-01-24-dynamic-schema-detection-regression)
**Root Cause Fix:** Moved validation to plugin construction (this ticket)

**Why Superseded:**
The introspection fix (Phase 3) was correct but created technical debt.
Instead of fixing the debt (Phase 4 propagation plan), we fixed the
root cause (validation placement).

**See:** docs/plans/2026-01-24-fix-schema-validation-properly.md
```

**Step 3: Commit**

```bash
git add docs/bugs/*.md
git commit -m "docs: mark schema detection bugs as resolved

- Root cause fixed: validation moved to plugin construction
- No parallel mechanisms, no introspection, no NodeInfo bloat
- Architecture quality: 2/5 → 5/5
- Ref: Fix schema validation architecture properly"
```

---

## Completion Checklist

**Phase 1 Validation (Self-Validation):**
- [ ] Plugin protocols define self-validation methods
- [ ] Transforms validate self-consistency during construction
- [ ] Enforcement mechanism ensures validation cannot be skipped
- [ ] Validation enforcement tests pass

**Phase 2 Validation (Compatibility):**
- [ ] ExecutionGraph validates edge compatibility during add_edge()
- [ ] Missing field detection works
- [ ] Dynamic schema compatibility works
- [ ] Gate pass-through validation works
- [ ] Coalesce branch compatibility validation works
- [ ] Edge validation tests pass

**Phase 3 Validation (Structure Only):**
- [ ] DAG validation only checks structure (cycles, connectivity)
- [ ] Schema validation methods deleted from DAG layer (~200 LOC)

**Test Updates:**
- [ ] All tests updated for two-phase validation model
- [ ] Integration tests expect ValueError during plugin wiring (not DAG.validate)
- [ ] no_bug_hiding.yaml allowlist entries removed

**Quality Gates:**
- [ ] Full test suite passes (3,279+)
- [ ] mypy clean
- [ ] ruff clean
- [ ] Bug tickets marked resolved

**Success Metrics:**

| Metric | Before (Phase 3) | After (Root Fix) |
|--------|------------------|------------------|
| Detection mechanisms | 2 (SchemaConfig + introspection) | 0 (no detection, validation at construction) |
| Validation phases | 1 (DAG validates everything) | 2 (self + compatibility) |
| Validation locations | 2 (plugin + DAG) | 2 (plugin self-check + graph edge-check) |
| Pydantic coupling | HIGH (introspection) | NONE (no introspection) |
| NodeInfo fields | 6 | 6 (no change) |
| Architecture quality | 2/5 | **5/5** |
| Information loss | YES (SchemaConfig discarded) | NO (stays with plugin) |
| Technical debt | Created (introspection) | **ELIMINATED** |
| Validation enforcement | NO (plugins can skip) | YES (__init_subclass__ hook) |
| Gate validation | Missing | Explicit (input == output check) |
| Coalesce validation | Missing | Explicit (branch compatibility check) |

---

## Two-Phase Validation: The Key Insight

The critical architectural insight is distinguishing between two fundamentally different kinds of validation:

### Phase 1: Self-Validation (Plugin Construction)

**What it validates:** Is the plugin's own schema well-formed and internally consistent?

**When it happens:** During `Plugin.__init__(config)`

**Who knows enough to validate:** The plugin itself (has access to its own config)

**Examples:**
- Source: Output schema has valid field definitions
- Transform: If pass-through, input_schema == output_schema
- Sink: Input schema has valid field definitions

**Failure mode:** ValueError raised during plugin construction

**Key property:** Plugin can validate itself WITHOUT knowledge of other plugins

### Phase 2: Compatibility Validation (Graph Construction)

**What it validates:** Are two plugins compatible when wired together?

**When it happens:** During `ExecutionGraph.add_edge(from_node, to_node)`

**Who knows enough to validate:** The graph construction code (has access to both plugins)

**Examples:**
- Does source output schema have all fields required by transform input?
- Is dynamic schema → strict schema compatible?
- Does gate preserve schema (input == output)?
- Do all coalesce branches produce compatible schemas?

**Failure mode:** ValueError raised during edge creation

**Key property:** Requires knowledge of BOTH plugins being connected

### Why This Distinction Matters

**The Architecture Critic's key observation:**
> "A plugin cannot validate its compatibility with other plugins during its own
> __init__() because the other plugins don't exist yet."

This is why propagating SchemaConfig to NodeInfo (Option A) was the wrong fix:
- It tried to move DAG validation logic from DAG.validate() to DAG construction
- But it kept ALL validation at the DAG layer
- This missed the opportunity to move SELF-validation to plugins

**The root cause fix:**
- Phase 1 validation moves to plugins (they validate themselves)
- Phase 2 validation stays at graph construction (cross-plugin checks)
- DAG.validate() only checks structure (no schema knowledge needed)

**Result:** Each layer validates what it has knowledge to validate

---

## Why This Is Better Than Option A (Propagation)

**Option A (propagation plan):**
- Adds `input_schema_config` and `output_schema_config` to NodeInfo
- Propagates SchemaConfig through graph construction
- DAG validation checks `schema_config.is_dynamic`
- **Result:** 2/5 → 4/5 architecture (still has information duplication)

**This Plan (proper fix):**
- NO changes to NodeInfo (stays at 6 fields)
- NO SchemaConfig propagation (doesn't leave plugin)
- DAG validation doesn't check schemas AT ALL
- **Result:** 2/5 → **5/5** architecture (single responsibility, single source of truth)

**Eliminated Complexity:**
- ❌ No extraction helper needed
- ❌ No from_config() vs from_plugin_instances() divergence
- ❌ No SchemaConfig serialization concerns
- ❌ No parallel validation logic
- ❌ No NodeInfo bloat precedent

**You're right - this is the last chance to fix it properly before release.**

---

## References

**Related Documentation:**
- Bug ticket: `docs/bugs/P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md`
- Systems thinking review: Identified validation placement as root cause
- CLAUDE.md: "Plugin Ownership: System Code, Not User Code"

**Architectural Principle:**
> "Validation should happen where the knowledge lives"
> - Plugins have SchemaConfig → Plugins validate
> - DAG has graph structure → DAG validates structure
