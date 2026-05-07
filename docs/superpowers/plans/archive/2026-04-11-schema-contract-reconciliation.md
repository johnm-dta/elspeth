# Schema Contract Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three-way disagreement between builder, SchemaContract.merge(), and sink validation on what "required" means after a coalesce union merge.

**Architecture:** Change SchemaContract.merge() from OR to AND semantics for `required` (aligning with builder.py). Add a `required_field_names` property for ergonomic access. Add build-time validation that coalesce output satisfies downstream sink requirements. Make sink validation error messages contract-aware.

**Tech Stack:** Python dataclasses, pytest, SchemaContract (contracts/schema_contract.py), ExecutionGraph (core/dag/graph.py), SinkExecutor (engine/executors/sink.py)

**Filigree issue:** `elspeth-c746590d2b`

---

## Background

Three components disagree on `required` semantics after a coalesce union merge:

| Component | File | Semantics | Logic |
|-----------|------|-----------|-------|
| Builder (build-time) | `builder.py:916-918` | AND — optional if optional in ANY branch | Conservative |
| SchemaContract.merge() (runtime) | `schema_contract.py:489` | OR — required if required in EITHER branch | Aggressive |
| Sink validation (runtime) | `sink.py:177-184` | Independent — uses plugin's `declared_required_fields` | Ignores both |

**Why AND is correct:** For `best_effort`/`quorum` coalesces, a branch that guarantees field X might be lost. The merged row then lacks X even though OR logic would say it's required. AND is safe for all policies. For `require_all`, OR would also be correct (all branches arrive), but using AND uniformly is simpler and safer.

**The real gap:** No build-time check ensures coalesce output fields satisfy downstream sink requirements. A sink that declares `declared_required_fields = {"x"}` will crash at runtime if the coalesce output has `x` as optional.

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/elspeth/contracts/schema_contract.py` | Modify | Fix merge() required semantics, add required_field_names |
| `src/elspeth/core/dag/graph.py` | Modify | Add coalesce→sink required-field build-time validation |
| `src/elspeth/engine/executors/sink.py` | Modify | Contract-aware error messages |
| `tests/unit/contracts/test_schema_contract.py` | Modify | Fix OR→AND test, add required_field_names tests |
| `tests/unit/core/test_dag_coalesce_optionality.py` | Modify | Add sink-required-field validation tests |
| `tests/unit/engine/test_executors.py` | Modify | Add contract-aware error message tests |

---

### Task 1: Fix SchemaContract.merge() required semantics (OR → AND)

**Files:**
- Modify: `src/elspeth/contracts/schema_contract.py:489`
- Modify: `tests/unit/contracts/test_schema_contract.py:945-959`

- [ ] **Step 1: Update the existing test to expect AND semantics**

The test at line 945 is called `test_merge_required_if_either_required` and expects OR. Rename it and flip the expectation:

```python
def test_merge_required_only_if_both_required(self) -> None:
    """Field is required only if required in BOTH paths (AND semantics).

    Why AND: For best_effort/quorum coalesces, a branch that guarantees
    field X might be lost. The merged output can only guarantee X if ALL
    branches that produce X guarantee it.
    """
    c1 = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
        locked=True,
    )
    c2 = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("x", int, original_name="X", required=False, source="inferred"),),
        locked=True,
    )
    merged = c1.merge(c2)

    assert merged.fields[0].required is False
```

- [ ] **Step 2: Run the test to verify it fails (proving current OR behavior)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_schema_contract.py::TestSchemaMerge::test_merge_required_only_if_both_required -v`

Expected: FAIL — `assert True is False` (current OR logic returns True)

- [ ] **Step 3: Add a positive AND test — required when both branches require it**

Add to the same test class:

```python
def test_merge_required_when_both_branches_require(self) -> None:
    """Field is required when required in BOTH paths."""
    c1 = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
        locked=True,
    )
    c2 = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("x", int, original_name="X", required=True, source="declared"),),
        locked=True,
    )
    merged = c1.merge(c2)

    assert merged.fields[0].required is True
```

- [ ] **Step 4: Run both tests to verify the new one also fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_schema_contract.py::TestSchemaMerge::test_merge_required_when_both_branches_require tests/unit/contracts/test_schema_contract.py::TestSchemaMerge::test_merge_required_only_if_both_required -v`

Expected: First PASSES (True and True → True under both OR and AND), second FAILS.

- [ ] **Step 5: Fix the merge() method — change OR to AND**

In `src/elspeth/contracts/schema_contract.py`, line 489, change:

```python
# BEFORE (line 483-489):
                # Use the one that's required if either is
                # Use declared source if either is declared
                merged_fields[name] = FieldContract(
                    normalized_name=name,
                    original_name=self_fc.original_name,
                    python_type=self_fc.python_type,
                    required=self_fc.required or other_fc.required,
```

to:

```python
# AFTER:
                # Required only if BOTH branches require it (AND semantics).
                # Why: for best_effort/quorum coalesces, the branch that
                # guarantees the field might be lost. Safe for require_all too.
                # Use declared source if either is declared
                merged_fields[name] = FieldContract(
                    normalized_name=name,
                    original_name=self_fc.original_name,
                    python_type=self_fc.python_type,
                    required=self_fc.required and other_fc.required,
```

Also update the docstring at line 443 — change rule 2:

```python
        """Merge two contracts at a coalesce point.

        Rules:
        1. Mode: Most restrictive wins (FIXED > FLEXIBLE > OBSERVED)
        2. Fields present in both: Types must match; required only if both require (AND)
        3. Fields in only one: Included but marked non-required
        4. Locked: True if either is locked
```

- [ ] **Step 6: Run both merge tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_schema_contract.py::TestSchemaMerge -v`

Expected: ALL PASS

- [ ] **Step 7: Run the full schema contract test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_schema_contract.py -v`

Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/contracts/schema_contract.py tests/unit/contracts/test_schema_contract.py
git commit -m "fix: SchemaContract.merge() — change required from OR to AND semantics

OR logic (required if either branch requires) is wrong for best_effort/quorum
coalesces where the guaranteeing branch might be lost. AND aligns with
builder.py's conservative semantics: a field is only guaranteed in the merged
output if ALL branches that produce it guarantee it.

Part of elspeth-c746590d2b."
```

---

### Task 2: Add required_field_names property to SchemaContract

**Files:**
- Modify: `src/elspeth/contracts/schema_contract.py`
- Modify: `tests/unit/contracts/test_schema_contract.py`

- [ ] **Step 1: Write the test for required_field_names**

Add to the appropriate test class in `tests/unit/contracts/test_schema_contract.py`:

```python
class TestRequiredFieldNames:
    """Tests for SchemaContract.required_field_names property."""

    def test_returns_required_fields_only(self) -> None:
        """Property returns frozenset of names where required=True."""
        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("a", int, original_name="A", required=True, source="declared"),
                make_field("b", str, original_name="B", required=False, source="declared"),
                make_field("c", float, original_name="C", required=True, source="declared"),
            ),
            locked=True,
        )
        assert contract.required_field_names == frozenset({"a", "c"})

    def test_empty_when_no_required_fields(self) -> None:
        """Property returns empty frozenset when no fields are required."""
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                make_field("x", int, original_name="X", required=False, source="inferred"),
            ),
            locked=True,
        )
        assert contract.required_field_names == frozenset()

    def test_empty_contract(self) -> None:
        """Property returns empty frozenset for empty contract."""
        contract = SchemaContract(mode="OBSERVED", fields=(), locked=True)
        assert contract.required_field_names == frozenset()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_schema_contract.py::TestRequiredFieldNames -v`

Expected: FAIL — `AttributeError: 'SchemaContract' object has no attribute 'required_field_names'`

- [ ] **Step 3: Implement the property**

Add to `SchemaContract` class in `src/elspeth/contracts/schema_contract.py`, after the existing properties:

```python
    @property
    def required_field_names(self) -> frozenset[str]:
        """Normalized names of all fields with required=True."""
        return frozenset(fc.normalized_name for fc in self.fields if fc.required)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_schema_contract.py::TestRequiredFieldNames -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/schema_contract.py tests/unit/contracts/test_schema_contract.py
git commit -m "feat: add SchemaContract.required_field_names property

Ergonomic accessor for the set of field names where required=True.
Used by downstream build-time validation and sink error reporting.

Part of elspeth-c746590d2b."
```

---

### Task 3: Add build-time coalesce→sink required-field validation

**Files:**
- Modify: `src/elspeth/core/dag/graph.py` (add validation in `_validate_coalesce_compatibility` or a new method)
- Modify: `tests/unit/core/test_dag_coalesce_optionality.py`

This is the critical missing check: after computing the coalesce output schema, verify that all downstream sinks' `declared_required_fields` are satisfied.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/core/test_dag_coalesce_optionality.py`:

```python
import pytest
from elspeth.core.dag.graph import ExecutionGraph, GraphValidationError


def test_coalesce_to_sink_required_field_mismatch_raises(
    graph_with_asymmetric_union_coalesce,
) -> None:
    """Build-time validation catches coalesce output missing a sink's required field.

    If coalesce union merge marks field X as optional (branch-exclusive or
    AND-downgraded), but a downstream sink declares X as required, validation
    must fail at build time rather than crash at runtime.
    """
    graph = graph_with_asymmetric_union_coalesce  # fixture: branch A has field X, branch B doesn't

    with pytest.raises(GraphValidationError, match="required by sink.*but optional in coalesce output"):
        graph.validate_schemas()
```

Note: this test needs a fixture. We'll build it in Step 3.

- [ ] **Step 2: Write the fixture for asymmetric coalesce → sink**

Add the fixture to `tests/unit/core/test_dag_coalesce_optionality.py` (or a conftest if one exists for this directory). The fixture needs to build a DAG where:
- A fork produces two branches
- Branch A has field `x` (required), branch B does not
- Union merge coalesce combines them → field `x` is optional in merged output
- A downstream sink declares `x` in `declared_required_fields`

```python
@pytest.fixture
def graph_with_asymmetric_union_coalesce() -> ExecutionGraph:
    """DAG: source → fork → [branch_a(has x), branch_b(no x)] → coalesce(union) → sink(requires x)."""
    from tests.fixtures.graph_builders import build_fork_coalesce_graph

    return build_fork_coalesce_graph(
        branch_a_fields=[("id", "str", True), ("x", "int", True)],
        branch_b_fields=[("id", "str", True)],
        merge_strategy="union",
        sink_required_fields=frozenset({"id", "x"}),
    )
```

**Important:** Check `tests/fixtures/` for existing graph builder helpers before creating `build_fork_coalesce_graph`. If none exist, implement a minimal one that constructs the graph programmatically using `ExecutionGraph.from_plugin_instances()` or the builder. The fixture must use production code paths (per CLAUDE.md: "Never bypass production code paths in tests").

- [ ] **Step 3: Verify the test fails (no validation exists yet)**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_coalesce_optionality.py::test_coalesce_to_sink_required_field_mismatch_raises -v`

Expected: FAIL — `GraphValidationError` not raised (validation doesn't exist yet)

- [ ] **Step 4: Implement build-time validation in graph.py**

Add a new method to `ExecutionGraph` and call it from `validate_schemas()` (line 960, after coalesce compatibility checks):

```python
    def _validate_coalesce_sink_requirements(self) -> None:
        """Validate coalesce output satisfies downstream sink required fields.

        For each coalesce node with union merge strategy, check that all
        downstream sinks' declared_required_fields are a subset of the
        coalesce output's guaranteed (required) fields. If a sink requires
        a field that the coalesce marks optional, validation fails at build
        time rather than crashing at runtime.
        """
        coalesce_nodes = [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data["info"].node_type == NodeType.COALESCE
        ]

        for coalesce_id in coalesce_nodes:
            node_info = self.get_node_info(coalesce_id)
            schema_config = node_info.output_schema_config
            if schema_config is None or schema_config.fields is None:
                continue  # Observed schemas — can't validate at build time

            # Build set of guaranteed fields from coalesce output schema
            guaranteed = frozenset(
                fd.name for fd in schema_config.fields if fd.required
            )

            # Check each downstream sink
            for _, successor_id in self._graph.out_edges(coalesce_id):
                successor_info = self.get_node_info(successor_id)
                if successor_info.node_type != NodeType.SINK:
                    continue

                sink_instance = successor_info.plugin_instance
                if sink_instance is None:
                    continue

                sink_required = sink_instance.declared_required_fields
                if not sink_required:
                    continue

                missing = sink_required - guaranteed
                if missing:
                    # Check which missing fields exist but are optional
                    all_field_names = frozenset(fd.name for fd in schema_config.fields)
                    optional_but_required = missing & all_field_names
                    absent_entirely = missing - all_field_names

                    parts = []
                    if optional_but_required:
                        parts.append(
                            f"fields {sorted(optional_but_required)} are optional in "
                            f"coalesce output (branch-exclusive or AND-downgraded)"
                        )
                    if absent_entirely:
                        parts.append(
                            f"fields {sorted(absent_entirely)} are absent from coalesce output entirely"
                        )

                    raise GraphValidationError(
                        f"Sink '{successor_info.name}' has fields "
                        f"required by sink {sorted(missing)} but optional in coalesce output "
                        f"from '{coalesce_id}': {'; '.join(parts)}. "
                        f"Fix: ensure all branches produce the required fields, "
                        f"or remove them from the sink's declared_required_fields."
                    )
```

Then add the call in `validate_schemas()`, after line 960:

```python
        # Validate all coalesce nodes (must have compatible schemas from all branches)
        coalesce_nodes = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == NodeType.COALESCE]
        for coalesce_id in coalesce_nodes:
            self._validate_coalesce_compatibility(coalesce_id, _schema_cache=schema_cache)

        # Validate coalesce output satisfies downstream sink requirements
        self._validate_coalesce_sink_requirements()
```

**Note:** The `coalesce_nodes` list is computed twice. Extract it to a local variable above both loops to avoid duplication.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_coalesce_optionality.py::test_coalesce_to_sink_required_field_mismatch_raises -v`

Expected: PASS

- [ ] **Step 6: Add positive test — sink requirements satisfied**

```python
def test_coalesce_to_sink_validates_when_all_fields_guaranteed(
    graph_with_symmetric_union_coalesce,
) -> None:
    """Build-time validation passes when coalesce guarantees all sink-required fields."""
    graph = graph_with_symmetric_union_coalesce  # both branches have field x required
    graph.validate_schemas()  # Should not raise
```

With fixture:

```python
@pytest.fixture
def graph_with_symmetric_union_coalesce() -> ExecutionGraph:
    """DAG: source → fork → [branch_a(has x), branch_b(has x)] → coalesce(union) → sink(requires x)."""
    from tests.fixtures.graph_builders import build_fork_coalesce_graph

    return build_fork_coalesce_graph(
        branch_a_fields=[("id", "str", True), ("x", "int", True)],
        branch_b_fields=[("id", "str", True), ("x", "int", True)],
        merge_strategy="union",
        sink_required_fields=frozenset({"id", "x"}),
    )
```

- [ ] **Step 7: Run the positive test**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_coalesce_optionality.py::test_coalesce_to_sink_validates_when_all_fields_guaranteed -v`

Expected: PASS

- [ ] **Step 8: Run all DAG coalesce optionality tests**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_coalesce_optionality.py -v`

Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/core/dag/graph.py tests/unit/core/test_dag_coalesce_optionality.py
git commit -m "feat: build-time validation — coalesce output vs sink required fields

Adds _validate_coalesce_sink_requirements() to ExecutionGraph, called during
validate_schemas(). Catches at build time when a sink requires fields that
are optional in the coalesce output (branch-exclusive or AND-downgraded).

Previously this was only discovered at runtime as a PluginContractViolation,
with no indication that the root cause was coalesce merge semantics.

Part of elspeth-c746590d2b."
```

---

### Task 4: Make sink validation error messages contract-aware

**Files:**
- Modify: `src/elspeth/engine/executors/sink.py:152-184`
- Modify: `tests/unit/engine/test_executors.py`

The sink validation at lines 177-184 checks `declared_required_fields` against raw rows but gives a generic error message. When the contract shows a field is optional (post-merge), the error should say so.

- [ ] **Step 1: Write the test for contract-aware error message**

Add to the sink executor test class in `tests/unit/engine/test_executors.py`:

```python
def test_missing_required_field_after_coalesce_shows_contract_context(self) -> None:
    """When a required field is missing and the contract marks it optional,
    error message includes coalesce merge context."""
    factory = _make_factory()
    executor = SinkExecutor(factory.execution, factory.data_flow, _make_span_factory(), run_id="test-run")

    # Contract says 'name' is optional (result of coalesce merge)
    contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(
            make_field("id", str, original_name="id", required=True, source="declared"),
            make_field("name", str, original_name="name", required=False, source="declared"),
        ),
        locked=True,
    )
    token = _make_token(data={"id": "1"}, contract=contract)  # Missing 'name'
    sink = _make_sink()
    sink.declared_required_fields = frozenset({"id", "name"})
    ctx = make_context()
    pending = PendingOutcome(outcome=RowOutcome.COMPLETED)

    with pytest.raises(
        PluginContractViolation,
        match=r"optional in the row's schema contract.*coalesce merge",
    ):
        executor.write(sink, [token], ctx, step_in_pipeline=5, sink_name="out", pending_outcome=pending)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py::TestSinkExecutor::test_missing_required_field_after_coalesce_shows_contract_context -v`

Expected: FAIL — error message doesn't contain "optional in the row's schema contract"

- [ ] **Step 3: Update _validate_sink_input to include contract context**

Modify `src/elspeth/engine/executors/sink.py` — change `_validate_sink_input` signature and body:

```python
    @staticmethod
    def _validate_sink_input(
        sink: SinkProtocol,
        rows: list[dict[str, object]],
        *,
        skip_schema: bool = False,
        contracts: list[SchemaContract] | None = None,
    ) -> None:
        """Validate rows against a sink's input schema and required fields.

        Args:
            sink: Sink to validate against.
            rows: Row dicts to validate.
            skip_schema: If True, skip input_schema.model_validate() and only
                check required fields. Used for failsink validation where the
                executor injects enrichment fields (__diversion_*) that are
                outside the failsink's declared schema.
            contracts: Optional per-row SchemaContracts for context-aware error messages.
        """
        if not skip_schema:
            for row in rows:
                try:
                    sink.input_schema.model_validate(row)
                except ValidationError as e:
                    raise PluginContractViolation(
                        f"Sink '{sink.name}' input validation failed: {e}. This indicates an upstream transform/source schema bug."
                    ) from e

        if sink.declared_required_fields:
            for row_index, row in enumerate(rows):
                missing = sorted(f for f in sink.declared_required_fields if f not in row)
                if missing:
                    # Check if any missing fields are optional in the contract (coalesce merge artifact)
                    contract_context = ""
                    if contracts and row_index < len(contracts):
                        contract = contracts[row_index]
                        contract_field_names = {fc.normalized_name for fc in contract.fields}
                        optional_in_contract = [
                            f for f in missing
                            if f in contract_field_names and f not in contract.required_field_names
                        ]
                        if optional_in_contract:
                            contract_context = (
                                f" Fields {optional_in_contract} are optional in the row's "
                                f"schema contract (likely from coalesce merge). "
                                f"Fix: ensure all branches produce these fields as required."
                            )
                    raise PluginContractViolation(
                        f"Sink '{sink.name}' row {row_index} is missing required fields "
                        f"{missing}. This indicates an upstream transform/schema bug.{contract_context}"
                    )
```

- [ ] **Step 4: Update the call site in write() to pass contracts**

In `sink.py`, find where `_validate_sink_input` is called (around line 290-295 in the write method). Update to pass contracts:

```python
        # Pass per-row contracts for context-aware error messages
        row_contracts = [t.row_data.contract for t in tokens]
        self._validate_sink_input(sink, rows, contracts=row_contracts)
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py::TestSinkExecutor::test_missing_required_field_after_coalesce_shows_contract_context -v`

Expected: PASS

- [ ] **Step 6: Run all existing sink executor tests to check for regressions**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py -v -x`

Expected: ALL PASS (existing tests don't pass `contracts` — the parameter is optional with default `None`)

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/engine/executors/sink.py tests/unit/engine/test_executors.py
git commit -m "fix: sink validation — contract-aware error messages for missing required fields

When a sink's declared_required_fields check fails and the missing field is
optional in the row's SchemaContract (artifact of coalesce merge), the error
message now says so. Helps operators trace the root cause to coalesce
branch asymmetry rather than a generic 'upstream schema bug'.

Part of elspeth-c746590d2b."
```

---

### Task 5: Run full test suite and CI checks

**Files:** None (verification only)

- [ ] **Step 1: Run unit tests**

Run: `.venv/bin/python -m pytest tests/unit/ -x -q`

Expected: ALL PASS

- [ ] **Step 2: Run property tests**

Run: `.venv/bin/python -m pytest tests/property/ -x -q`

Expected: ALL PASS

- [ ] **Step 3: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/ -x -q`

Expected: ALL PASS

- [ ] **Step 4: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/schema_contract.py src/elspeth/core/dag/graph.py src/elspeth/engine/executors/sink.py`

Expected: No new errors

- [ ] **Step 5: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/contracts/schema_contract.py src/elspeth/core/dag/graph.py src/elspeth/engine/executors/sink.py`

Expected: No new warnings

- [ ] **Step 6: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`

Expected: PASS (no new cross-layer violations — SchemaContract is L0, graph.py is L1, sink.py is L2; all imports flow downward)

- [ ] **Step 7: Run config contracts check**

Run: `.venv/bin/python -m scripts.check_contracts`

Expected: PASS

- [ ] **Step 8: Run freeze guard enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_freeze_guards.py`

Expected: PASS (SchemaContract is already frozen; no new mutable container fields added)
