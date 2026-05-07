# Composer Schema Contract Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the composer reject pipelines with unsatisfied schema contracts at composition time, and expose per-edge contract data in the preview response.

**Architecture:** Add a new validation pass (pass 9) to `CompositionState.validate()` that walks the connection-field chain but reuses shared raw-config contract helpers extracted into `elspeth.contracts.schema` rather than re-implementing runtime semantics inline. The pass must validate connection-namespace integrity before building `producer_map` so invalid drafts fail closed instead of silently overwriting producers. The shared helpers are responsible for parsing schema config while preserving the existing `schema` / `schema_config` alias contract, computing producer guarantees, computing node consumer requirements (top-level `required_input_fields`, aggregation `options.required_input_fields`, then explicit `schema.required_fields`), computing sink consumer requirements via `schema.get_effective_required_fields()`, and applying the closed-list observed-only text-source deterministic guarantee rule. Unsatisfied contracts remain error-level and therefore must force `ValidationSummary.is_valid = False`. Separately, add a mechanical backstop on the LLM-facing YAML-export surfaces (`src/elspeth/composer_mcp/server.py` and `GET /api/sessions/{id}/state/yaml`): those entry points must call `state.validate()` and refuse YAML when `is_valid` is false. Do **not** move that guard into `yaml_generator.generate_yaml()`, which must stay a pure serializer for execution-time validation and run creation. Extend `ValidationSummary` with `EdgeContract` tuples, update `ExecutionGraph` to delegate its raw-config requirement extraction to the same helpers, refactor the gate/coalesce walk-back into a shared helper in `state.py`, and patch both pipeline-composer skill files with contract-aware completion criteria that remain stricter than today's behaviour.

**Tech Stack:** Python dataclasses (frozen), `SchemaConfig` from `elspeth.contracts.schema`, pytest

**Spec:** `docs/superpowers/specs/2026-04-14-composer-schema-contract-validation-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/elspeth/contracts/schema.py` | Modify | Add shared raw-config contract helpers used by both composer and runtime; centralize observed-only text-source guarantee rule and sink `get_effective_required_fields()` semantics |
| `src/elspeth/core/dag/graph.py` | Modify | Delegate raw-config consumer requirement extraction to the shared helpers instead of maintaining a second copy of the logic |
| `src/elspeth/web/composer/state.py` | Modify | Add `EdgeContract` dataclass, extend `ValidationSummary`, add `_check_schema_contracts()` connection-walk that calls the shared helpers, call it from `validate()` |
| `src/elspeth/web/composer/tools.py` | Modify | Serialize `edge_contracts` in `_execute_preview_pipeline()` |
| `src/elspeth/composer_mcp/server.py` | Modify | Refuse `generate_yaml` when current state validation fails; preserve `yaml_generator.generate_yaml()` as a pure serializer below that boundary |
| `src/elspeth/web/sessions/routes.py` | Modify | Apply the same validation gate to `GET /api/sessions/{id}/state/yaml` so MCP and HTTP export surfaces cannot bypass composer validation |
| `src/elspeth/plugins/sources/text_source.py` | Modify | Auto-declare `guaranteed_fields` for deterministic observed-schema output only, sync the builder-visible raw schema config, and add enforcement comment |
| `src/elspeth/web/composer/skills/pipeline_composer.md` | Modify | Add completion criteria item 4, align guidance with the observed-text special-case without weakening the general validation stance, fix flow example |
| `.claude/skills/pipeline-composer/SKILL.md` | Modify | Keep the Claude skill's overlapping composer guidance and plugin quick-reference content aligned with the web skill; do not leave a second stale skill copy behind |
| `tests/unit/composer_mcp/test_server.py` | Modify | Add YAML-export backstop test for invalid composition state |
| `tests/unit/web/composer/test_tools.py` | Modify | Add preview/explanation tests for edge contract telemetry and contract-specific `explain_validation_error` guidance |
| `tests/unit/web/composer/test_state.py` | Modify | Add `TestSchemaContractValidation` class with test cases |
| `tests/unit/web/sessions/test_routes.py` | Modify | Add HTTP YAML-export rejection test for invalid current state |
| `tests/unit/web/composer/test_schema_contract_enforcement.py` | Create | Heuristic enforcement test (test case 20) |
| `tests/integration/pipeline/test_composer_runtime_agreement.py` | Create | Composer/runtime agreement test (test case 19), colocated with pipeline/DAG contract integration coverage rather than web-route tests |

---

## Sub-Plan Structure

This plan is intentionally split into three batchable sub-plans. Do **not** split an individual task across sub-plans; the grouping is by ownership / execution slice, not by step-level decomposition.

| Sub-Plan | Tasks | Scope | Batch Outcome |
|----------|-------|-------|---------------|
| **Sub-Plan A: Contract Core** | 1-4 | Core composer contract model, shared-helper wiring, and unit coverage for topology / edge-contract semantics | Composer can compute and validate schema contracts with stable unit-level coverage |
| **Sub-Plan B: Agent-Facing Surfaces** | 5, 5B, 5C | Preview response, YAML export guardrails, and validation-explanation support | LLM/user-facing surfaces expose contract data and refuse invalid exports |
| **Sub-Plan C: Runtime Parity and Release Gate** | 6-11 | Observed-text runtime backing, agreement/regression tests, skill updates, and final verification | Runtime and composer agree on the shared contract rules, guidance is updated, and the full verification gate is green |

**Execution order:** Run the sub-plans in order `A -> B -> C`.

**Dependency notes:**
- Sub-Plan B assumes Sub-Plan A has already produced `edge_contracts` and blocking contract validation.
- Sub-Plan C assumes Sub-Plan A is in place and must not enable the observed-text positive cases until Task 6 lands.
- Task 7 remains a sequencing checkpoint inside Sub-Plan C even though it has no implementation body.

---

## Sub-Plan A: Contract Core

### Task 1: Add `EdgeContract` Dataclass and Extend `ValidationSummary`

**Files:**
- Modify: `src/elspeth/web/composer/state.py:198-229`
- Test: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write the failing test for EdgeContract**

Add to `tests/unit/web/composer/test_state.py`, after the existing `TestValidationSummary` class:

```python
class TestEdgeContract:
    def test_frozen(self) -> None:
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="add_world",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        with pytest.raises(AttributeError):
            ec.satisfied = False  # type: ignore[misc]

    def test_to_dict_uses_from_key(self) -> None:
        """EdgeContract.to_dict() serializes from_id as 'from' (JSON key)."""
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="add_world",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        d = ec.to_dict()
        assert d["from"] == "source"
        assert d["to"] == "add_world"
        assert d["producer_guarantees"] == ["text"]
        assert d["consumer_requires"] == ["text"]
        assert d["missing_fields"] == []
        assert d["satisfied"] is True

    def test_to_dict_empty_fields(self) -> None:
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="sink",
            producer_guarantees=(),
            consumer_requires=(),
            missing_fields=(),
            satisfied=True,
        )
        d = ec.to_dict()
        assert d["producer_guarantees"] == []
        assert d["consumer_requires"] == []
        assert d["missing_fields"] == []


class TestValidationSummaryEdgeContracts:
    def test_default_empty(self) -> None:
        vs = ValidationSummary(is_valid=True, errors=())
        assert vs.edge_contracts == ()

    def test_with_edge_contracts(self) -> None:
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="t1",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        vs = ValidationSummary(is_valid=True, errors=(), edge_contracts=(ec,))
        assert len(vs.edge_contracts) == 1
        assert vs.edge_contracts[0].satisfied is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestEdgeContract -v`
Expected: FAIL with `ImportError: cannot import name 'EdgeContract'`

- [ ] **Step 3: Implement EdgeContract and extend ValidationSummary**

In `src/elspeth/web/composer/state.py`, after `ValidationEntry` (around line 215), add:

```python
@dataclass(frozen=True, slots=True)
class EdgeContract:
    """Schema contract check result for a single producer->consumer edge.

    All fields are scalars or tuples of strings. frozen=True is sufficient.
    """

    from_id: str
    to_id: str
    producer_guarantees: tuple[str, ...]
    consumer_requires: tuple[str, ...]
    missing_fields: tuple[str, ...]
    satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON responses.

        Uses 'from'/'to' as JSON keys (not from_id/to_id) for readability.
        """
        return {
            "from": self.from_id,
            "to": self.to_id,
            "producer_guarantees": list(self.producer_guarantees),
            "consumer_requires": list(self.consumer_requires),
            "missing_fields": list(self.missing_fields),
            "satisfied": self.satisfied,
        }
```

Then modify `ValidationSummary` to add the new field:

```python
@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Stage 1 validation result.

    errors block execution. warnings are advisory but actionable.
    suggestions are optional improvements. edge_contracts shows per-edge
    schema contract check results. All are tuples for structured attribution.
    """

    is_valid: bool
    errors: tuple[ValidationEntry, ...]
    warnings: tuple[ValidationEntry, ...] = ()
    suggestions: tuple[ValidationEntry, ...] = ()
    edge_contracts: tuple[EdgeContract, ...] = ()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestEdgeContract tests/unit/web/composer/test_state.py::TestValidationSummaryEdgeContracts -v`
Expected: PASS

- [ ] **Step 5: Run full existing test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All existing tests PASS (EdgeContract default=() is backward-compatible)

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/state.py tests/unit/web/composer/test_state.py
git commit -m "feat(composer): add EdgeContract dataclass and extend ValidationSummary"
```

---

### Task 2: Implement Schema Contract Validation Pass

**Files:**
- Modify: `src/elspeth/contracts/schema.py`
- Modify: `src/elspeth/core/dag/graph.py`
- Modify: `src/elspeth/web/composer/state.py:460-775`
- Test: `tests/unit/web/composer/test_state.py`

This is the core implementation. We write tests first for the positive and negative cases, then extract shared raw-config contract helpers into `elspeth.contracts.schema`, delegate the runtime's raw-config requirement reads to those helpers, and finally implement `_check_schema_contracts()` on top of the shared layer.

**Dependency note:** The generic contract pass can land before the observed-text special case, but the observed-text acceptance rule must not be enabled in composer unless the Task 6 runtime-backing implementation lands in the same changeset or earlier. If tasks are committed atomically, keep the observed-text-positive composer tests (`test_text_heuristic_infers_guarantee`, Task 8 observed-text accept, and Task 9's reported scenario regression) out of the Task 2 commit and enable them only when the runtime has matching backing.

- [ ] **Step 1: Write failing tests — positive cases (1-5)**

Add to `tests/unit/web/composer/test_state.py`, a new class after `TestStage1Validation`:

```python
class TestSchemaContractValidation:
    """Tests for schema contract validation (pass 9) in CompositionState.validate()."""

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _make_source(
        self,
        on_success: str = "t1",
        plugin: str = "csv",
        options: dict[str, Any] | None = None,
    ) -> SourceSpec:
        opts = dict(options or {})
        if plugin == "csv":
            opts = {"path": "/data/input.csv", **opts}
        elif plugin == "text":
            opts = {"path": "/data/input.txt", "column": "text", **opts}
        return SourceSpec(
            plugin=plugin,
            on_success=on_success,
            options=opts,
            on_validation_failure="quarantine",
        )

    def _make_transform(
        self,
        id: str,
        input: str,
        on_success: str,
        plugin: str = "value_transform",
        options: dict[str, Any] | None = None,
    ) -> NodeSpec:
        opts = dict(options or {})
        if plugin == "value_transform":
            opts = {
                "schema": {"mode": "observed"},
                "operations": [{"target": "_placeholder", "expression": "row['text']"}],
                **opts,
            }
        return NodeSpec(
            id=id,
            node_type="transform",
            plugin=plugin,
            input=input,
            on_success=on_success,
            on_error=None,
            options=opts,
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_gate(
        self,
        id: str,
        input: str,
        routes: dict[str, str],
        condition: str = "row['x'] > 0",
    ) -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="gate",
            plugin=None,
            input=input,
            on_success=None,
            on_error=None,
            options={},
            condition=condition,
            routes=routes,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        )

    def _make_coalesce(
        self,
        id: str,
        input: str,
        on_success: str,
        branches: tuple[str, ...] = ("branch_a", "branch_b"),
        policy: str = "require_all",
        merge: str = "nested",
    ) -> NodeSpec:
        return NodeSpec(
            id=id,
            node_type="coalesce",
            plugin=None,
            input=input,
            on_success=on_success,
            on_error=None,
            options={},
            condition=None,
            routes=None,
            fork_to=None,
            branches=branches,
            policy=policy,
            merge=merge,
        )

    def _make_output(self, name: str = "main") -> OutputSpec:
        return OutputSpec(
            name=name,
            plugin="csv",
            options={"path": f"outputs/{name}.csv", "schema": {"mode": "observed"}},
            on_write_failure="discard",
        )

    def _make_edge(
        self,
        id: str,
        from_node: str,
        to_node: str,
        edge_type: EdgeType = "on_success",
    ) -> EdgeSpec:
        return EdgeSpec(id=id, from_node=from_node, to_node=to_node, edge_type=edge_type, label=None)

    # --- Positive cases ---

    def test_fixed_schema_satisfies_requirement(self) -> None:
        """Test 1: Source with fixed schema guaranteeing 'text' satisfies consumer."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors
        assert not any("contract" in e.message.lower() for e in result.errors)

    def test_text_heuristic_infers_guarantee(self) -> None:
        """Test 2: Text source with column + observed schema — heuristic infers guarantee.

        The composer infers {column} as a guaranteed field for text sources
        with observed schema and no explicit guaranteed_fields. This mirrors
        what TextSource does at runtime by updating both its normalized
        SchemaConfig and the builder-visible raw config dict, so both the
        composer and the runtime agree.

        Staging rule: this test must land with Task 6, not in the initial
        generic contract-pass commit, because runtime backing is added there.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            plugin="text",
            options={"column": "text", "schema": {"mode": "observed"}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_text_heuristic_respects_explicit_guaranteed_fields(self) -> None:
        """Test 3: Text source with explicit guaranteed_fields — heuristic defers."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            plugin="text",
            options={
                "column": "text",
                "schema": {"mode": "observed", "guaranteed_fields": ["text"]},
            },
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_no_required_input_fields_skips_check(self) -> None:
        """Test 4: Consumer with no required_input_fields — no error."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "observed"}},
        ))
        state = state.with_node(self._make_transform("t1", "t1", "main"))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_empty_required_input_fields_skips_check(self) -> None:
        """Test 5: Consumer with required_input_fields: [] and no schema.required_fields — no error."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "observed"}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": []},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    # --- Post-implementation evidence cases ---
    # These sink/direct-contract cases assert populated edge_contracts, so they are NOT part of
    # the pre-implementation "current behavior stays green" run below. They
    # become green only after EdgeContract exists and pass 9 populates it.

    def test_source_direct_to_sink_records_contract(self) -> None:
        """Test 5a: Source → sink with no transforms exercises the sink loop directly."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="main",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_output(OutputSpec(
            name="main",
            plugin="csv",
            options={
                "path": "outputs/main.csv",
                "schema": {"mode": "observed", "required_fields": ["text"]},
            },
            on_write_failure="discard",
        ))
        result = state.validate()
        assert result.is_valid, result.errors
        assert len(result.edge_contracts) == 1
        sink_contract = result.edge_contracts[0]
        assert sink_contract.from_id == "source"
        assert sink_contract.to_id == "output:main"
        assert sink_contract.satisfied is True

    def test_sink_required_fields_satisfied(self) -> None:
        """Test 5b: Source-only pipeline sink required_fields satisfied by upstream guarantees."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="main",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_output(OutputSpec(
            name="main",
            plugin="csv",
            options={
                "path": "outputs/main.csv",
                "schema": {"mode": "observed", "required_fields": ["text"]},
            },
            on_write_failure="discard",
        ))
        result = state.validate()
        assert result.is_valid, result.errors
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.satisfied is True
        assert "text" in sink_contract.consumer_requires

    def test_consumer_schema_required_fields_satisfied(self) -> None:
        """Test 5c: Consumer falls back to explicit schema.required_fields."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"schema": {"mode": "observed", "required_fields": ["text"]}},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors
        edge_contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        assert edge_contract.satisfied is True
        assert edge_contract.consumer_requires == ("text",)
```

- [ ] **Step 2: Run the pre-implementation positive cases (1-5 only)**

Run:

```bash
.venv/bin/python -m pytest \
  tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_fixed_schema_satisfies_requirement \
  tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_no_required_input_fields_skips_check \
  tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_empty_required_input_fields_skips_check \
  -v
```

Expected: PASS on current code. These cases only assert that current Stage 1 validation stays green when no contract error is expected.

Do **not** include `test_source_direct_to_sink_records_contract`, `test_sink_required_fields_satisfied`, or `test_consumer_schema_required_fields_satisfied` in this pre-implementation run. Those cases assert populated `edge_contracts`, so they are expected to fail until `EdgeContract` exists and `_check_schema_contracts()` is wired into `validate()`. Also keep the observed-text-positive cases staged with Task 6 when commits are split, per the dependency note above. These cases become part of the post-implementation verification in Step 7.

- [ ] **Step 3: Write failing tests — negative cases (6-10)**

Append to `TestSchemaContractValidation`:

```python
    # --- Negative cases ---

    def test_observed_schema_no_guarantees_fails(self) -> None:
        """Test 6: Observed schema with no guarantees, consumer requires 'text'."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "observed"}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema contract violation" in e.message.lower() for e in result.errors)
        assert any("text" in e.message for e in result.errors)

    def test_partial_match_fails(self) -> None:
        """Test 7: Producer guarantees ['text'], consumer requires ['text', 'score']."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text", "score"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("score" in e.message for e in result.errors)

    def test_optional_field_not_guaranteed(self) -> None:
        """Test 8: Optional field (text: str?) is NOT guaranteed."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str?"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)

    def test_no_schema_config_fails(self) -> None:
        """Test 9: Source with no schema config at all — no guarantees."""
        state = self._empty_state()
        state = state.with_source(self._make_source(options={}))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema contract violation" in e.message.lower() for e in result.errors)

    def test_malformed_schema_emits_error(self) -> None:
        """Test 10: Malformed schema dict — error emitted, is_valid false."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "invalid_mode"}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema" in e.message.lower() for e in result.errors)

    def test_sink_required_fields_violation_fails(self) -> None:
        """Test 10b: Sink required_fields violated by upstream guarantees."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="main",
            options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
        ))
        state = state.with_output(OutputSpec(
            name="main",
            plugin="csv",
            options={
                "path": "outputs/main.csv",
                "schema": {"mode": "observed", "required_fields": ["text"]},
            },
            on_write_failure="discard",
        ))
        result = state.validate()
        assert not result.is_valid
        assert any("sink" in e.message.lower() and "text" in e.message.lower() for e in result.errors)
        sink_contract = next(ec for ec in result.edge_contracts if ec.to_id == "output:main")
        assert sink_contract.satisfied is False
        assert "text" in sink_contract.missing_fields

    def test_consumer_schema_required_fields_violation_fails(self) -> None:
        """Test 10c: required_input_fields=[] still falls through to schema.required_fields."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={
                "required_input_fields": [],
                "schema": {"mode": "observed", "required_fields": ["text"]},
            },
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)

    def test_malformed_consumer_schema_emits_error(self) -> None:
        """Test 10d: Malformed consumer schema is a blocking error, not empty requirements."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"schema": {"mode": "invalid_mode"}},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("schema" in e.message.lower() for e in result.errors)
        assert not any(ec.to_id == "t1" for ec in result.edge_contracts)
```

- [ ] **Step 4: Run negative tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_observed_schema_no_guarantees_fails tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_partial_match_fails tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_no_schema_config_fails tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_consumer_schema_required_fields_violation_fails tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_malformed_consumer_schema_emits_error -v`
Expected: FAIL (current validate() doesn't check contracts)

- [ ] **Step 5: Add shared raw-config helpers and implement `_check_schema_contracts()`**

In `src/elspeth/contracts/schema.py`, add a small public helper surface that both the composer and runtime can reuse:

```python
def parse_raw_schema_config(raw_schema: object, *, owner: str) -> SchemaConfig | None: ...
def get_raw_producer_guaranteed_fields(
    plugin_name: str | None,
    options: Mapping[str, Any],
    *,
    owner: str,
) -> frozenset[str]: ...
def get_raw_node_required_fields(
    options: Mapping[str, Any],
    *,
    owner: str,
    node_type: str | None = None,
) -> frozenset[str]: ...
def get_raw_sink_required_fields(
    options: Mapping[str, Any],
    *,
    owner: str,
) -> frozenset[str]: ...
```

Rules for the shared helper layer:

- Each shared helper docstring must state its `ValueError` protocol explicitly: "Raises ValueError for conditions that should surface as pipeline validation errors, not for programming errors at the call site."
- `parse_raw_schema_config()` raises `ValueError` for malformed schema dicts.
- The shared helper layer must preserve the existing alias contract from `PluginConfig`: whenever raw options are inspected for schema, accept both the user-facing `schema` alias and the backwards-compatible `schema_config` field name. Do not centralize parsing in a way that strands alias handling in `state.py` or drops parity with plugin config parsing (`populate_by_name=True` / `from_dict()` prevalidation).
- Define the closed-list observed-text heuristic marker at module scope in `src/elspeth/contracts/schema.py`, not inside a helper body:
  `_TEXT_HEURISTIC_PLUGINS: frozenset[str] = frozenset({"text"})`
  Keep the existing "do not extend without design review" comment adjacent to that constant so discoverability survives refactors.
- `get_raw_producer_guaranteed_fields()` uses `SchemaConfig.get_effective_guaranteed_fields()`. The observed-text special case is staged with Task 6 and applies only when `plugin_name == "text"`, `column` is a non-empty string, `schema.mode == "observed"`, and `declares_guaranteed_fields is False`.
- `get_raw_node_required_fields()` mirrors today's runtime `ExecutionGraph.get_required_fields()` semantics exactly: top-level `required_input_fields` first, then aggregation-nested `options["required_input_fields"]` when `node_type == "aggregation"` and the raw config is wrapper-shaped, then explicit `schema.required_fields`. An empty list does not block the fallback; `required_input_fields: []` means "no Priority 1 requirements declared", not "ignore schema.required_fields". Reject bare-string `required_input_fields` with `ValueError` at either level.
- `get_raw_sink_required_fields()` is intentionally stricter: parse the sink schema and return `schema.get_effective_required_fields()`, so fixed/flexible typed sink fields are treated as required even when `required_fields` is omitted explicitly.
- `state.py` and `graph.py` must not keep independent copies of the observed-text rule or sink requirement extraction.
- The helper surface must be shape-tolerant for aggregation configs: composer passes `node.options` directly, runtime passes the wrapped aggregation config whose plugin options live under `"options"`. The shared helper owns that normalization so the two call sites stay coupled.
- Keep `tests/unit/web/composer/test_tools.py::test_preview_source_with_schema_config_field_name` green. If alias resolution moves fully into `elspeth.contracts.schema`, add direct helper-level coverage there too rather than relying only on the preview test as an indirect guard.
- Keep helper side effects explicit. If a nested helper can append diagnostics, pass `errors`/`warnings` into it explicitly rather than mutating hidden closure state. In practice `_register_producer(..., errors)` and `_walk_to_real_producer(..., warnings)` should advertise their side effects in their signatures.

In `src/elspeth/core/dag/graph.py`, update the raw-config requirement extraction path to delegate to the shared helper surface instead of re-implementing the parsing logic inline. Runtime graph validation still owns graph walking, coalesce semantics, and `GraphValidationError`; the shared helper only owns raw-config contract semantics.

In `src/elspeth/web/composer/state.py`, keep the topology walk but replace the inline schema logic with top-level imports of the shared helpers. Do **not** add a new deferred import here: `state.py` (L3) importing `elspeth.contracts.schema` (L0) is layer-legal, and CLAUDE.md explicitly treats new lazy imports as the "Shifting the Burden" archetype rather than a pattern to spread. Preserve the current `_source_options_have_schema()` behaviour by delegating to the same alias rule (or an extracted shared alias helper) so preview summaries and contract validation cannot disagree about whether a source carries schema:

```python
from elspeth.contracts.schema import (
    get_raw_node_required_fields,
    get_raw_producer_guaranteed_fields,
    get_raw_sink_required_fields,
)
```

Then implement `_check_schema_contracts()` with these rules:

- before any contract walk, validate connection namespace integrity and abort the contract pass on ambiguity. Mirror the runtime builder's fail-closed policy for the subset needed to keep telemetry trustworthy:
  - duplicate producers for the same connection name => blocking error
  - duplicate consumers for the same connection name => blocking error
  - connection names overlapping sink names => blocking error
  - on any of these, return no `EdgeContract` rows rather than picking a winner via dict overwrite
  - do not assume an earlier Stage 1 check already enforces these rules; `_check_schema_contracts()` must defend itself because current composer validation does not yet reject this entire class of namespace ambiguity
- define a small internal `_ProducerEntry(NamedTuple)` at module scope in `state.py` for `producer_map` values (for example `producer_id`, `plugin_name`, `options`). Do not define it inside `_check_schema_contracts()`; `_walk_to_real_producer(...)` should read through named fields and mypy should see a stable named type across helper signatures.
- producer helper `ValueError` => emit blocking `ValidationEntry`, mark the producer as uncheckable, and suppress downstream edge contracts sourced from that producer
- node consumer helper `ValueError` => emit blocking `ValidationEntry` on that node and skip its edge contract rather than fabricating an empty requirement set. This includes malformed consumer `schema` dicts; they must not silently collapse to `frozenset()`.
- sink helper `ValueError` => emit blocking `ValidationEntry` on that output and skip its edge contract
- build `node_by_id = {node.id: node for node in nodes}` once near the top of the function and use it for all producer lookups
- build the producer registry through an explicit `_register_producer(...)` helper; plain `producer_map[name] = ...` is forbidden because it can silently overwrite an earlier producer on invalid drafts
- extract the duplicated gate/coalesce walk-back from the node loop and sink loop into `_walk_to_real_producer(producer_id, *, producer_map, node_by_id, warnings)` so both paths share the same route-gate termination and coalesce warning/skip behavior
- build `EdgeContract` rows only when both sides parsed successfully
- `_check_schema_contracts()` may accumulate into local mutable lists, but its return type should be immutable tuples to match the surrounding frozen-dataclass design. Do not leak `tuple[list, list, list]` out of the helper boundary.

Add the internal named type at module scope in `state.py` (near the other validation-support types), not inside `_check_schema_contracts()`:

```python
class _ProducerEntry(NamedTuple):
    producer_id: str
    plugin_name: str | None
    options: Mapping[str, Any]
```

Key excerpt:

```python
def _check_schema_contracts(
    source: SourceSpec | None,
    nodes: tuple[NodeSpec, ...],
    outputs: tuple[OutputSpec, ...],
) -> tuple[
    tuple[ValidationEntry, ...],
    tuple[ValidationEntry, ...],
    tuple[EdgeContract, ...],
]:
    errors: list[ValidationEntry] = []
    contract_warnings: list[ValidationEntry] = []
    edge_contracts: list[EdgeContract] = []
    parse_failed_producers: set[str] = set()
    node_by_id = {node.id: node for node in nodes}

    producer_map: dict[str, _ProducerEntry] = {}
    producer_desc: dict[str, str] = {}

    def _register_producer(
        connection_name: str,
        producer_id: str,
        plugin_name: str | None,
        options: Mapping[str, Any],
        description: str,
    ) -> None:
        if connection_name in producer_map:
            errors.append(
                _err(
                    f"connection:{connection_name}",
                    f"Duplicate producer for connection '{connection_name}': "
                    f"{producer_desc[connection_name]} and {description}.",
                    "high",
                )
            )
            return
        producer_map[connection_name] = _ProducerEntry(
            producer_id=producer_id,
            plugin_name=plugin_name,
            options=options,
        )
        producer_desc[connection_name] = description

    # Validate namespace integrity before any contract walk. Invalid drafts must
    # fail closed instead of emitting misleading edge telemetry.
    consumer_claims: list[tuple[str, str, str]] = []
    ...
    if namespace_errors_detected:
        return tuple(errors), tuple(contract_warnings), ()

    def _walk_to_real_producer(
        producer_id: str,
        *,
        producer_map: Mapping[str, _ProducerEntry],
        node_by_id: Mapping[str, NodeSpec],
        warnings: list[ValidationEntry],
    ) -> _ProducerEntry | None:
        ...

    ...

    for node in nodes:
        try:
            consumer_required = get_raw_node_required_fields(
                node.options,
                owner=f"node:{node.id}",
                node_type=node.node_type,
            )
        except ValueError as exc:
            errors.append(_err(f"node:{node.id}", f"Invalid contract config: {exc}", "high"))
            continue

        actual_producer = _walk_to_real_producer(
            node.input,
            producer_map=producer_map,
            node_by_id=node_by_id,
            warnings=contract_warnings,
        )
        if actual_producer is None:
            continue

        ...

        try:
            producer_guaranteed = get_raw_producer_guaranteed_fields(
                actual_producer.plugin_name,
                actual_producer.options,
                owner=(
                    "source"
                    if actual_producer.producer_id == "source"
                    else f"node:{actual_producer.producer_id}"
                ),
            )
        except ValueError as exc:
            errors.append(
                _err(
                    (
                        "source"
                        if actual_producer.producer_id == "source"
                        else f"node:{actual_producer.producer_id}"
                    ),
                    f"Invalid contract config: {exc}",
                    "high",
                )
            )
            parse_failed_producers.add(actual_producer.producer_id)
            continue

        ...

    for output in outputs:
        try:
            sink_required = get_raw_sink_required_fields(
                output.options,
                owner=f"output:{output.name}",
            )
        except ValueError as exc:
            errors.append(_err(f"output:{output.name}", f"Invalid contract config: {exc}", "high"))
            continue

        actual_producer = _walk_to_real_producer(
            output.name,
            producer_map=producer_map,
            node_by_id=node_by_id,
            warnings=contract_warnings,
        )
        if actual_producer is None:
            continue

    return tuple(errors), tuple(contract_warnings), tuple(edge_contracts)
```

- [ ] **Step 6: Call `_check_schema_contracts()` from `validate()`**

In `CompositionState.validate()`, before the `return ValidationSummary(...)` at the end (around line 770), add the contract check:

```python
        # 9. Schema contract validation
        contract_errors, contract_warnings, edge_contracts = _check_schema_contracts(self.source, self.nodes, self.outputs)
        errors.extend(contract_errors)
        warnings.extend(contract_warnings)

        return ValidationSummary(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            suggestions=tuple(suggestions),
            edge_contracts=tuple(edge_contracts),
        )
```

Remove the old `return ValidationSummary(...)` that was there before.

- [ ] **Step 7: Run contract tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestSchemaContractValidation -v`
Expected: All contract tests in this class PASS

- [ ] **Step 8: Run full test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests PASS. The `test_validate_clean_pipeline_no_warnings` test should still pass because it has no `required_input_fields` on any node, so the contract check is skipped.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/contracts/schema.py src/elspeth/core/dag/graph.py src/elspeth/web/composer/state.py tests/unit/web/composer/test_state.py
git commit -m "feat(composer): add schema contract validation pass (pass 9)"
```

---

### Task 3: Add Topology Test Cases (11-14)

**Files:**
- Test: `tests/unit/web/composer/test_state.py`

Review note (2026-04-14): tighten this batch so `test_multi_sink_gate_routing` uses sinks with required fields and asserts the expected `edge_contracts` rows, and add an explicit aggregation wrapper-shaped `options.required_input_fields` regression here instead of leaving that branch covered only by later Task 8 parity work.

- [ ] **Step 1: Write topology tests**

Append to `TestSchemaContractValidation`:

```python
    # --- Topology cases ---

    def test_gate_inherits_source_guarantees(self) -> None:
        """Test 11: Gate route targets inherit source's guaranteed fields."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="gate_in",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_gate(
            "g1", "gate_in", {"high": "main", "low": "errors"},
        ))
        # Add a transform after the gate on the "high" route
        state = state.with_node(self._make_transform(
            "t1", "main", "out",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output("out"))
        state = state.with_output(self._make_output("errors"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_route_gate_two_routes_inherit_guarantees(self) -> None:
        """Test 12: Route gate with two routes — both paths inherit source guarantees.

        Note: This tests route gates (node.routes), NOT fork gates (node.fork_to).
        Fork gates create parallel DAG branches requiring branch-aware contract
        checking, which is out of scope for this pass.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="gate_in",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_gate(
            "g1", "gate_in", {"a": "path_a", "b": "path_b"},
        ))
        state = state.with_node(self._make_transform(
            "ta", "path_a", "out_a",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_node(self._make_transform(
            "tb", "path_b", "out_b",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))
        result = state.validate()
        assert result.is_valid, result.errors

    def test_fork_gate_contract_check_skips_with_warning(self) -> None:
        """Test 12b: fork gates are pinned as unresolved, not silently checked.

        Fork gates (node.fork_to) create parallel branches. Branch-aware
        contract checking is intentionally out of scope for this pass, so the
        composer must skip these edges with an explicit warning rather than
        pretending the downstream consumers were checked against the source.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="gate_in",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(NodeSpec(
            id="g1",
            node_type="gate",
            plugin=None,
            input="gate_in",
            on_success=None,
            on_error=None,
            options={},
            condition="True",
            routes=None,
            fork_to=("path_a", "path_b"),
            branches=None,
            policy=None,
            merge=None,
        ))
        state = state.with_node(self._make_transform(
            "ta", "path_a", "out_a",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_node(self._make_transform(
            "tb", "path_b", "out_b",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))

        result = state.validate()

        # Out of scope for this pass: unresolved warning, no fake edge rows.
        assert result.is_valid, result.errors
        assert any("fork" in w.message.lower() and "contract" in w.message.lower() for w in result.warnings)
        assert not any(ec.to_id in {"ta", "tb"} for ec in result.edge_contracts)

    def test_multi_hop_transform_no_schema_breaks_chain(self) -> None:
        """Test 13: Transform A has no schema — transform B's requirements fail."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="source_to_ta",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        # Connection-name reminder: on_success/input values are routing names,
        # not node IDs. Use a distinct name here to keep the test aligned with
        # the declarative wiring model.
        # Transform A: no schema, no requirements — passes through
        state = state.with_node(self._make_transform("ta", "source_to_ta", "ta_out"))
        # Transform B: requires 'text' but ta guarantees nothing
        state = state.with_node(self._make_transform(
            "tb", "ta_out", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "ta"))
        state = state.with_edge(self._make_edge("e2", "ta", "tb"))
        result = state.validate()
        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)

    def test_transform_then_gate_walk_back_terminates(self) -> None:
        """Test 14b: source → transform A (no schema) → gate → transform B (requires text).

        The gate walk-back should stop at transform A (not a gate), and
        transform A has no schema so it guarantees nothing. Transform B's
        requirement should fail.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="ta_in",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        # Transform A: no schema, no requirements
        state = state.with_node(self._make_transform("ta", "ta_in", "gate_in"))
        # Gate after transform A
        state = state.with_node(self._make_gate(
            "g1", "gate_in", {"high": "tb_in", "low": "sink"},
        ))
        # Transform B: requires 'text', downstream of gate
        state = state.with_node(self._make_transform(
            "tb", "tb_in", "out",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output("out"))
        state = state.with_output(self._make_output("sink"))
        state = state.with_edge(self._make_edge("e1", "source", "ta"))
        state = state.with_edge(self._make_edge("e2", "ta", "g1"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))
        result = state.validate()
        # Walk-back: tb's producer is g1 (gate) → walk back to g1's producer → ta (not gate, stop)
        # ta has no schema → guarantees nothing → contract violation
        assert not result.is_valid
        assert any("text" in e.message for e in result.errors)

    def test_multi_sink_gate_routing(self) -> None:
        """Test 14c: Gate routes to multiple sinks — contract check per route."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="gate_in",
            options={"schema": {"mode": "observed"}},
        ))
        state = state.with_node(self._make_gate(
            "g1", "gate_in", {"high": "sink_a", "low": "sink_b"},
        ))
        state = state.with_output(self._make_output("sink_a"))
        state = state.with_output(self._make_output("sink_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "sink_a"))
        state = state.with_edge(self._make_edge("e3", "g1", "sink_b"))
        result = state.validate()
        # No consumer nodes with required_input_fields — should be valid
        assert result.is_valid, result.errors

    def test_mixed_consumer_requirements_from_same_producer(self) -> None:
        """Test 14d: One producer feeds two consumers — one satisfied, one violated."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="gate_in",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_gate(
            "g1", "gate_in", {"a": "path_a", "b": "path_b"},
        ))
        # Consumer A: requires "text" — satisfied by source
        state = state.with_node(self._make_transform(
            "ta", "path_a", "out_a",
            options={"required_input_fields": ["text"]},
        ))
        # Consumer B: requires "score" — NOT guaranteed by source
        state = state.with_node(self._make_transform(
            "tb", "path_b", "out_b",
            options={"required_input_fields": ["score"]},
        ))
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))
        result = state.validate()
        # Pipeline should fail — consumer B's requirement is unsatisfied
        assert not result.is_valid
        assert any("score" in e.message for e in result.errors)
        # Consumer A's contract should be satisfied
        ta_contract = next(ec for ec in result.edge_contracts if ec.to_id == "ta")
        assert ta_contract.satisfied is True
        # Consumer B's contract should be violated
        tb_contract = next(ec for ec in result.edge_contracts if ec.to_id == "tb")
        assert tb_contract.satisfied is False
        assert "score" in tb_contract.missing_fields

    def test_aggregation_consumer_required_input_fields_fail(self) -> None:
        """Test 14e: Aggregation consumers honor required_input_fields contracts."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="agg1",
            options={"schema": {"mode": "fixed", "fields": ["line: str"]}},
        ))
        state = state.with_node(NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_stats",
            input="agg1",
            on_success="main",
            on_error=None,
            options={
                "value_field": "value",
                "required_input_fields": ["value"],
                "schema": {"mode": "observed"},
            },
            condition=None,
            routes=None,
            fork_to=None,
            branches=None,
            policy=None,
            merge=None,
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "agg1"))
        result = state.validate()
        assert not result.is_valid
        assert any("value" in e.message for e in result.errors)

    def test_coalesce_producer_emits_skip_warning(self) -> None:
        """Test 14f: Coalesce producer emits a warning and skips fake contract checks."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="branch_a",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_coalesce(
            "c1",
            "branch_a",
            "after_merge",
        ))
        state = state.with_node(self._make_transform(
            "t1", "after_merge", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "c1"))
        state = state.with_edge(self._make_edge("e2", "c1", "t1"))
        result = state.validate()
        assert result.is_valid, result.errors
        assert any(
            "coalesce node" in w.message.lower() and "runtime validator will check" in w.message.lower()
            for w in result.warnings
        )
        assert not any(ec.to_id == "t1" for ec in result.edge_contracts)

    # --- Guard tests ---

    def test_node_id_source_is_reserved(self) -> None:
        """Test 14g: A node with id='source' triggers an error and aborts the pass.

        The id 'source' is a reserved sentinel used in the producer map and
        walk-back termination. If a node used this id, the walk-back would
        terminate prematurely (thinking it reached the source) and the
        producer map would silently conflict with the real source entry.
        The guard detects this and returns early with an error.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        # Node with reserved id="source" — should be caught by the guard
        state = state.with_node(self._make_transform(
            "source", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "source"))
        result = state.validate()
        assert not result.is_valid
        assert any("reserved" in e.message.lower() for e in result.errors)

    def test_bare_string_required_input_fields_emits_error(self) -> None:
        """Test 14h: required_input_fields as a bare string emits a validation error.

        If required_input_fields is 'text' instead of ['text'], frozenset('text')
        would produce {'t', 'e', 'x'} — a silent, hard-to-debug bug. The guard
        in the shared raw-config consumer helper detects bare strings and emits a ValidationEntry
        instead of silently corrupting the field set.
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": "text"},  # bare string, not list
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()
        assert not result.is_valid
        assert any("bare string" in e.message.lower() for e in result.errors)

    def test_duplicate_producer_connection_emits_error_and_skips_contracts(self) -> None:
        """Test 14i: Duplicate producers fail closed instead of overwriting producer_map."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="gate_in",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_gate(
            "g1", "gate_in", {"a": "dup", "b": "path_b"},
        ))
        state = state.with_node(self._make_transform(
            "ta", "dup", "out_a",
            options={"required_input_fields": ["text"]},
        ))
        # Second producer for the same connection name "dup"
        state = state.with_node(self._make_transform(
            "tb", "path_b", "dup",
        ))
        state = state.with_output(self._make_output("out_a"))
        state = state.with_edge(self._make_edge("e1", "source", "g1"))
        state = state.with_edge(self._make_edge("e2", "g1", "ta"))
        state = state.with_edge(self._make_edge("e3", "g1", "tb"))
        result = state.validate()
        assert not result.is_valid
        assert any("duplicate producer" in e.message.lower() for e in result.errors)
        assert result.edge_contracts == ()

    def test_duplicate_consumer_connection_emits_error_and_skips_contracts(self) -> None:
        """Test 14j: Duplicate consumers fail closed instead of fabricating two edge checks."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="shared",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "ta", "shared", "out_a",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_node(self._make_transform(
            "tb", "shared", "out_b",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output("out_a"))
        state = state.with_output(self._make_output("out_b"))
        state = state.with_edge(self._make_edge("e1", "source", "ta"))
        state = state.with_edge(self._make_edge("e2", "source", "tb"))
        result = state.validate()
        assert not result.is_valid
        assert any("duplicate consumer" in e.message.lower() for e in result.errors)
        assert result.edge_contracts == ()

    def test_connection_name_overlaps_sink_name_emits_error_and_skips_contracts(self) -> None:
        """Test 14k: Connection/sink namespace overlap aborts contract telemetry."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            on_success="t1",
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
        ))
        # Invalid: "main" is used as both a sink name and a connection name
        state = state.with_node(self._make_transform(
            "t2", "main", "out",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output("main"))
        state = state.with_output(self._make_output("out"))
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        state = state.with_edge(self._make_edge("e2", "t1", "t2"))
        result = state.validate()
        assert not result.is_valid
        assert any("disjoint" in e.message.lower() or "overlap" in e.message.lower() for e in result.errors)
        assert result.edge_contracts == ()
```

- [ ] **Step 2: Run topology tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestSchemaContractValidation -v -k "gate or route_gate or fork_gate or multi_hop or multi_sink or mixed_consumer or aggregation or coalesce or reserved or bare_string or duplicate or overlap or namespace"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_state.py
git commit -m "test(composer): add topology test cases for schema contract validation"
```

---

### Task 4: Add EdgeContract Data Integrity Tests (15-16)

**Files:**
- Test: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write edge contract data tests**

Append to `TestSchemaContractValidation`:

```python
    # --- Data integrity ---

    def test_edge_contracts_populated_correctly(self) -> None:
        """Test 15: ValidationSummary.edge_contracts has correct entries."""
        state = self._empty_state()
        state = state.with_source(self._make_source(
            options={"schema": {"mode": "fixed", "fields": ["text: str"]}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={"required_input_fields": ["text"]},
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))
        result = state.validate()

        assert len(result.edge_contracts) >= 1
        contract = next(ec for ec in result.edge_contracts if ec.to_id == "t1")
        assert contract.from_id == "source"
        assert "text" in contract.producer_guarantees
        assert "text" in contract.consumer_requires
        assert contract.satisfied is True

    def test_edge_contract_to_dict_serialization(self) -> None:
        """Test 16: EdgeContract.to_dict() uses 'from'/'to' keys."""
        from elspeth.web.composer.state import EdgeContract

        ec = EdgeContract(
            from_id="source",
            to_id="t1",
            producer_guarantees=("text",),
            consumer_requires=("text",),
            missing_fields=(),
            satisfied=True,
        )
        d = ec.to_dict()
        # 'from' is a Python keyword — verify it's used as JSON key
        assert "from" in d
        assert "to" in d
        assert "from_id" not in d
        assert "to_id" not in d
```

- [ ] **Step 2: Run data integrity tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestSchemaContractValidation -v -k "edge_contract"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_state.py
git commit -m "test(composer): add edge contract data integrity tests"
```

---

## Sub-Plan B: Agent-Facing Surfaces

### Task 5: Expose `edge_contracts` in Preview Response

**Files:**
- Modify: `src/elspeth/web/composer/tools.py:2599-2624`
- Test: `tests/unit/web/composer/test_tools.py`

- [ ] **Step 1: Write failing test for preview response**

Add to the preview test class in `tests/unit/web/composer/test_tools.py` (alongside `test_preview_valid_pipeline`):

```python
    def test_preview_pipeline_includes_edge_contracts(self) -> None:
        """Test 17: preview_pipeline response includes edge_contracts with field data."""
        state = _empty_state()
        catalog = _mock_catalog()
        # Build a pipeline with source (fixed schema) + transform (required_input_fields)
        r1 = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv", "schema": {"mode": "fixed", "fields": ["text: str"]}},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        r2 = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "value_transform",
                "input": "t1",
                "on_success": "main",
                "options": {
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text']"}],
                    "schema": {"mode": "observed"},
                },
            },
            r1.updated_state,
            catalog,
        )
        r3 = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                "on_write_failure": "discard",
            },
            r2.updated_state,
            catalog,
        )
        r4 = execute_tool("preview_pipeline", {}, r3.updated_state, catalog)

        assert r4.success is True
        assert "edge_contracts" in r4.data
        contracts = r4.data["edge_contracts"]
        assert len(contracts) >= 1
        # Find the source→t1 contract
        source_to_t1 = next(c for c in contracts if c["to"] == "t1")
        assert source_to_t1["from"] == "source"
        assert "text" in source_to_t1["producer_guarantees"]
        assert "text" in source_to_t1["consumer_requires"]
        assert source_to_t1["satisfied"] is True
        # Verify consistency: is_valid should be True since contract is satisfied
        assert r4.data["is_valid"] is True
```

This pass intentionally exposes the raw contract evidence, not redundant convenience booleans. Do not add `contracts_all_satisfied` to the preview response here; clients can derive it from `edge_contracts`. Likewise, do not add `has_unverified_edges` to `ValidationSummary` in this pass; skipped checks remain represented by warnings plus the absence of a matching `EdgeContract`.

- [ ] **Step 2: Modify `_execute_preview_pipeline()` to include edge_contracts**

In `src/elspeth/web/composer/tools.py`, in the `_execute_preview_pipeline()` function, add `edge_contracts` to the summary dict (after line 2608):

```python
    summary: dict[str, Any] = {
        "is_valid": validation.is_valid,
        "errors": [e.to_dict() for e in validation.errors],
        "warnings": [e.to_dict() for e in validation.warnings],
        "suggestions": [e.to_dict() for e in validation.suggestions],
        "edge_contracts": [ec.to_dict() for ec in validation.edge_contracts],
        "source": None,
        "node_count": len(state.nodes),
        "output_count": len(state.outputs),
        "nodes": [{"id": n.id, "node_type": n.node_type, "plugin": n.plugin} for n in state.nodes],
        "outputs": [{"name": o.name, "plugin": o.plugin} for o in state.outputs],
    }
```

- [ ] **Step 3: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v -k "edge_contract"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "feat(composer): expose edge_contracts in preview_pipeline response"
```

---

### Task 5B: Add Mechanical Backstop to YAML Export Surfaces

**Files:**
- Modify: `src/elspeth/composer_mcp/server.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Test: `tests/unit/composer_mcp/test_server.py`
- Test: `tests/unit/web/sessions/test_routes.py`

`generate_yaml()` itself is a pure serializer and must stay that way. The backstop belongs at the LLM-facing export surfaces that hand YAML back to agents and users. Those callers must re-run `state.validate()` and refuse to emit YAML when `validation.is_valid` is `false`. This backstop depends on unsatisfied contracts remaining error-level; do not add contract-warning paths that leave `is_valid=True` without revisiting this task.

- [ ] **Step 1: Write failing tests for invalid-state YAML export**

In `tests/unit/composer_mcp/test_server.py`, extend the existing state import to include `NodeSpec`, `OutputSpec`, and `SourceSpec`, then add to `TestDispatchTool`:

```python
    def test_generate_yaml_rejects_invalid_contract_state(self, scratch_dir: Path) -> None:
        invalid_state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="t1",
                    node_type="transform",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error=None,
                    options={
                        "required_input_fields": ["text"],
                        "operations": [{"target": "out", "expression": "row['text']"}],
                        "schema": {"mode": "observed"},
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = _dispatch_tool(
            "generate_yaml",
            {},
            invalid_state,
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()
        assert result["validation"]["is_valid"] is False
```

Add to `tests/unit/web/sessions/test_routes.py`, in `TestYamlEndpoint`:

```python
    @pytest.mark.asyncio
    async def test_yaml_returns_409_when_current_state_is_invalid(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "t1",
                    "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                    "on_validation_failure": "quarantine",
                },
                nodes=[
                    {
                        "id": "t1",
                        "node_type": "transform",
                        "plugin": "value_transform",
                        "input": "t1",
                        "on_success": "main",
                        "on_error": None,
                        "options": {
                            "required_input_fields": ["text"],
                            "operations": [{"target": "out", "expression": "row['text']"}],
                            "schema": {"mode": "observed"},
                        },
                        "condition": None,
                        "routes": None,
                        "fork_to": None,
                        "branches": None,
                        "policy": None,
                        "merge": None,
                    },
                ],
                outputs=[
                    {
                        "name": "main",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Invalid Contract Pipeline", "description": ""},
                is_valid=False,
            ),
        )

        resp = client.get(f"/api/sessions/{session.id}/state/yaml")
        assert resp.status_code == 409
        assert "invalid" in resp.json()["detail"].lower()
```

Expected: both tests FAIL on current code because the MCP session tool and HTTP route still serialize invalid state.

- [ ] **Step 1b: Add positive backstop tests for valid pipelines with empty `edge_contracts`**

These tests pin the success case where `validation.is_valid` is `true` but no explicit field contracts were declared, so `edge_contracts == ()` / `[]`. The export guard must key off `validation.is_valid`, not `validation.edge_contracts`.

In `tests/unit/composer_mcp/test_server.py`, add to `TestDispatchTool`:

```python
    def test_generate_yaml_allows_valid_state_with_no_edge_contracts(self, scratch_dir: Path) -> None:
        valid_state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/in.csv", "schema": {"mode": "observed"}},
                on_validation_failure="quarantine",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = _dispatch_tool(
            "generate_yaml",
            {},
            valid_state,
            _mock_catalog(),
            scratch_dir,
        )

        assert result["success"] is True
        assert isinstance(result["data"], str)
```

In `tests/unit/web/sessions/test_routes.py`, strengthen the existing happy-path YAML test so it uses a plugin-valid state with no declared `required_input_fields`:

```python
    @pytest.mark.asyncio
    async def test_yaml_returns_yaml_when_state_exists(self, tmp_path) -> None:
        """Returns generated YAML for a valid state even when edge_contracts is empty."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "on_success": "out",
                    "options": {"path": "/data.csv", "schema": {"mode": "observed"}},
                    "on_validation_failure": "quarantine",
                },
                outputs=[
                    {
                        "name": "out",
                        "plugin": "csv",
                        "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                        "on_write_failure": "discard",
                    }
                ],
                metadata_={"name": "Test Pipeline", "description": ""},
                is_valid=True,
            ),
        )

        resp = client.get(f"/api/sessions/{session.id}/state/yaml")
        assert resp.status_code == 200
        body = resp.json()
        assert "yaml" in body
        assert "csv" in body["yaml"]
```

Expected: on the fixed implementation, both tests PASS even though the validated state has no edge-contract rows.

- [ ] **Step 2: Guard the MCP `generate_yaml` tool**

In `src/elspeth/composer_mcp/server.py`, change the `generate_yaml` session tool branch to:

1. Compute `validation = state.validate()` before any serialization.
2. If `validation.is_valid` is `False`, return `success=False` with an error such as `"Current composition state is invalid. Fix validation errors before calling generate_yaml."`
3. Include structured validation telemetry in the failure payload:
   - `is_valid`
   - `errors`
   - `warnings`
   - `suggestions`
   - `edge_contracts`
4. Only call `generate_yaml(state)` on the valid path.

This is the LLM-facing backstop. It must key off `validation.is_valid`, not off a second derived flag such as `contracts_all_satisfied`.

- [ ] **Step 3: Guard the HTTP YAML route**

In `src/elspeth/web/sessions/routes.py`, in `get_state_yaml(...)`:

1. Reconstruct the `CompositionState` as today.
2. Compute `validation = state.validate()`.
3. If `validation.is_valid` is `False`, raise `HTTPException(status_code=409, detail="Current composition state is invalid. Fix validation errors before exporting YAML.")`
4. Only then call `generate_yaml(state)`.

Keep the serializer pure. Do **not** move validation into `elspeth.web.composer.yaml_generator.generate_yaml()`, because execution-time validation and run creation depend on that function staying a deterministic serialization primitive.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/composer_mcp/test_server.py tests/unit/web/sessions/test_routes.py -v -k "generate_yaml or state_yaml"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/composer_mcp/server.py src/elspeth/web/sessions/routes.py tests/unit/composer_mcp/test_server.py tests/unit/web/sessions/test_routes.py
git commit -m "feat(composer): block yaml export for invalid composition state"
```

---

### Task 5C: Add Schema Contract Pattern to `explain_validation_error`

**Files:**
- Modify: `src/elspeth/web/composer/tools.py`
- Test: `tests/unit/web/composer/test_tools.py`

The new contract pass will emit a new high-signal validation error family. The LLM tooling needs a matching `explain_validation_error` catalogue entry so agents get a concrete diagnosis and fix path instead of falling back to the generic "not in the known pattern catalogue" branch.

- [ ] **Step 1: Write a failing explain-validation test**

Add to `tests/unit/web/composer/test_tools.py`, in `TestExplainValidationError`:

```python
    def test_explains_schema_contract_violation(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "explain_validation_error",
            {
                "error_text": (
                    "Schema contract violation: 'source' -> 'add_world'. "
                    "Consumer requires ['text']; producer guarantees []."
                )
            },
            state,
            catalog,
        )
        assert result.success is True
        assert "upstream" in result.data["explanation"].lower()
        assert "preview_pipeline" in result.data["suggested_fix"]
        assert "patch_source_options" in result.data["suggested_fix"] or "patch_node_options" in result.data["suggested_fix"]
```

Expected: FAIL on current code because the contract-violation string falls through to the generic explanation path.

- [ ] **Step 2: Add a contract-violation pattern to `_VALIDATION_ERROR_PATTERNS`**

In `src/elspeth/web/composer/tools.py`, add a new tuple to `_VALIDATION_ERROR_PATTERNS` matching schema contract failures, for example:

```python
    (
        r"Schema contract violation:",
        "A downstream node requires fields that its upstream producer does not guarantee.",
        "Call preview_pipeline to inspect edge_contracts, then update the producer schema with patch_source_options or patch_node_options and re-preview until the edge shows satisfied=true.",
    ),
```

Keep this terse and mechanical. It only needs to recognize the contract-family error and route the agent to `preview_pipeline` plus the schema-patch tools.

- [ ] **Step 3: Run the explain-validation tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v -k "explain_validation_error"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "feat(composer): teach explain_validation_error about schema contract violations"
```

---

## Sub-Plan C: Runtime Parity and Release Gate

### Task 6: TDD the Observed-Text Runtime Backing in `text_source.py`

**Files:**
- Modify: `src/elspeth/plugins/sources/text_source.py`
- Create: `tests/unit/web/composer/test_schema_contract_enforcement.py`

**Prerequisite note:** This task is the runtime backing for the observed-text special case. The critical requirement is builder visibility: `build_execution_graph()` currently reads source schema config from `source.config["schema"]`, not from a private normalized field on the plugin instance. If tasks are committed separately, do not enable the observed-text-positive tests from Task 2, Task 8, or Task 9 until this task lands.

- [ ] **Step 1: Write the failing enforcement tests first**

Add `tests/unit/web/composer/test_schema_contract_enforcement.py` with:

```python
"""Enforcement test tying the shared observed-text contract rule to the plugin.

The shared contract helpers in elspeth.contracts.schema infer that an observed
text source with column='X' guarantees field 'X'. TextSource.__init__ must make
that guarantee visible in both its normalized schema config and the raw config
dict consumed by DAG construction. This test verifies both sides agree: the
plugin produces the key and declares it as guaranteed for observed schemas only.
If either side changes, this test fails.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from elspeth.contracts.contexts import SourceContext
from elspeth.plugins.sources.text_source import TextSource


class TestTextSourceHeuristicEnforcement:
    def test_text_source_produces_configured_column_key(self) -> None:
        """Test 20a: Text source output key matches the shared contract rule.

        The shared raw-config helper infers that a text source with
        column='text' guarantees field 'text' in its output. This test verifies
        that TextSource actually produces rows with that key.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "observed"},
            }
            source = TextSource(config)
            ctx = MagicMock(spec=SourceContext)
            ctx.record_validation_error = MagicMock()

            rows = list(source.load(ctx))
            assert len(rows) >= 1
            # The critical assertion: the row contains the configured column key
            first_row = rows[0].row
            assert "text" in first_row, (
                f"TextSource with column='text' must produce rows with key 'text'. "
                f"Got keys: {list(first_row.keys())}. "
                f"The shared contract helper in elspeth.contracts.schema depends on this."
            )
            assert first_row["text"] == "hello"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_text_source_auto_declares_guaranteed_fields(self) -> None:
        """Test 20b: TextSource auto-declares {column} and updates raw config for observed schemas only.

        The runtime reads SchemaConfig from the plugin. TextSource.__init__
        must auto-populate guaranteed_fields for the observed-text special case
        and keep source.config["schema"] in sync, because DAG construction reads
        the raw config dict rather than the private _schema_config field.
        Without this, the shared helper would say "valid" while runtime rejects.

        White-box note: this test intentionally inspects both
        source._schema_config and source.config["schema"]. There is no public
        accessor for the plugin's normalized constructor-time SchemaConfig, and
        the behavior under test lives in __init__ before row loading begins.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "observed"},
            }
            source = TextSource(config)

            # Intentional white-box assertions against both the normalized
            # internal config and the builder-visible raw config dict.
            # This test is pinning constructor-time contract state, not public row output.
            assert source._schema_config.guaranteed_fields == ("text",), (
                "TextSource must set the exact observed-text guarantee on its "
                "normalized SchemaConfig using mechanical dataclass state, not "
                "only via a derived effective-guarantees view."
            )
            raw_schema = source.config["schema"]
            assert raw_schema["guaranteed_fields"] == ["text"], (
                "TextSource must also write the observed-text guarantee back to "
                "source.config['schema'] so DAG construction sees it."
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_text_source_preserves_explicit_guaranteed_fields(self) -> None:
        """Test 20c: TextSource does not override explicit guaranteed_fields.

        White-box by design for the same reason as test 20b: the contract state
        is normalized in __init__ and has no public accessor.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "observed", "guaranteed_fields": ["custom_field"]},
            }
            source = TextSource(config)

            # Explicit guaranteed_fields should be preserved, not overridden
            guaranteed = source._schema_config.get_effective_guaranteed_fields()
            assert "custom_field" in guaranteed, (
                "Explicit guaranteed_fields must be preserved"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_text_source_observed_only_auto_declare_does_not_touch_fixed_schema(self) -> None:
        """Test 20d: Fixed schemas are not rewritten into explicit guarantees.

        `declares_guaranteed_fields` tracks only whether the schema made an
        explicit guaranteed_fields declaration. It does NOT mean "has no
        guarantees at all" for fixed/flexible schemas — required typed fields
        still contribute through get_effective_guaranteed_fields().

        White-box by design for the same reason as test 20b.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "fixed", "fields": ["text: str"]},
            }
            source = TextSource(config)

            assert source._schema_config.declares_guaranteed_fields is False, (
                "declares_guaranteed_fields tracks explicit guaranteed_fields "
                "only. Fixed typed fields remain implicit guarantees and must "
                "not be rewritten into an explicit observed-text declaration."
            )
            assert "text" in source._schema_config.get_effective_guaranteed_fields()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
```

- [ ] **Step 2: Run the enforcement tests to verify they fail on current code**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_schema_contract_enforcement.py -v`
Expected: FAIL on current code. `test_text_source_produces_configured_column_key` may already be green, but the suite as a whole should be red because the observed-text runtime backing does not yet materialize the guarantee into the constructor state that the shared contract relies on, especially the raw config path consumed by DAG construction.

- [ ] **Step 3: Add the enforcement comment**

In `src/elspeth/plugins/sources/text_source.py`, at line 109 (the `yield from self._validate_and_yield({self._column: value}, ctx)` line), add a comment:

```python
                    # Shared composer/runtime contract helper depends on this:
                    # elspeth.contracts.schema.get_raw_producer_guaranteed_fields()
                    # infers {self._column} for observed text sources only.
                    # If you change which key the row uses, update that helper
                    # and its agreement tests.
                    yield from self._validate_and_yield(
                        {self._column: value},
                        ctx,
                    )
```

- [ ] **Step 4: Auto-declare `guaranteed_fields` in `__init__` and sync a normalized config copy**

In `TextSource.__init__()`, restructure the current config / schema setup so the observed-text guarantee is normalized **before** `BaseSource` stores the config dict. Auto-declare the column as a guaranteed field only when the schema is `observed` and does not already declare `guaranteed_fields`. This makes the narrow shared observed-text rule mechanical in runtime code without changing the semantics of `fixed`/`flexible` schemas.

Critical detail: updating `self._schema_config` alone is **not sufficient**. The DAG builder currently materializes source schema contracts from `source.config["schema"]`, so the normalized schema must also be present in the raw plugin config dict that the builder reads. However, `BaseSource.__init__()` stores the incoming `config` dict by reference (`self.config = config`), so mutating `self.config["schema"]` after `super().__init__(config)` would leak a caller-visible aliasing side effect. Do **not** do that. Instead:

- parse and normalize the schema first,
- build a copied `normalized_config` dict,
- write the canonical `"schema"` entry into that copy,
- then call `super().__init__(normalized_config)` once.

Because `TextSourceConfig` inherits from `SourceDataConfig`, `cfg.schema_config` is guaranteed to be populated after `from_dict()`. Do **not** add a defensive `if schema_config is not None` guard here; that would misrepresent a constructor invariant as a runtime branch. Do **not** round-trip through `SchemaConfig.to_dict()` / `SchemaConfig.from_dict()` just to mutate a validated `SchemaConfig` you already hold. This is Tier 1 data in process memory. Update the frozen dataclass mechanically with `dataclasses.replace(...)`, then serialize only once when constructing the copied config passed to the base class.

```python
        # Add at module scope:
        # from dataclasses import replace
        cfg = TextSourceConfig.from_dict(config, plugin_name=self.name)
        schema_config = cfg.schema_config
        normalized_config = dict(config)

        # Auto-declare {column} as a guaranteed output field only for the
        # shared observed-text contract case: observed schema, no explicit
        # guaranteed_fields, non-empty column. TextSource always produces
        # {column: value} for every row, so this is a provable invariant.
        # Keep this narrow: fixed/flexible schemas already express their
        # guarantees through normal SchemaConfig semantics and must not be
        # rewritten into an explicit guaranteed_fields declaration.
        if (
            schema_config.mode == "observed"
            and not schema_config.declares_guaranteed_fields
            and cfg.column
        ):
            # Update the typed, already-validated SchemaConfig directly.
            # Avoid a dict round-trip and avoid caller-visible mutation of
            # self.config after BaseSource stores the config reference.
            schema_config = replace(
                schema_config,
                guaranteed_fields=(cfg.column,),
            )

        # DAG builder currently reads source.config["schema"], not the
        # plugin's private _schema_config. Normalize the copied config before
        # BaseSource stores it so runtime validation sees the same observed-text
        # contract the composer reports to the LLM, without mutating the
        # caller-supplied config dict.
        normalized_config["schema"] = schema_config.to_dict()
        super().__init__(normalized_config)

        self._schema_config = schema_config
```

- [ ] **Step 5: Run the enforcement suite to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_schema_contract_enforcement.py -v`
Expected: PASS — TextSource now auto-declares observed-text guarantees and leaves fixed schemas untouched

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/sources/text_source.py tests/unit/web/composer/test_schema_contract_enforcement.py
git commit -m "feat(text_source): add observed-text runtime backing with enforcement tests"
```

---

### Task 7: Reserved Checkpoint

Task 7's former enforcement suite has been intentionally folded into Task 6 so the observed-text runtime backing is driven by failing tests instead of landing first and being tested later. There is no separate implementation work here. Do not proceed to Task 8 until Task 6 Step 5 is green and the Task 6 commit includes both the runtime change and its enforcement tests.

---

### Task 8: Composer/Runtime Agreement Test

**Files:**
- Create: `tests/integration/pipeline/test_composer_runtime_agreement.py`

- [ ] **Step 1: Write the agreement test**

This test builds the same pipeline configuration, validates it through both the composer and the runtime DAG validator, and asserts they agree on the shared-contract cases covered here. It is not a proof that composer and runtime are globally identical. If the composer later adds intentionally stricter preflight checks that runtime does not mirror, document those in separate asymmetry tests rather than forcing them into this agreement suite. Cover the node-consumer failure case, the observed-text acceptance case, the strict-sink parity case that motivated the shared helper extraction, and an aggregation-consumer failure case that proves the runtime's nested `config["options"]["required_input_fields"]` path stays coupled to the composer's flat `node.options` view. Place it under `tests/integration/pipeline/`, not `tests/integration/web/`, because this is pipeline/DAG contract agreement work rather than HTTP-route integration.

```python
"""Composer/runtime agreement test.

Verifies that the composer's schema contract validation and the runtime
DAG validator agree on pass/fail for the same pipeline configuration in the
shared-contract cases covered here. This suite does not claim global
equivalence; intentionally stricter composer-only checks should live in
separate tests with explicit documentation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import (
    AggregationSettings,
    ElspethSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
    TriggerConfig,
)
from elspeth.core.dag.graph import ExecutionGraph, GraphValidationError
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


class TestComposerRuntimeAgreement:
    """Test 19: Composer and runtime validators agree on schema contracts."""

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None, nodes=(), edges=(), outputs=(),
            metadata=PipelineMetadata(), version=1,
        )

    def _build_runtime_graph(
        self,
        source_plugin: str,
        source_options: dict[str, Any],
        sink_options: dict[str, Any],
        transform_options: dict[str, Any] | None = None,
        transform_plugin: str | None = "value_transform",
        aggregation_options: dict[str, Any] | None = None,
        aggregation_plugin: str | None = None,
    ) -> ExecutionGraph:
        """Build a runtime ExecutionGraph using the correct API.

        Uses ElspethSettings -> instantiate_plugins_from_config -> from_plugin_instances,
        matching the pattern in existing integration tests.
        """
        source_on_success = "agg1" if aggregation_plugin is not None else ("t1" if transform_plugin is not None else "main")
        transforms: list[TransformSettings] = []
        aggregations: list[AggregationSettings] = []
        if transform_plugin is not None:
            transforms.append(
                TransformSettings(
                    name="t1",
                    plugin=transform_plugin,
                    input="t1",
                    on_success="main",
                    on_error="discard",
                    options=transform_options or {},
                ),
            )
        if aggregation_plugin is not None:
            aggregations.append(
                AggregationSettings(
                    name="agg1",
                    plugin=aggregation_plugin,
                    input="agg1",
                    on_success="main",
                    on_error="discard",
                    trigger=TriggerConfig(count=1),
                    options=aggregation_options or {},
                ),
            )

        config = ElspethSettings(
            source=SourceSettings(
                plugin=source_plugin,
                on_success=source_on_success,
                options={**source_options, "on_validation_failure": "discard"},
            ),
            transforms=transforms,
            aggregations=aggregations,
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options=sink_options,
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins.source,
            source_settings=plugins.source_settings,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
        )
        return graph

    def test_both_reject_missing_required_field(self) -> None:
        """Both validators reject: observed source, consumer requires 'text'."""
        # --- Composer validation ---
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            state = self._empty_state()
            state = state.with_source(SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": tmp_path,
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            ))
            state = state.with_node(NodeSpec(
                id="t1", node_type="transform", plugin="value_transform",
                input="t1", on_success="main", on_error=None,
                options={
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text'] + ' world'"}],
                    "schema": {"mode": "observed"},
                },
                condition=None, routes=None, fork_to=None,
                branches=None, policy=None, merge=None,
            ))
            state = state.with_output(OutputSpec(
                name="main", plugin="csv",
                options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ))
            state = state.with_edge(EdgeSpec(
                id="e1", from_node="source", to_node="t1",
                edge_type="on_success", label=None,
            ))

            composer_result = state.validate()

            # Composer should reject — the observed-text special case can only
            # guarantee the configured column. Here the source column is
            # 'line', so downstream requirement 'text' is still unsatisfied.
            assert not composer_result.is_valid, (
                "Composer should reject: source column is 'line' but consumer requires 'text'"
            )
            assert any("schema contract violation" in e.message.lower() for e in composer_result.errors)

            # --- Runtime validation ---
            # Build the same pipeline through the runtime path and verify it also rejects.
            try:
                graph = self._build_runtime_graph(
                    source_plugin="text",
                    source_options={"path": tmp_path, "column": "line", "schema": {"mode": "observed"}},
                    transform_options={
                        "required_input_fields": ["text"],
                        "operations": [{"target": "out", "expression": "row['text'] + ' world'"}],
                        "schema": {"mode": "observed"},
                    },
                    sink_options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                )
                graph.validate_edge_compatibility()
                pytest.fail(
                    "Runtime should have rejected: consumer requires 'text' "
                    "but source column is 'line' (observed schema)"
                )
            except GraphValidationError as exc:
                assert "text" in str(exc).lower() or "required" in str(exc).lower(), (
                    f"Runtime rejected but for unexpected reason: {exc}"
                )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_both_accept_satisfied_contract(self) -> None:
        """Both validators accept: fixed source with guaranteed field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            state = self._empty_state()
            state = state.with_source(SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": tmp_path,
                    "column": "text",
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_validation_failure="quarantine",
            ))
            state = state.with_node(NodeSpec(
                id="t1", node_type="transform", plugin="value_transform",
                input="t1", on_success="main", on_error=None,
                options={
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text'] + ' world'"}],
                    "schema": {"mode": "observed"},
                },
                condition=None, routes=None, fork_to=None,
                branches=None, policy=None, merge=None,
            ))
            state = state.with_output(OutputSpec(
                name="main", plugin="csv",
                options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ))
            state = state.with_edge(EdgeSpec(
                id="e1", from_node="source", to_node="t1",
                edge_type="on_success", label=None,
            ))

            composer_result = state.validate()
            assert composer_result.is_valid, composer_result.errors

            # --- Runtime validation ---
            graph = self._build_runtime_graph(
                source_plugin="text",
                source_options={"path": tmp_path, "column": "text", "schema": {"mode": "fixed", "fields": ["text: str"]}},
                transform_options={
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text'] + ' world'"}],
                    "schema": {"mode": "observed"},
                },
                sink_options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
            )
            graph.validate_edge_compatibility()
            # If we reach here, runtime agrees with composer: both accept

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_both_accept_observed_text_source_with_auto_guarantee(self) -> None:
        """Both validators accept: text source with observed schema + column guarantee.

        This is the critical agreement case that validates the shared
        observed-text rule. The shared helper infers {column} as guaranteed
        for observed text sources. TextSource.__init__ writes that guarantee
        into the builder-visible raw schema config as well as its normalized
        internal SchemaConfig, so DAG construction and composer validation get
        the same answer.

        If either the shared helper or the TextSource auto-declaration is
        removed without removing the other, this test fails — catching the
        divergence that would cause false positives or false negatives.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            state = self._empty_state()
            state = state.with_source(SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": tmp_path,
                    "column": "text",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            ))
            state = state.with_node(NodeSpec(
                id="t1", node_type="transform", plugin="value_transform",
                input="t1", on_success="main", on_error=None,
                options={
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text'] + ' world'"}],
                    "schema": {"mode": "observed"},
                },
                condition=None, routes=None, fork_to=None,
                branches=None, policy=None, merge=None,
            ))
            state = state.with_output(OutputSpec(
                name="main", plugin="csv",
                options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ))
            state = state.with_edge(EdgeSpec(
                id="e1", from_node="source", to_node="t1",
                edge_type="on_success", label=None,
            ))

            # Composer should accept — the shared helper infers {column}
            composer_result = state.validate()
            assert composer_result.is_valid, (
                "Composer should accept: observed text rule infers 'text' from column"
            )

            # Runtime should also accept — TextSource makes the observed-text
            # guarantee builder-visible by syncing source.config["schema"].
            graph = self._build_runtime_graph(
                source_plugin="text",
                source_options={"path": tmp_path, "column": "text", "schema": {"mode": "observed"}},
                transform_options={
                    "required_input_fields": ["text"],
                    "operations": [{"target": "out", "expression": "row['text'] + ' world'"}],
                    "schema": {"mode": "observed"},
                },
                sink_options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
            )
            graph.validate_edge_compatibility()
            # If we reach here, both agree: accept

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_both_reject_strict_sink_typed_requirement_without_upstream_guarantee(self) -> None:
        """Both validators reject: strict sink requirement from typed sink schema."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            state = self._empty_state()
            state = state.with_source(SourceSpec(
                plugin="text",
                on_success="main",
                options={
                    "path": tmp_path,
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            ))
            state = state.with_output(OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": "outputs/out.csv",
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_write_failure="discard",
            ))

            composer_result = state.validate()
            assert not composer_result.is_valid, (
                "Composer should reject: sink fixed schema requires 'text' "
                "but upstream only guarantees 'line'"
            )
            assert any(
                ec.to_id == "output:main" and not ec.satisfied
                for ec in composer_result.edge_contracts
            )

            try:
                graph = self._build_runtime_graph(
                    source_plugin="text",
                    source_options={"path": tmp_path, "column": "line", "schema": {"mode": "observed"}},
                    sink_options={"path": "outputs/out.csv", "schema": {"mode": "fixed", "fields": ["text: str"]}},
                    transform_plugin=None,
                )
                graph.validate_edge_compatibility()
                pytest.fail(
                    "Runtime should have rejected: sink typed schema requires 'text' "
                    "but upstream guarantees only 'line'"
                )
            except GraphValidationError as exc:
                assert "sink" in str(exc).lower() or "requires" in str(exc).lower(), (
                    f"Runtime rejected but for unexpected reason: {exc}"
                )

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_both_reject_aggregation_nested_required_input_fields_without_upstream_guarantee(self) -> None:
        """Both validators reject: aggregation contract comes from nested runtime options."""
        state = self._empty_state()
        state = state.with_source(SourceSpec(
            plugin="csv",
            on_success="agg1",
            options={
                "path": "inputs/in.csv",
                "schema": {"mode": "fixed", "fields": ["line: str"]},
            },
            on_validation_failure="quarantine",
        ))
        state = state.with_node(NodeSpec(
            id="agg1",
            node_type="aggregation",
            plugin="batch_stats",
            input="agg1",
            on_success="main",
            on_error=None,
            options={
                "value_field": "value",
                "required_input_fields": ["value"],
                "schema": {"mode": "observed"},
            },
            condition=None, routes=None, fork_to=None,
            branches=None, policy=None, merge=None,
        ))
        state = state.with_output(OutputSpec(
            name="main", plugin="csv",
            options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
            on_write_failure="discard",
        ))
        state = state.with_edge(EdgeSpec(
            id="e1", from_node="source", to_node="agg1",
            edge_type="on_success", label=None,
        ))

        composer_result = state.validate()
        assert not composer_result.is_valid
        assert any("value" in e.message.lower() for e in composer_result.errors)

        try:
            graph = self._build_runtime_graph(
                source_plugin="csv",
                source_options={"path": "inputs/in.csv", "schema": {"mode": "fixed", "fields": ["line: str"]}},
                sink_options={"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                transform_plugin=None,
                aggregation_plugin="batch_stats",
                aggregation_options={
                    "value_field": "value",
                    "required_input_fields": ["value"],
                    "schema": {"mode": "observed"},
                },
            )
            graph.validate_edge_compatibility()
            pytest.fail("Runtime should reject: aggregation requires 'value' but source guarantees only 'line'")
        except GraphValidationError as exc:
            assert "value" in str(exc).lower() or "required" in str(exc).lower(), (
                f"Runtime rejected but for unexpected reason: {exc}"
            )
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/pipeline/test_composer_runtime_agreement.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/pipeline/test_composer_runtime_agreement.py
git commit -m "test(integration): add composer/runtime agreement test for schema contracts"
```

---

### Task 9: Regression Test for the Reported Text-Source Scenario

**Files:**
- Test: `tests/unit/web/composer/test_state.py`

- [ ] **Step 1: Write regression test**

Append to `TestSchemaContractValidation`:

```python
    # --- Regression test ---

    def test_text_heuristic_rescues_original_bug_scenario(self) -> None:
        """Test 18: The reported text-source scenario now passes via the shared rule.

        text source (column=text, observed schema) + value_transform
        (required_input_fields=["text"]) + csv output.

        The original bug family: composer had NO contract checking at all, so it
        reported is_valid=True for any pipeline regardless of field contracts.
        The regression guard for that bug is test_observed_schema_no_guarantees_fails
        (test 6), which uses a non-text source with no observed-text exception.

        This test confirms the narrow observed-text rule correctly infers
        'text' from column='text', so the reported scenario is now valid
        (correctly). The runtime agrees because TextSource updates the
        builder-visible schema config as well as its normalized internal
        SchemaConfig (Task 6 runtime backing).
        """
        state = self._empty_state()
        state = state.with_source(self._make_source(
            plugin="text",
            options={"column": "text", "schema": {"mode": "observed"}},
        ))
        state = state.with_node(self._make_transform(
            "t1", "t1", "main",
            options={
                "required_input_fields": ["text"],
                "operations": [{"target": "combined", "expression": "row['text'] + ' world'"}],
            },
        ))
        state = state.with_output(self._make_output())
        state = state.with_edge(self._make_edge("e1", "source", "t1"))

        result = state.validate()
        # With the shared observed-text rule, this now passes (correctly)
        assert result.is_valid, result.errors
        # edge_contracts must confirm the contract is satisfied
        assert any(
            ec.to_id == "t1" and ec.satisfied
            for ec in result.edge_contracts
        )
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestSchemaContractValidation::test_text_heuristic_rescues_original_bug_scenario -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_state.py
git commit -m "test(composer): add regression test for original false-positive schema bug"
```

---

### Task 10: Patch the Pipeline Composer Skill

**Files:**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md:102-113, 353-357, 116`
- Modify: `.claude/skills/pipeline-composer/SKILL.md` (update the overlapping guidance / quick-reference content so the repo does not carry two conflicting pipeline-composer skill copies)

- [ ] **Step 1: Add completion criteria item 4**

In `pipeline_composer.md`, after line 107 (`3. All required plugin options are filled with meaningful values (not empty)`), add. Mirror the same completion-rule intent into the Claude skill if that copy is also used to guide agents; do not treat the web skill as the only source of truth:

```markdown
4. **All edge contracts are satisfied** — every downstream step's `required_input_fields` must be guaranteed by its upstream producer, and sink schemas may impose their own required fields. Check `edge_contracts` in the preview response. If any edge shows `"satisfied": false`, the pipeline is not complete.

- **Empty `edge_contracts` is not success** — `edge_contracts: []` means no field contracts were declared by any node. It does **not** mean all contracts are satisfied.
- **Skipped checks are unresolved** — if preview warnings say a contract check was skipped (for example because the producer is a coalesce node), treat that as unresolved rather than satisfied and surface the warning to the user.

Pipelines without `required_input_fields` declarations are not verified by the composer's contract check; the runtime validator is the final authority.

`generate_yaml` is an export step, not the primary validator. After Task 5B it becomes a hard backstop and should refuse invalid states, but the agent must still use `preview_pipeline` to diagnose and fix contract failures before retrying export.
```

- [ ] **Step 2: Add text source contract rule**

After line 356 (`- When wiring a text file via \`set_source_from_blob\`, you MUST pass...`), add. Apply the same text-source contract clarification to the overlapping quick-reference section in `.claude/skills/pipeline-composer/SKILL.md`:

```markdown
- **Schema rule for text sources:** Prefer an explicit `fixed` or `flexible` schema when you know the text column shape; it gives the strongest contract and clearer types. Narrow exception: a `text` source with `{"schema": {"mode": "observed"}}` and a non-empty `column` is still treated as guaranteeing `{column}` by the shared composer/runtime contract helper when `guaranteed_fields` is not explicitly set. Do not generalize this exception to other observed sources.
```

- [ ] **Step 2b: Add forward-reference in schema mode section**

In the "Choosing the right mode" bullet list (around line 143), after the `- **Sources:** Match the mode to how well you know the input data...` bullet, add:

```markdown
  **Default:** If downstream steps declare `required_input_fields` or reference fields by name, prefer `fixed` or `flexible` so the contract is explicit. `text` is the only observed-source exception, and only for its configured `column`; see the text source contract rule in "Plugin Quick Reference > Sources > text" below.
```

- [ ] **Step 3: Add fix flow example**

After the "Tool Failure Recovery" section (after line 127), add a new subsection. If the Claude skill has an equivalent recovery/help section, mirror this flow there as well; otherwise add a short note there pointing agents back to the same contract-resolution sequence so the two skill files do not drift semantically:

```markdown
#### Fixing Schema Contract Violations

When `preview_pipeline` returns an unsatisfied edge contract, follow this sequence:

1. **Read the violation** — identify which edge failed, what fields are missing, and which node is the producer.
2. **Patch the producer contract** — usually by fixing the actual producer shape first, then making the schema explicit. For most sources this means `patch_source_options` to change from `observed` to `fixed`/`flexible` with the required fields declared:
   ```json
   patch_source_options({
     "patch": {"schema": {"mode": "fixed", "fields": ["text: str"]}}
   })
   ```
3. **Re-preview** — call `preview_pipeline` and verify the edge now shows `"satisfied": true`.
4. **Only then call `generate_yaml` or report success.** If `generate_yaml` still refuses the export, treat that as confirmation the pipeline remains unresolved and return to `preview_pipeline` rather than bypassing the gate.

**Example — csv source + value_transform:**
- `preview_pipeline` returns: `edge_contracts: [{"from": "source", "to": "add_world", "satisfied": false, "consumer_requires": ["text"], "producer_guarantees": []}]`
- Fix: `patch_source_options({"patch": {"schema": {"mode": "fixed", "fields": ["text: str"]}}})`
- Re-preview confirms: `"satisfied": true`

**Text-source note:** if the source plugin is `text` and the consumer requires the configured `column`, observed mode is already valid via the shared observed-text rule. If the required field and `column` do not match, fix the `column` or downstream field reference; do not invent a `fixed` schema that claims a different key than the plugin actually emits.

If `get_pipeline_state` and `preview_pipeline` disagree (e.g., state shows a field but preview shows an unsatisfied contract), treat this as unresolved. Do not report success. Re-run both tools, fix the discrepancy, and confirm before responding.

#### Known Limitation: Intermediate Transforms Break the Guarantee Chain

Transforms without explicit schema declarations report zero guaranteed fields to downstream consumers — even schema-preserving transforms like `passthrough`. If a transform sits between a source and a consumer with `required_input_fields`, the contract check will report a violation even though the data flows through unchanged.

**Fix:** Either add a `schema` to the intermediate transform declaring the fields it passes through, or move `required_input_fields` to the first transform in the chain (directly downstream of the source). The source→first-consumer edge is where contract checking is most reliable.

#### Non-Converging Contract Violations

If `preview_pipeline` still shows `"satisfied": false` after **2** producer-schema patch attempts for the same edge, **stop patching and explain the limitation to the user.** The most common cause is an intermediate transform that does not propagate schema guarantees (see above). Do not repeatedly call `patch_source_options` or `patch_node_options` trying different schema configurations — after 2 attempts, treat the issue as structural rather than a missing field declaration. Ask the user whether to:
1. Add an explicit `schema` declaration on the intermediate transform, or
2. Accept that this contract cannot be verified at composition time (the runtime validator will still check it).

If the same producer feeds multiple consumers with conflicting truthful requirements, do not loop trying to force one schema to satisfy all of them. Surface the conflict explicitly and ask whether to:
1. Split the path so each consumer gets its own producer contract,
2. Insert an intermediate transform or aggregation with an explicit schema on one branch, or
3. Relax or correct one of the downstream requirements if it was overstated.
```

- [ ] **Step 4: Run skill drift test to check for regressions**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_skill_drift.py -v`
Expected: PASS. Then manually review both skill files together: the current drift suite enforces shared plugin-registry parity, but it does not exhaustively compare all prose guidance, so Task 10 is not complete if the two files still disagree in overlapping sections.

- [ ] **Step 5: Add a contract-violation pattern to `explain_validation_error`**

Because the skill tells the agent to use `explain_validation_error` for unclear failures, the implementation must also teach the catalogue about the new error family emitted by the contract pass. Fold this into the same change if Task 5C has not already landed separately.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md .claude/skills/pipeline-composer/SKILL.md
git commit -m "docs(skill): add schema contract completion criteria, text source rule, fix flow example"
```

---

### Task 11: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/composer_mcp/test_server.py tests/unit/web/sessions/test_routes.py -v`
Expected: All PASS

- [ ] **Step 2: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/web/ tests/integration/pipeline/test_composer_runtime_agreement.py -v`
Expected: All PASS

- [ ] **Step 3: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/schema.py src/elspeth/core/dag/graph.py src/elspeth/plugins/sources/text_source.py src/elspeth/web/composer/state.py src/elspeth/web/composer/tools.py src/elspeth/composer_mcp/server.py src/elspeth/web/sessions/routes.py`
Expected: No errors

- [ ] **Step 4: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/contracts/schema.py src/elspeth/core/dag/graph.py src/elspeth/plugins/sources/text_source.py src/elspeth/web/composer/state.py src/elspeth/web/composer/tools.py src/elspeth/composer_mcp/server.py src/elspeth/web/sessions/routes.py`
Expected: No errors

- [ ] **Step 5: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: No new violations (top-level imports of the shared raw-config contract helpers from L0 into L3 are layer-legal and preferred over new lazy imports)

### What This Doesn't Catch Yet

Task 5B adds a mechanical backstop only at the LLM/user-facing YAML export surfaces:

- `src/elspeth/composer_mcp/server.py` `generate_yaml`
- `GET /api/sessions/{id}/state/yaml` in `src/elspeth/web/sessions/routes.py`

It does **not** make `yaml_generator.generate_yaml()` universally validation-aware. In particular, `src/elspeth/web/execution/service.py` currently calls the serializer directly during execute-path run creation. That path remains outside the Task 5B backstop and still relies on earlier validation / workflow discipline rather than a universal export guard. Do not describe this plan as "all generate_yaml call sites are protected" unless a future task explicitly changes the execute path too.

- [ ] **Step 6: Register follow-up issue for expanded agreement tests**

The composer/runtime agreement test (Task 8) covers four scenarios: reject (wrong field), accept (observed text source special-case), reject (strict sink typed-schema requirement), and reject (aggregation consumer requirement coming from the runtime's nested `options` shape). It still does NOT cover coalesce-merge topologies or type-level schema compatibility — cases where the runtime validator checks properties the composer does not. These remaining gaps are expected (documented in the spec's "What This Doesn't Catch" table) but should be tracked for future expansion.

Create a Filigree issue:
```bash
filigree create "Expand composer/runtime agreement tests — coalesce and type-level" \
  --type=task --priority=3 \
  --description="The agreement test in tests/integration/pipeline/test_composer_runtime_agreement.py covers four basic scenarios (node reject, observed-text accept, strict-sink reject, aggregation nested-options reject). Expand it to cover the remaining runtime-only gaps: (1) coalesce-merge branch intersection, (2) type-level schema compatibility (Pydantic schema construction). These are cases where runtime may reject but the composer currently cannot check. See docs/superpowers/specs/2026-04-14-composer-schema-contract-validation-design.md 'What This Doesn't Catch' table."
```

- [ ] **Step 7: Final doc + parity reminders**

Before closing the work, do both of these:

1. Update `docs/reference/composer-tools.md` so the reference docs match the new tool behavior (`preview_pipeline.edge_contracts`, the contract-aware `explain_validation_error` path, and the YAML-export refusal for invalid states).
2. Compare the composer-side connection-namespace checks with the runtime builder/graph behavior in `src/elspeth/core/dag/graph.py` (`_validate_connection_namespaces(...)`). Keep the semantics aligned for the intended subset; if the composer implementation starts carrying a broader independent copy, either re-couple it mechanically or register a follow-up to extract a shared helper instead of letting the two validators drift.
