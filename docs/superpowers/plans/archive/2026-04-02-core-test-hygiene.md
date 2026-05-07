# Core Test Hygiene — Bloat Removal and Gap Filling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ~98 low-value tests from `src/elspeth/core/` test suite while adding 4 high-value gap-filling tests — improving signal-to-noise without reducing safety.

**Architecture:** Six independent cleanup/addition tasks targeting distinct test files. Tasks 1–5 are pure deletions or consolidations of existing tests. Task 6 adds a new partial-purge-failure test to `tests/unit/core/retention/test_purge.py`. Each task is independently committable.

**Tech Stack:** pytest, SQLAlchemy Core (in-memory SQLite), structlog, `tests/fixtures/landscape.py` helpers, `tests/fixtures/multi_run.py` fixture, `tests/fixtures/stores.py` (`MockPayloadStore`, `_ControlledStore`)

**SSRF note:** The original analysis flagged IPv6-mapped bypass as a gap. Research showed this is already thoroughly tested in both `tests/unit/core/security/test_web_ssrf_network_failures.py` (`TestSSRFIPv4MappedIPv6MetadataBypass`) and `tests/property/plugins/web_scrape/test_ssrf_properties.py` (`TestIPv4MappedIPv6Bypass`). No action needed.

---

### Task 1: Remove mutation-gap defaults tests

**Files:**
- Modify: `tests/unit/core/landscape/test_models_mutation_gaps.py` (787 lines → ~90 lines)

**What stays:** Tests that validate `__post_init__` guards, enum type checks, or non-trivial construction logic. Specifically:
- `TestRunDataclass.test_status_is_required_run_status_enum` — validates enum type, not just default
- `TestRunDataclass.test_run_with_all_optional_fields_set` — smoke test for full construction
- `TestNodeDataclass.test_registered_at_is_required` — validates constructor contract (missing required field)

**What goes:** Every test whose body is `assert obj.field is None` (testing Python's `= None` default) and every test whose body is `with pytest.raises(TypeError)` for a missing keyword argument to a frozen dataclass. These test Python's dataclass machinery, not ELSPETH logic.

**Remove these classes entirely (they contain only defaults/required-field tests):**
- `TestRowDataclass` (3 tests: `test_created_at_is_required`, `test_source_data_ref_defaults_to_none`, fixture)
- `TestTokenDataclass` (6 tests: all `*_defaults_to_none` + `test_created_at_is_required`, fixture)
- `TestNodeStateOpenDataclass` (4 tests: `test_status_is_literal_open`, `test_started_at_is_required`, `test_context_before_json_defaults_to_none`, fixture)
- `TestNodeStateCompletedDataclass` (4 tests: `test_duration_ms_is_required`, `test_context_*_defaults_to_none` x2, fixture)
- `TestNodeStateFailedDataclass` (5 tests: `test_duration_ms_is_required`, `test_error_json_defaults_to_none`, `test_output_hash_defaults_to_none`, `test_context_*_defaults_to_none` x2, fixture)
- `TestNodeStatePendingDataclass` (5 tests: all defaults + `test_status_is_literal_pending`, fixture)
- `TestCallDataclass` (6 tests: all defaults + `test_created_at_is_required`, fixture)
- `TestArtifactDataclass` (2 tests: `test_idempotency_key_defaults_to_none`, fixture)
- `TestRoutingEventDataclass` (3 tests: `test_created_at_is_required`, `test_reason_*_defaults_to_none` x2, fixture)
- `TestBatchDataclass` (5 tests: all defaults + `test_created_at_is_required`, fixture)
- `TestCheckpointDataclass` (2 tests + 1 smoke: keep `test_checkpoint_with_aggregation_state` if it tests deserialization, remove if it just checks field assignment)
- `TestEdgeDataclass` (2 tests: `test_created_at_is_required`, `test_edge_with_all_fields`)
- `TestRowLineageDataclass` (2 tests: `test_row_lineage_with_payload_available`, `test_row_lineage_with_payload_purged`)

**Remove from `TestRunDataclass` (keep class with 2 survivors):**
- `test_completed_at_defaults_to_none`
- `test_reproducibility_grade_defaults_to_none`
- `test_export_status_defaults_to_none`
- `test_export_error_defaults_to_none`
- `test_exported_at_defaults_to_none`
- `test_export_format_defaults_to_none`
- `test_export_sink_defaults_to_none`

**Remove from `TestNodeDataclass` (keep class with 1 survivor):**
- `test_schema_hash_defaults_to_none`
- `test_sequence_in_pipeline_defaults_to_none`

- [ ] **Step 1: Read the file and verify test names match plan**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_models_mutation_gaps.py --collect-only -q | wc -l`
Expected: 57 tests collected (matches analysis)

- [ ] **Step 2: Run all tests to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_models_mutation_gaps.py -v --tb=short`
Expected: All 57 tests PASS

- [ ] **Step 3: Rewrite file keeping only high-value tests**

Replace the entire file content with:

```python
# tests/unit/core/landscape/test_models_mutation_gaps.py
"""Surviving tests from mutation-gap suite — only non-trivial validation.

Original file tested 57 dataclass defaults/required-field patterns.
Most tested Python's @dataclass machinery, not ELSPETH logic.
Retained: enum type validation, required-field contracts for audit models,
full-construction smoke test.

Removed defaults-to-None tests: git log for 2026-04-02 has rationale.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import (
    Determinism,
    ExportStatus,
    Node,
    NodeType,
    ReproducibilityGrade,
    Run,
    RunStatus,
)


class TestRunDataclass:
    """Verify Run dataclass non-trivial field contracts."""

    def test_status_is_required_run_status_enum(self) -> None:
        """status must be RunStatus enum instance, not string."""
        run = Run(
            run_id="run-001",
            started_at=datetime.now(UTC),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.RUNNING,
        )
        assert isinstance(run.status, RunStatus)
        assert run.status == RunStatus.RUNNING

    def test_run_with_all_optional_fields_set(self) -> None:
        """Verify all optional fields can be set explicitly."""
        now = datetime.now(UTC)
        run = Run(
            run_id="run-002",
            started_at=now,
            config_hash="abc123",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=RunStatus.COMPLETED,
            completed_at=now,
            reproducibility_grade=ReproducibilityGrade.FULL_REPRODUCIBLE,
            export_status=ExportStatus.COMPLETED,
            export_error=None,
            exported_at=now,
            export_format="csv",
            export_sink="output",
        )
        assert run.completed_at == now
        assert run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE
        assert run.export_status == ExportStatus.COMPLETED
        assert run.export_format == "csv"


class TestNodeDataclass:
    """Verify Node dataclass required-field contracts."""

    def test_registered_at_is_required(self) -> None:
        """registered_at is required (no default)."""
        with pytest.raises(TypeError):
            Node(  # type: ignore[call-arg]
                node_id="node-001",
                run_id="run-001",
                plugin_name="test",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="x",
                config_json="{}",
                # registered_at missing
            )
```

- [ ] **Step 4: Run tests to verify survivors pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_models_mutation_gaps.py -v --tb=short`
Expected: 3 tests PASS

- [ ] **Step 5: Run full landscape test suite to check for no regressions**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/ -x --tb=short -q`
Expected: All pass (removed tests were self-contained, no cross-file dependencies)

- [ ] **Step 6: Commit**

```bash
git add tests/unit/core/landscape/test_models_mutation_gaps.py
git commit -m "test: remove 54 low-value mutation-gap defaults tests from models

Tests verified Python's @dataclass defaults (assert x is None) and
required-field enforcement (missing kwarg → TypeError). These test
the language runtime, not ELSPETH logic. Retained: enum type
validation, required-field contract for Node.registered_at, and
full-construction smoke test for Run.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Remove stdlib NaN guard tests

**Files:**
- Modify: `tests/unit/core/landscape/test_data_flow_nan_rejection.py` (53 lines → ~35 lines)

**What stays:** `TestNoUnguardedJsonDumps.test_all_json_dumps_have_allow_nan_false` — the AST scanner that structurally verifies every `json.dumps` call in `data_flow_repository.py` passes `allow_nan=False`. This is a genuine invariant enforcer.

**What goes:** `TestAllowNanFalseGuard` (4 tests) — these test that Python's `json.dumps(allow_nan=False)` rejects NaN/Infinity. That's a stdlib guarantee; we don't need to verify it.

- [ ] **Step 1: Run tests to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_data_flow_nan_rejection.py -v --tb=short`
Expected: 5 tests PASS

- [ ] **Step 2: Remove the TestAllowNanFalseGuard class**

Delete the `TestAllowNanFalseGuard` class (lines 36-52) from `tests/unit/core/landscape/test_data_flow_nan_rejection.py`. Keep the `TestNoUnguardedJsonDumps` class (lines 12-33) and the module docstring.

The file should contain only:

```python
"""Tests for NaN/Infinity rejection on all audit-path json.dumps calls.

The Data Manifesto requires NaN/Infinity rejection at all Tier 1 boundaries.
Every json.dumps in data_flow_repository.py must pass allow_nan=False.
"""

import ast
from pathlib import Path

import elspeth.core.landscape.data_flow_repository as mod


class TestNoUnguardedJsonDumps:
    """Audit: every json.dumps in data_flow_repository.py must pass allow_nan=False."""

    def test_all_json_dumps_have_allow_nan_false(self) -> None:
        source = Path(mod.__file__).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dumps"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                kwarg_names = [kw.arg for kw in node.keywords]
                assert "allow_nan" in kwarg_names, f"json.dumps at line {node.lineno} is missing allow_nan=False"
```

- [ ] **Step 3: Run test to verify survivor passes**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_data_flow_nan_rejection.py -v --tb=short`
Expected: 1 test PASS

- [ ] **Step 4: Commit**

```bash
git add tests/unit/core/landscape/test_data_flow_nan_rejection.py
git commit -m "test: remove 4 stdlib-testing NaN guard tests

Removed TestAllowNanFalseGuard which tested that json.dumps(allow_nan=False)
rejects NaN — a Python stdlib guarantee. Kept the AST scanner that
structurally verifies all json.dumps calls in data_flow_repository.py
include allow_nan=False.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Consolidate where_exactness tests

**Files:**
- Delete: `tests/unit/core/landscape/test_where_exactness.py` (191 lines)
- Delete: `tests/unit/core/landscape/test_where_exactness_data_flow.py` (356 lines)
- Delete: `tests/unit/core/landscape/test_where_exactness_execution.py` (363 lines)
- Delete: `tests/unit/core/landscape/test_where_exactness_run_lifecycle.py` (248 lines)
- Create: `tests/unit/core/landscape/test_where_exactness_consolidated.py`

**Strategy:** The 4 files contain 80 tests, most following an identical 3-variant pattern (target run / excludes adjacent / excludes prior). The "excludes adjacent" and "excludes prior" variants test the same invariant (exact match, not range) from different directions — one is sufficient. We keep one "target run returns correct data" test per query method and drop the redundant directional exclusion variants.

The multi_run fixture (`tests/fixtures/multi_run.py`) and its dataclasses are preserved unchanged.

**Query methods that need a "target run" test (30 methods):**

From `test_where_exactness.py`:
1. `get_rows(run_id)`
2. `get_all_tokens_for_run(run_id)`
3. `get_all_node_states_for_run(run_id)`
4. `get_all_calls_for_run(run_id)`
5. `get_all_routing_events_for_run(run_id)`
6. `get_all_token_outcomes_for_run(run_id)`
7. `get_batches(run_id)`

From `test_where_exactness_data_flow.py`:
8. `get_node(run_id, node_id)` — composite key, needs extra test for node_id dimension
9. `get_edges(run_id)`
10. `get_token_outcome(run_id, token_id)`
11. `get_token_outcomes_for_row(run_id, row_id)` — composite key
12. `get_validation_errors_for_run(run_id)` — uses record_validation_error first
13. `get_transform_errors_for_run(run_id)` — uses record_transform_error first

From `test_where_exactness_execution.py`:
14. `get_node_state(state_id)` — state scoped, not run scoped
15. `get_node_states_for_token(token_id)` — token scoped
16. `get_calls(state_id)` — state scoped
17. `find_call_by_request_hash(run_id, request_hash)` — **LLM cache critical**, keep both tests
18. `get_routing_events(state_id)` — state scoped
19. `get_batch(batch_id)` — batch scoped
20. `get_batch_members(batch_id)` — batch scoped
21. `get_operation(operation_id)` — operation scoped
22. `get_operation_calls(operation_id)` — operation scoped

From `test_where_exactness_run_lifecycle.py`:
23. `get_run(run_id)`
24. `complete_run(run_id, ...)` — mutation, verify only target affected
25. `update_run_status(run_id, ...)` — mutation, verify only target affected
26. `update_run_contract(run_id, ...)` — mutation, verify only target affected
27. `get_run_contract(run_id)`
28. `get_source_field_resolution(run_id)` / `record_source_field_resolution`
29. `set_export_status(run_id, ...)` — mutation, verify only target affected
30. `get_secret_resolutions_for_run(run_id)` / `record_secret_resolutions`
31. `list_runs(status=...)` — filtering, not scoping

- [ ] **Step 1: Run all where_exactness tests to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_where_exactness*.py -v --tb=short -q`
Expected: 80 tests PASS

- [ ] **Step 2: Note which tests use recorder methods that need error/secret setup**

The data_flow tests for `get_validation_errors_for_run` and `get_transform_errors_for_run` call `record_validation_error` and `record_transform_error` within the test. Read these tests to copy their setup exactly.

Similarly, `record_secret_resolutions` and `record_source_field_resolution` in the run_lifecycle tests need their setup patterns preserved.

Read the four existing files fully before writing the consolidated version.

- [ ] **Step 3: Create consolidated test file**

Write `tests/unit/core/landscape/test_where_exactness_consolidated.py` with:
- One test per query method that verifies target-run isolation
- For composite key methods (`get_node`, `get_token_outcomes_for_row`), include the secondary key test
- For `find_call_by_request_hash`, keep both tests (LLM cache safety critical)
- For mutation methods (`complete_run`, `update_run_status`, etc.), verify the mutation applied to target only AND didn't affect other runs

The test structure should be:

```python
# tests/unit/core/landscape/test_where_exactness_consolidated.py
"""WHERE clause exactness tests — one per query method.

Verifies that SQL queries use ``==`` (exact match) rather than
``>=`` / ``<=`` (range) operators.  The multi-run fixture creates
three runs with lexicographically ordered IDs (run-A < run-B < run-C)
so that an inequality operator would silently include data from
adjacent runs.

Consolidation: The original 4 files (80 tests) had 3 near-identical
variants per method. This file keeps one definitive test per method.
"""

from __future__ import annotations

import pytest

from elspeth.contracts import (
    CallType,
    CallStatus,
    NodeType,
    RunStatus,
    RoutingMode,
)
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.schema import SchemaConfig
from tests.fixtures.multi_run import MultiRunFixture

pytest_plugins = ["tests.fixtures.multi_run"]

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


# ── Core recorder methods (test_where_exactness.py survivors) ──


class TestGetRowsExactness:
    def test_returns_only_target_run_rows(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        rows = fix.recorder.get_rows(target.run_id)
        assert len(rows) == 2
        assert all(r.run_id == target.run_id for r in rows)
        assert {r.row_id for r in rows} == set(target.row_ids)


class TestGetAllTokensExactness:
    def test_returns_only_target_run_tokens(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        tokens = fix.recorder.get_all_tokens_for_run(target.run_id)
        expected_ids = {t.token_id for t in target.tokens}
        assert {t.token_id for t in tokens} == expected_ids
        assert len(tokens) == 2


class TestGetAllNodeStatesExactness:
    def test_returns_only_target_run_states(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        states = fix.recorder.get_all_node_states_for_run(target.run_id)
        expected_state_ids = {t.state_id for t in target.tokens}
        assert {s.state_id for s in states} == expected_state_ids


class TestGetAllCallsExactness:
    def test_returns_only_target_run_calls(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        calls = fix.recorder.get_all_calls_for_run(target.run_id)
        expected_call_ids = {t.call_id for t in target.tokens if t.call_id}
        assert {c.call_id for c in calls} == expected_call_ids


class TestGetAllRoutingEventsExactness:
    def test_returns_only_target_run_events(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        events = fix.recorder.get_all_routing_events_for_run(target.run_id)
        expected_event_ids = {t.routing_event_id for t in target.tokens if t.routing_event_id}
        assert {e.event_id for e in events} == expected_event_ids


class TestGetAllTokenOutcomesExactness:
    def test_returns_only_target_run_outcomes(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        outcomes = fix.recorder.get_all_token_outcomes_for_run(target.run_id)
        expected_token_ids = {t.token_id for t in target.tokens}
        assert {o.token_id for o in outcomes} == expected_token_ids


class TestGetBatchesExactness:
    def test_returns_only_target_run_batches(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        batches = fix.recorder.get_batches(target.run_id)
        assert len(batches) == 1
        assert batches[0].batch_id == target.batch_id


# ── Data flow repository methods (test_where_exactness_data_flow.py survivors) ──


class TestGetNodeExactness:
    def test_returns_only_target_run_node(self, multi_run_landscape: MultiRunFixture) -> None:
        """Composite key: must match both run_id AND node_id."""
        fix = multi_run_landscape
        target = fix.run("B")
        node = fix.recorder.get_node(target.run_id, target.source_node_id)
        assert node is not None
        assert node.run_id == target.run_id
        assert node.node_id == target.source_node_id

    def test_rejects_mismatched_run_and_node(self, multi_run_landscape: MultiRunFixture) -> None:
        """run-A's node_id with run-B's run_id must return None."""
        fix = multi_run_landscape
        run_a = fix.run("A")
        run_b = fix.run("B")
        node = fix.recorder.get_node(run_b.run_id, run_a.source_node_id)
        assert node is None


class TestGetEdgesExactness:
    def test_returns_only_target_run_edges(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        edges = fix.recorder.get_edges(target.run_id)
        assert len(edges) == 2
        edge_ids = {e.edge_id for e in edges}
        assert edge_ids == {target.edge_id_source_to_transform, target.edge_id_transform_to_sink}


class TestGetTokenOutcomeExactness:
    def test_returns_only_target_token_outcome(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        tok = target.tokens[0]
        outcome = fix.recorder.get_token_outcome(target.run_id, tok.token_id)
        assert outcome is not None
        assert outcome.token_id == tok.token_id


class TestGetTokenOutcomesForRowExactness:
    def test_returns_only_target_row_outcomes(self, multi_run_landscape: MultiRunFixture) -> None:
        """Composite key: must match both run_id AND row_id."""
        fix = multi_run_landscape
        target = fix.run("B")
        outcomes = fix.recorder.get_token_outcomes_for_row(target.run_id, target.row_ids[0])
        assert len(outcomes) == 1
        assert outcomes[0].token_id == target.tokens[0].token_id


class TestValidationErrorExactness:
    def test_returns_only_target_run_errors(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        from tests.fixtures.factories import make_context

        # Record validation errors in runs A and B
        for suffix in ("A", "B"):
            run = fix.run(suffix)
            ctx = make_context(run_id=run.run_id, node_id=run.source_node_id, landscape=fix.recorder)
            ctx.record_validation_error(
                row={"bad": suffix},
                error=f"bad-{suffix}",
                schema_mode="observed",
                destination="discard",
            )

        errors_b = fix.recorder.get_validation_errors_for_run("run-B")
        assert len(errors_b) == 1
        assert errors_b[0].error == "bad-B"


class TestTransformErrorExactness:
    def test_returns_only_target_run_errors(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        # Record transform errors in runs A and B
        for suffix in ("A", "B"):
            run = fix.run(suffix)
            fix.recorder.record_transform_error(
                run_id=run.run_id,
                token_id=run.tokens[0].token_id,
                node_id=run.transform_node_id,
                error_type="ValueError",
                error_message=f"fail-{suffix}",
                state_id=run.tokens[0].state_id,
            )

        errors_b = fix.recorder.get_transform_errors_for_run("run-B")
        assert len(errors_b) == 1
        assert errors_b[0].error_message == "fail-B"


# ── Execution repository methods (test_where_exactness_execution.py survivors) ──


class TestGetNodeStateExactness:
    def test_returns_correct_state(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        state = fix.recorder.get_node_state(target.tokens[0].state_id)
        assert state is not None
        assert state.state_id == target.tokens[0].state_id

    def test_rejects_nonexistent_state(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        state = fix.recorder.get_node_state("nonexistent-state")
        assert state is None


class TestGetNodeStatesForTokenExactness:
    def test_returns_only_target_token_states(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        tok = target.tokens[0]
        states = fix.recorder.get_node_states_for_token(tok.token_id)
        assert all(s.token_id == tok.token_id for s in states)
        assert len(states) >= 1


class TestGetCallsForStateExactness:
    def test_returns_only_target_state_calls(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        tok = target.tokens[0]  # token 0 has a call
        calls = fix.recorder.get_calls(tok.state_id)
        assert len(calls) == 1
        assert calls[0].call_id == tok.call_id


class TestFindCallByRequestHashExactness:
    def test_returns_only_target_run_match(self, multi_run_landscape: MultiRunFixture) -> None:
        """LLM cache safety: cross-run request hash match must not leak responses."""
        fix = multi_run_landscape
        target = fix.run("B")
        tok = target.tokens[0]
        calls = fix.recorder.get_calls(tok.state_id)
        assert len(calls) == 1
        request_hash = calls[0].request_hash

        result = fix.recorder.find_call_by_request_hash(target.run_id, request_hash)
        assert result is not None
        assert result.call_id == tok.call_id

    def test_same_hash_different_run_returns_none(self, multi_run_landscape: MultiRunFixture) -> None:
        """Same request_hash in run-A must not match when querying run-B."""
        fix = multi_run_landscape
        run_a = fix.run("A")
        run_b = fix.run("B")
        calls_a = fix.recorder.get_calls(run_a.tokens[0].state_id)
        assert len(calls_a) == 1
        request_hash_a = calls_a[0].request_hash

        # Query with run-B's run_id but run-A's hash — must not match
        result = fix.recorder.find_call_by_request_hash(run_b.run_id, request_hash_a)
        # Should either return None or return run-B's call (not run-A's)
        if result is not None:
            assert result.call_id != calls_a[0].call_id


class TestGetRoutingEventsForStateExactness:
    def test_returns_only_target_state_events(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        tok = target.tokens[0]  # token 0 has a routing event
        events = fix.recorder.get_routing_events(tok.state_id)
        assert len(events) == 1
        assert events[0].event_id == tok.routing_event_id


class TestGetBatchExactness:
    def test_returns_correct_batch(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        batch = fix.recorder.get_batch(target.batch_id)
        assert batch is not None
        assert batch.batch_id == target.batch_id

    def test_rejects_nonexistent_batch(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        batch = fix.recorder.get_batch("nonexistent-batch")
        assert batch is None


class TestGetBatchMembersExactness:
    def test_returns_only_target_batch_members(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        members = fix.recorder.get_batch_members(target.batch_id)
        expected_token_ids = {t.token_id for t in target.tokens}
        assert {m.token_id for m in members} == expected_token_ids


# ── Run lifecycle repository methods (test_where_exactness_run_lifecycle.py survivors) ──


class TestGetRunExactness:
    def test_returns_only_target_run(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        run = fix.recorder.get_run("run-B")
        assert run is not None
        assert run.run_id == "run-B"


class TestCompleteRunExactness:
    def test_only_target_run_completed(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        fix.recorder.complete_run("run-B")
        run_b = fix.recorder.get_run("run-B")
        run_a = fix.recorder.get_run("run-A")
        assert run_b is not None and run_b.completed_at is not None
        assert run_a is not None and run_a.completed_at is None


class TestUpdateRunStatusExactness:
    def test_only_target_run_status_updated(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        from elspeth.contracts import RunStatus

        fix.recorder.update_run_status("run-B", RunStatus.INTERRUPTED)
        run_b = fix.recorder.get_run("run-B")
        run_a = fix.recorder.get_run("run-A")
        assert run_b is not None and run_b.status == RunStatus.INTERRUPTED
        assert run_a is not None and run_a.status == RunStatus.RUNNING


class TestSetExportStatusExactness:
    def test_only_target_run_export_updated(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        from elspeth.contracts import ExportStatus

        fix.recorder.set_export_status("run-B", ExportStatus.COMPLETED)
        run_b = fix.recorder.get_run("run-B")
        run_a = fix.recorder.get_run("run-A")
        assert run_b is not None and run_b.export_status == ExportStatus.COMPLETED
        assert run_a is not None and run_a.export_status is None


class TestSecretResolutionsExactness:
    def test_returns_only_target_run_secrets(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        fix.recorder.record_secret_resolutions(
            "run-A",
            [{"name": "KEY_A", "source": "env", "fingerprint": "fp-a"}],
        )
        fix.recorder.record_secret_resolutions(
            "run-B",
            [{"name": "KEY_B", "source": "env", "fingerprint": "fp-b"}],
        )
        secrets_b = fix.recorder.get_secret_resolutions_for_run("run-B")
        assert len(secrets_b) == 1
        assert secrets_b[0]["name"] == "KEY_B"


class TestListRunsExactness:
    def test_filters_by_status(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        from elspeth.contracts import RunStatus

        fix.recorder.complete_run("run-B")
        fix.recorder.update_run_status("run-B", RunStatus.COMPLETED)
        completed = fix.recorder.list_runs(status=RunStatus.COMPLETED)
        assert all(r.status == RunStatus.COMPLETED for r in completed)
```

**Important:** The code above is a template. The implementing agent MUST:
1. Read the original 4 test files fully to get exact method signatures
2. Verify each recorder method name matches the actual API (e.g., check whether it's `get_calls(state_id)` or `get_calls_for_state(state_id)`)
3. Check that `record_transform_error`, `record_secret_resolutions`, `get_secret_resolutions_for_run`, `record_source_field_resolution`, `get_source_field_resolution`, `get_run_contract`, `update_run_contract` exist with the exact signatures used
4. Run the tests and fix any API mismatches before deleting the originals

- [ ] **Step 4: Run the consolidated tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_where_exactness_consolidated.py -v --tb=short`
Expected: ~35 tests PASS

- [ ] **Step 5: Delete the original 4 files**

```bash
rm tests/unit/core/landscape/test_where_exactness.py
rm tests/unit/core/landscape/test_where_exactness_data_flow.py
rm tests/unit/core/landscape/test_where_exactness_execution.py
rm tests/unit/core/landscape/test_where_exactness_run_lifecycle.py
```

- [ ] **Step 6: Run full landscape test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/ -x --tb=short -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add tests/unit/core/landscape/test_where_exactness_consolidated.py
git add -u tests/unit/core/landscape/  # picks up the 4 deleted files
git commit -m "test: consolidate 80 where_exactness tests into ~35

The original 4 files tested each query method with 3 near-identical
variants (target run / excludes adjacent / excludes prior). The
'excludes adjacent' and 'excludes prior' variants test the same
invariant (exact match vs range operator) from different directions.

Consolidated to one definitive test per query method. Retained both
variants for find_call_by_request_hash (LLM cache safety critical)
and composite key methods (get_node, get_token_outcomes_for_row).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Consolidate noncanonical validation error tests

**Files:**
- Modify: `tests/unit/core/landscape/test_validation_error_noncanonical.py` (329 lines → ~150 lines)

**What stays (5 tests):**
1. `test_primitive_int_audit_record_verified` — full audit trail verification for canonical data
2. `test_nan_audit_record_uses_repr_fallback` — full audit trail verification for non-canonical data
3. `test_multiple_non_canonical_rows` — batch recording + unique IDs
4. `test_repr_hash_helper` — standalone utility test (not a duplicate)
5. `test_noncanonical_metadata_structure` — validates `NonCanonicalMetadata` contract

**What goes (7 tests):**
- `test_record_primitive_int` — subset of `test_primitive_int_audit_record_verified`
- `test_record_primitive_string` — thin (only asserts `error_id is not None`)
- `test_record_list` — thin (only asserts `error_id is not None`)
- `test_record_nan_value` — subset of `test_nan_audit_record_uses_repr_fallback`
- `test_record_infinity_value` — thin (only asserts `error_id is not None`)
- `test_record_negative_infinity` — thin (only asserts `error_id is not None`)
- `test_audit_trail_contains_repr_fallback` — duplicates `test_nan_audit_record_uses_repr_fallback` (both query same table, check same `__repr__` keys)

- [ ] **Step 1: Run tests to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_validation_error_noncanonical.py -v --tb=short`
Expected: 12 tests PASS

- [ ] **Step 2: Remove the 7 low-value tests from TestValidationErrorNonCanonical**

From the `TestValidationErrorNonCanonical` class, delete:
- `test_record_primitive_int` (lines 51-64)
- `test_record_primitive_string` (lines 109-119)
- `test_record_list` (lines 121-131)
- `test_record_nan_value` (lines 135-146)
- `test_record_infinity_value` (lines 190-200)
- `test_record_negative_infinity` (lines 202-213)
- `test_audit_trail_contains_repr_fallback` (lines 216-248)

Keep the class with the 3 surviving tests:
- `test_primitive_int_audit_record_verified`
- `test_nan_audit_record_uses_repr_fallback`
- `test_multiple_non_canonical_rows`

Also keep the two standalone functions:
- `test_repr_hash_helper`
- `test_noncanonical_metadata_structure`

- [ ] **Step 3: Run tests to verify survivors pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_validation_error_noncanonical.py -v --tb=short`
Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/unit/core/landscape/test_validation_error_noncanonical.py
git commit -m "test: consolidate 12 noncanonical validation error tests to 5

Removed 7 thin tests that only asserted error_id is not None or
duplicated the fuller audit-trail verification tests. Kept:
primitive int audit record, NaN repr fallback, batch recording,
repr_hash helper, NonCanonicalMetadata contract.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Remove enum overlap test

**Files:**
- Modify: `tests/unit/core/landscape/test_models_enums.py` (125 lines → ~100 lines)

**What goes:** `TestModelEnumTypes.test_enum_type_verified_not_just_value` — this test creates a `Run` and asserts `isinstance(run.status, RunStatus)`. The `TestModelEnumTier1Rejection` tests (which reject string/integer values) already prove these fields are enum-typed. If the field accepted strings, the rejection tests would fail.

**What stays:** All 5 tests in `TestModelEnumTier1Rejection` — these are high-value Tier 1 corruption guards.

- [ ] **Step 1: Run tests to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_models_enums.py -v --tb=short`
Expected: 6 tests PASS

- [ ] **Step 2: Remove TestModelEnumTypes class**

Delete the entire `TestModelEnumTypes` class (lines 16-37) from the file. Keep the imports that are used by `TestModelEnumTier1Rejection`.

After removal, verify the import block. `Run`, `RunStatus`, `ExportStatus` were only used by the deleted class. Clean up unused imports:
- Remove: `Run`, `RunStatus`, `ExportStatus` (if they're not used in `TestModelEnumTier1Rejection`)
- Keep: `Determinism`, `Edge`, `Node`, `NodeType`, `RoutingMode` (used by Tier 1 tests)

- [ ] **Step 3: Run tests to verify survivors pass**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_models_enums.py -v --tb=short`
Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/unit/core/landscape/test_models_enums.py
git commit -m "test: remove redundant enum isinstance test

TestModelEnumTypes.test_enum_type_verified_not_just_value is
superseded by TestModelEnumTier1Rejection which proves fields
are enum-typed by verifying string/integer rejection.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Add partial purge failure test

**Files:**
- Modify: `tests/unit/core/retention/test_purge.py` (add ~80 lines to existing `TestPurgePayloads` class)

**What we're testing:** When `purge_payloads` processes a mix of refs where some exist-fail, some delete-fail, some skip, and some succeed — verify accounting invariant, correct grade update scoping, and that partial failures don't poison the batch.

**Existing infrastructure:**
- `_ControlledStore` class (line 247) provides `fail_exists_for`, `fail_delete_for`, `false_delete_for` injection
- `MockPayloadStore` from `tests/fixtures/stores.py` provides `store()` method
- Database helpers (`_create_run`, `_create_node`, `_create_row`) at top of file
- Existing tests monkeypatch `_find_affected_run_ids` and `update_grade_after_purge`

- [ ] **Step 1: Run existing purge tests to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/retention/test_purge.py -v --tb=short`
Expected: All existing tests PASS (count the number)

- [ ] **Step 2: Write the new test**

Add the following test to the `TestPurgePayloads` class in `tests/unit/core/retention/test_purge.py`, after the existing `test_purge_payloads_empty_input` method:

```python
    def test_partial_failure_accounting_invariant(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed success/skip/failure: accounting invariant holds and grade updates scope correctly.

        Scenario: 7 refs total
          - 2 deleted successfully
          - 2 skipped (not in store)
          - 1 exists-check fails (OSError)
          - 1 delete-call fails (OSError)
          - 1 delete returns False

        Verify:
          1. deleted_count + skipped_count + len(failed_refs) == 7
          2. Grade updates run only for runs linked to deleted refs
          3. Failed refs don't trigger grade updates
        """
        store = _ControlledStore()

        # Store payloads that will be deleted, exist-failed, delete-failed, or false-deleted
        ok_ref_1 = store.store(b"ok-1")
        ok_ref_2 = store.store(b"ok-2")
        exists_fail_ref = store.store(b"exists-fail")
        delete_fail_ref = store.store(b"delete-fail")
        false_delete_ref = store.store(b"false-delete")

        store._fail_exists_for.add(exists_fail_ref)
        store._fail_delete_for.add(delete_fail_ref)
        store._false_delete_for.add(false_delete_ref)

        manager = PurgeManager(db, store)

        # Map deleted refs → affected run IDs
        def _mock_affected(refs: list[str]) -> set[str]:
            # Only deleted refs should arrive here
            affected = set()
            if ok_ref_1 in refs:
                affected.add("run-alpha")
            if ok_ref_2 in refs:
                affected.add("run-beta")
            # Failed/skipped refs must NOT appear
            assert exists_fail_ref not in refs
            assert delete_fail_ref not in refs
            assert false_delete_ref not in refs
            assert "missing-ref-1" not in refs
            assert "missing-ref-2" not in refs
            return affected

        monkeypatch.setattr(manager, "_find_affected_run_ids", _mock_affected)

        grade_updates: list[str] = []
        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            lambda db_obj, run_id: grade_updates.append(run_id),
        )

        all_refs = [
            ok_ref_1,
            "missing-ref-1",
            exists_fail_ref,
            ok_ref_2,
            delete_fail_ref,
            "missing-ref-2",
            false_delete_ref,
        ]

        result = manager.purge_payloads(all_refs)

        # Accounting invariant
        assert result.deleted_count == 2
        assert result.skipped_count == 2
        assert len(result.failed_refs) == 3
        assert result.deleted_count + result.skipped_count + len(result.failed_refs) == len(all_refs)

        # Failed refs are exactly the three failure modes
        assert set(result.failed_refs) == {exists_fail_ref, delete_fail_ref, false_delete_ref}

        # Grade updates ran for both runs linked to successful deletions
        assert set(grade_updates) == {"run-alpha", "run-beta"}

        # No grade update failures (all mocked to succeed)
        assert result.grade_update_failures == ()
```

- [ ] **Step 3: Run new test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/core/retention/test_purge.py::TestPurgePayloads::test_partial_failure_accounting_invariant -v --tb=short`
Expected: PASS

- [ ] **Step 4: Run full purge test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/retention/test_purge.py -v --tb=short`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/core/retention/test_purge.py
git commit -m "test: add partial purge failure accounting invariant test

Verifies that mixed success/skip/failure across 7 refs maintains the
accounting invariant (deleted + skipped + failed == total) and that
grade updates scope correctly to runs with successful deletions only.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Summary

| Task | Action | Tests Removed | Tests Added | Net |
|------|--------|--------------|-------------|-----|
| 1 | Mutation-gap defaults | -54 | 0 | -54 |
| 2 | Stdlib NaN guards | -4 | 0 | -4 |
| 3 | Where_exactness consolidation | -80 | +35 | -45 |
| 4 | Noncanonical consolidation | -7 | 0 | -7 |
| 5 | Enum overlap | -1 | 0 | -1 |
| 6 | Partial purge failure | 0 | +1 | +1 |
| **Total** | | **-146** | **+36** | **-110** |

Lines saved: ~1,800 lines of test code removed, ~300 lines added = ~1,500 net reduction.

All tasks are independent and can be executed in any order or in parallel.
