# ADR-019 Stage 2/3 — Phase 4: Cross-Table Invariants (I1a, I1b, I1c, I3)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phase 4 adds NEW Tier 1 invariants that did not exist under ADR-018. They are additive and don't break the build, but they may surface latent bugs in the engine that the old invariant set was not catching. Phase 5 follows in the same PR.

**Goal:** Implement the four cross-table invariants from ADR-019 § "Cross-check invariants" (lines 237-269). Two are real-time at recording (I1c, I3); two are deferred — verified after sink writes (I1a, I1b). The mechanism choice for deferred invariants is **post-sink sweep in fresh-run `Orchestrator._execute_run`, resume-path `_process_resumed_rows`, and the public `resume()` no-work terminalization branch** per overview decision D3. The first two hooks run immediately after `_flush_and_write_sinks(...)` returns; the no-work resume branch has no sink flush, so it runs the same sweep immediately before deriving/finalizing the terminal resume status.

**Hook location verified against current HEAD (post-Stage-3 worktree).** `_finalize_source_iteration` at `src/elspeth/engine/orchestrator/core.py:2511` is **not** the correct hook: current source calls it before `_flush_and_write_sinks(...)`, and sink child outcomes are recorded by `SinkExecutor.write()` during that later sink phase. The deferred-invariant sweep slots in `src/elspeth/engine/orchestrator/core.py::_execute_run` immediately after `_flush_and_write_sinks(...)` returns (currently around lines 3029-3040) and before final progress / `PhaseCompleted` emission. The sibling resume path `_process_resumed_rows` also calls `_flush_and_write_sinks(...)` (currently around lines 3492-3502) and must run the same sweep immediately after that call returns. Current `resume()` also has an early no-work branch (`not unprocessed_rows and not restored_state and restored_coalesce_state is None`, currently around lines 3247-3264) that derives/finalizes a terminal status without calling `_process_resumed_rows`; Phase 4 must sweep that branch before terminal finalization. This makes fresh completions, resumed completions, and no-op resume completions all pass through the same deferred invariant gate.

**Shutdown semantic.** On graceful shutdown `_flush_and_write_sinks(...)` raises `GracefulShutdownError` before the fresh-run post-sink sweep call site. That natural control-flow skip is the required gate: a shutdown can legitimately leave fork-parents without children and BATCH_CONSUMED tokens without flush-results, because resume will produce them. When resume later reaches `_process_resumed_rows` and its sink flush returns normally, the resume-path sweep runs. Do not add a separate call in `_finalize_source_iteration`; running the sweep there would crash valid resumable stops and valid pre-sink runs.

**Files touched in this phase:**

- Modify: `src/elspeth/core/landscape/data_flow_repository.py` (add `_validate_cross_table_invariants`, `find_orphaned_transient_parents`, `find_orphaned_batch_consumptions`, and `sweep_deferred_invariants_or_crash`; wire I1c and I3 checks into `record_token_outcome`)
- Modify: `src/elspeth/engine/orchestrator/core.py:2972-3084` (add deferred-invariant sweep to `_execute_run` immediately after `_flush_and_write_sinks(...)` returns)
- Modify: `src/elspeth/engine/orchestrator/core.py:3247-3264` (add deferred-invariant sweep to the no-op `resume()` terminalization branch before `_derive_resume_terminal_status_from_audit(...)` / `finalize_run(...)`)
- Modify: `src/elspeth/engine/orchestrator/core.py:3420-3515` (add the same sweep to `_process_resumed_rows` after resume sink writes)
- Test (RED-first): `tests/integration/test_adr_019_cross_table_invariants.py` (NEW) — exercises each invariant
- Test (RED-first): `tests/integration/test_adr_019_sweep_durability.py` (NEW) — verifies sweep crash propagation, FAILED finalization, evidence preservation, and no-op resume coverage
- Test (unit): `tests/unit/core/landscape/test_data_flow_repository.py` — extend with I1c and I3 unit tests

**Background reading:** ADR-019 lines 237-269 (the four invariants and their semantics). The existing single-row guards in `_validate_outcome_fields` (Phase 1's `_TERMINAL_PAIR_FIELD_CONSTRAINTS`) are necessary but not sufficient; the cross-table invariants verify *structural consistency between `token_outcomes`, `node_states`, and `artifacts`*.

**Prerequisites for Phase 4 RED tests:**
- **Phases 1-3 are complete before this phase starts.** Phase 4's integration tests call `record_token_outcome` with the new `outcome: TerminalOutcome | None` and `path: TerminalPath` parameters and expect the Phase 1-3 schema/producer/accumulator changes to be present. Do not patch Phase 4 tests to accept the old ADR-018 `RowOutcome`-only signature.
- **Phase 3 Task 3.0 helpers** (`tests/integration/_helpers.py`) are NOT used in Phase 4. Invariant violations require direct factory API construction — no well-formed pipeline produces an orphan FORK_PARENT or an unflushed BATCH_CONSUMED. Phase 3 helpers exist for behaviour-change (counter/predicate) tests only.
- Batch creation lives on `ExecutionRepository`, not `DataFlowRepository`. Use `landscape_factory.execution.create_batch(run_id, aggregation_node_id, batch_id=batch_id)` to open a batch, and `landscape_factory.execution.complete_batch(batch_id, status=BatchStatus.COMPLETED)` to complete it. `open_batch` does not exist; `complete_batch` on `DataFlowRepository` does not exist. The tests below use the verified API.

---

## The four invariants (semantics summary)

### I1a (lineage-paired) — DEFERRED

`(TRANSIENT, FORK_PARENT)` and `(TRANSIENT, EXPAND_PARENT)` require **at least one child `token_outcomes` row reachable through `token_parents.parent_token_id == this token_id`**. `token_outcomes` has no `parent_token_id` column; the parent/child edge lives in `token_parents`. The child must complete after the parent, so this cannot be checked at parent-record time. Verified at end of run.

### I1b (aggregate-paired) — DEFERRED

`(TRANSIENT, BATCH_CONSUMED)` requires the consuming batch row to have reached `BatchStatus.COMPLETED` by end of run. The result token created at flush time carries the lifecycle answer; the engine guarantees BATCH_CONSUMED ⇒ batch COMPLETED by transactional construction (aggregation.py:486 calls `complete_batch(COMPLETED)` on the only success path that reaches CONSUMED_IN_BATCH recording at processor.py:1150).

**Why not a paired `token_outcomes` row?** ADR-019 now makes the batch-status witness explicit. The result token (created by `expand_token` at processor.py:1127) does NOT set `batch_id` in its own `token_outcomes` record — only CONSUMED_IN_BATCH and BUFFERED outcomes carry `batch_id`. The correct reachability path is: CONSUMED_IN_BATCH row → `batch_id` → `batches.status == COMPLETED`. A query against `token_outcomes` for a non-TRANSIENT row with the same `batch_id` would always return empty, making every BATCH_CONSUMED row appear orphaned. The `batches.status` check is both the correct invariant and the one the engine construction satisfies.

Same deferral reason — flush happens after consumption. Verified at end of run.

### I1c (sink-fallback-paired) — REAL-TIME

`(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` requires:
1. The producer passes the exact `sink_node_id` for the failsink that absorbed the row, AND
2. A paired `NodeStateStatus.COMPLETED` `node_state` row exists for the same `token_id` at that failsink node, AND
3. The producer passes the exact `artifact_id` returned by the failsink `register_artifact(...)` call, and that artifact row exists for the same `run_id` and exact `sink_node_id`.

Important cardinality note: current `SinkExecutor` opens/completes one failsink `node_state` per diverted token, but registers one artifact for the failsink write batch, linked to the first failsink state via `artifacts.produced_by_state_id`. I1c must **not** require `artifacts.produced_by_state_id == this token's node_state_id` for every diverted token unless this phase also changes artifact cardinality. This plan preserves the existing artifact contract: per token, I1c proves the exact failsink completed a node_state for that token, and separately proves the producer-declared artifact id exists for the same run/sink write. A query for "any artifact for this run/sink" is too weak and is not allowed.

The recorder must not infer the failsink node from `sink_name`: names are not
unique witnesses, and a query that accepts "any completed sink for this token"
can accidentally validate the primary sink or a sibling sink. Phase 1 adds
`sink_node_id` and `artifact_id` as forward-compatible keywords to
`record_token_outcome`; Phase 2 requires failsink-mode producers to pass the known
failsink node id and exact artifact id; Phase 4 makes those witnesses mandatory
for this real-time invariant.

Per ADR-019 line 256: "Real-time verifiable — `engine/executors/sink.py:898-946` registers the artifact and completes the failsink node_state before `record_token_outcome()` at line 952." So the recorder can check at write time.

### I3 (discard-FAILURE) — REAL-TIME

`(FAILURE, SINK_DISCARDED)` requires:
1. `sink_name == "__discard__"`, AND
2. **No paired** `NodeStateStatus.COMPLETED` sink `node_state` for the same `token_id` (no failsink absorbed).

Per ADR-019 line 261: "Real-time verifiable — `sink.py:977-1003` does not call `register_artifact()` and completes the primary node_state at `NodeStateStatus.FAILED` (line 991)."

---

## Tasks

### Task 4.1: Write the cross-table invariant RED tests

**Files:**
- Create: `tests/integration/test_adr_019_cross_table_invariants.py`

**Step 1: Write the cross-table invariant RED suite**

```python
"""ADR-019 § Cross-check invariants: structural consistency between
token_outcomes, node_states, and artifacts.

These invariants did NOT exist under ADR-018. They surface today's silent
audit-DB inconsistencies as Tier 1 errors. Each test simulates a corrupted
audit-DB state (the kind of state a faulty plugin could produce) and
verifies the recorder/orchestrator crashes with AuditIntegrityError.

PREREQUISITES:
- Phases 1-3 have shipped the updated ``record_token_outcome`` signature
  that accepts ``outcome: TerminalOutcome | None`` and ``path: TerminalPath``
  as separate parameters, plus the ``path`` column in ``token_outcomes_table``.
  Phase 4 tests must use this two-axis signature directly; do not reintroduce
  ADR-018 ``RowOutcome`` compatibility into new test helpers.
- Phase 3 Task 3.0 helpers (``tests/integration/_helpers.py``) are NOT used
  here. Invariant violations cannot be produced by any well-formed pipeline;
  they require direct factory API surgery to construct the illegal DB state.
  Phase 3 helpers exist for behaviour-change (counter/predicate) tests only.
"""

from __future__ import annotations

import pytest

# AuditIntegrityError lives in elspeth.contracts.errors (NOT contracts.audit —
# audit.py holds the dataclasses; errors.py holds the exception hierarchy).
# Verified against existing imports in src/elspeth/core/landscape/data_flow_repository.py:33
# and src/elspeth/core/landscape/model_loaders.py:50.
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
# TokenRef / DISCARD_SINK_NAME: from elspeth.contracts.audit
# (used in test_tier1_integrity.py:29 — confirmed import path)
from elspeth.contracts import NodeType
from elspeth.contracts.enums import BatchStatus, NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.factory import RecorderFactory

# Dynamic schema for tests that do not care about field definitions.
# Pattern matches tests/integration/audit/test_recorder_grades.py:16.
_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _build_base_run(factory: RecorderFactory) -> tuple[str, str, str]:
    """Create run + source node + row + token. Returns (run_id, source_node_id, token_id).

    All Phase 4 invariant tests need this minimal scaffold before constructing
    cross-table state. Callers add sink nodes, node_states, and artifacts on top.
    """
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source.node_id,
        row_index=0,
        data={"x": 1},
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    return run.run_id, source.node_id, token.token_id


def _register_sink_node(factory: RecorderFactory, run_id: str, *, name: str = "failsink") -> str:
    """Register a SINK node and return its node_id."""
    node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name=name,
        node_type=NodeType.SINK,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    return node.node_id


def _record_completed_sink_state_with_artifact(
    factory: RecorderFactory,
    *,
    run_id: str,
    token_id: str,
    sink_node_id: str,
) -> tuple[str, str]:
    """Record a NodeStateStatus.COMPLETED node_state for a SINK node + register artifact.

    This is the concrete witness that I1c requires for the single-token case:
    a completed failsink node_state plus a same-run/same-sink artifact witness.
    Production writes may register one artifact for a multi-token failsink batch,
    linked to the first failsink state; I1c must preserve that cardinality.
    Returns (state_id, artifact_id) for callers that need exact witnesses.

    Pattern mirrors tests/integration/audit/test_recorder_artifacts.py:40-56.
    """
    state = factory.execution.begin_node_state(
        token_id=token_id,
        node_id=sink_node_id,
        run_id=run_id,
        step_index=0,
        input_data={},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"written": True},
        duration_ms=1.0,
    )
    artifact = factory.execution.register_artifact(
        run_id=run_id,
        state_id=state.state_id,
        sink_node_id=sink_node_id,
        artifact_type="test",
        path=f"memory://failsink/{token_id}",
        content_hash="deadbeef" * 2,
        size_bytes=0,
    )
    return state.state_id, artifact.artifact_id


# ---------------------------------------------------------------------------
# I1c: failsink-paired (real-time)
# ---------------------------------------------------------------------------


class TestI1cFailsinkPaired:
    """I1c: (TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires a paired failsink
    node_state at NodeStateStatus.COMPLETED for the same token and exact
    producer-declared sink_node_id, plus an artifact witness for that same
    run/sink write. The artifact is intentionally not per-token; current
    SinkExecutor registers one artifact for the failsink write batch.
    Verified at write time in record_token_outcome.

    Fixture: ``landscape_factory`` from tests/integration/conftest.py:67.
    The fixture provides a function-scoped in-memory RecorderFactory — no
    tmp_path, no file I/O.
    """

    def test_failsink_pair_present_passes(self, landscape_factory: RecorderFactory) -> None:
        """Happy path: failsink COMPLETED node_state + artifact exist — no crash.

        Constructs the full I1c-valid single-token state: completed sink
        node_state + same-run/same-sink artifact registered before
        record_token_outcome is called. The recorder checks cross-table state
        at write time and must NOT raise.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        sink_node_id = _register_sink_node(landscape_factory, run_id, name="failsink")
        _state_id, artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=sink_node_id,
        )

        outcome_id = landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            sink_name="failsink",
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
            error_hash="abcd1234abcd1234abcd1234abcd1234",
        )
        assert outcome_id is not None, (
            "record_token_outcome must return outcome_id for valid I1c state."
        )

    def test_failsink_node_state_missing_crashes(self, landscape_factory: RecorderFactory) -> None:
        """I1c violation: token has no COMPLETED SINK node_state — crash.

        The recorder checks for a paired NodeStateStatus.COMPLETED sink node_state
        at write time. Absence of that row violates the I1c structural invariant:
        the audit trail claims a failsink absorbed the row but no node_state witnesses it.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        # Deliberately skip: _record_completed_sink_state_with_artifact().
        # No sink node_state exists for this token — I1c violation.

        with pytest.raises(AuditIntegrityError, match="failsink"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id="missing-failsink-node",
                artifact_id="missing-artifact",
                error_hash="abcd1234abcd1234abcd1234abcd1234",
            )

    def test_failsink_artifact_missing_crashes(self, landscape_factory: RecorderFactory) -> None:
        """I1c violation: COMPLETED sink node_state exists but no artifact — crash.

        The recorder must verify BOTH the token's node_state AND a registered
        artifact for the same run/sink write. A failsink that writes node_states
        but no artifact is a bug in the sink executor; the I1c invariant
        surfaces it at record time.

        Construction: begin_node_state + complete_node_state, but deliberately
        skip register_artifact. The node_state exists but has no artifact row.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        sink_node_id = _register_sink_node(landscape_factory, run_id, name="failsink")

        # Create COMPLETED node_state WITHOUT registering an artifact.
        state = landscape_factory.execution.begin_node_state(
            token_id=token_id,
            node_id=sink_node_id,
            run_id=run_id,
            step_index=0,
            input_data={},
        )
        landscape_factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"written": True},
            duration_ms=1.0,
        )
        # Deliberately skip: factory.execution.register_artifact(...)

        with pytest.raises(AuditIntegrityError, match="artifact"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=sink_node_id,
                artifact_id="missing-artifact",
                error_hash="abcd1234abcd1234abcd1234abcd1234",
            )

    def test_failsink_wrong_sink_node_crashes(self, landscape_factory: RecorderFactory) -> None:
        """I1c violation: witness exists for one sink, producer passes another.

        This protects the exact-node witness. The recorder must not accept
        "any completed sink node_state for this token" or infer the sink from
        ``sink_name``. The producer-declared ``sink_node_id`` must be the exact
        failsink node that completed the node_state.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        failsink_node_id = _register_sink_node(landscape_factory, run_id, name="failsink")
        sibling_sink_node_id = _register_sink_node(landscape_factory, run_id, name="other_failsink")
        _state_id, artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=failsink_node_id,
        )

        with pytest.raises(AuditIntegrityError, match="node|failsink"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=sibling_sink_node_id,
                artifact_id=artifact_id,
                error_hash="abcd1234abcd1234abcd1234abcd1234",
            )

    def test_failsink_wrong_artifact_crashes(self, landscape_factory: RecorderFactory) -> None:
        """I1c violation: artifact exists, but for a different sink write.

        The artifact witness must be the exact producer-declared artifact for
        the same run and exact failsink node. A query that accepts any artifact
        in the run, or any artifact for a sibling sink, is too weak.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        failsink_node_id = _register_sink_node(landscape_factory, run_id, name="failsink")
        sibling_sink_node_id = _register_sink_node(landscape_factory, run_id, name="other_failsink")
        _state_id, _correct_artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=failsink_node_id,
        )
        _other_state_id, wrong_artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=sibling_sink_node_id,
        )

        with pytest.raises(AuditIntegrityError, match="artifact"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=failsink_node_id,
                artifact_id=wrong_artifact_id,
                error_hash="abcd1234abcd1234abcd1234abcd1234",
            )


# ---------------------------------------------------------------------------
# I3: discard-no-failsink (real-time)
# ---------------------------------------------------------------------------


class TestI3DiscardNoFailsink:
    """I3: (FAILURE, SINK_DISCARDED) requires sink_name='__discard__' AND
    no paired NodeStateStatus.COMPLETED sink node_state for the same token.

    Fixture: ``landscape_factory`` from tests/integration/conftest.py:67.
    """

    def test_discard_with_failed_sink_state_passes(self, landscape_factory: RecorderFactory) -> None:
        """Happy path: discard with sink_name='__discard__' and no COMPLETED sink
        node_state for the token — I3-valid state, no crash.

        Production discard completes the primary sink node_state as FAILED before
        recording SINK_DISCARDED. I3 forbids a paired COMPLETED sink state, not
        the FAILED primary sink state that witnesses the discard attempt.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        primary_sink_node_id = _register_sink_node(landscape_factory, run_id, name="primary_sink")
        failed_state = landscape_factory.execution.begin_node_state(
            token_id=token_id,
            node_id=primary_sink_node_id,
            run_id=run_id,
            step_index=0,
            input_data={},
        )
        landscape_factory.execution.complete_node_state(
            state_id=failed_state.state_id,
            status=NodeStateStatus.FAILED,
            output_data={"discarded": True},
            duration_ms=1.0,
        )

        outcome_id = landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.SINK_DISCARDED,
            sink_name=DISCARD_SINK_NAME,
            error_hash="abcd1234abcd1234abcd1234abcd1234",
        )
        assert outcome_id is not None, (
            "record_token_outcome must return outcome_id for valid I3 state."
        )

    def test_discard_with_completed_sink_state_crashes(self, landscape_factory: RecorderFactory) -> None:
        """I3 violation: a COMPLETED SINK node_state exists alongside the discard record.

        Discard and failsink absorption are mutually exclusive. A COMPLETED sink
        node_state proves a failsink absorbed the row; discard claims it did not.
        The invariant says these cannot both be true — crash.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        sink_node_id = _register_sink_node(landscape_factory, run_id, name="failsink")
        _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=sink_node_id,
        )
        # Now attempt to record a SINK_DISCARDED outcome on the SAME token.
        # The failsink node_state contradicts the discard claim — I3 violation.

        with pytest.raises(AuditIntegrityError, match="discard"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name=DISCARD_SINK_NAME,
                error_hash="abcd1234abcd1234abcd1234abcd1234",
            )

    def test_discard_wrong_sink_name_rejected_by_scalar_guard(self, landscape_factory: RecorderFactory) -> None:
        """Scalar guard: SINK_DISCARDED requires sink_name='__discard__'.

        The sentinel name '__discard__' is the ONLY valid sink_name for the
        SINK_DISCARDED path. Any other name violates the semantic: discard is
        a specific system-managed outcome, not a sink-specific write. Phase 1's
        _TERMINAL_PAIR_FIELD_CONSTRAINTS already reject this before the Phase 4
        cross-table I3 query runs.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        # No COMPLETED sink node_state needed — the sink_name check fires first.

        with pytest.raises(ValueError, match="sink_name|__discard__"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name="other_sink_name",  # must be '__discard__'
                error_hash="abcd1234abcd1234abcd1234abcd1234",
            )


# ---------------------------------------------------------------------------
# I1a: fork-parent orphan (deferred)
# ---------------------------------------------------------------------------


class TestI1aForkParentDeferred:
    """I1a (deferred): (TRANSIENT, FORK_PARENT) at end of run requires ≥1 child
    token_outcomes row referencing this parent's token_id via token_parents.

    Tested in two layers (per project convention):
    1. Repository helper: find_orphaned_transient_parents returns the orphan.
    2. Sweep integration: _sweep_deferred_invariants_or_crash raises AuditIntegrityError.

    Fixture: ``landscape_factory`` from tests/integration/conftest.py:67.
    """

    def _build_fork_parent_orphan(self, landscape_factory: RecorderFactory) -> tuple[str, str]:
        """Record a (TRANSIENT, FORK_PARENT) token with NO children in token_parents.

        Returns (run_id, token_id). This simulates a faulty plugin that emits
        FORK_PARENT but never spawns children — the exact I1a violation shape.

        Uses fork_token to obtain valid fork_group_id metadata, then records only
        the parent outcome without children. The engine's normal fork_token call
        creates both parent record AND child tokens; here we skip the child step.
        """
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)

        # fork_token creates the parent<->child linkage in token_parents AND
        # returns the fork_group_id needed for record_token_outcome. We create
        # children via fork_token but do NOT record outcomes for them — the test
        # cares only about the FORK_PARENT row in token_outcomes having no child
        # token_outcomes rows (token_parents links alone are not sufficient).
        #
        # Simpler approach: record the parent outcome directly with a known
        # fork_group_id. The invariant query looks for (TRANSIENT, FORK_PARENT)
        # rows whose token_id has no child token_outcomes witness through token_parents.
        # A fresh token with no fork_token call has exactly this shape.
        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.FORK_PARENT,
            fork_group_id="fg_orphan_001",
        )
        return run_id, token_id

    def test_orphan_parent_detected_by_repository_helper(
        self, landscape_factory: RecorderFactory
    ) -> None:
        """Repository layer: find_orphaned_transient_parents returns the orphan row.

        Tests the new DataFlowRepository helper in isolation — RED before Phase 4
        adds find_orphaned_transient_parents() to data_flow_repository.py.
        """
        run_id, token_id = self._build_fork_parent_orphan(landscape_factory)

        orphans = landscape_factory.data_flow.find_orphaned_transient_parents(run_id)
        assert len(orphans) == 1, (
            f"Expected 1 orphaned FORK_PARENT, got {len(orphans)}. "
            f"find_orphaned_transient_parents must return rows where "
            f"no child token_outcomes row references the parent's token_id through token_parents."
        )
        assert orphans[0].token_id == token_id
        # path is returned as the enum value stored in the DB; compare by value.
        assert orphans[0].path == TerminalPath.FORK_PARENT.value, (
            f"Expected path={TerminalPath.FORK_PARENT.value!r}, got {orphans[0].path!r}."
        )

    def test_fork_parent_with_child_not_flagged(
        self, landscape_factory: RecorderFactory
    ) -> None:
        """Happy path: a FORK_PARENT with at least one child token_outcomes row
        must NOT appear in find_orphaned_transient_parents results.
        """
        run_id, _source_node_id, parent_token_id = _build_base_run(landscape_factory)
        row = landscape_factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=_source_node_id,
            row_index=1,
            data={"x": 2},
        )
        # fork_token creates children, populates token_parents, and records the
        # parent (TRANSIENT, FORK_PARENT) outcome. Do not record the parent a
        # second time; token_outcomes has a one-terminal-row-per-token unique
        # index.
        children, _fork_group_id = landscape_factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=parent_token_id, run_id=run_id),
            row_id=row.row_id,
            branches=["branch_a", "branch_b"],
        )
        # The ADR requires a child token_outcomes row, not merely a token_parents
        # link. Record one child lifecycle answer before asserting happy path.
        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=children[0].token_id, run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )

        orphans = landscape_factory.data_flow.find_orphaned_transient_parents(run_id)
        assert len(orphans) == 0, (
            f"A FORK_PARENT with a child token_outcomes row must not appear in "
            f"find_orphaned_transient_parents. Got: {orphans!r}."
        )

    def test_fork_parent_orphan_crashes_sweep(
        self, landscape_factory: RecorderFactory
    ) -> None:
        """Sweep layer: sweep_deferred_invariants_or_crash raises AuditIntegrityError.

        RED before Phase 4 adds sweep_deferred_invariants_or_crash() to
        DataFlowRepository. Until it lands, this test fails with AttributeError —
        that is the correct RED state.

        The sweep logic lives on DataFlowRepository (not Orchestrator) because the
        sweep body is pure repository SQL + raise. Orchestrator calls
        factory.data_flow.sweep_deferred_invariants_or_crash(run_id) from the
        real finalization path. Tests call the repository method directly — no
        Orchestrator construction needed.
        """
        run_id, _token_id = self._build_fork_parent_orphan(landscape_factory)

        orphans = landscape_factory.data_flow.find_orphaned_transient_parents(run_id)
        assert orphans, "Precondition: orphan must exist before sweep runs."

        # RED until Task 4.3 adds sweep_deferred_invariants_or_crash to DataFlowRepository.
        # After Phase 4 lands this becomes the actual sweep call; the AttributeError
        # on the missing method IS the meaningful RED signal — it proves the sweep
        # is not yet wired, rather than the test silently passing with a simulated raise.
        with pytest.raises(AuditIntegrityError, match="I1a"):
            landscape_factory.data_flow.sweep_deferred_invariants_or_crash(run_id)


# ---------------------------------------------------------------------------
# I1b: batch-consumed orphan (deferred)
# ---------------------------------------------------------------------------


class TestI1bBatchConsumedDeferred:
    """I1b (deferred): (TRANSIENT, BATCH_CONSUMED) at end of run requires the
    consuming batch row to have reached BatchStatus.COMPLETED before run-end.

    The I1b witness is BatchStatus.COMPLETED on the batches row, not a paired
    token_outcomes row. Result tokens created at flush time (via expand_token at
    processor.py:1127) do not carry batch_id in their token_outcomes records;
    only CONSUMED_IN_BATCH and BUFFERED outcomes set batch_id. A query for a
    non-TRANSIENT token_outcomes row with matching batch_id would always return
    empty. The correct check is batches.status == COMPLETED, which the engine
    guarantees by construction: aggregation.py:486 is the only success path that
    reaches the CONSUMED_IN_BATCH recording at processor.py:1150.

    Fixture: ``landscape_factory`` from tests/integration/conftest.py:67.
    """

    def _build_batch_consumed_orphan(
        self, landscape_factory: RecorderFactory
    ) -> tuple[str, str, str]:
        """Record a (TRANSIENT, BATCH_CONSUMED) token whose batch_id has no
        corresponding batch in BatchStatus.COMPLETED.

        Returns (run_id, token_id, batch_id). Simulates a faulty aggregation
        path that records BATCH_CONSUMED tokens but the batch never flushes.
        """
        run_id, source_node_id, token_id = _build_base_run(landscape_factory)
        batch_id = "batch_orphan_001"

        # Create a batch row (DRAFT) without completing it — satisfies the FK
        # target needed for token_outcomes.batch_id. The batch never reaches
        # COMPLETED, which is the I1b violation being tested.
        # API: landscape_factory.execution.create_batch (ExecutionRepository),
        # NOT data_flow. open_batch does not exist.
        landscape_factory.execution.create_batch(
            run_id=run_id,
            aggregation_node_id=source_node_id,
            batch_id=batch_id,
        )

        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch_id,
        )
        return run_id, token_id, batch_id

    def test_orphan_batch_detected_by_repository_helper(
        self, landscape_factory: RecorderFactory
    ) -> None:
        """Repository layer: find_orphaned_batch_consumptions returns the orphan batch_id.

        RED before Phase 4 adds find_orphaned_batch_consumptions() to
        data_flow_repository.py.
        """
        run_id, _token_id, batch_id = self._build_batch_consumed_orphan(landscape_factory)

        orphan_batch_ids = landscape_factory.data_flow.find_orphaned_batch_consumptions(run_id)
        assert batch_id in orphan_batch_ids, (
            f"Expected {batch_id!r} in orphaned batch IDs, got {orphan_batch_ids!r}. "
            f"find_orphaned_batch_consumptions must return batch_ids that have "
            f"BATCH_CONSUMED token_outcomes rows but no BatchStatus.COMPLETED row."
        )

    def test_batch_consumed_with_completed_batch_not_flagged(
        self, landscape_factory: RecorderFactory
    ) -> None:
        """Happy path: a BATCH_CONSUMED token whose batch reaches COMPLETED
        must NOT appear in find_orphaned_batch_consumptions.
        """
        run_id, source_node_id, token_id = _build_base_run(landscape_factory)
        batch_id = "batch_complete_001"

        # Create and complete the batch before recording the BATCH_CONSUMED outcome.
        # API: ExecutionRepository (landscape_factory.execution), not data_flow.
        landscape_factory.execution.create_batch(
            run_id=run_id,
            aggregation_node_id=source_node_id,
            batch_id=batch_id,
        )
        landscape_factory.execution.complete_batch(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED,
        )

        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch_id,
        )

        orphan_batch_ids = landscape_factory.data_flow.find_orphaned_batch_consumptions(run_id)
        assert batch_id not in orphan_batch_ids, (
            f"A batch that reached COMPLETED must not appear in orphaned batch IDs. "
            f"Got: {orphan_batch_ids!r}."
        )

    def test_batch_consumed_orphan_crashes_sweep(
        self, landscape_factory: RecorderFactory
    ) -> None:
        """Sweep layer: sweep_deferred_invariants_or_crash raises AuditIntegrityError on I1b.

        Mirrors test_fork_parent_orphan_crashes_sweep — validates the sweep
        integration layer in addition to the repository helper.

        Batch state is constructed via landscape_factory.execution.create_batch /
        complete_batch (ExecutionRepository). DataFlowRepository has no batch
        creation methods; open_batch does not exist anywhere.

        RED until Task 4.3 adds sweep_deferred_invariants_or_crash to DataFlowRepository.
        Fails with AttributeError until Phase 4 lands — that is the correct RED state.
        """
        run_id, _token_id, batch_id = self._build_batch_consumed_orphan(landscape_factory)

        orphan_batch_ids = landscape_factory.data_flow.find_orphaned_batch_consumptions(run_id)
        assert orphan_batch_ids, "Precondition: orphan batch must exist before sweep runs."

        # RED until Task 4.3 adds sweep_deferred_invariants_or_crash to DataFlowRepository.
        # The sweep runs BOTH the I1a and I1b checks; the I1b branch raises here because
        # the I1a check passes (no orphaned fork/expand parents in this fixture state).
        with pytest.raises(AuditIntegrityError, match="I1b"):
            landscape_factory.data_flow.sweep_deferred_invariants_or_crash(run_id)
```

**Step 2: Run RED**

Expected RED shape:
- Cross-table violation tests fail because Phase 4 has not added `_validate_cross_table_invariants`, `find_orphaned_transient_parents`, `find_orphaned_batch_consumptions`, and `sweep_deferred_invariants_or_crash` yet.
- Scalar producer-contract tests, such as the wrong `SINK_DISCARDED` sink name, are already owned by Phase 1's `_validate_outcome_fields` / `_TERMINAL_PAIR_FIELD_CONSTRAINTS`; they should raise `ValueError`, not `AuditIntegrityError`.
- Happy-path tests document the legal state shape and may pass once Phases 1-3 are present. Do not count them as required failures.

**Definition of Done:**
- [ ] I1c, I3, I1a, and I1b test cases written, including positive legal-state cases and negative cross-table violation cases
- [ ] Cross-table violation tests fail before Phase 4's edits land; scalar guard tests raise the existing scalar exception class
- [ ] Test fixtures construct the required cross-table state (node_states, artifacts) for the happy and violation paths

---

### Task 4.2: Implement I1c (failsink-paired) and I3 (discard-FAILURE) in the recorder

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py::record_token_outcome`

**Step 1: Add the cross-table check method**

```python
def _validate_cross_table_invariants(
    self,
    ref: TokenRef,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None,
    sink_node_id: str | None,
    artifact_id: str | None,
) -> None:
    """ADR-019 § Cross-check invariants — real-time invariants verified at write.

    I1c (sink-fallback-paired): (TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires
    a producer-declared exact sink_node_id witness, a paired
    NodeStateStatus.COMPLETED node_state at that failsink node for this token,
    AND the exact producer-declared artifact_id for the same run/sink write. The
    artifact witness is sink-write scoped, not per-token, matching
    SinkExecutor's current one-artifact-per-write-batch contract.

    I3 (discard-no-failsink): (FAILURE, SINK_DISCARDED) requires
    sink_name='__discard__' AND no paired NodeStateStatus.COMPLETED sink
    node_state for the same token (no failsink absorbed).

    Raises:
        AuditIntegrityError: if either invariant fails.
    """
    pair = (outcome, path)

    # I1c: failsink-paired
    if pair == (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK):
        if sink_node_id is None:
            raise AuditIntegrityError(
                f"I1c violation for token {ref.token_id}: "
                f"(TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires the producer "
                f"to pass the exact failsink node_id witness."
            )
        if artifact_id is None:
            raise AuditIntegrityError(
                f"I1c violation for token {ref.token_id}: "
                f"(TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires the producer "
                f"to pass the exact failsink artifact_id witness."
            )
        # Find the COMPLETED node_state for the exact failsink node that the
        # producer says absorbed the row. This prevents a different completed
        # sink for the same token from satisfying the invariant.
        completed_sink_state = self._ops.execute_fetchone(
            select(node_states_table.c.state_id, node_states_table.c.node_id)
            .select_from(
                node_states_table.join(
                    nodes_table,
                    and_(
                        node_states_table.c.node_id == nodes_table.c.node_id,
                        node_states_table.c.run_id == nodes_table.c.run_id,
                    ),
                )
            )
            .where(node_states_table.c.token_id == ref.token_id)
            .where(node_states_table.c.run_id == ref.run_id)
            .where(node_states_table.c.node_id == sink_node_id)
            .where(node_states_table.c.status == NodeStateStatus.COMPLETED.value)
            .where(nodes_table.c.node_type == NodeType.SINK.value)
        )
        if completed_sink_state is None:
            raise AuditIntegrityError(
                f"I1c violation for token {ref.token_id}: "
                f"(TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires a paired "
                f"NodeStateStatus.COMPLETED sink node_state, none found. The "
                f"failsink absorbed-the-row claim has no structural witness."
            )
        # The failsink write must have the exact producer-declared artifact.
        # Do not require produced_by_state_id == completed_sink_state.state_id
        # here: current SinkExecutor registers one artifact for the failsink
        # write batch, linked to the first failsink state, while every diverted
        # token still gets its own completed failsink node_state.
        artifact_row = self._ops.execute_fetchone(
            select(
                artifacts_table.c.artifact_id,
                artifacts_table.c.sink_node_id,
                artifacts_table.c.produced_by_state_id,
            )
            .where(artifacts_table.c.artifact_id == artifact_id)
            .where(artifacts_table.c.run_id == ref.run_id)
            .where(artifacts_table.c.sink_node_id == completed_sink_state.node_id)
        )
        if artifact_row is None:
            raise AuditIntegrityError(
                f"I1c violation for token {ref.token_id}: "
                f"failsink node {completed_sink_state.node_id} has no registered "
                f"artifact_id={artifact_id!r} for this run. Tier 1: a "
                f"failsink-paired diversion requires the exact durable artifact "
                f"witness returned by the sink write."
            )

    # I3: discard-no-failsink
    if pair == (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
        if sink_name != DISCARD_SINK_NAME:
            raise AuditIntegrityError(
                f"I3 violation for token {ref.token_id}: "
                f"(FAILURE, SINK_DISCARDED) requires sink_name={DISCARD_SINK_NAME!r}, "
                f"got sink_name={sink_name!r}. The discard semantic is "
                f"reserved for the sentinel sink name."
            )
        # Verify no COMPLETED sink node_state exists for this token.
        # (Discard mode does NOT call register_artifact and completes the
        # primary node_state at FAILED — see sink.py:991, sink.py:977-1003.
        # Presence of a COMPLETED sink node_state would mean a failsink
        # absorbed the row, contradicting the discard semantic.)
        #
        # The "sink node_state" filter joins node_states with nodes on
        # (run_id, node_id) and filters by node_type == NodeType.SINK.value.
        # Verified against current HEAD: nodes_table is at schema.py:97
        # with node_type: String(32), values from the NodeType enum
        # (contracts/enums.py:91 — SOURCE/TRANSFORM/GATE/AGGREGATION/COALESCE/SINK).
        completed_sink_state = self._ops.execute_fetchone(
            select(node_states_table.c.state_id)
            .select_from(
                node_states_table.join(
                    nodes_table,
                    and_(
                        node_states_table.c.node_id == nodes_table.c.node_id,
                        node_states_table.c.run_id == nodes_table.c.run_id,
                    ),
                )
            )
            .where(node_states_table.c.token_id == ref.token_id)
            .where(node_states_table.c.run_id == ref.run_id)
            .where(node_states_table.c.status == NodeStateStatus.COMPLETED.value)
            .where(nodes_table.c.node_type == NodeType.SINK.value)
        )
        if completed_sink_state is not None:
            raise AuditIntegrityError(
                f"I3 violation for token {ref.token_id}: discard-mode "
                f"recording attempted but a paired NodeStateStatus.COMPLETED "
                f"sink node_state exists ({completed_sink_state.state_id}). "
                f"Discard requires no failsink absorption — the layers "
                f"disagree about whether the row failed."
            )
```

**Imports for the I3 query** (added in the same file edit):

```python
from sqlalchemy import and_, select  # `and_` for the multi-key join condition
from sqlalchemy.engine import Row as SQLAlchemyRow

from elspeth.contracts import NodeType
from elspeth.contracts.enums import BatchStatus, NodeStateStatus, TerminalOutcome, TerminalPath
# These imports are NEW for Phase 4 — not currently in data_flow_repository.py
# (current imports verified at data_flow_repository.py:47-56):
from elspeth.core.landscape.schema import artifacts_table, batches_table, node_states_table
# The existing imports already cover: nodes_table, token_outcomes_table,
# token_parents_table (data_flow_repository.py:48-55).
```

Verify with: `.venv/bin/python -c "from elspeth.contracts import NodeType; print(NodeType.SINK.value)"` — expect `sink`.

**Step 2: Wire into `record_token_outcome`**

Add the call immediately after `_validate_token_run_ownership(ref)` and before the INSERT:

```python
# Showing only the changed call-site context. Full signature at
# data_flow_repository.py:802 — all params preserved; only adding path:TerminalPath.
def record_token_outcome(
    self,
    ref: TokenRef,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None = None,
    sink_node_id: str | None = None,
    artifact_id: str | None = None,
    batch_id: str | None = None,
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
    error_hash: str | None = None,
    context: Mapping[str, object] | None = None,
) -> str:
    self._validate_outcome_fields(
        outcome=outcome,
        path=path,
        sink_name=sink_name,
        batch_id=batch_id,
        fork_group_id=fork_group_id,
        join_group_id=join_group_id,
        expand_group_id=expand_group_id,
        error_hash=error_hash,
    )
    self._validate_token_run_ownership(ref)
    # NEW: Phase 4 cross-table invariants for real-time-verifiable cases.
    self._validate_cross_table_invariants(
        ref,
        outcome,
        path,
        sink_name=sink_name,
        sink_node_id=sink_node_id,
        artifact_id=artifact_id,
    )
    # Existing INSERT body follows unchanged (outcome_id, is_terminal, etc.)
```

**Step 3: GREEN**

Run: `.venv/bin/python -m pytest tests/integration/test_adr_019_cross_table_invariants.py::TestI1cFailsinkPaired tests/integration/test_adr_019_cross_table_invariants.py::TestI3DiscardNoFailsink -v`

Expected: all I1c and I3 tests pass.

**Definition of Done:**
- [ ] `_validate_cross_table_invariants` method added
- [ ] Wired into `record_token_outcome`
- [ ] `sink_node_id` witness is mandatory for `SINK_FALLBACK_TO_FAILSINK`; I1c
      checks the exact producer-declared failsink node and never infers from
      `sink_name` or from "any completed sink"
- [ ] I1c and I3 unit + integration tests pass
- [ ] mypy clean

---

### Task 4.3: Implement deferred I1a / I1b sweep after sink writes in fresh and resumed runs

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py:2972-3084` (`_execute_run`)
- Modify: `src/elspeth/engine/orchestrator/core.py:3420-3515` (`_process_resumed_rows`)

**Step 1: Locate the post-sink hook**

```bash
grep -n "_flush_and_write_sinks" src/elspeth/engine/orchestrator/core.py
```

Expected: `_execute_run` calls `_run_main_processing_loop(...)` first, then `_flush_and_write_sinks(...)` around lines 3029-3040, then emits final progress and `PhaseCompleted`. `_process_resumed_rows` calls `_run_resume_processing_loop(...)`, then `_flush_and_write_sinks(...)` around lines 3492-3502, then exits to the public resume wrapper. The deferred-invariant sweep slots immediately after `_flush_and_write_sinks(...)` returns in **both** paths.

Also inspect the public `resume()` no-work terminalization branch around the `if not unprocessed_rows and not restored_state and restored_coalesce_state is None:` guard. That branch does not call `_process_resumed_rows`; it derives/finalizes from existing audit rows directly. Phase 4 must run the same sweep in that branch immediately before `_derive_resume_terminal_status_from_audit(...)` / `finalize_run(...)`, otherwise a fully processed resumed run can finalize successfully with pre-existing orphaned deferred invariant rows.

Do not insert this sweep in `_finalize_source_iteration`. Current source calls `_finalize_source_iteration` before `_flush_and_write_sinks(...)`; sink child outcomes are recorded during `SinkExecutor.write()`, so a pre-sink sweep can reject valid fork/expand runs as orphaned.

The sweep is naturally skipped when `_flush_and_write_sinks(...)` raises before returning. Graceful shutdown is the expected benign case: it can legitimately leave fork-parents without children, and the resume path completes them before running its own post-sink sweep. Any other unhandled sink-flush exception also skips the sweep intentionally because the postconditions for I1a/I1b are not stable when sink flushing did not finish; the outer failure ceremony handles the run failure.

**Step 2: Add the sweep in `_execute_run` — exact insertion point after `_flush_and_write_sinks(...)`**

Insert this block immediately after the `_flush_and_write_sinks(...)` call in `_execute_run` and before the existing final-progress block:

```python
            # Existing sink-write phase.
            self._flush_and_write_sinks(
                factory,
                run_id,
                loop_ctx,
                artifacts.sink_id_map,
                artifacts.edge_map,
                loop_result.interrupted,
                on_token_written_factory=self._make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
                shutdown_checkpoint_source_id=artifacts.source_id,
            )

            # ADR-019 Phase 4: deferred cross-table invariants. MUST run after
            # sink writes because child sink token_outcomes are recorded inside
            # SinkExecutor.write(). `_finalize_source_iteration` is too early.
            # Any unhandled sink-flush exception skips this naturally because
            # I1a/I1b postconditions are not stable until sink writes finish.
            # GracefulShutdownError is the expected benign shutdown case.
            factory.data_flow.sweep_deferred_invariants_or_crash(run_id)

            # Existing final progress + PROCESS PhaseCompleted block follows.
```

**Step 2b: Add the same sweep in `_process_resumed_rows`**

Insert the call immediately after the resume path's `_flush_and_write_sinks(...)`
returns and before the `except GracefulShutdownError` / generic exception handler
can be left behind:

```python
            # Existing resume sink-write phase.
            self._flush_and_write_sinks(
                factory,
                run_id,
                loop_ctx,
                artifacts.sink_id_map,
                artifacts.edge_map,
                interrupted,
                on_token_written_factory=self._make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
                shutdown_checkpoint_source_id=artifacts.source_id,
            )

            # ADR-019 Phase 4: resumed runs reach stable I1a/I1b postconditions
            # only after resume sink writes finish. Mirror the fresh-run sweep.
            factory.data_flow.sweep_deferred_invariants_or_crash(run_id)
```

**Step 2c: Add the sweep in the public `resume()` no-work terminalization branch**

Move or create `resume_start_time = time.perf_counter()` before the no-work
branch so the failure ceremony has a valid duration anchor, then insert the
sweep before `_derive_resume_terminal_status_from_audit(...)`:

```python
        resume_start_time = time.perf_counter()

        if not unprocessed_rows and not restored_state and restored_coalesce_state is None:
            # ADR-019 Phase 4: no-op resume still reaches a terminal audit state.
            # It bypasses _process_resumed_rows, so it must run the deferred
            # invariant sweep here before deriving/finalizing the terminal
            # status from audit rows. If the sweep raises, use the same failed
            # ceremony contract as the row-processing resume path; otherwise the
            # run could be finalized as COMPLETED/PARTIAL_SUCCESS with orphaned
            # deferred invariant evidence still present.
            try:
                factory.data_flow.sweep_deferred_invariants_or_crash(run_id)
            except Exception:
                try:
                    self._emit_failed_ceremony(run_id, factory, resume_start_time)
                except Exception:
                    slog.debug("Failure ceremony failed — original exception preserved", run_id=run_id)
                self._safe_flush_telemetry()
                raise

            (
                terminal_status,
                audit_rows_processed,
                audit_rows_succeeded,
                audit_rows_failed,
                audit_rows_routed_success,
                audit_rows_routed_failure,
                audit_rows_quarantined,
            ) = self._derive_resume_terminal_status_from_audit(factory, run_id)
            factory.run_lifecycle.finalize_run(run_id, status=terminal_status)
            # Existing RunFinished, RunSummary, checkpoint deletion, and
            # RunResult return block follows unchanged.
```

Do not leave a second `resume_start_time = time.perf_counter()` assignment below
the no-work branch; the row-processing path should reuse the earlier timestamp.

**Step 2d: Add the positive fork/coalesce false-positive regression**

Add this concrete test body to `tests/integration/test_adr_019_sweep_durability.py`
or to `tests/integration/test_adr_019_cross_table_invariants.py` if that file
already owns the positive Orchestrator cases:

```python
def test_valid_fork_coalesce_run_does_not_false_positive_after_sink_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-sink sweep must accept a valid fork/coalesce run.

    This catches the wrong-hook regression: if the sweep runs from
    _finalize_source_iteration before SinkExecutor.write() records child sink
    outcomes, the run false-positives as I1a orphaned even though the final
    post-sink audit state is valid.
    """
    from elspeth.core.config import CoalesceSettings, GateSettings
    from elspeth.contracts import RunStatus
    from elspeth.core.landscape.data_flow_repository import DataFlowRepository
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from tests.fixtures.base_classes import as_sink, as_source
    from tests.fixtures.pipeline import build_fork_pipeline

    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    sweep_calls: list[str] = []

    original_sweep = DataFlowRepository.sweep_deferred_invariants_or_crash

    def _spy_sweep(self: DataFlowRepository, run_id: str) -> None:
        sweep_calls.append(run_id)
        original_sweep(self, run_id)

    monkeypatch.setattr(
        DataFlowRepository,
        "sweep_deferred_invariants_or_crash",
        _spy_sweep,
    )

    gate = GateSettings(
        name="fork_gate",
        input="list_source_out",
        condition="True",
        routes={"true": "fork", "false": "fork"},
        fork_to=["path_a", "path_b"],
    )
    coalesce = CoalesceSettings(
        name="merge_results",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
        on_success="default",
    )
    source, _transforms, sinks, graph = build_fork_pipeline(
        [{"id": 1, "value": 7}],
        gate=gate,
        branch_transforms={
            "path_a": [],
            "path_b": [],
        },
        coalesce_settings=[coalesce],
    )
    config = PipelineConfig(
        source=as_source(source),
        transforms=[],
        sinks={name: as_sink(sink) for name, sink in sinks.items()},
        coalesce_settings=[coalesce],
        gates=[gate],
    )

    result = Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

    assert result.status == RunStatus.COMPLETED
    assert sweep_calls == [result.run_id]
```

This is RED if the sweep runs before sink writes, because
`original_sweep(...)` sees valid fork parents without child sink outcomes yet.
It is GREEN only when the sweep is wired after `_flush_and_write_sinks(...)`.

**Step 3: Add the sweep body to DataFlowRepository**

The sweep body is pure repository SQL + raise. It belongs on `DataFlowRepository`, not on `Orchestrator` — consistent with the project's "SQL surface centralized in the repository layer" contract. `_execute_run` already has the repository `factory`, so it calls `factory.data_flow.sweep_deferred_invariants_or_crash(run_id)` directly. Tests can call the sweep without constructing a full `Orchestrator` instance (they use `landscape_factory.data_flow` directly).

**3a: Add `sweep_deferred_invariants_or_crash` to `DataFlowRepository` in `data_flow_repository.py`**

Place this method alongside `find_orphaned_transient_parents` and `find_orphaned_batch_consumptions` (all three are added in the same Phase 4 commit, after `record_token_outcome`):

```python
def sweep_deferred_invariants_or_crash(self, run_id: str) -> None:
    """ADR-019 § Cross-check invariants I1a, I1b — end-of-run sweep.

    I1a: every (TRANSIENT, FORK_PARENT) and (TRANSIENT, EXPAND_PARENT)
         parent must have ≥1 child token_outcomes row referencing it via
         token_parents.
    I1b: every (TRANSIENT, BATCH_CONSUMED) row must have its consuming
         batch row in BatchStatus.COMPLETED. The result token created at
         flush time does not carry batch_id in its own token_outcomes row;
         the link is: BATCH_CONSUMED.batch_id → batches.status == COMPLETED.

    Called from Orchestrator._execute_run and _process_resumed_rows through
    factory.data_flow after _flush_and_write_sinks(...) returns. Graceful
    shutdown skips the fresh-run call because _flush_and_write_sinks raises
    GracefulShutdownError before the post-sink sweep site; the resume path then
    completes missing children and batches before running its own post-sink sweep.

    Tests call this method directly via ``landscape_factory.data_flow`` without
    constructing a full Orchestrator instance.

    Raises:
        AuditIntegrityError: if any invariant is violated.
    """
    # I1a: orphaned FORK_PARENT / EXPAND_PARENT.
    orphan_parents = self.find_orphaned_transient_parents(run_id)
    if orphan_parents:
        formatted = ", ".join(
            f"{r.token_id} (path={r.path})" for r in orphan_parents[:10]
        )
        raise AuditIntegrityError(
            f"ADR-019 I1a violation: {len(orphan_parents)} fork/expand parent "
            f"token(s) have no children at run-end. Examples: {formatted}. "
            f"Every (TRANSIENT, FORK_PARENT|EXPAND_PARENT) row must have at "
            f"least one child token_outcomes row through token_parents."
        )

    # I1b: orphaned BATCH_CONSUMED.
    orphan_batches = self.find_orphaned_batch_consumptions(run_id)
    if orphan_batches:
        formatted = ", ".join(orphan_batches[:10])
        raise AuditIntegrityError(
            f"ADR-019 I1b violation: {len(orphan_batches)} batch_id(s) had "
            f"BATCH_CONSUMED tokens but the batch never reached "
            f"BatchStatus.COMPLETED. Examples: {formatted}. Every "
            f"(TRANSIENT, BATCH_CONSUMED) token requires its consuming batch "
            f"to reach COMPLETED before end of run."
        )
```

**Step 4: Add the two repository query helpers**

Place these above `sweep_deferred_invariants_or_crash` in `data_flow_repository.py`:

```python
from sqlalchemy.engine import Row as SQLAlchemyRow


def find_orphaned_transient_parents(self, run_id: str) -> list[SQLAlchemyRow]:
    """ADR-019 I1a: parent tokens (TRANSIENT, FORK_PARENT|EXPAND_PARENT)
    with no child token_outcomes row for the given run."""
    parent_paths = (
        TerminalPath.FORK_PARENT.value,
        TerminalPath.EXPAND_PARENT.value,
    )
    child_outcomes = token_outcomes_table.alias("child_outcomes")
    query = (
        select(token_outcomes_table.c.token_id, token_outcomes_table.c.path)
        .where(token_outcomes_table.c.run_id == run_id)
        .where(token_outcomes_table.c.path.in_(parent_paths))
        .where(token_outcomes_table.c.outcome == TerminalOutcome.TRANSIENT.value)
        .where(
            ~select(child_outcomes.c.outcome_id)
            .select_from(
                token_parents_table.join(
                    child_outcomes,
                    and_(
                        child_outcomes.c.token_id == token_parents_table.c.token_id,
                        child_outcomes.c.run_id == run_id,
                    ),
                )
            )
            .where(token_parents_table.c.parent_token_id == token_outcomes_table.c.token_id)
            .exists()
        )
    )
    return list(self._ops.execute_fetchall(query))


def find_orphaned_batch_consumptions(self, run_id: str) -> list[str]:
    """ADR-019 I1b: distinct batch_ids that have BATCH_CONSUMED tokens
    but no row in batches with status=COMPLETED for the given run.

    The I1b invariant links CONSUMED_IN_BATCH rows to their consuming
    batch via batch_id → batches.status == COMPLETED. Result tokens
    created at flush time do not carry batch_id in token_outcomes, so
    the check is against the batches table, not against a paired
    token_outcomes row.

    Returns just the batch_id strings (empty list = no violation)."""
    query = (
        select(token_outcomes_table.c.batch_id)
        .distinct()
        .where(token_outcomes_table.c.run_id == run_id)
        .where(token_outcomes_table.c.path == TerminalPath.BATCH_CONSUMED.value)
        .where(token_outcomes_table.c.outcome == TerminalOutcome.TRANSIENT.value)
        .where(
            ~select(batches_table.c.batch_id)
            .where(batches_table.c.batch_id == token_outcomes_table.c.batch_id)
            .where(batches_table.c.run_id == run_id)
            .where(batches_table.c.status == BatchStatus.COMPLETED.value)
            .exists()
        )
    )
    return [r.batch_id for r in self._ops.execute_fetchall(query)]
```

**Step 3: GREEN**

Run: `.venv/bin/python -m pytest tests/integration/test_adr_019_cross_table_invariants.py::TestI1aForkParentDeferred tests/integration/test_adr_019_cross_table_invariants.py::TestI1bBatchConsumedDeferred -v`

Expected: tests pass.

**Definition of Done:**
- [ ] `sweep_deferred_invariants_or_crash` method added
- [ ] Wired into `_execute_run` immediately after `_flush_and_write_sinks(...)` returns and before final progress / `PhaseCompleted`
- [ ] Wired into `_process_resumed_rows` immediately after resume `_flush_and_write_sinks(...)` returns
- [ ] Wired into the public `resume()` no-work terminalization branch before `_derive_resume_terminal_status_from_audit(...)` and `finalize_run(...)`, with sweep failures finalized through the failed ceremony rather than successful terminalization
- [ ] Concrete Orchestrator regression `test_valid_fork_coalesce_run_does_not_false_positive_after_sink_writes` proves a valid fork/coalesce-to-sink run does not false-positive because the sweep runs after sink outcomes are recorded
- [ ] Resume regression test proves a resumed run also invokes `sweep_deferred_invariants_or_crash(run_id)` after resume sink writes
- [ ] I1a and I1b integration tests pass
- [ ] No regression in other resume / orchestrator tests

---

### Task 4.4: Sweep-crash audit-trail durability — ordering + regression test

**Why this task exists:** When `DataFlowRepository.sweep_deferred_invariants_or_crash` raises `AuditIntegrityError`, the run must finalize as `RunStatus.FAILED` and the orphaned `token_outcomes` rows that triggered the crash must REMAIN in the audit DB. Per CLAUDE.md Auditability Standard (*"Every decision must be traceable to source data, configuration, and code version"*) and the attributability test (*"For any output, `explain(recorder, run_id, token_id)` must prove complete lineage back to source"*), an operator querying a sweep-crashed run must be able to inspect the preserved orphaned rows that triggered the failure. Durable error-message storage is not introduced by ADR-019; the exception message is re-raised to the caller, while the persisted run row records `FAILED` status.

The Phase 4 plan up to this point doesn't specify:
1. **Ordering**: where in the run-finalization sequence does the sweep run? It runs after `_flush_and_write_sinks(...)` has durably recorded sink outcomes, but before `_execute_run` returns to the public `run()` wrapper that finalizes the run as `COMPLETED`. The run is still `RUNNING` when the sweep fires.
2. **Audit-trail durability on crash**: the existing `Orchestrator` exception-handling path wraps exceptions from `_execute_run`, emits the failed ceremony, and calls `factory.run_lifecycle.finalize_run(run_id, status=RunStatus.FAILED)` (see `core.py:638`). The sweep's `AuditIntegrityError` must propagate through this same handler — NOT be caught locally and swallowed.
3. **Evidence preservation**: orphaned `token_outcomes` rows must NOT be deleted by the sweep. The sweep's job is to raise on detection, not to clean up. The orphaned rows are evidence of the bug that produced them — Tier 1 audit data, never deleted.
4. **Regression test**: there's no test that verifies the sweep-crash leaves a queryable audit trail.

**Files:**
- Verify (no edit): `src/elspeth/engine/orchestrator/core.py` exception handler around `_execute_run` / public `run()` failure ceremony (the existing path at `core.py:622-638` that calls `finalize_run(status=RunStatus.FAILED)` on uncaught exceptions). Confirm `AuditIntegrityError` is NOT caught earlier in `_execute_run` after the post-sink sweep call.
- Modify (only if the verification above fails): the exception-handler path so that `AuditIntegrityError` flows through to `finalize_run(FAILED)`.
- Test (RED-first): `tests/integration/test_adr_019_sweep_durability.py` (NEW)

**Step 1: Verify the existing exception path lets the sweep crash propagate to run-finalization**

```bash
grep -n "except\|finalize_run\|_execute_run\|_flush_and_write_sinks" src/elspeth/engine/orchestrator/core.py | head -60
rg -n "_RunFailedWithPartialResultError|original_error|with_traceback|from exc|from None" src/elspeth/engine/orchestrator/core.py
rg -n "except.*AuditIntegrityError|AuditIntegrityError" src/elspeth/engine/orchestrator/core.py
```

Walk the exception path from `DataFlowRepository.sweep_deferred_invariants_or_crash` outward:
- `DataFlowRepository.sweep_deferred_invariants_or_crash` (added in Task 4.3) raises `AuditIntegrityError`.
- Called from `_execute_run` immediately after `_flush_and_write_sinks(...)` returns (Task 4.3 edit). `_execute_run` MUST NOT catch and swallow `AuditIntegrityError`; it may wrap it in the existing partial-result failure wrapper as long as public `run()` still finalizes FAILED and re-raises.
- The Orchestrator's outer failure ceremony (search for `factory.run_lifecycle.finalize_run` calls) catches the unbounded exception path, calls `finalize_run(run_id, status=RunStatus.FAILED)` to durably record the crash, then re-raises so the CLI / API surfaces the failure. Note: `finalize_run` takes only `run_id` and `status` — there is no `error=` parameter and `Run` has no `error` field.

Document the exact wrapper semantics in the test comments:

- Internal `_execute_run` / `_process_resumed_rows` may raise
  `_RunFailedWithPartialResultError` with `exc.__cause__` set to the original
  `AuditIntegrityError` (`raise _RunFailedWithPartialResultError(...) from exc`).
  Any test that calls an internal method directly must assert that cause chain.
- Public `Orchestrator.run()` and resume entrypoints re-raise the original
  `AuditIntegrityError` after finalization and may suppress the wrapper cause
  (`from None`) to keep the operator-facing exception clean. Public-path tests
  must assert the caller sees `AuditIntegrityError`, the run row is finalized
  `FAILED`, and the evidence row remains queryable.

If any layer between the sweep and the outer handler catches `AuditIntegrityError` and swallows it, fix that catch to either re-raise or pass through. The whole point is that this exception class is NEVER recoverable per its `tier_1_error` decoration at `contracts/errors.py`.

**Step 2: Document the ordering contract**

Add this comment block immediately above the `factory.data_flow.sweep_deferred_invariants_or_crash` call site in `_execute_run`, just after `_flush_and_write_sinks(...)`:

```python
# ADR-019 Phase 4: deferred cross-table invariant sweep.
#
# AUDIT-TRAIL DURABILITY CONTRACT (CLAUDE.md Auditability Standard):
#
# 1. Run is in RunStatus.RUNNING when this sweep fires (the outer
#    finalize_run(COMPLETED) call has not yet executed).
# 2. If the sweep raises AuditIntegrityError, it propagates through _execute_run
#    to the Orchestrator failure ceremony, which calls
#    factory.run_lifecycle.finalize_run(run_id, status=RunStatus.FAILED).
#    `finalize_run` accepts only `run_id` and `status` — there is no
#    `error=` parameter and `Run` has no `error` field
#    (run_lifecycle_repository.py:814; contracts/audit.py:69-87).
#    Error-message durability is out of scope for ADR-019 — the run row
#    durably records FAILED status, and the preserved offending rows are the
#    queryable audit evidence. The exception is re-raised to the caller.
# 3. The orphaned token_outcomes rows that triggered the sweep crash
#    are NOT deleted. They remain as Tier 1 evidence: the operator's
#    explain(recorder, run_id, token_id) query returns the full lineage
#    including the orphaned parent / unflushed batch row that the sweep
#    detected. Deleting evidence on crash would violate CLAUDE.md
#    "no inference — if it's not recorded, it didn't happen" and erase
#    the auditor's ability to diagnose the bug.
# 4. Skipped on graceful shutdown because _flush_and_write_sinks raises
#    GracefulShutdownError before this post-sink call site. Resume completes
#    the missing children/batch-results.
factory.data_flow.sweep_deferred_invariants_or_crash(run_id)
```

**Step 3: Write the durability regression test (RED-first)**

Create: `tests/integration/test_adr_019_sweep_durability.py`

Required regression matrix (each row asserts propagation, `FAILED`
finalization, and evidence preservation):

| Test | Path | Required assertions |
| --- | --- | --- |
| `test_execute_i1a_sweep_crash_propagates_fails_and_preserves_orphan` | fresh run, I1a orphan parent | public `Orchestrator.run()` raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; planted `FORK_PARENT`/`EXPAND_PARENT` `token_outcomes` row remains queryable |
| `test_execute_i1b_sweep_crash_propagates_fails_and_preserves_orphan` | fresh run, I1b uncompleted batch | public `Orchestrator.run()` raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; planted `BATCH_CONSUMED` row and referenced non-completed batch remain queryable |
| `test_resume_i1a_sweep_crash_propagates_fails_and_preserves_orphan` | resume path, I1a orphan parent | public resume entrypoint raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; planted parent row remains queryable |
| `test_resume_i1b_sweep_crash_propagates_fails_and_preserves_orphan` | resume path, I1b uncompleted batch | public resume entrypoint raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; planted batch-consumed row remains queryable |
| `test_resume_noop_i1a_sweep_crash_propagates_fails_and_preserves_orphan` | no-work resume branch, I1a orphan parent | public resume entrypoint raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; planted parent row remains queryable; `_process_resumed_rows` is not called |
| `test_resume_noop_i1b_sweep_crash_propagates_fails_and_preserves_orphan` | no-work resume branch, I1b uncompleted batch | public resume entrypoint raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; planted batch-consumed row and referenced non-completed batch remain queryable; `_process_resumed_rows` is not called |
| `test_execute_i1c_realtime_crash_propagates_fails_and_preserves_witnesses` | fresh run, failsink producer passes wrong exact sink/artifact witness | public `Orchestrator.run()` raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; the completed failsink node_state and artifact rows written before `record_token_outcome` remain queryable |
| `test_execute_i3_realtime_crash_propagates_fails_and_preserves_failed_sink_state` | fresh run, discard producer violates I3 after primary sink state is written | public `Orchestrator.run()` raises `AuditIntegrityError`; run row is `RunStatus.FAILED`; the primary FAILED sink node_state remains queryable and no cleanup deletes discard evidence |

`test_sweep_skipped_on_graceful_shutdown` remains an additional shutdown-gate
test; it is not a substitute for any row-preservation matrix case.

The I1c/I3 durability rows must corrupt the producer boundary, not the invariant
method itself. For I1c, monkeypatch the failsink recording producer so the
failsink node_state/artifact are durably written but `record_token_outcome`
receives a sibling `sink_node_id` or sibling artifact id. For I3, monkeypatch
the discard producer so it reaches `record_token_outcome` with a contradictory
completed sink state or otherwise violates the real-time I3 condition after the
primary sink state has been durably completed as FAILED. Do not monkeypatch
`_validate_cross_table_invariants` to raise directly for these two rows; that
would skip the producer/evidence path the test is supposed to prove.

```python
"""ADR-019 Phase 4: sweep-crash audit-trail durability regression test.

Per CLAUDE.md Auditability Standard, when the deferred-invariant sweep
crashes mid-run, the Orchestrator's outer exception handler must call
finalize_run(FAILED) and the orphaned token_outcomes rows that triggered
the crash must remain as Tier 1 evidence. This test verifies the propagation
path through the ACTUAL outer handler — not a simulation of it.

	APPROACH: monkeypatch Orchestrator._initialize_database_phase so it calls the
	real initializer, then plants an orphan row before _execute_run starts. This
	gives the test a real run_id without writing from inside the later sweep
	closure (which would contend with the live Orchestrator transaction on
	SQLite). Separately monkeypatch
	DataFlowRepository.sweep_deferred_invariants_or_crash so it raises
	AuditIntegrityError when the post-sink hook fires. The outer handler at
	core.py:638 calls finalize_run(FAILED). The test asserts the run status AND
	that the pre-planted orphan token_outcomes rows survive the crash.

Normal pipelines cannot produce orphaned FORK_PARENT or BATCH_CONSUMED rows
by construction — the engine guarantees children/flush before advancing. The
only way to produce these in a test is via direct factory API surgery
(bypassing the engine), which the sweep tests in test_adr_019_cross_table_invariants.py
already cover. This file tests the propagation path, not the detection logic.

PREREQUISITES:
- Phases 1-3's ``record_token_outcome`` signature with ``outcome/path`` params.
- Use raw ``RecorderFactory`` + ``Orchestrator.run(config, ...)`` construction
  or a new ADR-019-compatible helper in this file. Do not reuse
  ``TestResumeLifecycle._setup_failed_run`` from
  ``tests/integration/pipeline/orchestrator/test_graceful_shutdown.py`` unless
  it has first been updated to record two-axis ``TerminalOutcome`` /
  ``TerminalPath`` rows; the current helper is an ADR-018 fixture and records
  ``RowOutcome.COMPLETED`` without ``path``.
- Phase 4's ``sweep_deferred_invariants_or_crash`` on DataFlowRepository
  (Task 4.3 Step 3a). RED until this method exists.

NOTE: These tests use a real Orchestrator run (not just landscape_factory)
so that the propagation through the actual outer exception handler at
core.py:638 is exercised. The monkeypatch injects the failure cleanly
without requiring a real orphan pipeline state.
"""

from __future__ import annotations

import pytest

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import BatchStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.factory import RecorderFactory

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _plant_orphan_fork_parent(factory: RecorderFactory, run_id: str, *, row_index: int = 999) -> str:
    """Plant a (TRANSIENT, FORK_PARENT) token with no children into an existing run.
    Returns the token_id. Called from the monkeypatched database-phase wrapper
    after the run row exists but before Orchestrator enters _execute_run, so the
    evidence exists before the sweep raises and without same-DB write-lock
    contention inside the sweep closure.

    NOTE: The patched sweep raises regardless after planting; detection logic is
    tested separately in test_adr_019_cross_table_invariants.py.
    """
    source = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="durability_evidence_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=source.node_id,
        row_index=row_index,
        data={"planted": True},
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=TerminalOutcome.TRANSIENT,
        path=TerminalPath.FORK_PARENT,
        fork_group_id="fg_durability_planted",
    )
    return token.token_id


def _plant_orphan_batch_consumed(factory: RecorderFactory, run_id: str, *, row_index: int = 998) -> str:
    """Plant a (TRANSIENT, BATCH_CONSUMED) token whose batch never completes.
    Called from the monkeypatched database-phase wrapper before _execute_run for
    the same durability reason as _plant_orphan_fork_parent.
    """
    source_node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="durability_evidence_source_i1b",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=source_node.node_id,
        row_index=row_index,
        data={"planted_i1b": True},
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    batch_id = "batch_durability_evidence_i1b"

    # Create the FK target but deliberately leave it non-COMPLETED. The I1b
    # violation is "batch exists but never completed", not "token_outcomes
    # violates its composite (batch_id, run_id) FK before the sweep can run".
    factory.execution.create_batch(
        run_id=run_id,
        aggregation_node_id=source_node.node_id,
        batch_id=batch_id,
    )
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=TerminalOutcome.TRANSIENT,
        path=TerminalPath.BATCH_CONSUMED,
        batch_id=batch_id,
    )
    return token.token_id


def _setup_adr019_failed_resume_run(
    db,
    payload_store,
    run_id: str,
    *,
    num_rows: int,
    processed_count: int,
):
    """ADR-019-compatible failed-run fixture for resume durability tests.

    Implement this as a local clone of
    ``TestResumeLifecycle._setup_failed_run`` from
    ``tests/integration/pipeline/orchestrator/test_graceful_shutdown.py``:
    keep its production ``build_linear_pipeline`` graph construction, manual
    runs/nodes/edges/rows/tokens SQL bootstrap, and checkpoint creation, but
    replace the stale ADR-018 terminal recording loop with:

        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=f"t{i}", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )

    The helper must not import the old one-axis enum or call the stale resume
    fixture. Verify with:
    ``rg -n "outcome=RowOutcome|TestResumeLifecycle\\(\\)\\._setup_failed_run" tests/integration/test_adr_019_sweep_durability.py``.
    """
    import json as json_mod
    from datetime import UTC, datetime

    from sqlalchemy import insert

    from elspeth.contracts.contract_records import ContractAuditRecord
    from elspeth.contracts.enums import Determinism, RoutingMode
    from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
    from elspeth.contracts.schema_contract import FieldContract, SchemaContract
    from elspeth.core.canonical import canonical_json
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.landscape.schema import (
        edges_table,
        nodes_table,
        rows_table,
        runs_table,
        tokens_table,
    )
    from elspeth.engine.orchestrator import prepare_for_run
    from tests.fixtures.base_classes import as_transform
    from tests.fixtures.pipeline import build_linear_pipeline
    from tests.fixtures.plugins import PassTransform

    now = datetime.now(UTC)
    prepare_for_run()
    runtime_val_manifest_json = canonical_json(build_runtime_val_manifest())

    source_data = [{"value": i} for i in range(num_rows)]
    transform = PassTransform()
    _, _, _, graph = build_linear_pipeline(source_data, transforms=[as_transform(transform)])

    source_nid = graph.get_source()
    assert source_nid is not None
    transform_id_map = graph.get_transform_id_map()
    sink_id_map = graph.get_sink_id_map()
    xform_nid = str(transform_id_map[0])
    sink_nid = str(next(iter(sink_id_map.values())))

    source_schema_json = json_mod.dumps({"properties": {"value": {"type": "integer"}}, "required": ["value"]})
    contract = SchemaContract(
        mode="FIXED",
        fields=(
            FieldContract(
                normalized_name="value",
                original_name="value",
                python_type=int,
                required=True,
                source="declared",
            ),
        ),
        locked=True,
    )
    audit_record = ContractAuditRecord.from_contract(contract)

    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.FAILED,
                source_schema_json=source_schema_json,
                schema_contract_json=audit_record.to_json(),
                schema_contract_hash=contract.version_hash(),
                runtime_val_manifest_json=runtime_val_manifest_json,
            )
        )

        for node_id, plugin_name, node_type in [
            (source_nid, "list_source", NodeType.SOURCE),
            (xform_nid, "passthrough", NodeType.TRANSFORM),
            (sink_nid, "collect_sink", NodeType.SINK),
        ]:
            conn.execute(
                insert(nodes_table).values(
                    node_id=node_id,
                    run_id=run_id,
                    plugin_name=plugin_name,
                    node_type=node_type,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

        for edge_id, from_node, to_node in [
            ("e1", source_nid, xform_nid),
            ("e2", xform_nid, sink_nid),
        ]:
            conn.execute(
                insert(edges_table).values(
                    edge_id=edge_id,
                    run_id=run_id,
                    from_node_id=from_node,
                    to_node_id=to_node,
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )

        for i in range(num_rows):
            row_data = {"value": i}
            ref = payload_store.store(json_mod.dumps(row_data).encode())
            conn.execute(
                insert(rows_table).values(
                    row_id=f"r{i}",
                    run_id=run_id,
                    source_node_id=source_nid,
                    row_index=i,
                    source_data_hash=f"h{i}",
                    source_data_ref=ref,
                    created_at=now,
                )
            )
            conn.execute(
                insert(tokens_table).values(
                    token_id=f"t{i}",
                    row_id=f"r{i}",
                    run_id=run_id,
                    created_at=now,
                )
            )

    factory = RecorderFactory(db)
    for i in range(processed_count):
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=f"t{i}", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="default",
        )

    if processed_count > 0:
        CheckpointManager(db).create_checkpoint(
            run_id=run_id,
            token_id=f"t{processed_count - 1}",
            node_id=xform_nid,
            sequence_number=processed_count - 1,
            graph=graph,
        )

    return graph


class TestSweepCrashAuditTrailDurability:
    def test_orphaned_fork_parent_crashes_run_with_durable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagation path: sweep AuditIntegrityError reaches outer handler,
        run finalizes as FAILED, orphaned token_outcomes row persists.

        Strategy: monkeypatch Orchestrator._initialize_database_phase to plant
        a durable orphan row before _execute_run starts, then monkeypatch
        DataFlowRepository.sweep_deferred_invariants_or_crash to raise
        AuditIntegrityError("ADR-019 I1a violation: ..."). Run a minimal 1-row
        pipeline. The patched sweep fires after sink writes, the outer handler
        at core.py:638 calls finalize_run(FAILED), and the test asserts:
        (a) Orchestrator.run() raises AuditIntegrityError,
        (b) run status is RunStatus.FAILED,
        (c) the pre-planted orphan token_outcomes row persists as Tier 1 evidence.

        RED until Task 4.3 adds sweep_deferred_invariants_or_crash to DataFlowRepository.
        Fails with AttributeError on monkeypatch.setattr until Phase 4 lands.

        PREREQUISITE: use an ADR-019-compatible direct pipeline construction
        pattern. Do not reuse stale RowOutcome-based resume helpers.
        """
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository

        captured_run_ids: list[str] = []
        planted_token_ids: list[str] = []

        # Canonical integration-test pattern from
        # tests/integration/audit/test_source_boundary_orchestrator.py:19-35.
        # LandscapeDB.in_memory() + MockPayloadStore + build_linear_pipeline
        # is the minimal viable setup that exercises Orchestrator.run() end-to-end
        # including the outer except Exception handler at core.py:1684 that calls
        # _emit_failed_ceremony → finalize_run(FAILED) at core.py:638, then re-raises.
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from tests.fixtures.base_classes import as_sink, as_source
        from tests.fixtures.pipeline import build_linear_pipeline
        from tests.fixtures.stores import MockPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()

        original_init = Orchestrator._initialize_database_phase

        def _init_and_plant_orphan(
            self: Orchestrator,
            config,
            payload_store,
            secret_resolutions,
            *,
            run_id=None,
        ):
            factory, run = original_init(
                self,
                config,
                payload_store,
                secret_resolutions,
                run_id=run_id,
            )
            planted_token_ids.append(_plant_orphan_fork_parent(factory, run.run_id))
            return factory, run

        def _patched_sweep(self: DataFlowRepository, run_id: str) -> None:
            captured_run_ids.append(run_id)
            raise AuditIntegrityError(
                f"ADR-019 I1a violation: 1 fork/expand parent token(s) have "
                f"no children at run-end. [monkeypatched for propagation test]"
            )

        monkeypatch.setattr(Orchestrator, "_initialize_database_phase", _init_and_plant_orphan)
        # RED: AttributeError here until Task 4.3 adds the method.
        monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _patched_sweep)

        # Single-row linear pipeline — the simplest shape that reaches the
        # post-sink _execute_run sweep. No transforms needed; source → sink
        # with one row is sufficient.
        source, _tx_list, sinks, graph = build_linear_pipeline(
            [{"value": 1}], transforms=[]
        )
        sink = sinks["default"]

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        # The patched sweep fires after sink writes and raises AuditIntegrityError.
        # The outer handler catches it,
        # calls _emit_failed_ceremony → finalize_run(FAILED) at core.py:638,
        # then re-raises ("raise  # CRITICAL: Always re-raise").
        # AuditIntegrityError propagates to the caller — not swallowed.
        with pytest.raises(AuditIntegrityError, match="I1a"):
            Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

        assert len(captured_run_ids) == 1, (
            "Patched sweep must have been called exactly once — confirms the sweep "
            "wiring in _execute_run after sink writes reached the patched method."
        )
        assert len(planted_token_ids) == 1, (
            "The patched database phase must plant the orphan before _execute_run "
            "starts; planting inside the sweep closure risks same-DB write-lock contention."
        )
        run_id = captured_run_ids[0]

        # Verify the outer handler recorded RunStatus.FAILED before re-raising.
        # factory.run_lifecycle.get_run is at run_lifecycle_repository.py:227.
        factory = RecorderFactory(db)
        run_row = factory.run_lifecycle.get_run(run_id)
        assert run_row is not None, (
            "finalize_run(FAILED) must have written the run row before re-raising."
        )
        assert run_row.status == RunStatus.FAILED, (
            f"Expected RunStatus.FAILED after sweep crash, got {run_row.status!r}. "
            f"The outer handler at core.py:638 must call finalize_run(FAILED) "
            f"before re-raising AuditIntegrityError."
        )

        # The orphan was planted before the sweep raised. The post-crash query
        # must still see it; otherwise the outer failure path rolled back or
        # erased Tier 1 evidence.
        token_id = planted_token_ids[0]

        # data_flow_repository.py:882 — returns the most recent outcome for a token.
        orphan_outcome = factory.data_flow.get_token_outcome(token_id)
        assert orphan_outcome is not None, (
            "The orphan token_outcomes row must persist after the sweep crash. "
            "Tier 1 evidence is never deleted — CLAUDE.md Auditability Standard."
        )
        assert orphan_outcome.outcome == TerminalOutcome.TRANSIENT
        assert orphan_outcome.path == TerminalPath.FORK_PARENT
        assert orphan_outcome.fork_group_id == "fg_durability_planted"

    def test_orphaned_batch_consumed_crashes_run_with_durable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagation path: I1b-shaped AuditIntegrityError reaches outer handler.

        Symmetric to the I1a durability test. Plant the orphan in the patched
        database phase before _execute_run starts; the patched sweep raises
        specifically with "I1b" to verify the propagation path is labelled for
        the right invariant.

        Assertions mirror test_orphaned_fork_parent_crashes_run_with_durable_error:
        (a) Orchestrator.run() raises AuditIntegrityError matching "I1b",
        (b) run status is RunStatus.FAILED,
        (c) the pre-planted BATCH_CONSUMED token_outcomes row persists.

        RED until Task 4.3 adds sweep_deferred_invariants_or_crash to DataFlowRepository.
        """
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository

        captured_run_ids: list[str] = []
        planted_token_ids: list[str] = []

        # Same construction pattern as test_orphaned_fork_parent_crashes_run_with_durable_error.
        # Canonical source: tests/integration/audit/test_source_boundary_orchestrator.py:19-35.
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from tests.fixtures.base_classes import as_sink, as_source
        from tests.fixtures.pipeline import build_linear_pipeline
        from tests.fixtures.stores import MockPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()

        original_init = Orchestrator._initialize_database_phase

        def _init_and_plant_orphan(
            self: Orchestrator,
            config,
            payload_store,
            secret_resolutions,
            *,
            run_id=None,
        ):
            factory, run = original_init(
                self,
                config,
                payload_store,
                secret_resolutions,
                run_id=run_id,
            )
            planted_token_ids.append(_plant_orphan_batch_consumed(factory, run.run_id))
            return factory, run

        def _patched_sweep(self: DataFlowRepository, run_id: str) -> None:
            captured_run_ids.append(run_id)
            raise AuditIntegrityError(
                f"ADR-019 I1b violation: 1 batch_id(s) had BATCH_CONSUMED tokens "
                f"but the batch never reached BatchStatus.COMPLETED. "
                f"[monkeypatched for propagation test]"
            )

        monkeypatch.setattr(Orchestrator, "_initialize_database_phase", _init_and_plant_orphan)
        # RED: AttributeError here until Task 4.3 adds the method.
        monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _patched_sweep)

        source, _tx_list, sinks, graph = build_linear_pipeline(
            [{"value": 1}], transforms=[]
        )
        sink = sinks["default"]

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        # outer handler at core.py:1684 → _emit_failed_ceremony → finalize_run(FAILED)
        # at core.py:638 → re-raise at core.py:1711.
        with pytest.raises(AuditIntegrityError, match="I1b"):
            Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

        # Verify the outer handler recorded RunStatus.FAILED before re-raising.
        # factory.run_lifecycle.get_run is at run_lifecycle_repository.py:227.
        factory = RecorderFactory(db)
        assert len(captured_run_ids) == 1, (
            "Patched sweep must have been called exactly once — confirms the sweep "
            "wiring in _execute_run after sink writes reached the patched method."
        )
        assert len(planted_token_ids) == 1, (
            "The patched database phase must plant the BATCH_CONSUMED orphan "
            "before _execute_run starts; post-crash planting would not test "
            "rollback durability."
        )
        run_id = captured_run_ids[0]

        run_row = factory.run_lifecycle.get_run(run_id)
        assert run_row is not None, (
            "finalize_run(FAILED) must have written the run row before re-raising."
        )
        assert run_row.status == RunStatus.FAILED, (
            f"Expected RunStatus.FAILED after I1b sweep crash, got {run_row.status!r}. "
            f"The outer handler at core.py:638 must call finalize_run(FAILED)."
        )

        # The orphan was planted before the sweep raised. The post-crash query
        # must still see it; otherwise the outer failure path rolled back or
        # erased Tier 1 evidence.
        orphan_outcome = factory.data_flow.get_token_outcome(planted_token_ids[0])
        assert orphan_outcome is not None, (
            "The BATCH_CONSUMED orphan token_outcomes row must persist after the "
            "sweep crash — Tier 1 evidence is never deleted."
        )
        assert orphan_outcome.outcome == TerminalOutcome.TRANSIENT
        assert orphan_outcome.path == TerminalPath.BATCH_CONSUMED
        assert orphan_outcome.batch_id == "batch_durability_evidence_i1b"
        orphan_batch = factory.execution.get_batch("batch_durability_evidence_i1b")
        assert orphan_batch is not None
        assert orphan_batch.status != BatchStatus.COMPLETED

    @pytest.mark.parametrize(
        ("label", "plant_orphan"),
        [
            ("I1a", _plant_orphan_fork_parent),
            ("I1b", _plant_orphan_batch_consumed),
        ],
        ids=[
            "resume-i1a-orphan-parent",
            "resume-i1b-uncompleted-batch",
        ],
    )
    def test_resume_sweep_crash_propagates_fails_and_preserves_orphan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        plant_orphan,
    ) -> None:
        """Resume sibling path: _process_resumed_rows runs the sweep after sink flush.

        This is the resume half of the durability matrix. It uses the public
        resume() entrypoint, plants Tier 1 evidence before resume starts, then
        monkeypatches the deferred sweep to raise the labelled AuditIntegrityError.
        Assertions:
        (a) resume() raises AuditIntegrityError matching I1a/I1b,
        (b) the run is finalized as RunStatus.FAILED,
        (c) the planted token_outcomes row remains queryable,
        (d) resume sink flush happens before the sweep call.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.null_source import NullSource
        from elspeth.plugins.transforms.passthrough import PassThrough
        from tests.fixtures.base_classes import as_sink, as_source, as_transform
        from tests.fixtures.plugins import CollectSink
        from tests.fixtures.stores import MockPayloadStore

        run_id = f"adr019-resume-sweep-{label.lower()}"
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(
            CheckpointSettings(enabled=True, frequency="every_row")
        )

        # Local ADR-019-compatible helper. It creates a failed run with
        # unprocessed rows and builds the graph through
        # ExecutionGraph.from_plugin_instances(), but records processed rows with
        # TerminalOutcome/TerminalPath, not stale RowOutcome.
        graph = _setup_adr019_failed_resume_run(
            db,
            payload_store,
            run_id,
            num_rows=4,
            processed_count=2,
        )
        factory = RecorderFactory(db)
        planted_token_id = plant_orphan(factory, run_id)

        recovery = RecoveryManager(db, checkpoint_mgr)
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        pass_through = PassThrough({"schema": {"mode": "observed"}})
        pass_through.on_success = "default"
        pass_through.on_error = "discard"
        null_source = NullSource({})
        null_source.on_success = "default"
        resume_sink = CollectSink()
        resume_config = PipelineConfig(
            source=as_source(null_source),
            transforms=[as_transform(pass_through)],
            sinks={"default": as_sink(resume_sink)},
        )

        call_order: list[str] = []
        sweep_calls: list[str] = []
        original_flush = Orchestrator._flush_and_write_sinks

        def _flush_spy(self: Orchestrator, factory, run_id_arg, loop_ctx, *args, **kwargs):
            call_order.append("flush")
            return original_flush(self, factory, run_id_arg, loop_ctx, *args, **kwargs)

        def _sweep_raises(self: DataFlowRepository, run_id_arg: str) -> None:
            call_order.append("sweep")
            sweep_calls.append(run_id_arg)
            raise AuditIntegrityError(
                f"ADR-019 {label} violation: monkeypatched resume durability test"
            )

        monkeypatch.setattr(Orchestrator, "_flush_and_write_sinks", _flush_spy)
        # RED: AttributeError here until Task 4.3 adds the method.
        monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _sweep_raises)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        with pytest.raises(AuditIntegrityError, match=label):
            orchestrator.resume(
                resume_point=resume_point,
                config=resume_config,
                graph=graph,
                payload_store=payload_store,
            )

        assert sweep_calls == [run_id]
        assert call_order[-2:] == ["flush", "sweep"]

        run_row = factory.run_lifecycle.get_run(run_id)
        assert run_row is not None
        assert run_row.status == RunStatus.FAILED

        orphan_outcome = factory.data_flow.get_token_outcome(planted_token_id)
        assert orphan_outcome is not None
        assert orphan_outcome.outcome == TerminalOutcome.TRANSIENT
        if label == "I1a":
            assert orphan_outcome.path == TerminalPath.FORK_PARENT
            assert orphan_outcome.fork_group_id == "fg_durability_planted"
        else:
            assert orphan_outcome.path == TerminalPath.BATCH_CONSUMED
            assert orphan_outcome.batch_id == "batch_durability_evidence_i1b"
            orphan_batch = factory.execution.get_batch("batch_durability_evidence_i1b")
            assert orphan_batch is not None
            assert orphan_batch.status != BatchStatus.COMPLETED

    @pytest.mark.parametrize(
        ("label", "plant_orphan"),
        [
            ("I1a", _plant_orphan_fork_parent),
            ("I1b", _plant_orphan_batch_consumed),
        ],
        ids=[
            "resume-noop-i1a-orphan-parent",
            "resume-noop-i1b-uncompleted-batch",
        ],
    )
    def test_resume_noop_sweep_crash_propagates_fails_and_preserves_orphan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        plant_orphan,
    ) -> None:
        """No-work resume branch: sweep runs even when _process_resumed_rows is skipped.

        This covers the public resume shortcut where all rows were already
        processed and no restored aggregation/coalesce state exists. Without the
        explicit Phase 4 sweep in that branch, the run can derive/finalize a
        successful terminal status from stale audit rows while orphaned I1a/I1b
        evidence remains in the database.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.null_source import NullSource
        from elspeth.plugins.transforms.passthrough import PassThrough
        from tests.fixtures.base_classes import as_sink, as_source, as_transform
        from tests.fixtures.plugins import CollectSink
        from tests.fixtures.stores import MockPayloadStore

        run_id = f"adr019-resume-noop-sweep-{label.lower()}"
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(
            CheckpointSettings(enabled=True, frequency="every_row")
        )

        graph = _setup_adr019_failed_resume_run(
            db,
            payload_store,
            run_id,
            num_rows=2,
            processed_count=2,
        )
        factory = RecorderFactory(db)
        # Keep evidence outside the resume query's "after last checkpoint"
        # range so this test stays on the no-work terminalization branch.
        planted_token_id = plant_orphan(factory, run_id, row_index=-1)

        resume_point = RecoveryManager(db, checkpoint_mgr).get_resume_point(run_id, graph)
        assert resume_point is not None

        pass_through = PassThrough({"schema": {"mode": "observed"}})
        pass_through.on_success = "default"
        pass_through.on_error = "discard"
        null_source = NullSource({})
        null_source.on_success = "default"
        resume_sink = CollectSink()
        resume_config = PipelineConfig(
            source=as_source(null_source),
            transforms=[as_transform(pass_through)],
            sinks={"default": as_sink(resume_sink)},
        )

        process_calls: list[str] = []
        sweep_calls: list[str] = []

        def _fail_if_processed(self: Orchestrator, *args, **kwargs):
            process_calls.append("process")
            raise AssertionError("no-work resume branch must not call _process_resumed_rows")

        def _sweep_raises(self: DataFlowRepository, run_id_arg: str) -> None:
            sweep_calls.append(run_id_arg)
            raise AuditIntegrityError(
                f"ADR-019 {label} violation: monkeypatched no-op resume durability test"
            )

        monkeypatch.setattr(Orchestrator, "_process_resumed_rows", _fail_if_processed)
        monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _sweep_raises)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        with pytest.raises(AuditIntegrityError, match=label):
            orchestrator.resume(
                resume_point=resume_point,
                config=resume_config,
                graph=graph,
                payload_store=payload_store,
            )

        assert process_calls == []
        assert sweep_calls == [run_id]

        run_row = factory.run_lifecycle.get_run(run_id)
        assert run_row is not None
        assert run_row.status == RunStatus.FAILED

        orphan_outcome = factory.data_flow.get_token_outcome(planted_token_id)
        assert orphan_outcome is not None
        assert orphan_outcome.outcome == TerminalOutcome.TRANSIENT
        if label == "I1a":
            assert orphan_outcome.path == TerminalPath.FORK_PARENT
            assert orphan_outcome.fork_group_id == "fg_durability_planted"
        else:
            assert orphan_outcome.path == TerminalPath.BATCH_CONSUMED
            assert orphan_outcome.batch_id == "batch_durability_evidence_i1b"
            orphan_batch = factory.execution.get_batch("batch_durability_evidence_i1b")
            assert orphan_batch is not None
            assert orphan_batch.status != BatchStatus.COMPLETED

    def test_sweep_skipped_on_graceful_shutdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Graceful shutdown: sweep must NOT run when sink flushing raises shutdown.

        This is a real Orchestrator.run() regression test. It monkeypatches the
        post-sink sweep to fail if called, then monkeypatches
        Orchestrator._flush_and_write_sinks to raise GracefulShutdownError at the
        same control-flow point a real shutdown uses. The assertion is direct:
        Orchestrator.run raises GracefulShutdownError and the sweep spy was never
        invoked.
        """
        from elspeth.contracts.errors import GracefulShutdownError
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from tests.fixtures.base_classes import as_sink, as_source
        from tests.fixtures.pipeline import build_linear_pipeline
        from tests.fixtures.stores import MockPayloadStore

        sweep_calls: list[str] = []

        def _fail_if_swept(self: DataFlowRepository, run_id: str) -> None:
            sweep_calls.append(run_id)
            raise AssertionError(
                "Deferred ADR-019 sweep ran after GracefulShutdownError; "
                "shutdown must exit before the post-sink sweep call site."
            )

        def _raise_shutdown_from_flush(self: Orchestrator, factory, run_id, loop_ctx, *args, **kwargs) -> None:
            raise GracefulShutdownError(
                rows_processed=loop_ctx.counters.rows_processed,
                run_id=run_id,
                rows_succeeded=loop_ctx.counters.rows_succeeded,
                rows_failed=loop_ctx.counters.rows_failed,
                rows_quarantined=loop_ctx.counters.rows_quarantined,
                rows_routed_success=loop_ctx.counters.rows_routed_success,
                rows_routed_failure=loop_ctx.counters.rows_routed_failure,
            )

        monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _fail_if_swept)
        monkeypatch.setattr(Orchestrator, "_flush_and_write_sinks", _raise_shutdown_from_flush)

        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source, _tx_list, sinks, graph = build_linear_pipeline(
            [{"value": 1}], transforms=[]
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sinks["default"])},
        )

        with pytest.raises(GracefulShutdownError):
            Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

        assert sweep_calls == [], (
            "sweep_deferred_invariants_or_crash must not be called after "
            "_flush_and_write_sinks raises GracefulShutdownError."
        )
```

**Step 4: Run RED**

```bash
.venv/bin/python -m pytest tests/integration/test_adr_019_sweep_durability.py -v
```

Expected RED states:
- Before Task 4.3: `monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", ...)` fails with `AttributeError` — the method does not exist yet. This is the correct RED signal: the test cannot be faked green because `monkeypatch.setattr` asserts the attribute exists before overwriting it.
- After Task 4.3 (method exists but only the fresh-run hook is wired): the parametrized resume cases fail because `sweep_calls == []` or the flush/sweep ordering assertion fails.
- After Task 4.3 (fresh/resume row-processing hooks wired but no-op resume omitted): the no-op resume cases fail because `_process_resumed_rows` is not called and `sweep_calls == []`.
- GREEN: the durability matrix passes after the full propagation path is wired: fresh I1a, fresh I1b, resume I1a, resume I1b, no-op resume I1a, no-op resume I1b, I1c real-time producer crash, I3 real-time producer crash, and graceful-shutdown skip.

**Step 5: Final propagation audit**

Before committing Phase 4, run:

```bash
# Confirm no swallowed AuditIntegrityError in the orchestrator path:
grep -A 3 "except.*AuditIntegrityError" src/elspeth/engine/orchestrator/core.py
# Expected: every match either re-raises or routes through finalize_run(FAILED).

# Confirm the sweep's exception class is tier_1_error decorated (cannot be
# accidentally caught by a generic ``except Exception``):
grep -B 2 "class AuditIntegrityError" src/elspeth/contracts/errors.py
# Expected: @tier_1_error(...) decorator immediately above.
```

**Definition of Done:**
- [ ] Verified the exception propagation path from `DataFlowRepository.sweep_deferred_invariants_or_crash` to `factory.run_lifecycle.finalize_run(run_id, status=RunStatus.FAILED)` — no intermediate handler swallows `AuditIntegrityError`
- [ ] `_RunFailedWithPartialResultError.__cause__` semantics documented and verified for any internal-path assertion
- [ ] Audit-trail durability comment block added at the sweep call site
- [ ] Durability matrix tests pass: fresh I1a, fresh I1b, resume I1a, resume I1b, no-op resume I1a, no-op resume I1b, I1c real-time producer crash, and I3 real-time producer crash; each asserts propagation, `FAILED` finalization, and evidence preservation
- [ ] Graceful-shutdown skip regression passes and documents that the sweep is intentionally not run when `_flush_and_write_sinks(...)` raises `GracefulShutdownError`
- [ ] Orphaned `token_outcomes` rows verified to PERSIST after sweep crash (Tier 1 evidence preservation)
- [ ] No regression in graceful-shutdown handling

---

### Task 4.5: Phase 4 commit

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest \
    tests/integration/test_adr_019_cross_table_invariants.py \
    tests/integration/test_adr_019_sweep_durability.py \
    tests/unit/core/landscape/test_data_flow_repository.py \
    -v
.venv/bin/python -m mypy \
    src/elspeth/core/landscape/data_flow_repository.py \
    src/elspeth/engine/orchestrator/core.py
.venv/bin/python -m ruff check \
    src/elspeth/core/landscape/data_flow_repository.py \
    src/elspeth/engine/orchestrator/core.py \
    tests/integration/test_adr_019_cross_table_invariants.py \
    tests/integration/test_adr_019_sweep_durability.py \
    tests/unit/core/landscape/test_data_flow_repository.py
.venv/bin/python -m ruff format --check \
    src/elspeth/core/landscape/data_flow_repository.py \
    src/elspeth/engine/orchestrator/core.py \
    tests/integration/test_adr_019_cross_table_invariants.py \
    tests/integration/test_adr_019_sweep_durability.py \
    tests/unit/core/landscape/test_data_flow_repository.py
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check \
    --root src/elspeth \
    --allowlist config/cicd/adr019_symbol_inventory
```

Expected: all focused Phase 4 invariant and durability tests pass. The full
`pytest tests/ -q --timeout=120` gate remains owned by Phase 5 after the
repo-wide schema-dependent/assertion-only test triage.

**If any pre-existing test fails because the engine produced an orphaned parent or unflushed batch under prior code paths**, it is surfacing a latent bug. Per CLAUDE.md no-legacy-code policy: fix the underlying engine code, do NOT relax the invariant. Surface to user if root cause is unclear.

**Step 2: Commit**

```bash
git add src/elspeth/core/landscape/data_flow_repository.py \
        src/elspeth/engine/orchestrator/core.py \
        tests/integration/test_adr_019_cross_table_invariants.py \
        tests/integration/test_adr_019_sweep_durability.py \
        tests/unit/core/landscape/test_data_flow_repository.py

git commit -m "$(cat <<'EOF'
feat(adr-019): phase 4 — cross-table invariants I1a, I1b, I1c, I3

ADR-019 Stage 2/3 Phase 4 of 5 (see docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md).

Four NEW Tier 1 cross-table invariants per ADR-019 § Cross-check invariants:

Real-time at recording (data_flow_repository.py::_validate_cross_table_invariants):
- I1c (sink-fallback-paired): (TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires
  a producer-declared exact failsink `sink_node_id` witness, a paired
  NodeStateStatus.COMPLETED node_state for that sink node, plus registered
  artifact. Verified at write time per ADR-019 line 256.
- I3 (discard-no-failsink): (FAILURE, SINK_DISCARDED) requires
  sink_name='__discard__' and no paired sink-completion node_state. Verified
  at write time per ADR-019 line 261.

Deferred / end-of-run sweep (orchestrator/core.py::_execute_run and
_process_resumed_rows call `factory.data_flow.sweep_deferred_invariants_or_crash(run_id)`
immediately after `_flush_and_write_sinks(...)` returns; the public resume
no-work terminalization branch calls it before deriving/finalizing terminal
status):
- I1a (lineage-paired): every fork/expand parent must have ≥1 child token by
  end of run. Children land later than parents, so write-time check is impossible.
- I1b (aggregate-paired): every BATCH_CONSUMED token's consuming batch must have
  reached BatchStatus.COMPLETED by end of run. Check is against the batches row,
  not a paired token_outcomes row (result tokens don't carry batch_id).

Two new repository helpers on DataFlowRepository host the SQL:
find_orphaned_transient_parents(run_id) and find_orphaned_batch_consumptions(run_id).
Keeps the SQL surface centralized in the repository layer per the project's
landscape-access conventions.

These invariants did not exist under ADR-018; they surface today's silent
audit-DB inconsistencies as Tier 1 errors. If a pre-existing engine path
violates the invariants, this commit's tests catch it as a fixable bug
rather than an invariant relaxation per CLAUDE.md no-legacy-code policy.

Plus a sweep-crash audit-trail durability contract (Task 4.4):
- AuditIntegrityError from the sweep propagates through Orchestrator's
  outer exception handler to factory.run_lifecycle.finalize_run(
  run_id, status=RunStatus.FAILED). The run row durably records FAILED.
  `Run` has no `error` field and `finalize_run` has no `error=` param;
  error-message durability is out of scope for ADR-019.
- Orphaned token_outcomes rows that triggered the sweep crash REMAIN
  in the DB as Tier 1 evidence for operator explain() queries.
- Shutdown-gated: graceful shutdown skips the sweep because
  `_flush_and_write_sinks(...)` raises before the post-sink sweep site,
  preserving the resume contract.
- Durability matrix tests in tests/integration/test_adr_019_sweep_durability.py
  pin fresh/resume/no-op-resume I1a and I1b crash paths, I1c/I3 real-time
  producer crash paths, evidence preservation, and FAILED finalization. A
  separate shutdown-skip test records the graceful-shutdown behaviour.

Refs: elspeth-edb60744f0 (Stage 3 ticket — producer + accumulator)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md § Cross-check invariants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Definition of Done:**
- [ ] All four invariants implemented (two real-time, two deferred)
- [ ] Cross-table invariant suite and durability matrix pass, including no-op resume, I1c, and I3 durability rows
- [ ] ADR-019 AST symbol inventory gate passes with `--allowlist config/cicd/adr019_symbol_inventory`
- [ ] No regression in other tests, OR latent bugs surfaced and fixed (not relaxed)
- [ ] Phase 4 commit landed
