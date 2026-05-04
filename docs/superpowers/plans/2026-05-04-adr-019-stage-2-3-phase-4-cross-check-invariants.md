# ADR-019 Stage 2/3 — Phase 4: Cross-Table Invariants (I1a, I1b, I1c, I3)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phase 4 adds NEW Tier 1 invariants that did not exist under ADR-018. They are additive and don't break the build, but they may surface latent bugs in the engine that the old invariant set was not catching. Phase 5 follows in the same PR.

**Goal:** Implement the four cross-table invariants from ADR-019 § "Cross-check invariants" (lines 237-269). Two are real-time at recording (I1c, I3); two are deferred — verified at end of run (I1a, I1b). The mechanism choice for deferred invariants is **end-of-run sweep in `Orchestrator._finalize_source_iteration`** per overview decision D3.

**Hook location verified against current HEAD (post-Stage-1 commit `60d30551`).** The end-of-source hook is `_finalize_source_iteration` at `src/elspeth/engine/orchestrator/core.py:2511`. (An earlier draft of this plan referenced `_post_source_iteration_work` from the ADR text — that symbol does not exist; the ADR's prose used the conceptual name. The actual method runs end-of-source aggregation flushes, coalesce flushes, and deferred field-resolution recording. The deferred-invariant sweep slots in AFTER the flushes complete (so I1b can observe batches reaching `BatchStatus.COMPLETED`) and BEFORE the deferred field-resolution recording.)

**Shutdown semantic.** `_finalize_source_iteration` takes an `interrupted_by_shutdown: bool` keyword. On graceful shutdown the existing code SKIPS aggregation flush and coalesce flush (lines 2554, 2584) because shutdown is resumable — buffered state is intentionally preserved. The deferred-invariant sweep MUST also be gated on `not interrupted_by_shutdown` for the same reason: a shutdown can legitimately leave fork-parents without children and BATCH_CONSUMED tokens without flush-results, because resume will produce them. Running the sweep on shutdown would crash the engine on every graceful stop.

**Files touched in this phase:**

- Modify: `src/elspeth/core/landscape/data_flow_repository.py` (add I1c and I3 checks to `record_token_outcome`)
- Modify: `src/elspeth/engine/orchestrator/core.py:2511-2603` (add deferred-invariant sweep to `_finalize_source_iteration`, gated on `not interrupted_by_shutdown`)
- Test (RED-first): `tests/integration/test_adr_019_cross_table_invariants.py` (NEW) — exercises each invariant
- Test (unit): `tests/unit/core/landscape/test_data_flow_repository.py` — extend with I1c and I3 unit tests

**Background reading:** ADR-019 lines 237-269 (the four invariants and their semantics). The existing single-row guards in `_validate_outcome_fields` (Phase 1's `_REQUIRED_FIELDS_BY_PAIR`) are necessary but not sufficient; the cross-table invariants verify *structural consistency between `token_outcomes`, `node_states`, and `artifacts`*.

---

## The four invariants (semantics summary)

### I1a (lineage-paired) — DEFERRED

`(TRANSIENT, FORK_PARENT)` and `(TRANSIENT, EXPAND_PARENT)` require **at least one child `token_outcomes` row with `parent_token_id == this.token_id`**. The child must complete after the parent, so this cannot be checked at parent-record time. Verified at end of run.

### I1b (aggregate-paired) — DEFERRED

`(TRANSIENT, BATCH_CONSUMED)` requires the consuming batch's batch-result token to be recorded at flush time. Same deferral reason — flush happens after consumption. Verified at end of run.

### I1c (sink-fallback-paired) — REAL-TIME

`(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` requires:
1. A paired `NodeStateStatus.COMPLETED` `node_state` row exists for the same `token_id` at the failsink node, AND
2. That `node_state` has a registered `artifacts` row.

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

**Step 1: Write the four failing tests**

```python
"""ADR-019 § Cross-check invariants: structural consistency between
token_outcomes, node_states, and artifacts.

These invariants did NOT exist under ADR-018. They surface today's silent
audit-DB inconsistencies as Tier 1 errors. Each test simulates a corrupted
audit-DB state (the kind of state a faulty plugin could produce) and
verifies the recorder/orchestrator crashes with AuditIntegrityError.
"""

import pytest

# AuditIntegrityError lives in elspeth.contracts.errors (NOT contracts.audit —
# audit.py holds the dataclasses; errors.py holds the exception hierarchy).
# Verified against existing imports in src/elspeth/core/landscape/data_flow_repository.py:33
# and src/elspeth/core/landscape/model_loaders.py:50.
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError


class TestI1cFailsinkPaired:
    """I1c: (TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires a paired failsink
    node_state COMPLETED + registered artifact for the same token."""

    def test_failsink_pair_present_passes(self, audit_repo, ...) -> None:
        """Happy path: failsink node_state exists, artifact registered."""
        # ... arrange the failsink completion + artifact ...
        outcome_id = audit_repo.record_token_outcome(
            ref=token_ref,
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            sink_name="failsink",
            error_hash="abcd1234abcd1234",
        )
        assert outcome_id is not None

    def test_failsink_node_state_missing_crashes(self, audit_repo, ...) -> None:
        """I1c violation: no failsink COMPLETED node_state — crash."""
        # Arrange: register the artifact, but skip the node_state.
        with pytest.raises(AuditIntegrityError, match="failsink"):
            audit_repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                error_hash="abcd1234abcd1234",
            )

    def test_failsink_artifact_missing_crashes(self, audit_repo, ...) -> None:
        """I1c violation: failsink node_state exists, but no artifact — crash."""
        with pytest.raises(AuditIntegrityError, match="artifact"):
            audit_repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                error_hash="abcd1234abcd1234",
            )


class TestI3DiscardNoFailsink:
    """I3: (FAILURE, SINK_DISCARDED) requires sink_name=__discard__ and
    NO paired failsink node_state for the same token."""

    def test_discard_with_failsink_pair_crashes(self, audit_repo, ...) -> None:
        """I3 violation: discard recorded but a failsink node_state exists for
        the same token — invariant says these are mutually exclusive."""
        with pytest.raises(AuditIntegrityError, match="discard"):
            audit_repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name="__discard__",
                error_hash="abcd1234abcd1234",
            )

    def test_discard_wrong_sink_name_crashes(self, audit_repo, ...) -> None:
        """I3 sub-invariant: SINK_DISCARDED requires sink_name='__discard__'."""
        with pytest.raises(AuditIntegrityError, match="discard"):
            audit_repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name="other_sink_name",  # not __discard__
                error_hash="abcd1234abcd1234",
            )


class TestI1aForkParentDeferred:
    """I1a (deferred): (TRANSIENT, FORK_PARENT) at end of run requires ≥1 child
    token_outcomes row referencing this parent's token_id. End-of-run sweep
    in Orchestrator._finalize_source_iteration catches violations."""

    def test_fork_parent_with_child_passes(self, ...) -> None:
        """Happy path: fork parent has a child token recorded."""
        ...

    def test_fork_parent_orphan_crashes_at_end_of_run(self, ...) -> None:
        """I1a violation: fork parent with no children — sweep catches at end."""
        with pytest.raises(AuditIntegrityError, match="fork.*orphan"):
            run_pipeline_to_end(...)


class TestI1bBatchConsumedDeferred:
    """I1b (deferred): (TRANSIENT, BATCH_CONSUMED) at end of run requires the
    batch-result token to be recorded. End-of-run sweep catches violations."""

    def test_batch_consumed_with_flush_token_passes(self, ...) -> None:
        ...

    def test_batch_consumed_orphan_crashes_at_end_of_run(self, ...) -> None:
        with pytest.raises(AuditIntegrityError, match="batch.*orphan"):
            run_pipeline_to_end(...)
```

**Step 2: Run RED**

Expected: all six tests fail. Phase 1's `_validate_outcome_fields` catches single-row missing-field violations but not cross-table; Phase 4 adds the cross-table layer.

**Definition of Done:**
- [ ] Six tests written
- [ ] All six fail before Phase 4's edits land
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
) -> None:
    """ADR-019 § Cross-check invariants — real-time invariants verified at write.

    I1c (sink-fallback-paired): (TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires
    a paired NodeStateStatus.COMPLETED node_state at the failsink node, AND
    that node_state has a registered artifacts row.

    I3 (discard-no-failsink): (FAILURE, SINK_DISCARDED) requires
    sink_name='__discard__' AND no paired NodeStateStatus.COMPLETED sink
    node_state for the same token (no failsink absorbed).

    Raises:
        AuditIntegrityError: if either invariant fails.
    """
    pair = (outcome, path)

    # I1c: failsink-paired
    if pair == (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK):
        # Find a COMPLETED SINK node_state for this token (the failsink
        # that absorbed the row). Filter by node_type == SINK so the check
        # cannot be spuriously satisfied by a COMPLETED transform/aggregation
        # node_state — those don't witness sink durability and are not
        # what the I1c invariant promises.
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
        if completed_sink_state is None:
            raise AuditIntegrityError(
                f"I1c violation for token {ref.token_id}: "
                f"(TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires a paired "
                f"NodeStateStatus.COMPLETED sink node_state, none found. The "
                f"failsink absorbed-the-row claim has no structural witness."
            )
        # The COMPLETED sink node_state must have a registered artifact.
        artifact_row = self._ops.execute_fetchone(
            select(artifacts_table.c.artifact_id)
            .where(artifacts_table.c.state_id == completed_sink_state.state_id)
        )
        if artifact_row is None:
            raise AuditIntegrityError(
                f"I1c violation for token {ref.token_id}: "
                f"failsink node_state {completed_sink_state.state_id} has no "
                f"registered artifact. Tier 1: a failsink-paired diversion "
                f"requires a durable artifact as the lifecycle witness."
            )

    # I3: discard-no-failsink
    if pair == (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
        if sink_name != "__discard__":
            raise AuditIntegrityError(
                f"I3 violation for token {ref.token_id}: "
                f"(FAILURE, SINK_DISCARDED) requires sink_name='__discard__', "
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

from elspeth.contracts.enums import NodeStateStatus, NodeType, TerminalOutcome, TerminalPath
from elspeth.core.landscape.schema import nodes_table, node_states_table, ...
```

Verify with: `.venv/bin/python -c "from elspeth.contracts.enums import NodeType; print(NodeType.SINK.value)"` — expect `sink`.

**Step 2: Wire into `record_token_outcome`**

Add the call immediately after `_validate_token_run_ownership(ref)` and before the INSERT:

```python
def record_token_outcome(
    self,
    ref: TokenRef,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None = None,
    # ... rest of params
) -> str:
    self._validate_outcome_fields(...)
    self._validate_token_run_ownership(ref)
    # NEW: Phase 4 cross-table invariants for real-time-verifiable cases.
    self._validate_cross_table_invariants(ref, outcome, path, sink_name=sink_name)
    # ... rest of the existing INSERT
```

**Step 3: GREEN**

Run: `.venv/bin/python -m pytest tests/integration/test_adr_019_cross_table_invariants.py::TestI1cFailsinkPaired tests/integration/test_adr_019_cross_table_invariants.py::TestI3DiscardNoFailsink -v`

Expected: all I1c and I3 tests pass.

**Definition of Done:**
- [ ] `_validate_cross_table_invariants` method added
- [ ] Wired into `record_token_outcome`
- [ ] I1c and I3 unit + integration tests pass
- [ ] mypy clean

---

### Task 4.3: Implement deferred I1a / I1b sweep in `_finalize_source_iteration`

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py:2511-2603` (`_finalize_source_iteration`)

**Step 1: Locate the existing finalize hook**

```bash
grep -n "_finalize_source_iteration" src/elspeth/engine/orchestrator/core.py
```

Expected: method definition at line 2511; called from `_run_main_processing_loop` at line 2812. The method runs aggregation flush (line 2563), coalesce flush (line 2585), then deferred field-resolution recording (line 2596). The deferred-invariant sweep slots in between the flushes and the field-resolution recording — AFTER flushes (so I1b sees batches reaching COMPLETED), BEFORE field resolution (no ordering dependency, but logically the sweep belongs to the audit-integrity layer, not the recorder layer).

The whole sweep MUST be gated on `not interrupted_by_shutdown`, mirroring the existing flush gates. Graceful shutdown legitimately leaves fork-parents without children; the resume path completes them.

**Step 2: Add the sweep — exact insertion point at line 2592 (between coalesce flush and field resolution)**

Insert this block AFTER the existing coalesce-flush block (which ends at line 2592, inside the `if not interrupted_by_shutdown:` body) and BEFORE the field-resolution recording at line 2596:

```python
        if not interrupted_by_shutdown:
            # ... existing aggregation flush at lines 2558-2581 ...
            # ... existing coalesce flush at lines 2584-2592 ...

            # ADR-019 Phase 4: deferred cross-table invariants. MUST run
            # after both flushes so I1b can observe batch-result tokens
            # landing at flush time. Skipped on graceful shutdown — the
            # resume path produces the missing children / batch-results.
            self._sweep_deferred_invariants_or_crash(run_id)

        # Record field resolution for empty sources (header-only files).
        # ... existing field_resolution_recorded block ...
```

**Step 3: Add the sweep helper as a new method on the Orchestrator class**

Place `_sweep_deferred_invariants_or_crash` as a sibling helper of `_finalize_source_iteration` (e.g., immediately after it, around line 2604).

```python
def _sweep_deferred_invariants_or_crash(self, run_id: str) -> None:
    """ADR-019 § Cross-check invariants I1a, I1b — end-of-run sweep.

    I1a: every (TRANSIENT, FORK_PARENT) and (TRANSIENT, EXPAND_PARENT)
         parent must have ≥1 child token_outcomes row referencing it via
         token_parents.
    I1b: every (TRANSIENT, BATCH_CONSUMED) row must have a paired
         batch-result token recorded for its batch_id (i.e., a batch row
         in BatchStatus.COMPLETED).

    These cannot be checked at parent-record / consume-record time because
    children/batch-results land later. This sweep is the first moment they
    can be verified. Called from _finalize_source_iteration ONLY on the
    non-interrupted-by-shutdown path — graceful shutdown intentionally
    leaves these unsatisfied because resume completes them.

    Raises:
        AuditIntegrityError: any orphaned parent / consumed token.
    """
    # I1a: orphaned FORK_PARENT / EXPAND_PARENT.
    # The data_flow repository owns the SQL surface; reuse its query
    # primitives rather than constructing raw SQLAlchemy here. The
    # Orchestrator's existing handle is self._data_flow (set during run
    # setup); add a public helper on DataFlowRepository named
    # ``find_orphaned_transient_parents(run_id)`` and ``find_orphaned_batch_consumptions(run_id)``
    # in the same Phase 4 commit (see Phase 4 supplementary edit below).
    orphan_parents = self._data_flow.find_orphaned_transient_parents(run_id)
    if orphan_parents:
        formatted = ", ".join(
            f"{r.token_id} (path={r.path.name})" for r in orphan_parents[:10]
        )
        raise AuditIntegrityError(
            f"ADR-019 I1a violation: {len(orphan_parents)} fork/expand parent "
            f"token(s) have no children at run-end. Examples: {formatted}. "
            f"Every (TRANSIENT, FORK_PARENT|EXPAND_PARENT) row must have at "
            f"least one child token in token_parents."
        )

    # I1b: orphaned BATCH_CONSUMED.
    orphan_batches = self._data_flow.find_orphaned_batch_consumptions(run_id)
    if orphan_batches:
        formatted = ", ".join(orphan_batches[:10])
        raise AuditIntegrityError(
            f"ADR-019 I1b violation: {len(orphan_batches)} batch_id(s) had "
            f"BATCH_CONSUMED tokens but the batch never reached "
            f"BatchStatus.COMPLETED. Examples: {formatted}. The batch-result "
            f"token must land before end of run."
        )
```

**Step 4: Add the two repository helper methods**

Place these next to the existing `record_token_outcome` in `data_flow_repository.py`:

```python
def find_orphaned_transient_parents(self, run_id: str) -> list[Row]:
    """ADR-019 I1a: parent tokens (TRANSIENT, FORK_PARENT|EXPAND_PARENT)
    with no children in token_parents for the given run."""
    parent_paths = (
        TerminalPath.FORK_PARENT.value,
        TerminalPath.EXPAND_PARENT.value,
    )
    query = (
        select(token_outcomes_table.c.token_id, token_outcomes_table.c.path)
        .where(token_outcomes_table.c.run_id == run_id)
        .where(token_outcomes_table.c.path.in_(parent_paths))
        .where(token_outcomes_table.c.outcome == TerminalOutcome.TRANSIENT.value)
        .where(
            ~select(token_parents_table.c.token_id)
            .where(
                token_parents_table.c.parent_token_id
                == token_outcomes_table.c.token_id
            )
            .exists()
        )
    )
    return list(self._ops.execute_fetchall(query))


def find_orphaned_batch_consumptions(self, run_id: str) -> list[str]:
    """ADR-019 I1b: distinct batch_ids that have BATCH_CONSUMED tokens
    but no row in batches with status=COMPLETED for the given run.
    Returns just the batch_id strings."""
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
- [ ] `_sweep_deferred_invariants_or_crash` method added
- [ ] Wired into `_finalize_source_iteration` at line ~2592, inside the `not interrupted_by_shutdown` block, after coalesce flush and before field-resolution recording
- [ ] I1a and I1b integration tests pass
- [ ] No regression in other resume / orchestrator tests

---

### Task 4.4: Sweep-crash audit-trail durability — ordering + regression test

**Why this task exists:** When `_sweep_deferred_invariants_or_crash` raises `AuditIntegrityError`, the run must finalize as `RunStatus.FAILED` with a queryable error message AND the orphaned `token_outcomes` rows that triggered the crash must REMAIN in the audit DB. Per CLAUDE.md Auditability Standard (*"Every decision must be traceable to source data, configuration, and code version"*) and the attributability test (*"For any output, `explain(recorder, run_id, token_id)` must prove complete lineage back to source"*), an operator querying a sweep-crashed run must be able to (a) see why the run failed and (b) inspect the orphaned rows that triggered the failure.

The Phase 4 plan up to this point doesn't specify:
1. **Ordering**: where in the run-finalization sequence does the sweep run? After `update_run_status(RUNNING)` but before `finalize_run(COMPLETED)` — i.e., the run is in `RUNNING` when the sweep fires.
2. **Audit-trail durability on crash**: the existing `Orchestrator` exception-handling path around `_run_main_processing_loop` calls `factory.run_lifecycle.finalize_run(run_id, status=RunStatus.FAILED)` on uncaught exceptions (see `core.py:638`). The sweep's `AuditIntegrityError` must propagate through this same handler — NOT be caught locally and swallowed.
3. **Evidence preservation**: orphaned `token_outcomes` rows must NOT be deleted by the sweep. The sweep's job is to raise on detection, not to clean up. The orphaned rows are evidence of the bug that produced them — Tier 1 audit data, never deleted.
4. **Regression test**: there's no test that verifies the sweep-crash leaves a queryable audit trail.

**Files:**
- Verify (no edit): `src/elspeth/engine/orchestrator/core.py` exception handler around `_run_main_processing_loop` (the existing path at `core.py:622-638` that calls `finalize_run(status=RunStatus.FAILED)` on uncaught exceptions). Confirm `AuditIntegrityError` is NOT caught earlier in `_finalize_source_iteration` or `_run_main_processing_loop`.
- Modify (only if the verification above fails): the exception-handler path so that `AuditIntegrityError` flows through to `finalize_run(FAILED)`.
- Test (RED-first): `tests/integration/test_adr_019_sweep_durability.py` (NEW)

**Step 1: Verify the existing exception path lets the sweep crash propagate to run-finalization**

```bash
grep -n "except\|finalize_run\|_finalize_source_iteration" src/elspeth/engine/orchestrator/core.py | head -30
```

Walk the exception path from `_sweep_deferred_invariants_or_crash` outward:
- `_sweep_deferred_invariants_or_crash` (added in Task 4.3) raises `AuditIntegrityError`.
- Called from `_finalize_source_iteration` at `core.py:~2592` (Task 4.3 edit). That method MUST NOT catch `AuditIntegrityError` — it must propagate.
- `_finalize_source_iteration` is called from `_run_main_processing_loop` at `core.py:2812`. Same: must not catch.
- The Orchestrator's outer `try/except` (search for `factory.run_lifecycle.finalize_run` calls) catches the unbounded exception, calls `finalize_run(status=FAILED, error=...)` to durably record the crash, then re-raises so the CLI / API surfaces the failure.

If any layer between the sweep and the outer handler catches `AuditIntegrityError` and swallows it, fix that catch to either re-raise or pass through. The whole point is that this exception class is NEVER recoverable per its `tier_1_error` decoration at `contracts/errors.py`.

**Step 2: Document the ordering contract**

Add this comment block immediately above the `_sweep_deferred_invariants_or_crash` call site at `core.py:~2592`:

```python
# ADR-019 Phase 4: deferred cross-table invariant sweep.
#
# AUDIT-TRAIL DURABILITY CONTRACT (CLAUDE.md Auditability Standard):
#
# 1. Run is in RunStatus.RUNNING when this sweep fires (the outer
#    finalize_run(COMPLETED) call has not yet executed).
# 2. If the sweep raises AuditIntegrityError, it propagates to the
#    Orchestrator's outer exception handler at core.py:622-638, which
#    calls factory.run_lifecycle.finalize_run(run_id, status=FAILED,
#    error=str(exc)). The run row is durably updated with the failure
#    reason BEFORE the exception is re-raised to the caller.
# 3. The orphaned token_outcomes rows that triggered the sweep crash
#    are NOT deleted. They remain as Tier 1 evidence: the operator's
#    explain(recorder, run_id, token_id) query returns the full lineage
#    including the orphaned parent / unflushed batch row that the sweep
#    detected. Deleting evidence on crash would violate CLAUDE.md
#    "no inference — if it's not recorded, it didn't happen" and erase
#    the auditor's ability to diagnose the bug.
# 4. Skipped on graceful shutdown — see the not interrupted_by_shutdown
#    gate above. Resume completes the missing children/batch-results.
self._sweep_deferred_invariants_or_crash(run_id)
```

**Step 3: Write the durability regression test (RED-first)**

Create: `tests/integration/test_adr_019_sweep_durability.py`

```python
"""ADR-019 Phase 4: sweep-crash audit-trail durability regression test.

Per CLAUDE.md Auditability Standard, when the deferred-invariant sweep
crashes mid-run, the operator must be able to query (a) the failed run's
status and error message, AND (b) the orphaned token_outcomes rows that
triggered the crash. Both are Tier 1 evidence; neither may be lost.

This test is RED before Phase 4's sweep + run-finalization wiring is
verified end-to-end and GREEN once both ordering and evidence-preservation
are confirmed.
"""

import pytest

from elspeth.contracts.enums import RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape import LandscapeDB


class TestSweepCrashAuditTrailDurability:
    def test_orphaned_fork_parent_crashes_run_with_durable_error(
        self, tmp_path
    ) -> None:
        """A fork-parent without children (I1a violation) crashes the run
        but leaves a queryable audit trail.

        Setup: directly insert a (TRANSIENT, FORK_PARENT) token_outcomes row
        with no entries in token_parents. This simulates a faulty plugin
        that emits a fork-parent without spawning children — exactly the
        bug shape the I1a invariant exists to catch.
        """
        # Build a minimal pipeline that completes normally, then directly
        # corrupt the audit DB to insert the orphan parent before the
        # sweep runs. (In production, this corruption would arise from a
        # faulty plugin emitting FORK_PARENT without children — we
        # simulate it deterministically here.)
        audit_db_url = f"sqlite:///{tmp_path / 'audit.db'}"
        run_id = "test_run_orphan_fork_parent"
        token_id = "tok_orphan_001"

        with LandscapeDB.connect(audit_db_url) as db:
            # Insert a run row in RUNNING state.
            # Insert a (TRANSIENT, FORK_PARENT) token_outcomes row with no
            # corresponding token_parents entry — the I1a violation.
            ...

            # Trigger _finalize_source_iteration directly OR run a tiny
            # pipeline that will hit the sweep at end-of-source.
            with pytest.raises(AuditIntegrityError, match="I1a"):
                # The sweep crashes — exception propagates out.
                ...

            # Audit-trail durability assertions:

            # 1. Run is finalized as FAILED.
            run_row = db.query.get_run(run_id)
            assert run_row.status == RunStatus.FAILED, (
                f"Expected RunStatus.FAILED after sweep crash, got "
                f"{run_row.status}. The sweep's AuditIntegrityError must "
                f"propagate to the run-finalization handler."
            )
            assert "I1a" in (run_row.error or ""), (
                f"Expected the I1a violation message in run.error for "
                f"operator queryability; got error={run_row.error!r}."
            )

            # 2. The orphaned token_outcomes row REMAINS in the DB as
            #    Tier 1 evidence. The sweep detected it; it must persist.
            orphan_outcome = db.data_flow.get_token_outcome(token_id)
            assert orphan_outcome is not None, (
                "The orphaned (TRANSIENT, FORK_PARENT) row was deleted by "
                "the sweep — Tier 1 violation. Evidence must persist for "
                "operator inspection per CLAUDE.md Auditability Standard."
            )
            assert orphan_outcome.outcome == TerminalOutcome.TRANSIENT
            assert orphan_outcome.path == TerminalPath.FORK_PARENT

    def test_orphaned_batch_consumed_crashes_run_with_durable_error(
        self, tmp_path
    ) -> None:
        """Symmetric I1b case: a BATCH_CONSUMED token whose batch never
        reached COMPLETED triggers an I1b violation; same durability
        assertions apply.
        """
        # ... mirror of the I1a test, simulating a batch-consume without
        # a flush-time batch-result token.
        ...

    def test_sweep_skipped_on_graceful_shutdown(self, tmp_path) -> None:
        """A run interrupted by graceful shutdown must NOT crash on
        orphaned fork-parents — resume produces the missing children.

        Setup: build a pipeline that buffers tokens, send SIGINT mid-run,
        verify the run finalizes as INTERRUPTED (not FAILED) and the
        unflushed parents remain queryable for resume.
        """
        # ... fixture for a buffered-mid-run state, signal-bounded
        # interruption, status assertion.
        ...
```

**Step 4: Run RED**

```bash
.venv/bin/python -m pytest tests/integration/test_adr_019_sweep_durability.py -v
```

Expected: tests fail until Phase 4's sweep + ordering + propagation is wired correctly. They GREEN once Step 1's verification confirms the exception path and Step 2's comment block lands.

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
- [ ] Verified the exception propagation path from `_sweep_deferred_invariants_or_crash` to `factory.run_lifecycle.finalize_run(status=FAILED, error=...)` — no intermediate handler swallows `AuditIntegrityError`
- [ ] Audit-trail durability comment block added at the sweep call site
- [ ] Three durability regression tests (I1a-orphan, I1b-orphan, shutdown-skip) pass
- [ ] Orphaned `token_outcomes` rows verified to PERSIST after sweep crash (Tier 1 evidence preservation)
- [ ] No regression in graceful-shutdown handling

---

### Task 4.5: Phase 4 commit

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -q --timeout=120
```

Expected: all tests pass. The new invariants are additive and should not break anything that was correct before.

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
  a paired NodeStateStatus.COMPLETED node_state plus registered artifact.
  Verified at write time per ADR-019 line 256.
- I3 (discard-no-failsink): (FAILURE, SINK_DISCARDED) requires
  sink_name='__discard__' and no paired sink-completion node_state. Verified
  at write time per ADR-019 line 261.

Deferred / end-of-run sweep (orchestrator/core.py::_finalize_source_iteration
calls a new helper _sweep_deferred_invariants_or_crash, gated on
not interrupted_by_shutdown):
- I1a (lineage-paired): every fork/expand parent must have ≥1 child token by
  end of run. Children land later than parents, so write-time check is impossible.
- I1b (aggregate-paired): every BATCH_CONSUMED token must have a paired
  batch-result (BatchStatus.COMPLETED) by end of run.

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
  status=FAILED, error=...). Run row durably reflects the failure.
- Orphaned token_outcomes rows that triggered the sweep crash REMAIN
  in the DB as Tier 1 evidence for operator explain() queries.
- Shutdown-gated: graceful shutdown skips the sweep entirely, preserving
  the resume contract.
- Three regression tests in tests/integration/test_adr_019_sweep_durability.py
  pin the I1a-orphan, I1b-orphan, and shutdown-skip behaviours.

Refs: elspeth-edb60744f0 (Stage 3 ticket — producer + accumulator)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md § Cross-check invariants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Definition of Done:**
- [ ] All four invariants implemented (two real-time, two deferred)
- [ ] Six integration tests pass
- [ ] No regression in other tests, OR latent bugs surfaced and fixed (not relaxed)
- [ ] Phase 4 commit landed
