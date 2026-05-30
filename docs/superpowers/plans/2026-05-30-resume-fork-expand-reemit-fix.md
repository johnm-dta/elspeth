# Resume Fork/Expand Re-emit Fix (F1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop checkpoint resume from re-emitting already-completed fork/expand branches (duplicate terminal outcomes + duplicate physical sink writes) by driving each *incomplete* child token to completion in place instead of restarting the whole row from source.

**Architecture:** On resume, a row with partially-completed fork/expand children is no longer restarted from source (which re-forks to *all* branches). Instead, `RecoveryManager` returns the incomplete non-delegation child tokens; `RowProcessor` reconstructs each as a `TokenInfo` from persisted columns + payload and re-drives it from its correct mid-DAG node — under the *original* fork/expand parent — via the existing `process_token` machinery. To coexist with run-1's append-only `node_states` records, the re-drive records at an elevated `attempt` carried on `WorkItem`. Expand children (whose per-child payload is not persisted today) are enabled by a new `tokens.token_data_ref` column written at expand time.

**Tech Stack:** Python 3.13, SQLAlchemy Core, SQLite audit DB (epoch-gated), pluggy, pytest + Hypothesis. Worktree: `/home/john/elspeth/.worktrees/fix-resume-fork-reemit` (branch `fix/resume-fork-reemit`, own `.venv` Python 3.13). Run tests with `.venv/bin/python -m pytest`.

**Authoritative spec:** `docs/superpowers/specs/2026-05-30-resume-partial-fork-reemit-fix-design.md` (read the REVISED DESIGN + ADDENDUM sections before starting).

---

## Build order & milestones

- **Phase 1** — harden the existing RED reproduction's oracle (still RED). Conservation law: never-zero, never-two, multiset == baseline, status COMPLETED, physical-write-once.
- **Phase 2** — fork-resume core (no schema change). Reconstruct incomplete child + `resume_attempt_offset` + mid-DAG dispatch → **RED test goes GREEN.** *(Primary P1 harm fixed; working/testable software milestone.)*
- **Phase 3** — schema change: `token_data_ref` column, epoch 10→11, `_REQUIRED_COLUMNS` F3 co-fix.
- **Phase 4** — expand persistence (`expand_token` writes per-child payload) + expand-resume reconstruction → expand-resume test GREEN.
- **Phase 5** — F2 counter-reconciliation assertion + remaining risk-ordered matrix + full gate run.

## File structure (created / modified)

- Modify `src/elspeth/core/landscape/schema.py` — add `tokens.token_data_ref`; bump `SQLITE_SCHEMA_EPOCH` 10→11 (Phase 3).
- Modify `src/elspeth/core/landscape/database.py` — extend `_REQUIRED_COLUMNS` (Phase 3, F3).
- Modify `src/elspeth/core/checkpoint/recovery.py` — new `IncompleteTokenSpec` + `get_incomplete_tokens_by_row()` (Phase 2/4).
- Modify `src/elspeth/engine/dag_navigator.py` — `WorkItem.resume_attempt_offset` field; `resolve_branch_first_node()` accessor (Phase 2).
- Modify `src/elspeth/engine/processor.py` — `resume_incomplete_token()`; relax `process_token` `current_node_id` to `NodeID | None`; thread offset through `_process_single_token` → handlers (Phase 2).
- Modify `src/elspeth/engine/executors/sink.py`, `src/elspeth/engine/executors/transform.py` (via `state_guard`), `src/elspeth/engine/coalesce_executor.py` — add offset to `begin_node_state` calls (Phase 2).
- Modify `src/elspeth/engine/orchestrator/resume.py` — dispatch: no-tokens → `process_existing_row`; incomplete-tokens → `resume_incomplete_token` (Phase 2).
- Modify `src/elspeth/core/landscape/data_flow_repository.py` — `expand_token` per-child payload persistence; expand-child contract read (Phase 4).
- Modify `src/elspeth/engine/tokens.py` — pass per-child payloads into `expand_token` (Phase 4).
- Modify `tests/property/audit/test_fork_join_balance.py` — oracle hardening + matrix (Phases 1, 2, 4, 5).
- Test `tests/unit/landscape/test_schema_epoch_and_required_columns.py` (new, Phase 3).

---

## Phase 1 — Harden the reproduction oracle (stays RED)

The existing `test_resume_does_not_reemit_completed_fork_branch` (test_fork_join_balance.py:737) checks only `n != 1`, which is **blind to the orphan failure mode** (a branch with zero outcomes) and discards `baseline` and the resume result status. Encode the full conservation law before fixing, so the GREEN bar is correct.

### Task 1: Strengthen the reproduction oracle

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py:824-889` (the assertion block of `test_resume_does_not_reemit_completed_fork_branch`)

- [ ] **Step 1: Add a zero-orphan-leaf helper near the other audit helpers** (after `get_fork_group_stats`, ~line 130)

```python
def count_nonterminal_leaf_tokens(db: LandscapeDB, run_id: str) -> int:
    """Count non-delegation leaf tokens that lack a completed terminal outcome.

    A leaf token is any token whose own outcome is NOT a delegation marker
    (FORK_PARENT / EXPAND_PARENT). After a fully-resumed run, every such token
    must carry exactly one completed=1 outcome — zero is an orphan (a different
    audit-integrity violation than double-emit, and invisible to a count!=1 check
    that only sees tokens which HAVE outcomes).
    """
    from elspeth.contracts.enums import TerminalPath

    delegation = (TerminalPath.FORK_PARENT.value, TerminalPath.EXPAND_PARENT.value)
    with db.connection() as conn:
        rows = conn.execute(
            text("""
                SELECT t.token_id AS token_id
                FROM tokens t
                WHERE t.run_id = :run_id
                  AND t.token_id NOT IN (
                      SELECT o.token_id FROM token_outcomes o
                      WHERE o.run_id = :run_id AND o.path IN (:fp, :ep)
                  )
                  AND t.token_id NOT IN (
                      SELECT o.token_id FROM token_outcomes o
                      WHERE o.run_id = :run_id AND o.completed = 1
                            AND o.path NOT IN (:fp, :ep)
                  )
            """),
            {"run_id": run_id, "fp": delegation[0], "ep": delegation[1]},
        ).fetchall()
    return len(rows)
```

- [ ] **Step 2: Replace the final assertion block (lines 880-889) with the full conservation law**

```python
        # CONSERVATION LAW (spec ADDENDUM test oracle):
        #  (a) multiset of completed (row_id, sink_name) outcomes == uninterrupted baseline
        #  (b) zero orphan leaf tokens (never-zero)
        #  (c) no (row_id, sink_name) carries two outcomes (never-two)
        #  (d) the resume result reports COMPLETED
        #  (e) the surviving branch's sink was physically written exactly once
        #  (f) fork-group shape is unchanged (guards against the Approach-2 double-parent)
        after = _outcome_counts()
        assert after == baseline, (
            f"Resume must conserve the terminal-outcome multiset. "
            f"baseline={baseline} after={after}"
        )
        assert count_nonterminal_leaf_tokens(db, run_id) == 0, (
            "Resume left a non-delegation leaf token with no terminal outcome (orphan)."
        )
        assert all(n == 1 for n in after.values()), after
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status
        assert len(sink_b.collected) == 1, (
            f"sink_b (completed before interruption) was physically written "
            f"{len(sink_b.collected)} times; resume must not re-write it."
        )
        post_resume_stats = get_fork_group_stats(db, run_id)
        assert post_resume_stats["total_fork_groups"] == baseline_fork_stats["total_fork_groups"], (
            f"Resume changed fork-group shape: {baseline_fork_stats} -> {post_resume_stats}"
        )
```

- [ ] **Step 3: Capture `baseline_fork_stats` and the resume result** — at line 824 (right after `baseline = _outcome_counts()`) add:

```python
        baseline_fork_stats = get_fork_group_stats(db, run_id)
```

  and change line 878 to capture the return value:

```python
        resume_result = resume_orchestrator.resume(
            resume_point, config, graph, payload_store=payload_store, settings=settings_obj
        )
```

- [ ] **Step 4: Add the `RunStatus` import** to the test's local imports (line 755-758 block):

```python
        from elspeth.contracts.enums import RunStatus
```

  (Verify the enum name: `rg -n "class RunStatus|COMPLETED" src/elspeth/contracts/enums.py`. If the orchestrator result exposes status differently, adapt `resume_result.status` to the actual attribute — confirm with `rg -n "class RunResult|status" src/elspeth/engine/orchestrator/types.py`.)

- [ ] **Step 5: Run the test — confirm it still FAILS, now on the conservation law**

Run: `.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_does_not_reemit_completed_fork_branch -v`
Expected: FAIL — `after != baseline` and/or `sink_b.collected` length 2 (double-emit). The failure proves the oracle detects the defect.

- [ ] **Step 6: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: harden resume fork re-emit oracle to full conservation law (F1)"
```

---

## Phase 2 — Fork-resume core (turns the RED test GREEN)

### Task 2: `IncompleteTokenSpec` + recovery selection

Reuse the existing incomplete-token semantics (`recovery.py:383-411` subqueries, `_DELEGATION_PATHS`) — do **not** write a second drifting completion query.

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py` (add dataclass near top imports; add method after `get_unprocessed_rows`, ~line 475)
- Test: `tests/property/audit/test_fork_join_balance.py` (new test in `TestForkRecoveryInvariant`)

- [ ] **Step 1: Write the failing test** (in `TestForkRecoveryInvariant`)

```python
    def test_get_incomplete_tokens_by_row_returns_only_incomplete_leaf(self) -> None:
        """Recovery surfaces exactly the non-delegation tokens lacking a terminal outcome."""
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource([{"value": 1}], on_success="sink_a")
        sink_a, sink_b = CollectSink("sink_a"), CollectSink("sink_b")
        gate = GateSettings(name="fork_gate", input="gate_in", condition="True",
                            routes={"true": "fork", "false": "sink_a"}, fork_to=["sink_a", "sink_b"])
        config = PipelineConfig(source=as_source(source), transforms=[],
                                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)}, gates=[gate])
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[], sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate], aggregations={}, coalesce_settings=[])
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={"sink_a": {"plugin": "test", "on_write_failure": "discard"},
                   "sink_b": {"plugin": "test", "on_write_failure": "discard"}}, gates=[gate])

        run = Orchestrator(db).run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        # Delete sink_a's outcome → its child becomes the sole incomplete leaf.
        with db.engine.connect() as conn:
            ids = conn.execute(text(
                "SELECT o.outcome_id AS oid, o.token_id AS tid FROM token_outcomes o "
                "JOIN tokens t ON t.token_id=o.token_id JOIN rows r ON r.row_id=t.row_id "
                "WHERE r.run_id=:rid AND o.sink_name='sink_a'"), {"rid": run.run_id}).fetchall()
            incomplete_token_id = ids[0].tid
            for r in ids:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == r.oid))
            conn.commit()

        recovery = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery.get_incomplete_tokens_by_row(run.run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert [s.token_id for s in all_specs] == [incomplete_token_id]
        spec = all_specs[0]
        assert spec.branch_name == "sink_a"
        assert spec.fork_group_id is not None
        assert spec.max_attempt >= 0  # sink_a child wrote a sink node_state at attempt 0
```

- [ ] **Step 2: Run it — fails** (method does not exist)

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_get_incomplete_tokens_by_row_returns_only_incomplete_leaf" -v`
Expected: FAIL — `AttributeError: 'RecoveryManager' object has no attribute 'get_incomplete_tokens_by_row'`

- [ ] **Step 3: Add the dataclass** at the top of `recovery.py` (after existing imports, near the `_DELEGATION_PATHS` constant at line 52):

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncompleteTokenSpec:
    """A non-delegation child token that lacks a terminal outcome on a resumed run.

    All identity fields are read directly from persisted columns (Tier-1: no
    defaults, no coercion — TokenInfo.__post_init__ rejects garbage downstream).
    ``token_data_ref`` is NULL for fork children (they share the parent/source
    payload, retrievable by ``row_id``) and set for expand children (Phase 4).
    ``max_attempt`` is the highest ``attempt`` already recorded for this token in
    ``node_states`` (-1 if none); the resume re-drive uses ``max_attempt + 1`` so
    its node_states coexist with the append-only run-1 records.
    """

    token_id: str
    row_id: str
    branch_name: str | None
    fork_group_id: str | None
    join_group_id: str | None
    expand_group_id: str | None
    token_data_ref: str | None
    max_attempt: int
```

- [ ] **Step 4: Add the selection method** after `get_unprocessed_rows` (~line 475). It reuses the delegation/terminal subqueries:

```python
    def get_incomplete_tokens_by_row(self, run_id: str) -> dict[str, list[IncompleteTokenSpec]]:
        """Return incomplete non-delegation child tokens, grouped by row_id.

        A token is incomplete when it is NOT a delegation marker
        (FORK_PARENT / EXPAND_PARENT) and has NO completed terminal outcome.
        Mirrors the completion semantics of get_unprocessed_rows (shared
        _DELEGATION_PATHS) so recovery selection and resume reconstruction
        cannot drift apart.
        """
        with self._db.engine.connect() as conn:
            delegation_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()
            terminal_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.completed == 1)
                .where(~token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()

            max_attempt_sq = (
                select(func.max(node_states_table.c.attempt))
                .where(node_states_table.c.token_id == tokens_table.c.token_id)
                .where(node_states_table.c.run_id == run_id)
                .correlate(tokens_table)
                .scalar_subquery()
            )

            query = (
                select(
                    tokens_table.c.token_id,
                    tokens_table.c.row_id,
                    tokens_table.c.branch_name,
                    tokens_table.c.fork_group_id,
                    tokens_table.c.join_group_id,
                    tokens_table.c.expand_group_id,
                    tokens_table.c.token_data_ref,
                    max_attempt_sq.label("max_attempt"),
                )
                .where(tokens_table.c.run_id == run_id)
                .where(~tokens_table.c.token_id.in_(delegation_tokens))
                .where(~tokens_table.c.token_id.in_(terminal_tokens))
                .order_by(tokens_table.c.step_in_pipeline, tokens_table.c.token_id)
            )
            rows = conn.execute(query).fetchall()

        by_row: dict[str, list[IncompleteTokenSpec]] = {}
        for r in rows:
            by_row.setdefault(r.row_id, []).append(
                IncompleteTokenSpec(
                    token_id=r.token_id,
                    row_id=r.row_id,
                    branch_name=r.branch_name,
                    fork_group_id=r.fork_group_id,
                    join_group_id=r.join_group_id,
                    expand_group_id=r.expand_group_id,
                    token_data_ref=r.token_data_ref,
                    max_attempt=-1 if r.max_attempt is None else int(r.max_attempt),
                )
            )
        return by_row
```

> **NOTE (Phase 3 dependency):** `tokens.token_data_ref` does not exist until Phase 3. To keep Phase 2 self-contained and GREEN before the schema change, in Phase 2 **omit** `tokens_table.c.token_data_ref` from the SELECT and set `token_data_ref=None` in the spec. Add the column to the SELECT in Phase 4, Task 9. (Fork children never need it.)

- [ ] **Step 5: Add imports** to `recovery.py`: ensure `func` and `node_states_table` are imported — `from sqlalchemy import func` and add `node_states_table` to the existing `from elspeth.core.landscape.schema import ...` line (confirm with `rg -n "^from sqlalchemy|landscape.schema import" src/elspeth/core/checkpoint/recovery.py`).

- [ ] **Step 6: Run the test — passes**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_get_incomplete_tokens_by_row_returns_only_incomplete_leaf" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/core/checkpoint/recovery.py tests/property/audit/test_fork_join_balance.py
git commit -m "feat(recovery): IncompleteTokenSpec + get_incomplete_tokens_by_row (F1)"
```

### Task 3: `resume_attempt_offset` on `WorkItem` + executor threading

**Files:**
- Modify: `src/elspeth/engine/dag_navigator.py:35-56` (WorkItem)
- Modify: `src/elspeth/engine/processor.py` (`_process_single_token` and the three handlers it calls)
- Modify: `src/elspeth/engine/executors/sink.py:453`, `:838`
- Modify: `src/elspeth/engine/coalesce_executor.py:500`, `:563`
- Modify: `src/elspeth/engine/executors/state_guard.py` usage in `src/elspeth/engine/executors/transform.py:273`

- [ ] **Step 1: Add the field to `WorkItem`** (`dag_navigator.py:43-47`)

```python
    token: TokenInfo
    current_node_id: NodeID | None
    coalesce_node_id: NodeID | None = None
    coalesce_name: CoalesceName | None = None  # Name of the coalesce point (if any)
    on_success_sink: str | None = None  # Inherited sink for terminal children (deagg)
    resume_attempt_offset: int = 0  # Added to every node_states.attempt written while
    # re-driving a reconstructed incomplete token on resume, so its records coexist with
    # the append-only run-1 records under UniqueConstraint(token_id, node_id, attempt).
    # 0 for all normal (non-resume) processing — the only nonzero source is
    # RowProcessor.resume_incomplete_token().
```

  `create_work_item` and `create_continuation_work_item` (dag_navigator.py:122, 254) must forward the param. Add `resume_attempt_offset: int = 0` to both signatures and pass it into the `WorkItem(...)` construction (line 156) and the nested `create_work_item` call (line 300).

- [ ] **Step 2: Thread the offset to `begin_node_state` sites.** All three executors must add the offset.

  **sink.py:453** (primary sink) and **:838** (failsink) — these pass no `attempt` today (default 0). Add `attempt=resume_attempt_offset`. The sink executor's `process` entry point receives the work items being flushed; plumb `resume_attempt_offset` from the `WorkItem` into the sink executor call and down to these two `begin_node_state(...)` calls. (Read the sink executor's public method that `_handle_terminal_token` invokes — `rg -n "def .*sink|SinkExecutor" src/elspeth/engine/executors/sink.py` — and add a `resume_attempt_offset: int = 0` kwarg threaded to lines 453 and 838.)

  **coalesce_executor.py:500, :563** — add `attempt=resume_attempt_offset` (default 0 kwarg on the coalesce executor entry points used during a re-drive).

  **transform via state_guard** — `transform.py:273` already passes `attempt=attempt`. Change the attempt it computes to `resume_attempt_offset + attempt` (the tenacity-derived 0-based number), so a node that retries during resume stays collision-free at `offset, offset+1, ...`. Thread `resume_attempt_offset` from the `WorkItem` into the transform executor entry point.

- [ ] **Step 3: Thread the offset through `_process_single_token`.** It already destructures the work item. Read `work_item.resume_attempt_offset` once and pass it to `_handle_transform_node` (processor.py:2890), `_handle_gate_node` (only forks create children — children get offset 0, see Step 4), `_maybe_coalesce_token` (processor.py:2856), and `_handle_terminal_token` (processor.py:2929). Add `resume_attempt_offset: int = 0` to each of those method signatures and forward to the executor calls from Step 2.

  Find the work-item unpack at the top of the drain loop: `rg -n "def _drain_work_queue|def _process_single_token|work_item\." src/elspeth/engine/processor.py` and read the call into `_process_single_token` to see how `current_node_id`/`coalesce_*` reach it; `resume_attempt_offset` rides the same path.

- [ ] **Step 4: Children created during a re-drive use offset 0.** When `_handle_gate_fork` (processor.py:2674) or deaggregation creates child work items via the nav factories, they get fresh token_ids (no prior node_states) → leave `resume_attempt_offset` at its default 0. Do **not** propagate the parent's offset to children. Add a one-line comment at `_handle_gate_fork` stating this.

- [ ] **Step 5: No standalone test here** — this field is exercised end-to-end by Task 5 (the RED test) and Task 6 (the crash-after-node_state case). Verify nothing broke:

Run: `.venv/bin/python -m pytest tests/unit/engine tests/property/audit/test_fork_join_balance.py -q`
Expected: PASS (all existing tests; `resume_attempt_offset` defaults to 0 so behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/dag_navigator.py src/elspeth/engine/processor.py src/elspeth/engine/executors/sink.py src/elspeth/engine/executors/transform.py src/elspeth/engine/executors/state_guard.py src/elspeth/engine/coalesce_executor.py
git commit -m "feat(engine): WorkItem.resume_attempt_offset threaded to node_states writes (F1)"
```

### Task 4: `resume_incomplete_token()` on `RowProcessor` + relax `process_token`

**Files:**
- Modify: `src/elspeth/engine/processor.py:1980-2001` (`process_token` signature) and add `resume_incomplete_token` near it
- Modify: `src/elspeth/engine/dag_navigator.py` (add `resolve_branch_first_node`)

- [ ] **Step 1: Relax `process_token`** (processor.py:1985) — change `current_node_id: NodeID` to `current_node_id: NodeID | None`. It delegates to `create_work_item`, which already accepts `NodeID | None` (dag_navigator.py:126), and to `_drain_work_queue` → `_process_single_token`, which already handles `None` + `_branch_to_sink` (processor.py:2827-2833). Add `resume_attempt_offset: int = 0` and pass it into `create_work_item`.

- [ ] **Step 2: Add `resolve_branch_first_node` accessor** to `DAGNavigator` (after `create_continuation_work_item`, ~line 310):

```python
    def resolve_branch_first_node(self, branch_name: str) -> NodeID:
        """First processing node for a fork branch routed to a coalesce.

        For identity branches this equals the coalesce node id; for transform
        branches it is the first transform in the chain. Raises if unknown —
        an audit/DAG inconsistency, never a default-route.
        """
        try:
            return self._branch_first_node[branch_name]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Unknown branch name '{branch_name}' — not in branch_first_node map. "
                f"Known branches: {sorted(self._branch_first_node.keys())}"
            ) from exc
```

- [ ] **Step 3: Add `resume_incomplete_token` to `RowProcessor`** (after `process_token`, ~line 2002). This mirrors `_handle_gate_fork`'s dispatch (processor.py:2674-2697):

```python
    def resume_incomplete_token(
        self,
        spec: "IncompleteTokenSpec",
        row_data: PipelineRow,
        ctx: PluginContext,
    ) -> list[RowResult]:
        """Drive a single reconstructed incomplete child token to completion.

        Reuses the persisted child token id (continuing under the ORIGINAL fork/
        expand parent) and re-drives it from its correct mid-DAG node, mirroring
        the live fork dispatch in _handle_gate_fork. node_states are written at an
        elevated attempt (spec.max_attempt + 1) so they coexist with run-1 records.
        """
        token = TokenInfo(
            row_id=spec.row_id,
            token_id=spec.token_id,
            row_data=row_data,
            branch_name=spec.branch_name,
            fork_group_id=spec.fork_group_id,
            join_group_id=spec.join_group_id,
            expand_group_id=spec.expand_group_id,
        )
        offset = spec.max_attempt + 1  # -1 -> 0 (no prior node_states), else bump

        branch = spec.branch_name
        if branch is not None and BranchName(branch) in self._branch_to_sink:
            # fork -> sink terminal branch: straight to the sink, no traversal.
            return self.process_token(token, ctx, current_node_id=None, resume_attempt_offset=offset)
        if branch is not None and BranchName(branch) in self._branch_to_coalesce:
            coalesce_name = self._branch_to_coalesce[BranchName(branch)]
            first_node = self._nav.resolve_branch_first_node(branch)
            return self.process_token(
                token, ctx,
                current_node_id=first_node,
                coalesce_name=coalesce_name,
                resume_attempt_offset=offset,
            )
        if spec.expand_group_id is not None:
            # Expand child (Phase 4): re-drive from the node after the expand step.
            # Resolved in Phase 4, Task 10 — until then this branch is unreachable
            # because get_incomplete_tokens_by_row only surfaces fork children.
            raise OrchestrationInvariantError(
                f"Expand-child resume not yet wired for token {spec.token_id} "
                f"(expand_group_id={spec.expand_group_id})."
            )
        raise OrchestrationInvariantError(
            f"Incomplete token {spec.token_id} has branch_name={branch!r} that maps to "
            f"neither a sink nor a coalesce, and is not an expand child — cannot resolve "
            f"a resume-start node. This is an audit/DAG inconsistency."
        )
```

- [ ] **Step 4: Add imports** to `processor.py`: `IncompleteTokenSpec` (under `TYPE_CHECKING` for the annotation), and confirm `BranchName`, `TokenInfo`, `PluginContext`, `RowResult`, `OrchestrationInvariantError` are already imported (they are — used throughout). For the `IncompleteTokenSpec` annotation use a string literal to avoid an L2→L1 runtime import if one doesn't already exist; `recovery.py` is L1 (`core/`), `processor.py` is L2 (`engine/`), so a real import is layer-legal — prefer the real import: `from elspeth.core.checkpoint.recovery import IncompleteTokenSpec`. Verify no import cycle: `rg -n "import.*processor|import.*engine" src/elspeth/core/checkpoint/recovery.py` (recovery must not import the engine).

- [ ] **Step 5: No standalone test** — exercised by Task 5. Type-check:

Run: `.venv/bin/python -m mypy src/elspeth/engine/processor.py src/elspeth/engine/dag_navigator.py`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/processor.py src/elspeth/engine/dag_navigator.py
git commit -m "feat(engine): resume_incomplete_token + NodeID|None process_token (F1)"
```

### Task 5: Resume dispatch — wire the new path; turn RED green

**Files:**
- Modify: `src/elspeth/engine/orchestrator/resume.py:169-204` (the per-row loop) and the LoopContext/setup that feeds it

- [ ] **Step 1: Make the incomplete-token map available to the loop.** The loop needs `dict[str, list[IncompleteTokenSpec]]`. Read how `run_resume_processing_loop` is called (`orchestrator/core.py:2816`) and how `unprocessed_rows` + `LoopContext` are built (`rg -n "get_unprocessed_rows|LoopContext|run_resume_processing_loop" src/elspeth/engine/orchestrator/core.py`). Compute `incomplete_by_row = recovery_manager.get_incomplete_tokens_by_row(run_id)` in the same place `get_unprocessed_rows` is called, and add it to `LoopContext` (a frozen field) or pass it as a new keyword arg to `run_resume_processing_loop`. Prefer adding to `LoopContext` if the recovery manager / run_id are available there; otherwise add a `*, incomplete_by_row: Mapping[str, list[IncompleteTokenSpec]]` parameter.

- [ ] **Step 2: Replace the per-row processing call (resume.py:194-204) with the dispatch:**

```python
        specs = incomplete_by_row.get(row_id)
        if specs:
            # Partial fork/expand completion: drive ONLY the incomplete children to
            # completion under the original parent. Restarting from source would
            # re-fork to ALL branches and re-emit the completed ones (F1).
            results = []
            for spec in specs:
                results.extend(processor.resume_incomplete_token(spec, pipeline_row, ctx))
        else:
            # No tokens for this row (never started) -> whole-row restart is correct.
            results = processor.process_existing_row(
                row_id=row_id,
                row_data=pipeline_row,
                transforms=config.transforms,
                ctx=ctx,
            )
        if results:
            loop_ctx.last_token_id = results[-1].token.token_id

        accumulate_row_outcomes(results, counters, pending_tokens)
```

> The `pipeline_row` for a fork child is built from the source payload (`row_data` already wrapped at resume.py:192) — correct, because fork children share the parent's deep-copied source data (`tokens.py` fork path). Expand children (Phase 4) need their own payload + contract and are handled in Task 10.

- [ ] **Step 3: Run the reproduction — now GREEN**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_does_not_reemit_completed_fork_branch" -v`
Expected: PASS — `after == baseline`, `sink_b.collected` length 1, zero orphans, status COMPLETED, fork-group shape unchanged.

- [ ] **Step 4: Run the whole fork-join + resume suite for regressions**

Run: `.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py tests/e2e/recovery -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator/resume.py src/elspeth/engine/orchestrator/core.py
git commit -m "fix(resume): drive incomplete fork children in place, no source restart (F1)

Closes the duplicate-emission defect: a resumed partial-fork row no longer
restarts from source (which re-forks to all branches and re-emits completed
ones). Recovery surfaces the incomplete child tokens; the processor re-drives
each from its mid-DAG node under the original parent, recording node_states at
an elevated attempt to coexist with the append-only run-1 records."
```

### Task 6: fork→sink crash-after-node_state regression (the attempt-collision case)

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: Write the failing test** — simulate a crash that left an OPEN sink node_state for the incomplete child (the case that would collide at attempt=0 without the offset):

```python
    def test_resume_fork_sink_with_orphan_open_node_state(self) -> None:
        """Re-driving an incomplete fork->sink child must not collide on node_states.

        The incomplete branch left a node_state at attempt=0 in run 1 (sink began
        but never recorded a terminal outcome). The resume re-drive must write its
        node_state at attempt=1, not crash on UniqueConstraint(token_id,node_id,attempt).
        """
        # ... same fork pipeline + interruption setup as
        # test_resume_does_not_reemit_completed_fork_branch, BUT after deleting
        # sink_a's terminal outcome, assert sink_a's child still has its OPEN
        # node_state from run 1 (do NOT delete it):
        #   SELECT COUNT(*) FROM node_states WHERE token_id = <sink_a child>
        # is >= 1 with attempt=0.
        # Then resume and assert:
        #   (a) the full conservation law (after == baseline, zero orphan, status COMPLETED)
        #   (b) sink_a's child now has a node_state at attempt=1 (the re-drive)
        #   (c) the attempt=0 record is preserved (append-only audit)
```

  Implement the body by copying the setup from `test_resume_does_not_reemit_completed_fork_branch`, then adding the node_states assertions:

```python
        with db.connection() as conn:
            attempts = [r.attempt for r in conn.execute(text(
                "SELECT attempt FROM node_states WHERE token_id=:tid ORDER BY attempt"),
                {"tid": incomplete_token_id}).fetchall()]
        assert 0 in attempts, "run-1 node_state (attempt 0) must be preserved (append-only)"
        assert 1 in attempts, "resume re-drive must record at attempt 1, not collide at 0"
```

- [ ] **Step 2: Run — confirm it PASSES** (Task 3+5 already implemented the offset).

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_fork_sink_with_orphan_open_node_state" -v`
Expected: PASS. (If it FAILS with a UniqueConstraint/IntegrityError, the offset threading in Task 3 missed the sink begin_node_state at sink.py:453 — fix that site.)

- [ ] **Step 3: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: resume fork->sink coexists with orphan open node_state at bumped attempt (F1)"
```

### Task 7: fork→coalesce resume regression

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: Write the failing test** — a fork where one branch routes to a coalesce, interrupt before the barrier, resume:

```python
    def test_resume_fork_to_coalesce_incomplete_branch(self) -> None:
        """A fork->coalesce branch interrupted before the barrier resumes from
        branch_first_node, merges normally, and yields the baseline outcome set."""
        # Build: gate forks to [branch routed to a coalesce 'merge', sink_b].
        # Use CoalesceSettings to wire 'merge' -> sink_a. One PassTransform on the
        # coalesce branch so branch_first_node != coalesce node (covers the
        # transform-branch path, not just identity).
        # Run, then delete the coalesce branch child's COALESCED outcome to mark it
        # incomplete. Checkpoint, mark failed, resume.
        # Assert the full conservation law + status COMPLETED + fork-group shape pin.
```

  Wire the coalesce using `CoalesceSettings` (already imported, line 30) and `wire_transforms` (line 39). Model the shape on the existing coalesce tests in this file (`rg -n "CoalesceSettings\(" tests/property/audit/test_fork_join_balance.py`).

- [ ] **Step 2: Run — expect PASS** (mechanism already built). If the re-drive collides on an intermediate transform's node_state, the offset must also reach the transform path (Task 3, transform.py:273) — verify.

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_fork_to_coalesce_incomplete_branch" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: resume fork->coalesce incomplete branch completes via barrier (F1)"
```

---

## Phase 3 — Schema change (`token_data_ref`, epoch bump, F3 co-fix)

### Task 8: Add `token_data_ref`, bump epoch, extend `_REQUIRED_COLUMNS`

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py:60`, `:210-224`
- Modify: `src/elspeth/core/landscape/database.py:32-45`
- Test: `tests/unit/landscape/test_schema_epoch_and_required_columns.py` (new)

- [ ] **Step 1: Write the failing test** (new file):

```python
"""Schema epoch + required-columns guards for the token_data_ref change (epoch 11)."""
from __future__ import annotations

import pytest
from sqlalchemy import inspect

from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, tokens_table
from elspeth.core.landscape.database import _REQUIRED_COLUMNS


def test_epoch_is_eleven() -> None:
    assert SQLITE_SCHEMA_EPOCH == 11


def test_tokens_has_token_data_ref_column() -> None:
    assert "token_data_ref" in tokens_table.c


def test_required_columns_include_token_data_ref_and_openrouter() -> None:
    required = set(_REQUIRED_COLUMNS)
    assert ("tokens", "token_data_ref") in required
    # F3 co-fix: the openrouter catalog columns (added at epoch 10) were never
    # added to the Postgres staleness backstop.
    assert ("runs", "openrouter_catalog_sha256") in required
    assert ("runs", "openrouter_catalog_source") in required
```

- [ ] **Step 2: Run — fails**

Run: `.venv/bin/python -m pytest tests/unit/landscape/test_schema_epoch_and_required_columns.py -v`
Expected: FAIL — epoch is 10; column/required entries missing.

- [ ] **Step 3: Add the column** (`schema.py:218`, inside `tokens_table`, after `expand_group_id`):

```python
    Column("branch_name", String(64)),
    # Payload-store ref for a token whose row_data differs from its source row
    # (expand/deaggregation children carry independently-transformed data). NULL
    # for fork children, which share the parent/source payload retrievable by
    # row_id. Enables faithful reconstruction of incomplete expand children on
    # resume (see resume fork/expand re-emit fix, epoch 11).
    Column("token_data_ref", String(64), nullable=True),
    Column("step_in_pipeline", Integer),  # Step where this token was created (fork/coalesce/expand)
```

- [ ] **Step 4: Bump the epoch** (`schema.py:57-60`) — add an `# 11 ->` comment line and set the constant:

```python
#  11 → resume fork/expand re-emit fix: tokens.token_data_ref persists per-token
#        output payloads so incomplete expand children are reconstructable on resume.
SQLITE_SCHEMA_EPOCH = 11
```

- [ ] **Step 5: Extend `_REQUIRED_COLUMNS`** (`database.py`, after the `("tokens","run_id")` entry ~line 47):

```python
    ("tokens", "token_data_ref"),
    # F3 co-fix: the OpenRouter catalog columns (epoch 10) were never added to the
    # Postgres staleness backstop, so a stale Postgres DB would slip past validation.
    ("runs", "openrouter_catalog_sha256"),
    ("runs", "openrouter_catalog_source"),
```

- [ ] **Step 6: Run the new test — passes**

Run: `.venv/bin/python -m pytest tests/unit/landscape/test_schema_epoch_and_required_columns.py -v`
Expected: PASS

- [ ] **Step 7: Run the landscape DB suite — confirm epoch gate + fresh-DB creation still pass**

Run: `.venv/bin/python -m pytest tests/unit/landscape tests/integration -k "schema or epoch or database or resume" -q`
Expected: PASS. (Existing tests that hardcode epoch 10, if any, must be updated to 11 — these are pinning tests of the constant, update them: `rg -rn "epoch.*10|== 10" tests/ | rg -i epoch`.)

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/core/landscape/schema.py src/elspeth/core/landscape/database.py tests/unit/landscape/test_schema_epoch_and_required_columns.py
git commit -m "feat(schema): tokens.token_data_ref + epoch 11 + F3 _REQUIRED_COLUMNS co-fix"
```

---

## Phase 4 — Expand persistence + resume

### Task 9: Persist per-child payload in `expand_token`

The expanded per-child payloads live in `tokens.py` (`expanded_rows`) and are created *after* `expand_token` returns. To persist `token_data_ref` atomically with the child INSERT, the payloads must flow **into** `expand_token`.

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:749-869` (`expand_token`)
- Modify: `src/elspeth/engine/tokens.py:366-394`
- Test: `tests/property/audit/test_fork_join_balance.py` (round-trip test)

- [ ] **Step 1: Write the failing test**

```python
    def test_expand_token_persists_per_child_payload(self) -> None:
        """expand_token stores each child's payload and writes tokens.token_data_ref."""
        from elspeth.core.landscape.data_flow_repository import DataFlowRepository  # adjust import
        from elspeth.core.landscape.schema import tokens_table
        from sqlalchemy import select
        # Build a minimal recorder with a MockPayloadStore, create a run+row+parent
        # token, then expand into 2 children with distinct payloads.
        # Assert: each child row has a non-null token_data_ref, and
        # payload_store.retrieve(ref) round-trips the child's payload bytes.
```

  Model construction on the existing recorder tests: `rg -n "DataFlowRepository\(|expand_token\(" tests/`.

- [ ] **Step 2: Run — fails** (expand_token takes `count`, writes no ref).

- [ ] **Step 3: Change `expand_token` signature** (data_flow_repository.py:749) from `count: int` to `child_payloads: Sequence[Mapping[str, object]]`, derive `count = len(child_payloads)`, and inside the child loop store each payload and write the ref:

```python
    def expand_token(
        self,
        parent_ref: TokenRef,
        row_id: str,
        child_payloads: Sequence[Mapping[str, object]],
        *,
        step_in_pipeline: int | None = None,
        record_parent_outcome: bool = True,
    ) -> tuple[list[Token], str]:
        count = len(child_payloads)
        if count < 1:
            raise ValueError("expand_token requires at least 1 child payload")
        self._validate_token_run_ownership(parent_ref)
        self._validate_token_row_ownership(parent_ref.token_id, row_id)
        expand_group_id = generate_id()
        children = []
        with self._db.connection() as conn:
            for ordinal, payload in enumerate(child_payloads):
                child_id = generate_id()
                timestamp = now()
                # Tier-1 audit write: persist this child's transformed payload so the
                # token is reconstructable on resume. Crash on store failure — no
                # best-effort. NULL token_data_ref is reserved for fork children.
                token_data_ref: str | None = None
                if self._payload_store is not None:
                    payload_bytes = canonical_json(payload).encode("utf-8")
                    token_data_ref = self._payload_store.store(payload_bytes)
                result = conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        run_id=parent_ref.run_id,
                        expand_group_id=expand_group_id,
                        token_data_ref=token_data_ref,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"expand_token: child token INSERT affected zero rows (token_id={child_id}, ordinal={ordinal})"
                    )
                # ... token_parents INSERT unchanged (lines 821-831) ...
                children.append(
                    Token(
                        token_id=child_id, row_id=row_id, expand_group_id=expand_group_id,
                        token_data_ref=token_data_ref, step_in_pipeline=step_in_pipeline,
                        created_at=timestamp, run_id=parent_ref.run_id,
                    )
                )
            # ... record_parent_outcome block unchanged (lines 850-869) ...
        return children, expand_group_id
```

  Add `token_data_ref: str | None = None` to the `Token` model (`rg -n "class Token\b" src/elspeth/core/landscape/`). Confirm `canonical_json`, `Sequence`, `Mapping` are imported (canonical_json is — used at create_row:439).

- [ ] **Step 4: Update the caller** (`tokens.py:366-374`). `expand_token` now needs the payload dicts. The caller has `expanded_rows` (the deaggregated rows). Pass them:

```python
        db_children, expand_group_id = self._data_flow.expand_token(
            parent_ref=TokenRef(token_id=parent_token.token_id, run_id=run_id),
            row_id=parent_token.row_id,
            child_payloads=[dict(r) for r in expanded_rows],
            step_in_pipeline=step,
            record_parent_outcome=record_parent_outcome,
        )
```

  Keep the existing `child_infos` construction (tokens.py:383-393) — it wraps each `expanded_rows` element in a deepcopy'd `PipelineRow` with `output_contract`. The persisted bytes must equal what `child_infos` carries, so persist `expanded_rows[i]` (pre-deepcopy is fine; deepcopy only guards in-memory sharing). Confirm `zip(db_children, expanded_rows, strict=True)` still aligns (same order, same count).

- [ ] **Step 5: Update all other `expand_token` call sites.** Find them: `rg -n "expand_token\(" src/elspeth tests`. The batch-aggregation path may also call it — change `count=` to `child_payloads=`. (No legacy shim — change every call site in this commit per the No-Legacy policy.)

- [ ] **Step 6: Run the round-trip test + the deaggregation suite**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_expand_token_persists_per_child_payload" tests/ -k "expand or deagg" -q`
Expected: PASS

- [ ] **Step 7: Add `tokens.token_data_ref` to the recovery SELECT** — in Task 2's `get_incomplete_tokens_by_row`, restore the `tokens_table.c.token_data_ref` column in the SELECT and set `token_data_ref=r.token_data_ref` in the spec (it was omitted in Phase 2). Run Task 2's test again to confirm still GREEN.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/core/landscape/data_flow_repository.py src/elspeth/engine/tokens.py src/elspeth/core/checkpoint/recovery.py tests/property/audit/test_fork_join_balance.py
git commit -m "feat(audit): expand_token persists per-child payload to token_data_ref (epoch 11)"
```

### Task 10: Expand-child resume reconstruction

**Files:**
- Modify: `src/elspeth/engine/processor.py` (`resume_incomplete_token` expand branch)
- Modify: `src/elspeth/engine/orchestrator/resume.py` (build expand-child payload + contract)
- Test: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: Write the failing test** — partial expand resume, full conservation law:

```python
    def test_resume_partial_expand_does_not_reemit(self) -> None:
        """A deaggregation row with one completed and one incomplete expanded child
        resumes the incomplete child only — no re-expansion, no double-emit."""
        # Build a pipeline with a deaggregation transform that expands one source row
        # into 2 children, each landing at a sink. Run, delete ONE child's terminal
        # outcome, checkpoint, mark failed, resume.
        # Assert full conservation law: after == baseline, zero orphan, one terminal
        # per expanded child, expanded-child count unchanged, status COMPLETED.
```

  Use the existing deaggregation fixtures (`rg -n "deagg|Deaggregat|expand" tests/fixtures/`).

- [ ] **Step 2: Run — fails** (expand branch raises `OrchestrationInvariantError` from Task 4 Step 3).

- [ ] **Step 3: Resolve the expand-child resume-start node + payload + contract.** In `resume.py`, when a spec has `expand_group_id is not None`, build the child's `PipelineRow` from its own payload and the expand step's output contract:

```python
        # Expand child: its payload differs from the source row and is persisted in
        # token_data_ref; its contract is the expand step's output_contract,
        # recoverable from nodes.output_contract_json.
        if spec.token_data_ref is None:
            raise OrchestrationInvariantError(
                f"Expand child {spec.token_id} has no token_data_ref — unreconstructable. "
                f"(Pre-epoch-11 data, or a recorder bug.)"
            )
        child_bytes = payload_store.retrieve(spec.token_data_ref)
        child_data = json.loads(child_bytes.decode("utf-8"))
        child_contract = recovery_manager.get_expand_child_contract(run_id, spec)  # see Step 4
        child_row = PipelineRow(data=child_data, contract=child_contract)
        results = processor.resume_incomplete_token(spec, child_row, ctx)
```

  Pass `child_row` instead of the source `pipeline_row` for expand children. (Fork children keep using `pipeline_row` from the source payload.)

- [ ] **Step 4: Add `get_expand_child_contract`** to `RecoveryManager` (recovery.py) — reuse the existing `output_contract_json` read path (`ContractAuditRecord.from_json`, data_flow_repository.py:1350-1351):

```python
    def get_expand_child_contract(self, run_id: str, spec: IncompleteTokenSpec) -> SchemaContract:
        """Reconstruct the output contract under which an expand child was created.

        Reads nodes.output_contract_json for the expand step that produced the
        child (resolved from the child's step_in_pipeline / expand_group_id).
        """
        # Resolve the expand node for spec.expand_group_id, load its
        # output_contract_json, and return ContractAuditRecord.from_json(...).to_contract().
        # Confirm the exact reader: rg -n "output_contract_json|ContractAuditRecord" src/elspeth/core/landscape/data_flow_repository.py
```

  Implement using the existing reader at data_flow_repository.py:1332-1351. If the expand node id is not directly on the token, derive it via `step_in_pipeline` → node (the recorder already maps steps to nodes; `rg -n "step_in_pipeline|step_resolver" src/elspeth/core/landscape/`).

- [ ] **Step 5: Wire the expand branch in `resume_incomplete_token`** (processor.py) — replace the Task-4 placeholder raise. The expand child re-drives from the node **after** the expand step (its `current_node_id` = the node that follows the expand transform), with `resume_attempt_offset=offset`. Resolve "node after expand step" via the navigator's `resolve_next_node` on the expand node:

```python
        if spec.expand_group_id is not None:
            next_node = self._nav.resolve_next_node(self._resolve_expand_node(spec))
            return self.process_token(token, ctx, current_node_id=next_node, resume_attempt_offset=offset)
```

  Add `_resolve_expand_node(spec)` (maps `expand_group_id`/`step_in_pipeline` → expand node id). If `next_node is None` the expand fed straight into a sink — pass `current_node_id=None` and rely on `on_success_sink` (confirm the deaggregation terminal path at `_process_single_token`).

- [ ] **Step 6: Run — GREEN**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_partial_expand_does_not_reemit" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/engine/processor.py src/elspeth/engine/orchestrator/resume.py src/elspeth/core/checkpoint/recovery.py tests/property/audit/test_fork_join_balance.py
git commit -m "feat(resume): reconstruct + re-drive incomplete expand children (epoch 11)"
```

---

## Phase 5 — Counter reconciliation, matrix completion, gates

### Task 11: F1/F2 counter-reconciliation assertion

Approach 1 changes the set/shape of `RowResult`s entering `accumulate_row_outcomes` (resume.py:204). Guard against a new `rows_processed` divergence.

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: Write the failing/guard test** — assert resumed counters + already-recorded run-1 counters reconcile to a single uninterrupted from-source run:

```python
    def test_resume_counters_reconcile_with_uninterrupted_run(self) -> None:
        """rows_succeeded/terminal-outcome totals after (run1 + resume) equal a
        single uninterrupted run of the same fork pipeline."""
        # Run pipeline A to completion uninterrupted -> record terminal-outcome totals.
        # Run pipeline B identically, interrupt one fork branch, resume.
        # Assert B's post-resume terminal-outcome multiset == A's.
        # (Reuses _outcome_counts; the multiset equality IS the reconciliation.)
```

- [ ] **Step 2: Run.** Expected PASS if Task 5 conserves outcomes. If it FAILS, the resume per-leaf `rows_processed` counting (F2) diverges — reconcile by counting per-source-row on the reconstruction path, not per-leaf. (Read `accumulate_row_outcomes` and `counters.rows_processed` semantics: `rg -n "rows_processed|def accumulate_row_outcomes" src/elspeth/engine/orchestrator/`.)

- [ ] **Step 3: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: resume counters reconcile with uninterrupted run (F1/F2 coupling)"
```

### Task 12: Remaining risk-ordered matrix

Add one test per remaining spec matrix item (spec §"Required test matrix"). Each follows the same RED→GREEN→commit micro-cycle; the mechanism is already built, so most should pass on first run (any failure localizes a real gap).

- [ ] **Step 1: matrix #2** — `test_resume_all_branches_incomplete_uses_restart`: a fork row where *no* branch completed (no terminal child outcomes) → `get_incomplete_tokens_by_row` returns the children, and re-driving each is still correct (or, if no tokens at all, restart). Assert conservation law.
- [ ] **Step 2: matrix #3** — `test_resume_three_way_fork_two_incomplete`: 1 done, 2 incomplete; assert deterministic completion (order by `step_in_pipeline`/`token_id`) and conservation law.
- [ ] **Step 3: matrix #4** — `test_resume_failure_during_resumed_branch_yields_failed_not_orphan`: make the resumed branch's sink fail → assert a FAILED terminal outcome, zero orphan.
- [ ] **Step 4: matrix #5** — `test_resume_aggregation_buffer_with_partial_fork`: mixed buffered + incomplete tokens on one row (exercises recovery.py:449-473 exclusion) → conservation law.
- [ ] **Step 5: matrix #6** — `test_resume_linear_pipeline_regression_audit`: a linear (no-fork) pipeline resume still uses `process_existing_row` and is correct, asserted on the **audit trail** (not just in-memory sink). Confirms the no-tokens dispatch branch.
- [ ] **Step 6: matrix #7** — `test_resume_of_resume_converges`: resume, then resume again → incomplete-leaf stock is monotonic non-increasing and reaches 0; idempotent.
- [ ] **Step 7: Run the whole new matrix**

Run: `.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant -v`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: complete risk-ordered resume fork/expand matrix (F1)"
```

### Task 13: Full gate run

- [ ] **Step 1: Targeted then full suite** (per memory: plain `pytest tests/` is the CI-equivalent selection — do NOT pass `-o addopts=""`):

```bash
.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py tests/e2e/recovery tests/unit/landscape -q
.venv/bin/python -m pytest tests/ -q
```
Expected: PASS (no new failures vs. the 8 known pre-existing RC5.2 failures — see memory `project_rc52_preexisting_test_failures_2026-05-29`).

- [ ] **Step 2: Type + lint**

```bash
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
```
Expected: clean.

- [ ] **Step 3: Tier-model lint (Python 3.13 venv — must match main)**

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```
Expected: pass. The new L2→L1 import (`processor.py` → `recovery.IncompleteTokenSpec`) is downward and layer-legal; if any allowlist fingerprint rotates due to body-index shifts, co-land the `web.yaml` fp update in this commit (see memory `feedback_ast_shift_fingerprint_rotation`).

- [ ] **Step 4: Freeze-guard gate** (WorkItem and IncompleteTokenSpec are frozen dataclasses; all fields are scalars/None, so no `deep_freeze` guard is required — verify the gate agrees):

```bash
.venv/bin/python scripts/cicd/enforce_freeze_guards.py
```
Expected: pass.

- [ ] **Step 5: Plugin hash gate** (only if any plugin file changed — none expected; if `expand_token` callers in plugins changed, refresh):

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules ... # plugin hash check
```

- [ ] **Step 6: Final commit if any gate reconciliation was needed**

```bash
git add -A && git commit -m "chore: reconcile gates for resume fork/expand fix (F1)"
```

---

## Done criteria

- The RED reproduction and all matrix tests are GREEN.
- The conservation law holds for fork→sink, fork→coalesce, and partial-expand resume.
- Epoch is 11; `_REQUIRED_COLUMNS` includes `token_data_ref` + the two openrouter columns (F3).
- Append-only audit preserved (run-1 node_states untouched; re-drive at bumped attempt).
- mypy / ruff / tier-model / freeze gates clean; full `pytest tests/` shows no new failures.
- Merge to RC5.2 with `--no-ff` (operator approval for the merge; bare push is non-destructive). Operator must delete the staging audit DB before next staging run (epoch 11 incompatibility) — surface this as an OPERATOR ACTION before any staging deploy.
