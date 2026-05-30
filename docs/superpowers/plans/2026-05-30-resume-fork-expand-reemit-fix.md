# Resume Fork/Expand Re-emit Fix (F1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop checkpoint resume from re-emitting already-completed fork/expand/coalesce branches (duplicate terminal outcomes + duplicate physical sink writes) by driving each *incomplete* child token to completion in place instead of restarting the whole row from source — while keeping every re-driven `node_states` record attributable to the resume that produced it.

**Architecture:** On resume, a row with partially-completed fork/expand/coalesce children is no longer restarted from source (which re-forks to *all* branches). Instead, `RecoveryManager` returns the incomplete non-delegation child tokens; `RowProcessor` reconstructs each as a `TokenInfo` from persisted columns + payload and re-drives it from its correct mid-DAG node — under the *original* parent — via the existing `process_token` machinery. To coexist with run-1's append-only `node_states` records, the re-drive records at an elevated `attempt` AND stamps a `resume_checkpoint_id` provenance marker, both carried on `WorkItem`. Tokens whose payload differs from their source row (expand children, post-coalesce merged tokens) are reconstructable via a new `tokens.token_data_ref` column written at expand/coalesce time.

**Tech Stack:** Python 3.13, SQLAlchemy Core, SQLite audit DB (epoch-gated), pluggy, pytest + Hypothesis. Worktree: `/home/john/elspeth/.worktrees/fix-resume-fork-reemit` (branch `fix/resume-fork-reemit`, own `.venv` Python 3.13). Run tests with `.venv/bin/python -m pytest`.

**Authoritative spec:** `docs/superpowers/specs/2026-05-30-resume-partial-fork-reemit-fix-design.md` — read REVISED DESIGN **and ADDENDUM 1 and ADDENDUM 2** before starting; the addenda refine the body and win on conflict.

> **Line-number discipline:** cited `file.py:NNN` ranges are approximate (the tree drifts). **Locate every edit site by symbol** (`rg -n "def <name>|class <name>"`), not by line number. The reality review confirmed all referenced *symbols* exist; only some line citations have drifted.

---

## Build order & milestones

Provenance (a `node_states` column written by the *first* re-drive) forces **schema-first** — the original "fix the bug before any schema change" ordering is no longer possible (spec ADDENDUM 2.D). The phases are TDD increments within one branch; the whole fix merges to RC5.2 as one unit.

- **Phase 1** — harden the RED reproduction's oracle to the full conservation law (still RED).
- **Phase 2** — schema: `tokens.token_data_ref` + `node_states.resume_checkpoint_id`, epoch 10→11, `_REQUIRED_COLUMNS` (+F3 co-fix), and `begin_node_state` accepts/writes the provenance marker.
- **Phase 3** — payload persistence: `expand_token` (per child) **and** `coalesce_tokens` (merged) write `token_data_ref`; migrate every test call site of both (No-Legacy).
- **Phase 4** — resume core: incomplete-token selection + `WorkItem` offset/provenance threading + generic reconstruction + `resume_incomplete_token` dispatch (fork→sink, fork→coalesce before-barrier, post-coalesce after-barrier, expand) → **RED test GREEN**, plus the regression/attributability cells.
- **Phase 5** — F2 counter-field reconciliation + remaining risk-ordered matrix + full gate run.

## File structure (created / modified)

- Modify `src/elspeth/core/landscape/schema.py` — add `tokens.token_data_ref` + `node_states.resume_checkpoint_id`; bump `SQLITE_SCHEMA_EPOCH` 10→11 (Phase 2).
- Modify `src/elspeth/core/landscape/database.py` — extend `_REQUIRED_COLUMNS` (Phase 2, +F3).
- Modify `src/elspeth/core/landscape/execution_repository.py` — `begin_node_state` accepts + writes `resume_checkpoint_id` (Phase 2).
- Modify `src/elspeth/core/landscape/data_flow_repository.py` — `expand_token` + `coalesce_tokens` per-token payload persistence; generic contract read for reconstruction (Phase 3/4).
- Modify `src/elspeth/contracts/audit.py` — `Token.token_data_ref` field (Phase 3).
- Modify `src/elspeth/engine/tokens.py` — pass per-child payloads into `expand_token`; pass merged payload into `coalesce_tokens` (Phase 3).
- Modify `src/elspeth/core/checkpoint/recovery.py` — `IncompleteTokenSpec` + `get_incomplete_tokens_by_row()` + `reconstruct_token_row()` (Phase 4).
- Modify `src/elspeth/engine/dag_navigator.py` — `WorkItem.resume_attempt_offset` + `WorkItem.resume_checkpoint_id`; `resolve_branch_first_node()`, `resolve_node_after()` accessors (Phase 4).
- Modify `src/elspeth/engine/processor.py` — `resume_incomplete_token()`; relax `process_token` `current_node_id` to `NodeID | None`; thread offset + provenance through `_process_single_token` → handlers (Phase 4).
- Modify `src/elspeth/engine/executors/sink.py`, `transform.py` (via `state_guard`), `coalesce_executor.py` — add offset + provenance to `begin_node_state` calls (Phase 4).
- Modify `src/elspeth/engine/orchestrator/resume.py` + `core.py` — dispatch: no-tokens → `process_existing_row`; incomplete-tokens → reconstruct + `resume_incomplete_token` (Phase 4).
- Modify `tests/property/audit/test_fork_join_balance.py` — oracle + matrix (Phases 1, 4, 5).
- Create `tests/unit/landscape/test_schema_epoch_and_required_columns.py` (Phase 2).
- Migrate test call sites of `expand_token` / `coalesce_tokens` (Phase 3) — enumerated in Task 4.

---

## Phase 1 — Harden the reproduction oracle (stays RED)

The existing `test_resume_does_not_reemit_completed_fork_branch` (`rg -n "def test_resume_does_not_reemit_completed_fork_branch" tests/property/audit/test_fork_join_balance.py`) checks only `n != 1`, which is **blind to the orphan failure mode** (a branch with zero outcomes) and discards `baseline` and the resume result status. Encode the full conservation law before fixing, so the GREEN bar is correct.

### Task 1: Strengthen the reproduction oracle

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py` (the assertion block of `test_resume_does_not_reemit_completed_fork_branch`)

- [ ] **Step 1: Add a zero-orphan-leaf helper near the other audit helpers** (after `get_fork_group_stats`)

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

> Verify the `TerminalPath` enum member names: `rg -n "FORK_PARENT|EXPAND_PARENT|class TerminalPath" src/elspeth/contracts/enums.py`.

- [ ] **Step 2: Replace the final assertion block with the full conservation law**

```python
        # CONSERVATION LAW (spec ADDENDUM 1 test oracle):
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

- [ ] **Step 3: Capture `baseline_fork_stats` and the resume result** — right after `baseline = _outcome_counts()` add:

```python
        baseline_fork_stats = get_fork_group_stats(db, run_id)
```

  and capture the resume return value:

```python
        resume_result = resume_orchestrator.resume(
            resume_point, config, graph, payload_store=payload_store, settings=settings_obj
        )
```

- [ ] **Step 4: Add the `RunStatus` import** to the test's local imports:

```python
        from elspeth.contracts.enums import RunStatus
```

  Verify: `rg -n "class RunStatus|COMPLETED" src/elspeth/contracts/enums.py`. Note `RunStatus.COMPLETED_WITH_FAILURES` also exists and `RunResult` enforces a status↔row-count biconditional — a clean fork resume should be `COMPLETED`; if a resumed leaf is counted as a failure/quarantine the assertion will (correctly) fail. Confirm the result attribute: `rg -n "class RunResult|status" src/elspeth/engine/orchestrator/*.py`.

- [ ] **Step 5: Run the test — confirm it still FAILS, now on the conservation law**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_does_not_reemit_completed_fork_branch" -v`
Expected: FAIL — `after != baseline` and/or `sink_b.collected` length 2 (double-emit). The failure proves the oracle detects the defect.

- [ ] **Step 6: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: harden resume fork re-emit oracle to full conservation law (F1)"
```

---

## Phase 2 — Schema (`token_data_ref`, `resume_checkpoint_id`, epoch bump, F3 co-fix)

### Task 2: Add both columns, bump epoch, extend `_REQUIRED_COLUMNS`, write provenance in `begin_node_state`

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` (`tokens_table`, `node_states_table`, `SQLITE_SCHEMA_EPOCH`)
- Modify: `src/elspeth/core/landscape/database.py` (`_REQUIRED_COLUMNS`)
- Modify: `src/elspeth/core/landscape/execution_repository.py` (`begin_node_state`)
- Test: `tests/unit/landscape/test_schema_epoch_and_required_columns.py` (new)

- [ ] **Step 1: Write the failing test** (new file):

```python
"""Schema epoch + required-columns + provenance-write guards (epoch 11)."""
from __future__ import annotations

from elspeth.core.landscape.schema import (
    SQLITE_SCHEMA_EPOCH,
    tokens_table,
    node_states_table,
)
from elspeth.core.landscape.database import _REQUIRED_COLUMNS


def test_epoch_is_eleven() -> None:
    assert SQLITE_SCHEMA_EPOCH == 11


def test_tokens_has_token_data_ref_column() -> None:
    assert "token_data_ref" in tokens_table.c


def test_node_states_has_resume_checkpoint_id_column() -> None:
    assert "resume_checkpoint_id" in node_states_table.c


def test_required_columns_include_new_columns_and_openrouter() -> None:
    required = set(_REQUIRED_COLUMNS)
    assert ("tokens", "token_data_ref") in required
    assert ("node_states", "resume_checkpoint_id") in required
    # F3 co-fix: the openrouter catalog columns (added at epoch 10) were never
    # added to the Postgres staleness backstop.
    assert ("runs", "openrouter_catalog_sha256") in required
    assert ("runs", "openrouter_catalog_source") in required
```

- [ ] **Step 2: Run — fails** (epoch is 10; columns/required entries missing).

Run: `.venv/bin/python -m pytest tests/unit/landscape/test_schema_epoch_and_required_columns.py -v`

- [ ] **Step 3: Add `tokens.token_data_ref`** (in `tokens_table`, after `branch_name`):

```python
    Column("branch_name", String(64)),
    # Payload-store ref for a token whose row_data differs from its source row:
    # expand/deaggregation children (independently-transformed data) AND post-coalesce
    # merged tokens (the merged row, computed in memory at barrier time). NULL for fork
    # children, which share the parent/source payload retrievable by row_id. Enables
    # faithful reconstruction of incomplete expand/coalesce tokens on resume (epoch 11).
    Column("token_data_ref", String(64), nullable=True),
    Column("step_in_pipeline", Integer),  # Step where this token was created (fork/coalesce/expand)
```

- [ ] **Step 4: Add `node_states.resume_checkpoint_id`** (in `node_states_table`, after `completed_at`, before the constraints):

```python
    Column("completed_at", DateTime(timezone=True)),
    # Resume provenance marker (epoch 11): NULL for every node_state written during the
    # original run; set to the resumed-from checkpoint id for every node_state written
    # while re-driving a reconstructed incomplete token on resume. Makes a resume re-drive
    # (which records at attempt = max+1 under the SAME run_id) provably distinguishable
    # from a run-1 tenacity retry — explain() filters on resume_checkpoint_id IS NULL.
    Column("resume_checkpoint_id", String(64), ForeignKey("checkpoints.checkpoint_id"), nullable=True),
```

> `ForeignKey` and `checkpoints_table` are already in this module (`checkpoints_table` defined later in the file; `ForeignKey` already imported — confirm with `rg -n "^from sqlalchemy|ForeignKey" src/elspeth/core/landscape/schema.py`). The FK references `checkpoints.checkpoint_id` (PK). If a string FK target ordering issue arises (table defined after `node_states`), SQLAlchemy resolves string FK targets lazily at mapper/DDL time — no reorder needed.

- [ ] **Step 5: Bump the epoch** (add an `# 11 ->` comment line and set the constant):

```python
#  11 → resume fork/expand/coalesce re-emit fix: tokens.token_data_ref persists per-token
#        payloads (expand children + coalesce merged tokens) and node_states.resume_checkpoint_id
#        marks resume re-drives, so incomplete tokens are reconstructable + attributable.
SQLITE_SCHEMA_EPOCH = 11
```

- [ ] **Step 6: Extend `_REQUIRED_COLUMNS`** (`database.py`, after the existing `("tokens", ...)` entries — `rg -n "_REQUIRED_COLUMNS" src/elspeth/core/landscape/database.py`):

```python
    ("tokens", "token_data_ref"),
    ("node_states", "resume_checkpoint_id"),
    # F3 co-fix: the OpenRouter catalog columns (epoch 10) were never added to the
    # Postgres staleness backstop, so a stale Postgres DB would slip past validation.
    ("runs", "openrouter_catalog_sha256"),
    ("runs", "openrouter_catalog_source"),
```

- [ ] **Step 7: Make `begin_node_state` accept + write the provenance marker** (`execution_repository.py`, `rg -n "def begin_node_state" src/elspeth/core/landscape/execution_repository.py`). Add a keyword param and include it in the INSERT:

```python
    def begin_node_state(
        self,
        ...,
        attempt: int = 0,
        resume_checkpoint_id: str | None = None,
    ) -> ...:
        ...
        # in the node_states INSERT .values(...):
        resume_checkpoint_id=resume_checkpoint_id,
```

  Default `None` preserves every existing caller (all run-1 writes) unchanged. Read the existing signature + INSERT and add the one param + one value. Add a focused unit assertion to the new test file that a `begin_node_state(..., resume_checkpoint_id="ck1")` row reads back with that value (construct a minimal recorder; model on `rg -n "begin_node_state\(" tests/`).

- [ ] **Step 8: Run the new test + the landscape DB suite**

Run: `.venv/bin/python -m pytest tests/unit/landscape/test_schema_epoch_and_required_columns.py tests/unit/landscape tests/integration -k "schema or epoch or database or resume or node_state" -q`
Expected: PASS. Any test hardcoding epoch 10 is a pinning test of the constant — update to 11. Find them: `rg -rn "SQLITE_SCHEMA_EPOCH|epoch.{0,4}10|== 10" tests/ | rg -i epoch`.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/core/landscape/schema.py src/elspeth/core/landscape/database.py src/elspeth/core/landscape/execution_repository.py tests/unit/landscape/test_schema_epoch_and_required_columns.py
git commit -m "feat(schema): token_data_ref + node_states.resume_checkpoint_id + epoch 11 + F3 (_REQUIRED_COLUMNS)"
```

---

## Phase 3 — Payload persistence (`expand_token` + `coalesce_tokens`)

`token_data_ref` has three writers: `expand_token` (per child) and `coalesce_tokens` (merged) set it; fork children leave it NULL. Both persist with the type-faithful `checkpoint_dumps` (Tier-1 — `canonical_json` would stringify `datetime`/`Decimal`). The payload dicts must flow **into** these recorders so the ref is written atomically with the token INSERT.

### Task 3: Persist per-token payloads + add `Token.token_data_ref`

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py` (`expand_token`, `coalesce_tokens`)
- Modify: `src/elspeth/contracts/audit.py` (`Token` model — `rg -n "class Token\b" src/elspeth/contracts/audit.py`)
- Test: `tests/property/audit/test_fork_join_balance.py` (round-trip tests)

- [ ] **Step 1: Add `token_data_ref` to the `Token` model** (`contracts/audit.py`, ~line 180). `Token` is a frozen, slotted, all-scalar dataclass → no `deep_freeze` guard needed:

```python
    token_data_ref: str | None = None
```

- [ ] **Step 2: Write the failing round-trip tests** (in `TestForkRecoveryInvariant`):

```python
    def test_expand_token_persists_per_child_payload(self) -> None:
        """expand_token stores each child's payload and writes tokens.token_data_ref."""
        # Build a minimal recorder with a MockPayloadStore, create run+row+parent token,
        # expand into 2 children with DISTINCT payloads. Assert each child row has a
        # non-null token_data_ref and payload_store.retrieve(ref) round-trips that child's
        # payload via checkpoint_loads.

    def test_coalesce_token_persists_merged_payload(self) -> None:
        """coalesce_tokens stores the merged payload and writes tokens.token_data_ref."""
        # Create run+row+two parent branch tokens, coalesce them with a merged payload.
        # Assert the merged token row has a non-null token_data_ref and the bytes
        # round-trip the merged payload via checkpoint_loads.
```

  Model construction on existing recorder tests: `rg -n "DataFlowRepository\(|\.expand_token\(|\.coalesce_tokens\(" tests/`.

  > **MockPayloadStore must serialize bytes** (store→retrieve round-trips actual bytes, not store-by-reference) or the type-fidelity check is theatre. Verify/strengthen: `rg -n "class MockPayloadStore" tests/`. If it stores objects by reference, fix it to keep `bytes`.

- [ ] **Step 3: Run — fails** (signatures take `count` / no merged payload; no ref written).

- [ ] **Step 4: Change `expand_token`** (`rg -n "def expand_token" src/elspeth/core/landscape/data_flow_repository.py`) from `count: int` to `child_payloads: Sequence[Mapping[str, object]]`; derive `count`; store each payload, write the ref:

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
        ...
        for ordinal, payload in enumerate(child_payloads):
            child_id = generate_id()
            # Tier-1 audit write: persist the child's transformed payload so the token is
            # reconstructable on resume. checkpoint_dumps (NOT canonical_json) preserves
            # datetime/Decimal type fidelity. Crash on store failure — no best-effort.
            token_data_ref = self._payload_store.store(checkpoint_dumps(payload).encode("utf-8"))
            # ... tokens_table.insert().values(..., token_data_ref=token_data_ref, ...)
            # ... Token(..., token_data_ref=token_data_ref, ...)
```

  `self._payload_store` is required for expand now (an expand child with no payload store is unreconstructable) — if it can be `None`, raise an `AuditIntegrityError` rather than writing NULL.

- [ ] **Step 5: Change `coalesce_tokens`** (`rg -n "def coalesce_tokens" src/elspeth/core/landscape/data_flow_repository.py`) to accept the merged payload and persist it:

```python
    def coalesce_tokens(
        self,
        parent_refs: list[TokenRef],
        row_id: str,
        merged_payload: Mapping[str, object],
        *,
        step_in_pipeline: int | None = None,
    ) -> Token:
        ...
        token_data_ref = self._payload_store.store(checkpoint_dumps(merged_payload).encode("utf-8"))
        # ... tokens_table.insert().values(..., token_data_ref=token_data_ref, ...)
        # ... return Token(..., token_data_ref=token_data_ref, ...)
```

  Add `from elspeth.core.checkpoint.serialization import checkpoint_dumps` (both `core/checkpoint` and `core/landscape` are L1 — layer-legal; confirm no cycle: `rg -n "import" src/elspeth/core/checkpoint/serialization.py | rg landscape`). Confirm `Sequence`, `Mapping` imported.

- [ ] **Step 6: Update the engine callers** (`tokens.py`):
  - `TokenManager.expand_token` (`rg -n "def expand_token" src/elspeth/engine/tokens.py`) — pass `child_payloads=[dict(r) for r in expanded_rows]` (it already has `expanded_rows`). Keep the existing `child_infos` deepcopy construction; the persisted bytes equal what `child_infos` carries (deepcopy only guards in-memory sharing). Confirm `zip(db_children, expanded_rows, strict=True)` still aligns.
  - The coalesce caller (`rg -n "\.coalesce_tokens\(" src/elspeth/engine`) — pass the merged row dict it computes as `merged_payload=...`.

- [ ] **Step 7: Migrate ALL test call sites (No-Legacy — same commit).** The signature changes break direct callers in the test suite. The recorder-level `expand_token` has exactly **one production caller** (`TokenManager.expand_token`), but **many test callers** that the original plan wrongly called "exactly one caller." Enumerate and migrate every `count=`→`child_payloads=` and every `coalesce_tokens(...)`→add `merged_payload=`:

```bash
rg -n "\.expand_token\(|\.coalesce_tokens\(" src/elspeth tests
```

  Known `expand_token(count=...)` test sites to convert (verify the list is complete with the grep above): `tests/unit/core/landscape/repository_integration/test_recorder_tokens.py`, `tests/unit/core/landscape/test_token_recording.py`, `tests/unit/core/landscape/test_data_flow_repository.py`, `tests/property/core/test_landscape_recording_properties.py`. Convert each `count=N` to `child_payloads=[{...}, ...]` with N representative dicts, and add `merged_payload={...}` to each `coalesce_tokens` call. Do NOT leave a compat shim accepting both.

- [ ] **Step 8: Run the round-trip tests + the deaggregation/coalesce suites**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_expand_token_persists_per_child_payload" "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_coalesce_token_persists_merged_payload" tests/ -k "expand or deagg or coalesce" -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/core/landscape/data_flow_repository.py src/elspeth/contracts/audit.py src/elspeth/engine/tokens.py tests/
git commit -m "feat(audit): expand_token + coalesce_tokens persist per-token payload to token_data_ref (epoch 11)"
```

---

## Phase 4 — Resume core (turns the RED test GREEN; all token kinds)

### Task 4: `IncompleteTokenSpec` + recovery selection (unfiltered) + reconstruction

Reuse the existing incomplete-token semantics and the shared `_DELEGATION_PATHS` predicate — do **not** write a second drifting completion query. The selection is **unfiltered**: it returns fork children, expand children, AND post-coalesce merged tokens. Every kind is dispatched in Task 6; filtering any kind out would drop it to `process_existing_row` → restart → reintroduce F1 (review finding B1).

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py`
- Test: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: Write the failing test**

```python
    def test_get_incomplete_tokens_by_row_returns_only_incomplete_leaf(self) -> None:
        """Recovery surfaces exactly the non-delegation tokens lacking a terminal outcome."""
        # Build the fork pipeline (gate forks each row to sink_a + sink_b), run to
        # completion, delete sink_a's terminal outcome so its child is the sole incomplete
        # leaf. (Model setup on test_resume_does_not_reemit_completed_fork_branch.)
        recovery = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery.get_incomplete_tokens_by_row(run.run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert [s.token_id for s in all_specs] == [incomplete_token_id]
        spec = all_specs[0]
        assert spec.branch_name == "sink_a"
        assert spec.fork_group_id is not None
        assert spec.token_data_ref is None          # fork child shares source payload
        assert spec.max_attempt >= 0                 # sink_a child wrote a node_state at attempt 0
```

- [ ] **Step 2: Run it — fails** (`AttributeError: ... 'get_incomplete_tokens_by_row'`).

- [ ] **Step 3: Add the dataclass** (near the `_DELEGATION_PATHS` constant):

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncompleteTokenSpec:
    """A non-delegation child token that lacks a terminal outcome on a resumed run.

    Identity fields read directly from persisted columns (Tier-1: no defaults, no
    coercion — TokenInfo.__post_init__ rejects garbage downstream). ``token_data_ref``
    is NULL for fork children (they share the parent/source payload, retrievable by
    ``row_id``) and set for expand children and post-coalesce merged tokens.
    ``max_attempt`` is the highest ``attempt`` already recorded for this token in
    ``node_states`` (-1 if none); the re-drive uses ``max_attempt + 1``.
    """

    token_id: str
    row_id: str
    branch_name: str | None
    fork_group_id: str | None
    join_group_id: str | None
    expand_group_id: str | None
    token_data_ref: str | None
    step_in_pipeline: int | None
    max_attempt: int
```

- [ ] **Step 4: Add `get_incomplete_tokens_by_row`** (after `get_unprocessed_rows`):

```python
    def get_incomplete_tokens_by_row(self, run_id: str) -> dict[str, list[IncompleteTokenSpec]]:
        """Return incomplete non-delegation child tokens, grouped by row_id.

        A token is incomplete when it is NOT a delegation marker (FORK_PARENT /
        EXPAND_PARENT) and has NO completed terminal outcome. Mirrors get_unprocessed_rows'
        completion semantics (shared _DELEGATION_PATHS) so recovery selection and resume
        reconstruction cannot drift apart. UNFILTERED: returns fork, expand, AND
        post-coalesce tokens — each is dispatched in resume_incomplete_token.
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
                    tokens_table.c.step_in_pipeline,
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
                    token_id=r.token_id, row_id=r.row_id, branch_name=r.branch_name,
                    fork_group_id=r.fork_group_id, join_group_id=r.join_group_id,
                    expand_group_id=r.expand_group_id, token_data_ref=r.token_data_ref,
                    step_in_pipeline=r.step_in_pipeline,
                    max_attempt=-1 if r.max_attempt is None else int(r.max_attempt),
                )
            )
        return by_row
```

- [ ] **Step 5: Add the generic reconstruction method** `reconstruct_token_row` after the selection:

```python
    def reconstruct_token_row(
        self,
        spec: IncompleteTokenSpec,
        run_id: str,
        source_row: PipelineRow,
        payload_store: PayloadStore,
    ) -> PipelineRow:
        """Build the PipelineRow to re-drive an incomplete token with.

        Fork children share the source payload → return source_row unchanged.
        Expand children / post-coalesce tokens carry their own payload in
        token_data_ref → restore it type-faithfully (checkpoint_loads) and wrap it in
        the contract under which the token was created (expand step output contract, or
        the coalesce output contract). Missing/garbage ref where one is required → raise
        (Tier-1, no coercion).
        """
        if spec.token_data_ref is None:
            return source_row  # fork child
        payload_bytes = payload_store.retrieve(spec.token_data_ref)
        data = checkpoint_loads(payload_bytes.decode("utf-8"))
        contract = self._resolve_token_contract(run_id, spec)
        return PipelineRow(data=data, contract=contract)

    def _resolve_token_contract(self, run_id: str, spec: IncompleteTokenSpec) -> SchemaContract:
        """Output contract for the step that created an expand/coalesce token.

        Reads nodes.output_contract_json for the node at spec.step_in_pipeline and
        returns ContractAuditRecord.from_json(...).to_contract(). Reuses the existing
        reader (rg -n "output_contract_json|ContractAuditRecord" src/elspeth/core/landscape/data_flow_repository.py).
        """
        # Resolve the node for spec.step_in_pipeline / expand_group_id|join_group_id,
        # load output_contract_json, return ContractAuditRecord.from_json(...).to_contract().
```

  Add imports: `from sqlalchemy import func` and add `node_states_table` to the existing `from elspeth.core.landscape.schema import ...` line (confirm: `rg -n "^from sqlalchemy|landscape.schema import" src/elspeth/core/checkpoint/recovery.py`). Add `PipelineRow`, `PayloadStore`, `SchemaContract` imports (all L0/L1 — confirm layer-legal). Verify the exact `ContractAuditRecord` reader and `to_contract` method name.

- [ ] **Step 6: Run the selection test — passes**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_get_incomplete_tokens_by_row_returns_only_incomplete_leaf" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/core/checkpoint/recovery.py tests/property/audit/test_fork_join_balance.py
git commit -m "feat(recovery): IncompleteTokenSpec + get_incomplete_tokens_by_row + reconstruct_token_row (F1)"
```

### Task 5: `WorkItem` offset + provenance + executor threading

**Files:**
- Modify: `src/elspeth/engine/dag_navigator.py` (`WorkItem`, `create_work_item`, `create_continuation_work_item`)
- Modify: `src/elspeth/engine/processor.py` (`_process_single_token` + the handlers it calls)
- Modify: `src/elspeth/engine/executors/sink.py`, `coalesce_executor.py`, `executors/state_guard.py` (via `transform.py`)

- [ ] **Step 1: Add two fields to `WorkItem`** (`rg -n "class WorkItem" src/elspeth/engine/dag_navigator.py`):

```python
    resume_attempt_offset: int = 0  # Added to every node_states.attempt written while
    # re-driving a reconstructed incomplete token on resume, so its records coexist with
    # the append-only run-1 records under UniqueConstraint(token_id, node_id, attempt).
    resume_checkpoint_id: str | None = None  # Stamped on every node_state written during a
    # resume re-drive (the resumed-from checkpoint id) so re-drives are provably
    # distinguishable from run-1 retries. Both default to the no-op (0 / None) for all
    # normal processing — the only nonzero/non-None source is resume_incomplete_token().
```

  Both are scalars → no `deep_freeze` guard. `create_work_item` and `create_continuation_work_item` forward both params (default 0 / None) into the `WorkItem(...)` construction and the nested `create_work_item` call.

- [ ] **Step 2: Thread both to the `begin_node_state` sites.**
  - **sink** (`rg -n "begin_node_state\(" src/elspeth/engine/executors/sink.py` → primary + failsink): add `attempt=resume_attempt_offset, resume_checkpoint_id=resume_checkpoint_id`. The sink executor has no retry loop, so the constant offset is collision-safe. Plumb both from the `WorkItem` into the sink executor entry point as `resume_attempt_offset: int = 0, resume_checkpoint_id: str | None = None`.
  - **coalesce** (`rg -n "begin_node_state\(" src/elspeth/engine/coalesce_executor.py`): same kwargs on the coalesce executor entry points.
  - **transform via state_guard** (`rg -n "begin_node_state\(|attempt=" src/elspeth/engine/executors/state_guard.py src/elspeth/engine/executors/transform.py`): the transform path threads tenacity `attempt`. Change the computed attempt to `resume_attempt_offset + attempt` (so a resume retry stays collision-free at `offset, offset+1, …`) and pass `resume_checkpoint_id`.

- [ ] **Step 3: Thread both through `_process_single_token`** (`rg -n "def _process_single_token|def _handle_transform_node|def _handle_terminal_token|def _maybe_coalesce_token" src/elspeth/engine/processor.py`). Read `work_item.resume_attempt_offset` and `work_item.resume_checkpoint_id` once and forward to `_handle_transform_node`, `_maybe_coalesce_token`, `_handle_terminal_token` (add `resume_attempt_offset: int = 0, resume_checkpoint_id: str | None = None` to each, forward to the executor calls in Step 2).

- [ ] **Step 4: Children created during a re-drive use the no-op defaults.** When `_handle_gate_fork` (`rg -n "def _handle_gate_fork" src/elspeth/engine/processor.py`) or deaggregation creates child work items, they get fresh token_ids (no prior node_states) → leave `resume_attempt_offset=0`, `resume_checkpoint_id=None`. Do NOT propagate the parent's values to children. Add a one-line comment stating this.

- [ ] **Step 5: No standalone test** (exercised end-to-end by Task 6/7). Verify nothing broke:

Run: `.venv/bin/python -m pytest tests/unit/engine tests/property/audit/test_fork_join_balance.py -q`
Expected: PASS (defaults preserve all existing behavior).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/dag_navigator.py src/elspeth/engine/processor.py src/elspeth/engine/executors/sink.py src/elspeth/engine/executors/transform.py src/elspeth/engine/executors/state_guard.py src/elspeth/engine/coalesce_executor.py
git commit -m "feat(engine): WorkItem.resume_attempt_offset + resume_checkpoint_id threaded to node_states writes (F1)"
```

### Task 6: `resume_incomplete_token` dispatch + relax `process_token`

**Files:**
- Modify: `src/elspeth/engine/processor.py` (`process_token` signature + `resume_incomplete_token`)
- Modify: `src/elspeth/engine/dag_navigator.py` (`resolve_branch_first_node`, `resolve_node_after`)

- [ ] **Step 1: Relax `process_token`** (`rg -n "def process_token" src/elspeth/engine/processor.py`) — `current_node_id: NodeID` → `NodeID | None`; add `resume_attempt_offset: int = 0, resume_checkpoint_id: str | None = None` and pass into `create_work_item`. It already delegates to `_process_single_token`, which handles `None` + `_branch_to_sink`.

- [ ] **Step 2: Add nav accessors** to `DAGNavigator` (after `create_continuation_work_item`):

```python
    def resolve_branch_first_node(self, branch_name: str) -> NodeID:
        """First processing node for a fork branch routed to a coalesce. Raises if unknown."""
        try:
            return self._branch_first_node[branch_name]
        except KeyError as exc:
            raise OrchestrationInvariantError(
                f"Unknown branch name '{branch_name}' — not in branch_first_node map. "
                f"Known: {sorted(self._branch_first_node.keys())}"
            ) from exc

    def resolve_node_after(self, node_id: NodeID) -> NodeID | None:
        """The single downstream node after node_id on a linear continuation, or None if
        it feeds a terminal sink. Raises on a fan-out (>1 successor) — a post-coalesce /
        post-expand token has exactly one continuation path. Reuses the existing successor
        lookup (rg -n "successors|_edges|resolve_next" src/elspeth/engine/dag_navigator.py)."""
```

- [ ] **Step 3: Add `resume_incomplete_token`** to `RowProcessor` (after `process_token`). The `row_data` is already reconstructed by the caller (Task 7); this method dispatches by token kind, mirroring `_handle_gate_fork`:

```python
    def resume_incomplete_token(
        self,
        spec: "IncompleteTokenSpec",
        row_data: PipelineRow,
        ctx: PluginContext,
        *,
        resume_checkpoint_id: str,
    ) -> list[RowResult]:
        """Drive one reconstructed incomplete child token to completion in place.

        Reuses the persisted token id (continuing under the ORIGINAL parent) and re-drives
        from the correct mid-DAG node. node_states are written at attempt spec.max_attempt+1
        and stamped with resume_checkpoint_id (provenance).
        """
        token = TokenInfo(
            row_id=spec.row_id, token_id=spec.token_id, row_data=row_data,
            branch_name=spec.branch_name, fork_group_id=spec.fork_group_id,
            join_group_id=spec.join_group_id, expand_group_id=spec.expand_group_id,
        )
        offset = spec.max_attempt + 1
        kw = {"resume_attempt_offset": offset, "resume_checkpoint_id": resume_checkpoint_id}
        branch = spec.branch_name

        if branch is not None and BranchName(branch) in self._branch_to_sink:
            # fork -> sink terminal branch: straight to the sink.
            return self.process_token(token, ctx, current_node_id=None, **kw)
        if branch is not None and BranchName(branch) in self._branch_to_coalesce:
            # fork -> coalesce, crashed BEFORE the barrier: re-run the branch from its
            # first node; the barrier fires normally via _maybe_coalesce_token.
            first_node = self._nav.resolve_branch_first_node(branch)
            return self.process_token(
                token, ctx, current_node_id=first_node,
                coalesce_name=self._branch_to_coalesce[BranchName(branch)], **kw,
            )
        if spec.expand_group_id is not None:
            # expand child: re-drive from the node after the expand step.
            after = self._nav.resolve_node_after(self._resolve_step_node(spec))
            return self.process_token(token, ctx, current_node_id=after, **kw)
        if spec.join_group_id is not None and spec.fork_group_id is None:
            # post-coalesce merged token, crashed AFTER the barrier (review finding B1):
            # re-drive downstream of the coalesce node. None => fed straight to a sink.
            after = self._nav.resolve_node_after(self._resolve_step_node(spec))
            return self.process_token(token, ctx, current_node_id=after, **kw)
        raise OrchestrationInvariantError(
            f"Incomplete token {spec.token_id} has branch_name={branch!r}, "
            f"fork_group_id={spec.fork_group_id!r}, join_group_id={spec.join_group_id!r}, "
            f"expand_group_id={spec.expand_group_id!r} — no resume-start node resolvable. "
            f"Audit/DAG inconsistency."
        )
```

  Add `_resolve_step_node(spec) -> NodeID` (maps `spec.step_in_pipeline` → node id via the recorder's step→node map; `rg -n "step_in_pipeline|step.*node|node.*step" src/elspeth/core/landscape/ src/elspeth/engine/dag_navigator.py`).

- [ ] **Step 4: Add the import** `from elspeth.core.checkpoint.recovery import IncompleteTokenSpec` (L2→L1, layer-legal; the annotation can be a string to avoid any cycle, but a real import is preferred — confirm recovery does not import engine: `rg -n "import.*engine" src/elspeth/core/checkpoint/recovery.py`).

- [ ] **Step 5: Type-check**

Run: `.venv/bin/python -m mypy src/elspeth/engine/processor.py src/elspeth/engine/dag_navigator.py`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/processor.py src/elspeth/engine/dag_navigator.py
git commit -m "feat(engine): resume_incomplete_token dispatch (fork/coalesce/expand) + NodeID|None process_token (F1)"
```

### Task 7: Resume dispatch wiring — turn RED green

**Files:**
- Modify: `src/elspeth/engine/orchestrator/resume.py` (per-row loop) + `core.py` (LoopContext / setup)

- [ ] **Step 1: Make the incomplete-token map + checkpoint id + payload store reachable in the loop.** Read how `run_resume_processing_loop` is called and how `unprocessed_rows`/`LoopContext` are built (`rg -n "get_unprocessed_rows|LoopContext|run_resume_processing_loop" src/elspeth/engine/orchestrator/core.py`). At the same site that calls `get_unprocessed_rows`, compute `incomplete_by_row = recovery_manager.get_incomplete_tokens_by_row(run_id)` and thread it, the `payload_store`, and `resume_point.checkpoint.checkpoint_id` to the loop (prefer `LoopContext` fields if `recovery_manager`/`payload_store` are already there; else add keyword params). **Confirm `payload_store` is reachable in the loop** — it is needed for expand/coalesce reconstruction (`rg -n "payload_store" src/elspeth/engine/orchestrator/resume.py src/elspeth/engine/orchestrator/core.py`).

- [ ] **Step 2: Replace the per-row processing call with the dispatch:**

```python
        specs = incomplete_by_row.get(row_id)
        if specs:
            # Partial fork/expand/coalesce completion: drive ONLY the incomplete children to
            # completion under the original parent. Restarting from source would re-fork to
            # ALL branches and re-emit the completed ones (F1).
            results = []
            for spec in specs:
                row = recovery_manager.reconstruct_token_row(
                    spec, run_id, source_row=pipeline_row, payload_store=payload_store
                )
                results.extend(processor.resume_incomplete_token(
                    spec, row, ctx, resume_checkpoint_id=resume_checkpoint_id
                ))
        else:
            # No tokens for this row (never started) -> whole-row restart is correct.
            results = processor.process_existing_row(
                row_id=row_id, row_data=pipeline_row, transforms=config.transforms, ctx=ctx,
            )
        if results:
            loop_ctx.last_token_id = results[-1].token.token_id
        accumulate_row_outcomes(results, counters, pending_tokens)
```

  Add `from elspeth.core.checkpoint.serialization import checkpoint_loads` if reconstruction is inlined here instead of in `recovery_manager` (it is in `recovery_manager` per Task 4 — no import needed in resume.py).

- [ ] **Step 3: Run the reproduction — now GREEN**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_does_not_reemit_completed_fork_branch" -v`
Expected: PASS — `after == baseline`, `sink_b.collected` length 1, zero orphans, status COMPLETED, fork-group shape unchanged.

- [ ] **Step 4: Regression sweep**

Run: `.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py tests/e2e/recovery -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator/resume.py src/elspeth/engine/orchestrator/core.py
git commit -m "fix(resume): drive incomplete fork/expand/coalesce children in place, no source restart (F1)"
```

### Task 8: Collision + provenance attributability regressions

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

> **RED-first discipline:** each cell below must be shown to fail for ITS OWN reason before relying on the mechanism. Since the mechanism is already built, demonstrate RED by **temporarily disabling the dispatch** — comment out the `if specs:` branch in `resume.py` (Task 7 Step 2) so the row falls to `process_existing_row` — run the cell (RED), then restore (GREEN). Record the observed RED reason in the test docstring.

- [ ] **Step 1: fork→sink collision, single-attempt** — `test_resume_fork_sink_coexists_with_run1_node_state`:

```python
    def test_resume_fork_sink_coexists_with_run1_node_state(self) -> None:
        """Re-driving an incomplete fork->sink child must not collide on node_states.

        run-1 left a CLOSED sink node_state at attempt=0 for the incomplete child (the
        sink ran; only its terminal OUTCOME was deleted to simulate the interruption —
        the node_state row remains). The re-drive must write at attempt=1, and the run-1
        record must be preserved (append-only). RED (dispatch disabled): double-emit
        breaks the conservation law.
        """
        # setup as test_resume_does_not_reemit..., then after resume:
        with db.connection() as conn:
            attempts = [r.attempt for r in conn.execute(text(
                "SELECT attempt FROM node_states WHERE token_id=:tid ORDER BY attempt"),
                {"tid": incomplete_token_id}).fetchall()]
        assert 0 in attempts, "run-1 node_state (attempt 0) preserved (append-only)"
        assert 1 in attempts, "resume re-drive recorded at attempt 1 (no collision)"
```

- [ ] **Step 2: fork→sink collision, MULTI-attempt** — `test_resume_offset_is_max_plus_one_not_hardcoded`:

```python
    def test_resume_offset_is_max_plus_one_not_hardcoded(self) -> None:
        """If the incomplete child already has TWO run-1 attempts (0 and 1, e.g. a
        tenacity retry before the crash), the re-drive must land at attempt=2 (max+1),
        not a hardcoded 1. Guards against offset = +1 / 'resume generation = 1'.
        """
        # After the run + outcome-delete, INSERT a second run-1 node_state for the
        # incomplete child at attempt=1 (simulating a run-1 retry) directly via SQL,
        # leaving NO terminal outcome. Resume, then:
        #   assert 2 in attempts and max(attempts) == 2
```

- [ ] **Step 3: provenance distinguishability** — `test_resume_redrive_is_query_separable_from_retry`:

```python
    def test_resume_redrive_is_query_separable_from_retry(self) -> None:
        """A resume re-drive node_state carries resume_checkpoint_id; the run-1 record at
        the same (token_id, node_id) does not. Proves explain() can separate them.
        """
        # After resume of the partial fork:
        with db.connection() as conn:
            rows = conn.execute(text(
                "SELECT attempt, resume_checkpoint_id FROM node_states "
                "WHERE token_id=:tid ORDER BY attempt"), {"tid": incomplete_token_id}).fetchall()
        run1 = [r for r in rows if r.resume_checkpoint_id is None]
        redrive = [r for r in rows if r.resume_checkpoint_id is not None]
        assert run1 and redrive, "both a run-1 (NULL) and a re-drive (non-NULL) record must exist"
        assert all(r.attempt == 0 for r in run1)
        assert all(r.attempt >= 1 for r in redrive)
```

- [ ] **Step 4: Run all three**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant" -k "coexists or offset_is_max or query_separable" -v`
Expected: PASS (after restoring the dispatch). If Step 1 fails with IntegrityError, the offset missed the sink `begin_node_state`; if Step 3's re-drive row is NULL, the provenance kwarg missed a site (Task 5 Step 2).

- [ ] **Step 5: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: resume node_states attempt-bump + provenance attributability (F1)"
```

### Task 9: fork→coalesce (before-barrier) + post-coalesce (after-barrier, B1) regressions

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: fork→coalesce before-barrier** — `test_resume_fork_to_coalesce_before_barrier`:

```python
    def test_resume_fork_to_coalesce_before_barrier(self) -> None:
        """A fork->coalesce branch interrupted BEFORE the barrier resumes from
        branch_first_node, merges normally, yields the baseline outcomes. Include ONE
        PassTransform on the coalesce branch so branch_first_node != coalesce node.
        RED (dispatch disabled): restart re-forks -> double-emit.
        """
```

- [ ] **Step 2: post-coalesce after-barrier (B1)** — `test_resume_post_coalesce_before_downstream`:

```python
    def test_resume_post_coalesce_before_downstream(self) -> None:
        """The barrier fired (merged token + COALESCED outcomes recorded) but the
        downstream sink did not. The merged token (join_group_id set, branch/fork/expand
        NULL) must re-drive downstream from its token_data_ref merged payload — NOT crash,
        NOT restart-and-re-fork. RED (B1 branch disabled): OrchestrationInvariantError.
        """
        # Build fork -> coalesce 'merge' -> sink_a (+ sink_b direct). Run to completion,
        # then delete the downstream sink_a terminal outcome of the MERGED token, leaving
        # the merged token incomplete. Resume; assert full conservation law + the merged
        # token now terminal + token_data_ref round-tripped the merged payload.
```

  > This cell's RED reason differs from the fork cells: with the post-coalesce branch disabled it raises `OrchestrationInvariantError`, not a conservation-law failure. Assert that specific RED in the docstring.

  Wire coalesce via `CoalesceSettings` (`rg -n "CoalesceSettings\(" tests/property/audit/test_fork_join_balance.py`).

- [ ] **Step 3: Run both**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant" -k "fork_to_coalesce_before or post_coalesce_before" -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: resume fork->coalesce (before-barrier) + post-coalesce (after-barrier, B1) (F1)"
```

### Task 10: expand resume + value-fidelity + side-effecting-transform attributability

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: partial expand resume + value fidelity** — `test_resume_partial_expand_does_not_reemit`:

```python
    def test_resume_partial_expand_does_not_reemit(self) -> None:
        """A deaggregation row with one completed and one incomplete expanded child resumes
        the incomplete child only — no re-expansion, no double-emit.

        VALUE FIDELITY (the conservation-law oracle counts outcomes, NOT values): give the
        two children DISTINCT payloads — child0 carries datetime(2021,1,1)+Decimal('1.5'),
        child1 carries datetime(2022,2,2)+Decimal('99.25'). After resuming the INCOMPLETE
        child, read its resumed row back and assert it carries ITS OWN values as
        datetime/Decimal instances (NOT str, NOT the sibling's values) — proving
        checkpoint_dumps/_loads preserved type fidelity AND value<->token alignment
        (zip(strict=True) guards length only).
        """
        import datetime, decimal
        # ... after resume of the incomplete child:
        assert isinstance(resumed_child_row["ts"], datetime.datetime)
        assert isinstance(resumed_child_row["amount"], decimal.Decimal)
        assert resumed_child_row["amount"] == decimal.Decimal("99.25")  # its OWN value
```

  Use the existing deaggregation fixtures (`rg -n "deagg|Deaggregat|expand" tests/fixtures/ tests/`).

- [ ] **Step 2: side-effecting transform on a re-driven branch carries provenance** — `test_resume_redriven_transform_external_call_is_attributable` (spec ADDENDUM 2.C):

```python
    def test_resume_redriven_transform_external_call_is_attributable(self) -> None:
        """A fork->coalesce branch whose transform makes a recorded operation_call is
        re-driven on resume; the call RE-FIRES (bounded non-goal) but the re-fired
        operation_call's node_state carries resume_checkpoint_id, so an auditor can prove
        it came from the resume — the duplication is honest, not silent.
        """
        # Use a transform that records an operation_call (rg -n "record.*operation|operation_call"
        # for a fixture/helper). Interrupt before the barrier, resume, then assert the
        # second operation_call's node_state has resume_checkpoint_id IS NOT NULL.
```

- [ ] **Step 3: Run both**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant" -k "partial_expand or redriven_transform" -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: resume partial-expand value fidelity + re-driven external-call attributability (F1)"
```

---

## Phase 5 — Counter reconciliation, matrix completion, gates

### Task 11: F1/F2 counter-field reconciliation

Approach 1 changes the set/shape of `RowResult`s entering `accumulate_row_outcomes`. Assert the actual **counter fields** reconcile (not just the outcome multiset — spec ADDENDUM 2.E).

**Files:**
- Modify: `tests/property/audit/test_fork_join_balance.py`

- [ ] **Step 1: Write the guard test** — `test_resume_counters_reconcile_with_uninterrupted_run`:

```python
    def test_resume_counters_reconcile_with_uninterrupted_run(self) -> None:
        """After (run1 + resume), the run's counter fields equal a single uninterrupted run.

        Run pipeline A uninterrupted -> capture counters (rows_processed, rows_succeeded,
        rows_failed) from the runs table / RunResult. Run B identically, interrupt one fork
        branch, resume. Assert B's reconciled counters == A's, AND the terminal-outcome
        multiset matches. Asserting the COUNTER FIELDS (not only outcomes) is the point:
        the resume path increments rows_processed per leaf (resume.py) which can diverge
        from A's per-source-row count.
        """
        # Read counters via: rg -n "rows_processed|rows_succeeded|rows_failed" src/elspeth/core/landscape/schema.py src/elspeth/engine/orchestrator/
```

- [ ] **Step 2: Run.** If it FAILS, the resume per-leaf `rows_processed` counting (F2) diverges — reconcile by counting per-source-row on the reconstruction path, not per-leaf (read `accumulate_row_outcomes` + `counters.rows_processed`: `rg -n "rows_processed|def accumulate_row_outcomes" src/elspeth/engine/orchestrator/`). Fixing F2 here is in scope (coupled, per spec).

- [ ] **Step 3: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py src/elspeth/engine/orchestrator/
git commit -m "test+fix: resume counter fields reconcile with uninterrupted run (F1/F2)"
```

### Task 12: Remaining risk-ordered matrix

Add one test per remaining spec matrix item. Each follows RED→GREEN→commit; demonstrate RED via the dispatch-disable technique (Task 8 header) and record each cell's RED reason.

- [ ] **Step 1: matrix #2** — `test_resume_all_branches_incomplete`: a fork row where *no* branch completed → all children surface as incomplete and each re-drives correctly; conservation law.
- [ ] **Step 2: matrix #3** — `test_resume_three_way_fork_two_incomplete`: 1 done, 2 incomplete; assert deterministic completion (order by `step_in_pipeline`/`token_id`) + conservation law.
- [ ] **Step 3: matrix #4** — `test_resume_failure_during_resumed_branch_yields_failed_not_orphan`: resumed branch's sink fails → a FAILED terminal outcome, zero orphan (verify bounded: the failed token gets `completed=1` and is not re-selected on a subsequent resume).
- [ ] **Step 4: matrix #5** — `test_resume_aggregation_buffer_with_partial_fork`: mixed buffered + incomplete tokens on one row (exercises the recovery mixed-state exclusion; `rg -n "BUFFERED|buffer" src/elspeth/core/checkpoint/recovery.py`) → conservation law.
- [ ] **Step 5: matrix #6 (multi-row)** — `test_resume_multi_row_partial_fork`: ≥2 source rows, one fully complete + one partial; assert by-row grouping in `get_incomplete_tokens_by_row` + the dispatch loop handle both, conservation law over all rows. (The single-row RED test does not exercise the by-row grouping.)
- [ ] **Step 6: matrix #6 (linear regression)** — `test_resume_linear_pipeline_regression_audit`: a linear (no-fork) resume still uses `process_existing_row`, asserted on the **audit trail** (not just in-memory sink). Confirms the no-tokens dispatch branch.
- [ ] **Step 7: matrix #7** — `test_resume_of_resume_converges`: resume, then resume again → incomplete-leaf stock monotonic non-increasing, reaches 0; idempotent.
- [ ] **Step 8: Run the whole matrix**

Run: `.venv/bin/python -m pytest "tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant" -v`
Expected: PASS (all)

- [ ] **Step 9: Commit**

```bash
git add tests/property/audit/test_fork_join_balance.py
git commit -m "test: complete risk-ordered resume fork/expand/coalesce matrix (F1)"
```

### Task 13: Full gate run

- [ ] **Step 1: Targeted then full suite** (plain `pytest tests/` is the CI-equivalent selection — do NOT pass `-o addopts=""`, per memory):

```bash
.venv/bin/python -m pytest tests/property/audit/test_fork_join_balance.py tests/e2e/recovery tests/unit/landscape tests/unit/core/landscape -q
.venv/bin/python -m pytest tests/ -q
```
Expected: PASS (no new failures vs. the known pre-existing RC5.2 failures — memory `project_rc52_preexisting_test_failures_2026-05-29`).

- [ ] **Step 2: Type + lint**

```bash
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
```
Expected: clean.

- [ ] **Step 3: Tier-model lint (Python 3.13 venv — matches main)**

```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```
Expected: pass. The new L2→L1 import (`processor.py` → `recovery.IncompleteTokenSpec`) is downward + layer-legal. If allowlist fingerprints rotate from body-index shifts, co-land the fp update in this commit (memory `feedback_ast_shift_fingerprint_rotation`).

- [ ] **Step 4: Freeze-guard gate** (WorkItem, IncompleteTokenSpec, Token are frozen; all new fields scalars/None → no `deep_freeze` guard needed; verify the gate agrees):

```bash
.venv/bin/python scripts/cicd/enforce_freeze_guards.py
```
Expected: pass.

- [ ] **Step 5: Plugin hash gate** (only if a plugin file changed — none expected; if a coalesce/expand plugin caller changed, refresh via the documented `scripts/cicd/plugin_hash` flow, memory `project_plugin_hash_gate_ci_only`).

- [ ] **Step 6: Final commit if any gate reconciliation was needed**

```bash
git add -A && git commit -m "chore: reconcile gates for resume fork/expand/coalesce fix (F1)"
```

---

## Done criteria

- The RED reproduction and all matrix tests are GREEN; each new cell was observed RED for its own reason before the mechanism was relied upon.
- The conservation law holds for fork→sink, fork→coalesce (before-barrier), post-coalesce (after-barrier, B1), and partial-expand resume.
- Resume re-drives are query-separable from run-1 retries (`node_states.resume_checkpoint_id`); a re-fired external call on a re-driven branch is attributable, not silent.
- Epoch is 11; `_REQUIRED_COLUMNS` includes `token_data_ref`, `resume_checkpoint_id`, and the two openrouter columns (F3).
- Append-only audit preserved (run-1 node_states untouched; re-drive at bumped attempt + provenance marker).
- Counter fields (`rows_processed`/`rows_succeeded`/`rows_failed`) reconcile with an uninterrupted run (F2).
- mypy / ruff / tier-model / freeze gates clean; full `pytest tests/` shows no new failures.
- Merge to RC5.2 with `--no-ff` (operator approval for the merge; bare push is non-destructive).
- **OPERATOR ACTION (gate before any staging deploy):** delete the staging audit DB before the next staging run — epoch 11 is incompatible with epoch-10 DBs and the SQLite epoch gate will reject them with migration guidance. Surface this explicitly; do not deploy to staging until done.
