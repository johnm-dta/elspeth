# ADR-019 Stage 2/3 — Phase 2: Producer Site Flip

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phase 2 alone leaves the accumulator and resume aggregation broken (they still match on `RowOutcome` while producers emit `(outcome, path)`). Phase 3 must follow in the same PR.

**Goal:** Flip every `RowOutcome.X` reference at the producer emit site to a `(outcome=TerminalOutcome.Y, path=TerminalPath.Z)` pair construction, per the canonical mapping table at `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`. After this phase, no `RowOutcome.X` references remain in `src/elspeth/engine/`, `src/elspeth/core/checkpoint/`, or anywhere else producers run — only `src/elspeth/contracts/enums.py` (defines RowOutcome through Stage 4) and `src/elspeth/testing/__init__.py` (re-exports through Stage 4).

**Files touched in this phase:**

- Modify: `src/elspeth/engine/processor.py` (28 emit sites)
- Modify: `src/elspeth/engine/executors/transform.py` (1 emit site)
- Modify: `src/elspeth/engine/executors/sink.py` (3 emit sites — primary, failsink-mode DIVERTED, discard-mode DIVERTED)
- Modify: `src/elspeth/engine/coalesce_executor.py` (4 emit sites)
- Modify: `src/elspeth/core/checkpoint/recovery.py` (resume reader — flips RowOutcome subquery to TerminalOutcome / TerminalPath)
- Test: targeted unit test that asserts each producer emits the expected pair (one test per emit-site cluster)

**Background reading:** ADR-019 § Mapping table (lines 99-115) is the canonical contract. The Stage 1 commit's mapping fixture at `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING` is the executable encoding — when in doubt about a translation, consult that dict.

---

## The canonical mapping (paste-table)

This is the table executing engineers use to translate. Identical encoding to Stage 1's `_ROW_OUTCOME_TO_TWO_AXIS_MAPPING` test fixture.

| Old `RowOutcome` | New `(outcome, path)` pair | Required fields |
| --- | --- | --- |
| `COMPLETED` | `(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)` | `sink_name` |
| `ROUTED` | `(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)` | `sink_name` |
| `ROUTED_ON_ERROR` | `(TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED)` | `sink_name`, `error` (FailureInfo) |
| `DROPPED_BY_FILTER` | `(TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED)` | — |
| `COALESCED` | `(TerminalOutcome.SUCCESS, TerminalPath.COALESCED)` | `sink_name`, `join_group_id` |
| `FAILED` | `(TerminalOutcome.FAILURE, TerminalPath.UNROUTED)` | `error_hash` (recorder-side) |
| `QUARANTINED` | `(TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE)` | `error_hash` |
| `DIVERTED` (failsink mode, `sink_name = failsink_name`) | `(TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK)` | `sink_name`, `error_hash` |
| `DIVERTED` (discard mode, `sink_name = "__discard__"`) | `(TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED)` | `sink_name`, `error_hash` |
| `FORKED` | `(TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT)` | `fork_group_id` |
| `EXPANDED` | `(TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT)` | `expand_group_id` |
| `CONSUMED_IN_BATCH` | `(TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED)` | `batch_id` |
| `BUFFERED` | `(None, TerminalPath.BUFFERED)` | `batch_id` |

**The two DIVERTED flavors are NOT auto-derivable** — the producer site must inspect `sink_name` to determine which flavor. See Task 2.4 for the worked example.

---

## Tasks

### Task 2.1: Producer flip — `processor.py` (28 sites)

**Files:**
- Modify: `src/elspeth/engine/processor.py`

**Step 1: Read the per-line enumeration**

Below is the exhaustive enumeration of every `outcome=RowOutcome.X` site in `processor.py` after Phase 1 commit. Each line maps to its target pair via the canonical table above.

| Line | Old | New `outcome=` | New `path=` | Notes |
| --- | --- | --- | --- | --- |
| 682 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | `error_hash` already wired; verify |
| 708 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | constructs `RowResult` via short-circuit; needs `path=` keyword |
| 911 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 951 | `RowOutcome.DROPPED_BY_FILTER` | `TerminalOutcome.SUCCESS` | `TerminalPath.FILTER_DROPPED` | no required field beyond path |
| 998 | `RowOutcome.DROPPED_BY_FILTER` | `TerminalOutcome.SUCCESS` | `TerminalPath.FILTER_DROPPED` | |
| 1064 | `RowOutcome.COMPLETED` | `TerminalOutcome.SUCCESS` | `TerminalPath.DEFAULT_FLOW` | `sink_name` already passed |
| 1145 | `RowOutcome.QUARANTINED` | `TerminalOutcome.FAILURE` | `TerminalPath.QUARANTINED_AT_SOURCE` | `error_hash` |
| 1152 | `RowOutcome.CONSUMED_IN_BATCH` | `TerminalOutcome.TRANSIENT` | `TerminalPath.BATCH_CONSUMED` | `batch_id` |
| 1181 | `RowOutcome.QUARANTINED` | `TerminalOutcome.FAILURE` | `TerminalPath.QUARANTINED_AT_SOURCE` | `error_hash` |
| 1206 | `RowOutcome.COMPLETED` | `TerminalOutcome.SUCCESS` | `TerminalPath.DEFAULT_FLOW` | `sink_name` |
| 1340 | `RowOutcome.BUFFERED` | `None` | `TerminalPath.BUFFERED` | non-terminal — `outcome=None` |
| 1404 | `RowOutcome.BUFFERED` | `None` | `TerminalPath.BUFFERED` | non-terminal — `outcome=None` |
| 1621 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 1901 | `RowOutcome.COALESCED` | `TerminalOutcome.SUCCESS` | `TerminalPath.COALESCED` | `sink_name`, `join_group_id` |
| 1923 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 1934 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 2005 | `RowOutcome.COALESCED` | `TerminalOutcome.SUCCESS` | `TerminalPath.COALESCED` | |
| 2029 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 2128 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 2142 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | |
| 2185 | `RowOutcome.DROPPED_BY_FILTER` | `TerminalOutcome.SUCCESS` | `TerminalPath.FILTER_DROPPED` | |
| 2232 | `RowOutcome.EXPANDED` | `TerminalOutcome.TRANSIENT` | `TerminalPath.EXPAND_PARENT` | `expand_group_id` |
| 2267 | `RowOutcome.QUARANTINED` | `TerminalOutcome.FAILURE` | `TerminalPath.QUARANTINED_AT_SOURCE` | |
| 2281 | `RowOutcome.QUARANTINED` | `TerminalOutcome.FAILURE` | `TerminalPath.QUARANTINED_AT_SOURCE` | |
| 2316 | `RowOutcome.ROUTED_ON_ERROR` | `TerminalOutcome.FAILURE` | `TerminalPath.ON_ERROR_ROUTED` | `sink_name`, `error` |
| 2385 | `RowOutcome.ROUTED` | `TerminalOutcome.SUCCESS` | `TerminalPath.GATE_ROUTED` | `sink_name` |
| 2498 | `RowOutcome.FORKED` | `TerminalOutcome.TRANSIENT` | `TerminalPath.FORK_PARENT` | `fork_group_id` |
| 2576 | `RowOutcome.COMPLETED` | `TerminalOutcome.SUCCESS` | `TerminalPath.DEFAULT_FLOW` | `sink_name` |

**Step 2: Worked example for the most common shape (FAILED → UNROUTED)**

```python
# OLD (e.g. line 682):
results.append(
    RowResult(
        token=token,
        final_data=token.row_data,
        outcome=RowOutcome.FAILED,
        error=failure,
    )
)

# NEW:
results.append(
    RowResult(
        token=token,
        final_data=token.row_data,
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.UNROUTED,
        error=failure,
    )
)
```

**Step 3: Worked example for ROUTED (gate MOVE)**

```python
# OLD (line 2382-2387):
current_result = RowResult(
    token=current_token,
    final_data=current_token.row_data,
    outcome=RowOutcome.ROUTED,
    sink_name=outcome.sink_name,
)

# NEW:
current_result = RowResult(
    token=current_token,
    final_data=current_token.row_data,
    outcome=TerminalOutcome.SUCCESS,
    path=TerminalPath.GATE_ROUTED,
    sink_name=outcome.sink_name,
)
```

**Step 4: Worked example for BUFFERED (non-terminal)**

```python
# OLD (line 1340):
self._record_token_outcome(
    ref=ref,
    outcome=RowOutcome.BUFFERED,
    batch_id=batch_id,
)

# NEW:
self._record_token_outcome(
    ref=ref,
    outcome=None,            # non-terminal
    path=TerminalPath.BUFFERED,
    batch_id=batch_id,
)
```

**Step 5: Worked example for `_emit_token_completed`**

`_emit_token_completed` is the telemetry call at processor sites that wrap a RowOutcome in a `TokenCompleted` event. Phase 1 retyped `TokenCompleted.outcome` to `TerminalOutcome | None` and added `path: TerminalPath`. Update each call:

```python
# OLD (e.g. line 696):
self._emit_token_completed(token, RowOutcome.FAILED)

# NEW: helper signature changed in Phase 1 to take (outcome, path).
self._emit_token_completed(
    token,
    outcome=TerminalOutcome.FAILURE,
    path=TerminalPath.UNROUTED,
)
```

**Note:** `_emit_token_completed`'s helper signature was updated in Phase 1 to accept `(outcome, path)` keywords. Phase 2 only updates callers.

**Step 6: Update imports**

These imports rely on Phase 1 Task 1.8 Step 1a having added `TerminalOutcome` and `TerminalPath` to `contracts/__init__.py`'s import block and `__all__` list. If executing phases in isolation, verify with `python -c 'from elspeth.contracts import TerminalOutcome, TerminalPath'` first — without that re-export the imports below crash with `ImportError: cannot import name 'TerminalOutcome' from 'elspeth.contracts'`.

Replace `from elspeth.contracts import RouteDestination, RowOutcome, RowResult, SourceRow, TokenInfo, TransformResult` (line 21) with:

```python
from elspeth.contracts import (
    RouteDestination,
    RowResult,
    SourceRow,
    TerminalOutcome,
    TerminalPath,
    TokenInfo,
    TransformResult,
)
```

`processor.py` no longer references `RowOutcome` after this task. The lint guard at `scripts/cicd/forbid_new_row_outcome.py` will start failing on `processor.py` if it remains in the migration allowlist — Phase 2 commit removes `src/elspeth/engine/processor.py` from `config/cicd/forbid_new_row_outcome/migration_files.yaml` to reflect the completed flip.

**Step 7: Verify the file compiles and unit tests run**

Run: `.venv/bin/python -c "from elspeth.engine import processor; print('OK')"`

Expected: `OK`.

Run: `.venv/bin/python -m pytest tests/unit/engine/ -q`

Expected: tests that exercise `processor.py` directly pass; tests that ASSERT on `RowOutcome.X` from processor outputs may fail (deferred to Stage 4 per Phase 5 triage).

**Definition of Done:**
- [ ] Verify `from elspeth.contracts import TerminalOutcome, TerminalPath` succeeds (depends on Phase 1 Task 1.8 Step 1a)
- [ ] All 28 emit sites in `processor.py` flipped to (outcome, path) pairs
- [ ] `_emit_token_completed` callers updated (~15 sites in `processor.py`)
- [ ] No `RowOutcome` references remain in `processor.py`
- [ ] `processor.py` removed from the lint guard allowlist
- [ ] `processor` module imports cleanly
- [ ] mypy passes on `src/elspeth/engine/processor.py`

---

### Task 2.2: Producer flip — `transform.py` (1 site)

**Files:**
- Modify: `src/elspeth/engine/executors/transform.py:147`

**Step 1: Apply the edit**

Line 147 (and surrounding context — read 140-155 for the construction site):

```python
# OLD:
results.append(
    RowResult(
        token=token,
        final_data=token.row_data,
        outcome=RowOutcome.FAILED,
        error=failure,
    )
)

# NEW:
results.append(
    RowResult(
        token=token,
        final_data=token.row_data,
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.UNROUTED,
        error=failure,
    )
)
```

**Step 2: Update imports**

Replace `RowOutcome,` (line 27) with `TerminalOutcome,\n    TerminalPath,`.

**Step 3: Remove from allowlist**

Remove `src/elspeth/engine/executors/transform.py` from `config/cicd/forbid_new_row_outcome/migration_files.yaml`.

**Step 4: Verify**

Run: `.venv/bin/python -c "from elspeth.engine.executors import transform; print('OK')"`

**Definition of Done:**
- [ ] One emit site flipped
- [ ] Imports updated
- [ ] Allowlist trimmed
- [ ] mypy passes

---

### Task 2.3: Producer flip — `coalesce_executor.py` (4 sites)

**Files:**
- Modify: `src/elspeth/engine/coalesce_executor.py`

**Step 1: Apply per-site edits**

| Line | Old | New `outcome=` | New `path=` | Required fields |
| --- | --- | --- | --- | --- |
| 535 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | `error_hash` |
| 728 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | `error_hash` |
| 1013 | `RowOutcome.COALESCED` | `TerminalOutcome.SUCCESS` | `TerminalPath.COALESCED` | `sink_name`, `join_group_id` |
| 1077 | `RowOutcome.FAILED` | `TerminalOutcome.FAILURE` | `TerminalPath.UNROUTED` | `error_hash` |

**Step 2: Update imports**

Replace `from elspeth.contracts.enums import NodeStateStatus, RowOutcome` (line 23) with:

```python
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
```

**Step 3: Remove from allowlist**

Remove `src/elspeth/engine/coalesce_executor.py` from the allowlist.

**Definition of Done:**
- [ ] Four emit sites flipped
- [ ] Imports updated
- [ ] Allowlist trimmed
- [ ] mypy passes

---

### Task 2.4: Producer flip — `sink.py` (3 sites with the load-bearing DIVERTED two-flavor split)

**Files:**
- Modify: `src/elspeth/engine/executors/sink.py`

**Step 1: Apply the primary-write site edit (line 301)**

```python
# OLD (line 301 area, primary-write FAILED handler):
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=RowOutcome.FAILED,
    error_hash=error_hash,
)

# NEW:
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=TerminalOutcome.FAILURE,
    path=TerminalPath.UNROUTED,
    error_hash=error_hash,
)
```

**Step 2: Apply the failsink-mode DIVERTED edit (line 952)**

This is the `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` flavor — the failsink absorbed the row for visibility. The lifecycle answer lives on the failsink's `NodeStateStatus.COMPLETED` `node_state` plus the registered `artifacts` row.

```python
# OLD (lines 952-957):
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=RowOutcome.DIVERTED,
    error_hash=error_hash,
    sink_name=failsink_name,
)

# NEW:
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=TerminalOutcome.TRANSIENT,
    path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
    error_hash=error_hash,
    sink_name=failsink_name,
)
```

**Step 3: Apply the discard-mode DIVERTED edit (line 998) — OPERATOR-VISIBLE BEHAVIOUR CHANGE**

This is the `(FAILURE, SINK_DISCARDED)` flavor. Per ADR-019 § Sub-decision 5, this becomes a **predicate input** — `rows_failed` increments, `failure_indicator` flips. The companion accumulator change lands in Phase 3.

```python
# OLD (lines 998-1003):
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=RowOutcome.DIVERTED,
    error_hash=error_hash,
    sink_name="__discard__",
)

# NEW:
# ADR-019 § Sub-decision 5 (round-3 panel-resolved): discard-mode DIVERTED
# becomes (FAILURE, SINK_DISCARDED) — a predicate-input failure rather
# than a TRANSIENT bookkeeping marker. Operator-visible RunStatus flip
# from COMPLETED to COMPLETED_WITH_FAILURES for runs containing discards;
# see docs/operator/migrations/adr-019.md.
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=TerminalOutcome.FAILURE,
    path=TerminalPath.SINK_DISCARDED,
    error_hash=error_hash,
    sink_name="__discard__",
)
```

**Step 4: Update imports**

Replace `RowOutcome,` (line 17) with `TerminalOutcome,\n    TerminalPath,`.

**Step 5: Remove from allowlist**

Remove `src/elspeth/engine/executors/sink.py` from the allowlist.

**Step 6: Verify**

Run: `.venv/bin/python -m pytest tests/unit/engine/executors/test_sink_executor.py -q` (and any related sink tests in `tests/integration/`).

Expected: tests pass IF they don't make `RowOutcome.X` assertions (those are deferred to Stage 4). Tests that read `TokenOutcome` from the audit DB now see `TerminalOutcome.FAILURE` / `TerminalPath.SINK_DISCARDED` for discards.

**Definition of Done:**
- [ ] Three emit sites flipped (primary, failsink-mode, discard-mode)
- [ ] Discard-mode site has the operator-visible-change comment block
- [ ] Imports updated
- [ ] Allowlist trimmed
- [ ] mypy passes

---

### Task 2.5: Resume reader — `recovery.py`

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py:370-412`

**Step 1: Read the existing query**

The recovery path queries `token_outcomes` to identify which rows have terminal outcomes. The query uses `RowOutcome.FORKED`, `RowOutcome.EXPANDED` as delegation markers and iterates `RowOutcome.is_terminal`.

Under ADR-019:
- "Delegation markers" = the two paths `FORK_PARENT` and `EXPAND_PARENT` (always paired with `outcome=TRANSIENT`).
- "Has terminal outcome" = `completed=1` (the renamed `is_terminal` column).
- "Excluded from terminal-row check" = the two delegation paths above.

**Step 2: Apply the edit**

```python
# OLD (lines 370-397):
delegation_tokens = (
    select(token_outcomes_table.c.token_id)
    .where(token_outcomes_table.c.run_id == run_id)
    .where(
        token_outcomes_table.c.outcome.in_(
            [
                RowOutcome.FORKED,
                RowOutcome.EXPANDED,
            ]
        )
    )
).scalar_subquery()

# Terminal outcomes that indicate row processing is complete.
# Derived from RowOutcome.is_terminal, excluding delegation markers
# (FORKED/EXPANDED delegate completion to child tokens).
_delegation = {RowOutcome.FORKED, RowOutcome.EXPANDED}
terminal_outcome_values = [o for o in RowOutcome if o.is_terminal and o not in _delegation]

# Subquery: Tokens with terminal outcomes
terminal_tokens = (
    select(token_outcomes_table.c.token_id)
    .where(token_outcomes_table.c.run_id == run_id)
    .where(token_outcomes_table.c.is_terminal == 1)
    .where(token_outcomes_table.c.outcome.in_(terminal_outcome_values))
).scalar_subquery()

# NEW:
# Delegation markers: FORK_PARENT and EXPAND_PARENT paths. Both carry
# outcome=TRANSIENT — the parent's lifecycle answer lives on its children.
_DELEGATION_PATHS = (TerminalPath.FORK_PARENT.value, TerminalPath.EXPAND_PARENT.value)

delegation_tokens = (
    select(token_outcomes_table.c.token_id)
    .where(token_outcomes_table.c.run_id == run_id)
    .where(token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
).scalar_subquery()

# Terminal-token query: rows that are completed (completed=1) AND not
# delegation markers. Under ADR-019, "completed" equates to the renamed
# is_terminal column; delegation paths are excluded so the parent doesn't
# count as terminating its row's processing — the child tokens do.
terminal_tokens = (
    select(token_outcomes_table.c.token_id)
    .where(token_outcomes_table.c.run_id == run_id)
    .where(token_outcomes_table.c.completed == 1)
    .where(~token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
).scalar_subquery()
```

**Step 3: Update the `rows_with_terminal` query (lines 400-412)**

```python
# OLD:
rows_with_terminal = (
    select(tokens_table.c.row_id)
    .distinct()
    .select_from(...)
    .where(token_outcomes_table.c.run_id == run_id)
    .where(token_outcomes_table.c.is_terminal == 1)
    .where(token_outcomes_table.c.outcome.in_(terminal_outcome_values))
).scalar_subquery()

# NEW:
rows_with_terminal = (
    select(tokens_table.c.row_id)
    .distinct()
    .select_from(...)  # unchanged join clause
    .where(token_outcomes_table.c.run_id == run_id)
    .where(token_outcomes_table.c.completed == 1)
    .where(~token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
).scalar_subquery()
```

**Step 4: Update imports**

Replace `RowOutcome,` (line 25) with `TerminalPath,`. The recovery module no longer needs `TerminalOutcome`; the path column alone identifies delegation markers.

**Step 5: Verify**

Run: `.venv/bin/python -m pytest tests/unit/core/checkpoint/test_recovery.py -q`

Expected: tests pass; the SQL query produces equivalent semantics with the renamed columns.

**Step 6: Remove from allowlist**

Remove `src/elspeth/core/checkpoint/recovery.py` from the allowlist.

**Definition of Done:**
- [ ] Recovery queries flipped to use `completed`, `path`, and the delegation-path filter
- [ ] Imports updated (RowOutcome dropped, TerminalPath added)
- [ ] Allowlist trimmed
- [ ] All recovery tests pass
- [ ] mypy passes

---

### Task 2.6: Phase 2 commit + verification

**Step 1: Run the lint guard to confirm src/ scope is now zero**

After Phase 2's edits, the `src/` migration scope drops from 134 references to 0. Run:

```bash
grep -rn "RowOutcome\." src/elspeth/engine/ src/elspeth/core/checkpoint/
```

Expected: empty output.

```bash
grep -rn "RowOutcome\." src/elspeth/
```

Expected: hits only in `src/elspeth/contracts/enums.py` (defines RowOutcome through Stage 4) and `src/elspeth/testing/__init__.py` (re-exports through Stage 4). The four contract files (`audit.py`, `events.py`, `engine.py`, `results.py`) should also be clean post-Phase-1.

**Step 2: Run the lint guard with updated allowlist**

```bash
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome
```

Expected: `exit=0`. The allowlist now lists only:
- `src/elspeth/contracts/enums.py` (RowOutcome lives here through Stage 4)
- `src/elspeth/testing/__init__.py` (Stage 4 trims the re-exports)
- `tests/` (Stage 4 mechanical sweep)

**Step 3: Run engine + integration tests**

```bash
.venv/bin/python -m pytest tests/unit/engine/ tests/integration/ -q --timeout=120
```

Expected: most tests pass. Integration tests that read `TokenOutcome.outcome` directly from the audit DB now see `TerminalOutcome.SUCCESS / FAILURE / TRANSIENT`. Some tests will fail with assertion mismatches — those are categorized in Phase 5 (schema-dependent vs assertion-only). Phase 5 fixes the schema-dependent ones.

**Step 4: Note that the accumulator and predicate are still on RowOutcome**

`outcomes.py::accumulate_row_outcomes` still pattern-matches on `RowOutcome.X`. After Phase 2's commit, the accumulator branches will all hit the `else` arm and crash because `RowResult.outcome` is now `TerminalOutcome`, not `RowOutcome`. **This is expected and only resolved by Phase 3.**

**Step 5: Commit**

```bash
git add src/elspeth/engine/ src/elspeth/core/checkpoint/recovery.py \
        config/cicd/forbid_new_row_outcome/migration_files.yaml

git commit -m "$(cat <<'EOF'
feat(adr-019): phase 2 — producer site flip, src/ scope reaches zero

ADR-019 Stage 2/3 Phase 2 of 5 (see docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md).

Flipped 36 producer emit sites across:
- engine/processor.py (28 sites — every RowResult / record_token_outcome construction)
- engine/executors/transform.py (1 site)
- engine/executors/sink.py (3 sites — primary FAILED, failsink-mode DIVERTED, discard-mode DIVERTED)
- engine/coalesce_executor.py (4 sites — flush + failure paths)

Plus the resume-reader query in core/checkpoint/recovery.py: flipped from
``RowOutcome.in_([FORKED, EXPANDED])`` to ``TerminalPath.in_([FORK_PARENT,
EXPAND_PARENT])`` and ``is_terminal == 1`` to ``completed == 1``.

The discard-mode DIVERTED flip at sink.py:998 ships the operator-visible
behaviour change from ADR-019 § Sub-decision 5: ``(FAILURE, SINK_DISCARDED)``
becomes a predicate input. The companion accumulator change lands in Phase 3.

Allowlist trimmed: src/elspeth/engine/* and src/elspeth/core/checkpoint/recovery.py
removed from config/cicd/forbid_new_row_outcome/migration_files.yaml.
Remaining allowlist scope: contracts/enums.py (defines RowOutcome through
Stage 4), testing/__init__.py (re-exports), tests/ (Stage 4 sweep).

This commit alone breaks accumulate_row_outcomes — every match arm there
still pattern-matches on RowOutcome.X but RowResult.outcome is now
TerminalOutcome. Phase 3 is the accumulator + predicate flip that restores
end-to-end engine execution. The merge is atomic per ADR-019 lines 318-320.

Refs: elspeth-edb60744f0 (Stage 3 ticket — producer + accumulator)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Definition of Done:**
- [ ] All 36 producer sites flipped
- [ ] Resume-reader query updated for `completed` + `path`
- [ ] Allowlist trimmed for the engine/ and recovery.py paths
- [ ] mypy clean
- [ ] Engine module imports cleanly
- [ ] Phase 2 commit landed
- [ ] Phase 3 starts in the next session/checkpoint
