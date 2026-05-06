# ADR-019 Stage 2/3 — Phase 2: Producer Site Flip

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phase 2 is a local checkpoint only, not a git commit boundary: it leaves the accumulator and resume aggregation broken because they still match on `RowOutcome` while producers emit `(outcome, path)`. Do NOT commit, push, or propose to land Phase 2 alone. Continue directly to Phase 3, and create the first git commit only after Phase 3 completes the atomic Stage 2/3 migration.

**Goal:** Flip every `RowOutcome.X` reference at the producer emit site to a `(outcome=TerminalOutcome.Y, path=TerminalPath.Z)` pair construction, per the canonical mapping table at `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`. After this phase, no `RowOutcome.X` references remain in `src/elspeth/engine/`, `src/elspeth/core/checkpoint/`, or anywhere else producers run — only `src/elspeth/contracts/enums.py` (defines RowOutcome through Stage 4) and `src/elspeth/testing/__init__.py` (re-exports through Stage 4).

**Files touched in this phase:**

- Modify: `src/elspeth/engine/processor.py` (28 emit sites)
- Modify: `src/elspeth/engine/executors/transform.py` (1 emit site)
- Modify: `src/elspeth/engine/executors/sink.py` (4 `record_token_outcome` emit sites — primary failure, primary durable success, failsink-mode DIVERTED, discard-mode DIVERTED)
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
| `DIVERTED` (failsink mode, `sink_name = failsink_name`) | `(TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK)` | `sink_name`, `sink_node_id`, `artifact_id`, `error_hash` |
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

Below is the exhaustive enumeration of every `outcome=RowOutcome.X` site in `processor.py` after the Phase 1 checkpoint. Each line maps to its target pair via the canonical table above.

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

**Step 5: Retype `_emit_token_completed` and update callers**

`_emit_token_completed` is the telemetry call at processor sites that wrap a `RowOutcome` in a `TokenCompleted` event. Phase 1 retyped `TokenCompleted.outcome` to `TerminalOutcome | None` and added `path: TerminalPath`, but the helper implementation itself lives in `processor.py` and must be retyped in this Phase 2 task before the callers are updated.

Update the helper signature and event construction first:

```python
# OLD:
def _emit_token_completed(
    self,
    token: TokenInfo,
    outcome: RowOutcome,
    sink_name: str | None = None,
) -> None:
    ...
    self._emit_telemetry(
        TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id=self._run_id,
            row_id=token.row_id,
            token_id=token.token_id,
            outcome=outcome,
            sink_name=sink_name,
        )
    )

# NEW:
def _emit_token_completed(
    self,
    token: TokenInfo,
    *,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    sink_name: str | None = None,
) -> None:
    ...
    self._emit_telemetry(
        TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id=self._run_id,
            row_id=token.row_id,
            token_id=token.token_id,
            outcome=outcome,
            path=path,
            sink_name=sink_name,
        )
    )
```

Then update each caller:

```python
# OLD (e.g. line 696):
self._emit_token_completed(token, RowOutcome.FAILED)

# NEW:
self._emit_token_completed(
    token,
    outcome=TerminalOutcome.FAILURE,
    path=TerminalPath.UNROUTED,
)
```

**Step 6: Update imports**

Replace `from elspeth.contracts import RouteDestination, RowOutcome, RowResult, SourceRow, TokenInfo, TransformResult` (line 21) with:

```python
from elspeth.contracts import (
    RouteDestination,
    RowResult,
    SourceRow,
    TokenInfo,
    TransformResult,
)
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
```

`processor.py` no longer references `RowOutcome` after this task. The lint guard at `scripts/cicd/forbid_new_row_outcome.py` will start failing on `processor.py` if it remains in the migration allowlist — the atomic Phases 1-3 commit removes `src/elspeth/engine/processor.py` from `config/cicd/forbid_new_row_outcome/migration_files.yaml` to reflect the completed flip.

**Step 7: Verify the file compiles and unit tests run**

Run: `.venv/bin/python -c "from elspeth.engine import processor; print('OK')"`

Expected: `OK`.

Add or update a focused engine producer test before this step that exercises
`_emit_token_completed` through a real processor path and asserts the emitted
`TokenCompleted` event carries both axes:

```python
assert event.outcome == TerminalOutcome.SUCCESS  # or FAILURE/TRANSIENT for the path under test
assert event.path == TerminalPath.DEFAULT_FLOW   # use the canonical pair for that producer site
```

The Phase 1 contract test only proves the event dataclass shape. This Phase 2
test proves the engine producer helper and its callers were actually migrated.

Run: `.venv/bin/python -m pytest tests/unit/engine/ -q`

Expected: tests that exercise `processor.py` directly pass. Tests that assert on `RowOutcome.X` from processor outputs may fail at this local checkpoint; Phase 5 translates those assertion-only sites in this PR before the full-suite gate.

**Definition of Done:**
- [ ] Verify `from elspeth.contracts.enums import TerminalOutcome, TerminalPath` succeeds (both symbols defined in Phase 1 Task 1.1)
- [ ] All 28 emit sites in `processor.py` flipped to (outcome, path) pairs
- [ ] `_emit_token_completed` helper signature retyped to keyword-only `(outcome, path)`
- [ ] `_emit_token_completed` callers updated (~15 sites in `processor.py`)
- [ ] Engine producer test asserts emitted `TokenCompleted` carries the expected `(outcome, path)` pair
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

### Task 2.4: Producer flip — `sink.py` (4 sites with the load-bearing DIVERTED two-flavor split)

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

**Step 2: Apply the normal durable primary-write success edit (line 633)**

The main successful sink path records the pending outcome for every primary token after the sink returns an artifact. This call currently passes only `pending_outcome.outcome`; after Phase 1 the recorder requires both axes. This is a live producer site and must be updated in Phase 2, not left for Phase 3.

```python
# OLD (lines 631-638):
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=pending_outcome.outcome,
    error_hash=pending_outcome.error_hash,
    sink_name=sink_name,
)

# NEW:
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=pending_outcome.outcome,
    path=pending_outcome.path,
    error_hash=pending_outcome.error_hash,
    sink_name=sink_name,
)
```

Add or update a sink-executor test that drives a successful primary write and asserts the recorded `token_outcomes` row carries `(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)` for a default pending outcome. This catches future omissions at the normal durable path, separate from the failsink/discard diversion paths.

**Step 3: Apply the failsink-mode DIVERTED edit (line 952)**

This is the `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` flavor — the failsink absorbed the row for visibility. The lifecycle answer lives on the failsink's `NodeStateStatus.COMPLETED` `node_state` plus the registered `artifacts` row. Capture the `Artifact` returned by `register_artifact(...)` and pass both the exact `failsink_node_id` and exact `artifact_id` into `record_token_outcome(...)` so Phase 4's I1c invariant validates the intended failsink write, not merely any completed sink or any artifact for the run.

```python
# OLD (lines 952-957):
self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=RowOutcome.DIVERTED,
    error_hash=error_hash,
    sink_name=failsink_name,
)

# NEW:
failsink_artifact = self._execution.register_artifact(
    run_id=self._run_id,
    state_id=first_fs_state.state_id,
    sink_node_id=failsink_node_id,
    artifact_type=failsink_artifact_info.artifact_type,
    path=failsink_artifact_info.path_or_uri,
    content_hash=failsink_artifact_info.content_hash,
    size_bytes=failsink_artifact_info.size_bytes,
)

self._data_flow.record_token_outcome(
    ref=TokenRef(token_id=token.token_id, run_id=self._run_id),
    outcome=TerminalOutcome.TRANSIENT,
    path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
    error_hash=error_hash,
    sink_name=failsink_name,
    sink_node_id=failsink_node_id,
    artifact_id=failsink_artifact.artifact_id,
)
```

**Step 4: Apply the discard-mode DIVERTED edit (line 998) — OPERATOR-VISIBLE BEHAVIOUR CHANGE**

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

**Step 5: Update imports**

Replace `RowOutcome,` (line 17) with `TerminalOutcome,\n    TerminalPath,`.

**Step 6: Remove from allowlist**

Remove `src/elspeth/engine/executors/sink.py` from the allowlist.

**Step 7: Verify**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_sink_executor_diversion.py -q` (and any related sink tests in `tests/integration/`). Do not use `tests/unit/engine/executors/test_sink_executor.py`; that file does not exist in current HEAD.

Expected: tests pass if they do not make stale `RowOutcome.X` assertions. Tests that still assert on `RowOutcome.X` are Category B and must be translated in Phase 5 before PR open. Tests that read `TokenOutcome` from the audit DB now see `TerminalOutcome.FAILURE` / `TerminalPath.SINK_DISCARDED` for discards.

**Definition of Done:**
- [ ] Four `record_token_outcome` emit sites flipped (primary failure, primary durable success, failsink-mode, discard-mode)
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

**Step 5: Add the RED-first delegation-path query test**

Add a focused regression in `tests/unit/core/checkpoint/test_recovery.py` before
changing the recovery query:

```python
def test_get_unprocessed_rows_uses_terminal_path_delegation_set(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    delegation_path: TerminalPath,
) -> None:
    """ADR-019: resume recovery classifies fork/expand parents by path, not outcome.

    The parent-token lifecycle row is now
    (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT/EXPAND_PARENT).
    A query that still looks for RowOutcome.FORKED/EXPANDED in
    token_outcomes.outcome silently misses the delegation marker and can mark
    a partially completed fork/expand row as done.
    """
    run_id = "run-resume-delegation"
    row_id = "row-resume-delegation"
    parent_token_id = "token-resume-delegation-parent"
    with db.transaction() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED)
        _insert_node(conn, run_id, "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, run_id, row_id, row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, parent_token_id, row_id)
        _insert_terminal_outcome(
            conn,
            run_id,
            parent_token_id,
            outcome=TerminalOutcome.TRANSIENT,
            path=delegation_path,
            completed=True,
        )

    recovery = RecoveryManager(db, checkpoint_manager)
    unprocessed = recovery.get_unprocessed_rows(run_id)
    assert [row.row_id for row in unprocessed] == [row_id]
```

Mirror the same shape for `TerminalPath.EXPAND_PARENT` or parameterize the test
over both paths. The test must fail before the query flips because the old
`RowOutcome.FORKED` / `RowOutcome.EXPANDED` outcome filter does not match the
new `outcome='transient', path IN _DELEGATION_PATHS` encoding. Keep
`_DELEGATION_PATHS` as a closed tuple of enum values and add a companion unit
assertion:

Extend the existing `_insert_terminal_outcome` helper in this test file to
accept `outcome: TerminalOutcome | None`, `path: TerminalPath`, and
`completed: bool`, because the helper is currently RowOutcome-shaped. That
helper edit is part of the RED setup; do not bypass the new schema by inserting
old `RowOutcome.FORKED` / `RowOutcome.EXPANDED` values.

```python
assert _DELEGATION_PATHS == (
    TerminalPath.FORK_PARENT.value,
    TerminalPath.EXPAND_PARENT.value,
)
```

**Step 6: Verify**

Run: `.venv/bin/python -m pytest tests/unit/core/checkpoint/test_recovery.py -q`

Expected: tests pass; the SQL query produces equivalent semantics with the renamed columns.

**Step 7: Remove from allowlist**

Remove `src/elspeth/core/checkpoint/recovery.py` from the allowlist.

**Definition of Done:**
- [ ] Recovery queries flipped to use `completed`, `path`, and the delegation-path filter
- [ ] RED-first recovery test proves `FORK_PARENT` and `EXPAND_PARENT` are detected through `_DELEGATION_PATHS`
- [ ] `_DELEGATION_PATHS` is pinned as the closed set `(TerminalPath.FORK_PARENT.value, TerminalPath.EXPAND_PARENT.value)`
- [ ] Imports updated (RowOutcome dropped, TerminalPath added)
- [ ] Allowlist trimmed
- [ ] All recovery tests pass
- [ ] mypy passes

---

### Task 2.6: Phase 2 local checkpoint + verification (no git commit)

**Step 1: Run the producer-scope RowOutcome check**

After Phase 2's edits, the producer and recovery emit/read sites are clean, but the accumulator and resume aggregation remain intentionally on `RowOutcome` until Phase 3. Run:

```bash
grep -rn "RowOutcome\." \
    src/elspeth/engine/processor.py \
    src/elspeth/engine/executors/transform.py \
    src/elspeth/engine/executors/sink.py \
    src/elspeth/engine/coalesce_executor.py \
    src/elspeth/core/checkpoint/recovery.py
```

Expected: empty output.

```bash
grep -rn "RowOutcome\." src/elspeth/engine/orchestrator/
```

Expected: hits remain in `outcomes.py` and the resume aggregation path in `core.py`. These are Phase 3-owned and must remain on the allowlist until the atomic Phases 1-3 commit.

**Step 2: Run the lint guard with the Phase 2 checkpoint allowlist**

```bash
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome
```

Expected: `exit=0`. The allowlist still includes the Phase 3-owned orchestrator accumulator/resume files:
- `src/elspeth/engine/orchestrator/outcomes.py`
- `src/elspeth/engine/orchestrator/core.py`
- `src/elspeth/contracts/enums.py` (RowOutcome lives here through Stage 5 deletion)
- `src/elspeth/testing/__init__.py` (Stage 5 trims the re-exports)
- `tests/` (compatibility fixtures and mapping-table tests only after Phase 5; no pytest-blocking `outcome == RowOutcome.X` assertions remain)

**Step 3: Run engine + integration tests**

```bash
.venv/bin/python -m pytest tests/unit/engine/ tests/integration/ -q --timeout=120
```

Expected: collection/import should get past the producer sites, but runtime tests that reach `accumulate_row_outcomes` or the resume aggregator may still fail until Phase 3. Integration tests that read `TokenOutcome.outcome` directly from the audit DB now see `TerminalOutcome.SUCCESS / FAILURE / TRANSIENT`. Some tests will fail with assertion mismatches — those are categorized in Phase 5 (schema-dependent vs assertion-only). Phase 5 fixes both categories before the PR opens.

**Step 4: Note that the accumulator and predicate are still on RowOutcome**

`outcomes.py::accumulate_row_outcomes` still pattern-matches on `RowOutcome.X`. At the Phase 2 checkpoint, the accumulator branches will all hit the `else` arm and crash because `RowResult.outcome` is now `TerminalOutcome`, not `RowOutcome`. **This is expected and only resolved by Phase 3. Do not create a git commit here.** Leave the changes in the worktree and continue directly to Phase 3.

**Step 5: Continue without committing**

Do not run `git commit` from the Phase 2 checkpoint. The Phase 1 and Phase 2 files are staged/committed only by the atomic Phases 1-3 commit in Phase 3.

**Definition of Done:**
- [ ] All 37 producer sites flipped
- [ ] Resume-reader query updated for `completed` + `path`
- [ ] Producer/recovery RowOutcome grep is empty
- [ ] Phase 3-owned orchestrator RowOutcome hits remain documented
- [ ] mypy clean
- [ ] Engine module imports cleanly
- [ ] No git commit created at the Phase 2 checkpoint
- [ ] Phase 3 starts in the next session/checkpoint
