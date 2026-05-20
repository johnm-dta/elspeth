# Multi-Source Token Scheduler Second-Pass Review

Target worktree: `/home/john/elspeth/.worktrees/multi-source-token-scheduler`

Scope: second-pass adversarial review against `notes/engine_deep_dive.md`, excluding the first-pass findings listed there. I reviewed the live worktree, committed diff against `origin/RC5.2`, uncommitted diff, and the two example directories. I used the determinism/replay lens for lease ownership, durable state transitions, replay coverage, external process outcomes, and example reproducibility.

## A. Unchecked Mutations Beyond Scheduler Repo

IMPORTANT PROBABLE-NON-ISSUE `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/run_lifecycle_repository.py:402`

Evidence:

```python
self._ops.execute_insert(
    run_sources_table.insert().values(...),
    context="run_sources",
)
```

Reasoning: `record_run_source()` uses the shared `DatabaseOps.execute_insert()` path, which turns SQLAlchemy insert failures and zero-row insert results into `LandscapeRecordError`, an `AuditIntegrityError` subclass. Duplicate `(run_id, source_name)` therefore does not leak as a generic `IntegrityError` at the repository boundary.

Proposed fix: none.

IMPORTANT PROBABLE-NON-ISSUE `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/data_flow_repository.py:433`

Evidence:

```python
row_id = row_id or generate_id()
source_row_index = row_index if source_row_index is None else source_row_index
ingest_sequence = row_index if ingest_sequence is None else ingest_sequence
```

Reasoning: `create_row()` still has the same legacy identity fabrication shape as the already-filed `_legacy_row_index_default` issue. I am not filing this separately because it is the same compatibility/fabrication root cause, but it is useful reinforcement for that existing ticket: fixing only the SQL default would still leave this Python-side default.

Proposed fix: add this call-site default to the existing first-pass legacy identity ticket.

## B. Lease Lifecycle Edges

CRITICAL DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/scheduler_repository.py:190`

Evidence:

```python
def recover_expired_leases(self, *, run_id: str, now: datetime) -> int:
    ...
    .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
    .where(token_work_items_table.c.lease_expires_at < now)
```

Related evidence:

```python
def mark_terminal(self, *, work_item_id: str, now: datetime) -> TokenWorkItem:
    return self._transition(work_item_id=work_item_id, ...)
```

and:

```python
conn.execute(
    update(token_work_items_table)
    .where(token_work_items_table.c.work_item_id == work_item_id)
    .values(**update_values)
)
```

Reasoning: expired leases are intentionally returned to `READY`, and another worker can reclaim the same `work_item_id`. The original worker can then finish late and call `mark_terminal()`, `mark_failed()`, `mark_blocked()`, or `mark_waiting()` by `work_item_id` only. `_transition()` does not constrain current status or `lease_owner`, so a stale owner can overwrite work that has already been recovered and leased to someone else. This is not the first-pass rowcount issue; even with rowcount checked, the wrong owner can mutate the row.

Proposed fix: carry the expected `lease_owner` from the claim into every claimed-item transition. Update with `work_item_id AND status = LEASED AND lease_owner = :owner`; zero rows should raise an audit/stale-lease conflict. If terminal idempotency remains, implement it as a no-op verification path, not an unconditional rewrite.

IMPORTANT DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/outcomes.py:454`

Evidence:

```python
for outcome in timed_out:
    if _validate_coalesce_outcome(outcome):
        _process_merged_coalesce_outcome(...)
    else:
        counters.rows_coalesce_failed += 1
        counters.rows_failed += len(outcome.consumed_tokens)
        _emit_failed_coalesce_telemetry(ctx, outcome.consumed_tokens)
```

Related success path:

```python
processor.mark_blocked_barrier_terminal(
    str(coalesce_name),
    tuple(token.token_id for token in outcome.consumed_tokens),
)
```

Reasoning: timeout-driven failed coalesces count the consumed tokens as failed, but they do not terminalize the durable `BLOCKED` scheduler rows at the point the failure is recorded. End-of-source cleanup can eventually sweep blocked barriers, but if the process stops after the timeout failure and before that cleanup, replay sees active blocked work for tokens whose coalesce failure was already emitted.

Proposed fix: in the timeout failure branch, call `processor.mark_blocked_barrier_terminal(str(coalesce_name), tuple(token.token_id for token in outcome.consumed_tokens))` and, after the known rowcount-discard issue is fixed, assert the count matches the consumed token count.

SUGGESTION REINFORCEMENT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/processor.py:453`

Evidence:

```python
self._scheduler_lease_owner = scheduler_lease_owner or f"row-processor:{run_id}"
```

Reasoning: the default lease owner is run-scoped, not worker-scoped. The current engine may be single-process in practice, but the schema already models leases and ownership. Once owner checks are added, two processors for the same run would still share the same default owner string.

Proposed fix: make the default owner process/session scoped, for example including a pid plus a UUID or orchestrator execution id.

## C. Exception Chain Integrity

IMPORTANT DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/processor.py:2422`

Evidence:

```python
except Exception:
    self._scheduler.mark_failed(work_item_id=claimed.work_item_id, now=datetime.now(UTC))
    raise
```

Reasoning: the broad catch intentionally records scheduler failure for any `_process_single_token()` exception, including audit/invariant failures. If `mark_failed()` itself raises, it masks the original processing exception and loses the causal chain at exactly the audit boundary that should be most explainable.

Proposed fix: catch as `exc`, wrap the scheduler failure write in its own `try/except`, and preserve both causes. If the scheduler write fails, raise an `AuditIntegrityError` chained from the scheduler exception and include the original exception type/message in diagnostic context; otherwise re-raise the original exception unchanged.

## D. Scheduler State Machine Gaps

CRITICAL DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/scheduler_repository.py:381`

Evidence:

```python
update_values = {"status": status.value, "updated_at": now, **values}
with self._engine.begin() as conn:
    conn.execute(
        update(token_work_items_table)
        .where(token_work_items_table.c.work_item_id == work_item_id)
        .values(**update_values)
    )
```

Related test evidence:

```python
completed = repo.mark_blocked_barrier_terminal(...)
assert completed == 1
...
blocked = repo.mark_blocked(work_item_id=item.work_item_id, ...)
assert blocked.status is TokenWorkStatus.BLOCKED
```

Reasoning: all four per-item transition methods route through an unconditional status rewrite. That means `TERMINAL -> BLOCKED`, `FAILED -> TERMINAL`, `TERMINAL -> WAITING`, and other impossible ordered pairs are allowed if the caller has a `work_item_id`. The unit test at `tests/unit/core/test_multi_source_foundation.py:1120` currently codifies reopening a terminal row to `BLOCKED`, so the illegal transition is not theoretical.

Proposed fix: add a mechanical transition table close to `TokenWorkStatus`, enforce legal `(from, to)` pairs in the repository with current-status predicates, and make `TERMINAL`/`FAILED` absorbing. Preserve `TERMINAL -> TERMINAL` only as a verified no-op if idempotency is required.

## E. Concurrent-Worker Assumptions

CRITICAL DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/processor.py:2392`

Evidence:

```python
self._scheduler.release_waiting(run_id=self._run_id, now=now)
self._scheduler.recover_expired_leases(run_id=self._run_id, now=now)
claimed = self._scheduler.claim_ready(
    run_id=self._run_id,
    lease_owner=self._scheduler_lease_owner,
    ...
)
```

Reasoning: the processor actively participates in lease recovery before claiming work, so the code is not merely single-process. But after claim, completion/block/fail paths do not pass the owner back to the repository. That makes replay schedule-dependent: the order in which two workers recover, reclaim, finish, and terminalize can change durable outcomes without any recorded schedule or ownership conflict.

Proposed fix: choose and document the concurrency strategy. If durable scheduler work is single-worker-only for this release, make that invariant mechanical. If multi-worker is supported, transitions need owner fencing and replay needs durable conflict outcomes rather than silent last-writer-wins status changes.

IMPORTANT PROBABLE-NON-ISSUE `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/core.py:3091`

Evidence:

```python
if not interrupted_by_shutdown and processor.has_scheduled_work():
    recovered_row_ids = frozenset(row[0] for row in unprocessed_rows)
    scheduled_row_ids = processor.active_scheduled_row_ids()
    uncovered_row_ids = recovered_row_ids - scheduled_row_ids
    if uncovered_row_ids:
        raise AuditIntegrityError(...)
```

Reasoning: I specifically checked the "scheduler work exists, so unrepresented resume rows are skipped" risk. The live tree now has a coverage assertion before clearing `unprocessed_rows`, so that issue is not present in the reviewed state.

Proposed fix: none.

## F. Schema Epoch And Migration

IMPORTANT DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/database.py:96`

Evidence:

```python
("token_work_items", "work_item_id"),
("token_work_items", "status"),
("token_work_items", "available_at"),
("token_work_items", "row_payload_json"),
("token_work_items", "on_success_sink"),
```

Related schema evidence:

```python
# 12 -> Durable scheduler resume identity
SQLITE_SCHEMA_EPOCH = 12
...
Column("branch_name", String(128)),
Column("fork_group_id", String(128)),
Column("join_group_id", String(128)),
Column("expand_group_id", String(128)),
Column("coalesce_node_id", String(NODE_ID_COLUMN_LENGTH)),
Column("coalesce_name", String(128)),
```

Related validation/stamp evidence:

```python
epoch_incompatible = present_landscape_tables and schema_epoch not in (0, SQLITE_SCHEMA_EPOCH)
...
if current_epoch < SQLITE_SCHEMA_EPOCH:
    self._set_sqlite_schema_epoch(SQLITE_SCHEMA_EPOCH)
```

Reasoning: epoch 12 adds durable scheduler resume identity columns, but `_REQUIRED_COLUMNS` only gates `token_work_items` through epoch 11's `on_success_sink`. Existing unversioned SQLite databases (`PRAGMA user_version = 0`) can pass validation without the epoch-12 columns and then be stamped as epoch 12. The next scheduler insert expects `branch_name`, `fork_group_id`, `join_group_id`, `expand_group_id`, `coalesce_node_id`, and `coalesce_name`, so the failure is delayed from schema-open time to runtime.

Proposed fix: add every runtime-required epoch-12 `token_work_items` column to `_REQUIRED_COLUMNS`. Add a regression where an unversioned DB with epoch-11 `token_work_items` shape is rejected before `_sync_sqlite_schema_epoch()`.

IMPORTANT PROBABLE-NON-ISSUE `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/schema.py:215`

Evidence:

```python
Column("source_row_index", Integer, nullable=False, default=_legacy_row_index_default),
UniqueConstraint("run_id", "source_node_id", "source_row_index"),
```

Reasoning: SQLite's "multiple NULLs are allowed under UNIQUE" behavior does not apply here because `source_row_index` is `NOT NULL`. The fabrication default is already a first-pass issue, but the NULL-tolerance trap is not present.

Proposed fix: none beyond the existing default-fabrication ticket.

## G. Process-Pool Isolation

IMPORTANT DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/plugins/transforms/rag/query.py:132`

Evidence:

```python
future = self._regex_pool.submit(run_regex_worker, self._compiled_pattern, extracted)
try:
    matched, group0, group1 = future.result(timeout=self._regex_timeout)
except FuturesTimeoutError:
    ...
    return QueryResult(
        error=TransformErrorReason(
            reason="no_regex_match",
            field=self._query_field,
            cause="regex_timeout",
        )
    )
```

Related contract/test evidence:

```python
"no_regex_match",
...
cause: NotRequired[str]
```

and:

```python
assert result.error["reason"] == "no_regex_match"
assert result.error["cause"] == "regex_timeout"
```

Reasoning: a deterministic semantic regex miss and a wall-clock/process-pool timeout are different replay classes, but the primary audit reason is the same. The cause side-channel helps humans, but downstream classifiers using the primary `reason` cannot distinguish "input did not match" from "resource/time budget interrupted evaluation."

Proposed fix: add a primary reason such as `regex_timeout` to the transform error reason contract and tests. Include `pattern`, `max_seconds`, and elapsed/worker outcome metadata where available.

SUGGESTION REINFORCEMENT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/plugins/transforms/rag/query.py:149`

Evidence:

```python
stuck_processes = list(self._regex_pool._processes.values())
self._regex_pool.shutdown(wait=False, cancel_futures=True)
...
self._regex_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
```

Reasoning: synchronous transform execution appears single-caller today, so I am not filing this as a present defect. If this builder becomes concurrent, one row can kill the shared worker used by another row, and timeout can include queue delay behind another regex. That scheduling dependency is not represented in row audit.

Proposed fix: either guard `build()` with an explicit single-caller lock and comment, or isolate each regex evaluation enough to separate queue wait from execution timeout.

## H. Examples As Contract

IMPORTANT DEFECT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/examples/multi_flow/README.md:12`

Evidence:

```bash
elspeth run --settings examples/multi_flow/settings.yaml --execute
```

Related settings:

```yaml
landscape:
  url: sqlite:///examples/multi_flow/runs/audit.db
```

Related local state checked during review:

```text
examples/multi_flow/runs/audit.db PRAGMA user_version = 11
examples/multi_source_queue/runs/audit.db PRAGMA user_version = 11
```

Reasoning: both example settings use fixed audit DB paths, and both current example `runs/audit.db` files are epoch 11 while the branch schema is epoch 12. The documented command is therefore not repeat-run safe on the live worktree: it will hit the project's fresh-DB/delete-old-DB policy before producing the README's spot-check output. Fresh runs may match the row counts, but the README contract omits the cleanup step needed on this branch.

Proposed fix: either remove the stale example DB artifacts and document `rm examples/.../runs/audit.db` before reruns/schema changes, or make the examples write to a fresh run-specific DB path.

SUGGESTION PROBABLE-NON-ISSUE `/home/john/elspeth/.worktrees/multi-source-token-scheduler/examples/multi_source_queue/README.md:19`

Evidence:

```text
Expected result: examples/multi_source_queue/output/combined.jsonl contains three rows, two from orders.csv and one from refunds.csv.
```

Reasoning: the README row-count claims align with the settings and inputs. The contract problem is repeat-run freshness, not the stated counts.

Proposed fix: none for the counts.

## I. Project Convention Drift

SUGGESTION REINFORCEMENT `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/core.py:3444`

Evidence:

```python
source_records_candidate = factory.run_lifecycle.get_run_source_resume_records(run_id)
if isinstance(source_records_candidate, Mapping) and source_records_candidate:
```

Reasoning: `get_run_source_resume_records()` is an internal typed repository call returning `dict[str, RunSourceResumeRecord]`. The `isinstance(..., Mapping)` guard means an internal contract drift would silently fall back to the legacy single-source resume path instead of crashing at the Tier-1 boundary. This is convention drift, not a standalone runtime defect observed in the current tree.

Proposed fix: trust the typed return and branch only on emptiness:

```python
source_records = factory.run_lifecycle.get_run_source_resume_records(run_id)
if source_records:
    ...
```

SUGGESTION PROBABLE-NON-ISSUE `/home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/types.py:1`

Evidence: I checked the new/changed frozen dataclasses with mapping/sequence fields. `PipelineConfig`, `GraphArtifacts`, `RunContext`, and `ResumeState` freeze their mapping and sequence containers in `__post_init__`.

Reasoning: no frozen-container drift found in this surface.

Proposed fix: none.

## What I Looked At And Ruled Out

- Read and grepped the target feature surface under `/home/john/elspeth/.worktrees/multi-source-token-scheduler`: scheduler repository, schema, data flow repository, run lifecycle repository, processor, orchestrator core/outcomes/types, scheduler contracts, and regex worker.
- Checked committed branch diff with `git diff --stat origin/RC5.2..HEAD -- ...` and uncommitted diff with `git diff --stat -- ...`.
- Checked `claim_ready()` rowcount/ownership verification and did not re-file the known first-pass rowcount issue.
- Checked `recover_expired_leases()` and `release_waiting()` for terminal-row reopening. Their status predicates avoid `TERMINAL -> READY`; the remaining defect is stale claimed-owner mutation after recovery.
- Checked `on_success_sink` persistence and rehydration. Enqueue and scheduler rehydration both preserve it, so I did not file a round-trip defect.
- Checked `run_sources` duplicate handling and row insert mechanics. They go through `DatabaseOps.execute_insert()` rather than unchecked raw inserts.
- Checked SQLite `UNIQUE(run_id, source_node_id, source_row_index)` NULL behavior. The column is `NOT NULL`; no NULL-tolerance trap.
- Checked resume with mixed scheduler and unrepresented rows. The live tree now raises `AuditIntegrityError` on uncovered rows instead of silently dropping them.
- Checked for `hasattr()`, object `getattr(..., default)`, and new `.get(...)` defensive lookups in the target surface. No actionable hits found.
- Checked broad exception handling and wrapper raises. Most wrapper raises preserve `from exc`; the actionable chain problem is the scheduler failure write inside the broad catch in `RowProcessor._drain_scheduler_claims()`.
- Checked the RAG regex worker path. Worker crashes are chained as `RuntimeError` and were not filed; the filed issue is the collapsed timeout/no-match audit reason.

Additional findings from another agent:
• Completed the second-pass static review of .worktrees/multi-source-token-scheduler against origin/RC5.2 plus dirty worktree state. I did not edit source files or run examples/tests; I only read code and updated Filigree.

  Created 8 new Filigree bugs:

  - elspeth-3f2e217f35 P0: aggregation buffered tokens are marked scheduler-terminal before flush consumption.
  - elspeth-8d96832650 P0: resume drains scheduler work by run-level presence and skips uncovered recovered rows.
  - elspeth-b3e139e322 P1: run_sources.source_node_id lacks graph-node FK.
  - elspeth-f1acdc34fb P1: epoch-12 scheduler resume columns missing from schema compatibility gate.
  - elspeth-7d6e5c6f4f P1: multi-source runs flush aggregation/coalesce state after each source, splitting cross-source batches.
  - elspeth-397e630eae P1: non-primary source-boundary failures are attributed to primary source.
  - elspeth-af72715071 P2: RAG regex timeout is top-level reported as no_regex_match.
  - elspeth-6ea361a160 P2: duplicate scheduler enqueue leaks raw DB errors instead of project audit errors.

  Updated/commented existing tickets:

  - elspeth-c1e298c4ed: parented under scheduler epic and added FK/sentinel-terminal reinforcement.
  - elspeth-cdf9b481cd: added lease-owner/source-status transition reinforcement.
  - elspeth-f9625e8692: added current resume-only source_on_success residual.
  - elspeth-a6c4f4e915: added README audit SQL ambiguity for persistent example DBs.

  Overall assessment: the branch has a promising shape, especially around explicit multi-source graph/root identity, but it is not merge-ready. The remaining scheduler/resume defects are not polish; they hit durable state, audit attribution, and recovery correctness. The P0/P1 items
  should be fixed before this work becomes an RC core branch. Goal usage: 544,593 tokens, about 15 minutes elapsed.
