# Design: Fix resume re-emitting completed fork/expand branches (F1)

**Date:** 2026-05-30
**Branch:** `fix/resume-fork-reemit` (off RC5.2)
**Status:** DRAFT — out for adversarial review (architecture / systems / python / quality)
**Severity:** P1 (audit-integrity / legal-record corruption)

## Problem

When a checkpoint-enabled pipeline containing a **fork** (gate `fork_to`) or
**expand** (deaggregation) is interrupted after one branch reaches a terminal
state (and is checkpointed) but before a sibling branch completes, resuming the
run **re-emits the already-completed branch**, producing duplicate terminal
outcomes under one `row_id` and a duplicate physical sink write.

### Confirmed reproduction (RED)

`tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_resume_does_not_reemit_completed_fork_branch`
runs a real fork pipeline (gate forks each row to `sink_a` + `sink_b`), deletes
`sink_a`'s terminal outcome to simulate an interrupted branch, then resumes via
the production `orchestrator.resume()` path. Result:

```
Resume re-emitted an already-completed fork branch: {(row_id, 'sink_b'): 2}.
```

`sink_b` (which completed before the interruption) ends with **two** terminal
outcomes. Reproduced from primary source; not inferred.

### Root cause (verified)

1. `recovery.py::get_unprocessed_rows` correctly returns a partial-fork row as
   "unprocessed" — a deliberate prior fix (`P2-2026-01-29-recovery-skips-partial-forks`)
   changed selection to "ALL non-delegation leaf tokens must be terminal."
2. `resume.py::run_resume_processing_loop` reprocesses each such row via
   `processor.process_existing_row(row_id, ...)`, which mints **one new token and
   restarts the whole row from the source** (`_record_source_and_start_traversal`,
   processor.py:1962-1978). It has no knowledge of which branches already
   completed.
3. The restart re-forks to **all** configured branches → the completed branch is
   re-executed and re-recorded.

## Two facts that constrain the fix

1. **Resume sinks append, they are not idempotent.** `SinkPlugin.configure_for_resume()`
   (base.py:1004-1008) switches resume-eligible sinks from truncate to **append**
   mode. So an audit-side dedup (suppress the duplicate `record_token_outcome`)
   would NOT stop the duplicate *physical* write — a CSV/Azure-append sink writes
   the row twice regardless. **The fix must prevent the completed branch from
   re-executing, not just re-recording.**

2. **The original incomplete child token must be accounted for.** The token that
   made recovery flag the row (the incomplete branch's child) persists with no
   terminal outcome. Any fix that mints a fresh parent and re-forks only the
   incomplete branch leaves that *original* child orphaned forever — trading
   double-emission for a never-terminal token (a different audit-integrity
   violation: every non-delegation leaf token must reach a terminal state).

## Relevant existing machinery (verified)

- `processor.process_token(token, ctx, *, current_node_id, coalesce_node_id, coalesce_name)`
  (processor.py:1980) already continues an **existing token from a mid-DAG node** —
  built for mid-pipeline coalesce merges.
- `tokens` table persists per token: `branch_name`, `fork_group_id`,
  `expand_group_id`, `step_in_pipeline`, `row_id` (schema.py:210). `token_parents`
  holds lineage. So an incomplete child token's destination is derivable
  (`branch_name` → `_branch_to_sink`) and its `row_data` is recoverable from the
  payload store.
- Fork-join-balance invariant (`test_fork_join_balance.py`) requires only that
  **every fork child has a parent link** — it does NOT require a fork group to
  have N children. So a fix does not have to re-create suppressed branches.

## Approaches

### Approach 1 — Mid-DAG continuation of incomplete branches (RECOMMENDED)

On resume, for a partial-fork/expand row, **do not restart from source**.
Instead:
1. Query the row's incomplete non-delegation child tokens (tokens with no
   terminal outcome, excluding FORK_PARENT/EXPAND_PARENT).
2. Reconstruct each as a `TokenInfo` (row_data from payload, `branch_name`,
   `fork_group_id`, parent link — all persisted).
3. Continue each from its destination node via `process_token`, under the
   **original** fork/expand parent.

Completed branches keep their run-1 tokens/outcomes untouched; the original
incomplete child is driven to completion in place. No new parent, no orphan, no
re-execution of completed branches.

- **Audit shape:** one fork parent, all children under it, the incomplete one now
  terminal — no second parent, no orphan. NOT "each branch resolved exactly once at
  the node level": re-driving an incomplete branch re-runs the nodes between the
  branch start and the crash point (their post-branch payload is unpersisted, so we
  cannot restart mid-branch). Those re-runs are append-only at an **elevated attempt**
  and carry a **resume-provenance marker** (`node_states.resume_checkpoint_id`, see
  ADDENDUM 2) so an auditor can prove which records came from the resume vs run-1.
  See the bounded non-goal in ADDENDUM 2 for the consequence on side-effecting
  transforms (re-fired external calls are attributable, not silent).
- **Fork-join balance:** preserved by construction.
- **Risk / open question:** requires faithful reconstruction of the incomplete
  child `TokenInfo` and resolution of its resume-start node from `branch_name`.
  Must handle (a) fork→sink branches, (b) fork→coalesce branches (coalesce_name +
  coalesce_node_id), (c) expand/deaggregation children (`expand_group_id`).
  `process_existing_row` (whole-row restart) is still correct for rows that never
  started (no tokens) — the new path applies only to partial-completion rows.

### Approach 2 — New parent for incomplete branches + terminalize the orphan

Mint a new parent that forks only to incomplete branches, AND record a new
"superseded/abandoned" terminal outcome for the original orphan child.

- **Audit shape:** honest but redundant — two fork parents per resumed row, plus a
  new terminal-outcome vocabulary for "abandoned."
- **Pros:** simpler fork-filtering logic.
- **Cons:** pollutes the legal record's shape and expands the audit terminal-state
  vocabulary (`CLOSED LIST` enum change with multi-consumer + DB CHECK sync — the
  exact archetype flagged elsewhere in the RC5.2 review). Wrong tradeoff for an
  auditability system.

## Recommendation

**Approach 1.** It produces the audit trail an auditor expects, reuses
`process_token`, preserves fork-join balance, and invents no new terminal
semantics. Approach 2 is easier to code but corrupts the *shape* of the legal
record and touches the closed terminal-outcome enum.

## Test strategy

- The RED reproduction (`test_resume_does_not_reemit_completed_fork_branch`) must
  go GREEN and assert: exactly one terminal outcome per `(row_id, sink_name)`
  after resume.
- Extend coverage to: (a) fork→coalesce branches, (b) expand/deaggregation
  partial completion, (c) multi-row partial forks, (d) the original incomplete
  child reaches a terminal state (no orphan), (e) fork-join-balance invariants
  (`count_fork_children_missing_parents == 0`, `get_fork_group_stats`) stay green
  after resume, (f) resume-of-resume (idempotent re-run).
- Also confirm the linear-pipeline resume path (`process_existing_row` whole-row
  restart) is unaffected — it remains correct for rows with no tokens.

## REVISED DESIGN (post-review, 2026-05-30) — AUTHORITATIVE

> **Precedence:** This section is refined by **ADDENDUM 1** (node_states attempt collision)
> and **ADDENDUM 2** (provenance marker, coalesce persistence, scope, test discipline) below.
> Where they conflict, the later addendum wins. Read all three before implementing.

Four reviewers (architecture-critic, systems pattern-recognizer, python-code-reviewer,
quality coverage-gap-analyst) reviewed the above. Verdict: **Approach 1, with
material corrections. Approach 2 rejected** (its new terminal-outcome value trips
the closed-enum / DB-CHECK / multi-consumer drift archetype — `_LEGAL_TERMINAL_PAIRS`
+ `accumulate_row_outcomes` + `derive_resume_terminal_status_from_audit` + `_DELEGATION_PATHS`
+ DB CHECK all need synchronized edits). The corrected design:

### Scope (operator decision 2026-05-30: fix BOTH fork and expand here)
- **IN SCOPE: fork branches** (gate `fork_to` → sink, and fork → coalesce).
- **IN SCOPE: partial-expand/deaggregation resume**, via a schema change to persist
  per-token output payloads (operator approved the larger scope).

### Schema change (the enabling piece) — TWO new columns, epoch 11
Two distinct unpersisted facts block a correct fix. Both are resolved by columns added in
the same epoch bump (one door opened once).

**(A) Per-token transformed payload — `tokens.token_data_ref` (THREE writers).**
Two token kinds carry `row_data` that differs from their source row and is created in
memory only, never persisted:
- **Expand children** (`tokens.py:383-393`): independently-transformed per-child rows.
  `expand_token` (`data_flow_repository.py:749`) persists count + ids only.
- **Post-coalesce merged tokens** (`coalesce_tokens`, `data_flow_repository.py:707-722`):
  the merged row is computed in memory at barrier time and the INSERT writes **no** payload
  ref. The COALESCED parent tokens are terminal (can't re-fire the merge), and the parent
  branch payloads are themselves unpersisted — so a merged token is unreconstructable. A
  crash *after* the barrier fires but *before* the post-merge sink/transform completes
  (spec matrix #1, "crash-after-merge-before-downstream") therefore cannot be resumed
  without this.

Fix: **new nullable column `token_data_ref String(64)` on `tokens_table`** (`schema.py:210`),
the payload-store ref for any token whose `row_data` differs from its source row.
- **`expand_token` persists each child's payload**; **`coalesce_tokens` persists the merged
  payload** — both via the existing `payload_store.store(payload_bytes)` pattern (mirror
  `create_row`). Both are Tier-1 audit writes — crash on store failure, no best-effort.
  NULL for **fork children** (they share the parent/source payload, retrievable by `row_id`).
  So three writers: expand + coalesce set it; fork leaves it NULL.
- **Type fidelity (Tier-1):** persist with `checkpoint_dumps` / read with `checkpoint_loads`
  (`core/checkpoint/serialization.py`), NOT `canonical_json` — the latter stringifies
  `datetime`/`Decimal`; the legal record must round-trip the exact Python types run 1 had.
- **Recovery retrieval (one generic path):** reconstructing an incomplete token reads
  `token_data_ref` when set (expand child or post-coalesce token) else `rows.source_data_ref`
  (fork child). Missing/garbage ref where one is required → raise (Tier-1, no coercion).
  Expand children and post-coalesce tokens share this reconstruction, keyed only on which
  contract to apply (expand step's output contract vs the coalesce output contract).

**(B) Resume provenance — `node_states.resume_checkpoint_id` (the attributability marker).**
On resume, the re-drive records `node_states` at an elevated `attempt` (ADDENDUM 1) under
the **same `run_id`** as run 1 (resume reuses the run; there is no distinct resume-run row).
At the `node_states` level, `(run_id, token_id, node_id, attempt=N+1)` is then
**indistinguishable from a run-1 tenacity retry** — only timestamps differ, and timestamps
are non-probative under the "no inference" doctrine. That is an attributability gap in the
legal record. Fix: **new nullable column `resume_checkpoint_id String(64)` on
`node_states_table`** (FK to `checkpoints.checkpoint_id`), NULL for all run-1 records and
set to the resumed-from checkpoint id (`ResumePoint.checkpoint.checkpoint_id`, reachable at
the resume loop) for every `node_state` written during a resume re-drive. `explain()` can
then query-separate a resume re-drive (`resume_checkpoint_id IS NOT NULL`) from a run-1
retry (`IS NULL`) at the same `(token_id, node_id)`. Operator decision 2026-05-30 (faithful
to "explain must prove complete lineage"). It rides the same `WorkItem` carrier and the same
`begin_node_state` sites as the attempt offset (ADDENDUM 1), so threading cost is shared.

**Epoch + backstop:**
- **Epoch bump:** `SQLITE_SCHEMA_EPOCH` 10 → 11 (existing DBs incompatible → operator
  deletes per the delete-the-old-DB policy; no Alembic for the audit store).
- **Co-fix review finding F3:** add **both** new columns AND the two omitted
  `runs.openrouter_catalog_*` columns to `_REQUIRED_COLUMNS` (`database.py`) so the Postgres
  staleness backstop (which bypasses the SQLite-only epoch gate) catches a stale DB. (F3 is
  an independent P1 from the same review; co-landing is cheap and correct.)

### Mechanism (corrected) — mirror the live fork dispatch
On resume, for a partial-fork row, do NOT restart from source. Instead:
1. **Reuse the existing incomplete-token selection** (`recovery.py:454-461`) and the
   shared `_DELEGATION_PATHS` predicate — do not write a second, drifting completion
   query (extract one shared predicate consumed by both recovery selection and resume
   reconstruction).
2. Build `TokenInfo` directly from persisted columns (`token_id, row_id, branch_name,
   fork_group_id, join_group_id, expand_group_id` — all in `tokens_table`; `row_data`
   from the source payload, valid for fork because fork children share the parent's
   deep-copied data). Read columns directly into `TokenInfo(...)` — no `.get`/`getattr`
   defaults; `TokenInfo.__post_init__` crashes on garbage (correct Tier-1 guard).
3. **Resolve the resume-start node by token kind (mirror `_handle_gate_fork`,
   processor.py:2674-2697, NOT `branch_name → destination`).** The dispatch is keyed on the
   persisted token shape; every case ends in `OrchestrationInvariantError` rather than a
   default-route:
   - **fork → sink** (`branch ∈ _branch_to_sink`) → terminal sink path
     (`current_node_id=None`); fork child uses the source payload (`token_data_ref` NULL).
   - **fork → coalesce, crash BEFORE barrier** (`branch ∈ _branch_to_coalesce`, token has a
     `branch_name`) → start at `_branch_first_node[branch]`, pass `coalesce_name`; the
     barrier then fires normally via `_maybe_coalesce_token`. Uses the source payload.
   - **post-coalesce merged token, crash AFTER barrier** (`join_group_id` set;
     `branch_name`/`fork_group_id`/`expand_group_id` all NULL — the shape `coalesce_tokens`
     inserts) → re-drive **downstream of the coalesce node** (resolve via `step_in_pipeline`
     → next node, or `current_node_id=None` if it fed straight to a sink). Payload from
     `token_data_ref` (the merged payload persisted at barrier time), contract = coalesce
     output contract. This is spec matrix #1's after-barrier case (review finding B1): it
     MUST be dispatched, never filtered — an un-dispatched post-coalesce token would fall to
     `process_existing_row` → restart → re-fork → reintroduce F1 for coalesce pipelines.
   - **expand child** (`expand_group_id` set) → re-drive from the node after the expand step;
     payload from `token_data_ref`, contract = expand step's output contract.
   - **none of the above** → audit/DAG inconsistency → **raise `OrchestrationInvariantError`**.
4. Continue under the **original** fork parent (reuse the persisted child token id; do
   NOT mint a new parent). Re-driving a branch re-runs nodes whose run-1 `node_states`
   already exist; the `node_states` unique constraint `(token_id, node_id, attempt)`
   forbids a colliding insert. This is resolved by recording the re-drive at an **elevated
   attempt** (see ADDENDUM 1) — *not* by requiring the resume to start strictly downstream
   of the last recorded position. (An earlier draft of this point asserted the opposite —
   "start strictly downstream… assert it." That clause is **WITHDRAWN**: it directly
   contradicts the attempt-bump mechanism and the RED test, which deletes only the terminal
   outcome and leaves the run-1 `node_state` in place precisely so the re-drive must coexist
   with it. ADDENDUM 1 is authoritative.)
5. `process_existing_row` (whole-row restart) remains correct ONLY for rows with **no
   tokens** (never started). The dispatch boundary (no-tokens → restart; partial → mid-DAG
   continuation) must be explicit.

### Signature change
`process_token(current_node_id: NodeID)` must accept `NodeID | None` (delegating to
`_process_single_token`, which already handles `None` + `_branch_to_sink`,
processor.py:2827-2833) to express the fork→sink terminal case — or add a sibling
continuation entry point. Relaxing the param is the smaller change.

### F1/F2 coupling (do not fully defer F2)
Approach 1 changes the set/shape of `RowResult`s entering `accumulate_row_outcomes`
(resume.py:204). The fix MUST include a **counter-reconciliation assertion** (resumed
counters + already-recorded audit counters == a single uninterrupted from-source run)
or it risks introducing a new `rows_processed` divergence. Treat F2 as coupled.

### Test oracle (corrected — the original was BLIND to the orphan failure mode)
The reproduction's `n != 1` check cannot see a branch with ZERO outcomes (orphan). The
regression suite MUST encode this two-part conservation law:
> After resume, every non-delegation leaf token of a forked row has exactly ONE terminal
> outcome — never zero (orphan), never two (double-emit) — AND the multiset of terminal
> `(row_id, sink_name|coalesce_name)` outcomes equals the uninterrupted baseline exactly.

Concretely: (a) `assert after == baseline` (multiset equality — the test already computes
`baseline` then discards it); (b) a token-level helper asserting zero non-delegation leaf
tokens lack a `completed=1` outcome; (c) assert the resume result status `== COMPLETED`
(currently discarded); (d) physical-write-once on a real append-mode file sink (not just
audit rows — the append-mode double-write is the harm audit-dedup can't fix); (e) pin
`get_fork_group_stats` total_fork_groups unchanged pre/post resume (guards against drift
toward the Approach-2 shape).

### Required test matrix (risk-ordered)
1. fork→coalesce, completed branch already at the barrier (crash-before-merge vs
   crash-after-merge-before-downstream) — turns open Q#3 into tests.
2. all-branches-incomplete (→ whole-row restart still correct) vs some-incomplete (→ mid-DAG).
3. multiple incomplete branches in one fork (3-way: 1 done, 2 incomplete) — determinism (order by persisted `step_in_pipeline`/`token_id`).
4. failure DURING the resumed branch → FAILED terminal, not orphan.
5. aggregation buffer × partial fork on the same row (`recovery.py:449-473` mixed-state exclusion).
6. linear-pipeline regression with an audit-trail assertion (not just in-memory sink).
7. resume-of-resume convergence (monotonic non-increasing incomplete-leaf stock).
8. **partial-expand/deaggregation resume** → full conservation law (one terminal outcome
   per expanded child, original incomplete expand child driven to terminal, expanded-child
   count after resume == original, no orphan, no double-emit) — same oracle as fork, now
   enabled by `token_data_ref` persistence.
9. **schema/recovery for `token_data_ref`**: expand child payload round-trips through
   `payload_store`; a stale DB (pre-epoch-11) is rejected at open with migration guidance
   (SQLite epoch gate) and `_REQUIRED_COLUMNS` rejects a Postgres DB missing the column.

## ADDENDUM 1 (2026-05-30, plan-grounding) — node_states attempt collision on re-drive (AUTHORITATIVE)

Grounding the plan against primary source surfaced a load-bearing gap the REVISED
DESIGN was silent on. It does **not** change the approved approach (Approach 1,
mid-DAG continuation under the original parent); it adds a required mechanism
*within* it. Verified facts:

- `node_states` carries two relevant unique constraints (`schema.py:312-313`):
  `UniqueConstraint("token_id","node_id","attempt")` and
  `UniqueConstraint("token_id","step_index","attempt")`.
- `attempt` is computed **in memory per traversal** (tenacity counts from 0;
  `retry.py:122-141`), **not** derived from the DB. A resume re-drive of a
  reconstructed token therefore restarts at `attempt=0`.
- The sink executor's `begin_node_state` (`sink.py:453`, and failsink `:838`)
  passes **no** `attempt` — it defaults to 0. Coalesce (`coalesce_executor.py:500,563`)
  likewise. Only the transform path threads `attempt` (`state_guard.py:105`).
- Consequence: re-driving an incomplete fork→sink child whose run-1 sink
  `node_state` already exists at `attempt=0` (the **exact** state the RED test
  produces — it deletes only the terminal *outcome*, not the `node_state`)
  collides on `(token_id, node_id, attempt)` and crashes. So correct attempt
  handling is **required to turn the RED test green** — not hypothetical.

### Why re-run from branch start (not mid-branch)
A fork→coalesce child's post-transform `row_data` is created in memory only and is
**not persisted per token** (the same root cause as expand's missing payload). So
we cannot resume from the crash point; we re-run the branch from `branch_first_node`
using the parent/source payload (which *is* persisted), via the existing fork-origin
routing in `DAGNavigator.create_continuation_work_item` (`dag_navigator.py:284-306`).
For fork→sink, the child re-drives straight to the sink (`current_node_id=None`).

### Why attempt-bump (not deletion, not new persistence)
Tier-1 audit is append-only — deleting the stale `node_state` is evidence tampering
(forbidden). Persisting every node's per-token output to enable true mid-branch
resume is a far larger change than the approved scope. The lawful, contained option
is to record the resume re-drive's `node_states` at an **elevated attempt** so they
coexist with run-1's records. All `node_states` written during one token's resume
re-drive share a single elevated `attempt` = a coherent "resume generation" marker.

### Mechanism
1. Add `resume_attempt_offset: int = 0` to the frozen `WorkItem`
   (`dag_navigator.py:35`). Default 0 preserves all existing (non-resume) behavior.
2. When reconstructing an incomplete token for re-drive, compute
   `offset = max(existing attempt over that token's node_states) + 1` (0 if the
   token has no node_states). This is a single Tier-1 read; missing/garbage → raise.
3. Every `begin_node_state` call on the re-drive path **adds the work item's
   `resume_attempt_offset` to its attempt**: transform (via `state_guard` `attempt`),
   sink (`:453` primary, `:838` failsink), coalesce (`:500`, `:563`). The two
   source-state calls (`processor.py:1696`, `core.py:1804`) are **not** on the
   reconstructed-token path and are left unchanged.
4. Within the re-drive, tenacity retries still increment from the offset
   (`base + (attempt_number - 1)`), so a node that retries during resume stays
   collision-free.

### Bounded non-goal (pre-existing resume semantics, not introduced here)
An incomplete append-mode sink branch that physically wrote in run-1 but crashed
before recording its outcome will be **physically written again** on re-drive. This
is the inherent at-least-once property of `configure_for_resume()` append mode and
already applies to the linear whole-row-restart resume path; this fix neither
introduces nor is obligated to solve it. The fix's contract is narrower and exact:
**stop re-emitting *completed* branches.** The conservation-law oracle asserts the
completed-branch invariant; it does not assert at-most-once for the in-flight branch.

## ADDENDUM 2 (2026-05-30, post-adversarial-review) — provenance, coalesce persistence, scope (AUTHORITATIVE)

Four independent reviewers (reality / architecture / systems / quality) reviewed the plan
against primary source. All returned GO-WITH-CHANGES; two findings were NO-GO-class for an
audit system. Operator adjudicated the two doctrine/scope calls on 2026-05-30. This addendum
records the resulting authoritative changes. Where it conflicts with the REVISED DESIGN body
or ADDENDUM 1, this addendum wins.

### A. Resume provenance marker (closes the attributability gap)
Verified: resume reuses the original `run_id` (no distinct resume-run row), so a re-driven
`node_state` at `attempt=N+1` is indistinguishable from a run-1 tenacity retry except by
timestamp (non-probative). **Decision: add `node_states.resume_checkpoint_id` (nullable, FK
to `checkpoints.checkpoint_id`), NULL for run-1, set to `ResumePoint.checkpoint.checkpoint_id`
for every re-drive `node_state`.** It threads on the same `WorkItem` carrier and through the
same `begin_node_state` sites as `resume_attempt_offset` (ADDENDUM 1). The attempt-bump is
still required (the unique constraint does **not** include the new column, so provenance
alone does not avoid the collision). A test MUST prove query-separation, not just non-null:
the same `(token_id, node_id)` shows a run-1 row with `resume_checkpoint_id IS NULL` and a
re-drive row with it set.

### B. Coalesce merged-payload persistence (closes review finding B1)
Verified: `coalesce_tokens` (`data_flow_repository.py:707-722`) persists no payload ref, so a
post-coalesce merged token is unreconstructable and the after-barrier resume case (matrix #1)
cannot be served. **Decision: `coalesce_tokens` persists the merged payload to
`token_data_ref`, making it a three-writer column (expand + coalesce; fork NULL).** The
post-coalesce token is dispatched (Mechanism point 3), never filtered — filtering drops it to
`process_existing_row` and reintroduces F1 for coalesce pipelines. Recovery reconstruction is
one generic path shared with expand children.

### C. Bounded non-goal — intermediate-transform re-execution (now attributable)
Re-driving an incomplete branch from its start re-runs the intermediate transforms between
branch start and the crash point (their post-branch payload is unpersisted). If such a
transform makes an external call (LLM/HTTP), that call **re-fires** on resume and its
`operation_calls` are recorded again. This is the same at-least-once class as the append-mode
sink double-write (already a bounded non-goal below) and is **not** solved here. What this fix
guarantees is that the duplication is **honest**: the re-fired transform's `node_states` (and
the `operation_calls` recorded under them) carry the `resume_checkpoint_id` provenance marker,
so an auditor can prove the second call came from the resume. A test MUST exercise a
**side-effecting** transform on a re-driven branch (not a no-op `PassTransform`) and assert
the duplicate carries the marker. The fix neither prevents nor is obligated to prevent the
physical re-fire; per-node mid-branch payload persistence (which would enable true mid-branch
resume) is a larger change than the approved scope.

### D. Build-order consequence (provenance forces schema-first)
Because the provenance column must be written by the **first** fork re-drive, the schema
change can no longer follow the fork-resume core. Authoritative phase order: oracle →
**schema (epoch 11, both columns)** → **payload persistence (expand + coalesce) + test-callsite
migration** → **resume core (all token kinds, RED→GREEN)** → counters/matrix/gates. No
add-then-remove query filters; persistence is a prerequisite consumed by the resume core.

### E. Test-discipline corrections (from the quality review)
- Every new regression/matrix cell must be demonstrated **RED for its own reason** before the
  mechanism is relied upon (fork cells fail the conservation law via double-emit; the
  post-coalesce cell, with its dispatch branch disabled, fails on `OrchestrationInvariantError`).
- The collision cell must exercise an offset `> 0→1`: a node with **≥2** run-1 attempts, with
  the re-drive landing at `max+1`, so a hardcoded `+1`/"resume generation = 1" cannot pass.
- The expand value-fidelity cell must use **distinct per-child values** (a `datetime` and a
  `Decimal` of different magnitudes) and assert each resumed child carries ITS OWN value —
  `zip(strict=True)` guards length, not value↔token alignment.
- F2 reconciliation must assert the actual counter fields (`rows_processed` /
  `rows_succeeded` / `rows_failed`), not only the terminal-outcome multiset.

## Open questions for reviewers

1. Is mid-DAG continuation under the original parent (Approach 1) the right audit
   end-state, or is the redundant-double-parent shape (Approach 2) acceptable?
2. Can the incomplete child `TokenInfo` always be faithfully reconstructed for all
   three child kinds (fork→sink, fork→coalesce, expand)? Any persisted state gap?
3. Does continuing a reconstructed token via `process_token` correctly re-enter
   coalesce handling for fork→coalesce branches?
4. Resume counter semantics (related finding F2): the resume path counts
   `rows_processed` per leaf, diverging from the live per-source-row count. Should
   this fix also reconcile that, or keep it separate?
5. Concurrency/determinism: ordering of reconstructed incomplete tokens across a
   multi-branch partial fork — any divergence risk vs the original run?

## ADDENDUM 3 (2026-05-30, mid-implementation) — contract reconstruction is via the payload envelope, NOT the nodes table (AUTHORITATIVE)

**Defect found during Task 4 code review (production-population check).** The plan's
`_resolve_token_contract` recovered an expand/coalesce token's output contract from
`nodes.output_contract_json` keyed by `nodes.sequence_in_pipeline == step_in_pipeline`.
Both are NULL in production:

- The live `register_node` call (`engine/orchestrator/landscape_registration.py`) passes
  **no `sequence=`** → `nodes.sequence_in_pipeline` is NULL for every prod node (only the
  synthesised/tutorial write path and test code populate it).
- `output_contract` is passed **only for the source node** (`if node_id == source_id`) →
  `nodes.output_contract_json` is NULL for every transform/aggregation/coalesce node.

So the resolver would raise `AuditIntegrityError` on *every* expand/coalesce resume. It was
latent only because no caller is wired until Task 7. The fork→sink core (original RED) is
unaffected (`current_node_id=None`; fork-child reconstruct returns `source_row` untouched).
There is **no prod-populated column** carrying a non-source node's output contract, so the
nodes-table approach is unsalvageable.

**Resolution (Option A — self-contained payload envelope):** `row.contract` is genuinely
consumed on re-drive (`executors/transform.py` sets `ctx.contract = token.row_data.contract`;
`executors/sink.py` merges `tokens[...].row_data.contract`), so a reconstructed token needs a
faithful contract. `SchemaContract.to_checkpoint_format()` / `.from_checkpoint()` already
exist with Tier-1 hash-integrity validation. Therefore:

- **`tokens.token_data_ref` stores an envelope**, not a bare data dict:
  `checkpoint_dumps({"data": <row data dict>, "contract": <SchemaContract.to_checkpoint_format()>})`.
  (Note: `PipelineRow.to_checkpoint_format()` stores only a contract *version reference*, not the
  full contract — it is NOT usable here; build the envelope from the full
  `SchemaContract.to_checkpoint_format()`.)
- The contract is available at both writer call sites: expand has a single locked
  `output_contract` (from `TransformResult.contract`, shared by all children); coalesce has
  `merged_data.contract` (a `PipelineRow`). `expand_token`/`coalesce_tokens` receive it and
  serialise the envelope.
- **`reconstruct_token_row`** restores both: `data = env["data"]`,
  `contract = SchemaContract.from_checkpoint(env["contract"])`, `PipelineRow(data, contract)`.
  Fork children (`token_data_ref is None`) still return `source_row` unchanged.
- **`_resolve_token_contract` is DELETED**, along with the `nodes_table` import if otherwise
  unused. `PayloadNotFoundError` from `payload_store.retrieve` is wrapped to a contextful
  `ValueError` (token_id, run_id, ref) mirroring `get_unprocessed_row_data`.

`step_in_pipeline` itself is sound (prod-populated via `graph.resolve_step`, aligned with
`_node_step_map`), so Task 6's `_resolve_step_node` is unaffected.

**Systemic lesson (applies to Tasks 6/7/9/10):** for every `tokens.*` / `nodes.*` column the
dispatch reads, verify it is *written in production*, not merely that the column/method exists.
The "symbol exists" spec check passed the broken resolver; the "is it populated in prod" check
caught it.

## ADDENDUM 4 (2026-05-30, mid-implementation) — resume offset/provenance live on TokenInfo, not WorkItem (AUTHORITATIVE)

**Defect found during Task 5 implementation (per-token vs batched check).** Task 5 first put
`resume_attempt_offset` / `resume_checkpoint_id` on `WorkItem` and threaded them to the
transform and coalesce `begin_node_state` writers. That works for those two (one WorkItem
processed at a time), but the **sink path was left unthreaded** — and the sink is the primary
F1 path's collision site: `SinkExecutor.write` buffers `TokenInfo`s from *multiple* WorkItems
and calls `begin_node_state` **per-token in a loop**. `WorkItem` context is destroyed at that
buffer boundary, and each resumed token's offset (`spec.max_attempt + 1`) differs — so a scalar
on `write()` cannot represent it. Unthreaded, a fork→sink re-drive writes its sink `node_state`
at `attempt=0`, colliding with the run-1 record at `(token_id, sink_node_id, 0)` →
`UniqueConstraint` violation or the silent collision the whole offset mechanism exists to
prevent (Task 8 asserts run-1 at attempt 0, re-drive at attempt 1).

**Resolution (carry resume state on the token):** put the two fields on **`TokenInfo`**
(`contracts/identity.py`, L0; frozen/slots; not hashed or used as a set/dict key, so adding
scalar defaulted fields is safe), **remove them from `WorkItem`**, and read them from the token
at every per-token `begin_node_state` site (sink loop, coalesce, transform/state_guard). The
sink's existing `TokenInfo` buffer then carries the offset for free — no parallel side-map (a
side-map would invite `.get(token_id, default)`, the banned defensive pattern). One mechanism,
not two.

**Propagation rule (must be encoded, verified by Tasks 8/9):**
- **token_id-preserving steps (row transform):** the continuation token re-traverses under the
  same `token_id` and re-writes the next node's state, so it **must keep** the offset/provenance
  (else it collides one node downstream). Ensure the continuation-token construction propagates
  the fields (free if built via `dataclasses.replace(token, …)`; copy explicitly if built fresh).
- **token_id-minting steps (fork / expand / coalesce):** children get **new** token_ids with no
  run-1 record, so `attempt=0` is correct and they must **not** inherit the parent's offset; new
  children default to `0 / None`.

The resume values are first set in Task 6's `resume_incomplete_token`, on the `TokenInfo` it
builds for the incomplete token (`resume_attempt_offset=spec.max_attempt+1`,
`resume_checkpoint_id=<resumed-from checkpoint id>`); from there they flow via the token.

**Forward lens (Tasks 6/7/9/10):** apply two checks at *design* time, not in review — "per-token
or batched?" (where does the state need to live to survive buffering) and "written/preserved in
prod?" (is the column/field actually populated and carried by the live path). This is the third
runtime-vs-plan gap (B1 crash → contract-NULL → sink batching); the reviews caught all three, but
mapping the resume drive granularity up front is cheaper than rediscovering it.

## ADDENDUM 5 (2026-05-30, operator decision) — resume_checkpoint_id is a marker-only id, NOT a FK (AUTHORITATIVE)

**Fourth runtime-vs-plan gap, found during Task 7.** The Task-2 schema made
`node_states.resume_checkpoint_id` a `ForeignKey("checkpoints.checkpoint_id")`. But
`CheckpointManager.delete_checkpoints` (run on every successful run/resume completion to clean up
progress checkpoints) would then crash `FOREIGN KEY constraint failed` whenever it deleted a
checkpoint that a resumed node_state references. A FK leaves only audit-hostile escapes
(ON DELETE CASCADE deletes audit node_states; SET NULL erases the provenance marker), so a FK
forces permanent retention of any resume-anchor checkpoint.

**Operator decision: drop the FK; `resume_checkpoint_id` is a marker-only `String(64)`.** No audit
read path resolves the checkpoint row — `explain()` only tests the marker's NULL-ness
(`resume_checkpoint_id IS NOT NULL`) to separate resume re-drives from run-1 retries. The marker id
endures on `node_states` as a durable provenance fact **like a content hash survives payload
deletion** (the project's existing Tier-1 doctrine): the checkpoint is deletable progress state;
the id is the enduring fact. Consequences:
- `node_states.resume_checkpoint_id` = `Column(String(64), nullable=True)` — no `ForeignKey`.
- `delete_checkpoints` is the simple unconditional per-run delete (no referenced-id preservation).
- Trade-off accepted: no referential integrity (the marker may name a checkpoint whose row was later
  purged) — consistent with hash-survives-payload-deletion, not a defect.
- Epoch stays 11 (epoch 11 is unreleased; dropping the FK is within the same bump).

This was a genuine policy fork (referential integrity vs the deletion doctrine), not a
single-correct-answer gap — so it went to the operator, who chose marker-only.
