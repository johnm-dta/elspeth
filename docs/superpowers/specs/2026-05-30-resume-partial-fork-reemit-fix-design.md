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

- **Audit shape:** clean — one fork parent, all children under it, the incomplete
  one now terminal. Matches what an auditor expects ("forked once, each branch
  resolved once").
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

### Expand-resume schema change (the enabling piece)
Expand children carry independently-transformed per-child `row_data` created in memory
(`tokens.py:383-393`) and never persisted — only the source row payload is retrievable
by `row_id` (`data_flow_repository.create_row:429-448` stores it via `payload_store.store`
→ `rows_table.source_data_ref`). `expand_token` (`data_flow_repository.py:749`) persists
count + ids only. So an expand child is currently unreconstructable. Fix:
- **New nullable column `token_data_ref String(64)` on `tokens_table`** (`schema.py:210`).
  Holds the payload-store ref for a token whose `row_data` differs from its source row.
  NULL for fork children (they share the parent/source payload, retrievable by `row_id`).
- **`expand_token` persists each child's payload** at expand time via the existing
  `payload_store.store(payload_bytes)` pattern (mirror `create_row:429-448`), writing the
  ref into `tokens.token_data_ref`. This is a Tier-1 audit write — crash on store failure,
  no best-effort.
- **Recovery retrieval:** reconstructing an incomplete child reads `token_data_ref` when
  set (expand child) else `rows.source_data_ref` (fork child). Missing/garbage ref where
  one is required → raise (Tier-1, no coercion).
- **Epoch bump:** `SQLITE_SCHEMA_EPOCH` 10 → 11 (existing DBs incompatible → operator
  deletes per the delete-the-old-DB policy; no Alembic for the audit store).
- **Co-fix review finding F3:** add the new column AND the two omitted
  `runs.openrouter_catalog_*` columns to `_REQUIRED_COLUMNS` (`database.py`) so the
  Postgres staleness backstop (which bypasses the SQLite-only epoch gate) catches a stale
  DB. (F3 is an independent P1 from the same review; co-landing is cheap and correct.)
- With per-child payload persisted, expand children reconstruct exactly like fork children
  (their resume-start node derives from the expand step, continuing under the original
  EXPAND_PARENT) — the mechanism below is shared, keyed on child kind only for which
  payload ref to read.

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
3. **Resolve the resume-start node by mirroring `_handle_gate_fork` (processor.py:2674-2697),
   NOT by `branch_name → destination`:**
   - `branch ∈ _branch_to_sink` → terminal sink path (`current_node_id=None`).
   - `branch ∈ _branch_to_coalesce` → start at `_branch_first_node[branch]`, pass
     `coalesce_name`; coalesce re-entry then fires normally via `_maybe_coalesce_token`.
   - branch in **neither** → audit/DAG inconsistency → **raise `OrchestrationInvariantError`**
     (reuse existing raising helpers; never default-route).
4. Continue under the **original** fork parent (reuse the persisted child token id; do
   NOT mint a new parent). **Resume must start strictly downstream of the token's last
   recorded `node_states` position** — re-driving an already-recorded `(token_id, node_id,
   attempt)` violates the `node_states` unique constraint and crashes. For the fork→sink
   and fork→first-coalesce-node shapes this holds by construction; assert it.
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

## ADDENDUM (2026-05-30, plan-grounding) — node_states attempt collision on re-drive (AUTHORITATIVE)

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
