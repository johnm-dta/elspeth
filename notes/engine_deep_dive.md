  You are conducting a SECOND-PASS adversarial review of the multi-source
  token-scheduler feature on the candidate branch
  `feat/multi-source-token-scheduler`. A first-pass review already produced
  14 filigree tickets under epic elspeth-651f15e3cb (P0 Tier-1 silent
  divergence, P0 multi-source routing, P1 fault-path test coverage, P2 drag-
  along). Your job is to KICK UP MORE DIRT in the same defect classes — not
  to re-enumerate the known findings, but to find the adjacent ones the
  first pass missed.

  ================================================================
  WORKING DIRECTORY DISCIPLINE — READ BEFORE ANYTHING ELSE
  ================================================================

  The code is in a git worktree, NOT the main checkout. EVERY absolute path
  below contains the segment `.worktrees/multi-source-token-scheduler/`.
  Do NOT rewrite paths to drop that segment. Do NOT `cd`. Read at the
  literal paths given. When grepping, anchor under
  `/home/john/elspeth/.worktrees/multi-source-token-scheduler/`, never under
  `/home/john/elspeth/`.

  ================================================================
  WHAT THE FIRST PASS ALREADY FOUND (don't re-file these)
  ================================================================

  Tier-1 silent divergence (scheduler write boundary):
  1. claim_ready never checks UPDATE rowcount → silent lease theft
  2. _transition never checks UPDATE rowcount (used by mark_waiting /
     mark_blocked / mark_terminal / mark_failed)
  3. mark_blocked_barrier_terminal returns rowcount via `or 0`; callers in
     processor.py + outcomes.py discard the return
  4. _drain_scheduler_claims marks BLOCKED with both queue_key AND
     barrier_key == None → forever-stranded row
  5. _legacy_row_index_default fabricates Tier-1 identity by silently
     copying row_index into source_row_index + ingest_sequence
  6. RowProcessor._source_on_success is single-valued → wrong sink on non-
     primary sources in source→sink (no-transform) configs

  Fault-path coverage gaps already filed: mark_failed untested,
  deserialize_row_payload AuditIntegrityError branches uncovered, the two
  new examples not executed end-to-end, _legacy_row_index_default untested,
  regex_worker untested.

  ================================================================
  THE SURFACE — go deeper on these files
  ================================================================

  Primary feature surface:
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/scheduler_repository.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/schema.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/data_flow_repository.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/landscape/run_lifecycle_repository.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/processor.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/core.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/outcomes.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/engine/orchestrator/types.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/contracts/scheduler.py
  - /home/john/elspeth/.worktrees/multi-source-token-scheduler/src/elspeth/core/regex_worker.py

  Diff base: origin/RC5.2. Working tree included — read files as they
  exist on disk and verify with `git diff origin/RC5.2..HEAD -- <path>` +
  `git diff -- <path>` to see committed and uncommitted deltas.

  ================================================================
  DEFECT CLASSES TO ATTACK (this is where the dirt is)
  ================================================================

  The first-pass pattern was: durable-write boundary discipline drops off
  even though the contract layer is clean. Look for *adjacent* instances:

  A. UNCHECKED MUTATIONS BEYOND THE SCHEDULER REPO
     The same rowcount-discard pattern flagged in scheduler_repository.py
     may exist in run_lifecycle_repository.py and data_flow_repository.py.
     Every `conn.execute(update(...).values(...))` and
     `conn.execute(insert(...).values(...))` in those files: does the
     caller verify the row actually changed? In particular:
     - Run lifecycle state transitions (RUNNING → COMPLETED / FAILED /
       CANCELLED) — does each transition assert the previous state was
       compatible, or does it silently overwrite?
     - run_sources table inserts (new on this branch) — what happens on
       duplicate (run_id, source_name)? The UNIQUE constraint will raise,
       but is that exception classified as AuditIntegrityError or does it
       leak as a generic IntegrityError?
     - rows_table inserts — does data_flow_repository.create_row verify
       post-insert that the row actually landed, or trust SQLAlchemy?

  B. LEASE LIFECYCLE EDGES NOT YET ENUMERATED
     - recover_expired_leases: rowcount discarded. The first pass flagged
       the *processor* discarding the rowcount; verify the *repository*
       side also doesn't have an issue: what if a row was already TERMINAL
       when its lease expired? Does it get yanked back to READY and
       re-execute?
     - release_waiting: same shape — rowcount discarded; what if a row's
       status is no longer WAITING when its available_at passes?
     - mark_terminal idempotency: documented as deliberate. Verify the
       idempotent path doesn't also accept LEASED→TERMINAL by a worker
       that doesn't own the lease (no lease_owner check before transition).
     - on_success_sink threading (the new uncommitted field): does it
       survive a LEASED→WAITING→READY round-trip without being silently
       dropped or rewritten?

  C. EXCEPTION CHAIN INTEGRITY
     The first pass flagged _legacy_row_index_default raising bare
     ValueError. Grep for every `raise` in the feature surface that does
     NOT use `from exc` or `from None` deliberately. Each one that re-
     raises a wrapped exception without a chain is an audit-trail leak.
     Also: every `except` block — does it catch too broadly? Are there
     any `except Exception:` that should be `except (AuditIntegrityError,
     SpecificError):`?

  D. SCHEDULER STATE MACHINE GAPS
     The TokenWorkStatus enum has 6 states: READY, LEASED, WAITING,
     BLOCKED, TERMINAL, FAILED. Enumerate every possible transition pair
     (6x6 = 36 ordered pairs). Which ones SHOULD be impossible? Of those,
     which ones does the code actually prevent vs. silently allow? E.g.
     TERMINAL → READY should be impossible (terminal is, well, terminal),
     but is there any code path — recover_expired_leases on a stale
     TERMINAL row? release_waiting on a TERMINAL row? — that could re-
     The repository docstring says "single-process friendly" but the
     `lease_owner` and `lease_expires_at` columns explicitly model multi-
     worker. Find every place the code is implicitly single-process but
     the schema is multi-worker — that's where the next-session-builds-
     multi-worker bug lands. Document them as "future-implicit single-
     process assumption" findings.
  The first-pass pattern was: durable-write boundary discipline drops off
  even though the contract layer is clean. Look for *adjacent* instances:

  A. UNCHECKED MUTATIONS BEYOND THE SCHEDULER REPO
     The same rowcount-discard pattern flagged in scheduler_repository.py
     may exist in run_lifecycle_repository.py and data_flow_repository.py.
     Every `conn.execute(update(...).values(...))` and
     `conn.execute(insert(...).values(...))` in those files: does the
     caller verify the row actually changed? In particular:
     - Run lifecycle state transitions (RUNNING → COMPLETED / FAILED /
       CANCELLED) — does each transition assert the previous state was
       compatible, or does it silently overwrite?
     - run_sources table inserts (new on this branch) — what happens on
       duplicate (run_id, source_name)? The UNIQUE constraint will raise,
       but is that exception classified as AuditIntegrityError or does it
       leak as a generic IntegrityError?
     - rows_table inserts — does data_flow_repository.create_row verify
       post-insert that the row actually landed, or trust SQLAlchemy?

  B. LEASE LIFECYCLE EDGES NOT YET ENUMERATED
     - recover_expired_leases: rowcount discarded. The first pass flagged
       the *processor* discarding the rowcount; verify the *repository*
       side also doesn't have an issue: what if a row was already TERMINAL
       when its lease expired? Does it get yanked back to READY and
       re-execute?
     - release_waiting: same shape — rowcount discarded; what if a row's
       status is no longer WAITING when its available_at passes?
     - mark_terminal idempotency: documented as deliberate. Verify the
       idempotent path doesn't also accept LEASED→TERMINAL by a worker
       that doesn't own the lease (no lease_owner check before transition).
     - on_success_sink threading (the new uncommitted field): does it
       survive a LEASED→WAITING→READY round-trip without being silently
       dropped or rewritten?

  C. EXCEPTION CHAIN INTEGRITY
     The first pass flagged _legacy_row_index_default raising bare
     ValueError. Grep for every `raise` in the feature surface that does
     NOT use `from exc` or `from None` deliberately. Each one that re-
     raises a wrapped exception without a chain is an audit-trail leak.
     Also: every `except` block — does it catch too broadly? Are there
     any `except Exception:` that should be `except (AuditIntegrityError,
     SpecificError):`?

  D. SCHEDULER STATE MACHINE GAPS
     The TokenWorkStatus enum has 6 states: READY, LEASED, WAITING,
     BLOCKED, TERMINAL, FAILED. Enumerate every possible transition pair
     (6x6 = 36 ordered pairs). Which ones SHOULD be impossible? Of those,
     which ones does the code actually prevent vs. silently allow? E.g.
     TERMINAL → READY should be impossible (terminal is, well, terminal),
     but is there any code path — recover_expired_leases on a stale
     TERMINAL row? release_waiting on a TERMINAL row? — that could re-
     open it?

  E. CONCURRENT-WORKER ASSUMPTIONS
     The repository docstring says "single-process friendly" but the
     `lease_owner` and `lease_expires_at` columns explicitly model multi-
     worker. Find every place the code is implicitly single-process but
     the schema is multi-worker — that's where the next-session-builds-
     multi-worker bug lands. Document them as "future-implicit single-
     process assumption" findings.

  F. SCHEMA EPOCH AND MIGRATION
     Two epoch bumps on one branch (9→10→11). Per project policy
     (memory: project_db_migration_policy / project_phase9_sqlite_only)
     migration = "delete the old DB"; no Alembic. Verify the schema
     factory builds the new tables from scratch correctly. Are there any
     indices, constraints, or defaults that depend on the order of CREATE
     statements? In particular: ForeignKeyConstraint between run_sources
     and runs — are both tables created in the right order by
     metadata.create_all()? Does the multi-column UNIQUE on rows_table
     (run_id, source_node_id, source_row_index) actually enforce what its
     name suggests on SQLite, or is there a NULL-tolerance trap?

  G. PROCESS-POOL ISOLATION
     regex_worker.py is correctly wired but the surface around it
     (rag/query.py:135-156) collapses "regex timed out" and "regex did
     not match" into the same QueryResult reason with a hidden cause
     discriminator. Audit query downstream gets the wrong answer. Look
     for other places where a multi-outcome failure is collapsed to a
     single result.code with a side-channel discriminator the audit
     trail will lose.

  H. EXAMPLES AS CONTRACT
     examples/multi_flow/ and examples/multi_source_queue/ are first-
     class user-facing artifacts (still untracked on the branch). Each
     README claims specific audit-DB spot-checks (SELECT statements with
     expected row counts). Read each README's SQL, then read the example's
     settings.yaml, then predict what the audit DB will actually contain
     and check whether it matches the README's claim. README-vs-reality
     drift is a documentation-as-contract violation.

  I. PROJECT CONVENTION DRIFT
     The branch was built fast by an agent. That class of work tends to
     drop convention-adherence at the edges. Spot-check for:
     - any new `.get(...)` defensive lookups on typed dataclass fields
     - any new `getattr(obj, "x", default)` calls
     - any new `hasattr(...)` (unconditionally banned)
     - any new `isinstance(...)` used as defensive guards (vs polymorphic
       dispatch, which is fine)
     - any frozen dataclass with new Mapping/Sequence/Set field that
       doesn't call freeze_fields() in __post_init__
     - any new layer-rule violation (L1 importing from L2, L2 from L3)

  ================================================================
  METHOD
  ================================================================

  1. Read every file in the feature surface in full at least once. Don't
     rely on prior context — this is your own pass.
  2. For each defect class A-I above, grep + read + verify with code
     reasoning. Cite file:line and quote the exact problematic code.
  3. For each finding, classify as:
     - DEFECT (filable as a new ticket, distinct from the known 14)
     - REINFORCEMENT (same root cause as a filed ticket; add to it)
     - PROBABLE-NON-ISSUE (looked at it; ruled out; document the
       reasoning so the next reviewer doesn't waste cycles)

  4. For each DEFECT, propose the minimum-disruption fix that follows the
     established offensive-programming pattern (raise AuditIntegrityError
     with diagnostic context, preserve exception chain, no defensive
     shims, no legacy compatibility layers).

  ================================================================
  OUTPUT
  ================================================================

  Structured markdown report. Group by defect class A-I. Lead each finding
  with one line: severity (CRITICAL / IMPORTANT / SUGGESTION),
  classification (DEFECT / REINFORCEMENT / PROBABLE-NON-ISSUE), and the
  file:line citation. Then evidence (code excerpt), reasoning, and
  proposed fix.

  End with a "what I looked at and ruled out" section so the operator can
  see the surface area covered, not just the findings list.

  DO NOT review: documentation deletions, fingerprint baseline rotation,
  CICD allowlist edits, PR_DESCRIPTION.md, composer/web changes outside
  the scheduler surface, or any test file that isn't directly testing the
  feature surface. The operator framed this as Python-feature-surface
  only.
