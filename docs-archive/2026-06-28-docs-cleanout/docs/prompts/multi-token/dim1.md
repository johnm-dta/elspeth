/goal Please do a detailed static analysis of the changed files in the `.worktrees/multi-source-token-scheduler` worktree.

Begin by identifying the changed files in the `.worktrees/multi-source-token-scheduler` worktree relative to its base branch. Review only changed files and directly related context needed to understand those changes. If the base branch cannot be determined, treat that as a hard precondition failure and report it.

Bring in and apply these skill packs:
- Python engineering
- systems thinking
- determinism and replay theory
- quality engineering
- embedded database, where persistence or replay state is involved

Focus specifically on the core engine upgrade from single-token processing to concurrent-token processing, and from single-source assumptions to multi-source scheduling.

Examine each changed file and section for:
- incorrect single-token assumptions that survived the migration
- incorrect single-source assumptions that survived the migration
- race conditions, ordering bugs, lifecycle bugs, unsafe shared state, or hidden coupling
- non-deterministic behaviour that would break replay, auditability, or reproducible investigations
- scheduler fairness, starvation, cancellation, timeout, backoff, and retry issues
- incorrect fan-out, fan-in, aggregation, source isolation, or partial-success handling
- conflation of source identity, token identity, run identity, job identity, or persisted state identity
- replay, idempotency, resumability, checkpointing, and recovery gaps
- embedded database usage issues, including transaction boundaries, locking, schema drift, migrations, indexes, and state consistency
- tests that no longer prove the intended concurrent, multi-source, deterministic, or replayable behaviour


CRITICAL RULE: Do not edit any files under any circumstances. Only stop early for hard precondition failures, such as the worktree being unavailable, changed files being impossible to identify, Filigree being unavailable, or the repository being unreadable. Do not stop merely because tests fail, docs are stale, or the branch is messy; create tickets for those issues instead. Because this audit will be run concurrently with other specialist audit passes, prefer high recall over strict deduplication. Create tickets for concrete findings even if they may overlap with another lens. Do not suppress a finding merely because another audit agent might also report it. The consolidation pass will de-duplicate and normalise severity later.

Create Filigree tickets for every concrete bug, architectural issue, code smell, fragile section, missing test, determinism risk, replay risk, persistence risk, or easy enhancement you find.

Use ticket titles in this format:
[area] concise issue summary

Examples:
[ux] UX not updated to handle multiple sources
[engine] Source identity can be lost during token retry
[replay] Concurrent source ordering is not deterministic
[docs] Operator guide still describes single-source secret references
[tests] Multi-source partial failure path lacks assertions

Each ticket should include:
- file and relevant section
- issue summary
- why it matters
- likely impact
- suggested fix
- relevant skill-pack lens used

For each ticket, include:
- severity: merge blocker, should fix before merge, should fix soon after merge, nice-to-have, documentation-only, or needs investigation
- confidence: high, medium, or low
- evidence type: direct code evidence, inferred design risk, missing test coverage, stale documentation, or operational concern

Include a short “dedupe key” field describing the underlying issue in stable terms, such as `single-source-assumption-in-token-state`, `non-deterministic-source-ordering`, `operator-docs-stale-secret-reference-flow`, or `sqlite-transaction-boundary-replay-risk`.

Do not create vague tickets that only say to “review”, “clean up”, or “consider improving” an area. If the issue is broad, identify the specific file, section, stale assumption, missing invariant, or concrete behaviour that makes it actionable.

At the end, provide a report summarising the tickets created and your overall assessment of the engine upgrade quality.
