# PRD-0001 — Web run audit-trail durability & truthfulness      Status: ready-for-planning
Decision: PDR-0002 (owed — records this session's re-scope DECIDE; append at checkpoint)
Bet (roadmap.md): Now (lead of "Web hardening to GA")   Target metric (metrics.md): run assurance completeness (north-star)

## Problem
**Who** — operators and auditors of web-initiated ELSPETH runs, whose entire reason
to use ELSPETH over an ordinary LLM workflow builder is that *every run is reviewable
and explainable after the fact*. **Their pain** — when a web run fails with no live
WebSocket subscriber attached, its terminal event is broadcast-only and never
persisted: the `run_events` table has a schema but **zero INSERT sites in the repo**,
so a later viewer of a failed run sees no event timeline at all (`f72b21e297`,
observed on staging run 8294aab2). Worse, even where audit rows *are* written, two
defects make them untrustworthy: startup schema validation compares CHECK constraints
by **name only**, so a session DB with a silently-weakened `ck_blobs_ready_hash` passes
validation and then accepts a `status='ready'` blob whose bytes are unverifiable
(`28bb7fcacb`); and post-compose, LLM-driven state advances are stamped with
`provenance='session_seed'`, so audit queries cannot tell an initial seed from a model
write (`24a7fb8e54`). **Desired outcome** — a web run's terminal outcome, and the
integrity and provenance of what it wrote, are durably and truthfully recorded such
that an auditor reconstructing the run after the fact gets the true story. **Why now** —
this is the lead bet of Web-to-GA; shipping the Composer path to real users while a
failed run can leave no queryable trace would ship the product with its core assurance
claim false on exactly the surface GA exposes.

## Success metric (the signal the bet paid off)
North-star — **run assurance completeness**. For this bet's class, TARGET = **0** web
runs that reach release with a missing, duplicated, or untrustworthy terminal audit
record, verified by regression tests at release. (The *aggregate* north-star percentage
is not yet instrumented on the `metrics.md` scoreboard — see Open questions; the per-
defect criteria below carry the falsification load independently and do not depend on
that instrumentation.)

## Acceptance criteria (falsifiable)
1. **PERSISTENCE** — Every web-initiated run reaching a terminal outcome (completed,
   failed, cancelled) persists ≥1 queryable terminal event in a durable store,
   independent of WebSocket subscriber presence. Verified by a regression test that runs
   a failing pipeline with **zero** subscribers and asserts the durable store holds the
   failure event for that run. Measured at release.
   *Reject branch:* any terminal path that can drop its event with no subscriber → unmet;
   bet rejected for that path.
2. **SINGLE-TERMINAL-STATE (regression guard)** — No run emits more than one terminal
   event; the late output-blob-finalization error path cannot produce a second,
   contradictory terminal. Regression test asserts exactly one terminal event per run
   across the happy path and a forced late-finalization error.
   *Reject branch:* >1 terminal event for any run → unmet. (Guards the already-landed
   `7fe9db9d97` fix against regression.)
3. **INTEGRITY** — `initialize_session_schema()` rejects a session DB whose CHECK
   constraints match by name but are weakened by expression; a `status='ready'` blob
   whose `content_hash` violates the 64-char lowercase-hex invariant cannot pass startup
   validation. Regression test clones the current schema with `ck_blobs_ready_hash`
   weakened and asserts validation **fails closed**.
   *Reject branch:* a weakened same-named CHECK passes validation → unmet.
4. **PROVENANCE** — `composition_states` audit rows distinguish session-seed/reseed from
   post-compose, LLM-driven state advances; no write path stamps a post-compose advance
   as `session_seed`. Regression test asserts `send_message` and the recompose path
   persist a non-seed provenance value.
   *Reject branch:* any post-compose advance recorded as `session_seed` → unmet.
5. **GUARDRAIL** — Over the change, the test battery stays ≥ 5507 pass / 0 hard
   failures (metrics.md guardrail) and the trust-tier red-gate stays green only on a
   signed state (no blind sign/bless).
   *Reject branch:* battery regresses to a hard failure, OR the gate is bypassed →
   bet rejected even if 1–4 pass.
6. **SCOPE** — Criteria 1–4 hold on 100% of web-initiated run paths, not a cohort- or
   flag-gated subset, at release.
   *Reject branch:* gated to a subset → this criterion is unmet.

## Non-goals (this bet)
- Oversized failed-login username audit bound (`1073b30450`, web-auth residue) — Next.
- Secret-lifecycle Landscape events (`149856079f`) — separate design/review cycle, Next.
- Blob session-quota write-skew under MVCC (`82281934aa`) — only bites a non-SQLite
  session DB, which is not shipped today — Next.
- The web-execution cluster's test-coverage and CLI-coupling debt
  (`0818bca84f`, `a7d0661310`, et al.) — quality follow-up, not a GA assurance gate.
- Any PostgreSQL session-DB migration work, and any UI/dashboard surfacing of events.

## Constraints & guardrails
- **Audit primacy (CLAUDE.md):** auditable actions belong in the durable Landscape/event
  store, not only in `slog`. The fix must persist, not merely log.
- **Session DB is SQLite-only today** (Phase 9 policy): the design must not *assume*
  PostgreSQL, but must not *regress* portability either.
- **Migration discipline:** persisting `run_events` writes may require a new session
  migration (the chain is at 006; next is 007 per the secrets cluster). DB migration =
  delete-old-DB policy; no schema-version probes.
- **Do not weaken the landed single-terminal-state invariant** (`7fe9db9d97`).

## Open questions / assumptions
- **North-star aggregate metric is uninstrumented** — its BASELINE/TARGET on
  `metrics.md` are placeholders. This bet is accepted on the per-defect regression
  criteria above; instrumenting the aggregate is a separate, owed item.
- **`f72b21e297` offers three resolution options** (broadcaster also writes `run_events`
  / delete the dead table / move to Landscape-native event sourcing). This PRD fixes the
  *outcome* (a durable, queryable terminal event) and routes the *choice* to
  `/axiom-solution-architect` — audit-native sourcing is the most consistent with audit
  primacy, but that is a design call, not a product one.
- **PDR-0002 is owed** — this PRD's bet was decided by this session's reconciliation
  (epic framing was stale); the decision must be recorded as a PDR at checkpoint.
- **Tracker reconciliation owed** — web-auth (`250f698aaf`) is one bug from done and
  web-execution's (`248536c9e6`) audit core is closed; both epics likely over-claim
  open scope. Reconcile at checkpoint.

## Handoff
- **Top item → `/axiom-planning`:** `f72b21e297` — persist web-run terminal events to a
  durable, queryable store regardless of subscriber presence. The spine; the other two
  in-scope bugs make the persisted record trustworthy.
- **Solution shape → `/axiom-solution-architect`:** choose among the three `f72b21e297`
  resolution options (broadcaster-writes-table vs. delete vs. Landscape-native) and the
  schema-validation-by-expression approach for `28bb7fcacb`.
- **Sequencing / dated forecast → `/axiom-program-management`:** this PRD emits no date.
- **In-scope tracker IDs:** `f72b21e297`, `28bb7fcacb`, `24a7fb8e54`.
  **Deferred (Next):** `1073b30450`, `149856079f`, `82281934aa`, `e213f73e9a`.
