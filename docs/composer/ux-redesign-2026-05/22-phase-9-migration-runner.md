# Phase 9 — Migration runner + caretaker-logic activation

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task once
> Steps will use checkbox (`- [ ]`) syntax for tracking. **DO NOT BEGIN IMPLEMENTATION**
> until remaining operator decisions (#2 post-ship rollback strategy, #3 fixture-generation
> approach) are resolved and Phase 1A → Phase 4A → Phase 5b are present in git history.

**Status:** PLAN READY — J1 APPROVED 2026-05-16 (Option (c) per-table preserve-on-recreate). Implementation gated on operator decisions #2 (post-ship rollback strategy), #3 (fixture-generation approach), and Phase 1A → Phase 4A → Phase 5b landed in git history.

**Upstream dependencies:**
- Phase 8 (polish + telemetry) shipped. Phase 9 is post-launch and must not block the demo
  critical path.
- A real-user deploy is in prospect (or imminent). Until "WE HAVE NO USERS YET" stops being
  true, the current `project_db_migration_policy` (delete the old DB) remains correct and
  Phase 9 is not yet *needed* — only *planned*.
- Plans 12 (Phase 1A), 21a (Phase 4A), 18a (Phase 5b) all shipped under the
  delete-the-DB policy. Their table/column/enum definitions are stable and Phase 9 does
  **not** rewrite them. See §8 "Coherence with existing redesign phases" below.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md) §A J1
row (open question), §D6 (cumulative DB-delete loop), §H finding 1 (cross-phase pattern).

---

## 1. Goal

Resolve the **cumulative DB-delete loop** introduced by three consecutive schema-adding
phases.

The composer UX redesign added schema in three separate phases under the existing
`project_db_migration_policy` ("delete the old DB at deploy"):

| Phase | Schema addition | User state lost on next delete |
|---|---|---|
| 1A | `user_preferences_table` (per-user composer preferences) | `default_composer_mode`, `banner_dismissed_at` |
| 4A | `user_preferences.tutorial_completed_at` column | Tutorial completion flag |
| 5b | `interpretation_events_table` + `composition_states.provenance` enum extension (`interpretation_resolve`) | Interpretation-event history; opt-out choices |

Each phase's DB-delete erases the **previous** phase's user state. Per roadmap §D6:

> A user who completes the tutorial after Phase 4 ships gets the tutorial again after
> Phase 5b ships.

The structural fix is a **migration runner** that preserves user-state tables across
schema changes. Phase 9 owns this fix. It is named in the §H finding-1 cross-phase
pattern, and explicitly deferred to in plans 12, 21a, and 18a.

Phase 9 also activates **caretaker logic**: on startup, detect schema drift between the
running code's `MetaData` and the live database; apply the chosen migration strategy;
record the migration as a first-class audit event. The caretaker is **new code** — no
caretaker logic exists in the codebase today (`project_db_migration_policy` deliberately
omitted it).

**Policy supersession.** Phase 9 is the planned departure from
`project_db_migration_policy`. The memory entry must be updated when Phase 9 ships to
reference Phase 9 as the new strategy. Phase 9 does **not** retroactively change the
behaviour of pre-Phase-9 deploys; those continue to require the operator-deletes-the-DB
ceremony.

---

## 2. J1 ADJUDICATION SECTION (explicit operator decision point)

**J1 question (from roadmap §A):** *Migration runner shape: SQL DDL diff, schema
migrations à la Alembic, or per-table preserve-on-recreate?*

Three options are evaluated below. **J1 APPROVED 2026-05-16 — Option (c) per-table
preserve-on-recreate.** Defensible cases exist for all three options; the trade-off
tables below document the comparison that informed the verdict.

The current policy (`project_db_migration_policy`) — *delete-the-DB* — is **incompatible
with Phase 9**. Phase 9 IS the policy change. The memory entry will be marked
"superseded by Phase 9" on ship.

### Option (a) — SQL DDL diff inferred from SQLAlchemy MetaData on startup

**Mechanism.** At app startup, compute the set difference between (1) the tables/columns
declared in `metadata` (compiled from `sessions/models.py`) and (2) what the live DB
reports via reflection. Emit SQL DDL to bring the live DB up to declared shape:
`CREATE TABLE` for missing tables, `ALTER TABLE ... ADD COLUMN` for missing columns,
enum-value `CHECK` constraint rewrites for closed-enum extensions. Run inside a
transaction. Record each DDL statement issued.

**Trade-offs.**

| Dimension | Verdict |
|---|---|
| Auditability | Good — every DDL statement can be recorded as a Landscape event before execution. Inferred from canonical `metadata`, so the source of truth is the code at the deployed revision. |
| Determinism | Weak — diff order depends on SQLAlchemy's reflection ordering. Mitigation: pin diff order by canonical sort. |
| Simplicity | Medium — ~300 LOC for diff engine, plus per-dialect DDL emitters. No version table, no migration files. |
| Reversibility | Hard — no down-migration. To revert, operator deletes the DB. |
| Coverage | Limited — handles add-table and add-column cleanly; **cannot** handle column renames, type changes, or data backfills. The composer redesign has none of these (every Phase added new shape, never modified existing), so coverage is sufficient for the current schema delta. |
| Risk to existing tables | Low — pure additive operations. The runner refuses to drop or alter existing columns. |
| Closed-enum extension (Phase 5b `interpretation_resolve`) | Needs a custom handler: detect `CHECK` constraint divergence on `composition_states.provenance`, rewrite the constraint. Non-trivial in SQLite (no `ALTER CHECK CONSTRAINT`); requires shadow-table-copy ceremony. |

**Why this might be the right call.** It matches the codebase's existing posture: the
canonical `metadata` is the source of truth, additions are the only schema change pattern
the project has ever performed, and there is no need for a separate migration-files
directory.

**Why this might not be.** The closed-enum extension case (Phase 5b's
`composition_states.provenance` adding `interpretation_resolve`) is the awkward edge:
SQLite requires a table-copy to rewrite a `CHECK` constraint. That ceremony is the same
ceremony Option (c) performs *always*, so Option (c) becomes simpler-by-uniformity at
that point.

### Option (b) — Alembic-style migration scripts checked into the repo

**Mechanism.** Reintroduce Alembic (or an Alembic-shaped homegrown equivalent). Each
schema change ships with a paired `up` migration file. Startup runs pending migrations
in order. A `alembic_version` table tracks applied revisions.

**Trade-offs.**

| Dimension | Verdict |
|---|---|
| Auditability | Excellent — each migration is a file under version control with an explicit revision id; the applied set is queryable. |
| Determinism | Excellent — migration order is hash-pinned in revision files. |
| Simplicity | **Poor — high ceremony.** Alembic adds: a config file, an `env.py`, a migrations directory, an autogeneration ceremony, revision-id collisions on parallel branches. The project deliberately avoided Alembic per `project_db_migration_policy`. Reintroducing it is a policy reversal of substantial weight. |
| Reversibility | Good — `down` migrations supported. |
| Coverage | Excellent — handles every schema change pattern, including renames, type changes, data backfills. |
| Risk to existing tables | Low — migrations are explicit; reviewer sees exactly what each one does. |
| Closed-enum extension | First-class — Alembic recipes for CHECK-constraint rewrites are well-trodden. |

**Why this might be the right call.** It is the industry-standard answer. If the project
expects to reach a point where renames, type changes, or backfills become common, Alembic
front-loads the discipline cost.

**Why this might not be.**

(a) **Auditability deficit.** Every caretaker action is a first-class Landscape event
with the table name, the action taken, and the SQL emitted. Alembic cannot natively
provide this: its version table records which revision was applied, but not the
individual DDL statements, their outcome records, or their pre/post metadata hashes.
Wrapping Alembic to emit Landscape events is additional plumbing that recreates what
Option (c) does natively.

(b) **Closed-enum extension uniformity.** The shadow-table-copy ceremony required by
SQLite for any `CHECK` constraint rewrite (Phase 5b's `composition_states.provenance`
adding `interpretation_resolve`) is handled as a first-class action in Option (c). In
Alembic, the same ceremony is achievable but is a per-migration hand-written migration
rather than a generalised mechanism. Option (c)'s single ceremony covers both
column-add and enum-rewrite uniformly; Alembic's revision-per-change model does not
collapse this to a uniform path.

Note that Alembic's per-dialect DDL and revision-graph serialization are well-trodden
and would address R2/R3 mitigations competently. The objection is not that Alembic is
defective — it is that Option (c) satisfies the project's auditability requirement
natively and covers the actual schema-change patterns without the revision-graph
ceremony.

### Option (c) — Per-table preserve-on-recreate

**Mechanism.** Classify each table as either **data table** (holds user state we MUST
preserve) or **definitional table** (holds derived/transient state we can drop). On
startup:

1. Walk the canonical `metadata`. Each `Table` carries an `info["migration_class"]`
   attribute set to `"data"` or `"definitional"`.
2. For each **definitional** table: drop and recreate from `metadata`. Idempotent if the
   table is already at declared shape (no-op).
3. For each **data** table: detect drift via reflection. If new columns are declared,
   `ALTER TABLE ADD COLUMN` them (additive only). If a column is dropped from
   `metadata`, refuse and crash with a loud error — that's a destructive change and
   Phase 9 does not perform destructive changes silently.
4. Closed-enum extensions: handled uniformly via the shadow-table-copy ceremony (create
   new table with new constraint; copy rows; drop old; rename). Since SQLite requires
   this for any `CHECK` constraint change anyway, treating it as the universal mechanism
   simplifies the runner.

**Classification for the current schema:**

| Table | Class | Rationale |
|---|---|---|
| `sessions` | data | User-owned session metadata; must survive. |
| `chat_messages` | data | Session history; user-visible conversation record. |
| `composition_states` | data | Per-session pipeline state — re-creatable in principle but expensive (would reset every user's WIP pipeline). Preserve. |
| `composition_proposals` | data | Per-session proposal history; user-authored pipeline intents. |
| `proposal_events` | data | Event trail for proposal lifecycle; audit-shaped. |
| `runs` | data | Pipeline run records linked to sessions; preserving provenance. |
| `blobs` | data | User-uploaded or system-generated binary/JSON blobs. |
| `blob_run_links` | data | Association table linking blobs to runs; loss severs provenance. |
| `run_events` | data | Per-run event stream; audit-shaped. |
| `user_secrets` | **data** | Holds encrypted user credentials. Misclassifying as `definitional` would destroy all stored credentials on the next caretaker run. This classification is non-negotiable. |
| `audit_access_log` | data | Access audit trail; preserve unconditionally per audit primacy. |
| `user_preferences` (added by Phase 1A; Phase 9 hard-blocks on Phase 1A shipping) | data | Per-user opt-out, banner-dismissal, tutorial-completion. |
| `interpretation_events` (added by Phase 5b; Phase 9 hard-blocks on Phase 5b shipping) | data | Audit-shaped interpretation-event history; preserve unconditionally. |
| `composer_completion_events` (added by Phase 6; protected by SQLite triggers) | data | Completion-gesture audit trail — preserve unconditionally per audit primacy. |
| `skill_markdown_history` (added by Phase 6) | data | Skill-markdown revision history; user-authored content trail. Preserve. |
| (any future caches, indexes, materialised views) | definitional | Re-derivable from data tables. |

**Trade-offs.**

| Dimension | Verdict |
|---|---|
| Auditability | Excellent — every operation (drop, recreate, add-column, shadow-copy) is recorded as a Landscape event with the table name, the class, and the SQL emitted. |
| Determinism | Good — classification is a code-level constant; iteration order is sorted. |
| Simplicity | Good — ~400 LOC. No revision table, no migration files. One uniform shadow-copy ceremony covers add-column AND enum-rewrite cases. |
| Reversibility | Hard — no down-migration. Symmetric with Option (a). Operator deletes the DB to revert. |
| Coverage | Sufficient — handles every schema change the project has performed (additive only). Refuses destructive changes loudly. |
| Risk to data tables | **Low — explicit refusal to drop.** Misclassification of a data table as definitional would be catastrophic, so the classification is a load-bearing review point. Mitigation: classification lives in `models.py` adjacent to the Table definition; classification changes require a deliberate diff. |
| Closed-enum extension | First-class via the shadow-copy ceremony. The Phase 5b case becomes mechanical. |

**Why this is the recommendation.**

1. **Auditability** — the project's prime directive. Every action the runner takes is a
   first-class audit event. The classification (data vs definitional) is itself
   defensible: "did Phase 9 preserve this table or recreate it?" has a one-word answer
   per table.
2. **Matches the project's actual schema change pattern.** Three schema changes; all
   additive. The runner covers the cases that have happened and the case that's next
   most likely (another additive table). Options (a) and (c) tie here.
3. **Closed-enum extensions become trivial.** Phase 5b's `interpretation_resolve` enum
   extension is the awkward case for Option (a); it's a no-op-by-mechanism for Option
   (c).
4. **Lower ceremony than Option (b).** No migrations directory, no revision graph, no
   `alembic upgrade head` in deploys. Matches the project's "no legacy code" stance.
5. **Refuses destructive changes loudly** rather than silently performing them. Aligns
   with the "offensive programming" principle: bugs surface as crashes, not silent data
   loss.

**J1 APPROVED 2026-05-16 — Option (c) per-table preserve-on-recreate. Roadmap §A J1 row updated.**
The remainder of this plan is written for Option (c) per the approved verdict.

---

## 3. Caretaker-logic activation

**Caretaker logic** is the runtime hook that runs the migration runner. Phase 9 introduces it
as **new code**. No caretaker hook exists today; `metadata.create_all()` is called
inertly at app startup and silently no-ops on existing tables.

**Activation site.** Caretaker is invoked from `create_app()` after session engine construction and before any router is mounted. This is the same call site as `initialize_session_schema` at `app.py:581`, which Task 5a displaces. No lifespan event is used; the call is synchronous inside `create_app()`.

**Activation contract:**

1. Caretaker reads `metadata` from `sessions/models.py`.
2. Caretaker connects to the configured sessions DB via the same engine used at
   runtime.
3. Caretaker classifies every table in `metadata` via `info["migration_class"]`. If a
   table has no classification, the caretaker defaults to `"data"` (preserve; never
   recreate) AND records the omission in every `MigrationPlanRecord` for that deploy
   (not once-tracked). Rationale: "if it's not recorded, it didn't happen" — every
   deploy's audit trail must independently document which tables were
   classification-defaulted. A "warned-once" optimisation creates a hidden state file
   that itself would need auditing; emit-every-deploy is the simplest audit-loudest
   behaviour.
4. Caretaker reflects the live DB shape.
4a. Caretaker inspects `engine.dialect.name`. If it is not `'sqlite'`, the caretaker refuses to start with `UnsupportedDialect(dialect_name, supported=['sqlite'])` and records a `MigrationUnsupportedDialectRefusal` Landscape event before raising. Phase 9 explicitly scopes the migration runner to SQLite only; Postgres support is deferred to a future phase. This refusal is mechanically enforced, not advisory.
5. Caretaker inspects reflected shape for orphan tables and orphan columns BEFORE
   computing the action plan. If reflection finds a table in the live DB that has no
   entry in `metadata.tables`, the caretaker refuses to start with
   `OrphanLiveTable(table_name, declared_tables)`. Likewise for orphan columns (a
   column in the live DB not present in `metadata`) on a table the caretaker manages:
   `OrphanLiveColumn(table_name, column_name)`. Per Tier 1 policy and offensive
   programming: a live DB shape that diverges from declared shape is an anomaly the
   caretaker must surface, not silently ignore.
6. Caretaker computes the action plan: per-table list of `("preserve_add_columns",
   "shadow_copy_enum_rewrite", "recreate", "noop")`.
7. Caretaker records the action plan as a `MigrationPlanRecord` Landscape event BEFORE
   executing.
8. Caretaker executes the plan action-by-action with per-action transactional discipline. Actions that SQLite transacts (`CREATE TABLE`, `ALTER TABLE ADD COLUMN`) are wrapped in their own transaction and either commit-or-rollback atomically. The shadow-copy ceremony is atomic at the *ceremony level*, not the SQLite-transaction level: SQLite's `PRAGMA foreign_keys` is connection-scoped and not transactional, so the ceremony cannot be wrapped in a single rollback-safe transaction. The ceremony's atomicity is enforced by the rename step (step 8 of the 12-step ceremony) being the only externally-visible commit point; pre-rename failures leave the original table untouched. See R3 for the SQLite documentation reference. The two failure record types (`MigrationFailedRecord` and `MigrationPartialFailureRecord`) fire under disjoint conditions defined in Task 7.
9. Caretaker records a `MigrationCompletedRecord` (or `MigrationFailedRecord` with
   exception details, chained via `from exc`) Landscape event AFTER execution. No retry:
   if the Landscape write fails, the caretaker crashes immediately with the offending
   exception. Landscape write failure aborts the caretaker; there is no recovery path.
10. On any failure: caretaker **does not start the app**. The deploy fails loudly. The
    operator sees the audit event, the SQL that failed, and the exception.

**Record-then-raise contract for refusal paths.** Every caretaker refusal MUST emit its corresponding `Migration*Refusal` Landscape event BEFORE raising the exception. The order is: (1) construct the refusal record; (2) call the recorder method; (3) raise the exception with the same context preserved via `from exc` if a wrapped exception triggered the refusal, or `from None` for fresh refusals. If the Landscape write itself fails during a refusal-path emission, the caretaker re-raises the Landscape write exception with the original refusal exception chained — per audit primacy, the inability to record the refusal is itself the headline failure. This is consistent with §6 R4's two-window analysis.

The refusal paths and their event types:

- Dialect check fails → `MigrationUnsupportedDialectRefusal` → `UnsupportedDialect`
- Orphan-plan scan finds an orphan → `MigrationOrphanPlanRefusal` → `OrphanMigrationPlanRefusal`
- Orphan live table → `MigrationOrphanLiveTableRefusal` → `OrphanLiveTable`
- Orphan live column → `MigrationOrphanLiveColumnRefusal` → `OrphanLiveColumn`
- Lock acquisition fails → `MigrationLockHeldRefusal` → `MigrationLockHeld`
- Column drop detected on data table → `MigrationDestructiveRefused` → `DestructiveMigrationRefused`
- Non-empty definitional table → `MigrationNonEmptyDefinitionalRefused` → `NonEmptyDefinitionalTable`
- Application-id mismatch → `MigrationWrongApplicationRefusal` → `WrongApplicationDatabase`
- Post-ceremony FK enforcement lost → `MigrationFKEnforcementLost` → same
- Trigger recreation incomplete after shadow-copy step 10 → `MigrationTriggerRecreationFailed` → same

**Bootstrap-ordering invariant (first deploy).** The caretaker requires a working Landscape engine to record `MigrationPlanRecord` BEFORE plan execution (step 7). On a truly fresh deploy where neither the sessions DB nor the Landscape DB has been initialised, the Landscape's own schema must be created BEFORE the caretaker runs. The ordering inside `create_app()` is:

1. Construct settings (incl. same-path validator from Task 4a).
2. Construct Landscape engine; call `LandscapeDB.bootstrap_schema(url, passphrase)` (idempotent — creates Landscape tables if absent). This is the existing pattern used elsewhere in the web subsystem.
3. Construct session engine.
4. Run caretaker (`run_caretaker(session_engine, metadata, recorder)`).
5. Mount routers.

Steps 2 and 4 are independent (Landscape DB ≠ sessions DB per Task 4a same-path validator); step 2 cannot fail in a way that leaves caretaker stranded mid-execution because step 4 has not yet started.

**Dry-run mode.** Caretaker accepts a `--dry-run` flag (env var
`ELSPETH_WEB__MIGRATION_DRY_RUN=1`, following the Dynaconf `ELSPETH_WEB__` nested-key
convention used throughout the web subsystem). In dry-run, steps 1-7 run, step 8 is
replaced with "log the SQL that *would* be issued," step 9 emits a
`MigrationDryRunRecord`. The app then exits cleanly. This is the deploy-time pre-flight:
operator runs dry-run, reviews the plan, then runs live.

**Idempotence.** Re-running the caretaker after a successful migration must be a no-op
(zero DDL emitted). This is verified by an integration test (§6).

**Landscape/sessions DB separation invariant.** The caretaker operates on the sessions
engine only. Landscape writes use the Landscape engine. These two MUST NOT share a DB
path — a misconfiguration that points both engines at the same SQLite file would
intermix schema with audit data. The integration test in Task 5 asserts this invariant
with a bad-config fixture (both engines sharing the same path): the caretaker must
refuse to start with a clear error identifying the conflict. A `@model_validator(mode='after')`
on `WebSettings` rejects configurations where `landscape_url` and `session_db_url`
resolve to the same SQLite path. This catches the misconfiguration
at config-construction time, before any engine is constructed.

---

## 4. Tasks (TDD-shaped: failing test → run-to-fail → implement → run-to-pass → commit)

> **Implements J1 verdict (c) per-table preserve-on-recreate (APPROVED 2026-05-16).**

### Task 0 — Ground-truth refresh (no-commit precondition gate)

**Goal.** This plan was authored over multiple cycles; line numbers, caller counts, and table inventories drift between cycles. Task 0 is a precondition gate: before any other task begins, the implementer runs a kickoff script that asserts the plan's load-bearing facts match HEAD. Failures BLOCK the rest of the plan until the plan text is refreshed.

**Mechanism.** Create `scripts/preconditions/check_phase9_ground_truth.py` (not committed; ephemeral) that asserts:

- `grep -c 'def initialize_session_schema' src/elspeth/web/sessions/schema.py` returns 1; capture the line number; assert it matches the value in the plan's Task 5a wording.
- `grep -c 'initialize_session_schema(session_engine)' src/elspeth/web/app.py` returns 1; capture the line number; assert it matches `app.py:581` (the value cited in §3 and Task 5a).
- `grep -c 'def _validate_partial_index_dialect_symmetry' src/elspeth/web/sessions/schema.py` returns 1; capture line number; assert it matches `schema.py:377` (the value cited in Task 5a).
- `grep -rln initialize_session_schema tests/ src/ | wc -l` returns a number; print it; assert the plan's Task 5a does NOT contain an enumerated caller list (the prior enumerated list was removed in this revision in favour of the authoritative grep directive).
- Count of `Table()` declarations in `src/elspeth/web/sessions/models.py` matches the number of classification entries in §2 plus the two future-cache placeholder rows.
- `_stamp_schema_sentinels` and `_assert_schema_sentinels` exist in `schema.py` at the expected line range.
- `_REQUIRED_SQLITE_TRIGGERS` exists in `schema.py:55-64` (the trigger constant referenced by Task 4's step-10 trigger-recreation specification).

If any assertion fails, the kickoff script exits non-zero with a clear message identifying which fact has drifted. The implementer refreshes the plan text BEFORE proceeding to Task 1.

**No commit.** Task 0 produces no code commit; it is a precondition check only. The kickoff script is run once at the start of the Phase 9 implementation session and discarded.

**Rationale.** Both the Cycle-1 and Cycle-2 plan reviews caught the same staleness defects (test-caller undercount; line numbers in `app.py`/`schema.py`). Moving load-bearing assertions from plan prose into an executable precondition prevents future cycles from inheriting the same staleness.

### Task 1 — Classify every existing table

**Goal.** Walk `sessions/models.py` and add `info={"migration_class": "data"}` (or
`"definitional"`) to every `Table` definition. Tables without classification default to
`"data"` at caretaker startup (conservative safe default); this task makes existing
tables explicitly classified so the caretaker records no implicit-default warnings in its
`MigrationPlanRecord`.

**TDD shape:**

- [ ] Write failing test: `tests/unit/web/sessions/test_migration_classification.py`
  asserts every `Table` in `metadata.tables.values()` has `tbl.info["migration_class"]
  in {"data", "definitional"}`. Also add a test verifying that a `Table` with no
  `migration_class` key in `info` is treated as `"data"` by the caretaker (the
  conservative default) and that the omission is recorded in **every**
  `MigrationPlanRecord` emitted for that deploy (emit-every-deploy, not once-tracked).
- [ ] Run test → fail (no classifications yet).
- [ ] Edit `sessions/models.py`: add `info={"migration_class": "data"}` to each Table
  per the classification table in §2 Option (c).
- [ ] Run test → pass.
- [ ] Commit: `migration: classify existing session tables for Phase 9 caretaker`.

### Task 2 — Schema-drift detector

**Goal.** Module `src/elspeth/web/sessions/migration/detector.py`. Pure function:
`detect_drift(metadata, engine) -> MigrationPlan`. Reads `metadata`, reflects the live
DB, returns a frozen plan dataclass listing per-table actions.

**TDD shape:**

- [ ] Write failing test: `tests/unit/web/sessions/migration/test_detector.py` —
  five scenarios: (a) live DB matches declared shape → plan is all-noop; (b) live DB
  missing the entire `user_preferences_table` which is a `data` table → plan has
  `create_table` (not `recreate`; absent data tables use fresh creation); (b2) live DB
  missing a `definitional` table → plan has `recreate`; (c) live DB has
  `user_preferences_table` but no `tutorial_completed_at` column → plan has
  `preserve_add_columns` with the column listed; (d) live DB has zero tables (first
  deploy; empty SQLite file) → plan has `create_table` for every
  table in `metadata.tables`. Verified for in-memory and on-disk SQLite. The
  `create_table` vs `recreate` distinction is load-bearing: the detector emits
  `create_table` for any table absent from the live DB regardless of class; it emits
  `recreate` only for definitional tables that ARE present in the live DB.
- [ ] Run → fail (module absent).
- [ ] Implement `detector.py`. Use SQLAlchemy reflection. Sort tables alphabetically
  for deterministic plan ordering.
- [ ] Run → pass.
- [ ] Commit: `migration: schema-drift detector (Phase 9 Task 2)`.

### Task 3 — Migration plan dataclass + Landscape event records

**Goal.** Frozen dataclasses `MigrationPlan`, `MigrationPlanRecord`, `MigrationCompletedRecord`, `MigrationFailedRecord`, `MigrationPartialFailureRecord`, `MigrationDryRunRecord`, `MigrationActionStartedRecord`, `MigrationActionCompletedRecord`, `MigrationOrphanCleared`, `MigrationFreshDatabaseStamped`, `MigrationEpochAdvanced`, plus the refusal-path records: `MigrationUnsupportedDialectRefusal`, `MigrationOrphanLiveTableRefusal`, `MigrationOrphanLiveColumnRefusal`, `MigrationLockHeldRefusal`, `MigrationOrphanPlanRefusal`, `MigrationDestructiveRefused`, `MigrationNonEmptyDefinitionalRefused`, `MigrationWrongApplicationRefusal`, `MigrationFKEnforcementLost`, `MigrationTriggerRecreationFailed` in
`src/elspeth/contracts/migration.py`. Per CLAUDE.md `deep_freeze` contract: containers
guarded via `freeze_fields`. Also define a `MigrationRecorder` Protocol in
`contracts/migration.py` declaring `record_migration_plan`, `record_migration_completed`, `record_migration_failed`, `record_migration_partial_failure`, `record_migration_dry_run`, `record_migration_orphan_cleared`, `record_migration_action_started`, `record_migration_action_completed`, `record_migration_fresh_database_stamped`, `record_migration_epoch_advanced`, plus refusal-path methods: `record_migration_unsupported_dialect_refusal`, `record_migration_orphan_live_table_refusal`, `record_migration_orphan_live_column_refusal`, `record_migration_lock_held_refusal`, `record_migration_orphan_plan_refusal`, `record_migration_destructive_refused`, `record_migration_non_empty_definitional_refused`, `record_migration_wrong_application_refusal`, `record_migration_fk_enforcement_lost`, `record_migration_trigger_recreation_failed`. The caretaker
accepts a `MigrationRecorder` parameter; the concrete Landscape recorder is wired at L3
startup. **Wiring strategy: per-call context manager.** The concrete recorder is constructed with `(url, passphrase)` and opens a fresh `LandscapeDB` via `LandscapeDB.from_url(url, passphrase=...)` per `record_*` call, matching the established per-call pattern at `auth/audit.py:174,197,226` and `composer/tutorial_service.py:268,371`. Rationale: the caretaker is a startup-blocking synchronous path; introducing a long-lived `app.state.landscape_engine` for this single use case would diverge from the rest of the subsystem and add a lifetime-management surface the project does not currently maintain. The cost (one open/close per audit event; ~26 cycles for a 12-step shadow-copy with per-action records) is acceptable on a startup path. A future optimisation — held-open engine for the duration of `run_caretaker` only — is documented as Phase-11 work but is NOT part of Phase 9. Keeps caretaker layer-pure and test doubles zero-cost.

**TDD shape:**

- [ ] Write failing introspection-based test: iterate `dataclasses.fields(record)` for
  each record dataclass; for every field whose type annotation is a `Mapping` or
  `Sequence`, assert `isinstance(getattr(record, field.name), (types.MappingProxyType,
  tuple, frozenset))`. This catches future fields added without `freeze_fields`
  automatically.
- [ ] Run → fail.
- [ ] Implement.
- [ ] Run → pass.
- [ ] Commit: `contracts: migration plan + audit record types (Phase 9 Task 3)`.

### Task 4 — Migration executor

**Goal.** Module `src/elspeth/web/sessions/migration/executor.py`. Pure function:
`execute_plan(plan, engine, *, dry_run: bool) -> MigrationOutcome`. Per-action handlers:
`_noop`, `_create_table`, `_preserve_add_columns`, `_shadow_copy_enum_rewrite`,
`_recreate_definitional`. Refuses destructive changes (column drop on a data table) with
a `DestructiveMigrationRefused` exception that includes the table name and the offending
diff.

`_recreate_definitional` handles the `recreate` action (definitional tables that ARE
present in the live DB) and is a critical-data-loss path. It must enforce its
precondition mechanically: before dropping the table, run `SELECT COUNT(*) FROM
<table>`. If the count is greater than zero, crash immediately with a
`NonEmptyDefinitionalTable` exception that names the table, the row count, and the
`migration_class` declaration source (the `info` dict and the file/line where it was
set). This guard applies ONLY to the `recreate` action. The `_create_table` handler is
the fresh-creation path (table absent from live DB) and requires no such guard — there
is nothing to drop.

The shadow-copy ceremony (`_shadow_copy_enum_rewrite`) follows the SQLite documentation
§7 "Making Other Kinds Of Table Schema Changes"
(https://www.sqlite.org/lang_altertable.html#otheralter) as its normative reference.
The executor must follow these 12 steps: (1) disable FK pragma; (2) create new table
from `metadata`; (3) copy rows from old table; (4) drop indices on old table; (5) drop
triggers on old table; (6) drop views that reference old table; (7) drop old table;
(8) rename new table to old name; (9) recreate indices; (10) recreate triggers;
(11) recreate views; (12) check FK integrity → re-enable FK pragma → commit. The
implementation file must include a comment block at the top of the shadow-copy handler
citing this reference URL and listing these 12 steps by number.

**Trigger recreation (step 10).** The session DB has six required SQLite triggers declared via `_REQUIRED_SQLITE_TRIGGERS` in `schema.py:55-64` and registered via SQLAlchemy `event.listen(Table, 'after_create', DDL(...))` listeners. These DDL events fire on `metadata.create_all()` but do NOT fire on direct `schema.CreateTable(tbl)` calls inside the shadow-copy ceremony. The executor MUST therefore enumerate the required triggers for the table being shadow-copied and emit each `CREATE TRIGGER` statement as part of step 10. The enumeration source is the same `_REQUIRED_SQLITE_TRIGGERS` constant (relocated alongside `_validate_partial_index_dialect_symmetry` into `caretaker.py` per Task 5a). A post-step-10 assertion confirms the trigger names present on the table match the expected set; mismatch raises `MigrationTriggerRecreationFailed`, which appears in §3's refusal-event list and Task 3's record-list and is bound to the record-then-raise contract.

DDL is emitted exclusively via SQLAlchemy schema objects (`schema.CreateTable(tbl)`,
`tbl.append_column()`, `AddColumn`, `text()` with bound parameters); raw f-string SQL in
DDL handlers is forbidden. CI lint enforces this on the migration module.

The shadow-copy executor obtains a raw DBAPI connection (`engine.raw_connection()`) for
the duration of the 12-step ceremony, bypassing the engine-level `connect` event listener
that re-enables `PRAGMA foreign_keys=ON`. The ceremony is responsible for restoring FK enforcement at step 12 before commit.

**Pool-poisoning prevention (load-bearing).** A naive implementation that toggles `PRAGMA foreign_keys=OFF` on a raw connection and then encounters an exception will return that connection to the SQLAlchemy pool with FK enforcement still disabled. Subsequent pool checkouts silently skip FK enforcement — a Tier-1 audit integrity failure per `engine.py:96-100`. The ceremony MUST therefore:

1. Wrap steps 1–12 in `try/finally`.
2. In the `finally` block, on ANY exception path (and only on the exception path), call `raw_conn.detach()` to force-discard the poisoned DBAPI connection from the pool before re-raising. The connect listener will re-stamp `foreign_keys=ON` on the next pool fill.
3. After commit (success path), acquire a NEW pooled connection via `engine.connect()` and assert `result = conn.execute(text('PRAGMA foreign_keys')).scalar(); assert result == 1, ...`. If the assertion fails, raise `MigrationFKEnforcementLost` and treat as a partial-migration failure. This converts a silent Tier-1 failure into an immediate detectable crash.

Both invariants — `detach()` on error path, `foreign_keys=1` assertion on success path — are independently tested in Task 4 (see TDD shape below).

**TDD shape:**

- [ ] Write failing tests for each handler in isolation, using in-memory SQLite engines.
  Include separate tests for the `create_table` path (table absent → created; no count
  guard called) and the `recreate` path (table present → count guard checked; non-empty
  definitional table → `NonEmptyDefinitionalTable` raised). The shadow-copy outcome test additionally asserts that after the ceremony completes, `SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=<table>` returns exactly the expected set of required triggers.
- [ ] Write failing test (`test_executor_raw_connection_pool_safety.py`): simulate an exception inside the shadow-copy ceremony (monkey-patch step 5 to raise); assert that on the next `engine.connect()` checkout, `PRAGMA foreign_keys` returns `1` (the poisoned connection was detached, the pool refilled, and the connect listener re-stamped FK enforcement). The test MUST FAIL without the `try/finally` + `detach()` discipline.
- [ ] Write failing test (`test_executor_fk_post_assertion.py`): construct a scenario where the ceremony forgets to re-issue `PRAGMA foreign_keys=ON` before commit (monkey-patch step 12 to no-op); assert `MigrationFKEnforcementLost` is raised by the post-ceremony assertion.
- [ ] Write failing test `test_executor_shadow_copy_step_order.py`: instrument the shadow-copy handler with a statement-capture recorder (custom DBAPI cursor that records every SQL statement); execute a shadow-copy against a small fixture; assert the captured sequence matches the normative 12-step order from sqlite.org/lang_altertable.html §7 exactly. Steps 1 (FK off) and 12 (FK on + check) are independently asserted as the first and last statements respectively. A correctly-outcome-but-mis-ordered implementation MUST fail this test.
- [ ] Run → fail.
- [ ] Implement handlers. The shadow-copy ceremony is SQLite-specific; no per-dialect adapter is needed. The executor follows the SQLite documentation §7 normative reference cited above. `_create_table` is the fresh-creation path; `_recreate_definitional` is the present-but-recreate path with the `SELECT COUNT(*)` guard.
- [ ] Run → pass.
- [ ] Commit: `migration: executor with five action handlers (Phase 9 Task 4)`.

### Task 4a — WebSettings fields + same-path model_validator

**Goal.** Wire the Phase 9 configuration into `src/elspeth/web/config.py`. The plan's §3 step 4a and §3 Landscape/sessions DB separation invariant currently assume these settings and the same-path validator already exist; they do not. This task ships them BEFORE Task 5 wires the caretaker.

Add to `WebSettings`:

- `migration_dry_run: bool = False` (env: `ELSPETH_WEB__MIGRATION_DRY_RUN`)
- `migration_orphan_scan_days: int = 90` (env: `ELSPETH_WEB__MIGRATION_ORPHAN_SCAN_DAYS`)

Add an `@model_validator(mode='after')` named `_reject_same_path_for_landscape_and_sessions`. It must:

- Parse both `landscape_url` and `session_db_url` as SQLAlchemy URLs.
- For each, if `dialect == 'sqlite'`, resolve the `database` component to an absolute filesystem path (`pathlib.Path.resolve()`).
- If both resolve to the same path, raise `ValueError("landscape_url and session_db_url resolve to the same SQLite path: {path}. Sharing one SQLite file between Landscape and sessions DBs is forbidden — it intermixes schema with audit data.")`.
- Skip the check entirely if either dialect is non-SQLite (no equivalence concern; SQLite is the only Phase 9 target).

**TDD shape:**

- [ ] Write failing unit test `tests/unit/web/test_config_migration_settings.py`: assert `WebSettings(migration_dry_run=True).migration_dry_run is True`; assert `ELSPETH_WEB__MIGRATION_DRY_RUN=1 elspeth-web` parses to `migration_dry_run=True`; assert same-path config raises `ValidationError` with a message containing both 'same SQLite path' and the conflicting path.
- [ ] Run → fail (fields and validator absent).
- [ ] Implement fields and `_reject_same_path_for_landscape_and_sessions` following the precedent at `config.py:447,462,472`.
- [ ] Run → pass.
- [ ] Commit: `migration: WebSettings fields + same-path model_validator (Phase 9 Task 4a)`.

The integration-level same-path test referenced in §3 ("the integration test in Task 5 asserts this invariant with a bad-config fixture") remains as Task 5's bad-config fixture and serves as a second coverage layer on top of this unit test.

### Task 4.5 — No-op caretaker tracer-bullet

**Goal.** Validate the caretaker activation contract end-to-end with a minimal no-op caretaker BEFORE the full executor (Task 4) is integrated into the startup path. This is a tracer bullet: it proves the `MigrationRecorder` Protocol, the `create_app()` call site, the per-call Landscape write pattern, and the orphan-scan guard all work together — under one integration test — before Task 5 commits to wiring the full caretaker.

Implementation:

- Module `src/elspeth/web/sessions/migration/_tracer.py` exposes `run_noop_caretaker(engine, metadata, recorder)`. It performs steps 1-7 of the §3 activation contract (read metadata; connect; classify; dialect check; orphan-scan; compute plan as all-noop; emit `MigrationPlanRecord`) and step 9 (`MigrationCompletedRecord` with empty actions). Steps 4a (dialect refusal), 5 (orphan-table refusal), and the orphan-plan startup guard ARE exercised; no DDL is emitted.
- The tracer is gated by an env var `ELSPETH_WEB__MIGRATION_TRACER=1` and is automatically replaced by `run_caretaker` (Task 5) when Task 5 lands.

**TDD shape:**

- [ ] Write integration test `tests/integration/web/test_caretaker_tracer.py` exercising: clean start (all-noop plan recorded); orphan-table (refusal recorded + raised); orphan-plan-record-from-prior-run (refusal recorded + raised); dialect=postgres (refusal recorded + raised); same-path config (refusal raised at config-validation time, before caretaker runs).
- [ ] Run → fail.
- [ ] Implement `_tracer.py` and the `create_app()` integration.
- [ ] Run → pass.
- [ ] Commit: `migration: no-op caretaker tracer-bullet (Phase 9 Task 4.5)`.

### Task 5 — Caretaker activation hook

**Goal.** Wire the caretaker into `create_app()` synchronously, immediately after session engine construction and before `include_router` is called. Module `src/elspeth/web/sessions/migration/caretaker.py` exports `run_caretaker(engine, metadata) -> None`. This is the same call site that `initialize_session_schema` (displaced by Task 5a) occupies at `app.py:581`; Task 5 replaces that call with `run_caretaker`. Dry-run mode honoured via `ELSPETH_WEB__MIGRATION_DRY_RUN`. Task 5 supersedes the Task 4.5 tracer-bullet by replacing the `_tracer.run_noop_caretaker` import with the full `caretaker.run_caretaker`. The tracer module is deleted in the same commit per the No Legacy Code Policy.

**TDD shape:**

- [ ] Write failing integration test:
  `tests/integration/web/test_caretaker_activation.py` — starts the app with a
  pre-seeded "old shape" DB (missing `tutorial_completed_at`), asserts the column
  exists after startup, asserts the audit events were recorded.
- [ ] Run → fail.
- [ ] Implement `caretaker.py` and the app-composition wiring.
- [ ] Run → pass.
- [ ] Commit: `migration: caretaker startup activation (Phase 9 Task 5)`.

### Task 5a — Displace `initialize_session_schema`

**Goal.** Remove `src/elspeth/web/sessions/schema.py`'s `initialize_session_schema`
function (called at `app.py:581`) — it crashes on any schema drift via
`SessionSchemaError`, which antagonises the caretaker. Per the No Legacy Code Policy,
the function is deleted entirely; the caretaker is the new schema gate.

**Decision (default per No Legacy Code Policy): option (a) — delete the function
entirely.** Option (b) — hollow it out to call the caretaker — introduces a wrapper
shim that survives exactly one refactor cycle before becoming confusing. Delete it.

**The `_validate_partial_index_dialect_symmetry` module-level helper is load-bearing**
(closes elspeth-obs-2ef48619d5). It is a module-level function at `schema.py:377` called from `initialize_session_schema` (defined at `schema.py:75`) via line 185; it is NOT a closure. It MUST NOT be deleted.
Relocate it to the caretaker module (`src/elspeth/web/sessions/migration/caretaker.py`)
as a module-level function; call it at the end of `run_caretaker` as a post-execution
sanity check. The caretaker calls it after every run regardless of whether any DDL was
issued.

**The schema-sentinel system at `schema.py:109-168` is also load-bearing** and MUST NOT be silently deleted with `initialize_session_schema`. It comprises two functions performing distinct safety roles:

- `_stamp_schema_sentinels(engine)` — writes `PRAGMA application_id = SESSION_DB_APPLICATION_ID` (0x454C5350 — the 'ELSP' identifier) and `PRAGMA user_version = SESSION_SCHEMA_EPOCH` to a freshly-created sessions DB. The application_id refuses opening a SQLite file that belongs to a different application; the user_version is a fast-path schema-generation watermark.
- `_assert_schema_sentinels(engine)` — refuses to proceed if the live DB's `PRAGMA application_id` does not match the expected value. This guards against accidentally pointing the caretaker at, for example, a Landscape DB file (a misconfiguration which would otherwise pass `OrphanLiveTable` and proceed to mutate the wrong file).

Relocate BOTH functions into `src/elspeth/web/sessions/migration/caretaker.py`:

- `_assert_schema_sentinels` runs FIRST in `run_caretaker`, before the dialect check (§3 step 4a), the orphan-table scan (§3 step 5), and plan computation. A mismatched `application_id` aborts the caretaker with `WrongApplicationDatabase(actual_application_id, expected_application_id, db_path)` and records the refusal as `MigrationWrongApplicationRefusal` (see §3's record-then-raise contract) before raising.
- `_stamp_schema_sentinels` runs in the fresh-DB path within the executor's `_create_table` handler (and only on the first action when reflection reports zero tables). The caretaker emits a `MigrationFreshDatabaseStamped` audit event after the stamps are written.

The `SESSION_SCHEMA_EPOCH` watermark continues to advance per release. The caretaker writes the current epoch on every successful run completion (a `MigrationEpochAdvanced` event records the old and new values). This preserves the existing rollback-detection capability that `_assert_schema_sentinels` provides today.

**TDD shape:**

- [ ] Write failing tests covering the displacement:
  - `tests/integration/web/test_caretaker_activation.py` (from Task 5) must be updated
    to assert that starting the app with a pre-seeded "old shape" DB does NOT call
    `initialize_session_schema`.
  - Confirm `_validate_partial_index_dialect_symmetry` is exercised by the caretaker
    path (existing `test_schema.py` assertions for it may need retargeting to
    `caretaker.py`).
- [ ] Run → fail (function still exists).
- [ ] Delete `initialize_session_schema` from `schema.py`. Move
  `_validate_partial_index_dialect_symmetry` into `caretaker.py`; call it at the end of
  `run_caretaker`. Update `app.py:581` to remove the `initialize_session_schema` call
  (the caretaker hook from Task 5 replaces it).
- [ ] Exhaustive caller list: at the start of Task 5a, run both of these greps and treat their output as the authoritative call-site enumeration:

  ```
  grep -rln initialize_session_schema tests/
  grep -rln initialize_session_schema src/
  ```

  At HEAD on 2026-05-19 these greps return approximately 50 test files and `src/elspeth/web/app.py` as the sole production caller. The exact set drifts as the codebase evolves — the prior plan revision's enumerated 9-file list became stale and was removed. The grep is authoritative.

  Update **every file returned by the greps** in the same commit. For each call site, replace `initialize_session_schema(engine)` with one of:

  - `metadata.create_all(engine)` — for tests/fixtures that only need a schema-initialised DB and are NOT exercising the caretaker path.
  - A shared `caretaker_initialised_engine` fixture — for tests that ARE exercising or asserting the caretaker activation path.

  Run the greps again BEFORE committing to confirm zero residual references. The commit MUST land with the grep returning empty.
- [ ] Run → pass.
- [ ] Commit: `migration: remove initialize_session_schema; caretaker is new schema gate (Phase 9 Task 5a)`.

### Task 6 — Audit logging of migration events

**Goal.** The caretaker emits `MigrationPlanRecord` BEFORE execution and
`MigrationCompletedRecord` / `MigrationFailedRecord` / `MigrationDryRunRecord` AFTER.
Each record contains the table name, the action, the SQL emitted (canonicalised), and a
SHA-256 hash of `sorted(metadata.tables.keys())` rendered as RFC-8785 canonical JSON,
then SHA-256 — matching the RFC-8785 + SHA-256 pattern used in
`contracts/composer_audit.py`. This hash is computed for both the pre-execution and
post-execution `metadata.tables` key sets.

Per CLAUDE.md audit primacy: audit fires synchronously and crashes on failure. If the
Landscape cannot accept a migration event, the caretaker aborts and the app does not
start.

In addition to the per-plan `MigrationPlanRecord` / `MigrationCompletedRecord` /
`MigrationFailedRecord`, the executor emits `MigrationActionStartedRecord` BEFORE and
`MigrationActionCompletedRecord` AFTER each individual action (e.g. each step of the
12-step shadow-copy). This allows forensic reconstruction of mid-ceremony SIGKILL state.
Both per-action records are synchronous, audit-primacy-compliant, and counted in
idempotence and orphan-scan logic.

**TDD shape:**

- [ ] Write failing test verifying record emission order (plan-before-execution,
  completion-after-execution), and that an audit-write failure aborts the caretaker.
- [ ] Write failing test (`test_refusal_path_events.py`): for each of the nine refusal paths above, construct a caretaker scenario that triggers the refusal; assert the corresponding Landscape event was recorded BEFORE the exception propagates (use a `MigrationRecorder` test double that records call order). Parametrize over all nine refusal paths.
- [ ] Run → fail.
- [ ] Implement Landscape recorder methods (`record_migration_plan`,
  `record_migration_completed`, `record_migration_failed`, `record_migration_dry_run`)
  and wire them into the caretaker.
- [ ] Run → pass.
- [ ] Commit: `migration: Landscape audit events for caretaker actions (Phase 9 Task 6)`.

### Task 7 — Fail-safe rollback

**Goal.** Wrap the executor in transactional discipline. If any per-table action fails
mid-plan, rollback the transaction and emit `MigrationFailedRecord` with the failing
action, the exception, and a "DB UNCHANGED" marker. SQLite supports transactional DDL
with partial constraints; the executor handles the documented SQLite-specific cases (e.g.
`CREATE TABLE` is transactional, `PRAGMA` is not). Where transactional DDL is unavailable
for a specific operation, the executor runs per-table transactions and emits a
`MigrationPartialFailureRecord` listing which tables succeeded and which did not.

**Disjoint failure-record conditions.** The two failure record types fire under non-overlapping conditions:

- `MigrationFailedRecord` — emitted when the plan fails BEFORE any action's transaction has committed, OR when a per-action rollback completes cleanly leaving the DB at its pre-plan state. The "DB UNCHANGED" marker is correct only in this case.
- `MigrationPartialFailureRecord` — emitted when AT LEAST ONE action's transaction has committed before a subsequent action fails. The record enumerates which actions committed (their SQL is durable) and which did not. The DB is in a *partially migrated* state and the operator's recovery path is the Task 10 runbook entry for partial-migration recovery.

Determining which record fires is a load-bearing executor concern: the executor maintains a `committed_actions: list[MigrationActionRecord]` accumulator and emits `MigrationFailedRecord` iff that list is empty at failure time.

The fail-safe must also handle the case where `MigrationFailedRecord` write itself
fails after a successful rollback. Per CLAUDE.md audit primacy: no retry; the caretaker
crashes immediately with the Landscape write exception preserved via `from exc`. The
sessions DB is in whatever state the rollback left it. The Landscape will contain an
orphan `MigrationPlanRecord` with no outcome record — this is the correct invariant:
an orphan `MigrationPlanRecord` means "caretaker crashed before outcome could be
recorded; DB state is whatever the rollback left." The operator inspects the
`MigrationPlanRecord` to determine what ran before the crash.

**Orphan plan-record detection (startup guard).** At caretaker startup, BEFORE plan
computation, the caretaker scans the Landscape for `MigrationPlanRecord` entries with
no paired outcome record (`MigrationCompletedRecord`, `MigrationFailedRecord`, or
`MigrationDryRunRecord`). The scan queries only `MigrationPlanRecord` entries within a
90-day lookback window (configurable via `ELSPETH_WEB__MIGRATION_ORPHAN_SCAN_DAYS`);
older records are assumed resolved by prior operator action and skipped. If any are
found, the caretaker refuses to start with `OrphanMigrationPlanRefusal(plan_record_id,
plan_recorded_at, sessions_db_path)`. Per CLAUDE.md Tier 1 policy: an orphan plan is
bad data in our own audit trail; we crash rather than silently recover. The operator
must clear the orphan via the runbook entry in Task 10 before the next deploy can proceed.

Add an `elspeth migration force-clear-orphan-plan <plan_id>` subcommand to the existing
`elspeth` CLI (NOT to `elspeth-mcp`, which is read-only). The subcommand writes a
`MigrationOrphanCleared` event linking the plan_id to operator-supplied justification
text, then exits. Documented in the Task 10 runbook step (3): `elspeth migration
force-clear-orphan-plan <id> --reason "<text>"`. The escape hatch preserves the audit
chain.

**Input sanitisation.** The `--reason "<text>"` argument is Tier 3 operator input and must be bounded before reaching the Landscape:

- Length cap: 500 characters. Typer annotation: `Annotated[str, typer.Option(..., max_length=500)]`.
- Character set: printable ASCII + common Unicode word-chars; control characters (anything matching `[\x00-\x1F\x7F]`) rejected.
- Empty/whitespace-only rejected.

A unit test (`tests/unit/cli/test_migration_force_clear_orphan_plan.py`) asserts each rejection path emits a clear Typer error and does NOT write any Landscape event.

**TDD shape:**

- [ ] Write failing test simulating an action-handler failure mid-plan; assert the
  DB shape matches pre-migration state.
- [ ] Add failing test: Landscape contains an orphan `MigrationPlanRecord` → caretaker
  startup raises `OrphanMigrationPlanRefusal` before any plan computation.
- [ ] Run → fail.
- [ ] Implement rollback wrapper and orphan-scan startup guard.
- [ ] Run → pass.
- [ ] Commit: `migration: fail-safe rollback on partial executor failure (Phase 9 Task 7)`.

### Task 7a — Cross-process advisory lock (SQLite)

**Goal.** Before plan computation, the caretaker acquires a process-exclusive advisory lock on a sidecar file `<session_db_path>.migration.lock` via `fcntl.flock(LOCK_EX | LOCK_NB)`. If acquisition fails, the caretaker refuses to start with `MigrationLockHeld(lock_path, holder_pid_if_readable)`. Lock released after the final outcome record is written. This serialises caretaker runs across concurrent processes (e.g. systemd flap, rolling restart) — the SQLite file-level locks alone do not protect against the multi-step ceremony being interleaved.

**TDD shape:**

- [ ] Failing test: two `multiprocessing.Process` instances start simultaneously against the same DB path; assert exactly one plan+completion pair lands in the Landscape; the other process raises `MigrationLockHeld`.
- [ ] Run → fail.
- [ ] Implement `flock`-based context manager in caretaker.py.
- [ ] Run → pass.
- [ ] Commit: `migration: cross-process advisory lock via flock sidecar (Phase 9 Task 7a)`.

### Task 8 — Idempotence

**Goal.** Running the caretaker against a DB already at the declared shape emits zero
DDL and records a single `MigrationCompletedRecord` with action list `[("*", "noop")]`.

**TDD shape:**

- [ ] Write failing test: run caretaker twice in sequence against an empty in-memory
  DB; assert the second run emits zero DDL. Also assert that if any table was
  classification-defaulted (no `migration_class` in `info`), the `MigrationPlanRecord`
  on the second run still records the omission (emit-every-deploy, not once-tracked).
- [ ] Run → fail (or pass-by-accident; if pass-by-accident, strengthen the assertion
  to also check Landscape emits exactly one `MigrationCompletedRecord` with all-noop
  actions on the second run, and that the classification-default omission is present
  in the second run's `MigrationPlanRecord`).
- [ ] Implement: detector returns all-noop plan when reflection matches metadata.
- [ ] Run → pass.
- [ ] Commit: `migration: caretaker idempotence verified (Phase 9 Task 8)`.

### Task 9 — Three-generation migration integration test

**Goal.** The flagship integration test (`tests/integration/web/test_three_generation_migration.py`):
boot the app against a DB at Phase 1A shape (only `user_preferences_table` with no
`tutorial_completed_at` column, no `interpretation_events_table`), seed real user data
into `user_preferences_table` and `sessions_table`, then run the caretaker at current
HEAD. Assert: all three generations of schema additions have applied; **all seeded user
data is preserved**; the Landscape contains one `MigrationCompletedRecord` per added
schema element with deterministic ordering.

The test runs against three DB fixtures representing each generation:

| Fixture | Tables present | Columns present |
|---|---|---|
| `gen_1_phase_1a_only.sqlite` | sessions, chat_messages, composition_states, user_preferences (no tutorial column) | per-Phase-1A schema |
| `gen_2_phase_1a_plus_4a.sqlite` | as gen-1 + tutorial_completed_at column | per-Phase-4A schema |
| `gen_3_phase_1a_plus_4a_plus_5b.sqlite` | as gen-2 + interpretation_events + provenance enum extended | per-Phase-5b schema |

The test matrix runs caretaker migrations: gen-1 → gen-3 (full path), gen-1 → gen-2
(stop after 4A), gen-2 → gen-3 (Phase-5b only). Each path asserts data preservation.

**TDD shape:**

- [ ] Write failing test with the matrix above. Fixtures generated by booting an app at
  the historical revision; checked into the repo as binary SQLite files.
- [ ] Run → fail.
- [ ] Iterate detector/executor until all matrix entries pass.
- [ ] Commit: `migration: three-generation integration test (Phase 9 Task 9)`.

### Task 10 — Dry-run smoke + operator runbook

**Goal.** Document `ELSPETH_WEB__MIGRATION_DRY_RUN=1 elspeth-web` for operators. Update
`docs/guides/` with a Phase-9 migration runbook: "before deploying a schema-changing
release, run with dry-run, review the Landscape `MigrationDryRunRecord`, then deploy
live."

The runbook must also document:

- **Emergency non-additive change path.** A schema change that the caretaker cannot
  handle (column rename, type change, data backfill) requires deleting the DB. After
  Phase 9 ships, this destroys all user state accumulated since the Phase 9 ship date.
  Operators must back up the sessions DB before performing any non-additive emergency
  change. See R8 in §6 for the accepted mitigation.

- **Post-ship defect recovery (mis-migration).** If Phase 9 ships with a defect
  discovered post-deploy after user state has accumulated: operator backs up the sessions
  DB, performs delete-the-DB, and restarts. This destroys accumulated user state since
  the Phase 9 ship date; that is the accepted cost of mis-migration recovery under the
  project's existing `project_db_migration_policy` retained as fallback.

- **Orphan plan-record cleanup.** If the caretaker refuses to start with
  `OrphanMigrationPlanRefusal`, the operator must: (1) inspect the orphan
  `MigrationPlanRecord` in the Landscape to determine what ran before the crash; (2)
  verify the sessions DB shape is consistent; (3) if DB is consistent, run
  `elspeth migration force-clear-orphan-plan <id> --reason "<text>"` to write a
  `MigrationOrphanCleared` event and clear the refusal; (4) retry deploy. If DB is
  inconsistent, fall back to backup + delete-the-DB.

- **Code rollback procedure.** Rolling back code to a revision whose `metadata` does not
  declare a newer table triggers `OrphanLiveTable` refusal. Documented procedure: (a)
  restore the sessions DB from backup taken before the schema-adding deploy, or (b)
  forward-patch the rollback target's `metadata` to include classification stubs for the
  newer tables (no DDL, classification only). Option (a) is preferred; option (b) is a
  code-level emergency override.

- **Mid-ceremony SIGKILL recovery.** If the process dies during the 12-step shadow-copy ceremony, the next caretaker boot's orphan-plan scan fires (Task 7). The operator uses the per-action audit records (Task 6) to reconstruct which ceremony steps committed. After running `elspeth migration force-clear-orphan-plan <id>` to clear the orphan-plan record, the next caretaker boot encounters a SECOND refusal: the live DB now contains a `<table>_new` orphan (if SIGKILL occurred between steps 2 and 8), which triggers `OrphanLiveTable`. Recovery procedure:
  1. Inspect per-action records to determine the last completed ceremony step.
  2. If steps 1-7 committed but step 8 (rename) did NOT: drop the `<table>_new` table manually (the original `<table>` is intact since drop was step 7, but the rename never happened — actually no, step 7 drops `<table>` and step 8 renames `<table>_new` to `<table>`, so this state is: no `<table>`, has `<table>_new`). Manually `ALTER TABLE <table>_new RENAME TO <table>` and run `PRAGMA foreign_key_check`; if clean, retry deploy.
  3. If steps 1-6 committed but step 7 did NOT: drop the orphan `<table>_new` (the original `<table>` is intact); retry deploy.
  4. If FK enforcement is in an indeterminate state (the process died after step 1 disabled FK but before step 12 re-enabled): the executor's post-ceremony assertion in Task 4 catches this on next boot (`MigrationFKEnforcementLost`); the operator runs `PRAGMA foreign_keys=ON; PRAGMA foreign_key_check;` to confirm integrity before retrying.
  5. In all paths, ensure `application_id` and `user_version` (the schema sentinels relocated by Task 5a) are intact via `PRAGMA application_id; PRAGMA user_version;` before retrying. If the sentinels were lost, restore from backup.

- **Landscape DB failure recovery.** If the Landscape DB itself is the source of caretaker refusal (disk full, file corruption, locked by another process), the `force-clear-orphan-plan` recovery tool will ALSO fail because it writes to the same Landscape. Recovery procedure:
  1. Diagnose the Landscape failure via direct SQLite inspection of the Landscape DB file (`sqlite3 <landscape_path> '.tables'`).
  2. If disk full: free space; re-run caretaker.
  3. If corruption: the operator MUST restore the Landscape DB from backup before any other recovery. Phase 9 has no path to write to a corrupt Landscape — by audit-primacy design, the caretaker cannot proceed without a working audit channel. The sessions DB is unmodified in this scenario (the caretaker aborted before plan execution).
  4. If locked by another process: identify the holder via `fuser <landscape_path>` or `lsof <landscape_path>`; resolve the conflict; re-run caretaker.
  5. After Landscape recovery, retry caretaker normally. If the sessions DB had an orphan plan from a prior crash, the orphan-plan-scan refusal will now succeed in recording a `MigrationOrphanPlanRefusal` and the operator can proceed with the orphan-cleanup procedure above.

Update `project_db_migration_policy` memory entry: mark superseded by Phase 9; reference
the runbook.

**TDD shape:**

- [ ] Write smoke test that runs the CLI in dry-run mode against a real shape-changing
  delta and asserts the app exits cleanly and the audit record is present.
- [ ] Run → fail.
- [ ] Implement CLI flag handling.
- [ ] Run → pass.
- [ ] Write runbook doc.
- [ ] Commit: `migration: dry-run mode + operator runbook (Phase 9 Task 10)`.

### Task 11 — CI fixture-freshness gate

**Goal.** CI job that boots the app at HEAD against each of the three generation
fixtures (§5 "Fixture DB generation") and asserts the caretaker's `MigrationPlanRecord`
is all-noop for the tables that already match HEAD schema. Detects fixture rot when new
schema changes land without corresponding fixture regeneration: if a Phase-10 schema
addition creates a new table, the gen-3 fixture no longer contains it, and the
fixture-freshness job fails until the fixture is regenerated.

**Second CI gate: synthetic-only + tamper-detection.** Phase 9 ships a second job (`ci-fixture-content-integrity`) that runs on every PR touching `tests/fixtures/migration/` or `FIXTURES.md`. The job:

1. Verifies each fixture binary's SHA-256 matches the entry in `tests/fixtures/migration/FIXTURES.md`. Mismatch → fail.
2. Opens each fixture in read-only mode and asserts the synthetic-only constraint: every text column matching session-content/user-identifier patterns (`session_id`, `user_id`, email fields, preference content text) is scanned. Rules: emails must end in `.invalid`, `.example`, or `.test`; UUIDs must come from a documented synthetic-UUID set declared in `tests/fixtures/migration/synthetic_uuid_set.json`. Violation → fail with the offending row+column identified.
3. The synthetic-UUID set is a CHECKED-IN allowlist (~50 UUIDs generated by `uuid.uuid5(namespace=DNS, name='elspeth-fixture-NNNN.invalid')`) — deterministic, regenerable, and easily auditable. Any UUID in a fixture not in the allowlist fails the gate. This addresses the systems reviewer's concern that UUIDs are otherwise indistinguishable from production-origin UUIDs.

**TDD shape:**

- [ ] Write failing test: `tests/integration/web/test_fixture_freshness.py` — parametrized
  over the three generation fixtures; for each, boot the app, run the caretaker, assert
  the resulting `MigrationPlanRecord` contains only expected schema-delta actions (none
  unexpected; fixture is fresh relative to HEAD schema minus the known generational
  delta). A fixture that requires unexpected actions → test failure.
- [ ] Write failing test `tests/integration/web/test_fixture_content_integrity.py`: parametrized over the three fixtures + a synthetic 'tampered' fixture; for the legitimate three, assert pass; for the tampered fixture (a fixture binary with one byte flipped), assert fail with a SHA-256 mismatch error; for a fixture seeded with a non-allowlist UUID, assert fail with a synthetic-only-violation error.
- [ ] Run → fail (both test infrastructure and CI scripts absent).
- [ ] Implement parametrized fixture-freshness test and wire into CI as a matrix job
  over all three fixtures (add to `.github/workflows/` or equivalent CI config as
  `ci-fixture-freshness` job with matrix `fixture: [gen_1, gen_2, gen_3]`).
- [ ] Implement the second CI job script `scripts/cicd/verify_migration_fixtures.py` and wire it into the workflow alongside the schema-freshness job (`ci-fixture-content-integrity`).
- [ ] Run → pass.
- [ ] Commit: `migration: CI fixture-freshness gate (Phase 9 Task 11)`.

---

## 5. Test strategy

### Layers

| Layer | Mechanism | Coverage |
|---|---|---|
| Unit | pytest + in-memory SQLite (`sqlite:///:memory:`) | detector, executor handlers, classification walker, dataclass freeze guards |
| Concurrency | pytest with `multiprocessing` + temp SQLite paths | cross-process advisory-lock correctness (Task 7a) |
| Integration | FastAPI TestClient against an on-disk SQLite created from a fixture | caretaker activation, audit-event emission, failure → app-does-not-start |
| Three-generation | Checked-in SQLite fixtures representing each schema generation | data preservation across gen-1 → gen-3 migration matrix (Task 9) |

### Fixture DB generation

The three generation fixtures are generated by:

1. Checking out the git revision at which Phase 1A landed; booting the app; creating
   sessions and user preferences; copying the resulting SQLite file to
   `tests/fixtures/migration/gen_1_phase_1a_only.sqlite`.
2. Checking out the revision at which Phase 4A landed; repeating; copying to
   `gen_2_phase_1a_plus_4a.sqlite` (with tutorial-completion seeded for a subset of
   users).
3. Checking out the revision at which Phase 5b landed; repeating; copying to
   `gen_3_phase_1a_plus_4a_plus_5b.sqlite` (with interpretation events seeded).

These fixtures are **load-bearing** for the project's auditability claim — they are the
evidence that the migration runner preserves user state across each schema generation.
They are checked into the repo under `tests/fixtures/migration/` with SHA-256 hashes
recorded in a `FIXTURES.md` so tampering is detectable.

### Synthetic-only constraint

Every row in every fixture MUST be synthetic — no real session content, no real user
identifiers, no real preferences. Fixture generation scripts must use a known-synthetic
seed set, e.g. `FixtureSeed("alice@example.invalid", "Sample preference content")`.
Using `.invalid` TLD usernames and clearly synthetic preference text is the required
convention. A required CI gate (NOT a bypassable pre-commit hook) verifies the fixtures contain no
production-origin tokens (no `@` domains other than `.invalid`, `.example`, or `.test`;
no session IDs that match the format of real production sessions) AND that each fixture
binary's SHA-256 matches the entry in `FIXTURES.md`. CI fails on divergence. This
constraint is documented in `FIXTURES.md` alongside the SHA-256 tamper-detection record.

### Property-style assertions

For each generation pair (gen-N, gen-N+1):

- Every row in every data table at gen-N is present and field-identical at gen-N+1.
- New columns added at gen-N+1 have their declared `server_default` applied for rows
  that pre-existed at gen-N.
- Enum-extension migrations (Phase 5b) preserve all existing enum values; new values
  are accepted by post-migration writes.
- The Landscape `MigrationCompletedRecord` for the migration step lists the exact set
  of actions performed; no extra, no missing.

### Failure-mode tests

- DB unreachable at caretaker startup → app does not start; clear error.
- Landscape write fails during caretaker → app does not start; the migration is
  considered un-recorded and therefore un-performed.
- A data table has no `migration_class` info → caretaker defaults to `"data"` and
  preserves the table; the missing classification is recorded in the `MigrationPlanRecord`
  for operator visibility.
- A `metadata` table declares a column that the live DB has *with a different type* →
  caretaker refuses (this is destructive); crash with the offending diff.
- An action-handler raises mid-plan → executor rolls back the transaction (or emits
  `MigrationPartialFailureRecord` on dialects without transactional DDL); app does not
  start.
- Live DB contains a table not present in `metadata.tables` → caretaker refuses to start
  with `OrphanLiveTable(table_name, declared_tables)` before computing any plan.
- Live DB contains a column on a managed table that is not present in `metadata` for
  that table → caretaker refuses to start with `OrphanLiveColumn(table_name,
  column_name)` before computing any plan.

---

## 6. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **Misclassifying a data table as definitional** silently deletes user data on next deploy. | Critical | Classification lives adjacent to the Table definition in `sessions/models.py`; classification changes require a deliberate diff. Task 1 test asserts every table has explicit classification. `_recreate_definitional` refuses to drop a non-empty table: it runs `SELECT COUNT(*)` first; if the count is >0, it crashes with the table name, row count, and the `migration_class` declaration source — preventing any silent data-loss path. |
| R2 | **Cross-process race on caretaker startup.** Two processes (systemd flap, rolling restart, dual operator invocation) racing the caretaker against the same SQLite file. SQLite's file-level lock serialises individual statements but not the multi-step shadow-copy ceremony — both processes could pass orphan-scan, both attempt step 1, etc. | High | Task 7a `flock`-based advisory lock on a sidecar file (`<session_db_path>.migration.lock`). If acquisition fails, the caretaker refuses with `MigrationLockHeld`. Postgres `pg_advisory_lock` is out of scope (Postgres support deferred). |
| R3 | **SQLite transactional-DDL boundaries.** Some operations (PRAGMA, certain DDL combinations) commit implicitly. The 12-step shadow-copy ceremony's normative reference (sqlite.org/lang_altertable.html §7) is the source of truth; deviation from those 12 steps is forbidden. | Medium | Executor follows the 12-step normative reference exactly; comment block in the handler cites the reference URL and lists all 12 steps. |
| R4 | **Audit-recorder failure during caretaker** covers two distinct failure windows. (a) Pre-execution: if `MigrationPlanRecord` write fails, the executor does not run at all — no migration, no partial state. (b) Post-execution: if the executor runs, mid-execution fails, rollback succeeds, and then `MigrationFailedRecord` write itself fails — the caretaker crashes with the original exception preserved via `from exc`; the sessions DB is in whatever state the rollback left it; the Landscape contains an orphan `MigrationPlanRecord` with no outcome record, which means "caretaker crashed before outcome could be recorded; DB state is whatever the rollback left." This is the correct invariant: no retry; Landscape write failure aborts the caretaker. On the **next deploy**, the orphan-scan startup guard (Task 7) detects the orphan and raises `OrphanMigrationPlanRefusal` — the operator must clear the orphan via the runbook entry in Task 10 before the next deploy can proceed. | High | No retry. Landscape write failure at either window aborts the caretaker immediately with the offending exception. The orphan `MigrationPlanRecord` is the audit evidence of the crash; the operator inspects it via the Task 10 runbook to determine what ran and clears the orphan before retrying. |
| R5 | **Three-generation fixture rot** — the checked-in fixture SQLite files become stale relative to the project's `metadata` at HEAD, masking real migration bugs. | Medium | `FIXTURES.md` records the git revision each fixture was generated from. CI job verifies that booting the app at HEAD against each fixture in turn produces a clean Landscape (no unexpected actions in the migration plan). If a phase later adds a new table, the fixtures need regeneration; CI fails until they're updated. |
| R6 | **The classification mechanism creates a new tier-1 footgun.** Adding a new Table without specifying `migration_class` is a silent omission. | Low | If `info["migration_class"]` is absent, the caretaker defaults to `"data"` (preserve; never recreate). This is the conservative safe default: an unclassified table is never dropped. Task 1 test verifies this default path. Task 1 ships a CI gate (`scripts/cicd/enforce_migration_classification.py`) that asserts every `Table` in `metadata.tables.values()` has explicit `info['migration_class']`. New tables without classification fail CI. |
| R7 | **Phase 9 ships and then a future schema change requires a non-additive operation** (column rename, type change). The runner refuses; the operator falls back to delete-the-DB. | Medium | This is acceptable — the runner is honest about its scope. The runbook explicitly states "the runner handles additive changes; non-additive changes require a planned migration ceremony that is out of scope for Phase 9." Phase 9 explicitly defers non-additive schema changes (column rename, type change, data backfill) as accepted technical debt. Future Phase 12 will revisit if/when a non-additive change is needed; until then, the runbook's emergency delete-the-DB path applies, with the cost named to the operator. |
| R8 | **Phase 9 ships with a defect discovered post-deploy after user state has accumulated** — a mis-migration bug is not caught before real user data is written to the migrated DB. | Medium | Operator follows the emergency runbook entry in Task 10 (post-ship defect recovery): back up the sessions DB, perform delete-the-DB, restart. This destroys accumulated user state since the Phase 9 ship date; that is the accepted cost of mis-migration recovery under the project's existing `project_db_migration_policy` retained as fallback. No down-migration path exists (symmetric with Options (a)/(c) reversibility verdict in §2). |

---

## 7. Coherence with existing redesign phases

Phase 9 retrofits the migration runner; it does not modify the table definitions added
by Phase 1A, Phase 4A, or Phase 5b. The three upstream phase plans are **correct as
written** under the delete-the-DB policy. Phase 9 changes the runtime behaviour, not the
schema.

| Phase | Schema artifact | Phase 9 change to artifact? |
|---|---|---|
| 1A (plan 12) | `user_preferences_table` (user_id PK, default_composer_mode, banner_dismissed_at, updated_at) | **None.** Phase 9 only adds `info={"migration_class": "data"}` to the `Table()` call (Task 1). |
| 4A (plan 21a) | `tutorial_completed_at` column on `user_preferences_table` | **None.** Phase 9's executor handles "new column on data table" via the `preserve_add_columns` action. |
| 5b (plan 18a) | `interpretation_events_table` + `composition_states.provenance` enum extension (`interpretation_resolve`) | **None to the table definition.** Phase 9 adds the `migration_class` info and handles the closed-enum extension via the shadow-copy ceremony. |
| 6 (shipped) | `composer_completion_events_table` + `skill_markdown_history_table` | **None.** Phase 9 only adds `info={"migration_class": "data"}` to both `Table()` calls (Task 1). |

The upstream plans explicitly defer to Phase 9 (plan 12 implicit via roadmap §D6; plan
21a §"DB-delete cadence (Phase 9 owns the structural fix)"; plan 18a §"Migration runner
ownership (deferred — see roadmap §D5)"). Phase 9 picks up exactly where they deferred.

**No co-ordinated re-deploy of Phase 1A/4A/5b is required.** Phase 9 ships standalone:
the caretaker runs against whatever shape the DB currently has and brings it to HEAD.
Operators who have already done the delete-the-DB ceremony for prior phases see Phase 9
as a transparent improvement — the caretaker runs, finds no drift, emits an all-noop
`MigrationCompletedRecord`, and the app starts.

Phase 9 is SQLite-only by design (operator decision 2026-05-16). Future Postgres portability for the sessions DB requires a separate phase that revisits dialect-specific items; Phase 9 does not block that future phase, only declines to ship it.

---

## 8. Implementation readiness

**BLOCKED** on:

- **Phase 1A shipping** — `user_preferences_table` is added by Phase 1A. Phase 9 cannot
  ship before Phase 1A is present in git history; the classification table and the
  three-generation fixture both depend on it.
- **Phase 5b shipping** — `interpretation_events_table` and the
  `composition_states.provenance` enum extension are added by Phase 5b. Phase 9 cannot
  ship before Phase 5b is present in git history.
- Phase 6 (already shipped at RC5.2 commit 93c374d63) — defines `composer_completion_events_table` and `skill_markdown_history_table`. Phase 9 classifies both as `data`.
- **Operator decision: fixture-generation approach (historical-revision-checkout vs
  synthetic-from-MetaData)** — the three-generation fixture used in Tasks 7/8/9 can be
  produced either by checking out a historical git revision to obtain old schema
  binaries, or by constructing a synthetic old-schema MetaData programmatically. The
  approach determines Task 7 implementation shape.

**NOT blocked** by:

- Real-user deploys — per CLAUDE.md and the "WE HAVE NO USERS YET" memory, staging
  deploys remain unblocked under the current delete-the-DB policy. Phase 9 can be
  planned and implemented in parallel with the demo critical path; it only becomes
  *required* at the point of real-user deploy.
- Phase 4A shipping independently. Phase 9 handles the `tutorial_completed_at` column
  via the `preserve_add_columns` action; it does not hard-block on Phase 4A being present
  (an absent column is simply not migrated).
- Alembic dependency — Option (c) does not introduce Alembic. Option (b) would.
- **Postgres support** — explicitly out of scope. Caretaker refuses to start against non-SQLite engines (§3 step 4a). Future Postgres support reopens BLOCKERS (advisory-lock dialect strategy, per-dialect DDL, FK pragma listener semantics, transactional-DDL coverage) and must be planned as its own phase.

**Once remaining operator decisions are resolved:** Tasks 1-11 (§4) are TDD-shaped,
sequenced, and ready to execute under `superpowers:subagent-driven-development`. Estimated
commit count: 11 (one per task), each with a paired failing-then-passing test cycle. No
frontend work; Phase 9 is backend-only. The frontend never sees the migration runner directly.

**On ship:** update `project_db_migration_policy` memory to mark superseded; update
roadmap §B Phase 9 row from NOT STARTED to SHIPPED.

---

2026-05-16 — Cycle-2 fix-up applied: 16 of 21 findings from 4-reviewer panel addressed. 5 items deferred to operator: sub-phase split, activation site (create_app vs lifespan), cross-process advisory lock strategy, plus operator-confirmation on §H1 cumulative-loop closure and policy-supersession memory update. 2026-05-16 — Operator decision: Phase 9 scope cut to SQLite only. Postgres deferred. Resolved: advisory-lock dialect strategy (Task 7a `flock` sidecar), per-dialect DDL adapter (removed), FK pragma fight (SQLite-only). Activation site resolved to `create_app()` synchronous path. Remaining operator items: sub-phase split (9A/9B/9C); policy-supersession memory update on ship.

2026-05-19 — Cycle-3 revision: 8 blockers from the 4-reviewer panel applied (transaction-scope contradiction, raw_connection pool poisoning, schema-sentinel preservation, MigrationRecorder wiring, WebSettings fields + same-path validator, Phase 6 classification gap, refusal-path event types, test-caller list staleness). 10 warnings applied (trigger recreation in shadow-copy, tracer-bullet Task 4.5, fixture content-integrity CI gate, step-ordering test, --reason input sanitisation, mid-SIGKILL refusal-ordering runbook, Landscape DB failure runbook, bootstrap-ordering invariant, Task 0 ground-truth refresh, line-number corrections). Cycle-3 sign-off pending.
