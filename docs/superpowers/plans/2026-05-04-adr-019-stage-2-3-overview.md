# ADR-019 Stage 2/3 — Recorder + Producer Flip (Overview)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.
>
> **CRITICAL — atomic merge:** This plan is split into FIVE phase documents for review and execution organization. **The merge into `main` is atomic per ADR-019 lines 318-320: "the migration plan's Stage 2/3 (merged) PR ships the accumulator change in lockstep with the `RunResult.__post_init__` predicate rewrite — neither edit is safe in isolation."** Phases 1 and 2 are local checkpoints only, not git commit boundaries: each leaves known engine breakage that Phase 3 resolves. The first legal git commit combines Phases 1-3 and must leave the tree compiling with the Stage 2/3 accumulator and predicate flip in place. Phases 4 and 5 may then land as follow-up commits in the same PR. Do NOT push, commit, or propose landing Phase 1 alone, or Phases 1-2 alone. The PR opens with all five phases complete and tests green end-to-end.

**Goal:** Replace the single-axis `RowOutcome` audit recording with the two-axis `(TerminalOutcome, TerminalPath, completed)` triple across the recorder, every producer emit site, the accumulator, the resume aggregator, the four contract dataclasses, the telemetry event payload, and the `RunResult.__post_init__` predicate. Ships the discard-mode behaviour change (operator-visible `RunStatus` flip from `COMPLETED` to `COMPLETED_WITH_FAILURES`) and the two accumulator counter-increment changes (`(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)`) per ADR-019 § Counter derivation contract.

**Architecture:** Producers emit `(outcome, path)` pairs at every recording site; the recorder writes the triple to `token_outcomes`; the loader reconstructs the dataclass and runs the new (outcome, path) cross-checks plus four NEW cross-table invariants (I1a/I1b/I1c/I3); the accumulator matches on `(outcome, path)` and increments per the canonical mapping; the predicate becomes `success_indicator = rows_succeeded > 0` and `failure_indicator = rows_failed > 0` — the bifurcated OR clauses go away. Stage 1 introduced `TerminalOutcome`/`TerminalPath`/`_LEGAL_TERMINAL_PAIRS` alongside the unchanged `RowOutcome` (commit `60d30551` on `RC5-UX-RoutingVocabFix`); Stages 4 (test mechanical translation) and 5 (delete `RowOutcome`) follow this PR.

**Tech Stack:** Python 3.13, SQLAlchemy Core, pytest, mypy, ruff, pluggy. Audit DB is SQLite or Postgres (no Alembic — operator replaces the Landscape audit store, for example `audit.db`, between this PR and any pre-Stage-2 audit schema per ELSPETH's project DB migration policy recorded for this plan; ADR-019 does not require deleting `sessions.db`).

**Prerequisites:**
- Stage 1 commit `60d30551` is on the branch (introduces `TerminalOutcome`, `TerminalPath`, `_LEGAL_TERMINAL_PAIRS`, `_NON_TERMINAL_PATHS`, the closed-set partition assertion, the property test, and the `forbid_new_row_outcome.py` lint guard with allowlist).
- Allowlist at `config/cicd/forbid_new_row_outcome/migration_files.yaml` covers the existing RowOutcome migration-window files. It is not expected to cover every downstream schema consumer touched by Phase 1, because many of those files read `token_outcomes.is_terminal`/wire fields without importing `RowOutcome`; the new ADR-019 AST inventory created in Phase 1 is the broader closeout gate for those surfaces.
- ADR-019 at `docs/architecture/adr/019-two-axis-terminal-model.md` (HEAD `a5144c01` post-round-4 amendment) is the canonical spec; line references in the phase docs are against that revision.

---

## Phase index

| Phase | Document | Scope |
| --- | --- | --- |
| 1 | [Phase 1 — Schema + recorder + loader + contract dataclasses + downstream consumers](2026-05-04-adr-019-stage-2-3-phase-1-schema-recorder.md) | **First task creates the AST inventory tool/config/tests** so every later phase boundary can use it. DB schema rename + new column; `TokenOutcome`, `RowResult`, `PendingOutcome`, `TokenCompleted` retype; recorder `record_token_outcome` signature flip; loader `TokenOutcomeLoader.load` cross-check rewrite; `contracts/__init__.py` and `testing/__init__.py` re-exports; operator migration doc stub so stale-schema errors point at an existing file. **Plus 9 downstream-consumer fixes** in `mcp/analyzers/{reports,diagnostics}.py`, `mcp/types.py`, `web/execution/{diagnostics,discard_summary}.py`, `contracts/export_records.py`, `core/landscape/{exporter,lineage,formatters}.py` — schema rename + B3 silent-zero-quarantine fix. Without these, MCP diagnose() lies about quarantine count (Tier 1 violation) and Web run-diagnostics crashes. |
| 2 | [Phase 2 — Producer site flip](2026-05-04-adr-019-stage-2-3-phase-2-producers.md) | Every `RowOutcome.X` reference in `processor.py`, `transform.py`, `coalesce_executor.py`, `sink.py`, `recovery.py` flips to `(outcome, path)` pair construction at the emit site. ~120 src/ references with the canonical mapping table embedded for mechanical translation. |
| 3 | [Phase 3 — Accumulator + L0 predicate + L3 Pydantic mirror + resume aggregation + behaviour changes](2026-05-04-adr-019-stage-2-3-phase-3-accumulator-predicate.md) | `accumulate_row_outcomes` matches on `(outcome, path)` and ships the `(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)` counter changes; the live source-quarantine path in `Orchestrator._handle_quarantine_row` also bumps `rows_failed` so `rows_quarantined` remains a subset of the exhaustive failure counter; `RunResult.__post_init__` and `derive_terminal_run_status` drop their bifurcated OR clauses while retaining routed/quarantine counters as guard-only subset inputs; **`web/execution/schemas.py::_validate_row_decomposition` formula drops `rows_routed_*` from the sum (post-Phase-3 they're non-disjoint subsets) and `_check_status_row_count_invariant` mirrors the L0 simplification — without these, `/api/runs/{rid}` returns 500 for every gate-MOVE / on_error-routed run because Pydantic rejects the new counter shape**; `_derive_resume_terminal_status_from_audit` reads new columns. The discard-mode operator-visible `RunStatus` flip and the B4 `/api/runs/{rid}` 500-regression are each exercised by RED-first integration tests before the predicate change lands. |
| 4 | [Phase 4 — Cross-table invariants (I1a/I1b/I1c/I3)](2026-05-04-adr-019-stage-2-3-phase-4-cross-check-invariants.md) | Four NEW deferred / real-time invariants per ADR-019 § "Cross-check invariants." I1c (failsink-pair) and I3 (discard-no-failsink) are real-time at recording. I1a/I1b are deferred (children land later) — verified via end-of-run sweep wired into `Orchestrator._execute_run` and `_process_resumed_rows` immediately after `_flush_and_write_sinks(...)` returns, plus the public `resume()` no-work terminalization branch before audit-derived terminal finalization. This is intentionally after sink writes when a sink flush occurs; `_finalize_source_iteration` is too early because child sink `token_outcomes` rows are written during the sink phase. |
| 5 | [Phase 5 — Test strategy + triage](2026-05-04-adr-019-stage-2-3-phase-5-test-strategy.md) | Audit `tests/` into schema-dependent, assertion-only, and direct-DB-read categories after the focused Phase 3 gate. Phase 3 does not run the full `pytest tests/` suite because Phase 5 owns the remaining repo-wide schema-dependent test migration. Phase 5 is the first full-suite gate; it must not carry known-red tests forward to PR open. Phase 5 expands the Phase 1 operator migration stub into the full deployment/rollback runbook, reruns the AST-backed source inventory as the authoritative closeout gate, and adds remaining non-blocking behavioural/property coverage for cross-table invariants and migration closeout. |

---

## Sequencing within the PR

The five phase documents are execution checkpoints, not five buildable commits. The only allowed git commit boundaries are:

1. **Atomic Stage 2/3 commit:** Phases 1-3 together. This is the first point at which the engine must compile and the accumulator / predicate contract is coherent. It runs the focused Phase 1-3 contract, integration, import-smoke, AST-inventory, frontend, and policy gates; the full `pytest tests/` gate is deliberately deferred to Phase 5.
2. **Phase 4 commit:** Cross-table invariants, additive on top of the buildable Stage 2/3 state.
3. **Phase 5 commit:** Post-green test-strategy closeout, additional non-blocking behavioural tests, and full operator migration documentation expansion.

Phase 5's behavioural tests are written FIRST per TDD discipline — they fail until the corresponding phase commit lands.

```
Stage 1 (already shipped, commit 60d30551)
   │
   ├── Phase 1 local checkpoint: AST inventory + schema + dataclasses + recorder + loader + downstream consumers
   │       ├── ❌ producer sites still pass RowOutcome-shaped values to record_token_outcome — type-check/runtime call sites remain broken
   │       └── no git commit; continue directly to Phase 2
   │
   ├── Phase 2 local checkpoint: producer flip
   │       ├── ❌ runtime tests fail because accumulator still matches RowOutcome
   │       └── no git commit; continue directly to Phase 3
   │
  ├── Atomic Stage 2/3 commit: Phases 1-3 together
  │       ├── ✅ engine compiles; accumulator, resume aggregation, L0 predicate, and L3 predicate mirror are coherent
  │       ├── ✅ engine import/runtime smoke passes from the Phase 3 hard gate
  │       ├── ✅ AST inventory gate has run from the Phase 1 tool
  │       └── ⚠️ full `pytest tests/` deferred to Phase 5 test triage
   │
  ├── Phase 4 commit: cross-table invariants (NEW behaviour, additive)
  │       └── ✅ AST inventory gate rerun before commit
   │
 └── Phase 5 commit: post-green test closeout + new behavioural tests + runbook expansion
                      ✅ AST inventory + frontend/Python/full project gates rerun
                      ✅ no remaining broken ``outcome == RowOutcome.X`` assertions in tests/
```

The PR is opened only after Phase 5 lands. Reviewers see three clean commits: the atomic Phases 1-3 migration, Phase 4 invariants, and Phase 5 tests/docs. The squash-or-keep choice is the reviewer's at merge time.

---

## Operator-visible changes that ship with this PR

### 1. Discard-sink `RunStatus` flip (ADR-019 § Behavior Change Notice)

**Before:** A pipeline with `discard` mode sinks (`sink_name="__discard__"`) and no other failures completes with `RunStatus.COMPLETED`. The discard-mode `DIVERTED` `token_outcomes` rows are silently classified as non-predicate-input.

**After:** The same pipeline completes with `RunStatus.COMPLETED_WITH_FAILURES` (or `RunStatus.FAILED` if every row discards). Discard-mode `DIVERTED` is reclassified as `(FAILURE, SINK_DISCARDED)` and bumps `rows_failed`. The token-outcome layer now agrees with the node-state layer (`sink.py:991` already classified discard at `NodeStateStatus.FAILED`).

**Operator action required:** if a pipeline uses discard as silent housekeeping (rows intentionally dropped without affecting run status), reconfigure to route those rows to a no-op success sink instead. If the new semantics are acceptable, no action needed beyond re-baselining dashboards.

### 2. Counter changes (`(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)`)

**Before:** `RowOutcome.ROUTED` increments only `rows_routed_success`. `RowOutcome.ROUTED_ON_ERROR` increments only `rows_routed_failure`.

**After:** `(SUCCESS, GATE_ROUTED)` increments BOTH `rows_succeeded` AND `rows_routed_success`. `(FAILURE, ON_ERROR_ROUTED)` increments BOTH `rows_failed` AND `rows_routed_failure`. Source quarantine rows increment BOTH `rows_failed` and `rows_quarantined` at the live quarantine-routing site rather than relying on the row-processing accumulator.

**Why:** Without this, the `RunResult.__post_init__` predicate has to retain the bifurcated OR clauses (`rows_succeeded > 0 OR rows_routed_success > 0`). The accumulator change makes `success_indicator = rows_succeeded > 0` exhaustive by construction, which makes the predicate change safe.

**Operator visibility:** dashboards reading `rows_succeeded` will see higher numbers for runs with gate-MOVE routing; dashboards reading `rows_failed` will see higher numbers for runs with transform `on_error` routing. Public API field names are preserved (ADR-019 § Counter derivation contract — public API field names preserved).

### 3. Audit DB schema change

The `token_outcomes` table changes:
- `is_terminal` (Integer 0/1) renamed to `completed` (Integer 0/1) — same semantics, mirrors the rename from "lifecycle audit terminology" to "operator vocabulary" per ADR-019 sub-decision 3.
- `outcome` (String 32) changes value space from `RowOutcome.value` (non-NULL) to `TerminalOutcome.value | NULL` — NULL means non-terminal (`BUFFERED`).
- `path` (String 64) added — always populated, never NULL.

Per ELSPETH's project DB migration policy recorded for this plan, ELSPETH does not run Alembic on schema changes. **Operator action required at deploy time:** replace the old-schema Landscape audit store before deploying this PR, with service stop, immutable snapshot/backup, destructive delete/drop, restart, health check, and rollback steps documented in `docs/operator/migrations/adr-019.md` (stub added in Phase 1; full runbook expanded in Phase 5). ADR-019 does not change the web session schema, so do not delete `sessions.db` unless a separate compatibility check proves it stale and the operator runbook is explicitly amended with session backup/restore steps. Agents must not run destructive database commands without explicit operator approval for the target environment.

---

## Decision log

The following decisions were made during planning and are pinned here so they are not re-litigated during execution.

### D1: `path` is producer-carried, not recorder-derived

Per ADR-019 § "Classification is producer-declared, not topology-derivable" (lines 211-226). The recorder cannot distinguish `(FAILURE, ON_ERROR_ROUTED)` from `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` from topology — both produce a paired `NodeStateStatus.COMPLETED` `node_state` plus an `artifacts` row at a different node. Only the producer knows whether the lifecycle answer is FAILURE or TRANSIENT. Therefore `path: TerminalPath` is a field on `RowResult` and `PendingOutcome` and `TokenCompleted` and `TokenOutcome` — every dataclass that carries an outcome through the engine.

### D2: Schema migration is rename + add (not in-place rebuild)

`is_terminal` → `completed`: rename. Same column type (Integer 0/1), same semantics. SQLAlchemy `Column("completed", Integer, nullable=False)` replaces the old line. Loaders/recorders update field names.

`path` is a NEW column with `String(64), nullable=False`. Always populated by the recorder.

`outcome` column is REUSED with new value space: was `RowOutcome.value` (always non-NULL), becomes `TerminalOutcome.value` (NULL for non-terminal — only `BUFFERED` today). Column type unchanged.

Cross-check changes from `is_terminal == RowOutcome.is_terminal` (line `model_loaders.py:539`) to `completed XOR (outcome IS NULL)` (the canonical Tier 1 invariant under the new model).

The "delete the DB" policy means we do not write a migration; operators delete and recreate. New tables are created on engine startup via `metadata.create_all()`.

### D3: Deferred invariants (I1a/I1b) verified via end-of-run sweep, not periodic probe

Per ADR-019 § "Cross-check invariants" (lines 237-269), I1a (FORK_PARENT requires ≥1 child) and I1b (BATCH_CONSUMED requires the consuming batch row to reach `BatchStatus.COMPLETED` by end of run) are *deferred* obligations. The mechanism: extend `Orchestrator._execute_run` (`src/elspeth/engine/orchestrator/core.py:2972-3084`) and the sibling resume row-processing path `_process_resumed_rows` (`core.py:3420-3515`) to call `factory.data_flow.sweep_deferred_invariants_or_crash(run_id)` immediately after `_flush_and_write_sinks(...)` returns. Also extend the public `resume()` no-work terminalization branch (`not unprocessed_rows and not restored_state and restored_coalesce_state is None`, currently around `core.py:3247-3264`) to run the same sweep immediately before `_derive_resume_terminal_status_from_audit(...)` and `finalize_run(...)`; that branch bypasses `_process_resumed_rows`, so without an explicit sweep it can finalize stale orphaned audit rows successfully. In the fresh-run path this is before final progress / `PhaseCompleted`; in the resume row-processing path it is before returning to the public `resume()` wrapper for terminal finalization; in the no-work resume path it is before audit-derived terminal finalization. Do **not** place the sweep in `_finalize_source_iteration`: current source calls `_finalize_source_iteration` before `_flush_and_write_sinks`, while child sink outcomes are recorded during `SinkExecutor.write()`. A pre-sink sweep would reject valid fork/expand runs as orphaned.

The sweep is naturally skipped whenever `_flush_and_write_sinks` raises before returning. Graceful shutdown is the benign expected case: buffered/forked work is resumable and must not be declared orphaned before resume has a chance to complete it. If resume later flushes sinks successfully, the resume row-processing sweep runs then; if resume discovers there is no work left, the no-work branch sweep runs before terminal finalization. Other unhandled sink-flush exceptions also skip the post-sink sweep intentionally because the postconditions for I1a/I1b are not stable after a failed flush; the outer failure ceremony owns that run failure. The repository sweep queries for orphaned `TRANSIENT` parent tokens (FORK_PARENT with no children) and orphaned BATCH_CONSUMED tokens (whose consuming batch is not yet `BatchStatus.COMPLETED`) via `DataFlowRepository` helpers (`find_orphaned_transient_parents`, `find_orphaned_batch_consumptions`). Crash with `AuditIntegrityError` if any are found. Run-end after sink writes, or no-work resume terminalization after all rows were already processed, is the first moment the invariant CAN be verified. See Phase 4 for the concrete sweep code and exact insertion point. Note on I1b semantics: the result token created at flush time does NOT carry `batch_id` in its own `token_outcomes` row (only CONSUMED_IN_BATCH and BUFFERED outcomes set `batch_id`), so the reachability path for the batch's lifecycle answer goes through `batches.status`, not a paired `token_outcomes` row. ADR-019 has been amended to make this batch-status witness canonical.

### D4: I1c (sink-fallback-paired) and I3 (discard-FAILURE) verified at recording time

Both are real-time verifiable per the ADR. I1c checks the failsink node_state + artifacts row exist at `record_token_outcome` call time. I3 checks `sink_name == "__discard__"` AND no failsink node_state exists for the same token at recording. Both are added to the existing `_validate_outcome_fields` block in `data_flow_repository.py` which already runs at write time. See Phase 4 for the concrete checks.

### D5: Operator release note location

The Behavior Change Notice is documented in three places:
1. **PR description** — top-level summary for reviewers.
2. **`docs/operator/migrations/adr-019.md`** (stub added in Phase 1; full runbook expanded in Phase 5) — durable operator reference: what changed, what action to take, how to identify affected pipelines, the DB-delete command.
3. **No CHANGELOG entry** — ELSPETH has no CHANGELOG today; commits + ADRs are the release record for this project.

### D7: Downstream-consumer sweep is mandatory at every phase boundary

**Discovered 2026-05-05 across two patch rounds:** The original Phase 1 + Phase 3 file lists missed 9 downstream consumers across THREE distinct bug classes. Each class is invisible to the Stage 1 lint guard because none of the offending sites import `RowOutcome`.

- **Bug class A: schema-column reads / wire-contract fields** (e.g. `token_outcomes_table.c.is_terminal` in raw SQL, `OutcomeDistributionEntry.is_terminal: bool`, `TokenOutcomeExportRecord.is_terminal: bool`) — would crash at deploy with `AttributeError` on the renamed column or fail mypy/wire-contract checks after the emitted payload switches to `completed` + `path`. Sites: `mcp/analyzers/reports.py`, `mcp/types.py`, `web/execution/diagnostics.py`, `web/execution/discard_summary.py`, `contracts/export_records.py`, `core/landscape/exporter.py`, `core/landscape/lineage.py`, `core/landscape/formatters.py`. Patched in Phase 1 Task 1.9.
- **Bug class B: hardcoded `RowOutcome.value` strings as SQL filters** (e.g. `outcome == "quarantined"` in `diagnostics.py:181`) — would silently match zero rows after the value-space change, returning confidently-wrong results to operators. Per CLAUDE.md "I don't know what happened" + Tier 1 audit-integrity, silent-wrong is the worst-class failure. Patched in Phase 1 Task 1.9.
- **Bug class C: Pydantic / L3 wire-schema mirrors of L0 predicates** (e.g. `web/execution/schemas.py::_validate_row_decomposition` copies the bifurcated sum-disjoint formula; `_check_status_row_count_invariant` copies the bifurcated OR predicate). When Phase 3's accumulator changes the *semantic* of which counters get bumped (without changing field names), the L3 mirror's formula breaks → HTTP 500 on the API. Distinct from class A because the column names ARE the same; the bug is at the predicate-formula layer. Patched in Phase 3 Task 3.5.
- **Bug class D: prose-driven hallucination — wrong import paths and wrong attribute references in plan-snippet code** (e.g. `from elspeth.contracts.audit import AuditIntegrityError` when the exception lives in `contracts.errors`; `self._counters.rows_failed += 1` when `SinkExecutor` has no `_counters` attribute and counter accumulation lives at the orchestrator level). These are NOT codebase bugs — they're plan-snippet bugs that would crash AT EXECUTION TIME if the plan is followed literally. The pattern: ADR / planning prose names a concept ("the SinkExecutor counter site") that doesn't map to a Python attribute, or imports a class from the file it's "about" rather than from the file that exports it. Patched in Phase 3 Task 3.3 Step 3 (counter relocation) and Phase 4 test-template imports. The mandatory check is: every code snippet in a plan that names a Python symbol must have that symbol grep-verified against current HEAD before the plan ships.

- **Bug class E: undefined / non-existent test-helper functions cited as if they exist** (e.g. Phase 3 referenced `build_test_pipeline_with_discard_sink`, `build_test_pipeline_with_gate_route`, `build_test_pipeline_with_on_error_route`, `run_pipeline` across three different import paths — none of which existed in the codebase). The plan-snippet cited the helpers from inconsistent locations (`elspeth.testing` vs `tests.integration._helpers` vs `tests.conftest`) and never specified which was canonical. **Plus:** the plan failed to mandate `ExecutionGraph.from_plugin_instances` + `instantiate_plugins_from_config` per CLAUDE.md "Never bypass production code paths in tests." Patched in Phase 3 Task 3.0 (a new prerequisite task that creates the canonical helpers ONCE at `tests/integration/_helpers.py`, mandates the production code path, and verifies plugin names against the registered set). The mandatory check is: any helper function cited by a plan code-block must either (a) exist in the codebase already (verify with `grep -rn "def <name>" src/ tests/`), or (b) have an explicit creation task in the plan with a single canonical location.

- **Bug class F: missing audit-trail durability contract on Tier 1 invariants** (e.g. Phase 4 introduced `AuditIntegrityError`-raising real-time and deferred checks but didn't specify run-finalization ordering, evidence preservation, or a durability regression test). Per CLAUDE.md Auditability Standard, every Tier 1 crash must (1) durably finalize the run as `RunStatus.FAILED`, (2) preserve queryable evidence rows that triggered the crash, and (3) be exercisable by a regression test. ADR-019 does not add a persisted run-error-message field; the queryable evidence is the preserved offending `token_outcomes`/batch/node rows plus the FAILED run status, while the exception message is re-raised to the caller. Patched in Phase 4 Task 4.4 (durability contract + regression matrix for fresh/resume/no-op-resume I1a/I1b, I1c/I3 real-time producer crashes, and shutdown-skip). The mandatory check is: every new exception-raising Tier 1 invariant must specify the run-finalization ordering, evidence-preservation policy, and have a regression test before the plan ships.

**Mandatory check before EVERY phase commit, not just Phase 1:**

Use the AST inventory added in Phase 1 Task 1.0 as the authoritative closeout gate:

```bash
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check \
  --root src/elspeth \
  --allowlist config/cicd/adr019_symbol_inventory
```

The older grep snippets below are local triage aids only. They are not
sufficient approval gates because they cannot see every syntactic form
(`TypedDict` declarations, keyword arguments, dict literal keys, and some
function-call shapes).

```bash
# Class A — schema-column residual reads:
grep -rn "token_outcomes_table.c.is_terminal" src/elspeth/
grep -rn "\\.is_terminal" src/elspeth/ | grep -v "/web/execution/progress.py" | grep -v "/contracts/"

# Class B — hardcoded RowOutcome.value SQL filters:
grep -rnE 'outcome\s*==\s*"(completed|routed|routed_on_error|forked|failed|quarantined|diverted|consumed_in_batch|dropped_by_filter|coalesced|expanded|buffered)"' src/elspeth/

# Class B (extended) — hardcoded RowOutcome.value string literals outside contracts/enums.py:
grep -rnE '"(completed|routed_on_error|consumed_in_batch|dropped_by_filter)"' src/elspeth/ \
    | grep -v "/contracts/enums.py" | grep -v test_

# Class C — Pydantic / wire-schema predicate mirrors:
grep -rn "rows_routed_success\|rows_routed_failure\|success_indicator\|failure_indicator" src/elspeth/ \
    | grep -v "test_\|/__pycache__\|/contracts/run_result.py\|/engine/orchestrator/" \
    | grep -v "/cli_formatters.py"  # display-only, no enforcement
# Inspect any hit outside the migration-window allowlist for predicate-formula or sum-decomposition
# logic. The web/execution/schemas.py site is the known one — additional hits mean a missed mirror.

# Class C (extended) — additional sum invariants over rows_processed:
grep -rnE 'rows_processed\s*[<>=]+\s*' src/elspeth/ | grep -v test_ | grep -v "/__pycache__/"

# Class D — plan-snippet hallucination check (run AGAINST the plan files,
# not src/). Every Python symbol used in a plan code-block must exist in
# the current source tree. Manual: for each ``import`` line and every
# ``self.X`` attribute reference in plan snippets, grep the source tree
# to confirm the symbol exists at the named path.
#
# Common patterns to verify before shipping plan edits:
#   - AuditIntegrityError is in contracts.errors (NOT contracts.audit).
#   - OrchestrationInvariantError is in contracts.errors (NOT in
#     contracts/results.py despite being raised from RowResult.__post_init__).
#   - SinkExecutor has no _counters attribute — orchestrator owns counters.
#   - End-of-source hook is _finalize_source_iteration, NOT
#     _post_source_iteration_work (the latter exists only in ADR prose).
#     Deferred I1a/I1b sweep is later still: post-sink in _execute_run /
#     _process_resumed_rows, plus the resume no-work terminalization branch.
#
# Adopt the discipline: for every named symbol in a plan snippet, run
# ``grep -rn "^class <Symbol>\|def <Symbol>" src/elspeth/`` to confirm
# the symbol exists at the path the plan claims, BEFORE the plan ships.
```

A non-zero result from the AST inventory after Phases 2-4 lands means a
downstream consumer was missed (classes A-C) or a plan-snippet bug was missed
(class D). STOP and patch the missing consumer in the same commit; do not let
the gap leak across the phase boundary. The pattern is recurring — string
literals, column names, and predicate-formula copies hide in places that don't
import `RowOutcome`, so the Stage 1 lint guard and grep triage do not catch them.

### D8: Closed-set contracts require property and boundary tests

ADR-019 introduces closed sets (`TerminalOutcome`, `TerminalPath`,
`_LEGAL_TERMINAL_PAIRS`, terminal run statuses, and the counter/predicate
truth table). Any new closed-set contract in this plan must include:

- A Hypothesis property sweep over the closed-set Cartesian product, asserting
  every illegal pair/shape is rejected or every legal pair/shape is accepted.
- Boundary examples for zero-cardinality, all-one-side, and mixed-success /
  mixed-failure cases when a predicate is simplified.
- A RED-first note naming which examples fail before the production change.

The concrete ADR-019 anchors are Phase 1 Task 1.2
(`test_all_illegal_completed_pairs_rejected`) and Phase 3 Task 3.2a
(`TestADR019PredicateBoundaryCases`). Treat those as the minimum pattern for
future closed-set additions; one hand-picked illegal example is not enough.

### D6: Test migration split (all pytest-blocking assertions move before the first commit)

Phase 5 owns the repo-wide `pytest tests/ -q` triage and is the first full-suite
gate. Phases 1-3 still must update every schema-dependent test that is in their
focused gate or in a touched module, but the atomic Stage 2/3 commit is not
required to make unrelated stale tests green. Short summary:
- **Tests that construct dataclass instances (`TokenOutcome(outcome=RowOutcome.X, ...)`)** — schema-dependent. Won't compile after Phase 1. Move with Stage 2/3.
- **Tests that assert on `result.outcome == RowOutcome.X` from a real engine execution** — assertion-only but pytest-blocking after Phase 1. These MUST move in this PR, translated mechanically against `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`. Cross-enum equality returns `False`, not `TypeError`, so leaving these assertions for Stage 4 would make the full-suite gate impossible.
- **Integration tests that exercise end-to-end engine execution and observe outputs** — schema-dependent if they read `token_outcomes` directly; assertion-only if they only check `RunResult` counters. Both pytest-blocking forms move in this PR.

The Phase 5 grep recipe distinguishes the categories deterministically so executing engineers don't have to invent it. Stage 4 remains useful only for non-blocking cleanup and eventual `RowOutcome` deletion prep; it is not allowed to carry failing `outcome == RowOutcome.X` assertions from this PR.

---

## Verification gates (run before opening PR)

After Phase 5 commit lands, all of these must pass before the PR opens:

```bash
# Unit + integration tests
.venv/bin/python -m pytest tests/ -q --timeout=120

# Type check across the engine
.venv/bin/python -m mypy src/elspeth

# Lint + format
.venv/bin/python -m ruff check src/ tests/ scripts/
.venv/bin/python -m ruff format --check src/ tests/ scripts/

# Project gates
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model --exclude "**/__pycache__/*"
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth
.venv/bin/python scripts/cicd/enforce_contract_manifest.py check --allowlist config/cicd/enforce_contract_manifest
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
.venv/bin/python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth --allowlist config/cicd/enforce_frozen_annotations
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check --root src/elspeth --allowlist config/cicd/adr019_symbol_inventory

# Stage 1's lint guard MUST still pass. The src/ side should drop FROM 134
# references to 0 by end of Phase 4 (TerminalOutcome / TerminalPath fully
# replaces RowOutcome in src/). Phase 5 also fixes assertion-only tests that
# would otherwise fail under cross-enum equality.
# FNR1 (RowOutcome.X attribute access) + FNR2 (hardcoded RowOutcome value-string
# comparisons, e.g. outcome == "quarantined") — Phase 5 Task 5.7 extends the script.
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome

# Frontend gates for the Phase 3 terminal-status UI fix
cd src/elspeth/web/frontend
npm run test
npm run build
cd /home/john/elspeth
```

After this PR merges, the migration files allowlist for `src/elspeth/` paths is now empty — every src/ migration site has flipped. Phase 5 also removes every pytest-blocking `outcome == RowOutcome.X` assertion. Stage 5 deletes the script entirely after `RowOutcome` itself is removed. Until Stage 5, the script + allowlist remain to guard against new `RowOutcome.X` introductions.

### Behavioural verification (NEW for this PR)

Per Phase 5, the new ADR-019 integration-test fixtures must exist and be green:

```bash
# Discard-mode RunStatus flip (the operator-visible change)
.venv/bin/python -m pytest tests/integration/test_adr_019_discard_mode_flip.py -v

# Cross-table invariants I1c (failsink-paired) and I3 (discard-no-failsink)
.venv/bin/python -m pytest tests/integration/test_adr_019_cross_table_invariants.py -v

# Counter changes for (SUCCESS, GATE_ROUTED) and (FAILURE, ON_ERROR_ROUTED)
.venv/bin/python -m pytest tests/integration/test_adr_019_counter_changes.py -v

# Sweep-crash durability and graceful-shutdown gate
.venv/bin/python -m pytest tests/integration/test_adr_019_sweep_durability.py -v
```

These tests are described in Phase 5; their existence is gated by a Phase 3 Definition of Done ("Behavioural test for discard-mode flip is RED before the predicate rewrite, GREEN after").

---

## Out of scope for this PR

- **`RowOutcome` enum deletion** — Stage 5 ticket `elspeth-774b1d3c2e`. RowOutcome continues to exist alongside `TerminalOutcome` until every assertion site has flipped. Deleting it is the final sweep.

- **The Stage 1 lint guard `forbid_new_row_outcome.py` and its allowlist** — Stage 5. The guard prevents drift during Stages 4 and earlier; once Stage 5 deletes `RowOutcome` itself, the guard becomes vacuous and is removed.

- **Counter rename** — separate ADR-020 conversation. Public API field names (`rows_succeeded`, `rows_routed_success`, etc.) are preserved in this PR per ADR-019 § Counter derivation contract.

- **Frontend counter-type redesign** — `web/frontend/src/types/index.ts` keeps the existing counter field names per ADR-019 § Counter derivation contract. The React layer is still in scope for the Phase 3 terminal-status fix: `SessionSidebar.tsx` must import the existing `TERMINAL_RUN_STATUS_VALUES` taxonomy and the frontend test/build gates must pass.

---

## How to use this plan

1. Read this overview.
2. Read each phase document in order.
3. Execute phase-by-phase using `superpowers:executing-plans`. Each phase has its own RED→GREEN test cycle and Definition of Done.
4. After Phase 5's Definition of Done is met, run all verification gates above. If all green, open the PR.
5. The PR description summarizes the change, links to ADR-019, and includes the operator deletion command from `docs/operator/migrations/adr-019.md`.

If a phase reveals a gap the ADR doesn't cover, STOP and re-read this overview's decision log. If the gap is genuinely new, surface to the user before silently expanding scope — same discipline as Stage 1 used for the allowlist deviation.
