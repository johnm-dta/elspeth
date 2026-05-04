# ADR-019 Stage 2/3 — Recorder + Producer Flip (Overview)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.
>
> **CRITICAL — atomic merge:** This plan is split into FIVE phase documents for review and execution organization. **The merge into `main` is atomic per ADR-019 lines 318-320: "the migration plan's Stage 2/3 (merged) PR ships the accumulator change in lockstep with the `RunResult.__post_init__` predicate rewrite — neither edit is safe in isolation."** Phases are sequenced commits within ONE PR; intermediate phases left on `main` would break the engine. Do NOT propose to land Phase 1 alone, or Phases 1-2 alone, etc. The PR opens with all five phases complete and tests green end-to-end.

**Goal:** Replace the single-axis `RowOutcome` audit recording with the two-axis `(TerminalOutcome, TerminalPath, completed)` triple across the recorder, every producer emit site, the accumulator, the resume aggregator, the four contract dataclasses, the telemetry event payload, and the `RunResult.__post_init__` predicate. Ships the discard-mode behaviour change (operator-visible `RunStatus` flip from `COMPLETED` to `COMPLETED_WITH_FAILURES`) and the two accumulator counter-increment changes (`(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)`) per ADR-019 § Counter derivation contract.

**Architecture:** Producers emit `(outcome, path)` pairs at every recording site; the recorder writes the triple to `token_outcomes`; the loader reconstructs the dataclass and runs the new (outcome, path) cross-checks plus four NEW cross-table invariants (I1a/I1b/I1c/I3); the accumulator matches on `(outcome, path)` and increments per the canonical mapping; the predicate becomes `success_indicator = rows_succeeded > 0` and `failure_indicator = rows_failed > 0` — the bifurcated OR clauses go away. Stage 1 introduced `TerminalOutcome`/`TerminalPath`/`_LEGAL_TERMINAL_PAIRS` alongside the unchanged `RowOutcome` (commit `60d30551` on `RC5-UX-RoutingVocabFix`); Stages 4 (test mechanical translation) and 5 (delete `RowOutcome`) follow this PR.

**Tech Stack:** Python 3.13, SQLAlchemy Core, pytest, mypy, ruff, pluggy. Audit DB is SQLite or Postgres (no Alembic — operator deletes `audit.db` + `sessions.db` between this PR and any pre-Stage-2 state per `MEMORY.md::project_db_migration_policy`).

**Prerequisites:**
- Stage 1 commit `60d30551` is on the branch (introduces `TerminalOutcome`, `TerminalPath`, `_LEGAL_TERMINAL_PAIRS`, `_NON_TERMINAL_PATHS`, the closed-set partition assertion, the property test, and the `forbid_new_row_outcome.py` lint guard with allowlist).
- Allowlist at `config/cicd/forbid_new_row_outcome/migration_files.yaml` already covers every file this plan touches.
- ADR-019 at `docs/architecture/adr/019-two-axis-terminal-model.md` (HEAD `a5144c01` post-round-4 amendment) is the canonical spec; line references in the phase docs are against that revision.

---

## Phase index

| Phase | Document | Scope |
| --- | --- | --- |
| 1 | [Phase 1 — Schema + recorder + loader + contract dataclasses + downstream consumers](2026-05-04-adr-019-stage-2-3-phase-1-schema-recorder.md) | DB schema rename + new column; `TokenOutcome`, `RowResult`, `PendingOutcome`, `TokenCompleted` retype; recorder `record_token_outcome` signature flip; loader `TokenOutcomeLoader.load` cross-check rewrite; `testing/__init__.py` re-exports. **Plus 8 downstream-consumer fixes** in `mcp/analyzers/{reports,diagnostics}.py`, `mcp/types.py`, `web/execution/{diagnostics,discard_summary}.py`, `core/landscape/{exporter,lineage,formatters}.py` — schema rename + B3 silent-zero-quarantine fix. Without these, MCP diagnose() lies about quarantine count (Tier 1 violation) and Web run-diagnostics crashes. |
| 2 | [Phase 2 — Producer site flip](2026-05-04-adr-019-stage-2-3-phase-2-producers.md) | Every `RowOutcome.X` reference in `processor.py`, `transform.py`, `coalesce_executor.py`, `sink.py`, `recovery.py` flips to `(outcome, path)` pair construction at the emit site. ~120 src/ references with the canonical mapping table embedded for mechanical translation. |
| 3 | [Phase 3 — Accumulator + L0 predicate + L3 Pydantic mirror + resume aggregation + behaviour changes](2026-05-04-adr-019-stage-2-3-phase-3-accumulator-predicate.md) | `accumulate_row_outcomes` matches on `(outcome, path)` and ships the `(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)` counter changes; `RunResult.__post_init__` and `derive_terminal_run_status` drop their bifurcated OR clauses; **`web/execution/schemas.py::_validate_row_decomposition` formula drops `rows_routed_*` from the sum (post-Phase-3 they're non-disjoint subsets) and `_check_status_row_count_invariant` mirrors the L0 simplification — without these, `/api/runs/{rid}` returns 500 for every gate-MOVE / on_error-routed run because Pydantic rejects the new counter shape**; `_derive_resume_terminal_status_from_audit` reads new columns. The discard-mode operator-visible `RunStatus` flip and the B4 `/api/runs/{rid}` 500-regression are each exercised by RED-first integration tests before the predicate change lands. |
| 4 | [Phase 4 — Cross-table invariants (I1a/I1b/I1c/I3)](2026-05-04-adr-019-stage-2-3-phase-4-cross-check-invariants.md) | Four NEW deferred / real-time invariants per ADR-019 § "Cross-check invariants." I1c (failsink-pair) and I3 (discard-no-failsink) are real-time at recording. I1a/I1b are deferred (children land later) — verified via end-of-run sweep wired into `Orchestrator._finalize_source_iteration` at `core.py:2511`, gated on `not interrupted_by_shutdown`. |
| 5 | [Phase 5 — Test strategy + triage](2026-05-04-adr-019-stage-2-3-phase-5-test-strategy.md) | Triage `tests/` into schema-dependent (must move with Stage 2/3, won't compile otherwise) vs assertion-only (deferred to Stage 4). Grep recipe + per-file expected categories. New behavioural tests for: cross-table invariants, discard-mode `RunStatus` flip, accumulator counter changes, predicate drop-OR-clause. |

---

## Sequencing within the PR

The five phases sequence as five commits in order; every commit boundary leaves the tree compiling and `pytest tests/unit/contracts/` + `pytest tests/unit/core/landscape/` green. Phase 5's behavioural tests are written FIRST per TDD discipline — they fail until the corresponding phase commit lands.

```
Stage 1 (already shipped, commit 60d30551)
   │
   ├── Phase 1 commit: schema + dataclasses + recorder + loader (engine compiles, producers broken)
   │       ├── ❌ producer sites still pass RowOutcome to record_token_outcome — engine module won't import
   │       └── this is why Phase 1 alone cannot land — Phase 2 must follow in the same PR
   │
   ├── Phase 2 commit: producer flip (engine module imports; runtime tests fail because accumulator still on RowOutcome)
   │
   ├── Phase 3 commit: accumulator + predicate + resume + behaviour changes (engine green end-to-end at unit level)
   │
   ├── Phase 4 commit: cross-table invariants (NEW behaviour, additive)
   │
   └── Phase 5 commit: schema-dependent test fixes + new behavioural tests
                        (Stage 4 leaves the ~143 ``outcome == RowOutcome.X`` assertion sites alone)
```

The PR is opened only after Phase 5 lands. Reviewers see five clean commits; the squash-or-keep choice is the reviewer's at merge time.

---

## Operator-visible changes that ship with this PR

### 1. Discard-sink `RunStatus` flip (ADR-019 § Behavior Change Notice)

**Before:** A pipeline with `discard` mode sinks (`sink_name="__discard__"`) and no other failures completes with `RunStatus.COMPLETED`. The discard-mode `DIVERTED` `token_outcomes` rows are silently classified as non-predicate-input.

**After:** The same pipeline completes with `RunStatus.COMPLETED_WITH_FAILURES` (or `RunStatus.FAILED` if every row discards). Discard-mode `DIVERTED` is reclassified as `(FAILURE, SINK_DISCARDED)` and bumps `rows_failed`. The token-outcome layer now agrees with the node-state layer (`sink.py:991` already classified discard at `NodeStateStatus.FAILED`).

**Operator action required:** if a pipeline uses discard as silent housekeeping (rows intentionally dropped without affecting run status), reconfigure to route those rows to a no-op success sink instead. If the new semantics are acceptable, no action needed beyond re-baselining dashboards.

### 2. Counter changes (`(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)`)

**Before:** `RowOutcome.ROUTED` increments only `rows_routed_success`. `RowOutcome.ROUTED_ON_ERROR` increments only `rows_routed_failure`.

**After:** `(SUCCESS, GATE_ROUTED)` increments BOTH `rows_succeeded` AND `rows_routed_success`. `(FAILURE, ON_ERROR_ROUTED)` increments BOTH `rows_failed` AND `rows_routed_failure`.

**Why:** Without this, the `RunResult.__post_init__` predicate has to retain the bifurcated OR clauses (`rows_succeeded > 0 OR rows_routed_success > 0`). The accumulator change makes `success_indicator = rows_succeeded > 0` exhaustive by construction, which makes the predicate change safe.

**Operator visibility:** dashboards reading `rows_succeeded` will see higher numbers for runs with gate-MOVE routing; dashboards reading `rows_failed` will see higher numbers for runs with transform `on_error` routing. Public API field names are preserved (ADR-019 § Counter derivation contract — public API field names preserved).

### 3. Audit DB schema change

The `token_outcomes` table changes:
- `is_terminal` (Integer 0/1) renamed to `completed` (Integer 0/1) — same semantics, mirrors the rename from "lifecycle audit terminology" to "operator vocabulary" per ADR-019 sub-decision 3.
- `outcome` (String 32) changes value space from `RowOutcome.value` (non-NULL) to `TerminalOutcome.value | NULL` — NULL means non-terminal (`BUFFERED`).
- `path` (String 64) added — always populated, never NULL.

Per `MEMORY.md::project_db_migration_policy`, ELSPETH does not run Alembic on schema changes. **Operator action required at deploy time:** delete `audit.db` and `sessions.db` from any production / staging environment before deploying this PR. The PR description and `docs/operator/migrations/adr-019.md` (added in Phase 5) document the required deletion command.

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

Per ADR-019 § "Cross-check invariants" (lines 237-269), I1a (FORK_PARENT requires ≥1 child) and I1b (BATCH_CONSUMED requires batch-result token at flush) are *deferred* obligations. The mechanism: extend `Orchestrator._finalize_source_iteration` (`src/elspeth/engine/orchestrator/core.py:2511`) — gated on `not interrupted_by_shutdown`, mirroring the existing aggregation-flush and coalesce-flush gates — to query for orphaned `TRANSIENT` parent tokens (no children, no batch-result) via two new helper methods on `DataFlowRepository` (`find_orphaned_transient_parents`, `find_orphaned_batch_consumptions`). Crash with `AuditIntegrityError` if any are found. Run-end is the first moment the invariant CAN be verified — earlier verification would race with child completion. The shutdown gate is essential: graceful shutdown legitimately leaves fork-parents without children because resume completes them. See Phase 4 for the concrete sweep code and exact insertion point.

### D4: I1c (sink-fallback-paired) and I3 (discard-FAILURE) verified at recording time

Both are real-time verifiable per the ADR. I1c checks the failsink node_state + artifacts row exist at `record_token_outcome` call time. I3 checks `sink_name == "__discard__"` AND no failsink node_state exists for the same token at recording. Both are added to the existing `_validate_outcome_fields` block in `data_flow_repository.py` which already runs at write time. See Phase 4 for the concrete checks.

### D5: Operator release note location

The Behavior Change Notice is documented in three places:
1. **PR description** — top-level summary for reviewers.
2. **`docs/operator/migrations/adr-019.md`** (NEW file added in Phase 5) — durable operator reference: what changed, what action to take, how to identify affected pipelines, the DB-delete command.
3. **No CHANGELOG entry** — ELSPETH has no CHANGELOG today (`MEMORY.md::feedback_no_calendar_shipping_commitments` codifies that release ceremony is not the project's idiom; commits + ADRs ARE the release record).

### D7: Downstream-consumer sweep is mandatory at every phase boundary

**Discovered 2026-05-05 across two patch rounds:** The original Phase 1 + Phase 3 file lists missed 9 downstream consumers across THREE distinct bug classes. Each class is invisible to the Stage 1 lint guard because none of the offending sites import `RowOutcome`.

- **Bug class A: schema-column reads** (e.g. `token_outcomes_table.c.is_terminal` in raw SQL, `OutcomeDistributionEntry.is_terminal: bool` in Pydantic TypedDict) — would crash at deploy with `AttributeError` on the renamed column. Sites: `mcp/analyzers/reports.py`, `mcp/types.py`, `web/execution/diagnostics.py`, `web/execution/discard_summary.py`, `core/landscape/exporter.py`, `core/landscape/lineage.py`, `core/landscape/formatters.py`. Patched in Phase 1 Task 1.9.
- **Bug class B: hardcoded `RowOutcome.value` strings as SQL filters** (e.g. `outcome == "quarantined"` in `diagnostics.py:181`) — would silently match zero rows after the value-space change, returning confidently-wrong results to operators. Per CLAUDE.md "I don't know what happened" + Tier 1 audit-integrity, silent-wrong is the worst-class failure. Patched in Phase 1 Task 1.9.
- **Bug class C: Pydantic / L3 wire-schema mirrors of L0 predicates** (e.g. `web/execution/schemas.py::_validate_row_decomposition` copies the bifurcated sum-disjoint formula; `_check_status_row_count_invariant` copies the bifurcated OR predicate). When Phase 3's accumulator changes the *semantic* of which counters get bumped (without changing field names), the L3 mirror's formula breaks → HTTP 500 on the API. Distinct from class A because the column names ARE the same; the bug is at the predicate-formula layer. Patched in Phase 3 Task 3.5.
- **Bug class D: prose-driven hallucination — wrong import paths and wrong attribute references in plan-snippet code** (e.g. `from elspeth.contracts.audit import AuditIntegrityError` when the exception lives in `contracts.errors`; `self._counters.rows_failed += 1` when `SinkExecutor` has no `_counters` attribute and counter accumulation lives at the orchestrator level). These are NOT codebase bugs — they're plan-snippet bugs that would crash AT EXECUTION TIME if the plan is followed literally. The pattern: ADR / planning prose names a concept ("the SinkExecutor counter site") that doesn't map to a Python attribute, or imports a class from the file it's "about" rather than from the file that exports it. Patched in Phase 3 Task 3.3 Step 3 (counter relocation) and Phase 4 test-template imports. The mandatory check is: every code snippet in a plan that names a Python symbol must have that symbol grep-verified against current HEAD before the plan ships.

- **Bug class E: undefined / non-existent test-helper functions cited as if they exist** (e.g. Phase 3 referenced `build_test_pipeline_with_discard_sink`, `build_test_pipeline_with_gate_route`, `build_test_pipeline_with_on_error_route`, `run_pipeline` across three different import paths — none of which existed in the codebase). The plan-snippet cited the helpers from inconsistent locations (`elspeth.testing` vs `tests.integration._helpers` vs `tests.conftest`) and never specified which was canonical. **Plus:** the plan failed to mandate `ExecutionGraph.from_plugin_instances` + `instantiate_plugins_from_config` per CLAUDE.md "Never bypass production code paths in tests." Patched in Phase 3 Task 3.0 (a new prerequisite task that creates the canonical helpers ONCE at `tests/integration/_helpers.py`, mandates the production code path, and verifies plugin names against the registered set). The mandatory check is: any helper function cited by a plan code-block must either (a) exist in the codebase already (verify with `grep -rn "def <name>" src/ tests/`), or (b) have an explicit creation task in the plan with a single canonical location.

- **Bug class F: missing audit-trail durability contract on Tier 1 invariants** (e.g. Phase 4 introduced an `AuditIntegrityError`-raising sweep but didn't specify run-finalization ordering, evidence preservation, or a durability regression test). Per CLAUDE.md Auditability Standard, every Tier 1 crash must (1) durably finalize the run as `RunStatus.FAILED` with a queryable error message, (2) preserve the evidence rows that triggered the crash, and (3) be exercisable by a regression test. Without these, an operator querying a sweep-crashed run gets an exception traceback but no audit trail explaining why. Patched in Phase 4 Task 4.4 (durability contract + regression test for I1a-orphan, I1b-orphan, and shutdown-skip cases). The mandatory check is: every new exception-raising Tier 1 invariant must specify the run-finalization ordering, evidence-preservation policy, and have a regression test before the plan ships.

**Mandatory check before EVERY phase commit, not just Phase 1:**

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
#   - Hook names in core.py are _finalize_source_iteration, NOT
#     _post_source_iteration_work (the latter exists only in ADR prose).
#
# Adopt the discipline: for every named symbol in a plan snippet, run
# ``grep -rn "^class <Symbol>\|def <Symbol>" src/elspeth/`` to confirm
# the symbol exists at the path the plan claims, BEFORE the plan ships.
```

A non-zero result from any of these greps after Phases 2-4 lands means a downstream consumer was missed (classes A-C) or a plan-snippet bug was missed (class D). STOP and patch the missing consumer in the same commit; do not let the gap leak across the phase boundary. The pattern is recurring — string literals, column names, and predicate-formula copies hide in places that don't import `RowOutcome`, so the Stage 1 lint guard does not catch them.

### D6: Test deferral split (Stage 2/3 vs Stage 4)

Phase 5 establishes the triage. Short summary:
- **Tests that construct dataclass instances (`TokenOutcome(outcome=RowOutcome.X, ...)`)** — schema-dependent. Won't compile after Phase 1. Move with Stage 2/3.
- **Tests that assert on `result.outcome == RowOutcome.X` from a real engine execution** — assertion-only. Defer to Stage 4. Stage 4's mechanical translation flips them.
- **Integration tests that exercise end-to-end engine execution and observe outputs** — schema-dependent if they read `token_outcomes` directly; assertion-only if they only check `RunResult` counters.

The Phase 5 grep recipe distinguishes the two categories deterministically so executing engineers don't have to invent it.

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

# Stage 1's lint guard MUST still pass — the migration files are still allowlisted, but
# the producer flip in Phase 2 may add new RowOutcome.X references in tests/ that
# Stage 4 will handle. The src/ side should drop FROM 134 references to 0 by end of
# Phase 4 (TerminalOutcome / TerminalPath fully replaces RowOutcome in src/).
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome
```

After this PR merges, the migration files allowlist for `src/elspeth/` paths is now empty — every src/ migration site has flipped. Stage 4 flips `tests/`; Stage 5 deletes the script entirely. Until Stage 5, the script + allowlist remain to guard against new `RowOutcome.X` introductions during the test-migration window.

### Behavioural verification (NEW for this PR)

Per Phase 5, three new integration-test fixtures must exist and be green:

```bash
# Discard-mode RunStatus flip (the operator-visible change)
.venv/bin/python -m pytest tests/integration/test_adr_019_discard_mode_flip.py -v

# Cross-table invariants I1c (failsink-paired) and I3 (discard-no-failsink)
.venv/bin/python -m pytest tests/integration/test_adr_019_cross_table_invariants.py -v

# Counter changes for (SUCCESS, GATE_ROUTED) and (FAILURE, ON_ERROR_ROUTED)
.venv/bin/python -m pytest tests/integration/test_adr_019_counter_changes.py -v
```

These tests are described in Phase 5; their existence is gated by a Phase 3 Definition of Done ("Behavioural test for discard-mode flip is RED before the predicate rewrite, GREEN after").

---

## Out of scope for this PR

- **The ~143 `outcome == RowOutcome.X` assertion sites in `tests/`** — Stage 4 ticket `elspeth-27ce7613fa`. These are mechanical translations against the canonical mapping table in `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`. The Phase 5 triage explicitly leaves them.

- **`RowOutcome` enum deletion** — Stage 5 ticket `elspeth-774b1d3c2e`. RowOutcome continues to exist alongside `TerminalOutcome` until every assertion site has flipped. Deleting it is the final sweep.

- **The Stage 1 lint guard `forbid_new_row_outcome.py` and its allowlist** — Stage 5. The guard prevents drift during Stages 4 and earlier; once Stage 5 deletes `RowOutcome` itself, the guard becomes vacuous and is removed.

- **Counter rename** — separate ADR-020 conversation. Public API field names (`rows_succeeded`, `rows_routed_success`, etc.) are preserved in this PR per ADR-019 § Counter derivation contract.

- **Frontend TypeScript types** — `web/frontend/src/types/index.ts` does not require changes (counter names preserved per ADR-019 § Counter derivation contract). **NOTE:** the Python web module `src/elspeth/web/execution/` IS in scope and DOES change — Pydantic schemas at `web/execution/schemas.py` are updated in Phase 3 Task 3.5; SQL queries at `web/execution/{diagnostics,discard_summary}.py` are updated in Phase 1 Task 1.9. The "frontend" caveat applies only to the React/TypeScript layer at `web/frontend/`.

---

## How to use this plan

1. Read this overview.
2. Read each phase document in order.
3. Execute phase-by-phase using `superpowers:executing-plans`. Each phase has its own RED→GREEN test cycle and Definition of Done.
4. After Phase 5's Definition of Done is met, run all verification gates above. If all green, open the PR.
5. The PR description summarizes the change, links to ADR-019, and includes the operator deletion command from `docs/operator/migrations/adr-019.md`.

If a phase reveals a gap the ADR doesn't cover, STOP and re-read this overview's decision log. If the gap is genuinely new, surface to the user before silently expanding scope — same discipline as Stage 1 used for the allowlist deviation.
