# ADR-019 Stage 2/3 — Phase 5: Test Strategy + Triage + Operator Migration Doc

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is the LAST phase of the five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). After Phase 5's commit, the PR opens — all gates must be green. Stage 4 (test mechanical translation) and Stage 5 (delete RowOutcome) are separate PRs landing AFTER this one.

**Goal:** Triage the `tests/` tree into three categories — schema-dependent (must move with this PR), assertion-only (defer to Stage 4), and freshly-added behavioural (already added in Phases 3 and 4). Update the schema-dependent tests so the suite is green end-to-end. Add the operator migration documentation. Open the PR.

**Files touched in this phase:**

- Modify: `tests/` — schema-dependent test fixtures (per the triage)
- Create: `docs/operator/migrations/adr-019.md` — operator-facing migration guide
- Modify: `config/cicd/forbid_new_row_outcome/migration_files.yaml` — final allowlist trim (after this PR, only `contracts/enums.py`, `testing/__init__.py`, `tests/` remain)
- Test: full suite green

**Background reading:** Phase 3 introduced two RED-first integration tests. Phase 4 added six. This phase ensures the rest of the suite compiles and passes; it does NOT do the Stage 4 mechanical translation of `outcome == RowOutcome.X` assertions.

---

## The grep recipe (executable triage)

The triage rule is mechanical. A `tests/` file falls into one of three categories:

### Category A — Schema-dependent (MUST move with this PR)

A test is schema-dependent if it CONSTRUCTS a contract dataclass instance. After Phase 1's retype, the old constructor calls fail to import:

```python
# Schema-dependent — won't compile after Phase 1
TokenOutcome(outcome=RowOutcome.COMPLETED, is_terminal=True, ...)
RowResult(outcome=RowOutcome.FAILED, ...)
PendingOutcome(outcome=RowOutcome.QUARANTINED, error_hash="...")
TokenCompleted(outcome=RowOutcome.ROUTED, sink_name="x")
```

Identify with this grep:

```bash
grep -rn "TokenOutcome(\|RowResult(\|PendingOutcome(\|TokenCompleted(" tests/ \
  | grep -v "^Binary" | sort -u > /tmp/category-a.txt

wc -l /tmp/category-a.txt
```

### Category B — Assertion-only (defer to Stage 4)

A test is assertion-only if it READS `result.outcome == RowOutcome.X` from a real engine output without constructing the dataclass:

```python
# Assertion-only — Stage 4 mechanical translation
assert result.outcome == RowOutcome.COMPLETED
assert outcome.outcome == RowOutcome.QUARANTINED
```

Identify with this grep:

```bash
grep -rn "outcome\s*==\s*RowOutcome\." tests/ \
  | grep -v "^Binary" | sort -u > /tmp/category-b.txt

wc -l /tmp/category-b.txt
```

### Category C — Audit-DB direct read (must move with this PR)

A test that reads `token_outcomes` rows directly via SQL (rather than through the loader) and checks `outcome` / `is_terminal` column values is schema-dependent — those columns no longer exist with the old names/values.

Identify with:

```bash
grep -rn "is_terminal\|token_outcomes_table" tests/ | head -30
```

Each hit needs evaluation: if it consults the renamed `completed` column or new `path` column semantics, fix it; if it just queries existence, no change needed.

---

## Tasks

### Task 5.1: Run the triage grep

**Step 1: Generate the category lists**

```bash
mkdir -p /tmp/adr-019-triage
grep -rn "TokenOutcome(\|RowResult(\|PendingOutcome(\|TokenCompleted(" tests/ \
  > /tmp/adr-019-triage/category-a.txt
grep -rn "outcome\s*==\s*RowOutcome\." tests/ \
  > /tmp/adr-019-triage/category-b.txt
grep -rn "is_terminal\|token_outcomes_table" tests/ \
  > /tmp/adr-019-triage/category-c.txt

wc -l /tmp/adr-019-triage/*.txt
```

**Step 2: Verify expected counts**

Per the Stage 1 recount, total `tests/` references to `RowOutcome.X` are ~645. Of those, ~143 are `outcome == RowOutcome.X` assertion sites (Category B — Stage 4). The remainder breaks down approximately as:
- Category A (constructor calls): ~80-120 expected
- Category C (direct DB reads): ~10-30 expected
- Category B (assertion-only): 143 (already known)
- The remaining count (~360) is fixture setup / mapping table imports / commentary references — most don't need changes; the lint guard's allowlist on `tests/` covers them through Stage 4.

If your category-A count is wildly different from this estimate (less than 30 or more than 200), STOP and surface to user — the triage might be missing a pattern.

**Step 3: Confirm category-A files compile**

For each file in `/tmp/adr-019-triage/category-a.txt`'s file list:

```bash
awk -F: '{print $1}' /tmp/adr-019-triage/category-a.txt | sort -u | while read f; do
  .venv/bin/python -c "import importlib.util; spec=importlib.util.spec_from_file_location('m','$f'); spec.loader.exec_module(importlib.util.module_from_spec(spec))" 2>&1 | grep -q "Error" && echo "FAILS: $f"
done
```

Files that fail to import are Category A — must be fixed in this phase.

**Definition of Done:**
- [ ] Category lists generated
- [ ] Counts within sanity bounds
- [ ] Expected Category A files identified

---

### Task 5.2: Update Category A tests (constructor flips)

**Files:** approximately 30-80 files under `tests/unit/contracts/`, `tests/unit/core/landscape/`, `tests/integration/`, `tests/unit/engine/`.

**Step 1: Identify the construction sites**

For each file in Category A, the typical fix is:

```python
# OLD:
record = TokenOutcome(
    outcome_id="o1",
    run_id="r1",
    token_id="t1",
    outcome=RowOutcome.COMPLETED,
    is_terminal=True,
    recorded_at=datetime.now(timezone.utc),
    sink_name="primary",
)

# NEW:
record = TokenOutcome(
    outcome_id="o1",
    run_id="r1",
    token_id="t1",
    outcome=TerminalOutcome.SUCCESS,
    path=TerminalPath.DEFAULT_FLOW,
    completed=True,
    recorded_at=datetime.now(timezone.utc),
    sink_name="primary",
)
```

Use the canonical mapping at `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING` to translate each `RowOutcome.X` to its `(TerminalOutcome, TerminalPath)` pair. For `DIVERTED`, inspect the test context to determine failsink-mode (`SINK_FALLBACK_TO_FAILSINK`) vs discard-mode (`SINK_DISCARDED`).

**Step 2: Fix imports**

Each updated file changes its import line from `from elspeth.contracts.enums import RowOutcome` to `from elspeth.contracts.enums import TerminalOutcome, TerminalPath`. (Keep `RowOutcome` import only if other parts of the same file still use it — Stage 4 will trim those.)

**Step 3: Run the impacted tests**

```bash
.venv/bin/python -m pytest tests/unit/contracts/ tests/unit/core/landscape/ tests/unit/engine/ tests/integration/ -q
```

Expected: all tests pass.

**Step 4: Confirm Category B tests STILL skip / xfail / pass**

Some Category B tests may pass anyway because the engine output now produces `TerminalOutcome.SUCCESS` which is `"success"`, while the test's `RowOutcome.COMPLETED.value` is `"completed"`. The string mismatch could:
- Hard-fail (assertion failure) — leave as-is for Stage 4 to fix.
- Silently pass because the test does something subtle — Stage 4 will catch and fix.

For tests where the failure is BLOCKING the suite from being green, either:
1. Fix the assertion as part of this PR (cross over from Category B → A).
2. Mark with `@pytest.mark.xfail(reason="ADR-019 Stage 4 mechanical translation")`.

Prefer option 1 unless the test count is overwhelming. Track each xfail in a Stage 4 comment.

**Definition of Done:**
- [ ] All Category A files fixed
- [ ] Imports cleaned per file
- [ ] Suite green; any persistent failures either fixed (cross-over) or xfail-tagged with Stage 4 reference

---

### Task 5.3: Update Category C tests (direct DB reads)

**Step 1: Audit direct-DB-read tests**

For each file in Category C, find queries like:

```python
# OLD:
result = conn.execute(
    select(token_outcomes_table.c.outcome, token_outcomes_table.c.is_terminal)
    .where(token_outcomes_table.c.token_id == "t1")
)

# NEW:
result = conn.execute(
    select(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
    )
    .where(token_outcomes_table.c.token_id == "t1")
)
```

Update SELECTs and WHEREs that reference `is_terminal` to `completed`; add `path` when needed for assertion clarity.

**Definition of Done:**
- [ ] All Category C files fixed
- [ ] Direct-DB-read tests pass

---

### Task 5.4: Add operator migration documentation

**Files:**
- Create: `docs/operator/migrations/adr-019.md`

**Step 1: Write the migration guide**

```markdown
# ADR-019 Operator Migration Guide

**Date:** 2026-05-04
**Stage:** Audit DB schema + behaviour change (Stages 2/3 of the five-stage rollout)

## What this PR does

Replaces the single-axis `RowOutcome` audit-DB recording with the two-axis
`(TerminalOutcome, TerminalPath, completed)` triple. See
[ADR-019](../../architecture/adr/019-two-axis-terminal-model.md) for the
rationale and the full mapping table.

## Action required for production / staging deploys

### 1. Delete the audit and sessions databases

ELSPETH does not run Alembic migrations
([MEMORY.md::project_db_migration_policy](../../../MEMORY.md)). The
`token_outcomes` table schema changes — column rename + new column + value
space change. **Before deploying this commit,** delete the existing
databases:

```bash
# Replace paths with your deployment-specific locations.
rm -f /var/lib/elspeth/audit.db
rm -f /var/lib/elspeth/sessions.db

# OR, if Postgres:
psql -d elspeth_audit -c "DROP TABLE IF EXISTS token_outcomes CASCADE;"
psql -d elspeth_sessions -c "DROP DATABASE IF EXISTS elspeth_sessions;"
```

The engine's startup `metadata.create_all()` recreates the new schema.

**No data migration is offered.** Pre-ADR-019 audit data is a different
shape and cannot be losslessly converted to the new model. If you need
historical audit records, snapshot the old DB before deletion.

### 2. Behaviour change: discard-mode `RunStatus` flip

Pipelines using discard-mode sinks (`sink_name="__discard__"`) will see
their `RunStatus` flip:

| Pre-ADR-019 | Post-ADR-019 |
| --- | --- |
| Some discards + some success → `COMPLETED` | → `COMPLETED_WITH_FAILURES` |
| All rows discarded → `COMPLETED` | → `FAILED` |

Per ADR-019 § Sub-decision 5: discard-mode is reclassified as a
predicate-input `(FAILURE, SINK_DISCARDED)`. The discarded rows now
increment `rows_failed` and flip `failure_indicator`.

#### How to identify affected pipelines

```bash
grep -rn '"__discard__"' your-pipeline-configs/
# OR (if discard is configured via an on_error block):
grep -rn 'on_error: discard\|on_error:\s*\(\|on_error:\s*$' your-pipeline-configs/
```

#### How to preserve old behaviour (if intentional)

If your pipeline uses discard as silent housekeeping (rows that should be
dropped without affecting run status), reconfigure to route those rows to
a no-op success sink:

```yaml
# Before (silent drop):
sinks:
  - name: __discard__

# After (explicit no-op success):
sinks:
  - name: silent_drop
    plugin: noop_sink   # writes nothing, returns success
```

If the new semantics are acceptable (discarded rows count toward
`rows_failed`), no action needed beyond re-baselining dashboards that
read `rows_succeeded` / `rows_failed`.

### 3. Counter changes — `(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)`

Two accumulator counter changes ship with this PR. Public API field
names are preserved per ADR-019 § Counter derivation contract.

| Counter | Pre-ADR-019 | Post-ADR-019 |
| --- | --- | --- |
| `rows_succeeded` | did NOT include gate MOVE rows | NOW includes gate MOVE rows |
| `rows_routed_success` | did include gate MOVE rows | UNCHANGED — still includes them |
| `rows_failed` | did NOT include transform `on_error` rows | NOW includes them |
| `rows_routed_failure` | did include transform `on_error` rows | UNCHANGED |

Dashboards reading `rows_succeeded` will see higher numbers for runs with
gate MOVE routing; dashboards reading `rows_failed` will see higher
numbers for runs with transform `on_error` routing.

**Recommended action:** re-baseline `rows_succeeded` / `rows_failed`
dashboard alerts after the first post-deploy run. The `rows_routed_*`
counter values are unchanged.

## Verifying the deployment

After deploy, run a known-good pipeline and confirm:

```bash
# 1. The audit DB has the new schema.
sqlite3 /var/lib/elspeth/audit.db ".schema token_outcomes" | grep -E "completed|path"
# Expected: both `completed` and `path` columns present.

# 2. The engine emits the new triple.
sqlite3 /var/lib/elspeth/audit.db \
  "SELECT outcome, path, completed FROM token_outcomes LIMIT 5;"
# Expected: outcome IN ('success', 'failure', 'transient', NULL),
# path IN (... 13 values), completed IN (0, 1).

# 3. The lint guard runs cleanly.
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check \
  --root . --allowlist config/cicd/forbid_new_row_outcome
# Expected: exit 0.
```

## Related work

- **Stage 4** (mechanical test translation) — separate PR that flips ~143
  `outcome == RowOutcome.X` assertion sites in `tests/`. Tracked in
  Filigree ticket `elspeth-27ce7613fa`. Not blocking this deploy.

- **Stage 5** (delete RowOutcome) — final sweep; deletes the enum and the
  lint guard. Tracked in `elspeth-774b1d3c2e`. Lands after Stage 4.

- **ADR-020** (potential counter rename) — separate breaking-API
  conversation. Not coupled to this PR.
```

**Step 2: Verify the doc renders cleanly**

```bash
.venv/bin/python -m markdown docs/operator/migrations/adr-019.md > /dev/null && echo OK
```

(If markdown lint is configured in pre-commit, run that against the new file.)

**Definition of Done:**
- [ ] `docs/operator/migrations/adr-019.md` written
- [ ] Renders without errors

---

### Task 5.5: Final allowlist trim

**Files:**
- Modify: `config/cicd/forbid_new_row_outcome/migration_files.yaml`

**Step 1: Verify the post-PR allowlist scope**

After Phases 1-4, src/ scope reaches zero RowOutcome references. The remaining allowlist is:

```yaml
allowed:
  - file: src/elspeth/contracts/enums.py
    justification: ADR-019 migration site — defines RowOutcome (kept through Stage 4); also defines TerminalOutcome / TerminalPath.
  - file: src/elspeth/testing/__init__.py
    justification: testing pack re-exports RowOutcome alongside TerminalOutcome/TerminalPath; Stage 5 trims to TerminalOutcome/TerminalPath only.
  - file: tests/
    justification: ~143 assertion sites + ~500 fixture references; Stage 4 mechanical sweep flips them.
```

Trim out every allowlist entry that no longer applies. The exact entries to remove are:

- `src/elspeth/contracts/audit.py` (Phase 1 retyped)
- `src/elspeth/contracts/events.py` (Phase 1 retyped)
- `src/elspeth/contracts/engine.py` (Phase 1 retyped)
- `src/elspeth/contracts/results.py` (Phase 1 retyped)
- `src/elspeth/core/landscape/data_flow_repository.py` (Phase 1 flipped + Phase 4 invariants)
- `src/elspeth/core/landscape/model_loaders.py` (Phase 1 loader retyped)
- `src/elspeth/core/checkpoint/recovery.py` (Phase 2 query flipped)
- `src/elspeth/engine/orchestrator/outcomes.py` (Phase 3 accumulator flipped)
- `src/elspeth/engine/orchestrator/core.py` (Phase 3 resume flipped + Phase 4 sweep)
- `src/elspeth/engine/executors/sink.py` (Phase 2 producer flipped + Phase 3 counter changes)
- `src/elspeth/engine/executors/transform.py` (Phase 2 producer flipped)
- `src/elspeth/engine/processor.py` (Phase 2 producer flipped)
- `src/elspeth/engine/coalesce_executor.py` (Phase 2 producer flipped)

These should already be removed across Phase 1-4 commits. Phase 5 verifies the allowlist file matches the final state.

**Step 2: Run the lint guard against the trimmed allowlist**

```bash
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check \
  --root . --allowlist config/cicd/forbid_new_row_outcome
```

Expected: `exit 0`. If any src/ file outside the final 2-entry allowlist contains `RowOutcome.X`, fix it (it was missed in Phases 1-4) or surface to user if root cause is unclear.

**Definition of Done:**
- [ ] Allowlist trimmed to its final 3-entry shape
- [ ] Lint guard passes

---

### Task 5.6: Final verification + Phase 5 commit + PR open

**Step 1: Run all verification gates**

```bash
.venv/bin/python -m pytest tests/ -q --timeout=120
.venv/bin/python -m mypy src/elspeth
.venv/bin/python -m ruff check src/ tests/ scripts/
.venv/bin/python -m ruff format --check src/ tests/ scripts/
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model --exclude "**/__pycache__/*"
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth
.venv/bin/python scripts/cicd/enforce_contract_manifest.py check --allowlist config/cicd/enforce_contract_manifest
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
.venv/bin/python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth --allowlist config/cicd/enforce_frozen_annotations
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome
```

ALL must pass.

**Step 2: Confirm the three new behavioural test fixtures are GREEN**

```bash
.venv/bin/python -m pytest \
  tests/integration/test_adr_019_discard_mode_flip.py \
  tests/integration/test_adr_019_counter_changes.py \
  tests/integration/test_adr_019_cross_table_invariants.py \
  -v
```

**Step 3: Commit**

```bash
git add tests/ docs/operator/migrations/adr-019.md \
        config/cicd/forbid_new_row_outcome/migration_files.yaml

git commit -m "$(cat <<'EOF'
feat(adr-019): phase 5 — schema-dependent test fixes + operator migration doc

ADR-019 Stage 2/3 Phase 5 of 5 (see docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md).

Schema-dependent test fixups (Category A — TokenOutcome/RowResult/PendingOutcome/
TokenCompleted constructor calls): every file that constructs these dataclasses
flipped to (outcome=TerminalOutcome.X, path=TerminalPath.Y) instead of
(outcome=RowOutcome.X, is_terminal=...). Per ADR-019 mapping table at
tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING.

Direct-DB-read tests (Category C): SELECTs and WHEREs against
token_outcomes.is_terminal renamed to .completed; .outcome value space
adjusted from RowOutcome.value to TerminalOutcome.value.

Assertion-only tests (Category B — ~143 ``outcome == RowOutcome.X`` sites)
deferred to Stage 4 ticket elspeth-27ce7613fa. The lint guard's tests/
allowlist entry covers them through that PR.

Operator-facing migration documentation at docs/operator/migrations/adr-019.md:
- DB delete commands (no Alembic per project policy)
- Discard-mode RunStatus flip + how to preserve old behaviour
- Counter changes for rows_succeeded / rows_failed
- Verification checklist for post-deploy

Final allowlist scope: contracts/enums.py + testing/__init__.py + tests/.
Stage 5 trims testing/__init__.py and deletes RowOutcome + the lint guard.

Refs: elspeth-949719575e (Stage 2), elspeth-edb60744f0 (Stage 3)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 4: Open the PR**

```bash
git push -u origin RC5-UX-RoutingVocabFix

gh pr create --title "feat(adr-019): two-axis terminal model — recorder + producer + accumulator + predicate" \
  --body "$(cat <<'EOF'
## Summary

Implements ADR-019 Stages 2/3 (merged) — replaces the single-axis `RowOutcome` audit recording with the two-axis `(TerminalOutcome, TerminalPath, completed)` triple across the recorder, every producer emit site, the accumulator, the resume aggregator, the four contract dataclasses, the telemetry event payload, and the `RunResult.__post_init__` predicate.

Ships three operator-visible behaviour changes (see [docs/operator/migrations/adr-019.md](docs/operator/migrations/adr-019.md)):

1. **Discard-mode `RunStatus` flip** — pipelines using `sink_name="__discard__"` now report `COMPLETED_WITH_FAILURES` (or `FAILED` if all rows discard) instead of `COMPLETED`, per ADR-019 § Sub-decision 5.
2. **`(SUCCESS, GATE_ROUTED)` counter doubling** — gate-routed rows now bump BOTH `rows_succeeded` and `rows_routed_success`.
3. **`(FAILURE, ON_ERROR_ROUTED)` counter doubling** — symmetric for transform `on_error` routes.

Plus four NEW Tier 1 cross-table invariants (I1a, I1b, I1c, I3) per ADR-019 § Cross-check invariants.

**Operator action required at deploy time:** delete `audit.db` and `sessions.db` per `docs/operator/migrations/adr-019.md`. ELSPETH does not run Alembic ([MEMORY.md::project_db_migration_policy](MEMORY.md)).

## Phase structure (5 commits)

1. Phase 1 — schema + recorder + loader + dataclass two-axis flip
2. Phase 2 — producer site flip (36 sites across processor, transform, sink, coalesce_executor, recovery)
3. Phase 3 — accumulator + predicate + resume aggregation + behaviour changes
4. Phase 4 — cross-table invariants I1a, I1b, I1c, I3
5. Phase 5 — schema-dependent test fixes + operator migration doc

The merge is atomic per ADR-019 lines 318-320 — accumulator change ships in lockstep with the predicate rewrite. Squash-or-keep at merge time is reviewer's choice.

## What's NOT in this PR

- ~143 `outcome == RowOutcome.X` assertion sites in `tests/` — Stage 4 ticket `elspeth-27ce7613fa` (mechanical translation against the canonical mapping at `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`).
- Deletion of `RowOutcome` itself — Stage 5 ticket `elspeth-774b1d3c2e`.
- Counter renames — separate ADR-020 conversation if pursued.

## Test plan

- [x] All ~16,010 tests pass (verified locally; `pytest tests/ -q --timeout=120`)
- [x] Three new behavioural integration test files green:
  - `tests/integration/test_adr_019_discard_mode_flip.py`
  - `tests/integration/test_adr_019_counter_changes.py`
  - `tests/integration/test_adr_019_cross_table_invariants.py`
- [x] mypy clean across 364 src files
- [x] ruff lint + format clean
- [x] All project gates pass (tier model, contract manifest, plugin hashes, freeze guards, frozen annotations, forbid-new-row-outcome)
- [x] Lint guard `forbid_new_row_outcome.py` passes; src/ scope reaches zero RowOutcome references; tests/ remain allowlisted for Stage 4

## Refs

- ADR: docs/architecture/adr/019-two-axis-terminal-model.md
- Plan: docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md
- Stage 1 (already shipped): commit 60d30551
- Stage 2 ticket: elspeth-949719575e
- Stage 3 ticket: elspeth-edb60744f0

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Definition of Done:**
- [ ] All verification gates green
- [ ] Phase 5 commit landed
- [ ] PR opened with comprehensive description
- [ ] Operator migration doc linked from PR description
- [ ] Stage 1, Stage 4, and Stage 5 ticket cross-references in PR description
