# ADR-019 Stage 2/3 — Phase 5: Test Strategy + Triage + Operator Migration Doc

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is the LAST phase of the five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). After Phase 5's commit, the PR opens — all gates must be green. There is no xfail/Stage-4 deferral for broken `outcome == RowOutcome.X` assertions in this PR; cross-enum equality returns `False`, so those tests would fail the required full-suite gate.

**Goal:** Triage the `tests/` tree into three categories — schema-dependent, assertion-only, and direct-DB-read. Schema-dependent and direct-DB-read tests must move with this PR because the old schema no longer exists. Assertion-only tests must also move with this PR because `TerminalOutcome.X == RowOutcome.Y` silently returns `False`; leaving those assertions for a later PR is incompatible with `pytest tests/ -q` exiting 0. Run and refresh the AST-backed source inventory introduced in Phase 1 so downstream schema/wire-contract misses cannot hide behind grep blind spots. Update the tests so the suite is green end-to-end. Expand the Phase 1 operator migration stub into the full deployment and rollback runbook. Open the PR.

**Files touched in this phase:**

- Modify: `tests/` — schema-dependent test fixtures (per the triage)
- Modify/use: `scripts/cicd/adr019_symbol_inventory.py` — AST-backed inventory created in Phase 1 for `is_terminal` declarations/accessors/kwargs/dict keys and hardcoded RowOutcome value comparisons
- Modify/use: `config/cicd/adr019_symbol_inventory/` — temporary allowlist directory created in Phase 1 for migration-window source inventory findings
- Test: `tests/unit/scripts/cicd/test_adr019_symbol_inventory.py`
- Modify: `docs/operator/migrations/adr-019.md` — operator-facing migration guide stub created in Phase 1; expand to full runbook here
- Modify: `config/cicd/forbid_new_row_outcome/migration_files.yaml` — final allowlist trim (after this PR, only `contracts/enums.py`, `testing/__init__.py`, `tests/` remain)
- Test: full suite green

**Background reading:** Phase 3 introduced two RED-first integration tests. Phase 4 added six. This phase ensures the rest of the suite compiles and passes, including the mechanical translation of `outcome == RowOutcome.X` assertion sites that would otherwise fail after the contract retype.

---

## The AST-backed triage recipe

The triage rule is mechanical. Grep remains useful for quick local slices, but
the source-closeout gate is AST-backed. This is load-bearing: grep-only sweeps
missed `contracts/export_records.py::TokenOutcomeExportRecord`, and the same
class of miss can recur in TypedDict declarations, function-call kwargs, dict
literal keys, and SQLAlchemy accessors. Do not close Phase 5 from grep output
alone.

A `tests/` file falls into one of three categories:

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

### Category B — Assertion-only (MUST move with this PR)

A test is assertion-only if it READS `result.outcome == RowOutcome.X` from a real engine output without constructing the dataclass:

```python
# Assertion-only — mechanical translation required before PR open
assert result.outcome == RowOutcome.COMPLETED
assert outcome.outcome == RowOutcome.QUARANTINED
```

Identify with this grep:

```bash
grep -rn "outcome\s*==\s*RowOutcome\." tests/ \
  | grep -v "^Binary" | sort -u > /tmp/category-b.txt

wc -l /tmp/category-b.txt
```

Translate every Category B assertion in this PR. Do not add xfail markers and do
not create a Stage 4 manifest for these sites. Cross-enum equality is a normal
`False` comparison, so these tests fail with `AssertionError`; deferring them
would make the full-suite green gate impossible.

### Category C — Audit-DB direct read (must move with this PR)

A test that reads `token_outcomes` rows directly via SQL (rather than through the loader) and checks `outcome` / `is_terminal` column values is schema-dependent — those columns no longer exist with the old names/values.

Identify with:

```bash
grep -rn "is_terminal\|token_outcomes_table" tests/ | head -30
```

Each hit needs evaluation: if it consults the renamed `completed` column or new `path` column semantics, fix it; if it just queries existence, no change needed.

### Category D — Source wire-contract / schema-field sweep (must move with this PR)

The downstream-consumer sweep is not test-only. It must also catch source-level
wire contracts and TypedDict declarations that never touch
`token_outcomes_table.c.is_terminal` directly. This is the class that previously
missed `contracts/export_records.py::TokenOutcomeExportRecord`.

Identify with these source-wide checks:

```bash
grep -rn "is_terminal: bool" src/elspeth/ \
  | grep -v "src/elspeth/contracts/audit.py" \
  | sort -u > /tmp/category-d-typedicts.txt

grep -rn "['\\\"]is_terminal['\\\"]" src/elspeth/ \
  | grep -v "src/elspeth/contracts/audit.py" \
  | sort -u > /tmp/category-d-dict-keys.txt

grep -rn "token_outcomes_table\\.c\\.is_terminal\\|\\.is_terminal" src/elspeth/ \
  | grep -v "src/elspeth/contracts/audit.py" \
  | sort -u > /tmp/category-d-accessors.txt
```

Do **not** use a directory-level `/contracts/` exclusion here. Use file-level
exclusions only for explicitly migrated source files, because `contracts/`
contains exported wire contracts as well as the dataclass under migration.

Every Category D hit must be classified as one of:

- in-scope Phase 1 consumer patch,
- in-scope Phase 5 test assertion translation, or
- false positive with the exact line and reason recorded in the PR notes.

---

## Tasks

### Task 5.0: Run and refresh AST-backed ADR-019 symbol inventory

**Why this task exists:** Phase 1 Task 1.0 creates the AST inventory before any
schema edits. Phase 5 uses it as the closeout gate after all producer,
accumulator, invariant, and test-triage edits have landed. The D7 grep recipe
has already missed source surfaces across multiple review rounds; the PR cannot
close on grep output alone.

**Files:**
- Modify/use: `scripts/cicd/adr019_symbol_inventory.py`
- Modify/use: `config/cicd/adr019_symbol_inventory/`
- Test: `tests/unit/scripts/cicd/test_adr019_symbol_inventory.py`

**Step 1: Confirm the Phase 1 inventory visitor still covers all required forms**

The script should already walk Python files under a root (default `src/elspeth`) and
report findings as JSON lines plus a non-zero exit code when non-allowlisted
findings remain. Use `ast.parse`; do not tokenize with regex.

Minimum finding kinds:

```python
class FindingKind(StrEnum):
    IS_TERMINAL_ANNOTATION = "is_terminal_annotation"       # AnnAssign target Name("is_terminal")
    IS_TERMINAL_ATTRIBUTE = "is_terminal_attribute"         # Attribute(attr="is_terminal")
    IS_TERMINAL_KEYWORD = "is_terminal_keyword"             # keyword.arg == "is_terminal"
    IS_TERMINAL_DICT_KEY = "is_terminal_dict_key"           # Dict key Constant("is_terminal")
    ROW_OUTCOME_STRING_COMPARE = "row_outcome_string_compare"  # outcome == "quarantined", etc.
    TERMINAL_OUTCOME_STRING_COMPARE = "terminal_outcome_string_compare"
    TERMINAL_PATH_STRING_COMPARE = "terminal_path_string_compare"
```

The terminal string checks are not optional. The migration must not merely
replace `outcome == "quarantined"` with `path == "quarantined_at_source"` or
`outcome == "failure"`; those are the same fragility class. The visitor should
flag equality / inequality comparisons where either side is a known
`TerminalOutcome` or `TerminalPath` value string outside `contracts/enums.py`
and explicitly allowlisted test fixtures.

Report fields: `kind`, `path`, `line`, `col`, `symbol`, and a short `context`
from `ast.unparse(node)` when available.

**Step 2: Re-run and extend focused tests if any syntactic gap appears**

The test file must include one test per finding kind and at least two false
positive guards:

```python
class OutcomeDistributionEntry(TypedDict):
    is_terminal: bool                 # annotation finding

record.is_terminal                    # attribute finding
record_token_outcome(is_terminal=True) # keyword finding
{"is_terminal": True}                 # dict-key finding
outcome == "quarantined"              # RowOutcome value compare finding
outcome == "failure"                  # TerminalOutcome value compare finding
path == "quarantined_at_source"       # TerminalPath value compare finding

terminal = True                       # no finding
payload = {"completed": True}         # no finding
outcome in {"completed", "failed"}    # no compare finding; membership is too broad
```

Use the committed fixture corpus from Phase 1
(`tests/fixtures/cicd/adr019_symbol_inventory/`) plus any additional temporary
files needed for edge cases. The fixture corpus must include positive and
negative examples and an import-presence case that imports `TerminalOutcome` /
`TerminalPath` while still using brittle string comparisons. Call the inventory
function directly for exact findings and exercise the CLI `check` command once
with `--allowlist config/cicd/adr019_symbol_inventory`.

**Step 3: Use the tool as the source closeout gate**

After the Phase 1 consumer patches and Phase 5 test triage have landed, run:

```bash
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check \
  --root src/elspeth \
  --allowlist config/cicd/adr019_symbol_inventory
```

Expected: zero findings outside the deliberately allowed migration files
(`contracts/enums.py`, `testing/__init__.py` until Stage 5). If the script
reports any `contracts/export_records.py`, MCP, Web, CLI formatter, or Landscape
exporter hit, STOP and patch the consumer in this PR.

**Definition of Done:**
- [ ] AST inventory script created in Phase 1 remains covered by unit tests
- [ ] Temporary allowlist directory exists and is limited to deliberate migration-window files
- [ ] Fixture corpus covers all finding kinds, false-positive guards, and the import-presence case
- [ ] Script detects annotations, attributes, kwargs, dict keys, hardcoded RowOutcome value comparisons, and hardcoded TerminalOutcome/TerminalPath value comparisons
- [ ] Script exits non-zero for non-allowlisted findings
- [ ] Phase 5 closeout uses this script; grep-only D7 checks are not sufficient for approval

---

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
grep -rn "is_terminal: bool" src/elspeth/ \
  | grep -v "src/elspeth/contracts/audit.py" \
  > /tmp/adr-019-triage/category-d-typedicts.txt
grep -rn "['\\\"]is_terminal['\\\"]" src/elspeth/ \
  | grep -v "src/elspeth/contracts/audit.py" \
  > /tmp/adr-019-triage/category-d-dict-keys.txt
grep -rn "token_outcomes_table\\.c\\.is_terminal\\|\\.is_terminal" src/elspeth/ \
  | grep -v "src/elspeth/contracts/audit.py" \
  > /tmp/adr-019-triage/category-d-accessors.txt

wc -l /tmp/adr-019-triage/*.txt
```

**Step 2: Verify expected counts**

Per the Stage 1 recount, total `tests/` references to `RowOutcome.X` are ~645. Of those, ~143 are `outcome == RowOutcome.X` assertion sites (Category B — mechanical translation required in this PR). The remainder breaks down approximately as:
- Category A (constructor calls): ~80-120 expected
- Category C (direct DB reads): ~10-30 expected
- Category B (assertion-only): 143 (must be updated before PR open)
- The remaining count (~360) is fixture setup / mapping table imports / commentary references — most don't need changes in this PR, but none may leave pytest-blocking `outcome == RowOutcome.X` assertions behind.

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
- [ ] Category D source-wire sweeps generated, with no directory-level `/contracts/` exclusion
- [ ] Counts within sanity bounds
- [ ] Expected Category A files identified

---

### Task 5.2: Update Category A and B tests (constructor flips + assertion translation)

**Files:** approximately 30-80 constructor-heavy files under `tests/unit/contracts/`, `tests/unit/core/landscape/`, `tests/integration/`, `tests/unit/engine/`, plus every file listed in `/tmp/adr-019-triage/category-b.txt`.

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

**Step 2: Translate Category B assertion-only sites**

For each `outcome == RowOutcome.X` assertion in `/tmp/adr-019-triage/category-b.txt`, translate to the new two-axis assertion using `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`:

```python
# OLD:
assert result.outcome == RowOutcome.ROUTED_ON_ERROR

# NEW:
assert result.outcome == TerminalOutcome.FAILURE
assert result.path == TerminalPath.ON_ERROR_ROUTED
```

For `RowOutcome.DIVERTED`, inspect the test context:

- failsink-mode diversion asserts `(TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK)`,
- discard-mode diversion asserts `(TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED)`.

Do not xfail these tests and do not defer them to Stage 4. If a Category B site
does not expose a path field today, the test must assert via the nearest
production loader/result object that does expose `(outcome, path)`, or the
implementation must surface the missing path through the existing contract shape
that Phase 1 retyped.

**Step 3: Fix imports**

Each updated file changes its import line from `from elspeth.contracts.enums import RowOutcome` to `from elspeth.contracts.enums import TerminalOutcome, TerminalPath`. Keep `RowOutcome` only for the canonical mapping-table tests or intentionally documented compatibility references; do not keep it solely for stale assertions.

**Step 4: Run the impacted tests**

```bash
.venv/bin/python -m pytest tests/unit/contracts/ tests/unit/core/landscape/ tests/unit/engine/ tests/integration/ -q
```

Expected: all tests pass.

**Step 5: Prove no xfail/deferred assertion escape hatch exists**

```bash
rg -n "outcome\\s*==\\s*RowOutcome\\." tests/
rg -n "xfail\\(.*ADR-019 Stage 4 mechanical translation" tests/
```

Expected: both commands return no results, except the first may report the
canonical mapping-table test if it is explicitly testing the deprecated enum
itself and not comparing a real engine output.

**Definition of Done:**
- [ ] All Category A files fixed
- [ ] All Category B assertion-only sites translated to `(TerminalOutcome, TerminalPath)` assertions in this PR
- [ ] Imports cleaned per file
- [ ] Suite green with no ADR-019 Stage 4 xfails
- [ ] `rg -n "outcome\\s*==\\s*RowOutcome\\." tests/` returns no real-engine-output assertions

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

### Task 5.4: Expand operator migration documentation

**Files:**
- Modify: `docs/operator/migrations/adr-019.md`

**Step 1: Expand the Phase 1 stub into the full migration guide**

```bash
mkdir -p docs/operator/migrations
```

```markdown
# ADR-019 Operator Migration Guide

**Date:** 2026-05-04
**Stage:** Audit DB schema + behaviour change (Stages 2/3 of the five-stage rollout)

## What this PR does

Replaces the single-axis `RowOutcome` audit-DB recording with the two-axis
`(TerminalOutcome, TerminalPath, completed)` triple. See
[ADR-019](../../architecture/adr/019-two-axis-terminal-model.md) for the
rationale and the full mapping table.

## Before deploy

### 1. Identify affected discard-mode runs and configs

Run this against the pre-ADR-019 audit DB before deleting it. It identifies
historical runs whose status/counter interpretation changes because discard
mode is now `(FAILURE, SINK_DISCARDED)`:

```sql
SELECT
  run_id,
  COUNT(*) AS discarded_rows
FROM token_outcomes
WHERE outcome = 'diverted'
  AND sink_name = '__discard__'
GROUP BY run_id
ORDER BY discarded_rows DESC, run_id;
```

Also search pipeline configuration repositories for discard-mode sinks:

```bash
grep -rn '"__discard__"' your-pipeline-configs/
rg -n 'on_error:\s*(discard)?\s*$|on_error:\s*\|\s*$' your-pipeline-configs/
```

### 2. Review counter non-disjointness

ADR-019 preserves the public counter field names but changes which counters are
base counters versus subset counters:

- `rows_routed_success` is now a subset of `rows_succeeded`.
- `rows_routed_failure` is now a subset of `rows_failed`.
- `rows_quarantined` remains a subset of `rows_failed`.

Do **not** compute totals with
`rows_succeeded + rows_failed + rows_routed_success + rows_routed_failure +
rows_quarantined`; that double-counts routed and quarantined rows. Use
`rows_processed` for total input rows, `rows_succeeded` / `rows_failed` for
base lifecycle counts, and `rows_routed_*` / `rows_quarantined` only as
breakdowns.

Re-baseline dashboard alerts that read `rows_succeeded`, `rows_failed`, or sum
row counters manually.

### 3. Confirm stale-database failure message

If the service, CLI, or resume path opens a pre-ADR-019 audit DB after this
commit, startup/recovery must fail fast with `SchemaCompatibilityError`. The
message must name `token_outcomes.completed`, `token_outcomes.path`, and this
document path (`docs/operator/migrations/adr-019.md`). A resume-across-migration
attempt must surface that same guidance; it must not degrade into a late
`AttributeError`, SQL "no such column", or generic checkpoint failure.

## Deploy

### 1. Replace the Landscape audit database

ELSPETH does not run Alembic migrations for this project. The
`token_outcomes` table schema changes — column rename + new column + value
space change. ADR-019 does not change the web session schema; preserve
`sessions.db` unless a separate web-session compatibility check fails and this
runbook is amended with explicit session backup/restore steps.

**Permission boundary:** these commands are destructive. Agents must not run
them unless the human operator gives explicit approval for the target
environment and database paths. Review the expanded commands with the operator
before execution.

#### SQLite deployment runbook

```bash
# Fail closed: every backup verification below must succeed before any delete.
set -euo pipefail

# 0. Stop the service that writes audit data.
sudo systemctl stop elspeth-web.service

# 1. Set deployment-specific paths and capture the app revision for rollback.
export ELSPETH_DATA_DIR=/var/lib/elspeth
export ELSPETH_CHECKOUT=/home/john/elspeth
export ADR019_BACKUP_DIR=/var/backups/elspeth/adr-019-$(date -u +%Y%m%dT%H%M%SZ)

# 2. Snapshot before deleting. Keep permissions and timestamps.
sudo mkdir -p "$ADR019_BACKUP_DIR"
git -C "$ELSPETH_CHECKOUT" rev-parse HEAD > "$ADR019_BACKUP_DIR/app-ref.txt"

# SQLite is configured with journal_mode=WAL. Stop writers first, then checkpoint
# and snapshot the main DB plus sidecars when present.
sudo sqlite3 "$ELSPETH_DATA_DIR/audit.db" "PRAGMA wal_checkpoint(TRUNCATE);"
for suffix in "" "-wal" "-shm"; do
  if sudo test -e "$ELSPETH_DATA_DIR/audit.db${suffix}"; then
    sudo cp -a "$ELSPETH_DATA_DIR/audit.db${suffix}" "$ADR019_BACKUP_DIR/audit.db${suffix}"
  fi
done

# 3. Verify the backup files exist before deleting anything. These `test -s`
# commands are the hard stop before the destructive rm step.
sudo test -s "$ADR019_BACKUP_DIR/audit.db"
test -s "$ADR019_BACKUP_DIR/app-ref.txt"

# 4. Delete the old-schema audit database. Startup recreates the new schema.
sudo rm -f \
  "$ELSPETH_DATA_DIR/audit.db" \
  "$ELSPETH_DATA_DIR/audit.db-wal" \
  "$ELSPETH_DATA_DIR/audit.db-shm"

# 5. Deploy/start the ADR-019 commit and smoke-check health.
sudo systemctl start elspeth-web.service
curl -fsS https://elspeth.foundryside.dev/api/health
```

#### Postgres deployment runbook

```bash
# Fail closed: every dump verification below must succeed before any dropdb.
set -euo pipefail

# 0. Stop writers first.
sudo systemctl stop elspeth-web.service

# 1. Snapshot the audit database. Use deployment-specific DB names/roles.
export ADR019_BACKUP_DIR=/var/backups/elspeth/adr-019-$(date -u +%Y%m%dT%H%M%SZ)
export ELSPETH_CHECKOUT=/home/john/elspeth
sudo mkdir -p "$ADR019_BACKUP_DIR"
git -C "$ELSPETH_CHECKOUT" rev-parse HEAD > "$ADR019_BACKUP_DIR/app-ref.txt"
pg_dump --format=custom --file="$ADR019_BACKUP_DIR/elspeth_audit.dump" elspeth_audit
# These `test -s` commands are the hard stop before the destructive dropdb step.
test -s "$ADR019_BACKUP_DIR/elspeth_audit.dump"
test -s "$ADR019_BACKUP_DIR/app-ref.txt"

# 2. Drop/recreate the old-schema audit store. Adjust ownership/permissions to match deployment.
dropdb elspeth_audit
createdb elspeth_audit

# 3. Deploy/start the ADR-019 commit and smoke-check health.
sudo systemctl start elspeth-web.service
curl -fsS https://elspeth.foundryside.dev/api/health
```

The engine's startup `metadata.create_all()` recreates the new schema on empty
stores.

**No data migration is offered.** Pre-ADR-019 audit data is a different
shape and cannot be losslessly converted to the new model. Keep the backup
artifacts until the release is accepted and rollback is no longer required.

## After deploy

### 1. Verify the deployment

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

### 2. Re-baseline behaviour changes

#### Discard-mode `RunStatus` flip

Pipelines using discard-mode sinks (`sink_name="__discard__"`) will see
their `RunStatus` flip:

| Pre-ADR-019 | Post-ADR-019 |
| --- | --- |
| Some discards + some success → `COMPLETED` | → `COMPLETED_WITH_FAILURES` |
| All rows discarded → `COMPLETED` | → `FAILED` |

Per ADR-019 § Sub-decision 5: discard-mode is reclassified as a
predicate-input `(FAILURE, SINK_DISCARDED)`. The discarded rows now
increment `rows_failed` and flip `failure_indicator`.

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

#### Counter changes — `(SUCCESS, GATE_ROUTED)` and `(FAILURE, ON_ERROR_ROUTED)`

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
counter values are unchanged, but they are no longer disjoint from the base
counters.

## Rollback

Rollback requires restoring the previous application commit **and** restoring
the pre-ADR-019 audit database snapshot. Do not run old code against the new schema
or new code against the old schema.

For source-checkout deployments, set `ADR019_PREVIOUS_REF` from the backup's
`app-ref.txt` or an equivalent release tag. Before switching the checkout,
verify the checkout is clean; if it is not clean, stop and use the deployment
system's normal artifact rollback instead of discarding local work.

### SQLite rollback

```bash
set -euo pipefail

# Stop writers, restore the previous application commit, restore DB snapshots,
# then restart and health-check.
export ELSPETH_DATA_DIR=/var/lib/elspeth
export ELSPETH_CHECKOUT=/home/john/elspeth
export ADR019_PREVIOUS_REF="$(cat "$ADR019_BACKUP_DIR/app-ref.txt")"

sudo systemctl stop elspeth-web.service
test -n "$ADR019_PREVIOUS_REF"
test -z "$(git -C "$ELSPETH_CHECKOUT" status --porcelain)"
git -C "$ELSPETH_CHECKOUT" switch --detach "$ADR019_PREVIOUS_REF"
test "$(git -C "$ELSPETH_CHECKOUT" rev-parse HEAD)" = \
  "$(git -C "$ELSPETH_CHECKOUT" rev-parse "$ADR019_PREVIOUS_REF")"

sudo test -s "$ADR019_BACKUP_DIR/audit.db"
sudo rm -f \
  "$ELSPETH_DATA_DIR/audit.db" \
  "$ELSPETH_DATA_DIR/audit.db-wal" \
  "$ELSPETH_DATA_DIR/audit.db-shm"
for suffix in "" "-wal" "-shm"; do
  if sudo test -e "$ADR019_BACKUP_DIR/audit.db${suffix}"; then
    sudo cp -a "$ADR019_BACKUP_DIR/audit.db${suffix}" "$ELSPETH_DATA_DIR/audit.db${suffix}"
  fi
done
sudo test -s "$ELSPETH_DATA_DIR/audit.db"
sudo systemctl start elspeth-web.service
curl -fsS https://elspeth.foundryside.dev/api/health
```

### Postgres rollback

```bash
set -euo pipefail

sudo systemctl stop elspeth-web.service
export ELSPETH_CHECKOUT=/home/john/elspeth
export ADR019_PREVIOUS_REF="$(cat "$ADR019_BACKUP_DIR/app-ref.txt")"

# Restore the previous application commit before restoring pre-ADR-019 data.
test -n "$ADR019_PREVIOUS_REF"
test -z "$(git -C "$ELSPETH_CHECKOUT" status --porcelain)"
git -C "$ELSPETH_CHECKOUT" switch --detach "$ADR019_PREVIOUS_REF"
test "$(git -C "$ELSPETH_CHECKOUT" rev-parse HEAD)" = \
  "$(git -C "$ELSPETH_CHECKOUT" rev-parse "$ADR019_PREVIOUS_REF")"

test -s "$ADR019_BACKUP_DIR/elspeth_audit.dump"
dropdb elspeth_audit
createdb elspeth_audit
pg_restore --dbname=elspeth_audit "$ADR019_BACKUP_DIR/elspeth_audit.dump"
sudo systemctl start elspeth-web.service
curl -fsS https://elspeth.foundryside.dev/api/health
```

## Related work

- **Stage 4 cleanup** — no pytest-blocking assertions remain for it. If kept,
  this follow-up may remove non-blocking compatibility fixtures or commentary
  references, but it is not part of this deploy gate.

- **Stage 5** (delete RowOutcome) — final sweep; deletes the enum and the
  lint guard. Tracked in `elspeth-774b1d3c2e`.

- **ADR-020** (potential counter rename) — separate breaking-API
  conversation. Not coupled to this PR.
```

**Step 2: Verify the doc renders cleanly**

```bash
test -f docs/operator/migrations/adr-019.md
grep -Fq "ADR-019 Operator Migration Guide" docs/operator/migrations/adr-019.md
grep -Fq "../../architecture/adr/019-two-axis-terminal-model.md" docs/operator/migrations/adr-019.md
grep -Fq "## Before deploy" docs/operator/migrations/adr-019.md
grep -Fq "## Deploy" docs/operator/migrations/adr-019.md
grep -Fq "## After deploy" docs/operator/migrations/adr-019.md
grep -Fq "## Rollback" docs/operator/migrations/adr-019.md
grep -Fq "SchemaCompatibilityError" docs/operator/migrations/adr-019.md
grep -Fq "docs/operator/migrations/adr-019.md" docs/operator/migrations/adr-019.md
grep -Fq "WHERE outcome = 'diverted'" docs/operator/migrations/adr-019.md
grep -Fq "Do **not** compute totals" docs/operator/migrations/adr-019.md
echo OK
```

This project does not currently depend on the Python `markdown` package; do not add a verification gate that requires undeclared tooling. If markdown lint is configured in pre-commit by the time this plan is executed, run that configured repo gate against the new file.

**Definition of Done:**
- [ ] `docs/operator/migrations/adr-019.md` expanded from the Phase 1 stub into the full runbook
- [ ] Migration doc has explicit `Before deploy`, `Deploy`, `After deploy`, and `Rollback` sections
- [ ] Migration doc includes a pre-deploy SQL query for historical discard-mode runs
- [ ] Migration doc warns that routed/quarantined counters are non-disjoint subsets and must not be summed with base counters
- [ ] Migration doc states that stale DB and resume-across-migration failures raise `SchemaCompatibilityError` pointing at `docs/operator/migrations/adr-019.md`
- [ ] Migration doc includes SQLite and Postgres rollback steps with application-version + DB-snapshot coupling
- [ ] SQLite backup/rollback commands handle `audit.db`, `audit.db-wal`, and `audit.db-shm` consistently after writers are stopped
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
    justification: ADR-019 migration site — defines RowOutcome until Stage 5; also defines TerminalOutcome / TerminalPath.
  - file: src/elspeth/testing/__init__.py
    justification: testing pack re-exports RowOutcome alongside TerminalOutcome/TerminalPath; Stage 5 trims to TerminalOutcome/TerminalPath only.
  - file: tests/
    justification: compatibility fixtures, mapping-table tests, and commentary references only; no real-engine-output `outcome == RowOutcome.X` assertions may remain after Phase 5.
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

Expected: `exit 0`. If any src/ file outside the final 3-entry allowlist shape
(two src entries plus `tests/`) contains `RowOutcome.X`, fix it (it was missed
in Phases 1-4) or surface to user if root cause is unclear.

**Definition of Done:**
- [ ] Allowlist trimmed to its final 3-entry shape
- [ ] Lint guard passes

---

### Task 5.6: Final verification + Phase 5 commit + PR open

**Execution order:** complete Task 5.7 before running this task. Task 5.7 is
documented after the PR section because it extends the existing FNR guard, but it
is a prerequisite for final verification, commit, and PR open. Do not run the
final gate set below until Task 5.7's Definition of Done is complete.

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
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check --root src/elspeth --allowlist config/cicd/adr019_symbol_inventory
cd src/elspeth/web/frontend
npm run test
npm run build
cd /home/john/elspeth
```

ALL must pass.

**Step 2: Confirm the new ADR-019 behavioural and guard fixtures are GREEN**

```bash
.venv/bin/python -m pytest \
  tests/integration/test_adr_019_discard_mode_flip.py \
  tests/integration/test_adr_019_counter_changes.py \
  tests/integration/test_adr_019_cross_table_invariants.py \
  tests/integration/test_adr_019_sweep_durability.py \
  tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
  tests/unit/scripts/cicd/test_forbid_new_row_outcome.py \
  -v
```

**Step 3: Commit**

```bash
git add tests/ docs/operator/migrations/adr-019.md \
        config/cicd/forbid_new_row_outcome/migration_files.yaml \
        scripts/cicd/forbid_new_row_outcome.py \
        tests/unit/scripts/cicd/test_forbid_new_row_outcome.py \
        scripts/cicd/adr019_symbol_inventory.py \
        config/cicd/adr019_symbol_inventory \
        tests/unit/scripts/cicd/test_adr019_symbol_inventory.py

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
translated in this PR to two-axis `(TerminalOutcome, TerminalPath)` assertions.
No ADR-019 Stage 4 xfails remain.

Operator-facing migration documentation at docs/operator/migrations/adr-019.md:
- DB delete commands (no Alembic per project policy)
- Discard-mode RunStatus flip + how to preserve old behaviour
- Counter changes for rows_succeeded / rows_failed
- Verification checklist for post-deploy
- Fail-closed backup/rollback commands for SQLite and Postgres

Guard updates:
- ADR-019 AST symbol inventory runs as a closeout gate for source wire-contract drift.
- forbid_new_row_outcome.py now detects hardcoded RowOutcome value-string comparisons in src/.

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

**Operator action required at deploy time:** replace the Landscape audit store per `docs/operator/migrations/adr-019.md`. ELSPETH does not run Alembic migrations for this project. ADR-019 does not require deleting `sessions.db`.

## Phase structure (3 commits)

1. Atomic Phases 1-3 — schema/recorder/loader/dataclasses, producer site flip, accumulator/predicate/resume changes
2. Phase 4 — cross-table invariants I1a, I1b, I1c, I3
3. Phase 5 — schema-dependent + assertion-only test fixes + operator migration doc

The merge is atomic per ADR-019 lines 318-320 — accumulator change ships in lockstep with the predicate rewrite. Squash-or-keep at merge time is reviewer's choice.

## What's NOT in this PR

- Deletion of `RowOutcome` itself — Stage 5 ticket `elspeth-774b1d3c2e`.
- Counter renames — separate ADR-020 conversation if pursued.

## Test plan

- [x] All ~16,010 tests pass (verified locally; `pytest tests/ -q --timeout=120`)
- [x] ADR-019 behavioural integration test files green:
  - `tests/integration/test_adr_019_discard_mode_flip.py`
  - `tests/integration/test_adr_019_counter_changes.py`
  - `tests/integration/test_adr_019_cross_table_invariants.py`
  - `tests/integration/test_adr_019_sweep_durability.py`
- [x] mypy clean across 364 src files
- [x] ruff lint + format clean
- [x] All project gates pass (tier model, contract manifest, plugin hashes, freeze guards, frozen annotations, forbid-new-row-outcome)
- [x] Lint guard `forbid_new_row_outcome.py` passes including hardcoded RowOutcome value-string detection; src/ scope reaches zero RowOutcome references; tests/ retain only deliberate compatibility/mapping references
- [x] ADR-019 AST symbol inventory passes with only deliberate migration-window allowlist entries
- [x] Frontend session terminal-status gates pass: `npm run test` and `npm run build`

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
- [ ] Stage 1 and Stage 5 ticket cross-references in PR description

---

### Task 5.7: Extend forbid_new_row_outcome.py to detect hardcoded RowOutcome value strings

**Context:** `src/elspeth/mcp/analyzers/diagnostics.py:181` contains `outcome == "quarantined"` — a string-literal comparison against a known RowOutcome value. Phase 1 fixes this specific site, but the existing lint guard (`forbid_new_row_outcome.py`) only detects `RowOutcome.X` AST attribute accesses. A future contributor could re-introduce a hardcoded value string without using `RowOutcome.X` syntax and the guard would not catch it. This task extends the guard to cover that pattern during the remaining RowOutcome-retention window.

**Lifecycle note:** This extension is explicitly bounded. `forbid_new_row_outcome.py` is deleted in Stage 5 (out-of-scope entry in the overview; script docstring line 25 states: "Stage 5 (post-migration) deletes both `RowOutcome` itself and this script"). Between this PR and Stage 5, however, the gate IS load-bearing — source scope must stay at zero RowOutcome references, and a hardcoded string bypass would silently defeat that guarantee.

**Files:**
- Modify: `scripts/cicd/forbid_new_row_outcome.py`
- Modify: `config/cicd/forbid_new_row_outcome/migration_files.yaml` (allowlist, if any src/ sites need listing)
- Create: `tests/unit/scripts/cicd/test_forbid_new_row_outcome.py`

**Step 1: Identify the known RowOutcome value strings**

The full set of `RowOutcome` string values (from `src/elspeth/contracts/enums.py`):

```python
_ROW_OUTCOME_VALUE_STRINGS: frozenset[str] = frozenset({
    "completed",
    "routed",
    "routed_on_error",
    "forked",
    "failed",
    "quarantined",
    "diverted",
    "consumed_in_batch",
    "dropped_by_filter",
    "coalesced",
    "expanded",
    "buffered",
})
```

**Step 2: Add regex detection for string-literal comparisons**

The pattern to detect is a binary comparison where one operand is the name `outcome` (or a field access ending in `.outcome`) and the other is a string literal whose value is in `_ROW_OUTCOME_VALUE_STRINGS`. Two canonical forms:

```python
outcome == "quarantined"          # forward
"quarantined" == outcome          # reversed
row.outcome == "completed"        # attribute on left
"completed" == row.outcome        # attribute on right
```

Detection approach: extend the script with a second AST visitor (`_HardcodedValueVisitor`) that walks `ast.Compare` nodes and checks:

1. Left operand is `ast.Name(id="outcome")` or `ast.Attribute(attr="outcome")`, AND right comparator is `ast.Constant(value=v)` where `v in _ROW_OUTCOME_VALUE_STRINGS`.
2. OR: left operand is `ast.Constant(value=v)` where `v in _ROW_OUTCOME_VALUE_STRINGS`, AND right comparator is `ast.Name(id="outcome")` or `ast.Attribute(attr="outcome")`.

Use `ast.Compare` only — do not flag `in` membership tests (`ast.In`) to avoid false positives on legitimate `outcome in {set_of_strings}` guards in non-migration code.

Define a second rule constant alongside the existing `RULE_ID = "FNR1"`:

```python
RULE_ID_2 = "FNR2"
RULE_NAME_2 = "no-hardcoded-row-outcome-value-string"
RULE_DESCRIPTION_2 = (
    "String-literal comparison against a known RowOutcome value — use "
    "(TerminalOutcome, TerminalPath) pairs in new code, or add this file "
    "to the migration allowlist with a justification."
)
```

Scope the FNR2 check to `src/elspeth/` only (exclude `tests/` and `scripts/` from FNR2 — tests legitimately compare strings, and the script itself imports the value list). Pass the scope as a `--src-root` flag (default `src/elspeth`) or hardcode a secondary path filter inside the check function.

**Step 3: Integrate with the existing allowlist mechanism**

The existing allowlist at `config/cicd/forbid_new_row_outcome/migration_files.yaml` uses a `file:` / `justification:` YAML structure. FNR2 findings respect the same allowlist — if a file is listed, both FNR1 and FNR2 violations are suppressed for that file. No structural changes to the allowlist format are needed.

After Phase 1 fixes `src/elspeth/mcp/analyzers/diagnostics.py:181`, the allowlist should contain ZERO entries from `src/elspeth/mcp/`. Confirm this before closing the Definition of Done.

**Step 4: Write a unit test for the new rule**

Test file: `tests/unit/scripts/cicd/test_forbid_new_row_outcome.py`

Follow the pattern established by the existing CICD script tests in `tests/unit/scripts/cicd/` (they construct a fake file tree under `tmp_path`, write Python source snippets, and invoke the script's check function directly).

Minimum test cases for FNR2:

```python
# FAILS: forward comparison
outcome == "quarantined"

# FAILS: reversed comparison
"quarantined" == outcome

# FAILS: attribute left
row.outcome == "completed"

# PASSES: not a RowOutcome value string
outcome == "active"

# PASSES: membership test (not a Compare with Eq — use ast.In, not flagged)
outcome in {"completed", "failed"}

# PASSES: file in allowlist → suppressed
```

Also confirm that FNR1 (existing `RowOutcome.X` attribute detection) still passes its existing test cases after the extension — the two visitors are independent.

**Step 5: Verify the guard in src/ scope**

After Phase 1 lands, run:

```bash
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check \
  --root . --allowlist config/cicd/forbid_new_row_outcome
```

Expected: exit 0 with zero FNR2 findings in `src/elspeth/`. If `src/elspeth/mcp/analyzers/diagnostics.py` still appears, Phase 1 missed the fix — surface to user.

**Definition of Done:**
- [ ] `_HardcodedValueVisitor` added to `forbid_new_row_outcome.py`; FNR2 rule defined
- [ ] FNR2 scoped to `src/elspeth/` only; `tests/` and `scripts/` excluded
- [ ] Existing allowlist mechanism covers FNR2 findings without format changes
- [ ] `tests/unit/scripts/cicd/test_forbid_new_row_outcome.py` created and passes
- [ ] `forbid_new_row_outcome.py` check returns exit 0 with zero FNR2 hits in `src/elspeth/`
- [ ] Allowlist contains zero entries from `src/elspeth/mcp/` (Phase 1 fixed the only known site)
