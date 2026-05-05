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
- Create/use: `scripts/cicd/adr019_test_inventory.py` — tests-only AST-backed inventory for RowOutcome comparisons, raw `token_outcomes` SQL, `is_terminal` schema reads, and hardcoded old outcome strings in tests
- Create/use: `config/cicd/adr019_test_inventory/` — temporary allowlist directory for deliberate compatibility/mapping/commentary test fixtures
- Test: `tests/unit/scripts/cicd/test_adr019_test_inventory.py`
- Modify: `docs/operator/migrations/adr-019.md` — operator-facing migration guide stub created in Phase 1; expand to full runbook here
- Modify: `config/cicd/forbid_new_row_outcome/migration_files.yaml` — final allowlist trim (after this PR, only `contracts/enums.py`, `testing/__init__.py`, `tests/` remain)
- Test: full suite green

**Background reading:** Phase 3 introduced two RED-first integration tests. Phase 4 added six. This phase ensures the rest of the suite compiles and passes, including the mechanical translation of `outcome == RowOutcome.X` assertion sites that would otherwise fail after the contract retype.

---

## The AST-backed triage recipe

The triage rule is mechanical. Grep/ripgrep remains useful for quick local
slices, but both source-closeout and test-closeout gates are AST-backed. This is
load-bearing: grep-only sweeps missed
`contracts/export_records.py::TokenOutcomeExportRecord`, and exact-pattern test
grep misses raw SQL, `actual == RowOutcome.X`, list equality, membership checks,
and hardcoded old outcome strings. Do not close Phase 5 from grep output alone.

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

Quick local slice:

```bash
rg -n --glob '*.py' "TokenOutcome\(|RowResult\(|PendingOutcome\(|TokenCompleted\(" tests/ \
  | sort -u > /tmp/category-a.txt

wc -l /tmp/category-a.txt
```

### Category B — Assertion/comparison-only (MUST move with this PR)

A test is assertion/comparison-only if it reads a real engine/audit output and
compares it to old single-axis vocabulary without constructing the dataclass.
This includes exact equality, reversed equality, list/tuple equality,
membership, set membership, and hardcoded old value strings:

```python
# Assertion/comparison-only — mechanical translation required before PR open
assert result.outcome == RowOutcome.COMPLETED
assert outcome.outcome == RowOutcome.QUARANTINED
actual = RowOutcome(row.outcome)
assert actual == RowOutcome.CONSUMED_IN_BATCH
assert outcomes == [RowOutcome.BUFFERED, RowOutcome.FAILED]
assert RowOutcome.COMPLETED in outcome_values
assert row.outcome == "routed"
```

The AST test inventory in Task 5.0a is authoritative for Category B. For quick
local slices only, use source-only ripgrep patterns that avoid `__pycache__` and
binary `.pyc` noise:

```bash
rg -n --glob '*.py' \
  "RowOutcome\.|outcome\s*(==|!=|in|not in)|path\s*(==|!=|in|not in)|\"(completed|routed|routed_on_error|forked|failed|quarantined|diverted|consumed_in_batch|dropped_by_filter|coalesced|expanded|buffered)\"" \
  tests/ \
  | sort -u > /tmp/category-b-rg-triage.txt

wc -l /tmp/category-b-rg-triage.txt
```

Translate every Category B assertion in this PR. Do not add xfail markers and do
not create a Stage 4 manifest for these sites. Cross-enum equality is a normal
`False` comparison, so these tests fail with `AssertionError`; deferring them
would make the full-suite green gate impossible.

### Category C — Audit-DB direct read (must move with this PR)

A test that reads `token_outcomes` rows directly via SQL (rather than through the loader) and checks `outcome` / `is_terminal` column values is schema-dependent — those columns no longer exist with the old names/values.

Identify with AST inventory plus a source-only ripgrep slice:

```bash
rg -n --glob '*.py' "is_terminal|token_outcomes_table|token_outcomes|SELECT .*outcome|SELECT .*is_terminal" tests/ \
  | sort -u > /tmp/category-c-rg-triage.txt

head -30 /tmp/category-c-rg-triage.txt
```

Each hit needs evaluation: if it consults the renamed `completed` column or new `path` column semantics, fix it; if it just queries existence, no change needed.

### Category D — Source wire-contract / schema-field sweep (must move with this PR)

The downstream-consumer sweep is not test-only. It must also catch source-level
wire contracts and TypedDict declarations that never touch
`token_outcomes_table.c.is_terminal` directly. This is the class that previously
missed `contracts/export_records.py::TokenOutcomeExportRecord`.

Quick local source-wide slices:

```bash
rg -n --glob '*.py' "is_terminal: bool" src/elspeth/ \
  | rg -v "src/elspeth/contracts/audit.py" \
  | sort -u > /tmp/category-d-typedicts.txt

rg -n --glob '*.py' "['\\\"]is_terminal['\\\"]" src/elspeth/ \
  | rg -v "src/elspeth/contracts/audit.py" \
  | sort -u > /tmp/category-d-dict-keys.txt

rg -n --glob '*.py' "token_outcomes_table\\.c\\.is_terminal|\\.is_terminal" src/elspeth/ \
  | rg -v "src/elspeth/contracts/audit.py" \
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
    ROW_OUTCOME_STRING_MEMBERSHIP = "row_outcome_string_membership"
    TERMINAL_OUTCOME_STRING_MEMBERSHIP = "terminal_outcome_string_membership"
    TERMINAL_PATH_STRING_MEMBERSHIP = "terminal_path_string_membership"
```

The terminal string checks are not optional. The migration must not merely
replace `outcome == "quarantined"` with `path == "quarantined_at_source"` or
`outcome == "failure"`; those are the same fragility class. The visitor should
flag equality / inequality comparisons and membership checks (`in` /
`not in`) where either side is a known `RowOutcome`, `TerminalOutcome`, or
`TerminalPath` value string outside `contracts/enums.py` and explicitly
allowlisted fixtures.

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
outcome in {"completed", "failed"}    # RowOutcome/TerminalOutcome membership finding
path in {"default_flow"}              # TerminalPath membership finding

terminal = True                       # no finding
payload = {"completed": True}         # no finding
status in {"completed", "failed"}     # no finding; not outcome/path vocabulary
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
- [ ] Script detects annotations, attributes, kwargs, dict keys, hardcoded RowOutcome value comparisons/memberships, and hardcoded TerminalOutcome/TerminalPath value comparisons/memberships
- [ ] Script exits non-zero for non-allowlisted findings
- [ ] Phase 5 closeout uses this script; grep-only D7 checks are not sufficient for approval

---

### Task 5.0a: Create and run AST-backed test migration inventory

**Why this task exists:** The source inventory does not cover every pytest
failure class. Phase 5 also needs a tests-only inventory that finds old
single-axis expectations even when they are not written as
`outcome == RowOutcome.X`. Live examples include raw SQL
`SELECT outcome FROM token_outcomes`, `result[0] == RowOutcome.FORKED`,
`actual == RowOutcome.CONSUMED_IN_BATCH`, list equality with RowOutcome
members, membership checks, and hardcoded string checks such as
`row.outcome == "routed"`.

**Files:**
- Create: `scripts/cicd/adr019_test_inventory.py`
- Create: `config/cicd/adr019_test_inventory/migration_files.yaml`
- Test: `tests/unit/scripts/cicd/test_adr019_test_inventory.py`

**Step 1: Write the RED tests first**

Before creating or changing `scripts/cicd/adr019_test_inventory.py`, add
`tests/unit/scripts/cicd/test_adr019_test_inventory.py` with fixture snippets
that fail under the missing tool. Minimum finding kinds:

```python
class TestFindingKind(StrEnum):
    ROW_OUTCOME_ATTRIBUTE = "row_outcome_attribute"      # RowOutcome.X anywhere in tests
    ROW_OUTCOME_COMPARE = "row_outcome_compare"          # actual == RowOutcome.X / reversed
    ROW_OUTCOME_COLLECTION = "row_outcome_collection"    # [RowOutcome.X], {RowOutcome.X}
    ROW_OUTCOME_MEMBERSHIP = "row_outcome_membership"    # RowOutcome.X in outcome_values / actual in {...}
    OLD_OUTCOME_STRING_COMPARE = "old_outcome_string_compare"
    OLD_OUTCOME_STRING_MEMBERSHIP = "old_outcome_string_membership"
    RAW_TOKEN_OUTCOMES_SQL = "raw_token_outcomes_sql"    # SQL text mentioning token_outcomes/outcome/is_terminal
    TOKEN_OUTCOMES_SCHEMA_READ = "token_outcomes_schema_read"  # token_outcomes_table.c.is_terminal/outcome
```

Minimum positive fixtures:

```python
assert result[0] == RowOutcome.FORKED
actual = RowOutcome(row.outcome)
assert actual == RowOutcome.CONSUMED_IN_BATCH
assert outcomes == [RowOutcome.BUFFERED, RowOutcome.FAILED]
assert RowOutcome.COMPLETED in outcome_values
assert row.outcome == "routed"
assert row.outcome in {"completed", "failed"}
text("SELECT outcome FROM token_outcomes WHERE token_id = :token_id")
select(token_outcomes_table.c.outcome).where(token_outcomes_table.c.is_terminal == 1)
```

Minimum negative fixtures:

```python
assert result.outcome == TerminalOutcome.SUCCESS
assert result.path == TerminalPath.DEFAULT_FLOW
assert status in {"completed", "failed"}  # not an outcome/path symbol
payload = {"completed": True}
```

Run the new test before implementation and confirm it fails because the tool is
missing or lacks the required finding kinds:

```bash
.venv/bin/python -m pytest tests/unit/scripts/cicd/test_adr019_test_inventory.py -v
```

Expected RED: failure before the tool exists or before it detects the required
finding kinds.

**Step 2: Implement the tests inventory**

Create `scripts/cicd/adr019_test_inventory.py` using `ast.parse`, matching the
report shape of `adr019_symbol_inventory.py`: JSON-line findings with `kind`,
`path`, `line`, `col`, `symbol`, and `context`; `check` exits non-zero for
non-allowlisted findings. It must scan `tests/` by default and ignore hidden
directories, `__pycache__`, `node_modules`, `build`, and `dist`.

The allowlist format is the same directory-form YAML style as the other CICD
guards. The initial allowlist may cover only deliberate compatibility and
mapping-table fixtures, for example `tests/unit/contracts/test_enums.py`, with
line-level rationale in the YAML justification. Do not allowlist whole
`tests/` for this guard.

**Step 3: Use the inventory as the test closeout gate**

After Category A/B/C edits land, run:

```bash
.venv/bin/python scripts/cicd/adr019_test_inventory.py check \
  --root tests \
  --allowlist config/cicd/adr019_test_inventory
```

Expected: zero findings outside deliberate compatibility/mapping/commentary
fixtures. If the tool reports raw SQL, `is_terminal`, arbitrary RowOutcome
comparisons, membership checks, or old outcome strings in a real engine-output
test, patch that test in this PR.

**Definition of Done:**
- [ ] RED test for `adr019_test_inventory.py` fails before implementation
- [ ] Inventory detects raw SQL, SQLAlchemy schema reads, arbitrary RowOutcome attribute/comparison/collection/membership uses, and hardcoded old outcome string compare/membership uses in tests
- [ ] Inventory ignores non-outcome strings and already migrated `(TerminalOutcome, TerminalPath)` assertions
- [ ] Allowlist is narrow and does not allowlist all `tests/`
- [ ] CLI `check` exits non-zero for non-allowlisted findings
- [ ] Phase 5 closeout uses this tool; exact grep patterns alone are not sufficient

---

### Task 5.1: Run the triage inventory

**Step 1: Generate the category lists**

```bash
mkdir -p /tmp/adr-019-triage
rg -n --glob '*.py' "TokenOutcome\(|RowResult\(|PendingOutcome\(|TokenCompleted\(" tests/ \
  > /tmp/adr-019-triage/category-a.txt
rg -n --glob '*.py' "RowOutcome\.|outcome\s*(==|!=|in|not in)|path\s*(==|!=|in|not in)" tests/ \
  > /tmp/adr-019-triage/category-b.txt
rg -n --glob '*.py' "is_terminal|token_outcomes_table|token_outcomes|SELECT .*outcome|SELECT .*is_terminal" tests/ \
  > /tmp/adr-019-triage/category-c.txt
rg -n --glob '*.py' "is_terminal: bool" src/elspeth/ \
  | rg -v "src/elspeth/contracts/audit.py" \
  > /tmp/adr-019-triage/category-d-typedicts.txt
rg -n --glob '*.py' "['\\\"]is_terminal['\\\"]" src/elspeth/ \
  | rg -v "src/elspeth/contracts/audit.py" \
  > /tmp/adr-019-triage/category-d-dict-keys.txt
rg -n --glob '*.py' "token_outcomes_table\\.c\\.is_terminal|\\.is_terminal" src/elspeth/ \
  | rg -v "src/elspeth/contracts/audit.py" \
  > /tmp/adr-019-triage/category-d-accessors.txt

.venv/bin/python scripts/cicd/adr019_test_inventory.py check \
  --root tests \
  --allowlist config/cicd/adr019_test_inventory \
  > /tmp/adr-019-triage/test-inventory.jsonl || true

wc -l /tmp/adr-019-triage/*.txt
wc -l /tmp/adr-019-triage/test-inventory.jsonl
```

**Step 2: Record live counts**

The counts in this document are historical estimates and must not be copied into
the PR as verified facts. Record the live counts from the commands above in the
PR notes. During the 2026-05-05 plan review, source-only counts were materially
different from older estimates (`RowOutcome.X` under tests was about 485;
exact `outcome == RowOutcome.X` was about 87; Category C was about 201), so the
executor must trust the live inventory, not stale prose.

If the AST inventory reports zero findings before any Category A/B/C edits, STOP
and fix the inventory; that is more likely a scanner miss than a clean suite.

**Step 3: Confirm affected files collect under pytest**

For each file named by the category lists and the AST inventory, use pytest
collection instead of direct `importlib` execution. Direct imports bypass pytest
configuration, strict markers, fixtures, and collection behavior.

```bash
awk -F: '{print $1}' /tmp/adr-019-triage/category-a.txt /tmp/adr-019-triage/category-b.txt /tmp/adr-019-triage/category-c.txt \
  | sort -u > /tmp/adr-019-triage/affected-files.txt
jq -r '.path' /tmp/adr-019-triage/test-inventory.jsonl 2>/dev/null \
  | sort -u >> /tmp/adr-019-triage/affected-files.txt
sort -u -o /tmp/adr-019-triage/affected-files.txt /tmp/adr-019-triage/affected-files.txt

if test -s /tmp/adr-019-triage/affected-files.txt; then
  xargs -a /tmp/adr-019-triage/affected-files.txt \
    .venv/bin/python -m pytest --collect-only -q
fi
```

Files that fail to collect are in scope for this phase and must be fixed before
the full-suite gate.

**Definition of Done:**
- [ ] Category lists generated
- [ ] AST test inventory generated and reviewed
- [ ] Category D source-wire sweeps generated, with no directory-level `/contracts/` exclusion
- [ ] Counts recorded from live scans, not copied from stale estimates
- [ ] Expected Category A files identified
- [ ] Affected files collect under pytest

---

### Task 5.2: Update Category A and B tests (constructor flips + assertion translation)

**Files:** every file listed by Task 5.1 and `adr019_test_inventory.py`,
including constructor-heavy files under `tests/unit/contracts/`,
`tests/unit/core/landscape/`, `tests/unit/engine/`, `tests/integration/`,
`tests/property/`, and `tests/e2e/`. Do not assume property/e2e tests are
unaffected; live review found old schema/value expectations there.

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

**Step 2: Translate Category B assertion/comparison-only sites**

For each Category B finding from `/tmp/adr-019-triage/category-b.txt` and
`adr019_test_inventory.py`, translate old single-axis expectations to new
two-axis assertions using
`tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`:

```python
# OLD:
assert result.outcome == RowOutcome.ROUTED_ON_ERROR

# NEW:
assert result.outcome == TerminalOutcome.FAILURE
assert result.path == TerminalPath.ON_ERROR_ROUTED
```

For list, set, membership, and raw SQL cases, assert against the actual
`TerminalOutcome` and `TerminalPath` values from the production result object or
database row:

```python
# OLD:
assert RowOutcome.COMPLETED in outcome_values

# NEW:
assert (TerminalOutcome.SUCCESS.value, TerminalPath.DEFAULT_FLOW.value) in {
    (row.outcome, row.path) for row in outcome_rows
}
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
if test -s /tmp/adr-019-triage/affected-files.txt; then
  xargs -a /tmp/adr-019-triage/affected-files.txt \
    .venv/bin/python -m pytest -q
fi

.venv/bin/python -m pytest tests/property/ tests/e2e/ -q
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
# Staging is a source-checkout deployment. Its systemd unit runs from
# /home/john/elspeth and grants write access to /home/john/elspeth/data.
export ELSPETH_CHECKOUT=/home/john/elspeth
export ELSPETH_DATA_DIR=/home/john/elspeth/data
export ELSPETH_AUDIT_DB="$ELSPETH_DATA_DIR/runs/audit.db"
export ADR019_BACKUP_PARENT=/var/backups/elspeth
export ADR019_BACKUP_DIR="$ADR019_BACKUP_PARENT/adr-019-$(date -u +%Y%m%dT%H%M%SZ)"

# If deploy/elspeth-web.env overrides ELSPETH_WEB__LANDSCAPE_URL, resolve that
# value first and set ELSPETH_AUDIT_DB to the exact sqlite file path. Do not
# print secret values from the env file while inspecting it.

# 2. Fail closed unless the target DB is present and definitely pre-ADR-019.
# sqlite3 can create a missing path unless guarded first.
sudo test -s "$ELSPETH_AUDIT_DB"
if sudo sqlite3 -readonly "$ELSPETH_AUDIT_DB" \
  "SELECT 1 FROM pragma_table_info('token_outcomes') WHERE name='completed';" \
  | grep -qx 1; then
  echo "Audit DB already has token_outcomes.completed; aborting destructive ADR-019 replace."
  exit 1
fi
sudo sqlite3 -readonly "$ELSPETH_AUDIT_DB" \
  "SELECT 1 FROM pragma_table_info('token_outcomes') WHERE name='is_terminal';" \
  | grep -qx 1

# 3. Snapshot before deleting. Keep permissions and timestamps. Use a unique
# backup directory and make it writable by the deployment operator before shell
# redirections write into it.
sudo install -d -m 0750 -o john -g john "$ADR019_BACKUP_PARENT"
test ! -e "$ADR019_BACKUP_DIR"
sudo install -d -m 0750 -o john -g john "$ADR019_BACKUP_DIR"
git -C "$ELSPETH_CHECKOUT" rev-parse HEAD > "$ADR019_BACKUP_DIR/app-ref.txt"

# SQLite is configured with journal_mode=WAL. Stop writers first, then checkpoint
# and snapshot the main DB plus sidecars when present.
sudo sqlite3 "$ELSPETH_AUDIT_DB" "PRAGMA wal_checkpoint(TRUNCATE);"
for suffix in "" "-wal" "-shm"; do
  if sudo test -e "$ELSPETH_AUDIT_DB${suffix}"; then
    sudo cp -a "$ELSPETH_AUDIT_DB${suffix}" "$ADR019_BACKUP_DIR/audit.db${suffix}"
  fi
done

# 4. Verify the backup files exist before deleting anything. These `test -s`
# commands are the hard stop before the destructive rm step.
sudo test -s "$ADR019_BACKUP_DIR/audit.db"
test -s "$ADR019_BACKUP_DIR/app-ref.txt"

# 5. Delete the old-schema audit database. Startup recreates the new schema.
sudo rm -f \
  "$ELSPETH_AUDIT_DB" \
  "$ELSPETH_AUDIT_DB-wal" \
  "$ELSPETH_AUDIT_DB-shm"

# 6. Deploy/start the ADR-019 commit and smoke-check health.
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
sudo install -d -m 0750 -o john -g john /var/backups/elspeth
test ! -e "$ADR019_BACKUP_DIR"
sudo install -d -m 0750 -o john -g john "$ADR019_BACKUP_DIR"
git -C "$ELSPETH_CHECKOUT" rev-parse HEAD > "$ADR019_BACKUP_DIR/app-ref.txt"

# Abort unless the DB is definitely pre-ADR-019 and not already current.
psql -d elspeth_audit -Atc \
  "SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'token_outcomes' AND column_name = 'is_terminal')
      AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'token_outcomes' AND column_name = 'completed');" \
  | grep -qx t

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
sqlite3 /home/john/elspeth/data/runs/audit.db ".schema token_outcomes" | grep -E "completed|path"
# Expected: both `completed` and `path` columns present.

# 2. The engine emits the new triple.
sqlite3 /home/john/elspeth/data/runs/audit.db \
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
dropped without affecting run status), do not configure the example
`plugin: noop_sink`: there is no built-in no-op sink today, and unknown sink
plugins fail validation with `PluginNotFoundError`.

To preserve old success semantics, choose one explicit operator action:

1. Route those rows to an existing supported success sink (`csv`, `json`,
   `database`, `azure_blob`, `dataverse`, or `chroma_sink`) and manage that
   output according to your retention policy.
2. Add a real no-op success sink as a separate, tested feature before using it
   in production.
3. Accept the new ADR-019 semantics and re-baseline dashboards/alerts.

Unsupported example, included only to show what NOT to do:

```yaml
# Before (silent drop):
sinks:
  - name: __discard__

# INVALID unless a real noop_sink plugin has been added and tested:
sinks:
  - name: silent_drop
    plugin: noop_sink
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
export ELSPETH_CHECKOUT=/home/john/elspeth
export ELSPETH_DATA_DIR=/home/john/elspeth/data
export ELSPETH_AUDIT_DB="$ELSPETH_DATA_DIR/runs/audit.db"
export ADR019_PREVIOUS_REF="$(cat "$ADR019_BACKUP_DIR/app-ref.txt")"

sudo systemctl stop elspeth-web.service
test -n "$ADR019_PREVIOUS_REF"
test -z "$(git -C "$ELSPETH_CHECKOUT" status --porcelain)"
git -C "$ELSPETH_CHECKOUT" switch --detach "$ADR019_PREVIOUS_REF"
test "$(git -C "$ELSPETH_CHECKOUT" rev-parse HEAD)" = \
  "$(git -C "$ELSPETH_CHECKOUT" rev-parse "$ADR019_PREVIOUS_REF")"

sudo test -s "$ADR019_BACKUP_DIR/audit.db"
sudo rm -f \
  "$ELSPETH_AUDIT_DB" \
  "$ELSPETH_AUDIT_DB-wal" \
  "$ELSPETH_AUDIT_DB-shm"
for suffix in "" "-wal" "-shm"; do
  if sudo test -e "$ADR019_BACKUP_DIR/audit.db${suffix}"; then
    sudo cp -a "$ADR019_BACKUP_DIR/audit.db${suffix}" "$ELSPETH_AUDIT_DB${suffix}"
  fi
done
sudo test -s "$ELSPETH_AUDIT_DB"
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
- [ ] Migration doc resolves the live Landscape audit DB path before backup/delete and uses the staging default `/home/john/elspeth/data/runs/audit.db` unless config overrides it
- [ ] Destructive SQLite/Postgres replacement commands abort unless the audit store is definitely pre-ADR-019 and abort if it is already current
- [ ] Backup commands use a unique backup directory and avoid non-sudo shell redirection into root-owned paths
- [ ] Migration doc does not recommend nonexistent `noop_sink`; preserving old discard semantics requires a supported success sink or a separate tested no-op sink feature
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

### Task 5.5a: Extend forbid_new_row_outcome.py to detect hardcoded RowOutcome value strings

**Context:** Historical migration misses included string-literal comparisons
against old `RowOutcome` values (for example an `outcome == "quarantined"`
shape). The existing lint guard (`forbid_new_row_outcome.py`) only detects
`RowOutcome.X` AST attribute accesses. A future contributor could re-introduce a
hardcoded value string without using `RowOutcome.X` syntax and the guard would
not catch it.

**Lifecycle note:** This extension is explicitly bounded. `forbid_new_row_outcome.py`
is deleted in Stage 5 (out-of-scope entry in the overview; script docstring line
25 states: "Stage 5 (post-migration) deletes both `RowOutcome` itself and this
script"). Between this PR and Stage 5, however, the gate IS load-bearing: source
scope must stay at zero RowOutcome references, and a hardcoded string bypass
would silently defeat that guarantee.

**Files:**
- Modify: `scripts/cicd/forbid_new_row_outcome.py`
- Modify: `config/cicd/forbid_new_row_outcome/migration_files.yaml` (allowlist, if any src/ sites need listing)
- Create: `tests/unit/scripts/cicd/test_forbid_new_row_outcome.py`

**Step 1: Write the RED tests first**

Before changing `scripts/cicd/forbid_new_row_outcome.py`, create
`tests/unit/scripts/cicd/test_forbid_new_row_outcome.py`. Follow the pattern
established by the existing CICD script tests in `tests/unit/scripts/cicd/`
(fake file tree under `tmp_path`, source snippets written to disk, scanner or
CLI invoked directly).

Minimum test cases for the new FNR2 rule:

```python
# FAILS: forward comparison
outcome == "quarantined"

# FAILS: reversed comparison
"quarantined" == outcome

# FAILS: attribute left
row.outcome == "completed"

# FAILS: membership against old RowOutcome value strings
outcome in {"completed", "failed"}
row.outcome not in {"diverted", "buffered"}

# PASSES: not a RowOutcome value string
outcome == "active"

# PASSES: not an outcome/path symbol
status in {"completed", "failed"}

# PASSES: file in allowlist -> suppressed
```

Also confirm that FNR1 (existing `RowOutcome.X` attribute detection) still passes
its existing cases after the extension. The two visitors are independent.

Run the new test before implementation and confirm it fails:

```bash
.venv/bin/python -m pytest tests/unit/scripts/cicd/test_forbid_new_row_outcome.py -v
```

Expected RED: FNR2 comparison and membership cases are not detected yet.

**Step 2: Identify the known RowOutcome value strings**

The full set of `RowOutcome` string values (from
`src/elspeth/contracts/enums.py`):

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

**Step 3: Add AST detection for string-literal comparisons and membership**

The pattern to detect is an equality/inequality or membership comparison where
one side is the name `outcome` (or a field access ending in `.outcome`) and the
other side is a string literal, or a literal collection of strings, whose value
is in `_ROW_OUTCOME_VALUE_STRINGS`. Canonical forms:

```python
outcome == "quarantined"          # forward
"quarantined" == outcome          # reversed
row.outcome == "completed"        # attribute on left
"completed" == row.outcome        # attribute on right
outcome in {"completed", "failed"}  # membership
row.outcome not in {"diverted"}      # negative membership
```

Detection approach: extend the script with a second AST visitor
(`_HardcodedValueVisitor`) that walks `ast.Compare` nodes and checks both:

1. `ast.Eq` / `ast.NotEq` comparisons between `ast.Name(id="outcome")` or
   `ast.Attribute(attr="outcome")` and a known RowOutcome string constant.
2. `ast.In` / `ast.NotIn` comparisons where the outcome symbol is compared
   against a literal `set`, `list`, or `tuple` containing one or more known
   RowOutcome string constants.

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

Scope the FNR2 check to `src/elspeth/` only (exclude `tests/` and `scripts/`
from FNR2; tests have their own `adr019_test_inventory.py`, and the script
itself imports the value list). Pass the scope as a `--src-root` flag (default
`src/elspeth`) or hardcode a secondary path filter inside the check function.

**Step 4: Integrate with the existing allowlist mechanism**

The existing allowlist at
`config/cicd/forbid_new_row_outcome/migration_files.yaml` uses a `file:` /
`justification:` YAML structure. FNR2 findings respect the same allowlist; if a
file is listed, both FNR1 and FNR2 violations are suppressed for that file. No
structural changes to the allowlist format are needed.

Confirm the live source tree contains zero FNR2 findings after Phase 1 consumer
fixes. If an MCP/Web/CLI/Landscape consumer appears, patch it in this PR rather
than allowlisting it.

**Step 5: Verify the guard in src/ scope**

```bash
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check \
  --root . --allowlist config/cicd/forbid_new_row_outcome
```

Expected: exit 0 with zero FNR2 findings in `src/elspeth/`.

**Definition of Done:**
- [ ] RED FNR2 tests fail before implementation
- [ ] `_HardcodedValueVisitor` added to `forbid_new_row_outcome.py`; FNR2 rule defined
- [ ] FNR2 detects equality, inequality, `in`, and `not in` hardcoded RowOutcome value-string checks for outcome symbols
- [ ] FNR2 scoped to `src/elspeth/`; tests are covered by `adr019_test_inventory.py`
- [ ] Existing allowlist mechanism covers FNR2 findings without format changes
- [ ] `tests/unit/scripts/cicd/test_forbid_new_row_outcome.py` passes
- [ ] `forbid_new_row_outcome.py` check returns exit 0 with zero FNR2 hits in `src/elspeth/`

---

### Task 5.6: Final verification + Phase 5 commit + PR open

**Execution order:** complete Task 5.0a and Task 5.5a before running this task.
Both inventory guards are prerequisites for final verification, commit, and PR
open. Do not run the final gate set below until both Definitions of Done are
complete.

**Step 1: Run all verification gates**

```bash
.venv/bin/pre-commit run --all-files
.venv/bin/python -m pytest tests/ \
  --cov=src/elspeth \
  --cov-report=xml \
  --cov-report=term-missing \
  --cov-fail-under=80 \
  -v \
  -m "not slow and not stress and not performance" \
  --timeout=120
.venv/bin/python -m mypy src/ tests/
.venv/bin/python -m ruff check src/ tests/ scripts/ examples/
.venv/bin/python -m ruff format --check src/ tests/ scripts/ examples/
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model --exclude "**/__pycache__/*"
.venv/bin/python scripts/cicd/enforce_audit_evidence_nominal.py check --root src/elspeth --allowlist config/cicd/enforce_audit_evidence_nominal
.venv/bin/python scripts/cicd/enforce_tier_1_decoration.py check --file src/elspeth/contracts/errors.py --allowlist config/cicd/enforce_tier_1_decoration
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth
.venv/bin/python scripts/cicd/enforce_contract_manifest.py check --allowlist config/cicd/enforce_contract_manifest
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
.venv/bin/python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth --allowlist config/cicd/enforce_frozen_annotations
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check --root src/elspeth --allowlist config/cicd/adr019_symbol_inventory
.venv/bin/python scripts/cicd/adr019_test_inventory.py check --root tests --allowlist config/cicd/adr019_test_inventory
cd src/elspeth/web/frontend
npm run test
npm run build
cd /home/john/elspeth
```

ALL must pass locally. The local gate intentionally mirrors CI scopes:
ruff includes `examples/`, mypy includes `tests/`, coverage enforces the CI
threshold, and policy gates include audit-evidence and Tier-1 decoration checks.
Because CI runs a Python 3.12/3.13 matrix, the PR is not done until the remote
matrix is green too.

**Step 2: Confirm the new ADR-019 behavioural and guard fixtures are GREEN**

```bash
.venv/bin/python -m pytest \
  tests/integration/test_adr_019_discard_mode_flip.py \
  tests/integration/test_adr_019_counter_changes.py \
  tests/integration/test_adr_019_cross_table_invariants.py \
  tests/integration/test_adr_019_sweep_durability.py \
  tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
  tests/unit/scripts/cicd/test_adr019_test_inventory.py \
  tests/unit/scripts/cicd/test_forbid_new_row_outcome.py \
  -v
```

**Step 3: Commit**

Before staging, verify the tree contains only intended Phase 5 changes and any
previous phase commits have already landed. Do not use `git stash`.

```bash
git status --short
git diff --name-only > /tmp/adr-019-phase-5-worktree-files.txt

# Review /tmp/adr-019-phase-5-worktree-files.txt before staging. Stop if it
# includes unrelated docs/source files or uncommitted previous-phase code.
```

```bash
git add tests/ docs/operator/migrations/adr-019.md \
        config/cicd/forbid_new_row_outcome/migration_files.yaml \
        scripts/cicd/forbid_new_row_outcome.py \
        tests/unit/scripts/cicd/test_forbid_new_row_outcome.py \
        scripts/cicd/adr019_symbol_inventory.py \
        config/cicd/adr019_symbol_inventory \
        tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
        scripts/cicd/adr019_test_inventory.py \
        config/cicd/adr019_test_inventory \
        tests/unit/scripts/cicd/test_adr019_test_inventory.py

git diff --cached --stat
git diff --name-only --cached > /tmp/adr-019-phase-5-staged-files.txt
git status --short

# Stop here if the staged files differ from the intended manifest or if
# unrelated unstaged changes remain. Do not let pre-commit or commit hooks hide
# unrelated dirty work behind a stash/pop cycle.

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

Assertion/comparison-only tests (Category B) translated in this PR to two-axis
`(TerminalOutcome, TerminalPath)` assertions. This includes equality,
membership, collection equality, raw SQL value checks, and hardcoded old outcome
strings identified by the live AST test inventory. No ADR-019 Stage 4 xfails
remain.

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

- [x] Full local test gate passes with CI-equivalent coverage command (`pytest tests/ ... --cov-fail-under=80 -m "not slow and not stress and not performance" --timeout=120`)
- [x] ADR-019 behavioural integration test files green:
  - `tests/integration/test_adr_019_discard_mode_flip.py`
  - `tests/integration/test_adr_019_counter_changes.py`
  - `tests/integration/test_adr_019_cross_table_invariants.py`
  - `tests/integration/test_adr_019_sweep_durability.py`
- [x] mypy clean across `src/ tests/`
- [x] ruff lint + format clean across `src/ tests/ scripts/ examples/`
- [x] All project gates pass (tier model, audit evidence nominal, Tier-1 decoration, contract manifest, plugin hashes, freeze guards, frozen annotations, forbid-new-row-outcome)
- [x] Lint guard `forbid_new_row_outcome.py` passes including hardcoded RowOutcome value-string detection; src/ scope reaches zero RowOutcome references; tests/ retain only deliberate compatibility/mapping references
- [x] ADR-019 AST symbol inventory passes with only deliberate migration-window allowlist entries
- [x] ADR-019 test inventory passes with only deliberate compatibility/mapping/commentary allowlist entries
- [x] Frontend session terminal-status gates pass: `npm run test` and `npm run build`
- [x] Remote GitHub checks green for the PR, including Python 3.12 and 3.13

## Refs

- ADR: docs/architecture/adr/019-two-axis-terminal-model.md
- Plan: docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md
- Stage 1 (already shipped): commit 60d30551
- Stage 2 ticket: elspeth-949719575e
- Stage 3 ticket: elspeth-edb60744f0

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"

gh pr checks --watch
```

**Definition of Done:**
- [ ] All verification gates green
- [ ] Phase 5 commit landed
- [ ] PR opened with comprehensive description
- [ ] Remote CI/checks green after PR creation, including Python 3.12 and 3.13 jobs
- [ ] Operator migration doc linked from PR description
- [ ] Stage 1 and Stage 5 ticket cross-references in PR description
