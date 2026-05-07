# ADR-019 Stage 2/3 — Phase 1: Schema + Recorder + Loader + Contract Dataclasses

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phase 1 is a local checkpoint only, not a git commit boundary: it leaves the recorder/producer contract intentionally inconsistent because producer sites still pass `RowOutcome`-shaped values to the new `record_token_outcome` API. Ordinary imports may still succeed; the expected failures are type-check/runtime call-site failures and stale schema-dependent tests. Do NOT commit, push, or propose to land Phase 1 alone. Continue directly to Phase 2, and create the first git commit only after Phase 3 completes the atomic Stage 2/3 migration and the Phase 3 import/runtime smoke passes.

**Goal:** Flip the audit foundation — DB schema (`is_terminal` → `completed`, add `path` column, repurpose `outcome` value space), the four contract dataclasses (`TokenOutcome`, `RowResult`, `PendingOutcome`, `TokenCompleted`), the recorder write path (`record_token_outcome` signature + `_validate_outcome_fields` rewrite), and the loader read path (`TokenOutcomeLoader.load` cross-checks per ADR-019 § Implementation Notes invariant-translation table).

**Files touched in this phase:**

Core schema + recorder + dataclasses:
- Create: `scripts/cicd/adr019_symbol_inventory.py` (AST-backed source inventory; must exist before every phase boundary)
- Create: `config/cicd/adr019_symbol_inventory/` (temporary migration-window allowlist directory; at minimum `migration_files.yaml`)
- Create: `tests/fixtures/cicd/adr019_symbol_inventory/` (committed positive/negative fixture corpus for the AST inventory)
- Test: `tests/unit/scripts/cicd/test_adr019_symbol_inventory.py`
- Modify: `docs/architecture/adr/019-two-axis-terminal-model.md:652-659` (align Implementation Notes with existing `error_hash` audit column; replace stale `quarantine_reason` wording before deriving the machine-readable constraint table)
- Modify: `src/elspeth/core/landscape/schema.py:180-216` (table definition)
- Modify: `src/elspeth/core/landscape/database.py:26-70, 427-441` (schema compatibility guard: epoch + required columns)
- Modify: `src/elspeth/contracts/audit.py:673-703` (TokenOutcome dataclass)
- Modify: `src/elspeth/contracts/results.py:379-421` (RowResult dataclass)
- Modify: `src/elspeth/contracts/engine.py:46-100` (PendingOutcome dataclass)
- Modify: `src/elspeth/contracts/events.py:242-249` (TokenCompleted dataclass)
- Modify: `src/elspeth/contracts/__init__.py` (runtime re-export `TerminalOutcome`/`TerminalPath` beside `RowOutcome`; explicit package-boundary decision)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:203-307, 570-580, 785-795, 802-880, 895-940` (recorder + internal FORKED/EXPANDED callers + read-side query column-rename sites at 899/930)
- Modify: `src/elspeth/core/landscape/model_loaders.py:525-609` (loader)
- Modify: `src/elspeth/testing/__init__.py:31-45, 507-540, 715-740` (test helpers re-export `TerminalOutcome`/`TerminalPath`)
- Create: `docs/operator/migrations/adr-019.md` (operator-facing migration guide stub; Phase 5 expands it, but Phase 1 error paths must never point to a missing document)

**Downstream consumers of the audit schema (added 2026-05-05 after consumer-surface sweep — the original plan missed these and Phase 1 would crash MCP/Web at deploy without them):**
- Modify: `src/elspeth/mcp/types.py:204-214, 360` (move `OutcomeDistributionEntry` above `RunSummaryReport`, then update `RunSummaryReport.outcome_distribution` — no forward references; wire-schema field rename `is_terminal` → `completed`, add `path: str`, change collapsed dict to path-aware entry list)
- Modify: `src/elspeth/mcp/analyzers/reports.py:120-127, 157, 659, 663, 700, 708, 709` (`get_run_summary` outcome distribution and `get_outcome_analysis` both become path-aware)
- Modify: `src/elspeth/mcp/analyzers/diagnostics.py:181` (hardcoded `outcome == "quarantined"` → `path == "quarantined_at_source"` — silent-zero-quarantine bug per CLAUDE.md Tier 1 audit integrity)
- Modify: `src/elspeth/web/execution/diagnostics.py:170` (JOIN condition `is_terminal == 1` → `completed == 1`)
- Modify: `src/elspeth/web/execution/discard_summary.py:20, 92` (import the canonical `DISCARD_SINK_NAME` from `contracts.audit`; WHERE filter `is_terminal == 1` → `completed == 1`)
- Modify: `src/elspeth/contracts/export_records.py:139-154` (`TokenOutcomeExportRecord` TypedDict — add `path`, rename `is_terminal` → `completed`, allow `outcome: str | None`)
- Modify: `src/elspeth/core/landscape/exporter.py:430` (JSONL token_outcome export field `is_terminal` → `completed`, add `path` field)
- Modify: `src/elspeth/core/landscape/lineage.py:118` (property read `o.is_terminal` → `o.completed` on `TokenOutcome` dataclass)
- Modify: `src/elspeth/core/landscape/formatters.py:170` (CLI formatter — print `path.name` alongside `outcome.name`; `is_terminal` line becomes `completed`)

Tests:
- Test: `tests/unit/core/landscape/test_database_compatibility_guards.py` (extend with stale ADR-018 token_outcomes schema rejection)
- Test: `tests/unit/core/landscape/test_data_flow_repository.py` (new tests for the (outcome, path) write path)
- Test: `tests/unit/core/landscape/test_model_loaders.py` (extend existing `TokenOutcomeLoader` coverage with the new cross-checks; do not create a parallel loader-test module unless this file is first split deliberately)
- Test: `tests/unit/contracts/test_audit.py` (TokenOutcome construction tests for the new shape)
- Test: `tests/unit/mcp/test_outcome_analysis.py` (new — verifies wire-schema rename + path column)
- Test: `tests/unit/mcp/test_diagnose_quarantine_count.py` (new — RED-first regression test for B3)
- Test: `tests/unit/web/execution/test_discard_summary.py` (new — direct regression for the completed-column rename and discard-only count)
- Test: `tests/unit/core/landscape/test_exporter.py` (existing — JSONL output assertion fixup)
- Test: `tests/unit/core/landscape/test_formatters.py` (existing — direct CLI formatter assertion for `Path:` + `Completed:` output)
- Test: `tests/unit/telemetry/test_contracts.py`, `tests/unit/telemetry/test_filtering.py`, and affected `tests/unit/telemetry/exporters/` modules (TokenCompleted emits/serializes both `outcome` and `path`)

**Background reading:** ADR-019 lines 99-115 (mapping table — the canonical contract), lines 237-269 (cross-check invariants), lines 638-660 (Implementation Notes table). The Stage 1 closed-set partition at `src/elspeth/contracts/enums.py::_LEGAL_TERMINAL_PAIRS` is THE source of truth for legal `(outcome, path)` pairs — every cross-check in this phase consults it.

---

## Schema decisions (read before editing)

### 1. `is_terminal` → `completed` is a rename, not a new column

Same SQLAlchemy `Column(Integer, nullable=False)` semantics. Same 0/1 stored values. Only the field name changes — and the cross-check that consults it. Per ADR-019 sub-decision 3 (panel-resolved), the `completed` field is materially redundant with `outcome IS NOT NULL` but preserved as a materialized column for query ergonomics and operator vocabulary, mirroring the existing `is_terminal` pattern.

### 2. `outcome` column changes value space

| Before | After |
| --- | --- |
| `String(32), nullable=False` | `String(32), nullable=True` |
| Stores `RowOutcome.value` (12 enum values, never NULL) | Stores `TerminalOutcome.value` (3 enum values: `success`, `failure`, `transient`) OR NULL when `completed=False` |
| Old cross-check: `is_terminal == RowOutcome(outcome).is_terminal` | New cross-check: `completed XOR (outcome IS NULL)` — i.e., `completed=True ↔ outcome IS NOT NULL` |

The same column is REUSED, not added. The schema migration is: rename one column, add one column (`path`), change one column's nullability and value space.

### 3. `path` is a new always-populated column

`Column("path", String(64), nullable=False)` — every row in `token_outcomes` has a path, including `BUFFERED` rows which carry `path="buffered"`. This makes the path column lookup-stable and avoids NULL handling at the loader.

### 4. DB migration is delete-and-recreate

Per ELSPETH's project DB migration policy recorded for this plan, ELSPETH does not run Alembic. The metadata defines the new schema; `metadata.create_all()` creates the new tables on engine startup. Operators replace the Landscape audit store (for example `audit.db`) between this PR and any pre-Stage-2 audit schema. Do not delete or replace `sessions.db` for ADR-019 unless a separate web-session migration proves that schema incompatible; ADR-019's schema changes are in the Landscape audit tables.

**Mechanical guard, not just docs:** `LandscapeDB` validates an existing database before `metadata.create_all()` and `create_all()` will not alter old tables. Therefore this phase must also bump `SQLITE_SCHEMA_EPOCH` and validate the full `token_outcomes` ADR-019 shape for existing schemas: `token_outcomes.completed` plus `token_outcomes.path` exist, `token_outcomes.outcome` is nullable, stale `token_outcomes.is_terminal` is rejected as an old-shape witness, and the terminal unique-index predicate uses `completed == 1` rather than `is_terminal == 1` wherever the dialect exposes the predicate. An existing epoch-6 ADR-018 database must fail fast with an operator-actionable `SchemaCompatibilityError`, before any recorder or query path can hit a late SQL/AttributeError. The same shape scan must run for non-SQLite backends when Landscape tables already exist; Postgres has no `PRAGMA user_version`, but it is equally vulnerable to `create_all()` silently leaving stale columns/indexes in place. The error must name the stale shape and point at an existing `docs/operator/migrations/adr-019.md` so staging/prod operators know this is the ADR-019 delete-and-recreate boundary, not a mysterious mid-pipeline crash. **Important ordering:** the epoch-incompatible branch must not short-circuit before the ADR-019 shape scan can contribute the migration-specific message.

---

## Tasks

### Task 1.0: Add AST-backed ADR-019 source inventory before touching schema

**Files:**
- Modify: `docs/architecture/adr/019-two-axis-terminal-model.md`
- Create: `scripts/cicd/adr019_symbol_inventory.py`
- Create: `config/cicd/adr019_symbol_inventory/`
- Create: `tests/fixtures/cicd/adr019_symbol_inventory/`
- Create: `tests/unit/scripts/cicd/test_adr019_symbol_inventory.py`

**Why this task is first:** the overview's D7 closeout gate is load-bearing at
every phase boundary. The tool cannot be introduced in Phase 5, because the
Phases 1-3 atomic commit and the Phase 4 commit already depend on it to catch
missed `is_terminal` / hardcoded RowOutcome-value surfaces. Implement this
inventory before the schema rename so every later checkpoint can run it.

**Step 0: Align ADR-019 Implementation Notes with the stored audit column**

Before deriving any machine-readable constraint table, patch
`docs/architecture/adr/019-two-axis-terminal-model.md` so the Implementation
Notes table uses the existing `error_hash` field for
`(FAILURE, QUARANTINED_AT_SOURCE)`. The current prose says
`quarantine_reason`, but `token_outcomes` has no such column and both the
existing ADR-018 schema and this Phase 1 schema use `error_hash` as the
stable single-hop error witness.

Required edit:

```diff
-   | `QUARANTINED requires quarantine_reason` | `(FAILURE, QUARANTINED_AT_SOURCE)` requires `quarantine_reason` |
+   | `QUARANTINED requires error_hash` | `(FAILURE, QUARANTINED_AT_SOURCE)` requires `error_hash` |
```

Then run:

```bash
grep -n "QUARANTINED" docs/architecture/adr/019-two-axis-terminal-model.md
```

Expected: the Implementation Notes row names `error_hash`, not
`quarantine_reason`. Do not proceed to schema/recorder code until the ADR and
the plan agree on the column name.

**Step 1: Implement the AST visitor**

The script walks Python files under a root (default `src/elspeth`) and reports
non-allowlisted findings as JSON lines plus a non-zero exit code. Use `ast.parse`;
do not tokenize with regex.

Minimum finding kinds:

```python
class FindingKind(StrEnum):
    IS_TERMINAL_ANNOTATION = "is_terminal_annotation"
    IS_TERMINAL_ATTRIBUTE = "is_terminal_attribute"
    IS_TERMINAL_KEYWORD = "is_terminal_keyword"
    IS_TERMINAL_DICT_KEY = "is_terminal_dict_key"
    ROW_OUTCOME_STRING_COMPARE = "row_outcome_string_compare"
    TERMINAL_OUTCOME_STRING_COMPARE = "terminal_outcome_string_compare"
    TERMINAL_PATH_STRING_COMPARE = "terminal_path_string_compare"
```

`ROW_OUTCOME_STRING_COMPARE` is the migration-window guard for old
`RowOutcome` value strings. The two terminal-value finding kinds are the
post-migration guard: any direct comparison to known `TerminalOutcome` values
(`"success"`, `"failure"`, `"transient"`) or `TerminalPath` values
(`"default_flow"`, `"gate_routed"`, `"on_error_routed"`,
`"filter_dropped"`, `"coalesced"`, `"unrouted"`,
`"quarantined_at_source"`, `"sink_fallback_to_failsink"`,
`"sink_discarded"`, `"fork_parent"`, `"expand_parent"`,
`"batch_consumed"`, `"buffered"`) must be flagged unless it is in the enum
definition or an explicitly allowlisted migration fixture. This prevents
replacing brittle `RowOutcome` string checks with equally brittle
`TerminalOutcome` / `TerminalPath` string checks.

Report fields: `kind`, `path`, `line`, `col`, `symbol`, and a short `context`
from `ast.unparse(node)` when available.

**Step 2: Add focused tests**

The unit test must cover every finding kind plus false-positive guards:

```python
class OutcomeDistributionEntry(TypedDict):
    is_terminal: bool

record.is_terminal
record_token_outcome(is_terminal=True)
{"is_terminal": True}
outcome == "quarantined"
outcome == "failure"
path == "quarantined_at_source"

terminal = True
payload = {"completed": True}
outcome in {"completed", "failed"}
```

Use the existing CICD-script test style under `tests/unit/scripts/cicd/`, but
do not rely only on inline snippets. Add a committed fixture corpus under
`tests/fixtures/cicd/adr019_symbol_inventory/` with:

- positive fixtures for every finding kind,
- negative fixtures for the false-positive guards above,
- at least one fixture that imports `TerminalOutcome` and `TerminalPath` so the
  script proves it flags brittle string comparisons even when the enum imports
  are present, and
- a CLI test that points `--allowlist` at the directory
  `config/cicd/adr019_symbol_inventory` rather than a flat YAML file.

The unit tests should load the fixture corpus, call the inventory function
directly for exact findings, and exercise the CLI `check` command once.

**Step 3: Seed the temporary allowlist**

Allow only deliberate migration-window files, with one-line justifications:

- `src/elspeth/contracts/enums.py` while `RowOutcome` still exists until Stage 5.
- `src/elspeth/testing/__init__.py` while it re-exports `RowOutcome` through the
  remaining RowOutcome-retention window. Phase 5 removes pytest-blocking
  assertions; Stage 5 removes the enum/re-export.

Do not allowlist MCP, Web, exporter, formatter, or Landscape consumer files.
Those are real missed consumers and must be fixed in this PR.

**Step 4: Verify the tool is usable before schema edits**

```bash
.venv/bin/python -m pytest tests/unit/scripts/cicd/test_adr019_symbol_inventory.py -v
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check \
  --root src/elspeth \
  --allowlist config/cicd/adr019_symbol_inventory
```

The first command must pass. The second may report known pre-migration findings
before Phase 1's consumer edits, but the output format and allowlist behavior
must be correct. After Phase 1 Task 1.9 and before the atomic Phases 1-3 commit,
the check must exit 0.

**Definition of Done:**
- [ ] AST inventory script created and covered by unit tests
- [ ] Temporary allowlist directory exists and is limited to deliberate migration-window files
- [ ] Fixture corpus covers positive/negative AST inventory cases and an import-presence case
- [ ] Script detects annotations, attributes, kwargs, dict keys, hardcoded RowOutcome value comparisons, hardcoded TerminalOutcome value comparisons, and hardcoded TerminalPath value comparisons
- [ ] Script exits non-zero for non-allowlisted findings
- [ ] Atomic Phases 1-3 commit staging includes the script, config, and tests

---

### Task 1.1: Update `token_outcomes` schema definition

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py:180-216`

**Step 1: Read the existing schema block to confirm line numbers**

Run: `grep -n "token_outcomes_table = Table" src/elspeth/core/landscape/schema.py`

Expected: line `180` (current HEAD).

**Step 2: Make the schema change**

Apply this edit:

```python
# OLD (lines 180-206 approximately):
token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("token_id", String(64), nullable=False, index=True),
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    Column("outcome", String(32), nullable=False),
    Column("is_terminal", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    Column("sink_name", String(128)),
    Column("batch_id", String(64)),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    Column("context_json", Text),
    Column("expected_branches_json", Text),
    ForeignKeyConstraint(["batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
)

# NEW:
token_outcomes_table = Table(
    "token_outcomes",
    metadata,
    # Identity
    Column("outcome_id", String(64), primary_key=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("token_id", String(64), nullable=False, index=True),
    # Composite FK: token_id and run_id belong together (prevents cross-run contamination)
    ForeignKeyConstraint(["token_id", "run_id"], ["tokens.token_id", "tokens.run_id"]),
    # ADR-019 two-axis terminal model. ``completed`` mirrors the prior
    # ``is_terminal`` column (sub-decision 3). ``outcome`` value space changed
    # from RowOutcome (12 values, non-NULL) to TerminalOutcome (3 values:
    # success / failure / transient) with NULL when completed=False
    # (only ``BUFFERED`` today). ``path`` is producer-declared per ADR-019
    # § "Classification is producer-declared, not topology-derivable" and
    # always populated, including ``path="buffered"`` for non-terminal rows.
    Column("outcome", String(32), nullable=True),
    Column("path", String(64), nullable=False),
    Column("completed", Integer, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False),
    # Outcome-specific fields (nullable based on (outcome, path) pair)
    Column("sink_name", String(128)),
    Column("batch_id", String(64)),
    Column("fork_group_id", String(64)),
    Column("join_group_id", String(64)),
    Column("expand_group_id", String(64)),
    Column("error_hash", String(64)),
    # Optional extended context
    Column("context_json", Text),
    Column("expected_branches_json", Text),
    ForeignKeyConstraint(["batch_id", "run_id"], ["batches.batch_id", "batches.run_id"]),
)
```

**Step 3: Update the partial unique index (lines ~210-216)**

The existing index uses `is_terminal == 1`; rename to `completed == 1`:

```python
# OLD:
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.is_terminal == 1),
    postgresql_where=(token_outcomes_table.c.is_terminal == 1),
)

# NEW:
Index(
    "ix_token_outcomes_terminal_unique",
    token_outcomes_table.c.token_id,
    unique=True,
    sqlite_where=(token_outcomes_table.c.completed == 1),
    postgresql_where=(token_outcomes_table.c.completed == 1),
)
```

**Step 4: Verify schema compiles**

Run: `.venv/bin/python -c "from elspeth.core.landscape.schema import token_outcomes_table; print([c.name for c in token_outcomes_table.columns])"`

Expected output:
```
['outcome_id', 'run_id', 'token_id', 'outcome', 'path', 'completed', 'recorded_at', 'sink_name', 'batch_id', 'fork_group_id', 'join_group_id', 'expand_group_id', 'error_hash', 'context_json', 'expected_branches_json']
```

**Definition of Done:**
- [ ] `is_terminal` column renamed to `completed`
- [ ] `outcome` column nullability changed to `True`
- [ ] `path` column added with `String(64), nullable=False`
- [ ] Partial unique index references `completed == 1` instead of `is_terminal == 1`
- [ ] Module imports cleanly
- [ ] No other references to `token_outcomes_table.c.is_terminal` remain in `schema.py` (grep verifies)

---

### Task 1.1a: Create the operator migration guide stub before any error path links to it

**Files:**
- Create: `docs/operator/migrations/adr-019.md`

**Why this task exists:** Task 1.1b deliberately points stale-schema failures at
`docs/operator/migrations/adr-019.md`. The first legal commit is the atomic
Phases 1-3 commit, while the original plan created the operator guide in Phase 5.
That leaves a valid intermediate commit where fail-fast runtime errors point at
a missing remediation document. Create the durable stub before writing any
schema-compatibility test or error branch that references it; Phase 5 expands it
into the full deployment runbook.

**Step 1: Create the minimal operator-facing document**

```markdown
# ADR-019 Operator Migration Guide

**Status:** Stub created by Phase 1. Phase 5 expands this into the complete
deployment, verification, and rollback runbook before the PR opens.

ADR-019 replaces the single-axis `RowOutcome` audit DB encoding with the
two-axis `(TerminalOutcome, TerminalPath, completed)` model. Existing
pre-ADR-019 `audit.db` / audit-store schemas are intentionally incompatible.
ELSPETH does not run Alembic migrations for this project; operators must replace
the Landscape audit store at the ADR-019 deployment boundary. ADR-019 does not
change the web session schema; do not delete or replace `sessions.db` for this
migration unless a separate web-session compatibility check fails and the Phase
5 runbook is amended with explicit session backup/restore steps.

If startup raises `SchemaCompatibilityError` naming `token_outcomes.completed`,
`token_outcomes.path`, stale `token_outcomes.is_terminal`, or a stale terminal
index predicate, stop the service and follow the full Phase 5 runbook before
deploying this PR.

See [ADR-019](../../architecture/adr/019-two-axis-terminal-model.md) for the
canonical mapping table and behaviour-change rationale.
```

**Step 2: Verify the link target exists**

```bash
test -f docs/operator/migrations/adr-019.md
grep -Fq "ADR-019 Operator Migration Guide" docs/operator/migrations/adr-019.md
grep -Fq "../../architecture/adr/019-two-axis-terminal-model.md" docs/operator/migrations/adr-019.md
```

**Definition of Done:**
- [ ] `docs/operator/migrations/adr-019.md` exists before any Task 1.1b test or implementation points errors at it
- [ ] Phase 5 Task 5.4 is updated to expand, not first-create, this document

---

### Task 1.1b: Update Landscape schema compatibility guard

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py:27-48`
- Modify: `src/elspeth/core/landscape/database.py:26-70`
- Test: `tests/unit/core/landscape/test_database_compatibility_guards.py`

**Why this task exists:** the migration policy is delete-and-recreate, but the runtime must still reject stale databases mechanically. Current startup validates before `metadata.create_all()`, and `create_all()` never mutates existing tables. Without this task, an epoch-6 SQLite database or a stale Postgres database with `token_outcomes.is_terminal` and no `path`/`completed` columns can pass compatibility checks and fail later on first write/read. That is an auditability failure: the operator sees a pipeline crash instead of a startup-time instruction to replace the pre-ADR-019 database.

**Step 1: Write the stale-schema RED test FIRST**

Extend `tests/unit/core/landscape/test_database_compatibility_guards.py` with a test that creates an ADR-018-shaped `token_outcomes` table, stamps it with the previous epoch, and asserts `LandscapeDB(...)` raises `SchemaCompatibilityError` naming both missing columns:

```python
def test_validate_schema_rejects_adr018_token_outcomes_shape(self, tmp_path: Path) -> None:
    db_path = tmp_path / "adr018_audit.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA user_version = 6")
        conn.execute(
            text(
                """
                CREATE TABLE token_outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    is_terminal INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL
                )
                """
            )
        )
    engine.dispose()

    with pytest.raises(SchemaCompatibilityError) as exc_info:
        LandscapeDB(f"sqlite:///{db_path}")

    msg = str(exc_info.value)
    assert "Landscape database schema is outdated." in msg
    assert "token_outcomes.completed" in msg
    assert "token_outcomes.path" in msg
    assert "docs/operator/migrations/adr-019.md" in msg
```

**Step 2: Bump the SQLite schema epoch**

In `src/elspeth/core/landscape/schema.py`, add an epoch-history entry and set:

```python
#   7 -> ADR-019 Stage 2/3: token_outcomes stores the two-axis terminal model
#        (`outcome`, `path`, `completed`) instead of RowOutcome + is_terminal.
SQLITE_SCHEMA_EPOCH = 7
```

**Step 3: Add the new required columns**

In `src/elspeth/core/landscape/database.py::_REQUIRED_COLUMNS`, add:

```python
# ADR-019 two-axis terminal model: old `is_terminal` DBs must fail fast.
("token_outcomes", "completed"),
("token_outcomes", "path"),
```

Do not remove the existing `("token_outcomes", "expected_branches_json")` guard.

**Step 3a: Make the stale-ADR-019 error operator-actionable and avoid epoch short-circuit**

Live `_validate_schema()` currently rejects incompatible SQLite epochs before it
collects missing required columns. After the epoch bump to 7, the RED test above
would otherwise take the generic epoch error branch and never mention
`token_outcomes.completed`, `token_outcomes.path`, or the ADR-019 migration
guide. Fix the ordering as part of this task:

1. Inspect tables and collect `_REQUIRED_COLUMNS` failures before raising for an
   incompatible non-zero epoch, OR
2. Preserve the epoch check but make it include the already-collected missing
   ADR-019 columns when present.

The required behavior is that an epoch-6 ADR-018-shaped DB reports both the epoch
mismatch and the ADR-019 missing-column action. Do not weaken the epoch check;
augment the error so operators get the specific remediation.

Extend the stale-schema message when the missing columns include
`token_outcomes.completed` or `token_outcomes.path`:

```python
if ("token_outcomes", "completed") in missing_columns or (
    "token_outcomes",
    "path",
) in missing_columns:
    error_parts.append(
        "ADR-019 changed token_outcomes from RowOutcome/is_terminal to "
        "(TerminalOutcome, TerminalPath, completed). See "
        "docs/operator/migrations/adr-019.md and replace the stale audit.db "
        "before starting this ELSPETH version."
    )
```

Keep the exception type as `SchemaCompatibilityError`; existing callers/tests
already treat schema incompatibility as that fail-fast startup class. The
important behaviour is timing and message quality: stale DBs fail during
`LandscapeDB(...)`, before any recorder write, loader query, MCP analyzer, or
web request can crash with a late `AttributeError`.

**Step 3b: Make required-column validation backend-agnostic**

Do not leave `_validate_schema()` as "SQLite full validation, non-SQLite table
existence only." Extract the required-column scan into a helper that accepts a
SQLAlchemy inspector and works for SQLite and Postgres:

```python
def _collect_missing_required_columns(inspector: Inspector) -> list[tuple[str, str]]:
    existing_tables = set(inspector.get_table_names())
    missing: list[tuple[str, str]] = []
    for table_name, column_name in _REQUIRED_COLUMNS:
        if table_name not in existing_tables:
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        if column_name not in existing_columns:
            missing.append((table_name, column_name))
    return missing
```

Required behavior:

1. New empty databases still reach `metadata.create_all()`.
2. Existing non-Landscape databases opened with `_require_existing_schema=True`
   still fail with the existing "does not contain any Landscape tables" message.
3. Existing SQLite or Postgres databases that contain any Landscape table and a
   stale `token_outcomes` table missing `completed` or `path` fail before
   `create_all()` with the ADR-019 operator-actionable `SchemaCompatibilityError`.

Add a dialect-agnostic unit test without requiring a live Postgres service:
monkeypatch SQLAlchemy inspection so `_validate_schema()` sees
`engine.dialect.name == "postgresql"`, `get_table_names()` returning
`["runs", "token_outcomes"]`, and `get_columns("token_outcomes")` returning the
ADR-018 column set. Assert the same `SchemaCompatibilityError` details as the
SQLite test: `token_outcomes.completed`, `token_outcomes.path`, and
`docs/operator/migrations/adr-019.md`.

**Step 3c: Validate the full ADR-019 token_outcomes shape, not just column names**

Column presence is not sufficient for an existing database. ADR-019 also changes
`token_outcomes.outcome` from `nullable=False` to `nullable=True` and changes the
terminal unique-index predicate from `is_terminal == 1` to `completed == 1`.
Implement a second helper that reports stale-shape failures separately from
missing-column failures:

```python
def _collect_token_outcomes_shape_errors(inspector: Inspector) -> list[str]:
    """Return ADR-019 shape errors for existing token_outcomes tables."""
    existing_tables = set(inspector.get_table_names())
    if "token_outcomes" not in existing_tables:
        return []

    columns = {column["name"]: column for column in inspector.get_columns("token_outcomes")}
    errors: list[str] = []

    if "is_terminal" in columns:
        errors.append("token_outcomes.is_terminal is stale; ADR-019 uses completed")
    if "outcome" in columns and columns["outcome"].get("nullable") is False:
        errors.append("token_outcomes.outcome must be nullable for BUFFERED rows")

    # Inspect index predicate where SQLAlchemy exposes dialect-specific options.
    # SQLite may require querying sqlite_master by index name from the live
    # connection; Postgres may expose the predicate through dialect_options.
    # If the predicate cannot be introspected, do not guess. Report only the
    # stale `is_terminal` column and nullable outcome failures above.
    ...
    return errors
```

Add RED tests for all mechanically inspectable stale-shape cases:

1. Existing SQLite table with `completed` and `path` present but `outcome TEXT NOT NULL`
   still fails with `SchemaCompatibilityError`.
2. Existing SQLite table with both `completed` and stale `is_terminal` still
   fails; this catches partial/manual schema edits that only add new columns.
3. Existing SQLite partial unique index whose SQL predicate still references
   `is_terminal` fails and names the stale predicate.
4. Existing dialect-agnostic/Postgres-shaped inspector data with stale
   `is_terminal` and `outcome nullable=False` fails without needing a live
   Postgres service.

Required error behavior:

- Include `docs/operator/migrations/adr-019.md` in every ADR-019 shape error.
- Include the specific stale witness (`token_outcomes.is_terminal`,
  `token_outcomes.outcome nullable`, or the stale index predicate) so operators
  can distinguish "wrong DB" from "application bug."
- Do not silently pass a table that has the new column names but the old
  nullability/predicate semantics.

**Step 4: GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/unit/core/landscape/test_database_compatibility_guards.py -v
```

**Definition of Done:**
- [ ] `SQLITE_SCHEMA_EPOCH` bumped and epoch history updated
- [ ] `_REQUIRED_COLUMNS` includes `token_outcomes.completed` and `token_outcomes.path`
- [ ] `_validate_schema()` does not raise the generic epoch-only error before collecting/reporting the ADR-019 missing-column details
- [ ] `_collect_token_outcomes_shape_errors()` rejects stale `is_terminal`, non-nullable `outcome`, and inspectable stale terminal-index predicates
- [ ] ADR-018-shaped SQLite stale DB fails with `SchemaCompatibilityError` before `create_all()`
- [ ] Partially edited SQLite stale DBs fail when `completed`/`path` exist but `outcome` is still NOT NULL or `is_terminal` remains present
- [ ] Stale terminal unique-index predicates that still reference `is_terminal` fail where the dialect exposes the predicate
- [ ] ADR-018-shaped Postgres/non-SQLite stale schema fails with the same ADR-019 missing-column `SchemaCompatibilityError`
- [ ] Stale-DB error names `token_outcomes.completed`, `token_outcomes.path`, stale-shape witnesses when present, and `docs/operator/migrations/adr-019.md`
- [ ] Compatibility guard tests pass

---

### Task 1.2: Retype `TokenOutcome` dataclass

**Files:**
- Modify: `src/elspeth/contracts/audit.py:673-703`

**Step 1: Write the failing test FIRST**

Create or extend: `tests/unit/contracts/test_audit.py`

```python
"""Tests for ADR-019 TokenOutcome dataclass shape change."""

from datetime import datetime, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elspeth.contracts.audit import TokenOutcome
from elspeth.contracts.enums import TerminalOutcome, TerminalPath, _LEGAL_TERMINAL_PAIRS


class TestTokenOutcomeTwoAxis:
    """ADR-019 Phase 1: TokenOutcome carries (outcome, path, completed)."""

    def test_completed_outcome_has_outcome_path_completed(self) -> None:
        """A completed-state TokenOutcome has all three two-axis fields."""
        record = TokenOutcome(
            outcome_id="out_test_01",
            run_id="run_001",
            token_id="tok_001",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=datetime.now(timezone.utc),
            sink_name="primary",
        )
        assert record.outcome == TerminalOutcome.SUCCESS
        assert record.path == TerminalPath.DEFAULT_FLOW
        assert record.completed is True

    def test_buffered_outcome_has_null_outcome_buffered_path(self) -> None:
        """A non-terminal (BUFFERED) TokenOutcome has outcome=None, path=BUFFERED, completed=False."""
        record = TokenOutcome(
            outcome_id="out_test_02",
            run_id="run_001",
            token_id="tok_001",
            outcome=None,
            path=TerminalPath.BUFFERED,
            completed=False,
            recorded_at=datetime.now(timezone.utc),
            batch_id="batch_001",
        )
        assert record.outcome is None
        assert record.path == TerminalPath.BUFFERED
        assert record.completed is False

    def test_completed_xor_outcome_invariant_completed_true_outcome_none(self) -> None:
        """Tier 1: completed=True with outcome=None is an invariant violation — crash."""
        with pytest.raises(ValueError, match="completed"):
            TokenOutcome(
                outcome_id="out_test_03",
                run_id="run_001",
                token_id="tok_001",
                outcome=None,
                path=TerminalPath.DEFAULT_FLOW,
                completed=True,  # mismatch
                recorded_at=datetime.now(timezone.utc),
            )

    def test_completed_xor_outcome_invariant_completed_false_outcome_set(self) -> None:
        """Tier 1: completed=False with outcome=SUCCESS is an invariant violation — crash."""
        with pytest.raises(ValueError, match="completed"):
            TokenOutcome(
                outcome_id="out_test_04",
                run_id="run_001",
                token_id="tok_001",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.BUFFERED,
                completed=False,  # mismatch
                recorded_at=datetime.now(timezone.utc),
            )

    def test_legal_pair_required(self) -> None:
        """Tier 1: an unknown (outcome, path) pair is an invariant violation — crash."""
        with pytest.raises(ValueError, match="_LEGAL_TERMINAL_PAIRS"):
            TokenOutcome(
                outcome_id="out_test_05",
                run_id="run_001",
                token_id="tok_001",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,  # SUCCESS+UNROUTED is not a legal pair
                completed=True,
                recorded_at=datetime.now(timezone.utc),
            )

    @given(
        outcome=st.sampled_from(list(TerminalOutcome)),
        path=st.sampled_from(list(TerminalPath)),
    )
    def test_all_illegal_completed_pairs_rejected(
        self,
        outcome: TerminalOutcome,
        path: TerminalPath,
    ) -> None:
        """Tier 1: every pair outside _LEGAL_TERMINAL_PAIRS is rejected."""
        if (outcome, path) in _LEGAL_TERMINAL_PAIRS:
            return

        with pytest.raises(ValueError):
            TokenOutcome(
                outcome_id="out_prop_illegal",
                run_id="run_001",
                token_id="tok_001",
                outcome=outcome,
                path=path,
                completed=True,
                recorded_at=datetime.now(timezone.utc),
            )

    @given(pair=st.sampled_from(list(_LEGAL_TERMINAL_PAIRS)))
    def test_all_legal_pairs_accepted(
        self,
        pair: tuple[TerminalOutcome | None, TerminalPath],
    ) -> None:
        """Tier 1: every closed-set legal pair is accepted at the shape layer.

        This dataclass test intentionally covers the completed/outcome/path shape
        only. Pair-specific discriminator fields such as sink_name, batch_id, and
        error_hash are enforced by the recorder write guard and loader read guard
        against _TERMINAL_PAIR_FIELD_CONSTRAINTS below.
        """
        outcome, path = pair
        TokenOutcome(
            outcome_id="out_prop_legal",
            run_id="run_001",
            token_id="tok_001",
            outcome=outcome,
            path=path,
            completed=outcome is not None,
            recorded_at=datetime.now(timezone.utc),
        )
```

**Step 2: Run the tests to verify they fail (RED)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py::TestTokenOutcomeTwoAxis -v`

Expected: All seven tests fail with `TypeError` ("unexpected keyword argument 'path'") — the dataclass doesn't have the new fields yet. The Hypothesis property tests may report a single shrunk example; the important RED signal is that every generated valid and invalid pair currently reaches the old dataclass shape.

**Step 3: Update the dataclass**

Apply this edit to `src/elspeth/contracts/audit.py:673-703`:

```python
# OLD:
@dataclass(frozen=True, slots=True)
class TokenOutcome:
    """Recorded terminal state for a token.

    Captures the moment a token reached its terminal (or buffered) state.
    Part of AUD-001 audit integrity - explicit rather than derived.
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: RowOutcome  # Direct type, not forward reference
    is_terminal: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on outcome type)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
    expected_branches_json: str | None = None  # Branch contract for FORKED/EXPANDED

    def __post_init__(self) -> None:
        """Validate enum and bool fields - Tier 1 crash on invalid types."""
        _validate_enum(self.outcome, RowOutcome, "outcome")
        if not isinstance(self.is_terminal, bool):
            raise TypeError(f"is_terminal must be bool, got {type(self.is_terminal).__name__}: {self.is_terminal!r}")

# NEW:
@dataclass(frozen=True, slots=True)
class TokenOutcome:
    """Recorded terminal state for a token (ADR-019 two-axis model).

    Captures the moment a token reached its terminal (or buffered) state.
    Part of AUD-001 audit integrity - explicit rather than derived.

    ``outcome`` is the lifecycle answer (TerminalOutcome) when ``completed=True``,
    or ``None`` when ``completed=False`` (only BUFFERED today). ``path`` is the
    provenance answer (TerminalPath), always populated. ``completed`` mirrors
    the prior ``is_terminal`` field per ADR-019 sub-decision 3.

    The ``__post_init__`` invariants enforce three Tier 1 constraints:
    1. ``completed XOR (outcome IS NULL)`` — bool/outcome consistency.
    2. ``(outcome, path) ∈ _LEGAL_TERMINAL_PAIRS`` when completed=True.
    3. ``path == BUFFERED`` when completed=False (only non-terminal path today).
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: TerminalOutcome | None
    path: TerminalPath
    completed: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on (outcome, path) pair)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
    expected_branches_json: str | None = None  # Branch contract for FORK_PARENT/EXPAND_PARENT

    def __post_init__(self) -> None:
        """Validate two-axis invariants — Tier 1 crash on invalid combinations."""
        # I0a: completed/outcome consistency
        if not isinstance(self.completed, bool):
            raise TypeError(
                f"completed must be bool, got {type(self.completed).__name__}: {self.completed!r}"
            )
        if self.completed and self.outcome is None:
            raise ValueError(
                f"TokenOutcome {self.outcome_id}: completed=True requires non-NULL outcome "
                f"(ADR-019 § Decision invariant: completed XOR (outcome IS NULL))"
            )
        if not self.completed and self.outcome is not None:
            raise ValueError(
                f"TokenOutcome {self.outcome_id}: completed=False requires outcome=None "
                f"(got outcome={self.outcome!r})"
            )

        # I0b: enum types
        if self.outcome is not None:
            _validate_enum(self.outcome, TerminalOutcome, "outcome")
        _validate_enum(self.path, TerminalPath, "path")

        # I0c: legal pair when terminal
        if self.completed:
            assert self.outcome is not None  # invariant from I0a
            if (self.outcome, self.path) not in _LEGAL_TERMINAL_PAIRS:
                raise ValueError(
                    f"TokenOutcome {self.outcome_id}: ({self.outcome!r}, {self.path!r}) "
                    f"is not in _LEGAL_TERMINAL_PAIRS — see ADR-019 § Mapping table."
                )
        else:
            # I0d: non-terminal path
            if self.path != TerminalPath.BUFFERED:
                raise ValueError(
                    f"TokenOutcome {self.outcome_id}: completed=False requires "
                    f"path=BUFFERED (got path={self.path!r})"
                )
```

Update the imports at the top of `contracts/audit.py`. The live file imports
`RowOutcome` inside a grouped enum import alongside still-used names such as
`BatchStatus` and `RunStatus`; remove only `RowOutcome` from that group and add
the terminal symbols while preserving the other enum imports:

```python
from elspeth.contracts.enums import (
    _LEGAL_TERMINAL_PAIRS,
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    ReproducibilityGrade,
    RoutingMode,
    RunStatus,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
```

**Step 4: Run the tests to verify they pass (GREEN)**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py::TestTokenOutcomeTwoAxis -v`

Expected: All seven tests pass.

**Step 5: Verify other tests in the contracts test suite still pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py -v`

Expected: every existing test that constructs `TokenOutcome` will fail because the old shape (`outcome=RowOutcome.X, is_terminal=True`) is gone. Update each existing test fixture in this file in the same commit to use `(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW, completed=True)` style. This is schema-dependent test fixup per Phase 5 § Test triage; do not defer it.

**Definition of Done:**
- [ ] Seven new tests in `TestTokenOutcomeTwoAxis` pass, including Hypothesis sweeps over every legal pair in `_LEGAL_TERMINAL_PAIRS` and every illegal `(TerminalOutcome, TerminalPath)` pair outside it
- [ ] All pre-existing tests in `test_audit.py` updated and passing
- [ ] `TokenOutcome.__post_init__` enforces completed/outcome/path I0 shape invariants; recorder and loader tests below enforce the required/exact/forbidden discriminator-field constraints
- [ ] No `RowOutcome` references remain in `contracts/audit.py` (grep verifies)
- [ ] mypy passes on `src/elspeth/contracts/audit.py`

---

### Task 1.3: Retype `RowResult` dataclass

**Files:**
- Modify: `src/elspeth/contracts/results.py:379-421`

**Step 1: Write the failing test FIRST**

Extend `tests/unit/contracts/test_results.py` with:

```python
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract


def _pipeline_row(data: dict[str, object]) -> PipelineRow:
    return PipelineRow(data, SchemaContract(mode="OBSERVED", fields=(), locked=False))


class TestRowResultTwoAxis:
    """ADR-019 Phase 1: RowResult carries (outcome, path) at the producer site."""

    def test_completed_row_result(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        result = RowResult(
            token=token,
            final_data=_pipeline_row({"k": "v"}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.DEFAULT_FLOW

    def test_routed_on_error_requires_error_field(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        with pytest.raises(OrchestrationInvariantError, match="ON_ERROR_ROUTED"):
            RowResult(
                token=token,
                final_data=_pipeline_row({"k": "v"}),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                sink_name="error_sink",
                error=None,  # missing — must crash
            )

    def test_buffered_row_result(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        result = RowResult(
            token=token,
            final_data=_pipeline_row({"k": "v"}),
            outcome=None,
            path=TerminalPath.BUFFERED,
        )
        assert result.outcome is None
        assert result.path == TerminalPath.BUFFERED

    def test_illegal_completed_pair_rejected_before_recording(self) -> None:
        token = TokenInfo(token_id="tok_001", row_id="row_001", run_id="run_001")
        with pytest.raises(OrchestrationInvariantError, match="legal"):
            RowResult(
                token=token,
                final_data=_pipeline_row({"k": "v"}),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,
                sink_name="primary",
            )
```

**Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_results.py::TestRowResultTwoAxis -v`

Expected fail: `TypeError: __init__() got an unexpected keyword argument 'path'`.

**Step 3: Update the dataclass**

Apply edit at `src/elspeth/contracts/results.py:379-421`:

```python
@dataclass(frozen=True, slots=True)
class RowResult:
    """Final result of processing a row through the pipeline (ADR-019 two-axis).

    Producers emit (outcome, path) pairs at the producer site per ADR-019
    § "Classification is producer-declared, not topology-derivable." The
    recorder writes the pair without re-derivation.

    Fields:
        token: Token identity for this row instance
        final_data: Final row data as PipelineRow (may be original if failed early)
        outcome: Lifecycle answer (None for non-terminal BUFFERED rows)
        path: Provenance answer (always populated)
        sink_name: For paths that reach a sink, the destination sink name
        error: For (FAILURE, ON_ERROR_ROUTED), type-safe error details for audit
    """

    token: TokenInfo
    final_data: PipelineRow
    outcome: TerminalOutcome | None
    path: TerminalPath
    sink_name: str | None = None
    error: FailureInfo | None = None

    def __post_init__(self) -> None:
        # The recorder will rerun the (outcome, path) legality check; here we
        # catch obvious construction-site bugs early. The post-Stage-2 contract
        # is: every legal terminal pair has its required fields documented in
        # the ADR Implementation Notes table; we mirror those here.
        if self.outcome is not None and (self.outcome, self.path) not in _LEGAL_TERMINAL_PAIRS:
            raise OrchestrationInvariantError(
                f"RowResult: illegal (outcome, path) pair: ({self.outcome!r}, {self.path!r})"
            )
        if self.outcome is None and self.path != TerminalPath.BUFFERED:
            raise OrchestrationInvariantError(
                f"RowResult: outcome=None requires path=BUFFERED, got path={self.path!r}"
            )
        if self.outcome is not None and self.path == TerminalPath.BUFFERED:
            raise OrchestrationInvariantError(
                f"RowResult: path=BUFFERED requires outcome=None, got outcome={self.outcome!r}"
            )
        if self.outcome is None and self.path == TerminalPath.BUFFERED and self.sink_name is not None:
            raise OrchestrationInvariantError(
                "RowResult: BUFFERED rows must not set sink_name before terminal recording"
            )

        # Per-pair sink_name and error invariants
        if self.path == TerminalPath.DEFAULT_FLOW and self.sink_name is None:
            raise OrchestrationInvariantError(
                "(SUCCESS, DEFAULT_FLOW) outcome requires sink_name to be set"
            )
        if self.path == TerminalPath.GATE_ROUTED and self.sink_name is None:
            raise OrchestrationInvariantError(
                "(SUCCESS, GATE_ROUTED) outcome requires sink_name to be set"
            )
        if self.path == TerminalPath.ON_ERROR_ROUTED:
            if self.sink_name is None:
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires sink_name to be set"
                )
            if self.error is None:
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires error (FailureInfo) to be set — "
                    "the originating transform error must be captured on the outcome record for "
                    "single-hop audit attributability."
                )
            if not isinstance(self.error, FailureInfo):
                raise OrchestrationInvariantError(
                    "(FAILURE, ON_ERROR_ROUTED) outcome requires error to be a FailureInfo instance"
                )
        if self.path == TerminalPath.COALESCED and self.sink_name is None:
            raise OrchestrationInvariantError(
                "(SUCCESS, COALESCED) outcome requires sink_name to be set"
            )
```

Update imports — replace `from elspeth.contracts.enums import RowOutcome` with
`from elspeth.contracts.enums import TerminalOutcome, TerminalPath, _LEGAL_TERMINAL_PAIRS`.
This is an intentional use of the Stage 1 closed-set partition; do not duplicate
or rederive the legal-pair set locally.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_results.py::TestRowResultTwoAxis -v`

Expected: all four tests pass.

**Step 5: Update existing RowResult tests in this file**

Pre-existing test fixtures construct `RowResult(outcome=RowOutcome.X, ...)`. Update each to the new shape with the canonical pair from `tests/unit/contracts/test_enums.py::_ROW_OUTCOME_TO_TWO_AXIS_MAPPING`. This is schema-dependent test fixup per Phase 5; do NOT defer.

**Definition of Done:**
- [ ] Four new TestRowResultTwoAxis tests pass
- [ ] All pre-existing RowResult tests updated and passing
- [ ] `__post_init__` enforces membership in `_LEGAL_TERMINAL_PAIRS` plus BUFFERED/non-terminal consistency invariants
- [ ] mypy passes
- [ ] No `RowOutcome` references remain in `contracts/results.py` (grep verifies)

---

### Task 1.4: Retype `PendingOutcome` dataclass

**Files:**
- Modify: `src/elspeth/contracts/engine.py:46-100`

**Step 1: Write the failing test FIRST**

Extend `tests/unit/contracts/test_engine_contracts.py`:

```python
class TestPendingOutcomeTwoAxis:
    """ADR-019 Phase 1: PendingOutcome carries (outcome, path) for sink-durable recording."""

    def test_pending_outcome_completed(self) -> None:
        po = PendingOutcome(
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
        assert po.outcome == TerminalOutcome.SUCCESS
        assert po.path == TerminalPath.DEFAULT_FLOW
        assert po.error_hash is None

    def test_pending_outcome_routed_on_error_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                error_hash=None,  # required for ON_ERROR_ROUTED path
            )

    def test_pending_outcome_failed_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
                error_hash=None,
            )

    def test_pending_outcome_quarantined_requires_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
                error_hash=None,
            )

    def test_pending_outcome_completed_must_not_have_hash(self) -> None:
        with pytest.raises(ValueError, match="error_hash"):
            PendingOutcome(
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                error_hash="abcd1234abcd1234",
            )

    def test_pending_outcome_rejects_illegal_completed_pair(self) -> None:
        with pytest.raises(ValueError, match="legal"):
            PendingOutcome(
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,
            )

    def test_pending_outcome_none_requires_buffered_path(self) -> None:
        with pytest.raises(ValueError, match="BUFFERED"):
            PendingOutcome(
                outcome=None,
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_pending_outcome_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            PendingOutcome(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
```

**Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_engine_contracts.py::TestPendingOutcomeTwoAxis -v`

Expected fail: `TypeError: __init__() got an unexpected keyword argument 'path'`.

**Step 3: Update the dataclass**

Apply edit at `src/elspeth/contracts/engine.py:46-100`:

```python
# Replace the entire PendingOutcome class:
@dataclass(frozen=True, slots=True, kw_only=True)
class PendingOutcome:
    """Pending token outcome waiting for sink durability confirmation (ADR-019).

    Carries (outcome, path) pairs through the pending_tokens queue for sink
    durability sequencing per the original ADR-018 motivation: token outcomes
    must only be recorded after sink write + flush complete successfully.

    The ``_REQUIRES_ERROR_HASH_PATHS`` set encodes the ADR-019 mapping:
    paths that require error_hash for single-hop audit attributability are
    (FAILURE, ON_ERROR_ROUTED), (FAILURE, UNROUTED), (FAILURE, QUARANTINED_AT_SOURCE),
    (TRANSIENT, SINK_FALLBACK_TO_FAILSINK), and (FAILURE, SINK_DISCARDED).
    """

    # Paths that require ``error_hash`` on PendingOutcome. Indexed by path
    # rather than (outcome, path) because the path uniquely identifies the
    # error-carrying scenarios under the new model.
    _REQUIRES_ERROR_HASH_PATHS: ClassVar[frozenset[TerminalPath]] = frozenset(
        {
            TerminalPath.ON_ERROR_ROUTED,
            TerminalPath.UNROUTED,
            TerminalPath.QUARANTINED_AT_SOURCE,
            TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            TerminalPath.SINK_DISCARDED,
        }
    )

    outcome: TerminalOutcome | None
    path: TerminalPath
    error_hash: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is None:
            if self.path != TerminalPath.BUFFERED:
                raise ValueError(
                    f"PendingOutcome with outcome=None requires path=BUFFERED, got {self.path.name}"
                )
        elif (self.outcome, self.path) not in _LEGAL_TERMINAL_PAIRS:
            raise ValueError(
                f"PendingOutcome has illegal (outcome, path) pair: ({self.outcome.name}, {self.path.name})"
            )

        if self.path in self._REQUIRES_ERROR_HASH_PATHS and (
            self.error_hash is None or not self.error_hash.strip()
        ):
            raise ValueError(
                f"PendingOutcome with path={self.path.name} requires non-empty error_hash"
            )
        if self.path not in self._REQUIRES_ERROR_HASH_PATHS and self.error_hash is not None:
            raise ValueError(
                f"PendingOutcome with path={self.path.name} must not have error_hash"
            )
```

Update imports: replace `from elspeth.contracts.enums import RowOutcome` with
`from elspeth.contracts.enums import TerminalOutcome, TerminalPath, _LEGAL_TERMINAL_PAIRS`.
This is load-bearing because `SinkExecutor.write()` consumes `PendingOutcome`
before the recorder's final `record_token_outcome(...)` call. Illegal pairs must
crash at PendingOutcome construction time, before any sink side effect can run.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_engine_contracts.py::TestPendingOutcomeTwoAxis -v`

Expected: eight tests pass.

**Step 5: Local contract checkpoint only**

Do not run `git add` or `git commit` here. This is an intra-phase verification
checkpoint so the executor can confirm the three contract dataclasses are
internally coherent before touching `TokenCompleted`. The first legal staging
and commit operation for these files is the atomic Phases 1-3 commit in Phase 3.

**Definition of Done:**
- [ ] Eight new TestPendingOutcomeTwoAxis tests pass
- [ ] `PendingOutcome` is keyword-only so old positional call sites fail clearly instead of silently shifting `error_hash` into `path`
- [ ] `PendingOutcome.__post_init__` enforces membership in `_LEGAL_TERMINAL_PAIRS` before sink side effects
- [ ] mypy passes
- [ ] No `RowOutcome` references remain in `contracts/engine.py`

---

### Task 1.5: Retype `TokenCompleted` telemetry event

**Files:**
- Modify: `src/elspeth/contracts/events.py:242-249`
- Modify: `src/elspeth/telemetry/serialization.py:61-86` (shared enum serialization already handles both fields; add regression coverage, not custom branching)
- Modify: `src/elspeth/telemetry/filtering.py:63` (row-level event match should still include `TokenCompleted`)
- Modify: `tests/unit/telemetry/test_contracts.py` (event factory uses `TerminalOutcome` + `TerminalPath`)
- Modify: `tests/unit/telemetry/test_filtering.py` (TokenCompleted helper uses the new two-axis payload)
- Modify: `tests/unit/telemetry/test_property_based.py` (TokenCompleted strategy/helper uses the new two-axis payload)
- Modify: affected exporter tests under `tests/unit/telemetry/exporters/` that construct or assert `TokenCompleted` payloads (`test_console.py`, `test_otlp.py`, `test_otlp_integration.py`, `test_azure_monitor.py`, `test_azure_monitor_integration.py`, `test_datadog.py`, `test_datadog_integration.py`)

**Step 1: Write the failing test FIRST**

Extend `tests/unit/contracts/test_events.py`:

```python
from datetime import datetime, timezone


class TestTokenCompletedTwoAxis:
    """ADR-019 Phase 1: TokenCompleted telemetry event carries (outcome, path)."""

    def test_token_completed_carries_outcome_and_path(self) -> None:
        evt = TokenCompleted(
            timestamp=datetime.now(timezone.utc),
            run_id="run_001",
            row_id="row_001",
            token_id="tok_001",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert evt.outcome == TerminalOutcome.SUCCESS
        assert evt.path == TerminalPath.DEFAULT_FLOW
```

**Step 2: Run RED**

Expected: `TypeError: __init__() got an unexpected keyword argument 'path'`.

**Step 3: Update the dataclass**

```python
# OLD (lines 242-249):
@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    """Emitted when a token reaches its terminal state."""

    row_id: str
    token_id: str
    outcome: RowOutcome
    sink_name: str | None

# NEW:
@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    """Emitted when a token reaches its terminal state (ADR-019 two-axis)."""

    row_id: str
    token_id: str
    outcome: TerminalOutcome | None
    path: TerminalPath
    sink_name: str | None
```

Update imports at the top of `events.py` to import `TerminalOutcome` and `TerminalPath` instead of `RowOutcome`.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_events.py::TestTokenCompletedTwoAxis -v`

Expected: pass.

**Step 5: Update telemetry consumers in this file and downstream**

Telemetry consumers that read `TokenCompleted.outcome` and downstream-produce
`RowOutcome`-named values exist in `tests/unit/telemetry/` and exporter
coverage. Grep:

```bash
grep -rn "TokenCompleted" src/elspeth/telemetry tests/unit/telemetry
```

For each pure telemetry consumer, update field reads from `evt.outcome.X`
(RowOutcome) to `(evt.outcome, evt.path)` pair handling. The shared
`serialize_event_attributes()` helper already serializes Enum values, so the
required regression is explicit coverage that a `TokenCompleted` event produces
both `"outcome": "success"` and `"path": "default_flow"` (or another legal
pair), and filtering still treats `TokenCompleted` as row-level telemetry.

Minimum concrete coverage:

```python
from datetime import UTC, datetime

from elspeth.contracts import TokenCompleted
from elspeth.contracts.enums import TerminalOutcome, TerminalPath, TelemetryGranularity
from elspeth.telemetry.filtering import should_emit
from elspeth.telemetry.serialization import serialize_event_attributes


def _token_completed_two_axis() -> TokenCompleted:
    return TokenCompleted(
        timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC),
        run_id="run-789",
        row_id="row-3",
        token_id="token-3",
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.DEFAULT_FLOW,
        sink_name="output_sink",
    )


def test_token_completed_serializes_outcome_and_path() -> None:
    attrs = serialize_event_attributes(_token_completed_two_axis())
    assert attrs["event_type"] == "TokenCompleted"
    assert attrs["outcome"] == "success"
    assert attrs["path"] == "default_flow"
    assert attrs["sink_name"] == "output_sink"


def test_token_completed_remains_row_level_telemetry() -> None:
    event = _token_completed_two_axis()
    assert should_emit(event, TelemetryGranularity.LIFECYCLE) is False
    assert should_emit(event, TelemetryGranularity.ROWS) is True
    assert should_emit(event, TelemetryGranularity.FULL) is True
```

Exporter tests that currently assert `"outcome": "completed"` must be updated
to a legal `TerminalOutcome.value` and must also assert the serialized `path`.
Do not rely on the contract-only `tests/unit/contracts/test_events.py` case to
cover telemetry; serialization/filtering/exporter paths are separate runtime
surfaces.

**Important phase boundary:** engine producer helpers that construct
`TokenCompleted` from row execution state (`processor.py`,
`engine/orchestrator/outcomes.py`, and any sink/transform producer helper) are
owned by Phase 2 alongside the `record_token_outcome(...)` producer-site flip.
Do not declare the engine producer surface green in Phase 1. Phase 1 proves the
event contract and pure telemetry consumers; Phase 2 must include the engine
producer tests that exercise `_emit_token_completed` with the new pair.

Do not add logger/structlog calls for row-level lifecycle decisions; the
Landscape audit trail remains the source of truth and telemetry is the
operational visibility channel.

**Definition of Done:**
- [ ] TestTokenCompletedTwoAxis passes
- [ ] Pure telemetry consumers in src/ updated; Phase 2 explicitly owns engine producer helpers
- [ ] `serialize_event_attributes(TokenCompleted(...))` regression asserts both `outcome` and `path`
- [ ] `should_emit(TokenCompleted(...), granularity)` regression still proves row-level filtering
- [ ] Exporter tests that construct `TokenCompleted` use `TerminalOutcome` + `TerminalPath` and assert the emitted `path`
- [ ] mypy passes

---

### Task 1.6: Update `record_token_outcome` recorder signature

**Files:**
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:203-307` (`_validate_outcome_fields`)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:802-880` (`record_token_outcome`)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:570-580, 785-795` (the two internal callers that emit `FORKED` and `EXPANDED` directly)
- Modify: `src/elspeth/core/landscape/data_flow_repository.py:895-940` (`get_token_outcome` ORDER BY at line 899; `get_token_outcomes_for_row` SELECT column list at line 930) — pure column rename `is_terminal` → `completed`; the loader at Task 1.7 already expects `row.completed`

**Step 1: Write the failing test FIRST**

Create or extend `tests/unit/core/landscape/test_data_flow_repository.py` with:

```python
class TestRecordTokenOutcomeTwoAxis:
    """ADR-019 Phase 1: recorder writes (outcome, path, completed) triple."""

    def test_record_completed_default_flow(self) -> None:
        _db, repo, _factory, _row_id, token_id = _make_repo_with_token()
        token_ref = TokenRef(token_id=token_id, run_id="run-1")

        outcome_id = repo.record_token_outcome(
            ref=token_ref,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert outcome_id
        loaded = repo.get_token_outcome(token_ref.token_id)
        assert loaded is not None
        assert loaded.outcome == TerminalOutcome.SUCCESS
        assert loaded.path == TerminalPath.DEFAULT_FLOW
        assert loaded.completed is True

    def test_record_buffered(self) -> None:
        _db, repo, _factory, _row_id, token_id = _make_repo_with_token()
        token_ref = TokenRef(token_id=token_id, run_id="run-1")

        outcome_id = repo.record_token_outcome(
            ref=token_ref,
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id="batch_001",
        )
        assert outcome_id
        loaded = repo.get_token_outcome(token_ref.token_id)
        assert loaded.outcome is None
        assert loaded.path == TerminalPath.BUFFERED
        assert loaded.completed is False

    def test_record_illegal_pair_crashes(self) -> None:
        _db, repo, _factory, _row_id, token_id = _make_repo_with_token()
        token_ref = TokenRef(token_id=token_id, run_id="run-1")

        with pytest.raises(ValueError, match=r"Unhandled \(outcome, path\) pair"):
            repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,  # illegal pair
                sink_name="x",
            )

    def test_record_default_flow_requires_sink_name(self) -> None:
        _db, repo, _factory, _row_id, token_id = _make_repo_with_token()
        token_ref = TokenRef(token_id=token_id, run_id="run-1")

        with pytest.raises(ValueError, match="sink_name"):
            repo.record_token_outcome(
                ref=token_ref,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name=None,
            )

    # Plus parametrized tests for every (outcome, path) constraint row from the
    # ADR Implementation Notes table at lines 638-660: required fields, exact
    # values, and forbidden sibling discriminator fields.
    #
    # Named coverage anchors required by the review:
    # - test_record_filter_dropped_requires_no_extra_fields
    # - test_record_expand_parent_requires_expand_group_id
    # - test_record_sink_discarded_requires_exact_discard_sink_name
    # - test_record_default_flow_rejects_error_hash
    # - test_record_buffered_rejects_sink_name
```

Use the live helper shape in `tests/unit/core/landscape/test_data_flow_repository.py`:
`_make_repo_with_token()` returns `(db, repo, factory, row_id, token_id)`. Do not
invent `audit_repo`, `run_id`, or `token_ref` pytest fixtures unless the plan
also adds them explicitly.

**Step 2: Run RED**

Expected: each test fails with `TypeError: record_token_outcome() got an unexpected keyword argument 'path'`.

**Step 3: Add shared terminal-pair field constraints and rewrite `_validate_outcome_fields`**

Replace the entire `_validate_outcome_fields` block (lines 203-307) with a `(outcome, path)`-driven version. The 13 if-branches in the old code map mechanically to the 13 legal pairs:

```python
# In contracts/audit.py, alongside TokenOutcome. This is the canonical discard
# sink sentinel for engine producers and web discard summaries; later tasks must
# import this constant instead of retaining "__discard__" literals or local copies.
DISCARD_SINK_NAME = "__discard__"


@dataclass(frozen=True, slots=True)
class TerminalPairFieldConstraints:
    """Column-level constraints for one ADR-019 (outcome, path) pair.

    ``required`` means the field must be non-NULL.
    ``exact`` means the field must equal the specified value.
    ``forbidden`` means the field must be NULL; extra context in a sibling
    discriminator column would make the audit row ambiguous.
    """

    required: tuple[str, ...] = ()
    exact: Mapping[str, object] = field(default_factory=dict)
    forbidden: tuple[str, ...] = ()


_DISCRIMINATOR_FIELDS = (
    "sink_name",
    "batch_id",
    "fork_group_id",
    "join_group_id",
    "expand_group_id",
    "error_hash",
)


def _forbid_except(*allowed: str) -> tuple[str, ...]:
    return tuple(field for field in _DISCRIMINATOR_FIELDS if field not in allowed)


# Column constraints per ADR-019 Implementation Notes table (lines 638-660),
# after Task 1.0 aligns the quarantine row to the existing error_hash column.
# This table is intentionally richer than "required fields": it also rejects
# fields whose presence would make the row ambiguous and pins discard mode to
# the sentinel sink name.
_TERMINAL_PAIR_FIELD_CONSTRAINTS: dict[
    tuple[TerminalOutcome | None, TerminalPath],
    TerminalPairFieldConstraints,
] = {
    (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW): TerminalPairFieldConstraints(
        required=("sink_name",),
        forbidden=_forbid_except("sink_name"),
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED): TerminalPairFieldConstraints(
        required=("sink_name",),
        forbidden=_forbid_except("sink_name"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED): TerminalPairFieldConstraints(
        required=("sink_name", "error_hash"),
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED): TerminalPairFieldConstraints(
        forbidden=_DISCRIMINATOR_FIELDS,
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.COALESCED): TerminalPairFieldConstraints(
        required=("sink_name", "join_group_id"),
        forbidden=_forbid_except("sink_name", "join_group_id"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.UNROUTED): TerminalPairFieldConstraints(
        required=("error_hash",),
        forbidden=_forbid_except("error_hash"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE): TerminalPairFieldConstraints(
        required=("error_hash",),
        forbidden=_forbid_except("error_hash"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK): TerminalPairFieldConstraints(
        required=("sink_name", "error_hash"),
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED): TerminalPairFieldConstraints(
        required=("sink_name", "error_hash"),
        exact={"sink_name": DISCARD_SINK_NAME},
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT): TerminalPairFieldConstraints(
        required=("fork_group_id",),
        forbidden=_forbid_except("fork_group_id"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT): TerminalPairFieldConstraints(
        required=("expand_group_id",),
        forbidden=_forbid_except("expand_group_id"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED): TerminalPairFieldConstraints(
        required=("batch_id",),
        forbidden=_forbid_except("batch_id"),
    ),
    (None, TerminalPath.BUFFERED): TerminalPairFieldConstraints(
        required=("batch_id",),
        forbidden=_forbid_except("batch_id"),
    ),
}


def _validate_outcome_fields(
    self,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None,
    batch_id: str | None,
    fork_group_id: str | None,
    join_group_id: str | None,
    expand_group_id: str | None,
    error_hash: str | None,
) -> None:
    """Validate required fields for the (outcome, path) pair.

    Per ADR-019 Implementation Notes invariant-translation table at
    docs/architecture/adr/019-two-axis-terminal-model.md lines 638-660.
    Defense-in-depth: producers SHOULD pass correct fields, but bugs in
    producer code crash here at write time rather than corrupting the
    audit DB.

    Raises:
        ValueError: If the pair is neither a legal terminal pair nor the
            non-terminal BUFFERED pair, or if a field constraint is violated.
    """
    pair = (outcome, path)
    if pair not in _TERMINAL_PAIR_FIELD_CONSTRAINTS:
        raise ValueError(
            f"Unhandled (outcome, path) pair in validation: {pair!r}. "
            f"See ADR-019 § Mapping table (lines 99-115) and update "
            f"_TERMINAL_PAIR_FIELD_CONSTRAINTS with the new pair."
        )
    constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
    field_values = {
        "sink_name": sink_name,
        "batch_id": batch_id,
        "fork_group_id": fork_group_id,
        "join_group_id": join_group_id,
        "expand_group_id": expand_group_id,
        "error_hash": error_hash,
    }
    for field_name in constraints.required:
        if field_values[field_name] is None:
            raise ValueError(
                f"({pair[0].name if pair[0] else 'NULL'}, {pair[1].name}) outcome "
                f"requires {field_name} but got None. "
                f"Contract violation — see ADR-019 § Implementation Notes table."
            )
    for field_name, expected in constraints.exact.items():
        if field_values[field_name] != expected:
            raise ValueError(
                f"({pair[0].name if pair[0] else 'NULL'}, {pair[1].name}) outcome "
                f"requires {field_name}={expected!r}, got {field_values[field_name]!r}. "
                f"Contract violation — see ADR-019 § Implementation Notes table."
            )
    for field_name in constraints.forbidden:
        if field_values[field_name] is not None:
            raise ValueError(
                f"({pair[0].name if pair[0] else 'NULL'}, {pair[1].name}) outcome "
                f"forbids {field_name}, got {field_values[field_name]!r}. "
                f"Contract violation — see ADR-019 § Implementation Notes table."
            )
```

Add `dataclass`, `field`, and `Mapping` imports in `contracts/audit.py` if they
are not already present for the new shared constraint dataclass. Update
`data_flow_repository.py` to import `_TERMINAL_PAIR_FIELD_CONSTRAINTS` from
`elspeth.contracts.audit`; do not leave a private duplicate table in the
repository module.

**Step 4: Rewrite `record_token_outcome` signature**

Replace the existing method (lines 802-880) with:

```python
def record_token_outcome(
    self,
    ref: TokenRef,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    *,
    sink_name: str | None = None,
    sink_node_id: str | None = None,
    artifact_id: str | None = None,
    batch_id: str | None = None,
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
    error_hash: str | None = None,
    context: Mapping[str, object] | None = None,
) -> str:
    """Record a token's (outcome, path) audit terminal in the audit trail.

    Called at the moment the producer determines the terminal pair. For
    BUFFERED tokens (outcome=None, path=BUFFERED), a second call records
    the actual lifecycle terminal when the batch flushes.

    Validates that the token belongs to the specified run_id before recording.
    Cross-run contamination crashes immediately per Tier 1 trust model.

    Per ADR-019 § "Classification is producer-declared, not topology-derivable":
    the (outcome, path) pair is the producer's declaration; the recorder
    writes it without re-derivation.

    Args:
        ref: TokenRef bundling token_id and run_id
        outcome: TerminalOutcome lifecycle answer, or None for BUFFERED
        path: TerminalPath provenance answer (always required)
        sink_name: For paths that reach a sink (REQUIRED for those)
        sink_node_id: Producer-declared sink node witness for failsink-paired
            outcomes. Phase 1 accepts it as a forward-compatible keyword only;
            Phase 4 makes it load-bearing for I1c real-time validation. It is
            not inserted into token_outcomes because the structural witness
            remains the node_states/artifacts rows.
        artifact_id: Producer-declared artifact witness for failsink-paired
            outcomes. Phase 1 accepts it as a forward-compatible keyword only;
            Phase 4 requires it for I1c so the recorder checks the exact
            artifact returned by the failsink write, not "any artifact for this
            run/sink." It is not inserted into token_outcomes.
        batch_id: For BATCH_CONSUMED / BUFFERED (REQUIRED)
        fork_group_id: For FORK_PARENT (REQUIRED)
        join_group_id: For COALESCED (REQUIRED)
        expand_group_id: For EXPAND_PARENT (REQUIRED)
        error_hash: For UNROUTED / QUARANTINED_AT_SOURCE / ON_ERROR_ROUTED /
                    SINK_FALLBACK_TO_FAILSINK / SINK_DISCARDED (REQUIRED)
        context: Optional additional context (stored as JSON)

    Returns:
        outcome_id for tracking

    Raises:
        ValueError: If (outcome, path) is illegal or required fields missing
        AuditIntegrityError: If token does not belong to the specified run, or
            if a cross-table invariant fails (Phase 4 will add I1c, I3 here)
        IntegrityError: If terminal outcome already exists for token
    """
    self._validate_outcome_fields(
        outcome,
        path,
        sink_name=sink_name,
        batch_id=batch_id,
        fork_group_id=fork_group_id,
        join_group_id=join_group_id,
        expand_group_id=expand_group_id,
        error_hash=error_hash,
    )
    self._validate_token_run_ownership(ref)

    outcome_id = f"out_{generate_id()[:12]}"
    completed = outcome is not None  # I0a invariant
    context_json = canonical_json(context) if context is not None else None

    self._ops.execute_insert(
        token_outcomes_table.insert().values(
            outcome_id=outcome_id,
            run_id=ref.run_id,
            token_id=ref.token_id,
            outcome=outcome.value if outcome is not None else None,
            path=path.value,
            completed=1 if completed else 0,
            recorded_at=now(),
            sink_name=sink_name,
            batch_id=batch_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            error_hash=error_hash,
            context_json=context_json,
        )
    )

    return outcome_id
```

**Step 5: Update the two atomic internal recorder inserts (FORKED at line 572, EXPANDED at line 787)**

These two sites do **not** call `record_token_outcome` today; they intentionally insert into `token_outcomes_table` inside the same `with self._db.connection() as conn:` transaction that creates the child tokens and `token_parents` rows. Preserve that atomicity. Do **not** replace these direct inserts with `self.record_token_outcome(...)`, because that would open a separate repository operation and reintroduce the crash window the current code explicitly avoids.

Update only the inserted column/value shape inside the existing transaction:

```python
# Line ~572 (FORKED parent recording inside fork_token's existing transaction):
result = conn.execute(
    token_outcomes_table.insert().values(
        outcome_id=outcome_id,
        run_id=parent_ref.run_id,
        token_id=parent_ref.token_id,
        outcome=TerminalOutcome.TRANSIENT.value,
        path=TerminalPath.FORK_PARENT.value,
        completed=1,
        recorded_at=now(),
        fork_group_id=fork_group_id,
        expected_branches_json=json.dumps(branches, allow_nan=False),
    )
)

# Line ~787 (EXPANDED parent recording inside expand_token's existing transaction):
result = conn.execute(
    token_outcomes_table.insert().values(
        outcome_id=outcome_id,
        run_id=parent_ref.run_id,
        token_id=parent_ref.token_id,
        outcome=TerminalOutcome.TRANSIENT.value,
        path=TerminalPath.EXPAND_PARENT.value,
        completed=1,
        recorded_at=now(),
        expand_group_id=expand_group_id,
        expected_branches_json=json.dumps({"count": count}, allow_nan=False),
    )
)
```

Keep the existing `rowcount == 0` checks after each insert. Add a focused test that monkeypatches the second write in each transaction to fail and proves children plus parent outcome do not commit partially.

**Step 5b: Rename `is_terminal` → `completed` in the read-side query sites (lines 899 and 930)**

The `record_token_outcome` rewrite in Step 4 changes the inserted column name from `is_terminal` to `completed`. Two read-side query sites in the same file still reference the old column name and must be renamed in lock-step or the queries will fail with a SQLAlchemy "no such column" error against the new schema. The loader at Task 1.7 already expects `row.completed`, so these are pure column renames with no semantic change.

```python
# Line ~899 (get_token_outcome ORDER BY clause):
# OLD:
.order_by(
    token_outcomes_table.c.is_terminal.desc(),  # Terminal first
    token_outcomes_table.c.recorded_at.desc(),  # Then by time
)

# NEW:
.order_by(
    token_outcomes_table.c.completed.desc(),    # Terminal first (ADR-019 column rename)
    token_outcomes_table.c.recorded_at.desc(),  # Then by time
)
```

```python
# Line ~930 (get_token_outcomes_for_row SELECT column list):
# OLD:
select(
    token_outcomes_table.c.outcome_id,
    token_outcomes_table.c.run_id,
    token_outcomes_table.c.token_id,
    token_outcomes_table.c.outcome,
    token_outcomes_table.c.is_terminal,
    token_outcomes_table.c.recorded_at,
    ...
)

# NEW:
select(
    token_outcomes_table.c.outcome_id,
    token_outcomes_table.c.run_id,
    token_outcomes_table.c.token_id,
    token_outcomes_table.c.outcome,
    token_outcomes_table.c.path,        # ADR-019 new column (always populated)
    token_outcomes_table.c.completed,   # ADR-019 column rename (was is_terminal)
    token_outcomes_table.c.recorded_at,
    ...
)
```

The `path` column must be added to this SELECT list because `TokenOutcomeLoader.load` (Task 1.7) reads `row.path` as a load-bearing field. Verify the SELECT lists every column the loader reads — `outcome_id`, `run_id`, `token_id`, `outcome`, `path`, `completed`, `recorded_at`, `sink_name`, `batch_id`, `fork_group_id`, `join_group_id`, `expand_group_id`, `error_hash`, `context_json`, `expected_branches_json`.

**Step 6: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_data_flow_repository.py::TestRecordTokenOutcomeTwoAxis -v`

Expected: all tests pass.

**Definition of Done:**
- [ ] `_validate_outcome_fields` rewritten against shared `_TERMINAL_PAIR_FIELD_CONSTRAINTS`
- [ ] `record_token_outcome` signature flipped to `(outcome, path)`
- [ ] Internal recorder callers updated for FORK_PARENT / EXPAND_PARENT
- [ ] Recorder tests include explicit FILTER_DROPPED and EXPAND_PARENT coverage anchors
- [ ] Read-side queries renamed: `get_token_outcome` ORDER BY (line ~899) and `get_token_outcomes_for_row` SELECT column list (line ~930) use `completed` (was `is_terminal`); SELECT list includes the new `path` column
- [ ] All TestRecordTokenOutcomeTwoAxis tests pass
- [ ] mypy passes
- [ ] No `RowOutcome` references remain in `data_flow_repository.py`
- [ ] No `is_terminal` references remain in `data_flow_repository.py` (verify with `grep -n "is_terminal" src/elspeth/core/landscape/data_flow_repository.py` returning empty)

---

### Task 1.7: Update `TokenOutcomeLoader.load` cross-checks

**Files:**
- Modify: `src/elspeth/core/landscape/model_loaders.py:525-609`
- Modify: `tests/unit/core/landscape/test_model_loaders.py`

**Step 1: Write the failing test FIRST**

Extend the existing loader-test module `tests/unit/core/landscape/test_model_loaders.py`:

```python
class TestTokenOutcomeLoaderTwoAxis:
    """ADR-019 Phase 1: loader runs two-axis cross-checks at read time."""

    def test_loads_completed_default_flow(self, audit_db) -> None:
        # Given a recorded (SUCCESS, DEFAULT_FLOW) outcome,
        # When the loader reads it,
        # Then we get a TokenOutcome with the correct fields.
        ...

    def test_completed_xor_outcome_violation_crashes(self, audit_db) -> None:
        # Given a tampered DB with completed=1 and outcome=NULL,
        # When the loader reads it,
        # Then AuditIntegrityError fires.
        ...

    def test_illegal_pair_in_db_crashes(self, audit_db) -> None:
        # Given a tampered DB with (SUCCESS, UNROUTED) — not legal,
        # Then AuditIntegrityError fires.
        ...

    def test_required_field_missing_crashes(self, audit_db) -> None:
        # Given a tampered DB with (SUCCESS, DEFAULT_FLOW) but sink_name=NULL,
        # Then AuditIntegrityError fires.
        ...

    @pytest.mark.parametrize("pair,constraints", _TERMINAL_PAIR_FIELD_CONSTRAINTS.items())
    def test_every_pair_constraint_row_is_enforced(self, audit_db, pair, constraints) -> None:
        outcome, path = pair
        # For every shared constraint row, direct-INSERT a DB row that violates
        # one required/exact/forbidden condition and assert the loader crashes.
        ...

    @pytest.mark.parametrize(
        "completed,outcome,path",
        [
            (1, None, "default_flow"),          # completed/outcome XOR violation
            (0, "success", "buffered"),        # completed/outcome XOR violation
            (0, None, "default_flow"),         # non-terminal must be BUFFERED
            (1, "success", None),              # path is never NULL
            (1, "success", "not_a_path"),      # invalid TerminalPath
            ("1", "success", "default_flow"), # completed must be int 0/1
            (True, "success", "default_flow"), # bool is not accepted as int
        ],
    )
    def test_tampered_shape_crashes(self, audit_db, completed, outcome, path) -> None:
        ...
```

(The fixtures in this test class can use direct INSERT statements through the SQLAlchemy connection — bypassing the recorder — to simulate the "tampered DB" scenario. This is a Tier 1 read guard test.)

**Step 2: Run RED**

Expected: tests fail because the loader still reads the old `is_terminal` column.

**Step 3: Rewrite `TokenOutcomeLoader.load`**

Replace lines 525-609 with:

```python
def load(self, row: SARow[Any]) -> TokenOutcome:
    """Load a TokenOutcome from a token_outcomes row, with Tier 1 cross-checks.

    Per ADR-019 § Cross-check invariants and CLAUDE.md "Three-Tier Trust Model":
    audit DB is OUR data; crash on any anomaly. The cross-checks fall into
    six layers, all run before constructing the dataclass:

    1. completed type check (must be int 0 or 1, never bool/str/etc.)
    2. outcome value coercion (TerminalOutcome.X or None)
    3. path value coercion (TerminalPath.X)
    4. completed XOR (outcome IS NULL) cross-check
    5. (outcome, path) ∈ _LEGAL_TERMINAL_PAIRS when completed
    6. required/exact/forbidden field cross-check per ADR-019 Implementation Notes table

    Raises:
        AuditIntegrityError: any cross-check violation
    """
    oid = row.outcome_id

    # 1. completed type check
    if type(row.completed) is not int or row.completed not in (0, 1):
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: invalid completed={row.completed!r} (expected int 0 or 1) "
            f"— audit integrity violation"
        )
    completed = row.completed == 1

    # 2. outcome value coercion (None for non-terminal)
    outcome: TerminalOutcome | None
    if row.outcome is None:
        outcome = None
    else:
        try:
            outcome = TerminalOutcome(row.outcome)
        except ValueError as exc:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: invalid outcome={row.outcome!r} not in TerminalOutcome — "
                f"audit integrity violation"
            ) from exc

    # 3. path value coercion (always non-NULL)
    if row.path is None:
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: path is NULL — audit integrity violation "
            f"(path is always populated under ADR-019)"
        )
    try:
        path = TerminalPath(row.path)
    except ValueError as exc:
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: invalid path={row.path!r} not in TerminalPath — "
            f"audit integrity violation"
        ) from exc

    # 4. completed XOR (outcome IS NULL)
    if completed != (outcome is not None):
        raise AuditIntegrityError(
            f"TokenOutcome {oid}: completed={completed} but outcome={outcome!r} — "
            f"completed must be true iff outcome is non-NULL "
            f"(ADR-019 § Decision invariant)"
        )

    # 5. (outcome, path) ∈ _LEGAL_TERMINAL_PAIRS when completed; else path == BUFFERED
    if completed:
        assert outcome is not None  # invariant from check 4
        if (outcome, path) not in _LEGAL_TERMINAL_PAIRS:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: ({outcome!r}, {path!r}) not in _LEGAL_TERMINAL_PAIRS "
                f"— see ADR-019 § Mapping table"
            )
    else:
        if path != TerminalPath.BUFFERED:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: completed=False requires path=BUFFERED, got {path!r} "
                f"— audit integrity violation"
            )

    # 6. Required/exact/forbidden field cross-check per ADR-019 Implementation
    # Notes table. Mirrors _TERMINAL_PAIR_FIELD_CONSTRAINTS in
    # contracts/audit.py — the read path repeats the write path's check because
    # audit-DB tampering bypasses the write check.
    pair: tuple[TerminalOutcome | None, TerminalPath] = (outcome, path)
    constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
    field_values = {
        "sink_name": row.sink_name,
        "batch_id": row.batch_id,
        "fork_group_id": row.fork_group_id,
        "join_group_id": row.join_group_id,
        "expand_group_id": row.expand_group_id,
        "error_hash": row.error_hash,
    }
    for field_name in constraints.required:
        if field_values[field_name] is None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: ({outcome!r}, {path!r}) requires {field_name} but "
                f"DB has NULL — audit integrity violation"
            )
    for field_name, expected in constraints.exact.items():
        if field_values[field_name] != expected:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: ({outcome!r}, {path!r}) requires "
                f"{field_name}={expected!r}, got {field_values[field_name]!r} — "
                f"audit integrity violation"
            )
    for field_name in constraints.forbidden:
        if field_values[field_name] is not None:
            raise AuditIntegrityError(
                f"TokenOutcome {oid}: ({outcome!r}, {path!r}) forbids {field_name}, "
                f"got {field_values[field_name]!r} — audit integrity violation"
            )

    return TokenOutcome(
        outcome_id=oid,
        run_id=row.run_id,
        token_id=row.token_id,
        outcome=outcome,
        path=path,
        completed=completed,
        recorded_at=row.recorded_at,
        sink_name=row.sink_name,
        batch_id=row.batch_id,
        fork_group_id=row.fork_group_id,
        join_group_id=row.join_group_id,
        expand_group_id=row.expand_group_id,
        error_hash=row.error_hash,
        context_json=row.context_json,
        expected_branches_json=row.expected_branches_json,
    )
```

Use the existing SQLAlchemy alias in `model_loaders.py`: the file imports
SQLAlchemy rows as `SARow`, while `Row` already names the audit dataclass from
`elspeth.contracts.audit`. The loader signature must remain
`def load(self, row: SARow[Any]) -> TokenOutcome:` to avoid shadowing the
dataclass import.

To avoid duplicating the field-constraint table in two files, keep
`_TERMINAL_PAIR_FIELD_CONSTRAINTS` in `src/elspeth/contracts/audit.py` (the
canonical location alongside `TokenOutcome`) and import it in both
`data_flow_repository.py` and `model_loaders.py`. The dict is the canonical
machine-readable encoding of ADR-019 § Implementation Notes table.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/core/landscape/test_model_loaders.py::TestTokenOutcomeLoaderTwoAxis -v`

Expected: all tests pass.

**Definition of Done:**
- [ ] Loader runs all six cross-checks
- [ ] Loader signature uses `SARow[Any]`, not `Row[Any]`
- [ ] `_TERMINAL_PAIR_FIELD_CONSTRAINTS` lives in `contracts/audit.py` and is shared
- [ ] Loader tamper tests cover every constraint-table row plus `path=None`, invalid path, invalid `completed` type, and completed/outcome XOR violations
- [ ] All TestTokenOutcomeLoaderTwoAxis tests pass
- [ ] mypy passes

---

### Task 1.8: Update public enum re-exports — `contracts/__init__.py` and `testing/__init__.py`

**Why this task exists:** Stage 1 added `TerminalOutcome` and `TerminalPath` to `src/elspeth/contracts/enums.py`. They are now cross-boundary contract types, while `RowOutcome` remains publicly exported from `elspeth.contracts` during the migration window. Make the public export decision explicit: add `TerminalOutcome` and `TerminalPath` to `contracts/__init__.py` beside `RowOutcome`, and add the testing-pack runtime exports used by fixtures.

The `elspeth.testing` package currently imports many contract enums only under `if TYPE_CHECKING`, which is **not** a runtime re-export surface; `from elspeth.testing import RowOutcome` fails on current HEAD. This task creates explicit runtime exports because several scaffolding helpers also need the enum defaults.

**Files:**
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/testing/__init__.py:31-45, 507-540, 715-740`

**Step 1: Add runtime enum imports for the contracts package**

In `src/elspeth/contracts/__init__.py`, import and list `TerminalOutcome` and
`TerminalPath` anywhere `RowOutcome` is already re-exported:

```python
from elspeth.contracts.enums import RowOutcome, TerminalOutcome, TerminalPath

__all__ = [
    ...
    "RowOutcome",
    "TerminalOutcome",
    "TerminalPath",
    ...
]
```

Verify:

```bash
.venv/bin/python -c "from elspeth.contracts import TerminalOutcome, TerminalPath; print('contracts: OK')"
```

**Step 2: Add runtime enum imports for the testing pack**

Do not edit only the `if TYPE_CHECKING:` block. Add real module-level imports near the existing runtime imports so `from elspeth.testing import TerminalOutcome, TerminalPath, RowOutcome` works:

```python
# Near the existing runtime imports, outside `if TYPE_CHECKING:`
from elspeth.contracts.enums import (
    RowOutcome,
    TerminalOutcome,
    TerminalPath,
)
```

**Step 3: Update default-outcome callsites in testing helper builders**

Lines 507-540 and 715-740 use `outcome or RowOutcome.COMPLETED` defaults in test scaffolding helpers. Update each to construct the (outcome, path) pair:

```python
# OLD (line ~520):
resolved_outcome = outcome or RowOutcome.COMPLETED

# NEW:
# Test helpers default to (SUCCESS, DEFAULT_FLOW) — the canonical
# "happy path" pair per ADR-019 mapping table.
if outcome is None and path is None:
    resolved_outcome = TerminalOutcome.SUCCESS
    resolved_path = TerminalPath.DEFAULT_FLOW
else:
    resolved_outcome = outcome
    resolved_path = path
```

The signature of these helpers will need a `path: TerminalPath | None = None` parameter alongside the existing `outcome` parameter. Each helper should accept both for caller flexibility.

**Step 4: Verify the testing re-export works**

```bash
.venv/bin/python -c "from elspeth.testing import TerminalOutcome, TerminalPath, RowOutcome; print('testing: OK')"
```

Expected output: `testing: OK`.

**Definition of Done:**
- [ ] `TerminalOutcome` and `TerminalPath` import at runtime from `elspeth.contracts`
- [ ] `TerminalOutcome`, `TerminalPath`, and `RowOutcome` import at runtime from `elspeth.testing`
- [ ] The enum imports are outside `if TYPE_CHECKING`
- [ ] Helper defaults updated to construct (outcome, path) pairs
- [ ] mypy passes
- [ ] ChaosEngine fixtures (in `tests/`) that depend on these helpers still compile

---

### Task 1.9: Fix downstream schema consumers (MCP analyzers, Web execution, exporter, lineage, formatters)

**Why this task exists:** Phase 1 changes the `token_outcomes` schema (column rename + new column + `outcome` value space). Downstream consumers in the MCP analyzer pack, the Web execution layer, the JSONL export contract/exporter, the lineage helper, and the CLI formatter read those columns directly via SQL or via the `TokenOutcome` dataclass's `is_terminal` property. The original plan missed these — they were discovered during a 2026-05-05 consumer-surface sweep. Without them in Phase 1, MCP `diagnose()` silently returns zero quarantines (Tier 1 audit-integrity violation per CLAUDE.md), `get_run_summary()` collapses the path axis, the Web run-diagnostics view crashes on the renamed column, and the export TypedDict drifts from the wire payload.

**Files:** see "Files touched in this phase" at the top of this document.

**Step 1: Write the silent-failure regression test (RED) for the diagnostics quarantine bug**

This is the most dangerous of the eight bugs: it doesn't crash, it silently lies. Encode it as a regression test FIRST so the fix is anchored.

Create: `tests/unit/mcp/test_diagnose_quarantine_count.py`

```python
"""ADR-019 B3: diagnose() must report the correct quarantine count under
the two-axis terminal model.

Pre-fix bug: mcp/analyzers/diagnostics.py:181 hardcodes
``outcome == "quarantined"``. After Phase 1's recorder writes
``outcome="failure"`` and ``path="quarantined_at_source"``, that filter
matches zero rows. diagnose() reports "0 quarantines" with confidence —
exactly the silent-wrong-result class CLAUDE.md Tier 1 forbids.
"""

import pytest

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.mcp.analyzers.diagnostics import diagnose
from tests.fixtures.landscape import make_factory, make_landscape_db


_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _create_completed_run_with_quarantines(
    db: LandscapeDB,
    factory: RecorderFactory,
    *,
    run_id: str = "quarantine-run",
    count: int = 3,
) -> None:
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=f"source-{run_id}",
        schema_config=_DYNAMIC_SCHEMA,
    )
    for row_index in range(count):
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=f"source-{run_id}",
            row_index=row_index,
            data={"col": f"bad-{row_index}"},
        )
        token = factory.data_flow.create_token(row.row_id)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run_id),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            error_hash=f"{row_index:064x}",
        )
    factory.run_lifecycle.complete_run(run_id, RunStatus.COMPLETED)


def test_diagnose_counts_quarantined_under_new_path() -> None:
    """A run with 3 quarantined rows reports 3 quarantines via diagnose()."""
    db = make_landscape_db()
    factory = make_factory(db)
    _create_completed_run_with_quarantines(db, factory, count=3)

    result = diagnose(db, factory)
    quarantine_problems = [
        p for p in result["problems"] if p["type"] == "quarantined_rows"
    ]
    assert len(quarantine_problems) == 1
    assert quarantine_problems[0]["count"] == 3, (
        f"Expected diagnose() to count 3 quarantined rows, got "
        f"{quarantine_problems[0]['count']}. The filter must match the "
        f"new TerminalPath.QUARANTINED_AT_SOURCE value, not the legacy "
        f"RowOutcome.QUARANTINED string 'quarantined'."
    )
```

Do not call `diagnose(audit_db_with_quarantines.conn)`: the live analyzer
signature is `diagnose(db: LandscapeDB, factory: RecorderFactory)`. The RED
signal must be the silent-wrong `count == 0` result from the legacy
`outcome == "quarantined"` filter, not a missing fixture or TypeError.

**Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/unit/mcp/test_diagnose_quarantine_count.py -v
```

Expected: test fails with `count == 0` (the bug). After the fix in Step 4, it passes.

**Step 3: Patch `mcp/analyzers/reports.py` (5 sites in `get_outcome_analysis`)**

```python
# Lines 657-665 area (the SELECT and GROUP BY):
# OLD:
stmt = (
    select(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.is_terminal,
        func.count(token_outcomes_table.c.outcome_id).label("count"),
    )
    .where(token_outcomes_table.c.run_id == run_id)
    .group_by(token_outcomes_table.c.outcome, token_outcomes_table.c.is_terminal)
)

# NEW:
# ADR-019: the outcome distribution is naturally keyed by (outcome, path)
# under the two-axis model. Group by both; the wire schema exposes both
# fields plus the renamed `completed` for query ergonomics.
stmt = (
    select(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
        func.count(token_outcomes_table.c.outcome_id).label("count"),
    )
    .where(token_outcomes_table.c.run_id == run_id)
    .group_by(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
    )
)
```

```python
# Lines 698-710 area (the dict construction + summary aggregation):
# OLD:
outcomes.append(
    {
        "outcome": row.outcome,
        "is_terminal": bool(row.is_terminal),
        "count": row.count,
    }
)
...
terminal_count = sum(o["count"] for o in outcomes if o["is_terminal"])
non_terminal_count = sum(o["count"] for o in outcomes if not o["is_terminal"])

# NEW:
outcomes.append(
    {
        "outcome": row.outcome,           # TerminalOutcome.value or NULL
        "path": row.path,                 # TerminalPath.value (always populated)
        "completed": bool(row.completed), # ADR-019 rename — replaces is_terminal
        "count": row.count,
    }
)
...
terminal_count = sum(o["count"] for o in outcomes if o["completed"])
non_terminal_count = sum(o["count"] for o in outcomes if not o["completed"])
```

**Step 4: Patch `mcp/analyzers/diagnostics.py:181` — the silent-failure fix**

```python
# OLD (line 181):
.where(token_outcomes_table.c.outcome == "quarantined")

# NEW:
# ADR-019: under the two-axis model, the quarantine signal lives on
# the path column, not the outcome column. (FAILURE, QUARANTINED_AT_SOURCE)
# is the canonical encoding; path alone is sufficient because no other
# legal pair uses QUARANTINED_AT_SOURCE.
.where(token_outcomes_table.c.path == TerminalPath.QUARANTINED_AT_SOURCE.value)
```

Add `from elspeth.contracts.enums import TerminalPath` to the imports.

**Step 5: Patch `mcp/types.py` — move `OutcomeDistributionEntry` before first use**

No forward references in this module. Do not add
`from __future__ import annotations`, do not quote `OutcomeDistributionEntry`,
and do not leave `RunSummaryReport` above the entry type. The import model
expects annotations to resolve at import time. Move `OutcomeDistributionEntry`
above `RunSummaryReport` before changing `RunSummaryReport.outcome_distribution`.

```python
# OLD (`RunSummaryReport`):
outcome_distribution: dict[str, int]  # dynamic outcome names

# OLD later definition:
class OutcomeDistributionEntry(TypedDict):
    outcome: str
    is_terminal: bool
    count: int

# NEW: place this class above `RunSummaryReport`, then remove the old later
# definition so `RunSummaryReport` has no forward reference.
class OutcomeDistributionEntry(TypedDict):
    """Single entry in outcome distribution (ADR-019 two-axis terminal model).

    Keyed by (outcome, path). ``outcome`` is the TerminalOutcome value or NULL
    for non-terminal rows. ``path`` is the TerminalPath value (always populated).
    ``completed`` mirrors the recorder's ``completed`` column — true iff the
    row reached a terminal state.
    """
    outcome: str | None  # TerminalOutcome.value or NULL (non-terminal)
    path: str            # TerminalPath.value
    completed: bool
    count: int


class RunSummaryReport(TypedDict):
    # ... existing fields unchanged ...
    outcome_distribution: list[OutcomeDistributionEntry]
```

**Wire-format note:** This is a breaking change to the MCP outcome-analysis response shape. Per CLAUDE.md "no legacy code" + "no users yet," breaking is acceptable; the renamed field flows from the recorder schema rename and the added field surfaces a load-bearing dimension that could not be exposed under the single-axis model. Operator MCP clients that destructured `is_terminal` must update to `completed`. Documented in `docs/operator/migrations/adr-019.md` (Phase 5).

**Step 5b: Patch `mcp/analyzers/reports.py::get_run_summary` path distribution**

`get_run_summary()` has a second outcome distribution that groups only by `token_outcomes.outcome` and returns a collapsed `{"outcome": count}` dictionary. Under the two-axis model this silently merges distinct lifecycle paths. Patch it in the same file as `get_outcome_analysis`:

```python
# OLD (lines 120-127):
outcome_query = (
    select(token_outcomes_table.c.outcome, func.count().label("count"))
    .where(token_outcomes_table.c.run_id == run_id)
    .group_by(token_outcomes_table.c.outcome)
)
outcome_distribution = {row.outcome: row.count for row in outcome_rows}

# NEW:
outcome_query = (
    select(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
        func.count().label("count"),
    )
    .where(token_outcomes_table.c.run_id == run_id)
    .group_by(
        token_outcomes_table.c.outcome,
        token_outcomes_table.c.path,
        token_outcomes_table.c.completed,
    )
)
outcome_distribution = [
    {
        "outcome": row.outcome,
        "path": row.path,
        "completed": bool(row.completed),
        "count": row.count,
    }
    for row in outcome_rows
]
```

Update `RunSummaryReport.outcome_distribution` and every test expectation for `get_run_summary()` so the report surface is path-aware. Do not keep a collapsed legacy distribution unless the field is explicitly renamed/deprecated and covered by tests. Verify the module imports without postponed annotations:

```bash
.venv/bin/python -c "from elspeth.mcp.types import RunSummaryReport, OutcomeDistributionEntry; print('types: OK')"
```

**Step 5c: Add DB-backed RED tests for both MCP report surfaces**

Existing analogous tests in `tests/unit/mcp/analyzers/test_reports.py` mock
SQLAlchemy rows. Keep those tests if useful, but they are not sufficient for
this migration because they do not prove the production query groups by
`(outcome, path, completed)` or that `RunSummaryReport`'s TypedDict matches the
real payload.

Add a concrete DB-backed test module, or extend
`tests/unit/mcp/analyzers/test_reports.py`, with real `LandscapeDB` +
`RecorderFactory` setup:

```python
from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.mcp.analyzers.reports import get_outcome_analysis, get_run_summary
from tests.fixtures.landscape import make_factory, make_landscape_db


def _record_token(factory, *, run_id: str, row_index: int, outcome, path, **fields):
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=f"source-{run_id}",
        row_index=row_index,
        data={"row": row_index},
    )
    token = factory.data_flow.create_token(row.row_id)
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=outcome,
        path=path,
        **fields,
    )


def test_outcome_reports_group_by_path_not_lifecycle_only() -> None:
    db = make_landscape_db()
    factory = make_factory(db)
    run_id = "two-axis-report-run"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=f"source-{run_id}",
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )
    _record_token(
        factory,
        run_id=run_id,
        row_index=0,
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.DEFAULT_FLOW,
        sink_name="primary",
    )
    _record_token(
        factory,
        run_id=run_id,
        row_index=1,
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.FILTER_DROPPED,
    )
    factory.run_lifecycle.complete_run(run_id, RunStatus.COMPLETED)

    outcome_analysis = get_outcome_analysis(db, factory, run_id)
    run_summary = get_run_summary(db, factory, run_id)

    for report in (outcome_analysis, run_summary):
        assert "error" not in report
        distribution = report["outcome_distribution"]
        buckets = {
            (entry["outcome"], entry["path"], entry["completed"]): entry["count"]
            for entry in distribution
        }
        assert buckets[("success", "default_flow", True)] == 1
        assert buckets[("success", "filter_dropped", True)] == 1
```

Before the SQL rewrite, this test must fail because `get_run_summary()` returns a
collapsed `dict[str, int]` and `get_outcome_analysis()` has no `path` field. The
post-fix payload must be an entry list for both surfaces.

**Step 6: Patch `web/execution/diagnostics.py:170`**

```python
# OLD:
.outerjoin(
    token_outcomes_table,
    and_(
        token_outcomes_table.c.token_id == tokens_table.c.token_id,
        token_outcomes_table.c.run_id == tokens_table.c.run_id,
        token_outcomes_table.c.is_terminal == 1,
    ),
)

# NEW:
.outerjoin(
    token_outcomes_table,
    and_(
        token_outcomes_table.c.token_id == tokens_table.c.token_id,
        token_outcomes_table.c.run_id == tokens_table.c.run_id,
        token_outcomes_table.c.completed == 1,
    ),
)
```

**Step 7: Patch `web/execution/discard_summary.py:92`**

First remove the local `DISCARD_SINK_NAME = "__discard__"` declaration from
`web/execution/discard_summary.py` and import the canonical sentinel from
`elspeth.contracts.audit`. The exact discard sentinel is now part of the
ADR-019 recorder contract, so web summaries and engine producers must not carry
their own copies.

```python
from elspeth.contracts.audit import DISCARD_SINK_NAME

# OLD:
.where(token_outcomes_table.c.is_terminal == 1)

# NEW:
.where(token_outcomes_table.c.completed == 1)
```

The discard-summary widget can ALSO benefit from a path-aware filter under the new model — the widget today counts rows where `sink_name == "__discard__"`, which conflates failsink-mode `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` (sink_name = the actual failsink name, NOT `__discard__`) with discard-mode `(FAILURE, SINK_DISCARDED)` (sink_name = `__discard__`). The existing `sink_name == DISCARD_SINK_NAME` filter already isolates the discard-mode case, so no semantic change is needed beyond the column rename. Verify the widget's count under the new model produces the same number as before for discard-mode-only runs.

**Step 7a: Add the direct discard-summary regression test**

Create `tests/unit/web/execution/test_discard_summary.py` instead of relying on
diagnostics tests to cover this separate query helper. The test must prove the
new query reads `completed` and still counts discard-mode rows exactly once:

```python
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.web.execution.discard_summary import load_discard_summaries_from_db
from tests.fixtures.landscape import make_factory, make_landscape_db


def test_discard_summary_counts_completed_discard_path() -> None:
    db = make_landscape_db()
    factory = make_factory(db)
    run_id = "discard-summary-run"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=f"source-{run_id}",
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=f"source-{run_id}",
        row_index=0,
        data={"id": "drop-me"},
    )
    token = factory.data_flow.create_token(row.row_id)
    factory.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=run_id),
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.SINK_DISCARDED,
        sink_name=DISCARD_SINK_NAME,
        error_hash="a" * 64,
    )

    summaries = load_discard_summaries_from_db(db, [run_id])

    assert summaries[run_id].total == 1
    assert summaries[run_id].sink_discards == 1
```

Before the query patch, this must fail because `token_outcomes_table.c.is_terminal`
no longer exists after the schema rename. After the patch, it passes via the
`completed == 1` filter.

**Step 7b: Patch `contracts/export_records.py::TokenOutcomeExportRecord`**

The JSONL exporter is typed by `TokenOutcomeExportRecord`; changing only the emitted payload leaves mypy and the wire contract inconsistent. Update the TypedDict before patching the exporter:

```python
# OLD:
class TokenOutcomeExportRecord(TypedDict):
    record_type: Literal["token_outcome"]
    run_id: str
    outcome_id: str
    token_id: str
    outcome: str
    is_terminal: bool
    recorded_at: str
    ...

# NEW:
class TokenOutcomeExportRecord(TypedDict):
    record_type: Literal["token_outcome"]
    run_id: str
    outcome_id: str
    token_id: str
    outcome: str | None
    path: str
    completed: bool
    recorded_at: str
    ...
```

The export tests must assert the TypedDict shape and the JSONL row shape together, so a future payload/contract drift fails in one place.

**Step 8: Patch `core/landscape/exporter.py:430`**

```python
# OLD (line 430):
"outcome": outcome.outcome.value,
"is_terminal": outcome.is_terminal,

# NEW:
# ADR-019: outcome value space is TerminalOutcome.value or None;
# completed mirrors the renamed is_terminal column; path is the new
# always-populated provenance field.
"outcome": outcome.outcome.value if outcome.outcome is not None else None,
"path": outcome.path.value,
"completed": outcome.completed,
```

The JSONL export is a wire format. The same breaking-change reasoning as the MCP TypedDict applies — operators regenerate exports from the new audit DB; old JSONL exports remain readable as historical snapshots but the new format adds `path` and renames `is_terminal` → `completed`.

**Step 9: Patch `core/landscape/lineage.py:118`**

```python
# OLD:
terminal_outcomes = [o for o in outcomes if o.is_terminal]

# NEW:
terminal_outcomes = [o for o in outcomes if o.completed]
```

Trivial property rename. The `TokenOutcome` dataclass exposes `completed: bool` after Phase 1 Task 1.2; `is_terminal` no longer exists.

**Step 10: Patch `core/landscape/formatters.py:170`**

```python
# OLD:
lines.append(f"Outcome: {result.outcome.outcome.name}")
if result.outcome.sink_name:
    lines.append(f"Sink: {result.outcome.sink_name}")
lines.append(f"Terminal: {result.outcome.is_terminal}")

# NEW:
# ADR-019: print both axes — operator CLI output should make the new
# (outcome, path) pair visible at a glance.
outcome_name = result.outcome.outcome.name if result.outcome.outcome else "NULL"
lines.append(f"Outcome: {outcome_name}")
lines.append(f"Path: {result.outcome.path.name}")
if result.outcome.sink_name:
    lines.append(f"Sink: {result.outcome.sink_name}")
lines.append(f"Completed: {result.outcome.completed}")
```

Add a direct assertion to `tests/unit/core/landscape/test_formatters.py` for the
operator-visible CLI text. Use the existing formatter fixtures in that module;
do not leave this covered only through exporter or lineage tests.

Minimum assertion shape:

```python
from datetime import UTC, datetime

from elspeth.contracts import RowLineage, Token, TokenOutcome
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.formatters import LineageTextFormatter
from elspeth.core.landscape.lineage import LineageResult


def test_format_explain_prints_two_axis_outcome_path_and_completed() -> None:
    now = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
    result = LineageResult(
        token=Token(token_id="tok-123", row_id="row-456", created_at=now, run_id="run-001"),
        source_row=RowLineage(
            row_id="row-456",
            run_id="run-789",
            source_node_id="src-node",
            row_index=0,
            source_data_hash="abc123",
            created_at=now,
            source_data={"id": 1},
            payload_available=True,
        ),
        node_states=(),
        routing_events=(),
        calls=(),
        parent_tokens=(),
        outcome=TokenOutcome(
            outcome_id="out-1",
            token_id="tok-123",
            run_id="run-789",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            sink_name="output",
            recorded_at=now,
        ),
    )

    rendered = LineageTextFormatter().format(result)

    assert "Outcome: SUCCESS" in rendered
    assert "Path: DEFAULT_FLOW" in rendered
    assert "Completed: True" in rendered
    assert "Terminal:" not in rendered
```

**Step 11: Final consumer-surface sweep**

After applying Steps 1-10, run a residual-hit grep to confirm zero misses:

```bash
set -euo pipefail

assert_no_hits() {
    label="$1"
    shift
    tmp_file="$(mktemp)"
    if "$@" >"$tmp_file"; then
        if [ -s "$tmp_file" ]; then
            printf '%s\n' "Unexpected findings for ${label}:"
            cat "$tmp_file"
            rm -f "$tmp_file"
            exit 1
        fi
    fi
    rm -f "$tmp_file"
}

assert_no_hits "residual token_outcomes_table.c.is_terminal/outcome.is_terminal reads" \
    grep -rn "token_outcomes_table.c.is_terminal\|outcome.is_terminal" src/elspeth/
# NOTE: the previous version of this sweep excluded data_flow_repository.py to silence
# noise from the in-scope sites at lines 573/788/859/868. That file-level exclusion
# also masked the read-side query sites at lines 899 and 930. Task 1.6 Step 5b now
# covers those lines, so the exclusion is removed — by end of Phase 1, the file is
# 100% migrated and any residual hit indicates a missed consumer.

assert_no_hits "residual hardcoded RowOutcome.value SQL filters" \
    grep -rn 'outcome\s*==\s*"\(completed\|routed\|routed_on_error\|forked\|failed\|quarantined\|diverted\|consumed_in_batch\|dropped_by_filter\|coalesced\|expanded\|buffered\)"' src/elspeth/

residual_is_terminal="$(grep -rn "is_terminal" src/elspeth/ || true)"
residual_is_terminal="$(printf '%s\n' "$residual_is_terminal" | grep -v "/web/execution/progress.py" | grep -v "/__pycache__/" || true)"
if [ -n "$residual_is_terminal" ]; then
    printf '%s\n' "Unexpected residual is_terminal references:"
    printf '%s\n' "$residual_is_terminal"
    exit 1
fi
```

The helper must exit 0 when grep finds no matches (grep's native no-match exit
code is 1) and exit 1 after printing findings when matches exist. If any sweep
shows a hit not addressed by Steps 1-10, STOP and surface to user — there's a
missed consumer.

**Step 12: GREEN — both regression tests + suite**

```bash
.venv/bin/python -m pytest tests/unit/mcp/test_diagnose_quarantine_count.py tests/unit/mcp/test_outcome_analysis.py -v
.venv/bin/python -m pytest \
    tests/unit/mcp/analyzers/test_reports.py \
    tests/unit/mcp/test_diagnostics.py \
    tests/unit/web/execution/test_diagnostics.py \
    tests/unit/web/execution/test_discard_summary.py \
    tests/unit/core/landscape/test_exporter.py \
    tests/unit/core/landscape/test_lineage.py \
    tests/unit/core/landscape/test_formatters.py \
    -q
```

Expected: all focused consumer tests green. Existing consumer tests with stale
assertions in these files (`is_terminal` field reads, hardcoded `"quarantined"`
string assertions) need fixture updates in the same commit because they are
co-located with the consumer code. Do not broaden this Phase 1 checkpoint to all
of `tests/unit/mcp/` unless the executor is also prepared to complete every
schema-dependent/assertion-only stale fixture that Phase 5 triages.

**Definition of Done:**
- [ ] Downstream consumer sites patched (`mcp/analyzers/reports.py` `get_outcome_analysis`, `mcp/analyzers/reports.py` `get_run_summary`, `mcp/types.py`, `mcp/analyzers/diagnostics.py`, `web/execution/diagnostics.py`, `web/execution/discard_summary.py`, `contracts/export_records.py`, `core/landscape/exporter.py`, `core/landscape/lineage.py`, `core/landscape/formatters.py`)
- [ ] `OutcomeDistributionEntry` is defined before `RunSummaryReport` in `mcp/types.py`; no quoted or postponed forward reference is used
- [ ] `.venv/bin/python -c "from elspeth.mcp.types import RunSummaryReport, OutcomeDistributionEntry; print('types: OK')"` passes
- [ ] B3 silent-zero-quarantine regression test passes
- [ ] B2 wire-schema test passes (new MCP outcome-analysis test)
- [ ] Direct discard-summary regression test proves `completed == 1` and discard-only count
- [ ] Direct formatter regression test proves `Outcome`, `Path`, and `Completed` appear and `Terminal` is gone
- [ ] Final consumer-surface sweep shows zero residual hits
- [ ] mypy clean across `mcp/`, `web/execution/`, `core/landscape/`
- [ ] No `RowOutcome` references remain in `src/elspeth/mcp/` or `src/elspeth/web/execution/`

---

### Task 1.10: Phase 1 local checkpoint (no git commit)

**Step 1: Run the focused Phase 1 tests**

```bash
.venv/bin/python -m pytest \
    tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
    tests/unit/core/landscape/test_database_compatibility_guards.py \
    tests/unit/contracts/test_audit.py::TestTokenOutcomeTwoAxis \
    tests/unit/contracts/test_results.py::TestRowResultTwoAxis \
    tests/unit/contracts/test_engine_contracts.py::TestPendingOutcomeTwoAxis \
    tests/unit/contracts/test_events.py::TestTokenCompletedTwoAxis \
    tests/unit/core/landscape/test_data_flow_repository.py::TestRecordTokenOutcomeTwoAxis \
    tests/unit/core/landscape/test_model_loaders.py::TestTokenOutcomeLoaderTwoAxis \
    tests/unit/mcp/test_diagnose_quarantine_count.py \
    tests/unit/mcp/test_outcome_analysis.py \
    tests/unit/mcp/analyzers/test_reports.py \
    tests/unit/web/execution/test_diagnostics.py \
    tests/unit/web/execution/test_discard_summary.py \
    tests/unit/core/landscape/test_exporter.py \
    tests/unit/core/landscape/test_lineage.py \
    tests/unit/core/landscape/test_formatters.py \
    tests/unit/telemetry/test_contracts.py \
    tests/unit/telemetry/test_filtering.py \
    tests/unit/telemetry/test_property_based.py \
    tests/unit/telemetry/exporters/test_console.py \
    tests/unit/telemetry/exporters/test_otlp.py \
    tests/unit/telemetry/exporters/test_otlp_integration.py \
    tests/unit/telemetry/exporters/test_azure_monitor.py \
    tests/unit/telemetry/exporters/test_azure_monitor_integration.py \
    tests/unit/telemetry/exporters/test_datadog.py \
    tests/unit/telemetry/exporters/test_datadog_integration.py \
    -v
```

All focused tests above must pass — including the downstream-consumer fixes and
the B3 regression test added in Task 1.9. Do **not** claim all of
`tests/unit/contracts/` or `tests/unit/core/landscape/` is green at the Phase 1
local checkpoint: Phase 5 still owns the repo-wide schema-dependent constructor
triage, assertion-only `RowOutcome` translations, and direct-DB-read test
updates needed for the final full-suite gate.

**Step 2: Run quality gates**

```bash
.venv/bin/python -m mypy src/elspeth/contracts/ src/elspeth/core/landscape/ src/elspeth/testing/ \
    src/elspeth/mcp/ src/elspeth/web/execution/ src/elspeth/telemetry/
.venv/bin/python -m ruff check src/elspeth/contracts/ src/elspeth/core/landscape/ src/elspeth/testing/ \
    src/elspeth/mcp/ src/elspeth/web/execution/ src/elspeth/telemetry/ \
    scripts/cicd/adr019_symbol_inventory.py tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
    tests/unit/contracts/ tests/unit/core/landscape/ tests/unit/mcp/ tests/unit/web/execution/ tests/unit/telemetry/
.venv/bin/python -m ruff format --check src/elspeth/contracts/ src/elspeth/core/landscape/ src/elspeth/testing/ \
    src/elspeth/mcp/ src/elspeth/web/execution/ src/elspeth/telemetry/ \
    scripts/cicd/adr019_symbol_inventory.py tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
    tests/unit/contracts/ tests/unit/core/landscape/ tests/unit/mcp/ tests/unit/web/execution/ tests/unit/telemetry/
```

All gates must pass.

**Step 3: Note that Phase 1 is still an intentionally incomplete producer/recorder state**

At this checkpoint, the tree is not a valid commit boundary because producer
sites still call `record_token_outcome(outcome=RowOutcome.X, ...)` while the
recorder contract now expects `(TerminalOutcome | None, TerminalPath)`. The
expected failure mode is type-check/runtime call-site breakage plus stale
schema-dependent tests, not necessarily a plain import failure. This is only
resolved by Phase 2's producer flip plus Phase 3's accumulator/predicate import
smoke. **Do not create a git commit here.** Leave the changes in the worktree
and continue directly to Phase 2. The first legal commit is the atomic Phases
1-3 commit in Phase 3 after its hard import/runtime smoke passes.

**Step 4: Continue without committing**

Do not run `git commit` from the Phase 1 checkpoint. The Phase 1 files are staged/committed only by the atomic Phases 1-3 commit in Phase 3.

**Definition of Done:**
- [ ] Focused Phase 1 test list above passes
- [ ] Broad `tests/unit/contracts/` and `tests/unit/core/landscape/` failures are not hidden or ignored; every known stale fixture class is either fixed in Phase 1 because it is touched here or explicitly covered by Phase 5 triage
- [ ] mypy / ruff / format clean on the affected paths
- [ ] No git commit created at the Phase 1 checkpoint
- [ ] Phase 2 starts immediately; Phase 1 is not marked complete in tracker state except as "local checkpoint reached, continuing to Phase 2"
