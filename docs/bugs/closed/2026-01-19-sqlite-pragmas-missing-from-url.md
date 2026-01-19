# Bug Report: SQLite PRAGMAs (foreign_keys/WAL) are not applied for `LandscapeDB.from_url()` / `in_memory()`

## Summary

- `LandscapeDB` enforces critical SQLite reliability settings via a connect hook (`PRAGMA foreign_keys=ON`, `PRAGMA journal_mode=WAL`), but that hook is **only installed in `LandscapeDB.__init__()`**.
- Both factory constructors (`LandscapeDB.from_url()` and `LandscapeDB.in_memory()`) **bypass `__init__()`** and never call `_configure_sqlite()`, so the PRAGMAs are not applied.
- The CLI and public engine examples use `LandscapeDB.from_url(...)`, so the default runtime path can run with **foreign key enforcement disabled**, violating the “no orphan records” audit invariant.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `370fc20862d6bab1bb77ebfe8c49527a12fa2aa8` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1 (repo targets >=3.11)
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: investigate repo bugs and write bug reports
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/core/landscape/database.py`, `src/elspeth/cli.py`, and design invariants

## Steps To Reproduce

1. Start Python and run the following:
   1. `db = LandscapeDB.from_url("sqlite:///:memory:")`
   2. Query `PRAGMA foreign_keys` on `db.engine` (it should report `0` / off).
2. Repeat with `db = LandscapeDB.in_memory()` (same result).
3. Compare against `db = LandscapeDB("sqlite:///:memory:")` which installs the connect hook and should report `1` / on.

(Alternative integrity repro)
1. Create a `LandscapeDB` via `from_url("sqlite:///...")`.
2. Insert a row into a table with a foreign key pointing to a non-existent parent (e.g., insert a `nodes` row with a bogus `run_id`).
3. Observe the insert succeeds when foreign keys are not enforced (audit DB can contain orphans).

## Expected Behavior

- All SQLite-backed `LandscapeDB` constructors enable the audit integrity settings:
  - `PRAGMA foreign_keys=ON` (required for “no orphan records” invariant)
  - `PRAGMA journal_mode=WAL` (expected by `LandscapeDB._configure_sqlite()` and existing tests)

## Actual Behavior

- `LandscapeDB.from_url()` and `LandscapeDB.in_memory()` do not configure SQLite connections, so:
  - foreign key constraints may not be enforced
  - WAL mode is not set

## Evidence

- SQLite connect hook exists, but is only installed by `LandscapeDB.__init__()`:
  - `src/elspeth/core/landscape/database.py:39-61` (`_setup_engine()` + `_configure_sqlite()` with PRAGMAs)
- Factory constructors bypass initialization and never call `_configure_sqlite()`:
  - `src/elspeth/core/landscape/database.py:92-128` (`in_memory()`)
  - `src/elspeth/core/landscape/database.py:130-158` (`from_url()`)
- CLI uses `LandscapeDB.from_url(...)` (high-impact call site):
  - `src/elspeth/cli.py:271-274`
- Design invariant explicitly requires foreign key enforcement:
  - `docs/design/architecture.md:274` (“No Orphan Records: Foreign keys enforced (`PRAGMA foreign_keys=ON` in SQLite)”)
- Existing tests validate WAL mode only for the `LandscapeDB(...)` constructor, not the factories:
  - `tests/core/landscape/test_database.py:20-55` (WAL test doesn’t cover `from_url()`/`in_memory()`)

## Impact

- User-facing impact: subtle and delayed failures; corrupted audit DB may only surface during `explain()`, export, or recovery.
- Data integrity / security impact: **high**. Orphaned `nodes`, `edges`, `tokens`, `node_states`, etc. can exist without a parent `run`/`row`, violating audit trail guarantees and undermining forensic trust.
- Performance or cost impact: WAL disabled can reduce concurrency and increase lock contention on SQLite.

## Root Cause Hypothesis

- `LandscapeDB.from_url()` and `LandscapeDB.in_memory()` construct instances via `__new__` and set `_engine` directly, skipping `_setup_engine()` and therefore skipping the SQLite connect event hook that sets PRAGMAs.

## Proposed Fix

- Code changes (modules/files):
  - Refactor `LandscapeDB` so **all constructors share a single engine-creation path** that:
    1. creates the engine
    2. installs SQLite PRAGMA hooks when `url.startswith("sqlite")`
    3. creates tables (when requested)
  - Concretely:
    - Add a private helper like `_create_engine(url) -> Engine` that applies `_configure_sqlite()` when needed, and call it from `__init__`, `in_memory`, and `from_url`.
    - Or: in `from_url`/`in_memory`, instantiate via `cls(url)` (and optionally skip `create_all`), rather than bypassing `__init__`.
- Config or schema changes: none.
- Tests to add/update:
  - Add assertions to `tests/core/landscape/test_database.py` that `PRAGMA foreign_keys == 1` for:
    - `LandscapeDB(...)`
    - `LandscapeDB.from_url(...)` (sqlite)
    - `LandscapeDB.in_memory()`
  - Optionally assert `PRAGMA journal_mode == wal` for `from_url(sqlite:///file.db)` as well.
- Risks or migration steps:
  - Ensure WAL setting is safe for all intended SQLite deployments (it should be, given it is already mandated in `_configure_sqlite()`).

## Architectural Deviations

- Spec or doc reference: `docs/design/architecture.md` (“No Orphan Records” invariant)
- Observed divergence: the default entrypoint DB constructor (`from_url`) does not enable foreign key enforcement.
- Reason (if known): factory methods bypassed `__init__` for convenience without reapplying sqlite configuration.
- Alignment plan or decision needed: none; this is a correctness/integrity fix.

## Acceptance Criteria

- For SQLite URLs, `LandscapeDB.from_url()` and `LandscapeDB.in_memory()`:
  - enable `PRAGMA foreign_keys=ON` for all connections
  - set WAL mode where applicable
- New tests fail before fix and pass after fix.

## Tests

- Suggested tests to run:
  - `pytest tests/core/landscape/test_database.py`
- New tests required: yes (foreign key + WAL assertions for factory constructors)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Resolution

**Fixed by:** Making `_configure_sqlite()` a static method and calling it from all construction paths.

**Changes:**
- `src/elspeth/core/landscape/database.py`: Converted `_configure_sqlite()` to static method, added calls in `from_url()` and `in_memory()`
- `tests/core/landscape/test_database.py`: Added 5 new PRAGMA tests covering all construction paths
- `tests/core/landscape/test_exporter.py`: Fixed FK violation in `test_exporter_final_hash_deterministic_with_multiple_records`
- `tests/cli/test_cli.py`: Fixed FK violation in `test_resume_shows_resume_point_info`
- `tests/engine/test_processor.py`: Fixed FK violation in `test_processor_buffers_restored_on_recovery`
- `tests/engine/test_integration.py`: Fixed FK violation in `test_explain_for_coalesced_row`

**Root cause confirmed:** Factory methods used `cls.__new__(cls)` to bypass `__init__()` without calling `_configure_sqlite()`.

**Bonus finding:** Enabling FK enforcement revealed four tests that were inserting orphan records - these have been fixed to properly set up parent records. This validates that the fix is working correctly.

**Closed:** 2026-01-20
