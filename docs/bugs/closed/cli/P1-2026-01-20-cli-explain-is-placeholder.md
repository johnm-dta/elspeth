# Bug Report: `elspeth explain` is effectively a placeholder (no DB query; JSON/text modes always error)

## Summary

- `elspeth explain` does not currently query Landscape to explain lineage:
  - `--json` mode always emits a hard-coded error and exits non-zero.
  - `--no-tui` (text mode) always emits a hard-coded error and exits non-zero.
  - TUI mode launches `ExplainApp`, but `ExplainApp` is a placeholder widget that does not load lineage data.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 1 (CLI), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/cli.py` and `src/elspeth/tui/explain_app.py`

## Steps To Reproduce

1. Run `elspeth explain -r latest --json`.
2. Observe a hard-coded “No runs found…” JSON error and exit code 1, regardless of whether runs exist.
3. Run `elspeth explain -r latest --no-tui`.
4. Observe a hard-coded “Run not found” text error and exit code 1, regardless of whether runs exist.
5. Run `elspeth explain -r latest` (TUI mode).
6. Observe the TUI shows placeholders and does not load lineage.

## Expected Behavior

- `explain` should query the Landscape audit DB and provide lineage for:
  - a run (`--run`)
  - optionally narrowed by `--row` or `--token`
- JSON/text output modes should work for scripting (CI, automation, debugging).
- TUI should show real lineage data.

## Actual Behavior

- JSON/text modes always error, and TUI is a placeholder.

## Evidence

- Hard-coded error branches for `--json` and `--no-tui`:
  - `src/elspeth/cli.py:186-200`
- TUI app is placeholder widgets (no DB/data layer):
  - `src/elspeth/tui/explain_app.py:16-73`

## Impact

- User-facing impact: `explain` is advertised by `--help` but cannot be used to inspect actual runs.
- Data integrity / security impact: none directly, but reduces auditability in practice.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Explain subsystem was stubbed pending later phase work; CLI options were created but not wired to Landscape queries.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`:
    - Add `--database` and/or `--settings` to locate the Landscape DB used by runs.
    - Implement JSON/text paths by opening `LandscapeDB` and querying via `LandscapeRecorder` read APIs (or a dedicated read service).
  - `src/elspeth/tui/explain_app.py`:
    - Replace placeholder widgets with data-backed views and periodic refresh.
- Config or schema changes: none.
- Tests to add/update:
  - Add a CLI test that creates a minimal Landscape DB with one run and asserts:
    - `elspeth explain --json --run <id> --database <db>` returns JSON with run metadata
    - `elspeth explain --no-tui ...` prints expected info
- Risks or migration steps:
  - Ensure explain queries respect the “Tier 1 audit DB = crash on anomaly” rules.

## Architectural Deviations

- Spec or doc reference: auditability standard in `CLAUDE.md` (“explain queries are simple and complete”)
- Observed divergence: `explain` is not available in non-TUI forms and TUI is placeholder.
- Reason (if known): feature work deferred.
- Alignment plan or decision needed: confirm expected CLI explain output contracts (JSON schema/text format).

## Acceptance Criteria

- `elspeth explain --json` returns real data for an existing run ID.
- `elspeth explain --no-tui` returns real data for an existing run ID.
- TUI mode shows non-placeholder lineage content for an existing run.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py`
- New tests required: yes (explain output)

## Notes / Links

- Related issues/PRs: N/A

## Verification (2026-01-25)

### Status: STILL VALID

The bug remains valid. While error messaging has been improved since the original report, the core functionality is still not implemented.

### Current Code Analysis

**src/elspeth/cli.py (lines 260-335):**
- The `explain` command implementation now explicitly acknowledges it is not implemented
- JSON mode returns `{"status": "not_implemented", ...}` with exit code 2 (changed from exit code 1)
- Text mode (`--no-tui`) prints clear "not yet implemented" message with exit code 2
- TUI mode launches `ExplainApp` with a note that it's a "preview" for "Phase 4"
- Code includes a comment explicitly referencing this bug report: `# See: docs/bugs/open/P1-2026-01-20-cli-explain-is-placeholder.md`

**src/elspeth/tui/explain_app.py (lines 60-74):**
- Still shows placeholder Static widgets: "Lineage Tree (placeholder)" and "Detail Panel (placeholder)"
- No database connection or data loading logic implemented
- Action handlers (refresh, help) only show notifications, no actual functionality

### Git History Evidence

**Commit timeline:**
- Bug filed: 2026-01-20 against commit `8cfebea` (main branch)
- RC1 release: 2026-01-22 (commit `c786410`) - improved error messages but functionality still unimplemented
- Current HEAD: 2026-01-25 - no changes to explain command implementation

**Changes since bug report:**
1. Error messages improved to be more honest ("not yet implemented" vs "No runs found")
2. Exit code changed from 1 (error) to 2 (not implemented) for clarity
3. Docstring added noting the command is not yet implemented
4. Comment added explicitly referencing this bug report

**Files unchanged since RC1:**
- `src/elspeth/tui/explain_app.py` - last modified in RC1, still contains placeholders
- `src/elspeth/cli.py` explain command - messaging improved in RC1, but no functional implementation added

### Backend Support Analysis

**The infrastructure EXISTS to implement this:**
- `src/elspeth/core/landscape/lineage.py` provides `explain()` function (lines 63-83)
- Function signature: `explain(recorder: LandscapeRecorder, run_id, token_id=None, row_id=None, sink=None) -> LineageResult | None`
- Multiple tests use this function successfully (e.g., `tests/core/landscape/test_lineage.py`)
- Pattern: Create `LandscapeRecorder(db)` and pass to `explain()`

**What's missing:**
1. CLI does not instantiate `LandscapeDB` or `LandscapeRecorder`
2. CLI does not call the `explain()` function
3. CLI does not format/output the `LineageResult` in JSON or text format
4. TUI does not load data from the database
5. No `--database` or `--settings` option to locate the Landscape DB

### Recommendation

**Status:** Keep as P1, STILL VALID

**Rationale:**
1. The bug description remains accurate - explain command is a placeholder
2. Error messaging improvements are cosmetic, not functional
3. Backend infrastructure exists and is tested, so implementation is straightforward
4. This directly impacts the auditability promise in CLAUDE.md

**Implementation path is clear:**
1. Add `--database` option to explain command (similar to `purge` and `resume` commands)
2. Instantiate `LandscapeDB` and `LandscapeRecorder`
3. Call `explain()` function with appropriate parameters
4. Format and output the `LineageResult` for JSON/text modes
5. Wire TUI to load and display actual lineage data

**Estimated effort:** Medium (2-3 hours)
- Backend logic exists and is tested
- Similar patterns exist in `purge` and `resume` commands for DB access
- Main work is formatting output and wiring TUI widgets to data
