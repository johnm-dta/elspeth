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
