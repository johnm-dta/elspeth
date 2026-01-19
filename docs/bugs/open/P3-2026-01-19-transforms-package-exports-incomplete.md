# Bug Report: `elspeth.plugins.transforms` does not export all built-in transforms (`BatchStats`, `JSONExplode` missing)

## Summary

- The transforms package describes itself as “Built-in transform plugins”, but it only re-exports `FieldMapper` and `PassThrough`.
- Other built-in transforms exist and are used elsewhere (notably `BatchStats` and `JSONExplode`), but `from elspeth.plugins.transforms import BatchStats` fails.
- This makes the public plugin API inconsistent and increases the chance of drift between “supported plugins” and what the package actually exposes.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. In Python, run: `from elspeth.plugins.transforms import BatchStats` (or `JSONExplode`).
2. Observe import fails because the names are not exported by `elspeth.plugins.transforms`.

## Expected Behavior

- `elspeth.plugins.transforms` should export the full set of built-in transforms (or clearly document that only a subset are part of the public API).

## Actual Behavior

- `elspeth.plugins.transforms` exports only `FieldMapper` and `PassThrough`, omitting other built-ins.

## Evidence

- Transforms package exports only two built-ins: `src/elspeth/plugins/transforms/__init__.py:1-10`
- Built-in transforms list includes `BatchStats` and `JSONExplode`: `src/elspeth/plugins/transforms/hookimpl.py:1-20`
- CLI imports `BatchStats` and `JSONExplode` directly (indicating they are “built-in” in practice): `src/elspeth/cli.py:225-233`

## Impact

- User-facing impact: confusing imports and unclear public API; users can’t rely on `elspeth.plugins.transforms` to provide built-ins.
- Data integrity / security impact: low direct risk, but mismatched plugin catalogs increase chance of configuration misunderstandings.
- Performance or cost impact: engineering overhead and avoidable confusion.

## Root Cause Hypothesis

- The transforms package `__init__.py` was not updated when new built-in transforms were added.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/plugins/transforms/__init__.py` to export all built-in transforms (at least those used by the CLI run path).
  - Consider adding a small test to keep exports in sync with the built-in hook list.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test asserting `elspeth.plugins.transforms.__all__` contains the built-in transforms list.
- Risks or migration steps:
  - Adding exports is backwards-compatible; removing exports would be breaking.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: built-in transforms exist but are not exposed by the transforms package API.
- Reason (if known): incremental addition of plugins without updating exports.
- Alignment plan or decision needed: define which imports are part of the public/stable API.

## Acceptance Criteria

- `from elspeth.plugins.transforms import BatchStats, JSONExplode` succeeds (or the project documents that these are intentionally not part of the public API).

## Tests

- Suggested tests to run: `pytest tests/`
- New tests required: yes

## Notes / Links

- Related ticket: `docs/bugs/open/2026-01-20-cli-plugins-hardcoded-out-of-sync.md` (CLI drift from PluginManager)
