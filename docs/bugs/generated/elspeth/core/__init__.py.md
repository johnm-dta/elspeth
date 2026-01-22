# Bug Report: `resolve_config` not exported from `elspeth.core`

## Summary

- `resolve_config` is defined in `src/elspeth/core/config.py` but is not imported or exported in `src/elspeth/core/__init__.py`, so `from elspeth.core import resolve_config` raises ImportError despite being a documented public API.

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/core/__init__.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): `shell_command` in read-only sandbox; approvals never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected `src/elspeth/core/__init__.py`, `src/elspeth/core/config.py`, and plan doc via `sed`/`rg`/`nl`

## Steps To Reproduce

1. In the project environment, run `python -c "from elspeth.core import resolve_config"`.
2. Observe ImportError for `resolve_config`.

## Expected Behavior

- `resolve_config` is importable via `from elspeth.core import resolve_config`.

## Actual Behavior

- ImportError: cannot import name `resolve_config` from `elspeth.core`.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/__init__.py:15`, `src/elspeth/core/__init__.py:47` (no `resolve_config` import/export)
- Minimal repro input (attach or link): `src/elspeth/core/config.py:1172` (definition of `resolve_config`); `docs/plans/completed/2026-01-17-chunk1-quick-wins.md:743` (documented expectation to export)

## Impact

- User-facing impact: Documented import path fails, breaking scripts that follow the published API surface.
- Data integrity / security impact: None.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `resolve_config` was added to `core.config` but `core.__init__` was not updated to re-export it.

## Proposed Fix

- Code changes (modules/files): Add `resolve_config` to the import list and `__all__` in `src/elspeth/core/__init__.py`.
- Config or schema changes: None.
- Tests to add/update: Add a small export test (e.g., `tests/core/test_init_exports.py`) or extend an existing core export test to assert `from elspeth.core import resolve_config` succeeds.
- Risks or migration steps: Low risk; additive export only.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/plans/completed/2026-01-17-chunk1-quick-wins.md:743`
- Observed divergence: Doc explicitly calls for exporting `resolve_config` from `core.__init__`, but it is not exported.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Export `resolve_config` in `src/elspeth/core/__init__.py`.

## Acceptance Criteria

- `from elspeth.core import resolve_config` succeeds.
- `resolve_config` appears in `elspeth.core.__all__`.

## Tests

- Suggested tests to run: `python -c "from elspeth.core import resolve_config"`
- New tests required: Yes, add/extend a unit test covering core re-exports.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/plans/completed/2026-01-17-chunk1-quick-wins.md:743`
