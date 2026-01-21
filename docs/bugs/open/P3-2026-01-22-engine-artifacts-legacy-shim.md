# Bug Report: `elspeth.engine.artifacts` is a legacy re-export shim

## Summary

- `src/elspeth/engine/artifacts.py` exists only to re-export `ArtifactDescriptor` from `elspeth.contracts`.
- This is a backwards-compatibility shim, which violates the repository's "No Legacy Code" policy and keeps an old import path alive without need.

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/artifacts.py`, find bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection + repo policy review

## Steps To Reproduce

1. Open `src/elspeth/engine/artifacts.py`.
2. Observe it only re-exports `ArtifactDescriptor` from `elspeth.contracts`.
3. Compare with the "No Legacy Code Policy" in `CLAUDE.md`, which forbids compatibility shims.

## Expected Behavior

- There is a single canonical import path for `ArtifactDescriptor` (`elspeth.contracts`).
- Legacy shim modules are removed and all call sites updated.

## Actual Behavior

- `elspeth.engine.artifacts` persists as a re-export shim with no engine-specific behavior.

## Evidence

- Shim implementation: `src/elspeth/engine/artifacts.py`
- Policy forbidding compatibility shims: `CLAUDE.md`
- Old import path still used in tests/docs: `rg "elspeth.engine.artifacts"`

## Impact

- User-facing impact: none directly.
- Data integrity / security impact: none.
- Performance or cost impact: negligible; main cost is policy violation and maintenance drift risk.

## Root Cause Hypothesis

- The module was left behind after moving `ArtifactDescriptor` into `elspeth.contracts.results` to avoid updating imports.

## Proposed Fix

- Code changes (modules/files):
  - Remove `src/elspeth/engine/artifacts.py`.
  - Update all imports to `from elspeth.contracts import ArtifactDescriptor`.
- Config or schema changes: none.
- Tests to add/update: update existing tests/docs that import `elspeth.engine.artifacts`.
- Risks or migration steps: none; this is an internal codebase path change.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (No Legacy Code Policy)
- Observed divergence: backward-compatibility shim retained in engine namespace.
- Reason (if known): convenience during refactor.
- Alignment plan or decision needed: remove shim and update call sites.

## Acceptance Criteria

- `elspeth.engine.artifacts` module is removed.
- All references updated to `elspeth.contracts.ArtifactDescriptor`.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_orchestrator.py -k ArtifactDescriptor`
- New tests required: no

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
