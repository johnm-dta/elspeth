# Bug Report: contracts/config.py imports core.config (contracts not leaf)

## Summary

- Importing `elspeth.contracts` pulls in `elspeth.core.config` because `src/elspeth/contracts/config.py` re-exports Core settings models, breaking the contracts package’s intended independence and increasing circular-import fragility.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `/home/john/elspeth-rapid/src/elspeth/contracts/config.py`.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): sandbox_mode=read-only, approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/contracts/config.py`, `src/elspeth/contracts/__init__.py`, `src/elspeth/core/config.py`.

## Steps To Reproduce

1. Run `python -c "import elspeth.contracts, sys; print('elspeth.core.config' in sys.modules)"`.
2. Observe `True`, indicating `elspeth.core.config` was imported as a side effect.
3. Inspect `elspeth.contracts.config.CheckpointSettings.__module__` and note it resolves to `elspeth.core.config`.

## Expected Behavior

- Importing `elspeth.contracts` should not import Core modules; contracts should remain a leaf/independent package.

## Actual Behavior

- Importing `elspeth.contracts` imports `elspeth.core.config` due to re-exported settings models.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/contracts/config.py:10`, `src/elspeth/core/config.py:16`, `src/elspeth/contracts/__init__.py:12`
- Minimal repro input (attach or link): `python -c "import elspeth.contracts, sys; print('elspeth.core.config' in sys.modules)"`

## Impact

- User-facing impact: Importing contracts can trigger unintended Core dependencies and circular import failures in edge cases.
- Data integrity / security impact: Low.
- Performance or cost impact: Increased import time and heavier transitive dependency loading for contracts-only consumers.

## Root Cause Hypothesis

- `src/elspeth/contracts/config.py` re-exports settings models from Core instead of defining them in Contracts, inverting dependency direction.

## Proposed Fix

- Code changes (modules/files): Move Pydantic settings models into `src/elspeth/contracts/config.py` (or a new contracts module) and have `src/elspeth/core/config.py` import them; remove Core imports from `src/elspeth/contracts/config.py`.
- Config or schema changes: Unknown
- Tests to add/update: Add a smoke test ensuring `import elspeth.contracts` does not import `elspeth.core.config`.
- Risks or migration steps: Update internal imports to the new contract-owned config models to avoid new cycles.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/arch-analysis-2026-01-21-1932/02-subsystem-catalog.md:28`
- Observed divergence: Contracts are no longer independent due to Core dependency via config re-exports.
- Reason (if known): Pydantic validation logic kept in Core, with contracts acting as a re-export layer.
- Alignment plan or decision needed: Decide ownership of config models (Contracts vs Core) and enforce a stable import boundary.

## Acceptance Criteria

- Importing `elspeth.contracts` does not load `elspeth.core.config`.
- Config model import path is stable and documented without load-bearing import order dependencies.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/contracts/test_config.py -k import`
- New tests required: Yes, add an import-boundary smoke test.

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P2-2026-01-20-contracts-config-reexport-breaks-leaf-boundary.md`
- Related design docs: `docs/arch-analysis-2026-01-21-1932/02-subsystem-catalog.md`
