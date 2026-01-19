# Bug Report: contracts/config.py imports core.config (contracts no longer a leaf; circular-import risk)

## Summary

- The Contracts subsystem is intended to be the leaf “shared types” module, but `elspeth.contracts.config` directly imports `elspeth.core.config` and re-exports Pydantic settings models from Core.
- Because `elspeth.contracts.__init__` re-exports these settings models, importing `elspeth.contracts` pulls in Core (and Core pulls in Engine submodules like the expression parser).
- `elspeth.contracts.__init__` explicitly notes that import order is “load-bearing” to avoid circular imports, indicating existing fragility.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A

## Steps To Reproduce

1. In a minimal environment, run `import elspeth.contracts`.
2. Observe that it imports `elspeth.core.config` (and its transitive dependencies), even if the caller only wants basic contracts like enums/results.

## Expected Behavior

- Contracts remain a leaf module: importing `elspeth.contracts` should not import Core/Engine modules.
- Configuration models (if considered contracts) should live in Contracts and be consumed by Core, not the other way around.

## Actual Behavior

- `elspeth.contracts` depends on `elspeth.core.config`, making Contracts non-leaf and increasing the risk of circular imports and heavy import-time side effects.

## Evidence

- Direct dependency from Contracts into Core:
  - `src/elspeth/contracts/config.py:10-23` (`from elspeth.core.config import ...`)
- Contracts package imports config re-exports:
  - `src/elspeth/contracts/__init__.py:75-90`
- `contracts/__init__.py` acknowledges load-bearing import order to avoid circular imports:
  - `src/elspeth/contracts/__init__.py:12-13`

## Impact

- User-facing impact: increased import time and potential circular import failures in edge cases.
- Data integrity / security impact: low.
- Performance or cost impact: higher startup overhead for any consumer importing contracts.

## Root Cause Hypothesis

- To keep Pydantic validation logic in Core, config models were defined there and re-exported through Contracts for import consistency, creating a dependency inversion.

## Proposed Fix

- Code changes (modules/files):
  - Option A (preferred): move the Pydantic settings models into Contracts (e.g., `src/elspeth/contracts/config_models.py`) and have `src/elspeth/core/config.py` import and provide the load/resolve functions and validators around those models.
  - Option B (minimal): stop re-exporting config models from `elspeth.contracts` and require config imports from `elspeth.core.config` directly, restoring Contracts as a leaf.
- Tests to add/update:
  - Add a smoke test asserting `import elspeth.contracts` does not import `elspeth.core.config` (if Option A/B pursued).

## Architectural Deviations

- Spec or doc reference: subsystem boundary described in `docs/arch-analysis-2026-01-20-0105/02-subsystem-catalog.md` (“Outbound: None” for Contracts)
- Observed divergence: Contracts imports Core.
- Alignment plan or decision needed: decide whether config models belong in Contracts or Core.

## Acceptance Criteria

- Importing `elspeth.contracts` does not import `elspeth.core.config`.
- Config model import path is stable and documented (either contracts-first or core-first), without load-bearing import order hacks.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/ -k import`
- New tests required: yes
