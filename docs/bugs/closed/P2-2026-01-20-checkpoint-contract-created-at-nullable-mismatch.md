# Bug Report: Checkpoint contract allows created_at=None despite DB NOT NULL

## Summary

- `contracts.audit.Checkpoint.created_at` is typed as `datetime | None`, but the Landscape schema defines `checkpoints.created_at` as `nullable=False`.
- This is a Tier-1 (audit DB) contract: allowing `None` undermines “crash on any anomaly” and can mask audit corruption or incomplete writes.

## Severity

- Severity: minor
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
- Data set or fixture: N/A

## Steps To Reproduce

N/A (static contract/schema mismatch).

## Expected Behavior

- Contracts for Tier-1 audit tables match schema nullability. If the DB contains NULL where forbidden, readers crash immediately.

## Actual Behavior

- Contract allows `created_at=None` for `Checkpoint`, conflicting with `nullable=False` in the schema.

## Evidence

- Contract type allows NULL:
  - `src/elspeth/contracts/audit.py:295-308` (`Checkpoint.created_at: datetime | None`)
- DB schema forbids NULL:
  - `src/elspeth/core/landscape/schema.py:318-328` (`Column("created_at", ..., nullable=False)`)

## Impact

- User-facing impact: low (mostly type-level), but it weakens strictness around audit DB invariants.
- Data integrity / security impact: medium: Type contracts should not suggest NULL is acceptable in Tier-1.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Contract was authored with optionality for convenience or legacy reasons, but schema has since been made strict (or vice versa).

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/audit.py`: Change `Checkpoint.created_at` to `datetime` (non-optional).
  - If any reader path currently encounters NULL, treat as corruption and raise immediately (Tier-1 policy).
- Tests to add/update:
  - Add a small contract/schema consistency test (if such a suite exists), or a targeted unit test ensuring `LandscapeRecorder.get_checkpoints()` rejects NULL created_at.

## Architectural Deviations

- Spec or doc reference: CLAUDE.md “Tier 1: Crash on any anomaly”
- Observed divergence: contract suggests NULL is acceptable
- Alignment plan or decision needed: none; this should be strict

## Acceptance Criteria

- `Checkpoint.created_at` is non-optional.
- Any attempt to construct or return a Checkpoint with NULL created_at results in a hard failure.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/landscape -k checkpoint`
- New tests required: yes

## Resolution

**Status:** CLOSED (2026-01-21)
**Resolved by:** Claude Opus 4.5

### Changes Made

**Code fix (`src/elspeth/contracts/audit.py`):**

Changed `Checkpoint.created_at` from optional to required:

```python
# Before (Bug):
created_at: datetime | None

# After (Fix):
created_at: datetime  # Required - schema enforces NOT NULL (Tier 1 audit data)
```

### Verification

```bash
.venv/bin/python -m pytest tests/ -k checkpoint -v
# 88 passed, 2 skipped

.venv/bin/python -m mypy src/elspeth/contracts/audit.py src/elspeth/core/checkpoint/manager.py
# No errors
```

### Notes

The database schema already enforces `nullable=False` on `checkpoints.created_at`, so NULL values cannot be stored. The contract type was incorrectly allowing `None`, which was misleading and violated the Tier 1 principle of having contracts match schema strictness. With this fix, the contract accurately reflects that `created_at` is always present.
