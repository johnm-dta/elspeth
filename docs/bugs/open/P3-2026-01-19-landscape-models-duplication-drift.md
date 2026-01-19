# Bug Report: `core/landscape/models.py` duplicates audit contracts but diverges from runtime contracts/schema (test drift + confusion)

## Summary

- The repo defines audit dataclasses twice:
  - `src/elspeth/contracts/audit.py` (strict, used by runtime code and public exports)
  - `src/elspeth/core/landscape/models.py` (additional/legacy models)
- `core/landscape/models.py` has drift vs contracts/schema (examples):
  - `Node` is missing `schema_mode` / `schema_fields` even though schema and runtime `Node` include them.
  - `RoutingEvent.mode` is `str` instead of `RoutingMode`.
  - `Checkpoint.created_at` is `datetime | None` despite schema requiring `created_at NOT NULL`.
- Tests import `elspeth.core.landscape.models` directly, so they can pass even when runtime contracts differ.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code + test inspection

## Steps To Reproduce

1. Compare audit contracts:
   - `src/elspeth/contracts/audit.py` vs `src/elspeth/core/landscape/models.py`.
2. Observe mismatched fields/types.
3. Note that tests import the legacy models module (e.g., `tests/core/landscape/test_models_enums.py`), so they can validate the wrong contract surface.

## Expected Behavior

- There is a single source of truth for audit record contracts (the contracts subsystem).
- Tests validate the same contracts used in runtime APIs (`elspeth.contracts.audit` / `elspeth.core.landscape` exports).

## Actual Behavior

- Duplicate model definitions exist and have drift, which can hide bugs and confuse contributors.

## Evidence

- Missing schema audit fields in legacy Node:
  - `src/elspeth/core/landscape/models.py:49-64` (no `schema_mode` / `schema_fields`)
  - `src/elspeth/contracts/audit.py:49-82` (includes them)
- Type mismatch example:
  - `src/elspeth/core/landscape/models.py:248-260` (`RoutingEvent.mode: str`)
  - `src/elspeth/contracts/audit.py` (`RoutingEvent.mode: RoutingMode`)
- Schema mismatch example:
  - `src/elspeth/core/landscape/models.py:297-312` (`Checkpoint.created_at: datetime | None`)
  - `src/elspeth/core/landscape/schema.py` `checkpoints.created_at` is `nullable=False`
- Tests importing legacy models:
  - `tests/core/landscape/test_models_enums.py`
  - `tests/core/landscape/test_models.py`

## Impact

- User-facing impact: low (runtime uses contracts), but developer-facing impact is significant.
- Data integrity / security impact: indirect. Drift makes it easier to introduce audit contract mismatches unnoticed.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Models were initially implemented within Landscape and later moved to Contracts, but the legacy module remained and was kept alive by tests.

## Proposed Fix

- Code changes (modules/files):
  - Prefer a single contract source:
    - Remove `src/elspeth/core/landscape/models.py` and update tests/imports to use `elspeth.contracts.audit` or `elspeth.core.landscape` exports, OR
    - Make `src/elspeth/core/landscape/models.py` a thin re-export of contracts (no duplicate dataclasses).
- Config or schema changes: none.
- Tests to add/update:
  - Update `tests/core/landscape/test_models*.py` to import from `elspeth.contracts.audit` (or `elspeth.core.landscape`) instead of the legacy module.
- Risks or migration steps:
  - If external users import `elspeth.core.landscape.models`, this is a breaking change; repo guidance discourages compatibility shims, so call sites should be updated.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (prohibition on legacy compatibility shims / duplicated interfaces)
- Observed divergence: duplicate “contract” definitions exist and drift.
- Reason (if known): module not removed after contract migration.
- Alignment plan or decision needed: decide whether `core/landscape/models.py` should exist at all.

## Acceptance Criteria

- There is exactly one authoritative audit model definition set.
- Tests validate the same dataclasses used by production code.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_models*.py`
- New tests required: no (but update existing)

## Notes / Links

- Related issues/PRs: N/A
