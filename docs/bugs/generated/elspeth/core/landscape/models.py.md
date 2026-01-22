# Bug Report: Landscape models drift from contracts/schema

## Summary

- `src/elspeth/core/landscape/models.py` is a legacy duplicate of audit contracts and has drifted (missing Node schema fields, enum-typed fields downgraded to `str`, optional `Checkpoint.created_at`, truncated RowLineage), so code/tests importing it can accept invalid or incomplete audit records and mask contract drift.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for `src/elspeth/core/landscape/models.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/core/landscape/models.py`, `src/elspeth/contracts/audit.py`, `src/elspeth/core/landscape/schema.py`, and model-related tests

## Steps To Reproduce

1. Inspect `src/elspeth/core/landscape/models.py` for Node, Call, RoutingEvent, Checkpoint, and RowLineage fields.
2. Compare those fields to `src/elspeth/contracts/audit.py` and `src/elspeth/core/landscape/schema.py`.
3. Observe tests importing legacy models (e.g., `tests/core/landscape/test_models.py`) and constructing objects with shapes that violate the schema (e.g., `Checkpoint.created_at=None`).

## Expected Behavior

- `src/elspeth/core/landscape/models.py` matches the authoritative audit contracts/schema (or is removed in favor of contracts) so enum types and required fields are enforced consistently.

## Actual Behavior

- `src/elspeth/core/landscape/models.py` omits schema fields, relaxes enum types to `str`, allows nullable `created_at`, and defines a reduced RowLineage shape, diverging from contracts/schema while tests still import it.

## Evidence

- Logs or stack traces: Unknown (static analysis only; no runtime logs).
- Artifacts (paths, IDs, screenshots): Contract/schema drift in code: Node fields end without schema_mode/fields in `src/elspeth/core/landscape/models.py:63` while contracts/schema include them at `src/elspeth/contracts/audit.py:69` and `src/elspeth/core/landscape/schema.py:64`; Call/RoutingEvent use `str` at `src/elspeth/core/landscape/models.py:219` and `src/elspeth/core/landscape/models.py:255` vs enums at `src/elspeth/contracts/audit.py:211` and `src/elspeth/contracts/audit.py:250`; Checkpoint `created_at` optional at `src/elspeth/core/landscape/models.py:308` vs NOT NULL at `src/elspeth/core/landscape/schema.py:351`; RowLineage `source_hash` at `src/elspeth/core/landscape/models.py:327` vs `source_data_hash` at `src/elspeth/contracts/audit.py:325`.
- Minimal repro input (attach or link): Tests construct legacy models and accept invalid shapes, e.g., `tests/core/landscape/test_schema.py:248` (Checkpoint created_at None) and `tests/core/landscape/test_models.py:12` (imports legacy models).

## Impact

- User-facing impact: Low in runtime (contracts used), but external callers importing `core.landscape.models` can rely on the wrong contract.
- Data integrity / security impact: If models.py is used to build audit records, missing schema fields and relaxed enum/required fields can permit invalid audit entries and hide contract drift.
- Performance or cost impact: None observed.

## Root Cause Hypothesis

- Legacy Landscape models were duplicated before contracts migration and have not been kept in sync.

## Proposed Fix

- Code changes (modules/files): Remove `src/elspeth/core/landscape/models.py` and update imports to use contracts, or replace it with a thin re-export of `elspeth.contracts.audit`; if retained, align fields/types with contracts (add schema_mode/schema_fields, use enums, make Checkpoint.created_at required, and match RowLineage fields).
- Config or schema changes: None.
- Tests to add/update: Update model tests to import from `elspeth.contracts` or `elspeth.core.landscape` exports instead of legacy models.
- Risks or migration steps: Potential breaking change for any external imports of `core.landscape.models`; update call sites in the same change.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:417` (No Legacy Code Policy).
- Observed divergence: Duplicate, drifting audit model definitions exist in `src/elspeth/core/landscape/models.py`.
- Reason (if known): Legacy module retained after contracts migration.
- Alignment plan or decision needed: Decide whether to delete the legacy module or make it a pure re-export of contracts.

## Acceptance Criteria

- There is a single authoritative audit model definition and `core/landscape/models.py` no longer diverges from contracts/schema.

## Tests

- Suggested tests to run: `./.venv/bin/python -m pytest tests/core/landscape/test_models.py` and `./.venv/bin/python -m pytest tests/core/landscape/test_schema.py`
- New tests required: No (update existing tests to validate the contracts surface).

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P3-2026-01-19-landscape-models-duplication-drift.md`
- Related design docs: Unknown
