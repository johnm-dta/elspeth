# Bug Report: Error tables lack foreign keys for node/token references

## Summary

- `validation_errors.node_id` and `transform_errors.token_id`/`transform_errors.transform_id` are defined without FK constraints, allowing orphan error records and violating the "No Orphan Records" audit invariant.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `schema.py`.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed schema definitions plus error-recording call sites and architecture invariants.

## Steps To Reproduce

1. Insert a row into `transform_errors` with a non-existent `token_id` or `transform_id`.
2. Insert a row into `validation_errors` with a non-existent `node_id`.
3. Observe inserts succeed because no FK constraints exist for these columns.

## Expected Behavior

- DB rejects orphan references: `transform_errors.token_id` should reference `tokens.token_id`, `transform_errors.transform_id` should reference `nodes.node_id`, and `validation_errors.node_id` should reference `nodes.node_id` when non-NULL.

## Actual Behavior

- Orphan error records are permitted; no FK constraints exist for these columns.

## Evidence

- `src/elspeth/core/landscape/schema.py:303` defines `validation_errors.node_id` without `ForeignKey("nodes.node_id")`.
- `src/elspeth/core/landscape/schema.py:322` defines `transform_errors.token_id` and `transform_errors.transform_id` without foreign keys.
- `src/elspeth/engine/executors.py:258` records `transform_id=transform.node_id`, so `transform_errors.transform_id` should be constrained to `nodes.node_id`.
- `docs/design/architecture.md:274` states the "No Orphan Records" FK enforcement invariant.

## Impact

- User-facing impact: lineage/explain queries can surface error records that cannot resolve to a token/node.
- Data integrity / security impact: Tier 1 audit DB can hold structurally inconsistent records, weakening audit defensibility.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- Error tables were added without enforcing referential integrity, likely deferred due to evolving identity semantics.

## Proposed Fix

- Code changes (modules/files): Add `ForeignKey("nodes.node_id")` to `validation_errors.node_id`, add `ForeignKey("tokens.token_id")` to `transform_errors.token_id`, and add `ForeignKey("nodes.node_id")` to `transform_errors.transform_id` in `src/elspeth/core/landscape/schema.py`.
- Config or schema changes: Add an Alembic migration (or SQLite rebuild guidance) to apply new constraints.
- Tests to add/update: Extend `tests/core/landscape/test_schema.py` to assert FK enforcement rejects orphan inserts with `PRAGMA foreign_keys=ON`.
- Risks or migration steps: Existing databases may contain orphan rows; migration should repair or fail loudly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:274`
- Observed divergence: Error tables allow orphan references due to missing FKs.
- Reason (if known): Unknown
- Alignment plan or decision needed: Add FKs and define migration handling for existing data.

## Acceptance Criteria

- Inserts into `validation_errors` with non-existent `node_id` (when non-NULL) fail.
- Inserts into `transform_errors` with non-existent `token_id` or `transform_id` fail.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_schema.py`
- New tests required: yes, FK enforcement coverage for error tables.

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P2-2026-01-19-error-tables-missing-foreign-keys.md`
- Related design docs: `docs/design/architecture.md`
