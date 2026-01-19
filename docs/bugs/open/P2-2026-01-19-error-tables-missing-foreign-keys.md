# Bug Report: `validation_errors` / `transform_errors` tables lack key foreign keys (orphan error records possible)

## Summary

- Landscape is Tier 1 “full trust” data; it should be structurally self-consistent and enforce referential integrity wherever feasible.
- The schema defines:
  - `validation_errors.node_id` as a plain string (no FK to `nodes.node_id`)
  - `transform_errors.token_id` as a plain string (no FK to `tokens.token_id`)
- This allows orphan error records (referencing missing nodes/tokens) to exist without DB enforcement, weakening the “no orphan records” integrity posture.

## Severity

- Severity: major
- Priority: P2

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
- Notable tool calls or steps: schema inspection

## Steps To Reproduce

1. Insert a row into `transform_errors` with a non-existent `token_id`.
2. Observe the insert succeeds because there is no FK constraint.
3. Later, queries that assume token existence (explain/export/lineage) cannot resolve the error record cleanly.

## Expected Behavior

- Where possible, the schema enforces referential integrity:
  - `transform_errors.token_id` should reference `tokens.token_id`.
  - `validation_errors.node_id` should reference `nodes.node_id` when non-NULL (nullable FK is OK).

## Actual Behavior

- No FK constraints exist for these columns.

## Evidence

- Schema definitions missing FKs:
  - `src/elspeth/core/landscape/schema.py:279-291` (`validation_errors.node_id` has no `ForeignKey(...)`)
  - `src/elspeth/core/landscape/schema.py:298-310` (`transform_errors.token_id` has no `ForeignKey(...)`)
- Design posture expects referential integrity:
  - `docs/design/architecture.md` (“No Orphan Records: Foreign keys enforced…”)

## Impact

- User-facing impact: harder to trace/resolve error records; increases the need for inference.
- Data integrity / security impact: moderate. Tier 1 audit DB can contain structurally inconsistent records.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Error tables were added without threading through FK constraints, possibly due to evolving identity semantics (see transform_id ambiguity ticket).

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/schema.py`:
    - Add `ForeignKey("tokens.token_id")` to `transform_errors.token_id`.
    - Add `ForeignKey("nodes.node_id")` to `validation_errors.node_id` (keep nullable).
  - Consider adding/checking FKs for other “identity” fields as semantics stabilize.
- Config or schema changes:
  - If introducing migrations, add the constraints via Alembic (or fail-fast schema checks until migrations exist).
- Tests to add/update:
  - Add integrity tests that attempt to insert orphan error rows and assert DB rejects them when using SQLite with `foreign_keys=ON`.
- Risks or migration steps:
  - Existing DBs may already contain orphan records; migration must decide whether to delete/repair them or fail.

## Architectural Deviations

- Spec or doc reference: `docs/design/architecture.md` + `CLAUDE.md` Tier 1 trust rules
- Observed divergence: error tables permit orphan references.
- Reason (if known): constraints omitted during initial implementation.
- Alignment plan or decision needed: coordinate with `transform_errors.transform_id` identity decision (see `docs/bugs/open/2026-01-19-transform-errors-ambiguous-transform-id.md`).

## Acceptance Criteria

- DB rejects `transform_errors` rows referencing missing `tokens`.
- DB rejects non-NULL `validation_errors.node_id` values referencing missing `nodes`.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_schema.py`
- New tests required: yes (FK enforcement coverage)

## Notes / Links

- Related issues/PRs:
  - `docs/bugs/open/2026-01-19-transform-errors-ambiguous-transform-id.md`
  - `docs/bugs/open/2026-01-19-validation-errors-missing-node-id.md`
