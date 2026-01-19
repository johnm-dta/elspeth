# Bug Report: Audit export fails when an existing Landscape DB is missing `tokens.expand_group_id` (schema migration gap)

## Summary

- `elspeth run` with `landscape.export.enabled: true` can fail at export time with `sqlite3.OperationalError: no such column: tokens.expand_group_id` when the configured Landscape SQLite database was created before `expand_group_id` was added to the `tokens` table schema.
- This breaks the `examples/audit_export` pipeline when it points at an existing `examples/audit_export/runs/audit.db` that is behind the current schema (e.g., created by an older build/version and kept on disk; `*.db` is gitignored).
- Root problem: the project currently relies on `metadata.create_all(...)` (no migrations). `create_all` does not evolve existing tables, so persisted audit DBs become incompatible as schema evolves.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (run fails during export; run id not printed by CLI)

## Environment

- Commit/branch: `83e1b90f94d103e41041c5b37138856b87ea1d59` (main)
- OS: Ubuntu 24.04.3 LTS (kernel 6.8.0-90-generic)
- Python version: 3.13.1 (repo targets >=3.11)
- Config profile / env vars: N/A (`ELSPETH_SIGNING_KEY` not set; export signing disabled in repro)
- Data set or fixture: `examples/audit_export/*`

## Agent Context (if relevant)

- Goal or task prompt: deep-dive repo to identify a bug and write a report
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps:
  - Ran `pytest` (all passing)
  - Reproduced via `elspeth run` on `examples/audit_export/settings.yaml`
  - Confirmed DB schema via `sqlite3 ... ".schema tokens"`

## Steps To Reproduce

1. From repo root, run:
   - `.venv/bin/elspeth run -s examples/audit_export/settings.yaml --execute`
2. Observe failure during post-run export with a SQLite error.
3. Confirm the configured DB is missing the column:
   - `sqlite3 examples/audit_export/runs/audit.db ".schema tokens"`

## Expected Behavior

- The `examples/audit_export` pipeline should run successfully and write:
  - `examples/audit_export/output/non_corporate.csv`
  - `examples/audit_export/output/corporate.csv`
  - `examples/audit_export/output/audit_trail.json`
- If a user points at an older Landscape DB, the system should either:
  - migrate the schema forward, or
  - fail early with a clear, actionable error explaining the schema mismatch and remediation.

## Actual Behavior

- The run fails during export with:
  - `sqlite3.OperationalError: no such column: tokens.expand_group_id`

## Evidence

- CLI output:
  - `Error during pipeline execution: (sqlite3.OperationalError) no such column: tokens.expand_group_id`
  - `SELECT tokens.token_id, ... tokens.expand_group_id, ... FROM tokens WHERE tokens.row_id = ? ...`
- Schema of the existing example DB:
  - `sqlite3 examples/audit_export/runs/audit.db ".schema tokens"` shows a `tokens` table without `expand_group_id`.
- Code schema expects the column:
  - `src/elspeth/core/landscape/schema.py` defines `tokens.expand_group_id` (For deaggregation).
- Query path selects all token columns:
  - `src/elspeth/core/landscape/recorder.py` `get_tokens()` does `select(tokens_table)` which includes `expand_group_id`.
- Workaround validation:
  - Re-running the same example with a fresh DB path (e.g., `sqlite:////tmp/elspeth-audit-export-test.db`) completes and produces `audit_trail.json`.

## Impact

- User-facing impact:
  - `examples/audit_export` is broken by default (first-run experience failure).
  - Any persisted Landscape DB created before a schema change can become unusable for export/explain-like features that read tokens.
- Data integrity / security impact:
  - Integrity is preserved by failing fast, but audit accessibility is compromised (older records cannot be exported/explained after upgrades).
- Performance or cost impact:
  - N/A (fails immediately).

## Root Cause Hypothesis

- `LandscapeDB` uses `metadata.create_all(...)` (table creation only) and does not run schema migrations.
- When a SQLite file already exists with an older `tokens` schema, `create_all` does not add the new `expand_group_id` column.
- Export triggers token reads via `LandscapeRecorder.get_tokens()` which selects `tokens_table` (including `expand_group_id`), causing SQLite to raise `no such column`.

## Proposed Fix

- Code changes (modules/files):
  - Short-term (improve UX + correctness):
    - Add a schema compatibility check during DB initialization for SQLite (and ideally all backends):
      - if required columns are missing, raise a dedicated error with remediation steps (e.g., “delete DB (dev) / run migration (prod)”).
  - Medium-term (proper upgrade path):
    - Implement Alembic migrations for Landscape schema changes and run upgrades on startup (or provide an explicit `elspeth landscape migrate` command).
- Config or schema changes:
  - Consider an explicit `landscape.auto_migrate` flag (default off for “legal record” safety), plus a documented manual migration command.
- Example hygiene:
  - Ensure example comments/docs instruct deleting `examples/**/runs/audit.db` before running when switching versions (until migrations exist), or add a helper script/command to reset example run state safely.
- Tests to add/update:
  - Add an integration test that:
    1. creates an “old schema” SQLite DB (tokens table without `expand_group_id`)
    2. runs export or `LandscapeRecorder.get_tokens`
    3. asserts the system fails with a clear “schema mismatch” error (or auto-migrates if that’s the chosen behavior).
- Risks or migration steps:
  - If introducing migrations: schema evolution must be explicit and audited; do not silently drop/alter audit data.

## Architectural Deviations

- Spec or doc reference: `docs/design/architecture.md` (“Audit Trail Export” expects orchestrator export to work post-run)
- Observed divergence: export can fail due to schema mismatch in an existing Landscape DB without an actionable error or migration path.
- Reason (if known): early implementation uses `create_all` rather than a migration system; committed example DB drifted behind schema.
- Alignment plan or decision needed:
  - Decide on a migration posture for “Tier 1” audit data:
    - strict: refuse to run until migrated (but provide a supported migration mechanism)
    - permissive: auto-migrate in-place (requires strong guarantees and careful logging)

## Acceptance Criteria

- Running `examples/audit_export/settings.yaml` succeeds on a clean checkout.
- When configured DB schema is behind code schema:
  - system detects mismatch at startup (before processing/export), and
  - surfaces a clear remediation message (or successfully migrates, per decision).

## Tests

- Suggested tests to run:
  - `pytest`
  - `./.venv/bin/elspeth run -s examples/audit_export/settings.yaml --execute`
- New tests required: yes (schema mismatch / migration behavior).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`
