# Bug Report: Run records empty resolved config

## Summary
- Pipeline runs store `{}` for resolved configuration because `PipelineConfig.config` is never populated in the CLI, so the Landscape run record loses reproducibility data.

## Severity
- Severity: major
- Priority: P1

## Reporter
- Name or handle: Codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment
- Commit/branch: 5c27593 (local)
- OS: Linux (dev env)
- Python version: 3.11+ (per project)
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if applicable)
- Goal or task prompt: N/A
- Model/version: N/A
- Tooling and permissions (sandbox/approvals): N/A
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: N/A

## Steps To Reproduce
1. Run a pipeline via the CLI using a settings YAML file.
2. Inspect the `runs` table (or `LandscapeRecorder.get_run(...)`).
3. Observe `settings_json` or equivalent stored config is `{}`.

## Expected Behavior
- The run record stores the fully resolved configuration used for the run (not just a hash).

## Actual Behavior
- The run record stores an empty object because `PipelineConfig.config` defaults to `{}` and is never set in the CLI.

## Evidence
- Orchestrator uses `config.config` when beginning a run: `src/elspeth/engine/orchestrator.py:163`.
- CLI builds `PipelineConfig` without `config=...`: `src/elspeth/cli.py:306`.

## Impact
- User-facing impact: Reproducibility and auditability are compromised; runs cannot be recreated from stored config.
- Data integrity / security impact: Audit trail lacks required configuration context.
- Performance or cost impact: N/A.

## Root Cause Hypothesis
- CLI and callers never populate `PipelineConfig.config`, so LandscapeRecorder stores `{}`.

## Proposed Fix
- Code changes (modules/files):
  - `src/elspeth/cli.py`: set `PipelineConfig.config` to the resolved config (e.g., `config.model_dump()` or equivalent normalized dict).
  - Optionally update `Orchestrator.run` to use `settings` as fallback when `config.config` is empty.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that runs the orchestrator with config and asserts `runs.settings_json` includes expected keys.
- Risks or migration steps:
  - Ensure sensitive values are redacted before storing if necessary.

## Architectural Deviations
- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:249` and `docs/design/architecture.md:271` (runs store resolved config).
- Observed divergence: run records contain empty config.
- Reason (if known): `PipelineConfig.config` not populated.
- Alignment plan or decision needed: populate run config at pipeline construction time.

## Acceptance Criteria
- Run records include non-empty, resolved configuration matching the settings used to execute.

## Tests
- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -k run`
- New tests required: yes (run config persistence).

## Notes / Links
- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`
