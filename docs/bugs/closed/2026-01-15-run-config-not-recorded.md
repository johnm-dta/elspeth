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
  - Treat the resolved config as a first-class audit artifact, not an optional field:
    - Add a single resolver in `src/elspeth/core/config.py` (or a new `core/config_resolve.py`) that returns a canonical, fully resolved run config dict plus its hash. This is the only supported entry point for building `runs.settings_json`.
    - The resolver must apply the secret-handling policy (HMAC fingerprints, no raw secrets) before serialization, so the stored config is audit-safe by default.
    - `src/elspeth/engine/orchestrator.py`: require a resolved config to begin a run; fail fast if missing instead of falling back to `{}`.
    - `src/elspeth/cli.py`: use the resolver output to populate `PipelineConfig.config` and pass the same resolved dict to the orchestrator. Avoid any implicit or silent defaults.
  - Optional (if we want stronger auditability): persist config provenance (settings file path, env var keys used) alongside the resolved config, but only if the schema allows it.
- Config or schema changes: none (unless adding provenance fields; then update the runs table explicitly).
- Tests to add/update:
  - Assert `runs.settings_json` equals the canonicalized resolved config.
  - Assert secrets are redacted/fingerprinted in `settings_json`.
  - Negative test: orchestrator refuses to start when resolved config is missing.
- Risks or migration steps:
  - Ensure redaction is deterministic and part of the canonicalization step so hashes remain stable.
  - Avoid storing raw secrets under any circumstance, even in debug paths.

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

## Resolution

**Fixed in:** 2026-01-19 (verified during triage)
**Fix:** CLI now properly populates `PipelineConfig.config` using `resolve_config()`:

**Evidence:**
- `src/elspeth/cli.py:293`: `config=resolve_config(config)` populates the config field
- `src/elspeth/engine/orchestrator.py:337,526`: Uses the populated `config.config`
