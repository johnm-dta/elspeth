# Bug Report: Source and sink lifecycle hooks never called

## Summary
- Orchestrator only invokes lifecycle hooks on transforms; source and sink `on_start`/`on_complete` hooks defined in protocols/base classes are never called, so plugin setup/teardown logic is skipped.

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
1. Implement a custom source or sink with `on_start`/`on_complete` that appends to a list.
2. Run `Orchestrator.run(...)` with that plugin.
3. Observe the list remains empty for source/sink hooks.

## Expected Behavior
- `SourceProtocol.on_start` is called before `load()`.
- `SinkProtocol.on_start` is called before any writes.
- `on_complete` is called for source/sinks even on failure (before `close()`).

## Actual Behavior
- Only transforms receive `on_start` and `on_complete` callbacks.
- Source/sink hooks are never invoked.

## Evidence
- Orchestrator only iterates `config.transforms` for lifecycle hooks: `src/elspeth/engine/orchestrator.py:303`, `src/elspeth/engine/orchestrator.py:398`.
- Source and sink lifecycle hooks are part of the plugin contract: `src/elspeth/plugins/protocols.py:20`, `src/elspeth/plugins/protocols.py:408`, `src/elspeth/plugins/base.py:8`.

## Impact
- User-facing impact: Plugins that rely on setup/teardown (opening files, DB connections, telemetry) never initialize or flush, causing missing output or resource leaks.
- Data integrity / security impact: Potentially incomplete writes or missing cleanup for sinks that buffer.
- Performance or cost impact: Retries or failed runs if initialization is required.

## Root Cause Hypothesis
- `Orchestrator._execute_run` only invokes lifecycle hooks for transforms, not for source or sink instances.

## Proposed Fix
- Code changes (modules/files):
  - Make lifecycle management an explicit, enforced contract (no ad‑hoc `hasattr` checks):
    - `src/elspeth/plugins/protocols.py`: require `on_start`/`on_complete` on Source/Sink/Transform protocols (with default no‑ops in Base*), so missing hooks are a plugin contract violation, not silently ignored.
    - `src/elspeth/engine/orchestrator.py`: centralize lifecycle handling in a `LifecycleManager` that:
      - calls `on_start` in deterministic order (source → transforms → sinks),
      - calls `on_complete` in reverse order (sinks → transforms → source) even on failure,
      - records hook failures explicitly and marks the run failed (do not suppress errors).
    - If a hook raises, capture the error details and include them in the run completion record (requires recorder support) rather than dropping exceptions.
  - Optional (recommended for auditability): add a `run_events` or `error_json` field on runs to store lifecycle hook failures with timestamp and plugin identity.
- Config or schema changes: only if adding run‑level error storage (`runs.error_json` or `run_events` table).
- Tests to add/update:
  - Assert lifecycle call order and that `on_complete` runs on failure.
  - Assert hook exceptions fail the run and are persisted in audit records.
- Risks or migration steps:
  - Enforcing hooks as mandatory may break third‑party plugins that omit them; provide a clear validation error at plugin registration time.
  - If adding run‑level error storage, migrate schema and update export logic to include lifecycle failures.

## Architectural Deviations
- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/requirements.md` (PLG-003: plugin instance lifecycle management).
- Observed divergence: lifecycle hooks are defined for sources/sinks but not executed.
- Reason (if known): likely oversight from transform-only hook implementation.
- Alignment plan or decision needed: update orchestrator lifecycle handling to cover all plugin types.

## Acceptance Criteria
- Source and sink hooks are invoked in the correct order for successful and failed runs.
- New tests cover source/sink `on_start` and `on_complete` behavior.

## Tests
- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -k lifecycle`
- New tests required: yes (source/sink lifecycle hooks).

## Notes / Links
- Related issues/PRs: N/A
- Related design docs: `src/elspeth/plugins/protocols.py`, `docs/design/requirements.md`
