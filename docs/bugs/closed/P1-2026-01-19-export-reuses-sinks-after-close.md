# Bug Report: Landscape export reuses sinks after on_complete/close (lifecycle violation; export can fail)

## Summary

- `Orchestrator._execute_run()` always calls `sink.on_complete(ctx)` and `sink.close()` in a `finally:` block for **all sinks**, then returns to `Orchestrator.run()`.
- `Orchestrator.run()` performs post-run landscape export after `_execute_run()` returns, reusing sink instances from `PipelineConfig.sinks` and calling `sink.write(...)` (JSON export) and `sink.close()` again.
- This violates sink lifecycle ordering (`on_complete` should occur after all writes) and can fail for sinks that do not support being used after `close()` (or behave incorrectly after being “completed”).

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: `settings.landscape.export.enabled: true`
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `orchestrator.py`

## Steps To Reproduce

1. Configure `landscape.export.enabled: true` with `format: json` and `sink: <some sink>`.
2. Use an export sink implementation whose `close()` makes the instance unusable (or whose `on_complete()` finalizes state and forbids further writes).
3. Run a pipeline.

## Expected Behavior

- Export uses a sink instance that is in a valid lifecycle state:
  - `on_start` → `write` → `flush` → `on_complete` → `close`
- Post-run export should not depend on reusing a previously closed sink instance.

## Actual Behavior

- Export can occur after `on_complete`/`close` were already called for the same sink instance, leading to failures or incorrect behavior.

## Evidence

- Post-run export occurs after `_execute_run()`:
  - `src/elspeth/engine/orchestrator.py:351` (calls `_execute_run`)
  - `src/elspeth/engine/orchestrator.py:365` (begins export block after `_execute_run` returns)
- `_execute_run()` closes sinks unconditionally:
  - `src/elspeth/engine/orchestrator.py:691` (finally block)
  - `src/elspeth/engine/orchestrator.py:698` (calls `sink.on_complete`)
  - `src/elspeth/engine/orchestrator.py:707` (calls `sink.close`)
- Export reuses the sink instance from `PipelineConfig.sinks`:
  - `src/elspeth/engine/orchestrator.py:763` (selects export sink from `sinks`)
  - `src/elspeth/engine/orchestrator.py:790` (calls `sink.write(...)` for JSON export)
  - `src/elspeth/engine/orchestrator.py:795` (calls `sink.close()` again)

## Impact

- User-facing impact: export may fail or write incorrect data depending on sink implementation; export sink lifecycle hooks run in the wrong order.
- Data integrity / security impact: export failures reduce audit accessibility; lifecycle misuse risks partial exports or corrupted artifacts.
- Performance or cost impact: re-runs may be needed to recover export artifacts.

## Root Cause Hypothesis

- The pipeline runtime treats export as an afterthought but reuses the same sink registry/instances that were already finalized as part of the run’s normal sink lifecycle.

## Proposed Fix

- Code changes (modules/files):
  - Option A (simplest): move export earlier (before sink `on_complete`/`close`), or defer closing sinks until after export completes.
  - Option B (cleanest): instantiate a dedicated export sink instance separate from `PipelineConfig.sinks` (and run full lifecycle hooks for export sinks independently).
  - Ensure export sinks receive a context consistent with sink expectations (may need `PluginContext.landscape`).
- Config or schema changes:
  - Consider separate `settings.landscape.export.sink` lifecycle from pipeline sinks (allow it to be absent from `PipelineConfig.sinks`).
- Tests to add/update:
  - Add a sink fixture whose `close()` makes it unusable; verify export runs successfully (or fails early with a clear error) under the chosen approach.
- Risks or migration steps:
  - Ensure existing sinks that rely on lazy open still behave; avoid double-closing resources.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/plugins/protocols.py` sink lifecycle expectations (flush/close), and `CLAUDE.md` audit export goals
- Observed divergence: export writes occur after `on_complete`/`close`.
- Alignment plan or decision needed: define whether export sinks are part of “pipeline sinks” or an independent post-run subsystem.

## Acceptance Criteria

- With export enabled, export completes without relying on writing to a sink instance that was already closed/completed.
- Sink lifecycle hooks occur in the correct order for both pipeline sinks and export sinks.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k export`
- New tests required: yes (lifecycle ordering and closed-sink behavior)

## Notes / Links

- Related issues/PRs: `docs/bugs/open/2026-01-19-export-fails-old-landscape-schema-expand-group-id.md` (separate export failure mode)
- Related design docs: `docs/design/architecture.md`
