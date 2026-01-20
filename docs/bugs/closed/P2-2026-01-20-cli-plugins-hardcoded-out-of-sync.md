# Bug Report: CLI plugin listing/instantiation is hard-coded and drifts from `PluginManager`

## Summary

- The CLI has two separate hard-coded plugin registries:
  1. `_execute_pipeline()` hard-codes instantiation of sources/sinks/transforms.
  2. `elspeth plugins list` hard-codes a `PLUGIN_REGISTRY` list for display.
- These are already out of sync (e.g., `batch_stats` is supported by `_execute_pipeline()` and registered as a built-in transform, but not shown by `plugins list`).
- The repository already contains a `PluginManager` that loads built-ins via pluggy hooks; CLI should use it as the single source of truth.

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

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 1 (CLI), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of CLI + plugins subsystem

## Steps To Reproduce

1. Run `elspeth plugins list --type transform`.
2. Observe that `batch_stats` is not listed.
3. Configure a pipeline using `row_plugins: [{plugin: batch_stats, ...}]` and run it.
4. Observe that it executes successfully (plugin is available to the run path).

## Expected Behavior

- `elspeth plugins list` should reflect the actual available plugins used at runtime.
- CLI should not duplicate plugin mapping logic; it should use `PluginManager` for discovery and lookup.

## Actual Behavior

- `plugins list` is incomplete and will drift as new plugins are added.
- `run` and `plugins list` each maintain their own hard-coded plugin catalogs.

## Evidence

- Hard-coded transform mapping includes `batch_stats`:
  - `src/elspeth/cli.py:236-241`
- Hard-coded display registry omits `batch_stats`:
  - `src/elspeth/cli.py:405-424`
- Built-in transforms include `BatchStats` via pluggy hook:
  - `src/elspeth/plugins/transforms/hookimpl.py:12-19`
- `PluginManager` exists and loads built-ins:
  - `src/elspeth/plugins/manager.py:110-146`

## Impact

- User-facing impact: `plugins list` lies; users may assume a plugin is unavailable when it is, or vice versa.
- Data integrity / security impact: low (but can lead to operators running a different pipeline than they think).
- Performance or cost impact: low.

## Root Cause Hypothesis

- CLI was implemented with a “Phase 4 static registry” approach and never refactored to use the already-present plugin manager.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`:
    - Replace hard-coded plugin lookups with `PluginManager`:
      - `pm = PluginManager(); pm.load_builtins()`
      - resolve plugin class by name (source/transform/sink) and instantiate with config options
    - Implement `plugins list` by querying `pm.get_sources()`, `pm.get_transforms()`, `pm.get_sinks()`
      and printing `name`, `plugin_version`, `determinism`, and schema hashes (optional).
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that `elspeth plugins list --type transform` includes `batch_stats`.
  - Add a regression test ensuring the CLI uses the same source of truth for execution and listing.
- Risks or migration steps:
  - Ensure CLI and tests don’t implicitly depend on hard-coded ordering; sort plugin names deterministically for output.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: single source of truth for plugins vs duplicated registries.
- Reason (if known): incremental implementation.
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- `plugins list` output matches the set of plugins that `run` can instantiate.
- Adding a new built-in plugin requires registering it once (hookimpl/PluginManager), not editing CLI.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
