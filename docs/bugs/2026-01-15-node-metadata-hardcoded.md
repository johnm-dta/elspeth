# Bug Report: Node registration ignores plugin metadata

## Summary
- Orchestrator registers all nodes with `plugin_version="1.0.0"` and default determinism, ignoring actual plugin metadata (version/determinism/schema), so Landscape audit records are inaccurate.

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
1. Create a plugin with `plugin_version = "0.2.0"` and `determinism = Determinism.NONDETERMINISTIC`.
2. Run a pipeline with that plugin.
3. Inspect the `nodes` table for the run.

## Expected Behavior
- `nodes.plugin_version` and `nodes.determinism` should reflect the plugin’s declared metadata (and schema hashes if available).

## Actual Behavior
- `plugin_version` is always `"1.0.0"` and `determinism` defaults to deterministic regardless of plugin metadata.

## Evidence
- Hard-coded version in Orchestrator: `src/elspeth/engine/orchestrator.py:243`.
- Recorder supports determinism and schema hashes but never receives them: `src/elspeth/core/landscape/recorder.py:313`.
- Plugin metadata exists on base classes: `src/elspeth/plugins/base.py:47`.

## Impact
- User-facing impact: Audit records misrepresent plugin versions and determinism, undermining reproducibility and compliance.
- Data integrity / security impact: Reproducibility grade and audit trail integrity are compromised.
- Performance or cost impact: N/A.

## Root Cause Hypothesis
- Orchestrator builds nodes from graph info only and does not read plugin instance metadata (version/determinism/schema hashes).

## Proposed Fix
- Code changes (modules/files):
  - `src/elspeth/engine/orchestrator.py`: use plugin instances (source/transforms/sinks) to supply `plugin_version`, `determinism`, and schema hashes when calling `register_node`.
  - Optionally extend `ExecutionGraph` to carry plugin metadata so registration is accurate.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test to assert `plugin_version` and `determinism` stored in `nodes` match plugin attributes.
- Risks or migration steps:
  - Ensure adapters expose underlying plugin metadata consistently (e.g., `SinkAdapter`).

## Architectural Deviations
- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:249` (audit trail captures plugin instances with metadata).
- Observed divergence: node records use hard-coded version and default determinism.
- Reason (if known): registration uses graph-only info without plugin metadata.
- Alignment plan or decision needed: define authoritative source for plugin metadata during node registration.

## Acceptance Criteria
- Node records contain actual plugin version and determinism for source/transform/gate/sink nodes.

## Tests
- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -k node`
- New tests required: yes (node metadata persistence).

## Notes / Links
- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`
