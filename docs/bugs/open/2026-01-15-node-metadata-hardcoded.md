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
  - Treat plugin metadata as a first-class compile-time artifact, not a runtime patch:
    - `src/elspeth/plugins/manager.py`: expose a public helper or `PluginSpec.from_plugin(...)` for metadata resolution (version, determinism, schema hashes) so the CLI and engine share a single source of truth.
    - `src/elspeth/core/dag.py`: extend `NodeInfo` to carry `plugin_version`, `determinism`, and a stable `schema_hash` computed from input/output schema hashes; populate these in `ExecutionGraph.from_config(...)` using the plugin registry.
    - `src/elspeth/engine/orchestrator.py`: register nodes from `NodeInfo` (no hard-coded defaults), ensuring the audit trail matches the validated execution graph.
    - `src/elspeth/plugins/protocols.py` and `src/elspeth/plugins/base.py`: add `plugin_version` and `determinism` to `SourceProtocol`/`BaseSource` to avoid source nodes becoming an audit exception.
  - Avoid any runtime fallback or hard-coded values; if metadata is missing, fail fast during graph compilation.
- Config or schema changes: none.
- Tests to add/update:
  - Orchestrator test asserting `nodes.plugin_version`, `nodes.determinism`, and `nodes.schema_hash` match plugin class metadata for source, transform, gate, and sink.
  - Registry test to assert schema hash composition is stable and deterministic.
- Risks or migration steps:
  - Requires wiring a plugin registry into graph compilation; keep this as the single metadata resolution path.
  - Ensure adapters (e.g., `SinkAdapter`) surface plugin identity without masking the underlying plugin class metadata.

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
