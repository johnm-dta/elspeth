# Bug Report: Aggregation nodes record hardcoded metadata instead of transform metadata

## Summary

- Aggregation nodes are registered with plugin_version="1.0.0" and determinism=DETERMINISTIC regardless of the actual batch-aware transform. This misrepresents non-deterministic aggregation transforms (e.g., LLM batch transforms) in the audit trail.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any pipeline with aggregation using non-deterministic transform

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation node using a non-deterministic transform (e.g., azure_batch_llm).
2. Run a pipeline and inspect nodes table metadata for the aggregation node.

## Expected Behavior

- Aggregation node metadata should reflect the actual transform's plugin_version and determinism.

## Actual Behavior

- Aggregation nodes are registered as deterministic with a hardcoded version.

## Evidence

- Aggregation nodes are hardcoded in `src/elspeth/engine/orchestrator.py:569-573`.
- Aggregation transforms exist in config.transforms with real metadata, but are not used for node registration.

## Impact

- User-facing impact: audit metadata misrepresents LLM batch transforms as deterministic.
- Data integrity / security impact: audit trail accuracy compromised.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Node registration treats aggregation nodes as metadata-only, ignoring the actual transform instance that executes.

## Proposed Fix

- Code changes (modules/files):
  - Resolve aggregation node metadata from the batch-aware transform instance with matching node_id.
- Config or schema changes: N/A
- Tests to add/update:
  - Assert aggregation node determinism/version match transform metadata.
- Risks or migration steps:
  - Ensure aggregation transforms are discoverable via node_id during registration.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit trail must reflect actual plugin metadata.
- Observed divergence: aggregation nodes use placeholders.
- Reason (if known): node registration skips aggregation transform instances.
- Alignment plan or decision needed: define authoritative metadata source for aggregation nodes.

## Acceptance Criteria

- Aggregation nodes record determinism and plugin_version from their transform.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k aggregation -v`
- New tests required: yes, aggregation node metadata test.

## Notes / Links

- Related issues/PRs: P2-2026-01-15-node-metadata-hardcoded (config gates only)
- Related design docs: CLAUDE.md auditability standard
