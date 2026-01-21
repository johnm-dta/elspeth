# Bug Report: Aggregation nodes record dynamic schema instead of configured schema

## Summary

- Orchestrator reads schema from node_info.config["schema"], but aggregation nodes store plugin options under "options". As a result, aggregation nodes are registered with a dynamic schema even when an explicit schema was configured.

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
- Data set or fixture: any aggregation config with explicit schema

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation with explicit schema under options (e.g., schema: {fields: strict}).
2. Run pipeline and inspect nodes table for aggregation node schema_config.

## Expected Behavior

- Aggregation node should record the configured schema from aggregation options.

## Actual Behavior

- Aggregation node is registered with schema {fields: dynamic}.

## Evidence

- Schema is taken from node_info.config["schema"] in `src/elspeth/engine/orchestrator.py:589-592`.
- Aggregation node config stores options under "options" in `src/elspeth/core/dag.py:309-324`.

## Impact

- User-facing impact: schema metadata in audit trail is wrong for aggregation nodes.
- Data integrity / security impact: audit trail cannot prove schema contract used for aggregation.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Aggregation node config wraps plugin options, so orchestrator misses schema nested under options.

## Proposed Fix

- Code changes (modules/files):
  - If node_type == aggregation, pull schema from node_info.config.get("options", {}).get("schema").
- Config or schema changes: N/A
- Tests to add/update:
  - Aggregation node schema recorded in nodes table matches configured schema.
- Risks or migration steps:
  - Ensure backward compatibility for aggregation configs without schema.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit trail should record configured schema per node.
- Observed divergence: aggregation schema recorded as dynamic.
- Reason (if known): schema lookup ignores options wrapper.
- Alignment plan or decision needed: define schema extraction rules for aggregation nodes.

## Acceptance Criteria

- Aggregation nodes register schema_config that matches aggregation options.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k aggregation -v`
- New tests required: yes, aggregation schema metadata test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
