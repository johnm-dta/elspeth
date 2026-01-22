# Bug Report: Resume uses synthetic edge IDs that do not exist in Landscape

## Summary

- Resume builds a new `edge_map` with fake edge IDs ("resume_edge_*"), but routing events require real `edges.edge_id` values. This causes FK errors or invalid audit records when config gates record routing during resume.

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
- Data set or fixture: any resume run with config gates

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Run a pipeline with config gates so routing events are recorded.
2. Force a failure and resume via Orchestrator.resume().
3. Observe resume processing when routing events are recorded.

## Expected Behavior

- Resume should use the actual edge IDs registered for the run so routing_events FK constraints are satisfied.

## Actual Behavior

- Resume uses synthetic edge IDs (e.g., "resume_edge_0"), which do not exist in `edges`, causing FK violations or broken audit records.

## Evidence

- Synthetic edge IDs created in `src/elspeth/engine/orchestrator.py:1204-1209`.
- routing_events.edge_id is a FK to edges.edge_id in `src/elspeth/core/landscape/schema.py:230-236`.

## Impact

- User-facing impact: resume crashes on routing event insert for pipelines with gates.
- Data integrity / security impact: routing audit trail cannot be recorded correctly.
- Performance or cost impact: recovery attempts fail, requiring manual intervention.

## Root Cause Hypothesis

- Resume path never loads the edge IDs already registered for the run, and instead fabricates IDs.

## Proposed Fix

- Code changes (modules/files):
  - Load edge IDs for the run from Landscape (by from_node/label) and rebuild edge_map with real IDs.
- Config or schema changes: N/A
- Tests to add/update:
  - Resume test with config gate routing that asserts routing_events insert succeeds.
- Risks or migration steps:
  - None; read-only query of existing edges.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): audit trail must record routing events with valid edge references.
- Observed divergence: resume uses non-existent edge IDs.
- Reason (if known): edge IDs are not reloaded on resume.
- Alignment plan or decision needed: define resume strategy for reusing graph edge IDs.

## Acceptance Criteria

- Resume uses real edge IDs and routing events insert without FK errors.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k resume -v`
- New tests required: yes, resume routing event coverage.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
