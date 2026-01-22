# Bug Report: Span names embed run_id and row_id, creating extreme cardinality

## Summary

- run_span and row_span use IDs directly in span names (e.g., "run:{run_id}", "row:{row_id}"). This creates unbounded span name cardinality, which can overwhelm tracing backends and span-derived metrics.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/spans.py for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed span naming in SpanFactory

## Steps To Reproduce

1. Run a pipeline with many rows while OpenTelemetry tracing is enabled.
2. Inspect exported spans or span-derived metrics.
3. Observe span names are unique per run and per row, causing high-cardinality label sets.

## Expected Behavior

- Span names should be stable (e.g., "run", "row") with run_id/row_id stored as attributes.

## Actual Behavior

- Span names embed run_id and row_id, creating unique span names per run/row.

## Evidence

- run_span name includes run_id: src/elspeth/engine/spans.py:92
- row_span name includes row_id: src/elspeth/engine/spans.py:134

## Impact

- User-facing impact: tracing backends may become slow or expensive due to high-cardinality span names.
- Data integrity / security impact: none.
- Performance or cost impact: increased storage, indexing, and metric cardinality costs.

## Root Cause Hypothesis

- SpanFactory uses IDs in span names instead of attributes.

## Proposed Fix

- Code changes (modules/files): src/elspeth/engine/spans.py
- Config or schema changes: N/A
- Tests to add/update: add tests asserting span names are stable and IDs stored as attributes.
- Risks or migration steps: tracing-only change; no runtime behavior impact.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): N/A (best-practice observability guidance)
- Observed divergence: span names are high-cardinality.
- Reason (if known): convenience naming.
- Alignment plan or decision needed: use stable names and keep IDs as attributes.

## Acceptance Criteria

- Span names are stable across runs and rows.
- run_id and row_id remain available as span attributes.

## Tests

- Suggested tests to run: pytest tests/engine/test_spans.py
- New tests required: yes, span name stability checks.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
