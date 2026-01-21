# Bug Report: Aggregation flushes emit transform spans only, losing batch_id and aggregation context

## Summary

- SpanFactory provides aggregation_span with batch_id support, but AggregationExecutor uses transform_span for flushes. As a result, aggregation flush spans are not distinguished from normal transform spans and do not carry batch_id attributes.

## Severity

- Severity: major
- Priority: P2

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
- Notable tool calls or steps: reviewed SpanFactory and AggregationExecutor usage

## Steps To Reproduce

1. Configure a batch aggregation (e.g., batch_stats) with a trigger.
2. Run with OpenTelemetry tracing enabled.
3. Observe flush operations emit transform spans without batch_id or aggregation-specific span names.

## Expected Behavior

- Aggregation flushes should emit aggregation spans (or transform spans with batch_id) so flushes are distinguishable and batch_id is recorded.

## Actual Behavior

- AggregationExecutor uses transform_span for flushes; aggregation_span is unused and batch_id is not recorded on spans.

## Evidence

- aggregation_span supports batch_id: src/elspeth/engine/spans.py:193-218
- execute_flush uses transform_span (no batch_id attribute): src/elspeth/engine/executors.py:894-944

## Impact

- User-facing impact: tracing cannot distinguish normal transform operations from aggregation flushes.
- Data integrity / security impact: observability cannot correlate spans to batch_id, weakening audit alignment.
- Performance or cost impact: harder to diagnose batch-trigger timing and backpressure issues.

## Root Cause Hypothesis

- AggregationExecutor reuses transform_span instead of aggregation_span, so aggregation-specific metadata is never attached.

## Proposed Fix

- Code changes (modules/files): src/elspeth/engine/executors.py, src/elspeth/engine/spans.py
- Config or schema changes: N/A
- Tests to add/update: add tests verifying aggregation spans include batch_id and correct naming.
- Risks or migration steps: none; tracing only.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): src/elspeth/engine/spans.py:7-16 (span hierarchy includes aggregation:flush)
- Observed divergence: aggregation spans are never emitted; batch_id not recorded on spans.
- Reason (if known): aggregation_span is defined but unused.
- Alignment plan or decision needed: use aggregation_span in execute_flush or add batch_id to transform spans for flushes.

## Acceptance Criteria

- Aggregation flushes emit spans clearly labeled as aggregation operations.
- batch_id is present on flush spans.

## Tests

- Suggested tests to run: pytest tests/engine/test_spans.py
- New tests required: yes, verify aggregation span usage and batch_id attribute.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: src/elspeth/engine/spans.py
