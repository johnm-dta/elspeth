# Bug Report: Row span token.id attribute becomes incorrect after fork/deaggregation

## Summary

- RowProcessor wraps the entire work queue in a single row_span created with the initial token_id. When transforms fork or deaggregate, child tokens have new token_id values but continue under the same row_span, so the token.id attribute is wrong for child token work.

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
- Notable tool calls or steps: reviewed SpanFactory and RowProcessor usage

## Steps To Reproduce

1. Configure a pipeline with a deaggregation transform (e.g., json_explode) or a forking gate.
2. Run with OpenTelemetry tracing enabled.
3. Inspect spans for child token processing; the row_span token.id attribute remains the parent token_id.

## Expected Behavior

- token.id should reflect the current token being processed, or row_span should omit token.id and use per-token spans for token-specific attributes.

## Actual Behavior

- token.id is set once on row_span using the initial token_id and never updated for child tokens.

## Evidence

- row_span records token.id from its input argument: src/elspeth/engine/spans.py:115-137
- RowProcessor opens a single row_span around the entire work queue: src/elspeth/engine/processor.py:531-656

## Impact

- User-facing impact: traces mislabel child token operations, making debugging fork/deaggregation flows unreliable.
- Data integrity / security impact: tracing metadata no longer matches token lineage.
- Performance or cost impact: none directly, but debugging time increases.

## Root Cause Hypothesis

- row_span is scoped to the parent token, but child tokens are processed within the same span and share the same token.id attribute.

## Proposed Fix

- Code changes (modules/files): src/elspeth/engine/spans.py, src/elspeth/engine/processor.py
- Config or schema changes: N/A
- Tests to add/update: add span tests for fork/deaggregation flows ensuring token.id matches the active token.
- Risks or migration steps: none; tracing only.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/design/subsystems/00-overview.md:840-843 (token_id used for fork/join identity)
- Observed divergence: spans report token.id that does not match actual token lineage for child tokens.
- Reason (if known): row_span covers the entire work queue rather than per token.
- Alignment plan or decision needed: introduce per-token spans or update token.id per work item.

## Acceptance Criteria

- Spans for child tokens carry the correct token.id (or omit token.id at row scope).
- Fork/deaggregation traces clearly distinguish parent vs child token processing.

## Tests

- Suggested tests to run: pytest tests/engine/test_spans.py
- New tests required: yes, trace metadata correctness in fork/deaggregation scenarios.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/design/subsystems/00-overview.md
