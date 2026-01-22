# Bug Report: Transform/gate/sink spans are ambiguous when multiple plugin instances share the same plugin type

## Summary

- SpanFactory names and attributes use only the plugin type (e.g., "field_mapper", "csv"), so multiple instances of the same plugin in a pipeline produce indistinguishable spans. This breaks traceability and makes it impossible to map spans back to specific node_ids or pipeline steps.

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
- Notable tool calls or steps: reviewed SpanFactory and executor usage

## Steps To Reproduce

1. Configure a pipeline with two transforms of the same plugin type (e.g., two FieldMapper steps) or multiple sinks using the same plugin type.
2. Run with OpenTelemetry tracing enabled.
3. Inspect spans for transforms/sinks and observe identical span names/attributes, with no way to tell which pipeline node produced which span.

## Expected Behavior

- Spans include a unique identifier per pipeline node (node_id, step_index, or configured instance name), so multiple plugin instances are distinguishable.

## Actual Behavior

- Spans use only plugin type names (e.g., "transform:field_mapper"), so multiple instances are indistinguishable.

## Evidence

- Span names/attributes use plugin type only: src/elspeth/engine/spans.py:139-239
- Executors pass plugin type names, not node_id or instance name: src/elspeth/engine/executors.py:174, src/elspeth/engine/executors.py:365, src/elspeth/engine/executors.py:1288

## Impact

- User-facing impact: traces cannot be used to debug pipelines with repeated plugin types.
- Data integrity / security impact: observability cannot be correlated to specific node_states, weakening audit alignment.
- Performance or cost impact: increased troubleshooting time and potential misinterpretation of trace data.

## Root Cause Hypothesis

- SpanFactory APIs accept only plugin type names and do not include node_id/step_index; executor usage passes plugin type rather than unique instance identity.

## Proposed Fix

- Code changes (modules/files): src/elspeth/engine/spans.py, src/elspeth/engine/executors.py
- Config or schema changes: N/A
- Tests to add/update: extend tests to ensure spans include node_id or step_index when duplicate plugin types exist.
- Risks or migration steps: none; spans are observability-only.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/design/subsystems/00-overview.md:859 (spans should mirror Landscape events)
- Observed divergence: spans cannot be mapped to specific node_ids or steps when plugin types repeat.
- Reason (if known): span naming uses plugin type only.
- Alignment plan or decision needed: include node_id/step_index or configured instance name in span name/attributes.

## Acceptance Criteria

- Duplicate plugin instances produce distinguishable spans (node_id/step_index or instance name present).
- Trace-to-node_state correlation is possible without guessing.

## Tests

- Suggested tests to run: pytest tests/engine/test_spans.py
- New tests required: yes, verify unique identifiers on spans for repeated plugin types.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/design/subsystems/00-overview.md
