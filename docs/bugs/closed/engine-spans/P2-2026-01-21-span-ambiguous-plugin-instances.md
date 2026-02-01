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

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- Span naming still uses plugin names only; no node_id or instance identity in span names/attributes. (`src/elspeth/engine/spans.py:140-239`)
- Executors still pass `transform.name` / `gate.name` / `sink.name` to span creation. (`src/elspeth/engine/executors.py:246`, `src/elspeth/engine/executors.py:517`, `src/elspeth/engine/executors.py:1699`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

Examined the current implementation in both `spans.py` and `executors.py`:

1. **SpanFactory API (src/elspeth/engine/spans.py:139-240)**:
   - `transform_span()` accepts only `transform_name: str` (line 141)
   - `gate_span()` accepts only `gate_name: str` (line 169)
   - `sink_span()` accepts only `sink_name: str` (line 223)
   - All span names use the pattern `f"{type}:{name}"` (e.g., `"transform:field_mapper"`)
   - Attributes set: `plugin.name` and `plugin.type` only

2. **Executor calls (src/elspeth/engine/executors.py)**:
   - Line 174: `self._spans.transform_span(transform.name, input_hash=input_hash)`
   - Line 365: `self._spans.gate_span(gate.name, input_hash=input_hash)`
   - Line 1408: `self._spans.sink_span(sink.name)`
   - All calls pass `plugin.name` which is the plugin type, NOT the node_id

3. **node_id is available but unused**:
   - Transform executor: `transform.node_id` is available (line 155, 161, 170)
   - Gate executor: `gate.node_id` is available (line 353, 359, 400)
   - Sink executor: `sink.node_id` is available (line 1392, 1395)
   - The node_id is recorded in Landscape (`begin_node_state()`) but NOT passed to spans

4. **Node ID structure (src/elspeth/core/dag.py:334-340)**:
   - Node IDs are deterministic: `f"{prefix}_{name}_{config_hash}"` (line 340)
   - Example: `transform_field_mapper_a1b2c3d4e5f6` for a FieldMapper with specific config
   - Two FieldMapper instances with different configs get different node_ids
   - Two FieldMapper instances with identical configs would get the SAME node_id (deterministic)

**Git History:**

No commits since the original SpanFactory implementation (commit 5099cf1) have modified the span naming behavior or added node_id tracking to spans. The code is identical to the original implementation.

**Root Cause Confirmed:**

YES - The bug is still present. The root cause is exactly as described:

1. SpanFactory methods accept only plugin name (type), not node_id
2. Executors pass `plugin.name` to span creation, even though `plugin.node_id` is available
3. When multiple instances of the same plugin type exist (e.g., two FieldMapper transforms), their spans are indistinguishable if they have the same config (same node_id) or confusing if they have different configs (different node_ids but span still shows generic "field_mapper")

**Specific scenario that demonstrates the bug:**

Pipeline with two FieldMapper transforms:
- Transform 0: FieldMapper with config `{"mapping": {"a": "b"}}` → node_id = `transform_field_mapper_abc123`
- Transform 1: FieldMapper with config `{"mapping": {"x": "y"}}` → node_id = `transform_field_mapper_def456`

Both produce spans named `"transform:field_mapper"` with attribute `plugin.name = "field_mapper"`. There is NO way to correlate these spans back to specific node_states or determine which FieldMapper instance created which span.

**Recommendation:**

Keep open. This is a valid P2 bug that degrades observability for pipelines with repeated plugin types. The fix is straightforward:

1. Add optional `node_id` parameter to span methods in SpanFactory
2. Update executors to pass `plugin.node_id`
3. Include node_id in span name (e.g., `"transform:transform_field_mapper_abc123"`) or as attribute (`node.id`)
4. Add test case with duplicate plugin types to verify span distinguishability

This change is low-risk (observability only, no audit impact) and high-value (enables proper trace correlation).

---

## FIX APPLIED: 2026-02-02

**Status:** FIXED

**Fix Summary:**

Added optional `node_id` parameter to all span methods in SpanFactory and updated all executor call sites to pass the appropriate node_id.

**SpanFactory Changes (spans.py):**
- `transform_span()`: Added `node_id: str | None = None` parameter
- `gate_span()`: Added `node_id: str | None = None` parameter
- `aggregation_span()`: Added `node_id: str | None = None` parameter
- `sink_span()`: Added `node_id: str | None = None` parameter
- All methods now set `span.set_attribute("node.id", node_id)` when provided

**Executor Changes (executors.py):**
- `TransformExecutor.execute_transform()`: Now passes `node_id=transform.node_id`
- `GateExecutor.execute_gate()`: Now passes `node_id=gate.node_id`
- `GateExecutor._execute_config_gate()`: Now passes `node_id=node_id`
- `AggregationExecutor.execute_flush()`: Now passes `node_id=node_id`
- `SinkExecutor.write()`: Now passes `node_id=sink_node_id`

**Result:**
- Multiple instances of the same plugin type now have distinguishable spans via `node.id` attribute
- Spans can be correlated with Landscape `node_states` table using `node.id`
- Backwards compatible: `node_id=None` omits the attribute (existing code unaffected)

**Tests Added:**
- `tests/engine/test_spans.py::TestNodeIdOnSpans::test_transform_span_includes_node_id`
- `tests/engine/test_spans.py::TestNodeIdOnSpans::test_gate_span_includes_node_id`
- `tests/engine/test_spans.py::TestNodeIdOnSpans::test_sink_span_includes_node_id`
- `tests/engine/test_spans.py::TestNodeIdOnSpans::test_aggregation_span_includes_node_id`
- `tests/engine/test_spans.py::TestNodeIdOnSpans::test_duplicate_plugins_distinguishable_by_node_id`
- `tests/engine/test_spans.py::TestNodeIdOnSpans::test_node_id_none_omits_attribute`

**Verified By:** Claude Opus 4.5 systematic debugging
