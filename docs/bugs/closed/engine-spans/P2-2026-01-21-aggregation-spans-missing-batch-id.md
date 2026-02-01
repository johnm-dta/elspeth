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

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- Aggregation flush still uses `transform_span()` instead of `aggregation_span()`. (`src/elspeth/engine/executors.py:1120-1122`)
- `aggregation_span()` still exists with `batch_id` support, but remains unused. (`src/elspeth/engine/spans.py:193-218`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

The bug is confirmed in the current codebase at `/home/john/elspeth-rapid/src/elspeth/engine/executors.py:935`:

```python
# Step 3: Execute with timing and span
with self._spans.transform_span(transform.name, input_hash=input_hash):
    start = time.perf_counter()
    try:
        result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]
```

The code should be using `aggregation_span` instead of `transform_span`. Evidence:

1. **batch_id is available**: Retrieved at line 882 of `execute_flush()` method
2. **aggregation_span exists**: Defined at `src/elspeth/engine/spans.py:193-218` with explicit support for `batch_id` parameter
3. **Documentation specifies it**: Span hierarchy doc at `spans.py:7-16` explicitly shows `aggregation:{agg_name}` → `flush` pattern
4. **Never used in engine**: Grep shows `aggregation_span` is only defined in `spans.py`, tested in `test_spans.py`, but never invoked in actual execution code

**Git History:**

Examined commits from initial implementation to present:
- `b0c3174` (2026-01-?): Initial `execute_flush` implementation - used `transform_span` from the start
- `54edba7` (2026-01-23): Recent defensive guard addition - did not fix span usage
- `0f21ecb` (2026-01-23): Added PENDING status for async batches - did not fix span usage

No commits have addressed this issue. The bug has existed since the original implementation of aggregation flushing.

**Root Cause Confirmed:**

Yes. `AggregationExecutor.execute_flush()` incorrectly uses `self._spans.transform_span()` when it should use `self._spans.aggregation_span()`. This results in:
- Aggregation flush operations appearing as generic transforms in traces
- Loss of `batch.id` attribute that would enable correlation with batch audit records
- Inability to distinguish normal transform operations from aggregation flushes in OpenTelemetry traces

The fix is straightforward - change line 935 from:
```python
with self._spans.transform_span(transform.name, input_hash=input_hash):
```
to:
```python
with self._spans.aggregation_span(transform.name, batch_id=batch_id):
```

**Recommendation:**

**Keep open** - bug is valid and should be fixed. The impact is observability degradation (P2 severity is appropriate) but does not affect data integrity. The fix is a simple one-line change with the `batch_id` variable already in scope.

---

## FIX APPLIED: 2026-02-02

**Status:** FIXED

**Fix Summary:**

Changed `execute_flush()` in `src/elspeth/engine/executors.py` to use `aggregation_span()` instead of `transform_span()`.

**Code Change (executors.py:1135-1144):**
```python
# BEFORE:
with self._spans.transform_span(transform.name, input_hash=input_hash, token_ids=batch_token_ids):

# AFTER:
with self._spans.aggregation_span(
    transform.name,
    node_id=node_id,
    batch_id=batch_id,
    token_ids=batch_token_ids,
):
```

**Result:**
- Aggregation flushes now emit spans with name `aggregation:{plugin_name}` (distinguishable from regular transforms)
- `batch.id` attribute is now included on aggregation spans
- `node.id` attribute added for disambiguation when multiple aggregations exist
- `token.ids` preserved for batch token tracking

**Tests Added:** `tests/engine/test_spans.py::TestNodeIdOnSpans::test_aggregation_span_includes_node_id`

**Verified By:** Claude Opus 4.5 systematic debugging
