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

---

## RESOLUTION: 2026-02-02

**Status:** FIXED

**Fixed By:** Claude Code (Opus 4.5)

**Solution:**

Added `token_id` and `token_ids` parameters to child span methods so they record the actual token being processed:

1. **spans.py changes:**
   - `transform_span()`: Added `token_id: str | None` for single-row transforms, `token_ids: Sequence[str] | None` for batch transforms
   - `gate_span()`: Added `token_id: str | None`
   - `sink_span()`: Added `token_ids: Sequence[str] | None` (sinks batch-write multiple tokens)
   - `aggregation_span()`: Added `token_ids: Sequence[str] | None`

2. **executors.py changes:**
   - `TransformExecutor.execute()`: Passes `token_id=token.token_id`
   - `GateExecutor.execute_gate()`: Passes `token_id=token.token_id`
   - `GateExecutor.execute_config_gate()`: Passes `token_id=token.token_id`
   - `AggregationExecutor.execute_flush()`: Passes `token_ids=[t.token_id for t in buffered_tokens]`
   - `SinkExecutor.write()`: Passes `token_ids=[t.token_id for t in tokens]`

3. **Tests added:**
   - `TestTokenIdOnChildSpans`: 5 tests verifying token.id tracking on child spans
   - `TestTokenIdEdgeCases`: 6 tests for edge cases (None, empty sequences, batch transforms)

**Architectural Decision:**

Kept `row_span` unchanged - it represents "all processing for tokens derived from this source row" which is semantically correct. Child spans (transform/gate/sink) now carry the precise `token.id` for the operation being performed.

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID (superseded by fix)

- RowProcessor still wraps the entire work queue in a single `row_span` created with the initial token. (`src/elspeth/engine/processor.py:383-405`)
- `row_span()` still sets `token.id` at span creation and never updates it for child tokens. (`src/elspeth/engine/spans.py:116-137`)

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

Examined current implementation in `/home/john/elspeth-rapid/src/elspeth/engine/processor.py` and `/home/john/elspeth-rapid/src/elspeth/engine/spans.py`:

1. **Row span created once per row** (processor.py:521 and processor.py:598):
   - Both `process_row()` and `process_existing_row()` create a single `row_span` with the initial token's `token_id`
   - Pattern: `with self._spans.row_span(token.row_id, token.token_id):`

2. **Work queue processes multiple tokens under same span**:
   - Work queue loop (lines 522-547 and 599-620) iterates through parent token + all child tokens
   - Each work item has its own `token.token_id`, but all execute under the original span

3. **Child tokens created via two mechanisms**:
   - **Deaggregation** (processor.py:822-839): `expand_token()` creates child tokens with new IDs, queued for processing
   - **Forking** (processor.py:688-705): `fork_token()` creates child tokens with new IDs via gate executor

4. **Span structure** (spans.py:115-137):
   - `row_span()` sets `token.id` attribute once at span creation (line 136)
   - No mechanism exists to update or create per-token spans for children
   - The attribute is fixed for the entire span lifetime

**Git History:**

No commits since 2026-01-21 have addressed span token.id tracking:
- Recent processor.py commits (0a9cf2a, 3399faf, c6afc31, e93e56c, 6befd9a) focus on outcome recording and payload storage
- No changes to spans.py since RC-1 release (c786410)
- Test suite (tests/engine/test_spans.py) validates span creation but not token.id correctness for child tokens

**Root Cause Confirmed:**

Yes, the bug is still present. The architectural issue is:

1. Row-scoped span is created with initial token ID
2. Child tokens (from fork/deaggregation) have different token IDs
3. All child token operations execute under parent span with wrong token.id attribute
4. OpenTelemetry traces will show incorrect token.id for all child token operations

**Example Flow:**
```text
Token parent-001 enters work queue
→ row_span created with token.id="parent-001"
→ Transform causes deaggregation into child-001, child-002, child-003
→ Children queued and processed
→ All child operations tagged with token.id="parent-001" (incorrect)
→ Actual processing is for child-001, child-002, child-003
```

**Impact Assessment:**

- **Severity**: Correctly marked as P2 (major but not critical)
- **User Impact**: Trace debugging for fork/deaggregation is unreliable
- **Audit Impact**: Audit trail (Landscape DB) is correct; only OpenTelemetry spans affected
- **Scope**: Only affects tracing metadata, not data integrity or pipeline execution

**Recommendation:**

**Keep open.** This is a legitimate tracing bug that should be fixed.

**Potential fixes:**
1. **Per-token spans**: Create new span for each token in work queue (increases span count)
2. **Token.id updates**: Update span attribute when processing new token (OpenTelemetry may not support this)
3. **Omit token.id**: Remove token.id from row_span, add to transform/gate/sink spans instead
4. **Hybrid**: Keep row_span for parent, create child_token_span() for fork/deaggregation children

Fix priority should align with observability roadmap - if OpenTelemetry tracing is actively used for debugging, prioritize higher. If tracing is currently unused, defer until observability work begins.
