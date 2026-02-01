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

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- `run_span()` still names spans as `run:{run_id}` and `row_span()` uses `row:{row_id}`. (`src/elspeth/engine/spans.py:92-137`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 6 (FINAL)

**Current Code Analysis:**

The bug is **confirmed and still present** in the current codebase. Analysis of `/home/john/elspeth-rapid/src/elspeth/engine/spans.py` shows:

1. **Line 92**: `run_span()` uses `f"run:{run_id}"` as the span name
   - Creates unbounded cardinality - every run gets a unique span name
   - run_id is also correctly stored as attribute `"run.id"` (line 93)

2. **Line 134**: `row_span()` uses `f"row:{row_id}"` as the span name
   - Creates unbounded cardinality - every row gets a unique span name
   - row_id is also correctly stored as attribute `"row.id"` (line 135)

Other span types follow correct low-cardinality naming patterns:
- `source:{source_name}` - bounded by number of source plugins (line 110)
- `transform:{transform_name}` - bounded by number of transforms (line 159)
- `gate:{gate_name}` - bounded by number of gates (line 186)
- `aggregation:{aggregation_name}` - bounded by number of aggregations (line 213)
- `sink:{sink_name}` - bounded by number of sinks (line 237)

**Git History:**

The SpanFactory was introduced in commit `5099cf1` (2026-01-12) with this exact pattern. No subsequent commits have modified the span naming behavior:
- `c786410` (RC1, 2026-01-22) - Added the file (merge commit)
- `07084c3` (2026-01-20) - Only formatting changes (line length)
- `0f0c38f` (2026-01-20) - Mypy fixes, no span naming changes

No commits have addressed this cardinality issue.

**Root Cause Confirmed:**

The root cause is exactly as described in the original bug report. The SpanFactory embeds variable IDs directly in span names instead of using fixed span names with IDs as attributes. This violates OpenTelemetry best practices:

- **Current (WRONG)**: Span name = `"run:abc123"`, attributes = `{"run.id": "abc123"}`
- **Correct**: Span name = `"run"`, attributes = `{"run.id": "abc123"}`

**Impact Assessment:**

In production with OpenTelemetry tracing enabled:
- A pipeline processing 10,000 rows will create 10,000 unique span names (`row:1`, `row:2`, ..., `row:10000`)
- Running 100 pipeline executions will create 100 unique span names (`run:001`, `run:002`, ..., `run:100`)
- This creates high-cardinality metrics in backends that derive metrics from spans (Prometheus, Datadog, etc.)
- Span indexing and storage costs increase proportionally
- Query performance degrades as span name dictionaries grow unbounded

**Test Coverage:**

The existing tests (`tests/engine/test_spans.py`) verify that spans are created and attributes are set, but do NOT verify span name stability. No tests check for low-cardinality span naming patterns.

**Recommendation:**

**Keep open** - This is a valid P3 bug that should be fixed before production deployment. The fix is straightforward:
1. Change span names from `f"run:{run_id}"` → `"run"` and `f"row:{row_id}"` → `"row"`
2. IDs are already stored as attributes, so no information is lost
3. Add tests verifying span names are stable (don't include variable IDs)

Priority P3 is appropriate - this doesn't break functionality but will cause operational issues (cost, performance) when running at scale with tracing enabled.

## Closure Report (2026-02-01)

**Status:** CLOSED (IMPLEMENTED)

### Fix Summary

- Run/row spans now use stable names; IDs remain attributes.

### Test Coverage

- `tests/engine/test_spans.py::TestSpanFactory::test_span_names_stable`
