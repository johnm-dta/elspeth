# Enhancement: Integrate pooling metadata into audit trail context_after_json

## Summary

The P2 and P3 pooling bugs have been fixed - ordering metadata and pool stats are now **available** from PooledExecutor but not yet **persisted** to the audit trail. This enhancement completes the audit trail integration.

## Severity

- Severity: minor (enhancement)
- Priority: P3

## Reporter

- Name or handle: Claude
- Date: 2026-02-02
- Related run/issue ID: elspeth-rapid-7wo

## Background

Two bugs were fixed on 2026-02-02:

### P2-2026-01-21 (Fixed)
`execute_batch()` now returns `list[BufferEntry[TransformResult]]` with ordering metadata:
- `submit_index`: Order row was submitted (0-indexed)
- `complete_index`: Order row completed (may differ from submit)
- `submit_timestamp`: time.perf_counter() when submitted
- `complete_timestamp`: time.perf_counter() when completed
- `buffer_wait_ms`: Time spent waiting in buffer after completion

### P3-2026-01-21 (Fixed)
`get_stats()` now includes:
- `pool_stats.max_concurrent_reached`: Peak concurrent workers during batch
- `pool_config.dispatch_delay_at_completion_ms`: Throttle delay at batch completion

## Current State

The metadata is computed and exposed via the API, but no code passes it to the Landscape recorder for persistence in `context_after_json`.

## Expected Behavior

- Pool stats and ordering metadata should be persisted to the audit trail
- Auditors should be able to verify reordering worked correctly
- Auditors should be able to identify "lost" rows by examining ordering gaps

## Proposed Implementation

### Code Changes

1. **LLM Transforms** (`azure_multi_query.py`, `openrouter_multi_query.py`):
   - Call `executor.get_stats()` after batch completion
   - Pass stats to recorder when creating/completing node state

2. **Recorder Integration**:
   - Include pool stats in `context_after_json` parameter
   - Decide on per-row ordering metadata storage location

### Design Decision Needed

Where to store per-row ordering metadata:

| Option | Location | Pros | Cons |
|--------|----------|------|------|
| A | `node_state.context_after_json` | Simple, no schema change | Per-row, may inflate payload |
| B | `external_calls` table | Already tracks per-call data | May not fit semantically |
| C | New dedicated table | Clean separation | Schema migration required |

**Recommendation:** Option A (context_after_json) for initial implementation. The payload size increase is modest (6 fields per row), and it keeps all audit data in one place.

## Acceptance Criteria

1. LLM transforms call `executor.get_stats()` after batch completion
2. Pool stats are included in `context_after_json` when recording node state
3. Per-row ordering metadata (submit_index, complete_index, buffer_wait_ms) is recorded
4. Integration test verifies metadata reaches the audit trail
5. Metadata can be queried via Landscape MCP `explain_token()` or `get_node_states()`

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/ -k pool`
- New tests required: Integration test verifying end-to-end audit trail persistence

## References

- Design spec: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:152-175`
- Fixed P2 bug: `docs/bugs/closed/engine-pooling/P2-2026-01-21-pooling-ordering-metadata-dropped.md`
- Fixed P3 bug: `docs/bugs/closed/engine-pooling/P3-2026-01-21-pooling-missing-pool-stats.md`
- Beads issue: elspeth-rapid-7wo
