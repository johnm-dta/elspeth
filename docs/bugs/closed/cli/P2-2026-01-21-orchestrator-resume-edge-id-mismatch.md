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

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 3

**Current Code Analysis:**

The bug is confirmed to still exist in the current codebase. The synthetic edge ID pattern is present at `/home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py:1413-1418`:

```python
# Build edge_map from graph edges
edge_map: dict[tuple[str, str], str] = {}
for i, edge_info in enumerate(graph.get_edges()):
    # Generate synthetic edge_id for resume (edges were registered in original run)
    edge_id = f"resume_edge_{i}"
    edge_map[(edge_info.from_node, edge_info.label)] = edge_id
```

In contrast, the normal run path at lines 686-699 properly registers edges and stores real edge IDs:

```python
for edge_info in graph.get_edges():
    edge = recorder.register_edge(
        run_id=run_id,
        from_node_id=edge_info.from_node,
        to_node_id=edge_info.to_node,
        label=edge_info.label,
        mode=edge_info.mode,
    )
    edge_map[(edge_info.from_node, edge_info.label)] = edge.edge_id
```

**Triggering Conditions Confirmed:**

1. Config gates ARE executed during resume (orchestrator.py:1517, processor.py:872-885)
2. Config gates DO record routing events via `GateExecutor._record_routing()` (executors.py:581, 610, 630)
3. `_record_routing()` uses `self._edge_map` to look up edge IDs (executors.py:682, 696)
4. Routing events insert edge_id with FK constraint to edges.edge_id (schema.py:235)

**FK Constraint Verification:**

```python
# From schema.py:230-236
routing_events_table = Table(
    "routing_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
    Column("edge_id", String(64), ForeignKey("edges.edge_id"), nullable=False),  # <-- FK CONSTRAINT
    ...
)
```

**Git History:**

The synthetic edge_id pattern was introduced in commit `b2a3518` ("fix(sources,resume): comprehensive data handling bug fixes") and has not been modified since. No subsequent commits have addressed this issue.

**Fix Infrastructure Available:**

The LandscapeRecorder already has a method to load edges by run_id:
- `recorder.get_edges(run_id)` at recorder.py:689-713
- Returns list of Edge models with edge_id, from_node_id, to_node_id, label

The resume path could call this method and rebuild edge_map using real edge IDs instead of synthetic ones.

**Root Cause Confirmed:**

YES. The resume path creates synthetic edge IDs that do not exist in the `edges` table. When config gates execute during resume and attempt to record routing events, the FK constraint `routing_events.edge_id -> edges.edge_id` will fail with an integrity error.

**Recommendation:**

**Keep open.** This is a valid P2 bug that will cause resume to fail for any pipeline using config gates. The fix is straightforward:
1. Call `recorder.get_edges(run_id)` in `_process_resumed_rows()`
2. Build `edge_map` from returned edges using `(from_node_id, label) -> edge_id`
3. This mirrors the normal run path but reads from DB instead of registering new edges
