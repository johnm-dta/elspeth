# Bug Report: Gate routes with multiple labels to same sink break routing

## Summary
- ExecutionGraph uses `networkx.DiGraph`, which cannot hold multiple edges between the same nodes, so if a gate maps multiple route labels to the same sink, only one label survives and the other raises `MissingEdgeError` at runtime.

## Severity
- Severity: major
- Priority: P1

## Reporter
- Name or handle: Codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment
- Commit/branch: 5c27593 (local)
- OS: Linux (dev env)
- Python version: 3.11+ (per project)
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if applicable)
- Goal or task prompt: N/A
- Model/version: N/A
- Tooling and permissions (sandbox/approvals): N/A
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: N/A

## Steps To Reproduce
1. Configure a gate with routes mapping two labels to the same sink (e.g., `{"high": "alerts", "medium": "alerts"}`).
2. Build the execution graph and run a row that returns `RoutingAction.route("high")`.
3. Observe `MissingEdgeError` because the edge label for `high` was overwritten.

## Expected Behavior
- All route labels configured for a gate should be valid, even if they map to the same sink.

## Actual Behavior
- The second edge overwrites the first in `DiGraph`, so only one label is registered; the other label causes `MissingEdgeError`.

## Evidence
- Graph is a `networkx.DiGraph` (no parallel edges): `src/elspeth/core/dag.py:29`.
- Gate route edges are added per label: `src/elspeth/core/dag.py:299`.
- Routing lookup requires edge by label: `src/elspeth/engine/executors.py:326`.

## Impact
- User-facing impact: Valid routing configs that map multiple labels to the same sink crash at runtime.
- Data integrity / security impact: Audit trail cannot record routing decisions for missing edges.
- Performance or cost impact: Run aborts early; no output produced.

## Root Cause Hypothesis
- `DiGraph` does not support multiple edges between the same nodes, so route labels collide.

## Proposed Fix
- Code changes (modules/files):
  - `src/elspeth/core/dag.py`: switch to `networkx.MultiDiGraph` and include label as edge key, or validate config to forbid multiple labels mapping to the same sink.
  - `src/elspeth/engine/orchestrator.py` / `src/elspeth/engine/executors.py`: update edge registration/lookup to include edge keys if MultiDiGraph is used.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test where two labels map to the same sink and ensure routing does not raise `MissingEdgeError`.
- Risks or migration steps:
  - Changing graph type may require updating any code that assumes single edges per node pair.

## Architectural Deviations
- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:240` (DAG edges are labeled for routing decisions).
- Observed divergence: labeled edges collide when mapping multiple labels to the same sink.
- Reason (if known): underlying graph type does not support parallel edges.
- Alignment plan or decision needed: decide whether to support multiple labels per sink or forbid in validation.

## Acceptance Criteria
- Gate routes with multiple labels to the same sink are supported or explicitly rejected with a clear validation error.

## Tests
- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -k gate`
- New tests required: yes (duplicate label to same sink).

## Notes / Links
- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Resolution

**Fixed in:** 2026-01-18
**Fix:** Migrated `ExecutionGraph` from `networkx.DiGraph` to `networkx.MultiDiGraph`, using edge labels as keys to allow multiple edges between the same node pair.

**Commits:**
- feat(dag): migrate ExecutionGraph from DiGraph to MultiDiGraph
- test(dag): add multi-edge scenario integration tests
