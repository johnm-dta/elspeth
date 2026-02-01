# Bug Report: Config gate routing can silently mis-record routing events when edge labels collide (edge_map invariant)

## Summary

- The runtime records routing events by looking up an `edge_id` via `edge_map[(from_node_id, label)]`, which assumes **outgoing edge labels are unique per node**.
- Config gates always have a `"continue"` edge to the next node, but config gate route labels and `fork_to` branch names are not validated to avoid collisions (e.g., a route label `"continue"` or a branch name `"continue"`).
- If multiple edges share the same `(from_node_id, label)` (possible in `MultiDiGraph` when `to_node` differs), `edge_map` is overwritten during registration and routing events can point at the wrong edge ID (audit corruption) without raising.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `25468ac9550b481a55b81a05d84bbf2592e6430c`
- OS: Linux (Ubuntu 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A (static analysis)
- Data set or fixture: N/A (static analysis)

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystems, identify hotspots, write bug reports
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected config gate execution, DAG construction, orchestrator edge registration, and routing event recording

## Steps To Reproduce

1. Configure at least two config gates (so the first has an outgoing `"continue"` edge to the second), or a single config gate plus the default output sink edge.
2. In a config gate, create a label collision for outgoing edges from that gate node, e.g.:
   - **Route-label collision:** include a route label `"continue"` that routes to a sink (so the gate has both a `"continue"` edge to the next node and a `"continue"` edge to a sink).
   - **Fork-branch collision:** set `fork_to` to include `"continue"` (or any other label that already exists as an outgoing edge label from that gate node).
3. Run the pipeline and inspect `routing_events` for that gate decision.
4. Observe routing events may reference the wrong edge ID because `edge_map[(gate_node_id, "continue")]` is ambiguous and was overwritten at registration time.

## Expected Behavior

- The system fails fast at graph/config validation time if a gate node has ambiguous outgoing labels, so routing events cannot be mis-recorded.

## Actual Behavior

- Ambiguous outgoing edge labels can exist (same `from_node` + same `label` to different `to_node`s). `edge_map` is keyed only by `(from_node, label)`, so it can be overwritten during edge registration and routing events can be written against the wrong edge without error.

## Evidence

- Graph uses `MultiDiGraph` and uses `label` as the edge key, which allows the same `label` to exist on multiple outgoing edges from the same node as long as `to_node` differs: `src/elspeth/core/dag.py:40-46`, `src/elspeth/core/dag.py:92-110`
- Orchestrator builds `edge_map[(from_node, label)] = edge_id`, dropping `to_node` and overwriting on duplicates: `src/elspeth/engine/orchestrator.py:474-488`
- Config gate executor records routing events by looking up `edge_map[(node_id, dest_label)]`:
  - For sink routes, `dest_label` is the route label: `src/elspeth/engine/executors.py:588-600`, `src/elspeth/contracts/routing.py:59-82`
  - For forks, `dest_label` is each `fork_to` branch value: `src/elspeth/engine/executors.py:549-587`, `src/elspeth/contracts/routing.py:85-97`
- Config validation does not prevent route labels or `fork_to` values from colliding with `"continue"` or with each other: `src/elspeth/core/config.py:171-226`
- Graph builder always adds a `"continue"` edge between sequential nodes, including between config gates: `src/elspeth/core/dag.py:346-348`

## Impact

- User-facing impact: difficult-to-debug `explain()`/lineage behavior because routing events may point to the wrong destination edge even when runtime outputs look correct.
- Data integrity / security impact: audit trail can become internally inconsistent (routing decision recorded against the wrong edge), violating ELSPETH’s “no inference” auditability standard.
- Performance or cost impact: investigations and re-runs to re-establish a reliable audit trail.

## Root Cause Hypothesis

- `edge_map` keying drops `to_node`, so it cannot represent multiple edges with the same label from the same node. The config/graph layer does not enforce the uniqueness invariant required by the recorder (`(from_node, label)` must be unique).

## Proposed Fix

- Code changes (modules/files):
-  Option A (recommended): validate the edge-label uniqueness invariant at graph build time:
   - for each node, ensure outgoing edge labels are unique (no duplicate `(from_node, label)` across different `to_node`s)
   - explicitly forbid route labels and `fork_to` values of `"continue"` (reserved for linear flow)
   - validate `fork_to` is unique and does not intersect with sink-route labels for the same gate
-  Option B: make `edge_map` key unambiguous by including `to_node` (or use the NetworkX edge key), and update routing actions / recorders to resolve to a specific edge deterministically.
- Config or schema changes: none (unless choosing Option B and changing routing event contract).
- Tests to add/update:
  - Add a validation test that rejects a config gate with a route label `"continue"` that targets a sink.
  - Add a validation test that rejects a config gate with `fork_to` containing `"continue"` (or any duplicate outgoing label).
- Risks or migration steps:
  - Existing configs using `"continue"` as a semantic label will need to be renamed (but that is preferable to ambiguous audit recording).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (audit trail is the source of truth; “No inference — if it’s not recorded, it didn’t happen”)
- Observed divergence: routing events can be recorded against the wrong edge due to ambiguous edge labeling, breaking the “audit trail is the source of truth” guarantee.
- Reason (if known): graph allows label collisions; edge_map assumes uniqueness but does not validate it.
- Alignment plan or decision needed: define reserved labels and enforce “unique outgoing label per node” as a hard validation rule.

## Acceptance Criteria

- Graph/config validation fails fast when a gate node has duplicate outgoing labels (including collisions with `"continue"`).
- For valid configs, routing events always reference the correct edge ID for the chosen destination.

## Tests

- Suggested tests to run: `pytest tests/`
- New tests required: yes (config gate label collision validation)

## Notes / Links

- This issue is about audit correctness (wrong edge IDs), not just runtime routing behavior.
