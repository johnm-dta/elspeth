# Bug Report: Gate “continue” decisions are not recorded in routing_events (audit gap)

## Summary

- When a gate decides to “continue,” the engine records no routing event.
- Because node_states do not store action/decision payloads directly (they store hashes and error/context metadata), the audit trail may not capture which route label/branch decision occurred for “continue” outcomes, especially for config gates where the condition result matters.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: pipelines using config gates (routes with `continue`)
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `GateExecutor` and `ExecutionGraph`

## Steps To Reproduce

1. Configure a config gate with routes where one branch is `continue`, e.g.:
   - `routes: {true: flagged_sink, false: continue}`
2. Run a pipeline where some rows evaluate the condition to `false`.
3. Inspect `routing_events` for those gate states.

## Expected Behavior

- Every gate decision should be auditable:
  - if the decision is to continue, record a routing event against the gate’s `continue` edge (and include the evaluated label/reason in the routing event’s reason payload).

## Actual Behavior

- No routing event is recorded for continue outcomes; “continue” is inferred from the absence of an event.

## Evidence

- Plugin gate continue is a no-op (no routing recorded):
  - `src/elspeth/engine/executors.py:386`
- Config gate destination `continue` also records no routing:
  - `src/elspeth/engine/executors.py:559`
- The graph has an explicit continue edge between pipeline stages (including gates):
  - `src/elspeth/core/dag.py:368` (gate is wired with an outgoing `continue` edge)
- `RoutingAction.continue_()` has `destinations=()` so `_record_routing` cannot record anything for continue:
  - `src/elspeth/contracts/routing.py:49`
  - `src/elspeth/engine/executors.py:660`

## Impact

- User-facing impact: `explain()` output and exported audit trail may not show the gate’s decision path for continue cases.
- Data integrity / security impact: audit relies on inference (“no routing event means continue”), which is weaker than explicit recording, especially if multiple continue-like branches exist or config evolves.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- The routing event recorder is keyed off `RoutingAction.destinations`, and `continue_()` intentionally has no destinations; the executor treats continue as “no event.”

## Proposed Fix

- Code changes (modules/files):
  - Record a routing event for continue outcomes against the `(node_id, "continue")` edge:
    - For config gates: include `{"condition": ..., "result": <route_label>}` in `reason`.
    - For plugin gates: include `action.reason` if provided.
  - Options:
    - (A) Change `RoutingAction.continue_()` to use `destinations=("continue",)` so `_record_routing` can handle it.
    - (B) Add a special-case branch in `GateExecutor` for `RoutingKind.CONTINUE` that looks up the `continue` edge explicitly and records an event.
- Config or schema changes: none.
- Tests to add/update:
  - Add test asserting a continue outcome produces a routing_event with edge label `continue` and includes the evaluation reason for config gates.
- Risks or migration steps:
  - Ensure graph validation guarantees a `continue` edge exists for gates in all supported graphs.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“No inference” standard for audit trail)
- Observed divergence: continue decisions are inferred rather than recorded.
- Alignment plan or decision needed: decide whether “continue via missing event” is acceptable; if not, implement explicit recording.

## Acceptance Criteria

- For gate evaluations that result in continue, at least one routing_event is recorded and includes the evaluation reason.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_config_gates.py`
  - `pytest tests/engine/test_executors.py -k gate`
- New tests required: yes

## Notes / Links

- Related issues/PRs: `docs/bugs/pending/2026-01-15-routing-copy-ignored.md` (routing semantics)
- Related design docs: `docs/design/architecture.md`
