# Bug Report: schema_validator assumes a linear pipeline and ignores DAG routing

## Summary

- `validate_pipeline_schemas` validates a linear chain and then checks all sinks against the *final* transform output.
- In real pipelines, gates can route to sinks from intermediate nodes, and DAG branches can feed different sinks.
- The validator can therefore accept incompatible sink schemas or report incorrect errors, depending on routing.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any pipeline using gates or branching

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/schema_validator.py`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Define a pipeline with a gate that routes some rows directly to sink `raw`.
2. Place a transform after the gate that adds a required field (e.g., `score`).
3. Configure sink `raw` to require `score`.
4. Run schema validation (or the pipeline).

## Expected Behavior

- Validation fails because the gate routes rows to `raw` *before* the transform adds `score`.

## Actual Behavior

- Validation passes because sinks are checked against the final transform output, not against the actual upstream node for each sink.

## Evidence

- Validator only uses linear lists and treats the last transform output as the producer for all sinks: `src/elspeth/engine/schema_validator.py:55-76`
- Orchestrator does not pass the execution graph to the validator, so routing edges are ignored: `src/elspeth/engine/orchestrator.py:408-437`
- Contract expects compatibility between connected nodes (not just linear order): `docs/contracts/plugin-protocol.md:1352-1356`

## Impact

- User-facing impact: incompatible sink schemas can pass validation and fail during execution.
- Data integrity / security impact: schema compatibility claims are incorrect for DAG pipelines; audit metadata becomes misleading.
- Performance or cost impact: reruns and manual debugging.

## Root Cause Hypothesis

- Schema validation is implemented for linear pipelines only and is disconnected from the execution graph.

## Proposed Fix

- Code changes (modules/files):
  - Extend schema validation to accept the `ExecutionGraph` and validate schema compatibility per edge.
  - For sinks, validate against the actual upstream node(s) that route to each sink.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test with a gate routing to a sink mid-pipeline; ensure validation fails when schemas are incompatible.
- Risks or migration steps:
  - Requires graph-aware validation; may surface new validation errors in existing DAG pipelines.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (compatibility between connected nodes)
- Observed divergence: validation ignores routing edges in the execution graph.
- Reason (if known): validator signature only accepts linear schema lists.
- Alignment plan or decision needed: define validation rules for gate routes and forked paths.

## Acceptance Criteria

- Schema validation evaluates compatibility along actual graph edges, including gate routes to sinks.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
