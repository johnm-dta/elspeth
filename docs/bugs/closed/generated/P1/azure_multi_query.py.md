# Bug Report: Batch Mode Uses Synthetic state_id, Breaking Call Audit FK

## Summary

- Batch processing fabricates per-row `state_id` values and uses them for audited LLM calls, but the engine only creates a single `node_state` per batch, so call recording violates the `calls.state_id -> node_states.state_id` foreign key and can crash or drop audit data.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0 (fix/rc1-bug-burndown-session-4)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any aggregation flush using `azure_multi_query_llm` with Landscape enabled

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation node that flushes to `azure_multi_query_llm` with Landscape enabled (batch-aware transform).
2. Run a pipeline so the transform receives a list of rows and `_process_batch()` executes.

## Expected Behavior

- All external LLM calls are recorded under the batch node’s single `state_id` and the run completes with a consistent audit trail.

## Actual Behavior

- The transform constructs per-row synthetic `state_id`s (e.g., `state-abc_row0`) and uses them in `AuditedLLMClient`, which violates the `calls.state_id` foreign key and can raise an IntegrityError or leave calls unrecorded.

## Evidence

- Per-row synthetic IDs created in batch path: `src/elspeth/plugins/llm/azure_multi_query.py:494-505`
- LLM client uses the provided `state_id` for audit recording: `src/elspeth/plugins/llm/azure_multi_query.py:207-217`
- Engine only creates one `node_state` for the batch flush and assigns it to `ctx.state_id`: `src/elspeth/engine/executors.py:920-935`
- `calls.state_id` is a foreign key to `node_states.state_id`: `src/elspeth/core/landscape/schema.py:192-197`

## Impact

- User-facing impact: Batch runs can crash or abort when recording external calls.
- Data integrity / security impact: External call audit trail can be missing or invalid, violating auditability requirements.
- Performance or cost impact: Retries or failed runs increase compute and API costs.

## Root Cause Hypothesis

- `_process_batch()` invents per-row `state_id`s instead of using the batch node’s `state_id`, so call records reference non-existent node states.

## Proposed Fix

- Code changes (modules/files):
  - Use `ctx.state_id` for all batch rows in `src/elspeth/plugins/llm/azure_multi_query.py`, and reuse a single `AuditedLLMClient` per batch so `call_index` remains unique.
  - Remove per-row `row_state_id` usage and adjust client cache cleanup accordingly.
- Config or schema changes: None.
- Tests to add/update:
  - Add an integration test that flushes a batch to `azure_multi_query_llm` with a real Landscape DB and asserts calls are recorded under the batch node state.
- Risks or migration steps:
  - Ensure `call_index` uniqueness across all calls in the batch (shared client or shared counter).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:23-28`, `src/elspeth/core/landscape/schema.py:192-197`
- Observed divergence: Calls are recorded with state IDs that are not valid `node_states`, violating audit trail constraints.
- Reason (if known): Batch implementation uses per-row synthetic state IDs.
- Alignment plan or decision needed: Record all batch calls under the batch node’s `state_id` and preserve uniqueness via shared call indices.

## Acceptance Criteria

- Batch-mode LLM calls are recorded under the batch node’s `state_id` without foreign key violations.
- No IntegrityError is raised during batch LLM call recording.
- Audit trail shows all external calls for the batch node state.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -k batch -v`
- New tests required: yes, integration test covering batch + Landscape call recording

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
---
# Bug Report: response_format Config Ignored in LLM Calls

## Summary

- The transform captures `response_format` from config but never passes it to the LLM API, so JSON mode is not enforced even when configured.

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0 (fix/rc1-bug-burndown-session-4)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any `azure_multi_query_llm` run with `response_format` set (e.g., `examples/multi_query_assessment/suite.yaml`)

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure `azure_multi_query_llm` with `response_format: json`.
2. Run a query and inspect the LLM request arguments (or observe non-JSON outputs).

## Expected Behavior

- `response_format` is forwarded to the LLM API so JSON responses are enforced.

## Actual Behavior

- The LLM call omits `response_format`, so output format is uncontrolled despite configuration.

## Evidence

- Config field captured in init: `src/elspeth/plugins/llm/azure_multi_query.py:101-103`
- LLM call omits response_format: `src/elspeth/plugins/llm/azure_multi_query.py:211-217`
- `response_format` defined in config model: `src/elspeth/plugins/llm/multi_query.py:155-158`
- Example config uses `response_format: json`: `examples/multi_query_assessment/suite.yaml:113`

## Impact

- User-facing impact: JSON enforcement doesn’t occur, leading to parse failures or inconsistent behavior.
- Data integrity / security impact: Increased risk of unparseable or malformed outputs entering the pipeline.
- Performance or cost impact: Additional retries or failures due to parsing errors.

## Root Cause Hypothesis

- `_response_format` is stored but never included in `chat_completion()` call parameters.

## Proposed Fix

- Code changes (modules/files):
  - Pass `response_format` through to `llm_client.chat_completion()` in `src/elspeth/plugins/llm/azure_multi_query.py`, mapping string values to provider-specific formats as needed.
- Config or schema changes: Validate allowed `response_format` values (e.g., `json`).
- Tests to add/update:
  - Add a test that asserts `chat_completion()` is called with the configured response format.
- Risks or migration steps:
  - Ensure mapping aligns with Azure OpenAI SDK expectations (e.g., `{"type": "json_object"}`).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/llm/multi_query.py:130-158`, `examples/multi_query_assessment/suite.yaml:113`
- Observed divergence: Configured `response_format` is ignored during LLM calls.
- Reason (if known): Missing parameter propagation in `_process_single_query()`.
- Alignment plan or decision needed: Forward `response_format` to the LLM API and cover with tests.

## Acceptance Criteria

- LLM calls include `response_format` when configured.
- JSON mode is enforced (or errors are explicit if unsupported).
- Tests validate the parameter is propagated.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -v`
- New tests required: yes, parameter propagation assertion

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `examples/multi_query_assessment/suite.yaml`
