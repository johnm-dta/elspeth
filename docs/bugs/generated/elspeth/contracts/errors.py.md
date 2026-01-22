# Bug Report: RoutingReason contract out of sync with GateExecutor reason payload

## Summary

- `RoutingReason` requires `rule` and `matched_value`, but GateExecutor emits routing reasons with `condition` and `result`, so the contract does not describe actual audit payloads and typed consumers will be misled.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/contracts/errors.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Read `src/elspeth/contracts/errors.py`, `src/elspeth/engine/executors.py` with `sed`/`cat`

## Steps To Reproduce

1. Open `src/elspeth/contracts/errors.py` and note `RoutingReason` requires `rule` and `matched_value`.
2. Open `src/elspeth/engine/executors.py` and observe GateExecutor sets `reason = {"condition": ..., "result": ...}` before recording routing events.
3. Run any config-driven gate or inspect the emitted routing reason payload; it contains `condition`/`result` rather than the contract’s required keys.

## Expected Behavior

- Routing reason payloads conform to `RoutingReason` (or the contract is updated to match emitted keys).

## Actual Behavior

- Routing reason payloads emitted by GateExecutor use `condition`/`result` and omit required `rule`/`matched_value` fields.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Any gate config; inspect reason payload in GateExecutor
- Code references: `src/elspeth/contracts/errors.py:27`, `src/elspeth/contracts/errors.py:28`, `src/elspeth/engine/executors.py:575`

## Impact

- User-facing impact: Routing explanations are inconsistent with the documented contract; tooling expecting `rule`/`matched_value` will show missing data.
- Data integrity / security impact: Audit reason payload schema is inconsistent with stated contract, weakening traceability guarantees.
- Performance or cost impact: None observed.

## Root Cause Hypothesis

- `RoutingReason` schema in `src/elspeth/contracts/errors.py` is stale/out of sync with GateExecutor’s emitted reason payload.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/contracts/errors.py` to reflect actual routing reason keys (`condition`, `result`) or make the schema permissive; update `tests/contracts/test_errors.py` to match. If the contract is intended to be authoritative, standardize GateExecutor to emit `rule`/`matched_value`.
- Config or schema changes: None
- Tests to add/update: Update contract tests to reflect the chosen schema; optional integration test verifying gate reason payload shape.
- Risks or migration steps: Contract change may affect downstream type checking or documentation; communicate the updated schema.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Runtime routing reasons use `condition`/`result` instead of contract-required `rule`/`matched_value`.
- Reason (if known): Contract not updated when gate reason payload format changed.
- Alignment plan or decision needed: Align `RoutingReason` schema with emitted payloads or adjust emitter to match schema.

## Acceptance Criteria

- `RoutingReason` schema matches emitted routing reason payload keys (or GateExecutor emits keys required by the schema).
- Updated contract tests pass.

## Tests

- Suggested tests to run: `python -m pytest tests/contracts/test_errors.py`
- New tests required: Update existing contract tests to reflect the agreed schema; optional integration test for gate reason payloads.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
---
# Bug Report: TransformReason contract out of sync with TransformResult.error payloads

## Summary

- `TransformReason` requires `action` (and optional `fields_modified`/`validation_errors`), but transforms emit `TransformResult.error()` reasons with keys like `message`, `reason`, and `error`, so the contract does not reflect actual error payloads.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/contracts/errors.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Read `src/elspeth/contracts/errors.py`, `src/elspeth/plugins/transforms/field_mapper.py`, `src/elspeth/plugins/llm/base.py` with `sed`/`cat`

## Steps To Reproduce

1. Open `src/elspeth/contracts/errors.py` and note `TransformReason` requires `action`.
2. Open `src/elspeth/plugins/transforms/field_mapper.py` and see `TransformResult.error({"message": ...})`.
3. Open `src/elspeth/plugins/llm/base.py` and see `TransformResult.error({"reason": ..., "error": ...})`.

## Expected Behavior

- Transform error reason payloads conform to `TransformReason` (or the contract is updated to match emitted keys).

## Actual Behavior

- Transform error reason payloads use `message`/`reason`/`error` keys and do not include required `action`.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Any transform that returns `TransformResult.error()` with `message` or `reason` keys
- Code references: `src/elspeth/contracts/errors.py:40`, `src/elspeth/plugins/transforms/field_mapper.py:114`, `src/elspeth/plugins/llm/base.py:220`

## Impact

- User-facing impact: Error reasons are inconsistent with the documented contract; consumers expecting `action` will misparse or drop details.
- Data integrity / security impact: Audit reason payload schema is inconsistent with stated contract, reducing interpretability.
- Performance or cost impact: None observed.

## Root Cause Hypothesis

- `TransformReason` schema in `src/elspeth/contracts/errors.py` is outdated relative to the actual shapes used by transforms.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/contracts/errors.py` to allow the actual error reason keys (`message`, `reason`, `error`, etc.) or make the schema permissive; update `tests/contracts/test_errors.py` accordingly. If the contract is authoritative, standardize transform error payloads to include `action`.
- Config or schema changes: None
- Tests to add/update: Update contract tests to reflect the chosen schema; optional integration test verifying transform error reason payload shape.
- Risks or migration steps: Contract change may affect downstream type checking or documentation; communicate the updated schema.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Transform error payloads omit contract-required `action` and use other keys instead.
- Reason (if known): Contract not updated as transform error payloads evolved.
- Alignment plan or decision needed: Align `TransformReason` schema with emitted payloads or enforce schema in emitters.

## Acceptance Criteria

- `TransformReason` schema matches emitted TransformResult.error payload keys (or emitters conform to the schema).
- Updated contract tests pass.

## Tests

- Suggested tests to run: `python -m pytest tests/contracts/test_errors.py`
- New tests required: Update existing contract tests to reflect the agreed schema; optional integration test for transform error payloads.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
