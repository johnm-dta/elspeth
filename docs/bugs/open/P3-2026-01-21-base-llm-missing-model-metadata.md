# Bug Report: Base LLM transform omits model metadata in output

## Summary

- BaseLLMTransform does not include the resolved response model in its output fields, unlike Azure/OpenRouter transforms, reducing audit traceability when a subclass relies on BaseLLMTransform behavior.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any BaseLLMTransform subclass

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Implement a simple subclass of BaseLLMTransform.
2. Run a row and inspect output fields.

## Expected Behavior

- Output should include the actual model used (e.g., `<response_field>_model`) for audit parity with other LLM transforms.

## Actual Behavior

- No model field is added in BaseLLMTransform output.

## Evidence

- Output fields are assembled without model metadata in `src/elspeth/plugins/llm/base.py:259`.

## Impact

- User-facing impact: missing model attribution in outputs.
- Data integrity / security impact: audit trail lacks model detail for BaseLLMTransform subclasses.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Base implementation predates model metadata convention used in Azure/OpenRouter.

## Proposed Fix

- Code changes (modules/files): add `output[f"{response_field}_model"] = response.model` in BaseLLMTransform.
- Config or schema changes: N/A
- Tests to add/update:
  - Add a unit test for BaseLLMTransform subclasses to assert model field presence.
- Risks or migration steps:
  - Ensure output schema allows the new field when strict schemas are used.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): auditability requires full external call attribution.
- Observed divergence: model not recorded in output row.
- Reason (if known): missing field in base implementation.
- Alignment plan or decision needed: align base output fields with Azure/OpenRouter patterns.

## Acceptance Criteria

- BaseLLMTransform outputs include `<response_field>_model`.

## Tests

- Suggested tests to run: N/A (no direct tests currently)
- New tests required: yes, base LLM output metadata test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
