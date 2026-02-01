# Bug Report: Schema Config Errors Escape Validator Instead of Returning Structured Errors

## Summary

- Invalid schema configs raise `PluginConfigError` instead of returning structured `ValidationError` lists, breaking the validator contract and manager error formatting.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2025-02-14
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Invalid schema config dict for a source/transform/sink (e.g., mode "invalid_mode")

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of /home/john/elspeth-rapid/src/elspeth/plugins/validation.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Call `PluginConfigValidator().validate_source_config("csv", {"path":"/tmp/test.csv","schema":{"mode":"invalid_mode","fields":["id: int"]},"on_validation_failure":"quarantine"})`.
2. Observe that a `PluginConfigError` is raised instead of receiving a list of `ValidationError`.

## Expected Behavior

- Validation returns a list of structured `ValidationError` objects for schema errors so callers can format messages and proceed consistently.

## Actual Behavior

- `PluginConfigError` escapes from `validate_source_config` (and similar methods), bypassing the structured error path.

## Evidence

- `src/elspeth/plugins/validation.py:72` only unwraps `PydanticValidationError` and re-raises other exceptions, so `PluginConfigError` caused by schema parsing propagates.
- `src/elspeth/plugins/config_base.py:61` calls `SchemaConfig.from_dict` and wraps `ValueError` into `PluginConfigError`, which is not handled by the validator.
- `src/elspeth/contracts/schema.py:195` shows invalid schema configs raise `ValueError`, which becomes the `PluginConfigError.__cause__`.
- `src/elspeth/plugins/manager.py:319` assumes validation returns an error list and only formats/raises `ValueError`, so `PluginConfigError` bypasses expected error handling.

## Impact

- User-facing impact: Invalid schema configs produce raw exceptions/stack traces instead of clear, field-level validation messages.
- Data integrity / security impact: None observed.
- Performance or cost impact: None observed.

## Root Cause Hypothesis

- `PluginConfigValidator` only converts `PydanticValidationError` (or its cause) to `ValidationError` lists and does not handle `PluginConfigError` with `ValueError` cause from `SchemaConfig.from_dict`.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/plugins/validation.py` to catch `PluginConfigError` (or `Exception` with `ValueError` cause) and convert schema-related errors into `ValidationError` entries (likely field `"schema"`).
- Config or schema changes: None.
- Tests to add/update:
  - Add a unit test to `tests/plugins/test_validation.py` that uses invalid `schema` inside a source/transform/sink config and asserts `validate_*_config` returns errors instead of raising.
- Risks or migration steps:
  - Low risk; only changes error handling/formatting for invalid config inputs.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/validation.py:6`
- Observed divergence: Validator contract says it returns structured errors, but schema parsing errors propagate as exceptions.
- Reason (if known): Missing handling of `PluginConfigError` with `ValueError` cause.
- Alignment plan or decision needed: Convert `PluginConfigError` (schema parse failures) to `ValidationError` list and keep non-validation exceptions as crashes.

## Acceptance Criteria

- Invalid schema inside a plugin config returns a non-empty `ValidationError` list from `validate_source_config`/`validate_transform_config`/`validate_sink_config`.
- `PluginManager.create_*` formats those errors into `ValueError` consistently, without leaking `PluginConfigError`.

## Tests

- Suggested tests to run: `pytest tests/plugins/test_validation.py -k schema`
- New tests required: yes, add coverage for invalid schema inside plugin configs.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-25-validation-subsystem-extraction.md`
