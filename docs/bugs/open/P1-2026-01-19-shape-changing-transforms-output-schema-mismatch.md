# Bug Report: Shape-changing transforms declare `output_schema` identical to `input_schema`, making schema validation and audit metadata unreliable

## Summary

- The orchestrator validates pipeline schema compatibility using each plugin’s declared `input_schema`/`output_schema`.
- Several built-in transforms change the row shape but still set `output_schema = input_schema` (single schema from config), creating a mismatch between declared schema and actual output:
  - `FieldMapper` can delete/rename/select fields.
  - `JSONExplode` removes the array field and adds `output_field`/`item_index`.
  - `BatchStats` outputs `{count,sum,mean,...}` rather than the incoming row shape.
- This can produce false positives (schema validation passes but runtime output lacks expected fields) and false negatives (schema validation fails even though transform output would be compatible).

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

Example for `JSONExplode` (similar applies to others):

1. Configure `json_explode` with an explicit schema that requires the array field `items`.
2. Configure a downstream sink/transform that expects `items` to exist (schema requires it).
3. Run the pipeline:
   - schema validation may pass because `output_schema` incorrectly still includes `items`
   - runtime output rows do *not* include `items`, so downstream code can fail unexpectedly

## Expected Behavior

- Each transform’s `output_schema` should describe the row shape it actually emits so that:
  - `validate_pipeline_schemas(...)` correctly accepts/rejects pipelines
  - node schema metadata in Landscape is meaningful for audit/repro

## Actual Behavior

- Built-in transforms that mutate row shape set `output_schema` to the same schema as `input_schema`, despite emitting different keys.

## Evidence

- Orchestrator validates schemas directly from plugin attributes: `src/elspeth/engine/orchestrator.py:323-339`
- `FieldMapper` sets `input_schema` and `output_schema` to the same generated schema but can remove/rename fields: `src/elspeth/plugins/transforms/field_mapper.py:64-115`
- `JSONExplode` sets `input_schema` and `output_schema` to the same generated schema but removes the array field and adds new fields: `src/elspeth/plugins/transforms/json_explode.py:102-151`
- `BatchStats` sets `input_schema` and `output_schema` to the same schema but returns a stats dict: `src/elspeth/plugins/transforms/batch_stats.py:84-147`

## Impact

- User-facing impact: confusing schema errors (or missing schema errors) when building pipelines; downstream crashes that schema validation should have caught.
- Data integrity / security impact: schema metadata in audit trail can be misleading (records indicate one schema while outputs differ).
- Performance or cost impact: wasted debugging time; potential reruns.

## Root Cause Hypothesis

- Transform configuration only supports a single `schema` config, so plugins use it for both input and output even when they change row shape.

## Proposed Fix

- Code changes (modules/files):
  - Add first-class support for distinct `input_schema` and `output_schema` (or `schema_in` / `schema_out`) in transform configs.
  - Update built-in transforms that change shape to declare correct output schemas.
  - Alternatively (minimal): require `schema: {fields: dynamic}` for any transform that changes row shape and enforce this at config validation time.
- Config or schema changes:
  - Extend `TransformDataConfig` to accept separate schema configs (input vs output).
- Tests to add/update:
  - Add tests ensuring schema validation fails when a transform declares an output schema missing downstream-required fields.
  - Add tests that shape-changing transforms either (a) require dynamic schema, or (b) correctly validate via separate schemas.
- Risks or migration steps:
  - Introducing separate schemas is a breaking config change; provide migration path or backwards-compatible defaults.

## Architectural Deviations

- Spec or doc reference: `docs/design/architecture.md` (schema compatibility and audit expectations); `CLAUDE.md` (auditability and “no inference”)
- Observed divergence: declared schemas do not match actual produced data for built-in transforms.
- Reason (if known): single-schema config simplification.
- Alignment plan or decision needed: decide whether schema configs are mandatory and bidirectional (input/output) for all transforms.

## Acceptance Criteria

- For each built-in transform, `output_schema` matches the shape of emitted row(s).
- Orchestrator schema validation produces consistent results with runtime behavior for pipelines using these transforms.

## Tests

- Suggested tests to run: `pytest tests/engine/test_schema_validator.py tests/plugins/transforms/`
- New tests required: yes

## Notes / Links

- Related issue: `docs/bugs/open/2026-01-19-json-explode-iterable-nonstrict-types.md` (JSONExplode type enforcement)
