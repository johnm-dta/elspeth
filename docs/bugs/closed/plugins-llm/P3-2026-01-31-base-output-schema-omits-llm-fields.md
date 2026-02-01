# Bug Report: BaseLLMTransform output_schema omits LLM-added fields

## Summary

- `BaseLLMTransform` sets `output_schema = input_schema` but `process()` adds fields like `_usage`, `_model`, `_template_hash` that aren't in the schema.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/llm/base.py:231-232` - `self.output_schema = schema` (same as input)
- Lines 329-341 - transform adds `_usage`, `_model`, etc.
- Output rows don't match declared schema

## Proposed Fix

- Generate output_schema including LLM-added fields

## Acceptance Criteria

- Output schema accurately reflects all emitted fields

## Resolution: INVALID (2026-02-02)

**Status: CLOSED - NOT A BUG**

### Investigation Findings

The bug report misunderstands ELSPETH's dual schema architecture:

1. **`output_schema` (Pydantic class)**: Used for type logging to audit trail and schema identity checks. NOT used for runtime row validation.

2. **`_output_schema_config` (SchemaConfig)**: Contains field contracts (guaranteed_fields, audit_fields) used for DAG validation. This IS correctly populated with LLM-added fields.

### Evidence

1. `BaseLLMTransform.__init__()` correctly builds `_output_schema_config` at lines 233-248 with:
   - `guaranteed_fields`: response field, `_usage`, `_model` suffixed fields
   - `audit_fields`: `_template_hash`, `_variables_hash`, `_lookup_hash`, etc.

2. DAG validation correctly reads `_output_schema_config`:
   - `dag.py:462`: `output_schema_config = getattr(transform, "_output_schema_config", None)`
   - `dag.py:1167-1168`: `node_info.output_schema_config` used for guaranteed field validation

3. The engine never validates rows against `output_schema`:
   - No `output_schema.model_validate()` calls exist in the engine
   - Only `input_schema.model_validate()` is used (in specific transforms)

### Why This Design Is Intentional

| Attribute | Purpose | Contains LLM fields? |
|-----------|---------|---------------------|
| `output_schema` | Type schema for audit logging | No (same as input) |
| `_output_schema_config` | Contract metadata for DAG validation | Yes |

The separation allows transforms to add fields at runtime while maintaining explicit contracts for compile-time DAG validation.

### Conclusion

No functional impact. Downstream transforms CAN correctly depend on LLM-added fields because DAG validation uses `_output_schema_config`, not `output_schema`.
