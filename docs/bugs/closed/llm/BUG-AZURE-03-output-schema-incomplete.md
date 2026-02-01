# Bug Report: AzureLLMTransform output_schema Omits LLM Response Fields

## Summary

- `output_schema` doesn't include actual LLM response fields (model, usage, finish_reason), causing schema validation mismatches.

## Status

- **RESOLVED** - 2026-01-30
- Commit: `05c2120`
- Branch: `fix/P2-aggregation-metadata-hardcoded`

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: Branch Bug Scan
- Date: 2026-01-25
- Related run/issue ID: BUG-AZURE-03

## Evidence

- `src/elspeth/plugins/llm/azure.py` - Schema missing response metadata fields

## Impact

- Schema completeness: Downstream transforms can't rely on schema contract

## Proposed Fix

- Add LLM response fields to output_schema:
  ```python
  output_schema: ClassVar[type[PluginSchema]] = LLMResponseSchema
  # Include: model, usage_tokens, finish_reason, etc.
  ```

## Acceptance Criteria

- output_schema includes all emitted fields

## Tests

- New tests required: yes, schema validation test

## Resolution

All 6 LLM transforms now declare `_output_schema_config` with proper field categorization:

**Guaranteed fields** (stable API contract):
- `{response_field}` - LLM response content
- `{response_field}_usage` - Token counts for cost/quota
- `{response_field}_model` - Model identifier

**Audit fields** (provenance, may evolve):
- `{response_field}_template_hash` - Prompt fingerprint
- `{response_field}_variables_hash` - Rendered variables fingerprint
- `{response_field}_template_source` - Config file path
- `{response_field}_lookup_hash` - Lookup data fingerprint
- `{response_field}_lookup_source` - Config file path
- `{response_field}_system_prompt_source` - Config file path

Changes:
- Added `audit_fields` attribute to `SchemaConfig`
- DAG construction extracts `_output_schema_config` from transforms
- Added helper functions in `plugins/llm/__init__.py`
- 71 new tests across 3 test files
