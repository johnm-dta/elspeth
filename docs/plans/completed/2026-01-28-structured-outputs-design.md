# Structured Outputs for Multi-Query LLM Transform

**Date:** 2026-01-28
**Status:** Approved
**Problem:** LLM responses truncated at `max_tokens` limit cause JSON parse failures (80% failure rate observed with 1000 token limit)

## Solution

Add structured outputs support to `azure_multi_query_llm` plugin. Structured outputs provide API-level schema enforcement, preventing truncation and guaranteeing valid JSON.

## Config Changes

### Breaking Change: `output_mapping` Format

Old format (removed):
```yaml
output_mapping:
  score: score
  rationale: rationale
```

New format (required):
```yaml
output_mapping:
  score:
    suffix: score
    type: integer
  rationale:
    suffix: rationale
    type: string
  confidence:
    suffix: confidence
    type: enum
    values: [low, medium, high]
```

### Response Format Options

- `response_format: standard` - Uses `{"type": "json_object"}`, types used for post-parse validation
- `response_format: structured` - Uses `{"type": "json_schema", ...}`, API enforces schema

### Supported Types

| Type | JSON Schema | Notes |
|------|-------------|-------|
| `string` | `{"type": "string"}` | Free text |
| `integer` | `{"type": "integer"}` | Whole numbers |
| `number` | `{"type": "number"}` | Floats |
| `boolean` | `{"type": "boolean"}` | true/false |
| `enum` | `{"enum": [...]}` | Requires `values` list |

## API Integration

For `response_format: structured`, build and send schema:

```python
llm_kwargs["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "rationale": {"type": "string"},
                "confidence": {"enum": ["low", "medium", "high"]}
            },
            "required": ["score", "rationale", "confidence"],
            "additionalProperties": False
        }
    }
}
```

All fields are required (no optional fields).

## Validation & Parsing

### Structured Mode
- JSON parsing still wrapped (API could still fail)
- Schema validation by API means we can trust structure
- Keep type validation as defense-in-depth (Tier 3 boundary principle)
- Remove markdown fence stripping (structured outputs don't wrap in code blocks)

### Standard Mode
- Keep markdown fence stripping (LLMs still do this in JSON mode)
- Add post-parse type validation using declared types
- Return error if type mismatch

### Error Format
```python
{
    "reason": "type_mismatch",
    "field": "score",
    "expected": "integer",
    "actual": "string",
    "value": "42"
}
```

## Files to Change

1. **`multi_query.py`** - New `OutputFieldConfig` model, update `MultiQueryConfig`
2. **`azure_multi_query.py`** - Schema builder, updated API call, type validation
3. **`openrouter_multi_query.py`** - Same changes (if OpenRouter supports structured outputs)

## Migration

This is a breaking change. Existing configs using the old `output_mapping` format will fail validation with a clear error message indicating the new required format.
