# Bug Report: Plugins cannot declare configuration-dependent required fields

## Summary

Many plugins have field requirements that depend on their **configuration** (templates, expressions, queries), but the schema system only supports **static** field declarations. This creates a **bidirectional validation gap** affecting:

**Plugins with config-dependent requirements:**
- LLM sinks with Jinja2 templates: `"Analyze {{ row.customer_id }}"` requires `customer_id`
- SQL sinks with dynamic queries: `"INSERT ... VALUES (${row.user_id})"` requires `user_id`
- Expression-based gates/transforms: `"row['amount'] > 100"` requires `amount`
- Filter transforms with field predicates: `"customer_type == 'premium'"` requires `customer_type`

Currently, none of these can declare their actual requirements in `input_schema`. This creates a gap:

**Consumer side (transforms/sinks with config-dependent requirements):**
1. Can't declare "I need fields A, B based on my configuration (template/expression/query)"
2. Must choose between:
   - `{"fields": "dynamic"}` → No validation, runtime failures
   - `{"fields": ["a", "b"]}` → Rejects extras, can't handle dynamic upstream
3. Config-time validation impossible

**Producer side (sources/transforms with dynamic + guaranteed fields):**
1. Can't declare "I produce dynamic fields BUT guarantee A, B, C are always present"
2. Dynamic source with guaranteed core fields has no schema representation
3. Must choose between:
   - `{"fields": "dynamic"}` → Consumer can't validate requirements met
   - `{"fields": ["a", "b", "c"]}` → Rejects dynamic extras from source

This violates schema validation principle: **both producers and consumers must be able to declare hybrid schemas (required + dynamic), and validation must verify required ⊆ guaranteed**

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: User (john)
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: Linux
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Bug triage and verification session
- Model/version: Claude Sonnet 4.5
- Tooling and permissions: Bug verification READ-ONLY
- Determinism details: N/A
- Notable tool calls or steps: User identified gap during triage discussion

## Steps To Reproduce

1. Create LLM sink with template that references fields: `"prompt": "Analyze {{ row.customer_id }} and {{ row.transaction_amount }}"`
2. Configure pipeline with transform that outputs `{"fields": "dynamic"}` (or schema without customer_id/transaction_amount)
3. Run DAG validation
4. Observe: Validation passes even though required fields are missing
5. Run pipeline with row missing those fields
6. Observe: Template rendering fails at runtime with unclear KeyError

## Expected Behavior

**Config-time validation:**
- LLM sink analyzes its template during initialization
- Extracts referenced field names (customer_id, transaction_amount)
- Declares `input_schema = {"fields": ["customer_id", "transaction_amount", "dynamic"]}`
- DAG validation catches missing fields before runtime

**Runtime behavior:**
- If schema is dynamic: template renders with available fields (or fails gracefully)
- If schema validation enabled: pipeline fails fast with clear error about missing required fields

## Actual Behavior

**Config-time:**
- LLM sinks declare `input_schema = {"fields": "dynamic"}` regardless of template requirements
- No field extraction from template
- DAG validation can't detect missing required fields

**Runtime:**
- Template rendering fails with `UndefinedError` or `KeyError`
- Error message doesn't clearly indicate which fields are missing
- Fails late in pipeline instead of at validation time

## Evidence

**Location:** `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/` (LLM sink implementations)

**Current pattern:**
```python
class SomeLLMSink(BaseSink):
    def __init__(self, config):
        self.template = config.prompt_template  # "Process {{ row.field_a }} and {{ row.field_b }}"
        # ❌ No field extraction from template

    @property
    def input_schema(self) -> dict:
        return {"fields": "dynamic"}  # ❌ Doesn't declare field_a, field_b as required
```

**What's needed (Consumer side - LLM sink):**
```python
class SomeLLMSink(BaseSink):
    def __init__(self, config):
        self.template = config.prompt_template
        # ✅ Extract referenced fields from template
        self._required_fields = self._extract_template_fields(self.template)

    @property
    def input_schema(self) -> dict:
        # ✅ Declare: "I REQUIRE fields A, B + ACCEPT dynamic extras"
        return {
            "fields": self._required_fields,
            "extras": "dynamic"  # or similar mechanism
        }

    def _extract_template_fields(self, template: str) -> list[str]:
        """Parse Jinja2 template to find {{ row.field_name }} references."""
        # Use jinja2.meta.find_undeclared_variables() or regex
        ...
```

**What's needed (Producer side - sources/transforms):**
```python
class CSVSource(BaseSource):
    @property
    def output_schema(self) -> dict:
        # ✅ Declare: "I GUARANTEE fields A, B, C + PRODUCE dynamic extras"
        return {
            "fields": ["customer_id", "transaction_id"],  # Guaranteed always present
            "extras": "dynamic"  # Other columns may vary by file
        }

# OR in transform config:
transforms:
  - name: extract_features
    type: custom_transform
    output_schema:
      fields: ["user_id", "timestamp"]  # Guaranteed
      extras: dynamic  # Plus computed features that vary
```

**Schema validation must check:**
1. Producer guarantees ⊇ Consumer requirements
2. `["customer_id", "transaction_id"] + dynamic` satisfies requirement for `["customer_id"] + dynamic` ✅
3. `["customer_id"] + dynamic` does NOT satisfy requirement for `["customer_id", "transaction_id"]` ❌

## Impact

**Pipeline Reliability:**
- Runtime failures instead of config-time validation
- Unclear error messages (Jinja2 UndefinedError vs "field X missing from schema")
- Wasted compute on pipelines that will fail

**Schema Validation:**
- Defeats purpose of schema validation
- Can't use strict mode to catch field mismatches
- Transform chains can silently drop required fields

**Developer Experience:**
- Hard to debug which fields LLM sink actually needs
- Trial-and-error to figure out required fields
- Template changes break pipelines unexpectedly

## Root Cause

1. **Missing template parsing:** Sinks don't analyze their templates to extract field references
2. **Generic schema declaration:** All LLM sinks use `{"fields": "dynamic"}` instead of declaring requirements
3. **No validation hook:** No mechanism for sinks to declare "I need fields X, Y, Z but also accept dynamic"

## Proposed Fix

### Option 1: Template Field Extraction (Recommended)

**Implementation:**
```python
import jinja2
from jinja2 import meta

def extract_template_fields(template_str: str) -> set[str]:
    """Extract row.field_name references from Jinja2 template."""
    env = jinja2.Environment()
    parsed = env.parse(template_str)
    variables = meta.find_undeclared_variables(parsed)

    # Filter for row.* references
    row_fields = set()
    for var in variables:
        if var.startswith("row."):
            field_name = var.split(".", 1)[1]
            row_fields.add(field_name)

    return row_fields

class BaseLLMSink:
    def __init__(self, config):
        self.template = config.prompt_template
        self._required_fields = list(extract_template_fields(self.template))

    @property
    def input_schema(self) -> dict:
        if not self._required_fields:
            return {"fields": "dynamic"}
        # Hybrid: required fields + dynamic
        return {"fields": self._required_fields + ["dynamic"]}
```

**Benefits:**
- Automatic field detection
- Config-time validation catches errors
- Clear contract: "I need A, B, C but accept other fields too"

**Risks:**
- Complex templates (conditionals, loops) might have optional fields
- Need to handle edge cases (row["field"] vs row.field syntax)

### Option 2: Manual Field Declaration

**Implementation:**
```yaml
sinks:
  - name: llm_classifier
    type: azure_batch
    prompt_template: "Classify {{ row.text }} from {{ row.source }}"
    required_fields:  # ✅ Explicit declaration
      - text
      - source
    allow_extra_fields: true
```

**Benefits:**
- Explicit, no magic parsing
- Developer controls exactly what's required
- Handles complex template logic

**Risks:**
- Manual maintenance (template changes require config updates)
- Can get out of sync with template

### Option 3: Hybrid (Recommended for Production)

1. **Extract fields from template** (Option 1) during sink initialization
2. **Allow override** in config (Option 2) for complex cases
3. **Validate at runtime**: If template references field not in extracted/declared set, log warning

## Related Bugs

- **P2-2026-01-21-schema-validator-skips-all-when-source-dynamic**: Related to dynamic schema validation gaps
- **P1-2026-01-21-csvsink-append-schema-mismatch**: Sink schema validation incomplete

## Test Cases Needed

```python
def test_llm_sink_declares_template_fields():
    """LLM sink must declare template-referenced fields in input_schema."""
    config = AzureBatchConfig(
        prompt_template="Analyze {{ row.customer_id }} for amount {{ row.amount }}"
    )
    sink = AzureBatchSink(config)

    schema = sink.input_schema
    assert "customer_id" in schema["fields"]
    assert "amount" in schema["fields"]
    assert "dynamic" in schema["fields"]  # Still accepts extras

def test_pipeline_validation_catches_missing_llm_fields():
    """DAG validation should fail if LLM sink's required fields are missing."""
    pipeline_config = {
        "source": {"type": "csv", "path": "input.csv"},
        "transforms": [
            {"name": "select", "type": "field_mapper", "output_schema": {"fields": ["user_id"]}}
        ],
        "sinks": [
            {
                "name": "classify",
                "type": "azure_batch",
                "prompt_template": "Classify user {{ row.user_id }} transaction {{ row.transaction_id }}"
            }
        ]
    }

    # Should fail: field_mapper only outputs user_id, but template needs transaction_id
    with pytest.raises(SchemaValidationError, match="missing required field: transaction_id"):
        dag = build_dag(pipeline_config)
        dag.validate()
```

## Acceptance Criteria

- [x] LLM sinks extract field references from templates (or accept manual declaration)
- [x] LLM sinks declare hybrid schema: `["field_a", "field_b", "dynamic"]`
- [x] DAG validation catches missing required fields at config-time
- [x] Runtime errors clearly indicate which fields are missing from template
- [x] Tests verify template field extraction and validation

## Notes

- This affects all LLM sinks: AzureBatch, LiteLLM, any sink using Jinja2 templates
- Should be backported to all plugin packs that use templates
- Consider making template field extraction a utility in core for reuse

## Verification Status

**Status:** CLOSED - FIXED (2026-01-29)

### Implementation Summary

Implemented Option 3 (Hybrid) with **explicit contracts** approach:

1. **Schema contracts** (`src/elspeth/contracts/schema.py`):
   - Added `guaranteed_fields` and `required_fields` to `SchemaConfig`
   - `get_effective_guaranteed_fields()` returns producer guarantees
   - `get_effective_required_fields()` returns consumer requirements

2. **Template field extraction** (`src/elspeth/core/templates.py`):
   - Dev-time utility `extract_jinja2_fields()` using Jinja2 AST parsing
   - Handles both `row.field` and `row["field"]` syntax
   - Documents limitations (conditionals, dynamic keys)

3. **Explicit declaration requirement** (`src/elspeth/plugins/llm/base.py`):
   - LLMConfig requires `required_input_fields` when template references row fields
   - Error-with-opt-out pattern: `None` = error, `[]` = explicit opt-out
   - Uses AST parser for accurate row field detection

4. **DAG validation** (`src/elspeth/core/dag.py`):
   - `_get_guaranteed_fields()` and `_get_required_fields()` helpers
   - `_validate_single_edge()` checks contracts before type validation
   - Clear error messages with actionable fix suggestions
   - Aggregation nodes properly checked via nested `options` dict

5. **Tests**:
   - `tests/core/test_templates.py`: Field extraction tests
   - `tests/core/test_dag_contract_validation.py`: Contract validation tests
   - `tests/integration/test_llm_contract_validation.py`: End-to-end tests
   - `tests/integration/test_aggregation_contracts.py`: Aggregation tests

### Additional Fixes (P2 bugs from Codex):
- Row field detection now uses AST parser (not substring matching)
- Aggregation nodes properly read `required_input_fields` from nested `options`
- Optional fields (marked with `?`) are NOT included in guaranteed fields
