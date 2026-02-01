# Multi-Query Case Study Assessment Example

This example demonstrates the `azure_multi_query_llm` transform, which evaluates
multiple case studies against multiple criteria in a single pipeline step.

## Configuration

- **2 case studies** per row (cs1, cs2)
- **5 criteria** (diagnosis, treatment, prognosis, risk, followup)
- **10 LLM queries per row** (2 × 5 matrix)
- **All queries run in parallel** (pool_size: 10)
- **All-or-nothing error handling** per row

## Input Format

CSV with columns:
- `user_id` - Unique identifier
- `cs1_background`, `cs1_symptoms`, `cs1_history` - Case study 1 data
- `cs2_background`, `cs2_symptoms`, `cs2_history` - Case study 2 data

See `input.csv` for sample data with medical case studies.

## Output Format

Original columns plus 20 assessment columns (5 criteria × 2 case studies × 2 fields):
- `cs1_diagnosis_score`, `cs1_diagnosis_rationale`
- `cs1_treatment_score`, `cs1_treatment_rationale`
- `cs1_prognosis_score`, `cs1_prognosis_rationale`
- `cs1_risk_score`, `cs1_risk_rationale`
- `cs1_followup_score`, `cs1_followup_rationale`
- `cs2_diagnosis_score`, `cs2_diagnosis_rationale`
- ... (same pattern for cs2)

Plus metadata columns per query (usage, model, template_hash, etc.)

## Running

```bash
# Set Azure OpenAI credentials
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

# Run the pipeline
uv run elspeth run -s examples/multi_query_assessment/suite.yaml --execute
```

## Customization

### Adding More Case Studies

Add entries to `case_studies` in suite.yaml:
```yaml
case_studies:
  - name: cs3
    input_fields: [cs3_background, cs3_symptoms, cs3_history]
```

### Adding More Criteria

Add entries to `criteria` in suite.yaml:
```yaml
criteria:
  - name: safety
    code: SAFE
    description: "Assess patient safety considerations"
    subcriteria:
      - Medication interactions
      - Fall risk
```

### Using Lookup Data

The `criteria_lookup.yaml` file contains weights and guidance for each criterion.
You can reference this in your template:
```jinja2
Guidance: {{ lookup[criterion.code].guidance }}
Weight: {{ lookup[criterion.code].weight }}
```
