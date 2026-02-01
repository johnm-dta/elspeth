# Multi-Query Case Study Assessment Example (OpenRouter)

This example demonstrates the `openrouter_multi_query_llm` transform, which evaluates
multiple case studies against multiple criteria using OpenRouter's unified LLM API.

## Why OpenRouter?

OpenRouter provides access to **100+ LLM models** through a single API:
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **OpenAI**: GPT-4o, GPT-4 Turbo
- **Meta**: Llama 3.1 (8B, 70B, 405B)
- **Mistral**: Mistral Large, Mixtral
- **Google**: Gemini Pro
- And many more...

This means you can switch models by changing one line in your config.

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
# Set OpenRouter API key (get one at https://openrouter.ai/keys)
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run the pipeline
uv run elspeth run -s examples/openrouter_multi_query_assessment/suite.yaml --execute
```

### Run with JSONL Change Journal

Enable the optional JSONL change journal for an append-only backup stream:

```bash
uv run elspeth run -s examples/openrouter_multi_query_assessment/suite_journal.yaml --execute
```

The journal is disabled by default. See [JSONL Change Journal](../../README.md#jsonl-change-journal-optional) for configuration details.

## Model Selection

Change the model in `suite.yaml` to use different providers:

```yaml
# Best quality (Claude 3 Opus)
model: "anthropic/claude-3-opus"

# Balanced quality/speed (Claude 3.5 Sonnet) - DEFAULT
model: "anthropic/claude-3-5-sonnet"

# Fast and capable (GPT-4o)
model: "openai/gpt-4o"

# Cost-effective (Llama 3.1 70B)
model: "meta-llama/llama-3.1-70b-instruct"

# Budget option (Llama 3.1 8B)
model: "meta-llama/llama-3.1-8b-instruct"
```

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

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

## Comparison with Azure Version

| Feature | Azure Multi-Query | OpenRouter Multi-Query |
|---------|------------------|------------------------|
| API | Azure OpenAI SDK | HTTP REST API |
| Models | Azure-hosted OpenAI | 100+ providers |
| Auth | Azure AD / API Key | API Key |
| Rate Limits | Per-deployment | Per-account |
| Error Codes | SDK exceptions | HTTP status codes |

Both transforms use identical:
- Query expansion (case_studies × criteria)
- Template rendering with `{{ row.input_N }}` and `{{ row.criterion }}`
- Output mapping (JSON field → column suffix)
- Pooled parallel execution with AIMD retry
- All-or-nothing row semantics
