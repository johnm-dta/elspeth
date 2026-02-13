# ChaosLLM Endurance Test Example

Demonstrates the `openrouter_multi_query_llm` transform under sustained fault injection using ChaosLLM.

## What This Shows

A medical assessment pipeline processes case studies against multiple evaluation criteria (diagnosis, treatment, prognosis, risk, follow-up). Each row generates N x M LLM calls (case studies x criteria) with a pool of 30 concurrent workers. ChaosLLM injects realistic failures to stress-test retry logic and error handling.

```
source ─(source_out)─> openrouter_multi_query_llm ─┬─(output)─> results.csv
                                                    └─(on_error)─> quarantined.json
```

## Prerequisites

Start the ChaosLLM server:

```bash
chaosllm serve --port 8199 --config examples/chaosllm_endurance/chaos_config.yaml
```

## Running

```bash
# Using the convenience script (starts ChaosLLM + runs pipeline)
./examples/chaosllm_endurance/run.sh

# Or manually
elspeth run --settings examples/chaosllm_endurance/settings.yaml --execute
```

## Output

- `output/results.csv` — Scored assessments with per-criterion scores and rationales
- `output/quarantined.json` — Rows that failed after all retry attempts

## Key Concepts

- **Multi-query transform**: Each input row generates multiple LLM calls (2 case studies x 5 criteria = 10 calls per row)
- **Pooled execution**: `pool_size: 30` for concurrent LLM calls within batches
- **Retry resilience**: 5 attempts with exponential backoff (0.5s initial, 10s max, base 2.0)
- **Template-driven prompts**: Jinja2 templates with `row.input_1`, `row.criterion.name`, etc.
- **Structured output mapping**: LLM JSON responses mapped to typed columns (`score` as integer, `rationale` as string)
