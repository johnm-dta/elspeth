# ChaosLLM Sentiment Analysis Example

Demonstrates sentiment analysis using the `openrouter_llm` transform against a local ChaosLLM server for testing without API keys.

## What This Shows

The same pipeline as [`openrouter_sentiment`](../openrouter_sentiment/), but configured to hit a local ChaosLLM server instead of the real OpenRouter API. This allows testing LLM pipeline behaviour — including error handling and retries — without consuming API credits.

```
source ─(source_out)─> openrouter_llm ─┬─(output)─> results.json
                                        └─(on_error)─> quarantined.json
```

## Prerequisites

Start the ChaosLLM server:

```bash
chaosllm serve --port 8199 --config examples/chaosllm_sentiment/chaos_config.yaml
```

## Running

```bash
# Using the convenience script (starts ChaosLLM + runs pipeline)
./examples/chaosllm_sentiment/run.sh

# Or manually
elspeth run --settings examples/chaosllm_sentiment/settings.yaml --execute
```

## Output

- `output/results.json` — Enriched rows with sentiment classification (JSONL format)
- `output/quarantined.json` — Rows that failed after all retry attempts

## Key Concepts

- **ChaosLLM testing**: Uses `base_url: http://127.0.0.1:8199/v1` with a fake API key — no real API credentials needed
- **Same pipeline, different backend**: Compare with [`openrouter_sentiment`](../openrouter_sentiment/) which uses the real API
- **Retry under faults**: ChaosLLM injects rate limits, timeouts, and malformed responses to exercise retry logic
