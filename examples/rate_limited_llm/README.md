# Rate-Limited LLM Example

Demonstrates rate limiting on external API calls to prevent flooding upstream services.

## What This Shows

A sentiment analysis pipeline with the `rate_limit` configuration section capping API calls to 30 requests per minute. Uses ChaosLLM so no real API key is needed.

```
source ─(source_out)─> openrouter_llm (30 req/min) ─┬─ results.json
                                                      └─ quarantined.json
```

## Prerequisites

Start the ChaosLLM server:

```bash
chaosllm serve --port 8199 --preset=realistic
```

## Running

```bash
elspeth run --settings examples/rate_limited_llm/settings.yaml --execute
```

## Output

- `output/results.json` — Enriched rows with sentiment analysis (JSONL)
- `output/quarantined.json` — Rows that failed after all retries

## Rate Limit Configuration

```yaml
rate_limit:
  enabled: true
  default_requests_per_minute: 30    # Global default
  services:                           # Optional per-service overrides
    openai:
      requests_per_minute: 100
    slow_api:
      requests_per_minute: 10
```

### Options

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable/disable rate limiting |
| `default_requests_per_minute` | `60` | Default limit for unconfigured services |
| `persistence_path` | `null` | SQLite path for cross-process rate limits |
| `services` | `{}` | Per-service overrides |

### When to Use

- **Always** when calling external APIs in production
- **Especially** with pooled execution (`pool_size > 1`) where concurrent workers can easily exceed API quotas
- **Cross-process**: Set `persistence_path` when running multiple pipeline instances against the same API

## Key Concepts

- **Sliding window**: Uses per-minute rolling window (not fixed windows)
- **Thread-safe**: Safe with pooled/concurrent execution
- **Per-service**: Different APIs can have different limits
- **Transparent**: Rate limiting is applied automatically — transforms don't need to know about it

## See Also

- [`chaosllm_sentiment`](../chaosllm_sentiment/) — Same pipeline without rate limiting, for comparison
- [`openrouter_sentiment`](../openrouter_sentiment/) — Real OpenRouter API (bring your own key)
