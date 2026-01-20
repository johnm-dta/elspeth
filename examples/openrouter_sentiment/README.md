# OpenRouter Sentiment Analysis Example

This example demonstrates using ELSPETH with the OpenRouter LLM transform to perform sentiment analysis on text data.

## What it does

1. Reads customer feedback text from `input.csv`
2. Sends each text through OpenRouter (using GPT-4o-mini by default)
3. Gets sentiment classification (positive/negative/neutral) with confidence scores
4. Writes enriched results to `output/results.csv`

## Prerequisites

1. Get an OpenRouter API key from https://openrouter.ai/
2. Set the environment variable:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

## Running the example

```bash
# From the repository root
uv run elspeth run -s examples/openrouter_sentiment/settings.yaml --execute
```

## Pooled (Multi-threaded) Execution

For higher throughput with larger datasets, use the pooled variant which processes multiple rows concurrently:

```bash
uv run elspeth run -s examples/openrouter_sentiment/settings_pooled.yaml --execute
```

**Key differences:**
- `pool_size: 3` - Processes 3 rows concurrently instead of sequentially
- **AIMD throttling** - Automatically backs off on rate limits (HTTP 429), then gradually increases concurrency
- **Order preservation** - Results maintain submission order despite concurrent processing
- Separate output: `output/results_pooled.csv` and audit: `runs/audit_pooled.db`

**When to use pooled execution:**
- Large datasets (100+ rows) where sequential processing is slow
- When you have API quota headroom for concurrent requests
- Production workloads where throughput matters

**Configuration options:**
```yaml
pool_size: 3                      # Number of concurrent workers
max_dispatch_delay_ms: 5000       # Maximum backoff delay (optional)
max_capacity_retry_seconds: 60    # Timeout for rate limit retries (optional)
```

## Output

The output CSV will contain the original columns plus:
- `sentiment_analysis` - The LLM's JSON response
- `sentiment_analysis_usage` - Token usage metadata
- `sentiment_analysis_template_hash` - Hash of the prompt template (for audit)
- `sentiment_analysis_variables_hash` - Hash of the input variables (for audit)
- `sentiment_analysis_model` - The actual model that responded

## Changing the model

Edit `settings.yaml` and change the `model` field. Popular options:
- `openai/gpt-4o-mini` - Fast, cheap, good for demos
- `anthropic/claude-3-haiku` - Fast, cheap Anthropic option
- `anthropic/claude-3-sonnet` - Better quality, moderate cost
- `openai/gpt-4o` - High quality, higher cost

See https://openrouter.ai/models for all available models.

## Batch Aggregation Mode

For workloads that benefit from processing rows in coordinated batches, use the batched variant:

```bash
uv run elspeth run -s examples/openrouter_sentiment/settings_batched.yaml --execute
```

**Key differences from pooled mode:**
- **Buffering**: Rows are collected until trigger fires (e.g., every 5 rows)
- **Batch processing**: Entire batch sent to `_process_batch()` at once
- **Parallel within batch**: `pool_size: 3` processes batch rows concurrently
- **Clear boundaries**: Each batch is a discrete unit for checkpointing/auditing
- Separate output: `output/results_batched.csv` and audit: `runs/audit_batched.db`

**When to use batch aggregation:**
- When you need coordinated processing across multiple rows
- When API rate limits are per-batch rather than per-request
- When you want clear batch boundaries for checkpointing or auditing
- When rows have dependencies that benefit from batch context

**Comparison of execution modes:**

| Mode | Config | Behavior |
|------|--------|----------|
| Sequential | `settings.yaml` | Process 1 row at a time (simple, slow) |
| Pooled | `settings_pooled.yaml` | Stream rows through concurrent workers |
| Batched | `settings_batched.yaml` | Buffer N rows, process batch in parallel |

**Configuration:**
```yaml
aggregations:
  - name: sentiment_batch
    plugin: openrouter_llm
    trigger:
      count: 5              # Buffer 5 rows before processing
    output_mode: passthrough  # N inputs → N outputs (enriched)
    options:
      pool_size: 3          # Concurrent workers within batch
```

## Audit trail

The pipeline records full audit data to `runs/audit.db`, including:
- Every input row processed
- The full LLM request (prompt, parameters)
- The full LLM response (content, tokens, latency)
- Content hashes for verification

Query the audit trail:
```bash
uv run elspeth explain -s examples/openrouter_sentiment/settings.yaml --run latest
```
