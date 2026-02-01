# Azure OpenAI Sentiment Analysis Example

This example demonstrates using ELSPETH with the Azure OpenAI LLM transform to perform sentiment analysis on text data.

## What it does

1. Reads customer feedback text from `input.csv`
2. Sends each text through Azure OpenAI (using your deployed model)
3. Gets sentiment classification (positive/negative/neutral) with confidence scores
4. Writes enriched results to `output/results.csv`

## Prerequisites

1. An Azure OpenAI resource with a deployed model (e.g., gpt-4o, gpt-4o-mini)
2. Set the environment variables:
   ```bash
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
   export AZURE_OPENAI_KEY="your-api-key-here"
   export AZURE_OPENAI_DEPLOYMENT="your-gpt-deployment-name"
   ```

## Running the example

```bash
# From the repository root
uv run elspeth run -s examples/azure_openai_sentiment/settings.yaml --execute
```

## Pooled (Multi-threaded) Execution

For higher throughput with larger datasets, use the pooled variant which processes multiple rows concurrently:

```bash
uv run elspeth run -s examples/azure_openai_sentiment/settings_pooled.yaml --execute
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
- `sentiment_analysis_model` - The deployment/model that responded

## Azure OpenAI Setup

### Creating a deployment

1. Go to Azure Portal > Azure OpenAI > your resource
2. Navigate to "Model deployments"
3. Click "Create new deployment"
4. Select a model (e.g., `gpt-4o-mini` for cost-effective demos)
5. Name your deployment (this is your `AZURE_OPENAI_DEPLOYMENT`)

### Finding your endpoint

Your endpoint URL is in the Azure Portal:
- Azure OpenAI resource > "Keys and Endpoint"
- Format: `https://<your-resource-name>.openai.azure.com`

### API versions

The default API version is `2024-10-21`. You can override this in the config:
```yaml
api_version: "2024-10-21"
```

See [Azure OpenAI API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference) for available versions.

## Comparison with OpenRouter

| Feature | Azure OpenAI | OpenRouter |
|---------|--------------|------------|
| Authentication | Azure API key + endpoint | OpenRouter API key |
| Model selection | Azure deployment name | Model string (e.g., `openai/gpt-4o`) |
| Rate limits | Per-deployment TPM/RPM | Per-account limits |
| Billing | Azure subscription | Pay-as-you-go |
| Best for | Enterprise, compliance, Azure ecosystem | Multi-provider access, prototyping |

## Audit trail

The pipeline records full audit data to `runs/audit.db`, including:
- Every input row processed
- The full LLM request (prompt, parameters)
- The full LLM response (content, tokens, latency)
- Content hashes for verification

Query the audit trail:
```bash
uv run elspeth explain --run latest --database examples/azure_openai_sentiment/runs/audit.db
```
