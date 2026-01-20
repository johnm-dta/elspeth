# Azure Blob Sentiment Analysis Example

This example demonstrates a fully Azure-native ELSPETH pipeline:
- **Source**: Azure Blob Storage (CSV file)
- **Transform**: Azure OpenAI (sentiment analysis)
- **Sink**: Azure Blob Storage (results CSV)

## What it does

1. Reads customer feedback from Azure Blob Storage (`input/sentiment_data.csv`)
2. Sends each text through Azure OpenAI for sentiment analysis
3. Gets sentiment classification (positive/negative/neutral) with confidence scores
4. Writes enriched results back to Azure Blob Storage (`output/{run_id}/results.csv`)

## Prerequisites

### 1. Azure OpenAI

You need an Azure OpenAI resource with a deployed model:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

### 2. Azure Storage Account

You need an Azure Storage account with a container:

```bash
# Get this from Azure Portal > Storage Account > Access Keys
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=youraccount;AccountKey=yourkey;EndpointSuffix=core.windows.net"
export AZURE_STORAGE_CONTAINER="elspeth-demo"
```

### 3. Upload Input Data

Upload the sample input file to your blob container:

```bash
# Using Azure CLI
az storage blob upload \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --container-name "$AZURE_STORAGE_CONTAINER" \
  --file examples/azure_blob_sentiment/input.csv \
  --name "input/sentiment_data.csv"
```

Or create `input/sentiment_data.csv` in your container with this content:

```csv
id,text
1,"I absolutely love this product! It exceeded all my expectations."
2,"The service was terrible and the staff were rude. Never coming back."
3,"It was okay, nothing special but nothing bad either."
4,"Amazing experience! Highly recommend to everyone."
5,"Completely disappointed. Waste of money."
```

## Running the example

```bash
# Sequential execution
uv run elspeth run -s examples/azure_blob_sentiment/settings.yaml --execute

# Pooled execution (concurrent)
uv run elspeth run -s examples/azure_blob_sentiment/settings_pooled.yaml --execute
```

## Output

Results are written to Azure Blob Storage at:
```
{container}/output/{run_id}/results.csv
```

The output includes:
- Original `id` and `text` columns
- `sentiment_analysis` - The LLM's JSON response
- `sentiment_analysis_usage` - Token usage metadata
- `sentiment_analysis_template_hash` - Hash of prompt template (for audit)
- `sentiment_analysis_variables_hash` - Hash of input variables (for audit)
- `sentiment_analysis_model` - The model that responded

### Viewing Results

```bash
# List output blobs
az storage blob list \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --container-name "$AZURE_STORAGE_CONTAINER" \
  --prefix "output/" \
  --output table

# Download results
az storage blob download \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --container-name "$AZURE_STORAGE_CONTAINER" \
  --name "output/{run_id}/results.csv" \
  --file results.csv
```

## Authentication Options

The Azure Blob plugins support three authentication methods:

### 1. Connection String (used in this example)

Simplest option, good for development:

```yaml
datasource:
  plugin: azure_blob
  options:
    connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
    container: "my-container"
    blob_path: "data/input.csv"
```

### 2. Managed Identity

Best for Azure-hosted workloads (VMs, App Service, Functions):

```yaml
datasource:
  plugin: azure_blob
  options:
    use_managed_identity: true
    account_url: "https://mystorageaccount.blob.core.windows.net"
    container: "my-container"
    blob_path: "data/input.csv"
```

### 3. Service Principal

Best for CI/CD pipelines:

```yaml
datasource:
  plugin: azure_blob
  options:
    tenant_id: "${AZURE_TENANT_ID}"
    client_id: "${AZURE_CLIENT_ID}"
    client_secret: "${AZURE_CLIENT_SECRET}"
    account_url: "https://mystorageaccount.blob.core.windows.net"
    container: "my-container"
    blob_path: "data/input.csv"
```

## Supported Formats

Both source and sink support:

| Format | Description |
|--------|-------------|
| `csv` | Comma-separated values (default) |
| `json` | JSON array of objects |
| `jsonl` | Newline-delimited JSON |

### Example: JSON format

```yaml
datasource:
  plugin: azure_blob
  options:
    connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
    container: "my-container"
    blob_path: "data/input.json"
    format: json
    json_options:
      encoding: utf-8
      data_key: "records"  # Optional: extract from nested key
```

## Dynamic Output Paths

The sink blob_path supports Jinja2 templates:

```yaml
sinks:
  output:
    plugin: azure_blob
    options:
      blob_path: "output/{{ run_id }}/results.csv"      # Per-run directory
      # Or: "output/{{ timestamp }}/results.csv"        # Timestamp-based
```

Available variables:
- `{{ run_id }}` - The unique run identifier
- `{{ timestamp }}` - ISO format timestamp at write time

## Audit Trail

The pipeline records full audit data locally to `runs/audit.db`, including:
- Every input row processed
- The full LLM request (prompt, parameters)
- The full LLM response (content, tokens, latency)
- Content hashes for verification
- Source and sink blob paths

Query the audit trail:
```bash
uv run elspeth explain -s examples/azure_blob_sentiment/settings.yaml --run latest
```

## Troubleshooting

### "ResourceNotFoundError: The specified container does not exist"

Create the container first:
```bash
az storage container create \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --name "$AZURE_STORAGE_CONTAINER"
```

### "ResourceNotFoundError: The specified blob does not exist"

Upload the input file:
```bash
az storage blob upload \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --container-name "$AZURE_STORAGE_CONTAINER" \
  --file examples/azure_blob_sentiment/input.csv \
  --name "input/sentiment_data.csv"
```

### "ClientAuthenticationError: Server failed to authenticate the request"

Check your connection string or credentials are correct and not expired.
