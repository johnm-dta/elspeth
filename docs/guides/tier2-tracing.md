# Tier 2 Plugin Tracing Guide

Tier 2 tracing provides deep LLM observability beyond the framework's Tier 1 telemetry. While Tier 1 captures latency, status, and content hashes for ALL external calls, Tier 2 captures full prompts, responses, and token-level metrics for LLM calls specifically.

> **Langfuse SDK v3.12+ Required** (as of ELSPETH RC-3)
>
> ELSPETH now uses Langfuse SDK v3 which is built on OpenTelemetry standards. If you're upgrading from an earlier ELSPETH version with Langfuse v2, update your SDK:
> ```bash
> uv pip install 'langfuse>=3.12,<4'
> ```
> No configuration changes are required - the migration is internal to ELSPETH.

## Overview

ELSPETH has two independent telemetry tiers:

| Aspect | Tier 1 (Framework) | Tier 2 (Plugin) |
|--------|-------------------|-----------------|
| **Scope** | All external calls (LLM, HTTP, SQL) | LLM calls only |
| **Data captured** | Hashes, latency, status, call counts | Full prompts, responses, token usage |
| **Destination** | Generic observability platforms | LLM-specific platforms |
| **Configuration** | `telemetry:` section in settings | Plugin `options.tracing:` block |
| **Dependency** | Optional exporters | Optional SDK per provider |

**Key principle:** Tier 1 and Tier 2 are complementary. Use Tier 1 for operational monitoring and alerting. Use Tier 2 for prompt engineering, cost tracking, and LLM-specific analytics.

## Supported Providers

### Real-Time Plugins

| Provider | azure_llm | azure_multi_query_llm | openrouter_llm | openrouter_multi_query_llm |
|----------|-----------|----------------------|----------------|---------------------------|
| Azure AI (App Insights) | Yes | Yes | No | No |
| Langfuse | Yes | Yes | Yes | Yes |

### Batch Plugins

| Provider | azure_batch_llm | openrouter_batch_llm | Notes |
|----------|----------------|---------------------|-------|
| Azure AI | No | No | Batch API runs in Azure infrastructure - no SDK instrumentation |
| Langfuse | Job-level only | Per-call | Azure Batch = async job submission; OpenRouter Batch = sync parallel HTTP |

**Why Azure AI doesn't work with OpenRouter:**
Azure AI tracing auto-instruments the OpenAI SDK. OpenRouter plugins use HTTP directly via `httpx`, so there's no SDK to instrument. Use Langfuse for OpenRouter tracing.

**Why Azure AI doesn't work with Azure Batch:**
The Azure Batch API submits jobs that run asynchronously in Azure's infrastructure. The OpenAI SDK is only used to submit and check status - the actual LLM inference happens outside your process. Langfuse can trace at the job level (submit/complete), but not individual row processing.

## Azure AI (Application Insights)

Azure AI tracing uses Azure Monitor OpenTelemetry to auto-instrument the OpenAI SDK. This captures full prompts and responses in Application Insights.

### Configuration

```yaml
transforms:
  - plugin: azure_llm
    options:
      deployment_name: gpt-4o
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      template: |
        Analyze: {{ row.text }}

      # Tier 2: Azure AI tracing
      tracing:
        provider: azure_ai
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
        enable_content_recording: true  # Capture prompts/responses (default: true)
        enable_live_metrics: false      # Live Metrics Stream (default: false)
```

### Required Dependency

```bash
uv pip install elspeth[tracing-azure]
# Or manually: uv pip install azure-monitor-opentelemetry
```

### Finding Your Connection String

1. Go to Azure Portal > Application Insights resource
2. Overview > Connection String (copy)
3. Set as environment variable:
   ```bash
   export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."
   ```

### What Gets Captured

With `enable_content_recording: true`:
- Full prompt messages (system + user)
- Full response content
- Token usage (prompt_tokens, completion_tokens, total_tokens)
- Latency metrics
- Model/deployment information
- Error details on failure

### Process-Level Warning

Azure Monitor OpenTelemetry configures at the process level. If multiple plugins configure `azure_ai` tracing, the first one to initialize wins. All subsequent configurations are ignored.

## Langfuse

Langfuse provides LLM-specific observability with prompt engineering analytics, cost tracking, and evaluation capabilities.

### Configuration

```yaml
transforms:
  - plugin: azure_llm  # Or any LLM plugin
    options:
      deployment_name: gpt-4o
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      template: |
        Analyze: {{ row.text }}

      # Tier 2: Langfuse tracing
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
        host: https://cloud.langfuse.com  # Default, or self-hosted URL
        tracing_enabled: true  # v3: Can be disabled per-plugin (default: true)
```

### Required Dependency

```bash
uv pip install elspeth[tracing-langfuse]
# Or manually: uv pip install 'langfuse>=3.12,<4'
```

> **Note:** Langfuse SDK v3.12+ is required. This version uses OpenTelemetry-based context managers for trace lifecycle management and is thread-safe for concurrent pipeline execution.

### Langfuse Credentials

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (or deploy self-hosted)
2. Create a project
3. Go to Settings > API Keys
4. Copy Public Key and Secret Key
5. Set as environment variables:
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-..."
   export LANGFUSE_SECRET_KEY="sk-..."
   ```

### What Gets Captured

For real-time plugins (azure_llm, openrouter_llm, etc.):
- Full prompt content
- Full response content
- Token usage (input, output, total)
- Latency per call
- Model information
- Token ID for correlation with Landscape

For multi-query plugins (azure_multi_query_llm, openrouter_multi_query_llm):
- Trace per row (not per query)
- Aggregate query count and success count
- Total latency for all queries in the row

For batch plugins:
- Azure Batch: Job-level trace (submit/complete events, not per-row)
- OpenRouter Batch: Per-call traces (full observability)

### Langfuse Advantages

Unlike Azure AI auto-instrumentation:
- **Per-instance configuration:** Each plugin can have different Langfuse settings
- **Works with any plugin:** HTTP-based manual instrumentation, not SDK-dependent
- **Self-hosted option:** Keep data on-premises if required
- **LLM-specific features:** Prompt versioning, A/B testing, evaluation scores
- **Thread-safe:** Uses OpenTelemetry thread-local context (safe for concurrent pipelines)

### Langfuse v3 Behavioral Note

In Langfuse SDK v3, traces are recorded **after** a successful LLM call, not wrapped around the call. This means:

- **LLM call succeeds:** Trace is created with full prompt, response, and usage
- **LLM call fails:** No trace is created in Langfuse

This is acceptable because:
1. Failed LLM calls are always recorded in the Landscape audit trail (source of truth)
2. Telemetry events are still emitted for operational visibility
3. Tier 2 tracing is for prompt engineering and cost tracking, not failure investigation

## Using Both Tiers Together

Tier 1 and Tier 2 serve different purposes and can (and should) be used together:

```yaml
# Tier 1: Framework telemetry to Datadog for operational monitoring
telemetry:
  enabled: true
  granularity: rows
  exporters:
    - name: datadog
      options:
        service_name: my-pipeline
        env: production

# Tier 2: LLM-specific tracing to Langfuse for prompt engineering
transforms:
  - plugin: azure_llm
    options:
      deployment_name: gpt-4o
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_key: ${AZURE_OPENAI_KEY}
      template: |
        Analyze: {{ row.text }}
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
```

**Workflow with both tiers:**

1. **Datadog alert fires:** "LLM latency P95 > 5s"
2. **Datadog trace shows:** `run_id=run-abc123`, `node_id=llm_classifier`
3. **Landscape query:** Get failed row IDs and error messages
4. **Langfuse analysis:** Review actual prompts, identify why responses were slow
5. **Root cause:** Prompts too long, hitting context limit

## Tier 1 + Tier 2 Azure AI Conflict

**Warning:** Using Azure AI for BOTH Tier 1 telemetry AND Tier 2 plugin tracing can cause conflicts.

### The Problem

Both systems use OpenTelemetry:
- Tier 1 `azure_monitor` exporter configures OTEL for general telemetry
- Tier 2 `azure_ai` tracing configures OTEL for LLM auto-instrumentation

OpenTelemetry's global tracer provider can only be set once. The second configuration may fail silently or overwrite the first.

### Detection

If you see this log warning:
```
Existing OpenTelemetry tracer detected - Azure AI tracing may conflict with Tier 1 telemetry
```

### Solutions

**Option 1: Use Langfuse for Tier 2 (Recommended)**
```yaml
# Tier 1: Azure Monitor for framework telemetry
telemetry:
  exporters:
    - name: azure_monitor
      options:
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}

# Tier 2: Langfuse for LLM tracing (no OTEL conflict)
transforms:
  - plugin: azure_llm
    options:
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
```

**Option 2: Use only Tier 2 Azure AI**
```yaml
# Skip Tier 1 azure_monitor exporter
telemetry:
  exporters:
    - name: datadog  # Use non-OTEL exporter
      options:
        service_name: my-pipeline

# Tier 2: Azure AI (now owns OTEL)
transforms:
  - plugin: azure_llm
    options:
      tracing:
        provider: azure_ai
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
```

**Option 3: Use only Tier 1 with full granularity**
```yaml
# Tier 1 at 'full' granularity captures external call details
telemetry:
  granularity: full  # Includes ExternalCallCompleted events
  exporters:
    - name: azure_monitor
      options:
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}

# No Tier 2 tracing - rely on Tier 1 for LLM visibility
transforms:
  - plugin: azure_llm
    options:
      # No tracing: block
```

## Multi-Plugin Scenarios

When multiple LLM plugins have different tracing configurations:

| Plugin A Config | Plugin B Config | Behavior |
|-----------------|-----------------|----------|
| `langfuse` | `langfuse` (same host) | Both trace to same project |
| `langfuse` (host A) | `langfuse` (host B) | Each traces to its own instance |
| `azure_ai` | `azure_ai` | First wins (process-level) |
| `azure_ai` | `langfuse` | Both work independently |
| `azure_ai` | `none` | Only Plugin A traces |
| `langfuse` | `none` | Only Plugin A traces |

**Best practice for multi-plugin pipelines:**
```yaml
transforms:
  # All LLM plugins trace to same Langfuse project
  - plugin: azure_llm
    options:
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}

  - plugin: openrouter_llm
    options:
      tracing:
        provider: langfuse
        public_key: ${LANGFUSE_PUBLIC_KEY}
        secret_key: ${LANGFUSE_SECRET_KEY}
```

## Graceful Degradation

Tier 2 tracing is designed to fail gracefully. If the required SDK is not installed:

1. Plugin logs a warning at startup:
   ```
   Langfuse tracing requested but package not installed
   hint="Install with: uv pip install elspeth[tracing-langfuse]"
   ```

2. Pipeline continues without tracing

3. No runtime errors - tracing methods become no-ops

**To explicitly disable tracing:**
```yaml
transforms:
  - plugin: azure_llm
    options:
      # Simply omit the tracing block, or:
      tracing:
        provider: none
```

**To make missing SDK a hard error:**
There is no configuration for this. If you require tracing, ensure the dependency is installed in your environment. Consider adding a pre-flight check in your deployment pipeline.

## Privacy Considerations

### PII in Prompts and Responses

Tier 2 tracing captures full prompts and responses. This may include:
- Personal identifiable information (PII) from input data
- Sensitive business data
- Medical, financial, or legal information

**Mitigations:**

1. **Disable content recording (Azure AI only):**
   ```yaml
   tracing:
     provider: azure_ai
     connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
     enable_content_recording: false  # Only capture metadata
   ```

2. **Use Langfuse with data masking:**
   Langfuse supports regex-based masking in the UI for viewing sensitive data.

3. **Self-hosted Langfuse:**
   Deploy Langfuse in your own infrastructure to keep data within your security boundary.

4. **No Tier 2 tracing:**
   Rely on Tier 1 telemetry (hashes only) and Landscape audit trail for investigations.

### Data Residency

| Provider | Data Location | Control |
|----------|---------------|---------|
| Azure AI (App Insights) | Your Azure subscription region | Full control |
| Langfuse Cloud | EU (default) or US | Choose region at signup |
| Langfuse Self-Hosted | Your infrastructure | Full control |

### Compliance Considerations

| Requirement | Recommendation |
|-------------|----------------|
| GDPR | Use EU-region Langfuse or self-hosted |
| HIPAA | Self-hosted Langfuse with BAA, or disable Tier 2 |
| SOC 2 | Azure AI (inherits Azure compliance) or self-hosted |
| Data minimization | `enable_content_recording: false` or no Tier 2 |

## Troubleshooting

### Tracing Not Appearing in Langfuse

**Check configuration:**
```yaml
tracing:
  provider: langfuse
  public_key: ${LANGFUSE_PUBLIC_KEY}  # Must resolve to actual key
  secret_key: ${LANGFUSE_SECRET_KEY}  # Must resolve to actual key
  host: https://cloud.langfuse.com    # Correct host URL
```

**Check environment variables:**
```bash
echo $LANGFUSE_PUBLIC_KEY  # Should print pk-...
echo $LANGFUSE_SECRET_KEY  # Should print sk-...
```

**Check for validation errors in logs:**
```
Tracing configuration error, error="langfuse tracing requires public_key..."
```

**Verify SDK is installed:**
```bash
.venv/bin/python -c "import langfuse; print(langfuse.__version__)"
```

### Azure AI Tracing Not Working

**Check for OTEL conflict warning:**
```
Existing OpenTelemetry tracer detected - Azure AI tracing may conflict...
```

**Verify Azure Monitor package:**
```bash
.venv/bin/python -c "from azure.monitor.opentelemetry import configure_azure_monitor"
```

**Check connection string format:**
```bash
# Should start with "InstrumentationKey=" or "IngestionEndpoint="
echo $APPLICATIONINSIGHTS_CONNECTION_STRING
```

### Traces Not Correlating with Landscape

Tier 2 traces include `token_id` in metadata for correlation. To link a Langfuse trace to Landscape:

1. Find the trace in Langfuse
2. Look in the span metadata for `token_id` (Langfuse v3 uses W3C Trace Context IDs, not custom observation IDs)
3. Query Landscape:
   ```bash
   elspeth explain --run <run_id> --token <token_id>
   ```

**Note on Langfuse v3:** The SDK now uses W3C Trace Context compliant IDs. Custom observation IDs are no longer supported. Use `metadata.token_id` for correlation with the ELSPETH Landscape audit trail.

### High Latency from Tracing

**Symptoms:**
- Pipeline slows down after enabling Tier 2 tracing
- Timeout errors during Langfuse flush

**Solutions:**

1. **Check network latency to tracing endpoint:**
   ```bash
   curl -o /dev/null -s -w "%{time_total}\n" https://cloud.langfuse.com
   ```

2. **Use regional endpoint:**
   For Langfuse Cloud, use the endpoint in your region (EU or US).

3. **Consider self-hosted:**
   Deploy Langfuse in the same region as your pipeline.

4. **Reduce tracing scope:**
   Only enable Tier 2 for debugging, not production.

### Langfuse Flush Errors

**Symptoms:**
```
Failed to flush Langfuse tracing, error="..."
```

**Causes:**
- Network connectivity issues
- Invalid credentials
- Langfuse service unavailable

**Solutions:**
- Verify credentials
- Check Langfuse status page
- Errors are logged but don't crash the pipeline
