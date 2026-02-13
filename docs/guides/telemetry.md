# Telemetry Guide

ELSPETH's telemetry system provides operational visibility into pipeline execution. It streams events to external observability platforms (Datadog, Azure Monitor, Jaeger, etc.) for real-time monitoring and alerting.

## Overview

### What Telemetry Provides

- **Real-time dashboards**: Watch pipeline progress as it happens
- **Alerting**: Get notified when pipelines fail or slow down
- **Performance metrics**: Identify slow transforms and bottlenecks
- **LLM usage tracking**: Monitor token usage and latency across providers

### Relationship to Landscape Audit Trail

| Aspect | Landscape | Telemetry |
|--------|-----------|-----------|
| **Purpose** | Legal record, complete lineage | Operational visibility |
| **Persistence** | Forever (audit requirement) | Ephemeral (observability retention) |
| **Completeness** | 100% - every operation recorded | Configurable granularity |
| **Source of truth** | Yes - for investigations | No - for monitoring |
| **Query interface** | SQL / MCP / `explain` CLI | External platform (Datadog, Grafana, etc.) |

**Key principle:** Landscape is the legal record. Telemetry complements it with operational visibility but never replaces it. When investigating issues, always verify findings against Landscape.

### When to Use Which

| Scenario | Use Landscape | Use Telemetry |
|----------|---------------|---------------|
| "Why did row 42 get routed to quarantine?" | Yes | No |
| "Are any pipelines currently failing?" | No | Yes |
| "What was the average LLM latency today?" | No | Yes |
| "Show complete lineage for this output" | Yes | No |
| "Alert me when error rate exceeds 5%" | No | Yes |
| "Reproduce the exact input to this LLM call" | Yes | No |

## Configuration

### Enabling Telemetry

Add a `telemetry` section to your pipeline settings:

```yaml
telemetry:
  enabled: true
  granularity: rows
  exporters:
    - name: console
      options:
        format: pretty
```

### Granularity Levels

Granularity controls which events are emitted:

| Level | Events Emitted | Typical Volume | Use Case |
|-------|----------------|----------------|----------|
| `lifecycle` | `RunStarted`, `RunFinished`, `PhaseChanged` | ~10-20 per run | Production (minimal overhead) |
| `rows` | Above + `RowCreated`, `TransformCompleted`, `GateEvaluated`, `TokenCompleted`, `FieldResolutionApplied` | N x M (rows x transforms) | Production (standard) |
| `full` | Above + `ExternalCallCompleted` with all details | High | Debugging, development |

**Performance guidance:**
- `lifecycle`: < 1% overhead
- `rows`: < 5% overhead
- `full`: < 10% overhead (LLM-heavy pipelines may vary)

### Backpressure Modes

When exporters can't keep up with event volume:

| Mode | Behavior | Trade-off |
|------|----------|-----------|
| `block` | Pause pipeline until exporters catch up | Complete telemetry, may slow pipeline |
| `drop` | Drop oldest events when buffer full | Fast pipeline, may lose events |

```yaml
telemetry:
  backpressure_mode: block  # Default - completeness over speed
```

**Recommendation:** Use `block` (default) unless you have strict latency requirements and can tolerate missing events.

### Failure Handling

```yaml
telemetry:
  # If true, crash the run when all exporters fail repeatedly
  # If false, log CRITICAL and continue without telemetry
  fail_on_total_exporter_failure: false  # Default
```

Telemetry uses aggregate logging to avoid "Warning Fatigue" - it logs every 100 dropped events, not every single one.

## Exporters

### Console Exporter

Writes events to stdout/stderr for testing and local debugging.

```yaml
telemetry:
  exporters:
    - name: console
      options:
        format: json    # json (default) | pretty
        output: stdout  # stdout (default) | stderr
```

**Example output (pretty format):**
```
[2026-01-30T10:15:30.123456] RunStarted: run-abc123 (config_hash=def456, source_plugin=csv)
[2026-01-30T10:15:30.234567] RowCreated: run-abc123 (row_id=row-001, token_id=tok-001)
```

**Example output (json format):**
```json
{"event_type": "RunStarted", "timestamp": "2026-01-30T10:15:30.123456", "run_id": "run-abc123", "config_hash": "def456", "source_plugin": "csv"}
```

### OTLP Exporter

Exports to any OpenTelemetry Protocol compatible backend (Jaeger, Tempo, Honeycomb, Grafana Cloud, etc.).

```yaml
telemetry:
  exporters:
    - name: otlp
      options:
        endpoint: ${OTEL_ENDPOINT}  # e.g., "http://localhost:4317"
        headers:
          Authorization: "Bearer ${OTEL_TOKEN}"
        batch_size: 100  # Events per batch (default: 100)
```

**Required dependency:**
```bash
uv pip install opentelemetry-exporter-otlp-proto-grpc
```

**Span mapping:**
- `span.name` = Event class name (e.g., "TransformCompleted")
- `span.trace_id` = Derived from `run_id` (consistent within run)
- `span.attributes` = All event fields
- `span.start_time` = Event timestamp (instant spans - events are points in time)

### Azure Monitor Exporter

Exports to Azure Application Insights for native Azure observability.

```yaml
telemetry:
  exporters:
    - name: azure_monitor
      options:
        connection_string: ${APPLICATIONINSIGHTS_CONNECTION_STRING}
        batch_size: 100  # Events per batch (default: 100)
        service_name: "my-pipeline"  # Service name in App Insights (default: "elspeth")
        service_version: "1.0.0"  # Service version (optional)
        deployment_environment: "production"  # Environment tag (optional)
```

**Required dependency:**
```bash
uv pip install azure-monitor-opentelemetry-exporter
```

**Azure-specific features:**
- Spans include `cloud.provider=azure` for filtering
- Full integration with Application Insights Distributed Tracing blade
- Compatible with Azure Monitor alerting rules
- Resource attributes (`service.name`, `service.version`, `deployment.environment`) for proper service identification

**Finding your connection string:**
1. Go to Azure Portal > Application Insights resource
2. Overview > Connection String (copy)
3. Set as environment variable: `export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=..."`

### Datadog Exporter

Exports to Datadog via the native ddtrace library for full APM integration.

```yaml
telemetry:
  exporters:
    - name: datadog
      options:
        service_name: elspeth-pipeline  # Default: "elspeth"
        env: production             # Default: "production"
        agent_host: localhost       # Default: "localhost"
        agent_port: 8126            # Default: 8126
        version: "1.0.0"            # Optional service version tag
```

**Required dependency:**
```bash
uv pip install ddtrace
```

**Datadog-specific features:**
- All event fields available as `elspeth.*` tags
- Native Datadog APM integration
- Works with local Datadog Agent

**Using with Datadog Agent (recommended):**
```bash
# Start Datadog Agent (Docker example)
docker run -d --name dd-agent \
  -e DD_API_KEY=${DD_API_KEY} \
  -e DD_APM_ENABLED=true \
  -p 8126:8126 \
  datadog/agent:latest
```

## Secrets Handling

**All secrets MUST come from environment variables, never hardcoded in config files.**

**Standard practice:** keep non-sensitive settings (endpoint, batch_size, service_name, env) in YAML; keep secrets (tokens, connection strings, API keys) in `.env` and reference them with `${VAR}`.

ELSPETH uses `${ENV_VAR}` substitution in pipeline config values:

```yaml
# settings.yaml - non-sensitive config only, secrets via env vars
telemetry:
  exporters:
    - name: otlp
      options:
        endpoint: http://localhost:4317
        headers:
          Authorization: "Bearer ${OTEL_TOKEN}"
```

```bash
# .env file (gitignored) - secrets for local development
OTEL_TOKEN=my-secret-token
```

| Secret Type | Environment Variable |
|-------------|---------------------|
| OTLP auth token | `OTEL_TOKEN` |
| Azure Monitor | `APPLICATIONINSIGHTS_CONNECTION_STRING` |
| Datadog Agent API key | `DD_API_KEY` (for the Datadog Agent process, not ELSPETH exporter options) |

## Correlation Workflow

### From External Alert to Landscape Explain

When an alert fires in your observability platform, follow this workflow to investigate:

1. **Extract `run_id` from the alert**

   Datadog example:
   ```
   Alert: ELSPETH pipeline failed
   Tags: elspeth.run_id:run-abc123, elspeth.event_type:RunFinished, status:failed
   ```

2. **Get failure context with Landscape MCP**

   ```bash
   # Start MCP server
   elspeth-mcp --database sqlite:///./runs/audit.db
   ```

   Then use `get_failure_context(run_id="run-abc123")` to see:
   - Failed node states
   - Error messages
   - Failed rows

3. **Investigate specific row lineage**

   If you need to see what happened to a specific row:
   ```bash
   elspeth explain --run run-abc123 --row row-001 --database ./runs/audit.db
   ```

   Or via MCP: `explain_token(run_id="run-abc123", row_id="row-001")`

4. **Review external call details**

   For LLM failures, check the `calls` table for request/response payloads.

### Example: Datadog Alert to Root Cause

**Scenario:** Datadog alert fires: "ELSPETH error rate > 5%"

1. **Click through to trace:**
   - Service: `elspeth-pipeline`
   - Span: `TransformCompleted`
   - Tag: `elspeth.status=failed`
   - Tag: `elspeth.run_id=run-xyz789`
   - Tag: `elspeth.node_id=llm_classifier`

2. **Query Landscape for details:**
   ```sql
   -- Via MCP query() tool
   SELECT state_id, status, error, input_hash, output_hash
   FROM node_states
   WHERE run_id = 'run-xyz789' AND node_id = 'llm_classifier' AND status = 'failed'
   LIMIT 10
   ```

3. **Get full error context:**
   ```
   explain_token(run_id="run-xyz789", row_id="row-042")
   ```

4. **Root cause identified:**
   - LLM returned malformed JSON
   - Transform correctly quarantined the row
   - Fix: Update prompt to be more explicit about JSON format

## Troubleshooting

### Events Not Appearing in Observability Platform

**Check telemetry is enabled:**
```yaml
telemetry:
  enabled: true  # Must be true
```

**Check granularity level:**
```yaml
telemetry:
  granularity: rows  # 'lifecycle' may not include the events you expect
```

**Check exporter configuration:**
- Verify endpoint is reachable
- Verify credentials are correct (check for typos in env var names)
- Check exporter logs for connection errors

**Enable console exporter for debugging:**
```yaml
telemetry:
  exporters:
    - name: console
      options:
        format: pretty
    - name: otlp
      options:
        endpoint: ${OTEL_ENDPOINT}
```

### Events Being Dropped

**Symptoms:**
- Log message: "Telemetry buffer overflow - events dropped"
- Missing events in observability platform

**Solutions:**
1. Reduce granularity: `full` -> `rows` -> `lifecycle`
2. Increase buffer size (if using custom BoundedBuffer)
3. Check exporter endpoint latency
4. Switch to `block` backpressure mode (if data completeness is critical)

### All Exporters Failing

**Symptoms:**
- Log message: "ALL telemetry exporters failing - events dropped"
- Log message (after 10 failures): "Telemetry disabled after repeated total failures"

**Solutions:**
1. Check network connectivity to all exporter endpoints
2. Verify all credentials are valid
3. Check exporter service status (Datadog Agent, OTLP collector, etc.)
4. Review exporter-specific logs

### Debug Mode

For detailed telemetry debugging, set log level to DEBUG:

```python
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))
```

This will show:
- Exporter configuration details
- Batch export counts
- Individual event processing

## Telemetry Health Metrics

The TelemetryManager exposes health metrics:

| Metric | Description | Warning Threshold |
|--------|-------------|-------------------|
| `events_emitted` | Successfully delivered events | N/A |
| `events_dropped` | Failed to deliver to any exporter | > 0 |
| `exporter_failures` | Per-exporter failure counts | Trend increasing |
| `consecutive_total_failures` | Current streak of all-exporter failures | > 5 |

These metrics are logged at pipeline shutdown and can be monitored via structured logging.

## Performance Tuning

| Scenario | Recommended Settings |
|----------|---------------------|
| Development/debugging | `granularity: full`, console exporter |
| Production (low volume) | `granularity: rows`, OTLP with batching |
| Production (high volume) | `granularity: lifecycle`, increase buffer |
| CI/CD pipelines | `enabled: false` or `granularity: lifecycle` |
| Debugging production issue | Temporarily enable `granularity: full` |

## Event Reference

### Lifecycle Events

| Event | When Emitted | Key Fields |
|-------|--------------|------------|
| `RunStarted` | Pipeline begins | `config_hash`, `source_plugin` |
| `RunFinished` | Pipeline finishes | `status`, `row_count`, `duration_ms` |
| `PhaseChanged` | Phase transition | `phase`, `action` |

### Row-Level Events

| Event | When Emitted | Key Fields |
|-------|--------------|------------|
| `RowCreated` | Source emits row | `row_id`, `token_id`, `content_hash` |
| `TransformCompleted` | Transform finishes | `node_id`, `plugin_name`, `status`, `duration_ms` |
| `GateEvaluated` | Gate routes row | `routing_mode`, `destinations` |
| `TokenCompleted` | Token reaches terminal | `outcome`, `sink_name` |

### External Call Events

| Event | When Emitted | Key Fields |
|-------|--------------|------------|
| `ExternalCallCompleted` | LLM/HTTP/SQL call finishes | `call_type`, `provider`, `status`, `latency_ms`, `token_usage` |
