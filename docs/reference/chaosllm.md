# ChaosLLM User Guide

ChaosLLM is a fake LLM server for load testing, stress testing, and fault injection. It provides OpenAI and Azure OpenAI compatible endpoints with configurable error injection, response generation, and metrics recording.

## Use Cases

- **Load testing** - Measure throughput and find bottlenecks without API costs
- **Stress testing** - Test AIMD throttle behavior under burst error conditions
- **Fault injection** - Verify error handling for rate limits, timeouts, malformed responses
- **Integration testing** - Run large-scale tests (10K-100K requests) without burning credits

## Quick Start

### CLI

```bash
# Start with gentle defaults (2% errors, 50ms latency)
chaosllm serve

# Start with a preset for stress testing
chaosllm serve --preset=stress_aimd

# List available presets
chaosllm presets

# Custom port and error rates
chaosllm serve --port=9000 --rate-limit-pct=20
```

> **Note:** All `chaosllm` commands also work as `elspeth chaosllm`.

### Pytest Fixture

```python
def test_my_pipeline(chaosllm_server):
    """Server starts automatically, isolated per test."""
    response = chaosllm_server.post_completion()
    assert response.status_code == 200

@pytest.mark.chaosllm(preset="stress_aimd")
def test_under_stress(chaosllm_server):
    """Use marker for custom configuration."""
    stats = chaosllm_server.get_stats()
    # ... run pipeline and check results
```

## API Endpoints

### LLM Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/openai/deployments/{deployment}/chat/completions` | POST | Azure OpenAI format |
| `/health` | GET | Liveness check |

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/config` | GET | View current configuration |
| `/admin/config` | POST | Update configuration at runtime |
| `/admin/stats` | GET | Request statistics summary |
| `/admin/reset` | POST | Clear metrics, start fresh run |
| `/admin/export` | GET | Export raw metrics data for external analysis |

> **Security Notice:** All admin endpoints are unauthenticated. ChaosLLM is a test
> tool intended for local or trusted-network use only. Do not expose admin endpoints
> to untrusted networks. There are no CORS restrictions or rate limits on admin
> endpoints. When running on shared infrastructure, bind to `127.0.0.1` (the default).

### Request Format

Standard OpenAI chat completion format:

```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### Success Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1706644800,
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Generated response"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}
}
```

## Presets

Built-in configurations for common testing scenarios:

| Preset | Purpose | Error Rate | Latency | Bursts |
|--------|---------|------------|---------|--------|
| `gentle` | Basic functionality testing | 2% total | 50ms | No |
| `realistic` | Mimics typical Azure behavior | ~7% total | 80ms | Occasional |
| `stress_aimd` | Stress test AIMD throttle | 15% rate limit | 50ms | Every 30s |
| `chaos` | Everything breaks constantly | 25%+ total | Variable | Frequent |
| `silent` | Zero errors, baseline throughput | 0% | 10ms | No |

```bash
# Use a preset
chaosllm serve --preset=stress_aimd

# Override preset values
chaosllm serve --preset=realistic --rate-limit-pct=25
```

## Error Injection

### HTTP Errors

| Error Type | Status | Config Key | Typical Use |
|------------|--------|------------|-------------|
| Rate limit | 429 | `rate_limit_pct` | Primary AIMD trigger |
| Model overloaded | 529 | `capacity_529_pct` | Azure-specific capacity |
| Service unavailable | 503 | `service_unavailable_pct` | Generic capacity |
| Bad gateway | 502 | `bad_gateway_pct` | Upstream failures |
| Gateway timeout | 504 | `gateway_timeout_pct` | Slow upstream |
| Internal error | 500 | `internal_error_pct` | Server bugs |

Rate limit (429) and capacity (529) errors include `Retry-After` header.

### Malformed Responses

Test your JSON parsing and error handling:

| Type | Config Key | Behavior |
|------|------------|----------|
| Invalid JSON | `invalid_json_pct` | Returns `{malformed...` |
| Truncated | `truncated_pct` | JSON cut off mid-stream |
| Empty body | `empty_body_pct` | 200 with empty response |
| Missing fields | `missing_fields_pct` | Valid JSON, no `choices` |
| Wrong content-type | `wrong_content_type_pct` | Returns `text/html` |

### Burst Patterns

Simulate real-world provider behavior where errors spike then recover:

```yaml
burst:
  enabled: true
  interval_sec: 30      # Time between bursts
  duration_sec: 5       # How long each burst lasts
  rate_limit_pct: 80    # Rate limit % during burst
  capacity_pct: 50      # 529 error % during burst
```

Bursts are essential for testing AIMD throttle recovery behavior.

## Response Modes

### Random Mode

Generates random words/sentences. Best for pure throughput testing.

```yaml
response:
  mode: random
  random:
    min_words: 10
    max_words: 200
    vocabulary: english  # or "lorem"
```

### Template Mode

Returns structured responses using Jinja2 templates. Useful when your pipeline parses JSON from LLM responses.

```yaml
response:
  mode: template
  template:
    body: |
      {"classification": "{{ random_choice(['positive', 'negative', 'neutral']) }}",
       "confidence": {{ random_float(0.7, 0.99) }},
       "summary": "{{ random_words(5, 15) }}"}
```

Available template functions:
- `random_choice(list)` - Pick random item from list
- `random_float(min, max)` - Random float in range
- `random_int(min, max)` - Random integer in range
- `random_words(min, max)` - Random word count

### Echo Mode

Reflects input back, useful for debugging:

```yaml
response:
  mode: echo
```

### Preset Mode

Returns responses from a JSONL preset bank (one JSON object per line). Each
object must have a `content` field with the response text.

Use this mode for deterministic responses in regression tests.

```yaml
response:
  mode: preset
  preset:
    file: "./examples/chaosllm/responses.jsonl"
    selection: sequential  # or "random"
```

**Example JSONL file** (`responses.jsonl`):

```jsonl
{"content": "{\"category\": \"fraud\", \"confidence\": 0.95}"}
{"content": "{\"category\": \"legitimate\", \"confidence\": 0.88}"}
```

Selection modes:
- `sequential`: Cycle through responses in order
- `random`: Pick randomly from the bank

### Per-Request Override

Override the server default for specific requests using headers:

```
X-Fake-Response-Mode: template
X-Fake-Template: {"result": "{{ random_choice(['A','B','C']) }}"}
```

## Configuration

### Precedence (highest to lowest)

1. CLI flags (`--rate-limit-pct=25`)
2. YAML config file (`--config=stress.yaml`)
3. Preset defaults (`--preset=stress_aimd`)
4. Built-in defaults

### Full YAML Structure

```yaml
server:
  host: "127.0.0.1"
  port: 8000

metrics:
  database: "./chaosllm-metrics.db"
  timeseries_bucket_sec: 1

response:
  mode: template  # random, template, echo, preset
  random:
    min_words: 10
    max_words: 100
    vocabulary: english
  template:
    body: '{"result": "{{ random_choice([''A'',''B'',''C'']) }}"}'

latency:
  base_ms: 50
  jitter_ms: 30

error_injection:
  # Capacity errors (trigger throttling)
  rate_limit_pct: 15
  capacity_529_pct: 5
  service_unavailable_pct: 2
  retry_after_sec: [1, 5]  # Random range for Retry-After header

  # Server errors
  internal_error_pct: 1
  bad_gateway_pct: 0.5
  gateway_timeout_pct: 0.5

  # Malformed responses
  invalid_json_pct: 0.3
  truncated_pct: 0.2
  empty_body_pct: 0.2
  missing_fields_pct: 0.3

  # Burst configuration
  burst:
    enabled: true
    interval_sec: 30
    duration_sec: 5
    rate_limit_pct: 80
    capacity_pct: 50
```

## Pytest Fixture Reference

### Basic Usage

```python
def test_basic(chaosllm_server):
    """Server auto-starts with gentle preset."""
    # Make requests to chaosllm_server.url
    response = requests.post(
        f"{chaosllm_server.url}/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
    )
    assert response.status_code == 200
```

### Configuration via Marker

```python
@pytest.mark.chaosllm(preset="stress_aimd")
def test_stress(chaosllm_server):
    """Use stress_aimd preset."""
    pass

@pytest.mark.chaosllm(rate_limit_pct=50, burst_enabled=True)
def test_custom(chaosllm_server):
    """Custom error rates."""
    pass
```

### Fixture Object

The `chaosllm_server` fixture provides:

```python
chaosllm_server.url          # "http://127.0.0.1:54321"
chaosllm_server.port         # 54321
chaosllm_server.metrics_db   # Path to SQLite metrics database
chaosllm_server.admin_url    # "http://127.0.0.1:54321/admin"

# Methods
chaosllm_server.get_stats()              # Fetch /admin/stats
chaosllm_server.update_config(**kwargs)  # POST to /admin/config
chaosllm_server.reset()                  # POST to /admin/reset
chaosllm_server.wait_for_requests(n)     # Block until n requests processed

# Convenience methods for making requests
chaosllm_server.post_completion(messages=None, model="gpt-4")
chaosllm_server.post_azure_completion(deployment, messages=None)
```

### Runtime Configuration Changes

```python
def test_degradation_recovery(chaosllm_server):
    # Start with low error rate
    chaosllm_server.update_config(rate_limit_pct=5)
    # ... run first batch, verify success

    # Spike errors
    chaosllm_server.update_config(rate_limit_pct=50)
    # ... run second batch, verify throttling

    # Recover
    chaosllm_server.update_config(rate_limit_pct=5)
    # ... verify throughput recovers
```

## Metrics Database

ChaosLLM records every request to SQLite for analysis.

### Schema

```sql
-- One row per request
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    deployment TEXT,
    model TEXT,
    outcome TEXT NOT NULL,      -- 'success', 'error_injected', 'error_malformed'
    status_code INTEGER,
    error_type TEXT,
    latency_ms REAL,
    injected_delay_ms REAL,
    message_count INTEGER,
    prompt_tokens_approx INTEGER,
    response_tokens INTEGER,
    response_mode TEXT
);

-- Time-series aggregates (one row per second)
CREATE TABLE timeseries (
    bucket_utc TEXT PRIMARY KEY,
    requests_total INTEGER,
    requests_success INTEGER,
    requests_rate_limited INTEGER,
    requests_capacity_error INTEGER,
    requests_server_error INTEGER,
    requests_malformed INTEGER,
    avg_latency_ms REAL,
    p99_latency_ms REAL
);

-- Run metadata
CREATE TABLE run_info (
    run_id TEXT PRIMARY KEY,
    started_utc TEXT NOT NULL,
    config_json TEXT NOT NULL,
    preset_name TEXT
);
```

### Direct Query

```bash
sqlite3 chaosllm-metrics.db "SELECT outcome, COUNT(*) FROM requests GROUP BY outcome"
```

## MCP Analysis Server

For Claude Code integration, ChaosLLM includes an MCP server with pre-computed analysis tools.

```bash
# Start MCP server (typically done automatically by Claude Code)
chaosllm-mcp --database=./chaosllm-metrics.db
```

See [ChaosLLM MCP Server](chaosllm-mcp.md) for the full tool reference.

### Key Tools

| Tool | Returns |
|------|---------|
| `diagnose()` | One-paragraph summary: success rate, top errors, patterns |
| `analyze_aimd_behavior()` | Recovery times, backoff effectiveness |
| `analyze_errors()` | Errors grouped by type with counts and samples |
| `get_burst_events()` | Detected burst periods with before/during/after stats |

### Example `diagnose()` Output

```json
{
  "summary": "50,000 requests over 180s (277 req/s avg). 89.2% success rate.",
  "error_breakdown": {
    "rate_limit_429": {"count": 3200, "pct": 6.4},
    "capacity_529": {"count": 1800, "pct": 3.6}
  },
  "patterns_detected": [
    "Burst at t=45s: 80% rejection for 5s, recovery took 15s",
    "Burst at t=105s: 80% rejection for 5s, recovery took 8s (improved)"
  ],
  "aimd_assessment": "AIMD throttle responding correctly"
}
```

## Common Scenarios

### Testing AIMD Throttle Recovery

```python
@pytest.mark.chaosllm(preset="stress_aimd")
def test_aimd_recovery(chaosllm_server, my_pipeline):
    """Verify throttle recovers after burst errors."""
    # Run pipeline with many concurrent requests
    results = my_pipeline.run(rows=1000, concurrency=50)

    # Check recovery behavior
    stats = chaosllm_server.get_stats()
    assert stats["success_rate"] > 0.80  # Should recover despite bursts
```

### Testing Malformed Response Handling

```python
@pytest.mark.chaosllm(
    invalid_json_pct=10,
    truncated_pct=10,
    missing_fields_pct=10
)
def test_malformed_handling(chaosllm_server, my_transform):
    """Verify transform handles malformed responses gracefully."""
    results = [my_transform.process(row) for row in test_rows]

    # Should quarantine bad responses, not crash
    errors = [r for r in results if r.is_error]
    assert all("malformed" in e.error_type or "parse" in e.error_type for e in errors)
```

### Baseline Throughput Measurement

```python
@pytest.mark.chaosllm(preset="silent")
def test_baseline_throughput(chaosllm_server, benchmark):
    """Measure throughput without any error injection."""
    def run_batch():
        for _ in range(100):
            chaosllm_server.post_completion()

    result = benchmark(run_batch)
    assert result.stats.mean < 0.5  # 100 requests in < 500ms
```

### Mid-Test Configuration Changes

```python
def test_progressive_degradation(chaosllm_server):
    """Test behavior as error rate increases."""
    for error_rate in [5, 15, 30, 50]:
        chaosllm_server.update_config(rate_limit_pct=error_rate)
        chaosllm_server.reset()  # Fresh metrics

        # Run workload
        for _ in range(100):
            chaosllm_server.post_completion()

        stats = chaosllm_server.get_stats()
        print(f"Error rate {error_rate}%: {stats['success_rate']*100:.1f}% success")
```

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Use a different port
chaosllm serve --port=9000
```

### No Metrics Recorded

```bash
# Check database exists and has data
sqlite3 chaosllm-metrics.db "SELECT COUNT(*) FROM requests"

# Check server logs for errors
chaosllm serve --log-level=debug
```

### Fixture Isolation Issues

Each test gets a fresh server instance with isolated metrics. If you see cross-test contamination:

```python
def test_isolation(chaosllm_server):
    # Always reset at start of test if needed
    chaosllm_server.reset()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ChaosLLM Server                               │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   Starlette │    │   Error     │    │   Response              │  │
│  │   ASGI App  │───▶│   Injector  │───▶│   Generator             │  │
│  │             │    │             │    │   (random/template/echo)│  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
│         │                                          │                 │
│         ▼                                          ▼                 │
│  ┌─────────────┐                        ┌─────────────────────────┐ │
│  │   Metrics   │                        │   Latency               │ │
│  │   Recorder  │                        │   Simulator             │ │
│  │   (SQLite)  │                        │                         │ │
│  └─────────────┘                        └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐         ┌─────────────────────────────────────────┐
│   metrics.db    │◀────────│   MCP Server (analysis tools)           │
└─────────────────┘         └─────────────────────────────────────────┘
```

**Why Starlette?** Lightweight, minimal dependencies, and we're mimicking an API rather than building one - no need for FastAPI's OpenAPI generation or request validation.
