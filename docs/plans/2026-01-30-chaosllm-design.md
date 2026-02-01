# ChaosLLM: Fake LLM Server for Load Testing

**Date:** 2026-01-30
**Status:** Approved
**Branch:** `feature/chaosllm`

## Overview

ChaosLLM is a fake LLM server designed for testing ELSPETH pipelines at scale (10K-100K queries) without hitting real LLM APIs. It provides:

- **Load/stress testing** - Measure throughput, concurrency limits, bottlenecks
- **Integration testing at scale** - Run large pipelines without burning API credits
- **Fault injection** - Test error handling (rate limits, timeouts, malformed responses)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ChaosLLM Server                               │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   Starlette │    │   Error     │    │   Response              │  │
│  │   ASGI App  │───▶│   Injector  │───▶│   Generator             │  │
│  │             │    │             │    │   (random/template/bank)│  │
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
│   metrics.db    │◀────────│   MCP Server (lightweight, last run)    │
└─────────────────┘         └─────────────────────────────────────────┘
```

### Components

| Component | Responsibility |
|-----------|----------------|
| **Starlette ASGI App** | HTTP routing, OpenAI + Azure endpoint compatibility |
| **Error Injector** | Decides per-request: success, 429, 529, 500, timeout (based on config + burst state) |
| **Response Generator** | Produces response content (random/template/preset modes) |
| **Latency Simulator** | Adds configurable delay before responding |
| **Metrics Recorder** | Writes every request to SQLite (timestamp, latency, error type, etc.) |
| **MCP Server** | Lightweight query interface over metrics.db |

**Why Starlette (not FastAPI)?** Lighter weight, fewer dependencies, and we don't need FastAPI's automatic OpenAPI docs or Pydantic request validation - we're mimicking an API, not building one.

## API Endpoints

### LLM Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/openai/deployments/{deployment}/chat/completions` | POST | Azure OpenAI format |

### Admin Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/admin/config` | GET/POST | View or update error injection config at runtime |
| `/admin/stats` | GET | Quick summary stats (JSON) |
| `/admin/reset` | POST | Clear metrics, start fresh run |
| `/health` | GET | Liveness check |

### Request Format (OpenAI-compatible)

```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### Success Response (200)

```json
{
  "id": "fake-12345",
  "object": "chat.completion",
  "created": 1706644800,
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Generated response here"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}
}
```

## Error Injection

### HTTP-Level Errors

| Error Type | Code | Config Key | Use Case |
|------------|------|------------|----------|
| Rate limit | 429 | `rate_limit_pct` | Primary AIMD trigger |
| Model overloaded | 529 | `capacity_529_pct` | Azure-specific capacity |
| Service unavailable | 503 | `service_unavailable_pct` | Generic capacity |
| Bad gateway | 502 | `bad_gateway_pct` | Upstream failures |
| Gateway timeout | 504 | `gateway_timeout_pct` | Slow upstream |
| Internal error | 500 | `internal_error_pct` | Server bugs |
| Forbidden | 403 | `forbidden_pct` | Auth/permission issues |
| Not found | 404 | `not_found_pct` | Endpoint issues |

All error responses include `Retry-After` header (configurable value) for 429/529 responses.

### Connection-Level Failures

| Error Type | Config Key | Behavior |
|------------|------------|----------|
| Connection timeout | `timeout_pct` | Accept connection, never respond (triggers client timeout) |
| Connection reset | `connection_reset_pct` | Accept, then RST the TCP connection mid-stream |
| Slow response | `slow_response_pct` | Respond, but take 10-60s (configurable) |

### Malformed Response Errors

| Error Type | Config Key | Behavior |
|------------|------------|----------|
| Invalid JSON | `invalid_json_pct` | Return 200 with `{malformed json...` |
| Truncated response | `truncated_pct` | Return 200 with JSON cut off mid-stream |
| Wrong content-type | `wrong_content_type_pct` | Return 200 with `text/html` instead of `application/json` |
| Empty body | `empty_body_pct` | Return 200 with empty response body |
| Missing fields | `missing_fields_pct` | Return 200 with valid JSON but missing `choices` or `usage` |

### Burst Patterns

Bursts simulate real-world provider behavior where errors spike then recover:

```yaml
burst:
  enabled: true
  interval_sec: 30            # Time between bursts
  duration_sec: 5             # How long burst lasts
  rate_limit_pct: 80          # Rate limit during burst
  capacity_pct: 50            # 529 errors during burst
```

## Response Generation

### Mode A: Random Text

Generates random words/sentences. Best for pure throughput testing.

```yaml
response:
  mode: random
  random:
    min_words: 10
    max_words: 200
    vocabulary: english  # or "lorem"
```

### Mode B: Template/Echo

Returns structured responses based on templates. Useful when transforms parse JSON.

```yaml
response:
  mode: template
  template:
    body: |
      {"classification": "{{ random_choice(['positive', 'negative', 'neutral']) }}",
       "confidence": {{ random_float(0.7, 0.99) }},
       "summary": "{{ random_words(5, 15) }}"}
```

### Mode C: Preset Bank

Cycle through or randomly select from canned responses.

```yaml
response:
  mode: preset
  preset:
    file: responses.jsonl
    selection: random  # or "sequential"
```

### Per-Request Override

The `X-Fake-Response-Mode` header can override the server default:
```
X-Fake-Response-Mode: template
X-Fake-Template: {"result": "{{ random_choice(['A','B','C']) }}"}
```

## Metrics Storage (SQLite)

### Schema

```sql
-- Core request log (one row per request)
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    deployment TEXT,
    model TEXT,

    -- Outcome
    outcome TEXT NOT NULL,            -- 'success', 'error_injected', 'error_malformed'
    status_code INTEGER,
    error_type TEXT,

    -- Timing
    latency_ms REAL,
    injected_delay_ms REAL,

    -- Request details
    message_count INTEGER,
    prompt_tokens_approx INTEGER,

    -- Response details
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
    requests_client_error INTEGER,
    requests_connection_error INTEGER,
    requests_malformed INTEGER,
    avg_latency_ms REAL,
    p99_latency_ms REAL
);

-- Run metadata (single row, replaced on /admin/reset)
CREATE TABLE run_info (
    run_id TEXT PRIMARY KEY,
    started_utc TEXT NOT NULL,
    config_json TEXT NOT NULL,
    preset_name TEXT
);
```

## MCP Server (Claude-Optimized)

The MCP server is designed with Claude Code as the primary consumer - pre-computing insights rather than returning raw data.

### High-Level Analysis Tools

| Tool | Returns | Context Cost |
|------|---------|--------------|
| `diagnose()` | One-paragraph summary: total requests, success rate, top 3 error types, notable patterns | ~100 tokens |
| `analyze_aimd_behavior()` | Recovery times after each burst, backoff effectiveness, throughput degradation % | ~150 tokens |
| `analyze_errors()` | Grouped by category with counts, % of total, sample timestamps | ~120 tokens |
| `analyze_latency()` | p50/p95/p99, slow request count, correlation with error injection periods | ~80 tokens |
| `find_anomalies()` | Auto-detected unusual patterns, unexpected errors, throughput cliffs | ~100 tokens |

### Drill-Down Tools

| Tool | Returns | Use Case |
|------|---------|----------|
| `get_burst_events()` | Detected burst periods with before/during/after stats | "What happened during the second burst?" |
| `get_error_samples(error_type, limit=5)` | Sample requests for a specific error type | "Show me some rate limit examples" |
| `get_time_window(start_sec, end_sec)` | Stats for a specific time range | "What happened between t=40 and t=60?" |

### Raw Access

| Tool | Returns | Use Case |
|------|---------|----------|
| `query(sql)` | Raw SQL results (auto LIMIT 100) | Ad-hoc investigation |

### Example `diagnose()` Output

```json
{
  "summary": "50,000 requests over 180s (277 req/s avg). 89.2% success rate.",
  "error_breakdown": {
    "rate_limit_429": {"count": 3200, "pct": 6.4},
    "capacity_529": {"count": 1800, "pct": 3.6},
    "timeout": {"count": 400, "pct": 0.8}
  },
  "patterns_detected": [
    "Burst at t=45s: 80% rejection for 5s, recovery took 15s",
    "Burst at t=105s: 80% rejection for 5s, recovery took 8s (improved)",
    "Throughput stable at 270-290 req/s outside burst windows"
  ],
  "aimd_assessment": "AIMD throttle responding correctly - faster recovery on second burst indicates learning"
}
```

## Configuration

### Precedence (highest to lowest)

1. CLI flags (`--rate-limit-pct=25`)
2. YAML config file (`--config stress.yaml`)
3. Preset defaults (`--preset=stress-aimd`)
4. Built-in defaults

### Built-in Presets

| Preset | Purpose | Key Settings |
|--------|---------|--------------|
| `gentle` | Basic functionality testing | 2% errors, no bursts, 50ms latency |
| `realistic` | Mimics typical Azure behavior | 5% rate limit, 2% capacity, occasional bursts |
| `stress-aimd` | Specifically stress the AIMD throttle | 15% rate limit, frequent bursts (every 30s) |
| `chaos` | Everything breaks constantly | 25% errors across all types, malformed responses, timeouts |
| `silent` | Zero errors, maximum throughput | 0% errors, 10ms latency, for baseline measurements |

### Full YAML Config Structure

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  workers: 4

metrics:
  database: "./chaosllm-metrics.db"
  timeseries_bucket_sec: 1

response:
  mode: template
  random:
    min_words: 10
    max_words: 100
    vocabulary: english
  template:
    body: '{"result": "{{ random_choice([''A'',''B'',''C'']) }}"}'
  preset:
    file: "./responses.jsonl"
    selection: random

latency:
  base_ms: 50
  jitter_ms: 30

error_injection:
  # Capacity errors
  rate_limit_pct: 15
  capacity_529_pct: 5
  service_unavailable_pct: 2
  retry_after_sec: [1, 5]

  # Server errors
  internal_error_pct: 1
  bad_gateway_pct: 0.5
  gateway_timeout_pct: 0.5

  # Client errors
  forbidden_pct: 0.2
  not_found_pct: 0.1

  # Connection-level
  timeout_pct: 0.5
  timeout_sec: [30, 60]
  connection_reset_pct: 0.2
  slow_response_pct: 1
  slow_response_sec: [10, 30]

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

## CLI Usage

```bash
# Quick start with preset
elspeth chaosllm --preset=stress-aimd

# Preset with overrides
elspeth chaosllm --preset=realistic --rate-limit-pct=25 --port=9000

# Full custom config
elspeth chaosllm --config=my-scenario.yaml

# Config with CLI overrides
elspeth chaosllm --config=base.yaml --burst-interval=15 --burst-duration=10

# With MCP server (separate port)
elspeth chaosllm --preset=stress-aimd --port=8000 --mcp-port=8001

# MCP server only (for analyzing existing metrics.db)
elspeth chaosllm-mcp --database=./chaosllm-metrics.db
```

## Pytest Integration

### Fixture

```python
@pytest.fixture
def chaosllm_server(request, tmp_path):
    """Spin up ChaosLLM server for the test.

    Usage:
        def test_something(chaosllm_server):
            # Server is running at chaosllm_server.url
            # Default: gentle preset, random port

        @pytest.mark.chaosllm(preset="stress-aimd", rate_limit_pct=30)
        def test_stress(chaosllm_server):
            # Custom configuration via marker
    """
```

### Server Info Object

```python
@dataclass
class ChaosLLMServer:
    url: str                      # "http://127.0.0.1:54321"
    port: int
    metrics_db: Path
    admin_url: str

    def get_stats(self) -> dict:
        """Fetch current stats from /admin/stats."""

    def update_config(self, **kwargs) -> None:
        """Update error injection config mid-test."""

    def reset(self) -> None:
        """Clear metrics, start fresh."""

    def wait_for_requests(self, count: int, timeout: float = 30.0) -> None:
        """Block until N requests have been processed."""
```

### Usage Examples

```python
# Basic usage - gentle defaults
def test_pipeline_completes(chaosllm_server, recorder):
    transform = AzureMultiQueryLLMTransform({
        "endpoint": chaosllm_server.url,
        "deployment_name": "gpt-4",
        "api_key": "fake-key",
        "template": "Classify: {{ row.text }}",
        ...
    })
    # Run pipeline...
    assert result.status == "success"


# Stress test with custom config
@pytest.mark.chaosllm(preset="stress-aimd", burst_interval_sec=10)
def test_aimd_recovery(chaosllm_server, recorder):
    # Run 1000 requests through pooled executor
    ...
    stats = chaosllm_server.get_stats()
    assert stats["recovery_times_avg_sec"] < 15


# Mid-test config change
def test_degradation_handling(chaosllm_server, recorder):
    chaosllm_server.update_config(rate_limit_pct=5)
    # Run first batch...

    chaosllm_server.update_config(rate_limit_pct=50)
    # Run second batch, verify throttling kicks in...

    chaosllm_server.update_config(rate_limit_pct=5)
    # Verify throughput recovers...
```

## File Layout

```
src/elspeth/
├── testing/
│   ├── __init__.py
│   ├── chaosllm/
│   │   ├── __init__.py
│   │   ├── server.py                 # Starlette app, endpoints, request handling
│   │   ├── error_injector.py         # Error decision logic, burst state machine
│   │   ├── response_generator.py     # Random/template/preset response modes
│   │   ├── latency_simulator.py      # Delay injection
│   │   ├── metrics.py                # SQLite recorder, time-series aggregation
│   │   ├── config.py                 # Pydantic config models, preset definitions
│   │   └── presets/
│   │       ├── gentle.yaml
│   │       ├── realistic.yaml
│   │       ├── stress_aimd.yaml
│   │       ├── chaos.yaml
│   │       └── silent.yaml
│   └── chaosllm_mcp/
│       ├── __init__.py
│       └── server.py                 # MCP server with analysis tools

tests/
├── fixtures/
│   ├── __init__.py
│   └── chaosllm.py                   # pytest fixture
├── integration/
│   └── test_chaosllm_server.py       # Tests for ChaosLLM itself
└── stress/
    ├── __init__.py
    ├── test_aimd_stress.py           # AIMD throttle stress tests
    └── test_high_volume.py           # 10K-100K request tests
```

## Dependencies

All already in the project or lightweight:

| Package | Purpose |
|---------|---------|
| `starlette` | ASGI framework |
| `uvicorn` | ASGI server |
| `pydantic` | Config validation |
| `jinja2` | Template responses |
| `mcp` | MCP server |

No new dependencies required.

## Implementation Phases

| Phase | Deliverable | Enables |
|-------|-------------|---------|
| 1 | Core server with random responses + basic error injection | Basic load testing |
| 2 | Full error injection (bursts, connection failures, malformed) | AIMD stress testing |
| 3 | Response modes (template, preset bank) | Integration tests with JSON parsing |
| 4 | Metrics SQLite + MCP server | Analysis and debugging |
| 5 | Pytest fixture + presets | Easy test integration |
| 6 | CLI polish + documentation | Production-ready tool |

## Success Criteria

1. **Scale**: Successfully handle 100K requests in a single test run
2. **AIMD Stress**: Demonstrate throttle recovery behavior with burst injection
3. **Error Coverage**: All error types in `_is_retryable_error()` are testable
4. **Analysis**: MCP server provides actionable insights with single `diagnose()` call
5. **Integration**: Pytest fixture works seamlessly with existing test infrastructure
