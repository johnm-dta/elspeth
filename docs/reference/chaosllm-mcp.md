# ChaosLLM MCP Server

The ChaosLLM MCP server provides Claude-optimized analysis tools for investigating test results. Tools pre-compute insights and return concise summaries designed for LLM consumption.

## Starting the Server

```bash
# Auto-discover database in current directory
chaosllm-mcp

# Specify database explicitly
chaosllm-mcp --database ./chaosllm-metrics.db

# As Python module
python -m elspeth.testing.chaosllm_mcp.server --database ./metrics.db
```

> **Note:** All `chaosllm-mcp` commands also work as `elspeth chaosllm-mcp`.

## Tool Reference

### High-Level Analysis Tools

These tools provide pre-computed insights with ~100-150 token responses.

#### `diagnose()`

**First tool to call.** One-paragraph diagnostic summary.

**Returns:**
- Total requests and success rate
- Top 3 error types with counts
- Detected patterns (bursts, anomalies)
- AIMD throttle assessment

**Example output:**
```json
{
  "summary": "50000 requests, 89.2% success. Top errors: rate_limit (3200), capacity_529 (1800), timeout (400). Patterns: burst_detected, recovery_improving. AIMD: MODERATE",
  "status": "OK",
  "total_requests": 50000,
  "success_rate_pct": 89.2,
  "rate_limit_pct": 6.4,
  "top_errors": [["rate_limit", 3200], ["capacity_529", 1800], ["timeout", 400]],
  "patterns_detected": ["burst_detected", "recovery_improving"],
  "aimd_assessment": "MODERATE: Noticeable 429s, AIMD should be adjusting"
}
```

#### `analyze_aimd_behavior()`

Analyzes AIMD throttle effectiveness during test.

**Returns:**
- Detected burst periods
- Recovery time after each burst
- Throughput during vs outside bursts
- Assessment of AIMD effectiveness

**Example output:**
```json
{
  "burst_count": 3,
  "bursts": [
    {"start_sec": 45, "duration_sec": 5, "recovery_sec": 15, "rejection_rate": 0.82},
    {"start_sec": 105, "duration_sec": 5, "recovery_sec": 8, "rejection_rate": 0.78}
  ],
  "avg_recovery_sec": 11.5,
  "throughput_during_burst": 45.2,
  "throughput_outside_burst": 277.3,
  "assessment": "IMPROVING: Recovery times decreasing, AIMD adapting well"
}
```

#### `analyze_errors()`

Groups errors by category with counts and timing.

**Returns:**
- Errors grouped by type
- Count and percentage for each
- First/last occurrence timestamps
- Sample request IDs for investigation

**Example output:**
```json
{
  "total_errors": 5400,
  "error_rate_pct": 10.8,
  "by_type": {
    "rate_limit": {
      "count": 3200,
      "pct": 6.4,
      "first_seen": "2026-01-30T10:15:23Z",
      "last_seen": "2026-01-30T10:18:45Z",
      "sample_ids": ["req-abc123", "req-def456"]
    },
    "capacity_529": {
      "count": 1800,
      "pct": 3.6,
      "first_seen": "2026-01-30T10:15:45Z",
      "last_seen": "2026-01-30T10:18:30Z"
    }
  }
}
```

#### `analyze_latency()`

Latency statistics and correlation with errors.

**Returns:**
- p50, p95, p99 latency values
- Latency during error periods vs clean periods
- Slow request count and threshold

**Example output:**
```json
{
  "p50_ms": 52.3,
  "p95_ms": 89.7,
  "p99_ms": 145.2,
  "max_ms": 892.1,
  "slow_requests": 234,
  "slow_threshold_ms": 200,
  "latency_during_bursts_ms": 78.4,
  "latency_outside_bursts_ms": 51.2
}
```

#### `find_anomalies()`

Auto-detects unusual patterns that may indicate issues.

**Returns:**
- Unexpected error types
- Throughput cliffs (sudden drops)
- Latency spikes
- Configuration mismatches

**Example output:**
```json
{
  "anomalies": [
    {
      "type": "throughput_cliff",
      "time": "2026-01-30T10:16:30Z",
      "description": "Throughput dropped from 280 to 45 req/s suddenly",
      "severity": "HIGH"
    },
    {
      "type": "unexpected_error",
      "time": "2026-01-30T10:17:00Z",
      "description": "403 Forbidden errors (not configured for injection)",
      "severity": "MEDIUM"
    }
  ],
  "assessment": "2 anomalies detected - investigate throughput cliff"
}
```

### Drill-Down Tools

For investigating specific areas in detail.

#### `get_burst_events()`

Detailed burst period analysis.

**Returns:**
- All detected burst periods
- Before/during/after statistics for each
- Request counts and error rates

**Example output:**
```json
{
  "bursts": [
    {
      "index": 1,
      "start_utc": "2026-01-30T10:15:45Z",
      "end_utc": "2026-01-30T10:15:50Z",
      "duration_sec": 5,
      "stats": {
        "before": {"requests": 280, "success_rate": 0.95},
        "during": {"requests": 45, "success_rate": 0.18},
        "after": {"requests": 210, "success_rate": 0.72}
      },
      "recovery_sec": 15
    }
  ]
}
```

#### `get_error_samples(error_type, limit=5)`

Retrieve sample requests for a specific error type.

**Parameters:**
- `error_type`: Type of error to sample (e.g., "rate_limit", "capacity_529")
- `limit`: Maximum samples to return (default 5)

**Example:**
```
get_error_samples("rate_limit", limit=3)
```

**Returns:**
```json
{
  "error_type": "rate_limit",
  "samples": [
    {
      "request_id": "req-abc123",
      "timestamp_utc": "2026-01-30T10:15:47Z",
      "endpoint": "/v1/chat/completions",
      "status_code": 429,
      "latency_ms": 12.3
    }
  ]
}
```

#### `get_time_window(start_sec, end_sec)`

Statistics for a specific time range (seconds since run start).

**Parameters:**
- `start_sec`: Start of window in seconds since run start
- `end_sec`: End of window in seconds since run start

**Example:**
```
get_time_window(40, 60)  # Seconds 40-60 of the run
```

**Returns:**
```json
{
  "window": {"start_sec": 40, "end_sec": 60},
  "requests": 5230,
  "success_rate": 0.72,
  "avg_latency_ms": 67.8,
  "error_breakdown": {
    "rate_limit": 1200,
    "capacity_529": 250
  }
}
```

### Raw Access Tools

For ad-hoc investigation when the high-level tools aren't enough.

#### `query(sql)`

Execute read-only SQL against the metrics database.

**Parameters:**
- `sql`: SELECT query (automatically limited to 100 rows)

**Safety:**
- Only SELECT statements allowed
- DROP, DELETE, UPDATE, INSERT blocked
- Results automatically limited

**Example:**
```sql
SELECT endpoint, COUNT(*) as count, AVG(latency_ms) as avg_latency
FROM requests
WHERE status_code = 429
GROUP BY endpoint
```

#### `describe_schema()`

Returns the database schema for writing custom queries.

**Returns:**
- Table names
- Column definitions with types
- Primary keys and foreign keys

## Common Workflows

### Quick Diagnostics

```
1. diagnose()           → Overview of what happened
2. analyze_errors()     → If error rate is high
3. analyze_aimd_behavior() → If testing AIMD
```

### Investigating a Failure

```
1. diagnose()                    → Identify the problem area
2. find_anomalies()              → Check for unexpected issues
3. get_error_samples("error_type") → Get specific examples
4. get_time_window(start, end)   → Focus on problem period
```

### AIMD Stress Test Analysis

```
1. diagnose()                → Overall health check
2. analyze_aimd_behavior()   → Recovery times and effectiveness
3. get_burst_events()        → Detailed burst-by-burst analysis
4. analyze_latency()         → Latency impact during throttling
```

### Custom Investigation

```
1. describe_schema()    → Understand available data
2. query("SELECT ...")  → Custom analysis
```

## Claude Code Integration

The MCP server is designed for Claude Code's context constraints:

- **Concise outputs** - High-level tools return ~100-150 tokens
- **Pre-computed insights** - Analysis done server-side, not in context
- **Progressive detail** - Start with `diagnose()`, drill down as needed
- **Structured JSON** - Easy to parse and reference

### Example Claude Code Session

```
User: "The AIMD stress test failed. What went wrong?"

Claude: Let me analyze the test results.
[calls diagnose()]

The test ran 50,000 requests with 89.2% success rate. Top errors were rate_limit (3200)
and capacity_529 (1800). AIMD assessment shows "MODERATE" pressure.

Let me check the AIMD behavior specifically.
[calls analyze_aimd_behavior()]

I see 3 burst events. Recovery times improved from 15s to 8s across bursts, indicating
AIMD is adapting. However, the first recovery took longer than expected.

[calls get_burst_events()]

The first burst at t=45s had 82% rejection rate and took 15s to recover. Subsequent
bursts recovered faster (8-10s). This suggests the AIMD parameters may need tuning
for faster initial adaptation.
```

## Database Location

The MCP server needs access to the SQLite metrics database created by ChaosLLM:

```bash
# Default location (when using chaosllm serve)
./chaosllm-metrics.db

# Custom location
chaosllm serve --metrics-db=/path/to/metrics.db
chaosllm-mcp --database=/path/to/metrics.db

# In pytest fixtures
chaosllm_server.metrics_db  # Returns Path to the database
```
