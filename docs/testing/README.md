# Testing Tools

This section documents testing utilities that ship with ELSPETH for load testing, stress testing, and fault injection.

## ChaosLLM

**ChaosLLM** is a fake LLM server for testing pipelines at scale without hitting real LLM APIs.

**Features:**
- OpenAI + Azure OpenAI compatible chat completion endpoints
- Error injection: rate limits (429), server errors (5xx), timeouts, disconnects, malformed JSON
- Latency simulation and burst patterns for AIMD testing
- Response modes: random, template (Jinja2), echo, preset bank (JSONL)
- SQLite metrics with MCP analysis tools

**Documentation:**
- [ChaosLLM User Guide](chaosllm.md) - Complete configuration reference
- [ChaosLLM MCP Server](chaosllm-mcp.md) - Analysis tools for Claude Code

### Quick Start

Start the server with a stress testing preset:

```bash
chaosllm serve --preset=stress_aimd
```

In another terminal, run your pipeline against it:

```bash
elspeth run --settings pipeline.yaml --execute
```

### CLI Examples

Run on custom port with 20% rate limit errors and 5% capacity errors:

```bash
chaosllm serve --port=9000 --rate-limit-pct=20 --capacity-529-pct=5
```

Generate structured JSON with Jinja2 templates:

```bash
chaosllm serve --response-mode=template
```

Use preset JSONL bank for deterministic responses:

```bash
chaosllm serve --response-mode=preset --config=./my-chaos.yaml
```

**Example preset config (JSONL bank):**

```yaml
response:
  mode: preset
  preset:
    file: "./examples/chaosllm/responses.jsonl"
    selection: sequential  # or "random"
```

### Pytest Fixture

```python
def test_pipeline_handles_errors(chaosllm_server):
    """ChaosLLM server auto-starts for the test."""
    # Configure your transform to use chaosllm_server.url
    transform = AzureMultiQueryLLMTransform({
        "endpoint": chaosllm_server.url,
        "api_key": "fake-key",
        ...
    })
```

### MCP Analysis

Analyze metrics from the last run:

```bash
chaosllm-mcp --database ./chaosllm-metrics.db
```

> **Note:** All `chaosllm` commands also work as `elspeth chaosllm`.

See the [full guide](chaosllm.md) for presets, configuration, and analysis tools.
