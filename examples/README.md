# ELSPETH Examples

This directory contains runnable pipeline examples demonstrating ELSPETH's features. Each example has its own `settings.yaml` and a `README.md` explaining what it demonstrates.

## Quick Start

```bash
# Run any example from the repository root
elspeth run --settings examples/<name>/settings.yaml --execute

# Explore the audit trail after a run
elspeth explain --run latest --database examples/<name>/runs/audit.db
```

## Example Index

### Pure Data Processing (no external APIs)

These examples run locally with no credentials or external services.

| Example | What It Demonstrates |
|---------|---------------------|
| [`threshold_gate`](threshold_gate/) | Simplest gate — one numeric threshold, two sinks |
| [`boolean_routing`](boolean_routing/) | Gate routing based on a string field value |
| [`explicit_routing`](explicit_routing/) | Declarative `on_success`/`input` wiring pattern |
| [`error_routing`](error_routing/) | `on_error` diversion to quarantine sinks |
| [`deep_routing`](deep_routing/) | 5 chained gates, 3 transforms, 7 sinks — complex decision tree |
| [`fork_coalesce`](fork_coalesce/) | Fork/join DAG pattern — parallel paths merged with configurable policy (includes ARCH-15 per-branch transforms variant) |
| [`batch_aggregation`](batch_aggregation/) | Count-triggered aggregation with group-by statistics |
| [`deaggregation`](deaggregation/) | 1-to-N row expansion via `batch_replicate` |
| [`json_explode`](json_explode/) | Expand nested JSON arrays into individual rows |
| [`database_sink`](database_sink/) | Write pipeline output to a SQLite database |
| [`checkpoint_resume`](checkpoint_resume/) | Crash recovery via checkpointing and `elspeth resume` |
| [`retention_purge`](retention_purge/) | Payload retention lifecycle and `elspeth purge` |
| [`audit_export`](audit_export/) | Export the Landscape audit trail to JSON |
| [`landscape_journal`](landscape_journal/) | Event journaling for real-time audit monitoring |
| [`schema_contracts_demo`](schema_contracts_demo/) | DAG-time schema validation (`guaranteed_fields` / `required_input_fields`) |
| [`large_scale_test`](large_scale_test/) | Performance testing with large datasets |
| [`threshold_gate_container`](threshold_gate_container/) | Docker-packaged pipeline deployment |

### OpenRouter LLM (real API — requires `OPENROUTER_API_KEY`)

These examples call the real OpenRouter API. Set your API key first:

```bash
export OPENROUTER_API_KEY="your-key-from-openrouter.ai"
```

| Example | What It Demonstrates |
|---------|---------------------|
| [`openrouter_sentiment`](openrouter_sentiment/) | Single-query sentiment analysis (sequential, pooled, and batched modes) |
| [`openrouter_multi_query_assessment`](openrouter_multi_query_assessment/) | Multi-query matrix (case studies x criteria) with stress/overflow variants |
| [`schema_contracts_llm_assessment`](schema_contracts_llm_assessment/) | LLM pipeline with DAG-time schema contract validation |
| [`template_lookups`](template_lookups/) | Jinja2 template-driven prompts with field extraction |

### Azure (requires Azure credentials)

| Example | What It Demonstrates |
|---------|---------------------|
| [`azure_openai_sentiment`](azure_openai_sentiment/) | Azure OpenAI endpoint (sequential and pooled) |
| [`azure_blob_sentiment`](azure_blob_sentiment/) | Azure Blob Storage source with LLM processing |
| [`azure_keyvault_secrets`](azure_keyvault_secrets/) | Secret resolution from Azure Key Vault |
| [`multi_query_assessment`](multi_query_assessment/) | Azure-backed multi-query assessment matrix |

### ChaosLLM / ChaosWeb (local fault injection — no API keys needed)

These examples use ELSPETH's built-in fault injection servers to test pipeline resilience without real API credentials.

| Example | What It Demonstrates |
|---------|---------------------|
| [`chaosllm_sentiment`](chaosllm_sentiment/) | Sentiment analysis against ChaosLLM (mirrors `openrouter_sentiment`) |
| [`chaosllm_endurance`](chaosllm_endurance/) | Multi-query endurance test with fault injection |
| [`rate_limited_llm`](rate_limited_llm/) | LLM pipeline with rate limiting (30 req/min cap) |
| [`chaosweb`](chaosweb/) | Web scraping resilience with ChaosWeb fault injection |
| [`chaosllm`](chaosllm/) | Response data used by ChaosLLM server (not a runnable pipeline) |

---

## If You Want to See...

| You want to learn about... | Look at... |
|---------------------------|-----------|
| **How wiring works** | [`explicit_routing`](explicit_routing/) — the canonical minimal example |
| **Simple routing** | [`threshold_gate`](threshold_gate/) or [`boolean_routing`](boolean_routing/) |
| **Complex decision trees** | [`deep_routing`](deep_routing/) — 5 gates, 7 sinks, 8-node-deep DAG |
| **Fork/join patterns** | [`fork_coalesce`](fork_coalesce/) — parallel paths with merge policies |
| **Error handling / quarantine** | [`error_routing`](error_routing/) — `on_error` diversion pattern |
| **Aggregation (N to 1)** | [`batch_aggregation`](batch_aggregation/) — count triggers, group-by stats |
| **Deaggregation (1 to N)** | [`deaggregation`](deaggregation/) or [`json_explode`](json_explode/) |
| **LLM integration (quick start)** | [`openrouter_sentiment`](openrouter_sentiment/) — simplest real LLM pipeline |
| **LLM without API keys** | [`chaosllm_sentiment`](chaosllm_sentiment/) — same pipeline, local ChaosLLM server |
| **Multi-query LLM matrices** | [`openrouter_multi_query_assessment`](openrouter_multi_query_assessment/) — case studies x criteria |
| **Pooled/concurrent execution** | [`openrouter_sentiment`](openrouter_sentiment/) — has `settings_pooled.yaml` variant |
| **Batch aggregation + LLM** | [`openrouter_sentiment`](openrouter_sentiment/) — has `settings_batched.yaml` variant |
| **Rate limiting** | [`rate_limited_llm`](rate_limited_llm/) — throttled API calls with ChaosLLM |
| **Schema contracts** | [`schema_contracts_demo`](schema_contracts_demo/) (pure data) or [`schema_contracts_llm_assessment`](schema_contracts_llm_assessment/) (with LLM) |
| **Jinja2 templates** | [`template_lookups`](template_lookups/) — field extraction and template-driven prompts |
| **Web scraping** | [`chaosweb`](chaosweb/) — fault-injected scraping with content gates |
| **Database output** | [`database_sink`](database_sink/) — write to SQLite (or PostgreSQL/MySQL) |
| **Crash recovery / resume** | [`checkpoint_resume`](checkpoint_resume/) — checkpoint + Ctrl-C + `elspeth resume` |
| **Graceful shutdown** | [`checkpoint_resume`](checkpoint_resume/) — covers Ctrl-C shutdown behaviour |
| **Payload retention** | [`retention_purge`](retention_purge/) — payload lifecycle and `elspeth purge` |
| **Audit trail export** | [`audit_export`](audit_export/) — JSON export with optional signing |
| **Event journaling** | [`landscape_journal`](landscape_journal/) — real-time audit event stream |
| **Azure integration** | [`azure_openai_sentiment`](azure_openai_sentiment/), [`azure_blob_sentiment`](azure_blob_sentiment/), [`azure_keyvault_secrets`](azure_keyvault_secrets/) |
| **Docker deployment** | [`threshold_gate_container`](threshold_gate_container/) — containerised pipeline |
| **Retry under faults** | [`chaosllm_endurance`](chaosllm_endurance/) — 5 retries with exponential backoff against ChaosLLM |
| **Stress testing** | [`large_scale_test`](large_scale_test/) or [`chaosllm_endurance`](chaosllm_endurance/) |
