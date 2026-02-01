# ELSPETH Feature Inventory - January 29, 2026

**Purpose:** Complete inventory of what ELSPETH actually does today, reconciled against the original architecture.md (Jan 12) and requirements.md (Jan 22).

**This document is the truth.** If code exists, it's listed. If it's listed but doesn't exist, that's a bug in this document.

---

## EVOLUTION SUMMARY

| Aspect | Original Vision (Jan 12) | Current Reality (Jan 29) |
|--------|-------------------------|-------------------------|
| Execution Model | Linear pipeline with gates | Full DAG with fork/coalesce |
| Routing | continue, route_to_sink | continue, route_to_sink, fork_to_paths |
| Aggregation | "accumulate state until trigger" | Count, timeout, condition triggers with flush |
| Token Identity | row_id sufficient | row_id + token_id + parent_token_id |
| LLM Integration | "Phase 6 future work" | 6 LLM transforms, structured outputs, batch API |
| Analysis Tools | "elspeth explain" | MCP server with 20+ analysis tools |
| Field Handling | Assumed clean input | Full normalization pipeline with collision detection |

---

## 1. PLUGIN INVENTORY

### 1.1 Sources (4)

| Plugin Name | Config Key | Purpose | Schema Modes | Special Features |
|-------------|-----------|---------|--------------|------------------|
| CSV Source | `csv` | Load from CSV files | dynamic, strict, free | Multiline quoted fields, delimiter config, skip rows, field normalization |
| JSON Source | `json` | Load from JSON/JSONL | dynamic, strict, free | Auto-format detection, data_key extraction, field normalization |
| Null Source | `null` | Yield no rows | N/A | For resume operations |
| Azure Blob Source | `azure_blob` | Load from Azure Blob Storage | dynamic, strict, free | SAS/connection string auth, path templating, format detection |

**Source Capabilities:**
- Schema coercion at Tier 3 boundary (external data)
- Validation with quarantine routing (`on_validation_failure`)
- Field normalization with collision detection
- Guaranteed fields declaration for downstream validation

### 1.2 Transforms (18)

#### Core Transforms (8)

| Plugin Name | Config Key | Type | I/O | Purpose |
|-------------|-----------|------|-----|---------|
| Passthrough | `passthrough` | Row | 1→1 | Pass rows unchanged (testing/debugging) |
| Field Mapper | `field_mapper` | Row | 1→1 | Rename/remap/select fields |
| Truncate | `truncate` | Row | 1→1 | Limit string field lengths |
| Keyword Filter | `keyword_filter` | Row | 1→1 | Filter rows by regex patterns |
| JSON Explode | `json_explode` | Row (forking) | 1→N | Expand JSON arrays to multiple rows |
| Batch Stats | `batch_stats` | Batch-aware | N→1 | Compute statistics over batch |
| Batch Replicate | `batch_replicate` | Batch-aware | N→M | Replicate rows N times |

#### Azure Transforms (2)

| Plugin Name | Config Key | Type | Purpose |
|-------------|-----------|------|---------|
| Azure Content Safety | `azure_content_safety` | LLM | Detect harmful content |
| Azure Prompt Shield | `azure_prompt_shield` | LLM | Detect prompt injection |

#### LLM Transforms (6)

| Plugin Name | Config Key | Type | Provider | Features |
|-------------|-----------|------|----------|----------|
| Azure LLM | `azure_llm` | LLM | Azure OpenAI | Single-row classification, template prompts |
| Azure Batch LLM | `azure_batch` | Batch LLM | Azure OpenAI Batch API | 50% cost savings, async processing |
| Azure Multi-Query | `azure_multi_query` | Batch LLM | Azure OpenAI | Multiple prompts per row, structured output |
| OpenRouter LLM | `openrouter` | LLM | OpenRouter (100+ providers) | Multi-provider access |
| OpenRouter Multi-Query | `openrouter_multi_query` | Batch LLM | OpenRouter | Multiple prompts, structured output |
| Multi-Query (Generic) | `multi_query` | LLM | Configurable | Cross-product evaluation |

**LLM Capabilities:**
- Jinja2 template-based prompting
- Structured output (JSON schema) via ResponseFormat.STRUCTURED
- Automatic retry with exponential backoff
- Rate limiting via pyrate-limiter
- AIMD throttle for adaptive rate control
- Reorder buffer for out-of-order completion
- Full request/response audit recording

### 1.3 Sinks (4)

| Plugin Name | Config Key | Purpose | Features |
|-------------|-----------|---------|----------|
| CSV Sink | `csv` | Write to CSV files | Append mode, deterministic headers |
| JSON Sink | `json` | Write to JSON/JSONL | Array or line-delimited format |
| Database Sink | `database` | Write to SQL database | Multi-backend (PostgreSQL, MySQL, SQLite) |
| Azure Blob Sink | `azure_blob` | Write to Azure Blob Storage | SAS auth, path templating |

**Sink Capabilities:**
- Idempotency key support (`{run_id}:{token_id}:{sink}`)
- Artifact recording with content hash
- Resume-aware (skip already-written rows)

### 1.4 Gates

Gates are configured in YAML, not as separate plugins. The engine provides:

| Gate Type | Configuration | Routing Options |
|-----------|--------------|-----------------|
| Expression Gate | `condition: "row['field'] > 10"` | continue, route_to_sink, fork_to_paths |
| Threshold Gate | `rules: [{field, operator, value}]` | Configurable route labels |
| Boolean Gate | `condition: "row['valid']"` | Enforced `true`/`false` labels |

**Gate Capabilities:**
- AST-based expression parsing (no eval)
- Safe field access via `row.get()` or `row['field']`
- Boolean operators: and, or, not
- Comparison operators: ==, !=, <, >, <=, >=, in, not in

---

## 2. ENGINE CAPABILITIES

### 2.1 DAG Execution

| Feature | Status | Evidence |
|---------|--------|----------|
| Linear pipeline | ✅ Working | Degenerates to single-path DAG |
| Multi-sink routing | ✅ Working | Gates route to named sinks |
| Fork to parallel paths | ✅ Working | `fork_to_paths` creates child tokens |
| Coalesce (join) | ✅ Working | Merges tokens by row_id |
| Topological execution | ✅ Working | NetworkX provides ordering |
| Cycle detection | ✅ Working | Rejects cyclic configurations |

### 2.2 Token Management

| Feature | Status | Description |
|---------|--------|-------------|
| row_id | ✅ | Stable source row identity |
| token_id | ✅ | Instance in specific DAG path |
| parent_token_id | ✅ | Lineage for forks |
| token_parents table | ✅ | Multi-parent joins |
| expand_group_id | ✅ | Deaggregation (1→N) lineage |
| deepcopy isolation | ✅ | Sibling tokens have independent data |

### 2.3 Aggregation

| Feature | Status | Description |
|---------|--------|-------------|
| Count trigger | ✅ | Fire after N rows |
| Timeout trigger | ⚠️ Partial | Fires on next row arrival, not true idle |
| Condition trigger | ✅ | Fire on matching row |
| End-of-source flush | ✅ | Implicit trigger at source completion |
| Multiple triggers | ✅ | OR logic - first wins |
| Batch state checkpoint | ✅ | Survives crash recovery |
| Output modes | ✅ | passthrough, transform (default: transform) |

### 2.4 Retry System

| Feature | Status | Description |
|---------|--------|-------------|
| Configurable max_attempts | ✅ | Per-transform or global |
| Exponential backoff | ✅ | Via tenacity |
| Attempt tracking | ✅ | Each attempt recorded separately |
| Transient vs permanent | ✅ | Only transient errors retry |
| MaxRetriesExceeded | ✅ | Clear terminal state |

### 2.5 Coalesce Policies

| Policy | Status | Behavior |
|--------|--------|----------|
| require_all | ✅ | Wait for all branches |
| quorum | ✅ | Wait for N branches |
| best_effort | ✅ | Wait until timeout |
| first | ✅ | Take first arrival |

| Merge Strategy | Status | Behavior |
|----------------|--------|----------|
| union | ✅ | Merge all fields |
| nested | ✅ | Nest by branch name |
| select | ✅ | Pick specific branch |

---

## 3. AUDIT TRAIL (LANDSCAPE)

### 3.1 Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| runs | Run metadata | run_id, status, config_hash, settings_json, reproducibility_grade |
| nodes | Execution graph vertices | node_id, run_id, plugin_name, determinism, schema_hash |
| edges | Graph connections | from_node, to_node, label, default_mode |
| source_rows | Source data | row_id, run_id, row_index, source_data_ref, source_data_hash |
| tokens | Row instances | token_id, row_id, parent_token_id, expand_group_id |
| token_parents | Multi-parent joins | token_id, parent_token_id |
| node_states | Processing records | state_id, token_id, node_id, status, input_hash, output_hash |
| calls | External calls | call_id, state_id, call_type, request_ref, response_ref, latency_ms |
| routing_events | Edge selections | token_id, from_node, to_node, route_label, reason |
| batches | Aggregation groups | batch_id, node_id, trigger_type, status |
| batch_members | Batch composition | batch_id, token_id |
| batch_outputs | Batch results | batch_id, output_token_id, output_type |
| artifacts | Sink outputs | artifact_id, run_id, sink_name, content_hash, size_bytes |
| validation_errors | Source validation failures | row_id, node_id, field_errors, raw_row |
| transform_errors | Transform failures | state_id, error_type, error_message, retryable |
| token_outcomes | Terminal states | token_id, outcome, recorded_at |
| checkpoints | Crash recovery | checkpoint_id, run_id, state_json |

### 3.2 Recording Points

| Boundary | What's Recorded | Method |
|----------|-----------------|--------|
| Source entry | Raw row data, hash | `create_source_row()` |
| Transform input | Input hash, timestamp | `create_node_state()` |
| Transform output | Output hash, status | `update_node_state()` |
| External call | Full request/response | `record_call()` |
| Routing decision | Edge, reason | `record_routing_event()` |
| Fork | Parent→children | `record_fork()` |
| Coalesce | Children→merged | `record_coalesce()` |
| Aggregation | Batch membership | `record_batch_member()` |
| Sink write | Artifact descriptor | `record_artifact()` |
| Terminal state | Outcome | `record_token_outcome()` |

### 3.3 Export

| Feature | Status | Description |
|---------|--------|-------------|
| CSV export | ✅ | Type-specific files |
| JSON export | ✅ | JSONL stream |
| HMAC signing | ✅ | Per-record + manifest |
| Running hash chain | ✅ | Tamper detection |

---

## 4. CLI COMMANDS

| Command | Status | Purpose |
|---------|--------|---------|
| `elspeth run` | ⚠️ Partial | Execute pipeline (PayloadStore not wired) |
| `elspeth validate` | ✅ | Validate configuration |
| `elspeth explain` | ⚠️ Partial | Lineage explorer (TUI preview) |
| `elspeth plugins list` | ✅ | List available plugins |
| `elspeth purge` | ✅ | Clean old payloads |
| `elspeth resume` | ✅ | Resume from checkpoint |
| `elspeth health` | ✅ | System health check |

---

## 5. CORE SUBSYSTEMS

### 5.1 Configuration (Dynaconf + Pydantic)

| Feature | Status |
|---------|--------|
| YAML loading | ✅ |
| Environment variable interpolation | ✅ |
| Default values | ✅ |
| Multi-source precedence | ✅ |
| Pydantic validation | ✅ |
| Secret fingerprinting | ✅ |
| Template file expansion | ✅ |
| Lookup file expansion | ✅ |

### 5.2 Canonical JSON

| Feature | Status |
|---------|--------|
| RFC 8785 serialization | ✅ |
| NaN/Infinity rejection | ✅ |
| numpy type conversion | ✅ |
| pandas type conversion | ✅ |
| Version tracking | ✅ |
| Cross-process stability | ✅ |

### 5.3 Payload Store

| Feature | Status |
|---------|--------|
| Filesystem backend | ✅ |
| put/get/exists API | ✅ |
| Content-addressed storage | ✅ |
| Hash verification on read | ✅ |
| S3/blob backend | ❌ Not implemented |

### 5.4 Rate Limiting

| Feature | Status |
|---------|--------|
| pyrate-limiter integration | ✅ |
| Per-plugin limits | ✅ |
| Registry for multiple limiters | ✅ |
| Engine wiring | ❌ Not wired |

### 5.5 Checkpoint & Recovery

| Feature | Status |
|---------|--------|
| Checkpoint creation | ✅ |
| Resume from checkpoint | ✅ |
| Aggregation state restore | ✅ |
| Compatibility checking | ✅ |

### 5.6 Security

| Feature | Status |
|---------|--------|
| HMAC secret fingerprinting | ✅ |
| Azure Key Vault integration | ✅ |
| Redaction profiles | ❌ Not implemented |
| Access control | ❌ Not implemented |

---

## 6. MCP ANALYSIS SERVER

Entry point: `elspeth-mcp`

| Tool | Purpose |
|------|---------|
| `list_runs` | List pipeline runs with status filter |
| `get_run` | Detailed run information |
| `get_run_summary` | Statistics (counts, durations, errors) |
| `list_nodes` | Plugin instances for a run |
| `list_rows` | Source rows with pagination |
| `list_tokens` | Token tracking |
| `explain_token` | Complete lineage |
| `get_errors` | Validation/transform errors |
| `get_node_states` | Processing records |
| `get_calls` | External calls |
| `query` | Ad-hoc SQL (read-only) |
| `describe_schema` | Database schema |
| `get_dag_structure` | DAG with Mermaid diagram |
| `get_performance_report` | Slow nodes, bottlenecks |
| `get_error_analysis` | Errors by type/node |
| `get_llm_usage_report` | LLM statistics |
| `get_outcome_analysis` | Terminal states distribution |
| `diagnose` | Emergency failure scan |
| `get_failure_context` | Deep dive on failures |
| `get_recent_activity` | Recent run timeline |

---

## 7. RECENT ADDITIONS (Post-Jan 22)

### 7.1 Field Normalization

**Files:** `plugins/sources/field_normalization.py`

- `normalize_field_name()` - Unicode NFC, lowercase, identifier conversion
- `check_normalization_collisions()` - Detect header collisions
- `resolve_field_names()` - Complete resolution pipeline
- Algorithm versioning: `NORMALIZATION_ALGORITHM_VERSION = "1.0.0"`

### 7.2 Identifier Validation

**Files:** `core/identifiers.py`

- `validate_field_names()` - Validate Python identifiers, keywords, duplicates

### 7.3 Structured Outputs

**Config:** `ResponseFormat.STRUCTURED` in LLM plugins

- JSON schema-based output validation
- Automatic parsing and field extraction

### 7.4 MCP Database Auto-Discovery

- Auto-discovers `.db` files in current directory
- Prioritizes `audit.db` in `runs/` directories
- Sorts by most recently modified

---

## 8. NOT IMPLEMENTED (Deferred)

| Feature | Original Phase | Status |
|---------|---------------|--------|
| Database source | Phase 4 | Not started |
| HTTP API source | Phase 6 | Not started |
| Message queue source | Phase 6+ | Not started |
| Webhook sink | Phase 6 | Not started |
| Profile system (`--profile`) | Phase 4 | Deferred |
| Pack defaults | Phase 6+ | Deferred |
| Redaction profiles | Phase 5 | Deferred |
| Access control (Operator/Auditor/Admin) | Phase 5+ | Deferred |
| S3/blob payload backend | Phase 7 | Not started |
| Multi-destination copy routing | Phase 7 | Not started |
| Concurrent processing integration | Phase 5 | Config exists, not wired |

---

## 9. DIVERGENCES FROM ORIGINAL SPEC

| Original Spec | Actual Implementation | Assessment |
|--------------|----------------------|------------|
| `landscape.backend` + `path` | SQLAlchemy URL format | ✅ Better |
| Split retention by type | Unified `retention_days` | ⚠️ Less granular |
| 1→1 transform only | 1→1 default + `success_multi()` for 1→N | ✅ Extended |
| Static hookimpl registration | Dynamic directory scanning | ✅ Better |
| LiteLLM for LLM access | Direct OpenAI SDK + custom clients | ⚠️ Different (works) |
| `elspeth explain --full` | `--json` and `--no-tui` | 🔀 Changed |
| Terminal states derived | Explicit `token_outcomes` table | ✅ Better for queries |

---

## 10. REQUIREMENTS.MD GAPS

### In requirements.md but not fully working:
- CLI-016: `elspeth run` PayloadStore wiring
- CRIT-03: Coalesce timeout calling in processor
- FAI-009: Every token reaches terminal state (some gaps)

### Working but not in requirements.md:
- Field normalization with collision detection
- `resolve_field_names()` complete pipeline
- Structured output support in LLM plugins
- MCP server with 20+ tools
- Aggregation trigger type in metadata
- `validate_field_names()` core utility

---

*Inventory completed: January 29, 2026*
*Next update: After RC-2 release*
