# ELSPETH Feature Inventory - March 1, 2026

> **ARCHIVED — Feature-inventory snapshot captured at 1 March 2026 (RC-3.3).**
> This document records the platform's plugin and capability inventory as it stood at RC-3.3.
> RC-3.4, RC-4.0, RC-4.1, RC-5.0, RC-5.1, and RC-5.2 have all shipped since — including the Web UX Composer, Dataverse and RAG plugins, ChromaSink, ten statistical batch plugins, guided composer, and composer progress persistence — and this document has not been refreshed.
>
> **For the current state, see** [`../elspeth-progress-rc1-to-rc5.md`](../elspeth-progress-rc1-to-rc5.md) (updated 19 May 2026, RC-5.2) and the canonical line-by-line record in [`../../../CHANGELOG.md`](../../../CHANGELOG.md).

**Version:** RC-3.3
**Snapshot date:** 1 March 2026 (all status tables in this document are valid as of this date only)
**Audience:** Engineering team and engineering leadership — historical reference
**Register:** Technical
**Purpose:** Complete inventory of what ELSPETH did at RC-3.3, reconciled against the original architecture.md (Jan 12) and requirements.md (Jan 22). Updated for RC-3.3 architectural remediation.

> Every `✅ / ⚠️ / ❌` status mark in this document carries the implicit suffix *[as of RC-3.3, 1 March 2026]*. Many items marked `❌ Not implemented` or `❌ Deferred` in §9 have since shipped — do not cite this document for current capability claims.

---

## EVOLUTION SUMMARY

| Aspect | Original Vision (Jan 12) | Current Reality (Mar 1, RC-3.3) |
|--------|-------------------------|-------------------------|
| Execution Model | Linear pipeline with gates | Full DAG with fork/coalesce, declarative wiring |
| Routing | continue, route_to_sink | Declarative `on_success`/`input` routing (ADR-004, ADR-005) |
| Gates | Plugin-based gate system | Config-driven only (GateSettings + ExpressionParser) |
| Aggregation | "accumulate state until trigger" | Count, timeout, condition triggers with flush |
| Token Identity | row_id sufficient | row_id + token_id + parent_token_id |
| LLM Integration | "Phase 6 future work" | Unified LLMTransform with provider dispatch (azure/openrouter), single/multi-query strategies, 2 batch transforms |
| Analysis Tools | "elspeth explain" | MCP server with 20+ analysis tools |
| Field Handling | Assumed clean input | Full normalization pipeline with collision detection |
| Shutdown | Not addressed | Cooperative graceful shutdown with checkpoint + resume |
| Telemetry | Not addressed | OpenTelemetry with OTLP, Azure Monitor, Datadog exporters |
| Testing | Manual | ~10,500 automated tests (unit, integration, e2e, property, performance) |

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

### 1.2 Transforms (13)

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
| Web Scrape | `web_scrape` | Row | 1→1 | Extract content from web pages with fingerprinting |

#### Azure Transforms (2)

| Plugin Name | Config Key | Type | Purpose |
|-------------|-----------|------|---------|
| Azure Content Safety | `azure_content_safety` | LLM | Detect harmful content |
| Azure Prompt Shield | `azure_prompt_shield` | LLM | Detect prompt injection |

#### LLM Transforms (3)

| Plugin Name | Config Key | Type | Provider | Features |
|-------------|-----------|------|----------|----------|
| LLM Transform | `llm` | LLM | Azure OpenAI, OpenRouter | Unified transform with provider dispatch (azure/openrouter), single-query and multi-query strategies, template prompts, structured output |

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

### 1.4 Gates (Config-Driven)

Gates are **config-driven system operations**, not plugins. Gate plugins were deliberately removed in RC-3 (2026-02-11). All routing is handled by `GateSettings` + `ExpressionParser`.

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
| Declarative DAG wiring | ✅ Working | Explicit `on_success`/`input` routing (ADR-004, ADR-005) |
| Explicit sink routing | ✅ Working | No implicit sink connections; all routes declared in YAML |
| DIVERT routing edges | ✅ Working | Quarantine/error sinks wired as `RoutingMode.DIVERT` edges |
| Quarantine sink DAG exclusion | ✅ Working | Prevents unreachable node errors for quarantine sinks |
| Structural node allowlists | ✅ Working | `NodeType` enum for type-safe node identification |

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

### 2.6 Graceful Shutdown (FEAT-05, FEAT-05b)

| Feature | Status | Description |
|---------|--------|-------------|
| Signal handling (SIGTERM/SIGINT) | ✅ | Cooperative shutdown via `threading.Event` |
| Processing loop shutdown | ✅ | Checks shutdown flag between rows and on quarantine path |
| Resume path shutdown | ✅ | Graceful shutdown during `elspeth resume` |
| Aggregation buffer flush | ✅ | Flushes pending aggregation buffers on shutdown |
| Checkpoint on interrupt | ✅ | Creates checkpoint for `elspeth resume` |
| Run status INTERRUPTED | ✅ | Clean status marking, resumable |
| Second Ctrl-C force kill | ✅ | Immediate exit on repeated signal |
| CLI exit code 3 | ✅ | Distinct exit code for interrupted runs |

### 2.7 DROP-Mode Sentinel Handling

| Feature | Status | Description |
|---------|--------|-------------|
| Sentinel requeue failure isolation | ✅ | DROP-mode sentinel requeue failures do not raise to callers |
| Flush join guarantee | ✅ | Preserved during DROP queue replacement |
| Shutdown sentinel race fix | ✅ | Race condition between shutdown and DROP-mode sentinel requeue resolved |

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
| operations | Source/sink audit trail | operation_id, run_id, node_id, operation_type, input_hash, output_hash |
| secret_resolutions | Secret loading audit | run_id, secret_name, vault_source, fingerprint, latency_ms |

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
| Source/sink operation | I/O hashes for source and sink operations | `create_operation()` / `complete_operation()` |
| Secret resolution | Vault source, HMAC fingerprint, latency | `record_secret_resolution()` |
| DIVERT routing | Quarantine/error routing events | `record_routing_event()` with DIVERT mode |
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
| `elspeth run` | ✅ Working | Execute pipeline with PayloadStore |
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
| Settings→Runtime*Config contracts | ✅ |
| Protocol-based config verification | ✅ |
| AST field-mapping checker | ✅ |

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
| Engine wiring | ✅ (CRIT-01) |

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
| Secret resolution audit trail | ✅ |
| SQLCipher encrypted databases | ✅ |
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

## 7. TELEMETRY

**Entry point:** Configured via `telemetry:` section in `settings.yaml`

| Feature | Status | Description |
|---------|--------|-------------|
| OpenTelemetry integration | ✅ | Structured event emission with trace/span correlation |
| OTLP exporter | ✅ | gRPC/HTTP export to any OTLP-compatible backend |
| Azure Monitor exporter | ✅ | Direct export to Azure Application Insights |
| Datadog exporter | ✅ | Export via `ddtrace` |
| Granularity filtering | ✅ | Row-level vs run-level event classification |
| FieldResolutionApplied events | ✅ | Classified as row-level telemetry |
| ExternalCallCompleted token-id | ✅ | Token-id correlation on external call events |
| DROP-mode backpressure | ✅ | Evicts oldest queued events under pressure |
| Queue accounting | ✅ | Hardened queue size tracking and hook validation |
| Empty flush acknowledgement | ✅ | OTLP and Azure Monitor explicitly handle empty flushes |
| No-silent-failures policy | ✅ | Every emission point sends or acknowledges absence |

---

## 8. RECENT ADDITIONS (Post-Jan 22)

### 8.1 Field Normalization

**Files:** `plugins/sources/field_normalization.py`

- `normalize_field_name()` - Unicode NFC, lowercase, identifier conversion
- `check_normalization_collisions()` - Detect header collisions
- `resolve_field_names()` - Complete resolution pipeline
- Algorithm versioning: `NORMALIZATION_ALGORITHM_VERSION = "1.0.0"`

### 8.2 Identifier Validation

**Files:** `core/identifiers.py`

- `validate_field_names()` - Validate Python identifiers, keywords, duplicates

### 8.3 Structured Outputs

**Config:** `ResponseFormat.STRUCTURED` in LLM plugins

- JSON schema-based output validation
- Automatic parsing and field extraction

### 8.4 MCP Database Auto-Discovery

- Auto-discovers `.db` files in current directory
- Prioritizes `audit.db` in `runs/` directories
- Sorts by most recently modified

### 8.5 Declarative DAG Wiring (ADR-004, ADR-005) — RC-3

**Commits:** Routing Trilogy (`d213fca3`, `00d3c6ba`, `5080ff1c`, et al.)

Three-phase routing overhaul replacing implicit sink connections with fully declarative wiring:

- **Phase 1 (Explicit Sink Routing):** All sink routes declared via `on_success` in YAML; no implicit connections
- **Phase 2 (Processor Node-ID Refactoring):** `NodeType` enum for structural node identification; `MappingProxyType` for frozen config
- **Phase 3 (Declarative DAG Wiring):** `on_success` and `input` routing on all transforms/gates; `WiredTransform` connection matching; connection namespace reservation
- Gate-to-gate route resolution
- Fail-closed routing (missing edge = error, not silent drop)

### 8.6 Gate Plugin Removal — RC-3

**Commits:** `7b61f3bb`, `ea0e208d`, et al.

All gate plugin infrastructure deliberately removed (~3,000 lines deleted):

- `GateProtocol`, `BaseGate`, `execute_gate()`, `PluginGateReason` deleted
- Gate plugin registration, discovery, and factory code removed
- Routing is config-driven only via `GateSettings` + `ExpressionParser`
- Dead plugin protocols (`CoalesceProtocol`, `PluginProtocol`, `CoalescePolicy`) also removed

### 8.7 Graceful Shutdown (FEAT-05, FEAT-05b) — RC-3

**Commits:** `6286367f` (processing loop), `d2a6cbef` (resume path), `9a9fd8ca` (quarantine path)

- Cooperative shutdown via `threading.Event` and SIGTERM/SIGINT handlers
- Processing loop, resume path, and quarantine path all check shutdown flag
- On shutdown: flush aggregation buffers, write pending sinks, create checkpoint, mark run INTERRUPTED
- Second Ctrl-C force-kills; CLI exits with code 3
- 16 tests (9 unit + 7 integration)

### 8.8 DROP-Mode Sentinel Handling — RC-3

**Commits:** `bbc2f515`, `3af35d0d`, `1acec271`

- DROP-mode sentinel requeue failures isolated from callers
- Flush join guarantee preserved during DROP queue replacement
- Shutdown sentinel requeue race condition resolved

### 8.9 DIVERT Routing for Audit Completeness — RC-3

**Commits:** `7126d457`, `8d2b7bb7`, `73d40fd7`, `d5d8e4df`, et al.

- `RoutingMode.DIVERT` edges for quarantine/error sinks in DAG
- Transform error paths emit DIVERT routing events
- Source quarantine paths emit routing events with `SourceQuarantineReason`
- MCP `explain_token` annotates DIVERT routing in lineage response
- Mermaid diagrams render DIVERT edges as dashed arrows
- DAG validation warns on DIVERT + `require_all` coalesce

### 8.10 Contract Propagation for Complex Fields — RC-3

**Commits:** `c21cdb27`, `7e6285b3`, `16cb0726`

- `dict` and `list` fields preserved in propagated contracts
- Schema contract factory for consistent contract creation
- Contract propagation utilities for transform pipeline
- JSONExplode contract handles complex `output_field` values

### 8.11 ExternalCallCompleted Token-ID Correlation — RC-3

**Commit:** `f38cfa2c`

- External call telemetry events now include `token_id` for correlation with Landscape audit trail

### 8.12 FieldResolution Telemetry Granularity — RC-3

**Commit:** `047d8975`

- `FieldResolutionApplied` events classified as row-level telemetry for granularity filtering

### 8.13 Test Suite v2 Migration — RC-3

**Commits:** 7-phase migration (`f62aa1a3` through `9c657fb7`)

- ~10,563 tests at RC-3.3 (grown from initial 8,138 at cutover)
- Phase 0+1: Scaffolding + factories
- Phase 2A: Contracts (1,142 tests)
- Phase 2B+2C+2D: Core + engine + plugins (3,823 tests)
- Phase 3: Property tests (1,057 tests)
- Phase 4: Integration tests (482 tests)
- Phase 5: E2E tests (48 tests)
- Phase 6: Performance tests (67 tests)
- Phase 7: Cutover — deleted v1 suite (7,487 tests, 507 files, 222K lines), renamed `tests_v2/` to `tests/`

### 8.14 ExecutionGraph Public API — RC-3

**Commit:** `1c31869b` (TEST-01)

- 13 public setters + 5 method renames on `ExecutionGraph`
- 60+ private attribute accesses eliminated from tests
- Clean public API boundary between engine internals and test code

### 8.15 Landscape Recorder Decomposition — RC-3

**Commit:** `85f94895`

- `LandscapeRecorder` god class decomposed into 8 focused mixins
- Cleaner separation of recording responsibilities

### 8.16 Operation I/O Hashes — RC-3

**Commit:** `264ab2cf`

- Operation I/O hashes survive payload purge
- Integrity verification possible even after retention policies delete payloads

### 8.17 SQLCipher Support — RC-3

**Commits:** `186b7507`, `5e338057`, `a428da13`, `bdb99c33`, `d4d7a85c`

- SQLCipher encrypted audit databases
- Backend validation and key escaping
- CLI passphrase support
- MCP server passphrase forwarding
- Empty passphrase rejection

### 8.18 Secret Resolution Audit Trail — RC-3

**Commit:** `b758b299`, `16cb0726`

- `secret_resolutions` Landscape table
- Records vault source, HMAC fingerprint, latency for every secret loaded
- Azure Key Vault and environment variable sources audited

### 8.19 Per-Branch Transforms (ARCH-15) — RC-3

**Commit:** `83a6d40a`

- Per-branch transforms between fork and coalesce nodes
- Enables distinct processing on each parallel path
- 23 tests

### 8.20 LLM Transform Consolidation (T10) — RC-3.3

- Unified 6 separate LLM transforms into single `LLMTransform` with provider dispatch
- Provider selection via config: `azure` or `openrouter`
- Strategy selection: single-query or multi-query
- (Originally retained two batch-LLM plugins separate from the unified
  transform; both were retired 2026-05-06 per ADR-020 due to conops
  misalignment with per-row audit attribution.)

### 8.21 Plugins Restructure — RC-3.3

**Commits:** `ec1f668c`, `4045362e`, `8e120814`

- Restructured `plugins/` into 4 SDA-aligned subfolders: `infrastructure/`, `sources/`, `transforms/`, `sinks/`
- Updated all imports and patch targets across test suite

---

## 9. NOT IMPLEMENTED (Deferred) — as of RC-3.3, 1 March 2026

> **Stale section warning.** Many entries below have since shipped (notably access control, blob storage backend, and Dataverse / RAG / Chroma sources). This list is preserved verbatim for historical context only. **Do not cite this list as current.** For the current capability surface see the progress report.

| Feature | Original Phase | Status at RC-3.3 (1 March 2026) |
|---------|---------------|----------------------------------|
| Database source | Phase 4 | Not started |
| HTTP API source | Phase 6 | Not started |
| Message queue source | Phase 6+ | Not started |
| Webhook sink | Phase 6 | Not started |
| Profile system (`--profile`) | Phase 4 | Deferred |
| Pack defaults | Phase 6+ | Deferred |
| Redaction profiles | Phase 5 | Deferred |
| Access control (Operator/Auditor/Admin) | Phase 5+ | Deferred *(shipped in a different shape at RC-5 as three-provider authentication; redaction-profile work tracked separately)* |
| S3/blob payload backend | Phase 7 | Not started *(blob storage manager shipped at RC-5)* |
| Multi-destination copy routing | Phase 7 | Not started |
| Concurrent processing integration | Phase 5 | Config exists, not wired |
| Circuit breaker for retry logic | RC-3+ | FEAT-06 deferred |
| CLI `status`/`export`/`db migrate` | RC-3+ | FEAT-04 deferred |
| Prometheus/pull metrics | RC-3+ | OBS-04 deferred |

---

## 10. DIVERGENCES FROM ORIGINAL SPEC

| Original Spec | Actual Implementation | Assessment |
|--------------|----------------------|------------|
| `landscape.backend` + `path` | SQLAlchemy URL format | ✅ Better |
| Split retention by type | Unified `retention_days` | ⚠️ Less granular |
| 1→1 transform only | 1→1 default + `success_multi()` for 1→N | ✅ Extended |
| Static hookimpl registration | Dynamic directory scanning | ✅ Better |
| LiteLLM for LLM access | Direct OpenAI SDK + custom clients | ⚠️ Different (works) |
| `elspeth explain --full` | `--json` and `--no-tui` | 🔀 Changed |
| Terminal states derived | Explicit `token_outcomes` table | ✅ Better for queries |
| Plugin-based gates | Config-driven gates (GateSettings) | ✅ Simpler, no plugin overhead |
| Implicit sink routing | Declarative `on_success`/`input` routing | ✅ Explicit, auditable |

---

## 11. REQUIREMENTS.MD GAPS — as of RC-3.3, 1 March 2026

> **Stale section warning.** `REQUIREMENTS.md` was deleted in RC-3 as part of the documentation remediation (F-05) — `pyproject.toml` and `CHANGELOG.md` are the single sources of truth. The "gaps" tracked below are historical and have no current binding.

### In requirements.md but not fully implemented:
- FAI-009: Every token reaches terminal state (edge case gaps remain)

### Working but not in requirements.md:
- Field normalization with collision detection
- `resolve_field_names()` complete pipeline
- Structured output support in LLM plugins
- MCP server with 20+ tools
- Aggregation trigger type in metadata
- `validate_field_names()` core utility
- Graceful shutdown with checkpoint + resume
- Declarative DAG wiring (ADR-004, ADR-005)
- DIVERT routing for audit completeness
- Telemetry subsystem (OTLP, Azure Monitor, Datadog)
- SQLCipher encrypted audit databases
- Secret resolution audit trail
- Contract propagation for complex fields
- Per-branch transforms between fork/coalesce (ARCH-15)
- Unified LLM transform with provider dispatch (T10 consolidation)
- Plugins restructured into SDA-aligned subfolders

---

*Inventory completed: January 29, 2026*
*Updated: March 2, 2026 (RC-3.3 architectural remediation)*
