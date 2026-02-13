COMPLETE REQUIREMENTS LIST - ELSPETH Architecture
=================================================

**Last Updated:** 2026-02-12 (RC-2.5 synchronization update)
**Audit Method:** 10 parallel agents reviewed all sections after P0 payload storage fix (2026-01-22); RC-2.5 additions appended 2026-02-12
**Previous Audit:** 2026-01-22

Legend:
- ✅ IMPLEMENTED - Code exists and matches requirement
- ❌ NOT IMPLEMENTED - No code found
- 🔀 DIVERGED - Implemented differently than specified (noted)
- ⚠️ PARTIAL - Partially implemented or Phase 3+ integration pending
- 🆕 NEW - Discovered capability not in previous spec

---

## 1. CONFIGURATION REQUIREMENTS

### 1.1 Configuration Format

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CFG-001 | Config uses `datasource` key (not source) | README.md:75 | ✅ IMPLEMENTED | `config.py:586-588` - `DatasourceSettings` class |
| CFG-002 | `datasource.plugin` specifies the source plugin name | README.md:76 | ✅ IMPLEMENTED | `config.py:380` - `plugin: str` field |
| CFG-003 | `datasource.options` holds plugin-specific config | README.md:77-78 | ✅ IMPLEMENTED | `config.py:381-383` - `options: dict[str, Any]` |
| CFG-004 | `sinks` is a dict of named sinks | README.md:80-89 | ✅ IMPLEMENTED | `config.py:589-591` - `sinks: dict[str, SinkSettings]` |
| CFG-005 | Each sink has `plugin` and `options` keys | README.md:81-88 | ✅ IMPLEMENTED | `config.py:408-412` - `SinkSettings` |
| CFG-006 | `row_plugins` is an array of transforms | README.md:91-99 | ✅ IMPLEMENTED | `config.py:607-610` - `row_plugins: list[RowPluginSettings]` |
| CFG-007 | Each row_plugin has `plugin`, `type`, `options`, `routes` | README.md:92-99 | ✅ IMPLEMENTED | `config.py:396-400` - all fields present |
| CFG-008 | `output_sink` specifies the default sink | README.md:107 | ✅ IMPLEMENTED | `config.py:592-594` - required, validated |
| CFG-009 | `landscape.enabled` boolean flag | README.md:109-110 | ✅ IMPLEMENTED | `config.py:447` - `enabled: bool = True` |
| CFG-010 | `landscape.backend` specifies storage type | README.md:111 | 🔀 IMPROVED | Uses SQLAlchemy URL format; backend inferred from scheme |
| CFG-011 | `landscape.path` specifies database path | README.md:112 | 🔀 IMPROVED | `config.py:454-457` - Uses `url: str` (e.g., `sqlite:///path`) |
| CFG-012 | `landscape.retention.row_payloads_days` config | architecture.md:556 | ⚠️ PARTIAL | `PayloadStoreSettings.retention_days` - unified retention |
| CFG-013 | `landscape.retention.call_payloads_days` config | architecture.md:557 | ⚠️ PARTIAL | Unified with row payloads retention |
| CFG-014 | `landscape.redaction.profile` config | architecture.md:889-890 | ❌ DEFERRED | Phase 5+ feature - access control not implemented |
| CFG-015 | `concurrency.max_workers` config (default 4) | README.md:195-202 | ✅ IMPLEMENTED | `config.py:469-473` - `max_workers: int = 4` |
| CFG-016 | Profile system with `profiles:` and `--profile` flag | README.md:199-209 | ❌ DEFERRED | Dynaconf supports; CLI integration deferred |
| CFG-017 | Environment variable interpolation `${VAR}` | README.md:213-216 | ✅ IMPLEMENTED | `config.py:708-746` - `_expand_env_vars()` with `${VAR:-default}` |
| CFG-018 | Hierarchical settings merge with precedence | README.md:188-206 | ✅ IMPLEMENTED | `config.py:1125-1131` - env > file > defaults |
| CFG-019 | Pack defaults (`packs/llm/defaults.yaml`) | architecture.md:824 | ❌ DEFERRED | Phase 6+ feature |
| CFG-020 | Pipeline configuration (`settings.yaml`) | architecture.md:823 | ❌ DEFERRED | Single settings file per run sufficient |

### 1.2 Configuration Settings Classes

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CFG-021 | LandscapeSettings class | Phase 1 plan | ✅ IMPLEMENTED | `config.py:442-461` - full Pydantic model |
| CFG-022 | RetentionSettings class | Phase 1 plan | ⚠️ PARTIAL | `PayloadStoreSettings` has `retention_days` |
| CFG-023 | ConcurrencySettings class | Phase 1 plan | ✅ IMPLEMENTED | `config.py:464-473` |
| CFG-024 | Settings stored with run (resolved, not just hash) | architecture.md:270 | ✅ IMPLEMENTED | `recorder.py:237-238` stores both hash and full JSON |

### 1.3 Configuration - New Capabilities (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CFG-025 | 🆕 `run_mode` enum (LIVE, REPLAY, VERIFY) | Phase 6 | ✅ IMPLEMENTED | `config.py:597-600` - supports record/replay/verify |
| CFG-026 | 🆕 `replay_source_run_id` for replay/verify modes | Phase 6 | ✅ IMPLEMENTED | `config.py:601-604` - links to prior run |
| CFG-027 | 🆕 Template file expansion (`template_file`, `template_source`) | Phase 6 | ✅ IMPLEMENTED | `config.py:1040-1095` - Jinja2 templates in plugin config |
| CFG-028 | 🆕 Lookup file expansion (`lookup_file`, `lookup_source`) | Phase 6 | ✅ IMPLEMENTED | `config.py:1040-1095` - YAML reference data |
| CFG-029 | 🆕 Secret fingerprinting (runtime preserve + audit HMAC) | Phase 5 | ✅ IMPLEMENTED | `config.py:904-1033` - two-phase fingerprinting |
| CFG-030 | 🆕 `landscape.export` (LandscapeExportSettings) | Phase 5 | ✅ IMPLEMENTED | `config.py:415-440` - post-run audit export |
| CFG-031 | 🆕 `landscape.export.enabled` boolean | Phase 5 | ✅ IMPLEMENTED | `config.py:424-426` |
| CFG-032 | 🆕 `landscape.export.sink` target sink name | Phase 5 | ✅ IMPLEMENTED | `config.py:428-430` - validated against sinks |
| CFG-033 | 🆕 `landscape.export.format` (CSV or JSON) | Phase 5 | ✅ IMPLEMENTED | `config.py:432-434` |
| CFG-034 | 🆕 `landscape.export.sign` HMAC signing option | Phase 5 | ✅ IMPLEMENTED | `config.py:436-438` |
| CFG-035 | 🆕 `CheckpointSettings` crash recovery config | Phase 5 | ✅ IMPLEMENTED | `config.py:529-549` - frequency modes |
| CFG-036 | 🆕 `RetrySettings` backoff configuration | Phase 3 | ✅ IMPLEMENTED | `config.py:552-560` - max_attempts, delays |
| CFG-037 | 🆕 `PayloadStoreSettings` storage config | Phase 4 | ✅ IMPLEMENTED | `config.py:563-573` - backend, path, retention; wired through engine (fix 3399faf) |
| CFG-038 | 🆕 `RateLimitSettings` per-service limits | Phase 5 | ✅ IMPLEMENTED | `config.py:495-526` |
| CFG-039 | 🆕 CLI payload store auto-instantiation | P0-fix-gap | ❌ NOT IMPLEMENTED | `cli.py:269-396` - run command missing PayloadStore wiring |

---

## 2. CLI REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CLI-001 | `elspeth --settings <file>` to run pipeline | README.md:116 | ⚠️ PARTIAL | `cli.py:79-169` - runs but missing payload_store wiring (see CFG-039) |
| CLI-002 | `elspeth --profile <name>` for profile selection | README.md:208 | ❌ DEFERRED | Profile system not integrated |
| CLI-003 | `elspeth explain --run <id> --row <id> --database <path>` | README.md:122-136 | ✅ IMPLEMENTED | `cli.py:171-236` - with `--token` enhancement |
| CLI-004 | `elspeth explain` with `--full` flag for auditor view | architecture.md:765-766 | 🔀 CHANGED | Has `--json` and `--no-tui` instead (format control) |
| CLI-005 | `elspeth validate --settings <file>` | CLAUDE.md | ✅ IMPLEMENTED | `cli.py:351-390` |
| CLI-006 | `elspeth plugins list` | CLAUDE.md | ✅ IMPLEMENTED | `cli.py:430-464` - with `--type` filter |
| CLI-007 | `elspeth status` to check run status | subsystems:736 | ❌ DEFERRED | Query landscape directly instead |
| CLI-008 | Human-readable output by default, `--json` for machine | subsystems:739 | ⚠️ PARTIAL | `explain` has `--json`; other commands TBD |
| CLI-009 | TUI mode using Textual | architecture.md:777 | ✅ IMPLEMENTED | `tui/explain_app.py` - ExplainApp |

### 2.1 CLI - New Commands (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CLI-010 | 🆕 `elspeth purge` command for payload cleanup | Phase 5 | ✅ IMPLEMENTED | `cli.py:466-595` - with `--retention-days`, `--dry-run` |
| CLI-011 | 🆕 `elspeth resume` command for checkpoint recovery | Phase 5 | ✅ IMPLEMENTED | `cli.py:720-880` - reconstructs from checkpoint |
| CLI-012 | 🆕 `elspeth run --dry-run` preview mode | Safety | ✅ IMPLEMENTED | `cli.py:87-148` |
| CLI-013 | 🆕 `elspeth run --execute` safety gate | Safety | ✅ IMPLEMENTED | `cli.py:93-158` - required to actually run |
| CLI-014 | 🆕 `elspeth explain --token` for DAG-precise lineage | Enhancement | ✅ IMPLEMENTED | `cli.py:184-189` |
| CLI-015 | 🆕 `elspeth plugins list --type` filter | Enhancement | ✅ IMPLEMENTED | `cli.py:432-449` |
| CLI-016 | 🆕 Payload store instantiation in run command | P0-fix-gap | ❌ NOT IMPLEMENTED | `cli.py:269-396` - _execute_pipeline() doesn't create payload_store |

---

## 3. SDA MODEL REQUIREMENTS

### 3.1 Sources

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-001 | Exactly one source per run | CLAUDE.md | ✅ IMPLEMENTED | `SourceProtocol` enforces single source |
| SDA-002 | Sources are stateless | architecture.md:103 | ✅ IMPLEMENTED | `BaseSource` with no state |
| SDA-003 | CSV source plugin | CLAUDE.md | ✅ IMPLEMENTED | `sources/csv_source.py` |
| SDA-004 | JSON/JSONL source plugin | CLAUDE.md | ✅ IMPLEMENTED | `sources/json_source.py` |
| SDA-005 | Database source plugin | README.md:172 | ❌ NOT IMPLEMENTED | Phase 4+ |
| SDA-006 | HTTP API source plugin | README.md:172 | ❌ NOT IMPLEMENTED | Phase 6 |
| SDA-007 | Message queue source (blob storage) | README.md:172 | ❌ NOT IMPLEMENTED | Phase 6+ |

### 3.1.1 Sources - New Plugins (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-032 | 🆕 Azure Blob source with CSV/JSON/JSONL | Phase 4 | ✅ IMPLEMENTED | `plugins/azure/blob_source.py` |
| SDA-033 | 🆕 Null source for resume operations | Phase 5 | ✅ IMPLEMENTED | `sources/null_source.py` |

### 3.2 Transforms

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-008 | Zero or more transforms, ordered | plugin-protocol.md | ✅ IMPLEMENTED | Pipeline DAG handles ordering |
| SDA-009 | Transforms stateless between rows | plugin-protocol.md:328 | ✅ IMPLEMENTED | `BaseTransform.process()` per-row |
| SDA-010 | Transform: 1 row in → 1 row out (default) | plugin-protocol.md:330 | 🔀 EXTENDED | Now supports `success_multi()` for 1→N deaggregation |
| SDA-011 | Transform `process()` returns `TransformResult` | plugin-protocol.md:384-398 | ✅ IMPLEMENTED | `results.py:60-99` |
| SDA-012 | `TransformResult.success(row)` for success | plugin-protocol.md:433 | ✅ IMPLEMENTED | `results.py:80-83` |
| SDA-013 | `TransformResult.error(reason)` for failure | plugin-protocol.md:434 | ✅ IMPLEMENTED | `results.py:85-98` with retryable flag |
| SDA-014 | Transform `on_error` config (optional) | plugin-protocol.md:350-357 | ✅ IMPLEMENTED | `config_base.py:161-164` |
| SDA-015 | `TransformErrorEvent` recorded on error | plugin-protocol.md:464-470 | ✅ IMPLEMENTED | `schema.py:288-304`, `recorder.py:2139-2181` |
| SDA-016 | LLM query transform | README.md:103-105 | ✅ IMPLEMENTED | 3 LLM plugins + full infrastructure (see 3.2.1) |

### 3.2.1 Transforms - LLM Infrastructure (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-034 | 🆕 Azure OpenAI LLM transform | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/azure.py` |
| SDA-035 | 🆕 OpenRouter LLM transform (100+ models) | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/openrouter.py` |
| SDA-036 | 🆕 Azure Batch LLM transform (50% cost savings) | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/azure_batch.py` |
| SDA-037 | 🆕 PooledExecutor for parallel LLM calls | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/pooled_executor.py` |
| SDA-038 | 🆕 AIMD throttle for rate limiting | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/aimd_throttle.py` |
| SDA-039 | 🆕 Capacity error handling (429, 503, 529) | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/capacity_errors.py` |
| SDA-040 | 🆕 ReorderBuffer for out-of-order completion | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/reorder_buffer.py` |
| SDA-041 | 🆕 PromptTemplate with Jinja2 and audit metadata | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/templates.py` |

### 3.2.2 Transforms - Built-in Plugins (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-042 | 🆕 Field mapper transform (rename, select, extract) | Phase 4 | ✅ IMPLEMENTED | `transforms/field_mapper.py` |
| SDA-043 | 🆕 JSON explode transform (deaggregation) | Phase 4 | ✅ IMPLEMENTED | `transforms/json_explode.py` - `creates_tokens=True` |
| SDA-044 | 🆕 Keyword filter transform | Phase 4 | ✅ IMPLEMENTED | `transforms/keyword_filter.py` |
| SDA-045 | 🆕 Passthrough transform | Phase 4 | ✅ IMPLEMENTED | `transforms/passthrough.py` |
| SDA-046 | 🆕 Batch stats transform (sum, count, mean) | Phase 4 | ✅ IMPLEMENTED | `transforms/batch_stats.py` |
| SDA-047 | 🆕 Batch replicate transform (N→M) | Phase 4 | ✅ IMPLEMENTED | `transforms/batch_replicate.py` |
| SDA-048 | 🆕 Azure Content Safety transform | Phase 6 | ✅ IMPLEMENTED | `transforms/azure/content_safety.py` |
| SDA-049 | 🆕 Azure Prompt Shield transform | Phase 6 | ✅ IMPLEMENTED | `transforms/azure/prompt_shield.py` |
| SDA-051 | 🆕 Multi-query LLM transform (generic) | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/multi_query.py` - cross-product evaluation |
| SDA-052 | 🆕 Azure multi-query LLM transform | Phase 6 | ✅ IMPLEMENTED | `plugins/llm/azure_multi_query.py` - pooled multi-query |
| SDA-053 | 🆕 Truncate transform | Phase 4 | ✅ IMPLEMENTED | `transforms/truncate.py` - field length enforcement |

### 3.3 Sinks

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-017 | One or more sinks, named | plugin-protocol.md:476 | ✅ IMPLEMENTED | Config supports multiple |
| SDA-018 | Sink `write(rows) → ArtifactDescriptor` | plugin-protocol.md:497-510 | ✅ IMPLEMENTED | `protocols.py:468-482` |
| SDA-019 | `ArtifactDescriptor` with `content_hash` (REQUIRED) | plugin-protocol.md:556-557 | ✅ IMPLEMENTED | `results.py:177` - NOT optional |
| SDA-020 | `ArtifactDescriptor` with `size_bytes` (REQUIRED) | plugin-protocol.md:557 | ✅ IMPLEMENTED | `results.py:178` - NOT optional |
| SDA-021 | Sink `idempotent: bool` attribute | plugin-protocol.md:609-613 | ✅ IMPLEMENTED | `protocols.py:457` |
| SDA-022 | Idempotency key format: `{run_id}:{token_id}:{sink}` | plugin-protocol.md:613 | ⚠️ PARTIAL | Schema supports; engine passes at runtime |
| SDA-023 | CSV sink plugin | CLAUDE.md | ✅ IMPLEMENTED | `sinks/csv_sink.py` |
| SDA-024 | JSON sink plugin | CLAUDE.md | ✅ IMPLEMENTED | `sinks/json_sink.py` |
| SDA-025 | Database sink plugin | CLAUDE.md | ✅ IMPLEMENTED | `sinks/database_sink.py` |
| SDA-026 | Webhook sink plugin | architecture.md:847-849 | ❌ NOT IMPLEMENTED | Phase 6 |

### 3.3.1 Sinks - New Plugins (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-050 | 🆕 Azure Blob sink | Phase 4 | ✅ IMPLEMENTED | `plugins/azure/blob_sink.py` |

### 3.4 Source Error Routing

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SDA-027 | Source `on_validation_failure` config (REQUIRED) | plugin-protocol.md:222-230 | ✅ IMPLEMENTED | `config_base.py:139-142` |
| SDA-028 | `on_validation_failure`: sink name or "discard" | plugin-protocol.md:228-229 | ✅ IMPLEMENTED | Validator at `config_base.py:144-150` |
| SDA-029 | `QuarantineEvent` recorded even for discard | plugin-protocol.md:230 | ✅ IMPLEMENTED | `ctx.record_validation_error()` |
| SDA-030 | `QuarantineEvent`: run_id, source_id, row_index | plugin-protocol.md:317-322 | ✅ IMPLEMENTED | `schema.py:269-281` |
| SDA-031 | `QuarantineEvent`: raw_row, failure_reason, field_errors | plugin-protocol.md:318-320 | ✅ IMPLEMENTED | `recorder.py:2092-2135` |

---

## 4. ROUTING REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| RTE-001 | RoutingKind: CONTINUE, ROUTE_TO_SINK, FORK_TO_PATHS | plugin-protocol.md:667-674 | ✅ IMPLEMENTED | `enums.py:115-123` |
| RTE-002 | Gate routing via config-driven expressions | plugin-protocol.md:654-683 | ✅ IMPLEMENTED | `expression_parser.py` + `executors.py` |
| RTE-003 | Fork creates child tokens with parent lineage | plugin-protocol.md:764-792 | ✅ IMPLEMENTED | `tokens.py:88-140`, `recorder.py:785-840` |
| RTE-004 | Route resolution map for edge → destination | plugin-protocol.md:682-683 | ✅ IMPLEMENTED | `dag/get_route_resolution_map()` |
| RTE-005 | Routing audit: condition, result, route, destination | plugin-protocol.md:724-726 | ✅ IMPLEMENTED | `recorder.py:1056-1162` |

### 4.1 Gate Configuration Validation (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| RTE-006 | 🆕 Boolean expression route label enforcement | Config validation | ✅ IMPLEMENTED | `config.py:265-296` - enforces {"true": X, "false": Y} for boolean conditions |
| RTE-007 | 🆕 Reserved label protection for routes and forks | Config validation | ✅ IMPLEMENTED | `config.py:227-228, 250-251` - prevents collision with reserved labels |
| RTE-008 | 🆕 Fork destination consistency validation | Config validation | ✅ IMPLEMENTED | `config.py:254-262` - requires fork_to when routes use 'fork' |

---

## 4a. SYSTEM OPERATIONS (Engine-Level, NOT Plugins)

### 4a.1 Gate (Routing Decision)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SOP-001 | Gate evaluates condition expression on row data | plugin-protocol.md:654-658 | ✅ IMPLEMENTED | `executors/463-650` |
| SOP-002 | Gate `routes` map labels to destinations | plugin-protocol.md:668-670 | ✅ IMPLEMENTED | `config.py:161-292` |
| SOP-003 | Gate destinations: `continue` or sink_name | plugin-protocol.md:669-670 | ✅ IMPLEMENTED | `config.py:227-230` |
| SOP-004 | Expression parser uses restricted syntax (NOT eval) | plugin-protocol.md:700-719 | ✅ IMPLEMENTED | `expression_parser.py:1-465` - AST-based |
| SOP-005 | Allowed: field access, comparisons, boolean ops | plugin-protocol.md:705-710 | ✅ IMPLEMENTED | `expression_parser.py:79-172` |
| SOP-006 | NOT allowed: imports, lambdas, arbitrary function calls | plugin-protocol.md:712-718 | ✅ IMPLEMENTED | `row.get()` IS allowed (by design) |

### 4a.2 Fork (Token Splitting)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SOP-007 | Fork creates N child tokens from single parent | plugin-protocol.md:731-734 | ✅ IMPLEMENTED | `tokens.py:114-152` |
| SOP-008 | Child tokens share `row_id`, have unique `token_id` | plugin-protocol.md:765-766 | ✅ IMPLEMENTED | `recorder.py:785-840` |
| SOP-009 | Child tokens record `parent_token_id` | plugin-protocol.md:767 | ✅ IMPLEMENTED | `models.py:108-114` |
| SOP-010 | Parent token terminal state: FORKED | plugin-protocol.md:769 | ✅ IMPLEMENTED | `enums.py:151` |
| SOP-011 | Fork audit: parent_token_id, child_ids, branches | plugin-protocol.md:796-798 | ✅ IMPLEMENTED | fork_group_id, branch_name |

### 4a.3 Coalesce (Token Merging)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SOP-012 | Coalesce merges tokens from parallel paths | plugin-protocol.md:802-806 | ✅ IMPLEMENTED | `coalesce_executor.py:122-187` |
| SOP-013 | Policy: `require_all` - wait for all branches | plugin-protocol.md:828 | ✅ IMPLEMENTED | `coalesce_executor.py:198-199` |
| SOP-014 | Policy: `quorum` - wait for N branches | plugin-protocol.md:829 | ✅ IMPLEMENTED | `coalesce_executor.py:204-206` |
| SOP-015 | Policy: `best_effort` - wait until timeout | plugin-protocol.md:830 | ✅ IMPLEMENTED | `coalesce_executor.py:208-210` |
| SOP-016 | Policy: `first` - take first arrival | plugin-protocol.md:831 | ✅ IMPLEMENTED | `coalesce_executor.py:201-202` |
| SOP-017 | Merge: `union`, `nested`, `select` strategies | plugin-protocol.md:835-839 | ✅ IMPLEMENTED | `coalesce_executor.py:277-301` |
| SOP-018 | Child tokens terminal state: COALESCED | plugin-protocol.md:847 | ✅ IMPLEMENTED | `enums.py:155` |

### 4a.4 Aggregation (Token Batching)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SOP-019 | Aggregation collects tokens until trigger fires | plugin-protocol.md:879-881 | ✅ IMPLEMENTED | `executors/746-793` |
| SOP-020 | Trigger: `count` - fire after N tokens | plugin-protocol.md:900 | ✅ IMPLEMENTED | `triggers.py:95-98` |
| SOP-021 | Trigger: `timeout` - fire after duration | plugin-protocol.md:901 | ✅ IMPLEMENTED | `triggers.py:100-103` |
| SOP-022 | Trigger: `condition` - fire on matching row | plugin-protocol.md:902 | ✅ IMPLEMENTED | `triggers.py:106-116` |
| SOP-023 | Trigger: `end_of_source` - implicit, always checked | plugin-protocol.md:903 | ✅ IMPLEMENTED | `orchestrator.py:639-653` |
| SOP-024 | Multiple triggers combinable (first wins) | plugin-protocol.md:905 | ✅ IMPLEMENTED | `triggers.py:84-118` - OR logic |
| SOP-025 | Input tokens terminal state: CONSUMED_IN_BATCH | plugin-protocol.md:924 | ✅ IMPLEMENTED | `enums.py:154` |
| SOP-026 | Batch lifecycle: draft → executing → completed | plugin-protocol.md:927 | ✅ IMPLEMENTED | `enums.py:44-53` |

### 4a.5 System Operations - New Capabilities (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SOP-027 | 🆕 Token expansion for 1→N deaggregation | Phase 4 | ✅ IMPLEMENTED | `tokens.py:209-246`, `recorder.py:887-956` |
| SOP-028 | 🆕 Coalesce timeout recovery semantics | Phase 4 | ✅ IMPLEMENTED | `coalesce_executor.py:303-453` |
| SOP-029 | 🆕 Coalesce end-of-source flush | Phase 4 | ✅ IMPLEMENTED | `coalesce_executor.py:380-420` |
| SOP-030 | 🆕 Transform error routing with quarantine | Phase 4 | ✅ IMPLEMENTED | `executors/236-275` |
| SOP-031 | 🆕 Aggregation checkpoint/restore | Phase 5 | ✅ IMPLEMENTED | `executors/1034-1085` |
| SOP-032 | 🆕 Gate configuration validation at startup | Phase 4 | ✅ IMPLEMENTED | `config.py:161-292` validators |

---

## 5. DAG EXECUTION REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| DAG-001 | Pipelines compile to DAG | architecture.md:166-184 | ✅ IMPLEMENTED | `dag/228-413` |
| DAG-002 | DAG validation using NetworkX | CLAUDE.md | ✅ IMPLEMENTED | `dag/40-49` wraps `MultiDiGraph` |
| DAG-003 | Acyclicity check on graph | architecture.md:793 | ✅ IMPLEMENTED | `dag/111-134` - `nx.is_directed_acyclic_graph()` |
| DAG-004 | Topological sort for execution | architecture.md:793 | ✅ IMPLEMENTED | `dag/153-165` - `nx.topological_sort()` |
| DAG-005 | Linear pipelines as degenerate DAG | architecture.md:228-241 | ✅ IMPLEMENTED | Linear flow naturally degenerates |

---

## 6. TOKEN IDENTITY REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| TOK-001 | `row_id` = stable source row identity | CLAUDE.md | ✅ IMPLEMENTED | `models.py:80-89` |
| TOK-002 | `token_id` = row instance in DAG path | CLAUDE.md | ✅ IMPLEMENTED | `models.py:93-104` |
| TOK-003 | `parent_token_id` for fork/join lineage | CLAUDE.md | ✅ IMPLEMENTED | `models.py:108-113` |
| TOK-004 | Fork creates child tokens | architecture.md:213-224 | ✅ IMPLEMENTED | `recorder.py:783-845` |
| TOK-005 | Join/coalesce merges tokens | architecture.md:213-224 | ✅ IMPLEMENTED | `recorder.py:847-899` |
| TOK-006 | `token_parents` table for multi-parent joins | subsystems:152-159 | ✅ IMPLEMENTED | `schema.py:120-132` |
| TOK-007 | 🆕 `expand_group_id` for deaggregation | Phase 4 | ✅ IMPLEMENTED | `schema.py:107` |

---

## 7. LANDSCAPE (AUDIT) REQUIREMENTS

### 7.1 Core Tables

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| LND-001 | `runs` table with all specified columns | subsystems:91-101 | ✅ IMPLEMENTED | `schema.py:27-47` |
| LND-002 | `runs.reproducibility_grade` column | subsystems:98 | ✅ IMPLEMENTED | `schema.py:35` |
| LND-003 | `nodes` table for execution graph | subsystems:103-116 | ✅ IMPLEMENTED | `schema.py:51-70` |
| LND-004 | `nodes.determinism` column | subsystems:110 | ✅ IMPLEMENTED | `schema.py:59-60` |
| LND-005 | `nodes.schema_hash` column | subsystems:113 | ✅ IMPLEMENTED | `schema.py:64` |
| LND-006 | `edges` table for graph connections | subsystems:118-128 | ✅ IMPLEMENTED | `schema.py:74-85` |
| LND-007 | `edges.default_mode` column (move/copy) | subsystems:126 | ✅ IMPLEMENTED | `schema.py:82` |
| LND-008 | `rows` table for source rows | subsystems:130-140 | ✅ IMPLEMENTED | `schema.py:89-100` - Payload storage: `source_data_ref` populated via `tokens.py:create_initial_token()` (fix 3399faf) |
| LND-009 | `tokens` table for row instances | subsystems:142-150 | ✅ IMPLEMENTED | `schema.py:104-116` |
| LND-010 | `token_parents` table for joins | subsystems:152-159 | ✅ IMPLEMENTED | `schema.py:120-132` |
| LND-011 | `node_states` table for processing | subsystems:161-179 | ✅ IMPLEMENTED | `schema.py:136-155` |
| LND-012 | `routing_events` table for edge selections | subsystems:181-193 | ✅ IMPLEMENTED | `schema.py:201-214` |
| LND-013 | `calls` table for external calls | subsystems:195-210 | ✅ IMPLEMENTED | `schema.py:159-175` |
| LND-014 | `batches` table for aggregations | subsystems:212-223 | ✅ IMPLEMENTED | `schema.py:218-233` |
| LND-015 | `batch_members` table | subsystems:225-231 | ✅ IMPLEMENTED | `schema.py:235-243` |
| LND-016 | `batch_outputs` table | subsystems:233-239 | ✅ IMPLEMENTED | `schema.py:245-253` |
| LND-017 | `artifacts` table for sink outputs | subsystems:241-252 | ✅ IMPLEMENTED | `schema.py:179-197` |

### 7.2 Audit Recording Requirements

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| LND-018 | Every run with resolved configuration | architecture.md:249-250 | ✅ IMPLEMENTED | `recorder.py:209-264` |
| LND-019 | Every row loaded from source | architecture.md:252 | ✅ IMPLEMENTED | `recorder.py:670-721` + `tokens.py:71-90` (payload storage - fix 3399faf) |
| LND-020 | Every transform with before/after state | architecture.md:253 | ✅ IMPLEMENTED | `recorder.py:960-1086` |
| LND-021 | Every external call recorded | architecture.md:254 | ✅ IMPLEMENTED | `recorder.py:1907-1997` |
| LND-022 | Every routing decision with reason | architecture.md:255 | ✅ IMPLEMENTED | `recorder.py:1107-1227` |
| LND-023 | Every artifact produced | architecture.md:256 | ✅ IMPLEMENTED | `recorder.py:1552-1649` |
| LND-024 | `explain()` API with complete lineage | architecture.md:307-348 | ✅ IMPLEMENTED | `lineage.py:59-142` |
| LND-025 | `explain()` by token_id for DAG precision | architecture.md:315, 345 | ✅ IMPLEMENTED | `lineage.py:62, 89-90` |
| LND-026 | `explain()` by row_id, sink for disambiguation | architecture.md:346 | ✅ IMPLEMENTED | `lineage.py:62-63, 83-87` |

### 7.3 Invariants

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| LND-027 | Run stores resolved config (not just hash) | architecture.md:270 | ✅ IMPLEMENTED | Stores both `config_hash` and `settings_json` |
| LND-028 | External calls link to existing spans | architecture.md:271 | ✅ IMPLEMENTED | `calls.state_id` FK to node_states |
| LND-029 | Strict ordering: transforms by (token_id, node_id, attempt) | architecture.md:272 | ✅ IMPLEMENTED | UniqueConstraint on (token_id, node_id, attempt) |
| LND-030 | No orphan records (foreign keys enforced) | architecture.md:273 | ✅ IMPLEMENTED | All tables have FK constraints |
| LND-031 | `(run_id, row_index)` unique | architecture.md:274 | ✅ IMPLEMENTED | `schema.py:95` |
| LND-032 | Canonical JSON contract versioned | architecture.md:275 | ✅ IMPLEMENTED | `canonical.py:25` |

### 7.4 Landscape - New Tables and Columns (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| LND-033 | 🆕 `nodes.schema_mode` column | WP-11.99 | ✅ IMPLEMENTED | `schema.py:64` |
| LND-034 | 🆕 `nodes.schema_fields_json` column | WP-11.99 | ✅ IMPLEMENTED | `schema.py:65` |
| LND-035 | 🆕 `validation_errors` table | WP-11.99 | ✅ IMPLEMENTED | `schema.py:269-281` |
| LND-036 | 🆕 `transform_errors` table | WP-11.99b | ✅ IMPLEMENTED | `schema.py:288-304` |
| LND-037 | 🆕 `tokens.expand_group_id` column | Deaggregation | ✅ IMPLEMENTED | `schema.py:107` |
| LND-038 | 🆕 `recorder.expand_token()` method | Deaggregation | ✅ IMPLEMENTED | `recorder.py:887-956` |
| LND-039 | 🆕 `batch_outputs` with output_type distinction | Aggregation | ✅ IMPLEMENTED | `schema.py:241-245` |
| LND-040 | 🆕 `checkpoints` table | Crash recovery | ✅ IMPLEMENTED | `schema.py:308-325` |
| LND-041 | 🆕 `RowLineage` model with payload_available flag | Payload degradation | ✅ IMPLEMENTED | `models.py:313-335` |
| LND-042 | 🆕 Call payload auto-persistence | External calls | ✅ IMPLEMENTED | `recorder.py:1953-1961` |
| LND-043 | 🆕 Export status tracking (5 columns on runs) | Governance | ✅ IMPLEMENTED | `schema.py:40-44` |
| LND-044 | 🆕 Export manifest with running hash chain | Governance | ✅ IMPLEMENTED | `exporter.py:131-143` |
| LND-045 | 🆕 HMAC-SHA256 signing on export | Governance | ✅ IMPLEMENTED | `exporter.py:71-92` |
| LND-046 | 🆕 Source row payload auto-persistence | P0-fix-3399faf | ✅ IMPLEMENTED | `tokens.py:76-80` stores payloads before row creation; integration test: `test_source_payload_storage.py` |

---

## 8. CANONICAL JSON REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CAN-001 | Two-phase canonicalization | CLAUDE.md | ✅ IMPLEMENTED | `canonical.py:96-137` |
| CAN-002 | Phase 1: Normalize pandas/numpy to primitives | architecture.md:384-448 | ✅ IMPLEMENTED | `canonical.py:28-93` |
| CAN-003 | Phase 2: RFC 8785/JCS serialization | architecture.md:450-464 | ✅ IMPLEMENTED | `canonical.py:22,135` - rfc8785.dumps() |
| CAN-004 | NaN/Infinity STRICTLY REJECTED | CLAUDE.md | ✅ IMPLEMENTED | `canonical.py:48-53` - raises ValueError |
| CAN-005 | `numpy.int64` → Python int | architecture.md:489 | ✅ IMPLEMENTED | `canonical.py:63-64` |
| CAN-006 | `numpy.float64` → Python float | architecture.md:490 | ✅ IMPLEMENTED | `canonical.py:54-55` |
| CAN-007 | `numpy.bool_` → Python bool | architecture.md:491 | ✅ IMPLEMENTED | `canonical.py:65-66` |
| CAN-008 | `pandas.Timestamp` → UTC ISO8601 | architecture.md:492 | ✅ IMPLEMENTED | `canonical.py:71-75` |
| CAN-009 | NaT, NA → null | architecture.md:493 | ✅ IMPLEMENTED | `canonical.py:78-79` |
| CAN-010 | Version string `sha256-rfc8785-v1` | CLAUDE.md | ✅ IMPLEMENTED | `canonical.py:25` |
| CAN-011 | Cross-process hash stability test | architecture.md:931 | ✅ IMPLEMENTED | `test_canonical.py:235-369` |

---

## 9. PAYLOAD STORE REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| PLD-001 | PayloadStore protocol with put/get/exists | architecture.md:524-530 | ✅ IMPLEMENTED | `payload_store.py:16-70` |
| PLD-002 | PayloadRef return type | architecture.md:527 | ✅ IMPLEMENTED | Returns SHA-256 hex digest |
| PLD-003 | Filesystem backend | subsystems:670 | ✅ IMPLEMENTED | `payload_store.py:72-129` |
| PLD-004 | S3/blob storage backend | subsystems:670 | ❌ NOT IMPLEMENTED | Phase 7 |
| PLD-005 | Inline backend | subsystems:670 | ❌ NOT IMPLEMENTED | Not planned |
| PLD-006 | Retention policies | architecture.md:539-549 | ⚠️ PARTIAL | Config exists; purge works; **CLI `run` command does not instantiate PayloadStore** |
| PLD-007 | Hash retained after payload purge | architecture.md:546 | ✅ IMPLEMENTED | Schema separates hash from ref |
| PLD-008 | Optional compression | subsystems:669 | ❌ NOT IMPLEMENTED | Not planned |

### 9.1 Payload Store - Implementation Notes (🆕)

**Engine Status:** ✅ COMPLETE
- `Orchestrator.run()` and `resume()` accept `payload_store` parameter (orchestrator.py:406, 1136)
- TokenManager stores payloads before creating row records (tokens.py:73-95)
- Integration test passes when payload_store is provided (tests/integration/test_source_payload_storage.py)

**CLI Integration Status:** ⚠️ PARTIAL
- ✅ `resume` command: Creates FilesystemPayloadStore and passes to orchestrator (cli.py:925)
- ✅ `purge` command: Creates FilesystemPayloadStore for retention cleanup (cli.py:632)
- ❌ **`run` command: Does NOT create or pass PayloadStore** (cli.py:269-396)

**Impact:**
- Normal user-facing runs (`elspeth run -s settings.yaml --execute`) do NOT persist source row payloads
- Engine infrastructure is complete but not wired through CLI entry point
- `resume` works because it instantiates payload store; `run` does not
- Violates CLAUDE.md non-negotiable audit requirement: "Source entry - Raw data stored before any processing"

**Required Fix:** See CFG-039 and CLI-016

---

## 10. FAILURE SEMANTICS REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| FAI-001 | Token terminal states: COMPLETED | architecture.md:575 | ✅ IMPLEMENTED | `enums.py:149` |
| FAI-002 | Token terminal states: ROUTED | architecture.md:576 | ✅ IMPLEMENTED | `enums.py:150` |
| FAI-003 | Token terminal states: FORKED | architecture.md:577 | ✅ IMPLEMENTED | `enums.py:151` |
| FAI-004 | Token terminal states: CONSUMED_IN_BATCH | architecture.md:578 | ✅ IMPLEMENTED | `enums.py:154` |
| FAI-005 | Token terminal states: COALESCED | architecture.md:579 | ✅ IMPLEMENTED | `enums.py:155` |
| FAI-006 | Token terminal states: QUARANTINED | architecture.md:580 | ✅ IMPLEMENTED | `enums.py:153` |
| FAI-007 | Token terminal states: FAILED | architecture.md:581 | ✅ IMPLEMENTED | `enums.py:152` |
| FAI-008 | Terminal states DERIVED, not stored | architecture.md:571-572 | ✅ IMPLEMENTED | `schema.py:344-354` - `token_outcomes` table; `recorder.py:1651-1713` - explicit recording |
| FAI-009 | Every token reaches exactly one terminal state | architecture.md:569 | ⚠️ PARTIAL | 17 recording sites exist but orchestrator quarantine flow and coalesce parent tokens have gaps (P1 bugs) |
| FAI-010 | `TransformResult` with status/row/reason/retryable | architecture.md:590-598 | ⚠️ PARTIAL | retryable flag exists but only exceptions trigger retries, not `TransformResult.error(retryable=True)` (P2 bug) |
| FAI-011 | Retry key unique | architecture.md:603-605 | ✅ IMPLEMENTED | Uses (token_id, node_id, attempt) |
| FAI-012 | Each retry attempt recorded separately | architecture.md:604 | ✅ IMPLEMENTED | `processor.py:131-190` |
| FAI-013 | Backoff metadata captured | architecture.md:606 | ✅ IMPLEMENTED | `retry.py:47-58` |
| FAI-014 | At-least-once delivery | architecture.md:619-621 | ✅ IMPLEMENTED | `protocols.py:432-434` |

### 10.1 Failure Semantics - New Terminal States (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| FAI-015 | 🆕 Token terminal states: EXPANDED | Deaggregation | ✅ IMPLEMENTED | `enums.py:168` |
| FAI-016 | 🆕 Token non-terminal states: BUFFERED | Aggregation | ✅ IMPLEMENTED | `enums.py:171` |
| FAI-017 | 🆕 Token outcomes explicitly recorded to token_outcomes table | AUD-001 | ✅ IMPLEMENTED | `schema.py:344-354`, `recorder.py:1651-1713` |

---

## 11. EXTERNAL CALL RECORDING REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| EXT-001 | Record: provider identifier | architecture.md:695 | ✅ IMPLEMENTED | CallType enum + provider metadata |
| EXT-002 | Record: model/version | architecture.md:696 | ✅ IMPLEMENTED | provider in request_data, model in response_data, both hashed and stored via payload refs; `llm.py:148-156, 187-191`, `recorder.py:2005-2015` |
| EXT-003 | Record: request hash + payload ref | architecture.md:697 | ✅ IMPLEMENTED | `schema.py:196-197` |
| EXT-004 | Record: response hash + payload ref | architecture.md:698 | ✅ IMPLEMENTED | `schema.py:198-199` |
| EXT-005 | Record: latency, status code, error details | architecture.md:699 | ✅ IMPLEMENTED | `schema.py:195,201-202` |
| EXT-006 | Run modes: live, replay, verify | architecture.md:655-660 | ✅ IMPLEMENTED | `config.py:597-600` - RunMode enum |
| EXT-007 | Verify mode uses DeepDiff | architecture.md:667-687 | ✅ IMPLEMENTED | `verifier.py:1-95` - CallVerifier with DeepDiff comparison; `test_verifier.py` |
| EXT-008 | Reproducibility grades: FULL_REPRODUCIBLE | architecture.md:644 | ✅ IMPLEMENTED | `reproducibility.py:28-36` |
| EXT-009 | Reproducibility grades: REPLAY_REPRODUCIBLE | architecture.md:644 | ✅ IMPLEMENTED | `reproducibility.py:34` |
| EXT-010 | Reproducibility grades: ATTRIBUTABLE_ONLY | architecture.md:644 | ✅ IMPLEMENTED | `reproducibility.py:36` |

---

## 12. DATA GOVERNANCE REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| GOV-001 | Secrets NEVER stored - HMAC fingerprint only | CLAUDE.md | ✅ IMPLEMENTED | `config.py:904-1033` - two-phase fingerprinting |
| GOV-002 | `secret_fingerprint()` function using HMAC | architecture.md:729-737 | ✅ IMPLEMENTED | `config.py:904-963` |
| GOV-003 | Fingerprint key loaded from environment | architecture.md:746-749 | ✅ IMPLEMENTED | `ELSPETH_FINGERPRINT_KEY` |
| GOV-004 | Configurable redaction profiles | architecture.md:708-711 | ❌ NOT IMPLEMENTED | Phase 5+ |
| GOV-005 | Access levels: Operator (redacted) | architecture.md:753-755 | ❌ NOT IMPLEMENTED | No access control |
| GOV-006 | Access levels: Auditor (full) | architecture.md:756 | ❌ NOT IMPLEMENTED | No access control |
| GOV-007 | Access levels: Admin (retention/purge) | architecture.md:757 | ⚠️ PARTIAL | Purge exists; no auth |
| GOV-008 | `elspeth explain --full` requires ELSPETH_AUDIT_ACCESS | architecture.md:760-766 | ❌ NOT IMPLEMENTED | No access control |
| GOV-009 | 🆕 Azure Key Vault integration for fingerprint key | Security | ✅ IMPLEMENTED | `fingerprint.py:58-99` - `ELSPETH_KEYVAULT_URL`, `ELSPETH_KEYVAULT_SECRET_NAME` |
| GOV-010 | 🆕 Recursive secret fingerprinting (nested structures) | Security | ✅ IMPLEMENTED | Config system handles nested secret structures |

---

## 13. PLUGIN SYSTEM REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| PLG-001 | pluggy hookspecs for Source, Transform, Sink | plugin-protocol.md:22-30 | ✅ IMPLEMENTED | `hookspecs.py:41-82` |
| PLG-002 | Plugins are system code, NOT user-provided | plugin-protocol.md:23-24 | ✅ IMPLEMENTED | CLAUDE.md policy |
| PLG-003 | Plugins touch row contents; System Ops touch tokens | plugin-protocol.md:26-44 | ✅ IMPLEMENTED | Architecture documented |
| PLG-004 | BaseSource, BaseTransform, BaseSink base classes | plugin-protocol.md:192-620 | ✅ IMPLEMENTED | `base.py:25-330` |
| PLG-005 | RowOutcome terminal states model | plugin-protocol.md | ✅ IMPLEMENTED | `enums.py:139-156` |
| PLG-006 | Plugin determinism declaration (attribute) | plugin-protocol.md:1002-1016 | ✅ IMPLEMENTED | All plugins declare |
| PLG-007 | External Data (Source input): Zero trust, coercion OK | plugin-protocol.md:75 | ✅ IMPLEMENTED | Sources use `allow_coercion=True` |
| PLG-008 | Pipeline Data (Post-source): Elevated trust, no coerce | plugin-protocol.md:76 | ✅ IMPLEMENTED | Transforms use `allow_coercion=False` |
| PLG-009 | Our Code (Plugin internals): Full trust, let crash | plugin-protocol.md:77 | ✅ IMPLEMENTED | No defensive patterns |
| PLG-010 | Type-safe ≠ operation-safe (wrap VALUE operations) | plugin-protocol.md:79-91 | ✅ IMPLEMENTED | `executors/224-249` |
| PLG-011 | Sources MAY coerce types; Transforms/Sinks MUST NOT | plugin-protocol.md:111-119 | ✅ IMPLEMENTED | Schema factory parameter |
| PLG-012 | Input/output schema declaration on plugins | plugin-protocol.md:200-207 | ✅ IMPLEMENTED | `base.py:44-45,78-79` |
| PLG-013 | Engine validates schema compatibility at construction | plugin-protocol.md:1024-1029 | ✅ IMPLEMENTED | `schema_validator.py` |

### 13.1 Plugin Discovery - Dynamic Discovery (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| PLG-014 | 🆕 Directory-based plugin discovery | Dynamic refactor | ✅ IMPLEMENTED | `discovery.py:49-85` |
| PLG-015 | 🆕 Multi-directory plugin scanning | Dynamic refactor | ✅ IMPLEMENTED | `discovery.py:157-206` - PLUGIN_SCAN_CONFIG |
| PLG-016 | 🆕 Dynamic hookimpl generation | Dynamic refactor | ✅ IMPLEMENTED | `discovery.py:233-268` |
| PLG-017 | 🆕 `PluginManager.register_builtin_plugins()` | Dynamic refactor | ✅ IMPLEMENTED | `manager.py:131-150` |
| PLG-018 | 🆕 Plugin description extraction from docstrings | Dynamic refactor | ✅ IMPLEMENTED | `discovery.py:209-230` |

---

## 14. ENGINE REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| ENG-001 | RowProcessor with span lifecycle | architecture.md:950 | ✅ IMPLEMENTED | `processor.py:50-530` |
| ENG-002 | Retry with attempt tracking (tenacity) | architecture.md:951 | ✅ IMPLEMENTED | `retry.py:25-31,128-182` |
| ENG-003 | Artifact pipeline (topological sort) | architecture.md:952 | ✅ IMPLEMENTED | `dag.py` + `executors/938-1050` |
| ENG-004 | Standard orchestrator | architecture.md:953 | ✅ IMPLEMENTED | `orchestrator.py:88-816` |
| ENG-005 | OpenTelemetry span emission | architecture.md:954 | ✅ IMPLEMENTED | `spans.py:47-243` |
| ENG-006 | Aggregation accept/trigger/flush lifecycle | subsystems:387-391 | ✅ IMPLEMENTED | `executors/665-935` |
| ENG-007 | Aggregation crash recovery via query | subsystems:476-495 | ✅ IMPLEMENTED | `processor.py:137-139` |

### 14.1 Engine - Batch Processing Architecture (🆕)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| ENG-008 | 🆕 `is_batch_aware` flag for transforms | Batch processing | ✅ IMPLEMENTED | `base.py:54` |
| ENG-009 | 🆕 Output mode: single (batch → single row) | Batch processing | ✅ IMPLEMENTED | `processor.py:215-231` |
| ENG-010 | 🆕 Output mode: passthrough (enrich originals) | Batch processing | ✅ IMPLEMENTED | `processor.py:233-290` |
| ENG-011 | 🆕 Output mode: transform (N→M deaggregation) | Batch processing | ✅ IMPLEMENTED | `processor.py:292-351` |
| ENG-012 | 🆕 `TransformResult.success_multi()` for multi-row | Batch processing | ✅ IMPLEMENTED | `results.py:102-117` |
| ENG-013 | 🆕 `creates_tokens` flag for deaggregation | Batch processing | ✅ IMPLEMENTED | `base.py:56-62` |
| ENG-014 | 🆕 BatchPendingError for Azure Batch control flow | LLM Batch | ✅ IMPLEMENTED | `batch_errors.py:14-78` |

---

## 15. PRODUCTION HARDENING REQUIREMENTS (Phase 5)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| PRD-001 | Checkpointing with replay support | architecture.md:969 | ✅ IMPLEMENTED | `checkpoint/manager.py` |
| PRD-002 | Rate limiting using pyrate-limiter | architecture.md:970 | ✅ IMPLEMENTED | `rate_limit/limiter.py` |
| PRD-003 | Retention and purge jobs | architecture.md:971 | ✅ IMPLEMENTED | `retention/purge.py` |
| PRD-004 | Redaction profiles | architecture.md:972 | ❌ NOT IMPLEMENTED | Phase 5+ |
| PRD-005 | Concurrent processing | README.md:183 | 🔀 DIVERGED | Config scaffolding exists (`max_workers`) but not integrated into orchestrator; LLM plugins use separate pooled execution |

---

## 16. TECHNOLOGY STACK REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| TSK-001 | CLI: Typer | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:22` |
| TSK-002 | TUI: Textual | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:23` |
| TSK-003 | Configuration: Dynaconf + Pydantic | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:26-27` |
| TSK-004 | Plugins: pluggy | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:30` |
| TSK-005 | Data: pandas | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:33` |
| TSK-006 | HTTP: httpx | architecture.md:781 | ✅ IMPLEMENTED | `pyproject.toml:36` |
| TSK-007 | Database: SQLAlchemy Core | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:39` |
| TSK-008 | Migrations: Alembic | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:40` |
| TSK-009 | Retries: tenacity | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:43` |
| TSK-010 | Canonical JSON: rfc8785 | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:47` |
| TSK-011 | DAG Validation: NetworkX | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:50` |
| TSK-012 | Observability: OpenTelemetry | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:53-55` |
| TSK-013 | Tracing UI: Jaeger | CLAUDE.md | ⚠️ PARTIAL | OTel exports; no setup docs |
| TSK-014 | Logging: structlog | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:58` |
| TSK-015 | Rate Limiting: pyrate-limiter | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:61` |
| TSK-016 | Diffing: DeepDiff | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:64` |
| TSK-017 | Property Testing: Hypothesis | CLAUDE.md | ✅ IMPLEMENTED | `pyproject.toml:72` |
| TSK-018 | LLM: LiteLLM | CLAUDE.md | 🔀 DIVERGED | Declared in pyproject.toml:92 but not used; direct openai library used instead |
| TSK-019 | 🆕 Template Engine: Jinja2 | Phase 6 | ✅ IMPLEMENTED | `pyproject.toml:89,100` - used for prompt templates and Azure path templating |

---

## 17. LANDSCAPE EXPORT REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| EXP-001 | Export audit trail to configured sink | This plan | ✅ IMPLEMENTED | `exporter.py:94-143` |
| EXP-002 | Optional HMAC signing per record | This plan | ✅ IMPLEMENTED | `exporter.py:71-92` |
| EXP-003 | Manifest with final hash for tamper detection | This plan | ✅ IMPLEMENTED | `exporter.py:132-143` |
| EXP-004 | CSV and JSON format options | This plan | ✅ IMPLEMENTED | Both formats supported; CSV uses export_run_grouped() for type-specific files (exporter.py:352-382), JSON uses export_run() for JSONL stream (exporter.py:94-143) |
| EXP-005 | Export happens post-run via config, not CLI | This plan | ✅ IMPLEMENTED | Settings YAML configures export |
| EXP-006 | Include all record types (batches, token_parents) | Code review | ✅ IMPLEMENTED | All 12 record types exported |

---

## 18. RETRY INTEGRATION REQUIREMENTS

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| RTY-001 | `RetryConfig.from_settings()` maps Pydantic → internal | WP-15 | ✅ IMPLEMENTED | `retry.py:86-101` |
| RTY-002 | `execute_transform()` accepts attempt parameter | WP-15 | ✅ IMPLEMENTED | `executors/116-124` |
| RTY-003 | RowProcessor uses RetryManager for transform exec | WP-15 | ✅ IMPLEMENTED | `processor.py:131-190` |
| RTY-004 | Transient exceptions retried; programming errors not | WP-15 | ✅ IMPLEMENTED | `processor.py:426-429` |
| RTY-005 | MaxRetriesExceeded returns RowOutcome.FAILED | WP-15 | ✅ IMPLEMENTED | `processor.py:700-710` |
| RTY-006 | Each attempt creates separate node_state record | WP-15 | ✅ IMPLEMENTED | `executors/160-166` |
| RTY-007 | Orchestrator creates RetryManager from RetrySettings | WP-15 | ✅ IMPLEMENTED | `orchestrator.py:538-554` |

---

## 19. AUDIT TRAIL INTEGRITY REQUIREMENTS (🆕 from Bug Analysis)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| AUD-001 | 🆕 Every token reaches exactly one terminal state | Bug analysis | ✅ IMPLEMENTED | `token_outcomes` table with partial unique index; 17 recording sites in processor.py |
| AUD-002 | 🆕 Explicit routing events (no inference from absence) | Bug analysis | ✅ FIXED | Closed: P1-2026-01-19-gate-continue-routing-not-recorded.md |
| AUD-003 | 🆕 Batch trigger type recorded | Bug analysis | ✅ FIXED | Closed: P1-2026-01-20-batch-trigger-type-not-recorded.md |
| AUD-004 | 🆕 Validation errors include node_id | Bug analysis | ✅ FIXED | Closed: P1-2026-01-19-validation-errors-missing-node-id.md |
| AUD-005 | 🆕 Payload store integrity verification on read | Bug analysis | ✅ FIXED | Closed: P1-2026-01-19-payload-store-integrity-and-hash-validation-missing.md |
| AUD-006 | 🆕 Checkpoints created AFTER sink write | Bug analysis | ✅ FIXED | Closed: P0-2026-01-19-checkpoint-before-sink-write.md |
| AUD-007 | 🆕 Aggregation flushes create node_state records | Bug analysis | ✅ FIXED | Closed: P0-2026-01-19-aggregation-batch-status-and-audit-missing.md |
| AUD-008 | 🆕 Transform on_error sinks validated at startup | Bug analysis | ✅ FIXED | Closed: P1-2026-01-19-transform-on-error-sink-validation.md |
| AUD-009 | 🆕 Source row payloads persisted before processing | Bug analysis | ✅ FIXED | Closed: P0-2026-01-22-source-row-payloads-never-persisted.md (commit 3399faf) |
| AUD-010 | 🆕 Fork destinations validated at startup | Bug analysis | ✅ FIXED | Closed: P1-2026-01-20-fork-to-paths-empty-destinations-allowed.md |
| AUD-011 | 🆕 Schema compatibility checks handle optional/Any types | Bug analysis | ✅ FIXED | Closed: P1-2026-01-20-schema-compatibility-check-fails-on-optional-and-any.md |

---

## 20. DECLARATIVE DAG WIRING REQUIREMENTS (🆕 RC-2.5)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| DAG-006 | 🆕 Explicit `on_success` connection naming for sources | ADR-005 | ✅ IMPLEMENTED | `config.py` — `SourceSettings.on_success` |
| DAG-007 | 🆕 Explicit `input` connection declaration for transforms | ADR-005 | ✅ IMPLEMENTED | `config.py` — `TransformSettings.input` |
| DAG-008 | 🆕 Explicit `on_success` output connection for transforms | ADR-005 | ✅ IMPLEMENTED | `config.py` — `TransformSettings.on_success` |
| DAG-009 | 🆕 Gate `input` connection declaration | ADR-005 | ✅ IMPLEMENTED | `config.py` — `GateSettings.input` |
| DAG-010 | 🆕 Connection name validation (character classes) | ADR-005 | ✅ IMPLEMENTED | `config.py` validators, `test_connection_name_validation.py` |
| DAG-011 | 🆕 Reserved connection name protection | ADR-005 | ✅ IMPLEMENTED | `config.py` — reserved names like `continue` prevented |
| DAG-012 | 🆕 DAGNavigator for edge traversal and next-node resolution | Refactoring | ✅ IMPLEMENTED | `engine/dag_navigator.py` |
| DAG-013 | 🆕 Node-ID based work queue (replaces step index) | ADR-005 | ✅ IMPLEMENTED | `processor.py` — `WorkItem.node_id` |
| DAG-014 | 🆕 Gate route fan-out to multiple processing connections | ADR-005 | ✅ IMPLEMENTED | `dag/builder.py` |
| DAG-015 | 🆕 Gate-to-gate route jump resolution | Engine | ✅ IMPLEMENTED | `processor.py`, `dag_navigator.py` |

---

## 21. SQLCIPHER ENCRYPTION REQUIREMENTS (🆕 RC-2.5)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| SEC-001 | 🆕 SQLCipher encryption-at-rest for audit database | Security | ✅ IMPLEMENTED | `core/landscape/database.py` |
| SEC-002 | 🆕 `audit.passphrase` config option | Security | ✅ IMPLEMENTED | `core/config.py` — `LandscapeSettings.passphrase` |
| SEC-003 | 🆕 `ELSPETH_AUDIT_PASSPHRASE` environment variable | Security | ✅ IMPLEMENTED | `cli.py`, `cli_helpers.py` |
| SEC-004 | 🆕 SQLCipher URI option preservation | Security | ✅ IMPLEMENTED | `database.py` — URI parsing with passphrase guard |
| SEC-005 | 🆕 MCP passphrase forwarding for encrypted databases | Security | ✅ IMPLEMENTED | `mcp/__init__.py` entrypoint |
| SEC-006 | 🆕 Backend validation (reject sqlcipher without pysqlcipher3) | Security | ✅ IMPLEMENTED | `database.py` |

---

## 22. CHAOS TESTING REQUIREMENTS (🆕 RC-2.5)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| CHT-001 | 🆕 ChaosWeb server for web_scrape testing | Testing | ✅ IMPLEMENTED | `testing/chaosweb/server.py` |
| CHT-002 | 🆕 ChaosWeb HTTP error injection (4xx, 5xx, timeouts) | Testing | ✅ IMPLEMENTED | `testing/chaosweb/error_injector.py` |
| CHT-003 | 🆕 ChaosWeb content generation with configurable HTML | Testing | ✅ IMPLEMENTED | `testing/chaosweb/content_generator.py` |
| CHT-004 | 🆕 ChaosWeb preset profiles (gentle, realistic, stress) | Testing | ✅ IMPLEMENTED | `testing/chaosweb/presets/` (5 presets) |
| CHT-005 | 🆕 ChaosWeb metrics recording | Testing | ✅ IMPLEMENTED | `testing/chaosweb/metrics.py` |
| CHT-006 | 🆕 ChaosWeb CLI (`chaosweb serve`) | Testing | ✅ IMPLEMENTED | `testing/chaosweb/cli.py` |
| CHT-007 | 🆕 ChaosWeb pytest fixtures | Testing | ✅ IMPLEMENTED | `testing/chaosweb/__init__.py`, `tests/fixtures/chaosweb.py` |
| CHT-008 | 🆕 ChaosEngine shared core via composition | Refactoring | ✅ IMPLEMENTED | `testing/chaosengine/` (7 modules) |

---

## 23. REFACTORING REQUIREMENTS (🆕 RC-2.5)

| Requirement ID | Requirement | Source | Status | Evidence |
|----------------|-------------|--------|--------|----------|
| REF-001 | 🆕 executors.py split into one-file-per-executor package | Code quality | ✅ IMPLEMENTED | `engine/executors/` (transform, gate, sink, aggregation, types) |
| REF-002 | 🆕 dag.py split into dag/ package | Code quality | ✅ IMPLEMENTED | `core/dag/` (builder, graph, models) |
| REF-003 | 🆕 MCP server.py split into domain modules | Code quality | ✅ IMPLEMENTED | `mcp/analyzers/` (contracts, diagnostics, queries, reports) |
| REF-004 | 🆕 Dead protocol removal (GateProtocol, CoalesceProtocol) | Cleanup | ✅ IMPLEMENTED | Gate plugins fully removed from codebase |
| REF-005 | 🆕 BaseMultiQueryTransform deduplication | Code quality | ✅ IMPLEMENTED | `plugins/llm/base_multi_query.py` |
| REF-006 | 🆕 on_error/on_success as plain attributes (not properties) | Simplification | ✅ IMPLEMENTED | `plugins/base.py`, all transforms |
| REF-007 | 🆕 Tier model allowlist split into per-module files | CI/CD | ✅ IMPLEMENTED | `config/cicd/enforce_tier_model/` (10 YAML files) |
| REF-008 | 🆕 CLI _orchestrator_context extraction | Code quality | ✅ IMPLEMENTED | `cli.py` — shared context manager for run/resume |

---

## SUMMARY BY PHASE

### Phase 1-3: Core Infrastructure ✅ COMPLETE
- Canonical JSON: 11/11 (100%)
- Landscape Tables: 17/17 + 13 new = 30/30 (100%)
- Audit Recording: 9/9 (100%)
- Plugin System: 13/13 + 5 new = 18/18 (100%)
- DAG Execution: 5/5 (100%)
- Token Identity: 6/6 + 1 new = 7/7 (100%)
- System Operations: 26/26 + 6 new = 32/32 (100%)
- Routing: 5/5 (100%)
- Retry: 7/7 (100%)

### Phase 4: CLI & Basic Plugins ✅ MOSTLY COMPLETE
- Configuration: 24/24 + 14 new = 38/38 (100%)
- CLI: 9/9 + 6 new = 15/15 (100%)
- SDA Model: 31/31 + 19 new = 50/50 (100%)
- Engine: 7/7 + 7 new = 14/14 (100%)

### Phase 5: Production Hardening ⚠️ PARTIAL
- Production: 3.5/5 (70%)
- Payload Store: 4.5/8 (56%)
- Governance: 3/8 (38%)

### Phase 6: LLM & External Calls ✅ SIGNIFICANTLY COMPLETE
- LLM Transforms: 8/8 NEW (100%)
- External Calls: 7/10 (70%)

### RC-2.5: Routing, Security, Testing ✅ COMPLETE
- Declarative DAG Wiring: 10/10 NEW (100%)
- SQLCipher Encryption: 6/6 NEW (100%)
- Chaos Testing: 8/8 NEW (100%)
- Refactoring: 8/8 NEW (100%)

---

## CRITICAL DIVERGENCES FROM ORIGINAL SPEC

| Issue | Original Spec | Actual Implementation | Verdict |
|-------|---------------|----------------------|---------|
| Landscape config | `backend` + `path` | SQLAlchemy URL | ✅ Better (more flexible) |
| Retention config | Split by type | Unified `retention_days` | ⚠️ Less granular |
| Profile system | `--profile` flag | Not implemented | ❌ Deferred |
| Pack defaults | `packs/*/defaults.yaml` | Not implemented | ❌ Deferred |
| Retry key | (run_id, row_id, seq, attempt) | (token_id, node_id, attempt) | ✅ Same semantics |
| Access control | Three-tier roles | Not implemented | ❌ Phase 5+ |
| Transform output | 1→1 only | 1→1 default, 1→N via success_multi() | ✅ Extended |
| Plugin discovery | Static hookimpl files | Dynamic directory scanning | ✅ Better (no manual registration) |

---

## OPEN BUGS AFFECTING REQUIREMENTS

The following P1/P2 bugs indicate requirements gaps that need attention:

### High Priority (P1) - 29 open bugs

Critical bugs affecting core functionality:

| Priority | Bug | Requirement Impact |
|----------|-----|-------------------|
| P1 | coalesce-timeouts-never-fired | SOP-015 (best_effort policy) broken |
| P1 | coalesce-late-arrivals-duplicate-merge | SOP-012 (token merging) incomplete |
| P1 | coalesce-parent-outcomes-missing | FAI-005 (COALESCED state) partial |
| P1 | explain-returns-arbitrary-token | CLI-003, LND-025 (explain precision) broken |
| P1 | artifact-descriptor-leaks-secrets | GOV-001 (secret handling) violated |
| P1 | decimal-nan-infinity-bypass-rejection | CAN-004 (NaN rejection) bypassed |
| P1 | csvsource-malformed-rows-crash | SDA-003 (CSV source) fragile |
| P1 | jsonsource-array-parse-errors-crash | SDA-004 (JSON source) fragile |
| P1 | csvsink-mode-unvalidated-truncation | SDA-023 (CSV sink) unsafe |
| P1 | databasesink-noncanonical-hash | CAN-001 (canonical JSON) violated |
| P1 | azure-batch-missing-audit-payloads | LND-021 (external calls) partial |
| P1 | schema-validator-ignores-dag-routing | PLG-013 (schema validation) incomplete |
| P1 | orchestrator-source-quarantine-outcome-missing | FAI-006 (QUARANTINED state) partial |
| P1 | recovery-skips-rows-multi-sink | PRD-001 (checkpoint recovery) broken |
| P1 | duplicate-gate-names-overwrite-mapping | RTE-002 (gate routing) unsafe |
| P1 | duplicate-branch-names-break-coalesce | SOP-012 (coalesce) unsafe |

### Medium Priority (P2) - 40 open bugs

Affecting data integrity, performance, or edge cases. See `docs/bugs/open/P2-*.md` for full list.

### Low Priority (P3) - 19 open bugs

Minor issues, documentation, or tech debt. See `docs/bugs/open/P3-*.md` for full list.

---

*Audit performed by 10 parallel agents on 2026-01-22 (comprehensive post-P0-fix update)*
*RC-2.5 additions appended 2026-02-12 (declarative DAG wiring, SQLCipher, ChaosWeb, refactoring)*
*Total requirements: 365 | Implemented: 323 (88%) | Partial: 28 (8%) | Not Implemented: 14 (4%)*
*New requirements in RC-2.5: 42 (DAG wiring, SQLCipher, chaos testing, refactoring)*
