# RC-1 Changelog

Historical changelog reconstructed from branch history. Covers the initial framework build (Pre-RC1) and the hardening sprint leading to RC2.

For RC2 sub-releases, see [CHANGELOG-RC2.md](CHANGELOG-RC2.md).
For RC3+, see [CHANGELOG.md](CHANGELOG.md).

---

## Pre-RC1 (Jan 12–22, 2026) — Framework Build

762 commits taking ELSPETH from an empty scaffold to a working auditable SDA pipeline framework with CLI, plugin system, DAG execution, LLM integration, Azure ecosystem support, and crash recovery.

### Highlights

- **Canonical JSON and audit hashing** — built on RFC 8785 (JCS), with two-phase normalization handling pandas, numpy, datetime, bytes, Decimal, NaN/Infinity rejection, and golden hash stability tests
- **Landscape audit trail** — full SQLAlchemy schema, recorder, exporter, lineage explorer, token outcomes (AUD-001), schema compatibility checking, and call recording
- **DAG execution engine** — linear pipelines, fork/join with coalesce, aggregation with count/timeout triggers, deaggregation, batch processing, retry with backoff, checkpoint/resume, and pooled LLM execution with AIMD throttling
- **Azure LLM ecosystem** — Azure OpenAI, OpenRouter, multi-query assessment, batch API, content safety, prompt shield, Azure Blob Storage source/sink, Key Vault fingerprint key, and SAS token auth
- **Plugin system** — pluggy with dynamic discovery, protocol enforcement via mypy, and 20+ plugins across sources, transforms, and sinks

### Added

- **Canonical JSON:** Two-phase canonicalization — Phase 1 normalizes pandas/numpy/datetime/bytes/Decimal types; Phase 2 delegates to `rfc8785` for RFC 8785/JCS deterministic serialization. NaN and Infinity strictly rejected. Stable hashing via SHA-256 of canonical form. Golden hash stability tests for cross-version consistency
- **Landscape:** SQLAlchemy table definitions and dataclass models for the full audit schema. `LandscapeRecorder` with run lifecycle, node state recording, `record_call()` for external call audit, token outcome recording (AUD-001), and lineage `explain()`. `LandscapeExporter` for portable audit exports. Schema compatibility checking for SQLite databases. SQLite PRAGMAs applied in all constructors
- **DAG:** `ExecutionGraph` backed by NetworkX with acyclicity validation, topological sort, source/sink constraints, `from_config()` construction, coalesce node creation, and fork-child-to-coalesce linking
- **Plugin System:** pluggy-based registration with `PluginProtocol` v1.5 (multi-row output, `creates_tokens`, `is_batch_aware`). Dynamic plugin discovery replacing static `PLUGIN_REGISTRY`. `PluginManager` with import error crash and duplicate name detection
- **Configuration:** Pydantic settings schema with Dynaconf-based multi-source loading. Template file expansion (`_expand_template_files`). Run mode setting (live/replay/verify). `.env` auto-loading via python-dotenv. Secret fingerprinting with fail-closed recursive detection and HMAC-based audit. Azure Key Vault support for fingerprint key storage
- **Engine:** `Orchestrator` with full run lifecycle, `RowProcessor` with DAG traversal and work queue, `RetryManager` with tenacity-based backoff. `AggregationExecutor` with `execute_flush()`, buffer management, and end-of-source flush to prevent data loss. `CoalesceExecutor` wired into processor. `BatchAdapter` with passthrough and transform output modes, BUFFERED semantics, and deaggregation support. Checkpoint creation moved to after sink writes for crash safety
- **Checkpoint:** `RecoveryManager` with `get_unprocessed_row_data()`, topology hash validation, and NullSource for resume operations. CSVSink append mode for resume support. CLI `--execute` flag on resume command
- **Sources:** CSVSource, JSONSource (with JSONDecodeError handling and JSONL parse error quarantine), NullSource for resume, AzureBlobSource with SAS token and managed identity auth
- **Transforms:** OpenRouterLLMTransform, AzureLLMTransform, AzureBatchLLMTransform with batch API call audit. AzureMultiQueryLLMTransform with parallel row processing, all-or-nothing semantics, case study context, and JSON parsing. KeywordFilter with pattern matching and context extraction. AzureContentSafety with threshold checking. AzurePromptShield with jailbreak and prompt injection detection. FieldMapper, Passthrough, JSONExplode (deaggregation), BatchReplicate, Truncate transform for field length enforcement
- **LLM Infrastructure:** `BaseLLMTransform` base class. Jinja2 prompt template engine with lookup/source namespaces, source metadata fields, and `system_prompt_file` support. Audited client infrastructure for LLM and HTTP calls. CallReplayer and CallVerifier for replay/verify modes. Pooled execution with AIMD throttle (backoff on capacity errors, recovery on success, statistics for audit), reorder buffer with timing, `PooledExecutor` with batch execution and per-row state. LLM batch aggregation with polymorphic dispatch
- **Sinks:** CSVSink (with append mode), JSONSink, DatabaseSink, AzureBlobSink
- **Content Filtering:** Azure Content Safety transform with configurable thresholds. Azure Prompt Shield with fail-closed security posture. Pooled execution for both content filtering transforms
- **Security:** Secret fingerprinting via HMAC with fail-closed behavior. Azure Key Vault integration for fingerprint key retrieval. Multiple Azure authentication options (managed identity, SAS token)
- **CLI:** Typer-based CLI with `run --execute`, `resume --execute`, `validate`, `plugins list`. Pretty progress output for pipeline execution. Automatic `.env` loading. Dynamic plugin discovery replacing static registry
- **Telemetry:** Token outcomes table and recording (AUD-001) with all outcome types (COMPLETED, ROUTED, FORKED, CONSUMED_IN_BATCH, COALESCED, QUARANTINED, FAILED). Outcome included in `explain()` lineage results
- **Docker/CI:** Dockerfile with Python 3.12, `/app/runs` for default audit.db location. GitHub Actions CI pipeline. Pre-commit hooks. Mutation testing runner script. Line-length 140 formatting standard
- **Examples:** OpenRouter sentiment analysis (standard and pooled/batched variants), template lookups, Azure pipeline examples, multi-query assessment

### Fixed

- **Canonical:** NaN/Infinity rejection enforced at validation boundary
- **Configuration:** Fail-closed secret fingerprinting with recursive detection. Runtime secrets preserved across config reload. Azure template metadata correctly propagated
- **Engine:** Checkpoint creation moved to after sink writes (preventing data loss on crash). Aggregation buffers flushed at end-of-source. Silent data loss in aggregation flush extracted to helper. Trigger evaluator reset in `execute_flush()`. RowProcessor wired to `execute_flush()` for batch audit. Transform `on_error` sink validation at startup and resume. Coalesce step positions computed correctly
- **LLM:** `max_tokens=null` no longer sent to OpenAI SDK. Call payloads auto-persisted when payload store configured. Batch checkpoint and replay mode issues resolved. `AuditedClientBase._next_call_index` made thread-safe. Semaphore deadlock prevented in pooled executor. Double-sleeping and 503/529 retry inconsistency fixed. Cached clients evicted after row completion. Call index collision resolved in pooled execution. Azure Content Safety threshold comparison corrected
- **Plugins:** Import errors crash instead of silent degradation. Duplicate plugin names detected and rejected. Missing audit fields added to LLM transform outputs. Keyword filter skips missing fields instead of crashing. OpenRouter malformed API response handling
- **Sources:** JSONDecodeError handled gracefully in JSON array files. JSONL parse errors quarantine rows instead of crashing
- **Landscape:** SQLite PRAGMAs applied in all `LandscapeDB` constructors. HTTP response body fully recorded in audit trail. Sensitive response headers filtered from audit trail
- **Audit:** Five audit integrity gaps closed (AUD-001 through AUD-005). Gate reason/mode preserved in continue routing events. Explicit `retryable` key in all error dicts for audit consistency. Batch aggregation outcome recordings completed
- **Routing:** Empty destinations rejected in `fork_to_paths`
- **CI:** Pre-commit hooks use direct venv paths. Correct root path for `no-bug-hiding` check. `[all]` installed in Lint job for complete mypy coverage
- **Docker:** Python 3.12 to match `requires-python`. Secrets handled via env block in bash scripts. Default audit.db path aligned with documented `/app/state` mount. Container build dependency issues resolved
- **Contracts:** Eight bugs resolved in contract code review

### Changed

- **Python Version:** Dropped Python 3.11 support, requiring 3.12+. PEP 695 type parameter syntax adopted for generics
- **Plugin System:** `PluginProtocol` updated to v1.5 with multi-row output and new attributes. Legacy hookimpl files deleted in favor of dynamic discovery. `PluginManager` used for all plugin instantiation
- **Aggregation:** `AggregationProtocol` and `BaseAggregation` removed in structural cleanup. `flush_buffer()` renamed to `_get_buffered_data()` for internal-only use
- **Enums:** `RowOutcome` changed to `(str, Enum)` for AUD-001 storage compatibility
- **Code Quality:** Full ruff lint and format pass. Line-length set to 140. Bug-hiding patterns removed from CI allowlist
- **Pooling:** Pooling infrastructure extracted to shared location for reuse across LLM transforms

### Tests

- Comprehensive test regime (Phases 1–3): unit, integration, and mutation testing
- Golden hash stability tests for canonical JSON
- Contract tests for all plugins (keyword filter, content safety, prompt shield, multi-query)
- Integration tests for LLM transforms, deaggregation pipelines, fork/coalesce pipelines, resume cycle, and pooled execution
- Property-based tests for reorder buffer
- Token outcome recording tests (AUD-001)
- Aggregation batch audit trail tests
- Secret fingerprinting tests (including Key Vault integration)

---

## RC-1 Hardening (Jan 22 – Feb 2, 2026) — Bug Burndown + Azure LLM Ecosystem

765 commits focused on systematic bug burndown, telemetry implementation, test infrastructure hardening, contract boundary enforcement, and rate limiting. Multiple triage and hunt sessions drove closure of 100+ bugs across all subsystems.

### Highlights

- **Telemetry subsystem** — built from scratch: event protocol, `TelemetryManager` with async export loop, bounded buffer with DROP/BLOCK backpressure modes, console/OTLP/Azure Monitor/Datadog exporters, and full orchestrator integration
- **ChaosLLM test infrastructure** — fake OpenAI/Azure server with error injector (burst state machine), latency simulator, response generator, metrics recorder (SQLite + time-series), MCP server for analysis, and pytest fixture
- **Systematic bug burndown** — 100+ bugs triaged and closed across 8+ sessions covering core engine, audit trail, LLM plugins, telemetry, contracts, sources/sinks, checkpoint, DAG, and configuration
- **Contract boundary hardening** — `AuditIntegrityError` and `OrchestrationInvariantError` error hierarchy, `RoutingReason` discriminated union, `TransformErrorReason` TypedDict, type soup cleanup, and tier model enforcement renamed and expanded
- **Rate limiting and pooling** — rate limiting wired through audited clients with two-layer architecture (per-service + per-endpoint), pool metadata integrated into audit trail

### Added

- **Telemetry:** Complete telemetry subsystem with event protocol, `TelemetryManager` for event coordination, `BoundedBuffer` for thread-safe event batching, async export loop with failure isolation. Backpressure modes: DROP (evicts oldest when full) and BLOCK (back-pressures producers). Exporters: `ConsoleExporter` for debugging, OTLP for OpenTelemetry collectors, Azure Monitor, and Datadog (ddtrace 4.x API). Queue metrics in `health_metrics`. Orchestrator integration emitting `RowCreated`, `TransformCompleted`, `TokenCompleted`, `RunCompleted` events. PARTIAL status emission on export failure
- **ChaosLLM:** Error injector with burst state machine and weighted error selection. Latency simulator for delay injection. Response generator with multiple modes (lorem, echo, structured JSON). Metrics recorder with SQLite storage and time-series aggregation. MCP server for Claude-optimized metrics analysis. pytest fixture and CLI for server lifecycle. Slow response handling and metrics export
- **Landscape:** Operations table for source/sink audit trail with XOR invariant (`state_id` vs `operation_id`). Transform success reasons recorded
- **Contracts:** `AuditIntegrityError` and `OrchestrationInvariantError` for domain-specific error hierarchy. `RoutingReason` 2-variant discriminated union. `TransformErrorReason` TypedDict with `Literal` reason discriminator. `ExceptionResult` moved to `contracts/results.py`. Telemetry events moved to `contracts/events.py`. `TokenOutcome` dataclass. Type soup cleanup with explicit field semantics
- **Engine:** Runtime enforcement for `expected_output_count` in processor. Identity-based audit assertion helpers for tests
- **Configuration:** Telemetry configuration contracts and queue_size internal default. `TelemetrySettings` registered in alignment registry
- **CLI:** Pretty error formatting for `validate` command. Backend validation and try/finally cleanup on resume path
- **CI/CD:** Contract enforcement CI stage. Stress test infrastructure. Tier model enforcement (renamed from `no_bug_hiding`). Python 3.13 CI upgrade. Scan group files for high-risk modules
- **Rate Limiting:** Two-layer rate control architecture wired through audited clients. Per-service rate limiting with `NoOpLimiter` timeout parameter for API parity. Underscore service name compatibility
- **Documentation:** Comprehensive telemetry documentation (design, trust model guide, backpressure modes). ChaosLLM user documentation. Rate limiting configuration guide. Two-layer rate control architecture guide. User-facing documentation suite updates. Module documentation generated for bug hunting

### Fixed

- **Bug Burndown Sessions:** Systematic triage and closure across all subsystems. Multiple numbered sessions (P1, P2, P3 priority). Bug reports generated via static analysis, mutation testing gaps, and manual review
- **Engine:** Source row payloads stored for audit compliance. Deferred `operation_id` clear and sinks moved outside `track_operation`. `operation_id` restored at end of source iteration. Three P2/P3 regressions resolved from code review. `max_workers` wired through `RowProcessor`. QUARANTINED outcome recorded after sink durability check. Coalesce double terminal outcome recording prevented. Asserts replaced with proper exception raising. Leaf boundary handling corrected
- **LLM/Plugins:** OpenRouter batch critical bugs fixed. Empty batch crash prevented. Template errors audited. Provider parameter wiring completed for batch transforms. OpenRouter multi-query guaranteed fields bug fixed. 503/529 treated as capacity errors for pooled retry. Silent fallback eliminated. Dead code and unused config fields removed from openrouter_batch. Explicit `plugin_version` added to sources and transforms
- **Telemetry:** Shutdown hang and credential leak resolved. `ReadableSpan` properties added for SDK compatibility. Azure Monitor `ProxyTracerProvider` bug fixed. Payload observability added. Datadog exporter updated for ddtrace 4.x API. Duplicate `RunCompleted` emission on export failure removed. `TransformCompleted` emitted for aggregation flushes. `TransformCompleted`/`TokenCompleted` ordering corrected for transform-mode aggregation. Unused `GateOutcome` import removed
- **Contracts:** Node metadata and leaf boundary P2 bugs fixed. Payload store typed properly (replacing `Any`). Type ignores eliminated with `TypeGuard`. `from __future__ import annotations` added for `TYPE_CHECKING` blocks. Contract fields validated as subsets of declared fields. `model_config["strict"]` accessed directly. Two P2 and two P3 contract bugs closed systematically
- **Sources:** NaN/Infinity rejected at JSONSource parse boundary. `data_key` structural errors quarantined in JSONSource
- **DAG:** Gates inherit computed schema guarantees from upstream. Structural schema comparison for coalesce validation. Mixed dynamic/explicit schemas rejected on pass-through nodes
- **Configuration:** 23 `cast(SchemaConfig)` calls eliminated via `DataPluginConfig` restructure. `output_mode='single'` removed, replaced with `expected_output_count`. Environment variable overrides restored for sink settings. Malformed retry policy types rejected with clear errors
- **Checkpoint:** Two P2 bugs closed (topology hash and state restoration)
- **Security:** `content_hash` validated in payload store to prevent path traversal. Key Vault fingerprint key cached. Key Vault operational errors propagated instead of masked as not-found. Composable secret loaders added
- **Coalesce:** `require_all` timeout handled in `check_timeouts()`. Fork/join handling improved in coalesce executor
- **Expression Parser:** Evaluation errors wrapped as `ExpressionEvaluationError`
- **Triggers:** "First to fire wins" semantics implemented. Boolean conditions enforced. Fire times preserved on resume
- **MCP:** LLM usage status check corrected. Fork/join counts scoped to run
- **Retention:** Failed runs included in purge. I/O errors handled gracefully
- **Spans:** Node ID disambiguation added. Aggregation span used for flushes
- **Retry:** `before_sleep` hook used for correct `on_retry` semantics
- **Logging:** Azure SDK and third-party HTTP connection spam silenced. Stdlib loggers routed through structlog for consistent JSON output
- **Rate Limiting:** Rate limiting wired through audited clients. Pool metadata and ordering metadata preserved. Missing pool stats added
- **Routing:** Broken factory methods repaired after `_freeze_dict` rename
- **Sinks:** Strict schemas enforced for CSV and Database sinks
- **CI:** Tier model allowlist drift and duplicates resolved. Allowlist updated for Python 3.12 parsing. Elspeth installed in contract-enforcement job
- **Docker:** Locked dependencies synced. Installation commands updated
- **Tests:** TelemetryManager thread cleanup via autouse fixture. Stale assertion fixed (plugin name vs node\_id). Token info initialization corrected. Batch test `time.sleep` replaced with Event-based coordination. CLI validation error test updated for rich output format. All tests updated for `output_mode='single'` removal and `TransformErrorReason` TypedDict

### Changed

- **Tier Model Enforcement:** `no_bug_hiding` renamed to `enforce_tier_model`. Tier 1 trust model enforced in TUI, engine, and property tests. Allowlist entries updated for trust boundary patterns
- **Contracts:** `ExceptionResult` and telemetry events relocated to `contracts/` layer. `RoutingReason` TypedDict replaced with discriminated union. `RunCompleted` naming collision resolved between telemetry and landscape events. Payload store and `IntegrityError` imported from contracts
- **Engine:** `output_mode='single'` removed; `expected_output_count` added as explicit count-based enforcement. Type narrowing used for `SchemaConfig` access
- **Tests:** Duplicate P1 test code deduplicated (Phase 2). Test suite consolidated and redundant property tests removed. Parametrized common tests added for `Runtime*Config`. Fixtures updated for valid `RoutingReason` variants and `TransformErrorReason`
- **Telemetry:** Lazy-imported exporters typed properly. All exporters registered in `__init__.py`
- **CLI:** Type ignores eliminated with explicit protocol typing. Legacy `token_id`/`row_id` accessors removed from `RowResult`
- **Plugins:** Unused `PluginSpec` class and schema hashing removed. Type narrowing used for `SchemaConfig` access
- **Documentation:** Plans reorganized (completed plans moved to `completed/` directory). Bug reports triaged and closed systematically. Architecture analysis refreshed

### Tests

- **Telemetry:** Property-based tests with Hypothesis. EventBus re-entrance safety. Concurrent close during export. Re-entrance deadlock prevention. Lock contention for `_events_dropped`. DROP and BLOCK mode backpressure. Graceful shutdown. Thread liveness checks. Task-done exception safety. OTLP and Azure Monitor integration tests. Datadog integration tests. Contract tests. Client telemetry tests. RowProcessor telemetry tests
- **ChaosLLM:** Integration tests for ChaosLLM server. Weighted selection, slow response, and metrics export tests. Error injector tests with deterministic Random instance
- **Stress:** LLM stress testing framework with ChaosLLM HTTP server (row counts tuned for CI 15-minute timeout)
- **Landscape:** Token outcome constraint tests. Operations table tests
- **Engine:** Processor iteration guard tests. Identity-based regression tests for batch aggregation
- **Checkpoint:** Topology hash mismatch tests
- **Security:** Fingerprint key vault failure tests
- **Verifier:** Duplicate element handling, nested list ordering, dict key ordering, empty list edge case, realistic LLM tool call ordering (with `ignore_order` support)
- **Integration:** OpenRouter tests updated for retry exception semantics. Batch test coordination improved (Event-based instead of `time.sleep`)
