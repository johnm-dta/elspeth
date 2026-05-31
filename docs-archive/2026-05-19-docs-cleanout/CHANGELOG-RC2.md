# RC-2 Changelog

Historical changelog reconstructed from branch history. Covers the six RC2.x sub-releases on main (Feb 2–12, 2026).

For Pre-RC1 and RC1 hardening, see [CHANGELOG-RC1.md](CHANGELOG-RC1.md).
For RC3+, see [CHANGELOG.md](CHANGELOG.md).

---

## RC2 Post-Implementation Cleanup (Feb 2, 2026)

Light cleanup after the RC2 squash-and-restart. PR #14 and #15 addressed stale allowlists, missing sink features, and a Tier 1 audit integrity fix.

### Added

- **Sinks** — Display header support for CSV, JSON, and Azure Blob sinks (field normalization with original-name display)
- **Telemetry** — `FieldResolutionApplied` event emitted at run start
- **Azure Blob Source** — Field normalization support matching CSV source behavior

### Fixed

- **Landscape** — Crash on corrupt field resolution instead of returning `None` (Tier 1 integrity)
- **Executor** — Pass `context_after` to failed node states
- **Sinks** — Display header timing, validation, and resume with `restore_source_headers`

### Changed

- Removed 84 stale tier model allowlist entries
- Excluded performance tests from the default test suite
- Removed outdated audit and telemetry analysis documents

---

## RC2.1 (Feb 2–3, 2026) — Key Vault Secrets + Documentation

PR #17. 142 commits. Major infrastructure release introducing Azure Key Vault secrets, schema contracts, Langfuse tracing for LLM plugins, and the orchestrator decomposition.

### Highlights

- **Azure Key Vault secrets backend** — Secrets loaded from Key Vault with HMAC fingerprints recorded in a new `secret_resolutions` audit table. Integrated into `run`, `resume`, and `validate` CLI commands.
- **Schema contracts** — First-row-inferred field contracts with O(1) name resolution, propagated through transforms, recorded in the audit trail, and verified on checkpoint resume.
- **Orchestrator decomposition** — Extracted monolithic `orchestrator.py` into a focused package with `validation.py`, `export.py`, `aggregation.py`, and `types.py` modules.

### Added

- **Secrets** — `SecretsConfig` Pydantic model, `config_secrets.py` Key Vault loader, `secret_resolutions` Landscape table, deferred recording of secret resolutions
- **Schema contracts** — `FieldContract`, `SchemaContract`, `ContractBuilder` for first-row inference, contract-aware `PipelineRow` with dual-name template access, transform contract propagation, sink header mode resolution (`headers: contract`, `headers: original`), contract audit columns in Landscape tables, MCP contract analysis tools, checkpoint integrity verification
- **Tracing** — Tier 2 Langfuse tracing for Azure OpenAI, OpenRouter, and batch LLM plugins with span creation and lifecycle management
- **CI/CD** — Per-file whitelisting for tier model enforcement
- **Examples** — Schema contracts demo pipeline, Azure Key Vault secrets example pipeline
- **Documentation** — Secrets configuration reference, Key Vault runbook, Tier 2 plugin tracing guide, ChaosLLM design document, WebScrape design document

### Fixed

- **Engine** — Removed bug-hiding `.get()` patterns in `AggregationExecutor`, wired aggregation state into checkpoint creation
- **Contracts** — `PipelineRow.__contains__` respects contract boundaries, schema contract integrity hash includes `locked`/`source` fields, `any` type validation, non-primitive type handling, deferred schema contract recording to first valid row
- **Sinks** — `headers: original` mode resolved from `ctx.contract`, infer-and-lock pattern for all schema modes
- **Landscape** — Eliminated N+1 query pattern in batch export
- **Engine** — JSON schema reconstruction handles nullable and format types
- **CLI** — Auto-create output directories, improved error handling in purge command settings loading
- **Contracts** — Restored leaf boundary with lazy numpy/pandas imports
- **Azure** — Missing tracing flush in `azure_multi_query.close()`
- **Contracts** — Consolidated duplicate `ContractAwareRow` into `PipelineRow`

### Changed

- **Orchestrator** — Extracted into package with `types.py`, `validation.py`, `export.py`, `aggregation.py` modules
- Migrated schema mode naming and standardized example files
- Moved pytest options to conftest, updated `SecretsConfig` API
- Simplified fingerprint key to use environment variable only
- Consolidated `ContractAwareRow` into `PipelineRow`

---

## RC2.2 (Feb 3, 2026) — Langfuse Upgrade

PR #18. 3 commits. Migrated Langfuse tracing from SDK v2 to v3 (OpenTelemetry-based). Failed LLM calls now record Langfuse traces. Removed slow `test_fifo_ordering_preserved` stress test.

---

## RC2.3 (Feb 3–7, 2026) — PipelineRow

PR #19. 106 commits. Introduced the typed `PipelineRow` wrapper, replacing raw `dict` usage throughout the entire pipeline — from source through transforms, gates, aggregations, and sinks. Also delivered the WebScrape transform, SSRF prevention, and DIVERT routing.

### Highlights

- **PipelineRow migration** — Replaced raw `dict[str, Any]` with typed `PipelineRow` across all plugin signatures, executors, engine, and test suite. Checkpoint version bumped to 2.0.
- **WebScrape transform** — New HTTP scraping transform with SSRF prevention (URL scheme validation, IP blocklist, DNS timeout), HTML extraction (markdown/text/raw), and content fingerprinting.
- **DIVERT routing** — Added `RoutingMode.DIVERT` edges for quarantine/error sink paths in the DAG, with coalesce branch-loss notification and MCP lineage annotations.

### Added

- **WebScrape** — `WebScrapeTransform` with error hierarchy (retryable classification), HTML extraction via html2text/BeautifulSoup, content fingerprinting with normalization, SSRF prevention, and audit trail
- **Security** — URL scheme validation, IP validation with DNS timeout and blocklist for SSRF prevention
- **Routing** — `RoutingMode.DIVERT` edges in DAG, DIVERT routing events for transform error paths, quarantine routing events with `SourceQuarantineReason`, DAG validation warning for DIVERT + `require_all` coalesce, coalesce branch-loss notification for eager fork/join resolution, DIVERT annotation in MCP `explain_token` lineage
- **Engine** — Per-row source `node_states` for complete audit lineage, `shutdown_requested` in `TransformErrorCategory`
- **Protocols** — `BatchTransformProtocol` for type-safe batch transforms
- **HTTP client** — GET method on `AuditedHTTPClient` with audit trail
- **Build** — `[web]` optional dependency group (html2text, beautifulsoup4)

### Fixed

- **P0 security** — DNS rebinding TOCTOU in SSRF prevention, atomic write for JSON array sink (prevents data loss on crash), content safety and prompt shield fail-closed validation
- **P1 security** — Secret scrubbing, ReDoS guard, Jinja2 sandboxing, template path traversal blocked
- **P1 correctness** — Tier 1 crash semantics enforced, key collision validator, `max_tokens` handling, `LLMClientError` propagation, `extra="forbid"` on Settings models, atomic payload store, `batch_members` primary key
- **Landscape** — Routing reason payloads stored in payload store, fail-fast on partially missing schema
- **DAG** — Disconnected graph detection
- **Coalesce** — Union collision tracking, nested coalesce contract uses FIXED mode with branch-key fields
- **Redirects** — Resolve relative redirects against hostname URL, not IP-based connection URL; record redirect hops as individual `CallType.HTTP_REDIRECT` audit entries
- **Batch** — Batch lineage query ordering by `(step_index, attempt)` not `state_id`
- **Types** — Resolved all mypy errors in production code and test suite, migrated `(str, Enum)` to `StrEnum`
- **Tests** — Converted manual graph construction tests to production path, consolidated `_make_pipeline_row` into shared helper, repaired property test strategies

### Changed

- **PipelineRow** — Updated `TokenInfo.row_data` type from `dict` to `PipelineRow`, updated `BaseTransform`/`BaseGate` signatures, all `dict(row)` calls migrated to `row.to_dict()`
- **LandscapeRecorder** — Decomposed god class into 8 focused mixins
- **MCP** — Typed server returns with TypedDicts, removing 24 whitelist entries; validated tool arguments at Tier 3 boundary
- **CLI** — Extracted event formatters to separate module to reduce duplication
- **Engine** — Deduplicated processor work queue loops, used `Counter` for `routed_destinations` tracking
- **Contracts** — Consolidated cross-boundary types into `contracts/`
- **CI** — Pinned `ruff==0.15.0`, aligned pre-commit and CI lint scopes, pre-commit upgraded to v4
- Shared `httpx.Client` for connection pooling
- Deleted dead `BoundedBuffer` in telemetry
- Resolved N+1 query patterns in Landscape layer
- Chunked row metadata queries to avoid SQLite bind limit
- Extracted shared orchestrator run/resume finalization

---

## RC2.4 (Feb 7–9, 2026) — Bug Sprint

PR #20. 107 commits. Intensive bug burndown sprint: 178-bug triage, test suite v2 migration (8,138 tests), and security hardening across the entire codebase.

### Highlights

- **178-bug triage** — Systematic triage of 140 validated bugs from static analysis, with 28 resolved in this sprint across P0–P2 priorities.
- **Test suite v2** — Complete 7-phase migration: scaffolding, factories, unit tests (contracts/core/engine/plugins), property tests, integration tests, E2E tests, performance/stress tests, and v1 cutover. 8,138 tests collected, 8,037 passed.
- **Security hardening** — Ephemeral `httpx.Client` for SSRF-safe requests (prevents TLS/SNI leakage), secrets injection prevention, data leakage fixes.

### Added

- **Test suite v2** — Factory architecture with centralized `make_context()`, `make_recorder_with_run()`, `register_test_node()` factories; property tests for SSRF, ChaosLLM, DAG topologies, token ops, triggers, routing, schema contracts, reorder buffer, orchestrator lifecycle, landscape recording, LLM templates, Azure safety; integration tests for trust boundaries and contracts; E2E pipeline tests; performance benchmarks and stress tests
- **ChaosLLM** — Examples and test data generator
- **Engine** — Quarantine sink DAG exclusion to prevent unreachable node errors

### Fixed

- **Correctness** — OpenRouter, `azure_batch`, and `batch_replicate` bugs; `azure_batch` `error_file_id` check; `batch_replicate` quarantine handling; `batch_stats` and `json_explode` data integrity; `on_error` encapsulation; `TransformResult` validation
- **Security** — Ephemeral `httpx.Client` for SSRF-safe requests; secrets, injection, and data leakage hardening; `keyword_filter` positional metadata
- **Engine** — Batch transform type mismatch and audit integrity; pooled executor bugs; AIMD-aware dispatch gate double-penalization reverted; coalesce union collision tracking; journal recovery; SQL blocklist word-boundary matching
- **WebScrape** — Tier 3 boundary protection for `extract_content` failures
- **Landscape** — Fail-fast on partially missing schema
- **Tests** — Defensive programming violations removed; checkpoint version references updated; `TransformResult` types tightened to `PipelineRow`; 38 tests updated for `extra=forbid` and schema contract enforcement
- **DAG** — Dead DAG warning; multi-row contract validation; `batch_replicate` audit integrity

### Changed

- **Test suite v2 cutover** — Deleted v1 suite (7,487 tests, 507 files, 222K lines), renamed `tests_v2/` to `tests/`, rewrote 204 imports across 123 files, fixed 47 pre-existing lint issues
- Removed defensive programming violations and dead code
- Removed obsolete export state JSON files

---

## RC2.5 (Feb 9–12, 2026) — SQLite Migration

PR #21. 134 commits. Declarative DAG wiring, SQLCipher encryption-at-rest, ChaosWeb fake server, executor decomposition, and comprehensive hardening across the routing and plugin layers.

### Highlights

- **Declarative DAG wiring** — Three-phase routing overhaul: explicit sink routing (Phase 1), processor node-ID refactoring (Phase 2), and declarative `on_success`/`on_error` connection matching with `WiredTransform` (Phase 3). Every edge explicitly named and validated at construction time.
- **SQLCipher encryption-at-rest** — Landscape audit database encryption with passphrase management across CLI, MCP, and resume paths. Empty passphrases rejected.
- **ChaosWeb** — Fake web server with configurable error injection, content generation, metrics recording, pytest fixtures, and 265 tests.

### Added

- **DAG wiring** — `WiredTransform` connection matching, gate route fan-out to multiple processing connections, `DAGTraversalContext` for processor traversal, `StepResolver` dependency injection, route destination invariants and route-edge traversal enforcement, character-class validation on route labels/fork branches/sink names
- **SQLCipher** — Encryption-at-rest for Landscape database, backend validation, key escaping, CLI passphrase prompts, URI option preservation
- **ChaosWeb** — Config model, error injector, content generator, server, metrics recorder, CLI, pytest fixtures, test utilities, scraping pipeline demo
- **Engine** — `DAGNavigator` extracted from `RowProcessor`, aggregation flush path deduplication
- **Examples** — `deep_routing` pipeline with 5 gates and 3 transforms
- **Tests** — 71 trust-boundary and contract integration tests, 13 `openrouter_batch` integration tests, orchestrator regression coverage, sink name assertions, call-index concurrency stress tests, error routing decision tree documentation

### Fixed

- **P0 routing** — `NodeInfo` immutability, `on_error` validation, sink name invariants, fork-to-sink routing bypass
- **P1 routing** — Character validation, exhaustiveness checks, gate-to-gate route jump resolution crash, Tier 1 trust violations in processor coalesce handling, symmetric coalesce resolution, stale refs and dead code removal
- **Routing** — Continue edge materialization for gate routes targeting continue, connection namespace enforcement, DAG topology and coalesce step ordering regressions, routing resolution and sink preloading invariants
- **SQLCipher** — Backend validation regressions, key escaping, CLI passphrase guard, empty passphrase rejection, URI options preserved
- **Plugins** — `truncate` strict mode rejects non-string fields consistently, `database_sink` stale replace and portability, `json_explode` empty array handling, `blob_sink` re-raise, `web_scrape` HTTP config validated with Pydantic sub-model
- **Security** — NaN/Infinity rejected at JSON parse and schema validation boundaries; quarantine NaN/Infinity crash; content safety fail-closed fixes; rate limiter, prompt shield, LLM client, and type inference fixes
- **Engine** — Condition trigger latching, pre-row flush, hash verification; DAG schema propagation; `on_success` and `on_error` required on `TransformSettings`
- **MCP** — Narrowed exception handler, `query()` read-only SQL guard strengthened, sink map fix, SQLCipher passphrase forwarding in `elspeth-mcp` entrypoint
- **Telemetry** — Fail-closed enforcement, unknown-event forward compatibility filtering, deterministic span IDs replaced with random
- **Contracts** — Non-enum `RoutingAction` constructor rejected, `TriggerType` enum validation enforced
- **Landscape** — Deferred payload storage in `complete_operation` until status check passes; 6 Tier 1 audit timing, TOCTOU race, shallow copy, contract inference, key exposure, and telemetry race bugs
- **CLI** — Smoke-test config regression, sensitive header detection gap, `--format json` added to `resume`; `batch_stats` preserves int precision; `_orchestrator_context` extracted from run/resume paths

### Changed

- **DAG** — Extracted `ExecutionGraph` builder into `dag/` package; lifted routing fields from plugin config to settings level
- **Processor** — Refactored work queue items to node-ID state, unified node-ID traversal in main loop, work-item continuations and coalesce trigger checks moved to node traversal
- **Executors** — Split `executors.py` into one-file-per-executor package
- **MCP** — Split `server.py` into domain modules via facade pattern
- **LLM** — Extracted `BaseMultiQueryTransform` to deduplicate Azure/OpenRouter plugins; migrated `OpenRouterBatchLLMTransform` to `AuditedHTTPClient`
- **Plugins** — Replaced `on_error`/`on_success` properties with plain attributes; replaced unvalidated dict with Pydantic sub-model for `web_scrape` HTTP config
- **ChaosEngine** — Extracted shared core via composition, normalized timeseries buckets to UTC, isolated metrics failures
- **Engine** — Eliminated defensive `hasattr`/`getattr`, extended expression parser
- **CI/CD** — Split tier model allowlist into per-module YAML files; installed `libsqlcipher-dev` in all CI jobs

### Removed

- Dead plugin protocols (`CoalesceProtocol`, `GateProtocol`, `PluginProtocol`, `CoalescePolicy`)
- Gate plugin dead code from DAG builder and docs
- 4 redundant per-type uniqueness validators
- Backward-compatibility language from `dag` package docstring
- Generated bug docs (replaced with manually-written reports)
- Batching infrastructure examples
