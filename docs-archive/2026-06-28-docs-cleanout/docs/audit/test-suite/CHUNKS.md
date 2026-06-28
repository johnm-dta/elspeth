# Test Suite Audit — Chunk Plan & Progress

**Goal:** Identify pointless, duplicate, low-effort, defective, and underperforming tests across the 741-file ELSPETH test suite, alongside major coverage gaps.

**Method:** Each chunk is reviewed by a parallel wave of 5 specialist agents:
1. `ordis-quality-engineering:test-suite-reviewer` — anti-patterns
2. `axiom-sdlc-engineering:quality-assurance-analyst` — VER/VAL theatre
3. `axiom-python-engineering:python-code-reviewer` — Python-specific smells
4. `pr-review-toolkit:pr-test-analyzer` — scenario coverage
5. `ordis-quality-engineering:coverage-gap-analyst` — SUT coverage gaps

**Cross-reference task:** From wave 3 (I-1) onward, the gap-analyst and pr-test-analyzer prompts include an explicit cross-reference against prior-wave findings, to (a) confirm gaps persist at this layer or (b) downgrade prior issues that turn out to be covered elsewhere.

**Sizing rule:** Each chunk is 15–35 files. Subsets are used only when the parent cluster crosses ~40 files. Never split a leaf directory across chunks.

---

## Progress

| # | Chunk | Files | Status | Synthesis | Filigree issues |
|---|---|---|---|---|---|
| 1 | U-CONTRACTS-1 (Plugin & registry contracts) | 25 | ✅ Done | [findings](u-contracts-1-findings.md), [raw](u-contracts-1-raw-reports.md) | `f92ba560ad`, `8d5558dc25`, `7b7fe68836` |
| 2 | U-CORE-1 (Landscape audit-DB recording) | 32 | ✅ Done | [findings](u-core-1-findings.md), [raw](u-core-1-raw-reports.md) | `297dafdf47`, `499100db05`, `82c7c028a8`, `f6f50e9394` |
| 3 | I-1 (Audit integration tests) | 32 | ✅ Done | [findings](i-1-findings.md), [raw](i-1-raw-reports.md) | `ae9f541775` closed via taxonomy gate, `4a013f9833`; `f6f50e9394` scope-revised |
| 4 | U-ENGINE-1 partial continuation | partial | 🔎 Partial | [findings](u-engine-1-findings.md), [raw](u-engine-1-raw-reports.md) | `8bf288792a`, `bd49237412`, `e4281a36d8`, `e0afd080cc`, `462f50680f`, `f295b77e76`, `9a1262dbc7`, `ff85897f8f`, `314eb3552e`, `2744d69903`, `0c1c7d5cec`, `eb12769648`, `975f45dfcb` |
| 5 | U-ENGINE-2 partial continuation | partial | 🔎 Partial | [findings](u-engine-2-findings.md), [raw](u-engine-2-raw-reports.md) | `958f307f29`, `68cd1876d0`, `aa7781f802`, `c5add729fa`, `786291485f` |
| 6 | _next full wave_ | — | ⏳ Pending | — | — |

---

## Tier I — Unit tests (530 files → 16 chunks)

### Set U-CONTRACTS — L0 contracts (110 files → 4 subsets)

| ID | Files | Scope |
|---|---|---|
| U-CONTRACTS-1 ✅ | ~25 | `unit/contracts/{transform,source,sink}_contracts/`, `test_plugin_*`, `test_registry_*`, `test_tier_*`, `test_plugin_semantics*` |
| U-CONTRACTS-2 | ~24 | `test_schema*`, `test_contract*`, `test_declaration_contracts`, `test_compose_propagation`, `test_propagation_walkers_agree`, `test_*_row_contract`, `test_field_contract`, `test_telemetry_contracts`, `test_engine_contracts`, `test_runtime_val_manifest` |
| U-CONTRACTS-3 | ~22 | `test_audit*`, `test_hashing`, `test_freeze*`, `test_secrets`, `test_secret_scrub`, `test_composer_audit*`, `test_token_*`, `test_identity`, `test_probes`, `test_call_data`, `test_record_call_guards` |
| U-CONTRACTS-4 | ~34 | `test_coalesce*`, `test_checkpoint*`, `test_batch_checkpoint`, `test_routing`, `test_results`, `test_run_result`, `test_diversion`, `test_diverted_outcome`, `test_gate_result_contract`, `test_error*`, `test_new_errors`, `test_contract_violation*`, `test_control_flow_exceptions`, `test_validation_error_noncanonical`, `test_post_init_validations`, `unit/contracts/config/`, `test_cli`, `test_config`, `test_data`, `test_enums`, `test_events`, `test_header_modes`, `test_type_normalization` |

### Set U-CORE — L1 core (90 files → 4 subsets)

| ID | Files | Scope |
|---|---|---|
| U-CORE-1 ✅ | 32 | `unit/core/landscape/test_*.py` (recording, repositories, DB ops, schema, models, serialization, querying, factory/journal/lineage/exporter) |
| U-CORE-2 | ~13 | (Reabsorbed into U-CORE-1 in execution; see U-CORE-1 findings) |
| U-CORE-3a | ~33 | Core root: `test_canonical*`, `test_config*`, `test_dag*`, `test_dag_registry`, `test_payload_store`, `test_secrets_config`, `test_resolve_secret_refs`, `test_template_extraction_dual`, `test_templates`, `test_token_outcomes`, `test_identifiers`, `test_logging`, `test_operations`, `test_events`, `test_edge_validation`, `test_explicit_sink_routing_safeguards`, `test_sink_settings_on_write_failure`, `test_dependency_config`, `test_connection_name_validation` |
| U-CORE-3b | ~24 | `unit/core/dag/`, `unit/core/checkpoint/`, `unit/core/security/`, `unit/core/retention/`, `unit/core/rate_limit/` |

### Set U-ENGINE — L2 engine (60 files → 2 subsets)

| ID | Files | Scope |
|---|---|---|
| U-ENGINE-1 🔎 partial | ~25 | Declaration/dispatch/processor: `test_declaration_*`, `test_declared_*_fields_contract`, `test_pass_through_*`, `test_processor*`, `test_dependency_resolver`, `test_executors`, `test_flush_dispatcher_routing`, `test_sink_*`, `test_failsink_validation`, `test_can_drop_rows_contract`, `test_boundary_dispatch_inputs`, `test_record_flush_violation_failure`, `test_state_guard_audit_evidence_discriminator` |
| U-ENGINE-2 🔎 partial | ~35 | Engine root part 2 + orchestrator: `test_batch_*`, `test_coalesce_*`, `test_cross_check_flush_output`, `test_dag_navigator`, `test_retry*`, `test_clock`, `test_commencement`, `test_expression_parser`, `test_plugin_*`, `test_routing_enums`, `test_row_outcome`, `test_schema_config_mode_contract`, `test_source_guaranteed_fields_contract`, `test_spans`, `test_token_manager_pipeline_row`, `test_tokens`, `test_transform_success_reason`, `test_triggers`, `test_audit_wrapper_scope`, `test_bootstrap_preflight`, `test_adr019_phase2_producer_pairs`, `test_orchestrator_registry_bootstrap`, `unit/engine/orchestrator/` |

### Set U-PLUGINS — L3 plugins (166 files → 5 subsets)

| ID | Files | Scope |
|---|---|---|
| U-PLUGINS-1a | ~26 | Plugin framework root: `unit/plugins/test_*.py` (base, base_signatures, builtin_plugin_metadata, config_base, context, discovery, hookimpl_registration, manager, manager_singleton, node_id_protocol, post_init_validations, protocols, results, schema_factory, schemas, sink_header_config, utils, validation, validation_integration, validation_path_agreement, etc.) |
| U-PLUGINS-1b | ~24 | Clients/pooling/batching/config/infrastructure: `unit/plugins/{clients,pooling,batching,config,infrastructure}/` |
| U-PLUGINS-2 | ~28 | Sources + Sinks: `unit/plugins/sources/` (10) + `unit/plugins/sinks/` (18) |
| U-PLUGINS-3 | ~22 | Pipeline transforms (excluding web_scrape): batch_replicate/stats, field_*, json/line_explode, keyword_filter, passthrough, safety_utils, truncate, type_coerce, value_transform, backward/forward_invariant_probes |
| U-PLUGINS-4 | 5 | Web scrape (Tier-3 boundary): `test_web_scrape*` |
| U-PLUGINS-5a | ~17 | LLM providers/templates/registration/config: azure*, openrouter*, providers, templates, contract-aware-template, prompt_template_contract, plugin_registration, llm_config, config_schema, validation, transform |
| U-PLUGINS-5b | ~18 | LLM batch/multi-query/pooling/tracing: multi_query*, batch*, pool*, tracing*, langfuse_tracer, aimd_throttle, capacity_errors, reorder_buffer, audit_metadata_functions, success_reason, p1_bug_fixes, provider_lifecycle |

### Set U-WEB — Web/composer (75 files → 3 subsets)

| ID | Files | Scope |
|---|---|---|
| U-WEB-1 | 30 | Composer + execution: `unit/web/composer/` (17) + `unit/web/execution/` (13) |
| U-WEB-2 | 22 | Sessions + auth + middleware: `unit/web/sessions/` + `unit/web/auth/` + `unit/web/middleware/` |
| U-WEB-3 | 23 | Secrets + blobs + catalog + root |

### Set U-OPERATOR-SURFACES — CLI / TUI / MCP / Composer-MCP (41 files → 2 subsets)

| ID | Files | Scope |
|---|---|---|
| U-OPS-1 | 21 | CLI + composer_mcp: `unit/cli/` + `unit/composer_mcp/` |
| U-OPS-2 | 20 | MCP + TUI: `unit/mcp/` + `unit/tui/` |

### Set U-CROSSCUTTING — Telemetry / scripts / regression / fixtures (42 files → 2 subsets)

| ID | Files | Scope |
|---|---|---|
| U-CROSS-1 | 17 | Telemetry: `unit/telemetry/` |
| U-CROSS-2 | ~25 | Scripts + regression + root: `unit/scripts/` + `unit/regression/` + `unit/test_*.py` root |

---

## Tier II — Integration tests (100 files → 5 chunks)

| ID | Files | Scope |
|---|---|---|
| I-1 ✅ | 32 | Audit: `integration/audit/` (root 33 + recorder/) |
| I-2 | 32 | Pipeline + orchestrator: `integration/pipeline/` root (16) + `pipeline/orchestrator/` (16) |
| I-3 | 19 | Plugins: `integration/plugins/{sources,sinks,transforms,llm}/` |
| I-4 | 21 | Cross-cutting infra: `integration/{cli,config,contracts,core,checkpoint,rate_limit,telemetry,web}/` + `integration/_helpers.py` |
| I-5 | 6 | ADR-019 migration suite: `integration/_adr019_test_plugins.py` + `integration/test_adr_019_*.py` (5 files: counter-changes, cross-table invariants, discard-mode flip, sweep durability, resume parity) |

---

## Tier III — Property-based tests (70 files → 3 chunks)

Property tests need a different audit lens (Hypothesis strategy quality, shrinker traps, deadline/health-check usage, false-positive immunity).

| ID | Files | Scope |
|---|---|---|
| P-1 | ~52 | Core/canonical/engine/contracts: `property/core/` (25) + `property/canonical/` (4) + `property/engine/` (16) + `property/contracts/` (7) |
| P-2 | 18 | Plugins: `property/plugins/{llm,web_scrape,transforms,sources,sinks,batching}/` + plugins root |
| P-3 | ~18 | Audit/sinks/sources/integration/telemetry/root: `property/audit/` (6) + `property/sinks/` (4) + `property/sources/` (2) + `property/integration/` (2) + `property/telemetry/` (3) + root + conftest |

---

## Tier IV — Higher-order suites (40 files → 3 chunks)

| ID | Files | Scope |
|---|---|---|
| H-1 | 16 | E2E: `e2e/{audit,examples,external,pipelines,recovery}/` |
| H-2 | 18 | Performance / scalability / stress / memory: `performance/{benchmarks,scalability,stress,memory}/` |
| H-3 | ~12 | Invariants + helpers + fixtures + strategies: `invariants/` (6) + `tests/helpers/` + `tests/fixtures/` + `tests/strategies/` + top-level `tests/conftest.py` |

---

## Recommended dispatch order

1. **U-CONTRACTS-1 ✅** (calibration; tautology-rich)
2. **U-CORE-1 ✅** (audit-DB; high stakes)
3. **I-1 ✅** (cross-reference test introduced)
4. **U-ENGINE-1** ← _recommended next_: central engine, structurally different from prior chunks
5. **U-PLUGINS-4** (web_scrape) — closes `elspeth-7b7fe68836` SSRF gap if covered
6. **U-PLUGINS-5a** (LLM providers) — closes `elspeth-8d5558dc25` Azure threshold gap if covered
7. **I-5** (ADR-019 migration) — confirms `elspeth-f6f50e9394` downgrade and validates active-migration coverage
8. **I-2** (Pipeline integration) — extends I-1's structural finding to a different integration directory
9. Remaining chunks in any order

## Resume protocol for a future session

To continue this audit in a new session:

1. Read `docs/audit/test-suite/CHUNKS.md` (this file) for the chunk plan and progress.
2. Read `docs/audit/test-suite/README.md` for the cross-chunk patterns and methodology.
3. Read the most recent `docs/audit/test-suite/<chunk>-findings.md` for the immediate context.
4. Pick the next chunk per the recommended order, or per current priorities.
5. Dispatch 5 parallel agents using the prompts established in the prior waves (template: see any of the prior wave dispatches in transcript history; the structure is identical except for the file list and the cross-reference task).
6. Synthesise into `docs/audit/test-suite/<chunk>-findings.md` and `docs/audit/test-suite/<chunk>-raw-reports.md`.
7. Update this file's progress table.
8. File new filigree issues for production-code gaps; update existing issues if a cross-reference revises their scope.

## Cross-chunk patterns being tracked

These appear in multiple chunks; they should be addressed once with a CI rule or sweep PR rather than per-chunk. (Authoritative list lives in `docs/audit/test-suite/README.md`; copied here for convenience.)

- **`hasattr()` in tests violates CLAUDE.md** (U-CONTRACTS-1: ~15 sites; U-CORE-1: 5 sites; U-ENGINE-1 partial: at least one weak assertion). Filed `elspeth-2f4978ffbc`; the R3 detector exists, but current gates scan `src/elspeth`, not `tests/`.
- **Spec-less `Mock()`/`MagicMock()`** (U-CONTRACTS-1, U-CORE-1, I-1, U-ENGINE-1 partial). Filed `elspeth-e984600f90`; recommend a CI grep/lint to flag behavioral mocks without `spec=` or a real fake in `tests/`.
- **Hash-without-binding theatre** (U-CORE-1, I-1, U-ENGINE-1 partial). Filed `elspeth-e0afd080cc`; recommend a shared fixture asserting `(actual_hash) == stable_hash(input)`.
- **Dataclass-machinery tautology cluster** (all chunks, including U-ENGINE-1 partial). Tests construct a `@dataclass`, set fields, read them back. Delete on sight.
- **Production-code-path bypass in integration tests** (I-1: 26 of 32 files). Filed as `elspeth-ae9f541775`; false-confidence taxonomy fixed 2026-05-20 by moving repository-level tests out of `tests/integration/audit/`.
- **Regression-dump test files** (I-1: `test_fixes.py`). Any file named after a sprint, ticket, or "fixes" warrants scrutiny.

## Calibration learnings (across 3 waves)

- **The gap-analyst lens is the single most valuable agent** — has surfaced 9/9 critical production-code gaps; no other agent saw them on first pass.
- **The qa-analyst (theatre) lens is the second most valuable** — caught the I-1 structural defect (26-of-32 bypass) that other lenses missed at chunk-level.
- **The cross-reference task is high-leverage** — added in I-1, validated 4 prior gaps as still open and downgraded 1 issue. Add it to every wave's gap-analyst and pr-test-analyzer prompts.
- **Compression target if needed:** gap-analyst + qa-analyst + python-code-reviewer covers ~85% of what 5 agents produce.
- **Cost per chunk:** ~5 agents × ~3-5 minutes wall-clock × ~700K-1M tokens for ~25-30 files. For all 27 chunks dispatched serially: ~2-3 hours wall-clock, substantial token budget.
