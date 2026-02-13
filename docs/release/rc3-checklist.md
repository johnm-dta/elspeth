# ELSPETH RC-3 Release Validation Checklist

**Purpose:** What MUST work before shipping RC-3. This is not a feature list - it's the minimum bar for release.

**Criterion:** If any item fails, RC-3 cannot ship.

**Branch:** `RC3-quality-sprint`

**Legend:** `[ ]` unchecked (awaiting verification); `[KNOWN LIMITATION]` acceptable RC-3 limitation (see list below).

---

## 1. AUDIT INTEGRITY (Non-Negotiable)

These are the core promises of ELSPETH. Failure here is a showstopper.

### 1.1 Source Data Capture

- [ ] Every source row has `source_data_ref` populated before any processing
- [ ] Source row payload persisted to PayloadStore (not just hash)
- [ ] `elspeth run` command wires PayloadStore
- [ ] Quarantined rows recorded with original data and failure reason
- [ ] Quarantined rows include structured reason from source validation

### 1.2 Transform Boundaries

- [ ] Every transform execution creates node_state record
- [ ] Input hash recorded before transform runs
- [ ] Output hash recorded after successful transform
- [ ] Failed transforms record error details with retryable flag
- [ ] Operation I/O hashes survive payload purge (P2-2026-02-05 fix)

### 1.3 External Call Recording

- [ ] LLM calls record full request/response payloads
- [ ] HTTP calls record request/response with status codes
- [ ] Call latency captured
- [ ] Calls linked to correct node_state
- [ ] ExternalCallCompleted events include `token_id` correlation (telemetry fix)

### 1.4 Terminal States

- [ ] Every token reaches exactly one terminal state
- [ ] No silent drops (row enters system -> row has recorded outcome)
- [ ] Terminal states: COMPLETED, ROUTED, FORKED, CONSUMED_IN_BATCH, COALESCED, QUARANTINED, FAILED, EXPANDED

### 1.5 Lineage Query

- [ ] `explain_token()` returns complete lineage for any token
- [ ] Lineage includes: source_row -> tokens -> node_states -> calls -> routing_events -> outcome
- [ ] Fork/coalesce lineage traversable (parent_token_id chain)

---

## 2. DECLARATIVE DAG WIRING (New in RC-3)

The routing trilogy (ADR-004, ADR-005) replaced implicit pipeline wiring with explicit declarative routing.

### 2.1 Explicit Sink Routing

- [ ] `on_success` routing directive connects transforms to named sinks
- [ ] `input` routing directive declares where a node receives data from
- [ ] Sink references in routes validated at configuration time
- [ ] Invalid sink references rejected with clear error message

### 2.2 DAG Construction

- [ ] `ExecutionGraph.from_plugin_instances()` builds DAG from declarative config
- [ ] Topological sort produces valid execution order
- [ ] Cycle detection rejects invalid DAGs at construction time
- [ ] Schema contracts validated: upstream `guaranteed_fields` satisfy downstream `required_input_fields`

### 2.3 Node-ID Refactoring

- [ ] Processor uses node IDs consistently (not positional indices)
- [ ] NodeType enum comparisons used throughout engine
- [ ] MappingProxyType used for frozen configuration dictionaries
- [ ] Structural node allowlist enforced

### 2.4 Gate Routing (Config-Driven Only)

- [ ] Gates evaluate expressions via ExpressionParser (AST-based, no `eval`)
- [ ] `continue` passes row to next node
- [ ] `route_to_sink` sends row to named sink with reason
- [ ] `fork_to_paths` creates child tokens with correct parent linkage
- [ ] Routing events recorded in audit trail
- [ ] No gate plugin infrastructure exists (GateProtocol, BaseGate, execute_gate, PluginGateReason all removed)

---

## 3. GRACEFUL SHUTDOWN (New in RC-3)

### 3.1 Signal Handling (FEAT-05)

- [ ] SIGINT (Ctrl-C) triggers cooperative shutdown via `threading.Event`
- [ ] Processing loop checks shutdown flag between rows
- [ ] Second SIGINT force-kills the process
- [ ] CLI exits with code 3 on graceful shutdown

### 3.2 Shutdown Behavior

- [ ] Aggregation buffers flushed on shutdown
- [ ] Pending tokens written to sinks
- [ ] Checkpoint created on interrupted run
- [ ] Run marked INTERRUPTED in landscape
- [ ] Interrupted run resumable via `elspeth resume`

### 3.3 Resume Path Shutdown (FEAT-05b)

- [ ] Shutdown flag checked on resume path
- [ ] Shutdown flag checked on quarantine path in processing loop
- [ ] DROP-mode sentinel requeue failure does not raise to callers

---

## 4. CORE ENGINE FUNCTIONALITY

### 4.1 Linear Pipeline

- [ ] Source -> Transform chain -> Sink works
- [ ] Multiple transforms execute in sequence
- [ ] Output sink receives all non-routed rows

### 4.2 Fork/Coalesce

- [ ] Fork creates child tokens with correct parent linkage
- [ ] Each branch executes independently
- [ ] Coalesce merges tokens when all branches complete
- [ ] Coalesce timeout fires (not just at end-of-source)
- [ ] Late-arrival tokens after coalesce deadline recorded as failed outcomes

### 4.3 Aggregation

- [ ] Count trigger fires at threshold
- [ ] Timeout trigger fires (with known limitation: only on next row arrival)
- [ ] End-of-source flushes remaining buffers
- [ ] Batch members linked to batch in audit trail
- [ ] Trigger type recorded in metadata
- [ ] Batch-only trigger condition keys enforced in config

### 4.4 Retry Logic

- [ ] Transient failures retry with backoff
- [ ] Max retries respected
- [ ] Each attempt recorded separately
- [ ] Non-retryable errors fail immediately
- [ ] Capacity errors (PromptShield, LLM) classified as retryable

### 4.5 Checkpoint Recovery

- [ ] Checkpoints created at processing boundaries
- [ ] `elspeth resume` continues from last checkpoint
- [ ] Already-processed rows not reprocessed
- [ ] Aggregation state restored
- [ ] Call index seeded from MAX(call_index) on resume (SAFE-05, prevents UNIQUE violations)

---

## 5. PLUGIN CORRECTNESS

### 5.1 Sources

- [ ] CSV source handles multiline quoted fields
- [ ] CSV source quarantines malformed header parse errors
- [ ] CSV source handles `csv.Error` in skip_rows
- [ ] JSON source handles both array and JSONL formats
- [ ] JSON source handles surrogate-escape quarantine for UTF-16/32
- [ ] JSON source `data_key` structural errors use parse schema mode
- [ ] JSONL multibyte decoding handled correctly
- [ ] Field normalization produces valid Python identifiers
- [ ] Collision detection reports clear errors

### 5.2 Core Transforms

- [ ] Passthrough passes rows unchanged
- [ ] Field mapper renames fields correctly
- [ ] Field mapper preserves original-name lineage for renames
- [ ] Truncate respects length limits
- [ ] Keyword filter honors PipelineRow dual-name field lookup
- [ ] JSON explode handles complex output_field values correctly

### 5.3 LLM Transforms (if using LLM pack)

- [ ] Azure LLM transform calls API and records response
- [ ] Template variables substituted correctly
- [ ] Structured output mode returns parsed JSON
- [ ] Rate limiting prevents 429 errors
- [ ] Batch telemetry attributed correctly (OpenRouter batch fix)
- [ ] Multi-query dual-name resolution works for PipelineRow inputs
- [ ] Duplicate multi-query output suffixes rejected
- [ ] BaseLLMTransform contract includes usage metadata

### 5.4 Sinks

- [ ] CSV sink writes valid CSV
- [ ] JSON sink writes valid JSON/JSONL
- [ ] Database sink inserts rows correctly
- [ ] Azure Blob sink keeps retries idempotent after upload errors
- [ ] Azure Blob sink handles multi-batch overwrite data loss fix
- [ ] Blob sink CSV schema drift and rename metadata propagation handled

### 5.5 Safety Transforms

- [ ] Azure Content Safety fails closed on missing explicit safety fields
- [ ] Azure Prompt Shield capacity errors treated as row-level retryable results
- [ ] Content safety unknown categories do NOT pass through as safe

---

## 6. CONTRACTS AND SCHEMA

### 6.1 Schema Contracts

- [ ] Sources declare guaranteed output fields
- [ ] Transforms declare required input fields
- [ ] DAG construction validates field compatibility
- [ ] Mismatches rejected before execution
- [ ] Deterministic contract audit ordering after merges

### 6.2 Contract Propagation

- [ ] Contracts propagate through transform chain
- [ ] Complex fields (dict/list) preserved as `python_type=object` in propagated contracts
- [ ] Strict field lookup enforced (no silent fallbacks)
- [ ] Immutable schema indices maintained
- [ ] FLEXIBLE run contract infer-and-lock initialization works correctly
- [ ] ORIGINAL header contract lookup corruption fails fast

### 6.3 Config Contracts

- [ ] Settings->Runtime protocol enforcement passes (`.venv/bin/python -m scripts.check_contracts`)
- [ ] All Settings fields mapped in `from_settings()` methods
- [ ] `FIELD_MAPPINGS` and `INTERNAL_DEFAULTS` up to date
- [ ] `pytest tests/core/test_config_alignment.py` passes

---

## 7. TELEMETRY

### 7.1 Core Telemetry

- [ ] Telemetry events include `run_id` and `token_id`
- [ ] ExternalCallCompleted includes token_id correlation
- [ ] FieldResolutionApplied classified as row-level telemetry (filtering fix)
- [ ] No silent failures: emit or explicitly acknowledge "nothing to send"

### 7.2 Exporter Correctness

- [ ] OTLP exporter acknowledges empty flushes
- [ ] Azure Monitor exporter acknowledges empty flushes
- [ ] Datadog unsupported `api_key` option rejected
- [ ] Tracing providers validated (no silent disable)
- [ ] OTLP exporter config types validated
- [ ] Telemetry exporter hook discovery works correctly

### 7.3 Backpressure

- [ ] DROP-mode evicts oldest queued events (not newest)
- [ ] Queue accounting hardened
- [ ] Hook validation enforced
- [ ] PluginContext telemetry payloads snapshotted (no mutation after emit)

---

## 8. CLI COMMANDS

### 8.1 `elspeth run`

- [ ] `--settings` loads configuration
- [ ] `--execute` required to actually run (safety gate)
- [ ] `--dry-run` validates without executing
- [ ] PayloadStore instantiated and passed to engine
- [ ] Graceful shutdown on SIGINT (exit code 3)

### 8.2 `elspeth validate`

- [ ] Validates YAML syntax
- [ ] Validates plugin references exist
- [ ] Validates sink references in routes (declarative DAG wiring)
- [ ] Reports clear error messages
- [ ] Schema plugin config errors returned as structured validation results

### 8.3 `elspeth resume`

- [ ] Loads checkpoint from previous run
- [ ] Resumes from last known good state
- [ ] PayloadStore wired correctly
- [ ] Graceful shutdown works on resume path (FEAT-05b)

### 8.4 `elspeth explain`

- [ ] `--run` and `--row` parameters work
- [ ] Returns lineage data (JSON mode minimum)
- [ ] TUI mode functional

### 8.5 `elspeth plugins list`

- [ ] Lists all available plugins
- [ ] `--type` filter works
- [ ] Plugin name contract enforced during discovery

### 8.6 `elspeth purge`

- [ ] `--retention-days` respected
- [ ] `--dry-run` shows what would be deleted
- [ ] Preserves hashes after payload deletion
- [ ] Operation payload refs included in retention purge coverage
- [ ] SQLite URI purge validation correct

---

## 9. DATA INTEGRITY

### 9.1 Canonical JSON

- [ ] NaN rejected with clear error (not silently converted)
- [ ] Infinity rejected with clear error
- [ ] numpy types converted correctly
- [ ] pandas Timestamp -> UTC ISO8601
- [ ] Hash stable across process restarts
- [ ] JSONFormatter rejects coercion and non-finite values

### 9.2 Payload Store

- [ ] `put()` stores data and returns ref
- [ ] `get()` retrieves data by ref
- [ ] `exists()` returns correct boolean
- [ ] Hash verification on read

### 9.3 Database Integrity

- [ ] Foreign keys enforced
- [ ] No orphan records
- [ ] Unique constraints respected
- [ ] Composite primary key (node_id, run_id) on nodes table handled correctly
- [ ] NodeRepository schema_fields_json shape validated
- [ ] Scalar CSV audit validation enforced

---

## 10. CONFIGURATION

### 10.1 Basic Loading

- [ ] YAML syntax parsed correctly
- [ ] Environment variable interpolation `${VAR}` works
- [ ] Default values applied

### 10.2 Validation

- [ ] Invalid plugin names rejected
- [ ] Invalid sink references in routes rejected
- [ ] Missing required fields reported
- [ ] Unknown plugin hooks rejected during registration
- [ ] Non-dict plugin config input fails fast
- [ ] Rate-limit protocol aligned with registry contract

---

## 11. TEST SUITE

### 11.1 Test Suite v2 Migration (Complete)

- [x] v2 migration complete: `tests_v2/` renamed to `tests/`
- [x] Old v1 suite deleted (507 files, 222K lines)
- [ ] 8,138+ tests collected
- [ ] All imports rewritten (204 imports across 123 files)
- [ ] pyproject.toml test configuration updated

### 11.2 Test Coverage by Phase

- [ ] Unit tests pass (Phase 2: contracts, core, engine, plugins)
- [ ] Property tests pass (Phase 3: Hypothesis-based)
- [ ] Integration tests pass (Phase 4)
- [ ] E2E tests pass (Phase 5)
- [ ] Performance tests pass (Phase 6: benchmarks, stress, scalability, memory)

### 11.3 Quality Gates

- [ ] `mypy src/` passes
- [ ] `ruff check src/` passes
- [ ] `.venv/bin/python -m scripts.check_contracts` passes
- [ ] `enforce_tier_model.py check` passes

---

## 12. DOCUMENTATION (RC-3 Remediation)

### 12.1 Version Strings (F-01)

- [ ] All version references updated to RC-3 / 0.3.0
- [ ] `pyproject.toml` version is `0.3.0`
- [ ] `src/elspeth/__init__.py` version matches pyproject.toml

### 12.2 Gate Plugin Removal (F-03)

- [ ] No references to `GateProtocol`, `BaseGate`, `execute_gate()`, `PluginGateReason` outside of archive
- [ ] Contract docs updated (system-operations.md, execution-graph.md)
- [ ] CLAUDE.md plugin ownership section reflects gates as config-driven, not plugins

### 12.3 Source Layout (F-02)

- [ ] CLAUDE.md source tree diagram matches actual directory structure

### 12.4 Release Docs (F-09)

- [ ] `docs/release/guarantees.md` versioned as RC-3
- [ ] RC-3 version table entry includes: declarative DAG wiring, graceful shutdown, DROP-mode handling, gate plugin removal, telemetry hardening, test suite v2 migration
- [ ] Feature inventory updated with RC-3 additions

### 12.5 Project Layout (F-04, F-05, F-06, F-07)

- [ ] mutmut paths in pyproject.toml resolve to actual files/packages
- [ ] REQUIREMENTS.md deleted or aligned with pyproject.toml
- [ ] Broken links in docs/README.md resolved
- [ ] Placeholder URLs updated (`your-org` -> `johnm-dta`)

### 12.6 Examples (DOC-01, DOC-02)

- [ ] All example directories have README files
- [ ] Master `examples/README.md` index exists
- [ ] Fork/coalesce example created (with ARCH-15 limitation documented)

---

## KNOWN ISSUES ACCEPTABLE FOR RC-3

These are documented limitations, not blockers:

1. **Timeout triggers** - Fire only when next row arrives (documented limitation, same as RC-2)
2. **Concurrent processing** - Single-threaded only (acceptable for RC-3)
3. **True idle timeout** - Aggregation buffers not flushed during completely idle periods without heartbeat rows
4. **Circuit breaker** - RetryManager has no circuit breaker; many rows against a dead service retry individually (FEAT-06, deferred)
5. **CLI surface gaps** - `elspeth status`, `elspeth export`, `elspeth db migrate` not yet implemented (FEAT-04, deferred)
6. **AggregationExecutor parallel dictionaries** - 6 parallel dicts instead of consolidated dataclass (ARCH-06, deferred)
7. **Resume schema verification** - No schema fingerprint comparison between original and resumed run (ARCH-14, deferred)
8. **LLM boundary `.get()` chains** - 4 files use `data.get("usage") or {}` on external responses; technically correct at Tier 3 but produces silent empty-dict fallbacks (CRIT-02, deferred)
9. **Prometheus metrics** - No pull-based metrics endpoint for operational dashboards (OBS-04, deferred)

---

## VERIFICATION PROCEDURE

For each item above:

1. Write or identify existing test
2. Run test against clean database
3. Verify audit trail contains expected records
4. Mark checkbox when confirmed

**Minimum verification commands:**

```bash
# Full test suite
.venv/bin/python -m pytest tests/ -q

# Type checking
.venv/bin/python -m mypy src/

# Linting
.venv/bin/python -m ruff check src/

# Config contracts
.venv/bin/python -m scripts.check_contracts

# Tier model enforcement
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

**Release gate:** All non-"acceptable limitation" items checked. All quality gate commands exit 0.
