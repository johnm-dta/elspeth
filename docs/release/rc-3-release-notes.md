# ELSPETH RC-3 Release Notes

**Date:** February 2026
**Version:** 0.3.0
**Branch:** `RC3-quality-sprint`
**Commits since RC-2:** 117

---

## Highlights

- **Routing trilogy completed** -- three-phase architectural overhaul replaces implicit `default_sink` routing with fully declarative DAG wiring (ADR-004, ADR-005). Every edge in the pipeline is now explicit.
- **Gate plugins removed entirely** -- routing is config-driven only via `GateSettings` and `ExpressionParser`. Approximately 3,000 lines of gate plugin infrastructure deleted.
- **Test suite v2 migration complete** -- 7-phase cutover from v1 (7,487 tests, 507 files, 222K lines deleted) to v2 (8,138 tests). Property testing expanded with Hypothesis.
- **Graceful shutdown** -- cooperative signal handling (SIGTERM/SIGINT) with aggregation buffer flush, checkpoint creation, and resumable interrupted runs.
- **65+ remediation items resolved** from the RC-2 comprehensive architecture analysis (75+ original findings).

---

## Breaking Changes

These changes are incompatible with RC-2 configurations and checkpoints. Since ELSPETH has no external users yet, no migration tooling is provided.

### Explicit `on_success` routing replaces implicit `default_sink`

The `default_sink` configuration field has been removed. Every terminal node must declare an explicit `on_success` target. Source nodes also require `on_success`.

**Before (RC-2):**
```yaml
default_sink: output
source:
  plugin: csv
  options: { path: input.csv }
transforms:
  - plugin: classifier
    on_error: quarantine
```

**After (RC-3):**
```yaml
source:
  plugin: csv
  options:
    path: input.csv
  on_success: output
transforms:
  - plugin: classifier
    on_error: quarantine
    on_success: output
```

See [ADR-004: Explicit Sink Routing](../architecture/adr/004-adr-explicit-sink-routing.md) for the full rationale.

### Explicit `input` wiring replaces positional ordering

Transforms, gates, and aggregation nodes now declare their input connection by name via the `input:` field. Positional ordering in the YAML no longer determines dataflow.

```yaml
source:
  plugin: csv
  options: { path: input.csv }
  on_success: raw_data

transforms:
  - plugin: enricher
    name: enricher
    input: raw_data
    on_success: enriched
  - plugin: classifier
    name: classifier
    input: enriched
    on_success: output

sinks:
  output: { plugin: csv, options: { path: results.csv } }
```

Connection names and sink names occupy separate namespaces. A collision between the two is a `GraphValidationError`. See [ADR-005: Declarative DAG Wiring](../architecture/adr/005-adr-declarative-dag-wiring.md).

### Gate plugins removed

Gate plugins (`GateProtocol`, `BaseGate`, `execute_gate()`, `PluginGateReason`) have been deleted from the codebase. Routing is exclusively config-driven via `GateSettings` with `ExpressionParser` (AST-based, no `eval`). This is a hard prohibition -- gate plugin infrastructure will not be reintroduced.

### `on_success` lifted to settings level

The `on_success` field has been moved from plugin `options:` to the top-level settings for all node types, aligning with how `on_error` is already configured. This applies to transforms, sources, aggregations, and coalesce nodes.

### Node ID derivation changed

Node IDs are now derived from `settings.name` instead of `plugin_name + sequence_number`. This makes node IDs position-independent and human-readable in audit records. Existing checkpoints are incompatible.

### Test suite restructured

The `tests/` directory has been completely restructured. The v1 suite (507 files, 222K lines) was deleted and replaced with the v2 suite. Import paths changed across 123 files (204 import rewrites).

### Pre-2026-01-24 checkpoints incompatible

All checkpoints created before 2026-01-24 are invalid due to node ID format changes introduced in the routing refactor. Attempting to resume from a pre-2026-01-24 checkpoint will fail with a clear error message. Delete old checkpoint files and re-run affected pipelines.

---

## New Features

### Graceful Shutdown (FEAT-05, FEAT-05b)

Cooperative shutdown via `threading.Event`. Signal handler sets the event; the processing loop checks between rows.

- First SIGINT/SIGTERM: flush aggregation buffers, write pending tokens to sinks, create checkpoint, mark run `INTERRUPTED`
- Second SIGINT: force-kill
- CLI exits with code 3 on graceful shutdown
- Interrupted runs are resumable via `elspeth resume`
- Shutdown flag checked on quarantine path and resume path
- 16 tests (9 unit + 7 integration)

**Commits:** `6286367f` (FEAT-05), `9a9fd8ca` (quarantine path), `d2a6cbef` (FEAT-05b resume path)

### Declarative DAG Wiring (ADR-004, ADR-005)

The routing trilogy -- three sequential phases that replaced ELSPETH's implicit routing model with fully declarative wiring:

1. **Phase 1** (ADR-004, `elspeth-rapid-o639`): Explicit `on_success` routing. Removed `default_sink` and its ~472 references across 50 test files and 30 example YAMLs.
2. **Phase 2** (`elspeth-rapid-hscm`): Processor node-ID refactoring. Replaced `start_step` integer indexing with `current_node_id`-based DAG traversal.
3. **Phase 3** (ADR-005, `elspeth-rapid-tbia`): Declarative `input:` wiring. Replaced positional edge inference with named connection matching. ~900+ reference updates.

Key design decisions:
- Connection and sink namespaces are separate (collision is a validation error)
- `settings.name` drives node IDs (position-independent, human-readable)
- `branch_name` is lineage metadata only, never used for routing
- Aggregations stay post-transform; mid-chain aggregations deferred

### Per-Branch Fork Transforms (ARCH-15)

Fork branches can now have intermediate transforms before reaching coalesce. `CoalesceSettings.branches` accepts dict format (`{branch_name: input_connection}`) to wire per-branch transform chains. List format (`[a, b]`) normalizes to identity dict (`{a: a, b: b}`) — fully backward compatible.

- `ExecutionGraph.get_branch_first_nodes()` provides O(1) dispatch to first transform per branch
- DAG builder: conditional producer registration (identity branches use COPY edges, transform branches use connection resolution)
- Strategy-aware schema validation: union merges check type compatibility across branches; nested/select skip cross-branch checks
- 23 new tests (10 unit, 8 property, 5 integration)

**Commit:** `83a6d40a`

### DROP-Mode Sentinel Requeue Handling

Prevents DROP-mode sentinel requeue failure from raising to callers. Fixes a race condition where shutdown sentinels could fail to requeue in telemetry DROP mode.

**Commits:** `bbc2f515`, `3af35d0d` (race fix), `1acec271` (flush join guarantee)

### Call Index Seeding on Resume (SAFE-05)

Call index is now seeded from `MAX(call_index)` on resume, preventing UNIQUE constraint violations when resuming runs that made external calls.

**Commit:** `b60cdcd9`

### ExecutionGraph Public API (TEST-01)

13 public setter methods and 5 method renames on `ExecutionGraph`. Over 60 private attribute accesses eliminated from tests, replaced with proper public API calls.

**Commit:** `1c31869b`

---

## Bug Fixes

### Critical and High Priority

| Fix | Commit | Description |
|-----|--------|-------------|
| Schema reconstruction crash | `b29f87e8` | Fixed crash on `Optional[NestedModel]` fields and stale batch token IDs |
| Operation I/O hashes | `264ab2cf` | Added operation I/O hashes to survive payload purge (P2-2026-02-05) |
| Resume schema fidelity | `4b6dee2f` | Fixed resume schema fidelity and archived stale bug tickets |
| Batch telemetry mis-attribution | `e4ff9cf0` | Corrected batch telemetry mis-attribution in OpenRouter batch transform |
| CSV header parse errors | `7c547838` | Quarantine malformed header parse errors instead of crashing |
| Sink contract sync | `78f8ecec` | Sync sink `ctx.contract` from sink-bound token contracts |
| Azure safety fails closed | `ae4eb816` | Fail closed on missing explicit Azure safety fields (was failing open) |
| PromptShield capacity errors | `129c08d2` | Handle PromptShield capacity errors as row-level retryable results |
| Coalesce late arrivals | `bfef176e` | Record late-arrival failed outcomes immediately |
| JSON source encoding | `8d239c8e` | Fix JSONSource surrogateescape quarantine for UTF-16/32 |

### Contract and Telemetry Hardening

| Fix | Commit | Description |
|-----|--------|-------------|
| Contract field propagation | `c21cdb27` | Preserve dict/list fields in propagated contracts |
| Strict field lookup | `250ffee0` | Strict field lookup and immutable schema indices |
| Header contract corruption | `6a42515e` | Fail fast on ORIGINAL header contract lookup corruption |
| Telemetry queue accounting | `778b9efc` | Harden queue accounting and hook validation |
| Contract violation keys | `cc8de379` | Normalize violation error keys |
| TransformErrorReason alignment | `2d7b7302` | Align TransformErrorReason with violation helpers |
| OTLP empty flushes | `7ac85893`, `5afd33f8` | Acknowledge empty OTLP and Azure Monitor flushes |
| Telemetry token correlation | `f38cfa2c` | Add `token_id` correlation to external call events |
| Telemetry row-level classification | `047d8975` | Classify `FieldResolutionApplied` as row-level telemetry |
| Telemetry registration | `71c7e1d0` | Handle `ValueError` in telemetry registration |
| Telemetry drop-mode eviction | `337405b8` | Fix telemetry drop-mode to evict oldest queued events |
| Plugin name enforcement | `57c5084b` | Enforce plugin name contract during discovery |
| Plugin hook registration | `44482e86` | Reject unknown plugin hooks during registration |

### LLM Plugin Fixes

| Fix | Commit | Description |
|-----|--------|-------------|
| Azure LLM context token | `78f8ecec` | Fail fast when Azure LLM context token is missing |
| Azure blob sink retries | `7a99c8be` | Keep retries idempotent after upload errors |
| Azure blob sink conflicts | `ae6d44bd` | Audit overwrite conflict failures in blob sink |
| Blob sink schema drift | `6cdc20a4` | Fix blob sink CSV schema drift and rename metadata propagation |
| Batch stats empty batches | `bc208c3b` | Honor `compute_mean` for empty batches |
| OpenRouter batch state_id | `ab4f6bbd` | Fail fast when openrouter batch `state_id` is missing |
| Multi-query duplicate suffixes | `8cbae041` | Reject duplicate multi-query output suffixes |
| BaseLLMTransform usage metadata | `08150b46` | Ensure BaseLLMTransform contract includes usage metadata |
| PromptTemplate hash consistency | `5e2011e1` | Freeze PromptTemplate lookup snapshot for audit hash consistency |
| JSONExplode complex output | `50422dc2` | Fix JSONExplode contract for complex `output_field` values |

### Other Fixes

| Fix | Commit | Description |
|-----|--------|-------------|
| Rate-limit protocol alignment | `119bd537` | Align rate-limit protocol with registry contract |
| Service principal auth | `985d32e6` | Reject partial service principal auth when mixed with other methods |
| Schema plugin config errors | `6c049f45` | Convert schema plugin config errors to structured validation results |
| OTLP exporter config types | `e872a1f9` | Validate OTLP exporter config types |
| Tracing provider validation | `2b947834` | Validate tracing providers and close silent-disable bug |
| CSV audit validation | `d16f56ab` | Enforce scalar CSV audit validation |
| PluginContext telemetry snapshots | `3e51467c` | Snapshot PluginContext telemetry payloads |
| NodeRepository schema fields | `e34d1acb` | Validate `schema_fields_json` shape in NodeRepository |
| LLM retry classification | `3f94be33` | Fix LLM retry classification false positives |
| Observed schema mode | `6b6fb334` | Reject non-list fields in observed schema mode |
| Sink merge failure state | `62cd2bd2` | Fix sink merge failure state closure and prompt shield capacity retries |
| Optional Annotated float | `bef6298b` | Fix optional `Annotated` float type extraction in contracts |
| Expand token lock | `67908a7a` | Validate `expand_token` contract lock before recorder writes |
| Operation status literals | `a38ca874` | Validate Operation status/type literals in audit contract |
| Flexible run contract | `d85b2e9e` | Fix flexible run contract persistence and blob sink retry state |
| Deterministic contract ordering | `d891319e` | Fix deterministic contract audit ordering after merges |
| Datadog api_key rejection | `32edbfef` | Reject unsupported Datadog `api_key` exporter option |

---

## Architecture Changes

### Routing Trilogy (3 Phases)

The routing trilogy is the single largest architectural change in RC-3. It touched every layer of the system:

- **Phase 1 -- Explicit Sink Routing:** Removed `default_sink` and 40+ fallback chains. Separated `branch_name` (lineage) from routing (now `on_success` only). ~472 references updated.
- **Phase 2 -- Processor Node-ID Refactoring:** Replaced `start_step` integer indexing with `current_node_id` DAG traversal. Processor now follows graph edges rather than list positions.
- **Phase 3 -- Declarative DAG Wiring:** Replaced positional edge inference with named connection matching via `input:` fields. ~900+ references updated.

### Gate Plugin Removal (~3,000 Lines Deleted)

`GateProtocol`, `BaseGate`, `execute_gate()`, and `PluginGateReason` were deleted from source and tests. Routing is exclusively config-driven via `GateSettings` + `ExpressionParser` (AST-based, no `eval`). This is a hard prohibition documented in project memory.

### Engine Package Refactoring

Single-file modules refactored into packages:
- `engine/orchestrator.py` became `engine/orchestrator/` (core, aggregation, export, outcomes, types, validation)
- `engine/executors.py` became `engine/executors/` (transform, gate, sink, aggregation, types)

### Contract Hardening

Extensive work on the contracts subsystem throughout the sprint:
- Strict field lookup with immutable schema indices
- Normalized violation error keys
- Deterministic contract audit ordering
- Plugin name contract enforcement during discovery
- Unknown plugin hook rejection during registration

### Unused Session Parameter Removal (ARCH-13)

Removed unused `session` parameter from all Repository classes.

**Commit:** `b60cdcd9`

---

## Test Suite

### v2 Migration Complete (7 Phases)

| Phase | Content | Tests |
|-------|---------|-------|
| Phase 0+1 | Scaffolding + Factories | -- |
| Phase 2A | Contracts | 1,142 |
| Phase 2B+2C+2D | Core + Engine + Plugins | 3,823 |
| Phase 3 | Property (Hypothesis) | 1,057 |
| Phase 4 | Integration | 482 |
| Phase 5 | End-to-End | 48 |
| Phase 6 | Performance (benchmarks, stress, scalability, memory) | 67 |
| Phase 7 | Cutover | -- |
| **Total** | | **8,138 collected** |

**Final stats:** 8,037 passed, 16 skipped, 3 xfailed.

### Cutover Details (Phase 7)

- Deleted v1 suite: 7,487 tests, 507 files, 222K lines
- Renamed `tests_v2/` to `tests/` via `git mv`
- Rewrote 204 imports across 123 files
- Fixed 47 pre-existing lint issues (import sorting, formatting, unused vars, `zip` strict)
- Updated `pyproject.toml` (removed v2 per-file-ignore, fixed mutmut paths)

### Property Testing Expansion

Three rounds of gap analysis added comprehensive Hypothesis-based property tests:
- Round 1: SSRF (19), DAG topologies (20), ChaosLLM (63), token ops (11), web scrape (15)
- Round 2: Triggers (54), routing (23), schema contracts (15), reorder buffer (19), +92 others
- Round 3: Orchestrator lifecycle (37), landscape recording (32), LLM templates (21), multi-query (11), Azure safety (29)

---

## Documentation

### RC-3 Documentation Remediation (20 Findings)

A full project audit identified 20 documentation and metadata findings (F-01 through F-20), organized into 4 priority phases:

**Phase 1 (Silent Breakage):**
- F-01: Version strings normalized from RC-2/RC-2.5 to RC-3 across 10+ files
- F-02: CLAUDE.md source layout diagram updated to reflect package refactoring
- F-03: Gate plugin references removed from contract docs
- F-04: mutmut paths in pyproject.toml fixed (pointed to deleted single-file modules)
- F-05: REQUIREMENTS.md deleted (pyproject.toml is the single source of truth)
- F-06: Broken links in docs/README.md fixed

**Phase 2 (Release Alignment):**
- F-07: Placeholder URLs updated (`your-org` to `johnm-dta`)
- F-08: ADR-004 and ADR-005 added to docs/README.md index
- F-09: Release docs updated for RC-3
- F-10: Feature inventory updated with graceful shutdown and related features
- F-11: ARCHITECTURE.md LOC statistics updated (~75,800 source, ~207,000 test)
- F-12: pyproject.toml classifier updated from Alpha to Beta
- F-20: Telemetry OTLP exporter version reference fixed

**Phase 3 (Internal Consistency):**
- F-13: TEST_SYSTEM.md title updated (no longer "v2")
- F-14: Contradictory FEAT-05 state in remediation plan reconciled
- F-15: Plans index metadata corrected
- F-16: Bug tracking docs marked for archival (beads is active tracker)

**Phase 4 (Polish):**
- F-17: 12 example READMEs created; master examples/README.md index added
- F-18: Archive breadcrumb fix (stale reference to non-existent directory)
- F-19: Stale architecture analysis docs deleted

### Other Documentation

- LLM execution models documented (accept vs process, batch lifecycle) -- ARCH-02
- Plugin lifecycle hooks documented (on_start/on_complete/close ordering) -- ARCH-07
- Access control limitations documented in release guarantees -- DOC-03
- Pre-2026-01-24 checkpoint incompatibility documented -- DOC-04
- Audit export signing recommendation documented -- DOC-05
- 5 new examples filling coverage gaps: fork_coalesce, checkpoint_resume, database_sink, rate_limited_llm, retention_purge

### ADRs

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-004](../architecture/adr/004-adr-explicit-sink-routing.md) | Explicit Sink Routing | Approved, Implemented |
| [ADR-005](../architecture/adr/005-adr-declarative-dag-wiring.md) | Declarative DAG Wiring | Approved, Implemented |

---

## Known Issues

### Open (10 remaining from RC-2 remediation)

| ID | Summary | Priority |
|----|---------|----------|
| CRIT-02 | `.get("usage") or {}` chains on LLM API responses (4 files) | MEDIUM |
| FEAT-04 | Missing CLI commands (`status`, `export`, `db migrate`) | MEDIUM |
| FEAT-06 | No circuit breaker in retry logic | MEDIUM |
| ARCH-06 | AggregationExecutor maintains 6 parallel dictionaries | MEDIUM |
| ARCH-14 | Resume schema verification gap (no schema fingerprint) | MEDIUM |
| QW-10 | CLI event formatters not extracted (~2,000 lines monolithic) | LOW |
| OBS-04 | No Prometheus-style pull metrics | LOW |

### Known Flaky Test

`tests/core/rate_limit/test_limiter.py::test_acquire_within_limit` -- timing-dependent, occasionally fails under CI load.

### Known Limitation

Aggregation timeout triggers fire when the next row arrives, not during completely idle periods. If no rows arrive after the timeout elapses, buffered data waits until either a new row arrives or the source completes. Combine timeout with count triggers, or implement heartbeat rows at the source level.

---

## Infrastructure Changes

- Version bumped to `0.3.0` (from `0.1.0`)
- pyproject.toml classifier updated to `Development Status :: 4 - Beta`
- REQUIREMENTS.md deleted (pyproject.toml is the single source of truth)
- Obsolete `.codex` configuration file removed
- Stale architecture analysis docs (`docs/arch-analysis-2026-02-02-1114/`) deleted
- Old migration script deleted

---

## Release Guarantees

The full set of RC-3 guarantees is documented in [docs/release/guarantees.md](../release/guarantees.md). Key additions for RC-3:

- Declarative DAG wiring with explicit input/output connections
- Graceful shutdown with checkpoint creation and resumable interrupted runs
- DROP-mode telemetry handling
- Gate plugin removal (config-driven routing only)
- Telemetry hardening across OTLP, Azure Monitor, and Datadog exporters

---

## Codebase Statistics

| Metric | Value |
|--------|-------|
| Source lines | ~75,800 across 234 Python files |
| Test lines | ~207,000 (2.7:1 test-to-source ratio) |
| Tests collected | 8,138 |
| Tests passing | 8,037 |
| Tests skipped | 16 |
| Commits (RC-3 branch) | 117 |
| Remediation items resolved | ~65 of 75 |
| Remaining open items | 10 |
