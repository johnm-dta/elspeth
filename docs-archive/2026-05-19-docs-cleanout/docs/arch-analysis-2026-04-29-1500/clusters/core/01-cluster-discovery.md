# 01 — core/ cluster discovery (L2 holistic scan)

## Cluster shape

The core cluster is **49 .py files, 20,791 LOC**, distributed across **6 sub-packages and 11 standalone modules**. Sub-packages account for **16,025 LOC (~77%)** and standalone modules account for **4,766 LOC (~23%)**. LOC concentration is heavily Landscape-weighted: `landscape/` alone holds **9,384 LOC across 18 files (~45% of cluster)**, with the next-largest sub-area (`dag/`) at 17%, and the four flagged deep-dive candidates (`config.py` 2,227 + `dag/graph.py` 1,968 + `landscape/execution_repository.py` 1,750 + `landscape/data_flow_repository.py` 1,590 = 7,535 LOC) holding **~36% of the cluster** in four files. This concentration is exactly the §7 P3 framing of "cascade-prone" risk — the four files own configuration loading, DAG construction/validation, and the two heaviest Landscape repositories.

Sub-areas sorted by LOC (verified inventory):

| LOC | Sub-area | Files |
|---:|---|---:|
| 9,384 | `landscape/` | 18 (incl. `execution_repository.py` 1,750 — flag, `data_flow_repository.py` 1,590 — flag) |
| 3,549 | `dag/` | 5 (incl. `graph.py` 1,968 — flag) |
| 2,227 | `config.py` (standalone) | 1 (flag) |
| 1,237 | `checkpoint/` | 5 |
| 940 | `security/` | 4 |
| 820 | `expression_parser.py` (standalone) | 1 |
| 470 | `rate_limit/` | 3 |
| 445 | `retention/` | 2 |
| 309 | `canonical.py` (standalone) | 1 |
| 293 | `templates.py` (standalone) | 1 |
| 208 | `operations.py` (standalone) | 1 |
| 185 | `logging.py` (standalone) | 1 |
| 183 | `payload_store.py` (standalone) | 1 |
| 173 | `dependency_config.py` (standalone) | 1 |
| 124 | `secrets.py` (standalone) | 1 |
| 111 | `events.py` (standalone) | 1 |
| 100 | `__init__.py` | 1 |
| 33 | `identifiers.py` (standalone) | 1 |

## Entry points and runtime surfaces

`core/` ships **no console scripts** — it is a library tier consumed by `engine/` (L2) and the L3 surfaces. Public symbols are re-exported through `core/__init__.py` (100 lines, read in full). The `__all__` list is 41 names long and groups into:

- **Configuration façade:** `ElspethSettings`, `load_settings`, plus 11 setting dataclasses (`CheckpointSettings`, `ConcurrencySettings`, `DatabaseSettings`, `LandscapeExportSettings`, `LandscapeSettings`, `PayloadStoreSettings`, `RateLimitSettings`, `RetrySettings`, `SecretsConfig`, `ServiceRateLimit`, `SinkSettings`, `SourceSettings`, `TransformSettings`) plus `SecretFingerprintError`.
- **Canonicalisation:** `CANONICAL_VERSION`, `canonical_json`, `stable_hash`.
- **Checkpointing:** `CheckpointManager`, `RecoveryManager`, `ResumeCheck`, `ResumePoint` (the latter two re-exported from `contracts/`).
- **DAG:** `ExecutionGraph`, `GraphValidationError`, `GraphValidationWarning`, `NodeConfig`, `NodeInfo`, `WiredTransform`.
- **EventBus:** `EventBus`, `EventBusProtocol`, `NullEventBus`.
- **Expression parser:** `ExpressionEvaluationError`, `ExpressionParser`, `ExpressionSecurityError`, `ExpressionSyntaxError` (Phase-1 ADR-006 relocation per `KNOW-ADR-006b`).
- **Logging:** `configure_logging`, `get_logger`.
- **Payload store:** `FilesystemPayloadStore` (`PayloadStore` *protocol* is re-exported from `contracts/`, not `core/`).
- **Integrity primitives:** `IntegrityError` (re-exported from `contracts/`).

The Landscape primitives — `RecorderFactory`, `LandscapeDB`, `ExecutionRepository`, `DataFlowRepository`, `QueryRepository`, `RunLifecycleRepository`, `LineageResult`/`explain`, the schema tables, formatters, and audit-trail dataclasses — are **NOT** re-exported through `core/__init__.py`. Callers reach them via `elspeth.core.landscape.*` directly, and the documented entry surface is `RecorderFactory` (the facade) per `KNOW-A29`. This is intentional encapsulation: callers go through the factory, not direct repository imports.

## Foundation-tier orientation

`core/` is the system's **foundation tier** as framed in `KNOW-A22` (Container Responsibilities), `KNOW-A29`–`KNOW-A33` (Landscape Components), `KNOW-C27` (key subsystems list), `KNOW-C28` (DAG execution model), `KNOW-C45` (configuration precedence), and `KNOW-A53`–`KNOW-A54` (dependency graph). Its responsibilities cluster around six themes:

1. **Audit backbone (Landscape).** The 4-repository design (`KNOW-A30`) under a `RecorderFactory` facade (`KNOW-A29`) backed by `LandscapeDB` (SQLAlchemy Core, SQLite/SQLCipher dev, PostgreSQL prod per `KNOW-A4`/`KNOW-A49`). The audit DB has 20 tables per `landscape/schema.py` (`[DIVERGES FROM KNOW-A24]` which claims 21 — recorded as one-table drift; defer to doc-correctness pass). This is **Tier 1 / "Our Data" / FULL TRUST** territory per `KNOW-C11`–`KNOW-C12`: every read crashes on anomaly, never coerces.

2. **DAG construction and validation.** `ExecutionGraph.from_plugin_instances()` (the canonical entry point per `KNOW-C44`) plus the schema-contract two-phase validation in `dag/builder.py` (`KNOW-A55`–`KNOW-A57`, `KNOW-ADR-003`/`-005`). Coalesce-merge logic for fork/join is in `dag/coalesce_merge.py` (extracted from `builder.py` for testability per its own docstring) implementing the policy-driven semantics (`KNOW-A47`–`KNOW-A48`).

3. **Configuration loading.** `config.py` (2,227 LOC, deep-dive flag) houses Dynaconf+Pydantic settings (`KNOW-C36`) including the `ElspethSettings` root and 11+ child dataclasses. `dependency_config.py` adds Pydantic models for cross-pipeline dependencies and commencement gates (with ExpressionParser pre-validation at config time). Configuration precedence is `KNOW-C45` (runtime → pipeline → profile → pack defaults → system defaults).

4. **Determinism + integrity primitives.** `canonical.py` implements two-phase canonical JSON (KNOW-C8 hashing): Phase 1 is our `_normalize_value()` walker handling pandas/numpy types with **strict NaN/Infinity rejection** (defense-in-depth for audit integrity); Phase 2 is `rfc8785.dumps()` for deterministic JSON per RFC 8785/JCS. The same module owns `compute_full_topology_hash()` for checkpoint validation. `payload_store.py` (`FilesystemPayloadStore`) is content-addressable storage with timing-safe hash comparison (`hmac.compare_digest`), atomic write via tempfile+fsync, and path-traversal defense.

5. **Security boundaries.** `security/web.py` is sophisticated SSRF defense — `validate_url_for_ssrf()` returns an `SSRFSafeRequest` dataclass with **IP-pinning** to eliminate TOCTOU vulnerabilities in DNS rebinding attacks; `ALWAYS_BLOCKED_RANGES` includes IPv4-mapped IPv6 metadata endpoints (`::ffff:169.254.0.0/112`, the bypass vector). `security/secret_loader.py` provides a 4-implementation `SecretLoader` hierarchy (Cached/Composite/Env/KeyVault) plus `SecretRef` DTO. `security/config_secrets.py` orchestrates two-phase Key Vault loading with mandatory `ELSPETH_FINGERPRINT_KEY` precondition — secrets are fingerprinted before plaintext leaves the loader. `core/secrets.py` (separate from `security/`) provides the **runtime** secret-ref resolver (`resolve_secret_refs`) used by web composer-authored configs to thread `{"secret_ref": "NAME"}` references through the resolver+audit path.

6. **Run-lifecycle support primitives.** `checkpoint/` (5 files, 1,237 LOC) for crash-recovery state with **strict topology equality** (`compute_full_topology_hash` + `checkpoint_node_config_hash` — one_run_id = one configuration invariant). `rate_limit/` (3 files, 470 LOC) wraps `pyrate-limiter` with `NoOpLimiter` parity (Protocol-based, not inheritance — preventing accidental substitution bugs per its docstring) and SQLite persistence. `retention/purge.py` (436 LOC, the largest single file in `retention/`) deletes expired payloads while preserving Landscape hashes (the integrity-after-deletion invariant per `KNOW-C8`). `events.py` is a synchronous in-process EventBus for orchestrator → CLI formatter decoupling (Protocol-based for the same reason — `NullEventBus` does NOT inherit from `EventBus`).

## Layer position and dependency surface

`core/` is **L1** (per `enforce_tier_model.py:238` `"core": 1`, also `KNOW-C47`). It may import only from `{contracts}` (L0). The Phase 0 L3↔L3 oracle filtered to `core/` is **necessarily empty** — `temp/intra-cluster-edges.json` reports `intra_node_count = 0`, `intra_edge_count = 0`, `inbound_cross_cluster_edges = []`, `outbound_cross_cluster_edges = []`. **This is expected and correct, not a gap**: the source oracle filters L3-only and `core/` is L1; its downward edges to `contracts/` are layer-permitted by construction and not graph-enumerated by Phase 0. Per Δ L2-3, the cluster catalog records `core → contracts` outbound at sub-subsystem granularity from per-file imports as they become relevant; cross-cluster oracle data is not the right tool here.

Concrete contracts-of-interest cited by per-file inspection (recorded for the `[CITES]`/cross-cluster-observation channel, not asserted as cross-cluster verdicts here):

- `contracts.payload_store.PayloadStore` (Protocol) + `IntegrityError` + `PayloadNotFoundError` — `core/payload_store.py` implements the Protocol; `core/__init__.py` re-exports `PayloadStore`/`IntegrityError`.
- `contracts.errors.AuditIntegrityError` + `contracts.errors.TIER_1_ERRORS` — used in `checkpoint/manager.py`, `checkpoint/serialization.py`, `core/operations.py`, `retention/purge.py` for Tier-1 audit-integrity exceptions and the `tier_1_error` registry per `KNOW-ADR-010b`.
- `contracts.hashing.CANONICAL_VERSION` — re-exported from `core/canonical.py`; `KNOW-ADR-006b` Phase 2 extracted hashing primitives to `contracts/`.
- `contracts.freeze.{deep_freeze, deep_thaw, freeze_fields, require_int}` — used in `dependency_config.py` for the `deep_freeze` contract (KNOW-C61–C65).
- `contracts.schema.{FieldDefinition, SchemaConfig}` — used by `dag/coalesce_merge.py` for branch-schema merge logic.
- `contracts.schema_contract.PipelineRow` + `contracts.security.{secret_fingerprint, get_fingerprint_key}` + `contracts.secrets.{ResolvedSecret, WebSecretResolver}` + `contracts.SecretResolutionInput` — secret pipeline.
- `contracts.{Checkpoint, ResumeCheck, ResumePoint, RunStatus, BatchPendingError, Operation, Artifact, Batch, ...}` — the dataclass DTO surface for audit recording.

## Layer conformance status

Two artefacts in `temp/`:

- `layer-check-core.txt` — scoped run with the production allowlist (`config/cicd/enforce_tier_model/`).
- `layer-check-core-empty-allowlist.txt` — re-run against an empty allowlist to isolate genuine layer-import findings from already-allowlisted defensive patterns.

The empty-allowlist run reports **205 findings re-surfaced** (rules R5/R6/R1/R4 — `isinstance`, silent except, `dict.get`, broad except). The scoped exit-1 status comes from these allowlist key prefixes not matching when the scope narrows below the allowlist's whole-tree key prefix; it is **not** a layer-import failure. The load-bearing claim — verified by `grep -c '^  Rule: L1'` and `grep -c '^  Rule: TC'` against the empty-allowlist file — is:

> **0 L1 layer-import violations, 0 TC TYPE_CHECKING layer warnings inside `core/`.**

Interpretation: `core/` is fully layer-conformant with respect to upward imports. The 205 defensive-pattern findings are all in-cluster code-quality items already governed by per-file allowlists with owner/reason/safety annotations. The 170 R5 (`isinstance`) findings concentrate in `landscape/` and reflect Tier-1 read guards at the Audit DB → Python boundary — legitimate per CLAUDE.md "Tier 1: crash on any anomaly" because the alternative (`row.foo`) would not provide a meaningful error message when the DB returned the wrong shape. The catalog wave inherits these as known and does not need to re-triage them.

## Test surface

`find tests/unit/core -maxdepth 2 -type d` yields:

- `tests/unit/core/`
- `tests/unit/core/checkpoint/` (5 test files: compatibility, manager, recovery, serialization, version_validation)
- `tests/unit/core/dag/` (6 test files: builder_validation, failsink_edges, graph, graph_validation, models_post_init, output_schema_enforcement)
- `tests/unit/core/landscape/` (≥20 test files including database_compatibility_guards, database_sqlcipher, data_flow_nan_rejection, exporter, factory, formatters, lineage, journal, model_loaders, models_enums, query_repository, reproducibility, run_lifecycle_repository, schema_contracts_audit, …)
- `tests/unit/core/rate_limit/`
- `tests/unit/core/retention/`
- `tests/unit/core/security/`

Plus `tests/unit/core/test_canonical_mutation_gaps.py` at the cluster root and `tests/integration/core/dag/` at integration depth.

Test-to-source coverage at sub-subsystem granularity is dense — every sub-package has its own test directory mirroring the module structure. Counting `find tests/unit/core -name 'test_*.py' | wc -l` is deferred to the validator; eyeballing `landscape/` alone yields ≥20 test files mapped 1:1 onto the 18 landscape source files (some files have multiple test files for distinct concerns — e.g. `test_database_compatibility_guards.py` and `test_database_ops.py` both target `landscape/database.py` aspects).

**Standing note (`KNOW-C44` / CLAUDE.md "Critical Implementation Patterns"):** integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`. The catalog wave should spot-check whether `dag/` tests honour this (the `dag/` builder is the production graph constructor, so any test that bypasses it is using a non-production code path).

## Open questions for the catalog wave

1. **Contracts/core boundary post-ADR-006 (L1 open question Q1).** The catalog records `core → contracts` imports observed at L2 depth (listed under "Layer position" above). Whether `secrets.py` should logically live under `security/` (alongside `secret_loader.py`/`config_secrets.py`/`web.py`) is an open structural question; its current root-level placement appears intentional — it is the runtime resolver consumed by the web composer's secret-ref pipeline, not the secret-loader hierarchy used at config-load time. Catalog will surface but defer.
2. **`config.py` 2,227 LOC cohesion (deep-dive flag).** Single Pydantic settings file holding 12+ dataclasses plus loader logic. Whether this is essential complexity (Pydantic settings tend to concentrate in a single file for cross-validation) or accidental sprawl is the central deep-dive question for `core/`. The catalog records the surface (re-export count, settings family) and flags the file; internals out of scope.
3. **`dag/graph.py` 1,968 LOC and the cascade-prone framing.** `graph.py` owns `ExecutionGraph` — the runtime DAG type used by every executor in `engine/`. §7 P3 frames this file as cascade-prone because changes here propagate to every plugin via the schema-contract validation flow. The catalog records the file's existence and `dag/__init__.py`'s `ExecutionGraph` re-export; internals out of scope.
4. **Landscape repository concentration.** `execution_repository.py` (1,750 LOC) and `data_flow_repository.py` (1,590 LOC) are 2 of the 4 repositories. Are these repositories cohesive (one concern each) or have they become god-classes? `KNOW-A31` claims `ExecutionRepository` is the largest at ~1,480 LOC — current verified figure is 1,750, drift is real but within reason. Catalog records and defers.
5. **20 vs 21 tables (KNOW-A24 drift).** `landscape/schema.py` defines 20 tables; KNOW-A24 says 21. Either the docs are stale by one or one was renamed/dropped — recorded as `[DIVERGES FROM KNOW-A24]` for the doc-correctness pass.
6. **`_validate_run_lifecycle_rows` Tier-1 invariant locus.** `retention/purge.py` enforces the Tier-1 invariant that `RunStatus.RUNNING` implies `completed_at is None` and `RunStatus.COMPLETED` implies `completed_at is not None` before any purge query (lines 75–97). This is one of the strongest Tier-1 read-guard examples in `core/` — the catalog should locate the corresponding test that locks it in.

## L1 cross-references

Supplements `02-l1-subsystem-map.md` §2 (core/) — adds sub-subsystem inventory, public-surface enumeration, deep-dive-candidate confirmation (4 files matching L1's flag list), and the 20-vs-21-tables divergence.

Supplements `04-l1-summary.md` §7 P3 — confirms the cascade-prone framing of `core/` (4 deep-dive candidates spanning config, DAG, and the two heaviest Landscape repositories) and ratifies `core/` as the L2 P3 cluster.

Closes — at L2 depth, but not the broader Q1 question — the structural side of L1 open question Q1: `core/` outbound imports from `contracts/` are layer-clean (0 L1, 0 TC) and concentrated in expected primitive surfaces (`payload_store`, `errors`, `freeze`, `hashing`, `schema`, `schema_contract`, `secrets`, `security`). The semantic question (is the responsibility cut correct?) requires reading both clusters and is a post-L2 synthesis concern.

Standing note from §7.5: ~97% of L3 edges are unconditional runtime coupling. The `core/` cluster is L1, so the standing note applies to `core/`'s L3 callers (every L3 surface inbound) when they're analysed in their own clusters; it does not relax suspicion of conditional/TYPE_CHECKING coupling **inside** `core/`, which is governed instead by the layer-check artefacts above (0 TC warnings).
