# 02 — core/ cluster sub-subsystem catalog

Per Δ L2-3, one entry per sub-subsystem (immediate subdirectory or coherent file group) and **nothing deeper**. Files >1,500 LOC are flagged as L3 deep-dive candidates and not opened. Each entry budgets 300–500 words. Internal coupling cites `temp/intra-cluster-edges.json` (empty by design — see Δ L2-2 in `00-cluster-coordination.md`); external coupling cites the L3 oracle path and the layer model. Per Δ L2-4, no claims cross the `core/` boundary except via the deferral channel in `04-cluster-report.md`.

---

## Entry 1 — landscape/

**Path:** `src/elspeth/core/landscape/`
**Responsibility:** The audit backbone — recorder facade, four domain repositories, schema definitions for 20 audit tables, lineage explainer, exporter (CSV/JSON/text), and Tier-1 read guards at the SQLAlchemy ↔ Python boundary.
**File count, LOC:** 18 files, 9,384 LOC (~45% of cluster).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json` (oracle is L3-only; `core/` is L1). Per-file imports observed: `landscape/__init__.py` re-exports from all 18 internal files; `database.py` is the foundation (other landscape files import its `LandscapeDB`); `factory.py` (`RecorderFactory`) constructs and returns instances of the 4 repositories; `schema.py` (598 LOC) is the table-definition module imported by every repository and by external `core/` modules that touch the DB (`checkpoint/manager.py`, `retention/purge.py`).
**External coupling (cross-cluster):** Inbound from `engine/` (per `KNOW-A29`/`KNOW-A30` — `RecorderFactory` is the engine's audit surface), `cli` (`elspeth explain` lineage queries via `LineageResult`/`explain`), `mcp/` (`elspeth-mcp` analyser is read-only against this DB per `KNOW-C35`), `tui/` (audit-trail TUI per `KNOW-C34`), `web/execution/` (per L3 oracle confirms `web/execution → web/composer` and `web/execution → web/sessions`; the indirect `web/execution → core/landscape` is layer-permitted but not graph-enumerated since `core/` is L1). Outbound to `contracts/` only — every model class (`Run`, `Node`, `Token`, `Artifact`, `Batch`, `RoutingEvent`, `NodeState`, `ContractAuditRecord`, `RowLineage`, etc.) is re-exported through `landscape/__init__.py` from `contracts/`, confirming `KNOW-A53`'s leaf invariant.
**Patterns observed:**
- **Facade pattern:** `RecorderFactory` (`factory.py:338` LOC) constructs the 4 repositories from `LandscapeDB` per `KNOW-A29`. Public callers go through the facade — the 4 repository classes are NOT re-exported through `core/__init__.py` (only through `core/landscape/__init__.py`).
- **Composite PK denormalisation per `KNOW-A33`:** `node_states.run_id` is denormalised so queries don't need to join `nodes (node_id, run_id)` ambiguously. Visible in `retention/purge.py:212–223` (joins use `node_states.run_id` directly, not `nodes`).
- **NaN/Infinity rejection at write boundary:** confirmed in `tests/unit/core/landscape/test_data_flow_nan_rejection.py` — locks the Tier-1 invariant that NaN/Infinity never enters the audit DB.
- **SQLCipher support:** `tests/unit/core/landscape/test_database_sqlcipher.py` — locks the encryption-at-rest surface per `KNOW-A4`/`KNOW-A49`.
**Concerns:**
- Two L3 deep-dive candidates: `execution_repository.py` (1,750 LOC; KNOW-A31 said ~1,480 — drift is real) and `data_flow_repository.py` (1,590 LOC). Combined: 3,340 LOC across 2 files = 36% of the landscape sub-area in 2 files.
- `landscape/schema.py` has **20** `Table` definitions (counted via grep `^[a-z_]+_table = Table`); `KNOW-A24` claims **21**. Recorded as `[DIVERGES FROM KNOW-A24]` for doc-correctness pass.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2 — confirms the 4-repository architecture (`KNOW-A29`/`KNOW-A30`) and refines KNOW-A31's `ExecutionRepository` LOC estimate (1,480 → 1,750). Closes the L1 framing of `landscape/` as the largest core sub-area.
**[CITES KNOW-A4, KNOW-A24, KNOW-A29, KNOW-A30, KNOW-A31, KNOW-A33, KNOW-A49, KNOW-A53, KNOW-C9, KNOW-C11, KNOW-C12, KNOW-C32, KNOW-C35, KNOW-C36]; [DIVERGES FROM KNOW-A24]** (20 tables, not 21); **[DIVERGES FROM KNOW-A31]** (`ExecutionRepository` 1,750 LOC, not ~1,480).
**Confidence:** **High** — public surface verified by `__init__.py` reading; LOC verified by `wc -l`; table count verified by grep; tests cited verify Tier-1 invariants. The 1,500+ LOC files are not opened; deep claims about their internals are correctly deferred.

---

## Entry 2 — dag/

**Path:** `src/elspeth/core/dag/`
**Responsibility:** Execution-graph construction, validation, and coalesce-merge logic — the runtime DAG type used by every executor in `engine/`. Owns the schema-contract two-phase validation per `KNOW-A55`–`KNOW-A57`.
**File count, LOC:** 5 files, 3,549 LOC (~17% of cluster).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. Per-file imports observed: `dag/__init__.py` re-exports `ExecutionGraph` (from `graph.py`), `GraphValidationError`/`GraphValidationWarning`/`NodeConfig`/`NodeInfo`/`WiredTransform` (from `models.py`), and `merge_guaranteed_fields`/`merge_union_fields` (from `coalesce_merge.py`). `models.py` (242 LOC) is the leaf — imports `contracts.enums.NodeType`, `contracts.freeze.freeze_fields`, `contracts.schema.SchemaConfig`, `contracts.types.{CoalesceName, NodeID}`, plus `core.landscape.schema.NODE_ID_COLUMN_LENGTH` (a single constant for `node_id` length validation — confirms the 64-char node-ID cap is shared with the audit DB schema). `coalesce_merge.py` (243 LOC) imports `dag/models.GraphValidationError` and `contracts.schema.{FieldDefinition, SchemaConfig}`. `builder.py` (1,071 LOC) is the production entry point for graph construction (called by `engine/orchestrator/`); `graph.py` (1,968 LOC, deep-dive flag) houses `ExecutionGraph` itself.
**External coupling (cross-cluster):** Inbound from `engine/orchestrator/` (the canonical entry per `KNOW-C44`: `ExecutionGraph.from_plugin_instances()`), `engine/executors/` (every executor reads `NodeInfo` for its `output_schema_config`/`passes_through_input` flag), `web/execution/` (validation per L3 oracle showing `web/execution → web/composer → ...` chain ultimately reaching DAG construction). Outbound to `contracts/` only.
**Patterns observed:**
- **Two-phase schema validation at DAG construction time per `KNOW-A55`–`KNOW-A57`:** Phase 1 verifies upstream `guaranteed_fields ⊇ downstream required_fields`; Phase 2 verifies field types are compatible. Failures crash immediately. The validator walk lives in `builder.py`/`graph.py` (deep-dive deferred).
- **Coalesce merge policy per `KNOW-A47`–`KNOW-A48`:** `coalesce_merge.py:28` `merge_guaranteed_fields(branch_schemas, *, require_all)` — union semantics for `require_all=True`, intersection otherwise. `merge_union_fields` (lines 70–243) handles per-field collisions with `first_wins`/`last_wins`/`fail` policies and propagates the partial-arrival nullable invariant correctly (line 209 comment is a P1 soundness fix).
- **Pass-through contract propagation per `KNOW-ADR-007`/`009`/`010`:** `NodeInfo.passes_through_input` (`models.py:149`) is the contract flag; `models.py:187–193` rejects non-False on non-TRANSFORM/AGGREGATION nodes with `GraphValidationError` (offensive programming).
- **Deterministic node IDs per `KNOW-ADR-005c`:** comment in `models.py:104` references `elspeth-c3a98c358c` and the position-independent `{prefix}_{settings_name}_{config_hash}` derivation.
- **`GraphValidationError` carries structured `component_id`/`component_type`** (`models.py:24–48`) so callers don't need to regex-parse `str(exc)` — supports the offensive-programming rule (`KNOW-C60`).
**Concerns:**
- One L3 deep-dive candidate: `graph.py` (1,968 LOC). The cascade-prone framing in §7 P3 lands here.
- `builder.py` at 1,071 LOC is below the deep-dive threshold but is the most behaviour-dense module in `dag/` — the L3 deep-dive candidate `graph.py` may turn out to be `builder.py`'s passive partner (graph data + add_node/add_edge mechanics) while `builder.py` does the orchestration.
**Tests cited (Δ L2-5):**
- `tests/unit/core/dag/test_models_post_init.py` locks `NodeInfo.__post_init__` invariants (component_id/component_type capture, length cap, declared_required_fields-sink-only, passes_through_input-transform-or-aggregation-only).
- `tests/unit/core/dag/test_builder_validation.py` and `test_graph_validation.py` lock the Phase 1/Phase 2 schema validation behaviour.
- `tests/unit/core/dag/test_failsink_edges.py` locks the explicit-sink-routing invariant from `KNOW-ADR-004`.
- `tests/integration/core/dag/` — at integration depth (verified to exist; per-file content not analysed by this pass).
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2; confirms `KNOW-A55`–`KNOW-A57` Phase-1/2 framing and refines KNOW-ADR-005c node-ID derivation specifics.
**[CITES KNOW-A55, KNOW-A56, KNOW-A57, KNOW-A47, KNOW-A48, KNOW-C28, KNOW-C44, KNOW-C60, KNOW-ADR-003, KNOW-ADR-005c, KNOW-ADR-007]**.
**Confidence:** **High** for the public surface and validation-pattern attributions (verified from `models.py` and `coalesce_merge.py`); **Medium** for the cascade-prone framing claim about `graph.py` (deferred to deep-dive).

---

## Entry 3 — checkpoint/

**Path:** `src/elspeth/core/checkpoint/`
**Responsibility:** Crash-recovery state — checkpoint creation, retrieval, deletion, compatibility validation against current graph topology, and type-preserving JSON serialisation for aggregation/coalesce state.
**File count, LOC:** 5 files, 1,237 LOC.
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. Per-file imports observed: `manager.py` (287 LOC) imports `checkpoint/serialization.checkpoint_dumps`, `core/canonical.{compute_full_topology_hash, stable_hash}`, `core/landscape/database.LandscapeDB`, and `core/landscape/schema.{checkpoints_table, tokens_table}`. `compatibility.py` (116 LOC) imports `core/canonical.{compute_full_topology_hash, stable_hash}` and `core/dag.ExecutionGraph`. `recovery.py` (562 LOC) is the largest file in the sub-package — owns "can this run be resumed?" logic. `serialization.py` (244 LOC) handles round-trip-safe checkpoint JSON with collision-safe envelope keys.
**External coupling (cross-cluster):** Inbound from `engine/orchestrator/` (calls `CheckpointManager.create_checkpoint` at row boundaries), `cli` (`elspeth resume` per `KNOW-C34` invokes `RecoveryManager`). Outbound to `contracts/` for `Checkpoint`, `ResumeCheck`, `ResumePoint`, `AuditIntegrityError`, and the `aggregation_checkpoint`/`coalesce_checkpoint` typed-dict modules.
**Patterns observed:**
- **One run = one configuration invariant** (per `manager.py:27–31` docstring). Enforced via:
  - **Cross-run contamination guard** (`manager.py:94–104`): `CheckpointManager.create_checkpoint` verifies the token belongs to the specified run via a `tokens` lookup before accepting; mismatch raises `AuditIntegrityError("Cross-run checkpoint contamination is audit corruption")`.
  - **Full-DAG topology hash** (`canonical.compute_full_topology_hash`): `compatibility.py:71–75` rejects resume if ANY node or edge changed (not just upstream). Comment at `compatibility.py:24–30` explains this changed from upstream-only after multi-sink DAGs allowed sibling-branch changes to go undetected.
  - **Format version equality** (`manager.py:280–287`): Both older AND newer format versions are rejected; cross-version resume is unsupported.
- **TOCTOU-safe transactional pattern:** `manager.py:93` opens a transaction, performs token lookup + topology hashing + insert atomically; `begin()` auto-commits on clean exit, auto-rollbacks on exception.
- **Type-preserving JSON envelopes** (`serialization.py:43–80`): `{"__elspeth_type__": "datetime", "__elspeth_value__": iso_string}` instead of shape-based `{"__datetime__": iso_string}` — collision-safe against user dicts. `_escape_reserved_keys` recursively escapes user dicts that coincidentally contain the reserved key. Tuples are also envelope-typed (`serialization.py:124–130`).
- **Strict NaN/Infinity rejection in checkpoint JSON** (`serialization.py:82–105`): same audit-integrity rule as `canonical.py`.
- **Tier-1 surfacing on corruption** (`manager.py:196–199`): `Checkpoint.__init__` failures are wrapped as `CheckpointCorruptionError` with the original `ValueError` chained via `from e` (per `KNOW-C60`).
**Concerns:** None observed at L2 depth — sub-1500-LOC files only; the largest file (`recovery.py`, 562 LOC) is well within budget.
**Tests cited (Δ L2-5):** `tests/unit/core/checkpoint/test_manager.py` (lifecycle), `test_recovery.py` (resumability), `test_compatibility.py` (topology-hash matching), `test_serialization.py` (round-trip + envelope handling), `test_version_validation.py` (format-version equality).
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2; confirms `KNOW-A20`'s "~600 LOC checkpoint" framing (verified 1,237 — KNOW-A20 was Landscape-conscious; KNOW-A20 stale by ~2x).
**[CITES KNOW-A20, KNOW-C8, KNOW-C42, KNOW-C45, KNOW-C60, KNOW-C61]**; **[DIVERGES FROM KNOW-A20]** (1,237 LOC vs ~600 — drift).
**Confidence:** **High** — every claim verified from `manager.py`, `compatibility.py`, `serialization.py` directly.

---

## Entry 4 — rate_limit/

**Path:** `src/elspeth/core/rate_limit/`
**Responsibility:** Rate-limiting wrapper around `pyrate-limiter` with optional SQLite persistence for cross-process coordination, plus a no-op limiter for the disabled case.
**File count, LOC:** 3 files, 470 LOC (`limiter.py` 326, `registry.py` 135, `__init__.py` 9).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. Per-file imports: `registry.py` imports `limiter.RateLimiter`. `__init__.py` re-exports `RateLimiter`, `NoOpLimiter`, `RateLimitRegistry`.
**External coupling (cross-cluster):** Inbound from `plugins/infrastructure/clients/` (audited HTTP/LLM clients per `KNOW-A39` use rate limiters), `plugins/transforms/llm/` and `plugins/transforms/azure/` (per L3 oracle showing both `→ plugins/infrastructure` heavy edges; the rate-limit registry is part of the runtime services available to plugin clients). Outbound to `contracts/config/protocols.RuntimeRateLimitProtocol`.
**Patterns observed:**
- **Protocol-based no-op parity** (`registry.py:33–66`): `NoOpLimiter` does NOT inherit from `RateLimiter` — both classes implement the same surface (`acquire`, `try_acquire`, `close`, context manager) so structural typing satisfies callers. The docstring at `events.py:88–103` (sister pattern in `EventBus`) explicitly states why: inheritance would hide bugs where callers expected real rate-limiting but got the no-op silently.
- **SQL-injection-safe naming** (`limiter.py:26`, `_VALID_NAME_PATTERN`): rate-limiter names are validated against `^[A-Za-z][A-Za-z0-9_]*$` because they're interpolated into SQL table names (`SQLiteBucket`'s `table_name`). `registry.py:13–26` `_sanitize_limiter_name` adapts arbitrary service names (e.g. hostnames with dots) to the safe form.
- **`pyrate-limiter` cleanup-race suppression** (`limiter.py:40–95`, `_install_hook`/`_custom_excepthook`): pyrate-limiter's Leaker thread can raise `AssertionError` during cleanup. The module installs a custom `threading.excepthook` only while suppressions are pending, deregisters per-thread by ident (not name) to avoid suppressing unrelated threads, and uninstalls when idle. This is one of the few `core/` examples of structurally suppressing a third-party library's known benign exception while preserving safety.
- **Type-strict argument validation** (`limiter.py:191–196` `_validate_weight`): uses `type(weight) is not int` (NOT `isinstance(weight, int)`) — explicit rejection of `bool` (which `isinstance(True, int)` returns True for) per the offensive-programming rule.
- **Lazy-resolved excepthook** (`limiter.py:40–55`): the hook is only installed when there's something to suppress, then uninstalled immediately when no suppressions are pending. Avoids replacing the global `threading.excepthook` at import time.
**Concerns:** None observed at L2 depth.
**Tests cited (Δ L2-5):** `tests/unit/core/rate_limit/` directory exists; per-file content not analysed by this pass — defer to validator.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2; refines `KNOW-A21` (~300 LOC vs verified 470 — drift).
**[CITES KNOW-A21, KNOW-C36, KNOW-C57, KNOW-C60]**; **[DIVERGES FROM KNOW-A21]** (470 LOC vs ~300).
**Confidence:** **High** — all claims verified from source.

---

## Entry 5 — retention/

**Path:** `src/elspeth/core/retention/`
**Responsibility:** PayloadStore retention enforcement — identifies expired payloads from completed runs and deletes blobs while preserving Landscape hashes for audit integrity (the `KNOW-C8` "hashes survive payload deletion" invariant).
**File count, LOC:** 2 files, 445 LOC (`purge.py` 436, `__init__.py` 9).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. `purge.py` imports `core/landscape/reproducibility.update_grade_after_purge` and the schema modules `calls_table`, `node_states_table`, `operations_table`, `routing_events_table`, `rows_table`, `runs_table`.
**External coupling (cross-cluster):** Inbound from `cli` (`elspeth purge --retention-days N` per `KNOW-C34`). Outbound to `contracts/{RunStatus, errors.AuditIntegrityError, errors.TIER_1_ERRORS, payload_store.PayloadStore}`.
**Patterns observed:**
- **Tier-1 read guard before any purge query** (`purge.py:75–97` `_validate_run_lifecycle_rows`): scans all run rows and crashes immediately on any `RunStatus.RUNNING && completed_at is not None` or `RunStatus.COMPLETED && completed_at is None` — these states are impossible per the run lifecycle contract; encountering them in the audit DB indicates corruption. Status enum values that don't parse raise `AuditIntegrityError`. **This is one of the strongest Tier-1 read-guard examples in the cluster.**
- **Active-run safety via UNION + set difference** (`purge.py:225–261`): expired-runs UNION query and active-runs UNION query are computed separately; deleted set = `expired_refs - active_refs`. Comments explain the SQL-EXCEPT-vs-Python-set-difference trade-off (SQLite EXCEPT performance + result-set size).
- **Composite-PK denormalisation discipline** (`purge.py:209–223`): every join uses `node_states.run_id` directly per `KNOW-A33` — never joins through `nodes` (which has composite PK and would be ambiguous when `node_id` is reused across runs). Comment at line 211 documents this explicitly.
- **Bind-variable-limit chunking** (`purge.py:265, 289–291`): `_PURGE_CHUNK_SIZE = 100` keeps `IN`-clause chunks well under SQLite's `SQLITE_MAX_VARIABLE_NUMBER` (default 999) given 8 sub-queries per chunk.
- **Hashes survive deletion + reproducibility-grade downgrade** (`purge.py:357–417`): for each successfully deleted ref, `update_grade_after_purge` downgrades affected runs from `REPLAY_REPRODUCIBLE` → `ATTRIBUTABLE_ONLY`. Failures don't crash — they're tracked in `PurgeResult.grade_update_failures` for operator visibility (each grade update wrapped because payloads are already irreversibly deleted; one failure must not block remaining grade updates). **Tier-1 errors propagate** (`purge.py:416–417`).
- **`PurgeResult` is `@dataclass(frozen=True, slots=True)`** with `__post_init__` validation (negative-counts rejected) per `KNOW-C61`.
**Concerns:** None observed at L2 depth.
**Tests cited (Δ L2-5):** `tests/unit/core/retention/` directory exists; per-file content not analysed by this pass.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2 (which named `retention/` only by category in the sub-area string — this pass identifies the specific Tier-1 invariants enforced).
**[CITES KNOW-A33, KNOW-C8, KNOW-C9, KNOW-C11, KNOW-C12, KNOW-C61]**.
**Confidence:** **High** — all claims verified from `purge.py:1–437`.

---

## Entry 6 — security/

**Path:** `src/elspeth/core/security/`
**Responsibility:** Security boundaries for ELSPETH — SSRF prevention with TOCTOU-safe DNS resolution, secret loader hierarchy (Cached/Composite/Env/KeyVault), `SecretRef` DTO, and config-time Key Vault loading with mandatory fingerprint key.
**File count, LOC:** 4 files, 940 LOC (`web.py` 355, `secret_loader.py` 317, `config_secrets.py` 212, `__init__.py` 56).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. Per-file imports: `__init__.py` re-exports from all 3 internal modules plus the `contracts.security.{secret_fingerprint, get_fingerprint_key}` primitives. `config_secrets.py` lazy-imports `secret_loader.{KeyVaultSecretLoader, SecretNotFoundError}` and the Azure SDK exception types.
**External coupling (cross-cluster):** Inbound from `plugins/transforms/llm/`, `plugins/sources/`, `plugins/sinks/` (any plugin that constructs HTTP-based clients), `web/` (the web composer threads `{"secret_ref": ...}` references through `core/secrets.resolve_secret_refs` — sister to but distinct from the `security/` loaders here). Outbound to `contracts/security`.
**Patterns observed:**
- **TOCTOU-safe SSRF defence via IP pinning** (`web.py:183–235`, `SSRFSafeRequest` dataclass): traditional SSRF defences resolve DNS twice (validate-then-fetch), allowing DNS rebinding attacks. `validate_url_for_ssrf()` resolves once, validates ALL resolved IPs, returns an `SSRFSafeRequest` with `connection_url` (hostname → IP), `host_header` (preserves vhost), and `sni_hostname` (TLS SNI). Caller connects to the pinned IP.
- **Three-tier IP check order** (`web.py:138–180`): (1) `ALWAYS_BLOCKED_RANGES` (unconditional, no allowlist bypass) → (2) `allowed_ranges` (operator-supplied bypass) → (3) `BLOCKED_IP_RANGES` (standard blocklist). Comment at `web.py:65–75` explains why `::ffff:169.254.0.0/112` (IPv4-mapped IPv6 metadata endpoint) is in `ALWAYS_BLOCKED_RANGES` while `::ffff:0:0/96` (the broader IPv4-mapped IPv6 family) is in `BLOCKED_IP_RANGES` — operator-allowed `::ffff:10.x.x.x` should work, but the metadata endpoint must never be reachable.
- **Bounded DNS thread pool** (`web.py:86–88`): `_dns_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dns_resolve")`. Under repeated timeouts (e.g. blackholed resolver), at most 8 threads are blocked; further requests queue rather than spawning unbounded threads.
- **Fail-closed unparseable IPs** (`web.py:158–163`): zone-scoped IPv6 (`fe80::1%eth0`) raises `SSRFBlockedError` rather than allowing the request through.
- **Two-phase Key Vault loading with atomic apply** (`config_secrets.py:140–210`): Phase 1 fetches all secrets and computes fingerprints WITHOUT mutating `os.environ`; Phase 2 applies all env vars only if Phase 1 succeeded. Prevents partial-state leak on failure.
- **`ELSPETH_FINGERPRINT_KEY` precondition** (`config_secrets.py:78–97`): preflight check before any Key Vault call — without the key, fingerprints can't be computed and audit recording would silently lose secret-resolution events. Either env var or in-mapping presence satisfies the check.
- **Plaintext discipline:** `config_secrets.py:117–122` comment — "plaintext values are fingerprinted and discarded within this loop iteration — they never accumulate in the resolutions list". The `SecretResolutionInput` DTO carries fingerprint only.
**Concerns:**
- The split between `core/secrets.py` (runtime resolver) and `core/security/{secret_loader,config_secrets}.py` (config-time loaders) is intentional but undocumented. A reader landing on `secrets.py` may not know `security/` exists. Consider catalog-level cross-reference (this is a doc-correctness item, not a code change).
**Tests cited (Δ L2-5):** `tests/unit/core/security/` directory exists; per-file content not analysed by this pass — security testing is high-stakes and a deeper L3 deep-dive on `web.py` SSRF defence is justified for security-architect review.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2; the L1 entry mentioned `security/` only in the sub-area summary line — this pass identifies the SSRF-defence sophistication and the SecretLoader hierarchy.
**[CITES KNOW-C8 (audit hashes), KNOW-C36 (tech stack), KNOW-A24 (audit DB context), KNOW-G9 (composer-specific opt-in)]**.
**Confidence:** **High** for `web.py` SSRF and `config_secrets.py` Key Vault patterns (verified directly); **Medium** for `secret_loader.py`'s 4-implementation hierarchy (read only the `__init__.py` re-export list — file internals not opened).

---

## Entry 7 — configuration_family (config.py + dependency_config.py + secrets.py)

**Path:** `src/elspeth/core/{config.py, dependency_config.py, secrets.py}` (file group)
**Responsibility:** Configuration loading and runtime resolution — `ElspethSettings` Pydantic root + 11+ child dataclasses (`config.py`); cross-pipeline dependencies and commencement gates with config-time expression validation (`dependency_config.py`); runtime `{"secret_ref": "NAME"}` resolver for web composer-authored configs (`secrets.py`).
**File count, LOC:** 3 files, 2,524 LOC (`config.py` 2,227 — flag, `dependency_config.py` 173, `secrets.py` 124).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. Per-file imports: `dependency_config.py` lazy-imports `core.expression_parser.ExpressionParser` (line 63) for config-time gate-expression validation. `secrets.py` is leaf-of-cluster — only imports `contracts.secrets.{ResolvedSecret, WebSecretResolver}`.
**External coupling (cross-cluster):** Inbound from every layer above (engine, plugins, web, mcp, composer_mcp, telemetry, tui, cli) — `ElspethSettings` is the system's configuration root. Outbound to `contracts/freeze` (for `deep_freeze`/`deep_thaw`/`freeze_fields`/`require_int`), `contracts/config/*`, `contracts/secrets`, `contracts/SecretResolutionInput`.
**Patterns observed:**
- **Pydantic v2 `frozen=True` + `extra="forbid"`** (`dependency_config.py:25, 44, 77`): every config dataclass is frozen and rejects unknown keys. `model_validator(mode="after")` deep-freezes mutable fields like `provider_config` per `KNOW-C61`–`KNOW-C65`; `field_serializer` thaws them back to dict for Pydantic JSON serialisation.
- **Config-time expression parsing** (`dependency_config.py:54–67`): `CommencementGateConfig.condition` is parsed by `ExpressionParser` at config-load time (with `allowed_names=["collections", "dependency_runs", "env"]`). Syntax/security errors surface at config load, not at gate evaluation. Lazy import of `ExpressionParser` avoids a `core/dependency_config → core/expression_parser` import order dependency at module-import time.
- **Result-not-failure encoding** (`dependency_config.py:131–155`): `CommencementGateResult.result` is always `True` — gate failures raise `CommencementGateFailedError` instead of returning `result=False`. The field exists so the audit trail records an explicit pass verdict, not just the absence of a failure. The `__post_init__` rejects `result=False` with a clear error message — offensive-programming pattern per `KNOW-C60`.
- **Runtime secret-ref resolution** (`secrets.py:27–53`, `resolve_secret_refs`): walks a config dict tree replacing `{"secret_ref": "NAME"}` with resolved values via a `WebSecretResolver`. Returns `(resolved_config, list_of_resolutions)` and raises `SecretResolutionError` listing **all** missing refs (not one at a time — better operator UX). The original config is not mutated (deepcopy pattern).
- **Exact-string env-var ref opt-in** (`secrets.py:65–80`): `${NAME}` strings are also treated as secret refs when `NAME` is in an explicit `env_ref_names` allowlist; embedded `prefix-${NAME}` is intentionally left alone. This routes web-authored exact references through the resolver/audit path instead of blind config env-var expansion.
**Concerns:**
- One L3 deep-dive candidate: `config.py` (2,227 LOC). Single Pydantic settings file holding 12+ dataclasses plus loader logic. Per Δ L2-3, internals out of scope for this pass.
- `secrets.py` topical placement (root) vs `security/{secret_loader,config_secrets}.py` (subpackage): they have related vocabulary but disjoint roles (runtime resolver vs config-time loaders). Open structural question for the post-L2 synthesis.
**Tests cited (Δ L2-5):** Direct `tests/unit/core/test_config.py` (or similar) was not located via the directory listing in this pass; defer to validator. `tests/unit/core/security/` covers the security/ side.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2; confirms `config.py` as the largest standalone file in `core/` and locates `secrets.py` distinctly from `security/`. Refines `KNOW-A22`'s "Core ~5,000 LOC: Config, canonical JSON, DAG package, payload store" framing — this pass confirms the L1 explanation that `KNOW-A22` excluded Landscape (KNOW-A22 + KNOW-A17 sums to ~13.3k, near-but-still-below verified 20.8k).
**[CITES KNOW-A22, KNOW-A27, KNOW-C45, KNOW-C57, KNOW-C60, KNOW-C61, KNOW-C62, KNOW-C63, KNOW-G9, KNOW-ADR-006b]**.
**Confidence:** **High** for `dependency_config.py` + `secrets.py` (verified directly); **Medium** for `config.py` (deep-dive deferred — claims about its responsibility are derived from its `__init__.py` re-exports and the L1 entry, not from reading `config.py` itself).

---

## Entry 8 — canonicalisation_and_templating (canonical.py + templates.py + expression_parser.py)

**Path:** `src/elspeth/core/{canonical.py, templates.py, expression_parser.py}` (file group)
**Responsibility:** Determinism and template-side primitives — RFC 8785/JCS canonical JSON with strict NaN/Infinity rejection (`canonical.py`), Jinja2 field-extraction development helper (`templates.py`), and the secured expression parser (`expression_parser.py`) used by config-time gate validation, runtime gates in `engine/`, and the composer.
**File count, LOC:** 3 files, 1,422 LOC (`canonical.py` 309, `templates.py` 293, `expression_parser.py` 820).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. Per-file imports: `canonical.py` re-exports `CANONICAL_VERSION` from `contracts.hashing` (per `KNOW-ADR-006b` Phase 2 extraction) and TYPE_CHECKING-imports `core.dag.ExecutionGraph` for `compute_full_topology_hash`. `templates.py` TYPE_CHECKING-imports `contracts.schema_contract.SchemaContract` for the contract-aware variant. `expression_parser.py` (deep-dive deferred — 820 LOC is below the 1,500 threshold but is a security-critical file worth a dedicated read) — at L2 depth, only its public surface is recorded: `ExpressionParser`, `ExpressionEvaluationError`, `ExpressionSecurityError`, `ExpressionSyntaxError`.
**External coupling (cross-cluster):** Inbound — `canonical.py` is used by `engine/orchestrator/`, `engine/executors/`, `core/checkpoint/{manager,compatibility}`, `core/dag/`, every plugin that emits hashed payloads (e.g. `plugins/sinks/azure_blob_sink.py`'s `content_hash`). `expression_parser.py` is used by `engine/` gate executors, `core/dependency_config.py`, `web/composer/` (per `KNOW-A27` + `KNOW-ADR-006b`). `templates.py` is a development helper — `engine/` does not auto-extract at runtime per the `templates.py:14–18` docstring. Outbound to `contracts/{hashing, schema_contract, schema}`.
**Patterns observed:**
- **Two-phase canonical JSON** (`canonical.py:1–17` docstring + `canonical_json:180–199`): Phase 1 normalises pandas/numpy types to JSON-safe primitives (our code in `_normalize_value`/`_normalize_for_canonical`); Phase 2 serialises per RFC 8785/JCS via `rfc8785.dumps()`.
- **Strict NaN/Infinity rejection in audit data** (`canonical.py:60–101`): NaN and Infinity are **not coerced to None** — they raise `ValueError` immediately. This is defense-in-depth for audit integrity per `KNOW-C8`. The exception is `sanitize_for_canonical` (`canonical.py:286–309`), which IS allowed to coerce NaN/Infinity → None at the **Tier 3 quarantine boundary** (sources may coerce per `KNOW-C19`); the quarantine error message records what was originally wrong, preserving auditability.
- **`np.longdouble` finiteness** (`canonical.py:65–73`): uses `np.isfinite(obj)` (NumPy-native) rather than `math.isnan(float(obj))` because `float()` downcast overflows for `np.longdouble` values outside IEEE 754 double range, falsely treating finite values as inf. Subtle correctness fix worth noting.
- **`compute_full_topology_hash`** (`canonical.py:215–252`): hashes the FULL DAG (every node + every edge with its mode/label) for checkpoint validation, supporting the "one_run_id = one configuration" invariant. The edge serialiser uses **explicit defaults** for `label`/`mode` and treats edge data as Tier 1 — crashes on missing attributes (`canonical.py:275–283`). Visible enforcement of `KNOW-ADR-002`'s ROUTE/MOVE-only mode rule (mode ends up in the hash, so any change invalidates checkpoints).
- **Jinja2 field extraction is explicit, not auto-runtime** (`templates.py:14–28` docstring): `extract_jinja2_fields()` is a development helper; developers must add the discovered fields to `required_input_fields` manually. Auto-extraction at runtime would violate auditability (KNOW-C7) by silently inferring contract content from templates. Bracket syntax (`row["Original Name"]`) returns names verbatim — the contract-aware variant `extract_jinja2_fields_with_names()` resolves original→normalised via a `SchemaContract`.
- **PipelineRow API name exclusion** (`templates.py:100–106`): the `_PIPELINE_ROW_API_NAMES` frozenset (`get`, `contract`, `to_dict`) is the **closed list** of names that can never be valid data field names. Comment explicitly notes that `keys`/`items`/`values` are NOT excluded because they can be legitimate column names.
**Concerns:**
- `expression_parser.py` (820 LOC) is below the deep-dive threshold but is security-sensitive (`ExpressionSecurityError` is part of its surface). A future security-architect review may want a focused read.
**Tests cited (Δ L2-5):** `tests/unit/core/test_canonical_mutation_gaps.py` exists at the cluster-test root — locks behaviour of `canonical.py`'s mutation/normalisation gaps; specific assertions not analysed by this pass.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2; verifies `KNOW-A27`'s "ExpressionParser at `core/expression_parser.py` (~652 LOC)" — current verified 820 LOC, drift ~25%, but the file location is correct (`KNOW-ADR-006b` Phase 1 relocation confirmed).
**[CITES KNOW-A27, KNOW-A58, KNOW-A69 (RFC 8785), KNOW-C7, KNOW-C8, KNOW-C16, KNOW-C19, KNOW-ADR-002, KNOW-ADR-006b]**; **[DIVERGES FROM KNOW-A27]** (820 LOC vs ~652 — drift).
**Confidence:** **High** for `canonical.py` and `templates.py` (verified directly); **Medium** for `expression_parser.py` (only public surface recorded; deferred for a focused review).

---

## Entry 9 — cross_cutting_primitives (events.py + identifiers.py + logging.py + operations.py + payload_store.py + __init__.py)

**Path:** `src/elspeth/core/{events.py, identifiers.py, logging.py, operations.py, payload_store.py, __init__.py}` (file group)
**Responsibility:** Cross-cutting primitives that don't fit the other entries — synchronous EventBus for orchestrator → CLI formatter decoupling, identifier validation, structured logging configuration (structlog + stdlib via `ProcessorFormatter`), source/sink operation lifecycle context manager, content-addressable filesystem payload store, and the cluster's public-surface re-export module.
**File count, LOC:** 6 files, 720 LOC (`operations.py` 208, `logging.py` 185, `payload_store.py` 183, `events.py` 111, `__init__.py` 100, `identifiers.py` 33).
**Internal coupling:** Empty per `temp/intra-cluster-edges.json`. `__init__.py` imports from every other sub-area (canonical, checkpoint, config, dag, events, expression_parser, logging, payload_store) — it is the cluster's public surface. `operations.py` TYPE_CHECKING-imports `core.landscape.execution_repository.ExecutionRepository`. `payload_store.py` imports `contracts.payload_store.{IntegrityError, PayloadNotFoundError}`.
**External coupling (cross-cluster):** Inbound from every L2+ surface. `EventBus` is consumed by the CLI/TUI per the orchestrator → formatter decoupling story. `track_operation` is consumed by `engine/orchestrator/` and `engine/executors/sink.py`. `FilesystemPayloadStore` is the production payload store (per `KNOW-A49` deployment view). `configure_logging`/`get_logger` is consumed by every L3 surface. Outbound to `contracts/{Operation, plugin_context.PluginContext, BatchPendingError, errors.TIER_1_ERRORS, payload_store}`.
**Patterns observed:**
- **Protocol-based no-op parity** (`events.py:14–28, 88–111`): same pattern as `rate_limit/registry.py:33–66`. `NullEventBus` does NOT inherit from `EventBus` — both implement `EventBusProtocol` structurally. The docstring explicitly says: "If someone subscribes expecting callbacks, inheritance would hide the bug. Protocol-based design makes the no-op behavior explicit." This is a recurring `core/` design rule worth surfacing at the cluster level.
- **Operation context manager with audit-completion guarantee** (`operations.py:74–209`, `track_operation`): wraps source/sink I/O in begin/complete operation lifecycle; **audit integrity rule** (`operations.py:182–204`): if `complete_operation` fails because of a DB error, the run MUST fail — silently dropping an audit record violates Tier-1 trust. Tier-1 errors propagate regardless of any original exception (`operations.py:199–200`); other DB errors propagate only if the original op succeeded (otherwise the original exception propagates and the DB error is logged).
- **`BatchPendingError` is control flow, not failure** (`operations.py:152–157`): batch-aware transforms that need to wait for async results raise `BatchPendingError`; the operation status is recorded as `"pending"` (not `"failed"`) and the exception is re-raised. Distinguishes legitimate async-batch waits from genuine errors.
- **`OperationHandle` slots-protected output_data** (`operations.py:47–72`): only `output_data` is writable; the `operation` field is read-only via `@property` because mutating it would corrupt audit trail linkage. `__slots__ = ("_operation", "output_data")` enforces no-extra-attrs.
- **`FilesystemPayloadStore` content-addressable storage with timing-safe verify** (`payload_store.py:34–183`):
  - Path traversal defense (`payload_store.py:65–86`): explicit `re.fullmatch` (NOT `re.match`, which would accept `\n`-terminated strings) plus runtime path-resolution check that resolved path is under `base_path`.
  - **Atomic write via tempfile + fsync** (`payload_store.py:118–143`): write to `.tmp`, fsync content, `os.replace` atomic rename, fsync parent directory (survives power loss).
  - **Timing-safe hash comparison** (`payload_store.py:111, 163`): `hmac.compare_digest` instead of `==` to prevent timing attacks that could allow attackers to incrementally discover expected hashes.
  - **Integrity-on-store** (`payload_store.py:105–115`): if the file already exists with content matching the hash, the store call is idempotent; if it exists with a mismatched hash, `IntegrityError` is raised — defends against silent corruption.
- **Structured logging routing** (`logging.py:93–172`, `configure_logging`): configures BOTH structlog AND stdlib `logging` to use the same `ProcessorFormatter` chain so `logging.getLogger(__name__)` and `structlog.get_logger()` produce consistent output. ID-set tracking of ELSPETH-owned handlers (`_elspeth_handler_ids`) enables selective removal during reconfiguration without touching pytest caplog handlers. Lazy `_LazyStdoutStream` (`logging.py:23–40`) resolves `sys.stdout` at write time so pytest stdout swaps don't leave stale references.
- **Noisy logger silencing** (`logging.py:47–66, 167–172`): hardcoded list of third-party noisy loggers (Azure SDK, urllib3, OpenTelemetry, httpx) is silenced to `WARNING` even when ELSPETH runs in DEBUG. `noisy_level = max(log_level, logging.WARNING)` — never makes noisy loggers less restrictive than the configured root level.
- **`identifiers.validate_field_names`** (`identifiers.py:13–33`): deduplicates + validates Python identifiers + rejects keywords. Error messages include the offending index. Exists in `core/` to be importable from any subsystem without creating cross-subsystem imports (per its docstring) — the kind of leaf utility that should be cited as evidence for the L1 layering discipline.
**Concerns:** None observed at L2 depth.
**Tests cited (Δ L2-5):** Direct unit tests for `payload_store.py` and `operations.py` exist under `tests/unit/core/` but specific files were not enumerated in the pass's directory listings; defer to validator.
**L1 cross-reference:** Supplements `02-l1-subsystem-map.md` §2 — identifies the **recurring "Protocol-based no-op parity" pattern** in `core/` (`EventBus`/`NullEventBus`, `RateLimiter`/`NoOpLimiter`) and the **ID-set-tracking-not-getattr** discipline in `logging.py:70` (cited explicitly as comply with the defensive-programming ban per `KNOW-C57`/`KNOW-C58`).
**[CITES KNOW-A49, KNOW-C7, KNOW-C8, KNOW-C9, KNOW-C11, KNOW-C12, KNOW-C36, KNOW-C38, KNOW-C39, KNOW-C40, KNOW-C42, KNOW-C57, KNOW-C58, KNOW-C60]**.
**Confidence:** **High** — every claim verified from source files directly.
