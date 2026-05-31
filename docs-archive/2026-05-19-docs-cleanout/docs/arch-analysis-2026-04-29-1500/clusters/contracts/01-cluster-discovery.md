# L2 #5 — `contracts/` cluster discovery findings

## File census

The cluster contains **63 Python files totalling 17,403 LOC** under `src/elspeth/contracts/`:

- **Top-level modules:** 58 files at `src/elspeth/contracts/*.py`.
- **Sub-package:** 1 directory (`config/` — 5 files, 1,231 LOC: `__init__.py`, `alignment.py`, `defaults.py`, `protocols.py`, `runtime.py`).
- **`__init__.py` size:** 535 LOC; the public surface re-exports 208 names (counted via `grep -cE '^    "' src/elspeth/contracts/__init__.py`).

Census derived by `find src/elspeth/contracts/ -name "*.py" -exec wc -l {} +` and verified against the L1 entry's "63 files / 17,403 LOC" headline.

## L3-deep-dive candidates

Per Δ L2-3 (do not open files >1,500 LOC for inline summary):

| File | LOC | Disposition |
|------|-----|-------------|
| `errors.py` | 1,566 | **L3 deep-dive candidate.** Only file in cluster crossing the 1,500-LOC threshold. Header-only inspection (lines 1–40); body deferred. |
| `declaration_contracts.py` | 1,323 | **Near-threshold (~80%).** Header read in full (`:1-58`); concrete content cited by file:line elsewhere in catalog and report; body NOT opened. |
| `audit.py` | 922 | Header-only inspection (`:1-25`). Body NOT opened. |
| `schema.py` | 851 | Header-only inspection (cited via downstream usage in `schema_contract.py:20` and `plugin_protocols.py:20`); body NOT opened. |
| `schema_contract.py` | 797 | Header-only inspection (`:1-50`). Body NOT opened. |
| `plugin_protocols.py` | 753 | Header-only inspection (`:1-50`). Body NOT opened. |
| `config/runtime.py` | 655 | Header-only inspection (`:1-60`); one targeted read at `:325-345` to verify a freeze-pattern observation (the `MappingProxyType(dict(self.services))` shallow-wrap). |

Six other top-level files have 300–550 LOC each (`results.py` 632, `plugin_context.py` 550, `__init__.py` 535, `events.py` 435, `url.py` 396, `data.py` 385). None reached the L3 threshold; treated at L2 depth.

## Sub-subsystem grouping (proposed for catalog)

Per Δ L2-3 (one entry per immediate subdirectory or coherent file group), the 63 files partition into **14 entries**:

1. **`__init__.py` — public surface** (1 file, 535 LOC): the re-exporting hub; 208-name `__all__`; explicit "Settings classes are NOT re-exported" comment block at lines 79–87 anchoring the post-ADR-006 boundary.
2. **`config/` sub-package** (5 files, 1,231 LOC): Settings→Runtime alignment primitives — `runtime.py` (655 LOC), `protocols.py` (209), `alignment.py` (181), `__init__.py` (112), `defaults.py` (74). Per `config-contracts-guide` skill.
3. **Freeze + numeric-validation primitives** (1 file, 172 LOC): `freeze.py` — `deep_freeze`, `freeze_fields`, `deep_thaw`, `require_int`. Pattern source for KNOW-C62/63/65.
4. **Hashing primitive** (1 file, ~150 LOC): `hashing.py` — RFC-8785 canonical JSON + `CANONICAL_VERSION = "sha256-rfc8785-v1"`. Per ADR-006 Phase 2 (extracted from core to break circular dependency).
5. **Audit DTO surface — `audit.py`** (1 file, 922 LOC, L3 deep-dive candidate by header): the strict-typed dataclass surface for Landscape rows (Run, Token, Operation, Call, Batch, Node, NodeState*, etc.). Header-only treatment.
6. **Audit-evidence framework** (3 files, ~1,700 LOC combined): `audit_evidence.py` (AuditEvidenceBase ABC; ADR-010 §Decision 1), `audit_protocols.py` (cross-layer recorder protocols), `declaration_contracts.py` (1,323 LOC; ADR-010 §Decision 3 — 4-site nominal-ABC dispatcher framework; near-threshold). Headers + class-level inspection only.
7. **Tier-1 registry primitives** (2 files): `tier_registry.py` (Tier-1 exception registry, module-prefix allowlist, freeze flag; ADR-010 §Decision 2; defines `FrameworkBugError`), `registry_primitive.py` (`FrozenRegistry` shared mechanics — ordered list, auxiliary map, freeze flag, single RLock).
8. **Schema contracts** (5 files, ~2,000 LOC combined): `schema.py` (851), `schema_contract.py` (797), `schema_contract_factory.py`, `contract_builder.py`, `contract_propagation.py`, `contract_records.py`, `transform_contract.py`, `type_normalization.py`. Schema-contract subsystem per Unified Schema Contracts design.
9. **Plugin-side L0 surface** (5 files, ~1,750 LOC combined): `plugin_protocols.py` (753; Source/Transform/Sink protocols + `PluginConfigProtocol`), `plugin_context.py` (550; `PluginContext` + tokens), `plugin_roles.py`, `plugin_semantics.py`, `plugin_assistance.py`, `contexts.py` (phase-typed protocols — `LifecycleContext`, `SinkContext`, `SourceContext`, `TransformContext`).
10. **Errors / reasons / DTOs** (1 file, 1,566 LOC, L3 deep-dive candidate): `errors.py`. Header-only inspection: re-exports `FrameworkBugError` from `tier_registry` to break a circular import; applies `@tier_1_error` decoration; defines `ExecutionError` (Tier-2 frozen audit DTO). Body deferred.
11. **Identity / lineage / token usage** (3 files): `identity.py` (`TokenInfo`), `token_usage.py` (`TokenUsage`), `secret_scrub.py` (declaration-violation payload scrubbing per ADR-010 §Decision 3 — last-line-of-defence redaction).
12. **Checkpoint family** (4 files): `checkpoint.py`, `batch_checkpoint.py`, `coalesce_checkpoint.py`, `aggregation_checkpoint.py`. Tier-1 audit DTOs for resume.
13. **Pipeline runner protocol** (1 file, ~25 LOC): `pipeline_runner.py` — `PipelineRunner` Protocol enabling L2 → L3 callback without upward import. Bookmarked by engine cluster's cross-cluster observations (`bootstrap.py` and `dependency_resolver.py` consume this protocol).
14. **Misc top-level types** (~30 files): `data.py`, `types.py`, `enums.py`, `results.py`, `run_result.py`, `events.py`, `node_state_context.py`, `header_modes.py`, `diversion.py`, `routing.py`, `probes.py`, `coalesce_enums.py`, `coalesce_metadata.py`, `call_data.py`, `engine.py`, `sink.py`, `cli.py`, `payload_store.py`, `database_url.py`, `url.py`, `runtime_val_manifest.py`, `export_records.py`, `secrets.py`, `security.py`. Per-file LOC ≤ 632 (results.py); most ≤ 400. Treated as a single coherent group of "shared types and small protocols" with file-level citations where invariants land.

This 14-entry partition keeps catalog entries within the 300–500-word budget and avoids per-file fragmentation.

## Layer conformance status

- **Authoritative whole-tree check:** `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` → **exit 0**, "No bug-hiding patterns detected. Check passed." This is the load-bearing artefact for both:
  - The L0 leaf invariant ([CITES] KNOW-A53, KNOW-C47): no contracts/ → core/, contracts/ → engine/, or contracts/ → L3 imports exist at runtime. The TYPE_CHECKING-only imports from core (e.g., `config/runtime.py:30-37` imports `CheckpointSettings`, `ConcurrencySettings`, `RateLimitSettings`, `RetrySettings`, `TelemetrySettings` under `if TYPE_CHECKING:`) do NOT create runtime coupling per `__init__.py:8-10`.
  - The R-rule conformance: every `dict.get`, `getattr`, `isinstance`, `silent-except` call site outside trust boundaries has a justifying allowlist entry under `config/cicd/enforce_tier_model/`.
- **Cluster-scoped corroborative check:** `enforce_tier_model.py check --root src/elspeth/contracts ...` → exit 1, 225 R-rule findings (R5=184, R2=20, R1=12, R6=9). These are NOT layer-import violations. Two confirming pieces of evidence:
  - The same script run with `--allowlist config/cicd/enforce_tier_model` (engaged) STILL produces 225 findings; running it WITHOUT the allowlist (`temp/layer-check-contracts-empty-allowlist.txt`) produces the **same 225-finding count**. If the allowlist were doing its job, the count would differ; it doesn't. The cluster-scoped run is reading allowlist YAMLs whose `key` field is full-path-prefixed against the whole `src/elspeth` tree (e.g., `key: contracts/type_normalization.py:R5:normalize_type_for_contract:fp=...`), and `--root src/elspeth/contracts` reports paths relative to the `--root`, not the tree, so the prefix match never fires.
  - The whole-tree run is clean; same allowlist; same code; the only delta is `--root`. Therefore the 225 findings are an **allowlist-prefix-mismatch artefact**, not cluster-specific debt. Same artefact the engine cluster pass identified and the engine validator V6 confirmed.
- **TYPE_CHECKING warnings (TC-rules):** zero. Recorded in `temp/layer-conformance-contracts.json` as `type_checking_layer_warnings_TC: []`.
- **L0 outbound count:** zero. Recorded in `temp/intra-cluster-edges.json` as `stats.outbound_edge_count = 0`.

## SCC status

Verified absence: `[ORACLE: temp/l3-import-graph.json strongly_connected_components — no entry contains any 'contracts' or 'contracts/*' node]`. The five SCCs in the L3 oracle (`web/*`, `plugins/transforms/llm/*`, `mcp/*`, `telemetry/*`, `tui/*`) are L3-only by construction, since the oracle is the L3↔L3 import graph. Δ L2-7 SCC handling does not apply to this cluster.

## Test coverage map

`tests/unit/contracts/` contains 87 files; `tests/unit/contracts/config/` contains 6 sub-files; `tests/integration/contracts/` contains 1 file (`test_build_runtime_consistency.py`). Direct unit coverage for every catalog entry's central invariant:

| Catalog entry | Test module |
|---|---|
| `freeze.py` | `test_freeze.py`, `test_freeze_regression.py` |
| `hashing.py` | `test_hashing.py` |
| `tier_registry.py` | `test_tier_registry.py`, `test_tier_registry_migration.py` |
| `registry_primitive.py` | `test_registry_primitive.py` |
| `secret_scrub.py` | `test_secret_scrub.py` |
| `audit_evidence.py` | `test_audit_evidence.py`, `test_audit_evidence_nominal_scanner.py` |
| `audit.py` | `test_audit.py` |
| `audit_protocols.py` | `test_audit_protocols.py` |
| `declaration_contracts.py` | `test_declaration_contracts.py` |
| `plugin_protocols.py` | `test_plugin_protocols.py` |
| `plugin_context.py` | `test_plugin_context_recording.py` |
| `schema_contract.py`, `schema_contract_factory.py` | `test_schema_contract.py`, `test_schema_contract_factory.py` |
| `contract_builder.py`, `contract_propagation.py`, `contract_records.py` | `test_contract_builder.py`, `test_contract_propagation.py`, `test_compose_propagation.py`, `test_contract_narrowing.py`, `test_contract_records.py` |
| `contract_violations` | `test_contract_violation_error.py`, `test_contract_violations.py` |
| `config/runtime.py` | `tests/unit/contracts/config/test_runtime_*.py` (5 files: alignment, common, retry, concurrency, rate_limit, checkpoint) |
| Sink/source contracts | `tests/unit/contracts/sink_contracts/`, `tests/unit/contracts/source_contracts/` |

A few catalog entries have no dedicated unit test file (e.g., `pipeline_runner.py` — protocol-only, ~25 LOC; `errors.py` — exercised through the violation-classes test suite). Flagged in catalog "Concerns" for the affected entries.

## ADR mapping (cited from `00b-existing-knowledge-map.md`)

- **ADR-006** (KNOW-ADR-006 — Accepted 2026-02-22): Layer dependency remediation — strict 4-layer model, 10 violations → 0, CI enforcement. Phases per KNOW-ADR-006b: Phase 1 ExpressionParser → core; **Phase 2 extracts `contracts/hashing.py`** (visible in this cluster as the single-purpose canonical-JSON helper that breaks the contracts↔core/canonical cycle); Phase 3 fingerprint+DSN handling; Phase 4 `RuntimeServiceRateLimit` (visible at `config/runtime.py:291` as `@dataclass(frozen=True, slots=True)`); Phase 5 CI gate (visible at `enforce_tier_model.py:237 "contracts": 0`).
- **ADR-010** (KNOW-ADR-010 — Accepted 2026-04-19, amended 2026-04-20): Declaration-trust framework. L0 surface lives in this cluster: `audit_evidence.py` (Decision 1, §nominal ABC), `tier_registry.py` (Decision 2, `@tier_1_error` decorator + frozen registry, KNOW-ADR-010b), `declaration_contracts.py` (Decision 3, 4-site dispatcher framework, payload-schema H5 = KNOW-ADR-010h). The engine cluster owns the dispatch-site mechanics; this cluster owns the L0 vocabulary the dispatcher uses.

## Reading-order discipline

Catalog written in the order the L0 surface composes:

1. Public surface (`__init__.py`) — the contract that L1+ depends on.
2. Primitives (`freeze`, `hashing`, `registry_primitive`) — used by every other contracts module.
3. Tier-1 registry (`tier_registry`) — registered exceptions; depended on by `errors.py` for `FrameworkBugError`.
4. Audit-evidence framework (`audit_evidence`, `declaration_contracts`, `audit_protocols`) — ADR-010 §Decision 1/3.
5. Errors / reasons / DTOs (`errors.py`) — header only.
6. Audit DTO surface (`audit.py`) — header only.
7. Schema contracts.
8. Plugin-side surface (`plugin_protocols`, `plugin_context`, `contexts`).
9. Identity / token usage / secret scrub.
10. Checkpoint family.
11. `pipeline_runner` protocol.
12. Config sub-package.
13. Misc top-level types.

This isn't a topological sort — circular header references exist (e.g., `errors.py` imports from `audit_evidence`, `declaration_contracts`, `tier_registry`; `audit_protocols.py` imports from `audit`, `call_data`, `errors`, `schema_contract`). The order above is **conceptual**: primitives first, then framework, then derived types.
