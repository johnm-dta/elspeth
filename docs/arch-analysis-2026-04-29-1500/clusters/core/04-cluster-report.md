# 04 — core/ cluster report (synthesis, risks, debt candidates)

This report synthesises the L2 cluster pass on `core/` (49 files, 20,791 LOC, layer L1 — the foundation tier). Inputs: the 9 sub-subsystem entries in `02-cluster-catalog.md`, the holistic scan in `01-cluster-discovery.md`, the empty-by-design `temp/intra-cluster-edges.json` (oracle is L3-only), and the `temp/layer-check-core{,-empty-allowlist}.txt` artefacts (0 L1 violations, 0 TC warnings — `core/` is fully layer-conformant).

## 1. Layer-conformance verdict

`core/` is fully layer-conformant. Two artefacts:

- `temp/layer-check-core.txt` — production allowlist run, 205 findings, 0 L1, 0 TC.
- `temp/layer-check-core-empty-allowlist.txt` — empty allowlist run, 205 findings (identical), 0 L1, 0 TC.

The 205 findings are R5 (170 × `isinstance`), R6 (18 × silent except), R1 (15 × `dict.get`), R4 (2 × broad except). All are pre-allowlisted defensive patterns that re-surface when the scope narrows below the allowlist's whole-tree key prefix — same artefact behaviour as the engine cluster. The 170 R5 sites concentrate in `landscape/` and reflect Tier-1 read guards at the SQLAlchemy ↔ Python boundary, which are legitimate per CLAUDE.md "Tier 1: crash on any anomaly" because alternative access patterns (`row.foo`) would not yield meaningful errors when the DB returns the wrong shape.

The Δ L2-6 `dump-edges` byte-equality assertion is **N/A** because the cluster is L1, not L3.

## 2. Sub-subsystem map (recap)

| # | Sub-subsystem | Path(s) | Files | LOC | Deep-dive flags | Confidence |
|---|---|---|---:|---:|---:|---|
| 1 | landscape | `landscape/` | 18 | 9,384 | 2 (`execution_repository.py`, `data_flow_repository.py`) | High |
| 2 | dag | `dag/` | 5 | 3,549 | 1 (`graph.py`) | High |
| 3 | checkpoint | `checkpoint/` | 5 | 1,237 | 0 | High |
| 4 | rate_limit | `rate_limit/` | 3 | 470 | 0 | High |
| 5 | retention | `retention/` | 2 | 445 | 0 | High |
| 6 | security | `security/` | 4 | 940 | 0 | High (web.py + config_secrets.py) / Medium (secret_loader.py) |
| 7 | configuration_family | `config.py + dependency_config.py + secrets.py` | 3 | 2,524 | 1 (`config.py`) | Medium (config.py deferred) / High (others) |
| 8 | canonicalisation_and_templating | `canonical.py + templates.py + expression_parser.py` | 3 | 1,422 | 0 (expression_parser.py 820 LOC is below threshold but security-sensitive) | High (canonical+templates) / Medium (expression_parser deferred) |
| 9 | cross_cutting_primitives | `events.py + identifiers.py + logging.py + operations.py + payload_store.py + __init__.py` | 6 | 720 | 0 | High |
| **Total** | | | **49** | **20,691** (vs L1 20,791; ~0.5% drift, see 00-coordination) | **4** | |

## 3. Highest-confidence claims (top 3 — for stitched-report propagation)

1. **`core/` is fully layer-conformant; the 4-layer model is mechanically respected at L1.** Verified by `enforce_tier_model.py check --root src/elspeth/core --allowlist <production>` (205 findings, all R5/R6/R1/R4 defensive-pattern items already governed by per-file allowlists; **0 L1, 0 TC**), corroborated by re-run with empty allowlist (identical histogram). Public surface of `core/__init__.py` (100 LOC) confirms outbound to `contracts/` only — `IntegrityError`, `PayloadStore`, `ResumeCheck`, `ResumePoint`, plus DTOs are re-exported from `contracts/`. **Implication:** Any future cross-layer need MUST follow the `KNOW-ADR-006d` "Violation #11 Protocol" (move down → extract primitive → restructure caller → never lazy-import). The single observed lazy intra-cluster import (`dependency_config.py:63` → `ExpressionParser`) is intra-`core/` and not a layer violation.

2. **The Landscape sub-area is the cluster's hub and matches the documented 4-repository facade pattern.** `landscape/__init__.py` (154 LOC, read in full) re-exports `RecorderFactory` (the facade per `KNOW-A29`), `LandscapeDB`, and exactly the 4 repositories named in `KNOW-A30` (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`, `RunLifecycleRepository`). `landscape/schema.py` defines **20 tables** (verified by grep: runs, nodes, edges, rows, tokens, token_outcomes, token_parents, node_states, operations, calls, artifacts, routing_events, batches, batch_members, batch_outputs, validation_errors, transform_errors, checkpoints, secret_resolutions, preflight_results) — `[DIVERGES FROM KNOW-A24]` which claims 21. Repositories are **NOT** re-exported through the cluster's `core/__init__.py`; callers reach them via `elspeth.core.landscape.*` per the encapsulation discipline. **Implication:** the documented "facade + 4 repositories" architecture is real and load-bearing — refactoring proposals that bypass `RecorderFactory` should be challenged.

3. **A "Protocol-based no-op parity" pattern recurs across `core/` and is a deliberate offensive-programming choice.** `EventBus`/`NullEventBus` (`events.py:14–28, 88–111`) and `RateLimiter`/`NoOpLimiter` (`rate_limit/registry.py:33–66`) both implement a protocol structurally rather than via inheritance. The `NullEventBus` docstring is explicit (`events.py:88–103`): "If someone subscribes expecting callbacks, inheritance would hide the bug. Protocol-based design makes the no-op behavior explicit." Combined with the pervasive use of `freeze_fields` / `__post_init__` validation per `KNOW-C61`–`KNOW-C65` and the offensive-programming `__post_init__` checks in `dag/models.py:151–193` and `dependency_config.py:144–155`, this is a coherent design discipline. **Implication:** new no-op or null-object additions to `core/` MUST follow the Protocol pattern; inheritance from the real class is the wrong default.

## 4. Highest-uncertainty questions (top 3 — agenda for post-L2 synthesis)

1. **`config.py` (2,227 LOC) cohesion — essential or accidental complexity?** The single Pydantic settings file holds 12+ child dataclasses (`CheckpointSettings`, `ConcurrencySettings`, `DatabaseSettings`, `ElspethSettings`, `LandscapeExportSettings`, `LandscapeSettings`, `PayloadStoreSettings`, `RateLimitSettings`, `RetrySettings`, `SecretsConfig`, `ServiceRateLimit`, `SinkSettings`, `SourceSettings`, `TransformSettings`) plus the `load_settings()` loader. Pydantic settings tend to concentrate for cross-validation reasons, but 2,227 LOC of single-file configuration is substantial. **Open question:** does internal structure factor cleanly (e.g., per-domain validator clusters, source/transform/sink groupings) or has it accreted by addition? **Resolution path:** L3 deep-dive on `config.py` followed by an architecture-pack proposal (split or keep) — **not** a `core/` archaeology decision.

2. **Contracts/core boundary post-ADR-006 (L1 open question Q1).** This pass observed `core/`'s outbound imports from `contracts/` and found them concentrated in expected primitive surfaces (`payload_store`, `errors`, `freeze`, `hashing`, `schema`, `schema_contract`, `secrets`, `security`). The structural side (no upward imports, layer-clean) is verified. The semantic side — *should* the responsibility cut be different? are there primitives currently in `core/` that belong in `contracts/`, or vice-versa? — requires reading both clusters and is a post-L2 synthesis concern. **Specific candidate for review:** `core/secrets.py` (124 LOC, runtime resolver) lives at `core/` root while `core/security/{secret_loader,config_secrets}.py` (529 LOC combined) live in the subpackage. Their topical relationship (both about secrets) but role disjointness (runtime vs config-time) is intentional but may benefit from explicit naming or co-location.

3. **`dag/graph.py` (1,968 LOC) cascade-prone risk — concrete blast radius?** The §7 P3 "cascade-prone" framing is qualitatively correct: `ExecutionGraph` is consumed by every executor in `engine/`, by `web/composer/_semantic_validator.py` (per L3 oracle), by `web/execution/validation.py`, by `core/checkpoint/{manager,compatibility}`, and indirectly by every plugin via the schema-contract validation flow. **Open question:** what is the test surface that locks `ExecutionGraph`'s public contract? `tests/unit/core/dag/test_graph.py` and `test_graph_validation.py` exist; their assertion density vs the file's behavioural surface area is the deferred deep-dive question. The `tests/unit/core/dag/test_models_post_init.py` evidence is encouraging (it locks the `NodeInfo` invariants), but `graph.py` itself was not opened by this pass.

## 5. Risks and debt candidates (deferred to architecture-pack)

These are **observations**, not prescriptions. Per Δ L2-7 / archaeology vs architecture discipline, this pass surfaces; the architecture-pack pass prescribes.

| ID | Risk / debt | Evidence | Owner for resolution |
|----|-------------|----------|----------------------|
| R-1 | `landscape/execution_repository.py` 1,750 LOC + `data_flow_repository.py` 1,590 LOC = 3,340 LOC in 2 files (36% of landscape) | `wc -l` verified; KNOW-A31 claimed 1,480 — drift +18% | Architecture-pack via L3 deep-dive |
| R-2 | `core/config.py` 2,227 LOC — single Pydantic settings file | `wc -l` verified | Architecture-pack via L3 deep-dive |
| R-3 | `core/dag/graph.py` 1,968 LOC — cascade-prone choke point | `wc -l` verified, §7 P3 framing | Architecture-pack via L3 deep-dive |
| R-4 | KNOW-A24 claims 21 audit tables; verified 20 | grep `^[a-z_]+_table = Table` `landscape/schema.py` → 20 | Doc-correctness pass (NOT architecture) |
| R-5 | KNOW-A20 claims `checkpoint/` ~600 LOC; verified 1,237 LOC (≈2× drift) | `wc -l` per-file totals | Doc-correctness pass |
| R-6 | KNOW-A21 claims `rate_limit/` ~300 LOC; verified 470 LOC | `wc -l` per-file totals | Doc-correctness pass |
| R-7 | KNOW-A27 claims `expression_parser.py` ~652 LOC; verified 820 LOC | `wc -l` | Doc-correctness pass |
| R-8 | `core/secrets.py` (root) and `core/security/{secret_loader,config_secrets}.py` (subpackage) split is intentional but undocumented; reader landing on one may not know the other exists | Per-file inspection | Doc-correctness or minor in-cluster comment addition |
| R-9 | 170 R5 (`isinstance`) findings in `core/` (mostly `landscape/`) are pre-allowlisted Tier-1 read guards; legitimate but the volume warrants future review of whether the read-guard idiom can be consolidated into a helper | `temp/layer-check-core-empty-allowlist.txt` rule histogram | Code-quality pass (not blocking) |
| R-10 | `core/expression_parser.py` (820 LOC) is below the deep-dive threshold but is security-sensitive (`ExpressionSecurityError` is part of its surface). Used by config-time gate validation, runtime gates, and the composer | Public surface enumeration; `KNOW-ADR-006b` Phase 1 origin | Future security-architect review (not L2 archaeology) |

## 6. Cross-cluster observations for synthesis (Δ L2-4 deferral channel)

These are claims I would have wanted to make about other clusters but are forbidden by Δ L2-4. Recorded here verbatim for the post-L2 synthesis pass.

- **(Synthesis-1) Contracts↔core boundary inventory.** `core/` imports the following identifiers from `contracts/`, observed during this pass: `IntegrityError`, `PayloadStore` (Protocol), `PayloadNotFoundError`, `AuditIntegrityError`, `TIER_1_ERRORS`, `BatchPendingError`, `Operation`, `Artifact`, `Batch`, `BatchMember`, `BatchOutput`, `Call`, `CallStatus`, `CallType`, `Checkpoint`, `ContractAuditRecord`, `Edge`, `FieldAuditRecord`, `Node`, `NodeState`, `NodeStateCompleted`, `NodeStateFailed`, `NodeStateOpen`, `NodeStateStatus`, `ReproducibilityGrade`, `ResumeCheck`, `ResumePoint`, `RoutingEvent`, `RoutingSpec`, `Row`, `RowLineage`, `Run`, `RunStatus`, `Token`, `TokenParent`, `ValidationErrorWithContract`, `SecretResolutionInput`, `CANONICAL_VERSION` (hashing), `deep_freeze`, `deep_thaw`, `freeze_fields`, `require_int` (freeze), `FieldDefinition`, `SchemaConfig` (schema), `PipelineRow`, `SchemaContract` (schema_contract), `ResolvedSecret`, `WebSecretResolver` (secrets), `secret_fingerprint`, `get_fingerprint_key` (security), `NodeType` (enums), `CoalesceName`, `NodeID` (types), and the `aggregation_checkpoint` / `coalesce_checkpoint` typed-dict modules. The contracts cluster's pass should verify that **every name above is present in its `__init__.py` `__all__` list** — if any is missing, that's a contracts-side debt item.

- **(Synthesis-2) `KNOW-A24` 20-vs-21-tables question.** `core/landscape/schema.py` defines 20 tables; the doc says 21. Possible explanations: (a) one table was renamed/dropped after `KNOW-A24` was written; (b) one is conditionally created (e.g., dialect-specific); (c) the doc was always off by one. **Defer to doc-correctness pass.** The contracts cluster does not own this question (table definitions live in `core/landscape/schema.py`, not `contracts/`).

- **(Synthesis-3) Engine cluster will need the `core/` outbound surface this pass enumerates.** When the engine cluster catalog asserts "engine imports `RecorderFactory`, `ExecutionGraph`, `compute_full_topology_hash`, `CheckpointManager`, …" it should cite this report's section 6 for the `contracts/`-side primitives that engine reaches **through** core (e.g. engine → core → contracts.errors.TIER_1_ERRORS for the `tier_1_error` registry per `KNOW-ADR-010b`).

- **(Synthesis-4) `core/secrets.py` ↔ `web/composer/`** — the runtime secret-ref resolver in `core/secrets.py` is consumed by the web composer when threading `{"secret_ref": ...}` references through resolved configs. The composer cluster's catalog should record `web/composer/* → core/secrets` as one of its outbound `core/` edges.

- **(Synthesis-5) MCP/composer_mcp cluster's separation rationale is reinforced by this pass.** `core/landscape/__init__.py` is the read-only audit DB surface (`elspeth-mcp` consumes this per `KNOW-C35`), distinct from `composer_mcp/` which has no Landscape coupling at all. This pass does not assert anything about the MCP clusters — merely confirms that the structural separation in `core/` (Landscape sub-area is encapsulated, `RecorderFactory`-fronted) supports the L1 "do not merge" guidance.

## 7. Departures from L1 (recorded honestly)

- `KNOW-A24` claims 21 audit tables; verified 20. `[DIVERGES FROM KNOW-A24]` recorded in entry 1 and section 6 above.
- `KNOW-A20` checkpoint LOC: ~600 claimed, 1,237 verified. `[DIVERGES FROM KNOW-A20]` recorded in entry 3.
- `KNOW-A21` rate_limit LOC: ~300 claimed, 470 verified. `[DIVERGES FROM KNOW-A21]` recorded in entry 4.
- `KNOW-A27` expression_parser LOC: ~652 claimed, 820 verified. `[DIVERGES FROM KNOW-A27]` recorded in entry 8.
- `KNOW-A31` ExecutionRepository LOC: ~1,480 claimed, 1,750 verified. `[DIVERGES FROM KNOW-A31]` recorded in entry 1.
- LOC sum delta: 20,691 (per-file `wc -l` summed) vs L1's 20,791. ~0.5% drift attributable to `find … -print0 | xargs -0 cat | wc -l` (L1's method) vs per-file `wc -l` summed (this pass). Recorded in `00-cluster-coordination.md`.

All other L1 claims about `core/` are **confirmed** at L2 depth, including:

- 49 files verified by direct count.
- COMPOSITE classification per Δ4 (≥4 sub-pkgs, ≥10k LOC, ≥20 files all fire).
- 4 deep-dive candidates from the L1 flag list are exactly the 4 verified at L2 (`config.py` 2,227, `dag/graph.py` 1,968, `landscape/execution_repository.py` 1,750, `landscape/data_flow_repository.py` 1,590).
- Layer L1 with outbound to `{contracts}` only.
- Inbound from `{engine, plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` (any L2+).

## 8. Validator handoff

The Δ L2-8 validation gate's contract:

- **Sub-subsystem entries correspond to actual directories or coherent file groups** — verifiable by `ls src/elspeth/core/`. 6 subpackages (checkpoint/, dag/, landscape/, rate_limit/, retention/, security/) and 11 standalone .py files (canonical, config, dependency_config, events, expression_parser, identifiers, logging, operations, payload_store, secrets, templates, plus __init__.py = 12 with __init__) — file groups 7/8/9 enumerate the standalone modules.
- **Oracle citations resolve.** `temp/intra-cluster-edges.json` is empty by design and cited explicitly in the cluster-coordination Δ L2-2 section; the L3 oracle is cited only for cross-cluster external-coupling claims about who imports `core/` (which are themselves derived from the L3 oracle's nodes/edges with `core/` and `contracts/` excluded).
- **`[CITES]` / `[DIVERGES FROM]` references resolve in `00b-existing-knowledge-map.md`.** Citations span KNOW-A4, A20–A33, A47–A49, A53–A58, A69; KNOW-C7–C12, C16, C19, C28, C32, C34–C36, C38–C42, C44, C45, C47, C57, C58, C60–C65; KNOW-G9; KNOW-ADR-002, ADR-003, ADR-004, ADR-005c, ADR-006b, ADR-006d, ADR-007, ADR-009, ADR-010b. Divergences enumerated in section 7.
- **No claim crosses the cluster boundary** except via section 6 (the deferral channel). The validator should specifically check section 1 and entries 1–9 for accidental cross-cluster claims about contracts/, engine/, plugins/, web/, mcp/, composer_mcp/, telemetry/, tui/, testing/, or cli internals.
- **SCC handling (Δ L2-7):** N/A — `core/` does not appear in the L3 oracle's `strongly_connected_components` list. Confirmed by direct grep on the JSON.
- **Layer-check oracle output captured and clean.** Both `temp/layer-check-core.txt` and `temp/layer-check-core-empty-allowlist.txt` are present; the load-bearing claim (0 L1, 0 TC) is verifiable by re-running `grep -c '^  Rule: L1'` and `grep -c '^  Rule: TC'` on either file.

The validator should NOT re-run the L1 validation contract.

## Highest-confidence claims

(see Section 3 above — these are the verbatim entries to be propagated to the eventual stitched report.)

## Highest-uncertainty questions

(see Section 4 above — these are the agenda for the post-L2 synthesis pass.)

## Cross-cluster observations for synthesis

(see Section 6 above — these are the Δ L2-4 deferrals.)
