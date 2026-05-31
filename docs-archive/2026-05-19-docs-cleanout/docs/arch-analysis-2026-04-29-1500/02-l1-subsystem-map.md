# 02 — L1 Subsystem Map

[Δ4 note] 11 subsystems verified, not 12 — recorded as `[DIVERGES FROM]` the scope-override prose count. The Δ4 expected-classification list itself enumerates only 11; treating the prose count as a typo.

## Conventions

- **Layer claims** cite the oracle artefact (`temp/tier-model-oracle.txt`) and the path→layer table at `scripts/cicd/enforce_tier_model.py:237–248` (`LAYER_HIERARCHY` lines 237–241 + `LAYER_NAMES` lines 243–248). The enforcer ran clean (`Check passed`), so the layer model is the authoritative cross-layer dependency truth-source.
- **Knowledge claims** cite `00b-existing-knowledge-map.md` ids (e.g. `[CITES KNOW-C12]`) or contradict them with one-line justification (`[DIVERGES FROM KNOW-A22]`).
- **L3↔L3 dependency edges** are layer-permitted but unconstrained at L1 depth — flagged "deferred to L2 dispatch wave", not enumerated by grep (per Δ5 ban + Δ2 depth cap).
- **"Internal sub-areas"** is a single line for composites only — no per-file breakdown.
- **Composite triggers** per Δ4 heuristic: ≥4 sub-pkgs OR ≥10k LOC OR ≥20 files. The trigger that fired is named per entry.
- **L2-deep-dive flags**: any single file >1,500 LOC is named-and-deferred, never read.

## 1. contracts/

**Location:** `src/elspeth/contracts/`
**Responsibility:** Leaf module owning shared types, protocols, enums, errors, and frozen-dataclass primitives that every higher layer imports — no outbound dependencies, by construction. [CITES KNOW-A53] [CITES KNOW-C47]
**Layer:** L0 — leaf, no upward outbound permitted [`enforce_tier_model.py:237` `"contracts": 0`].
**Composite/Leaf:** **COMPOSITE** — triggered by ≥10k LOC (17,403) AND ≥20 files (63). One sub-package only (`config/`), so the file-count and LOC heuristics dominate.
**Size:** 63 files, 17,403 LOC.
**Internal sub-areas (single line):** Top-level contracts modules (audit_evidence, declaration_contracts, plugin_context, plugin_protocols, errors, freeze, hashing, schema_contract, security, …) plus the `config/` subpackage (alignment, defaults, protocols, runtime).
**Inbound dependencies (subsystem names only):** Layer model permits inbound from any L1+. Concrete confirmed inbound = `{core, engine, plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}`.
**Outbound dependencies (subsystem names only):** ∅ (leaf invariant from clean enforcer status; KNOW-A53 explicitly states "ZERO outbound dependencies").
**Highest-risk concern (≤1 sentence):** L2-deep-dive candidate inside this subsystem: `contracts/errors.py` (1,566 LOC per discovery findings) — flag for later, do not open.
**Confidence:** **High** — layer claim verified against oracle artefact; responsibility cited from KNOW-A23, KNOW-A53, KNOW-C47, KNOW-C61–KNOW-C65 (freeze contract is owned here); ARCHITECTURE.md's ~8,300 LOC figure (KNOW-A23) is roughly **half** the verified 17,403 — `[DIVERGES FROM KNOW-A23]` ARCHITECTURE.md count is stale (it predates accumulated growth).

## 2. core/

**Location:** `src/elspeth/core/`
**Responsibility:** L1 foundation primitives — Landscape (audit DB recorder + 4 repositories), DAG construction/validation, Dynaconf+Pydantic configuration, canonical-JSON hashing, payload store, retention, rate limiting, security, expression parser. [CITES KNOW-A22] [CITES KNOW-A29] [CITES KNOW-A30] [CITES KNOW-C27]
**Layer:** L1 — outbound subset of `{contracts}` only [`enforce_tier_model.py:238` `"core": 1`].
**Composite/Leaf:** **COMPOSITE** — triggered by ≥4 sub-pkgs (`checkpoint/`, `dag/`, `landscape/`, `rate_limit/`, `retention/`, `security/` = 6 sub-packages) AND ≥10k LOC (20,791) AND ≥20 files (49). All three heuristics fire.
**Size:** 49 files, 20,791 LOC.
**Internal sub-areas (single line):** `landscape/`, `dag/`, `checkpoint/`, `rate_limit/`, `retention/`, `security/`, plus top-level modules (config.py, expression_parser.py, canonical JSON, payload store, templates).
**Inbound dependencies (subsystem names only):** `{engine, plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` (any L2+ permitted).
**Outbound dependencies (subsystem names only):** `{contracts}` only (per layer model + clean enforcer).
**Highest-risk concern (≤1 sentence):** Three L2-deep-dive candidates inside this subsystem — `core/config.py` (2,227 LOC), `core/dag/graph.py` (1,968 LOC), `core/landscape/execution_repository.py` (1,750 LOC), and `core/landscape/data_flow_repository.py` (1,590 LOC); ARCHITECTURE.md's ~5,000 LOC figure for "Core" (KNOW-A22) excludes Landscape and is a containerisation accounting choice, not a divergence.
**Confidence:** **High** — layer + sub-area structure verified against `ls`; responsibility cross-cited from KNOW-A17, KNOW-A22, KNOW-A24–KNOW-A33, KNOW-C27. `[DIVERGES FROM KNOW-A22]` only in the LOC accounting (5k vs 20.8k); KNOW-A17 (Landscape ~8.3k) plus KNOW-A22 sums roughly to the verified figure once `dag/`, `checkpoint/`, `rate_limit/`, `security/`, `retention/` are folded in.

## 3. engine/

**Location:** `src/elspeth/engine/`
**Responsibility:** L2 SDA engine — Orchestrator, RowProcessor, executors (transform/coalesce/pass-through), RetryManager, ArtifactPipeline, SpanFactory, Triggers; owns run lifecycle and DAG execution per row/token. [CITES KNOW-A15] [CITES KNOW-A25] [CITES KNOW-A26] [CITES KNOW-C27]
**Layer:** L2 — outbound subset of `{core, contracts}` [`enforce_tier_model.py:239` `"engine": 2`].
**Composite/Leaf:** **COMPOSITE** — triggered by ≥10k LOC (17,425) AND ≥20 files (36) AND 2 sub-pkgs (`executors/`, `orchestrator/`). LOC and file-count heuristics dominate.
**Size:** 36 files, 17,425 LOC.
**Internal sub-areas (single line):** `orchestrator/`, `executors/`, plus top-level modules (processor.py, coalesce_executor.py, retry_manager, artifact_pipeline, span_factory, triggers, expression evaluators).
**Inbound dependencies (subsystem names only):** `{plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` (any L3).
**Outbound dependencies (subsystem names only):** `{contracts, core}` (per layer model + clean enforcer).
**Highest-risk concern (≤1 sentence):** Three L2-deep-dive candidates inside this subsystem — `engine/orchestrator/core.py` (3,281 LOC), `engine/processor.py` (2,700 LOC), `engine/coalesce_executor.py` (1,603 LOC); together these exceed 7,500 LOC, ~43% of the engine, and are explicitly called out as quality risks in KNOW-A70.
**Confidence:** **High** — layer + sub-area verified; responsibility cross-cited from KNOW-A15, KNOW-A25–KNOW-A28, KNOW-ADR-009b (engine/executors/pass_through.py is the cross-check site), KNOW-ADR-010i (4 dispatch sites). `[DIVERGES FROM KNOW-A15]` LOC figure ~12k is stale relative to verified 17.4k — drift has happened during ADR-007/008/009/010 work.

## 4. plugins/

**Location:** `src/elspeth/plugins/`
**Responsibility:** L3 plugin ecosystem — system-owned (not user-extensible) Sources, Transforms, Sinks, plus shared infrastructure (audited HTTP/LLM clients, hookspecs, base classes). [CITES KNOW-A16] [CITES KNOW-A35] [CITES KNOW-C21] [CITES KNOW-P2]
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}` [`enforce_tier_model.py:241–242` everything-else implicitly L3].
**Composite/Leaf:** **COMPOSITE** — triggered by ≥4 sub-pkgs (`infrastructure/`, `sources/`, `transforms/`, `sinks/` = 4) AND ≥10k LOC (30,399) AND ≥20 files (98). All three heuristics fire; this is the largest subsystem.
**Size:** 98 files, 30,399 LOC.
**Internal sub-areas (single line):** `infrastructure/` (hookspecs, audited clients, base classes), `sources/`, `transforms/`, `sinks/`.
**Inbound dependencies (subsystem names only):** Other L3 only — `{web, mcp, composer_mcp, tui, cli, testing}` (engine instantiates plugins via the registry, but the registry itself is exposed through L3 surfaces).
**Outbound dependencies (subsystem names only):** `{contracts, core, engine}` plus L3↔L3 edges (notably to `telemetry/` for audited clients) — **deferred to L2 dispatch wave**.
**Highest-risk concern (≤1 sentence):** L2-deep-dive candidate `plugins/transforms/llm/azure_batch.py` (1,592 LOC); plugin-count tension between `[CITES KNOW-A35]` "25 plugins" and `[DIVERGES FROM KNOW-A72]` summary count "46 plugins" surfaced by knowledge-ingestion (do not resolve, just flag).
**Confidence:** **High** — sub-package structure verified by `ls`; responsibility cross-cited from KNOW-A16, KNOW-A34–KNOW-A40, KNOW-C21–KNOW-C25, KNOW-P1–KNOW-P33; LOC ~30k is consistent with KNOW-A16's ~20.6k once growth + the audited-clients infrastructure is folded in.

## 5. web/

**Location:** `src/elspeth/web/`
**Responsibility:** L3 FastAPI web UI server — pipeline composer, run execution, catalog, blobs, secrets, sessions, auth, middleware. The `webui` extra (PyJWT, bcrypt, websockets, FastAPI, uvicorn, litellm) gates this subsystem. [CITES KNOW-G6] [CITES KNOW-G7]
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}` (oracle line 77).
**Composite/Leaf:** **COMPOSITE** — triggered by ≥4 sub-pkgs (`auth/`, `blobs/`, `catalog/`, `composer/`, `execution/`, `middleware/`, `secrets/`, `sessions/` = 8 backend sub-pkgs, plus `frontend/` which is **out of scope** per Δ6) AND ≥10k LOC (22,558 Python-only) AND ≥20 files (72). All three heuristics fire.
**Size:** 72 files, 22,558 LOC (Python only — TypeScript/React in `frontend/` is excluded per Δ6).
**Internal sub-areas (single line):** `composer/` (LLM-driven pipeline composer), `execution/`, `catalog/`, `auth/`, `sessions/`, `secrets/`, `blobs/`, `middleware/`; SPA dist served from `frontend/dist/` after API/WS routes [CITES KNOW-G7].
**Inbound dependencies (subsystem names only):** External (uvicorn ASGI loader at `elspeth.web.app:create_app` per KNOW-G6); not imported by any other subsystem.
**Outbound dependencies (subsystem names only):** `{contracts, core, engine}` plus L3↔L3 edges (`plugins`, possibly `composer_mcp` and `telemetry`) — **deferred to L2 dispatch wave**.
**Highest-risk concern (≤1 sentence):** Two L2-deep-dive candidates inside this subsystem — `web/composer/tools.py` (3,804 LOC, the single largest file in the tree) and `web/composer/state.py` (1,710 LOC); the `composer/` cluster strongly resembles a tool-registry parallel to `composer_mcp/` (Q2 in discovery findings) — relationship deferred.
**Confidence:** **Medium** — layer claim is High; sub-area enumeration is High (verified by `ls`); responsibility is Medium because the institutional documentation set has very few claims about `web/` specifically (only KNOW-G4–KNOW-G11 cover the staging deployment). No KNOW-A* claim covers the web subsystem because ARCHITECTURE.md predates web-UI maturity.

## 6. mcp/

**Location:** `src/elspeth/mcp/`
**Responsibility:** L3 read-only Model Context Protocol server for Landscape audit-DB analysis; exposes `diagnose()`, `get_failure_context(run_id)`, `explain_token(run_id, token_id)` and domain-specific analyzers for debugging pipeline failures. [CITES KNOW-A14] [CITES KNOW-C35]
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Composite/Leaf:** **LEAF** — 9 files, 4,114 LOC, 1 sub-pkg (`analyzers/`); below all three composite thresholds.
**Size:** 9 files, 4,114 LOC.
**Entry point:** `elspeth-mcp = "elspeth.mcp:main"` (pyproject `[project.scripts]`); top-level files = `analyzer.py`, `server.py`, `types.py`, `__init__.py`, plus the `analyzers/` sub-package.
**Inbound dependencies (subsystem names only):** External (MCP client) only — invoked as a console script, not imported.
**Outbound dependencies (subsystem names only):** `{contracts, core}` (reads Landscape) and possibly `{engine}` for query helpers — **L3 outbound deferred to L2 dispatch wave**.
**Distinct purpose vs `composer_mcp/`:** `mcp/` is the **post-hoc audit analyser** (read-only consumer of the Landscape) [CITES KNOW-C35]; `composer_mcp/` is the **pipeline-construction MCP** (interactive YAML composition via the `mcp__elspeth-composer__*` tool family). Separate console scripts (`elspeth-mcp` vs `elspeth-composer`), separate runtime concerns (read-only vs stateful sessions), separate dependency surfaces (Landscape vs Composer state). A future L2 reviewer should NOT merge them.
**Highest-risk concern (≤1 sentence):** None observed at L1 depth — sub-1500-LOC files only.
**Confidence:** **High** — layer + responsibility cross-cited from KNOW-A14, KNOW-C35; entry point verified from pyproject; sub-area shape from `ls`.

## 7. composer_mcp/

**Location:** `src/elspeth/composer_mcp/`
**Responsibility:** L3 stateful MCP server backing the LLM-driven pipeline composer — sessions, plugin assistance, YAML generation, validation, source/transform/sink discovery. [CITES KNOW-G9] (provider-error opt-in is composer-specific) and the `mcp__elspeth-composer__*` tool family in the agent toolbelt.
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Composite/Leaf:** **LEAF** — 3 files, 824 LOC; below all thresholds.
**Size:** 3 files, 824 LOC.
**Entry point:** `elspeth-composer = "elspeth.composer_mcp:main"` (pyproject `[project.scripts]`); files = `server.py`, `session.py`, `__init__.py`.
**Inbound dependencies (subsystem names only):** External (MCP client) only — console-script entry, not imported.
**Outbound dependencies (subsystem names only):** Likely `{contracts, core, engine, plugins}` and possibly `{web}` (composer/state.py at 1,710 LOC suggests shared composer state with the web UI) — **L3↔L3 deferred to L2**.
**Distinct purpose vs `mcp/`:** Composer is **interactive pipeline construction** (mutations: `set_source`, `upsert_node`, `upsert_edge`, `set_output`, `generate_yaml`); `mcp/` is **read-only audit analysis**. The two share the MCP transport but nothing else known at L1 depth. Sibling-vs-nested question (Q3 in discovery findings): the file-count + LOC asymmetry (3 / 824 vs 9 / 4,114) and the wholly disjoint tool surfaces support keeping them sibling. Confirming the `web/composer/` ↔ `composer_mcp/` relationship is the key L2 task.
**Highest-risk concern (≤1 sentence):** None observed at L1 depth, though the relationship to `web/composer/` (3,804 + 1,710 LOC) is an open architectural question (deferred to L2).
**Confidence:** **Medium** — layer + entry point are High; responsibility derives from the agent toolbelt and KNOW-G9 only (no KNOW-A* coverage). The institutional docs predate `composer_mcp/`.

## 8. telemetry/

**Location:** `src/elspeth/telemetry/`
**Responsibility:** L3 operational telemetry pipeline — circuit breaker, exporters (OTLP/Datadog/Azure Monitor), filtering, manager, hookspecs, serialization; **emits AFTER Landscape recording** (audit primacy). [CITES KNOW-A19] [CITES KNOW-A43] [CITES KNOW-A50] [CITES KNOW-A51] [CITES KNOW-A52] [CITES KNOW-C38]
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Composite/Leaf:** **LEAF** — 14 files, 2,884 LOC, 1 sub-pkg (`exporters/`); below all thresholds.
**Size:** 14 files, 2,884 LOC.
**Entry points (informational, not console_script):** Top-level files = circuit_breaker, errors, factory, filtering, hookspecs, manager, protocols, serialization, plus `exporters/`.
**Inbound dependencies (subsystem names only):** Other L3 — at minimum `{plugins}` (audited clients emit telemetry) and probably `{engine, web, cli}` — **deferred to L2**.
**Outbound dependencies (subsystem names only):** `{contracts, core}` plus possibly `{engine}` — **deferred to L2**.
**Highest-risk concern (≤1 sentence):** None observed at L1 depth — sub-1500-LOC files only; the audit-primacy invariant (KNOW-C38) is enforced at integration points outside this subsystem and is not visible at L1.
**Confidence:** **High** — layer + responsibility cross-cited from KNOW-A19, KNOW-A43, KNOW-A50–KNOW-A52, KNOW-C38–KNOW-C41. `[DIVERGES FROM KNOW-A19]` ~1,200 LOC figure is stale vs verified 2,884 — drift has happened.

## 9. tui/

**Location:** `src/elspeth/tui/`
**Responsibility:** L3 Textual-based TUI for interactive lineage exploration; surfaces audit-trail traversal under the `elspeth explain` CLI subcommand (no separate console_script). [CITES KNOW-A13] [CITES KNOW-C34]
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Composite/Leaf:** **LEAF** — 9 files, 1,175 LOC, 2 sub-pkgs (`screens/`, `widgets/`); below LOC + file-count thresholds.
**Size:** 9 files, 1,175 LOC.
**Entry point:** `tui/explain_app.py` is the audit-trail explorer entry, launched via `elspeth explain --run <run_id> --row <row_id>`. Top-level files = `constants.py`, `explain_app.py`, `types.py`, `__init__.py`, plus `screens/` and `widgets/` sub-packages.
**Inbound dependencies (subsystem names only):** `{cli}` only (Typer subcommand wires TUI launch).
**Outbound dependencies (subsystem names only):** `{contracts, core}` (reads Landscape) — **L3↔L3 edges deferred**.
**Highest-risk concern (≤1 sentence):** None observed at L1 depth — small subsystem, sub-1500-LOC files only.
**Confidence:** **High** — layer + responsibility cross-cited from KNOW-A13 (~800 LOC, near verified 1,175), KNOW-A9 (Auditor actor), KNOW-C34 (`elspeth explain` is the entry).

## 10. testing/

**Location:** `src/elspeth/testing/`
**Responsibility:** L3 in-tree test infrastructure exposed as a pytest plugin — provides `pytest_xdist_auto` for automatic parallel-worker configuration. NOT to be confused with the `tests/` directory (out of scope per Δ6) or with the **chaos** test servers (ChaosLLM/ChaosWeb/ChaosEngine — those live in `tests/`, not here, despite KNOW-A18). [DIVERGES FROM KNOW-A18] ARCHITECTURE.md says "Testing subsystem ~9,500 LOC" and lists chaos servers; the verified `src/elspeth/testing/` is 877 LOC across 2 files and contains only the pytest-xdist auto-detector — KNOW-A18 is conflating `src/elspeth/testing/` with `tests/`.
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Composite/Leaf:** **LEAF** — 2 files, 877 LOC; below all thresholds.
**Size:** 2 files, 877 LOC.
**Entry point:** `elspeth-xdist-auto = "elspeth.testing.pytest_xdist_auto"` (pyproject `[project.entry-points.pytest11]`); files = `pytest_xdist_auto.py`, `__init__.py`.
**Inbound dependencies (subsystem names only):** External (pytest plugin discovery via entry point).
**Outbound dependencies (subsystem names only):** Likely none in `{contracts, core, engine}` — pure pytest tooling — **confirmation deferred to L2**.
**Highest-risk concern (≤1 sentence):** Doc tension flagged above — KNOW-A18 conflates this subsystem with `tests/chaos*`; the chaos servers must be located in a future test-architecture pass.
**Confidence:** **Medium** — entry point + size are High; responsibility cuts against the institutional documentation (`[DIVERGES FROM KNOW-A18]`), which lowers overall confidence until the catalog wave verifies the chaos-servers location.

## 11. cli (root files)

**Location:** `src/elspeth/cli.py`, `src/elspeth/cli_helpers.py`, `src/elspeth/cli_formatters.py`, `src/elspeth/__init__.py`
**Responsibility:** L3 Typer-based CLI — top-level commands `run`, `resume`, `validate`, `explain`, `plugins list`, `purge`; routes to engine, web, tui, mcp surfaces; also hosts the `TRANSFORM_PLUGINS` plugin registry [CITES KNOW-P22]. [CITES KNOW-A12] [CITES KNOW-C34]
**Layer:** L3 — outbound subset of `{contracts, core, engine, other L3}`.
**Composite/Leaf:** **LEAF** — treated as a single subsystem per task brief (Δ4 line 38); 4 root-level files, 2,942 LOC; below file-count threshold despite >1,500 LOC.
**Size:** 4 files, 2,942 LOC.
**Entry point:** `elspeth = "elspeth.cli:app"` (pyproject `[project.scripts]`).
**Inbound dependencies (subsystem names only):** External (shell invocation) only.
**Outbound dependencies (subsystem names only):** `{contracts, core, engine}` plus L3↔L3 edges to `{plugins, tui, telemetry}` (and possibly `{mcp, web}`) — **deferred to L2**.
**Highest-risk concern (≤1 sentence):** L2-deep-dive candidate `cli.py` at **2,357 LOC** — single-file sprawl houses the entire Typer app plus the `TRANSFORM_PLUGINS` dict (KNOW-P22 explicitly couples plugin registration to this file); KNOW-A70 calls out single-file size as a quality risk.
**Confidence:** **High** — entry point, responsibility, and registry coupling cross-cited from KNOW-A12, KNOW-C34, KNOW-P22; ARCHITECTURE.md's ~2,200 LOC figure for "CLI" (KNOW-A12) is consistent with verified 2,942 (the four-file aggregation includes helpers/formatters).

## Closing

### Files ≥1,500 LOC observed at L1 (deep-dive candidates)

From `01-discovery-findings.md` top-10 list, attributed to subsystems:

| File | LOC | Subsystem |
|---|--:|---|
| `web/composer/tools.py` | 3,804 | web |
| `engine/orchestrator/core.py` | 3,281 | engine |
| `engine/processor.py` | 2,700 | engine |
| `cli.py` | 2,357 | cli |
| `core/config.py` | 2,227 | core |
| `core/dag/graph.py` | 1,968 | core |
| `core/landscape/execution_repository.py` | 1,750 | core |
| `web/composer/state.py` | 1,710 | web |
| `engine/coalesce_executor.py` | 1,603 | engine |
| `plugins/transforms/llm/azure_batch.py` | 1,592 | plugins |
| `core/landscape/data_flow_repository.py` | 1,590 | core (next-after-top-10) |
| `contracts/errors.py` | 1,566 | contracts (next-after-top-10) |

Twelve files exceed the threshold across five subsystems (`web`, `engine`, `cli`, `core`, `plugins`, `contracts`). All are L2-deep-dive candidates; none have been opened by this catalog wave.

### L3↔L3 edges intentionally not enumerated

Per Δ5 (no grep for the dependency graph) and Δ2 (depth cap), this catalog records L3↔L3 edges as "deferred to L2 dispatch wave" rather than guessing them. The L2 cluster pass — likely covering at minimum the `web/composer/` ↔ `composer_mcp/` relationship and the `plugins/` ↔ `telemetry/` relationship — is the correct place to enumerate those edges with import-graph tools.

### Doc tensions flagged (do NOT resolve, just record)

1. **Plugin count drift** — KNOW-A35 says "25 plugins across 4 categories" while KNOW-A72 (same document, summary section) says "46 plugins". The per-category enumeration in KNOW-A36–KNOW-A39 sums to 25. Knowledge-ingestion already surfaced this; left for a doc-correctness pass.
2. **ADR table staleness** — ARCHITECTURE.md's ADR table covers ADR-001 through ADR-006 only (KNOW-A62), but ADRs 007–017 are accepted (KNOW-ADR-007 through KNOW-ADR-017a). The architecture overview is at least one major iteration behind the ADR set.
3. **Schema-mode vocabulary drift** — PLUGIN.md describes schema modes as `dynamic`/`strict`/`free` in one table (KNOW-P23) but uses `observed`/`fixed`/`free` in the YAML examples (KNOW-P24). The runtime vocabulary is the truth; the prose is stale.
4. **Subsystem-LOC drift in ARCHITECTURE.md** — Verified counts diverge from KNOW-A* figures for several subsystems (contracts 17.4k vs ~8.3k, engine 17.4k vs ~12k, telemetry 2.9k vs ~1.2k). ARCHITECTURE.md is dated 2026-04-03 and the codebase has grown ~17% (KNOW-A6 ~103.9k vs verified 121.4k).
5. **`testing/` subsystem misidentification (KNOW-A18)** — ARCHITECTURE.md's "Testing subsystem ~9,500 LOC including ChaosLLM/ChaosWeb/ChaosEngine" describes `tests/` (out of scope per Δ6) rather than `src/elspeth/testing/` (verified 877 LOC, a single pytest-xdist plugin). Recorded under entry 10's `[DIVERGES FROM KNOW-A18]`.
