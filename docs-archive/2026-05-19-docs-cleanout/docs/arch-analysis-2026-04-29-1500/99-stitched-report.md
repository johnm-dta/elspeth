# ELSPETH Architecture — Stitched Synthesis Report

## §1 Executive summary

ELSPETH is a "domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines" [CLAUDE.md "Project Overview"], built around a three-tier trust model in which "the audit trail is the legal record" [CLAUDE.md "Auditability Standard"]. Every output must be traceable through the Landscape audit database back to source data, configuration, and code version. The system is ~121,392 production Python LOC distributed across 11 top-level subsystems under `src/elspeth/` [L1: 02-l1-subsystem-map.md §Closing], with a strict, CI-enforced 4-layer model (L0 contracts → L1 core → L2 engine → L3 application surfaces) that ran clean during this analysis [ORACLE: tier-model-oracle.txt "No bug-hiding patterns detected. Check passed."].

This synthesis is the apex of a hierarchical exploration: 11 L1 subsystems were inventoried [L1: 02-l1-subsystem-map.md §1–11] and partitioned into 5 L2 clusters (engine, core, composer, plugins, contracts), an L3 import-graph oracle was extracted as a deterministic JSON artefact [ORACLE: temp/l3-import-graph.json], the five clusters were analysed in parallel under cluster-isolation discipline (Δ L2-4), reconciliation was run across them [RECONCILED in temp/reconciliation-log.md], and the 54 cluster-level claims (15 confidence + 15 uncertainty + 24 cross-cluster) were curated into a single manifest [temp/synthesis-input-manifest.md] from which this report draws.

Five headline findings emerge. First, the 4-layer model is **mechanically clean at every L1 boundary** — zero `L1` upward-import violations and zero `TC` TYPE_CHECKING layer warnings inside any cluster's enforcer scope [CLUSTER:engine "Highest-confidence claims" item 1; CLUSTER:core §3 item 1; CLUSTER:contracts "Highest-confidence claims" item 1; CLUSTER:plugins §10 item 1]. Second, the **ADR-010 declaration-trust framework is end-to-end consistent**: contracts owns the L0 vocabulary (`AuditEvidenceBase` ABC, `@tier_1_error` registry, `DeclarationContract` 4-site framework with payload-schema H5 enforcement), engine implements a single dispatcher across 4 sites × 7 adopters, drift is locked by an AST-scanning unit test [CLUSTER:engine item 2; CLUSTER:contracts item 3]. Third, the **`web/*` sub-packages form a 7-node strongly-connected component** that is structurally load-bearing rather than accidental — the FastAPI app-factory pattern wires every sub-package's router into `web/app.py` and sub-packages reach back via `from elspeth.web.config import WebSettings`; decomposition is non-trivial [CLUSTER:composer §5 item 2; ORACLE: strongly_connected_components[4]; PHASE-0.5 §7.5 F4]. Fourth, **`plugins/infrastructure/` is the structural spine of the plugin ecosystem**, with `plugins/sinks → plugins/infrastructure` (weight 45) the heaviest single L3 edge in the codebase [ORACLE: edge weights; CLUSTER:plugins §10 item 1; PHASE-0.5 §7.5 F3]. Fifth, **the cross-cluster join is structurally sound**: zero contradictions surfaced across the five independently-produced cluster reports [RECONCILED in temp/reconciliation-log.md §Verdict].

Confidence in this synthesis is **High** at the L1 and L2 layer-conformance level (oracle-cited at every step, byte-equivalent on re-derivation), **High** for the cross-cluster handshakes (14 already-aligned, see §3), and **Medium-Low** for several deferred deep-dive questions (six files >1,500 LOC remain unread; see §5 and §8).

## §2 System anatomy

### §2.1 The 4-layer model

ELSPETH partitions all production Python into four layers with strictly downward import direction. Layer assignment is path-based and embedded in `scripts/cicd/enforce_tier_model.py:237–248` (LAYER_HIERARCHY 237–241 + LAYER_NAMES 243–248), enforced on every commit [ORACLE: tier-model-oracle.txt §LAYER SCHEMA].

**L0 — `contracts/`** is the leaf. It owns shared types, protocols, enums, errors, and frozen-dataclass primitives. Outbound dependencies are empty by construction; the leaf invariant is mechanically verified [CLUSTER:contracts item 1; ORACLE: tier-model-oracle.txt]. Inbound is permitted from any L1+, and concretely realised by `{core, engine, plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` [L1: 02-l1-subsystem-map.md §1].

**L1 — `core/`** holds the foundation primitives: Landscape (audit DB recorder + 4 repositories), DAG construction/validation, Dynaconf+Pydantic configuration, canonical-JSON hashing, payload store, retention, rate limiting, security, expression parser. Outbound is restricted to `{contracts}` [L1: 02-l1-subsystem-map.md §2]. Verified clean by `enforce_tier_model.py check --root src/elspeth/core` [CLUSTER:core §3 item 1].

**L2 — `engine/`** is the SDA execution layer: Orchestrator, RowProcessor, executors (transform/coalesce/pass-through), RetryManager, ArtifactPipeline, SpanFactory, Triggers. It owns run lifecycle and DAG execution per row/token. Outbound is `{core, contracts}`, verified by clean enforcer status [CLUSTER:engine item 1].

**L3 — application surfaces** comprises everything else: `plugins/`, `web/`, `mcp/`, `composer_mcp/`, `telemetry/`, `tui/`, `testing/`, and the `cli` root files. Outbound is unconstrained across `{contracts, core, engine, other L3}`; L3↔L3 edges are the topology that the Phase 0 oracle enumerates [ORACLE: temp/l3-import-graph.json].

### §2.2 The 11 L1 subsystems

The L1 inventory verified 11 subsystems [L1: 02-l1-subsystem-map.md §Conventions]:

1. **`contracts/`** (L0, COMPOSITE, 63 files, 17,403 LOC) — leaf primitives [L1: §1].
2. **`core/`** (L1, COMPOSITE, 49 files, 20,791 LOC) — foundation; six sub-packages [L1: §2].
3. **`engine/`** (L2, COMPOSITE, 36 files, 17,425 LOC) — Orchestrator + executors [L1: §3].
4. **`plugins/`** (L3, COMPOSITE, 98 files, 30,399 LOC) — largest subsystem; four sub-packages [L1: §4].
5. **`web/`** (L3, COMPOSITE, 72 files, 22,558 LOC Python-only) — FastAPI server, 8 backend sub-packages [L1: §5].
6. **`mcp/`** (L3, LEAF, 9 files, 4,114 LOC) — read-only Landscape audit-DB analyser [L1: §6].
7. **`composer_mcp/`** (L3, LEAF, 3 files, 824 LOC) — stateful pipeline-construction MCP [L1: §7].
8. **`telemetry/`** (L3, LEAF, 14 files, 2,884 LOC) — operational telemetry, post-Landscape [L1: §8].
9. **`tui/`** (L3, LEAF, 9 files, 1,175 LOC) — Textual TUI for `elspeth explain` [L1: §9].
10. **`testing/`** (L3, LEAF, 2 files, 877 LOC) — pytest plugin (`elspeth-xdist-auto`) [L1: §10].
11. **`cli` root files** (L3, LEAF, 4 files, 2,942 LOC) — Typer CLI [L1: §11].

Five subsystems are COMPOSITE per the Δ4 heuristic (≥4 sub-pkgs OR ≥10k LOC OR ≥20 files); six are LEAF [L1: 04-l1-summary.md §1]. Twelve individual files exceed the 1,500-LOC L2-deep-dive threshold and were flagged-not-opened at L1 depth [L1: 02-l1-subsystem-map.md §Closing]; six remain unread after L2 (see §5/§7/§8).

### §2.3 The 5 L2 clusters

The L2 dispatch reduced the 11 subsystems to 5 risk-weighted clusters [L1: 04-l1-summary.md §7.5]; each was analysed independently and produced a structured report with confidence, uncertainty, and cross-cluster sections.

The **engine cluster** verified that engine is layer-conformant (zero L1 / zero TC), that the ADR-010 dispatch surface is faithfully implemented across 4 sites × 7 adopters with a single dispatcher and an AST-scanning drift test, and that the terminal-state-per-token invariant is structurally guaranteed via `engine/executors/state_guard.py:NodeStateGuard` (a context-manager pattern, not a convention) [CLUSTER:engine "Highest-confidence claims" items 1–3].

The **core cluster** verified that `core/` is fully layer-conformant (clean enforcer with empty allowlist), that the Landscape sub-area matches the documented 4-repository facade (`landscape/__init__.py` re-exports `RecorderFactory` + the 4 repositories named in KNOW-A30, with 20 schema tables observed `[DIVERGES FROM KNOW-A24]`), and that a "Protocol-based no-op parity" pattern (`EventBus`/`NullEventBus`, `RateLimiter`/`NoOpLimiter`) is a deliberate offensive-programming discipline [CLUSTER:core §3 items 1–3].

The **composer cluster** (web/ + composer_mcp/, per Phase 0 F1) verified that `web/composer/state.py` (1,710 LOC) and `tools.py` (3,804 LOC) own the composer state machine with three internal consumers (composer_mcp/ weight 12, web/sessions/ weight 15, web/execution/ weight 9), that the 7-node SCC is the FastAPI app-factory pattern, and that the cluster has 0 inbound edges from any other cluster [CLUSTER:composer §5 items 1–3].

The **plugins cluster** verified that plugins/ is layer-conformant and structurally clean (intra-cluster edges all flow toward `infrastructure/`, F3 reading-order verified empirically), that trust-tier discipline is documented and structurally encoded (every source repeats "ONLY place coercion is allowed", every sink repeats "wrong types = upstream bug = crash", and the discipline is encoded in the `allow_coercion` config flag), and that SCC #1 (transforms/llm ↔ transforms/llm/providers) is a module-level provider-registry pattern with deferred runtime instantiation [CLUSTER:plugins §10 items 1–3].

The **contracts cluster** verified that the L0 leaf invariant is mechanically confirmed (zero outbound edges; layer-conformance JSON empty for both L1 and TC findings; KNOW-A53 stands), that ADR-006 phase artefacts are visible in-cluster and the post-relocation boundary is materially clean, and that the ADR-010 declaration-trust framework's L0 surface is complete (the engine-cluster bookmarks for `pipeline_runner` and the ADR-010 payload typedicts are closed by this pass) [CLUSTER:contracts "Highest-confidence claims" items 1–3].

### §2.4 The L3 import topology

The Phase 0 oracle enumerates 77 L3 edges across 33 nodes, with 5 strongly-connected components, zero TYPE_CHECKING-only edges, two conditional edges, and zero re-export edges [ORACLE: stats.total_edges=77, stats.scc_count=5, stats.type_checking_edges=0, stats.conditional_edges=2, stats.reexport_edges=0]. The largest SCC has 7 nodes (the web/* cluster); the other four are 2-node SCCs [ORACLE: stats.largest_scc_size=7].

```mermaid
flowchart LR
    subgraph _[.]
        _[".<br/><sub>2958 LOC</sub>"]
    end
    subgraph composer_mcp[composer_mcp]
        composer_mcp["composer_mcp<br/><sub>824 LOC</sub>"]
    end
    subgraph mcp[mcp]
        mcp["mcp<br/><sub>1661 LOC</sub>"]
        mcp_analyzers["mcp/analyzers<br/><sub>2453 LOC</sub>"]
    end
    subgraph plugins[plugins]
        plugins["plugins<br/><sub>8 LOC</sub>"]
        plugins_infrastructure["plugins/infrastructure<br/><sub>3804 LOC</sub>"]
        plugins_infrastructure_batching["plugins/infrastructure/batching<br/><sub>1024 LOC</sub>"]
        plugins_infrastructure_clients["plugins/infrastructure/clients<br/><sub>3790 LOC</sub>"]
        plugins_infrastructure_clients_retrieval["plugins/infrastructure/clients/retrieval<br/><sub>1031 LOC</sub>"]
        plugins_infrastructure_pooling["plugins/infrastructure/pooling<br/><sub>1133 LOC</sub>"]
        plugins_sinks["plugins/sinks<br/><sub>3515 LOC</sub>"]
        plugins_sources["plugins/sources<br/><sub>3519 LOC</sub>"]
        plugins_transforms["plugins/transforms<br/><sub>4125 LOC</sub>"]
        plugins_transforms_azure["plugins/transforms/azure<br/><sub>1125 LOC</sub>"]
        plugins_transforms_llm["plugins/transforms/llm<br/><sub>5740 LOC</sub>"]
        plugins_transforms_llm_providers["plugins/transforms/llm/providers<br/><sub>663 LOC</sub>"]
        plugins_transforms_rag["plugins/transforms/rag<br/><sub>922 LOC</sub>"]
    end
    subgraph telemetry[telemetry]
        telemetry["telemetry<br/><sub>1497 LOC</sub>"]
        telemetry_exporters["telemetry/exporters<br/><sub>1387 LOC</sub>"]
    end
    subgraph testing[testing]
        testing["testing<br/><sub>877 LOC</sub>"]
    end
    subgraph tui[tui]
        tui["tui<br/><sub>307 LOC</sub>"]
        tui_screens["tui/screens<br/><sub>352 LOC</sub>"]
        tui_widgets["tui/widgets<br/><sub>516 LOC</sub>"]
    end
    subgraph web[web]
        web["web<br/><sub>1072 LOC</sub>"]
        web_auth["web/auth<br/><sub>1224 LOC</sub>"]
        web_blobs["web/blobs<br/><sub>1952 LOC</sub>"]
        web_catalog["web/catalog<br/><sub>407 LOC</sub>"]
        web_composer["web/composer<br/><sub>8189 LOC</sub>"]
        web_composer_skills["web/composer/skills<br/><sub>85 LOC</sub>"]
        web_execution["web/execution<br/><sub>3748 LOC</sub>"]
        web_middleware["web/middleware<br/><sub>257 LOC</sub>"]
        web_secrets["web/secrets<br/><sub>1011 LOC</sub>"]
        web_sessions["web/sessions<br/><sub>4080 LOC</sub>"]
    end
    _ -->|7| plugins_infrastructure
    _ -->|2| plugins_sources
    _ --> telemetry
    _ -.->|cond| tui
    composer_mcp --> web
    composer_mcp --> web_catalog
    composer_mcp ==>|12| web_composer
    mcp -->|4| mcp_analyzers
    mcp_analyzers ==>|29| mcp
    plugins_infrastructure --> plugins_infrastructure_clients_retrieval
    plugins_infrastructure_clients_retrieval -->|3| plugins_infrastructure_clients
    plugins_sinks ==>|45| plugins_infrastructure
    plugins_sinks -->|4| plugins_infrastructure_clients
    plugins_sinks --> plugins_infrastructure_clients_retrieval
    plugins_sources ==>|17| plugins_infrastructure
    plugins_sources -->|5| plugins_infrastructure_clients
    plugins_transforms ==>|40| plugins_infrastructure
    plugins_transforms --> plugins_infrastructure_clients
    plugins_transforms_azure -->|6| plugins_infrastructure
    plugins_transforms_azure -->|2| plugins_infrastructure_batching
    plugins_transforms_azure --> plugins_infrastructure_clients
    plugins_transforms_azure -->|2| plugins_infrastructure_pooling
    plugins_transforms_azure -->|2| plugins_transforms
    plugins_transforms_llm ==>|17| plugins_infrastructure
    plugins_transforms_llm -->|2| plugins_infrastructure_batching
    plugins_transforms_llm -->|3| plugins_infrastructure_clients
    plugins_transforms_llm -->|4| plugins_infrastructure_pooling
    plugins_transforms_llm -->|5| plugins_transforms_llm_providers
    plugins_transforms_llm_providers ==>|12| plugins_infrastructure_clients
    plugins_transforms_llm_providers ==>|10| plugins_transforms_llm
    plugins_transforms_rag -->|4| plugins_infrastructure
    plugins_transforms_rag -->|9| plugins_infrastructure_clients_retrieval
    telemetry -->|2| telemetry_exporters
    telemetry_exporters ==>|18| telemetry
    testing -->|4| plugins_infrastructure
    tui -->|4| tui_screens
    tui_screens -->|2| tui
    tui_screens -->|2| tui_widgets
    tui_widgets -->|7| tui
    web --> plugins_infrastructure
    web -->|6| web_auth
    web -->|2| web_blobs
    web -->|3| web_catalog
    web -->|3| web_composer
    web -->|3| web_execution
    web -->|2| web_middleware
    web -->|5| web_secrets
    web -->|5| web_sessions
    web_auth -->|6| web
    web_auth --> web_middleware
    web_blobs --> web
    web_blobs -->|2| web_auth
    web_blobs -->|4| web_sessions
    web_catalog -->|3| plugins_infrastructure
    web_composer ==>|22| plugins_infrastructure
    web_composer -->|4| web
    web_composer -->|5| web_blobs
    web_composer -->|4| web_catalog
    web_composer -->|2| web_composer_skills
    web_composer -->|5| web_sessions
    web_execution -->|3| _
    web_execution -->|4| plugins_infrastructure
    web_execution -.->|cond| telemetry
    web_execution ==>|15| web
    web_execution -->|6| web_auth
    web_execution -->|5| web_blobs
    web_execution -->|9| web_composer
    web_execution -->|7| web_sessions
    web_secrets -->|6| web
    web_secrets -->|2| web_auth
    web_secrets --> web_sessions
    web_sessions -->|3| web
    web_sessions -->|2| web_auth
    web_sessions -->|2| web_blobs
    web_sessions ==>|15| web_composer
    web_sessions -->|2| web_execution
    web_sessions -->|2| web_middleware
```

The five SCCs are:

- **SCC#0** — `mcp` ↔ `mcp/analyzers` (2 nodes); analyser sub-package has weight-29 inverted dependency on parent `mcp/` namespace [ORACLE: strongly_connected_components[0]; edge mcp/analyzers → mcp weight 29].
- **SCC#1** — `plugins/transforms/llm` ↔ `plugins/transforms/llm/providers` (2 nodes); provider-registry pattern with deferred runtime instantiation [ORACLE: strongly_connected_components[1]; CLUSTER:plugins §10 item 3].
- **SCC#2** — `telemetry` ↔ `telemetry/exporters` (2 nodes); exporter sub-package re-uses telemetry types via weight-18 inverted edge [ORACLE: strongly_connected_components[2]; edge telemetry/exporters → telemetry weight 18].
- **SCC#3** — `tui` ↔ `tui/screens` ↔ `tui/widgets` (3 nodes); screens and widgets reach back into the tui root [ORACLE: strongly_connected_components[3]].
- **SCC#4** — `web` ↔ `web/auth` ↔ `web/blobs` ↔ `web/composer` ↔ `web/execution` ↔ `web/secrets` ↔ `web/sessions` (7 nodes); the FastAPI app-factory pattern, structurally load-bearing [ORACLE: strongly_connected_components[4]; CLUSTER:composer §5 item 2; PHASE-0.5 §7.5 F4].

For the Container view of cross-cluster coupling, see `99-cross-cluster-graph.md`.

### §2.5 The trust-tier model

ELSPETH's trust posture has three tiers, each with distinct handling rules [CLAUDE.md "Three-Tier Trust Model"]:

- **Tier 1 — Our data (Landscape audit DB / checkpoints)**: full trust, must be 100% pristine; bad data crashes immediately; "no coercion, no defaults, no silent recovery" [CLAUDE.md].
- **Tier 2 — Pipeline data (post-source)**: elevated trust; types are trustworthy because the source validated/coerced them; transforms and sinks expect conformance and do not coerce [CLAUDE.md].
- **Tier 3 — External data (source input)**: zero trust; "validate at the boundary, coerce where possible, record what we got" [CLAUDE.md]; absence is recorded as `None`, not fabricated.

The trust topology is structurally encoded in the layer model and the plugin discipline. Source plugins are the ONLY place coercion is allowed; transforms and sinks treat incoming row data as Tier 2 (no coercion, wrap operations on values); the Landscape audit DB at L1 treats stored data as Tier 1 (crash on any anomaly) [CLUSTER:plugins §10 item 2; CLUSTER:contracts item 1; CLUSTER:core §3 item 2]. The `enforce_tier_model.py` CI tool is named for this model and additionally enforces the layer hierarchy on import direction [ORACLE: tier-model-oracle.txt §INTERPRETATION].

## §3 Cross-cluster findings

### §3.1 Coupling surfaces

Cluster-level coupling is the structure of the L3 directed import graph plus the L0–L2 downward import flows. At the L3 boundary, the oracle records 77 unconditional runtime edges, 2 conditional edges, and 0 TYPE_CHECKING-only edges [ORACLE: stats.total_edges=77, stats.conditional_edges=2, stats.type_checking_edges=0; PHASE-0.5 §7.5 F5]. The two conditional edges are `. → tui` (cli root, gated on availability) and `web/execution → telemetry` [ORACLE: edges array; CLUSTER:composer §7 bullet 2].

A methodological clarification applies: the oracle counts only edges visible to runtime imports; TYPE_CHECKING-guarded references are excluded from edge enumeration entirely [RECONCILED in temp/reconciliation-log.md §R2]. The contracts cluster identified one such reference in-cluster (`plugin_context.py:31` imports `core.rate_limit.RateLimitRegistry` inside a `TYPE_CHECKING` block) directly from source, not as a tool-flagged warning [CLUSTER:contracts "Highest-uncertainty questions" item 1; CLUSTER:contracts "Cross-cluster observations for synthesis" bullet 1]. Both stand: the oracle's `type_checking_edges = 0` means no edge is TYPE_CHECKING-only at the L3 boundary; cluster catalogs may still surface annotation-only references as cluster-internal observations.

The dominant cross-cluster surface is `composer → plugins`. `web/composer` has a single weight-22 edge into `plugins/infrastructure`, the heaviest cross-cluster inbound to plugins/ [ORACLE: edge web/composer → plugins/infrastructure weight 22; CLUSTER:plugins §9 bullet 1; CLUSTER:composer §7 bullet 1]. The composer cluster does not import directly from `engine/` at the package-collapse granularity; it routes through `plugins/infrastructure/` and through plugin metadata, consistent with the "engine instantiates plugins via the registry" pattern [CLUSTER:composer §7 bullet 1; CLUSTER:plugins §10 item 1]. The cli root similarly couples to plugins via two edges (`. → plugins/infrastructure` weight 7, `. → plugins/sources` weight 2), confirming KNOW-P22's `TRANSFORM_PLUGINS` registry pattern [CLUSTER:plugins §9 bullet 2; ORACLE: edges from .].

### §3.2 Strongly-connected zones

Five SCCs sit at the L3 boundary; four are 2- or 3-node and confined within a single L1 subsystem (mcp, plugins/transforms/llm, telemetry, tui), while SCC#4 spans a 7-node cluster of `web/*` sub-packages [ORACLE: strongly_connected_components; PHASE-0.5 §7.5 F4]. The smaller SCCs share a structural cause: import-time registry or re-export patterns where a sub-package needs the parent namespace's types. The plugins cluster catalog frames its own SCC explicitly as a provider-registry with runtime decoupling cited from `transform.py:9-13`; the architecture-pack pass owns the break-or-keep decision [CLUSTER:plugins §10 item 3, §11 item 1]. Whether the four small SCCs share a common cause (registry pattern) or are incidental is a synthesis question that the plugins catalog flags but does not resolve [CLUSTER:plugins §9 bullet 4].

The 7-node web SCC (SCC#4) is qualitatively different. It is the FastAPI app-factory pattern made structural: `web/app.py:create_app(...)` outwardly imports every sub-package's router (the wiring leg), and sub-packages reach back via `from elspeth.web.config import WebSettings` and `run_sync_in_worker` (the shared-infrastructure leg) [CLUSTER:composer §5 item 2]. Both directions are intentional. The composer cluster's L2 reading at symbol level enriches (rather than contradicts) Phase 0 finding F1: "F1's 'thin transport' framing is correct but understates the structural role — `web/composer/` is the cluster's data backbone, not just an MCP target" [CLUSTER:composer §5 item 1; PHASE-0.5 §7.5 F1; RECONCILED in temp/reconciliation-log.md §R1]. The cycle is load-bearing in its current form, and decomposition is non-trivial — left to the architecture-pack pass [CLUSTER:composer §5 item 2].

### §3.3 Trust-boundary topology

The three trust tiers are not abstractions; they are mapped to concrete architectural boundaries. Tier 3 (external data, zero trust) lives at source plugin ingress; Tier 2 (pipeline data, elevated trust) flows through transforms and sinks; Tier 1 (audit data, full trust) lives in the Landscape audit DB and the L0 audit DTOs [CLAUDE.md "Three-Tier Trust Model"]. The plugins cluster verifies that the trust-tier discipline is documented, repeated, and structurally encoded: every source module repeats the "ONLY place coercion is allowed" contract; every sink module repeats the "wrong types = upstream bug = crash" contract; the discipline is also encoded in the `allow_coercion` config flag [CLUSTER:plugins §10 item 2].

The audit-side of the boundary is owned by the contracts and core clusters jointly. The contracts cluster confirms that L0 owns the audit DTO surface (`contracts/audit.py`, 922 LOC, header-only at L2 depth) — these are the row-level records that the Landscape persists [CLUSTER:contracts "Cross-cluster observations" bullet 3]. The core cluster confirms that the Landscape facade pattern is real: `landscape/__init__.py` re-exports `RecorderFactory` and exactly the four repositories named by KNOW-A30 (`DataFlowRepository`, `ExecutionRepository`, `QueryRepository`, `RunLifecycleRepository`); repositories are not re-exported through `core/__init__.py`, enforcing the encapsulation discipline that callers reach the Landscape via `RecorderFactory` rather than around it [CLUSTER:core §3 item 2]. The trust-boundary topology is therefore: Tier 3 → (sources coerce) → Tier 2 → (transforms/sinks pass through) → Tier 1 (Landscape via `RecorderFactory`).

### §3.4 Audit-trail completeness

The attributability test from CLAUDE.md states: "for any output, `explain(recorder, run_id, token_id)` must prove complete lineage back to source" [CLAUDE.md "Auditability Standard"]. Three cluster-level invariants jointly satisfy this. First, the engine cluster verifies that the **terminal-state-per-token invariant is structurally guaranteed**: `engine/executors/state_guard.py:NodeStateGuard` implements "every row reaches exactly one terminal state" as a context-manager pattern, locked by `test_state_guard_audit_evidence_discriminator.py` and `test_row_outcome.py` [CLUSTER:engine "Highest-confidence claims" item 3]. Second, the core cluster verifies that the Landscape facade pattern owns the audit write/read mechanics through the 4 repositories named in KNOW-A30, and that 20 schema tables persist the row/token lifecycle [CLUSTER:core §3 item 2]. Third, the contracts cluster verifies that the L0 audit DTO surface (`contracts/audit.py`) is complete and the L0/L2 split is clean — engine consumes the L0 vocabulary, core persists it [CLUSTER:contracts items 1 and 3].

The ADR-010 declaration-trust framework reinforces audit completeness end-to-end. Contracts owns the L0 vocabulary (`AuditEvidenceBase` ABC, `@tier_1_error` decorator + frozen registry, `DeclarationContract` 4-site framework with bundle types and payload-schema H5 enforcement); engine implements a 4-site × 7-adopter dispatcher with audit-complete (collect-then-raise) semantics, locked by an AST-scanning unit test [CLUSTER:contracts item 3; CLUSTER:engine item 2; KNOW-ADR-010e; KNOW-ADR-010f]. Two engine-cluster cross-cluster bookmarks (the `pipeline_runner` Protocol and the ADR-010 payload typedicts) are closed by this synthesis.

### §3.5 The plugin spine

`plugins/infrastructure/` is the structural spine of the plugin ecosystem. The oracle confirms three of the four heaviest L3 edges land on it: `plugins/sinks → plugins/infrastructure` weight 45 (the heaviest single L3 edge in the codebase), `plugins/transforms → plugins/infrastructure` weight 40, and `plugins/sources → plugins/infrastructure` weight 17 [ORACLE: edges array; PHASE-0.5 §7.5 F3]. The plugins cluster verifies F3's reading-order claim empirically: 23 intra-cluster edges all flow toward `infrastructure/`; sinks/sources/transforms are clients of it [CLUSTER:plugins §10 item 1].

The spine is also the dominant cross-cluster inbound destination. `web/composer → plugins/infrastructure` weight 22 is the heaviest cross-cluster inbound edge [ORACLE; CLUSTER:plugins §9 bullet 1; CLUSTER:composer §7 bullet 1]. The cli root (weight 7), `web/` root (weight 1), `web/catalog` (weight 3), `web/execution` (weight 4), and `testing` (weight 4) all land here too [ORACLE: edges to plugins/infrastructure; CLUSTER:plugins §9 bullets 2–3]. The synthesis question that survives the L2 pass: should the testing harness depend on `contracts/` protocols rather than on `plugins/infrastructure/` directly? [CLUSTER:plugins §9 bullet 3].

### §3.6 Configuration & contracts flow

Configuration flows from `settings.yaml` through `contracts/config/` (alignment, defaults, protocols, runtime) into `core/config.py` (the Dynaconf+Pydantic loader) and from there into `engine/` and `plugins/`. The contracts cluster verifies that ADR-006 phase artefacts are visible in-cluster: Phase 2's `hashing.py` extraction, Phase 4's `RuntimeServiceRateLimit` dataclass, and Phase 5's CI gate (`enforce_tier_model.py:237`); the post-relocation boundary is materially clean and KNOW-ADR-006a–d are ratified [CLUSTER:contracts item 2].

Two open questions surround this flow. First, the `core/` cluster pass produced an inventory of 50+ identifiers imported from `contracts/` (errors, payload protocols, freeze primitives, schema/schema_contract/secrets/security types, audit DTOs, checkpoint family, enums, types) [CLUSTER:core §6 Synthesis-1]; the contracts cluster's confidence claim that the L0 surface is complete is corroborated by this inventory at the boundary, but the contracts validator flagged a minor inventory-completeness gap (`guarantee_propagation.py`, `reorder_primitives.py` not enumerated in catalog Entry 14) [RECONCILED in temp/reconciliation-log.md §handshake table]. Second, `core/config.py` at 2,227 LOC is itself a deep-dive candidate; whether its internal structure factors cleanly (per-domain validator clusters) or has accreted by addition is a post-L2 question [CLUSTER:core §4 item 1].

The web composer composes with secrets through this flow as well. The runtime secret-ref resolver in `core/secrets.py` (124 LOC) is consumed by the web composer when threading `{"secret_ref": ...}` references through resolved configs [CLUSTER:core §6 Synthesis-4; CLUSTER:composer §6 item 2 — alignment, not contradiction]. The composer cluster's `web/secrets/` sub-package has zero outbound edges to other clusters at package-collapse granularity, suggesting credential flow happens via `WebSettings` injection at request time rather than via static imports — an L3 inspection question [CLUSTER:composer §6 item 2].

## §4 Highest-confidence system-level claims (top 5–7)

These are claims the synthesis can make that no single cluster could make alone.

- **S4.1** The 4-layer model is **mechanically clean across all five L2 cluster scopes**. Engine, core, and contracts each verified zero L1 / zero TC findings within their cluster; plugins verified the whole-tree run is clean; composer is unconstrained at the L3 layer but does not introduce upward imports [CLUSTER:engine item 1; CLUSTER:core §3 item 1; CLUSTER:contracts item 1; CLUSTER:plugins §10 item 1; ORACLE: tier-model-oracle.txt]. The layer model is not aspirational — it is structurally enforced and currently honoured.

- **S4.2** **ADR-010 declaration-trust is end-to-end consistent across L0 and L2.** The contracts cluster confirms the L0 vocabulary is complete (`AuditEvidenceBase` ABC, `@tier_1_error` registry, `DeclarationContract` 4-site framework, payload-schema H5 enforcement, secret-scrub last-line-of-defence); the engine cluster confirms a single dispatcher implements the 4 sites × 7 adopters with audit-complete (collect-then-raise) semantics, locked by an AST-scanning unit test [CLUSTER:contracts item 3; CLUSTER:engine item 2; KNOW-ADR-010e; KNOW-ADR-010i]. Two cross-cluster bookmarks (the `pipeline_runner` Protocol and the ADR-010 payload typedicts) are closed.

- **S4.3** **The 7-node web/* SCC is the FastAPI app-factory pattern, not accidental tangling.** The composer cluster's symbol-level analysis demonstrates that both legs (app.py outward to routers; sub-packages back to `WebSettings`/`run_sync_in_worker`) are intentional; the oracle confirms the cycle topology and the dispatch queue raised effort accordingly to "Very Large (5–7 hr)" [CLUSTER:composer §5 item 2; ORACLE: strongly_connected_components[4]; PHASE-0.5 §7.5 F4]. Decomposition decisions belong to the architecture pack, not to L2 archaeology.

- **S4.4** **`plugins/infrastructure/` is the spine of the plugin ecosystem and the dominant cross-cluster inbound destination.** Three of the four heaviest L3 edges land there (sinks weight 45, transforms 40, sources 17); the heaviest cross-cluster inbound is `web/composer → plugins/infrastructure` weight 22; the cli root, `web/catalog`, `web/execution`, and `testing` all couple here directly [ORACLE: edges; PHASE-0.5 §7.5 F3; CLUSTER:plugins §10 item 1, §9 bullet 1].

- **S4.5** **The terminal-state-per-token audit invariant is structurally guaranteed end-to-end.** Engine encodes it via `NodeStateGuard` (context-manager pattern with locking tests); core's Landscape facade persists exactly the 8 terminal/non-terminal states across the 20 schema tables; contracts owns the L0 audit DTO vocabulary they use [CLUSTER:engine item 3; CLUSTER:core §3 item 2; CLUSTER:contracts items 1 and 3]. The "every row reaches exactly one terminal state" rule is not a convention; it is mechanically enforced at the boundary between engine and core.

- **S4.6** **The composer cluster is a structural leaf in the import graph.** It has 0 inbound edges from any other cluster; only the two console-script entry points (`elspeth-web`, `elspeth-composer`) consume it [ORACLE: cross_cluster_inbound_edges=[]; CLUSTER:composer §5 item 3, §7 bullet 4]. Architectural changes to composer cannot break library callers elsewhere.

- **S4.7** **The cross-cluster reconciliation surfaced zero contradictions.** Five independently-produced cluster reports, working under cluster-isolation discipline (Δ L2-4), agreed at every named cross-cluster boundary; 14 cross-cluster handshakes verified aligned; 2 near-miss reconciliations (R1 F1 enrichment, R2 TYPE_CHECKING accounting) are methodological, not factual; 0 reconciliation entries triggered per Δ 8-6 [RECONCILED in temp/reconciliation-log.md §Verdict; §handshake table]. This is a notable structural-quality finding in itself.

## §5 Highest-uncertainty system-level questions (top 5–7)

Each entry names the cluster(s) raising it and what would resolve it.

- **S5.1** **Is the engine `processor.py` (2,700 LOC) cohesion essential or accidental complexity?** The docstring claims one cohesive responsibility (RowProcessor end-to-end); the LOC and import-section breadth (DAG navigation, retry classification, terminal-state assignment, ADR-009b cross-check, batch error handling, quarantine routing) suggest several. **Resolution:** an L3 deep-dive on `processor.py` is the priority-1 follow-up; without it KNOW-A70's quality-risk verdict cannot be refined to essential-vs-accidental [CLUSTER:engine "Highest-uncertainty questions" item 1].

- **S5.2** **Does `engine/declaration_dispatch.py:137,142` R6 silent-except behaviour honour the inline "every violation is recorded" claim?** The catalog's interpretation is that the swallowing is intentional aggregation (collect all violations, then raise); verification requires reading the dispatcher body and confirming via `test_declaration_dispatch.py` that both `DeclarationContractViolation` and `PluginContractViolation` actually arrive in the aggregation list rather than the silent-except branch. **Resolution:** test-debt audit; this is the highest-stakes verification gap in the engine cluster [CLUSTER:engine "Highest-uncertainty questions" item 2].

- **S5.3** **Does engine integration testing honour the CLAUDE.md/KNOW-C44 production-path rule?** No `tests/integration/engine/` directory exists, but the integration suite covers engine paths through other directories. The engine cluster cannot determine whether those tests use `ExecutionGraph.from_plugin_instances()` consistently or include `MockCoalesceExecutor`-style bypass patterns. **Resolution:** a cross-cluster integration-tier audit, distinct from any single L2 cluster [CLUSTER:engine "Highest-uncertainty questions" item 3, "Cross-cluster observations" bullet 5].

- **S5.4** **Does `core/config.py` (2,227 LOC) factor cleanly internally, or has it accreted by addition?** Pydantic settings concentrate for cross-validation reasons, but 2,227 LOC of single-file configuration is substantial; 12+ child dataclasses span checkpoint, concurrency, database, landscape, payload-store, rate-limit, retry, secrets, sinks, sources, transforms. **Resolution:** L3 deep-dive on `config.py` followed by an architecture-pack proposal (split or keep) [CLUSTER:core §4 item 1].

- **S5.5** **What is `core/dag/graph.py` (1,968 LOC)'s actual blast radius and test-lock coverage?** `ExecutionGraph` is consumed by every executor in `engine/`, by `web/composer/_semantic_validator.py`, by `web/execution/validation.py`, by `core/checkpoint/{manager,compatibility}`, and indirectly by every plugin via the schema-contract validation flow. **Resolution:** L3 deep-dive on `graph.py`'s public-contract test surface; the existing `test_graph.py` and `test_graph_validation.py` files exist but their assertion density vs the file's behavioural surface is unknown [CLUSTER:core §4 item 3].

- **S5.6** **Should `web/composer/tools.py` (3,804 LOC) and `web/composer/state.py` (1,710 LOC) be decomposed, and along which seams?** Together with the 7-node web SCC, these are the system's largest concentration of composer logic. **Resolution:** L3 deep-dive on both files paired with the SCC#4 architecture-pack decision; isolated decomposition without the SCC context risks producing a worse cycle [CLUSTER:composer §5 item 1; PHASE-0.5 §7.5 F4].

- **S5.7** **Is the trust-tier discipline structurally enforced at runtime, not just documented?** Verbal/structural enforcement is in place at every plugin module; cross-cluster invariant tests (e.g., a fixture that injects a transform observed to coerce and asserts the run fails) are not. **Resolution:** a runtime-probe test suite paired with an architecture-pack decision on whether the discipline should be CI-enforced beyond static patterns [CLUSTER:plugins §11 item 2].

- **S5.8** **Are `errors.py` (1,566 LOC) and `plugin_context.py:31` TYPE_CHECKING reference candidates for the ADR-006d Violation #11 protocol?** The `errors.py` file holds Tier-1 raiseable exceptions, Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported `FrameworkBugError`; the Tier-1/Tier-2 distinction is currently encoded by inline comments, not by file split. The TYPE_CHECKING smell at `plugin_context.py:31` is the cluster's only cross-layer reference and is the strongest L1-Q1 evidence the L2 pass surfaced. **Resolution:** architecture-pack remediation per ADR-006d ("move down → extract primitive → restructure caller → never lazy-import") [CLUSTER:contracts "Highest-uncertainty questions" items 1 and 2; KNOW-ADR-006d].

## §6 Reconciled tensions

**R1 — F1 framing depth (composer cluster enriches Phase 0 F1).** Phase 0 finding F1 (the L1 amendment that `composer_mcp/` is coupled to `web/composer/` rather than an independent sibling) was framed as a single L3 oracle edge of weight 12. The composer cluster's L2 reading at symbol level enriches this: `composer_mcp/server.py:1–40` imports state types directly, and "F1's 'thin transport' framing is correct but understates the structural role — `web/composer/` is the cluster's data backbone, not just an MCP target" [CLUSTER:composer §5 item 1; PHASE-0.5 §7.5 F1]. This is enrichment, not contradiction. Both stand: F1 anchors the cross-cluster topology in §3.2 and §3.5; composer's enrichment supplies the SCC's internal-structure depth claim. No catalog amendment is required [RECONCILED in temp/reconciliation-log.md §R1].

**R2 — TYPE_CHECKING accounting methodology (contracts cluster vs Phase 0 F5).** F5 reports `stats.type_checking_edges = 0` (out of 77 edges) as evidence that ~97% of L3 edges are unconditional runtime coupling. The contracts cluster identifies `plugin_context.py:31` as the only cross-layer TYPE_CHECKING import in-cluster, framed as a candidate for ADR-006d remediation. Strict reading would suggest tension; methodologically there is none. The oracle excludes TYPE_CHECKING-guarded edges from enumeration entirely (per extraction methodology); cluster catalogs reading source directly may, and should, surface them as cluster-internal observations. The synthesis carries this clarification at §3.1: oracle edges and TYPE_CHECKING annotations are different visibility tiers; both stand. No catalog amendment is required [RECONCILED in temp/reconciliation-log.md §R2; CLUSTER:contracts "Highest-uncertainty questions" item 1].

No additional cross-cluster contradictions surfaced; the five L2 cluster reports were structurally consistent at the boundaries they describe — see `temp/reconciliation-log.md` §"Already-aligned cross-cluster handshakes" for the 14 already-aligned cross-cluster claims verified during the scan.

## §7 Architectural debt candidates

Catalogued, not remediated. Severity reflects the originating cluster's own assessment.

- **S7.1 — engine `processor.py` 2,700 LOC.** Quality risk per KNOW-A70; cohesion question is open. Severity: cluster-flagged quality risk. Scope: single-cluster (engine). Already known: yes [KNOW-A70]. [CLUSTER:engine "Highest-uncertainty questions" item 1].

- **S7.2 — engine `declaration_dispatch.py:137,142` R6 silent-except.** Test-debt candidate #3, highest-stakes verification gap in engine. Severity: high (test-debt; verification gap on audit-completeness claim). Scope: single-cluster. Newly surfaced by L2 [CLUSTER:engine "Highest-uncertainty questions" item 2].

- **S7.3 — engine integration testing audit gap.** No `tests/integration/engine/` directory; KNOW-C44 production-path rule cannot be verified from within the engine cluster. Severity: high (audit-trail integrity at the test-architecture level). Scope: cross-cluster (integration-tier audit). Newly surfaced [CLUSTER:engine "Highest-uncertainty questions" item 3].

- **S7.4 — core `config.py` 2,227 LOC.** Single-file Pydantic settings holding 12+ child dataclasses plus `load_settings()`. Severity: cluster-flagged structural concern (essential-vs-accidental open). Scope: single-cluster. Already known: yes [KNOW-A70]. [CLUSTER:core §4 item 1].

- **S7.5 — core `dag/graph.py` 1,968 LOC, cascade-prone.** Public-contract test surface unknown; consumed by every executor + composer + execution + checkpoint paths. Severity: cascade-prone (P3 per the cluster's own §7 framing). Scope: system-wide blast radius. Already known: yes [KNOW-A70]. [CLUSTER:core §4 item 3].

- **S7.6 — composer cluster's 7-node SCC is structurally load-bearing.** SCC#4 cannot be acyclically decomposed without architectural work; effort raised UP to "Very Large" in the dispatch queue. Severity: high (decomposition decision is non-trivial). Scope: cross-cluster (entire web/* SCC). Newly characterised at this depth by L2 [CLUSTER:composer §5 item 2; PHASE-0.5 §7.5 F4].

- **S7.7 — composer `tools.py` 3,804 LOC + `state.py` 1,710 LOC.** Joint-largest concentration of composer logic; symbol-level decomposition is paired with the SCC#4 decision. Severity: cluster-flagged (largest single file in the tree). Scope: single-cluster (composer). Already known: yes [KNOW-A70]. [CLUSTER:composer §5 item 1].

- **S7.8 — plugins SCC #1 module-level cycle (`plugins/transforms/llm` ↔ `plugins/transforms/llm/providers`).** Provider-registry pattern; runtime decoupling is in place; the architecture pack will need to compare the cost of moving shared types into `infrastructure/` versus leaving the cycle visible. Severity: low-medium (module-level only). Scope: single-cluster. Newly characterised at this depth [CLUSTER:plugins §10 item 3, §11 item 1].

- **S7.9 — plugins doc-rot (KNOW-A35 25 vs KNOW-A72 46 vs verified 29).** Four post-doc plugins were added without a doc update. Severity: doc-correctness (governance question). Scope: documentation set, not architecture. Already known: yes (governance-drift candidate) [CLUSTER:plugins §11 item 3].

- **S7.10 — contracts `errors.py` 1,566 LOC.** Tier-1 raiseable exceptions, Tier-2 frozen audit DTOs, structured-reason TypedDicts, re-exported `FrameworkBugError` mixed in one file. Severity: cluster-flagged structural concern; CI-enforced split (e.g., `errors_tier1.py` vs `errors_dtos.py`) would mechanise the discipline currently encoded in inline comments. Scope: single-cluster. Newly surfaced [CLUSTER:contracts "Highest-uncertainty questions" item 2].

- **S7.11 — contracts `plugin_context.py:31` TYPE_CHECKING smell.** The only cross-layer reference in the cluster; ADR-006d Violation #11 candidate (extracted `RateLimitRegistryProtocol` in `contracts.config.protocols` would eliminate the TYPE_CHECKING block). Severity: low (annotation-only; ADR-006d-shaped fix is well-understood). Scope: single-cluster, with cross-cluster signal. Already-known protocol: yes [KNOW-ADR-006d]. [CLUSTER:contracts "Highest-uncertainty questions" item 1].

- **S7.12 — contracts schema_contract sub-package promotion.** Catalog Entry 8 (8 files, ~3,500 LOC) has high internal cohesion; promoting to `contracts/schema_contracts/` sub-package would mirror the `config/` partition. Severity: organisational hygiene (non-blocking). Scope: single-cluster [CLUSTER:contracts "Highest-uncertainty questions" item 3].

- **S7.13 — doc-correctness items.** The 20 vs 21 audit tables divergence (`[DIVERGES FROM KNOW-A24]` in core), plugin-count drift (KNOW-A35 25 vs KNOW-A72 46 vs verified 29), and the 10-citation editorial defect in the contracts cluster catalog all flow to a doc-correctness pass. Severity: documentation correctness, not architecture. Scope: cross-cluster (knowledge map maintenance). [RECONCILED in temp/reconciliation-log.md §"Already-resolved divergences"; CLUSTER:core §6 Synthesis-2].

## §8 Open architectural questions

These are questions the synthesis cannot answer from its input set; each names the input gap and the resolution path.

- **S8.1 — Q1 (still-open after Phase 0):** Responsibility cut between `contracts/` (L0) and `core/` (L1) post-ADR-006. The structural side (no upward imports) is verified; the semantic side (should the responsibility cut be different?) requires reading both clusters' file-level details. Specific candidate raised by core: `core/secrets.py` (124 LOC, runtime resolver) lives at `core/` root while `core/security/{secret_loader,config_secrets}.py` (529 LOC combined) live in the subpackage. Resolution: architecture-pack pass owning the boundary [L1: 04-l1-summary.md §7.5 Still open Q1; CLUSTER:core §4 item 2].

- **S8.2 — Q4 (still-open after Phase 0):** Plugin-count drift in ARCHITECTURE.md (KNOW-A35 = 25 vs KNOW-A72 = 46 vs verified 29). Not a graph question; requires a doc-correctness pass [L1: 04-l1-summary.md §7.5 Still open Q4; CLUSTER:plugins §11 item 3].

- **S8.3 — Q5 (still-open after Phase 0):** Whether the engine LOC concentration (`orchestrator/core.py` 3,281 + `processor.py` 2,700 + `coalesce_executor.py` 1,603 = ~43% of engine in three files) is structural (orchestrator doing too much) or accidental (large but cleanly factored). The L2 engine cluster pass touched the question but explicitly left it for L3 deep-dive [L1: 04-l1-summary.md §7.5 Still open Q5; CLUSTER:engine "Highest-uncertainty questions" item 1].

- **S8.4 — L3 deep-dive candidates flagged by L2.** Six files (`tools.py` 3804, `state.py` 1710, `config.py` 2227, `dag/graph.py` 1968, `processor.py` 2700, `errors.py` 1566) and one file referenced by KNOW-A70 (`orchestrator/core.py` 3281) remain unread. Resolution: axiom-system-archaeologist deep-dive passes per file. None of these are architecture-pack questions until they have been read [CLUSTER:engine §"uncertainty"; CLUSTER:core §4; CLUSTER:composer §5 item 1; CLUSTER:contracts "Highest-uncertainty questions" item 2].

- **S8.5 — `web/execution → .` (root) edge purpose.** The composer cluster cannot diagnose this weight-3 edge without source inspection; if it is a re-export of a public symbol from `elspeth/__init__.py`, benign; if it is a deferred-import hack to bypass an explicit cluster dependency, that is a different finding. Resolution: L3 deep-dive [CLUSTER:composer §6 item 1; ORACLE: edge web/execution → . weight 3].

- **S8.6 — Composer secrets credential flow.** `web/secrets/` has zero outbound edges to other clusters at package-collapse granularity, yet composer/execution rely on LLM-provider credentials. Whether the credential flow happens via `WebSettings` injection at request time (not visible to the import graph) or some other mechanism is L3 inspection territory [CLUSTER:composer §6 item 2; CLUSTER:core §6 Synthesis-4].

- **S8.7 — `web/sessions → web/composer` weight 15 edge direction.** The composer cluster interprets this as "sessions persists composer drafts" based on file names, but symbol-level evidence has not been inspected. Confirming the data-flow direction is L3 scope [CLUSTER:composer §6 item 3].

## §9 Recommended downstream packs

- **axiom-system-architect** — owns architecture critique and the improvement roadmap. Specifically: SCC#4 decomposition decision (motivated by §3.2 / §7.6); SCC #1 break-or-keep decision (§3.2 / §7.8); `errors.py` Tier-1/Tier-2 split (§7.10); `config.py` split or keep (§7.4); `plugin_context.py:31` ADR-006d Violation #11 remediation (§7.11); engine LOC-concentration triage (§5.1 / §8.3).

- **ordis-security-architect** — owns threat modelling. Scope: trust-tier topology end-to-end (§3.3); audit-trail completeness across the engine→core→contracts join (§3.4); ADR-010 declaration-trust verification at the dispatcher's audit-complete boundary (§5.2); the credential-flow question for composer secrets (§8.6). The audit-trail and trust-tier surfaces are the highest-stakes security territory because "the audit trail is the legal record".

- **axiom-system-archaeologist deep-dives (per file)** — six (or seven) L3 candidates flagged by L2: `web/composer/tools.py` (3,804 LOC), `web/composer/state.py` (1,710 LOC), `core/config.py` (2,227 LOC), `core/dag/graph.py` (1,968 LOC), `engine/processor.py` (2,700 LOC), `contracts/errors.py` (1,566 LOC), and `engine/orchestrator/core.py` (3,281 LOC, KNOW-A70 named, not in cluster manifest). Pre-requisites for the architecture pack's decomposition decisions (§7.4–§7.7, §7.10).

- **Doc-correctness pass** — owns the three editorial backlog items: (1) the 20-vs-21 audit-tables divergence in ARCHITECTURE.md (`[DIVERGES FROM KNOW-A24]`); (2) plugin-count drift (KNOW-A35 25 vs KNOW-A72 46 vs verified 29); (3) the contracts cluster catalog's 10 mis-cited KNOW-A* references (citation IDs resolve but inline rationales mismatch). Distinct from architecture work; must not be bundled into any L2 cluster.

- **Cross-cluster integration-tier audit** — owns S7.3 / S5.3 (engine integration testing's KNOW-C44 production-path conformance). Distinct from any single L2 cluster because the integration suite spans cluster boundaries.

## §10 Provenance & confidence ledger

| Claim ID | Section | Sources cited | Confidence | Why |
|----------|---------|---------------|------------|-----|
| S3.1.1 | §3.1 | ORACLE stats; PHASE-0.5 §7.5 F5 | High | Oracle JSON fields directly cited |
| S3.1.2 | §3.1 | RECONCILED §R2; CLUSTER:contracts cross-cluster bullet 1; ORACLE methodology | High | Methodological clarification, both sources confirm |
| S3.1.3 | §3.1 | ORACLE edge weight 22; CLUSTER:plugins §9 bullet 1; CLUSTER:composer §7 bullet 1 | High | Three independent sources agree |
| S3.1.4 | §3.1 | CLUSTER:composer §7 bullet 1; CLUSTER:plugins §10 item 1 | High | Two cluster sources + oracle topology |
| S3.1.5 | §3.1 | CLUSTER:plugins §9 bullet 2; ORACLE edges from `.`; KNOW-P22 | High | Multiple cluster + oracle |
| S3.2.1 | §3.2 | ORACLE strongly_connected_components; PHASE-0.5 §7.5 F4 | High | Oracle-direct |
| S3.2.2 | §3.2 | CLUSTER:plugins §10 item 3, §11 item 1; CLUSTER:plugins §9 bullet 4 | High | Cluster source + cross-cluster question framing |
| S3.2.3 | §3.2 | CLUSTER:composer §5 item 2; CLUSTER:composer §5 item 1; PHASE-0.5 §7.5 F1; RECONCILED §R1 | High | Multi-cluster + oracle + reconciliation |
| S3.3.1 | §3.3 | CLAUDE.md "Three-Tier Trust Model"; CLUSTER:plugins §10 item 2 | High | System docs + cluster verification |
| S3.3.2 | §3.3 | CLUSTER:contracts cross-cluster bullet 3; CLUSTER:core §3 item 2 | High | Two cluster sources |
| S3.4.1 | §3.4 | CLUSTER:engine item 3; CLUSTER:core §3 item 2; CLUSTER:contracts items 1,3 | High | Three cluster sources |
| S3.4.2 | §3.4 | CLUSTER:contracts item 3; CLUSTER:engine item 2; KNOW-ADR-010e/f/i | High | Cluster sources + ADR knowledge |
| S3.5.1 | §3.5 | ORACLE edges; PHASE-0.5 §7.5 F3; CLUSTER:plugins §10 item 1 | High | Oracle-direct + cluster |
| S3.5.2 | §3.5 | ORACLE; CLUSTER:plugins §9 bullets 1–3; CLUSTER:composer §7 bullet 1 | High | Multi-source |
| S3.5.3 | §3.5 | CLUSTER:plugins §9 bullet 3 | Medium | Single cluster source raises the question |
| S3.6.1 | §3.6 | CLUSTER:contracts item 2 | Medium | Single cluster source for ADR-006 phase artefacts |
| S3.6.2 | §3.6 | CLUSTER:core §6 Synthesis-1; RECONCILED §handshake table; CLUSTER:core §4 item 1 | High | Cluster + reconciliation |
| S3.6.3 | §3.6 | CLUSTER:core §6 Synthesis-4; CLUSTER:composer §6 item 2 | Medium | Two cluster sources, alignment not contradiction |
| S4.1 | §4 | CLUSTER:engine item 1; CLUSTER:core §3 item 1; CLUSTER:contracts item 1; CLUSTER:plugins §10 item 1; ORACLE | High | All five clusters + oracle |
| S4.2 | §4 | CLUSTER:contracts item 3; CLUSTER:engine item 2; KNOW-ADR-010e/i | High | Two cluster sources + ADR knowledge |
| S4.3 | §4 | CLUSTER:composer §5 item 2; ORACLE strongly_connected_components[4]; PHASE-0.5 §7.5 F4 | High | Cluster + oracle + L1 amendment |
| S4.4 | §4 | ORACLE edges; PHASE-0.5 §7.5 F3; CLUSTER:plugins §10 item 1, §9 bullet 1 | High | Oracle + cluster |
| S4.5 | §4 | CLUSTER:engine item 3; CLUSTER:core §3 item 2; CLUSTER:contracts items 1,3 | High | Three cluster sources |
| S4.6 | §4 | ORACLE cross_cluster_inbound_edges=[]; CLUSTER:composer §5 item 3, §7 bullet 4 | High | Oracle-direct + cluster |
| S4.7 | §4 | RECONCILED §Verdict; RECONCILED §handshake table | High | Reconciliation log direct |
| S5.1 | §5 | CLUSTER:engine "uncertainty" item 1 | Medium | Single cluster source raises the question; KNOW-A70 corroborates |
| S5.2 | §5 | CLUSTER:engine "uncertainty" item 2 | Medium | Single cluster source; high-stakes verification gap |
| S5.3 | §5 | CLUSTER:engine "uncertainty" item 3, "Cross-cluster" bullet 5 | Medium | Single cluster + cross-cluster bullet |
| S5.4 | §5 | CLUSTER:core §4 item 1 | Medium | Single cluster source |
| S5.5 | §5 | CLUSTER:core §4 item 3 | Medium | Single cluster source; cascade scope is multi-cluster |
| S5.6 | §5 | CLUSTER:composer §5 item 1; PHASE-0.5 §7.5 F4 | High | Cluster + L1 amendment |
| S5.7 | §5 | CLUSTER:plugins §11 item 2 | Medium | Single cluster source raises the question |
| S5.8 | §5 | CLUSTER:contracts "uncertainty" items 1,2; KNOW-ADR-006d | Medium | Cluster + ADR knowledge |
| S6.R1 | §6 | RECONCILED §R1; CLUSTER:composer §5 item 1; PHASE-0.5 §7.5 F1 | High | Reconciliation log direct |
| S6.R2 | §6 | RECONCILED §R2; CLUSTER:contracts cross-cluster bullet 1; ORACLE | High | Reconciliation log direct |
| S7.1 | §7 | CLUSTER:engine "uncertainty" item 1; KNOW-A70 | High | Cluster + already-known |
| S7.2 | §7 | CLUSTER:engine "uncertainty" item 2 | Medium | Single cluster source; newly surfaced |
| S7.3 | §7 | CLUSTER:engine "uncertainty" item 3 | Medium | Single cluster source; cross-cluster scope |
| S7.4 | §7 | CLUSTER:core §4 item 1; KNOW-A70 | High | Cluster + already-known |
| S7.5 | §7 | CLUSTER:core §4 item 3; KNOW-A70 | High | Cluster + already-known |
| S7.6 | §7 | CLUSTER:composer §5 item 2; PHASE-0.5 §7.5 F4 | High | Cluster + L1 amendment |
| S7.7 | §7 | CLUSTER:composer §5 item 1; KNOW-A70 | High | Cluster + already-known |
| S7.8 | §7 | CLUSTER:plugins §10 item 3, §11 item 1 | Medium | Single cluster source; well-characterised |
| S7.9 | §7 | CLUSTER:plugins §11 item 3 | Medium | Single cluster source; doc-correctness scope |
| S7.10 | §7 | CLUSTER:contracts "uncertainty" item 2 | Medium | Single cluster source; newly surfaced |
| S7.11 | §7 | CLUSTER:contracts "uncertainty" item 1; KNOW-ADR-006d | Medium | Cluster + ADR knowledge |
| S7.12 | §7 | CLUSTER:contracts "uncertainty" item 3 | Medium | Single cluster source; organisational hygiene |
| S7.13 | §7 | RECONCILED §"Already-resolved divergences"; CLUSTER:core §6 Synthesis-2 | High | Reconciliation log + cluster |
