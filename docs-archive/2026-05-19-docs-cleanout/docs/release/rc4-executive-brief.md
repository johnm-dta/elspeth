# RC 4.0 Planning Brief — Semi-Autonomous Pipeline Platform

> **ARCHIVED — Planning brief captured at 3 March 2026 (RC-4.0 planning).**
> This document records the planned RC-4 work package as it stood at 3 March 2026.
> Scope evolved during execution; the semi-autonomous-platform deliverables described here ultimately shipped as part of the **RC-5 cut on 3 April 2026** (Web UX Composer + Composer MCP server + audited tool loop). The version-number framing ("RC 4 / v4.0.0") in this document does not match how the work was eventually released.
>
> **For what actually shipped, see** [`../elspeth-progress-rc1-to-rc5.md`](../elspeth-progress-rc1-to-rc5.md) (Periods 4–5) and [`../../../CHANGELOG.md`](../../../CHANGELOG.md) ([0.4.0], [0.4.1], [0.5.0] sections). A mapping from each planned feature in this brief to its actual RC-5 outcome appears in the *What actually shipped* appendix at the end of this document.

**Date:** 2026-03-03
**Release:** RC 4 (v4.0.0) — **superseded; work shipped as RC-5 (v0.5.0) on 3 April 2026**
**Status:** Planning complete *(superseded — see banner above)*
**Prepared by:** Architecture analysis, synthesized from design document and Filigree work package
**Audience:** Project stakeholders evaluating scope, sequencing, and risk for the planned 4.0 release — preserved for historical context
**Register:** Technical / engineering-stakeholder

---

## What RC 4.0 Delivers

A non-technical user describes a data processing task in natural language. The system generates a complete ELSPETH pipeline, presents it as a visual graph for review, and executes it with **full audit trail guarantees** — identical to any hand-written pipeline.

**Core invariant:** The semi-autonomous layer is a configuration generator. Once the config is produced, the standard ELSPETH engine executes it with zero relaxation of audit, lineage, or trust tier guarantees. There is no "auto-generated pipeline" mode that cuts corners.

**Plugin exploitation, not generation:** The LLM composes pipelines from the existing plugin library. It does not generate code.

---

## Work Package Summary

The 4.0 work package comprises **9 new features** (the semi-autonomous platform) plus **5 existing items** pulled forward from the backlog as enablers. Two Future items (server mode, visual pipeline designer) were closed as superseded — their intent is absorbed by the new features.

| Category | Count | Effort |
|----------|-------|--------|
| Semi-autonomous features | 9 | 2M + 5L + 2XL |
| Enablers (pulled from backlog) | 5 | 3M + 2L |
| Closed as superseded | 2 | — |
| **Total active items** | **14** | |

---

## Feature Map

### Semi-Autonomous Platform (new)

| # | Feature | Size | Summary |
|---|---------|------|---------|
| 1 | Engine API Extraction | M | Programmatic pipeline interface decoupled from CLI |
| 2 | Pipeline Composition API | L | Tool interface for step-by-step pipeline building (LLM-independent) |
| 3 | LLM Pipeline Composer | L | Agentic tool-use loop with decision-space prompt engineering |
| 4 | Conversation Service | L | FastAPI chat API, config store, workflow integration |
| 5 | Review Classification & Meta-Audit | L | Plugin trust tiers (transparent → approval_required), fail-closed |
| 6 | Workflow & Worker Infrastructure | L | Temporal orchestration, K8s worker pods, crash recovery |
| 7 | Real-Time Telemetry Pipeline | M | Redis exporter + WebSocket gateway for live execution |
| 8 | Frontend | XL | React Flow graph editor, summary reports, live visualization |
| 9 | Shared Storage & Task Database | M | PostgreSQL lifecycle DB, S3/Azure blob storage |

### Enablers (pulled forward from backlog)

| Item | Original home | Why 4.0 needs it |
|------|---------------|-------------------|
| Plugin registry pattern | Architecture Refactoring | Discovery tools (Epic 2) need plugin self-registration, not if/elif dispatch |
| LLM template rewrite | Template & Plugin | Composer (Epic 3) generates pipelines for any domain — templates must be topic-agnostic |
| Telemetry exporter cleanup | Architecture Refactoring | Redis exporter (Epic 7) needs shared serialization and per-exporter circuit breakers |
| Recorder facade evaluation | Architecture Refactoring | API extraction (Epic 1) should resolve facade-vs-DI before building the programmatic surface |
| Config decomposition | Configuration & Tooling | API extraction (Epic 1) decouples config loading — cleaner as separate submodules |

---

## Dependency Graph and Sequencing

```
                READY NOW (5 parallel starting points)
                ┌─────────────────────────────────────┐
                │                                     │
    Recorder facade eval ─┐     Plugin registry ────────────────┐
    Config decomposition ─┴→ Engine API (1) ─┬→ Composition API (2) ─┬→ LLM Composer (3) ──→ Conversation (4) ─┬→ Workflow (6) → Telemetry (7) ─┐
                                             │                       │                                         │                               │
                                             │                       └→ Review Classification (5) ─────────────┼───────────────────────────────→ Frontend (8)
                                             │                                                                 │                               │
                                             └→ Shared Storage (9)              LLM template rewrite ──────────┘   Telemetry cleanup ──────────┘
```

**Critical path length:** 8 items (facade eval → Engine API → Composition API → Composer → Conversation → Workflow → Telemetry → Frontend)

**Parallelism available:** 5 items can start immediately with zero contention. After Engine API lands, 3 independent tracks open (Composition API chain, Shared Storage, Workflow track).

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pipeline composition model | **Tool-use** (not free-form YAML generation) | LLM uses structured tool calls for per-step validation; prompt engineering focuses on decisions, not formatting |
| Plugin review enforcement | **Fail-closed** — unclassified plugins default to approval_required | Generated pipelines must not bypass review for untested plugin combinations |
| Telemetry bridge | **Redis pub/sub + WebSocket** (implements existing TelemetryExporter protocol) | Zero engine changes — plugs into existing telemetry infrastructure |
| Workflow orchestration | **Temporal** | Durable execution, crash recovery, approval gates, native heartbeat support |
| Frontend framework | **React Flow** (ComfyUI-inspired) | Mature DAG visualization library with custom node support |

---

## Risk Profile

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM generates invalid pipeline configs | Medium | Tool-use model validates per-step; submit_pipeline runs full DAG + schema check before finalization |
| Critical path length (8 deep) | Medium | 5 parallel starting points reduce wall-clock time; enablers can start before any new feature code |
| Prompt engineering iteration cycles | Medium | Decision-space focus (plugin selection, routing, field wiring) is more constrained than open-ended generation |
| New infrastructure dependencies (Temporal, Redis, K8s, PostgreSQL) | High | Scoped to semi-autonomous platform only — core ELSPETH remains dependency-light (SQLite, local filesystem) |
| Frontend is blocked by 6 predecessors | Low | Frontend work (component library, design system) can start in parallel even if data integration waits |

---

## What's NOT in 4.0

- Plugin code generation (composes existing plugins only)
- Multi-pipeline orchestration (single pipeline per task)
- Streaming/continuous mode (Future — depends on Conversation Service)
- Multi-tenant RBAC (Future — depends on Conversation Service)
- TUI lineage explorer enhancements (independent track, not gated by 4.0)
- Landscape Repository CQRS split (1.0 architecture target)

---

## Starting Conditions

**Branch:** `RC4-user-interface` (current at time of writing)
**Design document:** `docs/architecture/semi-autonomous/design.md`
**Filigree tracking:** 14 items under milestones Autonomous Pipeline + Code Quality & Architectural Remediation
**Immediate next actions:** Begin the 5 unblocked items — recorder facade evaluation, config decomposition, plugin registry pattern, LLM template rewrite, telemetry exporter cleanup

---

## Appendix: What actually shipped

This appendix maps each of the 9 planned semi-autonomous features above to the deliverable that ultimately shipped. The mapping is approximate; some scope was absorbed into adjacent items, some was deferred, and some shipped under a different framing.

| # | Planned (3 March 2026) | What shipped (RC-5, 3 April 2026 onwards) | Notes |
|---|---|---|---|
| 1 | Engine API Extraction | Refactored engine surface used by the composer; no separate "Engine API" published as a public package | The composer drives the engine via the composition primitives (#2) rather than a separate programmatic API |
| 2 | Pipeline Composition API | Frozen `SourceSpec` / `NodeSpec` / `EdgeSpec` / `OutputSpec` / `PipelineMetadata` DTOs + composition tools + YAML generator | Shipped as planned; reachable both from the composer and from the MCP server |
| 3 | LLM Pipeline Composer | `ComposerService` with LLM tool-use loop; sub-4x hardening (dual-counter loop guard, discovery cache, partial state recovery, rate limiting, tool registry) | Shipped as planned; later (RC-5.1) extended with the advisor-escalation contract and forced-repair loop |
| 4 | Conversation Service | Web sessions subsystem (SQLAlchemy Core tables, `SessionServiceImpl` with CRUD + versioning + run enforcement, fork-from-message) — embedded in the `elspeth web` FastAPI app rather than published as a standalone "Conversation Service" | Naming evolved; behaviour matches the original intent |
| 5 | Review Classification & Meta-Audit | Three-provider authentication shipped (Local / OIDC / Entra); pipeline-review trust-tier framework deferred and partially absorbed into the audit-MANIFEST work in RC-5.2 | Original "plugin trust tiers (transparent → approval_required)" framing not adopted; the substantive audit-completeness work happened differently |
| 6 | Workflow & Worker Infrastructure | **Not shipped as Temporal + K8s.** Background pipeline execution shipped using FastAPI's `BackgroundTasks` + WebSocket progress streaming; durable execution via the existing checkpoint/resume mechanism | The Temporal + K8s + PostgreSQL infrastructure was the largest scope cut; the resulting deployment surface is much lighter than originally planned |
| 7 | Real-Time Telemetry Pipeline | WebSocket progress streaming directly from the execution service; no Redis pub/sub layer | The Redis + WebSocket gateway design was simplified to a direct WebSocket because the simpler design met the requirement |
| 8 | Frontend | React SPA with AGDS theming, catalog drawer, inspector panel, run-evidence widgets, recovery panel, guided composer | Shipped substantially as planned; framework evolved from "ComfyUI-inspired React Flow editor" to a graph view + inspector pattern after early iteration |
| 9 | Shared Storage & Task Database | Blob storage manager (6 phases — data model, REST API, frontend integration, composer tools, execution integration, schema inference); session DB on SQLite | PostgreSQL deferred; SQLite became the deployable session-DB target; Postgres portability validated by RC-5.2's testcontainer lane but Postgres is not the default deployment |

**Enablers (pulled forward from backlog):** All five enablers (plugin registry pattern, LLM template rewrite, telemetry exporter cleanup, recorder facade evaluation, config decomposition) landed in some form across RC-3.3 and RC-4 — the recorder evaluation produced the repository pattern (T19), the LLM template rewrite landed as the T10 consolidation, and the plugin registry pattern landed as the SDA-aligned plugin tree.

**Net effect on the original scope:** The semi-autonomous platform shipped, with a significantly lighter infrastructure footprint than the planning brief anticipated (no Temporal, no K8s, no Redis, no PostgreSQL by default), and with the audit-completeness story landing later (RC-5.1 and RC-5.2) than originally bundled into the cut. The "configuration generator with zero relaxation of audit, lineage, or trust-tier guarantees" invariant held throughout.
