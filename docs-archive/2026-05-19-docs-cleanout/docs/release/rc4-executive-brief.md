# RC 4.0 Executive Brief — Semi-Autonomous Pipeline Platform

**Date:** 2026-03-03
**Release:** RC 4 (v4.0.0)
**Status:** Planning complete, ready to begin implementation
**Prepared by:** Architecture analysis, synthesized from design document and Filigree work package
**Intended audience:** Project stakeholders evaluating scope, sequencing, and risk for the 4.0 release

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

**Branch:** `RC4-user-interface` (current)
**Design document:** `docs/architecture/semi-autonomous/design.md`
**Filigree tracking:** 14 items under milestones Autonomous Pipeline + Code Quality & Architectural Remediation
**Immediate next actions:** Begin the 5 unblocked items — recorder facade evaluation, config decomposition, plugin registry pattern, LLM template rewrite, telemetry exporter cleanup
