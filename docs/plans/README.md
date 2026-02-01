# ELSPETH Implementation Plans

This directory tracks implementation plans across all development phases.

## Directory Structure

```
docs/plans/
├── in-progress/                 # 3 plans currently being worked on
├── paused/                      # 3 plans temporarily on hold
├── completed/                   # 127 fully implemented plans
│   └── plugin-refactor/         # 28 plans from the plugin refactor work
├── cancelled/                   # 1 cancelled plan
├── superseded/                  # 14 superseded plans
│   └── schema-validation-attempts-jan24/
├── *.md                         # 1 plan at root level (various states)
└── README.md                    # This file
```

## In-Progress Plans (3)

Plans actively being worked on:

| Plan | Description |
|------|-------------|
| `in-progress/2026-01-26-recorder-refactoring.md` | Landscape recorder refactor into repository modules |
| `in-progress/2026-01-30-tier2-tracing-implementation.md` | Tier 2 plugin tracing (Azure AI + Langfuse) |
| `in-progress/RC2-remediation.md` | RC-2 remediation and phased hardening roadmap |

## Root-Level Plans (1)

Plans at the root level in various states:

| Plan | Description |
|------|-------------|
| `2026-01-30-chaosllm-design.md` | ChaosLLM fake LLM server design |

## Paused Plans (3)

Plans temporarily on hold:

- `paused/2026-01-12-phase7-advanced-features.md`
- `paused/2026-01-17-chunk3-refactors-tui.md`
- `paused/2026-01-20-world-class-test-regime.md`

## Completed Plans (127)

### Summary
- **99 plans** directly in `completed/`
- **28 plans** in `completed/plugin-refactor/`

### Major Phases (All Completed)
- **Phase 1:** Foundation (canonical JSON, DAG, config)
- **Phase 2:** Plugin system (pluggy, hookspecs, discovery)
- **Phase 3A:** Landscape (audit trail, 12 tables)
- **Phase 3B:** Engine (orchestrator, processor, retry)
- **Phase 4:** CLI and I/O (Typer, CSV/JSON/DB sources/sinks)
- **Phase 5:** Production hardening (checkpointing, rate limiting, retention)
- **Phase 6:** External calls (LLM integration, Azure, pooling)

See `completed/` directory for full list.

## Superseded Plans (14)

Plans replaced by better approaches:

- `superseded/schema-validation-attempts-jan24/` - 14 schema validation attempts that led to the current validation subsystem design

## Plan Lifecycle

```
Created → In Progress → Paused/Cancelled/Completed/Superseded
```

### Status Definitions

| Status | Meaning | Location |
|--------|---------|----------|
| **In Progress** | Currently being implemented | `in-progress/` subdirectory |
| **Paused** | On hold, may resume | `paused/` subdirectory |
| **Cancelled** | Abandoned, not implemented | `cancelled/` subdirectory |
| **Superseded** | Replaced by a better approach | `superseded/` subdirectory |
| **Completed** | Fully implemented and tested | `completed/` subdirectory |

### Moving Plans

When marking a plan complete:
1. Update plan header: `**Status:** IMPLEMENTED (YYYY-MM-DD)`
2. Add "Implementation Summary" section with evidence
3. Move to `completed/` directory: `git mv plan.md completed/`

## Evidence-Based Status

All plan status updates verified against:
- Schema inspection (`schema.py`, migrations)
- Implementation presence (`grep` for APIs, patterns)
- Test coverage (`test_*.py` files)
- Bug directory (closed vs open)
- GitHub Actions workflows (CI/CD)
