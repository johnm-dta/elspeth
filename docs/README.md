# ELSPETH Documentation

ELSPETH is a domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines. Every decision is traceable to its source data, configuration, and code version.

**Current Status:** RC-3

---

## Start Here

| You are... | Read this first |
|------------|----------------|
| New to ELSPETH | [Your First Pipeline](guides/your-first-pipeline.md) |
| Building a pipeline | [User Manual](guides/user-manual.md) then [Configuration](reference/configuration.md) |
| Understanding the architecture | [Overview](architecture/overview.md) then [Token Lifecycle](architecture/token-lifecycle.md) |
| Developing plugins | [Plugin Protocol](contracts/plugin-protocol.md) |
| Operating in production | [Runbooks](runbooks/index.md) then [Troubleshooting](guides/troubleshooting.md) |
| Preparing a release | [Guarantees](release/guarantees.md) then [RC-3 Checklist](release/rc3-checklist.md) |

---

## Architecture

How the system works — design documents, subsystem overviews, decision records.

- [System Overview](architecture/overview.md) — Core architecture
- [Requirements Matrix](architecture/requirements.md) — Feature status and requirements
- [Subsystems](architecture/subsystems.md) — Component deep-dives
- [Token Lifecycle](architecture/token-lifecycle.md) — Row identity through forks/joins
- [Landscape System](architecture/landscape.md) — Audit trail architecture
- [Landscape Entry Points](architecture/landscape-entry-points.md) — Where audit records are created
- [Telemetry](architecture/telemetry.md) — Operational visibility (exporters, emission points, gaps)
- [Audit Remediation](architecture/audit-remediation.md) — Remediation epic
- **ADRs** — Architecture Decision Records
  - [ADR-001: Plugin-Level Concurrency](architecture/adr/001-plugin-level-concurrency.md)
  - [ADR-002: Routing Copy Mode Limitation](architecture/adr/002-routing-copy-mode-limitation.md)
  - [ADR-003: Schema Validation Lifecycle](architecture/adr/003-schema-validation-lifecycle.md)
  - [ADR-004: Explicit Sink Routing](architecture/adr/004-adr-explicit-sink-routing.md)
  - [ADR-005: Declarative DAG Wiring](architecture/adr/005-adr-declarative-dag-wiring.md)

## Contracts

Formal protocol definitions and token outcome guarantees.

- [Plugin Protocol](contracts/plugin-protocol.md) — Plugin development guide
- [System Operations](contracts/system-operations.md) — Operation contract definitions
- [Execution Graph](contracts/execution-graph.md) — DAG construction contracts
- **Token Outcomes** — Terminal state guarantees
  - [Token Outcome Contract](contracts/token-outcomes/00-token-outcome-contract.md)
  - [Outcome Path Map](contracts/token-outcomes/01-outcome-path-map.md)
  - [Audit Sweep](contracts/token-outcomes/02-audit-sweep.md)
  - [Test Strategy](contracts/token-outcomes/03-test-strategy.md)
  - [Investigation Playbook](contracts/token-outcomes/04-investigation-playbook.md)
  - [CI Gates and Metrics](contracts/token-outcomes/05-ci-gates-and-metrics.md)

## Guides

How-to guides and tutorials.

- [Your First Pipeline](guides/your-first-pipeline.md) — Step-by-step tutorial
- [User Manual](guides/user-manual.md) — Installation, configuration, running pipelines
- [Test System](guides/test-system.md) — Testing strategy and conventions
- [Data Trust and Error Handling](guides/data-trust-and-error-handling.md) — Three-tier trust model
- [Telemetry Guide](guides/telemetry.md) — User-facing telemetry configuration
- [Tier-2 Tracing](guides/tier2-tracing.md) — Pipeline data tracing
- [Landscape MCP Analysis](guides/landscape-mcp-analysis.md) — Audit database analysis with Claude
- [Troubleshooting](guides/troubleshooting.md) — Common errors and solutions
- [Docker](guides/docker.md) — Container deployment

## Reference

Lookup material — configuration, environment variables, tool documentation.

- [Configuration Reference](reference/configuration.md) — All YAML settings
- [Environment Variables](reference/environment-variables.md) — API keys, database URLs
- [ChaosLLM](reference/chaosllm.md) — Fake LLM server for testing pipelines
- [ChaosLLM MCP Server](reference/chaosllm-mcp.md) — Analysis tools for ChaosLLM metrics
- [Web Scrape Transform](reference/web-scrape-transform.md) — Web scraping transform reference

## Runbooks

Operational procedures for production environments.

- [Runbook Index](runbooks/index.md)
- [Resume Failed Run](runbooks/resume-failed-run.md)
- [Investigate Routing](runbooks/investigate-routing.md)
- [Incident Response](runbooks/incident-response.md)
- [Database Maintenance](runbooks/database-maintenance.md)
- [Backup and Recovery](runbooks/backup-and-recovery.md)
- [Configure Key Vault Secrets](runbooks/configure-keyvault-secrets.md)

## Release

Release management — checklists, guarantees, and release notes.

- [Guarantees](release/guarantees.md) — What ELSPETH promises
- [Feature Inventory](release/feature-inventory.md) — Complete feature list
- [RC-3 Checklist](release/rc3-checklist.md) — Current release checklist
- [RC-2 Checklist](release/rc2-checklist.md) — Previous release baseline
- [RC-3 Release Notes](release/rc-3-release-notes.md)
- [RC-2 Checkpoint Fix](release/rc-2-checkpoint-fix.md)

## Plans

Active implementation work.

- [Plans Index](plans/README.md)
- [RC-3 Remediation](plans/RC3-remediation.md)
- [ARCH-15 Design](plans/ARCH-15-design.md)
- [Contract Propagation](plans/2026-02-13-contract-propagation-complex-fields.md)
