# ELSPETH Documentation

Index of the documentation shipped in this repository.

**Framework status:** `0.5.1` (RC-5.1 line)
**Tracking note:** active delivery work lives in Filigree; many docs under `release/`,
`plans/`, `analysis/`, `audits/`, and `arch-analysis-*` are intentionally
point-in-time snapshots.

---

## Start Here

| You are... | Read this first |
|------------|----------------|
| New to ELSPETH | [Your First Pipeline](guides/your-first-pipeline.md) then [User Manual](guides/user-manual.md) |
| Building or operating pipelines | [Configuration Reference](reference/configuration.md), [Runbooks](runbooks/index.md), and [Troubleshooting](guides/troubleshooting.md) |
| Investigating audit data | [Landscape MCP Analysis](guides/landscape-mcp-analysis.md) and [Architecture Overview](../ARCHITECTURE.md) |
| Developing plugins | [Data Trust and Error Handling](guides/data-trust-and-error-handling.md), [Plugin Development Guide](../PLUGIN.md), then [Plugin Protocol](contracts/plugin-protocol.md) |
| Contributing to the codebase | [Contributing](../CONTRIBUTING.md) and [CLAUDE.md](../CLAUDE.md) |

---

## Architecture

Current architecture and design references.

- [Architecture Overview](../ARCHITECTURE.md) — C4 model, data flows, and system-level orientation
- [System Overview](architecture/overview.md) — subsystem map and architectural narrative
- [Requirements Matrix](architecture/requirements.md) — audited requirement coverage snapshot
- [Subsystems](architecture/subsystems.md) — component deep-dives
- [Token Lifecycle](architecture/token-lifecycle.md) — row identity through forks and joins
- [Landscape System](architecture/landscape.md) — audit trail architecture
- [Landscape Entry Points](architecture/landscape-entry-points.md) — where audit records are created
- [Telemetry](architecture/telemetry.md) — operational visibility architecture
- [ADR Index](architecture/adr/README.md) — accepted architecture decisions

## Contracts

Formal protocol definitions and token outcome guarantees.

- [Plugin Protocol](contracts/plugin-protocol.md)
- [System Operations](contracts/system-operations.md)
- [Execution Graph](contracts/execution-graph.md)
- [Token Outcome Assurance](contracts/token-outcomes/README.md)

## Guides

Tutorials and operator/developer how-to material.

- [Your First Pipeline](guides/your-first-pipeline.md)
- [User Manual](guides/user-manual.md)
- [Test System](guides/test-system.md)
- [Data Trust and Error Handling](guides/data-trust-and-error-handling.md)
- [Telemetry Guide](guides/telemetry.md)
- [Tier-2 Tracing](guides/tier2-tracing.md)
- [Landscape MCP Analysis](guides/landscape-mcp-analysis.md)
- [Troubleshooting](guides/troubleshooting.md)
- [Docker](guides/docker.md)

## Reference

Lookup material for configuration, tools, and plugin-specific behavior.

- [Configuration Reference](reference/configuration.md)
- [Environment Variables](reference/environment-variables.md)
- [Composer Tools](reference/composer-tools.md)
- [ChaosLLM](reference/chaosllm.md)
- [ChaosLLM MCP Server](reference/chaosllm-mcp.md)
- [Web Scrape Transform](reference/web-scrape-transform.md)

## Operations

Runbooks and production procedures.

- [Runbook Index](runbooks/index.md)
- [Resume Failed Run](runbooks/resume-failed-run.md)
- [Investigate Routing](runbooks/investigate-routing.md)
- [Incident Response](runbooks/incident-response.md)
- [Database Maintenance](runbooks/database-maintenance.md)
- [Backup and Recovery](runbooks/backup-and-recovery.md)
- [Configure Key Vault Secrets](runbooks/configure-keyvault-secrets.md)

## Historical Snapshots

Intentional point-in-time documents retained for reference.

- [Release docs](release/) — RC2 through RC4 checklists, notes, guarantees, and briefs
- [Plans index](plans/README.md) — curated in-tree design and implementation plans
- [Audits](audits/) — audit reports and verification sweeps
- [Analysis](analysis/) — analysis and posture briefs
- [Architecture analysis workspace](arch-analysis-2026-02-22-0446/) — pre-remediation snapshot cited by later ADRs and plans
- [Superpowers specs and plans](superpowers/) — internal assistant-driven planning/spec artifacts
