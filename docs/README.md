# ELSPETH Documentation

Index of the documentation shipped in this repository.

**Framework status:** `0.5.3` (RC-5.3 line)
**Archive note:** current release, architecture, contract, guide, reference, and
runbook docs remain visible here. Point-in-time audit, architecture-pack,
generated-review, and handover snapshots from earlier workstreams were moved to
[`docs-archive/2026-05-19-docs-cleanout/`](../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md).

---

## Start Here

| You are... | Read this first |
|------------|----------------|
| New to ELSPETH | [Your First Pipeline](guides/your-first-pipeline.md) then [User Manual](guides/user-manual.md) |
| Building or operating pipelines | [Configuration Reference](reference/configuration.md), [Runbooks](runbooks/index.md), and [Troubleshooting](guides/troubleshooting.md) |
| Investigating audit data | [Landscape MCP Analysis](guides/landscape-mcp-analysis.md) and [Architecture Overview](../ARCHITECTURE.md) |
| Developing plugins | [Data Trust and Error Handling](guides/data-trust-and-error-handling.md), [Plugin Development Guide](../PLUGIN.md), then [Plugin Protocol](contracts/plugin-protocol.md) |
| Contributing to the codebase | [Contributing](../CONTRIBUTING.md) |
| Evaluating ELSPETH | [Executive Summary](release/executive-summary.md), [Composer Guide](release/composer-guide.md), [Platform Architecture](release/platform-architecture.md), [Public-Sector Assessment Mapping](release/assessment-mapping.md), and [Audit and Lineage Guarantees](release/guarantees.md) |

---

## Architecture

Current architecture and design references.

- [Architecture Overview](../ARCHITECTURE.md) — C4 model, data flows, and system-level orientation
- [System Overview](architecture/overview.md) — compatibility pointer to the maintained root architecture overview
- [Requirements Matrix](architecture/requirements.md) — compatibility pointer to current requirement and contract sources
- [Subsystems](architecture/subsystems.md) — compatibility pointer to current subsystem diagrams and ADRs
- [Token Lifecycle](architecture/token-lifecycle.md) — row identity through forks and joins
- [Landscape System](architecture/landscape.md) — audit trail architecture
- [Landscape Entry Points](architecture/landscape-entry-points.md) — where audit records are created
- [Telemetry](architecture/telemetry.md) — operational visibility architecture
- [Barrier Machinery](architecture/barrier-machinery.md) — aggregation and coalesce as structural twins; paired-surfaces table and paired-change checklist
- [ADR Index](architecture/adr/README.md) — accepted architecture decisions

## Contracts

Formal protocol definitions and token outcome guarantees. The narrative
assurance surface is [`release/guarantees.md`](release/guarantees.md); the
documents in this section formalise specific contracts that the engine, plugin
authors, and integrators must uphold.

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
- [Clarion for Codex Agents](guides/clarion-for-codex-agents.md)
- [Troubleshooting](guides/troubleshooting.md)
- [Docker](guides/docker.md)

## Composer Evidence

Evaluation reports and investigation notes that still support current composer
development or product explanation.

- [Composer Evidence Index](composer/evidence/README.md)

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
- [Ansible Ubuntu Deployment](runbooks/ansible-ubuntu-deployment.md)

## Release History

Audience-facing release, progress, velocity, and evaluation documents. See the
[release docs README](release/README.md) for the full index.

- [Executive Summary](release/executive-summary.md) — capability and assurance brief for Australian public-sector evaluators *(DRAFT — awaiting operator sign-off)*
- [Composer Guide](release/composer-guide.md) — current user-facing guide to the web authoring surface
- [Platform Architecture](release/platform-architecture.md) — current platform architecture, trust-boundary, and operational-responsibility overview
- [Public-Sector Assessment Mapping](release/assessment-mapping.md) — current evidence map for government evaluation touchpoints
- [Progress Report: RC-1 to RC-5](release/elspeth-progress-rc1-to-rc5.md) — current cumulative-output view (RC-5.2, May 2026): what shipped, period by period
- [Velocity Report: RC-1 to RC-5](release/elspeth-velocity-rc1-to-rc5.md) — current per-day commit volume (RC-5.2, May 2026): tempo and peak-day attribution
- [Audit and Lineage Guarantees](release/guarantees.md) — long-lived assurance narrative; refreshed per release (current contract surface; §1–§10 RC-3 base, §11–§14 RC-5.2 additions)
- [Archived RC snapshots](../docs-archive/2026-05-19-docs-cleanout/docs/release/) — `feature-inventory.md` (RC-3.3), `rc4-executive-brief.md` (RC-4.0 planning), `rc-3-release-notes.md`, `rc-2-checkpoint-fix-postmortem.md` — historical context only; see the [2026-05-19 cleanout MANIFEST](../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md) for full relocation map

## Historical Snapshots

Intentional point-in-time documents retained for reference.

- [RC-3 Release Notes](../docs-archive/2026-05-19-docs-cleanout/docs/release/rc-3-release-notes.md) — RC-3 release-history snapshot, relocated into the 2026-05-19 dated docs archive
- [Plans index](plans/README.md) — curated in-tree design and implementation plans
- [Composer UX Redesign](composer/ux-redesign-2026-05/) — RC-5.2 composer planning and implementation phase documents
- [Archived docs cleanout manifest](../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md) — relocation map for archived release docs, audits, frozen architecture packs, generated review sidecars, and completed handovers
