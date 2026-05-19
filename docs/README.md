# ELSPETH Documentation

Index of the documentation shipped in this repository.

**Framework status:** RC-5.2 workstream (pre-release)
**Tracking note:** active delivery work lives in Filigree. Point-in-time
release, audit, architecture-pack, generated-review, and handover snapshots
from earlier workstreams were moved to
[`docs-archive/2026-05-19-docs-cleanout/`](../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md).

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

- [Assurance Contract](contracts/assurance-contract.md)
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

## Current Audit Work

Active audit syntheses and ticket-filing packets that still drive current work.

- [CI/CD Allowlist Audit](audit/2026-05-19-cicd-allowlist-audit.md) — live gate inventory and burn-down findings for `elspeth-297b8f5c5d`
- [CI/CD Allowlist Findings](audit/findings/) — SME reports and draft subticket list retained until filing is complete
- [Test Suite Audit](audit/test-suite/README.md) — incomplete but still actionable test-quality audit waves and filed issue map

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

## Historical Snapshots

Intentional point-in-time documents retained for reference.

- [RC-3 Release Notes](release/rc-3-release-notes.md) — retained because it is still directly cited as a release-history snapshot
- [Plans index](plans/README.md) — curated in-tree design and implementation plans
- [Composer UX Redesign](composer/ux-redesign-2026-05/) — current RC5.2 composer planning and implementation phase documents
- [Superpowers specs and active plans](superpowers/) — internal assistant-driven planning/spec artifacts retained while still load-bearing
- [Archived docs cleanout manifest](../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md) — relocation map for archived release docs, audits, frozen architecture packs, generated review sidecars, and completed handovers
