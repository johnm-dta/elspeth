# ELSPETH Documentation

ELSPETH is a domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines. Every decision is traceable to its source data, configuration, and code version.

**Current Status:** RC-2

---

## Quick Start

- **[User Manual](USER_MANUAL.md)** - Installation, configuration, running pipelines
- **[Your First Pipeline](guides/your-first-pipeline.md)** - Step-by-step tutorial
- **[Docker Guide](guides/docker.md)** - Container deployment

## Reference

- **[Configuration Reference](reference/configuration.md)** - All YAML settings
- **[Environment Variables](reference/environment-variables.md)** - API keys, database URLs
- **[Plugin Protocol](contracts/plugin-protocol.md)** - Plugin development guide

## Operations

- **[Runbooks](runbooks/index.md)** - Operational procedures index
  - [Resume Failed Run](runbooks/resume-failed-run.md)
  - [Investigate Routing](runbooks/investigate-routing.md)
  - [Incident Response](runbooks/incident-response.md)
  - [Database Maintenance](runbooks/database-maintenance.md)
  - [Backup and Recovery](runbooks/backup-and-recovery.md)
- **[Troubleshooting](guides/troubleshooting.md)** - Common errors and solutions
- **[Landscape MCP Analysis](guides/landscape-mcp-analysis.md)** - Audit database analysis with Claude

## Architecture

- **[System Design](design/architecture.md)** - Core architecture overview
- **[Requirements Matrix](design/requirements.md)** - Feature status and requirements
- **[ADRs](design/adr/)** - Architecture Decision Records
  - [ADR-001: Plugin-Level Concurrency](design/adr/001-plugin-level-concurrency.md)
  - [ADR-002: Routing Copy Mode Limitation](design/adr/002-routing-copy-mode-limitation.md)
  - [ADR-003: Schema Validation Lifecycle](design/adr/003-schema-validation-lifecycle.md)
- **[Subsystems](design/subsystems/)** - Component deep-dives
  - [Overview](design/subsystems/00-overview.md)
  - [Token Lifecycle](design/subsystems/06-token-lifecycle.md)

## Quality and Testing

- **[Test System](TEST_SYSTEM.md)** - Testing strategy and conventions
- **[Quality Audit](quality-audit/)** - Code quality findings
  - [Audit Plan](quality-audit/audit-plan.md)
  - [Test Suite Analysis](quality-audit/TEST_SUITE_ANALYSIS_2026-01-22.md)
  - [Mutation Testing Summary](quality-audit/MUTATION_TESTING_SUMMARY_2026-01-25.md)
  - [Integration Seam Analysis](quality-audit/INTEGRATION_SEAM_ANALYSIS_REPORT.md)

## Testing Tools

- **[Testing Tools Overview](testing/README.md)** - Load testing, stress testing, fault injection
- **[ChaosLLM](testing/chaosllm.md)** - Fake LLM server for testing pipelines at scale
- **[ChaosLLM MCP Server](testing/chaosllm-mcp.md)** - Analysis tools for ChaosLLM metrics

## Audit Trail

- **[Token Outcome Contract](audit-trail/tokens/00-token-outcome-contract.md)** - Terminal state guarantees
- **[Outcome Path Map](audit-trail/tokens/01-outcome-path-map.md)** - How rows reach terminal states
- **[Audit Sweep](audit-trail/tokens/02-audit-sweep.md)** - Verification procedures
- **[Test Strategy](audit-trail/tokens/03-test-strategy.md)** - Audit testing approach
- **[Investigation Playbook](audit-trail/tokens/04-investigation-playbook.md)** - Debugging audit issues

## Project Management

- **[Plans](plans/)** - Implementation roadmaps
  - [Plans Index](plans/README.md)
  - [RC-2 Remediation](plans/RC2-remediation.md)
- **[Bug Tracking](bugs/)** - Issue tracking
  - [Bug Index](bugs/README.md)
  - [Active Bugs](bugs/BUGS.md)
- **[Release](release/)** - RC-2 checklists and guarantees
  - [RC-2 Checklist](release/rc2-checklist.md)
  - [Feature Inventory](release/feature-inventory.md)
  - [Guarantees](release/guarantees.md)

## Archive

Historical analyses and completed work:

- **[Archived Docs](archive/)** - Completed analyses
  - [Architecture Analysis (2026-01-27)](archive/2026-01-27-arch-analysis/)
  - [Azure Performance Work (2026-01)](archive/2026-01-azure-performance/)

---

## Additional Resources

- **[Performance Baseline](performance/schema-refactor-baseline.md)** - Schema refactor benchmarks
- **[Release Notes](release-notes/)** - Version history
