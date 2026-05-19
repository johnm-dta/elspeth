# ELSPETH — Executive Summary (RC-5.2)

> **DRAFT — awaiting review (19 May 2026).**
> This document is the executive-tier briefing for ELSPETH, written for senior officials in the Australian public sector evaluating the platform for high-assurance pipeline work. It is not yet linked from the directory index; the operator is reviewing claims about assurance status, compliance posture, and residual risk before publication. Comments and corrections welcome.

**Document date:** 19 May 2026
**Release covered:** RC-5.2 (version 0.5.2)
**Audience:** Public-service executives, programme sponsors, assurance and risk staff
**Prepared by:** John Morrissey (sole contributor and author of record)

---

## At a glance

| | |
|---|---|
| **What ELSPETH is** | A data-processing platform where every decision is independently traceable to its source data, configuration, and code version. Designed for work where the audit trail must withstand formal inquiry. |
| **What it does** | Ingests data from external systems, applies structured decision logic (rules, model checks, or large language model assistance), and produces outputs with a complete evidence trail. |
| **Build state** | Working framework with a web authoring interface, three authentication providers, and integration with Microsoft Dataverse, ChromaDB vector storage, OpenRouter, and Azure OpenAI. |
| **Maturity** | Pre-production. 128 days of build to date, no production deployment in any Australian government agency, no independent third-party assurance assessment yet completed. |
| **Author** | Single developer. Continuity risk is material and is described under *Residual risk* below. |

---

## What ELSPETH is, in one paragraph

ELSPETH is a framework for building auditable data-processing pipelines — workflows where data is read from a source system, structured logic is applied (which may include calls to language models), and outputs are produced for downstream consumers. The design constraint that distinguishes it from general-purpose data tooling is that **every decision the system makes must be reconstructible from a permanent audit record**. Given any output, the system can prove which source row produced it, which configuration was active, which version of which model returned which response, and what controls were applied at each step. "I don't know what happened" is treated by the design as an unacceptable answer for any output the platform produces.

---

## What has been shipped (capability summary)

The following capabilities are present, tested, and documented in the current release. Detailed engineering provenance is in [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md).

- **Auditable pipeline engine.** Source → decide → act pipelines with full graph topology (parallel paths, branching, joining). All operations recorded before completion; checkpoint/resume after interruption.
- **Tamper-evident records.** Every output is hashed using a cryptographic standard (SHA-256 over RFC 8785 canonical form). Any undetected change to a record changes its hash.
- **Three-tier trust model.** The audit database is treated as fully trusted (crash on anomaly). Pipeline-internal data is type-validated. External data is validated at ingestion and quarantined on failure. The boundaries are enforced by code-review rules and automated checks.
- **Web authoring interface (Composer).** A chat-led authoring surface for non-engineering operators to build pipelines, themed to the **Australian Government Design System** for visual compliance. Accessibility features include skip-to-content, reduced-motion support, and screen-reader-safe status indicators.
- **Three authentication providers.** Local username/password (for development and air-gapped deployments), OpenID Connect (for federated identity), and Microsoft Entra ID (with tenant validation and group claims).
- **Secret-reference handling.** Credentials are referenced by name and resolved at run time from an Azure Key Vault or environment variable; the secret value never appears in pipeline configuration. Resolution is recorded in the audit trail by cryptographic fingerprint, not by value.
- **Integration coverage.** Microsoft Dataverse (read and upsert), ChromaDB (vector storage for retrieval-augmented generation), Azure OpenAI, OpenRouter, Azure Blob Storage, and CSV/JSON sources and sinks. Web-scraping transform with controls against server-side-request-forgery attacks.
- **Recovery surface.** The composer authoring surface persists in-progress work; if a session is interrupted, the operator can resume with the full transcript, redacted tool record, and a before/after comparison of what changed.

---

## Assurance posture

What the platform's design provides:

- **Audit-first writes.** Every operation writes to the audit trail before the operation is confirmed complete. If the audit write fails, the operation fails — there is no "best effort" audit path.
- **Lineage queries.** Given any output, the audit trail can return the source row, the configuration version, the model and prompt used (if any), the input and output of each transformation, and the principal that authored the pipeline.
- **Deliberate failure handling.** The system is designed to crash rather than continue when it detects internal inconsistency in its own data. This is by design: silent recovery from corruption is treated as a more dangerous failure mode than visible crash.
- **Quarantine, not silent skip.** External input that fails validation is recorded as quarantined, not dropped. The audit trail of "row 42 quarantined because field X was malformed" is itself a valid outcome.
- **Trust-boundary enforcement.** Automated code analysis prevents defensive patterns (`hasattr`, broad exception catches, `.get()` with default) on data the system itself produced — these are reserved for boundaries where external data enters the system.

What the platform's design does **not** yet provide:

- **No independent third-party assessment** to date. No IRAP (Information Security Registered Assessors Program) assessment. No DTA/AGDS conformance review beyond visual styling. No independent penetration test. The audit-trail and trust-tier claims are the *designed* behaviour and are exercised by an automated test suite of approximately 14,100 tests; they have not been independently certified.
- **No formal mapping** to the Protective Security Policy Framework (PSPF), the Information Security Manual (ISM), the Essential Eight, or the Digital Service Standard. The platform is designed to *support* these obligations — the audit trail provides evidence relevant to several ISM controls (system event logging, access control logging, change management) — but the mapping has not been formally compiled.

---

## Deployment readiness

| Dimension | State at RC-5.2 |
|-----------|----------------|
| Pilot deployment | Designed to support; not yet deployed in any Australian government agency |
| Air-gapped deployment | Supported (Local auth provider, no required external services) |
| Federated identity | Supported (OpenID Connect, Microsoft Entra) |
| Encryption at rest | Optional (SQLCipher passphrase, opt-in) |
| Encryption in transit | Required for external calls (validated at the trust boundary) |
| Manual upgrade steps | Yes — current release requires operator-administered database schema recreation between certain versions. An automated migration path is on the roadmap. |
| Operational documentation | Runbooks present for resume, routing investigation, incident response, database maintenance, backup, Key Vault configuration, and Ansible-based Ubuntu deployment |

---

## Residual risk

Honest enumeration. Each item is real, currently unmitigated, and visible to anyone evaluating the platform.

1. **Single-contributor continuity risk.** The platform has one developer. Loss of that contributor would halt development. The codebase, audit trail design, and runbooks are documented to a standard that allows another engineer to take over, but no second engineer has yet been onboarded.
2. **No independent assurance assessment.** The audit-integrity, trust-tier, and access-control claims have been internally tested. They have not been independently assessed by an IRAP-registered assessor or equivalent. Agencies adopting the platform under a high-assurance obligation will need to factor an independent assessment into their adoption plan.
3. **Pre-production maturity.** No agency has yet deployed ELSPETH in production. Operational characteristics under sustained agency load (concurrent users, audit-database growth, long-running pipelines under contended infrastructure) have been tested in simulation but not in deployment.
4. **Manual schema migration.** Upgrading between certain releases currently requires an operator action to recreate the session database. This is documented but is a manual step, not an automated migration.
5. **Default plugins include third-party dependencies.** Microsoft Dataverse, ChromaDB, Azure OpenAI, OpenRouter, and Microsoft Entra integrations depend on those vendors' SLAs and security postures. ELSPETH's audit trail records what was called and what was returned, but does not extend the audit boundary into those external systems.

---

## What an evaluator should consider next

For a decision-maker considering pilot adoption of ELSPETH, the following questions are not yet answered by this document or its companions and should be raised with the project:

- What independent assurance assessment is feasible in the next 90 days, and what would it cost?
- What does an agency-scale pilot look like — which use case, which agency, which data, which assurance threshold?
- What contingency exists for the single-contributor risk? Is the team plan to bring on additional engineers, or is the model that an adopting agency contributes engineering?
- What is the licence and intellectual property posture for an agency adopting the platform?

The companion documents — [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md) (cumulative engineering output) and [`elspeth-velocity-rc1-to-rc5.md`](elspeth-velocity-rc1-to-rc5.md) (delivery cadence) — provide the engineering provenance behind the capability claims in this brief. They are written for engineering reviewers and are not required reading for the questions above; they are available for an engineering team supporting an agency's evaluation.
