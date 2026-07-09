# ELSPETH — Executive Summary (0.7.0)

> **DRAFT — awaiting review (8 July 2026).** Once approved for release, this banner is replaced by the provenance footer at the foot of the document; until then, claims about assurance status, compliance posture, and residual risk remain subject to revision.

**Document date:** 8 July 2026
**Release covered:** 0.7.0
**Audience:** Public-service executives, programme sponsors, assurance and risk staff
**Register:** Australian Public Service / institutional
**Prepared by:** John Morrissey, Digital Transformation Agency

---

## Document scope

This brief is a capability and assurance summary for the ELSPETH platform at release 0.7.0. It is intended to support an evaluator considering whether the platform is suitable to assess for pilot adoption.

This document is not:

- a procurement proposal or a response to a tender;
- a commercial offer — ELSPETH is open-source software, released under the MIT licence (see `LICENSE` in the source tree). The platform originated as a personal project authored by John Morrissey and was subsequently adopted by the Digital Transformation Agency; the personal copyright has not been formally assigned to the Commonwealth because the MIT licence already grants the Commonwealth (and any other adopter) the rights it would otherwise require, without further instrument. An adopting agency requires no additional licensing agreement with the author or with the DTA in order to use, modify, or redistribute the platform;
- a certified assurance statement for the broader platform — an interim Authority to Operate (ATO) has been received for the orchestration component, but no independent third-party assessment of the broader platform has been completed (see *Assurance posture* below).

Readers requiring engineering detail are referred, at the close of this brief, to the root changelog, platform architecture, and guarantees appendix.

---

## At a glance

| | |
|---|---|
| **What ELSPETH is** | A data-processing platform in which every decision is independently traceable to its source data, configuration, and code version. Intended for processes subject to formal audit or external review. |
| **What it does** | Ingests data from external systems, applies structured decision logic (rules, model checks, or large language model assistance), and produces outputs accompanied by a complete evidence trail. |
| **Build state** | A working framework with YAML, CLI/TUI, and Web Composer authoring surfaces, three authentication providers, and integrations with Microsoft Dataverse, ChromaDB, OpenRouter, Azure OpenAI, Azure Blob Storage, and Azure Document Intelligence. |
| **Maturity** | An interim ATO has been received for the orchestration component, and that component has been operated under the interim ATO in a real-world pilot of approximately 2,200 rows. The full pipeline platform remains pre-production and has not been independently assessed beyond the orchestration component's interim ATO. |
| **Author** | A single DTA staff developer. The associated continuity risk is material, and is described under *Residual risk* below. |

---

## Platform description

ELSPETH is a framework for building auditable data-processing pipelines. A pipeline reads data from a source system, applies structured logic (which may include calls to large language models), and produces outputs for downstream consumers. The platform's central design constraint is that every decision the system makes must be reconstructible from a permanent audit record. Given any output, the system can identify the source row that produced it, the configuration that was active at the time, the version of any model invoked and the response that model returned, and the controls applied at each step. The platform is designed so that no output exists for which the system cannot account.

---

## Capabilities present in the current release

The following capabilities are present, tested, and documented in the current release. Detailed engineering provenance is recorded in the root [`CHANGELOG.md`](../../CHANGELOG.md).

- **Auditable pipeline engine.** Source-to-decide-to-act pipelines with full graph topology, including parallel paths, branching, and joining. All operations are recorded in the audit trail before they are confirmed complete, and pipelines can be checkpointed and resumed after interruption.
- **Cryptographic integrity of records.** Every output is hashed using SHA-256 over RFC 8785 canonical-JSON form. An undetected change to a record changes its hash, providing a basis for tamper-detection.
- **Three-tier trust model.** Data the platform itself produces (the audit database) is treated as fully trusted, and any anomaly causes the platform to stop. Data passing between pipeline steps is type-validated. Data ingested from external systems is validated at the boundary, and rows that fail validation are quarantined rather than discarded. The boundaries are enforced by code-review rules and by automated static analysis.
- **Web authoring interface (Composer).** A chat-based authoring surface intended for non-engineering operators to construct pipelines. In 0.7.0, guided creation is LLM-primary stage by stage, with a conversational builder, live graph verification, pending-interpretation gates, and advisor sign-off at the wiring stage. The interface's visual treatment is consistent with the Australian Government Design System (AGDS), though it has not been assessed for AGDS conformance. Accessibility features include skip-to-content navigation, support for reduced-motion preferences, and screen-reader-accessible status indicators.
- **Three authentication providers.** Local username and password (intended for development and air-gapped deployments), OpenID Connect (for federated identity), and Microsoft Entra ID (with tenant validation and group claim support).
- **Secret-reference handling.** Credentials are referenced by name in pipeline configuration and are resolved at run time from Azure Key Vault or from an environment variable. The secret value itself does not appear in pipeline configuration. The resolution event is recorded in the audit trail by cryptographic fingerprint, not by the value of the secret.
- **Integration coverage.** Microsoft Dataverse (read and upsert), ChromaDB (vector storage for retrieval-augmented generation), Azure OpenAI, OpenRouter, Azure Blob Storage, Azure Document Intelligence, blob-backed document ingestion transforms, and CSV and JSON sources and sinks. A web-scraping transform is provided, with controls intended to prevent requests against internal network targets (server-side request forgery).
- **Recovery surface.** The Composer authoring interface persists in-progress work. If a session is interrupted, the operator can resume with the full transcript, the redacted tool-call record, and a before-and-after comparison of changes to the pipeline state.

---

## Assurance posture

### What the platform's design provides

- **Audit-first writes.** Every operation writes to the audit trail before the operation is confirmed complete. If the audit write fails, the operation fails. There is no best-effort audit path.
- **Lineage queries.** Given any output, the audit trail can return the source row, the configuration version, the model and prompt used (where applicable), the input and output of each transformation, and the principal who authored the pipeline.
- **Failure handling by design.** The platform is designed to stop, rather than continue, when it detects an internal inconsistency in its own data. Silent recovery from corruption is treated as a more serious failure mode than a stop with a clear cause.
- **Quarantine of invalid input.** External input that fails validation is recorded as quarantined; it is not silently discarded. A record that a row was quarantined for a stated reason is a valid audit outcome.
- **Trust-boundary enforcement.** Automated static analysis prevents defensive coding patterns on data the system itself produced. These patterns are permitted only at the boundary where external data enters the system, where they are appropriate.
- **Interim Authority to Operate (ATO).** An interim ATO has been received for the orchestration component. The component has been operated under the interim ATO in a real-world pilot of approximately 2,200 rows, each accompanied by an audit record of the kind described above. The pilot is operational evidence that the audit-trail design behaves as specified at a meaningful scale, not only in test simulation.

### What the platform's design does not yet provide

- **Outside the orchestration scope, no broader independent third-party assessment** has been completed. Beyond the orchestration component's interim ATO: no Information Security Registered Assessors Program (IRAP) assessment of the broader platform, no Digital Transformation Agency or Australian Government Design System conformance review beyond visual styling, and no independent penetration test. The audit-trail and trust-tier claims are the designed behaviour and are exercised by the automated test suite and CI gates recorded for the release. Outside the orchestration component's interim ATO, they have not been independently certified.
- **No formal assessor-validated mapping** has been compiled to the Protective Security Policy Framework (PSPF), the Information Security Manual (ISM), the Essential Eight, or the Digital Service Standard. The 0.7.0 evidence map in [`assessment-mapping.md`](assessment-mapping.md) identifies current evidence and gaps, but it is not a conformance statement.

### Evidence produced by the audit trail, by ISM control family

The platform's audit trail produces material relevant to several ISM control families, notwithstanding that no formal mapping has been commissioned. An evaluator commissioning an IRAP or equivalent assessment should expect the following evidence to be available:

| ISM control family | Evidence produced by the audit trail |
|---|---|
| System monitoring and event logging | Every pipeline operation, every external call request and response, every routing decision, and every terminal state is recorded before the operation is acknowledged complete. |
| Cryptography | RFC 8785 canonical JSON with SHA-256 hashing throughout; SQLCipher available for encryption at rest; HMAC fingerprints recorded for every secret resolution. |
| Identification and authentication | Three authentication providers are supported, with audit records of the principal that authored each pipeline configuration and the principal associated with each session. |
| Access control logging | Session creation, preference mutation, and pipeline-execution attempts are recorded against the authenticated principal. |
| Change management | The configuration that produced any output is reconstructible from the audit record by hash, and the code version is recorded against each run. |

The mapping above identifies evidence that is already present. It is not a claim of conformance. A formal conformance statement requires the independent assessment that has not yet been commissioned.

---

## Deployment readiness

| Dimension | State at 0.7.0 |
|-----------|----------------|
| Pilot deployment | RC-3 deployed in orchestration-only mode under the interim ATO; pilot evaluation processed approximately 2,200 rows, each producing a complete audit record. Full pipeline platform not yet deployed in any Australian government agency. |
| Air-gapped deployment | Supported (Local auth provider, no required external services) |
| Federated identity | Supported (OpenID Connect, Microsoft Entra) |
| Encryption at rest | Optional (SQLCipher passphrase, opt-in) |
| Encryption in transit | Required for external calls (validated at the trust boundary) |
| Manual upgrade steps | Yes — 0.7.0 requires operator-administered recreation of both the web session database and the Landscape audit database before first start. An automated migration path is on the roadmap. |
| Operational documentation | Runbooks present for resume, routing investigation, incident response, database maintenance, backup, Key Vault configuration, and Ansible-based Ubuntu deployment |

---

## Residual risk

The following risks are present at 0.7.0. Mitigations in place are noted where they exist; residual exposure is described against each item.

1. **Single-contributor continuity.** The platform has been developed by a single DTA staff member. The departure or unavailability of that contributor would suspend further development of the platform. Mitigations in place include architectural-decision records covering each load-bearing design choice, a contracts subsystem enforced by automated static analysis (so that future engineers cannot silently breach the trust-tier model), an operational runbook set covering resume, routing investigation, incident response, database maintenance, backup, Key Vault configuration, and Ansible-based Ubuntu deployment, and an automated test suite that functions as an executable specification. Mitigations not yet in place include the onboarding of a second engineer, a documented succession arrangement, and a multi-party commitment to runbook and architecture-document currency. An adopting agency should treat continuity planning as a precondition for pilot adoption. Reasonable approaches include contributing engineering capacity, negotiating a documented succession arrangement, or scoping the engagement as a defined deliverable rather than as an ongoing capability.
2. **Independent assurance is scope-limited.** The orchestration component holds an interim ATO. The rest of the platform — audit-integrity, trust-tier, and access-control claims outside the orchestration scope — has been internally tested but not independently assessed by an IRAP-registered assessor or equivalent. An agency adopting beyond the orchestration scope under a high-assurance obligation should factor an independent assessment into its adoption plan.
3. **Deployment scope is pilot-only.** RC-3 has been deployed in orchestration-only mode for pilot evaluation under the interim ATO, processing approximately 2,200 rows. The full pipeline platform has not been deployed in production by any agency. Operational characteristics under sustained agency load — concurrent users, audit-database growth over time, and long-running pipelines on contended infrastructure — have been exercised in simulation; the pilot supplies real-world data for the orchestration-only scope at the volume noted above.
4. **Manual schema migration between certain releases.** Upgrading between some releases currently requires an operator to recreate the session database manually. The procedure is documented, but it is a manual step rather than an automated migration.
5. **Default plugins depend on third-party services.** The Microsoft Dataverse, ChromaDB, Azure OpenAI, OpenRouter, and Microsoft Entra integrations depend on the service levels and security postures of those external providers. The platform's audit trail records the calls made and the responses received, but the audit boundary does not extend into the external systems themselves.

---

## Matters for an evaluator to raise with the project

The following matters are not addressed by this document or its companions, and should be raised with the project team before any decision on pilot adoption:

- the scope, timing, and cost of an independent assurance assessment beyond the orchestration component's interim ATO, and the components the project would prioritise for inclusion;
- the shape of an agency-scale pilot beyond the orchestration-only scope already exercised, including the proposed use case, the participating agency, the data involved, and the assurance threshold to be met;
- the contingency arrangements for the single-contributor risk identified above, including whether the platform's continuing development is expected to be undertaken by the Digital Transformation Agency, by the adopting agency, or by both.

The root [`CHANGELOG.md`](../../CHANGELOG.md), [`platform-architecture.md`](platform-architecture.md), and [`guarantees.md`](guarantees.md) record the engineering provenance behind the capability claims in this brief. The contractual guarantees appendix documents the audit, lineage, and trust-model assurances using layered historical language from RC-3 onward, with later additions for authentication, secret references, multi-user sessions, composer authoring, and durable token scheduling.

---

## Provenance

**Released by:** *(awaiting operator sign-off — date and signatory to be inserted on publication)*
**Document version:** 1.0-draft
**Companion documents:** `CHANGELOG.md`, `platform-architecture.md`, `guarantees.md`
**Canonical sources of truth referenced:** `/CHANGELOG.md` (RC-3+), `/docs/release/guarantees.md`, `/docs/architecture/`
