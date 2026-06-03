# Semi-Autonomous Platform — Design Review Synthesis

**Date:** 2026-03-02
**Input:** `docs/architecture/semi-autonomous/design.md` (Draft)
**Method:** 3-round facilitated panel discussion with 6 specialized reviewers
**Epic:** `elspeth-rapid-ea33f5`

---

## Panel Composition

| Role | Perspective | Key Contribution |
|---|---|---|
| **Systems Thinker** | Feedback loops, emergent behaviors, leverage points | Identified Review Theater archetype, Oracle Problem, eroding goals dynamic |
| **Architecture Critic** | Structural quality, technology fitness, security surfaces | PipelineArtifact sealed model, static security enforcement, version pinning |
| **UX Specialist** | Cognitive load, review fatigue, workflow coherence | Anti-rubber-stamping patterns, refinement diff, reviewer workflow |
| **Requirements Engineer** | Completeness, testability, traceability | Formal requirement specifications with acceptance criteria |
| **Dr. Sarah Chen** (Research Analyst) | Day-to-day research workflow, publication attribution | Methodology export, templates, quarantine explanation |
| **Marcus Webb** (Compliance Officer) | Regulatory defensibility, four-eyes principle, audit integrity | Reference data versioning, classification config governance, test case libraries |

---

## Governance Model Revision: Three-Tier Enforcement

The panel unanimously endorsed replacing the design document's 4-tier review classification (transparent / visible / review-required / approval-required) with a **3-tier enforcement model**:

| Tier | Enforcement | Rationale |
|---|---|---|
| **No Review Required** | No human gate. Automatic preview runs; user can execute immediately. | Low-risk plugin combinations. Preview provides evidence without friction. |
| **Approval Required** | One authenticated signature before execution. | Medium-risk operations. Single accountable human reviews evidence and signs. |
| **Two Approval Required** | Two distinct authenticated signatures before execution. System-enforced server-side. | High-risk operations (external writes, sensitive data). Four-eyes principle. |

**Why the change:** The old "Review Required" tier (acknowledge checkbox) was identified as structurally equivalent to rubber-stamping — it recorded that a user saw a checkbox, not that they engaged with the content. The systems thinker identified this as a "Shifting the Burden" archetype; the UX specialist flagged it as a P0 review fatigue risk. Eliminating it and replacing with real signatures produces an honest governance model: either there's no gate, or there's a real commitment.

**Pipeline-level enforcement:** The pipeline tier is computed as `max(tier(plugin) for plugin in pipeline_plugins)`. A pipeline containing any Two-Approval-Required plugin requires two signatures for the entire pipeline — not per-node approval. Per-node indicators remain as UX signals explaining which plugin drives the elevated tier.

**System enforces, model presents:** The reasoning model builds pipelines and explains governance requirements to users. The platform computes and enforces the tier. The model has no mechanism to propose, influence, or override the computed tier. This separation was a critical product-owner clarification that resolved the panel's largest structural concern.

---

## Consensus Items (Unanimous Agreement, All 6 Panelists)

### C1. Sealed PipelineArtifact

Replace the "frozen YAML config" artifact boundary with a sealed, hash-verified record:

| Field | Purpose |
|---|---|
| `pipeline_yaml` | Complete executable configuration |
| `data_reference` + `content_hash` | Source data identity with integrity verification |
| `plugin_catalog_hash` | ELSPETH plugin versions at generation time |
| `classification_config_hash` | Review tier policy version active at generation time (added post-Round 2 based on Requirements Engineer REQ-TIER-002 and Marcus Webb's governance requirement; not in original Architecture Critic proposal) |
| `elspeth_version` | Engine version |
| `review_actions[]` | Timestamped approval signatures with actor identity |
| `canonical_hash` | RFC 8785 + SHA-256 over the complete artifact |

The artifact is write-once. Any modification (including refinement) creates a new artifact version with lineage to its predecessor. Hash verification at execution time ensures what was reviewed is what executes.

**Origin:** Architecture Critic (Round 2), endorsed by all panelists.

### C2. Automatic Micro-Execution (Preview by Default)

For pipelines at the No-Review-Required tier, the platform automatically executes a preview run on N sample rows (configurable, default 5) using the full ELSPETH engine before presenting the pipeline to the user. The reasoning model presents both the pipeline design AND the actual preview outputs in the same view.

For Approval-Required and Two-Approval-Required tiers, automatic preview does NOT execute until the approval gate is satisfied (to prevent unauthorized data processing).

Preview runs are full ELSPETH executions with complete Landscape recording. The preview `run_id` is recorded in the meta-audit log.

**Origin:** Product owner clarification (reasoning model actively builds flows), endorsed unanimously.

### C3. Dual-Layer Summaries

Pipeline summaries are presented in two distinct layers:

1. **Deterministic technical specification** — derived mechanically from the `GraphDescriptor`: plugin names, field mappings, routing rules, model selection. Audit-grade, reproducible, not generated by the LLM.
2. **LLM interpretation** — plain-English explanation of what the pipeline does and why. Explicitly labeled as AI-generated ("This description is AI-generated and may not be fully accurate").
3. **Trial results** — actual outputs from the preview run on sample rows.

The approver's signature binds to the deterministic spec hash, not the AI interpretation. The AI interpretation is context; the spec is the legal artifact.

**Origin:** Independent convergence — UX Specialist and Architecture Critic proposed this in Round 2; Marcus Webb refined the signature-binding detail in Round 3.

### C4. System-Enforced Execution Gates

The platform enforces approval gates server-side, independent of the frontend and the reasoning model:

- **No Review Required:** No gate. Execution available immediately after preview.
- **Approval Required:** Execution blocked until one authenticated signature is recorded.
- **Two Approval Required:** Execution blocked until two distinct authenticated signatures are recorded. The same user cannot provide both signatures under any circumstance.

The model cannot bypass, influence, or query the signature state. Gate enforcement is a platform function.

**Origin:** Product owner clarification, endorsed unanimously.

### C5. Plugin Floor Principle

Each plugin has an inviolable minimum review tier set by platform configuration. The reasoning model can upgrade a plugin's effective tier (by explaining additional risk) but can never downgrade it below the configured floor. The computation is:

```
effective_tier = max(platform_floor[plugin], model_advisory[plugin])
pipeline_tier = max(effective_tier for all plugins in pipeline)
```

New or unclassified plugins default to Approval Required (fail-closed).

**Origin:** Systems Thinker + Architecture Critic, confirmed by product owner.

### C6. Deterministic Preview Row Selection

Preview sample rows are selected by the platform using a deterministic, model-independent strategy — first N rows in source order (default) or stratified random sample with a seeded RNG. The model does not select or influence which rows are previewed. The selection policy, row identifiers, and seed (if applicable) are recorded in the meta-audit.

**Origin:** Architecture Critic + Systems Thinker + Requirements Engineer (independent convergence).

### C7. Hash-Based Signature Invalidation

Any change to the pipeline configuration invalidates all collected signatures. If the config hash changes (even from a minor parameter tweak), the approval cycle restarts from zero. Signatures bind to a specific artifact hash — not to the pipeline concept. Additionally, the approver's signature should reference the preview `run_id` alongside the pipeline config hash, so the audit trail records both what was designed and what evidence was seen before signing.

For refinement workflows, unchanged nodes may carry forward prior approvals via a structural diff mechanism (see Feature F3).

**Origin:** Marcus Webb (Round 2), Requirements Engineer (AC-REVIEW series).

### C8. Database-Enforced Tenant Isolation

- **Standard deployments:** Schema-per-tenant with PostgreSQL `SET search_path` for transparent isolation.
- **Regulated deployments:** Database-per-tenant with separate PostgreSQL instances.

The design documentation should explicitly recommend database-per-tenant for regulated financial services, healthcare (HIPAA), and any environment requiring physical data separation — not merely offer it as an option. Schema-per-tenant on a shared cluster may not survive regulatory examination due to shared WAL logs, autovacuum, `pg_stat_*` tables, and a single superuser that can bypass schema permissions.

**Origin:** Architecture Critic (Round 2), Marcus Webb (Round 3 precision note).

---

## Features (Prioritized by Panel Convergence)

### Tier 1: Architectural Necessities (Blocking — System Cannot Ship Without These)

#### F1. Classification Config Governance
**Panel support:** 3 of 6 panelists independently identified this (Systems Thinker, Marcus Webb, Requirements Engineer)

The plugin review tier configuration is the single highest-value governance artifact in the system. Changes to it must be governed with the same rigor as pipeline execution:

- **Version-controlled and immutable** — changes create a new version; old versions retained permanently
- **Two-administrator authorization** to modify any plugin's tier classification
- **Full event logging** — every change records: prior tier, new tier, both admin actors, timestamp, justification
- **Referenced by every pipeline run** — each `tier_computed` event records the classification config hash active at execution time

Without this, the structural enforcement model can be silently eroded through administrative configuration changes with no audit trail. An examiner asking "why did this pipeline require only one signature in March but two in January?" must be able to answer from the audit record.

#### F2. Two-Person Approval Workflow (Complete End-to-End)
**Panel support:** 2 of 6 (Architecture Critic, UX Specialist), with Marcus Webb's four-eyes requirement as the driver

Two-Approval-Required is now a first-class system tier. The workflow to satisfy it must be fully specified:

- **Reviewer discovery:** Requester specifies reviewer(s) by name or role at submission time
- **Async notification:** Email + in-app notification with direct link to self-contained review task
- **Reviewer independence:** Reviewer sees evidence (deterministic spec, trial outputs, business purpose) but NOT the requester's annotations or approval status until after submitting their own decision
- **Rejection and revision:** Rejected pipelines return to draft state with rejection reason visible to requester
- **Timeout/expiry:** Configurable maximum waiting period; expired pipelines return to draft
- **Audit record:** Both approval and rejection decisions recorded with timestamps, actor identity, and artifact hash

Without this, Two-Approval-Required pipelines are permanently blocked with no user path forward.

#### F3. Refinement Diff and Selective Re-Approval
**Panel support:** 1 of 6 (UX Specialist), but addresses a critical adoption risk under the new governance model

When a user refines a pipeline ("also extract dates"), the system computes a structural diff between old and new PipelineArtifact versions:

- Nodes with unchanged settings carry their prior approval forward (linked in the audit chain)
- Only changed or new nodes require fresh approval
- If the pipeline tier itself changes (e.g., refinement adds a database_sink), full approval restarts
- The meta-audit records which approvals were carried forward and which were fresh

Without this, any refinement forces a complete re-approval cycle — potentially across time zones for two-person flows. Users will stop refining and accept "good enough" first-generation pipelines.

#### F4. Static Security Policy Enforcement on Generated YAML
**Panel support:** 1 of 6 (Architecture Critic), but elevated to critical under 3-tier model

Before presenting any generated pipeline to the user, a deterministic, non-LLM security analysis pass validates the YAML against administrator-defined rules:

- HTTP-capable plugins may only reference URLs from a registered allowlist
- External write destinations must match registered, pre-approved targets
- No environment variable interpolation syntax (`@env_var`) in LLM-generated fields
- Violations block pipeline presentation entirely, before tier classification

Under the 3-tier model, No-Review-Required flows have zero human gate. This static pass is the primary technical control for pipelines that execute without human review. It guards against prompt injection attacks that construct dangerous behavior from low-risk-looking components.

#### F5. Plugin Catalog Version Pinning
**Panel support:** 1 of 6 (Architecture Critic)

The PipelineArtifact carries a `plugin_catalog_hash`. Before the Worker Pod executes, it compares its running catalog hash against the artifact's stored hash. On mismatch: execution fails with an explicit `CatalogVersionMismatch` error naming the artifact, expected hash, and running hash.

This prevents silent behavior changes when ELSPETH is updated between artifact generation and execution — especially critical for two-signature flows where approval may take days.

### Tier 2: Product Essentials (Required for Adoption by Target Users)

#### F6. Template Library with Organizational Sharing
**Panel support:** 2 of 6 (UX Specialist, Dr. Sarah Chen)

A curated library of pre-built pipeline patterns for common use cases (classification, extraction, aggregation, anomaly detection):

- Each template includes: sample prompt, sample input data, annotated expected outputs, review guidance, and displayed pipeline tier
- Templates available from home screen before any prompt is entered
- LLM generation context includes relevant templates as few-shot examples
- Users can save successful pipelines as personal templates
- Administrators can publish organization-level templates

Templates serve triple duty: onboarding (users see what "good" looks like), generation quality (few-shot examples improve LLM output), and tier visibility (users know what governance commitment a pattern requires before starting).

#### F7. Methodology Citation Export
**Panel support:** 1 of 6 (Dr. Sarah Chen), but addresses a core value proposition for the research user persona

After a completed run, the system generates a structured, human-readable document containing:

- Pipeline name, run date, run_id
- Model name, version, temperature setting
- Complete prompt text (verbatim)
- Total rows processed, quarantine count with reasons
- 5-row sample of input-output pairs
- Run_id for full lineage traceability

This turns the Landscape audit trail from a backend engineering asset into a user-facing publication artifact. The deterministic technical spec from Dual-Layer Summaries (C3) is the source material for this export — the methodology citation is the publication-facing rendering of the same data. Without this feature, researchers cannot cite their methodology in published work — the primary reason they would adopt this system over ad-hoc ChatGPT usage.

#### F8. Quarantine Explanation in User-Accessible Terms
**Panel support:** 1 of 6 (Dr. Sarah Chen)

Every quarantined row is presented with:

1. **Plain-English reason** — "This comment was empty after removing formatting characters"
2. **Suggested action** — "Refine the prompt to handle ambiguous comments, or add 'ambiguous' as a valid output category"
3. **Impact assessment** — when quarantine rate exceeds a configurable threshold (e.g., >5%): "15% of rows were quarantined. This may affect the validity of results."

For published research, quarantined rows are a methodological gap, not a technical artifact. Users need to assess whether quarantines are ignorable (junk data) or significant (systematic LLM failure that biases results).

#### F9. Reference Data Versioning
**Panel support:** 1 of 6 (Marcus Webb), but this was Marcus's **#1 blocker** for regulated deployment — without it, the audit trail cannot prove which reference data (sanctions lists, risk ratings) was active during a run

Every pipeline run records a cryptographic hash and immutable snapshot of reference data used for classifications (sanctions lists, risk ratings, jurisdiction tables, threshold values):

- New artifact input type: "Reference Dataset" with name and pinned version
- At execution time: hash reference data, store snapshot in payload store, record hash in run event log
- The gate expression language supports `reference_data('list_name', version='pinned')` rather than inline constants

Without this, the audit trail has a critical gap: it can prove the pipeline ran correctly but cannot prove it ran against the correct reference data. A sanctions screening run is legally indefensible without proof of which OFAC SDN List version was active.

### Tier 3: Governance Enhancements (Required for Regulated Deployment)

#### F10. User-Specified Validation Test Cases for Regulated Preview
**Panel support:** 2 of 6 (Marcus Webb, Systems Thinker by implication)

For Approval-Required and Two-Approval-Required pipelines, the preview supports user-specified test cases in addition to (or instead of) automatic sampling:

- Optional "validation case library" per pipeline template or organization
- System prompts the approver: "Run preview on [automatic sample] or [validation library: 24 cases]"
- Test cases exercise known edge cases (near-misses, threshold boundaries, known false positives/negatives)
- Selection choice and results recorded in the approval artifact

A random 5-row sample from 50,000 transactions has near-zero probability of exercising compliance-critical boundary cases. For regulated use, the preview sample is evidence — it should test something meaningful.

#### F11. Data Sensitivity Declaration as Tier Override
**Panel support:** 1 of 6 (Systems Thinker)

At pipeline submission time, users or platform policies can declare data sensitivity classification. The declaration can only upgrade the pipeline's tier above the plugin floor — never downgrade:

- User-provided: "this data contains PII", "this data is subject to HIPAA"
- Administrator-configured: "all pipelines in the HR workspace default to Two Approval Required"
- Declaration is immutable once set
- Tier elevation reason recorded: "elevated by data sensitivity declaration: PII"

The plugin floor governs what operations are performed. Data sensitivity governs what data is being processed. A `csv_source -> field_mapper -> csv_sink` pipeline at No-Review-Required becomes dangerous if the CSV contains medical records.

#### F12. Screening Aid / Research Finding Mode Toggle
**Panel support:** 1 of 6 (Dr. Sarah Chen), nuance on Marcus Webb's screening-aid default

Default operating mode is "screening aid / preliminary" — all LLM outputs are preliminary results requiring human disposition. For compliance contexts (Marcus's world), this is permanent.

For research contexts (Sarah's world), users can promote an output to "research finding" by attaching methodology documentation and completing a confirmation step. The promotion must be explicitly logged in the audit trail with: actor identity, timestamp, linked methodology documentation, and the specific output being promoted. This audit record is what distinguishes a preliminary screening result from an asserted research finding. The same system serves both personas.

---

## Formal Requirements (from Requirements Engineer)

The requirements engineer produced formal, testable requirement specifications. Key requirements with acceptance criteria:

| ID | Requirement | Testable Criterion |
|---|---|---|
| **REQ-TIER-001** | Pipeline tier computed server-side as max of all plugin tiers (precedence: Two Approval Required > Approval Required > No Review Required); model has no influence mechanism | Submit pipeline with mixed tiers; assert computed tier = max. Submit config with `review_tier_override` field; assert rejection. |
| **REQ-TIER-002** | Classification config changes require two-admin authorization with full audit logging | Attempt single-admin tier change; assert rejection. Complete with second admin; assert audit record. |
| **REQ-PREVIEW-001** | Automatic preview for No-Review-Required pipelines; no automatic preview for Approval-Required or Two-Approval-Required pipelines | Submit No-Review-Required pipeline; assert automatic preview. Submit Approval-Required; assert no preview until gate satisfied. |
| **REQ-PREVIEW-002** | Preview run records retained for minimum N days after task completion | Verify retention at N-1 days; verify policy application at N+1 days. |
| **REQ-PREVIEW-003** | Preview sample rows selected by platform, not model; selection recorded in audit | Verify `row_selection_policy` and `row_identifiers` in `preview_executed` event. |
| **REQ-AUTH-001** | Two-Approval-Required enforced server-side; same user cannot provide both signatures | Sign as creator, assert gate locked. Sign same user different session, assert `self_approval_forbidden`. Sign different reviewer, assert gate unlocks. |
| **AC-REVIEW-06** | Execution gate evaluates pipeline tier, not individual node acknowledgments | Acknowledge all nodes individually but don't satisfy pipeline gate; assert `can_execute() == False`. |
| **REQ-GEN-001** | Maintain benchmark suite of known-good pipeline generation tasks; LLM must achieve ≥90% structural accuracy before deployment | Run benchmark suite; assert ≥90% structural accuracy. Regression = deployment blocker. |
| **REQ-TENANT-001** | Database-per-tenant for regulated deployments, schema-per-tenant for standard | Submit query as Tenant A; assert no rows from Tenant B visible. |

---

## Risks and Dynamics Identified

### Emergent Behaviors (Systems Thinker)

| Dynamic | Description | Mitigation |
|---|---|---|
| **Review Theater** | Review checkboxes become performative rather than substantive | Eliminated by 3-tier model (no checkbox tier) |
| **Oracle Problem** | LLM generating both config AND summary creates correlated failure | Deterministic summaries + trial outputs (C3, C2) |
| **Eroding Goals** | Adoption pressure drives classification downgrades over time | Classification config governance (F1) |
| **Automation Complacency** | Users stop engaging with previews after repeated "correct" results (originally R4/R5 feedback loops from Systems Thinker Round 1) | Named risk; monitoring recommended |
| **Three-Tier Cliff Effect** | The cliff from No Review Required to Approval Required is steeper than the old 4-tier gradient. Administrators face pressure to classify plugins as No Review Required since there is no "soft" middle tier to absorb adoption pressure. The eroding goals dynamic concentrates at this single boundary. | Classification config governance (F1); monitoring of tier distribution over time |
| **Perverse Plugin Avoidance** | Users avoid approval-tier plugins (e.g., database_sink) and manually move data instead | Named by Sarah; monitoring recommended |
| **Plugin Substitution** | Users prompt model to use lower-tier plugin that accomplishes similar operation | Named risk; static security policy (F4) partially addresses |

### Architectural Risks (Architecture Critic)

| Risk | Description | Mitigation |
|---|---|---|
| **Redis pub/sub event loss** | Fire-and-forget drops events during reconnection | Replace with Redis Streams |
| **Stale mental model in design doc** | `settings.gates` reference reflects an obsolete plugin-based gate model; gate plugins were deliberately removed from ELSPETH. This is not just dead code — it indicates the design was written against a stale understanding of the engine. | Remove from design; verify no other gate-plugin assumptions persist |
| **Temporal cancel signal bug** | `_cancelled` set but never read; not initialized in `__init__` | Fix in implementation |
| **`TaskEvent.payload: dict[str, Any]`** | Violates ELSPETH's typed-boundary standards | Per-event-type frozen dataclasses |
| **`can_execute()` missing hash check** | Settings hash at execution time not verified against review-time hash | Add hash comparison (now covered by C1/C7) |
| **Schema-per-tenant WAL sharing** | Shared WAL/autovacuum/superuser in schema-per-tenant mode | Document limitations; recommend db-per-tenant for regulated (C8) |

---

## Design Document Corrections Required

The following items in the current design document need updating based on panel findings:

1. **Replace 4-tier classification with 3-tier enforcement model** — No Review Required / Approval Required / Two Approval Required
2. **Remove `settings.gates` reference** in `execute_pipeline` activity — gate plugins were removed from ELSPETH
3. **Fix `cancel_execution` signal handler** — initialize `_cancelled` in `__init__`, add read logic or remove dead signal
4. **Replace Redis pub/sub with Redis Streams** for telemetry channel — fire-and-forget is unacceptable for operational visibility
5. **Replace `PipelineDesign` with sealed `PipelineArtifact`** — add plugin_catalog_hash, classification_config_hash, elspeth_version, canonical_hash
6. **Add hash verification to `can_execute()`** — verify config hash at execution matches review-time hash
7. **Type `TaskEvent.payload`** — replace `dict[str, Any]` with per-event-type frozen dataclasses
8. **Clarify multi-tenancy recommendation** — schema-per-tenant (standard), database-per-tenant (regulated), with explicit guidance on which regulatory contexts require physical separation
9. **Resolve open design questions** — Q1 (multi-tenancy) resolved by C8, Q5 (templates) resolved by F6, Q7 (collaborative review) resolved by F2
10. **Remove AR-6 model_assessment concept** — The Architecture Critic's Round 2 AR-6 proposed `model_assessment` as an input to the tier `max()` computation; the product owner's clarification (system enforces, model has no influence) supersedes this. The model never touches enforcement.

---

## Open Design Questions (Remaining)

Of the original 8 open questions, 3 are resolved by this review. The remainder need product decisions:

| # | Question | Status |
|---|---|---|
| 1 | Multi-tenancy model | **Resolved** — schema-per-tenant (standard), db-per-tenant (regulated) |
| 2 | Data upload flow | **Open** — pre-upload to object store with reference is recommended path |
| 3 | LLM cost attribution | **Open** — per-task metering recommended; needs product decision |
| 4 | Iterative refinement scope | **Partially resolved** — refinement creates new artifact version; diff mechanism (F3) carries forward unchanged approvals |
| 5 | Template library | **Resolved** — yes, required for adoption (F6) |
| 6 | Offline/async execution | **Open** — Temporal supports this; async notification (F2) partially addresses |
| 7 | Collaborative review | **Resolved** — two-person approval workflow (F2) specifies the collaboration model |
| 8 | Plugin allow/deny per user/org | **Open** — panel recommends as a v2 feature; data sensitivity declaration (F11) partially addresses |

### New Questions Raised by This Review

| # | Question | Raised By |
|---|---|---|
| N1 | Does the preview run approval-required steps (e.g., database writes) or stop before them? Sarah recommends: preview should stop before approval-required nodes — the preview is a validation artifact, not an execution artifact, and running the sink would require approval before evaluation is possible. | Sarah Chen |
| N2 | Does the platform support sequential pipeline runs where Stage 1 output feeds Stage 2 input? | Sarah Chen |
| N3 | What is the naming for the "No Review Required" tier? Proposed alternatives: **"Auto-Approved"** (conveys the model completed its work) or **"Preview Only"** (conveys evidence is available without suggesting users should skip it). Both avoid implying users shouldn't review. | UX Specialist |
| N4 | When two plugins accomplish similar operations at different tiers, how is plugin substitution for tier-avoidance detected? | Systems Thinker |
| N5 | For the methodology citation export, what citation format standards should be supported? (APA, Chicago, BibTeX) | Sarah Chen |
| N6 | What formal model validation documentation is required for regulatory compliance (SR 11-7 equivalent for LLM-driven systems)? | Marcus Webb |
| N7 | What data residency agreements are needed with cloud LLM providers, and do they vary by jurisdiction? | Marcus Webb |
| N8 | What regulatory notification requirements apply to automated AML systems, and how do they vary by jurisdiction? | Marcus Webb |

---

## Priority Ranking (Panel-Weighted)

Items ranked by: (a) number of panelists who independently identified the need, (b) severity if missing, (c) blocking vs. enhancing.

| Priority | Item | Panelist Count | Blocking? |
|---|---|---|---|
| **P0** | 3-tier enforcement model (replaces 4-tier) | 6/6 | Yes — governance foundation |
| **P0** | System-enforced execution gates (C4) | 6/6 | Yes — cannot ship without |
| **P0** | Sealed PipelineArtifact (C1) | 6/6 | Yes — audit integrity |
| **P1** | Classification config governance (F1) | 3/6 | Yes — "who governs the governors" |
| **P1** | Two-person approval workflow (F2) | 2/6 | Yes — 2AR tier is unusable without it |
| **P1** | Automatic micro-execution (C2) | 6/6 | Yes — core value proposition |
| **P1** | Deterministic preview row selection (C6) | 3/6 | Yes — preview evidence quality |
| **P2** | Static security policy enforcement (F4) | 1/6 | Yes — primary control for no-gate flows |
| **P2** | Plugin catalog version pinning (F5) | 1/6 | Yes — execution integrity |
| **P2** | Refinement diff / selective re-approval (F3) | 1/6 | Adoption risk if missing |
| **P2** | Template library (F6) | 2/6 | Adoption risk if missing |
| **P2** | Dual-layer summaries (C3) | 6/6 | Quality — can launch without but shouldn't |
| **P3** | Methodology citation export (F7) | 1/6 | Research adoption |
| **P3** | Quarantine explanation (F8) | 1/6 | Research adoption |
| **P3** | Reference data versioning (F9) | 1/6 | Regulated deployment |
| **P3** | User-specified test cases (F10) | 2/6 | Regulated deployment |
| **P3** | Data sensitivity declaration (F11) | 1/6 | Context-aware risk |
| **P3** | Screening aid / research finding toggle (F12) | 1/6 | Persona flexibility |

---

## Panelist Final Positions

**Systems Thinker:** "The three-tier model plus system enforcement is a strong foundation. The 8 consensus points address the most acute architectural gaps." Top concern: tier reclassification governance has the longest time horizon of impact and the lowest implementation cost.

**Architecture Critic:** "If the design incorporates the 8 consensus points and these 3 additions, the remaining open questions are product decisions, not architectural blockers." Top concern: static security enforcement is the primary technical control for no-gate flows.

**UX Specialist:** "If the design document incorporates the 8 consensus points and these 3 additions, the major remaining risks are addressed." Top concern: refinement diff prevents re-approval fatigue that kills adoption.

**Requirements Engineer:** Produced 9 formal testable requirements with acceptance criteria (REQ-TIER-001/002, REQ-PREVIEW-001/002/003, REQ-AUTH-001, AC-REVIEW-06, REQ-GEN-001, REQ-TENANT-001). Top concern: the distinction between pipeline-level enforcement and per-node UX signals must be explicit in the implementation. **Noted gap:** No non-functional requirements for performance (concurrent users, max pipeline size), reliability (SLA, recovery time), or security beyond SSO — these need specification before architecture proceeds.

**Dr. Sarah Chen:** "Build those three [citation export, templates, quarantine explanation] and you have a tool I'd adopt. Build the eight consensus points without them and you have a technically sound system that non-technical users can't fully leverage." Top concern: the audit trail is only valuable if users can use it for professional accountability.

**Marcus Webb:** "I can make a credible case to my CCO for piloting this system in a controlled capacity — specifically for pre-screening workflows where human analysts review all flagged items before regulatory action. That's a meaningful change from my Round 1 position." Top concern: reference data versioning is the gap that makes the audit trail legally incomplete.

---

## Appendix: Consensus Formation Timeline

| Round | Key Development |
|---|---|
| **Round 1** | All 6 panelists independently identified "Review Theater" (rubber-stamping risk). Oracle Problem and frozen-YAML inadequacy surfaced. Marcus raised four-eyes as non-negotiable. |
| **Clarification 1** | Product owner: reasoning model actively builds flows; low-risk flows get automatic preview. |
| **Clarification 2** | Product owner: system enforces policy, model presents. Resolved largest structural concern. |
| **Clarification 3** | Product owner: pipeline tier = most restrictive plugin. |
| **Clarification 4** | Product owner: 3-tier model (no-review / approval / two-approval). Eliminated checkbox tier. |
| **Round 2** | Constructive proposals: PipelineArtifact (architect), prediction protocol (systems thinker), anti-rubber-stamping UX (UX specialist), minimum viable governance (Marcus), formal requirements (RE). |
| **Round 3** | Final convergence: 8 consensus points with unanimous consent. 18 individual top-3 items consolidated into 12 features across 3 priority tiers. All panelists shifted from critical to constructive posture. Marcus moved from "not ready for regulated deployment" to "credible case for controlled pilot." |
