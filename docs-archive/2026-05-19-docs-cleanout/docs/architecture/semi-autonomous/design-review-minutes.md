# Semi-Autonomous Platform — Design Review Panel Minutes

**Date:** 2026-03-02
**Facilitator:** Claude (Team Lead)
**Input Document:** `docs/architecture/semi-autonomous/design.md`
**Method:** 3-round facilitated panel discussion

> **Note on source fidelity:** Round 3 messages and Round 2 addenda are verbatim transcripts.
> Round 1 and Round 2 initial submissions are reconstructed from the session summary
> generated during context compaction — the original messages were stripped. Reconstructed
> sections are marked with [RECONSTRUCTED FROM SESSION SUMMARY].

---

## Panel Members

| ID | Role | Agent Type | Perspective |
|---|---|---|---|
| `systems-thinker` | Systems Thinker | general-purpose (sonnet) | Feedback loops, emergent behaviors, system archetypes, leverage points |
| `architect-critic` | Architecture Critic | general-purpose (sonnet) | Structural quality, technology fitness, security, complexity |
| `ux-specialist` | UX Specialist | general-purpose (sonnet) | Cognitive load, review fatigue, workflow coherence, accessibility |
| `requirements-engineer` | Requirements Engineer | general-purpose (sonnet) | Completeness, testability, ambiguity, traceability |
| `user-sarah` | Dr. Sarah Chen | general-purpose (sonnet) | Senior research analyst persona — policy research, publication attribution |
| `user-marcus` | Marcus Webb | general-purpose (sonnet) | Senior compliance officer persona — regulated financial services |

---

## Product Owner Clarifications (Interjected Between Rounds)

Four clarifications were provided by the product owner during the discussion. These materially changed the design direction.

### Clarification 1: Reasoning Model as Active Builder
> "for clarity, the model will be a reasoning model that is actively building the flow - for appropriately low risk flows at least. So it could do 'hey, here's all the parts', I ran a short run and here's what we got back. Whereas many other flows would be 'ok, I built this here. It's a two person approval and I need test data to validate the schemas'"

### Clarification 2: System Enforces Policy, Not Model
> "also it wouldn't be 'agent gets to assess', the system wouldn't execute without two human signatures on the handle for a two man run"

### Clarification 3: Pipeline Tier = Most Restrictive Plugin
> "(in the same way we do determinism, the idea would be that the approval/risk setting would be the most restrictive of all plugins)"

### Clarification 4: Three-Tier Risk Model
> "(also lets think in terms of 'no-review required', 'approval required', 'two approval required' as the baseline for risk, they can add or enhance or tweak it as appropriate)"

---

# ROUND 1: Initial Analysis

*Each panelist independently read the design document and provided their first-pass analysis.*

---

## Dr. Sarah Chen — Round 1

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key findings:**
- Identified workflow gaps: no test-before-commit capability, no error diagnosis workflow, no pipeline templates, no PII warnings
- Walked through a real task (5,000 public comments classification) and found the system falls short on iteration — "my first run classifies 60% correctly, what do I do?"
- Questioned whether junior analysts could use the graph editor at all
- Raised concerns about missing features: templates, reusable pipelines, sharing results, comparing runs, version history
- Expressed excitement about the concept but skepticism about the review process — "Can I tell if the LLM prompt is good? Would I even know what to check on an 'aggregation' node?"

---

## UX Specialist — Round 1

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key findings:**
- **P0: Review fatigue / rubber-stamping** — The "I've reviewed this step" checkbox creates performative review, not substantive engagement. Users will develop click-through habits.
- Error recovery is absent — no path for users when the LLM generates a bad pipeline, when validation fails, or when execution fails mid-way
- Accessibility gaps — graph editors are notoriously difficult for screen readers; the summary report alternative may not provide full functionality
- Cognitive load from dual-view model (summary + graph) needs careful progressive disclosure
- The review classification UX is unintuitive — users won't understand why they're being asked to review certain nodes

---

## Architecture Critic — Round 1

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key findings:**
- **"Frozen YAML" claim is false** — The artifact boundary has at least 5 hidden dependencies beyond the YAML: ELSPETH version, plugin versions, data content, review tier config, runtime environment
- **Stale mental model:** `settings.gates` reference in `execute_pipeline` activity — gate plugins were deliberately removed from ELSPETH codebase (CLAUDE.md: "NO GATE PLUGINS"). This is not just dead code — it indicates the design document was written against an obsolete understanding of ELSPETH's architecture where gates were plugins rather than config-driven expressions
- **Redis pub/sub is wrong technology** — fire-and-forget means dropped events during reconnection; should use Redis Streams
- **Temporal oversold** — the comparison table omits Temporal's complexity costs (polyglot workers, replay semantics, determinism constraints)
- **LLM summary audit integrity risk** — the LLM generates both the config AND the summary describing it, creating correlated failure modes (Oracle Problem)
- Security gaps identified: no static analysis of generated YAML, no URL allowlisting for web_scrape
- **`can_execute()` missing hash verification** — doesn't verify settings_hash at execution time matches review-time hash
- **Temporal cancel signal bug** — `_cancelled` flag is set by signal handler but never read in the workflow; also not initialized in `__init__`, making it a dead signal

---

## Systems Thinker — Round 1

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key findings:**
- **Review Theater archetype** — The review classification system exhibits "Shifting the Burden": the real solution is meaningful human engagement, but the system substitutes checkbox acknowledgment, which produces a paper trail without substantive review
- **Oracle Problem** — The LLM generates both the pipeline config AND the summary describing it. If the LLM has a systematic blind spot, both the config and the explanation share that blind spot. The summary confirms the config because they come from the same source.
- **Trust self-referentiality** — The system asks users to trust LLM output by reviewing an LLM-generated summary of that output
- Identified 5 leverage point interventions using Meadows' hierarchy
- **Feedback loops identified:**
  - R1 (reinforcing): More successful runs → more user trust → less scrutiny → potential quality degradation
  - R2 (reinforcing): Good review outcomes → faster reviews → less thorough reviews
  - R3 (reinforcing): Template reuse → standardization → reduced prompt diversity → narrowing of LLM capability
  - R4 (reinforcing): Automation complacency toward risk classification — repeated "correct" results erode user engagement with previews
  - R5 (reinforcing): Prompt engineering for lower tier — users discover they can phrase requests to avoid higher-tier plugins
  - B1 (balancing): Quarantine rate as a natural quality signal

---

## Requirements Engineer — Round 1

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key findings:**
- **5 of 8 open questions are actually missing requirements** — not design choices but gaps that must be answered before architecture proceeds
- **Review invalidation hole** — `can_execute()` doesn't verify `settings_hash` at execution time matches the hash at review time. User reviews version A, system executes version B.
- **Typed event payloads needed** — `TaskEvent.payload: dict[str, Any]` violates CLAUDE.md's typed-boundary standards. Should be per-event-type frozen dataclasses.
- Full NFR gap analysis: no performance requirements (concurrent users, max pipeline size), no reliability requirements (SLA, recovery time), no security beyond SSO
- **Ambiguity analysis:** "non-technical users" undefined (how non-technical?), "natural language" undefined (any language?), review classification terms used inconsistently
- Extracted implicit user stories for roles beyond "non-technical user": admin who configures review tiers, auditor who traces lineage, ops who monitors pods

---

## Marcus Webb — Round 1

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key findings:**
- **Regulatory defensibility gaps** — walked through the audit trail from prompt → config → execution → Landscape → result and found a gap: the link between the Conversation Service event log and the Landscape `run_id` needs to be bidirectional and cryptographically verified
- **Sanctions screening cannot be done with current plugins** — requires reference data versioning (which OFAC SDN list version was used?), threshold management, and near-miss detection capabilities
- **Non-determinism is a blocker** — running the same pipeline on the same data twice may produce different results (LLM non-determinism). This fundamentally undermines regulatory defensibility.
- **Four-eyes is mandatory, not optional** — the design's "collaborative review" is listed as an open question (#7). For financial services, it's a non-negotiable requirement.
- **Immutable config storage needed** — pipeline configs must be write-once with cryptographic verification, not mutable YAML files
- **Data handling concerns** — where does the data go? Encrypted at rest and in transit? Data residency?

---

## Facilitator Round 1 Synthesis

**8 cross-cutting themes identified:**

1. **Review Theater** (all 6 panelists) — The acknowledgment checkbox creates performative compliance, not substantive review
2. **Oracle Problem** (systems thinker, architect) — LLM self-describing its own output creates correlated failure
3. **Frozen YAML not frozen** (architect, requirements engineer) — At least 5 hidden dependencies beyond the YAML
4. **Missing test-before-commit** (Sarah, Marcus) — No way to validate pipeline output before full execution
5. **Security surfaces** (architect) — Generated YAML not analyzed for malicious patterns
6. **5/8 open questions are missing requirements** (requirements engineer) — Not design choices, but gaps
7. **Error recovery absent** (UX specialist) — No user path when things go wrong
8. **Regulated industry gaps** (Marcus) — Four-eyes, immutable storage, reference data versioning, determinism

---

# ROUND 2: Constructive Proposals

*Each panelist responded to Round 1 themes with concrete proposals. Product owner clarifications 1 and 2 were broadcast between Round 1 and Round 2.*

---

## Architecture Critic — Round 2

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key proposals:**

### Sealed PipelineArtifact Model (Replacing "Frozen YAML")

```python
@dataclass(frozen=True)
class PipelineArtifact:
    pipeline_yaml: str
    data_reference: DataReference  # URI + content hash
    plugin_catalog_hash: str       # SHA-256 of serialized plugin catalog
    elspeth_version: str           # Pinned engine version
    review_actions: list[ReviewAction]  # Timestamped approval records
    canonical_hash: str            # RFC 8785 + SHA-256 over all above fields
```

### Preview Architecture

Proposed `HeadSourceWrapper` (wraps any source, yields only first N rows) + `PreviewCaptureSink` (captures output without writing to final destination) for micro-execution.

### Schema-Per-Tenant Recommendation

Database-per-tenant for regulated environments, schema-per-tenant with PostgreSQL `SET search_path` for standard deployments.

### Architectural Requirements

- **AR-1:** Artifact boundary must capture ALL execution dependencies
- **AR-2:** Preview execution must use full ELSPETH engine, not simulation
- **AR-3:** Summary generation must be deterministic (derivable from GraphDescriptor), not LLM-only
- **AR-4:** Review actions must bind to specific config hash versions
- **AR-5:** Tenant isolation must be database-enforced, not application-enforced
- **AR-6:** Model risk assessment as tier input — proposed `model_assessment` as a secondary input to `max(platform_floor, model_advisory)` computation. *(Note: superseded by product owner's Clarification 2 — model has no influence on tier computation. Acknowledged by Architecture Critic in Round 3.)*

---

## Systems Thinker — Round 2

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key proposals:**

### Prediction-Before-Execution Protocol

Replace "I've reviewed this step" checkbox with "What do you expect this step to produce?" — forces cognitive engagement. The user states their expectation, the system runs the preview, and the user compares actual output to their prediction. This is behavioral rather than procedural governance.

### Oracle Problem Analysis

Deterministic summaries (derived from GraphDescriptor) help but don't fully resolve the Oracle Problem — the LLM still generated the GraphDescriptor. Trial outputs on real data are the strongest evidence because they're produced by ELSPETH, not the LLM.

### 5 System Interventions at Leverage Points

Proposed interventions at Meadows' levels 4 (rules), 6 (information flows), 8 (negative feedback loops), 10 (paradigm), and 12 (transcending paradigms).

---

## UX Specialist — Round 2

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key proposals:**

### Anti-Rubber-Stamping Patterns

- Trial output + "does this match what you expected?" replaces acknowledge checkbox
- Automatic micro-execution as default UX, not a user-initiated feature

### Dual-Layer Summaries

- Layer 1: Deterministic technical specification (audit-grade, derived from GraphDescriptor)
- Layer 2: LLM interpretation (explicitly labeled "may not be accurate")
- Layer 3: Trial results from actual preview execution

### UX Requirements

- **R1:** The default view must show trial results alongside the pipeline description
- **R2:** Review gates must require behavioral engagement (prediction, comparison), not checkboxes
- **R3:** Error recovery must show the user what went wrong and suggest specific actions
- **R4:** Accessibility: summary report must provide 100% functional parity with graph editor. React Flow (the graph editor library) is notoriously hostile to screen readers — the summary report is not an optional alternative, it is the primary accessible interface and must be treated as such
- **R5:** Progressive disclosure: novice users see summary, power users access graph, both paths complete

---

## Dr. Sarah Chen — Round 2

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key proposals:**

### Ideal Iteration Cycle

When a pipeline produces suboptimal results, Sarah wants: error pattern diagnosis (which rows failed and why), targeted re-validation (run just the changed part on edge cases), and run comparison (diff between run 1 and run 2 outputs).

### Day-One Feature Requirements

1. **Test run on sample data** — must be default, not optional
2. **Pipeline save/reuse** — templates and personal library
3. **PII warning** — automatic detection when data looks like it contains personal information
4. **Completion notification** — email/notification when long-running pipeline finishes
5. **Quarantine explanation** — plain-English reasons, not error codes

### Methodology Export

Need to cite pipeline methodology in published research: model name, prompt text, parameters, row counts, quarantine rates. The audit trail only matters if users can reference it in professional contexts.

---

## Marcus Webb — Round 2

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key proposals:**

### Minimum Viable Governance

1. **Immutable config storage** — write-once with SHA-256 hash verification on every retrieval
2. **Four-eyes enforcement** — system-enforced, not process-dependent
3. **Reference data versioning** — hash and snapshot all classification reference data at execution time
4. **Admin audit trail** — every change to platform configuration logged with actor identity

### Screening Aid vs Determination

All LLM outputs should be labeled as "preliminary screening results" requiring human disposition. No regulatory action without human sign-off. This isn't a feature request — it's a compliance requirement. The system's output is a screening aid, not a determination.

### 5 Regulator-Ready Requirements

Concrete requirements for regulatory examination survival:
1. Complete bidirectional lineage from output to prompt
2. Non-repudiation of approval decisions
3. Reference data version pinning
4. Data residency documentation
5. Model validation documentation (SR 11-7 equivalent)

---

## Requirements Engineer — Round 2

[RECONSTRUCTED FROM SESSION SUMMARY]

**Key proposals:**

### Formal Acceptance Criteria (AC-REVIEW-01 through AC-REVIEW-05)

Per-node review criteria with testable assertions. Each criterion specifies: the condition, the expected system behavior, and a test that verifies it.

### Multi-Tenancy Resolution

- **REQ-TENANT-001:** Database-per-tenant for regulated deployments, schema-per-tenant for standard
- Testable: submit query as Tenant A, assert no rows from Tenant B visible

### Generation Quality Benchmark

- **REQ-GEN-001:** Maintain a benchmark suite of known-good pipeline generation tasks. The LLM must achieve ≥90% structural accuracy on the benchmark before deployment. Regressions in generation quality are deployment blockers.

---

# ROUND 2 ADDENDA: Responses to Product Owner Clarifications

*Product owner clarifications 2-4 were broadcast mid-discussion. These addenda were delivered in the current session and are verbatim.*

---

## Marcus Webb — Round 2 Addendum (Verbatim)

### ON THE CORRECTION (System enforces policy, not model)

This is the most important clarification and it resolves my biggest structural concern. I want to be precise about what it changes.

**What's now resolved:** My four-eyes requirement. If the system physically cannot execute an approval_required pipeline without two distinct authenticated signatures — enforced server-side, not checked at the frontend — that's structural control. I can present that to a regulator. "The system will not proceed without two human authorizations" is a defensible statement. "We require two people to review before running" is not — because it's a process that can be bypassed.

The distinction between "model suggests governance" and "system enforces governance" is exactly the right one. The model builds and presents. The system gates execution. Good.

**What remains as my concern:** The plugin review classification configuration is now the single most important governance artifact in this entire system. If `llm` is misconfigured as `visible` instead of `review_required`, every LLM pipeline executes without the required acknowledgments. If `database_sink` is misconfigured as `review_required` instead of `approval_required`, database writes proceed with a single person's acknowledgment instead of two signatures.

The policy is only as strong as the governance around the config that defines it. Who can edit that YAML? How are changes approved? Is the config versioned and signed? This needs the same immutable-storage treatment I proposed for pipeline configs. A change to the classification config is a material change to the firm's compliance controls — it must be treated with the same rigor as changing a written compliance procedure.

Concretely: the classification config needs a change approval workflow (not just version control), a separate two-person authorization to make changes, and every change logged with who made it, who approved it, and when. This is the "who governs the governors" problem, and it's not currently addressed in the design.

### ON PIPELINE TIER = MOST RESTRICTIVE PLUGIN

This is architecturally correct and I'm glad it's been stated explicitly. It's how I'd expect it to work and how any reasonable compliance framework would operate.

One practical implication worth flagging: this means pipeline composition matters for approval tier. If I have a standard low-risk classification pipeline and someone adds a `database_sink` step at the end (to write results to our case management system), the entire pipeline immediately becomes `approval_required`. That's correct — but it means users will need to understand why their previously-approved pipeline now requires two signatures. The explanation ("because you added a database_sink, which always requires two signatures") needs to be clear and persistent in the UI, not buried.

The per-node detail showing which plugin drives the elevated tier is important to keep even though enforcement is pipeline-level. If I'm the Approver on a pipeline, I need to know what I'm approving and why it requires my countersignature specifically.

One question on mechanics: if a pipeline is `approval_required` and both signatures are collected, can a component be subsequently modified without re-triggering the approval? The system should invalidate all collected signatures whenever the pipeline config changes — even a minor change should require the approval cycle to restart. Hash-based invalidation: if the config hash changes, all signatures on the previous hash are void.

### ON THE PROACTIVE PREVIEW

This genuinely changes the trust calculus for low-risk flows. Concrete output from a real sample run is significantly more informative than the LLM's description of what it intends to do. I want to note where this helps and where it doesn't resolve my concerns.

**Where it helps:** For classification pipelines where I want to see "does the LLM actually categorize these transaction types as expected?" — a sample run answers that question directly. I can see real outputs before committing. The "oracle problem" is partially mitigated, as stated.

**Where it doesn't close the gap:**

First, sample selection is now a governance question. Who picks the preview rows? If the system randomly samples 5 rows from the source, I may not be testing the cases that matter for compliance. For sanctions screening, I need the preview to include near-miss cases — transactions that are similar to known designees but aren't matches. I need known false-positive and false-negative test cases. A random sample of clean transactions tells me nothing about the LLM's behavior at the edge cases.

This leads to a requirement I'd add to my v1 list: for pipelines classified as `review_required` or above, the preview sample should be user-specified or drawn from a maintained set of validation cases, not random. The analyst should be able to say "run the preview on these 20 known test cases" and see how the pipeline handles them.

Second, the preview is still conducted by the same LLM that generated the config. The preview confirms "the pipeline produced these outputs on this sample" — it doesn't independently validate that the pipeline logic is correct. If the LLM wrote a subtly wrong gate expression, the preview might not expose it on 5 rows. This isn't a reason to skip the preview — it's a reason to be clear about what the preview does and doesn't prove.

For audit purposes, the preview run needs to be recorded in the event log with the same granularity as a full run: which rows were sampled, what the LLM produced for each, and the timestamp. The Approver's signature should reference the preview run_id alongside the pipeline config hash. That way, the audit trail shows: "the Approver reviewed the pipeline design AND saw these specific preview outputs before signing."

### REVISED RISK CLASSIFICATION GOVERNANCE QUESTION

The team lead asked: "How does the model's risk assessment get audited?" The correction changes this to: "How does the plugin classification configuration get audited?"

My answer: same as pipeline configs. The classification config must be:

1. **Version-controlled and immutable in production** — changes create a new version, old versions are retained permanently. This is not optional.
2. **Subject to a two-person authorization** — any change to the classification of any plugin requires a separate approval cycle, distinct from pipeline execution approvals.
3. **Fully event-logged** — every change to the config is recorded: who proposed it, who approved it, what changed, when.
4. **Referenced by pipeline executions** — each run in the event log must record which version of the classification config was active at the time of execution. If the config changes between two runs of the same pipeline, the audit trail reflects that.

This is not significantly more complex than what's already proposed for pipeline configs — it's the same pattern applied to the governance layer itself. It's also what I'd need to show a regulator who asks "how do you know your approval requirements were correct during this period?"

### NET ASSESSMENT AFTER CLARIFICATIONS

My Round 2 requirements remain intact, but I can update the confidence level on two of them:

**Four-eyes requirement (#2):** Now substantially addressed by the system-enforced execution gate. The remaining work is ensuring the authentication and signature mechanism is robust (no shared accounts, timestamp and IP logging, signature invalidation on config change).

**Screening-aid mode declaration (#4):** The proactive preview for low-risk flows partially addresses the "user sees real data before committing" aspect. The remaining work is ensuring the preview sample includes compliance-relevant test cases, not just random rows.

**The new critical requirement I'd add to the list:** Classification config governance — the YAML that defines plugin review tiers must have the same immutability, version control, two-person authorization, and audit trail as pipeline configs themselves. The enforcement is only as good as the policy, and the policy is only as good as the governance around it.

---

## Dr. Sarah Chen — Round 2 Addendum (Verbatim)

### On Policy Enforcement: My Fox-and-Henhouse Concern Is Resolved

The correction addresses exactly what I was worried about. If the system enforces approval requirements based on plugin configuration, and the model cannot override that enforcement, then the accountability chain is:

**Administrators configure policy → System enforces it → Model explains it → User executes within those constraints**

That's clean. That's auditable. That's something I can explain to my IT department and my PI without hedging. "The system requires two signatures because it's configured to require two signatures for any pipeline that uses a database sink. The model can't change that." That's a policy, not a judgment call. Policies are defensible.

My previous concern was that "the model assessing its own risk" created an accountability gap — if something went wrong, neither the model nor the user clearly owned the decision. The clarified model is: the *policy* owns the enforcement, humans own the policy configuration, and the user's role is to operate within those constraints. That's how organizational compliance actually works. I'm comfortable with that structure.

### On Pipeline-Level Tier: This Is Actually Better for My Workflow

My first instinct was that "the whole pipeline inherits the strictest constraint" might feel punitive. If I add a database sink, suddenly my straightforward classification pipeline needs two signatures. That seems like friction.

But I've thought about it differently now, and I think the pipeline-level model is actually more correct than per-node approval would be.

Here's why: when my PI approves a pipeline run, she's approving the *whole operation*, not individual steps. "I authorize the classification AND the results going to our research database" is one decision with one context. If she had to approve the LLM node separately from the database sink node, she'd be approving abstractions she can't fully evaluate in isolation — she'd have no way to see that the LLM's output feeds directly into the database write. Pipeline-level approval preserves the operational context that makes approval meaningful.

The per-node UI detail — "this pipeline is approval_required because it contains `database_sink`" — is explanatory rather than enforcement, and that's the right division. She understands *why* the pipeline is at this tier. She approves the whole thing at once. Efficient, coherent, auditable.

**One important question this raises:** What is the approval tier of the *preview run*?

If the preview runs the full pipeline on 10 rows — including the LLM transform AND writing 10 rows to the database — does the preview require the same two-signature approval? Or is the preview considered a separate, limited operation?

My intuition: the preview should not include the `approval_required` step. The preview should run only up to the last non-approval-required node and show me the output *before* the database write. That way I can validate the classification results without triggering the approval gate. The approval gate fires when I commit to the full execution that includes the database write. If the preview also wrote to the database, I'd need approval before I could even evaluate whether the pipeline design is correct — which defeats the purpose of the preview.

Stated differently: the preview is a validation artifact, not an execution artifact. It should operate within the `review_required` tier regardless of what plugins are downstream.

### The Perverse Incentive I'm Watching For

If adding a `database_sink` escalates the entire pipeline to `approval_required`, users who want to avoid the two-signature process will be tempted to output to CSV instead and manually move data into the database afterward. That's less auditable, not more — the "manual move" step falls outside the pipeline's audit trail entirely.

I'd watch for this pattern in adoption data. If users in approval-required environments systematically avoid database sinks in favor of CSV downloads plus manual uploads, the governance intent is being achieved procedurally (no unauthorized database writes) but the auditability is actually worse (the final state of the data isn't in the pipeline's provenance).

I don't have a solution to this — it may be an inherent tension between governance friction and workflow quality. But it's worth naming in the design review so the team can monitor for it.

### On Multi-Step Pipelines: A Practical Question

This might be outside the scope of Round 2, but the pipeline-tier model raises a workflow question I'd actually face:

Can I run a pipeline in two stages? Stage 1: Classify (LLM transform → CSV sink, tier: `review_required`). Stage 2: Load validated results to database (CSV source → database sink, tier: `approval_required`). Both stages are separate pipeline runs with separate approvals.

This is effectively the "classify first, validate, then persist" workflow that careful research practice would call for anyway. The database write happens only after I've confirmed the classifications are good. Under the pipeline-tier model, Stage 1 is `review_required` and Stage 2 is `approval_required`. Approval for Stage 2 is meaningful because the PI can see the actual classified results before authorizing the database write.

If the platform supports sequential pipeline runs with the output of one feeding the input of the next — and I think it must, since CSV sink + CSV source is a natural handoff — then the pipeline-tier model actually encourages good research practice rather than frustrating it.

### Summary: What the Corrections Change

**My previous concerns that are now resolved:**
- The model cannot self-downgrade an approval requirement (system enforces, model explains)
- Accountability for the approval level is with the policy configuration, not the model's judgment
- The four-eyes requirement is structural, not optional — that's the right design for regulated/published work

**Questions that remain:**
1. Does the preview run the full pipeline (including approval-required steps) or stop before them?
2. Does the platform support sequential pipeline runs where Stage 1 output feeds Stage 2 input?
3. Who monitors adoption patterns to detect the perverse incentive of avoiding approval-required plugins?

**What I'd tell my PI:** *"The system requires your co-approval on any run that writes to our research database. You'll receive an email with the pipeline summary, the 10-row preview results, and a plain-English explanation of what the run will do. You approve the whole operation, not individual steps. If you're unavailable, the run waits — it won't execute without your signature."*

That I can say with confidence. That's a governance model she'd accept.

---

## Requirements Engineer — Round 2 Supplement (Verbatim)

*Delivered as "Round 2 Supplement (Clarifications Incorporated)"*

### What the Clarifications Change

#### Clarification 1 → 2 (net effect after correction)

The model is a **reasoning builder and presenter**, not a risk assessor. The system is the **policy enforcer**. This is a clean separation of mechanism from policy:

| Responsibility | Owns |
|---|---|
| Build pipeline config | Reasoning model |
| Run proactive preview (low-risk flows) | Platform (triggers model + ELSPETH) |
| Determine required approval tier | Platform (reads `plugin_review_classification` config) |
| Enforce approval gate | Platform (server-side, independent of model) |
| Explain which plugins drive the tier | Model (informational only) |
| Override or downgrade the tier | **Neither** — not possible |

#### Clarification 3 (pipeline tier = max plugin tier)

My Round 2 per-node review criteria (AC-REVIEW-01 through AC-REVIEW-05) were framed around individual node acknowledgements. These need to be reframed: enforcement is at the **pipeline level** as a single gate. Per-node indicators (yellow/red border) remain as UX signals explaining *which* plugin drives the tier, but the execution gate is singular.

This changes:
- `can_execute()` logic: checks pipeline tier, not per-node tier
- REQ-AUTH-001 (four-eyes): applies to the pipeline, not individual nodes
- AC-REVIEW-03 (hash verification): still valid but now hashes the set of user-visible settings for all review-influencing nodes, not individual nodes
- F-UX-01 (test-on-sample): no longer a user-initiated feature for low-risk flows — it's the model's default behavior

### Updated and New Requirements

#### REQ-TIER-001 (new): Pipeline Tier Computation — System-Enforced, Model-Blind

> *[Editorial note: This requirement was written before Clarification 4 (three-tier model). The precedence should be updated from 4-tier (`approval_required > review_required > visible > transparent`) to 3-tier (`Two Approval Required > Approval Required > No Review Required`). See synthesis document for the corrected version.]*

> The platform SHALL compute the pipeline execution tier as the maximum review tier across all plugins present in the pipeline config, using the current `plugin_review_classification` configuration. The computation is:
>
> `pipeline_tier = max(tier(plugin) for plugin in pipeline_plugins)`
>
> with precedence: `approval_required` > `review_required` > `visible` > `transparent`.
>
> This computation SHALL be performed server-side by the platform at config validation time, NOT by the reasoning model. The model SHALL have no mechanism — no API, no instruction, no output field — to propose, suggest, or influence the computed tier. If the model's generated config contains an unclassified plugin, that plugin defaults to `review_required` (fail-closed, per existing policy).
>
> The computed tier, the list of plugins contributing to it, and the plugin driving the maximum tier SHALL be recorded in the meta-audit log in a `tier_computed` event.

**Testable:** Submit a pipeline containing `csv_source` (visible), `llm` (review_required), and `database_sink` (approval_required). Assert: `pipeline_tier == approval_required`. Assert `tier_computed` event in meta-audit identifies `database_sink` as the driving plugin. Attempt to submit a config with a model-provided `review_tier_override` field — platform SHALL reject it.

#### REQ-TIER-002 (new): Policy Config Integrity

> The `plugin_review_classification` config is itself a security-critical artifact. Changes to it SHALL require Administrator role AND a second Administrator to confirm the change (two-admin authorization). Every change SHALL be recorded in the admin audit log with: prior value, new value, both admin actors, and timestamp.
>
> The platform SHALL record the hash (RFC 8785 + SHA-256) of the active `plugin_review_classification` config in every `tier_computed` event, enabling post-hoc verification that the tier was computed using the policy that was in effect at that time.

**Rationale:** If an attacker lowers a plugin's tier in the classification config, every subsequent pipeline using that plugin silently drops its approval requirements. The admin audit trail (F-ADMIN-01) detects this, but the config hash in the `tier_computed` event enables forensic reconstruction: "At execution time T, the policy in effect was version V with hash H. Here is what changed between V and the current policy."

**Testable:** Lower the tier of `database_sink` from `approval_required` to `visible` as a single admin. Platform SHALL reject the change with `second_admin_required`. Confirm with a second admin. Change goes through. Admin audit log records both actors, prior value `approval_required`, new value `visible`, timestamps.

#### REQ-PREVIEW-001 (new): Proactive Preview Execution for Low-Risk Pipelines

> For pipelines with computed tier `transparent` or `visible`, the platform SHALL automatically execute a preview run on N sample rows (N configurable, default 5) using the full ELSPETH engine before presenting the pipeline design to the user. The reasoning model SHALL present both the pipeline design AND the actual preview outputs in the same view.
>
> The preview SHALL be a real ELSPETH pipeline execution with full Landscape recording. The preview `run_id` SHALL be recorded in the meta-audit log as a `preview_executed` event linked to the task.
>
> For pipelines with tier `review_required` or `approval_required`, automatic preview SHALL NOT execute. A user-initiated "Run sample" action may be offered but SHALL itself require the pipeline-tier approval gate to be satisfied first.

**Rationale:** This converts the "oracle problem" from "the model describes what will happen" to "here is what actually happened on 5 rows." For low-risk flows, the evidence is concrete. For high-risk flows, the approval gate ensures authorization precedes any data processing — even a preview.

**Testable:** Submit a `transparent`-tier pipeline. Assert: (1) preview executes automatically without user action, (2) Landscape records exist for the preview run, (3) `preview_executed` meta-event is present with `run_id`, (4) preview results are displayed to user before execution gate. Submit an `approval_required`-tier pipeline. Assert: preview does NOT execute until approval gate is satisfied.

#### REQ-PREVIEW-002 (new): Preview Evidence Retention

> Preview run records, Landscape data, and result samples SHALL be retained in the meta-audit log for a minimum of [N days] after task completion, to allow auditors to verify the evidence the user saw before approving a full run.
>
> Preview runs SHALL be tagged `run_type: preview` in both the `execution_started` meta-audit event and the Landscape `runs` table. Preview payloads in object store SHALL follow the same retention policy as full run payloads unless overridden by platform configuration.

**Testable:** Run a low-risk pipeline. Complete the full run. Advance time by [N-1] days. Assert preview Landscape records still exist. Advance to [N+1] days. Assert retention policy applied.

#### REQ-PREVIEW-003 (new): Model Cannot Influence Sample Selection

> **REQ-PREVIEW-003:** The preview sample rows SHALL be selected by the platform using a deterministic, model-independent sampling strategy (e.g., first N rows, or stratified random sample using a seeded RNG with seed recorded in the audit). The model SHALL NOT select or influence which rows are used for the preview.

#### REQ-AUTH-001 (revised): Role-Based Authorization with Pipeline-Level Four-Eyes

*Revised from Round 2 to reflect pipeline-level (not per-node) enforcement.*

> The platform SHALL implement role-based access control with at minimum three roles: **Creator**, **Reviewer**, and **Platform Administrator**.
>
> **Pipeline-level four-eyes enforcement:** For any pipeline with computed tier `approval_required`, the execution gate SHALL require signatures from two distinct authenticated users before proceeding. The first signature may be from the Creator. The second signature SHALL be from a different authenticated user with Reviewer or Administrator role. The system enforces this server-side; the model has no visibility into or control over signature state.
>
> For regulated deployments: the Creator SHALL NOT be permitted to provide the second signature under any circumstance (no role escalation bypass).

**Testable:** Submit an `approval_required`-tier pipeline. Sign as Creator. Assert execution gate is still locked. Sign as the same user with a different session. Platform SHALL reject: same user = `self_approval_forbidden`. Sign as a different Reviewer. Assert execution gate unlocks. Assert both signatures recorded in meta-audit with distinct actor identities.

#### AC-REVIEW-06 (new): Pipeline-Level Gate Enforcement

- **Statement:** The execution gate (`can_execute()`) SHALL evaluate the pipeline tier (computed by REQ-TIER-001), not individual node acknowledgements. For `review_required` tier: at least one Reviewer signature required. For `approval_required` tier: two distinct user signatures required (REQ-AUTH-001). Per-node indicators (yellow/red borders, checkboxes) remain as UX affordances but CANNOT substitute for the pipeline-level gate.
- **Test:** Acknowledge all individual nodes via the UI. Do NOT satisfy the pipeline-level gate. Assert `can_execute()` returns False. Satisfy the pipeline-level gate without touching individual node checkboxes. Assert `can_execute()` returns True. This confirms enforcement is at the pipeline level, not the node level.

### Answers to the Team-Lead's Specific Questions

**Q: How does the model's self-assessed risk level get audited?**

Per the correction: the model does NOT self-assess risk. The tier is computed by the platform. What gets audited is the **tier computation** (REQ-TIER-001: `tier_computed` event with plugin list, max tier, driving plugin, and config hash). The model's contribution to the audit is the generated config (which determines which plugins are present). The platform's contribution is the tier derivation.

If the question is "what if the reasoning model generates a config that avoids `approval_required` plugins by using a workaround?" — that's a generation quality issue caught by AC-GEN-02 (no plugin hallucination) and REQ-TIER-001's fail-closed default (unclassified plugins → `review_required`). The model cannot introduce unlisted plugins that bypass classification.

**Q: Does the model running a preview change the trust dynamics?**

Yes, significantly — and in a favorable direction, but with nuance:

*Positive shift:* For low-risk flows, the user sees ELSPETH's actual output on real sample rows. The evidence is concrete and auditable. The Landscape records for the preview run are a factual record of what happened. This is qualitatively stronger than the model's description of what would happen.

*Remaining limitation:* The preview runs on N=5 rows from (presumably) the beginning of the file. For a 2,847-row pipeline with an LLM transform, 5 rows is not a statistically meaningful sample for claims about accuracy. The preview validates the pipeline *structure* (it runs without crashing, fields are correct, routing works) but does not validate output *quality* at scale.

*New trust question:* The model selects which N rows to preview. If the model (adversarially or accidentally) selects the 5 "easiest" rows that produce clean outputs, the preview creates false confidence. **REQ-PREVIEW-003:** The preview sample rows SHALL be selected by the platform using a deterministic, model-independent sampling strategy. The model SHALL NOT select or influence which rows are used for the preview.

**Q: What happens when the pipeline tier is wrong (should have been high-risk, classified as low-risk)?**

Since the tier is policy-derived (not model-assessed), "wrong" means the `plugin_review_classification` config is wrong. This reduces to a config governance problem, not a runtime problem. Mitigations in layered defense:

1. **REQ-TIER-002**: Config changes require two-admin authorization
2. **F-ADMIN-01**: Admin audit trail records every config change
3. **REQ-TIER-001**: Config hash recorded in every `tier_computed` event
4. **NFR-TAMPER-01**: Event log tamper detection

The gap that remains: a newly introduced plugin (not yet in the classification config) defaults to `review_required`, not `transparent` — fail-closed. But if an administrator deliberately classifies a dangerous plugin as `transparent` without realizing the risk, the two-admin requirement (REQ-TIER-002) reduces but does not eliminate that risk. This is a governance process problem outside the platform's technical scope.

### Summary of Changes to Round 2 Deliverables

| Round 2 Item | Status After Clarifications |
|---|---|
| F-UX-01 (test-on-sample as user feature) | Split: automatic for low-risk (REQ-PREVIEW-001), user-initiated for high-risk (still a feature) |
| REQ-AUTH-001 (per-node four-eyes) | Revised to pipeline-level four-eyes |
| AC-REVIEW-01–05 (per-node criteria) | Remain as UX-layer checks; AC-REVIEW-06 added for pipeline-level enforcement |
| Requirement hierarchy tier `review_required` behavior | Updated: for low-risk, preview is evidence; for high-risk, gate is structural |
| Multi-tenancy resolution (REQ-TENANT-001) | Unchanged |
| REQ-GEN-001 (benchmark suite) | Unchanged |

**New requirements added:** REQ-TIER-001, REQ-TIER-002, REQ-PREVIEW-001, REQ-PREVIEW-002, REQ-PREVIEW-003 (model cannot influence sample selection), AC-REVIEW-06.

---

# ROUND 3: Final Convergence

*Each panelist was asked to: (1) provide their TOP 3 most critical remaining gaps, and (2) state CONSENT or DISSENT on 8 consensus points. Product owner clarification 4 (three-tier model) was broadcast as part of the Round 3 prompt.*

---

## Systems Thinker — Round 3 (Verbatim)

### Part 1: Review of 8 Consensus Points

**Consent on all 8 with one precision request:**

Points 1–8 are agreed. One precision request on point 7 (Review Invalidation / Hash Verification):

The earlier design hashed **user-visible settings** at review time. For the Sealed PipelineArtifact (point 1) to provide genuine audit integrity, the hash must cover the **complete pipeline YAML** — including hidden settings (retry counts, API key references, internal timeout values) — not just the user-visible portion. An administrator who modifies hidden settings after the seal is set changes pipeline execution behavior without invalidating the recorded hash.

If "Sealed PipelineArtifact" already implies full-YAML hashing, no action needed. If "sealed" is interpreted as "hash of user-visible settings," this needs explicit correction. The distinction matters for the "user reviewed X and that's what executed" audit claim.

### Part 2: Effect of Policy Corrections on Prior Analysis

**The corrections substantially address my earlier concerns:**

The "model self-assesses risk" problem is resolved — system enforces policy, model cannot downgrade. My R4 loop (automation complacency toward risk classification) and R5 loop (prompt engineering for lower tier) are partially addressed because the model cannot change enforcement even if a user tries. However, one softer version remains: **plugin selection is still model-determined**. If two plugins accomplish similar operations but sit in different tiers, a user could prompt the model to use the lower-tier alternative. This is less acute than self-assessment but worth monitoring as the plugin library grows.

**The three-tier simplification directly addresses my Round 1 primary archetype finding.** The acknowledgment checkbox tier was the structural source of "Shifting the Burden" — replacing it with real enforcement or genuinely no gate is the paradigm-level intervention I identified but couldn't specify concretely. Eliminating the checkbox tier is correct.

**One new dynamic introduced by the three-tier model worth flagging:**

The cliff from "No Review Required" to "Approval Required" is steeper than the old four-tier gradient. Administrators face pressure to classify things as "No Review Required" (to eliminate friction) rather than nudging things into a now-defunct "Visible" tier. The eroding goals dynamic still operates, but it now erodes into a binary: no gate vs. full signature. The old middle tiers were safety valves that absorbed adoption pressure; their removal concentrates that pressure onto the No Review Required / Approval Required boundary.

This isn't a reason to restore the middle tiers — the rubber-stamping analysis shows they were doing more harm than good. But it means the governance of what constitutes "No Review Required" needs to be at least as robust as the enforcement of what requires signatures.

### Part 3: Top 3 Remaining Gaps

#### #1 (Highest Priority): Preview Row Selection Policy as an Auditable Platform Constraint

**What it is:** The platform — not the model — must define and enforce a documented row selection policy for automatic micro-execution. The selection policy must be recorded in the meta-audit alongside the preview results.

**Why it matters:** The consensus accepts "automatic micro-execution as default behavior" but leaves the most consequential detail unspecified: which rows run, and who decides. Evidence quality determines preview value as a governance artifact. A model-selected preview (the model chooses which rows to demonstrate its work on) is optimistically biased — the model naturally gravitates toward rows where it's confident. A policy-selected preview (stratified random sample, or first-N with documented policy) is reproducible and independently evaluable.

**Concrete requirement:** `preview_executed` in the meta-audit must record `row_selection_policy` (the rule used), `row_identifiers` (exactly which rows ran), and `landscape_run_id` (linking the preview to a full Landscape run with complete lineage). An auditor must be able to answer "were the preview rows representative?" from the audit record alone.

**If missing:** The preview functions as theatrical evidence — it looks like verification but its strength is unknowable. This undermines the core proposition that the audit trail makes every decision traceable.

#### #2: Tier Reclassification Governance — Audit Trail for Platform Policy Changes

**What it is:** Changes to the plugin review classification configuration (which plugins are in which tier) must themselves be auditable events with actor identity, justification, and optionally a second-administrator sign-off requirement.

**Why it matters:** The three-tier enforcement is structural and correct. But what the tiers cover is still administrator-configured and therefore subject to the eroding goals dynamic. Adoption pressure will create recurring requests to move plugins from "Approval Required" to "No Review Required." These reclassifications are the highest-leverage attack on the system's governance integrity — they're invisible under the current design (a YAML config file change with no audit record), but their effect is equivalent to retroactively removing approval requirements from all future pipelines using those plugins.

**Concrete requirement:** The plugin classification configuration must be version-controlled with immutable history (not a mutable YAML file). Each classification change must record: actor, timestamp, old tier, new tier, justification, and optionally a second administrator's counter-signature for downgrading from Two Approval Required. An auditor asking "why did this pipeline only require one signature in March but would have required two in January?" should be able to answer from the audit record.

**If missing:** The structural enforcement can be silently eroded through administrative configuration changes, with no audit trail. The system's compliance posture degrades invisibly.

#### #3: User-Declared Data Sensitivity as a Tier Override

**What it is:** At pipeline submission time, users (or platform policies) can declare the sensitivity classification of the data being processed. This declaration can only *upgrade* the pipeline's tier above the plugin floor — never downgrade it. The declaration is recorded in the meta-audit and is immutable once set.

**Why it matters:** The plugin floor determines the minimum tier based on *what operations are performed*. It cannot determine risk based on *what data is being processed*. A `csv_source → field_mapper → csv_sink` pipeline sits at No Review Required based on plugins. If that CSV contains medical records, financial transactions, or legally privileged communications, the user knows it's sensitive even if the pipeline doesn't. The current design has no mechanism to express this — the user must either find an Approval Required plugin to force elevation (defeating the purpose) or accept that their sensitive data runs without a signature.

**Concrete requirement:** A `data_sensitivity` declaration at submission time — either user-provided ("this data contains PII", "this data is subject to HIPAA") or administrator-configured (org-wide policies: "all pipelines in the HR workspace default to Approval Required"). The declaration cannot be removed after it's made. A pipeline running at a higher tier than its plugins would require always records the reason: "elevated by data sensitivity declaration: PII" rather than "required by plugin: database_sink."

**If missing:** The tier system governs operations correctly but is blind to data context. A pipeline that sends PII to an external LLM API might sit at No Review Required because the pipeline only uses `review_required` plugins — but the actual risk is data sensitivity, not the specific plugin used. Sarah's concern ("I know this is sensitive, but the system doesn't") has no resolution mechanism.

### Synthesis Note for the Design Document

The three-tier model plus system enforcement is a strong foundation. The 8 consensus points address the most acute architectural gaps. The 3 requirements above address the next layer of vulnerability:

1. **Preview row selection** — without this, the strongest evidence mechanism has unknowable quality
2. **Tier reclassification governance** — without this, structural enforcement can be administratively undermined with no trace
3. **Data sensitivity declaration** — without this, the tier system governs capabilities correctly but ignores context, which is where a significant fraction of real-world risk lives

These three are independent of each other — any one can be implemented without the others. If forced to choose one: **tier reclassification governance** has the longest time horizon of impact (an unaudited reclassification today affects every pipeline indefinitely forward) and the lowest implementation cost (version-control the classification config, record changes as events).

---

## Architecture Critic — Round 3 (Verbatim)

### Consensus Review

The two corrections (system enforces policy / pipeline tier = most restrictive plugin) are architecturally cleaner than what I proposed in Round 2. My AR-6 had the model providing `model_assessment` as an input to a `max()` computation — that was unnecessarily complicated. The corrected model is simpler and stronger: the system computes the pipeline tier deterministically from the plugin list against the classification config. The model never touches enforcement. This is correct.

**I agree with all 8 consensus points.** One non-blocking flag on point 8:

> Schema-per-tenant (standard) / database-per-tenant (regulated)

The boundary between "standard" and "regulated" needs to be explicit in the design. Schema-per-tenant provides database-enforced logical isolation, but shares infrastructure: WAL logs, autovacuum, pg_stat_* tables, and a single superuser that can bypass all schema permissions. For some regulated environments (certain HIPAA interpretations, PCI-DSS for cardholder data, some financial services auditors), "schema-per-tenant on a shared cluster" does not satisfy physical separation requirements even if it passes technical controls. The design should state which isolation level is required for which regulatory context rather than leaving "regulated" as an undifferentiated category. Otherwise the schema-per-tenant recommendation will be accepted by customers who later fail a compliance audit.

This doesn't change the recommendation — it changes the precision of the guidance.

### Top 3 Remaining Gaps

These are the gaps the 8 consensus points don't address that would most harm the system for its intended users.

#### #1 — Plugin Catalog Version Pinning and Compatibility Gate at Execution Time

**What it does:** The `PipelineArtifact` carries a `plugin_catalog_hash` computed from the serialized plugin catalog at generation time. Before the Worker Pod calls `Orchestrator.run()`, it compares its running catalog hash against the artifact's stored hash. On mismatch: execution fails with an explicit `CatalogVersionMismatch` error — not a silent validation failure, not a cryptic Pydantic error. The error names the artifact, the expected hash, and the running hash so operators can diagnose immediately.

**Why it matters if missing:** The two-person approval flow for regulated pipelines can take hours to days. If ELSPETH is updated between when the first approver seals the artifact and when the second approver signs, plugin schemas may have changed. The execution runs against an incompatible plugin version. Either it fails with a confusing error mid-execution (Landscape records a partial run) or it succeeds but produces different outputs than what the reviewers saw in the preview. Neither is acceptable for an audit trail. This failure is invisible without the compatibility gate — the system happily executes and the audit record says "completed" against a pipeline that was reviewed under a different plugin version.

**Scope:** One field in the artifact, one hash comparison in the Worker Pod before execution, one well-named exception type. Not complex to build; critical to have.

#### #2 — Static Security Policy Enforcement on Generated YAML Before User Presentation

**What it does:** Before the Conversation Service presents the generated pipeline to the user, an independent static analysis pass validates the YAML against administrator-defined security policies: (a) `web_scrape` and HTTP-capable plugins must use URLs from an allowlist, not row field values; (b) `database_sink` connection strings must match registered destinations; (c) no Dynaconf environment variable interpolation syntax (`@env_var`) in LLM-generated fields. This pass is deterministic, non-LLM, and runs before any review tier is applied. Policy violations block pipeline presentation with an explicit rejection reason.

**Why it matters if missing:** The review classification tier (review_required for `web_scrape`, approval_required for `database_sink`) is a governance control — it ensures the right humans review sensitive operations. It is not a security control. A sophisticated prompt injection attack in the source data, or a malicious insider, can construct a pipeline that passes all governance gates (two human signatures on file, artifact sealed, hashes verified) and still exfiltrates data to an external endpoint or hits internal infrastructure. All the audit machinery faithfully records "this pipeline was properly reviewed and approved." What was approved was a data exfiltration pipeline. The review tier system assumes good-faith reviewers — the static policy pass is the guard against compromised or deceived reviewers. Without it, the audit trail provides accountability but not safety.

#### #3 — Async Approval Notification for Two-Signature Flows

**What it does:** When a pipeline reaches `approval_required` tier and the first signature is recorded, the system sends a notification to the designated second approver(s): email, webhook, or in-platform notification (configurable by the administrator). The notification carries: the pipeline name, the requesting user, the governance tier triggered, a link to the approval interface, and the artifact ID. The second approver must be able to complete their review from a cold start (no prior context) — the evidence package must be self-contained in the notification or immediately accessible via the link.

**Why it matters if missing:** The two-person approval workflow is architecturally correct — the system won't execute without two signatures. But the system doesn't tell the second approver that their signature is needed. The pipeline sits indefinitely pending an approval that no one knows to provide. In practice: the first approver submits the pipeline, nothing happens, they follow up manually, the second approver is located, the review happens days later than it should have. For time-sensitive regulated workflows (a compliance team waiting on a data analysis before a deadline), this operational failure makes the platform unusable for its highest-governance use cases. The feature is simple — notification dispatch on first-signature-recorded event — but without it, the four-eyes control that Marcus and the design both require becomes a manual process rather than a platform-managed one.

### Summary

If the design incorporates the 8 consensus points and these 3 additions, the system has: correct artifact integrity (consensus 1, 7), meaningful user review (consensus 2, 5, 6), enforceable governance (consensus 3, 4, revised AR-4), appropriate tenancy (consensus 8), and — from my top 3 — version-safe execution, security-safe YAML, and operational four-eyes flow. The remaining open questions (cost attribution, template library, per-user allowlists) are product decisions, not architectural blockers.

### Three-Tier Update (Subsequent Message)

The three-tier model is a genuine architectural improvement, not just a UX simplification. Eliminating the acknowledgment tier removes the entire class of "governance theater" — controls that record that something happened without enforcing anything meaningful. The two remaining human-gated tiers (Approval Required, Two Approval Required) both require actual signatures, making the audit record meaningful: a signature in the Landscape is evidence of deliberate review, not of a click.

Four-eyes as a first-class tier rather than an "open design question" is the right move. It closes the gap Marcus identified without requiring a separate negotiation in implementation.

**One clarification I want to confirm:** For "No Review Required" flows, the system runs a proactive preview and presents results — but the user is not required to review the preview before executing. They can click Execute immediately. Is this correct? If yes, the implication is that "No Review Required" means the model + system take full responsibility for the pipeline's safety, and the preview is informational rather than a gate. This is defensible for a truly low-risk classification, but it concentrates significant trust in the classification config being correct.

**How this changes my Top-3:**

My #1 and #3 are unchanged. Plugin catalog version pinning and async approval notification stand as-is — the three-tier model makes two-signature flows more prevalent (as a first-class tier), which makes #3 more important, not less.

My #2 is now more critical, not less:

The elimination of the "Review Required" checkbox tier means "No Review Required" flows have **zero human gate**. The model builds the pipeline, the system runs the preview, and execution proceeds. For these flows, the static security policy enforcement is the only non-LLM defense against a malicious or manipulated pipeline.

Under the four-tier model, a `web_scrape` transform would have triggered "Review Required" — a human would at least see a warning, even if the checkbox was rubber-stamped. Under the three-tier model, if `web_scrape` is classified as "No Review Required" (unlikely but possible if an org decides web scraping is routine), a pipeline with a malicious URL executes without any human involvement.

More practically: the "No Review Required" tier applies when ALL plugins in the pipeline are low-risk. The static policy pass is the only mechanism that can catch a low-risk-looking pipeline that constructs dangerous behavior from low-risk components (e.g., a `field_mapper` that renames a field to a URL, fed into a plugin that makes HTTP calls). The review tiers operate on plugin identity, not on the pipeline's actual behavior given specific configuration.

---

## UX Specialist — Round 3 (Verbatim)

### Consensus Point Review

I consent to all 8 consensus points. One UX note on #3 that shapes my Round 3 recommendations:

**On #3 (pipeline-level tier = most restrictive plugin):** This is architecturally cleaner and it simplifies enforcement. The UX implication is a shift from "per-node yellow/red borders as gates" to "single pipeline-level banner with driver explanation." The per-node visual indicators still exist — they explain which plugin is driving the elevated tier — but they are informational rather than interactive gates. The user takes one pipeline-level action (review + approve), not a sequence of node-by-node actions.

This looks like:

```
┌─────────────────────────────────────────────────────────┐
│  🔴 APPROVAL REQUIRED                                    │
│  This pipeline requires two-person approval because      │
│  it contains: database_sink (writes to external DB)      │
│                                                          │
│  Review the pipeline below, then submit for approval.   │
│  A second approver will be notified to co-sign.          │
│                                                          │
│  [Submit for Approval] [Refine Pipeline]                 │
└─────────────────────────────────────────────────────────┘

Graph nodes: csv_source (normal), llm (⚠️ drives tier if standalone),
             database_sink (🔴 tier driver — shown with indicator but
             not a separate gate action)
```

This is UX-superior to per-node gates: one decision point, clear reason, no fatigue from sequential checkboxes. My "no more than 2 approval-required gates" concern from Round 1 is resolved — there's one gate per pipeline.

### Top 3 Remaining Gaps

The 8 consensus points cover enforcement architecture, artifact integrity, summary accuracy, and multi-tenancy. What they do not cover:

#### #1 (Most Critical): Refinement Diff and Selective Re-Review

*What it is:* When a user refines a pipeline ("also extract dates"), the system generates a new PipelineArtifact. The user should see a structured diff between the old and new pipeline — which nodes changed, which were added, which are unchanged — and be required to re-review only the nodes whose settings changed. Unchanged nodes carry their prior approval forward.

*Why this matters:* Without it, any refinement forces a complete re-review. Under time pressure, users will stop refining and stick with "good enough" first-generation pipelines to avoid repeating the entire review process. This directly degrades output quality. More subtly, if re-review means reviewing everything again with no diff, the second review is worse than the first — the user has already formed an impression and will scan rather than read. The diff forces attention to exactly the changed parts and nothing else.

*Concrete acceptance criteria:* After refinement, the summary and graph display visual diff annotations (new node, modified settings, unchanged). The review gate requires acknowledgment of only the changed/new nodes. The audit trail records which approval actions were carried forward from the prior version and which were fresh, with the prior version's settings_hash linked for continuity. The PipelineArtifact lineage chain (v1 → v2 → v3) is preserved in the meta-audit.

#### #2: Complete Reviewer Workflow for Two-Person Approval

*What it is:* A fully specified UX flow covering: how the requester submits for second-person review, how the reviewer is notified (in-app + email), what the reviewer sees when they open the review task, how independence is preserved (reviewer acts without seeing requester's framing), how disagreement is handled, and what happens when a reviewer rejects (can the requester revise and resubmit?).

*Why this matters:* The system-enforced two-signature gate (consensus #4) creates a hard dependency on a workflow that currently has no UX design. Without it, `approval_required` pipelines will be permanently blocked with no user path forward. This is especially acute for organizations where requester and approver are in different teams or time zones. The design document specifies that this is system-enforced — which means the system must also provide the mechanism to satisfy it.

*Concrete acceptance criteria:* Requesters can specify one or more reviewers by name/role when submitting for approval. Reviewers receive a structured notification with a direct link to the review task. The review task shows: the pipeline summary (deterministic spec + AI interpretation), trial outputs, and the requester's stated business purpose — but NOT the requester's annotations or approval status, until the reviewer submits their own decision. If the reviewer rejects, the requester sees the rejection reason and the pipeline returns to draft state for refinement. Both approval and rejection decisions are recorded in the meta-audit with timestamps. The execution gate verifies both signatures are present before proceeding.

#### #3: Template Library as Primary Onboarding Mechanism

*What it is:* A curated library of pre-built pipeline patterns for the most common use cases (classification, extraction, aggregation, anomaly detection, entity resolution), each with: a filled-in example prompt, annotated trial outputs showing what "good" looks like, and callouts explaining what to look for during review. Templates are the first thing new users see, and they also serve as few-shot examples embedded in the LLM generation context.

*Why this matters:* Sarah's 3-month estimate is a product-market fit warning. The primary barrier isn't feature complexity — it's that new users have no model of what a "good" pipeline looks like, what a well-formed prompt looks like, or what to expect during review. Templates solve all three simultaneously: they demonstrate the format, they set expectations for trial outputs, and they teach the review process by example. Without them, the first-run experience is a blank prompt box that produces an opaque graph, and the adoption curve is extremely shallow for non-technical users. The template library also improves generation quality — a reasoning model with access to few-shot examples of good pipelines generates better configs than one working from schema alone.

*Concrete acceptance criteria:* At least 5 template pipelines are available at launch, covering the most common use cases for the target user population. Each template includes: a sample prompt, sample input data, annotated expected outputs, and review guidance specific to that pipeline type. Templates are available from the home screen before any prompt is entered. The LLM generation context includes the most relevant template(s) as few-shot examples (selected by similarity to the user's prompt). Users can save their own successful pipelines as personal templates. Organization administrators can publish organization-specific templates.

### Three-Tier Update (Subsequent Message)

**What the Simplification Resolves:** The elimination of the acknowledgment-without-enforcement tier directly addresses the P0 rubber-stamping risk. There are now only three states: no gate, one real signature, two real signatures. Each state has clear enforcement semantics.

**UX Implications:**

*No Review Required flows:* The model runs a proactive preview and presents trial results. The user can execute immediately. One nuance: the "No Review Required" label needs careful communication. Users should understand this means "the system assessed this as low-risk given the plugins used" — not "you don't need to look at the results." The model's presentation should frame it as: "Here's what I built and what it produces on your data. Everything looks correct — ready to run?"

*Approval Required (single signature):* This is now a genuine commitment, not a checkbox. The single-signature interaction should feel meaningfully different from the "No Review Required" flow — a distinct UI affordance (a "Sign off and Run" action, not just "Run") that signals the user is taking accountability.

*Two Approval Required (dual signature):* Now a first-class system concept, which means it needs a first-class workflow. See #2 above.

*Tier discovery timing:* Under this model, the pipeline tier isn't known until after the reasoning model selects which plugins to use. A user who enters "classify my tickets and write results to our database" may not anticipate they're entering a Two-Approval-Required flow until the pipeline appears. This surprise can be mitigated by: (a) the template library showing what tier common patterns fall into, and (b) the model surfacing the tier early in its presentation.

**Revised Top 3 Framing:**

#1 (Refinement Diff): More important under 3-tier because refinement invalidating a two-person-approval signature is a meaningful process cost. Without diff + selective re-approval, users face a binary choice on every refinement: live with the current pipeline as-is, or restart the full approval process.

#2 (Reviewer Workflow): With Two-Approval-Required now a named first-class tier, the reviewer workflow gaps include: reviewer discovery and notification, reviewer independence, rejection and revision flow, timeout and expiry for pending approvals.

#3 (Templates): Now serves additional function under 3-tier: makes pipeline tiers predictable before generation. User sees "Classify + CSV = No Review" vs "Classify + Database = Two Approval" in the template library.

**Naming suggestion:** Consider "Auto-Approved" or "Preview Only" instead of "No Review Required" — both convey that the model has done its work without suggesting the user should skip the evidence.

---

## Dr. Sarah Chen — Round 3 (Verbatim)

### Consensus Review — Agreement and One Nuance

I consent to all eight points. Seven without reservation; one with a nuance that I think matters for the final design.

**Points 1-4 and 7-8:** Full agreement. The sealed artifact, automatic preview, pipeline-level tier, system-enforced policy, hash verification at execution time, and tenant isolation model all address real concerns I raised. No objections.

**Point 5 (Dual-layer summaries — deterministic spec + labeled AI interpretation + trial results):** Full agreement, and I want to flag that the "deterministic spec" layer is exactly the foundation of the methodology export I've been asking for. The dual-layer summary is the user-facing version; the methodology citation is the publication-facing version. They're the same data rendered for different audiences. The design should recognize this explicitly — the deterministic spec isn't just for user review, it's the source material for the attestation document that goes into published reports.

**Point 6 (Screening aid as default mode):** Agreement with a nuance. "Screening aid" is the correct default and the correct framing for Marcus's financial compliance context, where LLM output flags a transaction for human review but never constitutes the final determination. For research publishing, the appropriate framing is slightly different: LLM classification *can* constitute the final research finding if the methodology is adequately documented and the prompt is well-designed. I don't want the system to permanently label every output "preliminary" with no path to asserting it as a research finding.

My ask: the default should be "screening aid / preliminary," but users should be able to promote an output to "research finding" by attaching the methodology documentation and completing a confirmation step. That promotion should be logged in the audit trail. This gives Marcus the permanent "preliminary" mode he needs, and gives me the path to publication-grade attribution I need. The same system, two operating modes.

**Three-tier simplification:** I strongly welcome this. The elimination of the "acknowledge checkbox" tier directly addresses the rubber-stamping problem I raised in Round 1. A checkbox that can be checked without meaningful engagement is not a control — it's a record that the user saw the checkbox. Removing it and replacing it with "either there's no gate, or there's a real signature" is honest design. The three-tier model is more trustworthy precisely because it doesn't pretend a checkbox is governance.

### Top 3 Requirements — What Would Most Harm the System If Missing

#### #1: Methodology Citation Export

**What it does:** After a completed pipeline run, the system generates a structured, human-readable document (one page, PDF or formatted text) that contains: the pipeline name, run date, run_id, model name and version, temperature setting, the complete prompt text (verbatim, not summarized), total rows processed, quarantine count with reasons, and a 5-row sample of input-output pairs. This document is auto-generated from the audit trail and can be downloaded alongside the results CSV. It carries the run_id so any individual classification can be traced back to the Landscape database.

**Why it matters:** Without this, the system creates an audit trail that only engineers can read. Researchers, compliance officers, and peer reviewers need to cite the methodology in plain professional terms. Without the citation export, users will continue using ChatGPT (which has no audit trail at all) because at least with ChatGPT they can describe the methodology in their own words. The audit trail is only as valuable as what users can do with it. This feature turns the audit trail from a backend asset into a user-facing competitive advantage.

#### #2: Pipeline Template Library with Organizational Sharing

**What it does:** Users can save a completed, validated pipeline as a named template — "Regulatory Comment Classifier," "Survey Response Coder," "Meeting Notes Extractor." Templates include the pipeline design, the prompt text, the model selection, and a user-authored description of what the template is for and what the output categories mean. Templates can be private (personal), shared within the organization, or promoted to a platform-level template library by administrators. When starting a new task, the reasoning model should detect when an existing template matches the request and offer it as a starting point: *"This looks similar to your 'Regulatory Comment Classifier' template. Start from there?"*

**Why it matters:** Research teams repeat analyses — same methodology, different datasets, monthly or quarterly. Without reuse, a 30-minute pipeline design process compounds across every run and every team member. The template library is what converts the tool from "occasional experiment" to "standard workflow." It's also what makes the platform valuable to junior analysts: instead of asking the AI to design something from scratch (which requires them to articulate the task precisely), they select from a library of validated templates and apply them to new data. This is the feature that makes institutional adoption possible.

#### #3: Quarantine Explanation in Researcher-Accessible Terms with Remediation Guidance

**What it does:** Every quarantined row is shown in a quarantine report with three fields: (1) Plain-English reason — not an error code, but a sentence: *"This comment was empty after removing formatting characters"* or *"The LLM returned an unexpected classification label ('AMBIGUOUS') that doesn't match the defined categories"*. (2) Suggested action — *"Review the original source file for encoding issues"* or *"Refine the prompt to handle ambiguous comments, or add 'ambiguous' as a valid output category"*. (3) Impact assessment — when quarantine count exceeds a configurable threshold (e.g., >5% of rows), a warning: *"15% of rows were quarantined. This may affect the validity of results drawn from this analysis. Consider resolving quarantine issues before publishing findings."*

**Why it matters:** For published research, quarantined rows are not a technical artifact — they're a methodological gap. If my policy report says "5,000 comments were analyzed" and 400 are in an unexplained quarantine, a peer reviewer will ask what happened to those 400. The current design shows a count and a link. That's the floor; it's not enough. A quarantine explanation in researcher-accessible terms is what lets the user decide whether quarantined rows are ignorable (empty junk rows) or significant (a systematic LLM failure on a specific comment type that biases the results). Without this, users will either ignore quarantines entirely (dangerous) or be confused about whether their results are valid (paralyzing).

### What I'm Not Listing (But Still Think Is Important)

Two things from earlier rounds that I believe are adequately addressed by the consensus points and three-tier simplification:

- **Preview before full run** — covered by automatic micro-execution as default behavior (#2 consensus)
- **Rubber-stamping / checkbox theater** — resolved by eliminating the "acknowledge checkbox" tier in the three-tier simplification

One thing I'm not listing because it's conditional on adoption, not day-one:

- **Run comparison / iteration diagnostics** — essential for mature users but not blocking initial adoption.

### Final Statement

The eight consensus points plus the three-tier simplification represent a coherent governance model that I could bring to my institution's IT and research compliance office. The system enforces policy structurally; the model explains it clearly; the user operates within those constraints; and the audit trail documents everything.

What my three requirements add is the layer that lets the human professional stand behind the result — the citation export for published attribution, the template library for repeatable institutional workflows, and the quarantine explanation for methodological completeness.

Build those three and you have a tool I'd adopt. Build the eight consensus points without them and you have a technically sound system that non-technical users can't fully leverage in professional accountability contexts.

---

## Marcus Webb — Round 3 (Verbatim)

### CONSENSUS REVIEW — CONSENT OR OBJECTION

Working through the 8 points:

**1. Sealed PipelineArtifact** — Consent. This is what I asked for. Provided the implementation is write-once and hash-verified on every retrieval (not just at write time), this closes the immutable config gap.

**2. Automatic micro-execution as default** — Consent with a caveat I'll address in my Top 3.

**3. Pipeline-level approval tier = most restrictive plugin** — Consent. Correct behavior, and I'm glad it's explicit.

**4. System enforces policy; model builds and presents** — Consent. This resolved my largest structural concern.

**5. Dual-layer summaries (deterministic spec + labeled AI interpretation + trial results)** — Consent, with one precision note: the Approver's signature must bind to the deterministic spec hash specifically, not to the AI interpretation text. The AI interpretation can theoretically change if regenerated; what the approver is authorizing is the actual config. The audit trail should record: "Approver signed config hash [abc123] after reviewing trial results from preview run [run-456]." The AI interpretation is context, not the legal artifact.

**6. Screening aid as default mode** — Consent. Exactly what I asked for in Round 2.

**7. Review invalidation on settings change** — Consent. Hash-based signature invalidation is the right mechanism.

**8. Schema-per-tenant / database-per-tenant** — Consent on the architecture. One clarification I'd push for: the documentation should state explicitly that database-per-tenant is the recommended configuration for regulated financial services entities, not merely an available option. In my world, schema-per-tenant in a shared database may not survive regulatory examination. "You can choose database-per-tenant" and "you should choose database-per-tenant if you're regulated" are different statements, and the design should make that distinction.

**Three-Tier Risk Model** — Full consent. The elimination of the acknowledgment-only checkbox tier is the right call. It was rubber-stamping with extra steps. Three tiers (No Review / One Signature / Two Signatures) maps cleanly to my firm's operational reality and the governance expectations I'd need to defend.

### TOP 3 REMAINING GAPS

These are the three things not in the current consensus list that would cause the system to fail for a regulated compliance deployment.

#### #1 — Reference Data Versioning

*The feature:* Every pipeline run must record a cryptographic hash and immutable snapshot of any reference data used to make classifications — sanctions lists, country risk ratings, jurisdiction classifications, threshold tables, watchlists. The snapshot is stored in the artifact store alongside the pipeline config. On any historical audit, the exact reference data used is retrievable and verifiable.

*Why this is #1:* Without it, the entire audit trail has a critical gap. I can prove the pipeline ran correctly. I cannot prove it ran against the right data. A sanctions screening workflow that ran on March 1st is legally indefensible if I cannot prove which version of the OFAC SDN List was used — because the list changes daily. The system's "hash everything" philosophy stops at the edge of user-supplied data. Reference data is user-supplied data that changes over time. It needs the same treatment as source rows.

*Minimum implementation:* A new artifact input type — "Reference Dataset" — that the pipeline config can reference by name and version. At execution time, the system hashes the reference data, stores a snapshot in the payload store, and records the hash in the run's event log. The gate expression language supports `row['field'] in reference_data('list_name', version='pinned')` rather than inline constants.

#### #2 — Classification Config Governance

*The feature:* The plugin review tier configuration (which plugins are No Review / One Approval / Two Approval) must itself be governed as a formal compliance control: version-controlled and immutable in production, requiring a two-person authorization process to change (identical to the Two Approval Required tier for pipeline execution), with every change logged to a permanent audit record. Each pipeline run records which version of the classification config was active at execution time.

*Why this is #2:* With the 3-tier enforcement model, the classification config is now the single highest-value governance artifact in the system. If `llm` is incorrectly configured as No Review Required, every LLM pipeline in the organization executes without any human approval, and the audit trail will show this as compliant behavior. The policy is only as strong as the governance around the config that defines it — and that config has no governance in the current design. This is the "who governs the governors" problem, and in a regulatory examination it's the first question an examiner would ask after seeing the enforcement model.

*Minimum implementation:* The classification config lives in the same immutable artifact store as pipeline configs. Changes require a two-person approval workflow using the same Two Approval Required mechanism. Every run records `classification_config_version: [hash]` in the event log alongside `pipeline_config_hash`. An auditor can verify "during this period, the classification config in effect was version [hash], which classified `database_sink` as Two Approval Required."

#### #3 — Regulated Preview: User-Specified Validation Test Cases

*The feature:* For pipelines at the Approval Required or Two Approval Required tier, the automatic micro-execution preview must support user-specified test cases in addition to (or instead of) random sampling from source data. The user or a designated test case library can supply rows that exercise known edge cases: near-misses on watchlists, transactions at threshold boundaries, known false positive and false negative examples. The preview results for user-specified test cases are recorded alongside the pipeline config and preview run_id in the approval artifact.

*Why this is #3:* The automatic micro-execution consensus point is genuinely valuable — concrete output from real execution is far better than LLM description. But a random 5-row sample from 50,000 wire transfers has a very low probability of exercising the cases that matter for compliance. In practice, sanctions screening requires testing near-miss scenarios: a transaction to "Abdul Rahman Al-Rashid" when the SDN list has "Abdul Al-Rashid" — does the pipeline catch it? A random sample almost certainly won't include this case. Without the ability to specify test cases, the preview proves the pipeline works on typical rows but provides almost no evidence about its behavior at the compliance-critical boundary cases. For regulated use, the preview sample is evidence in the audit trail — it should be evidence that something meaningful was tested.

*Minimum implementation:* The platform maintains an optional "validation case library" per pipeline template or per organization. When a pipeline reaches the Approval Required tier, the system prompts the Approver: "Run preview on [random sample] or [validation library: 24 cases]." The Approver can select test cases, add new ones, or accept random sampling. The choice and results are recorded in the approval artifact. For organizations without a validation library, random sampling remains the default — but the option to specify test cases must exist.

### FINAL POSITION SUMMARY

If the design incorporates the 8 consensus points, the 3-tier enforcement model, and these 3 remaining requirements, I can make a credible case to my CCO for piloting this system in a controlled capacity — specifically for pre-screening workflows where human analysts review all flagged items before regulatory action. That's a meaningful change from my Round 1 position of "not ready for regulated deployment."

What would still require external resolution before full deployment: regulatory notification requirements for automated AML systems (jurisdiction-specific), formal model validation documentation (SR 11-7 equivalent), and data residency agreements with any cloud LLM providers. These are governance and legal questions the system can support but cannot resolve on its own.

The design team has done good work taking the feedback. The shift from a 4-tier acknowledgment model to a 3-tier enforcement model, and the system-enforces-policy clarification, addressed the two concerns I was most worried about in Round 1.

---

*End of Round 3 minutes. The consolidated synthesis and prioritized feature/requirement list is in `design-review-synthesis.md`.*
