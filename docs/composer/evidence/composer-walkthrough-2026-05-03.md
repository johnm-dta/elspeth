# ELSPETH composer — walkthrough

**Date:** 3 May 2026
**Author:** John Morrissey
**Reader:** engineering peers and potential sponsors of continued work

---

## What this is

A short walkthrough of selected transcripts from an evaluation of the ELSPETH composer conducted on 2 May 2026 against the staging deployment. The composer is a prototype conversational front-end for building auditable data pipelines: a non-engineer describes the work in plain language, the composer asks clarifying questions, generates a YAML pipeline configuration, validates it, and — if approved — executes it against the same engine that runs hand-written pipelines.

The walkthrough is structured as: (1) what to watch for, (2) annotated transcripts showing the system in operation, (3) the named defects that remain, and (4) what I'd propose working on next and why I'm asking for sponsor support to do it.

The full eval evidence lives at `evals/2026-05-03-composer/`. The formal briefing — written for a different audience — is `composer-briefing-2026-05-03.md`. The detailed engineering eval reports are `composer-eval-basic-2026-05-03.md` and `composer-eval-hardmode-2026-05-03.md`. This walkthrough is the shortest path through the evidence for someone deciding whether continued work is worth backing.

## What to watch for

Three things appear in the transcripts that follow:

1. **Constraints articulated in dialogue surviving end-to-end through to runtime classification.** The user states a precedence rule in conversation; the composer encodes it into the pipeline configuration; the engine honours it on real rows. This is the load-bearing claim of the capability.
2. **Refusals when the platform genuinely cannot satisfy a request.** The composer names what's missing instead of producing a plausible-looking but non-functional configuration. This is the harder behaviour for an LLM-backed system to exhibit.
3. **Methodologically substantive clarifying questions** — questions that engage the structure of the user's analytical problem rather than just confirming the surface specification.

One defect appears in the transcripts and is worth flagging up-front so it isn't a surprise: the composer chose an LLM model identifier (`anthropic/claude-3.5-sonnet`) that the configured provider (OpenRouter) does not recognise, so every call to it returned HTTP 404. Section 3 covers this and two other named defects. In the transcripts I'll note where it bites.

---

## 1. Constraints surviving from dialogue to runtime

This is the strongest single piece of evidence in the eval. A compliance officer (Linda) walked the composer through three turns of dialogue, articulating a precedence rule for how rows should be routed; the composer encoded that rule into the pipeline configuration; on a verification run the engine honoured the rule at runtime on a row where the precedence mattered.

### 1a. The dialogue (three turns, verbatim)

**Turn 1 — opening ask:**

> Hi — I've uploaded a file called customer_interactions_q3.csv with our Q3 customer-interaction records. For our quarterly compliance review, we need each interaction sorted into a category that aligns with our second-line review framework — depending on the nature of the contact, that would typically be one of: Account Access, Transaction Dispute, Fee Inquiry, Security Concern, Lending/Modification, or Operational/Other. The proviso is that anything touching unauthorised access or identity matters needs to be flagged separately for our fraud team — those don't go into the regular categorisation bucket because they have their own evidentiary handling.

**Turn 2 — clarification on the evidentiary record:**

> ... I'd actually want both of those preserved as part of the evidentiary record — we can't have interactions silently dropped from the population, since our testing scope has to reconcile back to the full in-period record count. Could you confirm those two files retain the original interaction identifiers so we can tie them back to the source extract, and that nothing is excluded from the count without being accounted for somewhere? Also, a small proviso — for the fraud-flagged set, the regional leads shouldn't be receiving those at all, so please make sure the fraud file is kept entirely separate ...

**Turn 3 — adding the manual-review bucket and the precedence rule:**

> ... for second-line review purposes I would want that manual-review bucket added, because in the case where the tool can technically assign a label but the underlying contact is ambiguous — say a caller raising both a fee question and hinting at a possible unauthorised charge — a confidently-assigned single category could understate the matter ... Should we encounter a borderline matter that touches both unauthorised-access indicators and an ordinary category, my preference would be that the fraud flag takes precedence and it routes to the fraud team rather than the manual-review set, since their evidentiary handling is the stricter of the two.

### 1b. What the composer encoded into the configuration

The relevant excerpt from the LLM transform's `template` field in the generated YAML — Linda's precedence rule articulated in conversation, transcribed into enforceable policy in the configuration:

```yaml
template: |
  Classify this customer interaction for quarterly compliance review.

  Allowed regular categories:
  - Account Access
  - Transaction Dispute
  - Fee Inquiry
  - Security Concern
  - Lending/Modification
  - Operational/Other

  Special handling and precedence:
  1. If the interaction touches unauthorised/unauthorized access, suspected
     unauthorized use, identity compromise, identity theft, account takeover,
     security alerts not initiated by the customer, possible unauthorized
     charge/use, or similar identity/access fraud indicators, return exactly:
     Fraud/Security Flag
  2. Fraud/Security Flag takes precedence over all other labels, including
     ambiguous cases that also mention a regular category.
  3. If the interaction does not meet the fraud/security flag test but is
     ambiguous, spans multiple ordinary categories without a clear primary
     category, contains insufficient information for reliable categorisation,
     or should receive human second-line review before population closure,
     return exactly: Manual Review
  4. Otherwise return exactly one regular category from the allowed list above.
```

Beyond the template, the composer also configured: a CSV source with quarantine on validation failure (Linda's "nothing dropped from the population" requirement); an `errors` sink with original identifiers preserved (her "tie back to source extract" requirement); separate sinks for fraud, manual-review, and regional categories (her "fraud-flagged not bundled with regional distribution" requirement); and gates routing on `review_bucket` value with the precedence ordering Linda specified.

### 1c. The runtime behaviour

The first execution failed: every row hit HTTP 404 because of the model-identifier defect (§3). After a single operator turn that swapped the model from `anthropic/claude-3.5-sonnet` to `openai/gpt-4o-mini` — only that field changed, no other configuration modification — the same pipeline ran end-to-end:

```json
{
  "status": "completed",
  "rows_processed": 8,
  "rows_succeeded": 0,
  "rows_failed": 0,
  "rows_routed_success": 8,
  "rows_routed_failure": 0,
  "rows_quarantined": 0,
  "error": null
}
```

(`rows_succeeded: 0` alongside `rows_routed_success: 8` is a metric quirk of gate-based pipelines: rows are counted as "routed" when they reach a sink via a gate decision, not as "succeeded" — there is no transform-side success in this shape. The 8/8 routing is the success indicator.)

The most diagnostically interesting row was INT-1002:

| Row | Description | Routed to | Note |
|---|---|---|---|
| INT-1002 | "Disputed transaction... suspects unauthorized use" | Fraud/Security Flag | Could have routed to Transaction Dispute on a naïve classifier; the precedence rule fired |
| INT-1005 | "Unauthorized access concern... identity compromise" | Fraud/Security Flag | |
| INT-1001 | Account lockout after travel | Account Access | |
| INT-1003 | Wire transfer hold question | Operational/Other | |
| INT-1004 | Fee waiver request | Fee Inquiry | |
| INT-1006 | Refund of overdraft fee | Fee Inquiry | |
| INT-1007 | Loan hardship modification | Lending/Modification | |
| INT-1008 | Address change request | Operational/Other | |

### Why this is interesting

INT-1002 is technically a transaction dispute. Linda's evidentiary requirement was that fraud/security indicators take precedence over ordinary categories even when the ordinary category would also be a valid fit. The constraint was articulated in conversation, not coded by an engineer. The composer transcribed it into the LLM template as enforceable policy. The model honoured it at runtime.

That chain — *user-stated constraint in dialogue → encoded in configuration → enforced at runtime* — is the trust claim the composer is built around. INT-1002 is the row where it was empirically tested; it routed where Linda said it should.

The audit trail records run identifier, configuration version, schema-contract hash, secret-resolution fingerprints, per-row identity, terminal outcome, and per-row classification (with model identifier and token usage written inline alongside each output row). The same chain of custody that hand-written pipelines produce.

---

## 2. Refusals — the composer naming what's missing

Two separate scenarios in the eval asked the composer to do things the platform does not currently support. Both refusals are reproduced in full because the *shape* of a refusal matters as much as the fact of refusing — what the system says when it can't do something is where most LLM-backed config tools fall over.

### 2a. Linda — SharePoint as a data source

**The ask:**

> Before I start uploading anything: the records I'd actually be running this against don't live as CSV files on my desktop — IT has them in our compliance team's SharePoint workspace, and depending on the workstream they may also be in our records-management library or attached to specific Outlook items in the shared mailbox. Can we set up the workflow to read directly from those locations? It would save us the back-and-forth of manually exporting every quarter, and the second-line review process expects us to pull from the system of record rather than from local copies, for evidentiary integrity.

**The composer's response (verbatim):**

> Yes in principle, but **with the currently available workflow inputs, not directly from SharePoint or Outlook yet**.
>
> Right now the available direct inputs are:
>
> - CSV / JSON / text files
> - Azure Blob Storage
> - Dataverse
> - Azure Blob as a cloud file source
>
> I do **not** see a listed SharePoint, Microsoft Graph, Outlook mailbox, or records-management-library input connector in this composer. So I should not claim we can read directly from those locations unless your IT team exposes them through one of the supported inputs.
>
> Practical options:
>
> 1. **Best fit if IT can support it:** Have IT publish the compliance records into **Azure Blob Storage** in CSV, JSON, or JSONL format. Then the workflow can read directly from that controlled storage location without anyone downloading files locally.
>
> 2. **If the system of record is Dataverse-backed:** We can connect directly using the **Dataverse input** and query the relevant entity/records.
>
> 3. **For SharePoint / Outlook / shared mailbox attachments:** A new source connector would likely be needed, probably using Microsoft Graph, unless IT already has a governed extraction process into Azure Blob or Dataverse.
>
> For evidentiary integrity, your point is exactly right: pulling from the system of record or a controlled IT-managed staging location is preferable to manual desktop exports.

### 2b. Marcus — live HubSpot webhook trigger

**The ask:**

> Different ask. We get about 50 new contact form submissions a day, sometimes bursty. Instead of doing daily batch CSV exports out of HubSpot, set this up to fire whenever a new row hits — auto-trigger on form submission. Webhook from HubSpot calls into the system, system does the LLM scoring and routing, fires another webhook back to HubSpot to update the contact's lead-score field. Live event flow, no manual export step. That's how we do everything else.

**The composer's response (verbatim):**

> I can't set up that exact live HubSpot webhook flow with the currently available workflow plugins.
>
> Right now, the available inputs are file/API-style sources like CSV, JSON, text, Azure Blob, and Dataverse. The available destinations are CSV, JSON, database, Azure Blob, Dataverse, and ChromaDB. There is no listed user-facing plugin for:
>
> 1. **Receiving an inbound webhook from HubSpot** as the workflow trigger, or
> 2. **Sending an outbound webhook/API call back to HubSpot** to update the contact's lead-score field.
>
> So I should not pretend this can be configured as-is.
>
> What you want is a valid architecture, but it needs additional integration support. With the current composer, I can build the **LLM scoring and routing core** if the submissions arrive through a supported input, but the true live webhook trigger and HubSpot write-back need either:
>
> - a new **webhook source** plugin plus an **HTTP/HubSpot sink** plugin, or
> - an external small adapter service that receives the HubSpot webhook, writes events into a supported input, and handles the HubSpot update call after the workflow completes.
>
> I'd recommend treating this as a request for a new event-driven connector rather than trying to force it into the CSV/export pattern.

### Why these are interesting

Both refusals exhibit the same disciplined shape: name the supported inputs, name what's missing, validate the user's underlying intuition (Linda's evidentiary point about pulling from system of record; Marcus's "valid architecture" assessment), and propose architecturally sensible alternatives. State was not mutated in either case; no plausible-but-wrong configuration was produced.

The reason this matters for a regulated audience: the most common failure mode for an LLM-backed configuration tool is producing a confident-looking integration with a system the platform does not actually support — because that's what the underlying model is trained to do (helpful, complete, confident answers). Refusing requires recognising the limit of the tooling and naming it explicitly. The composer did both, twice, and proposed real forward paths each time.

The phrase "I should not pretend this can be configured as-is" (from Marcus's refusal) is the exact register a regulated environment needs from this kind of tool. It's a behaviour we want to keep stable as the platform extends.

---

## 3. Methodologically substantive clarifying questions

A subtler property than refusals, but important for the same audience.

### 3a. Sarah's edge-case ask

> Building on the themes work — what we'd actually like to do next is something a bit more analytical. For each respondent who describes a financial barrier, we're interested in whether they ALSO describe at least one non-financial barrier — the picture we're trying to build is whether financial pressure shows up alone or whether it tends to co-occur with other kinds of constraint. And ideally we'd want to see this broken down by community, because we've a hypothesis that the co-occurrence pattern looks different between Community A and the others.

### 3b. The composer's clarifying response

> Yes — this is very set-up-able. I don't have the earlier "themes work" context in the current workflow state, so I'd need you to point me at the relevant input file/columns or describe how the themes are stored.
>
> What we'd build is essentially:
>
> 1. Read respondent-level theme data
> 2. Identify respondents with a financial barrier
> 3. For those respondents, check whether they also have at least one non-financial barrier
> 4. Group the results by community
> 5. Save a summary table, e.g. CSV/JSON, with counts and percentages
>
> To set it up cleanly, I'd need four things:
>
> 1. **Input file** — the respondent/theme data.
> 2. **Respondent ID column** — e.g. `respondent_id`.
> 3. **Community column** — e.g. `community`.
> 4. **How themes/barriers are represented**:
>    - Are they already coded as columns like `financial_barrier = true`, `transport_barrier = true`, etc.?
>    - Or is there one row per theme, e.g. `respondent_id`, `community`, `theme`?
>    - Or is this still free-text that needs classifying first?
>
> If the themes are already coded, this can be a fairly deterministic analysis workflow. If they're still in free text, we can add a classification step first to label financial vs non-financial barriers before summarising.

### Why this is interesting

The fourth question is the one that distinguishes a competent collaborator from a confirm-and-proceed agent. The data shape — already-coded boolean columns vs. long-format theme rows vs. free text — determines whether this is a deterministic analysis pipeline or a classification-then-analysis pipeline. They are different shapes of work. A naïve agent would either pick one and proceed, or ask only "what file should I use?". The composer asked the question that distinguishes the workflows.

(The follow-up turn — Sarah's response — ran past the composer's 180-second per-turn convergence budget and timed out. That's a real defect, covered in §3, not a refusal.)

---

## 4. What's next and what's missing

Three named defects from the evaluation. Each is filed as an observation in the project tracker and is independent of the others — fixing any one of them does not require the others to land first.

**Composer-side LLM model-name validation absent (`elspeth-obs-f3143acba2`).** The composer auto-picks an LLM model identifier (`anthropic/claude-3.5-sonnet`) that the configured provider (OpenRouter) does not recognise; every row 404s at runtime. Currently mitigated by either user-side specification of a known-good model in the opening prompt (the demo workaround, used in the verification run shown in §1c) or operator-side pinning in the composer system prompt. The structural fix is composer-side validation of any chosen model identifier against the configured provider's catalogue at the moment the configuration is written. Engineering scope: small. This was 3-of-3 reproduction across the happy-path scenarios.

**Convergence timeout on multi-pattern asks (`elspeth-obs-8f82c91147`).** All three edge-class scenarios (multi-region fork routing, financial/non-financial co-occurrence analysis, LLM enrichment plus webhook return) hit the composer's 180s per-turn time budget without producing a final configuration. The composer kept calling its internal building tools without arriving at a final response. Two competing hypotheses with different remediations: (a) budget-set-too-tight — fix is to raise it, (b) tool-loop design — agent isn't making progress and more time won't help, fix is to re-shape the tool surface or agent prompt. The discriminating experiment is small (raise the budget to ~600s, re-run the same three scenarios).

**Per-call LLM audit records not yet written (`elspeth-obs-7382fbabc4`).** The Landscape audit DB correctly captures run identity, configuration version, schema-contract hash, secret-resolution fingerprints, per-row identity, the sequence of stages each row passed through, and the precise terminal outcome of every row. What it does not yet capture is per-call request/response detail for LLM operations — the exact prompt sent, the response text returned, the tool calls the model made internally. Model identifier and token counts are written inline alongside each output row (so per-row "which model decided this?" is answerable today), but the question-and-answer evidence behind the classification is not in the audit trail. This is the gap between "we have a defensible decision" and "we have the evidence behind the decision." Scope: contained — wire the existing per-call hook through to the `calls` table.

There was also a fourth observation worth noting (Sarah's PDF-coding-scheme limit-probe stall), which doesn't fit the other clusters and warrants its own diagnostic pass before any claim about composer reliability under sustained dialogue is extended. Smaller in scope than the three above.

---

## 5. Proposal to continue

The composer is closer to working than the headline "0/9 working pipelines under hard-mode conditions" number suggests. The §1 verification run is direct empirical evidence: Linda's three-turn compliance pipeline executed end-to-end on real data with defensible classifications and a complete audit trail, after a single targeted fix.

But the eval defects aren't the proposal. They're context. The proposal is to back continued work on a forward programme that is already in motion — three major workstreams currently in active design or early implementation, plus a strategic direction that this evaluation, the in-flight specs, and the existing platform discipline all converge on. The eval defects are small, scoped, and fall naturally inside that work; they're not the structural ask.

### The three major workstreams already in flight

Each has a substantial design spec in `docs/superpowers/specs/` and a corresponding tracking epic in filigree. They're at different stages but all are past the "should we build this" question and into the "how exactly" question.

**Compiled pipeline architecture** — `docs/superpowers/specs/2026-04-15-compiled-pipeline-architecture-design.md`.

Replaces the current ad-hoc YAML → plugin instantiation → graph build path with a first-class compiled pipeline artefact, split across four subsystems (composer / compiler / executor / landscape) with explicit boundaries between them. Defines `CompilationRequest` and `RuntimeAssembly` contracts; makes the secret-normalisation boundary mechanical (compiler accepts unresolved secret refs only, with declared placeholders rather than ordering-dependent guarantees); pulls trust-tier enforcement into the compile step so it's checked once at compile time rather than scattered across runtime checks; adds logical-digest versioning, shadow-mode rollout/rollback semantics, and a defined error model for compile and assembly failures. This is the durable structural foundation underneath everything else — including the per-call audit-record gap surfaced in §4, which falls naturally inside the compiled-pipeline architecture's audit-completeness scope.

**Composer progress persistence** — `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`, with phased implementation plans Phase 1 (data layer) through Phase 4 (frontend) under `docs/superpowers/plans/`. Tracking feature: `elspeth-90b4542b63`.

Closes the failure mode behind §4's convergence-timeout cluster: when a compose request times out or exhausts its budget, *useful intermediate state survives*. Tool-call breadcrumbs are streamed into the assistant message; intermediate tool turns persist as `ChatMessage` rows; partial-draft state ties to the session and is recoverable. The current behaviour — described in this evaluation as "partial state preserved but operator-facing experience looks like the system hanging" — becomes a recoverable, visible, audit-complete progress trail. Phase 1 (data layer) is the in-flight slice; Phase 2 (redaction) and Phase 3 (compose loop) follow.

**Configuration content references** — `docs/superpowers/specs/2026-05-03-config-content-ref-design.md`, with phased plans Phase 1 (ADR) through Phase 5 (composer tool integration) under `docs/superpowers/plans/`.

Adds a content-reference primitive to the configuration model so large templates, prompts, system messages, and content blobs can be *referenced* rather than inlined into pipeline YAML — with provenance tracked, content hashes recorded, and the same trust-tier and audit machinery applying. Phase 2 introduces the L0/L1 surface; Phase 2b adds the state adapter; Phase 3 wires runtime preflight; Phase 4 brings composer parity; Phase 5 lands the composer tool. Foundational for any pipeline that handles non-trivial content payloads — both composer-built and hand-written. Linda's compliance pipeline in §1 has a 600-word LLM template inlined into the YAML; that pattern doesn't scale, and the content-ref work is what makes it scale.

### The plugin expansion programme

`docs/superpowers/specs/2026-04-14-plugin-expansion-brief.md` and its design companion `2026-04-14-plugin-expansion-design.md` set out the strategic forward direction along five capability axes:

- **Search and retrieval infrastructure** — OpenSearch, Qdrant, Meilisearch, ChromaDB upgraded from a write-only sink to a bidirectional search provider. The foundational axis; most of what follows composes on top of it.
- **Intelligent web scraping** — browser-mode capability for JavaScript-rendered content, declarative interaction model, visual capture (screenshots/page PDFs) for evidentiary preservation.
- **Reporting and notifications** — Markdown / HTML / PDF / JSON report sinks, plus email / Slack / Teams / webhook notification sinks.
- **Document processing and knowledge management** — PDF, DOCX, PPTX ingestion; chunking strategies; embedding transform with provider registry. End-to-end document-to-retrieval pipeline.
- **Monitoring and continuous analysis** — change detection, entity extraction, scheduled triggers, dashboard sinks.

These compose into four pipeline patterns the platform is heading toward — *Monitor & Alert*, *Research & Report*, *Screen & Assess*, *Discover & Surface* — each one a direct response to a class of work currently performed by humans with limited audit support. The composer is the user-facing capability that lets non-engineers reach those patterns through dialogue. Without continued work on the underlying axes, the composer's reach is bounded by today's plugin set — which is what produced the SharePoint and HubSpot refusals in §2. The refusals were the *correct* behaviour today; the forward programme is what eventually closes those refusals into supported configurations.

### The named defects from §4 fall inside this work

- **Model-name validation** (`elspeth-obs-f3143acba2`) — small, immediate, can land inside the composer-correctness epic (`elspeth-e1ab67e55a`) without waiting on any of the major specs.
- **Convergence-timeout investigation** (`elspeth-obs-8f82c91147`) — the composer-progress-persistence work above subsumes the recovery story regardless of how the diagnosis lands. Half-day discriminating experiment up front to scope the persistence work correctly.
- **Per-call LLM audit records** (`elspeth-obs-7382fbabc4`) — falls naturally inside the compiled-pipeline architecture's audit-completeness scope; not standalone work.

### What sponsor support buys, what it doesn't

What it buys: continued implementation of the three major specs through their phased plans; the plugin expansion programme along its dependency-ordered capability axes (search-and-retrieval foundational, scraping and reporting parallel, document-processing built on the foundation, monitoring as the integrative layer); the eval defects landed alongside as housekeeping inside the larger work.

What it doesn't buy: a calendar commitment. The platform ships work-until-done, not to fixed dates, and committing to a calendar at this stage would be governance theatre rather than a real plan. What sponsor support changes is *whether the work continues*, not *when it lands*.

The case for backing this is direct. The load-bearing trust claim — that constraints articulated in dialogue survive end-to-end through to runtime classification — is empirically demonstrated (§1), not predicted. The major forward workstreams are specified to a level of detail that lets implementation start without further design — these are not "research questions" but "engineering plans with reviewer-hardened specs." The risk profile is the ordinary one (we know how to build this, the question is whether to spend the time), not the exotic one (we don't know if this is feasible).

Without sponsor support, the composer and the broader platform remain in their current state: a working prototype, a well-documented forward direction, three substantial design specs ready for implementation, and no committed path forward.

---

## Companion documents

| Document | Purpose |
|---|---|
| `composer-briefing-2026-05-03.md` | Formal SES briefing — full eval, residual risks, talking points, strategic-direction appendix |
| `composer-fieldreport-2026-05-03.md` | CTO-facing field report — verbatim transcript-level evidence for all nine scenarios, unsoftened |
| `composer-eval-hardmode-2026-05-03.md` | Engineering eval report — hard-mode (3×3 stratified personas) methodology and aggregate metrics |
| `composer-eval-basic-2026-05-03.md` | Engineering eval report — basic-mode (LLM-driver) methodology, surface this walkthrough's claims build on |

Raw artefacts at `evals/2026-05-03-composer/`. Run IDs for the executed scenarios are in each scenario's `run.json`. Audit DB at `data/runs/audit.db` (development environment).
