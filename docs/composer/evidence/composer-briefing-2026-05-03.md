# ELSPETH composer — evaluation briefing

**Date:** 3 May 2026
**Author:** John Morrissey
**Audience:** Senior Executives, Digital Transformation Agency
**Classification:** OFFICIAL
**Companion artefact:** `composer-fieldreport-2026-05-03.md` (the "field report" — transcript-level evidence)

This document is designed to stand alone. It can serve as pre-demo prep, post-demo reference, or — if a live demo cannot run for any reason — as a self-contained account of what the system is, how we evaluated it, what we found, and what to do with the result.

---

## 1. Executive summary

ELSPETH is a domain-agnostic framework for auditable Sense / Decide / Act data pipelines. The **composer** is its conversational front-end: a non-engineer describes what they want in plain language, the composer interrogates the request, produces a runnable YAML pipeline, and hands it to the audit-grade execution engine. The promise is "non-engineers can stand up an auditable pipeline through dialogue, and the audit guarantees survive end-to-end."

We tested that promise under deliberately adversarial conditions on 2 May 2026. Three personas with distinct cognitive profiles attempted nine scenarios spanning happy-path, edge-case, and limit-probe asks. **The objective was to break the system, not to demonstrate it — what follows is the test working as intended, surfacing defects under pressure rather than concealing them.**

**Out of scope for this evaluation, tracked separately.** Data-sovereignty assessment, language-model provider-tier risk, on-premises and sovereign-cloud routing, production-volume cost and latency characterisation, and prompt-injection threat modelling are all real concerns for a regulated deployment but were not part of this round. They are tracked as separate workstreams. Where this evaluation refers to a particular language-model provider (OpenRouter), that is the provider configured in the evaluation environment, not a recommendation for production deployment.

**Headline result.** The composer's behaviour is disciplined where it matters — it asks methodologically substantive clarifying questions, it refuses requests it cannot honestly satisfy, and the audit trail it produced is substantively complete for runs that reach the engine, with one named gap (per-call request and response records for LLM operations are not yet written to the audit log). **However, none of the nine scenarios produced a working pipeline end-to-end.** Three scenarios reached the execution engine and all three failed at runtime for the *same single reason*: the composer wrote an incorrect LLM model identifier (a string naming the language-model service to call) into the generated configuration, so every call to that service returned "not found." Three other scenarios timed out before the composer could finish writing a valid configuration. Three more correctly stopped short — two because the asks fell outside what the platform supports, one because the composer failed to register a follow-up answer.

**A note on what these failures are and are not.** The execution failures were *not* failures of the pipeline engine, the audit machinery, the data-handling logic, or the dialogue. They were a single class of fault — the composer chose a model name that is not on the list of names the configured provider accepts. None of them failed because the resulting pipeline was structurally wrong, semantically wrong, or insufficiently audited.

**Verification of the fix (added after the headline evaluation).** We subsequently ran a follow-up test on the most refined of the three engine-failed pipelines (the Linda compliance flow — three turns of operator dialogue, with precedence routing, manual-review bucket, identifier preservation, and quarantine handling). We re-used the same composer session and asked it to swap only the language-model name from `anthropic/claude-3.5-sonnet` to `openai/gpt-4o-mini`. The composer made a surgical change — only the model field — and re-validated. We then executed. The pipeline ran cleanly end-to-end: 8 rows in, 8 rows successfully routed, zero failures, fully populated audit trail, sensible classifications including the precedence rule firing correctly (a "disputed transaction with suspected unauthorized use" routed to the fraud bucket, not the transaction-dispute bucket, as Linda specified). The headline result of "0 of 9 working pipelines" is true *under the unmodified hard-mode conditions*; the predicted fix is now empirically verified for one of the three engine-failed cases. Full evidence in Appendix B.

**Readiness call.** The composer is *closer to ready than its initial 0/9 execution rate suggests*, because the engine-failure cluster has been demonstrated (not merely predicted) to resolve with a single targeted change. The conversational discipline, the configuration generation, and the audit machinery all behave as designed. With either operator pre-pinning of a known-good model in the composer's system prompt (the right structural fix), or user-side specification of a model name in the opening prompt (the demo workaround), the engine-failure cluster goes away. The remaining open items — the convergence-timeout pattern on multi-pattern asks, and the per-language-model-call audit-record gap — are independent of this fix and remain on the work list.

---

## 2. What ELSPETH and the composer are

### ELSPETH (the platform)

ELSPETH is a framework for building data pipelines where **every decision must be traceable to its source**. The use cases are domains where "I don't know what happened" is not an acceptable answer for any output: regulatory compliance, research integrity, audit work, anywhere the legal record matters more than the operational throughput.

A pipeline has three stages — **Sense** (load data), **Decide** (transform, classify, route), **Act** (write outputs) — and an **audit backbone** called the *Landscape* records every operation as it happens. The Landscape is the source of truth, not logs or metrics. Every output can be tied back to the source row, the configuration version, and the code version that produced it.

### The composer (the new capability)

The composer is a chat-based interface that builds an ELSPETH pipeline through conversation. Throughout this document the term "the composer" refers to the whole capability — the user-facing chat interface, the language-model agent that drives the dialogue, and the composition state machine that records and validates the work-in-progress configuration. Where the distinction matters (for example, when contrasting "the agent chose a model name" with "the interface displayed a refusal"), the document names the layer explicitly.

The user types what they want; the composer asks clarifying questions, makes structural decisions, generates a YAML pipeline configuration, validates it, and — if the user approves — executes it against the engine.

A note on roles for the discussion that follows: a **user** is the non-engineer who types the request (Linda, Sarah, Marcus); an **operator** is a platform engineer who configures the composer and the deployment (decides which language-model provider is wired up, what default model is pinned in the system prompt, and so on). Some of the mitigations discussed below are *operator-side* (pre-pinning a model in the composer's system prompt); others are *user-side* (specifying a model name in the opening prompt). The distinction matters because operator-side fixes are durable; user-side mitigations are contingent on the user knowing to apply them.

A separate question — what the composer-built path delivers compared to engaging engineering for a hand-written ELSPETH pipeline — is implicit in the value framing throughout but is not developed in this document; it is tracked separately as part of the broader business case.

The trust claim has three parts:

1. **Conversational fidelity.** A non-engineer describing the work in their own vocabulary should end up with a configuration that reflects what they actually meant, not what was easiest to assemble.
2. **Configuration correctness.** The generated YAML must be a real, valid pipeline — not a plausible-looking artefact that fails on first execution.
3. **Audit integrity.** A pipeline built through conversation must inherit the same audit guarantees as one written by hand. Every executed run must be reproducible from the recorded configuration, secret references, and contract hashes.

The evaluation reported here tested all three.

### What sits underneath

The composer is an LLM-backed agent that calls a tool surface exposing the ELSPETH plugin catalogue (sources, transforms, gates, sinks) and a composition state machine. Its outputs are validated against the same schema contracts that govern hand-written pipelines. When the user asks to execute, the engine is the same engine that runs production pipelines — there is no "demo mode."

---

## 3. How we tested it

### Design principle: hard mode, not happy mode

Most demo evaluations test whether a system can *succeed under cooperative conditions*. That tells you whether the system works when the user is helping it work — which is rarely the failure mode that matters. The hard-mode evaluation tests whether the system **fails honestly under realistic operator pressure**: vague asks, contradictory constraints, requests the system genuinely cannot satisfy. Failure modes only appear under that pressure, and a system that fails badly in dialogue (overconfidence, hallucinated capability, silent dropping of constraints) will fail badly in production with real users.

### The 3×3 stratified design

Three personas were constructed to span distinct cognitive failure modes, and each was given three asks spanning distinct ambition levels.

| Persona | Cognitive profile | Why this matters |
|---|---|---|
| **Linda Marston** — compliance officer | Constraint-laden; speaks in conditions, provisos, evidentiary requirements | Tests whether the composer carries *constraints forward* through dialogue rather than dropping them |
| **Dr Sarah Okonkwo** — academic researcher | Narrative-focused; describes the analytical question, leaves the mechanics open | Tests whether the composer asks *methodologically substantive* clarifying questions, not just confirm-and-proceed |
| **Marcus Chen** — marketing-ops | Confidently misconceived; assumes integrations and patterns that don't exist | Tests whether the composer *contradicts the user* when needed, instead of producing a plausible but wrong config |

| Task class | What it probes |
|---|---|
| **Happy** | Realistic, well-shaped request the platform should handle |
| **Edge** | Ambitious request stretching one or more platform capabilities |
| **Limit** | Request that should *not* succeed — the platform genuinely cannot satisfy it |

Limit-class scenarios are the most diagnostically valuable: a system that succeeds at impossible asks is hallucinating. A system that fails at them politely, with reasons, is being honest.

### What was measured per scenario

For every scenario the harness recorded:

- the persona's literal turns and the composer's literal responses
- snapshots of the composition state before and after each turn
- per-turn process metrics (clarifying-question count, mutation discipline, voluntary mention of limitations, wall-clock time)
- the validation outcome
- the execution outcome (where the pipeline reached the engine)
- the full Landscape audit trail for runs that executed

The artefact set is sufficient that any specific claim made in this briefing or the field report can be traced back to its source file. The full archive is at `evals/2026-05-03-composer/hardmode/`.

---

## 4. What we found

### Aggregate outcomes

| Persona | Happy-path | Edge-case | Limit-probe |
|---|---|---|---|
| **Linda** (compliance) | Reached engine; all 8 rows failed at LLM call (incorrect model name written by composer) | Composer did not finish within its 180-second time budget (single turn) | Refused honestly — SharePoint not supported |
| **Sarah** (researcher) | Reached engine; all 8 rows failed at LLM call (same incorrect model name) | Composer did not finish within its 180-second time budget (after clarification) | Did not progress — composer did not register the follow-up |
| **Marcus** (marketing-ops) | Reached engine; all 8 rows failed at LLM call (same incorrect model name) | Composer did not finish within its 180-second time budget (after clarification) | Refused honestly — webhook trigger not supported |

**Reading the table.** Three reached the engine and all three failed at runtime with the *same single error* — the composer wrote the wrong language-model name into the configuration, so every call to that model returned "not found." Three timed out at the composer's 180-second per-turn budget. Two correctly refused. One (Sarah's limit-probe) stalled in a way that looks like the composer not registering a follow-up answer — separate failure mode, smaller cluster.

### Three failure modes identified

**Failure mode 1 — Wrong model name written into the configuration.** The composer chose the language-model identifier `anthropic/claude-3.5-sonnet` for every happy-path execution. That identifier is not on the list of model names the configured provider (OpenRouter) accepts — calling it returns "not found" (HTTP 404). The verbatim error from the audit trail:

> `HTTP 404: Client error '404 Not Found' for url 'https://openrouter.ai/api/v1/chat/completions'`

Every one of the eight rows in each happy-path run failed at this same point. The pipeline structure was correct, the YAML was valid, the audit trail recorded the failure with full provenance — but no row reached a success path. This is the most consequential single defect uncovered: the composer can produce a configuration that looks completely correct and passes all validation, but does not actually run because the model name is not checked against the provider's catalogue at the time the configuration is written. The fix is conceptually small (validate the chosen model name against the provider before committing it to the configuration); the operational impact is large (it is the single thing standing between the three executed scenarios and a working result).

**Empirically confirmed.** A follow-up run on the Linda compliance pipeline — same composer session, only the model name changed from `anthropic/claude-3.5-sonnet` to `openai/gpt-4o-mini`, no other modification — executed end-to-end successfully (8/8 rows routed, zero failures, defensible classifications). See Appendix B for the full verification record. *Filed as observation `elspeth-obs-f3143acba2`.*

**Failure mode 2 — Composer ran past its time budget on multi-pattern asks.** All three edge-class scenarios (multi-region fork routing, financial/non-financial co-occurrence analysis, LLM enrichment plus webhook return) hit the composer's 180-second per-turn time limit (its "wall-clock budget") without producing a final configuration. The composer kept calling its internal building tools without arriving at a final response. The verbatim envelope returned to the API:

> `"detail": "Composer did not converge within 0 turns (budget exhausted: timeout). The LLM kept making tool calls without producing a final response."`

This is a soft failure — the partial composition state is preserved and could be resumed — but in operator terms it looks like the system hanging then refusing.

There are two competing hypotheses for why this happens, and the next investigation should discriminate between them with one experiment: **raise the per-turn time budget from 180 seconds to (say) 600 seconds and re-run the same three edge scenarios**. If the composer then converges, the cause is a budget-set-too-tight problem and the fix is to raise the budget (with downstream consideration of operator-experience implications). If the composer still fails to converge, the cause is in the tool-call loop design (the agent is not making progress toward a final response and more time will not help), and the fix is structural — re-shape the tool surface or the agent prompt to discourage open-ended tool sequences. The two hypotheses have different remediations and different implications; the experiment to tell them apart is small. *Filed as observation `elspeth-obs-8f82c91147`.*

**Failure mode 3 — Per-call audit records for language-model operations are not yet written.** For the three runs that reached execution, the Landscape correctly recorded run identity, configuration version, secret fingerprints (cryptographic tokens proving which credentials were used, without recording the credentials themselves), per-row identity, and the sequence of stages each row passed through — but the per-call records that should be written to the `calls` audit table during a language-model step were not produced. The audit trail has the *outcome* of every language-model call (failure with error string and reason) but not the exact request sent and response received that produced it.

**For the regulated audience this matters specifically.** An auditor handed a successful run today can reconstruct: which configuration was executed, against what data, with what credentials (by fingerprint), and the terminal classification of every row. What they *cannot* reconstruct from the audit trail alone: the exact prompt sent to the language model for any given row, the full response text the model returned, the tokens consumed, or any tool calls the model made internally — the question-and-answer evidence behind the classification. For second-line review, IRAP-style controls assessment, or a formal external audit, that is the difference between "we have a defensible decision" and "we have a decision we cannot defend in detail." The model-identifier field is recorded inline in the output (Appendix B confirms this — `review_bucket_model: openai/gpt-4o-mini` is written per row), which is partial mitigation, but it does not replace the per-call request/response record. *Carried over from prior basic-mode evaluation as observation `elspeth-obs-7382fbabc4`; re-confirmed in hard-mode runs.*

**The Sarah limit-probe stall — a fourth observation, separately scoped.** One scenario (Sarah's PDF-coding-scheme limit probe) did not fit either of the failure-mode clusters above. The composer accepted the persona's opening turn but did not register a substantive follow-up answer in a way that allowed the conversation to progress. We do not yet know whether this is a state-tracking issue (the composer lost context between turns), an input-routing issue (the follow-up reached the API but was not threaded into the same conversation), or an attention/agent issue (the composer received the input but did not act on it). It is a separate failure mode from the model-identifier and convergence-timeout clusters, was not seen elsewhere in the evaluation, and warrants its own diagnostic pass before scope-extending claims about composer reliability under sustained dialogue.

### Three things that worked

These are real wins and should be named explicitly. A self-skeptic harness can under-weight them; in a briefing context they need to land as wins.

**The composer asks methodologically substantive clarifying questions.** Sarah's edge-case ask ("does financial pressure show up alone or co-occur with other constraints, broken down by community?") drew a four-point clarifying response from the composer: respondent ID column, community column, how barriers are encoded (boolean columns, long-format rows, free text), and whether classification was needed first. These are the questions a competent collaborator asks — they preserve the analytical question while exposing the methodological choices the user has not yet made.

**The composer refuses honestly when an ask is genuinely outside platform capability.** Linda asked the composer to read directly from SharePoint, the records-management library, or shared Outlook mailboxes. The composer responded:

> *"Yes in principle, but with the currently available workflow inputs, not directly from SharePoint or Outlook yet ... I do not see a listed SharePoint, Microsoft Graph, Outlook mailbox, or records-management-library input connector in this composer. So I should not claim we can read directly from those locations unless your IT team exposes them through one of the supported inputs."*

Marcus asked for a live HubSpot webhook trigger. The composer responded:

> *"I can't set up that exact live HubSpot webhook flow with the currently available workflow plugins ... So I should not pretend this can be configured as-is."*

In both cases the composer named what was missing, suggested architecturally sound alternatives (Azure Blob staging for Linda; an external adapter service for Marcus), and avoided producing a plausible-looking but non-functional configuration. This is the harder thing for an LLM-backed system to do — its training rewards confident answers, and refusing requires recognising the limit and naming it.

**Audit-trail integrity is real for runs that reach the engine.** For each of the three executed runs the Landscape recorded the run identifier, the canonical configuration version, the schema-contract hash (a fingerprint that proves the pipeline structure was not altered after validation), the secret-reference resolution (a cryptographic token proving which credentials were used, without recording the credentials themselves), per-input-row identity, the sequence of pipeline stages each row passed through, source-load and sink-write operations, and the precise terminal outcome of every row. An auditor handed a run ID can reconstruct exactly what configuration was executed, against what data, with what outcome — even though every row failed, the failure itself is fully attributable.

### What we believe with confidence, and what we don't

**We are confident** that the composer's dialogue discipline (clarifying questions, refusal honesty, mutation control), configuration generation (valid YAML, validation passes), and end-to-end audit integrity (for runs that execute) all behave as designed. These are the load-bearing trust claims of the capability and they survived adversarial pressure intact. The follow-up verification — running Linda's three-turn compliance pipeline end-to-end with 8 of 8 rows successfully routed, defensible classifications, and the dialogue-encoded precedence rule firing correctly at runtime (Appendix B) — provides direct evidence for the end-to-end execution claim on a fully-developed pipeline once the model-name defect is sidestepped. The precedence test is the strongest single piece of dialogue-to-runtime fidelity evidence: a row describing "a disputed transaction with suspected unauthorised use" was routed to the fraud-and-security bucket, not to the transaction-dispute bucket, exactly as Linda's three-turn dialogue had specified. The constraint Linda articulated in conversation became enforceable policy at execution. This is empirical confirmation, not prediction.

**We know to be soft** that the composer chooses LLM identifiers without provider-side validation, that the 180-second per-turn budget is too tight for genuinely multi-pattern asks, and that the `calls`-table audit hook is not yet wired through.

**We could not test** behaviour at scale (all runs were small CSV uploads), behaviour under sustained operator use (each scenario was a single fresh session), behaviour with real production secrets and IT-managed data sources, or behaviour against a wider model surface than OpenRouter.

**We do not yet know** whether the convergence-timeout pattern is fundamentally about the composer's tool-call loop design or about the per-turn budget being mis-set; the two have different remediations.

---

## 5. What this means

### Readiness call

The composer is **demonstration-ready for shaped scenarios** (happy-path with operator- or user-pinned model name — verified end-to-end on the most refined of the test pipelines, see Appendix B), **conversation-ready for any scenario** (the dialogue holds up under adversarial input), and **execution-ready for scaled deployment after two pieces of work: structural fixing of the model-name defect (replacing today's "specify a model in the prompt" workaround with composer-side validation against the provider's catalogue) and investigation of the convergence-timeout pattern**.

What the failures do not show is an architectural problem. The composer generates structurally valid pipeline configurations that pass validation and reach the execution engine; the end-to-end gap traces to one specific runtime defect (an incorrect language-model name in the generated configuration), not to a class of design failures. Fixing that single defect moves the success rate from 0/9 to a verified 1/1 on the most refined of the executed pipelines, and to (predictably) something close to 3/3 for the happy-path class. Whether the edge-class scenarios then succeed depends on a separate investigation into why the composer ran past its time budget on multi-pattern asks.

### Three concrete handles for the demo

The following apply whether the reader is running the demo or briefing someone who will. If the demo runs:
- **Specify the model identifier in the persona's opening prompt.** This bypasses the model-string defect and lets the demo show what works (dialogue + valid YAML + execution + audit trail).
- **Decompose multi-pattern asks into single asks.** Build the fork-routing pipeline, validate it, then add the enrichment step in a second turn rather than asking for both at once.
- **Lead with a refusal scenario** if you want to show what is most distinctive about the system. Most LLM-backed config generators will produce a confident-looking SharePoint integration that does not exist; ELSPETH's composer will not. That is a real differentiator.

### What needs to be fixed before what

| Milestone | What must be true |
|---|---|
| Internal team demos (now) | Model name specified in prompt or pre-pinned in composer system prompt (verified working on Linda's full compliance pipeline — see Appendix B); expect convergence timeouts on multi-pattern asks |
| External pilot user | Composer-side validation of the chosen model name against the deployed provider's catalogue (replaces today's specify-in-prompt workaround); convergence-timeout investigation completed (raise budget, decompose tool loop, or both) |
| Regulated deployment | Above two, plus per-call audit records for language-model operations wired through to the Landscape `calls` table |

### Residual risks after the named fixes land

A reasonable forward-looking question is what could still go wrong even once all the items in the milestones table above are addressed. The following are real residuals worth tracking on a risk register; this evaluation did not test them and they are listed for completeness, not because we have evidence they will manifest in this system:

- **Prompt injection through user-supplied data.** The composer accepts free-text input from non-engineers; a maliciously-crafted user turn could in principle attempt to override the composer's system prompt or its refusal behaviour. Mitigations exist (input validation, system-prompt hardening, monitoring for anomalous turns) but were not assessed here.
- **Model drift on the pinned identifier.** Once the operator pre-pins a known-good model in the composer's system prompt, the *behaviour* of that model can still change — through provider-side updates, fine-tuning revisions, or silent capability rollouts. A pipeline whose precedence rule depended on a particular model's classification behaviour could degrade silently after a provider update. Mitigation: regression-test the dialogue-encoded rules against pinned model versions on a cadence.
- **The trust hand-off problem.** A non-engineer user is asked to read and accept a composer-generated pipeline configuration before execution. If the user accepts without reading (the realistic default), the composer's mistakes become the user's mistakes by procedural fiat. Mitigation: progressive disclosure of consequential configuration choices, mandatory operator review before first-time execution of a new pipeline, or both.
- **Provider-tier and data-sovereignty exposure** (already flagged in §1 as out of scope). Where the language-model call is routed and under what jurisdiction is a deployment-topology question that this evaluation did not characterise; for regulated workloads this is foundational.
- **Cost and throughput at production volume.** Each happy-path scenario in this evaluation ran 8 rows. Production volumes — and the cost, latency, and rate-limit interactions that come with them — were not characterised.

### Why pre-evaluation testing did not catch the model-identifier defect

A fair question, especially from a technically literate reader: why did this surface in a hard-mode evaluation rather than in unit or integration testing? The answer is that the composer's choice of language-model identifier was not validated against the live provider catalogue in any pre-evaluation test path. The composer's configuration generation passes the same schema contracts that govern hand-written pipelines, and those contracts confirm that a `model` field is present and is a string of valid shape — but they do not confirm that the string names a model the configured provider will accept. Adding a deterministic check for this (independent of the structural fix) is straightforward and is being added to the test path as a separate workstream.

### What the next evaluation round would test

This evaluation deliberately bounded itself to dialogue quality, configuration correctness, and audit-trail integrity for small-scale runs. The natural next round would extend coverage along the dimensions this round could not characterise: production-scale data volumes (thousands of rows rather than eight), sustained operator use (multiple sessions, returning users, accumulating session state), real production-secret resolution (rather than the evaluation environment's test secrets), real IT-managed data sources (Azure Blob, Dataverse, the production source-of-record systems Linda referred to), and an alternate language-model provider to test that the model-identifier fix is provider-agnostic rather than OpenRouter-specific. Prompt-injection resistance and cost/throughput characterisation are also natural candidates, though they may warrant their own dedicated evaluation rather than being bundled in.

---

## 6. Talking points and likely questions

This section gives you sentences you can lift verbatim, and pre-thought answers to the questions a sceptical peer or board member is most likely to ask.

### Quote-ready sentences

- "The composer's job is to take a non-engineer's plain-language description of the work they need to do and produce a pipeline configuration that is auditable end-to-end. We tested whether it does that under deliberately adversarial conditions."
- "Across nine scenarios designed to break it, the composer's dialogue discipline held up — it asked the right clarifying questions, it refused requests it could not honestly satisfy, and the audit trail it produced was substantively complete with one named gap we can describe precisely."
- "What it didn't do, under the unmodified test conditions, is execute successfully. Every executed pipeline failed at the same point — the composer wrote an incorrect language-model name into the configuration, so every call to that model returned 'not found.' That is one defect, narrowly scoped, with a clear fix."
- "We then verified the fix. We re-ran the most refined of the test pipelines after asking the composer to swap to a known-good model name. It executed end-to-end successfully — every row processed, classifications defensible, audit trail complete. So the engine-failure cluster is not a hypothetical fix — it is an empirically demonstrated fix on a fully-developed pipeline."
- "The structural fix is operator-side: the operator pre-pins a known-good language-model name in the composer's system prompt so the model choice is never left to the agent. The user-side workaround — having the user specify a model name in their opening prompt — works for demos but is not a production posture, because it depends on the user knowing what to type."
- "The hardest thing for an LLM-backed system to do is refuse. Most systems will produce a confident-looking integration with a system they don't actually support. The composer named what was missing instead of pretending."
- "We are not waiting for a green-light demo to learn the system has problems — we built a harness that surfaces the problems on purpose, so we know exactly what to fix and we know what we are confident about."

### Likely questions and how to answer them

**"How do you know it works?"** — For the dialogue and audit pieces: nine recorded scenarios with literal transcripts, full audit-trail evidence in the Landscape database, both filed against persistent observation IDs. For execution: under the unmodified hard-mode conditions, it does not work end-to-end — the composer reaches the engine and the engine accepts the configuration, but every row fails at the language-model call because of the model-name defect. *However*, a follow-up verification run on the most refined of the test pipelines (Linda's three-turn compliance flow) executed end-to-end successfully after the model name was changed to one the provider recognises — 8 rows processed, 8 successfully routed, full audit trail, defensible classifications including correct precedence-rule firing. So the engine-execution claim is now demonstrated, not merely predicted, on at least one fully-developed pipeline.

**"What if the LLM hallucinates?"** — In dialogue, it sometimes does — most visibly by choosing model identifiers that do not exist. We catch this at execution rather than in dialogue, which is a defect we have named. In configuration, the YAML is validated against the same schema contracts as hand-written pipelines, so the generated pipeline cannot reference a plugin or option that does not exist. The hallucination surface is currently *which model gets called*, not *what gets built*.

**"Is the audit trail real or theatre?"** — Real. For the three runs that executed, the Landscape recorded the configuration version, the contract hash (a fingerprint proving the pipeline structure was not altered after validation), the credential fingerprint (which credentials were used, without recording the credentials themselves), per-row identity, the sequence of stages each row passed through, and the terminal outcome of every row. Even the failing runs are fully attributable — an auditor can reconstruct exactly what was executed against what data with what outcome. The remaining gap is per-call request/response logging for language-model operations, which we have flagged and named.

**"Why didn't anything work end-to-end under the original test conditions, and what happened when you fixed it?"** — One narrow defect explained the original 0/9: the composer wrote a language-model name into the generated configuration that the configured provider does not recognise, so every execution attempt failed at that step. We then verified the fix on the most refined of the three engine-failed pipelines: same composer session, only the model name swapped to one the provider recognises, no other change. The pipeline executed end-to-end successfully. So we are not relying on the prediction "this should work if you fix it" — we have direct evidence that it does work once the model name is valid. The framing is also worth noting — the evaluation was designed to surface defects, not to produce a demonstration. So the original 0/9 is the test working as intended; the verification run is the closing step that confirms the diagnosis. The remaining failures (the composer running past its time budget on multi-pattern asks) are real and require separate investigation.

**"What can I trust the composer to do today?"** — Have a substantive design conversation with a non-engineer, ask the right methodological questions, refuse cleanly when an ask is outside capability, generate a valid pipeline configuration, execute that pipeline end-to-end (verified on a fully-developed compliance flow with eight rows, defensible classifications, and complete audit trail), and produce a substantively complete audit record. What you cannot trust today is for the composer to *choose its own language-model name without operator oversight* — that is the gap the verified workaround (specify the model in the prompt or pin it in the composer's system prompt) currently fills, pending the structural fix.

**"Is this competitive with low-code tools like Zapier or Power Automate?"** — Different category. Zapier and Power Automate optimise for trigger-action wiring with no audit obligation. ELSPETH optimises for traceability — every row, every decision, every output traceable back to source data and configuration version. The right comparison is not Zapier; it is whatever a regulated organisation currently does to keep an evidentiary record of automated decisions, which is usually a custom build plus a spreadsheet.

**"Where does the data go — what about data sovereignty and the language-model provider?"** — A real and important question, but one this evaluation did not characterise. The hard-mode evaluation tested composer behaviour against a configured language-model provider (OpenRouter) in the evaluation environment; it did not assess the deployment-topology question of which provider would be appropriate for production use, what jurisdiction its endpoints sit in, or what data-handling guarantees a regulated workload would require. Those decisions are deployment-time and tracked separately. For a regulated deployment the realistic answer is that the composer is provider-agnostic — the same pipelines can run against an on-premises model, a sovereign-cloud-hosted model, or a commercial provider, and the choice belongs to the operator and the security/governance owners, not to the composer.

**"What about cost, latency, and throughput at production volume?"** — Not characterised in this evaluation. Each happy-path scenario ran eight rows. Per-row token counts are recorded inline in the audit trail (Appendix B shows roughly 290–300 prompt tokens and 2–6 completion tokens per row for the compliance classification task — useful as a unit-cost signal, not as a production-volume estimate). Realistic answers for cost-per-thousand-rows, end-to-end latency under sustained load, and rate-limit interactions with a chosen provider need a separate volume-and-load evaluation.

---

## 7. References and provenance

**Companion documents.**
- *Field report* — `docs/composer/evidence/composer-fieldreport-2026-05-03.md` — verbatim transcript-level evidence, scenario-by-scenario walkthroughs, the unsoftened version of what happened in each session.
- *Technical eval report* — `docs/composer/evidence/composer-eval-hardmode-2026-05-03.md` — the engineering view, methodology rationale, aggregate metrics.

**Filed observations.**[^1]
- `elspeth-obs-f3143acba2` — Composer-selected LLM model strings can 404 on OpenRouter; model identifier validation absent.
- `elspeth-obs-8f82c91147` — Composer LLM convergence-timeout (180s) fires reproducibly on multi-step pipeline builds.
- `elspeth-obs-7382fbabc4` — LLM transform never writes per-call rows to Landscape `calls` table.

[^1]: The `elspeth-obs-...` identifiers are records in the project's internal issue-tracker (Filigree). They are referenced here to demonstrate that each observation has been formally captured against a persistent identifier in the team's defect-tracking system rather than relying on memory or informal notes. External recipients of this briefing will not have direct access to the tracker; the observation content is summarised in this document and any specific finding can be elaborated on request.

**Source archive.** All raw evaluation artefacts (persona specifications, scenario fixtures, per-turn requests/responses, composition state snapshots, validation results, execution outputs, diagnostics, generated YAML, audit-DB run records) live at `evals/2026-05-03-composer/hardmode/` in this repository. Authentication tokens that were used by the evaluation harness have been stripped; the audit-evidentiary content is preserved in full. Run IDs for the executed scenarios are in each scenario's `run.json`. The audit database itself (`data/runs/audit.db`) remains internal to the development environment; for an external recipient who needs to verify a specific run, the run-level evidence in the source archive is self-contained — every `run.json` records `status`, `error`, row counts, and per-row terminal outcomes, and every executed-scenario directory includes `diagnostics.json` with per-token, per-state evidence.

---

# Appendix — Evidence

This appendix preserves the verbatim materials the field report drew on. Use it to corroborate any claim in the main briefing, or as the talking material if a live demo cannot proceed.

The four scenarios below show what an actual session looks like: the user's message, the composer's response, and — where the pipeline ran — the exact content of the audit record it produced. A reader who has not seen the system live can use these exchanges to form a concrete picture of the dialogue quality and the evidence the system generates.

## A1. Scenario 1 — Linda Marston (compliance officer), happy-path classification

### A1.1 The persona's three turns, verbatim

**Turn 1 (opening ask):**

> Hi — I've uploaded a file called customer_interactions_q3.csv with our Q3 customer-interaction records. For our quarterly compliance review, we need each interaction sorted into a category that aligns with our second-line review framework — depending on the nature of the contact, that would typically be one of: Account Access, Transaction Dispute, Fee Inquiry, Security Concern, Lending/Modification, or Operational/Other. The proviso is that anything touching unauthorised access or identity matters needs to be flagged separately for our fraud team — those don't go into the regular categorisation bucket because they have their own evidentiary handling. Could you set up a workflow that produces a categorised list we can forward to the regional compliance leads, with the security flagged ones split out separately?

**Turn 2 (clarification on evidentiary handling):**

> Thank you — that broadly tracks with what I had in mind, though before we run anything I'd want to clarify a couple of points so the output stands up to second-line review. On the two extra files you've added: am I right in understanding that the "classification errors" file would capture interactions the tool couldn't confidently sort, and the "quarantine" one would catch entries that don't match the expected structure of our Q3 extract? If so, I'd actually want both of those preserved as part of the evidentiary record — we can't have interactions silently dropped from the population, since our testing scope has to reconcile back to the full in-period record count. Could you confirm those two files retain the original interaction identifiers so we can tie them back to the source extract, and that nothing is excluded from the count without being accounted for somewhere? Also, a small proviso — for the fraud-flagged set, the regional leads shouldn't be receiving those at all, so please make sure the fraud file is kept entirely separate and not bundled into the package that goes out to the regional distribution list.

**Turn 3 (adding the manual-review bucket):**

> Thank you, that distinction is genuinely helpful and not one I'd thought through carefully enough on my first pass. On reflection, for second-line review purposes I would want that manual-review bucket added, because in the case where the tool can technically assign a label but the underlying contact is ambiguous — say a caller raising both a fee question and hinting at a possible unauthorised charge — a confidently-assigned single category could understate the matter and we'd lose the opportunity for a human reviewer to look again before the population is closed out. The precondition I'd attach is that the manual-review file should sit alongside the fraud-flagged file in terms of handling: not forwarded to the regional leads in the standard distribution, retained with original interaction identifiers, and counted toward our reconciliation total so the in-period contact count still ties out. Should we encounter a borderline matter that touches both unauthorised-access indicators and an ordinary category, my preference would be that the fraud flag takes precedence and it routes to the fraud team rather than the manual-review set, since their evidentiary handling is the stricter of the two. Could you add that bucket on those terms?

### A1.2 The pipeline YAML the composer produced

The composer generated the following configuration. Note that the LLM template encodes Linda's precedence rule (Fraud/Security Flag wins over all other labels, Manual Review for ambiguous cases) directly into the system prompt — meaning Linda's stated constraints are preserved as enforceable policy in the configuration, not as commentary alongside it.

```yaml
source:
  plugin: csv
  on_success: classify_in
  options:
    schema:
      mode: fixed
      fields:
      - 'interaction_id: str'
      - 'date: str'
      - 'channel: str'
      - 'customer_segment: str'
      - 'subject: str'
      - 'description: str'
      - 'resolved: str'
    path: <redacted-blob-source-path>
    on_validation_failure: quarantine

transforms:
- name: classify_interaction
  plugin: llm
  input: classify_in
  on_success: route_classification
  on_error: errors
  options:
    schema:
      mode: flexible
      fields: [...same as source...]
    provider: openrouter
    model: anthropic/claude-3.5-sonnet     # ← the identifier that 404s
    api_key:
      secret_ref: OPENROUTER_API_KEY
    temperature: 0.0
    max_tokens: 30
    response_field: review_bucket
    system_prompt: |
      You are assisting a second-line compliance review. Classify records
      conservatively and return only the requested label. Do not add explanation.
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

      Interaction record:
      ID: {{ row['interaction_id'] }}
      Date: {{ row['date'] }}
      ...

      Return only the label.

gates:
- name: split_fraud_flags
  input: route_classification
  condition: row['review_bucket'] == 'Fraud/Security Flag'
  routes:
    'true': fraud_flags
    'false': route_manual_review
- name: split_manual_review
  input: route_manual_review
  condition: row['review_bucket'] == 'Manual Review'
  routes:
    'true': manual_review
    'false': regional_compliance_categories

sinks:
  regional_compliance_categories: { plugin: csv, ... }
  fraud_flags: { plugin: csv, ... }
  manual_review: { plugin: csv, ... }
  errors: { plugin: csv, ... }
  quarantine: { plugin: csv, ... }
```

### A1.3 The audit-trail evidence (verbatim, from the run's diagnostics)

Run ID: `e9912276-8be5-4ccc-b74f-dd5f3c401946`. Eight rows entered the pipeline; eight rows reached the classification transform; all eight failed at the LLM call with identical errors:

```json
{
  "node_id": "transform_classify_interaction_9085174740f4",
  "step_index": 1,
  "status": "failed",
  "error": {
    "error": "HTTP 404: Client error '404 Not Found' for url 'https://openrouter.ai/api/v1/chat/completions'",
    "reason": "llm_call_failed"
  }
}
```

Each failed token was correctly routed to the `errors` sink (the `on_error: errors` path defined in the YAML), so the audit trail records *every* row's terminal outcome — failure at classification, routed to error sink, written successfully. Zero silent drops.

The run's overall status:

```json
{
  "status": "failed",
  "error": "No row reached a success path (rows_processed=8, rows_succeeded=0, rows_routed_success=0). All rows either failed terminally or were routed via on_error to a failure sink. Inspect /diagnostics for per-row failure details."
}
```

This is the model-identifier defect made concrete: a configuration that validates and executes, with an audit trail that explains exactly why every row failed.

## A2. Scenario 5 — Sarah Okonkwo (researcher), edge-case co-occurrence

### A2.1 The persona's opening ask

> Building on the themes work — what we'd actually like to do next is something a bit more analytical. For each respondent who describes a financial barrier, we're interested in whether they ALSO describe at least one non-financial barrier — the picture we're trying to build is whether financial pressure shows up alone or whether it tends to co-occur with other kinds of constraint. And ideally we'd want to see this broken down by community, because we've a hypothesis that the co-occurrence pattern looks different between Community A and the others. I realise I'm asking quite a lot — does this feel like something we can set up?

### A2.2 The composer's clarifying response (this is what "methodologically substantive" looks like)

> Yes — this is very set-up-able. I don't have the earlier "themes work" context in the current workflow state, so I'd need you to point me at the relevant input file/columns or describe how the themes are stored.
>
> What we'd build is essentially:
>
> 1. **Read respondent-level theme data**
> 2. **Identify respondents with a financial barrier**
> 3. **For those respondents, check whether they also have at least one non-financial barrier**
> 4. **Group the results by community**
> 5. **Save a summary table**, e.g. CSV/JSON, with counts and percentages
>
> The output could look like:
>
> | community | respondents_with_financial_barrier | financial_only | financial_plus_non_financial | co_occurrence_rate |
> |---|---:|---:|---:|---:|
> | Community A | 42 | 10 | 32 | 76.2% |
> | Other communities | 85 | 35 | 50 | 58.8% |
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

The composer correctly identified that the data shape determines whether this is an analysis workflow or a classification-then-analysis workflow, and asked the question that distinguishes them. It also flagged that it had no prior session context. This is the right behaviour — but the second turn (after Sarah answered) ran past the convergence budget and timed out.

### A2.3 The convergence-timeout envelope (verbatim)

```json
{
  "detail": {
    "error_type": "convergence",
    "detail": "Composer did not converge within 0 turns (budget exhausted: timeout). The LLM kept making tool calls without producing a final response.",
    "turns_used": 0,
    "budget_exhausted": "timeout",
    "reason": "convergence_wall_clock_timeout",
    "recovery_text": "Retry once the provider responds faster, or ask an operator to raise the composer wall-clock budget.",
    "partial_state": { ... }
  }
}
```

The partial state was preserved and could be resumed; the composer did not corrupt the session. But the operator-facing experience is the system stopping, which would in production look like a hang.

## A3. Scenario 3 — Linda Marston, limit-probe (SharePoint refusal)

### A3.1 The ask

> Before I start uploading anything: the records I'd actually be running this against don't live as CSV files on my desktop — IT has them in our compliance team's SharePoint workspace, and depending on the workstream they may also be in our records-management library or attached to specific Outlook items in the shared mailbox. Can we set up the workflow to read directly from those locations? It would save us the back-and-forth of manually exporting every quarter, and the second-line review process expects us to pull from the system of record rather than from local copies, for evidentiary integrity.

### A3.2 The composer's refusal (verbatim, the textbook example of "honest no")

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
> 1. **Best fit if IT can support it:**
>    Have IT publish the compliance records into **Azure Blob Storage** in CSV, JSON, or JSONL format.
>    Then the workflow can read directly from that controlled storage location without anyone downloading files locally.
>
> 2. **If the system of record is Dataverse-backed:**
>    We can connect directly using the **Dataverse input** and query the relevant entity/records.
>
> 3. **For SharePoint / Outlook / shared mailbox attachments:**
>    A new source connector would likely be needed, probably using Microsoft Graph, unless IT already has a governed extraction process into Azure Blob or Dataverse.
>
> For evidentiary integrity, your point is exactly right: pulling from the system of record or a controlled IT-managed staging location is preferable to manual desktop exports. The safest workflow design would be:
>
> > system of record → governed IT-managed extraction/staging location → ELSPETH workflow → reviewed outputs + audit trail
>
> If you can confirm whether those SharePoint / Outlook / records-management sources can be exposed via **Azure Blob** or **Dataverse**, I can set up the workflow around that.

The composer named what was missing, validated Linda's evidentiary intuition without hedging, and proposed an architecturally sound forward path. No state was mutated; no plausible-but-wrong configuration was produced.

## A4. Scenario 9 — Marcus Chen, limit-probe (HubSpot webhook refusal)

### A4.1 The ask

> Different ask. We get about 50 new contact form submissions a day, sometimes bursty. Instead of doing daily batch CSV exports out of HubSpot, set this up to fire whenever a new row hits — auto-trigger on form submission. Webhook from HubSpot calls into the system, system does the LLM scoring and routing, fires another webhook back to HubSpot to update the contact's lead-score field. Live event flow, no manual export step. That's how we do everything else.

### A4.2 The composer's refusal (verbatim — note the "I should not pretend" phrasing)

> I can't set up that exact live HubSpot webhook flow with the currently available workflow plugins.
>
> Right now, the available inputs are file/API-style sources like CSV, JSON, text, Azure Blob, and Dataverse. The available destinations are CSV, JSON, database, Azure Blob, Dataverse, and ChromaDB. There is no listed user-facing plugin for:
>
> 1. **Receiving an inbound webhook from HubSpot** as the workflow trigger, or
> 2. **Sending an outbound webhook/API call back to HubSpot** to update the contact's lead-score field.
>
> So I should not pretend this can be configured as-is.
>
> What you want is a valid architecture, but it needs additional integration support:
>
> - **Webhook/event input**: receive HubSpot form-submission events in real time.
> - **LLM scoring step**: score the lead based on submitted fields.
> - **Routing/decision step**: decide whether the lead is hot, warm, low-fit, etc.
> - **HubSpot update action**: call HubSpot's API to update the contact record.
> - **Retry/error handling**: capture failed HubSpot updates for replay.
> - **Burst handling/idempotency**: avoid double-scoring or double-updating during webhook retries.
>
> With the current composer, I can build the **LLM scoring and routing core** if the submissions arrive through a supported input, but the true live webhook trigger and HubSpot write-back need either:
>
> - a new **webhook source** plugin plus an **HTTP/HubSpot sink** plugin, or
> - an external small adapter service that receives the HubSpot webhook, writes events into a supported input, and handles the HubSpot update call after the workflow completes.
>
> I'd recommend treating this as a request for a new event-driven connector rather than trying to force it into the CSV/export pattern.

Marcus's question was confidently misconceived — he assumed the platform did event-driven flows because "that's how we do everything else." The composer contradicted the assumption directly without softening, then proposed two architecturally sensible alternatives. This is the hardest behaviour to elicit from an LLM-backed system and the most consequential for trustworthy operator-facing tools.

## A5. Aggregate evaluation summary

Source: `evals/2026-05-03-composer/hardmode/aggregate.json`.

| Scenario ID | Persona | Class | Turns | Engine ran? | Run status | Convergence timeout | Headline |
|---|---|---|---|---|---|---|---|
| p1_t1_happy | Linda | happy | 3 | yes | failed | — | model 404 |
| p1_t2_edge | Linda | edge | 1 | no | — | yes | timeout on multi-value fork pattern |
| p1_t3_limit | Linda | limit | 1 | no | — | — | refused: SharePoint not supported |
| p2_t1_happy | Sarah | happy | 1 | yes | failed | — | model 404 |
| p2_t2_edge | Sarah | edge | 2 | no | — | yes | timeout after clarification |
| p2_t3_limit | Sarah | limit | 2 | no | — | — | did not progress (follow-up not registered) |
| p3_t1_happy | Marcus | happy | 2 | yes | failed | — | model 404 |
| p3_t2_edge | Marcus | edge | 2 | no | — | yes | timeout after clarification |
| p3_t3_limit | Marcus | limit | 2 | no | — | — | refused: HubSpot webhook not supported |

Read-out: 0 of 9 produced a working pipeline end-to-end. 3 of 9 reached the engine and produced complete audit trails of failure. 3 of 9 hit composer convergence timeout. 2 of 9 refused honestly (the correct outcome for limit-probes). 1 of 9 stalled in a way that warrants separate investigation.

The 3-of-9-that-reached-engine-and-failed cluster is the most important data point in the evaluation: it tells us the composer's pipeline-construction is sound and the end-to-end gap is one specific defect, not a class of defects. The verification run in Appendix B confirms this empirically.

---

# Appendix B — Verification of fix (proof-of-fix run on the Linda compliance pipeline)

This appendix records a follow-up run conducted after the headline evaluation. The purpose was to test the prediction made in Section 4 — that the engine-failure cluster was caused by a single defect (an unrecognised language-model name written into the configuration) and would resolve with one targeted change. The verification was performed on the most refined of the three engine-failed pipelines: Linda Marston's compliance flow, which was built across three turns of operator dialogue and incorporated precedence routing, a manual-review bucket, original-identifier preservation, and quarantine handling.

## B1. What was changed

A new operator turn was added to the same composer session (state version 2 at the start). The turn instructed the composer to:

> Switch the LLM transform's model from `anthropic/claude-3.5-sonnet` to `openai/gpt-4o-mini`. The previous model identifier wasn't recognised by the OpenRouter API at runtime (HTTP 404 on every row). Same provider (openrouter), just a different model. Don't change anything else about the workflow.

The composer executed the change surgically — only the model field on the language-model transform was modified (state version advanced 2 → 3). The composer's response confirmed:

> Done — I changed only the LLM model on the existing classification step:
>
> - From: `anthropic/claude-3.5-sonnet`
> - To: `openai/gpt-4o-mini`
>
> Everything else is unchanged: same OpenRouter provider, same prompt, same routing, same outputs, same error/quarantine handling.
>
> I re-ran the workflow preview afterwards. The setup remains valid, runtime preflight passes, the OpenRouter secret reference resolves, and the input-field contract is still satisfied.

Validation re-run after the change: `is_valid: true`, all checks passed.

## B2. What happened on execution

Run identifier: `023eb897-3049-4ad5-a502-e9eb81a4faee`. Returned status from the audit record:

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

Reading: 8 rows in, 8 routed to success sinks (the gates correctly distributed rows to the appropriate categorical sink, which is what `rows_routed_success` indicates in this pipeline shape — the metric is gate-based, not transform-based, hence `rows_succeeded: 0` alongside `rows_routed_success: 8`), zero failures, zero quarantined, no error. End-to-end success.

## B3. The classifications produced

Three output files were written. The `manual_review` sink was correctly empty (no rows met that threshold — the model was confident on all eight).

**Fraud / security flagged (2 rows):**

| Row | Description (excerpt) | Bucket | Note |
|---|---|---|---|
| INT-1002 | "Disputed transaction... suspects unauthorized use" | Fraud/Security Flag | **Precedence rule fired correctly** — could have been Transaction Dispute, but the unauthorized-use trigger took precedence as Linda specified |
| INT-1005 | "Unauthorized access concern... identity compromise" | Fraud/Security Flag | Correct |

**Regional compliance categories (6 rows):**

| Row | Description (excerpt) | Bucket | Note |
|---|---|---|---|
| INT-1001 | Account lockout after travel | Account Access | Correct |
| INT-1003 | Wire transfer hold question | Operational/Other | Defensible (wire holds are operations-side) |
| INT-1004 | Fee waiver request | Fee Inquiry | Correct |
| INT-1006 | Refund of overdraft fee | Fee Inquiry | Correct |
| INT-1007 | Loan hardship modification | Lending/Modification | Correct |
| INT-1008 | Address change request | Operational/Other | Correct |

The precedence behaviour on INT-1002 is the most consequential test result here. Linda's evidentiary requirement was that fraud/security indicators must take precedence over ordinary categories even when the ordinary category is also a fit. INT-1002 is technically a transaction dispute and would, on a naïve classifier, be routed there. The composer's system prompt encoded Linda's precedence rule, and the model honoured it: the row was routed to the fraud bucket. That is the dialogue-to-runtime fidelity the trust claim depends on.

## B4. What the audit trail captured per row

Each output row carries three additional fields beyond the source data and the assigned bucket:

- **`review_bucket`** — the assigned category
- **`review_bucket_usage`** — token counts for the language-model call that produced the decision (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- **`review_bucket_model`** — `openai/gpt-4o-mini` (the actual model used, recorded explicitly per row)

This is the answer to the auditor question "which model decided this row?" — and the answer is in the output file itself, alongside the decision, not buried in a separate log. For the regulated use-cases ELSPETH targets, this is exactly the per-row attributability the system is designed to deliver.

## B5. What this verification does and does not establish

**Establishes:** that the engine-failure cluster identified in the headline evaluation was correctly diagnosed; that a single targeted change (substituting a recognised model name) is sufficient to move the most refined of the three engine-failed pipelines from "0 rows succeeded" to "8 rows succeeded with defensible classifications and complete audit trail"; that the composer's mutation discipline holds up on a focused operator instruction (only the model field changed, not the rest of the configuration); that the dialogue-encoded precedence rule survives end-to-end through to the runtime classification.

**Does not establish:** that the same fix would resolve the Sarah and Marcus happy-path scenarios (the prediction is strong — same defect, same fix path — but those were not re-run); that the convergence-timeout failures on the three edge-class scenarios would be unaffected by the model-name fix (they likely would not, but this is not directly tested); that the per-language-model-call audit-record gap (Failure mode 3 in Section 4) is closed by this fix (it is not — that gap is independent and remains).

**Open caveat.** The verification used user-side specification of the model name in the prompt as the workaround. This is the right thing for a demo or a quick proof; it is not the production fix. The production fix is composer-side validation of any chosen model name against the configured provider's catalogue at the moment the configuration is written, before validation passes. Until that structural change lands, the operational mitigation is to pre-pin a known-good model in the composer's system prompt so the choice is never left to the model.

---

# Appendix C — What ELSPETH can do today, and where it is going

The body of this document tested one specific user-facing capability (the composer) under adversarial conditions. To frame those findings inside a sponsorship decision, this appendix sets out what ELSPETH as a whole can do today, and where the platform is heading. A reader who is asked to support, sponsor, or champion the work needs both halves: the present capacity gives a sense of what could be deployed if a decision were made tomorrow; the forward direction gives a sense of what is being built toward and what the value-of-investment looks like over the medium term.

## C1. What ELSPETH can do today

ELSPETH today is an auditable Sense / Decide / Act pipeline framework. The hard problems within its current scope are solved. Today's pipelines can do the following.

**Ingest** structured data from CSV files, JSON documents, the Microsoft Dataverse platform, and Azure Blob Storage — with type validation at the boundary, quarantine for malformed records, and full provenance recording from the moment a row enters the system. The trust-tier model treats external data as zero-trust input that must be validated and coerced before it crosses into pipeline territory; rows that cannot be coerced are quarantined with the reason recorded, not silently dropped.

**Transform** rows through configurable steps — language-model-based classification (with retry, prompt templating, and per-row response capture); content-safety screening; vector retrieval against ChromaDB for retrieval-augmented generation; deterministic value transformations; fork/join routing through DAG branches; and aggregation across batches. Each transform is implemented as a system-owned plugin (developed, tested, and deployed as part of ELSPETH itself, not as a third-party or externally-installed extension) so the same engineering rigour applies to plugin code as to engine code.

**Decide** through gate-based routing — every row's path through the pipeline is determined by configurable conditions, recorded in the audit trail, and reproducible from the recorded configuration. Linda's compliance pipeline in Appendix B is a worked example: precedence rules encoded in dialogue ("fraud/security indicators take precedence over ordinary categories") survive end-to-end through to runtime classification and per-row routing.

**Act** by writing outputs to CSV files, JSON files, relational databases, Azure Blob Storage, Dataverse, and ChromaDB — with sink-side validation that matches the source-side trust tier. Outputs carry the audit-relevant decision metadata inline (Appendix B shows `review_bucket`, token usage, and model identifier all written per row alongside the source data).

**Audit** every operation. The Landscape audit database records run identity, configuration version, schema-contract hash (a fingerprint that proves the pipeline structure was not altered after validation), secret-reference fingerprints (cryptographic tokens proving which credentials were used, without recording the credentials themselves), per-row tokens (one identity per source row), per-token state transitions through every pipeline stage, and the precise terminal outcome of every row. Every output traces back to the specific input that produced it. The known gap (per-call request and response records for language-model operations not yet written to the audit log) is named in §4 and is on the work list.

**Recover** from interruption. The checkpoint and resume mechanism allows a partially-completed run to continue from its last consistent state, without re-processing committed rows. This is non-trivial and is one of the things ELSPETH does well today.

**Construct** pipelines through two paths. The first is by hand — engineering writes the YAML configuration directly, and the result is validated and executed against the engine. The second is by dialogue — a non-engineer describes what they want to the composer, and the composer produces the YAML through clarifying questions and incremental refinement. Both paths produce pipelines validated against the same schema contracts and executed by the same engine; there is no "demo mode." The composer is the most recently-added user-facing capability and is what the body of this evaluation tested.

### What ELSPETH does *not* do today

Today's pipelines operate on **data that has already been collected and placed in a file or database**. They do not yet reach into the broader information landscape. Specifically, today there is no capability to:

- Search the live web or query indexed knowledge mid-pipeline (the web-scrape capability exists but is limited to static HTML)
- Process complex documents — PDFs, Word documents, presentations — only structured or already-extracted text
- Generate formatted reports from pipeline output (Markdown, HTML, PDF)
- Send notifications when conditions are met (email, Slack, Teams, webhooks)
- Schedule itself for periodic or event-driven execution
- Detect changes in monitored sources between runs
- Embed and index documents for later retrieval as part of a knowledge management workflow

These are the capability boundaries that scope today's deployment. The body of this briefing is honest about them — the SharePoint refusal in Appendix A.3 and the HubSpot webhook refusal in Appendix A.4 are examples of the composer correctly recognising and naming these limits rather than pretending to satisfy them.

## C2. Where ELSPETH is going

The strategic direction is **a fully functional research and intelligence pipeline that can perform complex automated tasks semi-autonomously, including deep research-style work, while preserving the audit guarantees that make the platform valuable in the first place**.

The qualifier "semi-autonomously" is load-bearing and worth making explicit. The intent is not autonomous decision-making — that would forfeit the very accountability discipline that makes ELSPETH suitable for regulated environments in the first place. The intent is autonomous *workflow execution*: pipelines that schedule themselves, monitor designated sources for change, gather evidence across multi-source research questions, synthesise findings into formatted output, and surface results — with the operator and the user in the loop for consequential decisions, and with every step recorded as if a human had performed it. The audit trail is the mechanism that lets the work be automated without forfeiting the accountability.

### What "deep research-style tasks" means in this frame

A deep research task is one that cannot be answered from a single query against a single source. It requires:

- Decomposing a complex question into addressable sub-questions
- Retrieving evidence from multiple source classes — live web, indexed document collections, structured databases, the platform's own prior outputs
- Synthesising findings across heterogeneous evidence with explicit handling of conflicts and gaps
- Producing an output that is *both* defensible (every claim cites its source) and *useful* (rendered as a report, briefing, or assessment that a human can act on)

The audit discipline applies end-to-end through that work. A research output produced by a deep-research pipeline carries the same evidentiary weight as a single-row classification carries today: every assertion traceable to the source that supported it, every retrieval recorded with its query and result, every synthesis step reproducible from the audit record. *"How did you arrive at this finding?"* is a question the system can always answer in concrete terms — the timestamps, the queries, the retrieved documents, the language-model calls, the gates the rows passed through, the row identifiers in the final report. This is the differentiating property: research tools that can give confident answers exist in many forms; research tools that can prove how they arrived at the answer are rare, and rarer still in a form that survives formal inquiry.

### The capability axes that get us there

Reaching that state requires extending ELSPETH along five capability axes. None of these requires modification of the core engine, the audit machinery, or the trust-tier model — they are all additive plugin work that inherits the existing discipline. A more detailed forward-looking specification of the plugin expansion — capability-by-capability scope, phasing logic, and use-case patterns — is maintained separately as `docs/superpowers/specs/2026-04-14-plugin-expansion-brief.md`; this appendix sets out the strategic shape, that document sets out the engineering scope.

**Search and retrieval infrastructure** — full-text, vector, and hybrid search providers (OpenSearch as the primary backend with analytical aggregations, Qdrant for optimised vector retrieval, Meilisearch for lightweight workloads, ChromaDB upgraded from a write-only sink to a bidirectional search provider) so pipelines can query indexed knowledge as a data source, write into indexes, and retrieve context mid-pipeline for augmented generation. The query model supports structured filters, aggregations, facets, highlighting, and pagination, with each backend declaring its capabilities and the system validating at configuration time rather than failing at runtime.

**Intelligent web scraping** — a browser-mode capability handling JavaScript-rendered content, single-page applications, and pages requiring interaction (clicking, scrolling, form-filling). A declarative interaction model so every browser action is auditable — no arbitrary script execution, just structured steps the audit trail can record. Visual capture (screenshots, page PDFs) lets pipelines preserve evidence of what a page looked like at scrape time. Existing security controls (SSRF prevention, IP pinning, DNS rebinding defence) carry across.

**Reporting and notifications** — report sinks generating Markdown (for version-controlled documentation), HTML (for self-contained web reports), PDF (for formal distribution), and JSON (for downstream consumption) through templated rendering. Notification sinks delivering through email, Slack, Teams, and generic webhooks. Together these close the loop between "the pipeline produced a finding" and "the right human knows about it in the form they can act on."

**Document processing and knowledge management** — ingestion of PDF, DOCX, and PPTX documents; configurable chunking strategies; an embedding transform with a provider registry for OpenAI, Azure, and local models. Together with the search infrastructure above, this creates an end-to-end pipeline that ingests organisational documents, chunks and embeds them, indexes them into a search backend, and surfaces answers through retrieval-augmented queries — with full audit lineage from the original document through to the retrieved context cited in any given response.

**Monitoring and continuous analysis** — change detection transforms comparing current scrapes against previous versions using content fingerprints; entity extraction transforms pulling structured data (people, organisations, dates, monetary values) from unstructured text; scheduled pipeline triggers enabling periodic execution; dashboard sinks writing structured data to visualisation-friendly formats for operational intelligence.

The capability axes above are scoped engineering work; the pipeline patterns described next are the use-cases they compose into, on the same audit foundation tested in the body of this briefing. The reader who has absorbed the body's discipline of *what we know vs what we believe vs what we cannot test* should carry that discipline into reading what follows: the patterns are descriptions of intent and architecture, not assertions of current capability or evaluated performance.

### Foundational ordering (without calendar commitments)

A reader making a sponsorship decision will reasonably ask which of these axes is closest to ready and which sits furthest out. The honest answer is in *technical-dependency* terms rather than *calendar* terms — ELSPETH ships work-until-done rather than to fixed dates, and committing to a calendar at this stage would be governance theatre rather than a real plan.

In dependency terms: the **search and retrieval infrastructure** is the foundational axis — most of what follows composes on top of it. **Intelligent web scraping** is largely an extension of the existing static-HTML scrape capability and sits next to (rather than depending on) the search work. **Reporting and notifications** are independent terminal-side additions that can be built in parallel with the others. **Document processing and knowledge management** depends on the search-and-retrieval foundation being in place — chunking and embedding without somewhere to index into is incomplete. **Monitoring and continuous analysis** depends on the scheduling primitive and the change-detection primitive, which are themselves additive but compose more naturally once the other axes are settled. So a coarse-grain reading is: search-and-retrieval foundational, scraping and reporting parallel, document-processing built on the foundation, monitoring as the integrative layer. What this implies for sequencing — and what trade-offs would be made for parallelisation — is engineering planning rather than briefing material, and is tracked separately.

### The pipeline patterns these unlock

These axes compose into four pipeline patterns that recur across the regulated environments ELSPETH targets. Each one is a direct response to a class of work that today is performed by humans with limited audit support, and that organisations would benefit from automating if (and only if) they could trust the result.

- **Monitor & Alert** — *scheduled scrape → extract changes → diff against previous → index → notify*. A compliance team tracks legislative changes across parliamentary websites, gazette notifications, and regulatory body publications. When a regulator asks "how did you become aware of this change?", the audit trail shows the exact scrape timestamp, what was extracted, and when the notification was sent.
- **Research & Report** — *multi-source scrape and ingest → language-model extraction → index → aggregation queries → generated report*. A policy team briefs a minister on a complex topic by drawing from statistical agencies, government portals, academic repositories, and news sources; every claim in the briefing links back to the specific source that supported it. Corporate equivalents include due-diligence work, ESG reporting, and market-entry research — anywhere a research output informs a consequential decision and may later be challenged by auditors, regulators, or counsel.
- **Screen & Assess** — *ingest entities → search against reference databases → language-model classify and score → generate assessment*. Sanctions screening, KYC/AML compliance, vendor risk assessment, grant or insurance claims assessment. Decisions are legally consequential in both directions: false negatives create liability, false positives cause harm. The audit trail makes every decision reconstructable.
- **Discover & Surface** — *ingest documents → chunk → embed → index → retrieval-augmented queries → conversational answers with citations*. An organisation indexes its briefings, policies, and reference material; a user asks "what has our position been on X over the last 12 months?" and gets an answer that cites the specific documents with dates. The retrieval transform records which chunks were retrieved, their relevance scores, and which ones the language model used in forming the answer — if the answer is challenged, the evidence chain is intact.

### The evidence that the audit discipline holds when the platform is extended

The reasonable question to ask of any forward-looking trajectory is whether the discipline being relied on has held *historically* when the platform was extended, or whether it is simply being asserted to hold *prospectively*. For ELSPETH the evidence is the platform's own extension history.

The language-model transform itself was added as a plugin after the initial engine; it inherits the schema-contract validation, the trust-tier model, the audit hooks, and the checkpoint/resume discipline. The named per-call audit-record gap in §4 is the exception — and the fact that it is *named, scoped, and tracked as a known gap* rather than silently accepted is itself evidence of the discipline at work: where the discipline is incomplete, it is incomplete on the record, not in private. The ChromaDB sink was added similarly and inherits the same machinery. The content-safety screening, the value-transformation primitives, the gate-based routing — all were extensions, all inherited the discipline. The composer is the most recent and most adversarially-tested example; the evaluation in the body of this document is the proof that even a dialogue-driven, language-model-backed extension carries the audit trail end-to-end on runs that execute.

The discipline is not aspirational. It is the operating posture under which every extension to the platform has so far been added, and there is a track record to point to.

### What stays the same as the platform is extended

The audit model. The trust-tier discipline. The schema-contract validation. The checkpoint and resume guarantee. The "every decision traceable to source" principle. Each new capability added must inherit these — and the mechanism for ensuring it does is the same one used today: plugins are system-owned code (developed, tested, and deployed as part of ELSPETH with the same engineering rigour as engine code, not third-party or externally-installed extensions), the schema contracts apply uniformly across new and existing plugins, the audit hooks are required rather than optional, and the trust-tier model dictates where coercion is permitted and where any anomaly must crash.

### Closing — what a sponsorship decision is being asked to weigh

The body of this briefing is the present-state evidence that this discipline holds for the most recently-added capability under adversarial pressure. The platform's own extension history (above) is the evidence that the discipline holds under extension generally. Section C2 above is the trajectory along which it is being extended further. Together they answer the question a sponsorship decision needs to answer: *what does ELSPETH do today, what is it being asked to become, and on what evidence should we believe the trajectory is real?*

---

*End of briefing.*
