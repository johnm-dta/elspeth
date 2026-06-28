# 02 — Personas and Audiences

This document grounds the rest of the redesign in the documented users the
composer is built to serve. Every UI recommendation in this set traces back
to needs identified here.

## The README's two-audience model

ELSPETH explicitly serves two audiences over one runtime substrate:

| Audience | Authoring surface | Why |
|---|---|---|
| **Operators** in sensitive, regulated, transactional, operational, medical, security, or defence-adjacent workflows | Hand-edited YAML | The pipeline can be read, reviewed, versioned, and explained before it runs |
| **Knowledge workers** building document QA, classification, routing, extraction, reporting, or review workflows | Authenticated Web Composer | The LLM builds through audited tools, contracts, validation, preflight checks, and execution evidence rather than emitting unchecked config text |

The composer's primary audience is therefore knowledge workers. The
YAML-author surface remains first-class for skilled operators. Recommendations
in this redesign optimize for the knowledge-worker audience while preserving
clean handoff points where their composed pipeline becomes a reviewable
artifact for a skilled operator (or, more often, for the same person at
review time).

### A non-obvious gap in the README's model

The README's "operator → YAML" mapping assumes operators code. **Linda
Marston** (the compliance-officer persona, P1) is by domain an operator —
her work is in a regulated context, her output will be reviewed — but she
does not code. The composer is her on-ramp to the operator pipeline: she
composes, then hands the resulting YAML to a colleague who reviews and runs
it. This is the strongest argument for the **Save-for-review completion
gesture** documented in [09-completion-gestures.md](09-completion-gestures.md).

## The four documented personas

These personas live in `evals/composer-harness/personas/` and
`evals/2026-05-03-composer/hardmode/personas/`. They were written as test
fixtures for the composer's LLM behaviour and contain rich detail about each
user's cognitive style, vocabulary constraints, knowledge gaps, and stop
conditions. The summaries below are sufficient for design work; consult the
source files for fidelity.

### P1 — Linda Marston (compliance officer)

- **Role:** Senior Compliance Officer at a US mid-sized financial services
  firm. 14 years in GLBA/SOX work.
- **Skill:** Doesn't code. Comfortable in Excel, Outlook, SharePoint.
- **Cognitive style:** Hedge-and-condition driven. States restrictions
  before asks. Risk-averse, polite, persistent.
- **Vocabulary:** Domain-fluent (in scope, controls testing, evidentiary
  record, second-line review). Avoids technical product vocabulary.
- **Knowledge gaps:** Doesn't know what schema / transform / JSONL mean.
  Believes she can specify retention policies in conversation (sometimes
  she can; the system has the right concepts but not always at the right
  surface).
- **Composer journey:** Composes a pipeline she'll **hand off** to a
  colleague who reviews/runs it. Will not click Execute herself. Needs the
  audit story visible during composition because the audit story is her
  job.
- **Completion verb:** *Save & request review.*
- **Audit-readiness panel value:** **Highest.** Without it she cannot
  confirm to herself that what she's composing meets her professional
  standards.

### P2 — Dr. Sarah Okonkwo (academic researcher)

- **Role:** Senior Research Fellow in Applied Sociology. PI on a 4-year
  community-health study with 80,000+ open-ended survey responses.
- **Skill:** Was fluent in R / SPSS in 2019; rusty now. Picks up new tools
  on grad-student recommendation. Reads docs only after the third try.
- **Cognitive style:** Narrative. Tells the story of *why* before stating
  the ask. Frames work in research-question terms. Outcome-oriented.
- **Vocabulary:** Theoretically aware (thematic analysis, axial coding,
  saturation, lived experience). Treats the LLM as a research assistant.
- **Knowledge gaps:** Doesn't know what JSONL is — assumes she can open it
  in Excel. Treats LLM "thematic analysis" as open coding, which the
  product can only partly serve.
- **Composer journey:** Composes, **runs to see results**, iterates on
  categorization, re-runs. Needs results framed narratively — "what does
  this tell us about X" — not as a JSONL file download.
- **Completion verb:** *Run analysis.* Plus narrative result rendering.
- **Audit-readiness panel value:** Medium. She trusts the institution;
  doesn't need to read the audit chain herself.

### P3 — Marcus Chen (marketing ops)

- **Role:** Marketing Operations Manager at a B2B SaaS startup.
- **Skill:** 6 years on Zapier, HubSpot, Salesforce, Airtable. Built a
  GPT-powered Zap last quarter.
- **Cognitive style:** Assertive, opinionated, action-oriented. Knows what
  he wants. Doesn't hedge. Time-pressured.
- **Vocabulary:** Speaks Zapier (trigger, action, webhook, field mapping,
  automation, auto-route). His meanings of "schema" / "API" / "real-time"
  don't always match the product's.
- **Knowledge gaps:** Believes the product fires webhooks per row (it's
  batch); believes the LLM step is a generic GPT block with arbitrary
  if/then prompt logic; believes he can connect to HubSpot/Salesforce
  directly as a source/sink.
- **Composer journey:** Composes and **executes immediately**. Will push
  back hard on refusals. Needs the composer to gracefully reject
  Zapier-shaped impossible asks and translate his vocabulary to the
  product's.
- **Completion verb:** *Execute.* Run and ship.
- **Audit-readiness panel value:** Low. Not in his frame. He'll see it but
  it's not why he came.

### P4 — Dev Patel (senior engineer)

- **Role:** Staff data-platform engineer at a logistics company.
- **Skill:** 11 years of Airflow / dbt / Snowflake. Has used ELSPETH at a
  previous job. Knows the primitives by name.
- **Cognitive style:** Prescriptive. Names components ("CSV source, LLM
  transform with model X, fork on column Y"). Allergic to clarifying
  questions. Skim-reads composer output. Calls out hallucinations.
- **Vocabulary:** Technical-fluent. Uses snake_case identifiers, plugin
  kinds, and ADR numbers directly.
- **Knowledge gaps:** Out of date by ~6 months — may reference plugins
  that have been renamed or removed. Believes she can inline retry
  semantics in transform options when they need a retry block.
- **Composer journey:** Composes to **scaffold YAML quickly**, then
  hand-tunes the YAML in her own environment. Probably bypasses the
  composer entirely for serious work. When she uses it, she wants a fast
  YAML-emitter that pushes back on her stale plugin knowledge.
- **Completion verb:** *Copy YAML.*
- **Audit-readiness panel value:** Low. She'll read the YAML directly.

## Persona × surface decision matrix

This is the central table that justifies the surface-level recommendations.
Each cell answers: "does this surface earn its place for this persona?"

| Surface | Linda (compliance) | Sarah (researcher) | Marcus (ops) | Dev (engineer) |
|---|---|---|---|---|
| Catalog button (as **reference**) | Yes — orientation: "what can the system do?" written in compliance language | Yes — "what can the LLM step do?" | Yes — graceful rejection of Zapier-shaped asks; lists "we don't fire webhooks per row, but here's the audit-export sink" | Yes — currency: "is this plugin still called that this week?" |
| Catalog as **interactive toolkit** | No — she doesn't browse and pick | No — she describes; the LLM picks | No — he gives the LLM the goal | No — she names the plugin she wants directly |
| Spec tab | No | No | No | No (she reads YAML) |
| Graph view | Yes — "is this what I described?" | Yes — verification | Maybe — quick check before Execute | No (reads YAML) |
| YAML view | Yes — but as **export artifact** | Maybe — confirms it ran | No | Yes — primary |
| Runs tab | Yes — confirms audit recorded | Yes — **narrative results** | Yes — operational | No |
| Execute button | No — she hands off | Yes | **Yes — primary** | No |
| Validate button (manual) | No | No | No | No |
| Validation indicator (continuous) | Yes — load-bearing | Yes | Yes | Yes |
| Audit-readiness panel | **Yes — load-bearing** | Yes | Optional | Optional |
| Session sidebar (always on) | No | No | No | No |
| Templates / Example use cases | Yes — domain-shaped exemplars | Yes — research-shaped | Yes — Zap-shaped | No |
| Save-for-review completion | **Yes — primary** | Maybe | No | No |
| Run completion | Maybe | **Yes — primary** | **Yes — primary** | No |
| Export YAML completion | Yes — to share with colleague | Maybe | No | **Yes — primary** |
| Switch-to-guided affordance | Yes — default | Yes — default | After tutorial, she opts out | After tutorial, she opts out |

## Implication: three different "completion" stories

The composer's completion gesture is persona-dependent. The current single
Execute button cleanly serves only Marcus's flow. The recommendation in
[09-completion-gestures.md](09-completion-gestures.md) is a completion bar
that surfaces the right verb for each persona without forcing a mode-pick
beforehand.

## Implication: the audit-readiness panel is for Linda

Sarah and Marcus benefit from seeing it. Dev doesn't need it. **Linda
requires it** to use the tool at all. Without an in-composition view of the
audit story, Linda cannot do her job in the composer — she has to compose,
hand off the YAML, ask her colleague to verify the audit properties, and
wait. That defeats the purpose of giving her the on-ramp.

See [07-audit-readiness-panel.md](07-audit-readiness-panel.md) for the
panel's content design and Linda-vocabulary framing.

## Implication: default mode

Guided default favours Linda (needs structured questions) and Sarah
(under-specifies; needs to be asked). Freeform default favours Marcus
(action-oriented; will state the ask once) and Dev (prescriptive; allergic
to clarifying questions).

The committed call: **default-guided with a persistent per-user opt-out**,
set during the hello-world tutorial. Marcus and Dev opt out once; Linda and
Sarah never need to. See [05-modes-and-opt-out.md](05-modes-and-opt-out.md).

## Source references

Persona source files (in this repo):
- `evals/2026-05-03-composer/hardmode/personas/p1_compliance.md`
- `evals/2026-05-03-composer/hardmode/personas/p2_researcher.md`
- `evals/2026-05-03-composer/hardmode/personas/p3_marketingops.md`
- `evals/composer-harness/personas/p4_adversarial_engineer.md`

README positioning: `README.md` lines 84-91 ("Two first-class paths" table).

Memory entry: `project_composer_personas` (in
`~/.claude/projects/-home-john-elspeth/memory/`).
