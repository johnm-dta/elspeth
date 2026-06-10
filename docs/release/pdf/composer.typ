// composer.typ — ELSPETH Composer Reference (RC-5.2)
// Audience: engineering reviewers, UX reviewers, and operational
// stakeholders evaluating the Composer's authoring UX.
//
// This document compresses docs/composer/ux-redesign-2026-05/* into a
// release-PDF formatted reference. It is the release-set companion to
// docs/release/pdf/architecture.typ: the architecture document
// covers the engine and audit substrate; this document covers the
// authoring surface (Composer + chat protocol) layered above.
//
// Source-of-truth alignment:
//   docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md is
//   the canonical phase tracker. When a phase ships or its scope
//   shifts, update the roadmap FIRST, then carry the change here.
//
//   Status caveat (2026-05-20): the operator confirmed Phases 1–8 of
//   the UX-redesign track are implemented; the roadmap markdown is
//   stale and still labels several of those phases as "Plan
//   reviewed". The status column in §5 below reflects the
//   operator-confirmed current state, not the stale roadmap text.
//   Refreshing the roadmap markdown to match is queued as a
//   companion action.
//
// Visual treatment differs from executive-summary.typ:
//   - draft: false                — design committed; not under review
//   - h1-pagebreak: false         — linear reading, not slide-deck
//   - no cover-hero               — the Composer's defining surface is
//                                   the conversational chat, not a
//                                   diagram. A pill graphic would
//                                   misrepresent the product shape.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as data

#show: document-frame.with(
  title: "ELSPETH Composer",
  subtitle: "RC-5.2 -- " + data.doc-date,
  draft: false,
  h1-pagebreak: false,
)

#cover-page(
  title: "Composer Reference",
  subtitle: "Authoring surface for audit-bearing pipelines.",
  doc-date: data.doc-date,
  version: "RC-5.2",
  author: "John Morrissey, CTO Branch",
  affiliation: "Digital Transformation Agency",
  audience: "Engineering reviewers, UX reviewers, operational stakeholders",
  classification: default-classification,
  status: "Reference — current as of " + data.doc-date,
  distribution: "Public evaluation release copy",
)

#outline(
  title: text(font: font-body, size: size-h1, weight: "bold",
    fill: c-navy, "Contents"),
  indent: auto,
  depth: 2,
)
#pagebreak()

// ---------------------------------------------------------------------------
// 1. Framing
// ---------------------------------------------------------------------------

= Framing

The Composer is ELSPETH's authoring surface for knowledge workers
who need to build audit-bearing pipelines without writing YAML.
It is *not* a generic pipeline IDE: it is optimised for users in
audit-bearing contexts, to produce pipelines that can be reviewed,
executed, and explained.

The Composer's design is documented in detail under
`docs/composer/ux-redesign-2026-05/` (the UX-redesign-2026-05 track),
with the implementation roadmap at `00-implementation-roadmap.md`
and design rationale at `01-design-rationale.md`. This PDF is the
release-formatted reading copy; the markdown set is the working
source.

== The two-audience model

ELSPETH serves two authoring audiences over one runtime substrate:

#data-table(
  columns: 3,
  header: ("Audience", "Surface", "Why this surface"),
  align-rules: (left, left, left),
  ("Operators",
   "Hand-edited YAML",
   "Pipeline can be read, reviewed, versioned, and explained before it runs"),
  ("Knowledge workers",
   "Authenticated Web Composer",
   "LLM builds through audited tools, contracts, validation, preflight checks, and execution evidence rather than emitting unchecked config text"),
)

The Composer's primary audience is knowledge workers. The YAML-author
surface remains first-class for skilled operators.

#callout(kind: "note", title: "A non-obvious gap")[
  The "operator → YAML" mapping assumes operators code. Compliance
  officers in regulated contexts are by domain operators but do not
  code; the Composer is their on-ramp. They compose, then hand the
  resulting YAML to a colleague who reviews and runs it. This is
  the strongest argument for the *Save-for-review* completion
  gesture.
]

// ---------------------------------------------------------------------------
// 2. Personas
// ---------------------------------------------------------------------------

= Personas

The Composer's UX decisions are grounded in four documented personas
that live as eval fixtures at `evals/composer-harness/personas/` and
`evals/2026-05-03-composer/hardmode/personas/`. Each persona has a
distinct completion gesture and a distinct relationship to the audit
trail.

#data-table(
  columns: 4,
  header: ("Persona", "Role", "Completion gesture", "Audit-panel value"),
  align-rules: (left, left, left, left),
  ("Linda",   "Compliance officer", "Save & request review", "Highest"),
  ("Sarah",   "Academic researcher", "Run analysis (narrative results)",
   "Medium"),
  ("Marcus",  "Marketing ops",      "Execute immediately",   "Low"),
  ("Dev",     "Senior engineer",    "Copy YAML",             "Low"),
)

#callout(kind: "note", title: "Why four personas, not one")[
  No single completion gesture serves all four. Linda will not
  click Execute; Marcus will not save and wait; Dev wants the YAML
  and leaves. The completion bar surfaces the right verb for each
  persona without forcing a mode-pick beforehand. See
  `09-completion-gestures.md` for the design.
]

== The audit-readiness panel is for Linda

Sarah and Marcus benefit from seeing it; Dev doesn't need it. Linda
*requires* it to use the tool at all. Without an in-composition view
of the audit story, Linda cannot do her job in the Composer — she has
to compose, hand off the YAML, ask her colleague to verify the audit
properties, and wait. That defeats the purpose of giving her the
on-ramp. See `07-audit-readiness-panel.md`.

// ---------------------------------------------------------------------------
// 3. Default mode and first-run
// ---------------------------------------------------------------------------

= Default mode and first-run

== Default mode: guided, with opt-out

The committed call is *default-guided with a persistent per-user
opt-out*, set during the hello-world tutorial. Guided default
favours Linda (needs structured questions) and Sarah
(under-specifies; needs to be asked). Freeform default favours
Marcus (action-oriented; will state the ask once) and Dev
(prescriptive; allergic to clarifying questions). Marcus and Dev
opt out once; Linda and Sarah never need to.

The opt-out is reachable from two surfaces: in settings, and inline
during the first conversation ("disable; I'll go looking for it"). It
is per-user, persistent, and backed by a `user_preferences` table
introduced in Phase 1A.

== First-run: hello-world tutorial

The first session for a new user is a hello-world tutorial that
teaches source → transform → sink, demonstrates the
dynamic-source-from-chat feature, surfaces the audit trail, and flows
into the default-mode choice. It is *not* an interstitial; it is the
first conversation, structured. See `04-first-run-tutorial.md` and
`21-phase-4-hello-world-tutorial.md`.

== Dynamic source from chat

For simple inputs (a URL, a sentence, one record), the LLM creates a
one-row dynamic source from the chat text itself — no CSV upload.
This is the basis of the hello-world's source step and the smoothest
on-ramp for first-time users. The audit recorder treats the inline
content identically to any other source (`source_data_hash` is
recorded). See `17-phase-5a-dynamic-source-from-chat.md`.

// ---------------------------------------------------------------------------
// 4. Worked example: a composer exchange
// ---------------------------------------------------------------------------

= Worked example: a composer exchange

This is a real exchange from staging, session
`15216094-9faf-46e2-854e-e69ad168887b`, captured on 2026-05-19. The
user (a `dta_user` test principal) asked the Composer to build a
pipeline they did not know how to express in YAML. The exchange
illustrates how the tool-call surface, the continuous validator, and
the audit recorder cooperate.

== Turn 1 — user

#block(inset: (left: sp-3), stroke: (left: 2pt + c-action))[
  "Please go to the following web pages https:\/\/www.finance.gov.au,
  https:\/\/www.defence.gov.au, and https:\/\/www.dta.gov.au and read
  the HTML and have an LLM look up the colours for each page. Remove
  the HTML and then save the rest to a json file."
]

#callout(kind: "note", title: "What the Composer is doing")[
  The user has stated an outcome, not a pipeline. Naming the
  source, the transforms, the routing, or the sink is the
  Composer's job — and so is recognising that the URLs are
  *content* that the LLM can convert into a one-row dynamic source,
  not an upload prompt. The user does not need to know what a
  source plugin is.
]

== Turn 2 — assistant (parallel discovery)

The Composer's first move is a discovery turn — six tool calls
issued in parallel:

```text
list_secret_refs()
list_models(provider="openrouter/", limit=20)
get_plugin_schema(plugin_type="transform", name="web_scrape")
get_plugin_schema(plugin_type="transform", name="llm")
get_plugin_schema(plugin_type="transform", name="field_mapper")
get_plugin_schema(plugin_type="sink",      name="json")
```

#callout(kind: "note", title: "What the Composer is doing")[
  Discovery is structured, not free-form. The Composer learns
  which secrets are available, which models the operator has
  enabled, and the *exact* schema of each plugin it intends to
  use. Free-form recall of plugin options is how composers
  hallucinate; the schema fetch is how this Composer doesn't.
]

== Turn 3 — assistant (set_pipeline)

The Composer issues a single `set_pipeline` call describing the
full DAG: an inline-blob `text` source (the three URLs become a
77-byte inline blob), three transforms (`web_scrape` → `llm` →
`field_mapper`), and a `json` sink:

```text
source       text                  inline_blob: web_pages.txt (77 bytes)
fetch_pages  web_scrape            content_field: content
                                   http.abuse_contact: compliance@example.com
color_lookup llm                   provider: openrouter
                                   model:    anthropic/claude-sonnet-4.6
                                   prompt:   "identify main visible colours…"
remove_html  field_mapper          select_only: true
                                   mapping: (url, color_summary,
                                             content_fingerprint)
final_rows   json (sink)           path: outputs/web_page_colors.json
                                   collision_policy: auto_increment
```

The continuous validator runs and returns:

#callout(kind: "error", title: "Validation: high severity")[
  `web_scrape.http.abuse_contact` has domain `example.com` —
  RFC 2606/6761 reserves `example.com` for documentation /
  test use, so the value is not deliverable to anyone and would
  ship as a fabricated identity in the HTTP header to the
  scraped host. Set `abuse_contact` to an operator-supplied or
  deployment-identity-sourced email.
]

#callout(kind: "advisory", title: "What the Composer is doing")[
  The validator catches *real-world correctness* at compose time,
  not at run time. The pipeline would have technically run with
  `compliance@example.com`, but it would have transmitted a
  fabricated identity to three live government servers. The
  Composer treats this as a high-severity blocker because the
  audit trail would otherwise record a contact email the
  operator never authorised.
]

== Turn 4 — assistant (surface the blocker)

#block(inset: (left: sp-3), stroke: (left: 2pt + c-supported))[
  "I'm blocked on one required scraping header: the web fetch step
  needs a real abuse-report contact email, and the value I used was
  rejected because `example.com` is not deliverable.

  Send me the abuse contact email to use in the outbound HTTP
  header, and I'll finish the workflow."
]

#callout(kind: "note", title: "What the Composer is doing")[
  The Composer does not fabricate. It surfaces the missing
  fact to the user in domain language ("abuse-report contact
  email"), explains *why* the validator rejected its first
  guess, and stops. This is the Linda-readable failure mode:
  no silent default, no plausible-looking substitute.
]

== Turn 5 — user

#block(inset: (left: sp-3), stroke: (left: 2pt + c-action))[
  "great, john.morrissey\@dta.gov.au"
]

== Turn 6 — assistant (patch, preview, summarise)

The Composer issues a targeted `patch_node_options` (just the one
field, not a full pipeline rewrite), then a `preview_pipeline`
runtime preflight:

```text
patch_node_options(node_id="fetch_pages", patch={
    "http": {
        "abuse_contact":  "john.morrissey@dta.gov.au",
        "allowed_hosts":  "public_only",
        "scraping_reason": "Extract page HTML and identify
                            dominant colors for each government website"
    }
})
preview_pipeline()
```

Validator response: `is_valid: true`, the high-severity error is
recorded as `resolved_errors` in the validation delta, and the
pipeline version increments from 2 to 3.

The Composer's final summary turn (paraphrased from the live
transcript):

#block(inset: (left: sp-3), stroke: (left: 2pt + c-supported))[
  "Done — I updated the workflow and it now validates cleanly. \
  *What it does:* Reads the three government page URLs from the
  input file; fetches each page's HTML with `web_scrape`; sends
  the fetched content to an LLM to identify the main colours
  used on each page; removes the raw HTML by keeping only the
  cleaned fields; saves the result to
  `outputs/web_page_colors.json`. \
  *Data-loss / failure handling:* input rows that fail source
  validation are discarded; fetch / LLM / field-mapping / sink
  failures are discarded; dropped rows are only recorded in the
  audit trail. \
  *Decisions I made on your behalf:* `format = raw` (preserve
  HTML for the LLM), `allowed_hosts = public_only`,
  `temperature = 0` and `pool_size = 1` (deterministic),
  `collision_policy = auto_increment` (re-runs don't
  overwrite)."
]

== The audit trail this exchange produced

While the user saw four assistant turns, the audit recorder
captured nine `ComposerLLMCall` records:

#data-table(
  columns: 4,
  header: ("Turn", "Model", "Total tokens", "Cost (USD)"),
  align-rules: (left, left, right, right),
  ("Discovery + first build attempt",
   "gpt-5.4-mini",   "51,064", "\$0.0388"),
  ("Validator-driven reflection",
   "gpt-5.4-mini",   "66,428", "\$0.0184"),
  ("Surface blocker (formulate question)",
   "gpt-5.4-mini",   "66,919", "\$0.0057"),
  ("Surface blocker (compose reply)",
   "gpt-5.4-mini",   "67,365", "\$0.0058"),
  ("Re-state-fetch after user reply",
   "gpt-5.4-mini",   "52,167", "\$0.0057"),
  ("Schema-recheck for patch",
   "gpt-5.4-mini",   "53,453", "\$0.0053"),
  ("Patch + preview",
   "gpt-5.4-mini",   "56,519", "\$0.0067"),
  ("Validator interpretation",
   "gpt-5.4-mini",   "56,737", "\$0.0046"),
  ("Final summary",
   "gpt-5.4-mini",   "58,311", "\$0.0068"),
)

#callout(kind: "note", title: "What is recorded for each call")[
  Each `ComposerLLMCall` row carries the model requested and
  returned, prompt / completion / cached / reasoning token
  counts, latency, provider request id, the
  RFC-8785-canonicalised hash of the message stream and the
  tools spec, temperature, seed, cost, and any error class /
  message. The Composer session is, by construction, a
  recoverable authoring transcript.
]

The total spend on this exchange was approximately \$0.097 USD across
nine model calls and roughly 25 seconds of wall-clock LLM latency.
The user's first turn was a sentence; the final pipeline validates
clean, is preview-confirmed, and is ready to execute.

// ---------------------------------------------------------------------------
// 5. IA decisions
// ---------------------------------------------------------------------------

= Information architecture

The redesign treats the Composer's IA as load-bearing for the
product's positioning rather than as an inheritance from "what we
built first." The headline IA decisions are below; full rationale is
in `01-design-rationale.md` and `03-target-information-architecture.md`.

== Kept

#data-table(
  columns: 2,
  header: ("Surface", "Why kept"),
  align-rules: (left, left),
  ("Catalog button (as reference)",
   "Orientation: 'what can the system do?' Valuable across all four personas; the toolkit framing was the problem, not the underlying API"),
  ("Execute button",
   "Serves the second arm of the use case — Marcus's ad-hoc compose-and-run journey"),
  ("Runs tab",
   "Audit confirmation (Linda) and narrative result rendering (Sarah)"),
  ("Validation indicator (continuous)",
   "Load-bearing for all four personas"),
)

== Killed

#data-table(
  columns: 2,
  header: ("Surface", "Why killed"),
  align-rules: (left, left),
  ("Spec tab",
   "Exposes the engine's internal composition tree; serves no documented persona — only the engine team during debugging"),
  ("Manual Validate button",
   "Validation is continuous; the indicator dot does the work. The button is theatre"),
  ("Always-on session sidebar",
   "No persona opens the Composer to switch between pipelines they're building. A header switcher suffices"),
)

== Added

#data-table(
  columns: 2,
  header: ("Surface", "Why added"),
  align-rules: (left, left),
  ("Persistent audit-readiness panel",
   "ELSPETH's defining feature was previously invisible during composition"),
  ("Hello-world tutorial",
   "First-run experience that teaches the model, surfaces the audit trail, and sets the default-mode preference"),
  ("Completion bar",
   "Surfaces the persona-appropriate verb without forcing a mode-pick"),
  ("Catalog drawer reshape (browse / search / read / learn)",
   "Reframed from interactive toolkit to searchable system-capability reference"),
)

// ---------------------------------------------------------------------------
// 6. Implementation roadmap
// ---------------------------------------------------------------------------

= Implementation roadmap

The UX-redesign track has nine phases. As of #data.doc-date, the
operator has confirmed that Phases 1 to 8 are implemented; Phase 9
(the migration runner) is the remaining gate before any real-user
production deploy. The canonical roadmap markdown is at
`docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md`;
the table below reflects the operator-confirmed current state.

#data-table(
  columns: 3,
  header: ("Phase", "Subject", "Status"),
  align-rules: (left, left, left),
  ("1A",
   "Backend: user_preferences table + preferences API",
   "Shipped"),
  ("1B",
   "Frontend: store, opt-out surfaces, banner, smoke",
   "Shipped"),
  ("2A / 2B / 2C",
   "Audit-readiness panel (backend, frontend, integration)",
   "Shipped"),
  ("3A",
   "IA cleanup — removals (Spec / Validate / sidebar)",
   "Shipped"),
  ("3B",
   "IA cleanup — side-rail additions (session dropdown, command palette)",
   "Shipped"),
  ("4",
   "Hello-world tutorial",
   "Shipped"),
  ("5a",
   "Dynamic-source-from-chat",
   "Shipped"),
  ("5b",
   "Surface-the-LLM's-interpretation",
   "Shipped"),
  ("6A / 6B",
   "Completion gestures (Save-for-review, run-result rendering)",
   "Shipped"),
  ("7A / 7B / 7C",
   "Catalog reshape (backend, frontend, integration)",
   "Shipped"),
  ("8",
   "Polish + telemetry",
   "Shipped"),
  ("9",
   "Migration runner + caretaker-logic activation",
   "Plan reviewed; J1 verdict approved 2026-05-16 (per-table preserve-on-recreate, SQLite-only)"),
)

#callout(kind: "advisory", title: "Remaining ship gate")[
  Phase 9 (migration runner) is necessary before any real-user
  production deploy of any schema-adding phase. The staging
  deploy at `elspeth.foundryside.dev` runs on Phases 1–8 with the
  operator-managed "delete the old DB" migration policy in effect;
  production-class deploys are blocked on Phase 9.
]

== Per-step guided chat — Phase A

The per-step guided chat is a separate track that delivered the
slice-by-slice conversational interaction model the redesign assumes.
Phase A (slices 1 to 6 plus the slice-5.1 audit-persistence fix)
shipped on the `feat/composer-per-step-chat` branch; downstream phases
(A.5, B, C) remain per the plan. See the per-step chat plan under
the composer-pack docs.

// ---------------------------------------------------------------------------
// 7. The chat protocol
// ---------------------------------------------------------------------------

= The chat protocol

The Composer's runtime is a conversational protocol between the user,
the LLM, and a set of audited composer tools. The LLM does not emit
free-text YAML; every change to the in-progress pipeline goes through
a tool call that the audit recorder captures.

== Composer tool surface

The composer tools are exposed via the `elspeth-composer` MCP server
(shipped RC-5.0). The headline tools, grouped by function:

#data-table(
  columns: 2,
  header: ("Group", "Tools"),
  align-rules: (left, left),
  ("Session lifecycle",
   "new_session, load_session, save_session, list_sessions, delete_session"),
  ("Pipeline shape",
   "set_pipeline, get_pipeline_state, set_source, clear_source, upsert_node, remove_node, upsert_edge, remove_edge, set_output, remove_output"),
  ("Node config",
   "patch_source_options, patch_node_options, patch_output_options, set_metadata"),
  ("Catalog / introspection",
   "list_sources, list_transforms, list_sinks, list_models, list_recipes, get_plugin_schema, get_plugin_assistance, get_expression_grammar"),
  ("Validation / preflight",
   "preview_pipeline, diff_pipeline, generate_yaml, explain_validation_error, get_audit_info"),
)

Every tool call is audit-recorded. The composer's audit trail is a
peer of the engine's audit trail: a Composer session is itself an
auditable artefact.

== Why "tools" and not "free-form YAML"

The composer is positioned for users in audit-bearing contexts. A
free-form YAML emitter — even from a capable LLM — produces config
text that has to be re-validated, re-typed, and re-explained. The
tool-call surface lets the LLM compose against the *same* schema the
engine consumes, with the *same* validation pipeline. Every action
is auditable because every action is a structured event, not a string.

// ---------------------------------------------------------------------------
// 8. Audit surfaces inside the Composer
// ---------------------------------------------------------------------------

= Audit surfaces inside the Composer

The Composer's defining feature — that pipelines composed here are
audit-bearing by construction — is surfaced through three Composer
surfaces:

#data-table(
  columns: 2,
  header: ("Surface", "What it shows"),
  align-rules: (left, left),
  ("Audit-readiness panel",
   "Live readiness of the in-progress pipeline. Trust-tier checks, retention defaults, schema-contract coverage, sink-routing completeness. The compliance officer's primary surface."),
  ("Composer audit info tool",
   "Inspect the audit recording shape for any node or edge. Used by the LLM (and by reviewers) to answer 'what gets recorded if this runs?' without running the pipeline."),
  ("ComposerLLMCall audit channel",
   "Every LLM call the Composer makes is recorded — model, prompt, tool calls, latency, classification. The Composer session is itself a recoverable, replayable run."),
)

#callout(kind: "note", title: "The session is the artefact")[
  A Composer session is not ephemeral chat. It is a recoverable
  authoring transcript that — together with the pipeline YAML it
  produced — is the full provenance of the artefact. Reviewing a
  pipeline a colleague composed means reading the session, not just
  the YAML.
]

// ---------------------------------------------------------------------------
// 9. What is in scope; what is not
// ---------------------------------------------------------------------------

= Scope discipline

== In scope for this redesign

- The Web Composer's information architecture, default mode, first-run
  experience, completion gestures, and audit-readiness surfacing.
- The composer-tools API where the IA recommendations touch it.
- Hello-world tutorial and dynamic-source-from-chat.
- Catalog reshape from interactive toolkit to searchable reference.

== Out of scope

#data-table(
  columns: 2,
  header: ("Surface", "Why out of scope"),
  align-rules: (left, left),
  ("Hand-edited YAML path",
   "First-class parallel surface for operators; not the Composer's concern"),
  ("Chat protocol, session-state model, composer-tools API",
   "Recommendations note the touch-point but defer implementation strategy to follow-ups"),
  ("Engine, plugin system, audit recorder, BFF backend",
   "Owned by the architecture track; see docs/release/pdf/architecture.typ"),
  ("Mobile / touch-first redesign",
   "The Composer is a desktop authoring surface; responsive behaviour is in scope but a phone-first rethink is not"),
)

// ---------------------------------------------------------------------------
// 10. Where to read more
// ---------------------------------------------------------------------------

= Reading on

#data-table(
  columns: 2,
  header: ("Document", "Subject"),
  align-rules: (left, left),
  ("docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md",
   "Phase status, ship sequencing, open-question adjudication"),
  ("docs/composer/ux-redesign-2026-05/01-design-rationale.md",
   "Why the redesign exists; what changed during review; framing"),
  ("docs/composer/ux-redesign-2026-05/02-personas-and-audiences.md",
   "The four documented personas and the surface decision matrix"),
  ("docs/composer/ux-redesign-2026-05/04-first-run-tutorial.md",
   "Hello-world tutorial design (three-beat experience)"),
  ("docs/composer/ux-redesign-2026-05/09-completion-gestures.md",
   "Completion-bar design and persona-appropriate verbs"),
  ("docs/composer/ux-redesign-2026-05/14-phase-2-audit-readiness-panel.md",
   "Audit-readiness panel content design and Linda-vocabulary framing"),
  ("docs/composer/ux-redesign-2026-05/16-phase-7-catalog-reshape.md",
   "Catalog drawer reshape: interactive toolkit → searchable reference"),
  ("docs/composer/ux-redesign-2026-05/17-phase-5a-dynamic-source-from-chat.md",
   "Dynamic-source-from-chat design and audit-recorder verification"),
  ("docs/composer/ux-redesign-2026-05/18-phase-5b-surface-llm-interpretation.md",
   "Surface-the-LLM's-interpretation review surface"),
  ("docs/composer/ux-redesign-2026-05/22-phase-9-migration-runner.md",
   "Migration runner plan (per-table preserve-on-recreate, SQLite-only)"),
  ("evals/composer-harness/personas/, evals/2026-05-03-composer/hardmode/personas/",
   "Persona source files (eval fixtures)"),
)
