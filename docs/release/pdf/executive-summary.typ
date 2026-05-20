// executive-summary.typ — ELSPETH Executive Summary (RC-5.2)
// Audience: senior public-service officials evaluating ELSPETH.
//
// This document is the DTA-SES internal brief on ELSPETH at RC-5.2.
// It is audience-tiered: docs/release/executive-summary.md serves
// the external APS-evaluator audience and is structured around the
// matters that audience needs to raise with the project; this brief
// serves the internal DTA-SES audience and surfaces claims (interim
// ATO scope, pilot deployment volume, forward priorities) that the
// external brief deliberately handles differently.
// Keep the two documents FACT-aligned: when a substantive fact
// changes (a new interim ATO, a pilot scope shift, a residual-risk
// reassessment), update BOTH and review whether the claim belongs
// in each audience tier.
// DRAFT banner is required by the source's load-bearing draft marker.

#import "tokens.typ": *
#import "theme.typ": *
#import "data.typ" as data
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart

#show: document-frame.with(
  title: "ELSPETH Executive Summary",
  subtitle: "RC-5.2 -- " + data.doc-date,
  draft: true,
  draft-date: data.draft-date,
)

// `classification: default-classification` reads the module-level
// constant from theme.typ so the cover's bibliographic Classification
// row cannot drift from the running classification band's value.
#cover-page(
  title: "Executive Summary",
  subtitle: "An auditable data-processing platform for high-assurance work.",
  doc-date: data.doc-date,
  version: "RC-5.2",
  author: "John Morrissey, CTO Branch",
  affiliation: "Digital Transformation Agency",
  audience: "DTA SES and other stakeholders",
  classification: default-classification,
  status: "DRAFT — awaiting review",
  distribution: "Internal — Digital Transformation Agency only",
  draft: true,
  draft-date: data.draft-date,
  hero: cover-hero-sda(),
)

// Table of Contents — second page, before the body content begins.
#outline(
  title: text(font: font-body, size: size-h1, weight: "bold",
    fill: c-navy, "Contents"),
  indent: auto,
  depth: 2,
)
#pagebreak()

// ---------------------------------------------------------------------------
// 1. At a glance dashboard
// ---------------------------------------------------------------------------

= At a glance

A five-card snapshot of build status as of #data.draft-date.

// Page-2 dashboard — every card measured at its actual column width,
// then all five rendered at the global maximum natural height so the
// two rows align visually with no clipping. No magic numbers: when
// content changes, the height self-adjusts. Numeric values come from
// data.typ; do not hand-edit them here.
//
// The "Contributors" card uses the c-notyet stripe to signal the
// continuity risk; `value-colour: c-action` keeps the headline "1"
// in the action-blue typographic register so the number is not
// visually demoted by the risk-coloured stripe.

// Stripe-colour rotation across the dashboard expresses more of the
// DTA palette than the previous teal-dominant treatment. Each card's
// stripe encodes a semantic register:
//   c-action     — informational fact (period, volume)
//   c-supported  — positive outcome (tests pass, ATO received)
//   c-notyet     — continuity / risk signal (kept grey to match the
//                  residual-risk cards)
//   c-navy-soft  — structural fact (deployment, pilot status)
// The `value-colour` override on the Contributors card keeps the
// headline "1" in the action-blue typographic register so the
// numeric value isn't visually demoted by the risk-coloured stripe.
#let dash-cards-3 = (
  (label: "Build days", value: str(data.calendar-days),
   sub: data.project-start + " -- " + data.project-end,
   colour: c-action),
  (label: "Tests in automated suite", value: data.test-count-card,
   sub: "Approx. framework + composer; passing on RC-5.2",
   colour: c-supported),
  (label: "Contributors", value: "1",
   sub: "Single developer. Continuity risk is material.",
   colour: c-notyet, value-colour: c-action),
)

// `~2,200` here is the short visual form of `data.pilot-rows-prose`;
// the long form is used in body prose. Kept short here so the
// dashboard tile stays one line.
// FOLLOW-UP: introduce a `pilot-rows-short` constant in data.typ so
// this tile and the one other hardcoded reference (deployment-
// readiness table) source from a single constant rather than two
// duplicated literals.
#let dash-cards-2 = (
  (label: "Production deployments", value: "Pilot (RC-3)",
   sub: "RC-3 deployed in orchestration-only mode; pilot evaluation processed ~2,200 rows.",
   colour: c-navy-soft),
  (label: "Independent assurance", value: "Interim ATO",
   sub: "Received for the orchestration component only.",
   colour: c-supported),
)

#layout(size => {
  let g = sp-2
  let col-w-3 = (size.width - g * 2) / 3
  let col-w-2 = (size.width - g) / 2

  let measure-card-at = (c, w) => measure(
    block(width: w,
      metric-card(c.label, c.value, sub: c.sub, colour: c.colour))
  ).height

  let heights-3 = dash-cards-3.map(c => measure-card-at(c, col-w-3))
  let heights-2 = dash-cards-2.map(c => measure-card-at(c, col-w-2))
  let max-h = calc.max(..heights-3, ..heights-2)

  // Stack the two rows with explicit `spacing: g` so the vertical
  // gap between the rows is exactly one column-gutter wide. Without
  // stack, Typst's default `par(spacing:)` would add ~1.0em of block
  // spacing on top of any explicit v(g), making the vertical gap
  // noticeably larger than the horizontal column gutter.
  stack(dir: ttb, spacing: g,
    grid(columns: (1fr,) * 3, gutter: g,
      ..dash-cards-3.map(c => metric-card(c.label, c.value, sub: c.sub,
        colour: c.colour, height: max-h))),
    grid(columns: (1fr,) * 2, gutter: g,
      ..dash-cards-2.map(c => metric-card(c.label, c.value, sub: c.sub,
        colour: c.colour, height: max-h))),
  )
})

#v(sp-3)

#callout(kind: "success",
  title: "Current version: RC-5.2 — what's new since the RC-3 pilot")[
  - #strong[Web Composer authoring surface] — chat-led, LLM-assisted
    pipeline authoring for non-engineering operators, with a visual
    treatment consistent with the Australian Government Design System
    (not assessed for AGDS conformance).
  - #strong[Plugin and integration expansion] — Microsoft Dataverse,
    ChromaDB, Azure OpenAI, OpenRouter, Azure Blob Storage, and a
    web-research pipeline.
  - #strong[Engine hardening] — fork / join / coalesce DAG execution,
    transform-invariant migration, and Tier-1 audit-integrity guards.

  For engineering provenance, see the companion Progress and
  Velocity reports.
]

#v(sp-4)

#callout(kind: "advisory", title: "Document status")[
  Draft for review. Comments and corrections welcome.
]

// ---------------------------------------------------------------------------
// 2. What ELSPETH is, in one paragraph (with diagram)
// ---------------------------------------------------------------------------

= What ELSPETH is

ELSPETH is a framework for building auditable data-processing pipelines
— workflows where data is read from a source system, structured logic
is applied (which may include calls to language models), and outputs
are produced for downstream consumers. The design constraint that
distinguishes it from general-purpose data tooling is that #strong[every
decision the system makes must be reconstructible from a permanent
audit record]. Given any output, the system can prove which source row
produced it, which configuration was active, which version of which
model returned which response, and what controls were applied at each
step. "I don't know what happened" is treated by the design as an
unacceptable answer for any output the platform produces.

#v(sp-3)

// Native-Typst diagram (cetz). Replaces the previous PNG embed.
// Migration rationale recorded in theme.typ next to the function
// definition. cetz output participates in the tag tree as PDF
// vector ops; the figure caption carries the accessible description
// (same role the old `image(alt:)` string played). For full screen-
// reader fidelity the caption text below mirrors the original alt.
#figure(
  pdf.artifact(align(center, diagram-sda-flow())),
  caption: [Sense / Decide / Act pipeline with the audit trail as a
    parallel write target. An external source system (Dataverse, CSV,
    JSON, blob, queue, or API) feeds Sense, which loads and validates.
    Sense passes to Decide, which applies transforms and gates using
    rules, models, or LLM calls. Decide passes to Act, which writes
    to sinks. Output flows to a downstream consumer. Every stage
    records to the Audit trail (a Landscape database storing hashes,
    lineage, and provenance) before the operation is confirmed
    complete.],
)

// ---------------------------------------------------------------------------
// 3. What has been shipped (capability summary)
// ---------------------------------------------------------------------------

= What has been shipped

The following capabilities are present, tested, and documented in the
current release. Detailed engineering provenance is in the companion
#emph[Progress] document.

#stack(spacing: sp-1,
  callout(title: "Auditable pipeline engine")[
    Source / decide / act pipelines with full graph topology
    (parallel paths, branching, joining). All operations recorded
    before completion; checkpoint/resume after interruption.
  ],
  callout(title: "Tamper-evident records")[
    Every output is hashed using a cryptographic standard (SHA-256
    over RFC 8785 canonical form). Any undetected change to a record
    changes its hash.
  ],
  callout(title: "Three-tier trust model")[
    The audit database is treated as fully trusted (crash on anomaly).
    Pipeline-internal data is type-validated. External data is
    validated at ingestion and quarantined on failure. The boundaries
    are enforced by code-review rules and automated checks.
  ],
  callout(title: "Web authoring interface (Composer)")[
    A chat-led authoring surface for non-engineering operators to
    build pipelines. The visual treatment is consistent with the
    #strong[Australian Government Design System (AGDS)], though it
    has not been assessed for AGDS conformance. Accessibility
    features include skip-to-content, reduced-motion support, and
    screen-reader-safe status indicators.
  ],
  callout(title: "Three authentication providers")[
    Local username/password (for development and air-gapped
    deployments), OpenID Connect (for federated identity), and
    Microsoft Entra ID (with tenant validation and group claims).
  ],
  callout(title: "Secret-reference handling")[
    Credentials are referenced by name and resolved at run time from
    an Azure Key Vault or environment variable; the secret value
    never appears in pipeline configuration. Resolution is recorded
    in the audit trail by cryptographic fingerprint, not by value.
  ],
  callout(title: "Integration coverage")[
    Microsoft Dataverse (read and upsert), ChromaDB (vector storage
    for retrieval-augmented generation), Azure OpenAI, OpenRouter,
    Azure Blob Storage, and CSV/JSON sources and sinks. Web-scraping
    transform with controls against server-side-request-forgery
    attacks.
  ],
)

// ---------------------------------------------------------------------------
// 4. Assurance posture (split panel)
// ---------------------------------------------------------------------------

= Assurance posture

// Both panels rendered at the same height so the split-panel reads
// as visually balanced. Same `layout + measure + max-h` pattern as
// the page-2 dashboard.
#let posture-provides = block(
  width: 100%,
  inset: sp-3,
  fill: c-panel,
  radius: 3pt,
  stroke: (left: 3pt + c-supported),
  {
    text(size: size-eyebrow, fill: c-supported, weight: "bold",
      tracking: 1pt, upper("What the design provides"))
    v(sp-2)
    list(
      tight: false,
      spacing: sp-2,
      [#strong[Audit-first writes.] Every operation writes to the
       audit trail before the operation is confirmed complete. If
       the audit write fails, the operation fails — there is no
       "best effort" audit path.],
      [#strong[Lineage queries.] Given any output, the audit trail
       can return the source row, the configuration version, the
       model and prompt used (if any), the input and output of each
       transformation, and the principal that authored the
       pipeline.],
      [#strong[Deliberate failure handling.] The system is designed
       to crash rather than continue when it detects internal
       inconsistency in its own data. This is by design: silent
       recovery from corruption is treated as a more dangerous
       failure mode than visible crash.],
      [#strong[Quarantine, not silent skip.] External input that
       fails validation is recorded as quarantined, not dropped.
       The audit trail of "row 42 quarantined because field X was
       malformed" is itself a valid outcome.],
      [#strong[Trust-boundary enforcement.] Automated code analysis
       prevents defensive patterns (`hasattr`, broad exception
       catches, `.get()` with default) on data the system itself
       produced — these are reserved for boundaries where external
       data enters the system.],
      [#strong[Interim Authority to Operate (ATO).] An interim ATO
       has been received for the orchestration component, granting
       use within that scope. The component has been operated under
       the interim ATO in a real-world pilot of
       #data.pilot-rows-prose rows, each accompanied by an audit
       record of the kind described above. Broader-scope assurance
       remains outstanding (see right-hand column).],
    )
  },
)

#let posture-not-yet = block(
  width: 100%,
  inset: sp-3,
  fill: c-panel,
  radius: 3pt,
  stroke: (left: 3pt + c-notyet),
  {
    text(size: size-eyebrow, fill: c-notyet, weight: "bold",
      tracking: 1pt, upper("What the design does NOT yet provide"))
    v(sp-2)
    list(
      tight: false,
      spacing: sp-2,
      [#strong[Independent assurance is scope-limited.] An interim
       authority to operate has been received for the orchestration
       component only. Outside that scope: no IRAP (Information
       Security Registered Assessors Program) assessment, no
       DTA/AGDS conformance review beyond visual styling, and no
       independent penetration test. The audit-trail and trust-tier
       claims are the #emph[designed] behaviour and are exercised
       by an automated test suite of #data.test-count-prose tests;
       outside the orchestration-component interim ATO they have
       not been independently certified.],
      [#strong[No formal mapping] to the Protective Security Policy
       Framework (PSPF), the Information Security Manual (ISM), the
       Essential Eight, or the Digital Service Standard. The
       platform is designed to #emph[support] these obligations —
       the audit trail provides evidence relevant to several ISM
       controls (system event logging, access control logging,
       change management) — but the mapping has not been formally
       compiled.],
    )
  },
)

#layout(size => {
  let g = sp-3
  let col-w = (size.width - g) / 2
  let h-provides = measure(block(width: col-w, posture-provides)).height
  let h-not-yet = measure(block(width: col-w, posture-not-yet)).height
  let max-h = calc.max(h-provides, h-not-yet)
  grid(
    columns: (1fr, 1fr),
    column-gutter: g,
    block(width: 100%, height: max-h, posture-provides),
    block(width: 100%, height: max-h, posture-not-yet),
  )
})

// ---------------------------------------------------------------------------
// 5. Trust-tier diagram
// ---------------------------------------------------------------------------

= Three-tier trust model

The boundaries between tiers carry explicit handling contracts.
Reading from external-input to audit-store:

// Native-Typst diagram (cetz). The accessible description lives in
// the figure caption, mirroring the prose the previous `image(alt:)`
// string carried.
#figure(
  pdf.artifact(align(center, diagram-trust-tiers())),
  caption: [Three-tier trust model — the contract at each boundary.
    Tier 3 (External, zero trust) holds source plugins (CSV, JSON,
    Dataverse, Blob, Web); the framework validates, coerces, and
    quarantines, recording absence as None. The boundary into Tier 2
    enforces validate, coerce, quarantine. Tier 2 (Pipeline, elevated
    trust) holds transforms, gates, and aggregations on type-safe
    values; no coercion at this tier — operations are wrapped instead.
    The boundary into Tier 1 reads straight and writes atomically.
    Tier 1 (Our data, full trust) holds the Landscape audit DB,
    checkpoint state, and hashes; crashes on any anomaly and stays
    pristine at all times. A read-guard back-arrow (TIER_1_ERRORS)
    sits at the boundary back into Tier 2.],
)

// ---------------------------------------------------------------------------
// 6. Deployment readiness scorecard
// ---------------------------------------------------------------------------

= Deployment readiness

// Refactored to use `data-table()` (theme.typ) so any future
// header-tagging / alternating-fill / WCAG fix lands here too.
// `header-align` centres the column heads while the body keeps
// left-aligned dimension labels and centred status pills.
#data-table(
  columns: 3,
  header: ([Dimension], [Status], [Notes]),
  align-rules: (left + horizon, center + horizon, left + horizon),
  header-align: (center + horizon, center + horizon, center + horizon),
  ([Pilot deployment], status-pill("Not yet", kind: "notyet"),
   [RC-3 deployed in orchestration-only mode under the interim ATO;
    pilot evaluation processed \~2,200 rows, each producing a
    complete audit record.]),
  ([Air-gapped deployment], status-pill("Supported", kind: "supported"),
   [Local auth provider; no required external services.]),
  ([Federated identity], status-pill("Supported", kind: "supported"),
   [OpenID Connect, Microsoft Entra.]),
  ([Encryption at rest], status-pill("Optional", kind: "optional"),
   [SQLCipher passphrase, opt-in.]),
  ([Encryption in transit], status-pill("Required", kind: "supported"),
   [Validated at the trust boundary for external calls.]),
  ([Manual upgrade steps], status-pill("Partial", kind: "optional"),
   [Current release requires operator-administered database schema
    recreation between certain versions. An automated migration path
    is on the roadmap.]),
  ([Operational documentation], status-pill("Supported", kind: "supported"),
   [Runbooks shipped with the platform — see below.]),
)

#v(sp-5)

// Operational documentation included — uses the lower half of the
// page to enumerate the runbooks referenced by the table row above.
// Not a level-1 heading: keeps the ToC clean.

#text(size: size-eyebrow, fill: c-action, weight: "bold",
  tracking: 2pt, upper("Operational documentation included"))
#v(sp-1)
#line(length: 100%, stroke: 0.5pt + c-rule)
#v(sp-2)

Seven runbooks ship with the platform, covering the operational tasks
an agency would carry out across the lifecycle of a deployment.

#v(sp-2)

#let runbook-item(title, desc) = block(
  width: 100%,
  inset: (x: sp-3, y: sp-2),
  fill: c-panel,
  radius: 2pt,
  stroke: (left: 2pt + c-action),
  breakable: false,
  {
    text(size: size-small, weight: "bold", fill: c-navy, title)
    linebreak()
    text(size: size-small, fill: c-ink-soft, desc)
  },
)

#grid(
  columns: (1fr, 1fr),
  column-gutter: sp-2,
  row-gutter: sp-2,
  runbook-item("Resume",
    [Recover from an interrupted pipeline run.]),
  runbook-item("Routing investigation",
    [Trace why a row went where it did.]),
  runbook-item("Incident response",
    [Triage and remediate production failures.]),
  runbook-item("Database maintenance",
    [Audit-database tuning, vacuum, and retention.]),
  runbook-item("Backup",
    [Operational backup of audit data.]),
  runbook-item("Key Vault configuration",
    [Secret-reference setup against Azure Key Vault.]),
  runbook-item("Ansible-based Ubuntu deployment",
    [End-to-end deploy onto a sovereign Ubuntu host.]),
)

// ---------------------------------------------------------------------------
// 7. Residual risk (numbered cards)
// ---------------------------------------------------------------------------

= Residual risk

Honest enumeration. Each item is real, currently unmitigated, and
visible to anyone evaluating the platform.

// Risk-card stripe + numbered-badge use DTA warning orange (c-accent)
// rather than the previous neutral grey (c-notyet). "Risk" is a
// future-tense warning — orange matches; grey under-signalled. DTA's
// error red (c-error) stays reserved for actual error / failure
// semantics, never used for advisory/risk content.
#let risk-card(num, title, body) = block(
  width: 100%,
  inset: sp-3,
  radius: 3pt,
  fill: c-panel,
  stroke: (left: 3pt + c-accent),
  // Each risk card is a single visual unit — never split across pages.
  breakable: false,
  {
    grid(
      columns: (auto, 1fr),
      column-gutter: sp-3,
      align(top, box(width: 22pt, height: 22pt, fill: c-accent,
        radius: 11pt,
        align(center + horizon,
          text(fill: white, size: 11pt, weight: "bold", num)))),
      block({
        text(size: size-h3, fill: c-navy, weight: "bold", title)
        v(sp-1)
        text(size: size-body, fill: c-ink, body)
      }),
    )
  },
)

// All 5 risk cards rendered at the global maximum natural height so
// the grid reads as visually balanced. Same `layout + measure` pattern
// as the page-2 dashboard and the assurance-posture split panel.

#let risk-1 = risk-card("1", "Single-contributor continuity risk")[
  The platform has one developer. Loss of that contributor would
  halt development. The codebase, audit trail design, and runbooks
  are documented to a standard that allows another engineer to take
  over, but no second engineer has yet been onboarded.
]

#let risk-2 = risk-card("2", "Independent assurance is scope-limited")[
  The orchestration component holds an interim ATO. The rest of the
  platform — audit-integrity, trust-tier, and access-control claims
  outside the orchestration scope — has been internally tested but
  not independently assessed by an IRAP-registered assessor or
  equivalent. Agencies adopting beyond the orchestration scope
  under a high-assurance obligation will need to factor an
  independent assessment into their adoption plan.
]

#let risk-3 = risk-card("3", "Deployment scope is pilot-only")[
  RC-3 has been deployed in orchestration-only mode for pilot
  evaluation under the interim ATO, processing
  #data.pilot-rows-prose rows. The full pipeline platform has not
  been deployed in production. Operational characteristics under
  sustained agency load (concurrent users, audit-database growth,
  long-running pipelines under contended infrastructure) have been
  tested in simulation; the pilot supplies real-world data for the
  orchestration-only scope at the volume noted above.
]

#let risk-4 = risk-card("4", "Manual schema migration")[
  Upgrading between certain releases currently requires an operator
  action to recreate the session database. This is documented but
  is a manual step, not an automated migration.
]

#let risk-5 = risk-card("5", "Default plugins include third-party dependencies")[
  Microsoft Dataverse, ChromaDB, Azure OpenAI, OpenRouter, and
  Microsoft Entra integrations depend on those vendors' SLAs and
  security postures. ELSPETH's audit trail records what was called
  and what was returned, but does not extend the audit boundary into
  those external systems.
]

// Full-width single column: each card runs the full page width and
// stacks vertically. Content wraps to fewer lines than the 2-column
// version, so all five cards fit on a single page.
#stack(spacing: sp-2, risk-1, risk-2, risk-3, risk-4, risk-5)

// ---------------------------------------------------------------------------
// 8. What an evaluator should consider next
// ---------------------------------------------------------------------------

= Next steps

Three forward priorities, in scope of the platform's current
trajectory:

- #strong[Rigorous testing and UX enhancement.] Test the platform
  against real-world workloads and refine the user experience to
  meet operator needs.

- #strong[Plugin-extension program.] Open the platform to a curated
  set of plugins that extend ELSPETH into a deep-research system
  across more data sources, tools, and analytical operations.

- #strong[Sovereign deployment with IRAP assessment.] Deploy ELSPETH
  on a sovereign system and complete an IRAP (Information Security
  Registered Assessors Program) assessment so the platform can
  handle classified payloads.

The companion documents — #emph[Progress: cumulative engineering
output] and #emph[Velocity: delivery cadence] — provide the
engineering provenance behind the capability claims in this brief.
