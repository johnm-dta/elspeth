# 01 — Design Rationale

## Why this redesign exists

The composer's UI has accreted features without sufficient first-principles
review. Prior design passes were repeatedly described by the operator as
"waving through old decisions instead of relitigating them" — the canonical
example being the Catalog button, which has been a fixture of the inspector
chrome since early in RC-5 without anyone re-asking *what user actually wants
that surface*. Other surfaces in the same category: the Spec tab (a leaky
abstraction over the engine's internal composition model), the manual Validate
button (duplicates the continuous validation indicator), the Runs tab
(treated as a peer of the YAML view), and the freeform-default decision in
commit `82dd2e73b` (made without explicit reference to the documented
personas).

The result is a UI that is mechanically functional — every button has a
working handler, every panel has populated content — but whose chrome is
optimized for personas the product does not have, and whose defaults make
choices that are at odds with ELSPETH's stated value proposition of
audit-first knowledge-worker authoring.

This redesign treats the composer's IA as load-bearing for the product's
positioning rather than as an inheritance from "what we built first." The
goal is not polish; it is to align the composer's surfaces with the
documented audiences and use cases, and to surface ELSPETH's defining
feature — the audit trail — at the moments where users actually need it.

## The first-principles question

> If we were designing this composer today, with no inheritance from the
> current UI, which surfaces would each documented persona need?

That question is answered in detail in [02-personas-and-audiences.md](02-personas-and-audiences.md)
and in [03-target-information-architecture.md](03-target-information-architecture.md).
The short version: the composer needs fewer chrome surfaces than it has now,
two surfaces it doesn't have at all (a persistent audit-readiness panel and a
hello-world tutorial), and a richer interpretation of the chat input than the
current design treats it as.

## What changed during the review

The review's first pass produced several recommendations that were corrected
during operator-assistant dialogue. Documenting the corrections is part of
the planning record because future reviews should not repeat the
mis-recommendations:

| Recommendation | First pass (incorrect) | After correction |
|---|---|---|
| Catalog button | Kill from chrome (assumed nobody used it) | **Keep as button, reshape its drawer** from interactive toolkit to searchable system-capability reference. The information is valuable for orientation across all four personas; the *toolkit framing* was the problem, not the underlying API. |
| Execute button | Move out of composer (assumed compose-and-run were separate) | **Keep. It serves the second arm of the use case** — the ad-hoc compose-and-run journey. |
| Runs tab | Kill (assumed it belonged in an operator UI) | **Keep, with narrative-result rendering** for the researcher persona. |
| Freeform default | Auto-invert to guided default | **Adjudicate against expected user mix**, then settled to *guided default with persistent per-user opt-out* (replacing the freeform-default decision in commit `82dd2e73b`). |
| First-run experience | Treat as session zero with no special handling | **Hello-world tutorial** that teaches source/transform/sink, demonstrates the dynamic-source-from-chat feature, surfaces the audit trail, and flows into the default-mode choice. |

The lesson generalizable to future reviews: a first-principles pass that
doesn't read the README's positioning, the documented personas, and the
operator's design decisions is *insufficient*. Future composer reviews should
start by reading the seven memory entries this conversation produced rather
than re-deriving the framing from the code.

## What did not change

The first-principles framing held on these points:

- **Kill the Spec tab.** It exposes the engine's internal composition tree
  and serves no documented persona; only the engine team during debugging.
- **Remove the manual Validate button.** Validation should be continuous;
  the indicator dot does the work. The button is theater.
- **Remove the always-on session sidebar.** No persona opens the composer to
  switch between pipelines they're building. A header switcher suffices.
- **Build a persistent audit-readiness panel.** ELSPETH's defining feature
  is currently invisible during composition.
- **Replace generic ETL templates with audit-domain exemplars** from the
  README's `Example Use Cases` table.

## The product framing this redesign respects

From the README, the composer is positioned for:

> Knowledge workers building document QA, classification, routing,
> extraction, reporting, or review workflows.

With a parallel YAML-author surface for:

> Operators in sensitive, regulated, transactional, operational, medical,
> security, or defence-adjacent workflows.

The composer is therefore **not** a generic pipeline IDE. It is an authoring
surface for knowledge workers operating in audit-bearing contexts, optimized
to produce pipelines that can be reviewed, executed, and explained. Every
recommendation in this set is grounded in that framing.

For the deeper persona breakdown — including the compliance officer (Linda),
researcher (Sarah), marketing ops (Marcus), and senior engineer (Dev) who
appear in `evals/composer-harness/personas/` and
`evals/2026-05-03-composer/hardmode/personas/` — see
[02-personas-and-audiences.md](02-personas-and-audiences.md).

## Out of scope for this redesign

The following are explicitly **not** addressed:

- The hand-edited YAML path. It is the parallel operator surface and is
  documented as first-class in the README; this redesign concerns only the
  Web Composer.
- The chat protocol, session-state model, or composer-tool API. Where
  recommendations touch them, the docs note the touch-point but defer the
  implementation strategy to a follow-up.
- The engine, plugin system, audit recorder, or any backend surface that is
  not the composer's BFF.
- Mobile / touch-first redesign. The composer is a desktop authoring surface;
  responsive behaviour is in scope for keeping the existing breakpoints
  working, but a phone-first rethink is not.

## Companion documents

- [02-personas-and-audiences.md](02-personas-and-audiences.md) for the
  audience model that grounds every recommendation.
- [03-target-information-architecture.md](03-target-information-architecture.md)
  for the surface-by-surface target state.
- [10-implementation-phasing.md](10-implementation-phasing.md) for the
  recommended order of work.
- [11-open-questions.md](11-open-questions.md) for what still needs to be
  decided.
