# PDR-0001 — Bootstrap the product workspace from observed state

Date: 2026-06-14   Status: accepted   Author: claude (own-product)   Owner sign-off: yes (grant + primary bet confirmed)
Supersedes: none   Related: vision.md, roadmap.md, metrics.md, current-state.md

## Context

No `docs/product/` workspace existed. ELSPETH is a mature, multi-release
codebase (README at RC-5.3; release/0.6.0 line with ADR-030 landed) with a large
filigree backlog (925 ready, multiple open epics). Standing ownership needs a
durable workspace as its only cross-session memory, so one had to be constructed
rather than resumed. The risk: fabricating a remembered history the product never
had.

## Options considered

1. **Bootstrap from observed reality (README + git log + tracker), then confirm
   the two owner-only facts** — pro: honest, evidence-grounded, cheap; con: some
   strategy is inferred and marked as assumption.
2. **Interrogate the owner for vision/strategy from scratch** — pro: authoritative;
   con: slower, ignores the abundant observable direction, invites recited rather
   than real strategy.
3. **Do nothing / run ownership from memory** — pro: zero setup; con: defeats the
   entire continuity premise — a stateless agent with no workspace inherits
   nothing next session.

## The call

Option 1. Seeded all five artifacts from README/git/tracker. Escalated the two
facts that are genuinely the owner's (not inferable): the authority grant and the
primary Now bet. Owner confirmed the grant **as proposed** and selected **Web
hardening to GA** as primary — which overrode the observed inference (the
checked-out branch is plugins/0.6.0 work). Both are recorded.

## Rationale

Observed direction is cheaper and more honest than interrogation, and the README
already states purpose, audiences, and anti-goals (its "Consider Alternatives"
table *is* the anti-goal list). The only things reality could not settle —
delegation boundary and which bet leads — were the only things asked. The Web-
hardening-vs-0.6.0 fork is exactly why the primary-bet question mattered: the
branch said one thing, the owner said another.

## Reversal trigger

Revisit this entire workspace once the owner has reviewed the seeded vision and
metrics in situ. Specifically: if the owner sets real metric numbers that
contradict the seeded north-star framing, or re-ranks the Now bet away from Web
hardening, supersede the affected artifacts with a new PDR. Also revisit at the
first monthly grant review (next due 2026-07-14).
