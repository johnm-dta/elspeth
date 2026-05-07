# ELSPETH Architecture Pack

**Status:** Initial issue (unreviewed) · **Date:** 2026-04-29 · **Codebase HEAD:** `5a5e05d7`

A polished, evidence-anchored architecture description of ELSPETH — an
auditable Sense/Decide/Act pipeline framework written in ~122,500 LOC of
Python. This pack is the **readable face** of the underlying analysis;
it is the document set you would hand to:

- a new senior engineer joining the team
- an architecture-review board
- a downstream specialist pack (security, frontend, test-architecture)
- an executive sponsor reviewing posture before a major commitment

For the audit-grade workings (coordination plans, validation gates,
cluster discovery passes, reconciliation logs), see the peer folder
[`docs/arch-analysis-2026-04-29-1500/`](../arch-analysis-2026-04-29-1500/).
This pack is its presentation layer; the analysis remains the source of
truth.

---

## How to read this pack

**Pick your starting point:**

| If you are… | Read this first | Approx. time |
|---|---|---|
| An **executive** wanting the verdict and risk posture | [`00-executive-summary.md`](00-executive-summary.md) | 5 min |
| A **new senior engineer** building a mental model | [`02-architecture-overview.md`](02-architecture-overview.md) → [`03-container-view.md`](03-container-view.md) → [`subsystems/`](subsystems/) | 45 min |
| A **planner** mapping work to risk | [`06-quality-assessment.md`](06-quality-assessment.md) → [`07-improvement-roadmap.md`](07-improvement-roadmap.md) | 30 min |
| A **security reviewer** | [`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md) → [`06-quality-assessment.md`](06-quality-assessment.md) §3.4 | 25 min |
| A **subsystem owner** | [`subsystems/`](subsystems/) entry for your area | 10–15 min |
| A **downstream pack lead** | [`08-known-gaps.md`](08-known-gaps.md) → relevant section | 10 min |

---

## Pack contents

### Architectural narrative (read in order for full picture)

1. [`00-executive-summary.md`](00-executive-summary.md) — Verdict, headline findings, recommended next moves.
2. [`01-system-context.md`](01-system-context.md) — C4 Level 1: ELSPETH as a black box; actors, external systems.
3. [`02-architecture-overview.md`](02-architecture-overview.md) — The 4-layer model, three-tier trust model, SDA execution pattern.
4. [`03-container-view.md`](03-container-view.md) — C4 Level 2: the 11 subsystems, grouped by layer, with cross-cluster handshakes.
5. [`04-component-view.md`](04-component-view.md) — C4 Level 3: drilldown into the three structurally interesting zones (web SCC, plugin spine, audit backbone).
6. [`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md) — Security, audit-trail completeness, maintainability, testability.

### Assessment & action

7. [`06-quality-assessment.md`](06-quality-assessment.md) — Severity-rated findings per subsystem with confidence ledger.
8. [`07-improvement-roadmap.md`](07-improvement-roadmap.md) — Twelve prioritised recommendations with effort estimates.
9. [`08-known-gaps.md`](08-known-gaps.md) — Named limitations: frontend, test architecture, large-file deep-dives.

### Subsystem reference

10. [`subsystems/`](subsystems/) — One file per significant subsystem (5 deep clusters + 1 leaf-collection page).

### Reference data

11. [`reference/`](reference/) — Deterministic L3 import oracle, tier-model schema, ADR index, regeneration commands.

### Appendices

12. [`appendix/`](appendix/) — Glossary, methodology summary, provenance ledger.

---

## Headline findings

These are the load-bearing claims this pack rests on. Full evidence
chains live in the assessment ([`06-quality-assessment.md`](06-quality-assessment.md))
and subsystem files.

1. **The 4-layer model is mechanically clean.** Zero upward-import
   violations, zero TYPE_CHECKING layer warnings, CI-enforced by
   `scripts/cicd/enforce_tier_model.py`. Re-verified at this pack's
   HEAD. *(Confidence: High.)*

2. **ELSPETH's audit guarantees are encoded as mechanical invariants,
   not conventions.** Context-manager state guards, AST-scanning drift
   tests, allowlist-with-justification CI gates, deeply-frozen primitives.
   Unusually strong for a codebase of this size. *(Confidence: High.)*

3. **A 7-node strongly-connected component spans every `web/*`
   sub-package.** This is the FastAPI app-factory pattern made
   structural; no acyclic decomposition is possible without an
   architecture-pack decision. Decomposition is non-trivial, currently
   unowned, and the recommended freeze item before the next major
   `web/` addition. *(Confidence: High.)*

4. **`plugins/infrastructure/` is the structural spine of the plugin
   ecosystem.** The `plugins/sinks → plugins/infrastructure` edge is
   the heaviest single L3 edge in the codebase (weight 45). Sources,
   transforms, and sinks are clients of infrastructure, not peers.
   *(Confidence: High.)*

5. **13 files exceed 1,500 LOC** and concentrate ~23% of production
   Python in 0.6% of files. Per-file cohesion is not assessed in this
   pack (see [`08-known-gaps.md`](08-known-gaps.md) for the deep-dive
   backlog). *(Confidence: High on count; per-file verdicts open.)*

6. **Two material areas remain outside this pack's coverage**: the
   ~13k-LOC TS/React frontend under `web/frontend/`, and the ~351k-LOC
   `tests/` tree. These are named gaps, not silent omissions. Any
   "complete architecture" claim about ELSPETH requires the frontend
   pass and the test-architecture pass to land first.
   *(Documented in [`08-known-gaps.md`](08-known-gaps.md).)*

---

## Provenance and posture

Every load-bearing claim in this pack carries a citation back to either
a deterministic oracle artefact (under [`reference/`](reference/)), a
project source-of-truth document (`CLAUDE.md`, `ARCHITECTURE.md`, an
ADR), or a code `file:line` reference. Each section's confidence rating
appears in the per-document section or in [`appendix/C-provenance.md`](appendix/C-provenance.md).

Findings are rated by objective severity (Critical / High / Medium /
Low). Strengths are reported only where evidence warrants them, not as
diplomatic packaging for the bad news.

The pack is dated 2026-04-29 against codebase HEAD `5a5e05d7`. The
underlying analysis was performed at HEAD `47d3dd82`; structural claims
have been re-verified against the current tree. Minor numerical drift
(two new edges in the composer/execution corner of the web SCC, +504 LOC
on `web/sessions/routes.py`, +56 LOC on `web/composer/tools.py`) is
recorded in [`appendix/C-provenance.md`](appendix/C-provenance.md). The
structural verdicts are unchanged.

---

## What this pack is not

- **Not a security review.** Cross-cutting security findings are surfaced
  ([`05-cross-cutting-concerns.md`](05-cross-cutting-concerns.md) §1)
  but a STRIDE threat model is a separate downstream deliverable
  (recommendation R8 in [`07-improvement-roadmap.md`](07-improvement-roadmap.md)).
- **Not a test-architecture review.** The `tests/` tree is 2.9× the size
  of `src/`; whether that represents discipline or an inverted pyramid
  is unanswered (R7).
- **Not a frontend review.** `web/frontend/` (~13k LOC TS/React) is
  outside the Python-lens of this pack (R6).
- **Not a per-file deep-dive.** Thirteen files ≥1,500 LOC are flagged
  but not opened at component depth (R5).
- **Not a performance assessment.** No profiling has been performed.

The honest claim is that this pack is the strongest foundation available
without those additional passes — and names exactly where each one
should land.
