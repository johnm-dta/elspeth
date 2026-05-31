# Appendix B — Methodology

How this pack was produced. Brief, because the underlying audit-grade
analysis at
[`../../arch-analysis-2026-04-29-1500/`](../../arch-analysis-2026-04-29-1500/)
is the canonical methodology record.

---

## The two-stage approach

This pack is the **presentation layer** over a separate, audit-grade
**analysis** that already happened. The stages are:

1. **Analysis** (`docs/arch-analysis-2026-04-29-1500/`) — a frozen
   audit artefact set produced by adapting the
   `axiom-system-archaeologist` discipline for a very-large repository.
   Contains the coordination plans, validation gates, cluster discovery
   passes, reconciliation logs, and the deterministic L3 import oracle.
2. **Pack** (this directory) — the polished, scannable rendering of
   the analytical content. Process workings are dropped; analytical
   content is rewritten for clarity; citations are converted from
   internal-tag form to readable cross-references.

The analysis remains the source of truth. This pack inherits its
findings, re-verifies its load-bearing claims against the live tree,
and adds a documentation polish pass.

---

## The analysis (one-paragraph summary)

The analysis covered all ~121k LOC of production Python at HEAD
`47d3dd82`. It produced four contracted deliverables: a discovery and
inventory pass; five hierarchical cluster reports plus a synthesis;
the deterministic L3 import oracle; and a severity-rated quality
assessment with a 12-recommendation roadmap. The methodology was
**evidence-based** — every load-bearing claim carries a citation back
to either a deterministic oracle artefact, a `CLAUDE.md` /
`ARCHITECTURE.md` / ADR statement, or a code `file:line` reference.

The analysis explicitly named four out-of-scope areas (frontend, test
architecture, examples, scripts), refusing to make claims it could not
support. These same gaps are carried into this pack as
[`../08-known-gaps.md`](../08-known-gaps.md).

---

## What this pack adds

### Re-verification at the current HEAD

The analysis snapshot was taken at HEAD `47d3dd82`. This pack ships
against HEAD `5a5e05d7` (2026-04-29). Before any claim was carried
forward, three checks were performed:

1. `enforce_tier_model.py check` re-run — clean at this pack's HEAD.
2. `enforce_tier_model.py dump-edges --no-timestamp` re-run — byte-diffed
   against the snapshot oracle.
3. `wc -l` re-run on the 13 ≥1,500-LOC files plus the total tree.

Drift summary in [`../08-known-gaps.md#6`](../08-known-gaps.md#6-the-head-drift-caveat).
Structural claims (4-layer clean, 5 SCCs with same topology, plugin
spine pattern, composer cluster as import-graph leaf) all hold.
Numerical claims (LOC counts, edge weights) are re-derived against the
live tree throughout this pack.

### Restructuring and prose polish

The analysis is structured for **audit traceability**: dispatch trails,
phase chronologies, validation logs, reconciliation entries are
first-class deliverables alongside the analytical content. This is the
right shape for an audit artefact set. It is the wrong shape for a
document handed to an executive, an architecture board, or a new senior
engineer.

This pack inverts the priority:

- The verdict, headline findings, and roadmap come first.
- Cluster-level depth lives in dedicated [`../subsystems/`](../subsystems/)
  files.
- Process workings are dropped (or, where they are referenced, they
  point back to the analysis).
- Citations are rewritten from internal-tag form (e.g.,
  `[CLUSTER:engine item 3]`, `[PHASE-0.5 §7.5 F4]`) into readable
  cross-references (e.g., "see [`engine.md`](../subsystems/engine.md#5-strengths)").
- Code citations (`file:line`) are kept verbatim — they are load-bearing
  evidence.
- Oracle citations point to byte-stable JSON paths in the snapshot.

### The doc-correctness backlog

R10 in the analysis flagged six specific `ARCHITECTURE.md` drifts. This
pack does not modify `ARCHITECTURE.md` (out of scope) but provides
[`../reference/adr-index.md`](../reference/adr-index.md) as a
self-contained ADR table that R10 can consume.

### Findings carried forward unchanged

This pack does **not invent new findings**, **does not strengthen or
weaken existing severities**, and **does not modify the recommendation
set**. The verdict on each finding (severity, status, recommendation)
was set by the analysis. Where this pack found stale claims (notably
`web/sessions/routes.py` LOC growth from 1,563 to 2,067), the staleness
is surfaced explicitly rather than silently corrected.

---

## What this pack does not produce

- New findings beyond those the analysis surfaced.
- Per-file deep-dives. The 13 ≥1,500-LOC files remain backlogged
  ([R5](../07-improvement-roadmap.md#r5)).
- Frontend coverage. Out of scope for the analysis and out of scope
  here ([R6](../07-improvement-roadmap.md#r6)).
- Test-architecture coverage. Out of scope for the analysis and out of
  scope here ([R7](../07-improvement-roadmap.md#r7)).
- Performance assessments. Static analysis cannot make them.
- A security review. The architectural inputs are stable; a STRIDE
  threat model is the next pack ([R8](../07-improvement-roadmap.md#r8)).

---

## Tooling

| Tool | Role |
|------|------|
| `scripts/cicd/enforce_tier_model.py` | Layer-conformance check (`check`) and L3 import-graph generation (`dump-edges`). Both ran at this pack's HEAD; status documented in [`../reference/re-derive.md`](../reference/re-derive.md). |
| `find` / `wc -l` | LOC counts. |
| `jq` | JSON oracle queries. |
| `git log` | HEAD identification. |

No other automated tooling was used.

---

## Confidence ratings

Every section of this pack carries an implicit or explicit confidence
rating. The rating ladder:

- **High** — Multiple independent sources agree (deterministic oracle +
  cluster source, or oracle + live re-verification, or two cluster
  sources + ADR knowledge).
- **Medium** — One cluster source raises a question that this pack
  carries forward without independent corroboration.
- **Low** — No cluster source covers the area; the gap is named in
  [`../08-known-gaps.md`](../08-known-gaps.md).
- **N/A** — Out of scope.

Per-section confidence ratings appear in
[`C-provenance.md`](C-provenance.md).
