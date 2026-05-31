# Appendix C — Provenance & Confidence Ledger

Every load-bearing claim in this pack carries a citation chain back to
either:

1. A **deterministic oracle artefact** (under
   [`../reference/`](../reference/)).
2. A **project source-of-truth document** — `CLAUDE.md`,
   `ARCHITECTURE.md`, or an ADR (under
   [`../reference/adr-index.md`](../reference/adr-index.md)).
3. A **code `file:line` reference**.

This appendix is the per-section index of which evidence type underpins
each claim and the resulting confidence rating.

---

## Per-document confidence

| Document | Confidence | Why |
|----------|------------|-----|
| [`../README.md`](../README.md) | High | Navigation only; all substantive claims live in linked documents. |
| [`../00-executive-summary.md`](../00-executive-summary.md) | High | Inherits confidence from each subsection: layer model and structural claims High; performance and frontend claims explicitly N/A. |
| [`../01-system-context.md`](../01-system-context.md) | High | Console-script entries verified verbatim from `pyproject.toml`; actor enumeration cross-cited from `CLAUDE.md` and the project's MCP tool surfaces. |
| [`../02-architecture-overview.md`](../02-architecture-overview.md) | High | Layer model verified at this pack's HEAD; trust-tier model lifted verbatim from `CLAUDE.md`; SDA pattern verified against `core/dag/`, `engine/executors/`, and the `engine/__init__.py` 25-name `__all__`. |
| [`../03-container-view.md`](../03-container-view.md) | High | LOC and file counts re-derived at this pack's HEAD; layer-enforced edges deterministic from `enforce_tier_model.py:237–248`. |
| [`../04-component-view.md`](../04-component-view.md) | High | Cycle topology and edge weights from [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json) (re-verified byte-stable); cross-layer audit-backbone edges from per-cluster catalogs. |
| [`../05-cross-cutting-concerns.md`](../05-cross-cutting-concerns.md) | Medium-High | Defence-in-depth claims evidence-anchored; SEC2/SEC3 raise gaps the input set cannot close (named explicitly). |
| [`../06-quality-assessment.md`](../06-quality-assessment.md) | High per-finding (see below) | Each finding carries its own confidence inherited from its source. |
| [`../07-improvement-roadmap.md`](../07-improvement-roadmap.md) | High per-recommendation (each inherits its mapped finding) | — |
| [`../08-known-gaps.md`](../08-known-gaps.md) | High | The gaps themselves are verifiable; consequences are evidence-anchored claims about what cannot be said. |

---

## Per-finding citation chains

| Finding | Severity | Evidence |
|---------|----------|----------|
| **E1** ADR-010 dispatcher audit-completeness | Resolved | Direct read of `src/elspeth/engine/executors/declaration_dispatch.py:120–172`; `wc -l` of three test files (1,923 LOC total). |
| **E2** `processor.py` cohesion | Medium | Engine cluster catalog "Highest-uncertainty questions" item 1; live `wc -l` (2,700 LOC). |
| **E3** Engine integration tests have no in-cluster directory | Medium | `find tests/integration/` (no `engine/` subdirectory); `CLAUDE.md` "Critical Implementation Patterns" production-path rule. |
| **C1** `core/config.py` cohesion | Medium | Core cluster catalog §4 item 1; live `wc -l` (2,227 LOC). |
| **C2** `core/dag/graph.py` blast radius | Medium | Core cluster catalog §4 item 3; cross-cited consumers in engine, web/composer, web/execution, core/checkpoint. |
| **C3** Audit table count divergence | Medium | Direct count of schema tables in `core/landscape/`; `[DIVERGES FROM KNOW-A24]` reconciliation entry. |
| **C4** `core/secrets.py` placement | Low | Core cluster catalog §4 item 2; live verification of `core/secrets.py` (124 LOC) at root vs `core/security/` (529 LOC). |
| **P1** `azure_batch.py` unread | Medium | Live `wc -l` (1,592 LOC); plugins cluster catalog. |
| **P2** Trust-tier discipline not runtime-enforced | Medium | Plugins cluster catalog §11 item 2; absence of cross-cluster invariant tests in the integration tree. |
| **P3** Plugin-count drift | Low | Plugins cluster catalog §11 item 3; live verified count vs KNOW-A35 vs KNOW-A72. |
| **P4** SCC #1 module-level cycle | Low | [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json) `strongly_connected_components[1]`; `transform.py:9-13` runtime decoupling docstring. |
| **W1** 7-node `web/*` SCC | High | [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json) `strongly_connected_components[4]`; composer cluster catalog §5 item 2; `web/app.py:create_app()` direct read. |
| **W2** Largest concentration of composer logic | Medium | Live `wc -l`: `web/composer/tools.py` 3,860 LOC + `web/composer/state.py` 1,710 LOC. |
| **W3** `web/sessions/routes.py` missed | Medium | Live `wc -l` (2,067 LOC at this pack's HEAD; +504 LOC since the prior 1,563); cross-check against the prior inventory's 12-file list. |
| **W4** `composer_mcp/` mis-framed | Medium | [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json) edges (zero `mcp` ↔ `composer_mcp`; weight-13 `composer_mcp → web/composer`). |
| **W5** `web/execution → .` edge purpose unclear | Low | [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json) edge entry. |
| **K1** `errors.py` mixes Tier-1 / Tier-2 | Medium | Contracts cluster catalog "Highest-uncertainty questions" item 2; live `wc -l` (1,566 LOC). |
| **K2** `plugin_context.py:31` TYPE_CHECKING smell | Medium | Direct read of `contracts/plugin_context.py:31`; ADR-006d Violation #11 framework. |
| **K3** `schema_contract` sub-package promotion | Low | Contracts cluster catalog "Highest-uncertainty questions" item 3. |
| **K4** Catalog citation editorial defect | Low | Reconciliation log "Already-resolved divergences". |

---

## Live re-verification record

Performed at this pack's HEAD (`5a5e05d7`, 2026-04-29):

| Check | Result |
|-------|--------|
| `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` | "No bug-hiding patterns detected. Check passed." |
| `enforce_tier_model.py dump-edges --no-timestamp` byte-diff vs the snapshot | Two new edges in the composer/execution corner of SCC #4; line-number shifts on cited sample sites; SCC topology unchanged. |
| `wc -l` on all 13 ≥1,500-LOC files | Verified; `web/sessions/routes.py` grew +504 LOC. |
| `wc -l` total | 122,554 LOC (from 121,408 at the analysis HEAD). |
| `find docs/architecture/adr/ -name '*.md'` | 17 numbered ADRs plus template; ADR statuses verified per [`../reference/adr-index.md`](../reference/adr-index.md). |

---

## Inherited from the prior analysis

The following load-bearing claims are inherited from the analysis at
`docs/arch-analysis-2026-04-29-1500/` and have **not** been
independently re-verified at this pack's HEAD:

- The "1,923 LOC of dedicated test coverage" figure for ADR-010
  dispatcher tests (E1).
- The "23 intra-cluster edges all flow toward `infrastructure/`"
  claim for plugins (P-strength).
- The "20 schema tables" count for `core/landscape/` (C3).
- The "0 cross-cluster inbound edges to the composer cluster" claim
  (composer cluster strength).

Each is a **High-confidence claim in the analysis**, sourced from
direct reads / catalog enumerations performed during the cluster
discovery passes. They are carried forward in this pack with the
analysis as the citation chain. Re-verifying them would re-do the
cluster discovery — out of scope for this pack.

---

## Limitations of provenance

1. **The pack is dated 2026-04-29.** Subsequent merges may invalidate
   any specific `file:line` citation. The oracle re-derivation procedure
   in [`../reference/re-derive.md`](../reference/re-derive.md) is the
   recommended freshness check before relying on this document.
2. **Frontend, test architecture, and per-file cohesion** are not
   covered. See [`../08-known-gaps.md`](../08-known-gaps.md) for the
   gap catalogue and the downstream packs that resolve them.
3. **Performance and operability** are not covered. No profiling, no
   trace analysis. See [`../05-cross-cutting-concerns.md#2-performance--operability`](../05-cross-cutting-concerns.md#2-performance--operability).
