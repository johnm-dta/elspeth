# ELSPETH open-epic bucketing — 2026-05-22

Classification of the 40 currently-open epics across two parallel tracks: the RC5.2 → 1.0 release and the in-flight multi-source-token-scheduler upgrade (worktree `.worktrees/multi-source-token-scheduler`, branch `feat/multi-source-token-scheduler`, 22 commits ahead of RC5.2).

## Buckets

- **1.0-blocker** — must ship for 1.0 regardless of multi-token outcome.
- **multi-token-in-flight** — load-bearing for the multi-token upgrade; slips with multi-token if multi-token slips. Co-design required.
- **post-1.0** — deliberately not gating 1.0 (deferred plugin phases, parking lots, hygiene backlogs).
- **needs-operator-call** — genuinely ambiguous; tension named in the rationale.

**Multi-token coupling** records whether the epic's working set overlaps modules the worktree reshapes (`web/composer/{state,tools,validation,service,_producer_resolver,_semantic_validator,redaction,protocol}.py`, `web/execution/{accounting,fanout_guard,preflight,schemas,service,validation}.py`, `web/sessions/{service,models,protocol,converters,routes/composer}.py`, `web/shareable_reviews/`, frontend `sessionStore/executionStore/subscriptions/auditReadiness`, the deleted `interpretation_state.py`):

- **load-bearing** — overlaps reshaped surface; co-design required.
- **adjacent** — same subsystem, distinct surface.
- **none** — unrelated.

## Classification

| Epic ID | Pri | Type/WIP | Title | Bucket | MT coupling | Rationale |
|---|---|---|---|---|---|---|
| elspeth-528bde62bb | P0 | WIP | Composer LLM evaluation remediation — validator parity, runtime dry-run, operator visibility | 1.0-blocker | load-bearing | Labelled `rc5-blocker`; in_progress with codex; sweeping changes to composer validator/runtime dry-run/operator-visibility surfaces directly on `composer/{tools,service,validation}` reshaped by multi-token. |
| elspeth-e1ab67e55a | P0 | OPEN | Cluster — Composer correctness (validation + runtime) | 1.0-blocker | load-bearing | Labelled `rc5-blocker`; description names `composer/state.py` (validate, NodeSpec) and `composer/tools.py` (_compute_validation_delta) — both reshaped by multi-token. |
| elspeth-cf84eb1b52 | P0 | OPEN | Run-result row decomposition formula broken for fork/aggregate/coalesce/expand DAGs | 1.0-blocker | load-bearing | Labelled `blocks-rc5.1`; explicit acceptance criterion edits `web/execution/schemas.py` and `RunResultsResponse` — multi-token reshapes the same `execution/schemas.py`. Direct conflict surface. |
| elspeth-fdebcaa79a | P0 | OPEN | Audited content injection — widen blob_ref to inline_content for plugin config | 1.0-blocker | load-bearing | Description marks "P0 blocks RC5 release"; implementation touches `web/composer/tools.py` validation, `web/execution/validation.py` preflight, `web/composer/state.py` redaction — all reshaped by multi-token. |
| elspeth-de91358c30 | P1 | WIP | Cluster — RC5-UX frontend evaluation remediation (composer UI, theming, a11y) | 1.0-blocker | adjacent | RC5-UX merge-blocker; touches composer frontend (App.css, GraphView, CatalogDrawer) — multi-token reshapes frontend stores (`sessionStore/executionStore`) not the view layer; co-design needed only where ARIA contracts touch readiness panels. |
| elspeth-4cf3f22bc7 | P1 | WIP | Cluster — Phase 5 chat-data-entry worktree pre-merge remediation | needs-operator-call | load-bearing | RIVAL WORKTREE that extends `interpretation_state.py` while multi-token deletes it; cannot co-exist. Operator must decide which lands first; either Phase 5 lands then multi-token rebases over the deletion, or multi-token lands and Phase 5 reworks against new model. |
| elspeth-e20903300c | P1 | OPEN | Fork/coalesce audit integrity — schema reconciliation, field provenance, merge safety | 1.0-blocker | load-bearing | Audit-integrity cluster in fork/coalesce subsystem; coalesce/fork audit semantics are exactly what multi-token's scheduler barriers reshape. |
| elspeth-250f698aaf | P1 | OPEN | Cluster — Web auth hardening (OIDC / Entra / JWKS) | 1.0-blocker | none | Labelled `rc5-blocker`; lives in `web/auth/` which multi-token does not touch. |
| elspeth-ef52049338 | P1 | OPEN | Cluster — Web sessions, Alembic env, session-db config | 1.0-blocker | load-bearing | Labelled `rc5-blocker`; touches `web/sessions/{protocol,models}.py` and Alembic env — multi-token heavily reworks `sessions/{service,models,protocol,converters,routes/composer}.py`. |
| elspeth-0fd9dfcb7e | P1 | OPEN | Cluster — Web blobs integrity and MIME detection | 1.0-blocker | adjacent | Labelled `rc5-blocker`; lives in `web/blobs/` and session migrations — multi-token doesn't reshape blob service, only consumes it via composer schema contracts. |
| elspeth-16ddaa7d02 | P1 | OPEN | Cluster — Web secrets store concurrency and portability | 1.0-blocker | none | Labelled `rc5-blocker`; concerns `web/secrets/user_store.py` upsert + driver portability — unrelated to multi-token reshaped surface. |
| elspeth-248536c9e6 | P1 | OPEN | Cluster — Web execution service (terminal state + path allowlist) | 1.0-blocker | load-bearing | Labelled `rc5-blocker`; touches `web/execution/service.py` terminal-state semantics — same module multi-token reshapes for scheduler barriers. |
| elspeth-bf85fc8349 | P1 | OPEN | Cluster — Web execution audit integrity | 1.0-blocker | load-bearing | Cluster thesis is "web execution audit must match CLI"; member bugs sit on `web/execution/service.py` and orchestrator seam — multi-token reshapes both. |
| elspeth-4c2a2779ef | P1 | OPEN | Cluster — Engine/platform correctness (checkpoint, duplication, type safety, audit visibility) | 1.0-blocker | adjacent | Engine-layer cluster; checkpoint/aggregation/coalesce executors are engine-side, multi-token primarily reshapes web layer — overlaps via coalesce state at the boundary. |
| elspeth-a3ac5d88c6 | P1 | OPEN | Track 2 — Declaration-trust framework Phase 2B/2C | 1.0-blocker | none | Phase 2A landed; explicitly framed as "ELSPETH ships work-until-done; release number follows the work"; codex assigned. Contracts/L0 work, unrelated to multi-token web surface. |
| elspeth-783c9dede8 | P1 | OPEN | Composer simple-pipeline convergence — proof, repair, recipes | 1.0-blocker | load-bearing | Under composer-llm-eval umbrella; adds `web/composer/source_inspection.py`, upgrades `preview_pipeline`, adds repair loop in `service.py` — all reshaped surface. |
| elspeth-af7c270380 | P1 | OPEN | Stage-1 validator catch-list residuals — hard-mode eval gaps | 1.0-blocker | load-bearing | Labelled `rc5-blocker`; sits on `assemble_and_validate_pipeline_config()` / composer preflight — multi-token reshapes the validator path. |
| elspeth-9e6ee29c17 | P1 | OPEN | Composer LLM convergence failures — validator catches, LLM doesn't fix | 1.0-blocker | load-bearing | Labelled `rc5-blocker`; touches composer skill-pack + mutation tools + `composer/service.py` retry loop — all reshaped surface. |
| elspeth-3de9d9d5de | P1 | OPEN | Plugin Expansion — search, scraping, reporting, knowledge management | post-1.0 | none | Explicitly framed as RC 5.1 / RC 6 primary feature; phased delivery; new plugins, no overlap with multi-token surface. |
| elspeth-868c55d712 | P1 | OPEN | Plugin Expansion Phase 1 — Web research pipeline | post-1.0 | none | Phase 1 of plugin expansion; new contracts + plugins in `contracts/` and `plugins/`; no composer/sessions/execution reshape. |
| elspeth-2ac5fefeee | P1 | OPEN | OT&E harness — examples replication via plain-English novice intent | post-1.0 | adjacent | Eval harness under `evals/`; blocked_by elspeth-783c9dede8; harness consumes the composer surface but doesn't shape it. Could ship before or after 1.0; not load-bearing. |
| elspeth-d573f8e8ab | P1 | OPEN | Composer panel evals — persona × example matrix | post-1.0 | adjacent | Eval-harness work under `evals/composer-harness/`; supersedes scope of older dogfood task; not a 1.0 ship-gate. |
| elspeth-8843308cfe | P1 | OPEN | Consolidate CI/CD enforcement scripts into elspeth-lints | needs-operator-call | none | Labelled `pre-release` `refactor`; CI/CD discipline refactor with parity harness; pure structural — no enforcement change. Tension: "pre-release" label implies 1.0-blocker, but description says "does not change any enforced invariant" — could be safely post-1.0 if operator accepts deferring a 17→8 CI-job collapse. |
| elspeth-b9a3c59654 | P1 | OPEN | Test audit — folder risk sweep (145 children) | post-1.0 | none | 2026-05-20 folder-by-folder audit coordination; per-folder risk rating; not a 1.0 gate by description. |
| elspeth-4088c4c604 | P1 | OPEN | Bug risk epic — src/elspeth architecture smell + latent defect review | post-1.0 | none | 2026-05-20 source bug-risk sweep over `docs/audit/source-risk/`; review coordination, not a 1.0 deliverable. |
| elspeth-be398f0bcb | P2 | WIP | Remaining transform invariant migration | post-1.0 | none | Contract-invariants cluster; plugin-layer migration of remaining transforms into ADR-009/010 governance; not RC5.2-gated. |
| elspeth-rapid-ea33f5 | P2 | OPEN | RC 5.1 — Autonomous pipeline production hardening (governance, orchestration, infra, PG/S3) | post-1.0 | none | Explicitly RC 5.1 scope; production hardening (Temporal, K8s, PG/S3, content shield). |
| elspeth-a988edf5a5 | P2 | OPEN | Logging policy remediation — remove redundant logs, convert lifecycle to telemetry | post-1.0 | none | Parented under RC 5.1; phased remediation of audit-2026-03-12 findings. |
| elspeth-e8a660f5ed | P2 | OPEN | Web UI UX enhancements — light theme, command palette, etc | post-1.0 | none | Per project memory `project_ux_redesign_phases_1_8_shipped.md` Phases 1-8 ALL IMPLEMENTED on staging 2026-05-20 — flag for "likely shipped but not closed". The remaining listed P1s (light theme, command palette, interactive graph, templates) match the shipped UX-redesign track. See "Likely-shipped but open" below. |
| elspeth-644bb28566 | P2 | OPEN | Plugin Expansion Phase 2 — Vector search and notifications | post-1.0 | none | Phase 2; "RC 5.1 or RC 6" by description; blocked_by Phase 1. |
| elspeth-e6e02c97af | P2 | OPEN | Plugin Expansion Phase 3 — Document processing + lightweight search | post-1.0 | none | Phase 3; labelled `rc6`; blocked_by Phase 1. |
| elspeth-2f0a53bec5 | P2 | OPEN | Composer skill-pack failure routing — boundary semantics + briefing alignment | 1.0-blocker | adjacent | Parented under RC 5 (elspeth-d0ef7cbf54); description says "Parent: RC 5"; skill-pack alignment for composer correctness — adjacent to composer surface but doesn't reshape it. |
| elspeth-68b8794734 | P2 | OPEN | Pipeline-composer skill structural rewrite — five-part split | 1.0-blocker | none | Parented under composer-correctness umbrella (`elspeth-e1ab67e55a`); skill-prompt rewrite to fit 32 KB always-loaded budget; no code surface overlap with multi-token. |
| elspeth-ac647e88a4 | P2 | OPEN | Composer E2E test infrastructure (RC 5.1) | post-1.0 | adjacent | Explicitly parented under RC 5.1; Playwright unblockers + gating ratchet; test infra not composer correctness. |
| elspeth-6f4addce28 | P3 | OPEN | Observability event unification — standardise vocabulary, route lifecycle through telemetry | post-1.0 | none | Parented under Future parking lot (`elspeth-rapid-c9dc55a9b7`); architectural polish, no ship-gate. |
| elspeth-6193ecb7db | P3 | OPEN | Plugin Expansion Phase 4 — Advanced analytics + automation | post-1.0 | none | Phase 4; labelled `rc6`; blocked_by Phase 2. |
| elspeth-f0460a6594 | P3 | OPEN | Composer async/background execution model | post-1.0 | adjacent | Explicitly labelled `future-release`; description states "1 user today; value of B is speculative-on-future-scale, not load-bearing for current correctness"; ADR-first. |
| elspeth-cbb640bc7c | P3 | OPEN | RC4 bugsweep title-only cleanup backlog (113 children) | post-1.0 | none | Tracker-hygiene parking lot for title-only RC4 leftovers; per description "not confirmed-current defects". |
| elspeth-990cd1f2c2 | P3 | OPEN | Promoted observation follow-up backlog (30 children) | post-1.0 | none | Tracker-hygiene parking lot for promoted observations; validate-before-implement. |
| elspeth-0c212a5b34 | P4 | OPEN | Future — deferred work parking lot (53 children) | post-1.0 | none | The Future parking lot itself; explicit by description. |

## Counts

| Bucket | Count |
|---|---|
| 1.0-blocker | 18 |
| multi-token-in-flight | 0 |
| post-1.0 | 20 |
| needs-operator-call | 2 |
| **Total** | **40** |

| MT coupling | Count |
|---|---|
| load-bearing | 12 |
| adjacent | 9 |
| none | 19 |

Note: the brief asked for three buckets including `multi-token-in-flight`, but every epic that is load-bearing for multi-token is also load-bearing for 1.0 (composer correctness, web execution audit, fork/coalesce integrity, web sessions, etc.). There is no "multi-token-only, slip-with-multi-token" cohort in the open-epic set — the multi-token worktree extends RC5.2 work, not parallel scope. Per the slip rule (anything needed for multi-token AND multi-token-only slips together), the 12 load-bearing epics still bucket as `1.0-blocker` because they're also independently 1.0-load-bearing; co-design coordination is the operational outcome, not a slip.

## Cross-coupling and conflicts

- **elspeth-4cf3f22bc7 (Phase 5 chat-data-entry) vs the multi-token worktree.** Phase 5 extends `web/interpretation_state.py`; multi-token deletes it entirely. The two branches cannot both land; one rebases over the other's structural choice. Operator decision required before either branch merges. Captured in needs-operator-call.
- **elspeth-cf84eb1b52 (row decomposition) vs multi-token's `web/execution/schemas.py` rework.** Both add columns to `runs` and reshape `RunStatusResponse`/`RunResultsResponse`. Co-design required — either land the row-decomp fix first and multi-token rebases, or fold the row-decomp acceptance criteria into the multi-token schema design.
- **elspeth-e20903300c (fork/coalesce audit) vs multi-token's coalesce scheduler barriers.** Both touch coalesce executor semantics; merging required.
- **elspeth-fdebcaa79a (blob_ref widening) vs multi-token's composer `tools.py` / execution `validation.py` / composer `redaction.py`.** The widened-blob_ref ADR work edits the same surface multi-token reshapes for per-source schema contracts. Co-author the resolver and the redaction extension.
- **elspeth-528bde62bb, elspeth-783c9dede8, elspeth-af7c270380, elspeth-9e6ee29c17 (composer-correctness umbrella tickets) vs multi-token's composer reshape.** Four parallel composer-correctness streams all editing `composer/service.py`, `validation.py`, `tools.py`, `state.py`. Single coordination loop required, not pairwise merges.
- **elspeth-2f0a53bec5 (skill-pack failure routing) vs elspeth-68b8794734 (skill structural rewrite).** Both edit `src/elspeth/web/composer/skills/pipeline_composer.md`. The rewrite supersedes the boundary-alignment edits if not sequenced; merge the failure-routing semantics into the rewrite's draft, not vice versa.
- **elspeth-868c55d712 (Plugin Expansion Phase 1) blocks elspeth-3de9d9d5de, elspeth-644bb28566, elspeth-e6e02c97af, elspeth-6193ecb7db.** Phase 1 is the foundation; all phase epics declared post-1.0 here.

## Needs operator call

- **elspeth-4cf3f22bc7 (Phase 5 chat-data-entry pre-merge remediation).** Tension: Phase 5 extends `interpretation_state.py` which multi-token deletes. Either Phase 5 ships first and multi-token rebases over the deletion (operator absorbs the rework cost on multi-token), or multi-token ships first and the Phase 5 remediation reworks against the new contract (operator absorbs the rework cost on Phase 5). The 23 review findings under it include schema-epoch / append-only triggers that may be invariant-load-bearing for 1.0 regardless of which interpretation-state model wins. Cannot bucket as `1.0-blocker` without knowing which branch wins; cannot bucket as `post-1.0` because the audit-integrity critical findings are 1.0-class.
- **elspeth-8843308cfe (Consolidate CI/CD enforcement scripts into elspeth-lints).** Tension: labelled `pre-release` (operator's existing categorisation implies 1.0-blocker) but description states the refactor "does not change any enforced invariant" — pure structural. Could safely slip post-1.0 if the parity harness lands and the bespoke gates keep working. The "pre-release" tag may be aspirational; operator should confirm whether the 17→8 CI-job collapse is part of the 1.0 release-presentability story or post-1.0 polish.

## Likely-shipped but open

- **elspeth-e8a660f5ed (Web UI UX enhancements — light theme, command palette, interactive graph, mobile responsiveness).** Per `project_ux_redesign_phases_1_8_shipped.md` (2026-05-20 operator-confirmed): UX redesign Phases 1A/1B/2A/2B/2C/3A/3B/4/5a/5b/6A/6B/7A/7B/7C/8 are ALL IMPLEMENTED on staging. The P1 child list on this epic (light theme, command palette, interactive graph, pipeline templates, validation→node linking) matches the shipped UX-redesign feature set. Recommend a child-by-child audit: each P1 child likely closes immediately on inspection; the epic itself may be closeable with a short rollup comment. The "Phase 9 migration runner" still-pending gate from that memory is separate from this UX epic.

## Methodological notes

- Bucketing was performed by reading the issue body, labels, and parent chain — not by enumerating closed children. Where a memory entry (`project_ux_redesign_phases_1_8_shipped.md`) suggested likely-shipped status, the epic was flagged rather than re-bucketed, per the "status is suggestive, not definitive" guidance in the briefing.
- `elspeth-rapid-ea33f5` ID format verified non-standard but reads via MCP; bucketed as RC 5.1 production hardening per its description.
- All multi-token-coupling judgments cite the description's named modules or implementation surface against the worktree's reshaped surface list. Where coupling is "adjacent," the epic shares a subsystem (web/composer, web/sessions, web/execution) without overlapping the specific files multi-token rewrites.
