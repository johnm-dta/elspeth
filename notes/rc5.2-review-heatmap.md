# RC5.2 Change Heat Map — Review-Coverage Targeting

**Generated:** 2026-05-30
**Range:** `main...RC5.2` (merge-base `7bf364da3`)
**Scale:** 1,234 commits · 2,106 files · **+408,470 / −63,986** lines

This is a **targeting aid**, not a review. The branch is too large to read
end-to-end; this document points scarce review attention at the highest-risk
change concentrations. Rankings are **categorized before they are ranked** —
generated files, skill prompts, and styling are deliberately demoted so the
top tiers contain only code where a human reading the diff adds value.

## How to read this

Risk signals are presented as **independent columns, not a blended score**.
There is no validated weighting of churn × frequency × sensitivity, so any
single "risk number" would be fake precision. Judge each row on its columns:

| Signal | What it means | Why it matters |
|--------|---------------|----------------|
| **Churn** | lines added+removed in the net diff | size of the review surface |
| **Freq** | number of commits that touched the file | instability — reworked-many-times ≠ written-once |
| **New?** | net-new file (no prior baseline) vs modified | new code has *never* been reviewed; whole file is the surface |
| **Tier** | audit-critical path per CLAUDE.md doctrine | a bug in `contracts/`, `core/landscape/`, `engine/orchestrator/` corrupts the legal record |
| **Tests?** | does a same-stem test appear in the branch diff | absent tests → human review substitutes for the missing harness |

> **`Tests?` caveat:** computed by test-file *stem matching*, which is noisy for
> common stems (`service`, `core`, `composer` match dozens by substring). Trust
> the **zeros and ones**; treat large counts as "probably covered, unverified."

## Macro shape — where the mass is

```
src/elspeth/web ............ 138,234  (frontend 73,207 / backend 65,027)   ← ~⅓ of the branch
tests/unit/web .............  57,331
docs/composer/ux-redesign ..  51,249  (docs — not code review)
tests/unit/elspeth_lints ...  29,542
elspeth-lints/src ..........  23,495
docs-archive (cleanout) ....  20,490  (deletions — skip)
```

The branch is **overwhelmingly the web composer subsystem**. Within `src/`,
**327 files are net-new vs 257 modified** — the majority of production change
is greenfield code with no prior review history. That single fact sets the
priority order below.

---

## TIER 1 — Net-new logic in audit-critical paths *(review first)*

New code + high-stakes per the project's own audit doctrine. No baseline to
diff against; the entire file is the review surface.

| File | Churn | New? | Tier | Tests? | Note |
|------|------:|:----:|------|:------:|------|
| `core/blobs_inline.py` | 613 | **NEW** | **L1 core** | 2 | Inline-blob storage path; touches payload/audit boundary |
| `contracts/composer_interpretation.py` | 370 | mod | **L0 contracts** | — | Interpretation hash-domain (see ready bug `elspeth-2c846e322a`) |
| `contracts/blobs.py` / `blobs_inline.py` | 348 / 237 | mix | **L0 contracts** | — | Blob DTOs — frozen-dataclass + tier-1 read guards |
| `contracts/trust_boundary.py` | 221 | mix | **L0 contracts** | — | `@trust_boundary` machinery; migration deferred `elspeth-987f902911` |
| `engine/orchestrator/core.py` | 834 | mod | **L2 engine** | many | Orchestrator core; touched in 16 commits; decomposition in flight |
| `engine/orchestrator/resume.py` | 263 | mix | **L2 engine** | — | Resume/checkpoint path — replay integrity |
| `core/landscape/write_repository.py` | 163 | mod | **L1 landscape** | — | Audit *write* path — the legal record itself |

**Why first:** CLAUDE.md's Tier-1 doctrine — bad data in the audit trail must
crash, never coerce. New code on the contracts/landscape/orchestrator spine is
where an audit-integrity regression would hide.

---

## TIER 2 — High-frequency churned backend logic *(instability signal)*

Repeatedly reworked across many commits. High Freq = the design didn't settle
on the first try; each rework is a chance for a latent regression.

| File | Churn | Freq | New? | Tests? | Note |
|------|------:|:----:|:----:|:------:|------|
| `web/sessions/routes.py` | 3083 | **74** | mod | 16 | Most-touched file on the branch; HTTP surface for sessions |
| `web/composer/service.py` | 3944 | **68** | mod | 16 | Composer orchestration; recently decomposed (memory: service.py 5313→3653) |
| `web/sessions/service.py` | 4116 | **47** | mod | 16 | Session lifecycle/persistence |
| `web/composer/redaction.py` | 3066 | **37** | mod | 7 | **Redaction = security boundary**; web-hardening work landed here (2026-05-29) |
| `web/sessions/protocol.py` | 631 | **33** | mod | — | Settings→runtime protocol contract |
| `web/sessions/models.py` | 1234 | **30** | mod | — | ORM/DTO models — schema correctness |
| `web/composer/tools/_dispatch.py` | 596 | **27** | **NEW** | — | Tool dispatch routing — single source of truth (`elspeth-6c9972ccbf`) |
| `web/composer/guided/state_machine.py` | 850 | **24** | **NEW** | **1** | Guided-mode FSM; net-new, thin test-stem match |

**Why second:** `redaction.py` is a redaction/secret boundary — review with a
security lens (memory: web-hardening + redaction-bypass fix `9b63d8a80`).
`routes.py` at 74 touches is the branch's instability epicenter.

---

## TIER 3 — Large net-new logic, thin/absent direct tests

Big new files where same-stem test coverage is absent or minimal — human
review is the primary correctness check here.

| File | Churn | New? | Tests? | Note |
|------|------:|:----:|:------:|------|
| `web/sessions/routes/_helpers.py` | **4103** | **NEW** | **0** | Largest net-new file on the branch; **no same-stem test** (may be covered via route tests — verify) |
| `web/composer/tool_batch.py` | 1418 | **NEW** | **0** | Extracted in composer decomp; char-pinned per memory but **no named test file** |
| `web/composer/tools/sessions.py` | 1785 | **NEW** | — | Tool handlers — session ops |
| `web/composer/tools/blobs.py` | 1682 | **NEW** | — | Tool handlers — blob ops (redaction-adjacent) |
| `web/composer/tools/_common.py` | 1586 | **NEW** | — | Shared tool plumbing — blast radius across all tools |
| `web/composer/tools/generation.py` | 1550 | **NEW** | — | YAML generation path |
| `web/interpretation_state.py` | 1155 | **NEW** | **1** | Interpretation surfacing (memory: freeform deadlock class) |
| `web/audit_readiness/service.py` | 738 | **NEW** | 16 | Audit-readiness — better covered |
| `web/composer/tutorial_service.py` | 948 | **NEW** | — | First-run hello-world tutorial |

---

## TIER 4 — Modified backend logic with strong test churn *(lighter pass)*

High churn but well-exercised — churn here is likely *intentional, tested*
change. Spot-check the diffs; don't line-read.

| File | Churn | Freq | Tests? |
|------|------:|:----:|:------:|
| `web/composer/tools.py` | 5645 | 27 | 5 |
| `web/execution/validation.py` | 497 | 12 | 10 |
| `web/execution/service.py` | 481 | 17 | — |
| `plugins/infrastructure/base.py` | 435 | — | — |
| `plugins/transforms/llm/model_catalog.py` | 331 | — | — |
| `plugins/transforms/llm/providers/openrouter.py` | 310 | — | — |
| `plugins/transforms/web_scrape.py` | 154 | 17 | — |

---

## TIER 5 — Frontend (React/TS) *(separate reviewer / lens)*

73K churn. Correctness-relevant stores and API client first; components second.

| File | Churn | New? | Note |
|------|------:|:----:|------|
| `frontend/src/components/chat/ChatPanel.tsx` | 1454 | mod | Central chat surface |
| `frontend/src/stores/sessionStore.ts` | 969 | mod | State store — invariant-bearing |
| `frontend/src/stores/subscriptions.ts` | 365 | mod | Refresh/subscription (memory: freeform refreshAll deadlock) |
| `frontend/src/api/client.ts` | 566 | mod | API contract surface |
| `frontend/src/components/audit/AuditReadinessPanel.tsx` | 606 | **NEW** | Audit UI |
| `frontend/src/components/chat/guided/*` | ~1100 | **NEW** | Guided-mode UI |

---

## EXCLUDED from the code heat map *(by design — do not send reviewers here)*

| Path | Churn | Why excluded |
|------|------:|--------------|
| `frontend/package-lock.json` | 3463 | **Generated** — vendored dependency lock |
| `web/composer/skills/pipeline_composer.md` | 2712 | **Skill prompt** — reviewed by re-running the LLM, not diff-reading (memory: `feedback_no_tests_for_skill_prompts`) |
| `frontend/src/**/*.css` (App/chat/guided) | ~6700 | Styling — low correctness risk; visual review only |
| `docs/composer/ux-redesign-2026-05` | 51,249 | Documentation |
| `docs-archive/2026-05-19-docs-cleanout` | 20,490 | Mostly deletions (doc cleanout) |
| `tests/**` | ~160K | Test churn — review as **assertion validity** (do new tests pin behavior or pin bugs? — memory: `feedback_locked_in_buggy_expectations`), a distinct mode from logic review |

---

## Recommended review order

1. **Tier 1** — audit-critical spine (contracts / landscape / orchestrator / inline-blobs). Smallest surface, highest stakes.
2. **Tier 2** — `redaction.py` (security lens) → `routes.py`/`service.py` (instability).
3. **Tier 3** — net-new `_helpers.py` and `tool_batch.py` first (zero direct tests).
4. **Tier 4 / Tier 5** — lighter spot-check passes.

### Cross-references to open work
- `elspeth-2c846e322a` (P0) — contract whitelist / interpretation hash-domain → **Tier 1**
- `elspeth-985057ede1` / `elspeth-59efb74284` (P0) — tier_model allowlist + grandfathering → CI gate, not in this diff map but gates the merge
- `elspeth-6c9972ccbf` (epic) — ToolDeclaration paradigm → **Tier 2** `_dispatch.py`
- Web-hardening + redaction-bypass fixes (memory, 2026-05-29) already landed in **Tier 2** `redaction.py`

---

## Tier 1 Spine — Review Clusters

The audit-critical spine is the entire **L0 `contracts/` + L1 `core/` + L2
`engine/`** change set: **7,034 churn lines across 87 files**. No single file
exceeds 5,000 churn lines (largest is `orchestrator/core.py` at 834, a net
deletion from the decomposition), so the "one file = one cluster" rule never
binds. Clusters are cut on the **layer boundary** — the dependency-closed unit
the project already enforces (downward-only imports L0←L1←L2) — so each cluster
is a coherent review surface, all under the 5,000-line cap.

> **Cross-cutting thread:** the inline-blob payload path spans Cluster A
> (`contracts/blobs*.py`) and Cluster B (`core/blobs_inline.py`). Those two
> reviewers must coordinate on the payload→audit boundary — neither should
> assume the other checked it.

### Cluster A — L0 Contracts *(churn 2,956 · 36 files)*
The type / DTO / protocol surface. Frozen-dataclass `deep_freeze` discipline,
trust-boundary machinery, error taxonomy.
- **Blob/payload DTOs** *(cross-cut → B)*: `blobs.py` (348), `blobs_inline.py` (237)
- **Composer/interpretation**: `composer_interpretation.py` (370), `composer_progress.py` (193), `composer_llm_audit.py` (111), `composer_slots.py` (78), `plugin_assistance.py` (49)
- **Trust & audit DTOs**: `trust_boundary.py` (221), `wire_visible_identity.py` (43), `audit.py` (43), `synthesised_audit.py` (33), `security.py` (29), `advisory_locks.py` (53), `audit_protocols.py` (24), `auth.py` (11), `secrets.py` (8)
- **Config protocols**: `config/protocols.py` (151), `config/runtime.py` (28)
- **Type machinery**: `run_result.py` (167), `enums.py` (130), `plugin_protocols.py` (112), `errors.py` (95), `node_state_context.py` (93), `aggregation_checkpoint.py` (82), `type_normalization.py` (53), `schema.py` (46), `discriminated.py` (30), `declaration_contracts.py` (27), `__init__.py` (27), `contexts.py` (19), `plugin_context.py` (13), + small DTOs

### Cluster B — L1 Core *(churn 1,706 · 25 files)*
Audit/landscape **write path** (the legal record), checkpoint, secrets.
- **Inline-blob storage** *(cross-cut → A)*: `blobs_inline.py` (613)
- **Landscape audit write path**: `write_repository.py` (163), `auth_audit_repository.py` (150), `schema.py` (124), `run_lifecycle_repository.py` (119), `journal.py` (101), `database.py` (46), `exporter.py` (35), `factory.py` (28), `execution_repository.py` (20), `_database_ops.py` (18), + small repos
- **Checkpoint/recovery**: `checkpoint/recovery.py` (76), `checkpoint/manager.py` (75)
- **Secrets/security**: `secrets.py` (75), `security/config_secrets.py` (6), `security/web.py` (4), `security/secret_loader.py` (2)
- **Misc**: `operations.py` (10), `config.py` (10), `templates.py` (8)

### Cluster C — L2 Engine *(churn 2,372 · 26 files)*
Orchestrator, executors, resume/replay, processor.
- **Orchestrator core + lifecycle**: `orchestrator/core.py` (834), `run_status.py` (178), `landscape_registration.py` (150), `cleanup.py` (126), `graph_wiring.py` (120), `shutdown.py` (60), `runtime_preflight.py` (59), `types.py` (55), `preflight.py` (38), `outcomes.py` (22), `aggregation.py` (12), `export.py` (3)
- **Resume/replay**: `orchestrator/resume.py` (263)
- **Executors**: `executors/aggregation.py` (230), `coalesce_executor.py` (78), `executors/declaration_dispatch.py` (32), `executors/pass_through.py` (32), + 6 declared-field executors (1 each)
- **Processing/navigation**: `processor.py` (50), `dag_navigator.py` (19), `triggers.py` (5)

### Reviewer assignment (full pass — 2026-05-30)
| Cluster | python-code-reviewer | solution-design-reviewer | systems (pattern-recognizer) | specialists |
|---------|:--:|:--:|:--:|---|
| A — Contracts | ✓ | ✓ | ✓ | type-design-analyzer |
| B — Core/Landscape | ✓ | ✓ | ✓ | silent-failure-hunter · embedded-database-reviewer |
| C — Engine/Orchestrator | ✓ | ✓ | ✓ | silent-failure-hunter · determinism-reviewer |
