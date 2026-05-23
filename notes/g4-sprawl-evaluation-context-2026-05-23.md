# G4 Sprawl Evaluation — Context for SME review

Date: 2026-05-23. Branch: `feat/multi-source-token-scheduler`. Commits already landed: `f72e2fa33` (test fixtures), `3025168b2` (G1 lease self-steal), `3dcebe9ec` (G3 PENDING_SINK drain), `3cf44af3a` (ADRs 025+026), `4bf5c4e36` (G2 + G10).

## What was authorized

Orchestrator dispatched a subagent to implement filigree elspeth-af87655cdb (G4 in the consolidation note: "ElspethSettings.source legacy shim deletion"). The dispatch brief explicitly authorized **G4 + G5 + G8 in one coordinated change set** because they are inseparable:

- **G4 / elspeth-af87655cdb** — delete `ElspethSettings.source` field + `normalize_legacy_source` + `populate_legacy_source_view` validators.
- **G5 / elspeth-781e042709** — delete `legacy_single_source_invocation` branch + `source`/`source_settings` kwargs in `build_execution_graph`.
- **G8 / elspeth-bdc43c911e** — delete `get_source_id()` or rewrite per ADR-025 §6.

The brief **explicitly named as out-of-scope** (separate tickets):
- G6 (runs.contract_json dual-writer)
- G9 (composer state singular — `CompositionStateRecord`)
- G11 (TokenManager None defaults)
- G27 (CAS races)
- G29 (scheduler audit gap)

The brief said: *"If you find yourself touching files in `src/elspeth/engine/tokens.py`, `src/elspeth/core/landscape/data_flow_repository.py`, `src/elspeth/core/landscape/scheduler_repository.py`, or any composer surface (`src/elspeth/web/composer/` or composer skill files), stop and surface."*

The brief also said: *"If the inventory shows >50 production call sites, stop and surface — that's a sign the migration is more entangled than ADR-025 expected and may need an intermediate refactor."*

The brief also said: *"DO NOT commit. Leave staged for operator eyeball."*

## What the subagent delivered

**Headline numbers**:
- 217 files changed, +4487/-3675.
- 35 `config.source.*` production sites migrated.
- 8 `src/elspeth/` files touched plus 55 example YAML files plus ~150 test files.
- Narrow baseline: 299/299 (was 296 + 3 new tests).
- Wider baseline (`tests/unit/ tests/integration/`): 17961 passed, 170 failed, 5 skipped, 12 errors.
- Lint surface clean (ruff/mypy/tier_model/frozen_annotations).
- 2 tier-model allowlist fingerprints rotated for AST-shift consequences.
- Each Tier 1 / Tier 2 / Tier 3 ticket-set in the consolidation note has its own dedup-map entry — see `notes/branch-review-multi-source-token-scheduler-consolidation-2026-05-22.md`.

**Did stop and surface (per dispatch instructions)**:
- Touched `src/elspeth/web/composer/yaml_generator.py:82-95` because post-G4 the YAML loader rejects singular `source:` and composer-emitted YAML would no longer round-trip. Subagent presented this as operator-decision: keep the edit (composer round-trip works) or revert (composer breaks until G9 lands).

**Wider baseline failure triage by the subagent (170 failures)**:
- ~48 pre-existing orthogonal (G11 None defaults x40, G9 composer state x8) — explicitly out-of-scope per brief.
- ~20 migration fallout in composer test asserts (composer tests assert `parsed["source"]` against composer-generated YAML, but generator now emits `parsed["sources"]["primary"]`).
- ~100 "genuinely unknown" — distributed across `tests/integration/web/`, `tests/integration/pipeline/test_composer_runtime_agreement.py`, audit-related integration tests. Subagent did not triage these per "stop expanding scope" instruction.

**Subagent's own caveats**:
- Wider baseline pass count went from 5608 → 17961, a 3× increase that the subagent attributed to "xdist worker-count drift." The error count of 12 matches exactly across both runs; the subagent uses that as the equivalence claim. Failure count delta (30 → 170) is real, not a measurement artefact.
- DB-delete required: every existing audit DB / checkpoint becomes unreadable after this lands.

## Concrete sample of failure modes (from running pytest now)

Top patterns in the 170 failure list:

- `tests/unit/web/sessions/test_converters.py::TestStateFromRecord::*` — 9+ failures. State record serialization tests; CompositionStateRecord (G9 territory) is unchanged but consumers fail.
- `tests/unit/web/composer/test_tools.py::TestUpdateBlobActiveRunGuard::*` and `TestDeleteBlobActiveRunGuard::*` — composer blob storage guard tests.
- `tests/unit/web/composer/test_tutorial_service.py::test_parser_handles_list_form_*` — composer tutorial.
- `tests/unit/web/blobs/test_composition_references_blob.py::*` — blob storage.
- `tests/unit/web/execution/test_validation.py::TestValidatePipelineRuntimePathResolution::*` — composer→runtime path resolution.
- `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py::TestInterruptAndResume::test_interrupted_run_is_resumable` and adjacent — resume/checkpoint tests.
- `tests/integration/pipeline/orchestrator/test_orchestrator_core.py::test_first_row_inferred_source_contract_persisted_before_processing_failure` — main-path orchestrator.
- `tests/integration/audit/test_exporter_batch_queries.py::*` — audit exporter.
- `tests/integration/test_adr_019_resume_counter_parity.py::*` — ADR-019 resume parity.

## Operator clarification 2026-05-23 (post-dispatch)

Operator: *"its not surprising that we need to update the composer to work with the new enhanced engine but lets understand the damage"*

So the framing is **not** "is composer-touch scope drift" — composer requiring updates is *expected* given the engine surface changes. The framing is **damage assessment**:

- Of the 170 wider failures, which are *coherent consequences* of G4/G5/G8 landing (composer + sessions tests that need the same plural migration; these are healthy "the surface moved" failures)?
- Which are *latent regressions* hiding inside the change set (something G4 broke that wasn't supposed to break — particularly the ~40 audit/resume/orchestrator-core failures that look engine-side, not composer-side)?
- Is the **composer migration coherent enough to ship as part of G4**, or does it need a separate G9 pass to be done correctly?
- Are there **idiomatic-Python regressions** smuggled into the 35 `config.source.*` migration call sites (e.g., the `_initialize_database_phase` "first source via `next(iter(...))`" pattern — is this defensible "explicit fabrication-by-programmer" or backsliding to the singleton model)?

## Operator-decisional questions

1. Is the 217-file blast radius for G4+G5+G8 **proportionate** to ADR-025 §1/§2/§6 (which authorize this work but did not quantify scope)?
2. Was the **composer leak** (yaml_generator.py touch) the correct call by the subagent, or should it have been refused as scope drift?
3. Are the **~100 unclassified wider failures** pre-existing failures the wider scope newly captured, or G4 regressions introduced by the migration?
4. Is the **5608 → 17961 pass count tripling** legitimate xdist drift, or a sign that the baseline comparison is broken?
5. Should this commit at all, or be discarded and re-sequenced (e.g., G9 first, then G4 against a composer surface already plural-shaped)?

## Where to look

- Full staged diff: `git -C /home/john/elspeth/.worktrees/multi-source-token-scheduler diff --staged`
- Stat summary: `git -C /home/john/elspeth/.worktrees/multi-source-token-scheduler diff --staged --stat`
- ADR-025: `docs/architecture/adr/025-multi-source-ingestion.md` (recent commit `4bf5c4e36`)
- Consolidation note: `notes/branch-review-multi-source-token-scheduler-consolidation-2026-05-22.md`
- RC6 ticket list: `/home/john/elspeth/notes/RC6-large-list.md` (in main checkout, READ-ONLY)
- Run wider baseline yourself: `cd /home/john/elspeth/.worktrees/multi-source-token-scheduler && .venv/bin/python -m pytest tests/unit/ tests/integration/ -q --no-header 2>&1 | tail -80`

## Stand-down — DO NOT EDIT FILES OR COMMIT

This is an evaluation pass. No edits, no commits, no stage/unstage operations. The orchestrator (Claude Opus, holding session continuity) will gather all SME verdicts and synthesize for the operator.

## Memories worth knowing

- `feedback_tier1_explicit_vs_implicit_fabrication.md` (2026-05-23): T1 doctrine permits typed-exception-for-upstream-interpretation; forbids implicit `.get(k, default)` style fabrication.
- `feedback_default_to_worktree.md`: this worktree is the canonical work site.
- `feedback_no_invented_explanations_for_prior_sessions.md`: when asked about prior behaviour, say "I don't know" rather than confabulate. Applies to the 100 "unknown" wider failures — if you can't classify them, say so.
- ELSPETH's "WE HAVE NO USERS YET" + No Legacy Code Policy: structural deletion is the *correct* direction; the question is scope, not direction.
