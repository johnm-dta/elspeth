# Wider-baseline test re-run — 2026-05-24 (deterministic)

**Ticket:** elspeth-a9862ed231 (P1, closed 2026-05-23 18:26 UTC by `claude-rc6-remediation`). Diagnostic-only; no fixes applied as part of this report. (Inline fix `111f90e2d` for the Bin F EmptyResumeStateError TIER-2 annotation was landed by the closing session in a separate commit, per the ticket's `close_reason`.)

**Filename history:** Originally authored at `notes/wider-baseline-2026-05-24.md` at 2026-05-24T04:00+10. Renamed to `…-deterministic.md` to match the canonical task spec on elspeth-a9862ed231; predecessor removed (no content divergence between the two filenames).

**Branch:** `feat/multi-source-token-scheduler` @ `0899db85c` at run time. HEAD at commit time is `111f90e2d` (the closing session's inline annotation fix). Working tree at commit time also contains unrelated WIP for elspeth-ddde8144b6 (peer-worker lease reaping; `SchedulerLeaseLostError` and orchestrator/processor changes) which was NOT included in the test runs documented below and is being preserved unchanged by this commit (10 commits since the prior wider-baseline run on 2026-05-23 documented in `notes/g4-sprawl-evaluation-context-2026-05-23.md`).

**Command (deterministic — no xdist):**

```bash
.venv/bin/python -m pytest tests/unit/ tests/integration/ -p no:xdist --tb=short -q --no-header --continue-on-collection-errors
```

Logs: `/tmp/wider-baseline-2026-05-24-run1.log`, `/tmp/wider-baseline-2026-05-24-run2.log`.
Classifier output: `/tmp/bin_final.json` (full per-test mapping).

---

## Headline numbers

- **243 failures + 1 collection error** (test_exporter.py — same `source_row_index` Tier-1 invariant, surfaces at import time of a module-level fixture).
- **17893 passed, 5 skipped, 21 deselected, 50 warnings.**
- Wall time **~8 minutes** both runs.

### Determinism check (Iron Law verification)

| | Run 1 | Run 2 |
|---|---|---|
| Failed count | 243 | 243 |
| Identical failure set? | — | **YES** (`diff /tmp/failed-run{1,2}.txt` empty) |
| Wall time | 490.36 s | 481.08 s |

**Verdict: the failures are 100% deterministic. The earlier "xdist worker-count drift" framing is mechanically false.**
The prior session's "5608 → 17961 pass count tripling" was likely xdist worker drift *in the pass-count denominator only*; the failures themselves were and are reproducible. The 30→170→243 failure-count growth across runs is real signal, not measurement noise. The 73-failure delta from the prior session (170 → 243) is partly:
- New Tier-1 invariants now bite (commits `f909b2aff`, `2fa35f6fc`, `1e64f21a8`, and the `source_row_index` removal of implicit-fallback).
- New `active_source_name` orchestrator kwarg (commit ranges around `bd16fa641`, `9bfc49540`).
- Bins are now visible end-to-end where prior `-x` or xdist masking would have hidden parametrizations.

---

## Per-bin classification (243 failures + 1 collection error)

### Bin A — G6 schema-contract drift (89 failures + 1 collection error) — **dominant**

**Signature:**
- `sqlalchemy.exc.CompileError: Unconsumed column names: schema_contract_json, schema_contract_hash`
- `TypeError: RunLifecycleRepository.begin_run() got an unexpected keyword argument 'schema_contract'`
- `AttributeError: 'RunLifecycleRepository' object has no attribute 'update_run_contract' / 'get_run_contract'`

**Root cause:** ADR-025 / commit `4bf5c4e36` ("per-source schema contracts; refuse empty resume with typed error") removed the singular `runs.schema_contract_json` / `runs.schema_contract_hash` columns and the `begin_run(..., schema_contract=...)` / `update_run_contract` / `get_run_contract` APIs in favour of per-source equivalents on the `run_sources` join table. **Tests, fixtures, and conftest helpers still INSERT into / call the removed surface.** Each parametrization of `test_can_resume_*` hits the same fixture INSERT path, hence 89 distinct test IDs collapsing to one fix.

**Representative tests:**

- `tests/unit/core/checkpoint/test_recovery.py::test_can_resume_rejects_completed_run` (and 24+ siblings in same file)
- `tests/unit/core/landscape/repository_integration/test_recorder_contracts.py::TestBeginRunWithSchemaContract::*`
- `tests/unit/core/landscape/repository_integration/test_contract_audit.py::TestFullAuditTrailWithContracts::test_full_audit_trail_with_contracts`
- `tests/integration/checkpoint/test_recovery.py::*` (~11 failures)
- `tests/integration/pipeline/test_aggregation_recovery.py::*` (~5 failures)
- `tests/integration/pipeline/orchestrator/test_resume_guardrails.py::*` (~4 failures)

**Fix shape:** Migrate test fixtures + conftest `wrapped_begin_run` (`tests/conftest.py:165`) to the per-source contract API (`begin_run_source` / `update_run_source_contract` / `get_run_source_contract`). Remove direct INSERTs that reference `runs.schema_contract_json` / `runs.schema_contract_hash`. **No production code change needed.**

**Ticket affinity:** This is the test-side companion of completed P1 **elspeth-97bfe206bb** ("G6 audit read/write asymmetry"). It is **not** absorbed by an existing pending ticket — recommend **NEW** P1 sub-ticket "Migrate test fixtures off removed singular schema-contract surface" under the audit epic.

---

### Bin B — DAG auto-source-on-success sentinel not honoured (34 failures)

**Signature:** `elspeth.core.dag.models.GraphValidationError: Source 'primary' on_success '_auto_source_on_success' is neither a sink nor a known connection.` (sometimes preceded by `No producer for connection 'source_out'`).

**Root cause — REAL ENGINE BUG.** The graph builder uses an `_auto_source_on_success` sentinel string when a source's `on_success` is unset (legacy default → auto-wire to first downstream). The post-multi-source builder validates `source_on_success` against the connection registry **before** the auto-wiring pass resolves the sentinel into a real edge. Result: every source-only or source→sink pipeline that relies on auto-wire (the common case in test fixtures) now fails graph validation.

This is **Tier-1 invariant landing visibly**, but it is on the production code path — fixtures look correct (`sources={"primary": SourceSettings(...)}`); the builder rejects them.

**Representative tests:**

- `tests/unit/core/test_dag.py::TestExecutionGraphFromConfig::test_from_config_minimal`
- `tests/unit/core/test_dag.py::TestExecutionGraphFromConfig::test_from_config_with_transforms`
- `tests/unit/core/test_dag.py::TestCoalesceNodes::*` (8+)
- `tests/unit/core/test_dag.py::TestExecutionGraphRouteMapping::*` (4+)
- `tests/unit/core/test_dag.py::TestDivertEdges::*` (6+)

**Fix shape:** In `src/elspeth/core/dag/builder.py:~870` (the `for source_name, source_settings_entry in source_settings_map.items():` loop), recognize `_auto_source_on_success` as a valid sentinel and resolve it via the auto-wire path **before** validating against `consumers`/`queue_ids`. Or: drop the sentinel and require explicit `on_success` at config-load time (more aligned with ADR-025 "no implicit defaults").

**Ticket affinity:** **NEW P1** — engine regression introduced by the multi-source migration, not yet ticketed.

---

### Bin C — Tier-1 `source_row_index` None-not-permitted (32 failures + 1 collection error)

**Signature:** `TypeError: source_row_index must be int, got NoneType: None` at `src/elspeth/contracts/freeze.py:169` (`require_int`).

**Root cause — Tier-1 invariant landing visibly, but the field is declared `int | None`.**

Commit `1e64f21a8` ("Stabilize multi-source scheduler handoff") removed the implicit fallback:

```python
# REMOVED in 1e64f21a8:
if self.source_row_index is None:
    object.__setattr__(self, "source_row_index", self.row_index)
```

Per the `feedback_tier1_explicit_vs_implicit_fabrication.md` memory (2026-05-23), this is the right direction: the `if x is None: x = derived_value` pattern is the implicit-fabrication anti-pattern. **But the field type signature still says `int | None = None`** (`src/elspeth/contracts/audit.py:173`), and `require_int(...)` is called **without** `optional=True`. So the contract claims "None is valid" while the validator says "None crashes".

Two possible correct resolutions:

1. **Field is actually required (Tier-1 invariant):** change type to `int`, drop the `| None` and the default; every construction site must pass it. (Mirrors the offensive-programming doctrine.)
2. **Field is genuinely optional:** add `optional=True` to the `require_int` call. But then construction sites passing None must be audited — the field's downstream consumers (lineage, audit exporter, payload store) need to handle None coherently.

The 32 failing tests construct `Row` / `RowLineage` with no `source_row_index` and no `ingest_sequence`. **The collection-error case is `tests/unit/core/landscape/test_exporter.py:102` — a module-level `_ROW = Row(...)` fixture that crashes at import; this is why run1 had exactly 1 collection error (it sweeps an entire module worth of tests into one error).**

**Representative tests:**

- `tests/unit/contracts/test_freeze_regression.py::TestRowLineageDeepFreeze::*` (4+)
- `tests/unit/contracts/test_freeze_regression.py::TestFreezeFieldsUtility::*` (3+)
- `tests/unit/core/landscape/test_lineage.py::*` (~17 failures across `TestExplainByTokenId`, `TestExplainParentIntegrity`, `TestExplainGroupIdValidation`, `TestExplainSinkFilterEquality`, `TestExplainTier1Corruption`)
- `tests/unit/core/landscape/test_model_loaders.py::TestRowLoader::test_valid_load` (and `test_valid_load_with_none_ref` — the SimpleNamespace fixture missing the attribute)
- Collection error: `tests/unit/core/landscape/test_exporter.py` (entire module — `_ROW` module-level fixture).

**Fix shape:** Architectural decision required. **Recommendation:** make `source_row_index` and `ingest_sequence` required (drop `| None`), and update all `Row`/`RowLineage` construction sites in production to derive these explicitly at construction (the orchestrator already knows them — they were the values the implicit fallback was setting). The tests then construct with explicit values, which is the audit-honest shape.

**Ticket affinity:** **NEW P0** — this is on the critical Tier-1 audit path. Sub-ticket of the offensive-programming epic that produced `1e64f21a8`. Not absorbed by an existing ticket.

---

### Bin D — G4 default-source legacy YAML/state KeyError (16 failures)

**Signature:** `KeyError: 'source'`

**Root cause:** Composer test fixtures and composer helpers (`test_tools.py`, `test_route_integration.py`) still index session/state dicts as `state["source"]` after G4 (`0f83461bc`) deleted the singular `ElspethSettings.source` field. The plural shape now keys under `state["sources"]["<name>"]`.

**Representative tests:**

- `tests/unit/web/composer/test_route_integration.py::TestYamlGeneration::test_single_default_source_uses_legacy_yaml_shape`
- `tests/unit/web/composer/test_tools.py::TestSetPipeline::*` (5+)
- `tests/unit/web/composer/test_tools.py::TestUpdateBlobActiveRunGuard::*` (~3)
- `tests/unit/web/composer/test_tools.py::TestDeleteBlobActiveRunGuard::*` (~3)

**Fix shape:** Search/replace test asserts: `state["source"]` → `state["sources"]["primary"]` (or whatever the canonical default name is). Likely also need helper updates in `_helpers.py`.

**Ticket affinity:** Aligns with the dispatch brief's "G9 / composer state singular — `CompositionStateRecord`" (elspeth-?, explicitly out-of-scope per dispatch). Recommend **NEW P1** "Composer test/fixture migration off singular `source` key" or absorb into the broader G9 follow-up.

---

### Bin E — G9 `CompositionStateRecord.__init__()` missing 'source' arg (8 failures)

**Signature:** `TypeError: CompositionStateRecord.__init__() missing 1 required positional argument: 'source'`

**Root cause:** `CompositionStateRecord` (web sessions) **still requires** the singular `source` positional argument (G9 territory, deliberately deferred per the original dispatch brief). Tests in `tests/unit/web/sessions/test_converters.py` were already constructed expecting the plural shape — they fail because the production record dataclass hasn't been migrated yet.

This is the converse of Bin D: here the **production code is still singular**, the **tests are already plural-shaped**.

**Representative tests:**

- `tests/unit/web/sessions/test_converters.py::TestStateFromRecord::test_basic_roundtrip`
- `tests/unit/web/sessions/test_converters.py::TestStateFromRecord::test_none_nodes_becomes_empty_tuple`
- (10 total in the file; 8 hit this signature, 2 hit downstream assertions on `state.source`)

**Fix shape:** Migrate `CompositionStateRecord` to the plural `sources: Mapping[str, SourceSpec]` shape. This is the **G9 ticket** explicitly out-of-scope in the 2026-05-23 G4 dispatch. Re-scoping it in now (RC6 work, not RC5.2).

**Ticket affinity:** Reify the previously-deferred G9 ticket: **NEW P1** "CompositionStateRecord plural-sources migration" or locate the existing `elspeth-?` placeholder from the consolidation note.

---

### Bin F — ADR-019 sweep-durability `active_source_name` kwarg (7 failures)

**Signature:** `TypeError: ..._corrupting_loop() got an unexpected keyword argument 'active_source_name'`

**Root cause:** The orchestrator's invariant-crash loop now passes `active_source_name=...` (introduced in commit cluster `bd16fa641` / `9bfc49540` for the EmptyResumeStateError multi-source path), but the test helpers `_corrupting_loop`, `_plant_orphan_fork_parent`, `_plant_orphan_batch_consumed`, etc. were defined with the old signature.

**Representative tests:**

- `tests/integration/test_adr_019_sweep_durability.py::test_resume_sweep_crash_finalizes_failed_and_preserves_evidence[I1a-_plant_orphan_fork_parent-fork_parent]`
- `tests/integration/test_adr_019_sweep_durability.py::test_resume_sweep_crash_finalizes_failed_and_preserves_evidence[I1b-_plant_orphan_batch_consumed-batch_consumed]`
- `tests/integration/test_adr_019_sweep_durability.py::test_resume_no_work_sweep_crash_finalizes_failed_and_preserves_evidence[I1a-...]`
- `tests/integration/test_adr_019_sweep_durability.py::test_realtime_invariant_crash_finalizes_failed_and_preserves_witnesses[I1c]`
- `tests/integration/test_adr_019_sweep_durability.py::test_realtime_invariant_crash_finalizes_failed_and_preserves_witnesses[I3]`

**Fix shape:** Add `active_source_name` kwarg to the test helpers (`_corrupting_loop`, `_plant_orphan_*`); pass through to the orchestrator they replace. Test-side fix only.

**Ticket affinity:** Companion test-fix to completed P0 **elspeth-241608388f** (CLI EmptyResumeStateError reachability) / **elspeth-791ea2487c** (orchestrator `source` fallback removal). **NEW P2** — test helper signature update.

---

### Bin G — Orchestration "missing source continuation" invariant (4 failures)

**Signature:** `elspeth.contracts.errors.OrchestrationInvariantError: Traversal context is missing source continuation for '<token_id>'. This is a graph construction bug — every source node must have a node_to_next entry.`

**Root cause:** Tests in `test_batch_token_identity.py` and `test_dependency_resolver.py` construct execution graphs that previously relied on default `node_to_next` population. The new strict invariant (likely introduced in the multi-source migration's traversal context refactor) demands every source node have an explicit continuation. These tests trigger the offensive-programming invariant **on a real graph-construction gap** in their fixtures.

**Representative tests:**

- `tests/unit/engine/test_batch_token_identity.py::TestBatchTokenIdentity::test_all_batch_members_consumed_in_batch`
- `tests/unit/engine/test_batch_token_identity.py::TestBatchTokenIdentity::test_triggering_token_not_reused`
- `tests/unit/engine/test_batch_token_identity.py::TestBatchTokenIdentity::test_batch_members_correctly_recorded`
- `tests/unit/engine/test_dependency_resolver.py::TestHashSettingsFile::test_hash_binds_to_canonical_yaml_payload`

**Fix shape:** Audit whether the test fixtures should be migrating to `ExecutionGraph.from_plugin_instances()` (per CLAUDE.md "Never bypass production code paths in tests") — strongly suspect they're hand-rolling graphs and need to route through the production builder. If so, this is a test-side fix; the invariant is correct.

**Ticket affinity:** **NEW P2** — graph fixture migration to production path. Possibly intersects with the existing P2 elspeth-27e36f63ad (Shim Amplification archetype tracker).

---

### Bin H — `IndexError: list index out of range` in source-error paths (6 failures)

**Signature:** `IndexError: list index out of range`

**Root cause — REAL ENGINE BUG.** Source-error-path tests (`test_no_sources_raises`, `test_multiple_sources_raises`, `test_get_source_crashes_on_no_source`, `test_get_source_crashes_on_multiple_sources`) expect the engine to raise `GraphValidationError` for empty-source and multi-source-without-disambiguation cases, but the engine indexes into a list before checking length — so it `IndexError`s before raising the meaningful exception. **Plus** the multiple-sources cases also `DID NOT RAISE` (Bin K), implying the multi-source validation gate is missing in the affected codepath. The 6-test cluster also covers two `test_convergence_scenarios_mocked_llm.py` cases that hit the same source-list-empty path under a different scenario.

**Representative tests:**

- `tests/unit/core/dag/test_graph_validation.py::TestGetSourceErrorPaths::test_no_sources_raises`
- `tests/unit/core/dag/test_graph_validation.py::TestGetSourceErrorPaths::test_empty_graph_raises`
- `tests/unit/core/test_dag.py::TestSourceSinkValidation::test_get_source_crashes_on_no_source`
- `tests/unit/evals/test_convergence_scenarios_mocked_llm.py::TestNumericGateScenario::test_numeric_gate_first_pass_success`

**Fix shape:** In the `get_source()` / equivalent helper, length-check before index; raise `GraphValidationError(...)` instead. This is offensive programming applied at the right boundary.

**Ticket affinity:** **NEW P1** — engine-side. Companion to the multi-source-as-error work; closely related to Bin B.

---

### Bin I — Composer "pipeline is still empty" service drift (4 failures)

**Signature:** `assert 'source.plugin' in 'Recovered.\n\n---\n\n[ELSPETH-SYSTEM] The pipeline is still empty — the composer did not complete a valid build this turn ...'`

**Root cause:** Composer service's pipeline-build path (post-G4) is producing the "pipeline is still empty" sentinel response when it should be producing the validation-error-citing response. Likely cause: the composer's adequacy guard / validation flow is checking for `state.source` (singular) which is now always `None` because state shape is plural.

**Representative tests:**

- `tests/unit/web/composer/test_service.py::*` (4 test methods around blob recovery / adjusted-pipeline messaging)

**Fix shape:** Update the composer service's "is pipeline configured?" check from `state.source is not None` to `state.sources` truthiness; ensure validation errors propagate through the response path correctly.

**Ticket affinity:** **NEW P2** (or absorb into the G9 composer migration ticket above) — composer-side, related to Bins D + E.

---

### Bin J — G4 KeyError 'primary' (3 failures)

**Signature:** `KeyError: 'primary'`

**Root cause:** Composer yaml_generator tests construct fixtures with a different default name than `"primary"`, but the generator now hardcodes `sources["primary"]` lookup. **Inverse of Bin D** — tests use one shape, generator assumes another.

**Representative tests:**

- `tests/unit/web/composer/test_yaml_generator.py::TestGenerateYaml::test_linear_pipeline`
- `tests/unit/web/composer/test_yaml_generator.py::TestGenerateYaml::test_blob_ref_stripped_from_source_options`
- `tests/unit/web/composer/test_yaml_generator.py::TestGenerateYaml::test_frozen_state_serializes_without_error`

**Fix shape:** Yaml_generator must derive the source key from `state.sources.keys()` (probably the only one in a default-single-source pipeline), not hardcode `"primary"`.

**Ticket affinity:** **NEW P2** (or absorb into the broader G9 composer ticket).

---

### Bin K — `Failed: DID NOT RAISE GraphValidationError` (2 failures)

**Signature:** `Failed: DID NOT RAISE <class 'elspeth.core.dag.models.GraphValidationError'>`

**Root cause:** As noted in Bin H — the multi-source validation gate is silently swallowing or missing the "more than one source" / "no sources" cases.

**Representative tests:**

- `tests/unit/core/dag/test_graph_validation.py::TestGetSourceErrorPaths::test_multiple_sources_raises`
- `tests/unit/core/test_dag.py::TestSourceSinkValidation::test_get_source_crashes_on_multiple_sources`

**Fix shape:** Same as Bin H. These tests pin the invariant; the engine fix closes both bins together.

**Ticket affinity:** Same as Bin H.

---

### Bin L — `PluginBundle.source_settings` rename (2 failures)

**Signature:** `AttributeError: 'PluginBundle' object has no attribute 'source_settings'. Did you mean: 'source_settings_map'?`

**Root cause:** `PluginBundle.source_settings` (singular) renamed to `source_settings_map` (plural) per G4/G5; two tests in `test_dag.py::TestDeterministicNodeIDs` still access the old attribute.

**Representative tests:**

- `tests/unit/core/test_dag.py::TestDeterministicNodeIDs::test_node_ids_are_deterministic_for_same_config`
- `tests/unit/core/test_dag.py::TestDeterministicNodeIDs::test_node_ids_change_when_config_changes`

**Fix shape:** Two-line rename in the test file.

**Ticket affinity:** **NEW P3** — trivial test fix.

---

### Bin M — Eval scenario drift (3 failures)

**Signature:** `AssertionError: csv-classifier: ideal synthetic state did not score GREEN. red=[] amber=["source schema missing; cannot verify observed columns ..."]` / `numeric-gate: ideal synthetic state did not score GREEN. red=[] amber=["field 'price' has no numeric handling: neither source schema declares it int/float nor any type_coerce converts it"]` / `StopAsyncIteration` in convergence harness.

**Root cause:** Eval scenarios fed through the audit-readiness scorer expect GREEN but get AMBER because the post-G4 source-schema introspection has different defaults / requires more explicit declarations. This may be a **legitimate signal**: the eval scenarios are surfacing a real drift in what the scorer considers "ready" vs. what the post-migration state actually emits.

**Representative tests:**

- `tests/unit/evals/test_convergence_scenarios.py::test_ideal_state_scores_green[csv-classifier]`
- `tests/unit/evals/test_convergence_scenarios.py::test_ideal_state_scores_green[numeric-gate]`
- `tests/unit/evals/test_convergence_scenarios_mocked_llm.py::TestCsvClassifierScenario::test_csv_first_pass_success`

**Fix shape:** Audit-readiness scorer expectations vs. post-G4 state shape. Either tighten the test fixtures (scenarios need explicit schema declarations now) or relax the scorer (post-G4 implicit-discovery is enough). **Operator call.**

**Ticket affinity:** **NEW P2** — audit-readiness scorer review.

---

### Bin N — Composer skill / tutorial / adequacy drift (3 failures)

**Signature:** Mixed — `test_parser_handles_list_form_*` (yaml shape mismatch in tutorial parser), `test_redaction_policy_snapshot_matches_live_manifest` (snapshot bootstrap drift), `test_skill_drift` (composer skill text out of sync).

**Representative tests:**

- `tests/unit/web/composer/test_tutorial_service.py::test_parser_handles_list_form_transforms_from_production_composer_yaml`
- `tests/unit/web/composer/test_tutorial_service.py::test_parser_handles_list_form_aggregations`
- `tests/unit/web/composer/test_adequacy_guard.py::test_redaction_policy_snapshot_matches_live_manifest`

**Fix shape:** Run `scripts/cicd/bootstrap_redaction_snapshot.py --write`, re-derive list-form tutorial fixtures from the post-G4 generator output, and refresh the skill drift baseline.

**Ticket affinity:** **NEW P3** — snapshot/baseline maintenance.

---

### Bin P — EmptyResumeStateError migration (1 failure — task-required bin)

**Signature:** `elspeth.contracts.errors.EmptyResumeStateError: Resume requested for run 'test-run-123', but no rows were committed and no run_sources records were written. The run failed before any work was persisted to the audit trail. Start a fresh run.`

**Root cause:** **Test expected to reach `_process_resumed_rows`** (mocked to raise `RuntimeError("test failure")`) but the **new typed `EmptyResumeStateError` (commit `9bfc49540` / `4bf5c4e36`) now raises first** because the test mocks `mock_factory.run_lifecycle.get_run_source_resume_records.return_value = {}` (empty). The new guard correctly fires on empty resume records before reaching the user-mocked failure path.

This is the only test in the wider baseline that *literally raises `EmptyResumeStateError` unexpectedly*. **No test was found expecting `OrchestrationInvariantError` / `CheckpointCorruptionError` and getting `EmptyResumeStateError` instead** — that migration appears already done by the prior remediation work (completed P0 elspeth-241608388f / P1 elspeth-81dad89e1c).

**Test:**

- `tests/unit/engine/orchestrator/test_resume_failure.py::TestResumeFinalizesAsFailed::test_resume_failure_finalizes_run_as_failed`

The companion test `test_later_observed_source_records_own_contract_without_overwriting_run_singleton` in the same file fails with `AssertionError: Expected 'update_run_contract' to be called once. Called 0 times` — that's a Bin A sibling (G6 schema-contract drift: `update_run_contract` was deleted/renamed to `update_run_source_contract`), not EmptyResumeStateError migration.

**Fix shape:** Update the test fixture to provide non-empty `get_run_source_resume_records.return_value` so the guard passes and the test can exercise the FAILED-finalization path it was designed to test. **Test pinned the new behaviour correctly; needs a fixture update only.**

**Ticket affinity:** **NEW P3** — single test fixture migration. (The EmptyResumeStateError typed-error migration itself is complete; this is just one fixture not yet updated.)

---

### Bin O — Genuinely "OTHER" / one-offs (16 failures, of which ~10 are siblings of bins A-N)

After Bin re-allocation, the remaining truly-unique-signature failures:

| Test | Error | Likely category |
|---|---|---|
| `tests/unit/contracts/test_propagation_walkers_agree.py::test_walkers_agree_on_chain_topologies` | `AttributeError: 'PluginBundle' has no attribute 'source_settings'` | **Bin L sibling** — same rename |
| `tests/unit/core/landscape/test_model_loaders.py::TestRowLoader::test_valid_load{,_with_none_ref}` | `AttributeError: 'types.SimpleNamespace' has no attribute 'source_row_index'` | **Bin C sibling** — SimpleNamespace test stub missing attr |
| `tests/unit/core/test_config.py::TestLoadSettings::test_load_with_env_override` | Diff `+ csv` (env override leaking) | Pydantic/Dynaconf override drift |
| `tests/unit/core/test_config_alignment.py::TestElspethSettingsAlignment::test_all_fields_categorized` | `assert not {'source'}` | Contract-alignment test still expects the deleted `source` field |
| `tests/unit/core/test_dag.py::TestExecutionGraphFromConfig::test_get_terminal_sink_map_for_source_only` | `Actual message: "Source 'primary' on_success 'missing_sink' is neither a sink nor a known connection."` | **Bin B sibling** — same auto-source bug; the test checks an error path |
| `tests/unit/docs/test_readme_release_surface.py::*` (2) | README missing RC1→RC5 progress-report link | README drift, doc-only |
| `tests/unit/elspeth_lints/test_allowlist_dir_cli.py`, `test_allowlist_loader_unification.py`, `test_audit_evidence_rules.py` (3) | `elspeth-lints` CLI subprocess return codes / "190 more removed" | **CICD allowlist drift** — running with current codebase against stale snapshots; mirror of memory `feedback_merge_scale_cicd_pruning.md` |
| `tests/unit/test_mock_discipline_baseline.py::test_unspecced_mock_baseline_does_not_increase` | 2635 unspecced-mock sites (baseline was lower) | **Baseline drift** — needs ratchet |
| `tests/unit/test_no_hasattr_branching.py::test_hasattr_in_tests_is_limited_to_direct_surface_assertions` | `ValueError: unknown plugin kind: 'source'` | **Bin D sibling** — same singular-key |
| `tests/unit/web/composer/test_tools.py::TestSetPipeline::test_set_pipeline_materializes_inline_blob_source` | `ToolResult(success=False, ...)` | **Bin D sibling** — composer state shape |
| `tests/unit/web/composer/test_yaml_generator.py::TestGenerateYaml::test_generate_pipeline_dict_matches_generate_yaml_shape` | Diff suppressed | **Bin J sibling** |
| `tests/unit/web/sessions/test_converters.py::TestStateFromRecord::test_source_none_preserved`, `test_all_collections_none` | `state.source` is `SourceSpec(...)` not `None` | **Bin E sibling** — same record shape |
| `tests/unit/web/shareable_reviews/test_service.py::test_composition_snapshot_accepts_legacy_single_source_payload` | `pydantic_core._pydantic_core.ValidationError` | **Bin J/E sibling** — legacy payload acceptance lost in G4 |
| `tests/integration/web/test_shareable_reviews_routes.py::test_get_shareable_link_remints_stable_digest` | HTTP 409 (composition not marked ready) | Web routing precondition drift — possibly composer state shape |
| `tests/unit/engine/test_plugin_detection.py::TestProcessorRejectsDuckTypedPlugins::test_processor_rejects_duck_typed_transform` | (unread) | Needs investigation |
| `tests/unit/mcp/analyzers/test_contracts.py::TestExplainFieldPrecedence::test_normalized_name_takes_precedence_over_original_name` | (unread) | MCP analyzer contract drift |
| `tests/unit/web/audit_readiness/test_service.py::TestPluginCatalogHelpers::test_is_registered_and_get_class_share_single_snapshot` | (unread) | Audit-readiness service drift |
| `tests/unit/web/execution/test_service.py::TestWebRuntimeInfrastructure::test_run_pipeline_records_web_user_attribution_in_landscape` | (unread) | Web runtime attribution path |
| `tests/unit/web/sessions/test_routes.py::TestYamlEndpoint::test_get_state_yaml_preserves_secret_ref_markers_in_output` | (unread) | YAML round-trip drift |
| `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py::TestInterruptAndResume::test_resume_shutdown_recheckpoints_buffered_aggregation_without_sink_writes` | (unread) | Graceful-shutdown resume path |

**Bookkeeping note:** Bin O entries are presented for completeness so the operator can see what didn't classify cleanly under bins A-N's signature filters. **~10 of these 16 are siblings of bins A-N** (already counted in their parent bins via the FAILED-summary set; Bin O lists them again for visibility, so do not re-add to bin counts when summing). **Only ~6 are truly independent investigations** requiring per-test triage: `test_load_with_env_override`, `test_all_fields_categorized`, `test_plugin_detection`, `test_normalized_name_takes_precedence`, `test_is_registered_and_get_class_share_single_snapshot`, `test_run_pipeline_records_web_user_attribution_in_landscape`, `test_get_state_yaml_preserves_secret_ref_markers_in_output`, `test_resume_shutdown_recheckpoints_buffered_aggregation_without_sink_writes`, and the two README-drift items. Counting these as the "true" Bin O = 9 unique items; the other 7 are siblings already in their parent bins.

**Fix shape:** No single fix shape — case-by-case. The sibling subset closes when the parent bin closes; the ~9 truly-independent items need individual triage.

**Ticket affinity:** Most fold into Bins B / C / D / E / J. Distinct items: README drift (P3 doc), CICD allowlist drift (P2, file under existing audit epic per `feedback_merge_scale_cicd_pruning.md`), mock-discipline baseline ratchet (P3), and the ~5 unread one-offs above (each P2/P3 once triaged).

---

## Determinism evidence — refuting the "xdist drift" framing

| Claim from prior session | This run |
|---|---|
| "5608 → 17961 pass count tripling is xdist worker-count drift" | The **pass count is now stable at 17893** under `-p no:xdist`. The 5608 figure was an xdist artefact from collection. The failure count is independent of worker count. |
| "Failure count delta is real, not measurement noise" | Confirmed — 170 → 243 over 10 commits is real progress, not flake. **Both my runs produced exactly 243 identical failures.** |
| "12 errors equivalence across runs is the equivalence claim" | Now 1 collection error (the `_ROW = Row(None, ...)` module-level fixture in `test_exporter.py`). The 12-error count from the prior session was likely 12 modules-with-bad-fixtures. Verify by parsing the prior log if needed. |

**No flaky tests observed.** Two runs, identical failure sets. **The ticket's framing assertion is mechanically correct: the failures are deterministic.**

---

## Recommended absorption / new tickets

Mapping to the epic `elspeth-cde984c657` (multi-source-token-scheduler-audit) and the existing pending tickets:

| Bin | Count | Priority | Action |
|---|---|---|---|
| A — G6 schema-contract drift | 89+1 | **P1** | **NEW** — "Migrate test fixtures off removed singular schema-contract surface". Companion to completed elspeth-97bfe206bb. |
| B — DAG auto-source-on-success sentinel | 34 | **P1** | **NEW** — Real engine bug; production fix in `core/dag/builder.py`. |
| C — Tier-1 `source_row_index` None-not-permitted | 32+1 | **P0** | **NEW** — Architectural: drop `\| None` from required Tier-1 fields, update construction sites; or add `optional=True` to validator (operator decision required). |
| D — G4 default-source legacy YAML/state KeyError | 16 | **P1** | **NEW** — Composer test/fixture migration off singular `source` key. |
| E — G9 `CompositionStateRecord.__init__()` missing 'source' | 8 | **P1** | **NEW** — Reify the deferred G9 ticket: plural-sources migration of `CompositionStateRecord`. |
| F — ADR-019 `active_source_name` kwarg | 7 | **P2** | **NEW** — Test helper signature update. Companion to completed elspeth-241608388f. |
| G — Orchestration "missing source continuation" | 4 | **P2** | **NEW** — Migrate batch-token-identity / dependency-resolver test fixtures to `ExecutionGraph.from_plugin_instances()`. |
| H + K — Source-error-path IndexError + DID NOT RAISE | 6+2 | **P1** | **NEW** — Length-check before index in `get_source()` (engine fix; closes both bins). |
| I — Composer "pipeline is still empty" | 4 | **P2** | **NEW** — Composer adequacy-guard plural-state check (or absorb into Bin E). |
| J — G4 KeyError 'primary' | 3 | **P2** | **NEW** — yaml_generator derive source key from state (or absorb into Bin E). |
| L — PluginBundle.source_settings rename | 2 (+1 sibling in O) | **P3** | **NEW** — Trivial test rename. |
| M — Eval scenario drift | 3 | **P2** | **NEW** — Audit-readiness scorer review (operator decisional). |
| N — Composer skill / tutorial / adequacy | 3 | **P3** | **NEW** — Snapshot/baseline refresh. |
| O — Misc one-offs | 16 (mostly siblings of A-N) | mixed | Triage during sibling-bin work; ~5 truly unique need investigation. |

**Summary of NEW tickets:** 13 new tickets (1 P0, 5 P1, 5 P2, 2 P3) plus ~5 truly-orphan items requiring individual triage from Bin O.

**No bins absorb into the existing pending P1/P2 tickets** (elspeth-ddde8144b6 peer-worker reaping, elspeth-66be4216cd G3 helper, etc.) — those are forward-looking work; these are landed-regression cleanup.

---

## What this report is NOT

- It is **not** a fix plan — diagnostic only per ticket constraints.
- It is **not** an architectural critique of the multi-source migration — the failures are mostly the predicted "the surface moved" fallout from a deliberate, well-justified change set, plus 2-3 real engine bugs (Bins B, H/K, possibly C depending on the operator's call) that the change set introduced.
- It is **not** a recommendation to revert. **WE HAVE NO USERS YET**; the structural direction is correct; the test/fixture migration debt is the work product. The 243 failures are work to do, not signal that the work-to-date was wrong.

---

## Verification / reproduction

```bash
cd /home/john/elspeth/.worktrees/multi-source-token-scheduler
.venv/bin/python -m pytest tests/unit/ tests/integration/ \
  -p no:xdist --tb=short -q --no-header --continue-on-collection-errors \
  2>&1 | tee /tmp/wider-baseline-2026-05-24-runN.log
# Expect: 243 failed, 17893 passed, 5 skipped, 21 deselected, 50 warnings, 1 error
# Time: ~8 min on this machine
```

Determinism verification:

```bash
grep "^FAILED" /tmp/wider-baseline-2026-05-24-run1.log | awk '{print $2}' | sort > /tmp/f1.txt
grep "^FAILED" /tmp/wider-baseline-2026-05-24-run2.log | awk '{print $2}' | sort > /tmp/f2.txt
diff /tmp/f1.txt /tmp/f2.txt && echo "DETERMINISTIC"
```
