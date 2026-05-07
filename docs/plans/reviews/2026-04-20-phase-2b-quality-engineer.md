# Phase 2B Declaration-Trust Plan — Quality Engineer Review

**Reviewer:** quality-engineer (QA / test-strategy lens)
**Plan:** `/home/john/elspeth/docs/plans/2026-04-20-phase-2b-declaration-trust.md`
**Commit anchor:** `009b6009` on `h2-adr-010-amendment`

## Verdict: APPROVE-WITH-CHANGES

The plan gets the big structural calls right — rule-of-three gate ordered first, `can_drop_rows` on a separable critical path, ADR-first on `creates_tokens`, explicit bootstrap module ahead of adopter fan-out. But the Definition-of-Done sections drift systematically from the Track 2 epic DoD template (M5 finding) and the H2 §Acceptance bullets: they list green-path VER outputs without naming the three VAL requirements the panel already ratified (per-adopter Landscape round-trip, per-adopter manifest MC3a/b/c assertion, F-QA-5 Hypothesis property test per new dispatch surface). Aggregate-path (M>=2) coverage is not mentioned anywhere despite every 2B adopter overlapping `PassThroughDeclarationContract` on `applies_to`. Registry isolation and red/green discipline are assumed rather than enforced. The fixes below are all test-layer additions, not architecture changes — the skeleton is sound.

---

## QE-1 — DoD template drift from epic M5 (VER-vs-VAL, DoD drift) [CRITICAL]

**Finding.** No Task's DoD names the Landscape round-trip integration test that `elspeth-a3ac5d88c6` makes mandatory for every 2B/2C adopter. Tasks 2, 3, 4, 5 list the round-trip test *file* under "Files:" but the DoD checkboxes describe only VER outputs ("registers in production", "appears in `EXPECTED_CONTRACT_SITES`", "ADR lands"). "Done" is not operationalised as "auditor can run `explain(recorder, run_id, token_id)` on a mis-declaration and recover the violation."

**Evidence.**
- Plan lines 170-174 (Task 2 DoD), 220-223 (Task 3 DoD), 275-278 (Task 4 DoD), 317-320 (Task 5 DoD).
- Epic DoD template: `elspeth-a3ac5d88c6` description §"Definition of Done template (panel finding M5)": "Head-and-tail tests (unit-scope contract test + serializer round-trip) are insufficient."
- Round-trip pattern already in repo: `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py:82-176`.

**Recommendation.** Add a VAL DoD bullet to every adopter task, using exactly the epic template wording: "E2E Landscape round-trip: mis-declaration → dispatcher → `AuditEvidenceBase.to_audit_dict` → Landscape row persisted with scrubbed payload → `explain(recorder, run_id, token_id)` recovers the violation shape." Make it a checkbox the PR reviewer confirms, not prose.

---

## QE-2 — Aggregate-path (M>=2) coverage absent across all adopter tasks (coverage gap, VER) [CRITICAL]

**Finding.** Every 2B adopter added to the `post_emission_check` or `batch_flush_check` site will overlap `PassThroughDeclarationContract` on at least some plugin classes (`DeclaredOutputFieldsContract` + `PassThroughDeclarationContract` both fire when a `passes_through_input=True` transform declares output fields; `CanDropRowsContract` explicitly gates on `passes_through_input=True`). That is precisely what `AggregateDeclarationContractViolation` exists for. The plan exercises only the N=1 reference-equality path; it does not require any adopter to prove M>=2 aggregate round-trip.

**Evidence.**
- N=1 reference-equality anchor: `tests/unit/engine/test_declaration_dispatch.py:307` (`test_single_violation_raises_via_reference_identity`).
- Aggregate round-trip precedent: `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py:279` (`TestAggregateDeclarationContractViolationRoundTrip`).
- `CanDropRowsContract.applies_to` spec in plan line 194: `plugin.passes_through_input and not plugin.can_drop_rows` — direct overlap with `PassThroughDeclarationContract.applies_to`.
- Aggregate invariant requires N>=2: `src/elspeth/contracts/declaration_contracts.py:637-645`.

**Recommendation.** For Tasks 2, 3, 4 add a mandatory test: "aggregate round-trip — construct a row that triggers BOTH `PassThroughDeclarationContract` AND this adopter; assert the dispatcher raises `AggregateDeclarationContractViolation` whose `to_audit_dict()['violations']` contains both children's `to_audit_dict`s post-scrub; assert neither child's payload survives through a `except DeclarationContractViolation` (sibling-class invariant)." Extend `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py` with per-adopter classes rather than inventing a parallel file.

---

## QE-3 — F-QA-5 Hypothesis deferral not addressed (VER gap, H2 acceptance still open) [HIGH]

**Finding.** H2's §Acceptance explicitly requires "one Hypothesis property test per dispatch surface before accepting H2 as closed" (F-QA-5). The plan does not mention Hypothesis, does not name `@given` strategies for `PreEmissionInputs` / `PostEmissionInputs+Outputs` / `BatchFlushInputs+Outputs` / `BoundaryInputs+Outputs`, and does not schedule the work as a Task 0 prerequisite or Task 7 deliverable. If 2B adopters register without the per-surface property tests, H2 closes only nominally and the bundle-derivability invariant degrades to manual inspection.

**Evidence.**
- H2 §Acceptance: `filigree show elspeth-425047a599` — "each dispatch method's input bundle type MUST be a concrete dataclass or TypedDict whose `@given(...)` strategy can be derived without conditional `assume(x is not None)` guards."
- Plan Task 0 (lines 48-82): lists smoke-test pytest invocations only.
- Plan Task 7 (lines 362-399): lists `tests/invariants/test_contract_negative_examples_fire.py`, `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py`, benchmarks, ruff, mypy — no Hypothesis suite.

**Recommendation.** Insert a Task 0.5 (or make it part of Task 1, landing with the bootstrap module): one `tests/invariants/test_bundle_hypothesis_derivability.py` parametrised over the four bundle pairs, each under `@given(...)` with `hypothesis.strategies.builds(...)`. The test should fail if any bundle dataclass acquires a nullable field or an unannotated `Any` that forces an `assume(...)` guard. Tie its pass-state to the H2 close notes.

---

## QE-4 — Per-adopter manifest-drift test missing from Tasks 2-5 (coverage gap, VER) [HIGH]

**Finding.** The N1 MC3a/b/c CI scanner (`scripts/cicd/enforce_contract_manifest.py`) is the static-AST line of defence. `tests/unit/scripts/cicd/test_enforce_contract_manifest.py` already exists for framework-level coverage. The plan adds new contracts but does not require each adopter's PR to add a targeted MC3a/b/c regression test proving the scanner FAILS when the new contract is removed from `EXPECTED_CONTRACT_SITES` or when its `@implements_dispatch_site` marker is dropped. Without this, a future adopter refactor can silently re-introduce the exact drift MC3 was built to catch.

**Evidence.**
- Plan Task 2 "Files:" list (lines 125-132): modifies `tests/unit/engine/test_orchestrator_registry_bootstrap.py` but not `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`.
- Manifest: `src/elspeth/contracts/declaration_contracts.py:818-828`.
- Scanner contract: `scripts/cicd/enforce_contract_manifest.py` (MC3a/b/c per CLAUDE.md reference).

**Recommendation.** Add to every adopter task's "Files:" list: "Modify: `tests/unit/scripts/cicd/test_enforce_contract_manifest.py` — add one MC3a (manifest-lists-but-class-has-no-marker) case and one MC3b (class-marked-but-not-in-manifest) case per new contract." Add a DoD checkbox: "MC3a/b/c regression test for this contract lands in the same PR."

---

## QE-5 — Task 1 bootstrap test does not surface the "new adopter silently absent from bootstrap" failure mode (DoD drift, VER) [HIGH]

**Finding.** Task 1's DoD bullet says "The bootstrap test proves the manifest can be reconstructed from one authoritative import surface" (plan line 116). The test as described (clear registry → import only the bootstrap module → assert manifest fully restored) proves only that the *current* bootstrap file is internally consistent. It does NOT catch the regression we actually fear: a future 2B author adds `elspeth/engine/executors/new_contract.py` with `register_declaration_contract(...)` at module scope AND updates `EXPECTED_CONTRACT_SITES`, but forgets to add the import line to `declaration_contract_bootstrap.py`. Because `EXPECTED_CONTRACT_SITES` was updated, bootstrap fails — but only in production, not in this unit test (which exercises the already-drifted pair).

**Evidence.**
- Plan lines 110-117 (Task 1 Tests + DoD).
- Current registration side-effect pattern: `src/elspeth/engine/executors/pass_through.py:309`.
- Existing bootstrap-set equality tests: `tests/unit/engine/test_orchestrator_registry_bootstrap.py:99-218`.

**Recommendation.** Tighten the bootstrap test to a dual assertion: (a) importing the bootstrap module alone produces `EXPECTED_CONTRACT_SITES`-equal registry; (b) for every entry in `EXPECTED_CONTRACT_SITES`, a regex/AST scan of `declaration_contract_bootstrap.py` finds an `import` statement referencing the contract's defining module (read from registry metadata). Fail if any contract is in the manifest but has no corresponding import line in the bootstrap file. This is the only test shape that catches silent drift at the author-of-the-new-contract moment, not at the production bootstrap moment.

---

## QE-6 — Registry-isolation discipline assumed but not named (flakiness risk) [HIGH]

**Finding.** Every new adopter registers at module import time. Tasks 2-5 add unit tests that will almost certainly import the new contract module, and if they additionally clear/reload the registry inside individual tests (as `test_orchestrator_registry_bootstrap.py` does), cross-test pollution is a live risk. The plan does not name `_snapshot_registry_for_tests` / `_restore_registry_snapshot_for_tests` / `_require_pytest_process` or mandate a fixture pattern.

**Evidence.**
- Pattern already in repo: `tests/unit/engine/test_orchestrator_registry_bootstrap.py:26-66` (the `_isolate_both_registries` fixture).
- Helpers: `src/elspeth/contracts/declaration_contracts.py:989-1039`.
- Plan Tasks 2-5 do not reference isolation helpers.

**Recommendation.** Add a cross-cutting rule to Task 1 or Task 7: "Every new `tests/unit/engine/test_*_contract.py` file MUST use a `_isolate_both_registries`-equivalent fixture built on `_snapshot_registry_for_tests` / `_restore_registry_snapshot_for_tests`. Direct `_clear_registry_for_tests()` without snapshot/restore is a review-blocker." Add a ruff/grep CI check in Task 7 that greps for `_clear_registry_for_tests` without the snapshot/restore pair in the same file.

---

## QE-7 — `can_drop_rows` test matrix incomplete for interaction with pass-through exemption (coverage gap, governance-critical) [HIGH]

**Finding.** Task 3 positions `CanDropRowsContract` as the mechanism that retires the ADR-009 §Clause 3 empty-emission carve-out. The test matrix (plan lines 207-211) covers three cases: mis-declared filter fires, `can_drop_rows=True` does not fire, pass-through with 1 row does not fire. It is missing the interaction case the governance commitment actually targets: a second `passes_through_input=True + can_drop_rows=False` transform with external-call dependencies that emits zero rows — the dispatcher must raise BOTH `UnexpectedEmptyEmission` AND the pass-through contract's empty-emission-exempt short-circuit must not mask it. There is no test for "aggregate dispatch when `CanDropRowsContract` and `PassThroughDeclarationContract` both apply and one fires." Also missing: the non-fire invariant at the `positive_example_does_not_apply` level that `applies_to` returns False when `passes_through_input=False` (the contract should be scoped to pass-through transforms only — mis-scoping is a Tier-1 attribution bug per N2 Layer A).

**Evidence.**
- Plan lines 192-211 (Task 3 Contract semantics + Key tests).
- Empty-emission exemption in pass-through: `src/elspeth/engine/executors/pass_through.py:83-84` (`if not emitted_rows: return`).
- N2 Layer A invariant: `tests/invariants/test_contract_non_fire.py:37-50`.

**Recommendation.** Add to Task 3 tests: (a) aggregate case — mis-declared filter-style transform that ALSO drops input fields → `AggregateDeclarationContractViolation` carrying `UnexpectedEmptyEmission` AND `PassThroughContractViolation`; (b) scoping case — `passes_through_input=False` plugin with `can_drop_rows=False` and zero emission → `applies_to` returns False, no fire (this is the N2 Layer A non-fire harness contribution); (c) explicit test proving the pass-through contract's `if not emitted_rows: return` line does NOT short-circuit the dispatcher before `CanDropRowsContract` runs.

---

## QE-8 — Red/green discipline named only at Task 2; silent at Tasks 3-5 (DoD drift) [MEDIUM]

**Finding.** Task 2 has an explicit "Red test first" section (plan line 150). Tasks 3, 4, 5 each describe "Key tests" or "Tests to write first" but do not require the red-fail-observed step before implementation lands. The distinction matters: a `DeclaredRequiredFieldsViolation` test written *after* a green implementation can pass because the test itself is wrong (wrong assertion shape, wrong input construction), and no-one observes the failure-mode output.

**Evidence.**
- Plan line 150 (Task 2, "Red test first:").
- Plan lines 207-211 (Task 3, "Key tests" — no red-first wording).
- Plan lines 263-266 (Task 4, "Tests" — "Red:" prefix but no instruction to commit red first).
- Plan lines 306-309 (Task 5, "Tests to write first" — but "first" is ambiguous).

**Recommendation.** Add a cross-cutting DoD bullet to Task 7's verification checklist: "Each adopter's PR description must record the red-phase commit SHA where the unit test was observed failing with the expected-shape violation before the contract implementation landed." Or hoist the Task 2 "Red test first" wording into the plan's Execution Notes as a per-task requirement, not a one-off.

---

## QE-9 — Benchmark parametrisation impact not mentioned per-adopter (coverage gap, VER) [MEDIUM]

**Finding.** `test_dispatcher_overhead_scales_with_n` is parametrised over N ∈ {1, 2, 4, 8, 16} and today's production N is 1 (only `PassThroughDeclarationContract`). Each new adopter shifts live-production N upward. Task 7's verification checklist says "Performance benchmark still passes at N ∈ {1, 2, 4, 8, 16}" (line 376) but does not require each adopter's PR to verify that the benchmark's live-registry baseline (the non-parametrised `test_dispatcher_overhead_vs_direct_verify_pass_through` at N=1) is re-measured against the new production N or explicitly rebaselined if scope warrants.

**Evidence.**
- Benchmark: `tests/performance/benchmarks/test_cross_check_overhead.py:206-273` (N=1 live baseline) and `:335-` (parametrised scaling).
- Plan line 376 (Task 7 checklist).

**Recommendation.** Add to each adopter task's DoD: "Live-registry benchmark (`test_dispatcher_overhead_vs_direct_verify_pass_through`) re-run post-registration; either passes under the existing 27µs median / 54µs P99 budget, or the ADR-010 §NFR derivation receives an explicit update in the same PR." Prevents silent N creep degrading the NFR.

---

## QE-10 — Task 6 `creates_tokens` path 1 leaves harness without a proof artifact (coverage gap) [MEDIUM]

**Finding.** Task 6's recommended path 1 ("keep current protocol semantics; retype or replace the issue; no production contract") removes `CreatesTokensContract` from production. But the framework's "accepts a second-shape contract" invariant today relies on `tests/invariants/test_framework_accepts_second_contract.py` — which is named after precisely this preview contract. If the preview contract is removed or renamed and no replacement-shape fixture lands, the framework invariant silently narrows to "accepts one shape" (PassThroughDeclarationContract). The plan does not address this.

**Evidence.**
- Second-contract harness: `tests/invariants/test_framework_accepts_second_contract.py`.
- Plan lines 337-351 (Task 6 path 1 outcome).
- Filigree: `elspeth-cf2ee33808` description cites this harness.

**Recommendation.** Add to Task 6 path 1: "If `CreatesTokensContract` is closed as 'proof artifact, not production invariant', the existing `tests/invariants/test_framework_accepts_second_contract.py` must be either (a) re-pointed at one of the 2B production adopters (DeclaredOutputFields is the natural candidate) as proof that a second-shape contract registers, or (b) retired with a same-PR note citing which successor adopter now carries the shape-diversity invariant."

---

## QE-11 — Task 4's "Green: input row contract includes the field even if payload access would later fail" is under-specified (VER vs VAL confusion) [LOW]

**Finding.** Plan line 266: the non-fire case is "input row contract includes the field even if payload access would later fail for another reason." This conflates VER and VAL: the contract's job is to check declaration drift (contract-fields claim vs plugin-declared claim), not to check payload integrity — but the phrase "payload access would later fail" implies downstream crash. The test as specified does not clarify whether it asserts only "no `DeclaredRequiredFieldsViolation` raised" (correct) or "no exception raised at all" (wrong — a later payload access crash is the transform's problem, not the pre-emission contract's).

**Evidence.** Plan line 266.

**Recommendation.** Reword the Task 4 green case as: "contract includes the required field in `contract.fields` → `pre_emission_check` returns None; any subsequent payload-access failure inside `transform.process()` is not this contract's failure mode and is out of scope." The test should assert `pre_emission_check(...)` returns None cleanly on the input bundle; downstream failure modes belong to other tests.

---

## Findings index

| ID | Severity | Tag |
| --- | --- | --- |
| QE-1 | CRITICAL | VER-vs-VAL, DoD drift |
| QE-2 | CRITICAL | Coverage gap (aggregate-path) |
| QE-3 | HIGH | VER gap (F-QA-5 Hypothesis) |
| QE-4 | HIGH | Coverage gap (manifest MC3a/b/c) |
| QE-5 | HIGH | DoD drift (bootstrap silent-drift) |
| QE-6 | HIGH | Flakiness risk (registry isolation) |
| QE-7 | HIGH | Coverage gap (can_drop_rows interaction) |
| QE-8 | MEDIUM | DoD drift (red/green discipline) |
| QE-9 | MEDIUM | Coverage gap (benchmark per-adopter) |
| QE-10 | MEDIUM | Coverage gap (second-shape harness) |
| QE-11 | LOW | VER-vs-VAL confusion (wording) |
