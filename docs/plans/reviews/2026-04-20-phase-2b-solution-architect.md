# Phase 2B Declaration-Trust — Solution-Architect Review

**Verdict: APPROVE-WITH-CHANGES.**

The plan is structurally sound: sequencing is correct, the two-phase "ADR-first for CreatesTokens" proposal is the right call, the bootstrap task exists in the right slot, and the author's Reality Anchors (C1–C5) accurately reflect repo state I verified in this review. The load-bearing issues are (a) Task 1's urgency is under-stated — production registration of `PassThroughDeclarationContract` is already broken outside pytest (see SA-1), so Task 1 is a pre-Task-2 hotfix, not a housekeeping item; (b) the §Adoption State rule-of-three accounting for `pre_emission_check` is not fully satisfied by Phase 2B and the plan does not acknowledge the gap (SA-2); and (c) Task 4's runtime-attribute standardisation has a cross-layer placement question the plan skips (SA-4). A small set of ADR-discipline and reversibility items round out the findings. None block starting Task 0/1 immediately after these are resolved.

---

## SA-1 — CRITICAL — Task 1 is not "bootstrap cleanup," it is a production hotfix; it must land before Task 2

**Finding.** Task 1 (plan lines 84–117) treats explicit bootstrap as a hygienic step that "currently depends too much on incidental imports." Repo reality is stronger: the pass-through contract is not currently registered on any production import chain at all.

**Evidence.**
- `Grep pass_through src/` returns six hits, all comments or string literals. No `src/` module contains `import elspeth.engine.executors.pass_through` or `from elspeth.engine.executors.pass_through import …`. Verified against:
  - `src/elspeth/engine/executors/__init__.py:10-15` imports `aggregation`, `gate`, `sink`, `state_guard`, `transform`, `types` — no `pass_through`.
  - `src/elspeth/engine/executors/transform.py:103` only mentions `pass_through` in a comment.
  - `src/elspeth/engine/processor.py` imports `PassThroughContractViolation` from `errors`, not from `pass_through.py`.
  - `src/elspeth/engine/orchestrator/core.py:160-279` (`prepare_for_run`) asserts registry set-equality against `EXPECTED_CONTRACT_SITES`, which contains `"passes_through_input"`, so a production process that doesn't transitively import `pass_through` will raise `RuntimeError` at bootstrap.
- Registration today works only because `tests/conftest.py:53` (and several test modules) explicitly `import elspeth.engine.executors.pass_through`.

**Impact.** Any production-like invocation of `Orchestrator.run()` / `prepare_for_run()` outside the pytest process path fails fast with the N1 manifest-mismatch error. This turns Task 1 into a regression-blocker: without it, no 2B adopter can be verified end-to-end with a production-shaped orchestration test, because the orchestrator cannot bootstrap.

**Recommendation.** Reframe Task 1 as the first landing step after Task 0, explicitly "fixes production bootstrap," and add a regression test that runs `prepare_for_run()` in a subprocess without pulling `conftest.py` (e.g. `python -c "from elspeth.engine.orchestrator import prepare_for_run; prepare_for_run()"`). Add the missing import ordering to the Task 1 Definition of Done: "orchestrator.core (or a dedicated bootstrap module imported before `prepare_for_run()`) transitively reaches every module registered in `EXPECTED_CONTRACT_SITES`." Also import the bootstrap module from `src/elspeth/engine/orchestrator/core.py` top-of-file, not from `engine/executors/__init__.py` — the orchestrator is the canonical bootstrap entry point (matches the ADR-010 §Decision 3 prose at `docs/architecture/adr/010-declaration-trust-framework.md:166-172`).

---

## SA-2 — HIGH — §Adoption State accounting: `pre_emission_check` rule-of-three is NOT closed by Phase 2B, and the plan does not say so

**Finding.** Plan Task 4 (lines 225–278) lands `DeclaredRequiredFieldsContract` as "the first pre-emission adopter." Per ADR-010 §Adoption State (`docs/architecture/adr/010-declaration-trust-framework.md:249-254`) the `pre_emission_check` surface requires three adopters for rule-of-three. After Phase 2B (Tasks 2–5) the per-surface counts are:

| Surface | Pre-2B | Post-2B | Rule-of-three closed? |
|---|---|---|---|
| `pre_emission_check` | 0 | 1 (`DeclaredRequiredFieldsContract`) | **NO** (2 more needed) |
| `post_emission_check` | 1 | 4 (+ DeclaredOutputFields, CanDropRows, SchemaConfigMode) | YES |
| `batch_flush_check` | 1 | 4 (same three add batch coverage) | YES |
| `boundary_check` | 0 | 0 (2C paired landing) | NO (deferred; correct) |

**Impact.** The plan's Task 7 checklist (line 377) says "the pre-emission hot-path cost is justified only once `DeclaredRequiredFieldsContract` is registered." That is the narrower framework-overhead justification, but it does not match the §Adoption State gate: under F4 a single adopter cannot "validate the subtype's shape against exactly one example" (ADR-010 line 260). The same sharpening that forces the boundary paired-landing rule applies here: landing a single pre-emission adopter leaves the surface's shape unvalidated by multiple uses, and C5 (plan line 46) remains architecturally open after Phase 2B completes.

**Recommendation.** In Task 4 (or a new explicit section), record that Phase 2B closes rule-of-three for `post_emission_check` and `batch_flush_check` only; `pre_emission_check` remains provisional with one adopter, and the next two pre-emission adopters are 2C scope (candidates: a `creates_tokens`-tightened pre-emission adopter if Task 6 path 2 is chosen, plus a source-side pre-emission contract). Either update ADR-010 §Adoption State to explicitly treat the first pre-emission adopter as "provisional until two further adopters register," or state in ADR-013 that the pre-emission surface remains at rule-of-three-pending status post-Phase-2B. Do not silently ship a single-use dispatch surface.

---

## SA-3 — HIGH — Task 4 scope decision (batch-pre vs non-batch) is deferred inside the adopter ADR; it belongs in an ADR-010 amendment

**Finding.** Plan Task 4 lines 255–262 give ADR-013 the choice between (1) "scope to non-batch transforms" and (2) "add a new batch-pre-execution dispatch site." Option 2 is a `DispatchSite` enum extension — by ADR-010 §Adoption State line 263-267 ("A new contract introducing a new surface … would require an ADR amendment to name the surface, update `DispatchSite`, and restart the rule-of-three gate at that surface"), this is an ADR-010 amendment decision, not a ADR-013-local one.

**Impact.** If ADR-013 silently introduces a `batch_pre_emission_check` site, §Adoption State and `EXPECTED_CONTRACT_SITES` and the MC3 CI rules and the `_dispatch` helper all accumulate a new surface without an ADR-010 amendment banner. That would also reset rule-of-three on the new surface to zero — a hidden governance debt.

**Recommendation.** In Task 4, make option (2) contingent on an ADR-010 amendment banner landing first, in the same PR. Option (1) (non-batch scoping) does not touch ADR-010 and remains the lightweight path. Record in ADR-013 which option was chosen, with the rationale; make the ADR cite ADR-010 §Adoption State line 263-267 explicitly. The plan's own "my recommendation: option 1" (line 261) is defensible — endorse it and mark option 2 as a deferred follow-up requiring ADR-010 amendment.

---

## SA-4 — MEDIUM — Runtime attribute `declared_required_fields` on transforms: cross-layer placement needs a deliberate call

**Finding.** Plan Task 4 lines 243–250 propose adding `declared_required_fields: frozenset[str] = frozenset()` to `BaseTransform`. `declared_required_fields` already exists as a field on `SinkProtocol` (`src/elspeth/contracts/plugin_protocols.py:567`). If transforms gain the same attribute name with different semantics (transform *input* fields vs sink *input* fields at write-time), this is a name-collision across protocols the CI cannot catch.

**Evidence.**
- `plugin_protocols.py:567` — sinks: `declared_required_fields: frozenset[str]` (fields required at write boundary).
- `plugin_protocols.py:281,442` — transforms expose `passes_through_input` but not `declared_required_fields`.
- The plan's ADR-013 recommendation would add `declared_required_fields` to `TransformProtocol` too, with different semantics (fields the transform requires to be *present on the input row*).

**Impact.** Two protocols exposing the same attribute name with incompatible semantics is a pit of success for future plugin authors who cross-reference. A sink author reading the transform contract may assume field-at-write-boundary semantics and mis-declare.

**Recommendation.** Either (a) name the transform-side attribute differently — `declared_input_fields` or `declared_row_required_fields` — so the two surfaces are semantically distinguishable at grep time; or (b) explicitly document in ADR-013 (and update `plugin_protocols.py` docstrings for both sinks and transforms) that the semantics differ by plugin type, and add a CI assertion that sinks do not silently inherit a transform-style definition via any shared mixin. The plan should make this call explicitly rather than letting the adopter author default to parity-naming.

---

## SA-5 — MEDIUM — `CreatesTokensContract` path-1 outcome needs a positive artefact, not just a ticket retype

**Finding.** Plan Task 6 (lines 322–360) correctly identifies the C1 semantic conflict and recommends path 1 (keep current `creates_tokens=True = permission` semantics). The Definition-of-Done bullet "No production contract is implemented against a stale or impossible invariant" (line 359) leaves Task 6 as a negative result — "do nothing, retype the ticket."

**Impact.** A negative ADR-only outcome is easy to lose in 2C handover. The F4 §Adoption State table (ADR-010 line 249–254) still names `CreatesTokensContract` as a future adopter in reviewer memory even though the plan would close it as non-producible.

**Recommendation.** ADR-015 should still land (even under path 1) and should explicitly:
1. Reject `creates_tokens` as a production DeclarationContract adopter with the argument from plan lines 342–351.
2. Name the alternative runtime-VAL mechanism that IS honest at the dispatcher surface — for example, a `MultiRowEmissionCardinality` post-emission contract that applies only to transforms whose config declares "this transform MUST emit N>1 rows for inputs matching X." If no such mechanism is obviously needed, record "no replacement needed; protocol-docs suffice."
3. Update the §Adoption State table (ADR-010 line 249–254) to remove `CreatesTokensContract` from the pre-emission adopter candidates, or annotate it "rejected per ADR-015."
4. Close the Filigree task `elspeth-cf2ee33808` with an explicit "superseded by ADR-015" comment referencing the ADR.

---

## SA-6 — MEDIUM — Reversibility posture: each adopter should have its own ADR §Consequences reversibility clause

**Finding.** ADR-010 Amendment A2 (`docs/architecture/adr/010-declaration-trust-framework.md:22-23`) already records that reversibility weakens as adopters register. The plan's adopter ADRs (011/012/013/014/015) do not inherit this concern: the plan's ADR decisions checklist (Task 2 lines 134–138, Task 3 implicit, Task 4 line 276, Task 5 line 317) names violation class, payload schema, sites, and Tier-1 posture but does not require a per-adopter reversibility statement.

**Impact.** By the end of Phase 2B the framework has 4 registered adopters (post-2B) plus 2C's paired boundary landing. At six adopters Amendment A2's prose says "practical window closes" — but each individual ADR's consequences section does not record what removing that specific adopter would entail. The review date 2026-10-19 is a governance event that needs per-adopter reversibility notes to be meaningful.

**Recommendation.** Extend each adopter ADR's required sections to include a short §Reversibility subsection that names: the scalar flip that would disable the contract (`applies_to` short-circuit), the triage-SQL signatures it introduces, any runtime attribute on `BaseTransform` the contract depends on, and whether removal requires an `EXPECTED_CONTRACT_SITES` change. This is ~5 lines per ADR and makes the 2026-10-19 reversibility checkpoint auditable.

---

## SA-7 — LOW — `can_drop_rows` contract naming conflates declaration with carve-out retirement

**Finding.** Plan Task 3 (lines 176–223) frames `CanDropRowsContract` as both a declaration-trust contract and the mechanism that retires ADR-009 Clause 3. The two are subtly different: the ADR-009 Clause 3 carve-out is a behaviour in `verify_pass_through` (empty-emission exempt); the new contract is a declaration (`can_drop_rows: bool` on `BaseTransform`) that changes `applies_to` logic.

**Evidence.** `src/elspeth/engine/executors/pass_through.py:83-84` — the current `verify_pass_through` body short-circuits on `not emitted_rows` (this is Clause 3). The plan's Task 3 does not mention deleting or tightening this short-circuit, only adding a sibling contract.

**Impact.** If the Clause 3 `if not emitted_rows: return` remains in `verify_pass_through` while `CanDropRowsContract.post_emission_check` separately raises on zero rows for `passes_through_input=True + can_drop_rows=False`, a mis-declared transform will be caught by the new contract while still benefiting from the pass-through short-circuit — correct outcome, but the two checks now cover overlapping semantics in different code paths, which is a fertile surface for future drift.

**Recommendation.** In Task 3, either (a) delete the `if not emitted_rows: return` from `verify_pass_through` when `CanDropRowsContract` lands, because the new contract now owns the empty-emission governance; or (b) explicitly document in ADR-012 why the carve-out remains and record the redundancy as defence-in-depth. The plan's Task 3 Definition of Done ("the empty-emission carve-out is retired mechanically") reads like (a) — make it mechanical in the code, not just by registering a sibling contract that overlaps it.

---

## SA-8 — LOW — Integration with 2026-07-18 SLA: the plan does not mark the SLA-critical adopter

**Finding.** ADR-009 line 70 states the hard trigger for the SLA: `can_drop_rows` must land "within 90 days of Track 1 merge, OR upon registration of a second `passes_through_input=True` transform with external-call dependencies." The plan Task 3 lands `can_drop_rows` second in the sequence but does not cite the SLA explicitly as the motivating date.

**Impact.** The sequencing is already correct (Task 3 is SLA-critical and lands second). But a reader scanning the plan for "what must ship by 2026-07-18" finds no such tag. If Phase 2B slips past Task 2, a reviewer cannot tell from the plan alone whether the slip endangers an ADR SLA.

**Recommendation.** Add a one-line annotation at the top of Task 3 — "SLA: ADR-009 Clause 3 hard trigger 2026-07-18; this task retires the carve-out" — and reference it in the Recommended Landing Order (line 401–410) so the landing sequence ties to an ADR governance date.

---

## SA-9 — LOW — "One PR per adopter" is right, but verification gates need to be non-skippable per PR

**Finding.** Plan "Execution Notes" (lines 423–427) says "Land one production adopter per PR after Task 1" and "Task 7 verification accompanies each adopter." Task 7 itself (line 362) is listed as the eighth task, suggesting batch verification at the end.

**Impact.** Under the stated sequence, an adopter PR could land and the final Task 7 benchmark/invariant re-run could detect a regression introduced by the first adopter — by then, three more adopters may have piled on top. Bisecting the regression across four adopter PRs is avoidable.

**Recommendation.** Clarify Task 7 as a per-adopter PR gate, not a terminal task. Each adopter PR must: run the Task 7 benchmark + invariant tests; update `EXPECTED_CONTRACT_SITES` in the same PR as its `register_declaration_contract` call (already required by MC3 CI, but worth re-stating); and append the adopter's row to the per-surface rule-of-three accounting. Retain the terminal Task 7 for documentation/Filigree closure only.

---

## Confidence Assessment

**High confidence on SA-1** — verified by direct grep across `src/`. The production-bootstrap gap is mechanical and reproducible.

**High confidence on SA-2** — arithmetic directly off the §Adoption State table plus the plan's adopter list.

**Medium confidence on SA-3, SA-4, SA-5** — these depend on future ADR authoring choices; my claim is that the plan does not currently force the right discipline at those joints.

**Lower confidence on SA-7** — I did not trace every path through `verify_pass_through` to confirm the short-circuit is the sole Clause 3 mechanism; there may be additional guards I missed.

## Risk Assessment

Primary risk: starting Task 2 before Task 1 lands ships `DeclaredOutputFieldsContract` on a framework whose production bootstrap is already red (SA-1). Secondary risk: merging Phase 2B without surfacing SA-2 leaves `pre_emission_check` shipping with one adopter and a rule-of-three debt that 2C inherits without attribution.

## Information Gaps

- I did not read `scripts/cicd/enforce_contract_manifest.py`; the MC3a/b/c rules' interaction with a multi-module bootstrap (SA-1) may add constraints on the Task 1 module placement.
- I did not open `tests/unit/engine/test_orchestrator_registry_bootstrap.py`; if it already exercises the no-tests-imported path, SA-1 may be partially mitigated there, though my `Grep` of `src/` is conclusive that the production module-import chain does not reach `pass_through.py`.

## Caveats

This review is scoped to the solution-architect lens. Python-level concerns (attribute placement, TypedDict ergonomics), quality-engineer concerns (Hypothesis property shape, flake surface), and security-architect concerns (payload-schema TypedDict key choices per adopter) are out of scope and will be covered by the other panel reviewers.
