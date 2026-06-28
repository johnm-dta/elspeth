# Phase 2B Declaration-Trust Plan Review — Systems Thinker

**Reviewer role:** Systems pattern recognition (Meadows leverage hierarchy + Senge archetypes)
**Reviewed plan:** `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
**Branch:** `h2-adr-010-amendment` @ commit 009b6009

---

## Verdict: APPROVE-WITH-CHANGES

The plan is structurally sound and avoids the most serious archetypes. Task 1's explicit bootstrap module is a root-cause intervention, not a symptom-fix. The ADR-first gate on `CreatesTokensContract` correctly prevents a "Fixes that Fail" feedback loop. Two concerns warrant changes before coding starts: the `DeclaredRequiredFieldsContract` batch-gap creates a concealed blind spot that will compound over time (ST-1), and the `EXPECTED_CONTRACT_SITES` manifest faces a Tragedy of the Commons under concurrent 2B development that the plan does not structurally close (ST-3). These are HIGH, not CRITICAL, because the plan flags them as caveats — but flagging is not the same as resolving.

---

## Findings

### ST-1 — HIGH
**Archetype / leverage level:** Limits to Growth, Meadows L10 (Physical structure of the system)

**Finding:** The pre-emission dispatch site has no batch-pre-execution counterpart, so `DeclaredRequiredFieldsContract` provably does not cover the fastest-growing adopter category (aggregation-mode batch-aware transforms), and the coverage gap is structurally invisible to operators querying the audit trail.

**Evidence:** Plan lines 255-260 acknowledge the gap and recommend deferring to option 1 ("scope the contract to non-batch transform execution"). ADR-010 §Adoption State shows `pre_emission_check` at 0 adopters — the first adopter landing here sets the coverage expectation for downstream readers. `BatchFlushInputs` carries `buffered_tokens: tuple` not a pre-execution equivalent, confirming the physical structural absence. Once `DeclaredRequiredFieldsContract` registers at `pre_emission_check` only, every batch-aware transform carries an audit record implying "required fields checked" that the contract never actually evaluated for batch invocations.

**Recommendation:** ADR-013 must state explicitly in its "Scope" section that the contract fires only on single-row dispatch paths, and the `EXPECTED_CONTRACT_SITES` manifest entry for `declared_required_fields` must carry a `# NOTE: batch-pre-execution site absent` inline comment. This prevents the next contributor from reading the manifest and inferring full coverage. The structural fix (adding a 5th dispatch site) is correctly deferred; the audit-trail deception risk is not deferred — it is closed by making the scope gap machine-readable.

---

### ST-2 — HIGH
**Archetype / leverage level:** Fixes that Fail, Meadows L5 (Rules / incentive structures)

**Finding:** Deferring `CreatesTokensContract` pending ADR-015 is correct, but the four contracts landing before it all use `post_emission_check` on `emitted_rows: tuple[PipelineRow, ...]` — and none of them will verify child `TokenInfo` identity. This creates a progressive encoding of the assumption "post-emission rows are PipelineRow objects, not TokenInfo children" into the dispatch surface, which ADR-015 path 2 would need to reverse.

**Evidence:** Plan lines 339-347 describe path 2 requiring `docs/contracts/plugin-protocol.md` + processor test updates. `PostEmissionOutputs.emitted_rows: tuple[Any, ...]` (declaration_contracts.py lines 225-238) accepts whatever the executor passes. The dispatch surface sees `PipelineRow` today per plan line 33 (C1 caveat). As Tasks 2-5 land three more contracts against this surface, the "PipelineRow assumption" accumulates at 4 call sites, not 1. Each additional adopter raises the cost of ADR-015 path 2 and indirectly incentivises taking path 1 not on the merits but to avoid refactoring N adopters.

**Recommendation:** Task 6's ADR-015 should be ordered before Task 5 (SchemaConfigMode), not after. This ensures the semantic question is resolved before 4 adopters have embedded the PipelineRow assumption. The plan currently orders Task 6 last; the only rationale given (plan line 409) is "only after ADR-015 resolves" — that same reasoning argues for scheduling the ADR decision earlier in the sequence, not later.

---

### ST-3 — HIGH
**Archetype / leverage level:** Tragedy of the Commons, Meadows L6 (Information flows)

**Finding:** `EXPECTED_CONTRACT_SITES` (declaration_contracts.py lines 818-828) is a shared manifest that four concurrent 2B adopter tickets must each modify in the same commit as their registration. Under simultaneous branch development, two authors can each land a correct per-ticket manifest update in isolation while creating a merge conflict that silently drops one entry — and the bootstrap's set-equality assertion will catch the drop only if the bootstrap test runs post-merge.

**Evidence:** ADR-010 §H2 landing scope mandates "same commit" for manifest + registration + decorator markers (line 812-814 inline comment on the closed set). Plan lines 404-410 prescribe "one production adopter per PR" which partially addresses this. But the `MC3a`/`MC3b`/`MC3c` CI rules are AST scanners on source tree state; they fire on the post-merge tree, not on the merge conflict itself. The bootstrap test at `test_orchestrator_registry_bootstrap.py` is the only runtime gate, and it runs only if the merge is clean.

**Recommendation:** The plan should add an explicit rule: each adopter PR must include an integration-test assertion that exactly `N` contracts appear in `EXPECTED_CONTRACT_SITES` after the merge (where `N` is the cumulative count at that PR's landing position in the sequence). This turns the manifest from a shared resource with no depletion signal into one with a monotonically-checked gate per landing. This is a Meadows L6 intervention (information structure): replace after-the-fact set-equality check with per-PR count assertions that detect collisions before they become silent drops.

---

### ST-4 — MEDIUM
**Archetype / leverage level:** Success to the Successful, Meadows L5 (Rules)

**Finding:** The rule-of-three per dispatch site inadvertently advantages `post_emission_check` and `batch_flush_check` adopters, because `PassThroughDeclarationContract` already counts once at each of those sites. A new adopter at either of those sites needs only 2 additional registrations to satisfy rule-of-three, whereas `pre_emission_check` needs 3 from zero. This creates a structural incentive to route contracts toward the already-populated sites.

**Evidence:** ADR-010 §Adoption State table (lines 249-254): `post_emission_check` at 1, `batch_flush_check` at 1, `pre_emission_check` at 0, `boundary_check` at 0. Plan Task 2 and Task 3 both target `post_emission_check` and `batch_flush_check`, which is architecturally correct for those contracts — but the incentive asymmetry means a future contract author designing a contract that plausibly fits either `pre_emission_check` or `post_emission_check` will face implicit pressure toward the latter.

**Recommendation:** This is low-urgency because the 2B contracts are correctly matched to their semantically appropriate sites and the plan's author explicitly targeted `pre_emission_check` for Task 4. Record this asymmetry in ADR-010 §Adoption State as a governance note: "the pre-emission site's zero baseline means the rule-of-three gate provides weaker early coverage here — designs that plausibly fit pre-emission SHOULD prefer pre-emission to build the site's evidential weight." This is a Meadows L6 (information) intervention requiring one sentence in an existing ADR section.

---

### ST-5 — MEDIUM
**Archetype / leverage level:** Shifting the Burden, Meadows L5 (Rules)

**Finding:** Task 1's explicit bootstrap module is correctly identified as a root-cause fix — this is the OPPOSITE of Shifting the Burden and should be confirmed as such. However, the plan's "suggested code shape" (lines 103-108) places the bootstrap file inside `engine/executors/` rather than at the orchestrator layer, which means the fundamental solution (guaranteed registration before freeze) still depends on the orchestrator importing the bootstrap file at the right time. If the import is ever moved or removed, the explicit bootstrap silently reverts to incidental imports with no structural enforcement.

**Evidence:** Plan lines 88-117. `orchestrator/core.py` `prepare_for_run()` must import the bootstrap module — but no test in the definition-of-done asserts that removing the bootstrap import fails the bootstrap test without the test itself already importing production modules. The bootstrap test (plan line 111-113) clears the registry, imports only the bootstrap module, and asserts the manifest is restored — this test is correct and sufficient IF it is also run as part of the orchestrator test suite, not only as a standalone test.

**Recommendation:** The bootstrap test definition should require that it also passes when the orchestrator module is imported fresh (i.e., the test imports `orchestrator.core` rather than the bootstrap module directly), confirming the wire-up. This closes the "bootstrap exists but isn't wired" failure mode.

---

### ST-6 — LOW
**Archetype / leverage level:** Limits to Growth, Meadows L10 (Physical structure)

**Finding:** The NFR derivation's parametric benchmark (`N ∈ {1, 2, 4, 8, 16}`) was written against a single-pass-through registry. Phase 2B adds 3-4 contracts; each claims at most 2 dispatch sites. At N=4 dispatched per site (worst-case after 2B), the per-skip budget is 27 + 3 × 1.5 = 31.5 µs median. This is well within the ADR-008 50 µs P99 gate. At N=8 (2B + early 2C), the gate is 27 + 7 × 1.5 = 37.5 µs. The growth law is linear and the budget is intact through the 2026-10-19 review date.

**Evidence:** ADR-010 §NFR derivation lines 107-120. Plan Task 7 preserves the benchmark. No concern at 2B scale. The risk becomes relevant only if Phase 2C lands both source and sink boundary adopters (adding 2 more) and a second pass of 2B contracts for batch-pre-execution (adding another 3). That scenario would approach N=16 and reach 49.5 µs, leaving almost no headroom. The 2026-10-19 review date is the correct structural gate.

**Recommendation:** No change needed for 2B. Record in Task 7's definition-of-done that the benchmark's worst-case N=16 scenario at the review date should include the 2C boundary adopters in the parametric count, so the benchmark does not drift into theatre as 2C lands.

---

## Confidence Assessment

**High confidence (grounded in code):** ST-1 (structural absence of batch-pre-execution site is observable in declaration_contracts.py), ST-3 (manifest single-file shared resource confirmed in declaration_contracts.py line 818), ST-5 (bootstrap wiring gap is structurally present).

**Medium confidence (inferred from plan + ADR, not from runtime behaviour):** ST-2 (PipelineRow assumption accumulation), ST-4 (rule-of-three incentive asymmetry).

**Low confidence (scaling projection):** ST-6 (NFR extrapolation to 2C landing is speculative).

---

## Risk Assessment

**Highest risk:** ST-1 — because the audit trail will contain coverage-implying records for declarations the contract never evaluated. This is not a crash risk; it is an auditability risk, which is the project's stated highest concern.

**Second highest:** ST-3 — manifest drift under concurrent development can silently disable a contract post-merge without any failing test until the next pipeline run.

---

## Information Gaps

1. The actual `BatchTransformMixin` registration count is unknown. If no registered transforms are `is_batch_aware=True` before 2026-07-18, ST-1's practical risk is lower than assessed.
2. Whether the `MC3a`/`MC3b`/`MC3c` CI scanner runs on post-merge tree or per-PR is not visible from the plan. If it runs on every PR including merge commits, ST-3's risk is partially mitigated.
3. The `can_drop_rows` ADR-009 Clause 3 carve-out SLA (2026-07-18) depends on Task 3 landing. The plan's task ordering places Task 3 at position 4. If Tasks 0-2 slip, the SLA may be missed by the time Task 3 starts — the plan does not give Task 3 independent SLA tracking.

---

## Caveats

- This review did not execute tests. Assessment of "the bootstrap test is correct" is based on the definition-of-done description, not observed test output.
- The author-flagged caveats C1-C5 are all substantively addressed by the plan; this review finds no cases where a caveat was acknowledged but the plan's response was architecturally insufficient, except ST-1's batch-gap framing.
- Pattern names used are Meadows / Senge framework labels, not rigid diagnoses. The "Tragedy of the Commons" framing for ST-3 requires two agents sharing a resource with no depletion signal; under "one PR per adopter" discipline (plan line 425) the tragedy may not materialise in practice. The recommendation stands regardless because the structural protection costs one sentence per PR.
