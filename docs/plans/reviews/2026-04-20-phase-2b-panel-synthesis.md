# Phase 2B Declaration-Trust Plan — 5-Agent Panel Synthesis

**Date:** 2026-04-20
**Plan under review:** `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
**Plan state at review:** v1 427 lines, written grounded against commit 009b6009 (H2-B landing) on `h2-adr-010-amendment`.
**Plan state post-synthesis:** v2 1137 lines, all 42 findings incorporated with explicit ID citations.
**Synthesis authored by:** opus-4-7 (panel orchestrator), aggregating five parallel reviewer outputs.
**Mirrors:** the H2 comment #408 panel pattern on `elspeth-425047a599`.

---

## Reviewer Verdicts

| Discipline | Agent | Verdict | Load-bearing finding |
|---|---|---|---|
| Solution architect | axiom-solution-architect:solution-design-reviewer | APPROVE-WITH-CHANGES | **SA-1 CRITICAL** — production bootstrap of `PassThroughDeclarationContract` is already broken; only `tests/conftest.py:53` imports it. Task 1 is a hotfix, not hygiene. |
| Systems thinker | yzmir-systems-thinking:pattern-recognizer | APPROVE-WITH-CHANGES | **ST-1 HIGH** — `DeclaredRequiredFieldsContract` batch-gap creates a structurally-invisible coverage blind spot; audit records will imply coverage the contract never evaluated. |
| Python engineer | axiom-python-engineering:python-code-reviewer | APPROVE-WITH-CHANGES | **PY-1 HIGH** — `ExampleBundle` single-site tag leaves `batch_flush_check` path unexercised by the invariant harness on every multi-site adopter. |
| Quality engineer | axiom-sdlc-engineering:quality-assurance-analyst | APPROVE-WITH-CHANGES | **QE-1 / QE-2 CRITICAL** — every Task DoD drifts from the Track 2 epic M5 template; aggregate (M≥2) coverage absent despite every 2B adopter overlapping `PassThroughDeclarationContract`. |
| Security architect | ordis-security-architect:threat-analyst | APPROVE-WITH-CHANGES | **SEC-1 / SEC-2 / SEC-3 HIGH** — per-violation `payload_schema` TypedDicts not in ADRs; scrubber extension obligations unstated; `can_drop_rows` retires the ADR-009 Clause 3 carve-out without naming the terminal state for legitimate zero emission. |

**Verdict distribution: 5/5 APPROVE-WITH-CHANGES.** No REJECT, no unconditional APPROVE.

## Severity Distribution (42 total findings)

| Severity | Count | Share |
|---|---:|---:|
| CRITICAL | 3 | 7% |
| HIGH | 15 | 36% |
| MEDIUM | 14 | 33% |
| LOW / informational | 10 | 24% |

---

## Convergent Findings (≥2 disciplines independently surfaced)

These findings were raised by multiple reviewers from different disciplines without conferring — strong signal of load-bearing concerns.

### Convergent 1 — Bootstrap drift enforcement (SA-1, QE-5, SEC-6, ST-3)

Four reviewers independently pointed at the same failure mode via different lenses:
- **SA-1 CRITICAL:** production bootstrap is already broken; no `src/` module imports `pass_through`.
- **QE-5 HIGH:** Task 1's bootstrap test as originally described catches only positive consistency, not the "new adopter silently absent from bootstrap" failure mode.
- **SEC-6 MEDIUM:** a new adopter that forgets bootstrap + manifest BOTH is invisible to every existing CI rule.
- **ST-3 HIGH:** the `EXPECTED_CONTRACT_SITES` manifest is a shared resource with no per-PR depletion signal under concurrent development.

**Resolution in v2:** Task 1 reframed as production hotfix with (a) subprocess regression test, (b) orchestrator-top import wire-up, (c) AST drift test that scans `src/elspeth/engine/executors/` for `register_declaration_contract` call sites and asserts each is imported by the bootstrap module, (d) per-PR manifest-count assertion.

### Convergent 2 — `DeclaredRequiredFieldsContract` batch coverage gap (ST-1, SEC-4, author caveat C2)

- **ST-1 HIGH (Limits to Growth, Meadows L10):** the audit trail will contain coverage-implying records for declarations the contract never evaluated on batch-aware transforms.
- **SEC-4 MEDIUM (Repudiation):** if scoped out of batch-aware transforms, `applies_to` must crash rather than silently skip — same S2-003 pattern that forbids trivial bodies.
- **Author caveat C2:** plan flagged the gap but left the decision open.

**Resolution in v2:** Task 5 `applies_to` raises `FrameworkBugError` when passed a batch-aware transform with `declared_input_fields` populated. Silent skip explicitly rejected. ADR-013 cites ADR-010 §Adoption State 263-267 for the deferred fifth-site option.

### Convergent 3 — CreatesTokens ADR-first posture + sequencing (ST-2, SA-5, SEC-8)

- **ST-2 HIGH (Fixes that Fail, Meadows L5):** the four contracts landing before ADR-015 would each embed a `PipelineRow` assumption. Schedule ADR-015 earlier.
- **SA-5 MEDIUM:** Path-1 outcome ("no production contract") needs a positive artifact, not just a ticket retype.
- **SEC-8 LOW (informational):** ADR-first deferral is the correct Repudiation posture.

**Resolution in v2:** Task 4 = ADR-015 moved BEFORE Tasks 5-7. ADR-015 must (even under Path 1) explicitly reject, name an alternative mechanism, update §Adoption State, close the filigree task, and handle the `test_framework_accepts_second_contract.py` harness disposition (QE-10).

### Convergent 4 — Per-violation `payload_schema` + scrubber discipline (SEC-1, SEC-2, PY-3)

- **SEC-1 HIGH (Tampering / Info disclosure):** each ADR must inline its purpose-built TypedDict with `Required[...]`/`NotRequired[...]` wrappers BEFORE the first code PR.
- **SEC-2 HIGH (Info disclosure):** scrubber extension obligations unstated. Forbid `raw_schema_config`, `config_dict`, `options`, `sample_row` as payload keys.
- **PY-3 MEDIUM:** `UnexpectedEmptyEmissionPayload` is documented on the contract rather than as `payload_schema: ClassVar[type]` on the violation class — H5 Layer 1 gate bypass risk.

**Resolution in v2:** §DoD Template ADR-level checklist requires every ADR to inline the `TypedDict` schema + `§Scrubber-audit` subsection; code-level checklist requires the violation class to set `payload_schema: ClassVar[type]`.

### Convergent 5 — Aggregate-path (M≥2) coverage missing (QE-2, implicit across reviewers)

- **QE-2 CRITICAL:** every 2B adopter overlaps `PassThroughDeclarationContract` on `applies_to`; only N=1 reference-equality is tested; no aggregate E2E.
- Partially echoed by SEC-6 (defence-in-depth on aggregate triage), ST-3 (per-PR manifest-count and aggregate as shared-resource protection).

**Resolution in v2:** §DoD Template mandates an aggregate (M≥2) round-trip test per adopter; checkbox is "a row that triggers BOTH this contract AND `PassThroughDeclarationContract` produces `AggregateDeclarationContractViolation` whose `to_audit_dict()['violations']` carries both children's post-scrub payloads."

### Convergent 6 — Multi-site `ExampleBundle` blind spot (PY-1, QE-10)

- **PY-1 HIGH:** `ExampleBundle` carries one `site` tag; the plan's harness dispatches `bundle.site` without per-site iteration. Multi-site adopters get one dispatch site silently unexercised.
- **QE-10 MEDIUM:** `tests/invariants/test_framework_accepts_second_contract.py` is orphaned if CreatesTokens Path 1 is chosen.

**Resolution in v2:** §DoD Template requires multi-site adopters to return `list[ExampleBundle]` from `negative_example` OR provide `negative_example_<site>` classmethods. Task 4 (ADR-015) explicitly handles the second-shape harness disposition.

### Convergent 7 — F-QA-5 Hypothesis gap not scheduled (QE-3, PY-7)

- **QE-3 HIGH:** H2's F-QA-5 bullet is still open; 2B multiplies the gap (4 adopters × 2 sites).
- **PY-7 LOW:** plan should acknowledge in Task 7 DoD.

**Resolution in v2:** Task 1 (now "Production Bootstrap Hotfix + F-QA-5 Hypothesis Closure") lands four `@given`-driven property tests (one per dispatch surface) at `tests/property/engine/test_dispatch_bundle_properties.py`. H2 (`elspeth-425047a599`) closes in the same PR.

### Convergent 8 — VER vs VAL / DoD Template compliance (QE-1, QE-11, SEC-3)

- **QE-1 CRITICAL:** every Task DoD drifts from epic M5 template; Landscape round-trip is listed under Files but not DoD checkboxes.
- **QE-11 LOW:** Task 4's wording "payload access would later fail" conflates VER and VAL.
- **SEC-3 HIGH:** `can_drop_rows` retires the Clause-3 carve-out without naming the audit-trail terminal state for legitimate zero emission.

**Resolution in v2:** §DoD Template externalises 35+ checkboxes covering ADR, code, manifest+bootstrap, test-pyramid, process. Every Task inherits by reference. Task 7 ADR-012 names the terminal state for legitimate zero-emission rows and adds a dedicated test distinguishing from FAILED.

---

## Discipline-Specific Novel Findings

Each discipline surfaced concerns no other reviewer spotted. These add colour to the convergent findings and fill discipline-specific blind spots.

### Solution architect (SA-2, SA-3, SA-4, SA-6, SA-7, SA-8, SA-9)

- **SA-2 HIGH:** Phase 2B does NOT close rule-of-three for `pre_emission_check` (lands 1/3 adopters); §Adoption State must record provisional status.
- **SA-3 HIGH:** adding a fifth dispatch site is an ADR-010 amendment per §Adoption State 263-267, NOT a local ADR-013 decision.
- **SA-4 MEDIUM:** `declared_required_fields` collides with sink attribute name across protocols; use `declared_input_fields` for transforms.
- **SA-6 MEDIUM:** each adopter ADR needs a §Reversibility subsection.
- **SA-7 LOW:** retire the Clause-3 short-circuit in `pass_through.py` when `can_drop_rows` lands; don't leave overlapping checks in different code paths.
- **SA-8 LOW:** tag Task 7 with the ADR-009 Clause 3 2026-07-18 SLA.
- **SA-9 LOW:** Task 7 (now Task 8) is a per-PR gate, not a terminal batch.

### Systems thinker (ST-4, ST-5, ST-6)

- **ST-4 MEDIUM (Success to the Successful):** rule-of-three per-surface inadvertently favours `post_emission`/`batch_flush` because PassThrough already counts at both. Governance note in §Adoption State: "designs that plausibly fit pre-emission SHOULD prefer pre-emission."
- **ST-5 MEDIUM (Shifting the Burden, confirmed-not):** Task 1 bootstrap is the right root-cause fix; add a wire-up test that proves the orchestrator path reconstructs the manifest via transitive import (not by the test importing the bootstrap directly).
- **ST-6 LOW (Limits to Growth):** NFR derivation's N=16 worst case at 2C landing should include the 2C boundary adopters in the parametric count to avoid drift-to-theatre.

### Python engineer (PY-2, PY-4, PY-5, PY-6)

- **PY-2 HIGH:** the plan's `declared_required_fields: frozenset[str] = frozenset()` class default would tempt a `getattr(plugin, ..., frozenset())` pattern — CLAUDE.md §Offensive Programming forbids. Require `bool(plugin.declared_input_fields)` direct access.
- **PY-4 MEDIUM:** MC3b scanner is direct-body only; mixin-based multi-site adopters MUST carry `@implements_dispatch_site` on the concrete class per D1 correction.
- **PY-5 MEDIUM:** bootstrap at L2 importing L3 modules creates layer-rule risk; each adopter module must restrict imports to L0/L1/L2.
- **PY-6 LOW:** pre-existing `try/except AttributeError` in `declaration_dispatch._serialize_plugin_name` is a tempting template for adopters — explicitly forbid.

### Quality engineer (QE-4, QE-6, QE-7, QE-8, QE-9)

- **QE-4 HIGH:** per-adopter MC3a/b/c regression test missing from every adopter task's Files list. Catches the refactor case where a future author re-introduces the exact drift MC3 was built to catch.
- **QE-6 HIGH:** registry-isolation fixture (`_snapshot_registry_for_tests` / `_restore_registry_snapshot_for_tests`) pattern assumed but not named; flakiness risk from cross-test pollution.
- **QE-7 HIGH:** `can_drop_rows` test matrix incomplete — aggregate case, scoping non-fire, pass-through-exemption non-short-circuit all missing.
- **QE-8 MEDIUM:** red/green discipline named only at Task 2 in v1; silent at Tasks 3-5. Require red-phase commit SHA in PR description.
- **QE-9 MEDIUM:** benchmark live-registry re-baseline per-adopter not required; prevents silent N creep degrading the NFR.

### Security architect (SEC-5, SEC-7, SEC-9)

- **SEC-5 MEDIUM (Elevation of privilege):** Tier-1 classification per violation under-argued. Posture: Tier-1 for `DeclaredOutputFields` / `DeclaredRequiredInputFields` / `SchemaConfigMode`; Tier-2 for `CanDropRows`.
- **SEC-7 LOW (informational, Spoofing):** uniform `declared_input_fields` attribute is not a new Spoofing surface (the plugin cannot be its own witness; contract compares against caller-derived `effective_input_fields` per F1 resolution).
- **SEC-9 LOW (DoS):** each adopter's `applies_to` must be O(1); `SchemaConfigMode` is the one risk requiring explicit "single flag" body.

---

## Decision (opus-4-7 orchestrator-accepted)

All 42 findings incorporated into plan v2. No deferrals. Rationale for the absence of any deferral:

1. **3 CRITICAL findings** are each individually load-bearing; ignoring any one turns the plan into a green-CI lie or a broken-production deploy.
2. **15 HIGH findings** cluster around the DoD-template + bootstrap-drift + aggregate-coverage axes — incorporating them all costs one §DoD Template + two new test files + Task reshuffling, not architectural redesign.
3. **14 MEDIUM findings** fold cleanly into the DoD Template or into task-local constraint lists. Deferring them would cost more in scattered cleanup later than incorporating them now costs in plan length.
4. **10 LOW findings** are either informational confirmations of good plan choices or small textual improvements (SLA tag, O(1) `applies_to` requirement); each was cheap to add.

## Follow-up Actions

- **Plan v2 is the artifact.** File at `docs/plans/2026-04-20-phase-2b-declaration-trust.md`.
- **H2 closes in Task 1's PR.** F-QA-5 Hypothesis coverage lands there; no separate follow-up commit needed on the H2-B landing manifest.
- **ADR-015 must be authored BEFORE Tasks 5-7 start.** Engineer's choice between Path 1 (keep current permission semantics; no production contract) and Path 2 (tighten to must-expand; implement as Task 8). Recommendation: Path 1.
- **2026-07-18 SLA is Task 7's landing deadline.** ADR-009 Clause 3 empty-emission carve-out retirement is the SLA-gated item; re-sequence Task 7 earlier if Tasks 0-4 slip.
- **Filigree cleanup in Task 9 (cross-cutting verification)** includes `elspeth-cf2ee33808` disposition per ADR-015 outcome and the Track 2 epic (`elspeth-a3ac5d88c6`) closure audit.

## Pointers

- Plan v2: `docs/plans/2026-04-20-phase-2b-declaration-trust.md` (1137 lines)
- Individual reviews (authoritative evidence per finding):
  - `docs/plans/reviews/2026-04-20-phase-2b-solution-architect.md`
  - `docs/plans/reviews/2026-04-20-phase-2b-systems-thinker.md`
  - `docs/plans/reviews/2026-04-20-phase-2b-python-engineer.md`
  - `docs/plans/reviews/2026-04-20-phase-2b-quality-engineer.md`
  - `docs/plans/reviews/2026-04-20-phase-2b-security-architect.md`
- H2-B landing context: `docs/architecture/adr/010-declaration-trust-framework.md` Amendment A3; commit 009b6009 on `h2-adr-010-amendment`.
- Track 2 epic: `elspeth-a3ac5d88c6`.
