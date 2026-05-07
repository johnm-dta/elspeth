# Phase 2B Declaration-Trust Implementation Plan

> **Revision history.** v1 2026-04-20 (initial). v2 2026-04-20 incorporates the
> 5-agent panel review (solution architect, systems thinker, python engineer,
> quality engineer, security architect). Findings resolved inline; IDs cited in
> `(SA-n / ST-n / PY-n / QE-n / SEC-n)` tags. Individual reviews live at
> `docs/plans/reviews/2026-04-20-phase-2b-{solution-architect,systems-thinker,python-engineer,quality-engineer,security-architect}.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan task-by-task.

**Goal.** Land Phase 2B transform-level declaration VAL adopters on the
ADR-010 framework, plus the `can_drop_rows` governance contract, without
regressing audit completeness, dispatch performance, or runtime/schema
attribution. Close out the production-bootstrap hotfix (Task 1) and the
F-QA-5 Hypothesis follow-up before any adopter lands.

**Architecture.** Phase 2B extends the already-landed four-site declaration
dispatcher rather than introducing new ad hoc cross-check paths. Each
production adopter lives in its own module, registers through the existing
registry/manifest mechanism, exposes a purpose-built violation payload, and
passes the full Definition-of-Done template (§DoD Template below, derived
from Track 2 epic `elspeth-a3ac5d88c6` finding M5). Two items require
explicit ADR decisions before coding proceeds: the true semantics of
`creates_tokens` (ADR-015, Task 4) and the exact runtime surface of the
schema-mode contract (ADR-014, Task 6). ADR-015 is now scheduled BEFORE the
remaining multi-site adopters so the `PipelineRow` assumption cannot
accumulate across four contracts before the semantic question resolves
(ST-2).

**Tech Stack.** Python, pytest, Hypothesis, mypy, ruff, Filigree, ADR-010
declaration-contract registry/dispatcher, Landscape integration tests.

**Prerequisites.**
- Work on a branch/worktree that already contains the H2/F2/N1/N3 framework
  landing in code (commit 009b6009 on `h2-adr-010-amendment`).
- Use current framework entry points in
  `src/elspeth/contracts/declaration_contracts.py` and
  `src/elspeth/engine/executors/declaration_dispatch.py`; do NOT add new
  executor-local verification helpers.
- Before implementing any adopter, re-read ADR-010 and verify current repo
  reality with targeted tests rather than relying on ticket descriptions.

---

## Reality Anchors

Repo facts the implementation MUST treat as authoritative.

1. **Four-site dispatcher is live.** `pre_emission_check`,
   `post_emission_check`, `batch_flush_check`, and `boundary_check` are
   present in `src/elspeth/contracts/declaration_contracts.py`.
   `TransformExecutor` wires `run_pre_emission_checks(...)` before
   `transform.process()` (transform.py:276) and
   `run_post_emission_checks(...)` after success (transform.py:403).
   `RowProcessor._cross_check_flush_output()` routes through
   `run_batch_flush_checks(...)`.

2. **Pre-emission hot-path regression is real until the first pre-emission
   adopter lands.** Today there are no `pre_emission_check` adopters in
   `EXPECTED_CONTRACT_SITES`. Phase 2B either lands the first adopter (Task
   5) promptly or keeps the temporary hot-path guard local until that
   adopter merges.

3. **`creates_tokens` semantics are inconsistent across tracked artifacts.**
   Filigree task + proof contract assume `creates_tokens=True` means "must
   expand"; current protocol docs treat it as a permission flag
   (`success()` is valid). Post-emission dispatch sees `PipelineRow`
   emissions, not child `TokenInfo` — so the "`token_id` must differ"
   invariant is not directly implementable at the existing dispatch
   surface. **Task 4 resolves this via ADR-015 before the remaining
   multi-site adopters land (ST-2).**

4. **Transform-side required fields are not normalised onto a runtime
   attribute.** Sinks expose `declared_required_fields`
   (`plugin_protocols.py:567`); transforms expose `required_input_fields`
   via config models, not a uniform `BaseTransform` attribute. **Phase 2B
   standardises a NEW attribute name `declared_input_fields` (not
   `declared_required_fields`) to avoid cross-protocol naming collision
   (SA-4).** See Task 5.

5. **`SchemaConfig.mode` has three values, not four:** `fixed` / `flexible`
   / `observed`. `locked` is a runtime `SchemaContract` property derived
   from mode + first-row inference, not a fourth config-mode enum.

6. **Batch-aware transforms matter (ST-1, SEC-4).** `DeclaredOutputFieldsContract`,
   `SchemaConfigModeContract`, and `can_drop_rows` all need both
   `post_emission_check` and `batch_flush_check` coverage. There is no
   batch-pre-execution dispatch site, so `DeclaredRequiredFieldsContract`
   cannot honestly claim batch-aware coverage without either adding a
   fifth site (requires ADR-010 amendment per §Adoption State lines
   263-267 — SA-3) or scoping the ADR to non-batch transforms. Phase 2B
   chooses scoping; Task 5 MUST fail-closed (raise `FrameworkBugError`)
   when a batch-aware transform declares `declared_input_fields`, NOT
   silently return False from `applies_to` (SEC-4). The ST-1 audit-trail
   risk — batch-aware transforms producing records implying coverage
   that the contract never evaluated — is the reason silent-skip is
   rejected: the audit trail must not carry an "apparent coverage"
   signature for a check the framework structurally cannot run.

7. **Production bootstrap of `PassThroughDeclarationContract` is currently
   broken outside pytest (SA-1, CRITICAL).** `grep` across `src/` confirms
   no module imports `elspeth.engine.executors.pass_through`. Registration
   today depends on `tests/conftest.py:53`. Any production-like invocation
   of `Orchestrator.run()` / `prepare_for_run()` outside the pytest process
   raises `RuntimeError` at the N1 per-site manifest assertion. **Task 1 is
   a production hotfix, not hygiene; nothing else can ship until it lands.**

8. **`pre_emission_check` rule-of-three does NOT close in Phase 2B (SA-2,
   HIGH).** After Phase 2B completes, per-surface adopter counts are:

   | Surface | Post-2B | Rule-of-three closed? |
   |---|---|---|
   | `post_emission_check` | 4 (PassThrough, DeclaredOutputFields, CanDropRows, SchemaConfigMode) | YES |
   | `batch_flush_check` | 4 (same three add batch coverage) | YES |
   | `pre_emission_check` | 1 (DeclaredRequiredFields) | **NO — 2 more needed (2C scope)** |
   | `boundary_check` | 0 | NO (2C paired landing, correct) |

   Task 8 updates ADR-010 §Adoption State to record the pre-emission
   surface as "provisional until two further pre-emission adopters
   register." Do NOT silently ship a single-adopter dispatch surface.

9. **Surface asymmetry is a governance note (ST-4).** PassThrough counts
   toward both post_emission and batch_flush, creating an incentive pull
   toward those surfaces. Task 8 records in §Adoption State:
   "the pre-emission site's zero baseline means the rule-of-three gate
   provides weaker early coverage here — designs that plausibly fit
   pre-emission SHOULD prefer pre-emission to build the site's evidential
   weight."

10. **DoD template compliance is mandatory per adopter (QE-1).** See
    §Definition-of-Done Template below. Every Task DoD MUST tick those
    boxes. Green-path VER outputs alone are not sufficient.

---

## Definition-of-Done Template (Epic M5 compliance)

Every task landing a production adopter (Tasks 3, 5, 6, 7) MUST satisfy,
in the same PR, every box below. This template consolidates the panel's
ratified VAL requirements plus the inherited Track 2 epic DoD. Tasks
inherit this template by reference; each task's local DoD adds only
task-specific bullets on top.

**ADR-level (SEC-1, SEC-2, SEC-5, SA-6):**
- [ ] ADR-NNN lands and names: violation class, `payload_schema` TypedDict
      inline (every key annotated `Required[...]` / `NotRequired[...]`),
      dispatch sites claimed, Tier-1 classification + justification,
      `§Reversibility` subsection (scalar flip to disable, triage-SQL
      signatures introduced, `BaseTransform` runtime attribute
      dependency if any, manifest change required to remove), and
      `§Scrubber-audit` subsection (every payload key whose value could
      be sourced from plugin config or row data, confirmed covered by
      `secret_scrub._PATTERNS` / `_SECRET_KEY_NAMES` OR scrubber extended
      in the same PR). Forbidden payload keys: `raw_schema_config`,
      `config_dict`, `options`, `sample_row` (open-ended mapping sinks —
      values must be structural: field-name sets, mode strings, bool
      flags, counts).

**Code-level (PY-2, PY-3, PY-4, PY-5, SEC-9):**
- [ ] Violation class subclasses `DeclarationContractViolation` with
      `payload_schema: ClassVar[type] = <TypedDict>`. Base `__init__`
      applies `deep_freeze` and `_validate_payload_against_schema`
      automatically; do not re-implement.
- [ ] `applies_to` uses direct attribute access (no `getattr` default, no
      `hasattr`, no `try/except AttributeError` — follow
      `pass_through.py:141-145` not `declaration_dispatch._serialize_plugin_name`
      at line 88-94, which is a pre-existing defensive pattern adopters
      MUST NOT model on — PY-6).
- [ ] Every overriding dispatch method decorated with
      `@implements_dispatch_site("<site>")` ON THE CONCRETE CLASS, even
      if the body is inherited from a mixin (D1 correction — the AST
      scanner does not resolve MRO).
- [ ] `applies_to` body is O(1) in plugin attribute reads — no regex, no
      reflection, no nested `getattr` walks.
- [ ] Adopter module imports L0/L1/L2 types only. No L3 imports even
      under `TYPE_CHECKING` unless separately justified against
      `scripts/cicd/enforce_tier_model.py`. The bootstrap module remains
      at L2; it imports only adopter modules (side-effect) whose own
      imports follow this rule.

**Manifest + bootstrap same PR (QE-4, SEC-6, ST-3):**
- [ ] `EXPECTED_CONTRACT_SITES[<name>] = frozenset({<site>, ...})` added
      at `src/elspeth/contracts/declaration_contracts.py`.
- [ ] `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
      imports the new module.
- [ ] Bootstrap AST-drift test (Task 1) passes — asserts every
      `register_declaration_contract(...)` call site under
      `src/elspeth/engine/executors/` has a matching import line in the
      bootstrap module. This catches the "new adopter forgets the
      bootstrap import" failure mode; QE-5 rationale.
- [ ] Per-PR manifest-count assertion: the adopter PR includes an
      integration-test assertion that exactly N contracts appear in
      `EXPECTED_CONTRACT_SITES` after merge (ST-3 depletion signal).
- [ ] MC3a/b/c regression test: one MC3a case (marker without manifest) +
      one MC3b case (manifest without marker) per new contract lands in
      `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`.

**Test pyramid (QE-1, QE-2, QE-6, QE-9, QE-10, PY-1):**
- [ ] Unit (VER): `negative_example` fires expected violation;
      `positive_example_does_not_apply` confirms `applies_to` returns
      False (N2 Layer A non-fire).
- [ ] Non-fire harness: contract registered in
      `tests/invariants/test_contract_non_fire.py` coverage.
- [ ] N=1 reference-equality test: dispatcher raises `violations[0]`
      identity-preserved (type + `id()`) when only this contract fires.
- [ ] **Aggregate (M>=2) round-trip test (VAL):** a row triggers BOTH
      this contract AND `PassThroughDeclarationContract`; dispatcher
      raises `AggregateDeclarationContractViolation` whose
      `to_audit_dict()['violations']` carries both children's
      post-scrub payloads. `except DeclarationContractViolation` does
      NOT absorb the aggregate (sibling-class invariant / S2-001).
- [ ] **E2E Landscape round-trip test (VAL):** mis-declaration →
      dispatcher → `AuditEvidenceBase.to_audit_dict` → Landscape persisted
      row → `explain(recorder, run_id, token_id)` recovers the violation
      shape with scrubbed payload.
- [ ] Per-site `ExampleBundle` coverage for multi-site adopters: EITHER
      `negative_example` returns `list[ExampleBundle]` OR a dedicated
      `negative_example_<site>` classmethod exists per claimed site.
      Harness exercises every claimed site (PY-1; prevents batch-flush
      blind spot).
- [ ] Registry isolation: every new `tests/unit/engine/test_*_contract.py`
      uses `_snapshot_registry_for_tests` /
      `_restore_registry_snapshot_for_tests` wrapped in an
      `_isolate_both_registries`-equivalent fixture. Direct
      `_clear_registry_for_tests()` without snapshot/restore is a
      review-blocker (S2-008).
- [ ] Live-registry benchmark re-baseline:
      `test_dispatcher_overhead_vs_direct_verify_pass_through` re-run
      post-registration; either passes under 27µs median / 54µs P99 OR
      ADR-010 §NFR derivation is updated in the same PR.

**Process (QE-8):**
- [ ] Red-phase commit SHA recorded in PR description. Unit test was
      observed failing with the expected-shape violation before the
      implementation commit landed.

---

## Task 0: Verify Framework Prerequisites and Reconcile Board vs Workspace

**Files (read-only):**
- `docs/architecture/adr/010-declaration-trust-framework.md`
- `src/elspeth/contracts/declaration_contracts.py`
- `src/elspeth/engine/executors/declaration_dispatch.py`
- `src/elspeth/engine/executors/transform.py`
- `src/elspeth/engine/processor.py`
- `tests/unit/engine/test_declaration_dispatch.py`
- `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
- `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py`

**Step 1 — Framework smoke tests:**

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_declaration_dispatch.py \
  tests/unit/engine/test_orchestrator_registry_bootstrap.py \
  tests/integration/audit/test_declaration_contract_landscape_roundtrip.py
```

**Expected:** all pass on the current branch. If they do not, stop Phase 2B
and repair the framework landing first.

**Step 2 — Board/workspace reconciliation:**
- Confirm the H2-B landing sibling closures (N1, N3, F2, F3, F4, F5) are
  reflected in filigree. H2 (`elspeth-425047a599`) itself remains open
  pending F-QA-5 Hypothesis closure — see Task 1.
- Confirm no 2B adopter ticket is blocked on stale H2-B dependencies; add
  comments citing file/test references where filigree still carries
  closed-in-code acceptance criteria.

**Definition of Done:**
- [ ] Framework smoke tests green.
- [ ] Board state matches workspace state; drift documented, not silently
      worked around.
- [ ] No 2B contract work starts on top of an unverified framework
      foundation.

---

## Task 1: Production Bootstrap Hotfix + F-QA-5 Hypothesis Closure

> **SA-1 (CRITICAL) reframing:** production bootstrap of
> `PassThroughDeclarationContract` is already broken outside pytest. No
> module under `src/` imports `elspeth.engine.executors.pass_through`; the
> only import is at `tests/conftest.py:53`. Any production-like invocation
> of `prepare_for_run()` fails at the N1 set-equality assertion. This task
> is a pre-Task-2 hotfix.
>
> **QE-3, PY-7:** Phase 2B multiplies the F-QA-5 gap (four new adopters ×
> two sites) if left unaddressed. Close it here as a framework-layer
> prerequisite so adopter tasks inherit the property-test surface.

### Files
- **Create:** `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
- **Modify:** `src/elspeth/engine/orchestrator/core.py` — import bootstrap
  at module top (the orchestrator is the canonical bootstrap entry point
  per ADR-010 §Decision 3 prose; SA-1 recommendation).
- **Modify:** `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
- **Create:** `tests/unit/engine/test_declaration_contract_bootstrap_drift.py`
- **Create:** `tests/property/engine/test_dispatch_bundle_properties.py`
  (F-QA-5 per-surface Hypothesis coverage)
- **Modify:** `docs/architecture/adr/010-declaration-trust-framework.md` —
  amendment banner noting F-QA-5 closure

### Bootstrap module shape

```python
# src/elspeth/engine/executors/declaration_contract_bootstrap.py
"""Authoritative import surface for production DeclarationContract registrations.

Every production contract module with a module-level
``register_declaration_contract(...)`` call site MUST be imported here. The
drift test at ``tests/unit/engine/test_declaration_contract_bootstrap_drift.py``
AST-scans this file against ``src/elspeth/engine/executors/`` to fail CI when
a new contract module lands without its bootstrap import.

CLOSED SET — adding or removing an entry requires updating
``EXPECTED_CONTRACT_SITES`` in the same commit.
"""

import elspeth.engine.executors.pass_through  # noqa: F401
# Add future Phase 2B/2C adopters below as they land.
```

### Wire-up

`src/elspeth/engine/orchestrator/core.py` imports the bootstrap module at
module top (NOT from `engine/executors/__init__.py` — the orchestrator is
the canonical bootstrap entry point, per ADR-010 §Decision 3):

```python
# Top of orchestrator/core.py
import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401
```

### Tests

**1. Subprocess production-shape regression (SA-1):** spawn a subprocess
that executes `python -c "from elspeth.engine.orchestrator import
prepare_for_run; prepare_for_run()"` with NO pytest context. Assert exit
code 0. This catches the current broken state and prevents regression.

**2. Bootstrap wire-up test (ST-5):** the test MUST NOT import the
bootstrap module directly; it imports `orchestrator.core` fresh and
asserts the manifest is restored. This proves the wire-up is real, not
that the bootstrap file alone is consistent.

**3. AST drift test (QE-5, SEC-6) — the core mechanical enforcement:**

```python
# tests/unit/engine/test_declaration_contract_bootstrap_drift.py
def test_every_registration_has_matching_bootstrap_import():
    """AST-scan src/elspeth/engine/executors/ for
    register_declaration_contract(...) call sites. For each call site's
    defining module, assert declaration_contract_bootstrap.py carries a
    matching `import elspeth.engine.executors.<module>` line.

    Fails if any adopter module lands without its bootstrap import.
    """
```

**4. Manifest-count per-PR assertion (ST-3):** every adopter PR's test
suite asserts `len(EXPECTED_CONTRACT_SITES) == <expected_cumulative>` at
the point of that landing. Catches merge-conflict silent drops before
production bootstrap runs.

**5. F-QA-5 per-surface Hypothesis property tests (QE-3, PY-7):**

```python
# tests/property/engine/test_dispatch_bundle_properties.py
"""One @given property per dispatch surface, proving each bundle type's
Hypothesis strategy can be derived without conditional ``assume(x is not
None)`` guards. Closes H2's final F-QA-5 §Acceptance bullet.
"""

@given(inputs=builds(PreEmissionInputs, ...))
def test_pre_emission_bundle_derivable(inputs): ...

@given(inputs=builds(PostEmissionInputs, ...), outputs=builds(PostEmissionOutputs, ...))
def test_post_emission_bundle_derivable(inputs, outputs): ...

@given(inputs=builds(BatchFlushInputs, ...), outputs=builds(BatchFlushOutputs, ...))
def test_batch_flush_bundle_derivable(inputs, outputs): ...

@given(inputs=builds(BoundaryInputs, ...), outputs=builds(BoundaryOutputs, ...))
def test_boundary_bundle_derivable(inputs, outputs): ...
```

Any future bundle that acquires a nullable field or unannotated `Any` that
forces an `assume(...)` guard fails these tests immediately.

### Definition of Done
- [ ] Subprocess regression test passes; the bootstrap can reach every
      contract module without pytest.
- [ ] Orchestrator imports the bootstrap module at module top.
- [ ] AST drift test proves any future silent-drop between adopter
      creation and bootstrap update fails CI.
- [ ] Wire-up test proves the orchestrator path reconstructs the manifest
      via transitive import (not by the test importing the bootstrap
      itself).
- [ ] All four dispatch-site bundle types pass `@given`-based property
      tests; H2 (`elspeth-425047a599`) F-QA-5 acceptance bullet closed
      via a comment on H2 citing the new test file, and H2 itself is
      closed in the same PR.
- [ ] ADR-010 receives an amendment banner noting F-QA-5 closure.

---

## Task 2: Pre-Adopter Gate — DoD Template & Panel Compliance Check

This is a one-sitting review task, not a code task. Before Task 3 starts,
confirm the §Definition-of-Done Template (above) is internalised:

- [ ] Every reviewer of Tasks 3-7 has read the template.
- [ ] PR template for adopter PRs carries the full DoD checklist (not a
      subset).
- [ ] CI hook (optional but recommended) greps adopter PRs for all
      template boxes before allowing merge.

No code changes. Output: a CHANGELOG entry or PR-template addition
documenting the template.

---

## Task 3: ADR-011 — `DeclaredOutputFieldsContract` (Rule-of-Three Gate)

First production non-pass-through adopter. Validates the rule-of-three
gate on `post_emission_check` AND `batch_flush_check`.

### Files
- **Create:** `docs/architecture/adr/011-declared-output-fields-contract.md`
- **Create:** `src/elspeth/engine/executors/declared_output_fields.py`
- **Modify:** `src/elspeth/contracts/errors.py` (violation class + `@tier_1_error`)
- **Modify:** `src/elspeth/contracts/declaration_contracts.py`
  (`EXPECTED_CONTRACT_SITES` entry)
- **Modify:** `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
  (add import)
- **Modify:** `tests/invariants/test_contract_negative_examples_fire.py`
  (register new contract in non-fire / negative-example coverage)
- **Modify:** `tests/invariants/test_contract_non_fire.py`
- **Modify:** `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
- **Modify:** `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`
  (MC3a + MC3b regression fixtures — QE-4)
- **Create:** `tests/unit/engine/test_declared_output_fields_contract.py`
- **Create:** `tests/integration/audit/test_declared_output_fields_roundtrip.py`

### ADR-011 required content

In addition to the §DoD Template ADR-level checklist:

- **Tier-1 classification:** Tier-1 (SEC-5). A misdeclared output schema
  corrupts downstream lineage; `on_error` routing MUST NOT absorb.
- **Payload schema (SEC-1):**

```python
class DeclaredOutputFieldsPayload(TypedDict):
    declared: list[str]          # frozenset sorted for canonical audit
    runtime_observed: list[str]
    missing: list[str]
```

Forbidden: `raw_schema_config`, `config_dict`, `sample_row` (SEC-2).

- **Runtime invariant:**

```python
runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)
runtime_payload_fields = frozenset(emitted.keys())
runtime_observed = runtime_contract_fields & runtime_payload_fields
missing = frozenset(plugin.declared_output_fields) - runtime_observed
# Raise iff missing is non-empty.
```

- **Dispatch sites:** `post_emission_check` AND `batch_flush_check` (both
  decorated on the concrete class per D1 — PY-4).
- **§Reversibility:** flip `applies_to` to always return False; remove
  manifest entry + bootstrap import + violation class; no `BaseTransform`
  attribute change (uses existing `declared_output_fields` from
  `base.py:199`).
- **§Scrubber-audit:** all three payload keys carry field-name lists (no
  secret exposure); no scrubber extension needed.

### Tests

**Red phase first (QE-8):** land a failing unit test with a transform
declaring `declared_output_fields={"new_a", "new_b"}` but emitting only
`new_a`; expect `DeclaredOutputFieldsViolation` on both single-row and
batch-flush paths. Record the red-phase commit SHA in the PR description.

**Multi-site ExampleBundle (PY-1):** `DeclaredOutputFieldsContract`
provides two classmethods:

```python
@classmethod
def negative_example(cls) -> ExampleBundle:    # POST_EMISSION variant
@classmethod
def negative_example_batch_flush(cls) -> ExampleBundle:  # BATCH_FLUSH variant
```

The harness iterates both. Same pattern for Tasks 5 / 6.

**Aggregate round-trip (QE-2):** a `passes_through_input=True` transform
that ALSO mis-declares output fields triggers BOTH this contract AND
`PassThroughDeclarationContract`; assert `AggregateDeclarationContractViolation`
with both children's audit dicts in `violations`.

**E2E Landscape round-trip (QE-1):** follows the pattern at
`tests/integration/audit/test_declaration_contract_landscape_roundtrip.py::TestDeclarationContractViolationRoundTrip`.

**MC3a / MC3b regression fixtures (QE-4):** fail CI when:
- manifest lists `declared_output_fields` but the class drops the
  `@implements_dispatch_site` marker (MC3b)
- the class retains the marker but the manifest entry is removed (MC3a)

### Green verification

```bash
PYTHONPATH=src uv run pytest -q \
  tests/unit/engine/test_declared_output_fields_contract.py \
  tests/integration/audit/test_declared_output_fields_roundtrip.py \
  tests/unit/scripts/cicd/test_enforce_contract_manifest.py \
  tests/invariants/test_contract_negative_examples_fire.py \
  tests/invariants/test_contract_non_fire.py \
  tests/unit/engine/test_orchestrator_registry_bootstrap.py \
  tests/unit/engine/test_declaration_contract_bootstrap_drift.py
```

### Definition of Done
- [ ] Full §DoD Template passes.
- [ ] Both single-row and batch-flush misdeclarations fail with scrubbed,
      attributable audit records.
- [ ] Aggregate round-trip with PassThrough passes (QE-2).
- [ ] Rule-of-three gate for `post_emission_check` and `batch_flush_check`
      now advances to 2/3 each (toward closure at 3/3).

---

## Task 4: ADR-015 — `CreatesTokens` Semantic Resolution (ADR-first)

> **Moved up from Task 6 per ST-2.** Schedule ADR-015 BEFORE the remaining
> multi-site adopters land so Tasks 5-6 do not each embed a `PipelineRow`
> assumption that path-2 resolution would need to reverse across four
> contracts.

This is an ADR-only task. No production code ships here unless path 2 is
chosen; in that case, the code lands as Task 7.

### Files
- **Create:** `docs/architecture/adr/015-creates-tokens-contract.md`
- **Modify:** `docs/architecture/adr/010-declaration-trust-framework.md`
  (§Adoption State table update; ST-4 governance note)
- **Comment/close:** filigree `elspeth-cf2ee33808`

### Decision

Choose one of two semantically honest paths:

**Path 1 — Keep current protocol semantics (RECOMMENDED).**
`creates_tokens=True` means "multi-row expansion is permitted," not
"required." Rationale: current code and protocol docs both treat it as a
permission flag. Outcome: no production `CreatesTokensContract`.

**Path 2 — Tighten protocol semantics.** `creates_tokens=True` means
"this transform is a true deaggregation transform and must emit multi-row
output in the relevant success path." Outcome: update protocol docs,
processor tests, plus existing transforms that rely on "allowed single
output," then implement the contract in Task 7.

### Required positive artifact for Path 1 (SA-5)

Even under Path 1, ADR-015 MUST:

1. Explicitly reject `creates_tokens` as a production DeclarationContract
   adopter with the rationale.
2. Name the alternative runtime-VAL mechanism that IS honest at the
   dispatcher surface — e.g. a `MultiRowEmissionCardinality`
   post-emission contract that applies only to transforms whose config
   declares "MUST emit N>1 rows for inputs matching X." If no such
   mechanism is obviously needed, record "no replacement needed;
   protocol-docs suffice."
3. Update ADR-010 §Adoption State to remove `CreatesTokensContract` from
   pre-emission adopter candidates, annotated "rejected per ADR-015."
4. Close filigree task `elspeth-cf2ee33808` with a comment pointing at
   ADR-015.
5. **Decide the fate of `tests/invariants/test_framework_accepts_second_contract.py`
   (QE-10):** EITHER re-point it at one of the 2B production adopters
   (`DeclaredOutputFieldsContract` is the natural candidate — already
   landed in Task 3) as proof that a second-shape contract registers, OR
   retire it with a same-PR note citing which successor adopter now
   carries the shape-diversity invariant.

### Definition of Done
- [ ] ADR-015 resolves the semantic conflict explicitly (Path 1 or 2).
- [ ] Under Path 1: no production contract implemented; alternative
      mechanism named; §Adoption State updated; ticket closed; framework
      shape-diversity harness re-pointed or retired.
- [ ] Under Path 2: protocol docs + processor tests updated in the same
      PR as ADR-015; implementation scheduled for Task 7.
- [ ] Codebase, docs, and filigree all describe the same `creates_tokens`
      meaning.

---

## Task 5: ADR-013 — `DeclaredRequiredFieldsContract` + `declared_input_fields` Normalisation

First pre-emission adopter. Lands only after Tasks 3 and 4.

### Files
- **Create:** `docs/architecture/adr/013-declared-required-fields-contract.md`
- **Create:** `src/elspeth/engine/executors/declared_required_fields.py`
- **Modify:** `src/elspeth/contracts/errors.py`
- **Modify:** `src/elspeth/plugins/infrastructure/base.py` (add
  `declared_input_fields: frozenset[str] = frozenset()` to `BaseTransform`)
- **Modify:** `src/elspeth/contracts/plugin_protocols.py` (add to
  `TransformProtocol`)
- **Modify:** `src/elspeth/plugins/infrastructure/config_base.py`
  (populate `declared_input_fields` from config at construction)
- **Modify:** `src/elspeth/contracts/declaration_contracts.py`
  (`EXPECTED_CONTRACT_SITES` entry)
- **Modify:** `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
- **Modify:** `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`
  (MC3a + MC3b regression)
- **Modify:** `tests/invariants/test_contract_non_fire.py`
- **Create:** `tests/unit/engine/test_declared_required_fields_contract.py`
- **Create:** `tests/integration/audit/test_declared_required_fields_roundtrip.py`

### Runtime attribute naming (SA-4)

The attribute name is `declared_input_fields`, NOT
`declared_required_fields`. Sinks already use `declared_required_fields`
with different semantics (fields required at write boundary — see
`plugin_protocols.py:567`). Same name across protocols with different
semantics is a correctness trap for future plugin authors. `declared_input_fields`
on `BaseTransform` and `TransformProtocol` disambiguates at grep time.

### `applies_to` body (PY-2)

Direct attribute access, no defensive default:

```python
def applies_to(self, plugin: Any) -> bool:
    return bool(plugin.declared_input_fields)
```

A plugin missing `declared_input_fields` is a framework bug — let it
crash. (Do NOT write `getattr(plugin, "declared_input_fields",
frozenset())`.)

### Batch scope — fail-closed (SEC-4, SA-3)

Phase 2B scopes this contract to non-batch transforms (Reality Anchor 6).
Silent skip on batch-aware transforms is a Repudiation surface
indistinguishable from "checked and passed" (S2-003 pattern). MUST be
fail-closed:

```python
def applies_to(self, plugin: Any) -> bool:
    if not plugin.declared_input_fields:
        return False
    if getattr(plugin, "is_batch_aware", False):
        # Wrong to use getattr with default here in general, but this path
        # is guarded by the prior check; swap for direct access once
        # is_batch_aware lands on TransformProtocol.
        raise FrameworkBugError(
            f"Transform {plugin.name!r} declares declared_input_fields "
            f"but is batch-aware. No batch-pre-execution dispatch site "
            f"exists; scope mismatch forbidden per ADR-013 until a "
            f"batch_pre_emission_check site lands via ADR-010 amendment."
        )
    return True
```

Alternatively: reject at plugin construction (a batch-aware transform MUST
NOT have `declared_input_fields` populated until a batch-pre-emission site
exists). Either approach acceptable; silent skip is NOT.

### Adding a fifth site (Option 2) is an ADR-010 amendment (SA-3)

If a future ADR elects to add `batch_pre_emission_check`, that is an
ADR-010 amendment decision — `DispatchSite` enum extension, `_dispatch`
helper wrap, `EXPECTED_CONTRACT_SITES` value-type widening, MC3 CI rule
updates, new per-surface rule-of-three gate at zero. Do NOT introduce
silently inside ADR-013.

### ADR-013 required content

- **Tier-1 classification:** Tier-1 (SEC-5). A plugin running on input it
  didn't declare producing outputs means every downstream record is
  unattributable.
- **Payload schema (SEC-1):**

```python
class DeclaredRequiredInputFieldsPayload(TypedDict):
    declared: list[str]
    effective_input_fields: list[str]
    missing: list[str]
```

- **Runtime invariant:** `plugin.declared_input_fields ⊆
  inputs.effective_input_fields`. Fire before `transform.process()` so the
  audit record attributes the fault to declaration drift, not to a crash
  inside plugin logic.
- **Comparison source (SEC-7):** compare against
  `inputs.effective_input_fields` (caller-derived per F1 resolution), NOT
  against any plugin-derived view. The plugin cannot be its own witness.
- **§Scrubber-audit:** all three payload keys carry field-name lists; no
  scrubber extension needed.
- **§Reversibility:** remove `declared_input_fields` attribute from
  `BaseTransform` + `TransformProtocol`, drop manifest entry + bootstrap
  import; retype existing plugins that relied on the attribute.

### Tests

**Red (QE-11 — precise wording):** transform declares `declared_input_fields={"customer_id"}`;
input row contract lacks `customer_id` → `DeclaredRequiredInputFieldsViolation`
from the pre-emission dispatch site. Record red-phase commit SHA.

**Green (non-fire):** contract includes the required field in
`contract.fields` → `pre_emission_check` returns None cleanly. Any
subsequent payload-access failure inside `transform.process()` is NOT
this contract's failure mode and is out of scope (QE-11 clarification).

**Batch fail-closed (SEC-4):** a batch-aware transform configured with
`declared_input_fields` causes `applies_to` to raise `FrameworkBugError`;
pipeline refuses to start, audit trail carries the bootstrap-time failure
rather than silent per-row skip.

**Aggregate round-trip (QE-2):** `declared_input_fields` missing AND
`passes_through_input=True` with downstream field drop → dispatcher
raises aggregate carrying both this contract's and PassThrough's
violations.

**E2E Landscape round-trip (QE-1):** per template.

**MC3a/MC3b regression (QE-4):** per template.

### Definition of Done
- [ ] Full §DoD Template passes.
- [ ] `declared_input_fields` normalised on `BaseTransform` and
      `TransformProtocol`; populated from config at construction.
- [ ] Pre-emission adopter lands and §Adoption State (Task 8) records
      the surface as "provisional — 1/3 adopters; 2 more for rule-of-three
      closure."
- [ ] Batch-aware mis-configuration fails closed (SEC-4).
- [ ] ADR-013 cites ADR-010 §Adoption State lines 263-267 and records
      that Option 2 (fifth site) is a deferred ADR-010 amendment.

---

## Task 6: ADR-014 — `SchemaConfigModeContract`

The most structurally complex 2B adopter. ADR work before code.

### Files
- **Create:** `docs/architecture/adr/014-schema-config-mode-contract.md`
- **Create:** `src/elspeth/engine/executors/schema_config_mode.py`
- **Modify:** `src/elspeth/contracts/errors.py`
- **Modify:** `src/elspeth/contracts/declaration_contracts.py`
- **Modify:** `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
- **Modify:** `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`
- **Modify:** `tests/invariants/test_contract_non_fire.py`
- **Create:** `tests/unit/engine/test_schema_config_mode_contract.py`
- **Create:** `tests/integration/audit/test_schema_config_mode_roundtrip.py`

### ADR-014 required content

- **Tier-1 classification:** Tier-1 (SEC-5). A FIXED-mode transform
  emitting OBSERVED violates the contract auditors query by; fabrication
  surface.
- **Payload schema (SEC-1):**

```python
class SchemaConfigModePayload(TypedDict):
    declared_mode: str       # "fixed" / "flexible" / "observed"
    observed_mode: str
    declared_locked: bool
    observed_locked: bool
    undeclared_extra_fields: NotRequired[list[str]]
```

Forbidden payload keys: `raw_schema_config`, `config_dict`, `options`,
`sample_row` (SEC-2). Values must be structural (mode strings, bool
flags, field-name sets), never arbitrary config snapshots.

- **`applies_to` O(1) (SEC-9):** read a single flag
  (`plugin._output_schema_config is not None`). All mode-comparison
  logic belongs in `post_emission_check` after the applies-to filter has
  pruned.
- **Dispatch sites:** `post_emission_check` AND `batch_flush_check`
  (decorated on the concrete class — PY-4). Multi-site `ExampleBundle`
  strategy per template (PY-1).
- **Answer these ADR questions up front:**
  - For `fixed` mode: invariant = "runtime contract mode is FIXED and
    locked, and no undeclared output fields appear."
  - For `flexible` mode: invariant = "declared fields remain valid while
    extras are allowed."
  - For `observed` mode: invariant = "runtime contract shape matches"
    (and whether emitted contract locked after first-row inference
    belongs here or is a separate contract).
  - Is `locked` part of this contract's payload, or a separate invariant
    left to another contract? (ADR decides.)

### Implementation guidance
- Reuse current `SchemaContract` properties and the schema-factory
  mapping logic; do NOT invent a parallel mode-normalisation table.
- Prefer comparing actual emitted `PipelineRow.contract` semantics to
  `plugin._output_schema_config`, not raw plugin config dicts.

### Tests
- **Red first:** a `fixed`-mode transform emits an OBSERVED contract →
  violation.
- A transform with fixed schema emits undeclared extra fields →
  violation.
- A valid observed-mode transform infers fields at runtime → no violation.
- Aggregate round-trip with PassThrough per template.
- E2E Landscape round-trip per template.
- MC3a/MC3b regression per template.

### Definition of Done
- [ ] Full §DoD Template passes.
- [ ] ADR-014 precisely defines the runtime invariant using real repo
      types (`SchemaConfig`, `SchemaContract`, `locked`).
- [ ] Covers both single-row and batch-flush paths.
- [ ] Tests prove runtime schema semantics are checked, not string labels.

---

## Task 7: `can_drop_rows` Governance Contract (SLA-Critical)

> **SLA annotation (SA-8):** this task retires the ADR-009 Clause 3
> empty-emission carve-out. Hard trigger: **2026-07-18** (or registration
> of a second `passes_through_input=True` transform with external-call
> dependencies — whichever comes first). Phase 2B sequencing MUST land
> this task before 2026-07-18.

### Files
- **Create:** `docs/architecture/adr/012-can-drop-rows-contract.md`
- **Create:** `src/elspeth/engine/executors/can_drop_rows.py`
- **Modify:** `src/elspeth/contracts/errors.py`
- **Modify:** `src/elspeth/plugins/infrastructure/base.py`
  (add `can_drop_rows: bool = False`)
- **Modify:** `src/elspeth/contracts/plugin_protocols.py`
  (add to `TransformProtocol`)
- **Modify:** `src/elspeth/contracts/declaration_contracts.py`
  (`EXPECTED_CONTRACT_SITES` entry)
- **Modify:** `src/elspeth/engine/executors/declaration_contract_bootstrap.py`
- **Modify:** `src/elspeth/engine/executors/pass_through.py` (delete the
  `if not emitted_rows: return` Clause-3 short-circuit at lines 83-84 —
  SA-7; governance now lives in `CanDropRowsContract`)
- **Modify:** `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`
- **Modify:** `tests/invariants/test_contract_non_fire.py`
- **Modify:** `docs/contracts/plugin-protocol.md`
- **Create:** `tests/unit/engine/test_can_drop_rows_contract.py`
- **Create:** `tests/integration/audit/test_can_drop_rows_roundtrip.py`

### Contract semantics
- Add `can_drop_rows: bool = False` to `BaseTransform` and
  `TransformProtocol`.
- `CanDropRowsContract.applies_to(plugin)` body (PY-2):

```python
return plugin.passes_through_input and not plugin.can_drop_rows
```

- On `post_emission_check` and `batch_flush_check`, raise when
  `len(outputs.emitted_rows) == 0`.
- Keep this contract separate from `DeclaredOutputFieldsContract`; zero
  emitted rows is NOT "missing declared fields."
- Delete the Clause-3 carve-out (`pass_through.py:83-84`) in the same PR
  (SA-7). Governance now lives in this contract; two overlapping checks
  in different code paths is a future-drift surface.

### Terminal state for legitimate zero emission (SEC-3, HIGH)

A legitimately-dropped row (`can_drop_rows=True`, 0 emitted) MUST produce
a terminal state the auditor can query. "No record" is indistinguishable
from "we forgot to record" under audit-complete posture. ADR-012 MUST:

- Name the terminal state (CLAUDE.md names
  `COMPLETED`/`ROUTED`/`FORKED`/`CONSUMED_IN_BATCH`/`COALESCED`/`QUARANTINED`/`FAILED`/`EXPANDED`).
  Reuse `CONSUMED_IN_BATCH` or introduce a new `DROPPED_BY_FILTER` state.
- If a new terminal state is required, land it in the same PR as the
  contract (NOT as a follow-up).
- Task 7 test suite MUST include: "a `can_drop_rows=True` transform
  emitting 0 rows on a row produces a Landscape record whose terminal
  state is queryable and distinct from `FAILED`."

### Tier-1 classification (SEC-5)

Tier-2, NOT Tier-1. `CanDropRowsViolation` is a plugin bug (row-level),
same posture as base `PluginContractViolation`. Justification in ADR-012.

### ADR-012 required content
- **Payload schema (SEC-1):**

```python
class UnexpectedEmptyEmissionPayload(TypedDict):
    passes_through_input: bool
    can_drop_rows: bool
    emitted_count: int
```

Violation class explicitly: `class UnexpectedEmptyEmissionViolation(
DeclarationContractViolation): payload_schema: ClassVar[type] =
UnexpectedEmptyEmissionPayload` (PY-3).

- **§Scrubber-audit:** all three payload keys are structural (bool / int);
  no scrubber extension.
- **§Reversibility:** remove `can_drop_rows` attribute; drop manifest
  entry + bootstrap import; re-introduce Clause-3 carve-out at
  `pass_through.py` (git history exists).
- **§Clause-3 retirement:** state explicitly that the short-circuit at
  `pass_through.py:83-84` is deleted in this PR. The new contract owns
  empty-emission governance.

### Test matrix (QE-7 expansion)

- Red first: mis-declared filter (`passes_through_input=True,
  can_drop_rows=False`, 0 emitted) → violation.
- Legitimate filter (`can_drop_rows=True`, 0 emitted) → no violation +
  terminal state recorded (SEC-3).
- Pass-through transform returning one row → no violation.
- **Aggregate case (QE-7):** mis-declared filter that ALSO drops input
  fields → `AggregateDeclarationContractViolation` carrying both
  `UnexpectedEmptyEmissionViolation` AND `PassThroughContractViolation`.
- **Scoping non-fire (QE-7 / N2 Layer A):** `passes_through_input=False`
  plugin with `can_drop_rows=False` and 0 emission → `applies_to` returns
  False; no fire. Mis-scoping is a Tier-1 attribution bug.
- **No-short-circuit (QE-7 / SA-7):** after Clause-3 deletion, the
  `CanDropRowsContract` MUST NOT be short-circuited by the now-removed
  pass-through empty-emission exemption. Explicit test: a
  `passes_through_input=True, can_drop_rows=False` transform returning 0
  rows with the old short-circuit logic would not fire PassThrough's
  check; post-deletion the dispatcher runs `CanDropRowsContract` which
  fires.
- Multi-site ExampleBundle (PY-1) for `post_emission_check` and
  `batch_flush_check`.
- E2E Landscape round-trip (QE-1), aggregate round-trip (QE-2), MC3a/b
  regression (QE-4), registry isolation (QE-6), live-registry benchmark
  re-baseline (QE-9) per template.

### Definition of Done
- [ ] Full §DoD Template passes.
- [ ] Clause-3 carve-out deleted mechanically, not just documented (SA-7).
- [ ] `can_drop_rows` exists as a first-class runtime declaration.
- [ ] Legitimate zero-emission rows produce an auditor-queryable terminal
      state distinct from `FAILED` (SEC-3).
- [ ] A second externally-calling pass-through transform can no longer
      extend the carve-out silently.
- [ ] ADR-009 Clause 3 SLA 2026-07-18 is annotated as satisfied.

---

## Task 8: Cross-Cutting Verification, Benchmarking, and §Adoption State Update

### Files
- **Modify:** `tests/invariants/test_contract_negative_examples_fire.py`
- **Modify:** `tests/integration/audit/test_declaration_contract_landscape_roundtrip.py`
- **Modify:** `tests/unit/engine/test_orchestrator_registry_bootstrap.py`
- **Modify:** `tests/performance/benchmarks/test_cross_check_overhead.py`
- **Modify:** `docs/architecture/adr/010-declaration-trust-framework.md`
  (§Adoption State update + governance-asymmetry note — SA-2, ST-4)
- **Modify:** relevant filigree issues with closure notes and blockers

### §Adoption State update (mandatory)

Update the per-surface table at
`docs/architecture/adr/010-declaration-trust-framework.md:249-254` to
reflect post-2B counts and the pre-emission provisional status:

| Surface | Post-2B adopters | Rule-of-three closed? |
|---|---|---|
| `post_emission_check` | PassThrough + DeclaredOutputFields + CanDropRows + SchemaConfigMode (**4/3** — closed) | YES |
| `batch_flush_check` | same four (**4/3** — closed) | YES |
| `pre_emission_check` | DeclaredRequiredFields only (**1/3** — PROVISIONAL) | NO — 2 more for 2C |
| `boundary_check` | 0 | NO (2C paired landing) |

Add the ST-4 governance note: "the pre-emission site's zero baseline
meant the rule-of-three gate provided weaker early coverage there —
designs that plausibly fit pre-emission SHOULD prefer pre-emission to
build the site's evidential weight."

### Verification checklist (cross-cutting gates)

- [ ] Every production adopter appears in `EXPECTED_CONTRACT_SITES`.
- [ ] Every adopter has `negative_example()` +
      `positive_example_does_not_apply()` covering every claimed site
      (PY-1).
- [ ] Every adopter has an E2E Landscape round-trip test and an aggregate
      round-trip test with PassThrough.
- [ ] Every adopter has an MC3a + MC3b regression fixture in
      `tests/unit/scripts/cicd/test_enforce_contract_manifest.py`.
- [ ] Performance benchmark passes at `N ∈ {1, 2, 4, 8, 16}` with the
      production registry reflecting post-2B N (ST-6 parametric
      extrapolation still holds: `27 + 7 × 1.5 = 37.5 µs` median at
      N=8).
- [ ] Pre-emission hot-path cost is justified by `DeclaredRequiredFieldsContract`
      registration (Task 5).
- [ ] F-QA-5 per-surface Hypothesis property tests remain green (Task 1).
- [ ] Bootstrap drift AST test remains green with four additional
      adopters listed.
- [ ] Registry-isolation grep (ruff or a dedicated CI script) confirms no
      new `tests/unit/engine/test_*_contract.py` calls
      `_clear_registry_for_tests()` without a matching
      `_snapshot_registry_for_tests` / `_restore_registry_snapshot_for_tests`
      pair (QE-6 mechanical check).

```bash
PYTHONPATH=src uv run pytest -q \
  tests/invariants/test_contract_negative_examples_fire.py \
  tests/invariants/test_contract_non_fire.py \
  tests/unit/engine/test_orchestrator_registry_bootstrap.py \
  tests/unit/engine/test_declaration_contract_bootstrap_drift.py \
  tests/integration/audit/test_declaration_contract_landscape_roundtrip.py \
  tests/integration/audit/test_declared_output_fields_roundtrip.py \
  tests/integration/audit/test_can_drop_rows_roundtrip.py \
  tests/integration/audit/test_declared_required_fields_roundtrip.py \
  tests/integration/audit/test_schema_config_mode_roundtrip.py \
  tests/performance/benchmarks/test_cross_check_overhead.py \
  tests/property/engine/test_dispatch_bundle_properties.py
```

Then:

```bash
uv run ruff check src tests
uv run mypy src
.venv/bin/python scripts/cicd/enforce_contract_manifest.py check
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth \
  --allowlist config/cicd/enforce_tier_model
```

### Per-adopter PR gate vs terminal task (SA-9)

Task 8's verification is NOT a terminal batch step. Each adopter PR
(Tasks 3, 5, 6, 7) runs this full suite AT the PR gate, not at the end.
Task 8's terminal role is limited to:

- §Adoption State text update in ADR-010 (single PR at tranche end).
- Filigree closure notes + blocker cleanup across the 2B epic
  (`elspeth-a3ac5d88c6`).
- Any remaining documentation in `docs/contracts/plugin-protocol.md`.

### Definition of Done
- [ ] Every landed Phase 2B adopter satisfies §DoD Template.
- [ ] ADR-010 §Adoption State updated with post-2B counts + provisional
      pre-emission note + ST-4 governance note.
- [ ] Benchmark and static-analysis gates remain green at post-2B N.
- [ ] Filigree blockers and close notes reflect actual landed scope;
      `elspeth-cf2ee33808` closed or retyped per ADR-015 outcome.

---

## Recommended Landing Order

1. Task 0: verify prerequisites + board reality.
2. **Task 1: production bootstrap hotfix + F-QA-5 Hypothesis closure**
   (SA-1 CRITICAL). H2 closes in the same PR.
3. Task 2: DoD-template review gate (non-code).
4. Task 3: `DeclaredOutputFieldsContract` rule-of-three gate.
5. Task 4: ADR-015 `CreatesTokens` semantic resolution (ADR-first, ST-2).
6. Task 5: `DeclaredRequiredFieldsContract` + `declared_input_fields`
   normalisation.
7. Task 6: `SchemaConfigModeContract`.
8. Task 7: `can_drop_rows` governance contract (SLA 2026-07-18).
9. Task 8: §Adoption State update + cross-cutting verification +
   filigree cleanup.

> Task 7 can swap with Task 5 or 6 if the SLA deadline pressure exceeds
> semantic-priority concerns; the atomic requirement is that Task 7 lands
> before 2026-07-18.

---

## Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Production bootstrap broken outside pytest (SA-1) | Task 1 subprocess regression test + orchestrator-top-import wire-up + AST drift test. |
| Ticket descriptions do not match code reality | Treat ADR + current repo code as primary sources; update filigree in Task 0 + Task 8. |
| Pre-emission coverage silently excludes batch-aware transforms (SEC-4) | ADR-013 fail-closed `applies_to`; raise `FrameworkBugError` on batch-aware misconfiguration. |
| `DeclaredOutputFieldsContract` checks only contract metadata or only payload keys | Use the same "runtime observed = contract ∩ payload" posture as pass-through verification. |
| `CreatesTokensContract` lands on impossible semantics | ADR-015 is a hard gate before any production implementation (Task 4). |
| Bootstrap relies on accidental imports and silently drops contracts (SA-1, QE-5, SEC-6) | Task 1 explicit bootstrap module + AST drift test + orchestrator-top import + subprocess regression. |
| Manifest drift under concurrent adopter PRs (ST-3) | Per-PR manifest-count assertion landing with each adopter. |
| Performance regression with each new adopter | Keep live-registry benchmark re-baseline per-adopter (QE-9); avoid expensive `applies_to` logic (SEC-9). |
| Mixin-inherited dispatch methods miss `@implements_dispatch_site` markers (PY-4) | Decorator must be on the CONCRETE class even when the body is inherited. Reviewer discipline; enforced by MC3b CI. |
| `declared_required_fields` naming collides with sink attribute (SA-4) | Use `declared_input_fields` on transforms. |
| Clause-3 carve-out + `CanDropRowsContract` create redundant checks (SA-7) | Delete the Clause-3 short-circuit in Task 7's PR. |
| Legitimate zero emission lacks a queryable terminal state (SEC-3) | ADR-012 names the terminal state; dedicated test in Task 7 asserts queryability distinct from FAILED. |
| Invariant harness silently narrows if CreatesTokens is retired (QE-10) | Task 4 mandates re-pointing or retiring `tests/invariants/test_framework_accepts_second_contract.py` in the same PR. |
| Red-phase skipped (QE-8) | PR description records red-phase commit SHA; reviewer-enforced. |

---

## Execution Notes

- Land one production adopter per PR after Task 1. Task 2 is a non-code
  review gate. Tasks 4 and 8 are documentation/ADR-heavy.
- Each adopter PR is self-contained: ADR, violation class, contract
  module, manifest update, bootstrap import, negative-example coverage,
  round-trip tests (unit + aggregate + E2E), MC3 regression fixture,
  registry-isolation fixture, live-registry benchmark re-run, red-phase
  SHA in PR description, and filigree note.
- If Task 4 resolves to Path 1 ("no production contract"), record that
  explicitly, handle the second-shape harness orphan (QE-10), and remove
  `CreatesTokensContract` from the remaining Phase 2B critical path
  rather than leaving a zombie task.
- Task 7 is SLA-critical (2026-07-18, ADR-009 Clause 3). If Tasks 0-3
  slip, reconsider sequencing so Task 7 still lands before the SLA.
- The `pre_emission_check` rule-of-three remains open post-2B
  (1/3 adopters). Phase 2C MUST add two more pre-emission adopters (or
  elect, via ADR-010 amendment, to treat a single adopter as sufficient
  for that surface — with full §Alternatives rationale).

---

## Confidence and Caveats

- **CRITICAL and HIGH findings** from the panel review are folded into
  this revision with explicit citation tags. The F-QA-5 Hypothesis closure
  is scheduled here rather than deferred.
- **MEDIUM findings** are folded except where they duplicated
  recommendations already in the HIGH set. Each MEDIUM has a citation tag
  at the point of incorporation.
- **LOW / informational findings** are incorporated where cheap:
  SLA annotation (SA-8), per-PR verification gating (SA-9), multi-site
  harness disposition (QE-10), F-QA-5 inclusion note (PY-7),
  CreatesTokens ADR-first confirmation (SEC-8), DoS budget governance
  (SEC-9), NFR re-baseline at 2C landing (ST-6).
- No pytest / mypy / ruff invocation was performed during this revision;
  the plan is an intent document. Task 0 re-verifies the framework at
  the point of execution.
- The 5 individual review files at
  `docs/plans/reviews/2026-04-20-phase-2b-*.md` carry per-finding evidence
  and are the definitive source if any tag is ambiguous.
