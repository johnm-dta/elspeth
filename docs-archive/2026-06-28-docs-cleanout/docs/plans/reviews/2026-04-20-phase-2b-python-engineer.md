# Phase 2B Declaration-Trust — Python Engineer Review

**Verdict: APPROVE-WITH-CHANGES**

The plan is structurally sound and correctly identifies the key design decisions that must precede coding. The landing order, ADR-first posture for CreatesTokens, and explicit batch-coverage scoping are all correct engineering choices. Three findings require action before code ships: the `ExampleBundle` multi-site gap creates an untestable harness blind spot, the `declared_required_fields` class-attribute default creates a forbidden defensive-programming pattern, and every new violation `payload` TypedDict must be explicitly called out as requiring `deep_freeze` at construction (the base class handles it, but the plan's `can_drop_rows` shape is defined as a `TypedDict` on the contract, not on the violation class, leaving it ambiguous). The remaining findings are low-severity guidance items that can be resolved in-PR.

---

## Findings

### PY-1 — HIGH: `ExampleBundle` single-site constraint collides with multi-site adopters

**Severity:** HIGH

**Finding:** `DeclaredOutputFieldsContract` and `CanDropRowsContract` each claim two dispatch sites (`post_emission_check` + `batch_flush_check`), but `ExampleBundle` carries exactly one `site` tag and the plan's harness dispatches `bundle.site` without per-site iteration. Multi-site adopters exercised only via their `post_emission_check` bundle will have their `batch_flush_check` path silently unexercised by the negative-example harness.

**Evidence:** Plan Task 2 (line 136–174), Task 3 (line 176–224). `ExampleBundle` docstring in `src/elspeth/contracts/declaration_contracts.py` lines 379–381: "Contracts implementing multiple sites return the bundle for whichever site their negative_example primarily exercises." The plan does not address how the invariant harness covers the second site on these multi-site contracts.

**Recommendation:** Each multi-site adopter should return a `list[ExampleBundle]` from `negative_example` (or the harness must iterate both sites). The simplest path compatible with the existing harness is to define `negative_example_batch_flush` as a second classmethod and update `tests/invariants/test_contract_negative_examples_fire.py` to call it. Alternatively, document in each ADR that batch-flush negative-example coverage lives in the adopter's dedicated unit test rather than the shared invariant harness. Whichever choice is made, it must be explicit — the current plan leaves an ambiguity that will produce a silently unexercised dispatch path.

---

### PY-2 — HIGH: `declared_required_fields: frozenset[str] = frozenset()` class default violates CLAUDE.md §Offensive Programming

**Severity:** HIGH

**Finding:** The plan (Task 4, line 246–249) proposes adding `declared_required_fields: frozenset[str] = frozenset()` as a `BaseTransform` class-level default, and the `DeclaredRequiredFieldsContract.applies_to` would implicitly rely on transforms that do not set it yielding `frozenset()` (i.e. "no required fields declared, contract does not apply"). This is functionally equivalent to `getattr(plugin, "declared_required_fields", frozenset())` — a defensive default that masks a missing declaration. CLAUDE.md §Offensive Programming and the §Defensive Programming: Forbidden section both forbid this pattern.

**Evidence:** Plan lines 246–249: `declared_required_fields: frozenset[str] = frozenset()`. `src/elspeth/plugins/infrastructure/base.py` line 199 shows the existing `declared_output_fields: frozenset[str] = frozenset()` pattern on `BaseTransform` — this is an existing instance of the pattern, so replicating it for required fields would entrench it further. `PassThroughDeclarationContract.applies_to` at `src/elspeth/engine/executors/pass_through.py` line 145 correctly does direct attribute access: `return cast(bool, plugin.passes_through_input)` — that's the correct idiom.

**Recommendation:** The class-level default is acceptable IF and only if `frozenset()` is the semantically correct value for "this transform has no required fields" (i.e. all transforms start with an empty declared set and override it). This is the existing `declared_output_fields` pattern, so consistency argues for it. However, the plan must explicitly state that `applies_to` returns `True` when `plugin.declared_required_fields` is non-empty — not a `hasattr` guard. The contract's `applies_to` body must be:

```python
def applies_to(self, plugin: Any) -> bool:
    return bool(plugin.declared_required_fields)
```

If `declared_required_fields` is absent on a plugin, that is a framework bug and must crash, not silently skip.

---

### PY-3 — MEDIUM: `UnexpectedEmptyEmissionPayload` TypedDict is on the contract, not the violation class

**Severity:** MEDIUM

**Finding:** Task 3 (line 199–203) defines `UnexpectedEmptyEmissionPayload` as the payload shape, but it is presented as the contract's documentation of the payload rather than as an explicit `payload_schema: ClassVar[type] = UnexpectedEmptyEmissionPayload` on the violation class. The H5 Layer 1 deny-by-default gate (`_validate_payload_against_schema` in `src/elspeth/contracts/declaration_contracts.py` lines 502–535) enforces this at construction time — but only if `CanDropRowsViolation` (whatever it will be named) declares `payload_schema` and the violation is a `DeclarationContractViolation` subclass. The plan does not explicitly call out that the new violation class must (a) subclass `DeclarationContractViolation`, (b) override `payload_schema`, and (c) pass `deep_freeze(dict(payload))` already handled by the base class `__init__`. If the implementer writes a standalone exception class without the DCV hierarchy, the H5 gate is bypassed.

**Evidence:** Plan lines 199–203; `src/elspeth/contracts/declaration_contracts.py` lines 477–499 (base class `__init__` does `deep_freeze(dict(payload))` and calls `_validate_payload_against_schema`). Plan does not mention `payload_schema: ClassVar[type]` for Task 3's violation.

**Recommendation:** Add to each Task's "Suggested payload shape" section: "The violation class must subclass `DeclarationContractViolation` and set `payload_schema: ClassVar[type] = <TypedDict name>`. The base class `__init__` applies `deep_freeze` and `_validate_payload_against_schema` automatically." This is the rule-of-three gate the plan names Task 2 as establishing — it should be made explicit in every subsequent task rather than assumed from reading `pass_through.py`.

---

### PY-4 — MEDIUM: MC3b scanner will fail on mixin-based multi-site adopters unless concrete class carries both markers

**Severity:** MEDIUM

**Finding:** `_extract_marker_sites_and_trivial_bodies` in `scripts/cicd/enforce_contract_manifest.py` (lines 501–544) inspects only the direct class body — it explicitly documents "Per the D1 correction (comment #418 on H2), this scanner only inspects direct class body; mixin-inherited overrides are not resolved." If any Phase 2B adopter uses a mixin (e.g. a shared `BatchFlushMixin` that provides a shared `batch_flush_check` implementation), MC3b will fire for the `batch_flush_check` site unless the concrete class carries `@implements_dispatch_site("batch_flush_check")` directly on its override. The plan does not mention this constraint or flag it as a risk.

**Evidence:** `scripts/cicd/enforce_contract_manifest.py` lines 510–514 (D1 correction note). Plan Tasks 2–5 all propose two-site adopters without warning about the mixin constraint.

**Recommendation:** Add to the plan's Risks table: "Any adopter using mixin inheritance for `batch_flush_check` MUST carry `@implements_dispatch_site('batch_flush_check')` directly on the concrete class's override — the MC3b scanner does not resolve mixin-inherited markers." No code change to the scanner is needed; this is a discipline rule for implementers.

---

### PY-5 — MEDIUM: Bootstrap module at L2 imports L3 adopters — layer dependency risk

**Severity:** MEDIUM

**Finding:** `src/elspeth/engine/executors/declaration_contract_bootstrap.py` (Task 1) sits at L2 (`engine/`). The plan includes `import elspeth.engine.executors.can_drop_rows` in the bootstrap. If `can_drop_rows.py` imports anything from `plugins/` (L3) to access `BaseTransform` or `TransformProtocol` for type annotations, the bootstrap itself would be an L2 module with a transitive L3 dependency. The plan notes that Task 3 modifies `src/elspeth/plugins/infrastructure/base.py`, which is L3.

**Evidence:** Plan lines 103–106 (bootstrap imports); Task 3 line 193 (modifies `src/elspeth/plugins/infrastructure/base.py`). Layer rules in CLAUDE.md: L2 can import L0 and L1 only.

**Recommendation:** Each adopter module (`declared_output_fields.py`, `can_drop_rows.py`, etc.) must restrict its imports to L0/L1/L2 types only. Plugin attribute access via `Any`-typed `plugin` parameter avoids the cross-layer import. If type narrowing is needed, use `TYPE_CHECKING` imports and annotate with string literals — but note these are flagged as warnings by the tier enforcer. The bootstrap module itself imports only the adopter modules (side-effect imports), so it is safe as long as each adopter module does not itself import from L3. This must be called out as an implementer constraint in the plan.

---

### PY-6 — LOW: `_serialize_plugin_name` in dispatcher uses `try/except AttributeError` — inconsistent with CLAUDE.md

**Severity:** LOW (pre-existing, not introduced by plan, but plan adopters inherit the pattern)

**Finding:** `declaration_dispatch.py` line 88–94 contains a `try/except AttributeError` around `plugin.name`. This is a defensive pattern forbidden by CLAUDE.md §Offensive Programming. The plan does not introduce this, but Phase 2B adopters that model their `applies_to` on the dispatcher's own internal style might copy the pattern.

**Evidence:** `src/elspeth/engine/executors/declaration_dispatch.py` lines 88–94.

**Recommendation:** Do not model adopter `applies_to` implementations on `_serialize_plugin_name`'s defensive style. Use direct attribute access: `plugin.declared_required_fields`, `plugin.can_drop_rows`, `plugin.declared_output_fields`. The pass-through adopter's `applies_to` at `pass_through.py` line 145 is the canonical template.

---

### PY-7 — LOW: F-QA-5 Hypothesis property gap — plan is additive, not convergent

**Severity:** LOW

**Finding:** The open F-QA-5 gap (Hypothesis property test per dispatch surface) is not mentioned in the plan. Phase 2B adds four new adopters, each with two dispatch sites — this multiplies the gap rather than closing it. The plan's test coverage is unit-scope + Landscape round-trip only; no property-based testing of the runtime invariant logic.

**Evidence:** Author-flagged caveat C5 (plan preamble). Plan Tasks 2–5 test sections describe only red/green unit tests and round-trip integration tests.

**Recommendation:** Either add a Hypothesis-based property test for each adopter's core invariant (e.g. `hypothesis.given(st.frozensets(st.text())) → missing = declared - observed must be non-empty to fire`) or create a Filigree task to close F-QA-5 with explicit scope. Do not let Phase 2B close without acknowledging the gap in Task 7's Definition of Done.

---

## Confidence Assessment

**High confidence** on PY-1 (ExampleBundle single-site design is directly in the source), PY-2 (CLAUDE.md §Offensive Programming is explicit), PY-3 (H5 Layer 1 enforcement is directly in the base class), and PY-4 (MC3b scanner logic is directly in the CI script). **Medium confidence** on PY-5 (depends on which L3 types adopter modules will import — this is a risk, not a certainty). PY-6 and PY-7 are low-severity observations with no ambiguity.

## Risk Assessment

PY-1 and PY-2 have the highest exploit surface: PY-1 leaves a dispatch path unverified by the invariant harness (a contract appears to fire and audits as fired but the batch-flush path is untested); PY-2 risks entrenching a defensive access pattern that CLAUDE.md forbids. Both are fixable within the normal PR workflow.

## Information Gaps

- The plan does not specify the concrete class name for the `can_drop_rows` violation (Task 3 only names the payload TypedDict). The H5 Layer 1 gate needs both.
- The plan does not state whether `SchemaConfigModeContract` will have one or two `ExampleBundle` return values — the schema-mode contract has complex multi-site logic that may require per-site harness coverage.

## Caveats

This review is based on reading the plan and the current repo sources. No `mypy` or `pytest` were run. The `declared_output_fields: frozenset[str] = frozenset()` pattern already exists in `BaseTransform` (line 199 of `base.py`), so PY-2's concern about the class-level default is about the `applies_to` access pattern, not the default itself — both the existing field and the proposed `declared_required_fields` use the same pattern, and it is consistent with the existing codebase. The CLAUDE.md concern applies to the contract's `applies_to` implementation, not the field declaration.
