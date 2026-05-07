# RFC: Restoration Symmetry and Deep Immutability Rules

**Status:** Proposed
**Author:** ELSPETH project (consumer of wardline Python binding)
**Target:** Wardline specification v1.0 (prime) and Part II-A (Python binding)
**Date:** 2026-04-09

## Abstract

This RFC proposes three additions to the wardline framework:

1. **WL-009** (prime spec, §7): A structural verification rule requiring that integral-read paths through serialization boundaries declare commensurate restoration evidence. Closes a gap where `@integral_read` permits Tier 1 authority claims on deserialized data without evidence, contradicting the restoration model in §5.3.

2. **Non-normative note** (prime spec, §4 or §8): Guidance that bindings whose target language has shallow immutability mechanisms SHOULD define supplementary rules detecting false structural guarantees on Tier 1 data.

3. **PY-WL-010 and PY-WL-011** (Python binding, Part II-A): Two supplementary rules detecting shallow-freeze patterns in frozen dataclass `__post_init__` methods — bare `MappingProxyType` wrapping and `isinstance`-guarded freeze bypass.

All three proposals are derived from enforcement gaps discovered while applying wardline's tier model to a production audit-trail system (ELSPETH), where the write/read asymmetry and shallow immutability gaps produced integrity violations that the current rule set does not detect.

---

## 1. Motivation

### 1.1 The restoration symmetry gap

The wardline specification establishes a rigorous model for restoration boundaries (§5.3). Four evidence categories — structural, semantic, integrity, and institutional — determine what tier a deserialized representation may claim. WL-007 requires that declared restoration boundaries contain rejection paths. The model is sound.

The gap is upstream of the model: **there is no rule that requires a restoration boundary to exist when one is needed.**

Consider the following annotated code:

```python
@integral_writer
def write_audit_record(event: AuditEvent) -> None:
    """Write to the audit trail. Construction includes __post_init__
    validation: enum range checks, non-null invariants, hash verification."""
    db.insert(event.to_row())

@integral_read
def load_audit_record(run_id: str) -> AuditEvent:
    """Read from the audit trail. No validation — trusts the database."""
    row = db.fetch("audit_events", run_id=run_id)
    return AuditEvent(**row)
```

The write path constructs `AuditEvent` through validated construction — `__post_init__` enforces invariants. The read path deserializes from the database and stamps the output as INTEGRAL via `@integral_read`. But the read path performs no validation. The invariants established at construction time are not re-verified after deserialization.

This code is well-annotated. The scanner sees `@integral_read` and treats the output as Tier 1. WL-001 through WL-008 fire normally within the function body. But no rule detects the structural gap: **the read side claims the same authority as the write side without providing the same evidence.**

The consequence is precisely what §5.3 warns against: "a mere assertion of internal origin does not suffice." The `@integral_read` annotation is that mere assertion. Without a `@restoration_boundary` declaration specifying evidence categories, the scanner cannot verify — and governance cannot review — the adequacy of the restoration act.

### 1.2 Why this is a framework concern, not a binding concern

The restoration symmetry gap is language-agnostic:

- **Python:** `dataclass.__post_init__` validation on write, bare `dict(**row)` construction on read.
- **Java:** `@PrePersist` validation on JPA entity write, bare `ResultSet` mapping on read.
- **Go:** Struct constructor with validation on write, bare `json.Unmarshal` on read.

In every case, serialization strips runtime invariants, and the read path must re-establish them. The wardline already models this (§5.3) — the missing piece is a rule that detects when the model is not applied.

### 1.3 The shallow immutability problem

Python's `frozen=True` on dataclasses prevents attribute *reassignment* but does nothing about mutable *contents*. A frozen dataclass with a `dict` field is a false structural guarantee: the field reference cannot be reassigned, but the dict's contents can be freely mutated through the existing reference.

This matters for Tier 1 data because immutability is part of the integrity contract. An audit record whose fields can be silently mutated after construction — even if the dataclass is declared frozen — has a weaker integrity guarantee than the declaration implies. The `frozen=True` attribute is the Python-level assertion of immutability; if the assertion is false, downstream code that trusts it (including the wardline scanner, which may use `frozen=True` as a signal for structural soundness) is operating on a false premise.

The principle — "declared immutability must be deep, not shallow" — applies to any language where the immutability mechanism is shallow. But the detection is inherently language-specific because the immutability mechanisms differ across languages. This makes it a candidate for a framework-level non-normative note (establishing the principle) with binding-level supplementary rules (implementing detection).

---

## 2. Proposed Changes

### 2.1 WL-009: Integral restoration without declared evidence (prime spec)

#### 2.1.1 Rule definition

**Proposed addition to §7.1:**

| Rule | Pattern | Why It Is Dangerous |
|------|---------|---------------------|
| **WL-009** | Integral-read function on a serialization path without declared restoration evidence | The function claims Tier 1 authority for deserialized data on assertion alone — no evidence categories are declared, no governance review of the restoration act is possible, and the scanner cannot verify that write-side invariants are re-established on read. This is the restoration-symmetry failure: write-side construction includes validation that serialization strips, and the read side silently re-stamps the deserialized representation as authoritative without re-verifying the invariants. The severity of the gap is proportional to the write-side validation depth — a construction path with semantic validation, integrity checks, and cross-field invariants that is paired with a bare deserialization path creates a wider integrity gap than a construction path with only structural checks. |

#### 2.1.2 Detection semantics

WL-009 is a **structural verification rule** (like WL-007 and WL-008), not a pattern rule. It operates on the annotation topology, not on AST patterns within function bodies.

WL-009 fires when **all three** of the following conditions hold:

1. A function is declared `@integral_read` (Group 1).
2. The function's data source crosses a serialization boundary — determined by manifest declaration. The manifest's `boundary_declarations` or `dependency_taint` entries identify which data sources involve serialization (databases, files, message queues, caches). Functions whose `@integral_read` data source is an in-memory Tier 1 structure (no serialization) are not subject to WL-009.
3. The function does **not** co-declare `@restoration_boundary` (Group 17) with evidence categories, **and** the function's inputs do not trace through a declared restoration boundary with sufficient evidence within the two-hop analysis scope (§8.1).

**The manifest dependency is intentional.** WL-009 requires the manifest to declare which data sources involve serialization. Without this declaration, the scanner cannot distinguish an `@integral_read` on an in-memory cache (no serialization, no WL-009) from an `@integral_read` on a database query (serialization, WL-009 applies). This is consistent with the framework's design: the manifest declares the trust topology, and the scanner enforces rules against it.

#### 2.1.3 Severity matrix

WL-009 is a structural verification rule. Like WL-007 and WL-008, its severity is **ERROR/UNCONDITIONAL** across all eight taint states.

| Rule | INTEGRAL | ASSURED | GUARDED | Ext. Raw | Unk. Raw | Unk. Guarded | Unk. Assured | Mixed Raw |
|------|----------|---------|---------|----------|----------|--------------|--------------|-----------|
| **WL-009** | E/U | E/U | E/U | E/U | E/U | E/U | E/U | E/U |

**Rationale:** Like WL-007 (boundaries must reject) and WL-008 (semantic validation requires prior shape validation), WL-009 enforces a framework invariant — the restoration model's evidence requirement — rather than a context-dependent pattern judgement. A bare `@integral_read` on a serialization path is structurally unsound regardless of the enclosing taint state, because the deficiency is in the annotation topology, not in the code patterns within the function body.

#### 2.1.4 Interaction with existing rules

- **WL-007** continues to apply to declared `@restoration_boundary` functions — WL-009 ensures a restoration boundary exists; WL-007 ensures it contains a rejection path. The two rules are complementary: WL-009 catches the case where no restoration boundary is declared at all; WL-007 catches the case where one is declared but is structurally unsound.
- **§5.3 evidence categories** remain the authority on what evidence is required for each restoration tier. WL-009 does not duplicate that specification — it enforces that the evidence framework is *invoked*, not that the evidence is *sufficient*. Sufficiency remains a governance-reviewed claim (§12, residual risk 10).
- **Coherence checks (§9.2)** already detect orphaned annotations and undeclared boundaries. WL-009 is distinct: it detects a *missing* annotation (no restoration boundary) rather than a *mismatched* annotation (boundary declared but not found in code).

#### 2.1.5 Conformance implications

WL-009 is a structural verification rule. Under §14.2 criterion 3, conformant tools that implement structural verification MUST enforce WL-009 alongside WL-007 and WL-008. The Wardline-Core enforcement profile (§14.3.1) includes WL-009 when the tool's declared rule set includes structural verification rules.

**Golden corpus requirement (§10):** WL-009 requires specimens in the INTEGRAL taint state at minimum:

- **True positive:** `@integral_read` function reading from a manifest-declared serialized store, no `@restoration_boundary`.
- **True positive:** `@integral_read` function with `@restoration_boundary` that declares no evidence categories (degenerate restoration — structurally equivalent to no boundary).
- **True negative:** `@integral_read` function that co-declares `@restoration_boundary` with structural + semantic + integrity evidence.
- **True negative:** `@integral_read` function whose data source is not manifest-declared as a serialization boundary (in-memory Tier 1 read).
- **Adversarial false positive:** `@integral_read` function whose inputs trace through a restoration boundary in a called function (two-hop satisfaction — should not fire).

#### 2.1.6 Manifest schema change

The root manifest schema (`wardline.schema.json`) requires no change. The serialization-boundary declaration mechanism already exists in the overlay schema's `boundary_declarations` section. If an implementation finds the existing overlay structure insufficient for expressing "this data source involves serialization," a schema extension adding a `serialization_boundary: true` flag to `boundary_declarations` entries would be a minimal, backward-compatible addition.

#### 2.1.7 Adoption impact

WL-009 activates only when `@integral_read` is used. Projects that have not yet annotated their Tier 1 read paths are unaffected — the rule fires on the annotation, not on the code pattern. This means WL-009 has zero adoption friction for projects in early annotation stages, and creates a natural incentive to annotate read paths correctly: when you add `@integral_read`, you are prompted to also declare the restoration evidence.

For projects with existing `@integral_read` annotations that lack `@restoration_boundary` co-declarations, WL-009 findings surface the gap for governance review. The remediation path is to add `@restoration_boundary` with appropriate evidence categories — a governance decision, not a code change (though the restoration boundary function itself may need structural modification to include rejection paths per WL-007).

---

### 2.2 Non-normative note: Deep immutability at Tier 1 (prime spec)

#### 2.2.1 Proposed text

**Proposed addition to §4.1, after the coding posture model paragraph, or to §8 as a binding guidance note:**

> *Non-normative.* Language-level immutability mechanisms may provide only shallow guarantees — preventing attribute reassignment or reference rebinding while permitting mutation of contained values (nested dictionaries, lists, or equivalent mutable containers). When Tier 1 data structures rely on such mechanisms to establish their integrity contract, the declared immutability may be a false structural guarantee: downstream code — including the enforcement tool — treats the structure as immutable, but its contents can be silently modified after construction.
>
> Bindings whose target language has shallow immutability mechanisms SHOULD define supplementary rules that detect Tier 1 and Tier 2 data structures whose declared immutability does not achieve deep immutability. The detection criteria are necessarily language-specific — Python's `frozen=True` on dataclasses, Java's `final` on reference fields, Kotlin's `val` on collection properties — and belong in the binding, not the framework. The framework principle is: **if a data structure's immutability declaration is part of its Tier 1 integrity contract, the declaration must be truthful.**
>
> Bindings for languages where the immutability mechanism is inherently deep (e.g., Rust's ownership model, Haskell's persistent data structures) need not define such rules — the language guarantee is sufficient.

#### 2.2.2 Rationale for non-normative status

This note is non-normative because:

1. **Not all languages have the problem.** Rust's ownership model, Haskell's persistent data structures, and similar mechanisms provide deep immutability by default. A normative requirement would impose a burden on bindings where the problem does not exist.
2. **Detection is entirely language-specific.** The patterns that constitute "shallow freeze" differ fundamentally across languages. Python's `MappingProxyType` wrapping, Java's `Collections.unmodifiableMap()`, and Kotlin's `toList()` all share the same semantic problem but have no syntactic commonality.
3. **The principle cascades.** A non-normative note in the prime spec creates a clear mandate for bindings to address the gap, without prescribing how. Binding authors can cite the note as the framework-level justification for their supplementary rules.

---

### 2.3 PY-WL-010 and PY-WL-011: Shallow freeze detection (Python binding)

These rules implement the deep immutability principle from §2.2 for the Python binding.

#### 2.3.1 PY-WL-010: Bare MappingProxyType wrap in `__post_init__`

**Pattern:** `MappingProxyType(dict(self.x))` or `MappingProxyType(self.x)` in the `__post_init__` method of a frozen dataclass.

**Why it is dangerous:** `MappingProxyType` wraps a dictionary to prevent key/value *assignment* on the proxy, but nested mutable containers (dicts within dicts, lists as values) remain fully mutable through the proxy. `MappingProxyType(dict(self.x))` copies the outer dict (preventing caller mutation of the original) but leaves nested structures shared and mutable. For Tier 1 data, this means the integrity guarantee is only one level deep — a nested dict on an audit record can be silently modified after construction, and the modification is invisible to any code that trusts the `frozen=True` declaration.

**Detection:** AST visitor within `__post_init__` methods of classes decorated with `@dataclass(frozen=True)`. Match `ast.Call` nodes where the function resolves to `MappingProxyType` and the argument is either `self.<field>` or `dict(self.<field>)`. Exclude calls where the argument is already wrapped in `deep_freeze()` or equivalent recursive-immutability function (configurable via manifest or scanner configuration).

**Remediation:** Replace `MappingProxyType(dict(self.x))` with `deep_freeze(self.x)` or equivalent function that recursively converts `dict` to `MappingProxyType`, `list` to `tuple`, and `set` to `frozenset` through arbitrary nesting depth. The Python binding SHOULD document the expected signature and semantics of the deep-freeze function so that custom implementations can be recognised by the scanner.

#### 2.3.2 PY-WL-011: isinstance freeze guard bypass

**Pattern:** `isinstance(self.x, (dict, tuple, MappingProxyType, Mapping, frozenset))` used as a conditional guard for freezing in `__post_init__` methods of frozen dataclasses.

**Why it is dangerous:** Type guards used to conditionally skip freezing are fragile across two dimensions. First, they check the *container type* but not the *content types* — a `tuple` of mutable dicts passes an `isinstance(self.x, tuple)` guard but its contents are fully mutable. Second, they are not exhaustive — a `Mapping` subclass that is not `dict` or `MappingProxyType` may pass through unfrozen. The correct approach is an idempotent deep-freeze function that handles all container types without guards — if the value is already deeply frozen, the function returns it unchanged; if not, it freezes it. No type guard is needed because the operation is safe on all inputs.

**Detection:** AST visitor within `__post_init__` methods of classes decorated with `@dataclass(frozen=True)`. Match `isinstance` calls where:
- The first argument is `self.<field>`
- Any of the type arguments are in the set `{dict, tuple, MappingProxyType, frozenset, Mapping, list, set}`
- The `isinstance` call is the test expression of an `if` statement whose body contains `object.__setattr__` calls (the frozen-dataclass mutation pattern)

**Remediation:** Replace the `isinstance`-guarded conditional with an unconditional call to an idempotent deep-freeze function. The function should be identity-preserving on already-frozen values (no unnecessary copying) and handle all Python container types including arbitrary `Mapping` subclasses.

#### 2.3.3 Severity matrix

Both rules share the same severity profile. The rationale follows WL-001's gradient: shallow immutability is an integrity violation at Tier 1 (where the immutability contract is part of the authority guarantee), suspicious at Tier 2/3 (where it weakens but does not destroy structural guarantees), and suppressed at Tier 4 (where data has not been promoted and container mutability is expected).

| Rule | INTEGRAL | ASSURED | GUARDED | Ext. Raw | Unk. Raw | Unk. Guarded | Unk. Assured | Mixed Raw |
|------|----------|---------|---------|----------|----------|--------------|--------------|-----------|
| **PY-WL-010** | E/U | E/St | W/R | Su/T | Su/T | W/R | E/St | Su/T |
| **PY-WL-011** | E/U | E/St | W/R | Su/T | Su/T | W/R | E/St | Su/T |

**INTEGRAL (E/U):** Shallow immutability on a Tier 1 record is an unconditional integrity violation. The immutability declaration is part of the audit-trail contract. A frozen dataclass with mutable nested contents can be silently modified after construction — the audit record is not tamper-resistant as declared.

**ASSURED (E/St):** Tier 2 data has passed semantic validation. Shallow immutability weakens the structural guarantee that downstream code relies on, but the data is not part of the legal record. Standard exception governance applies — the project MAY accept this risk with documented rationale (e.g., the mutable contents are never accessed after construction in the project's specific usage).

**GUARDED (W/R):** Tier 3 data has structural guarantees but not semantic guarantees. Shallow immutability is suspicious but the structural guarantee is already limited. Relaxed exception governance.

**EXTERNAL_RAW, UNKNOWN_RAW, MIXED_RAW (Su/T):** Data at these taint states has not been promoted. Container mutability is expected — the data is being processed, not preserved. Suppress.

#### 2.3.4 Relationship to framework WL-001

PY-WL-010 and PY-WL-011 are **not** sub-rules of WL-001. WL-001 detects *access patterns* (member access with fallback default) that fabricate data. PY-WL-010/011 detect *construction patterns* (shallow freeze in `__post_init__`) that create false immutability guarantees. The semantic risk is different: WL-001 addresses data fabrication; PY-WL-010/011 address structural-guarantee fraud. They share a severity profile because the tier-sensitivity gradient is the same — the higher the tier, the more the false guarantee matters — but they are independently derived rules.

#### 2.3.5 Golden corpus specimens

**PY-WL-010 specimens:**

- **True positive (INTEGRAL):** Frozen dataclass with `MappingProxyType(dict(self.data))` in `__post_init__` where `data` is `Mapping[str, Any]` containing nested dicts.
- **True positive (INTEGRAL):** Frozen dataclass with `MappingProxyType(self.data)` (view, not copy) in `__post_init__`.
- **True negative (INTEGRAL):** Frozen dataclass with `deep_freeze(self.data)` in `__post_init__`.
- **True negative (INTEGRAL):** Frozen dataclass with scalar-only fields (no container fields) — no `__post_init__` freeze needed.
- **Adversarial false positive:** `MappingProxyType(dict(self.data))` in a non-frozen dataclass (not subject to freeze-guard rules).
- **Adversarial false positive:** `MappingProxyType` used outside `__post_init__` (not a freeze guard, may be legitimate view construction).

**PY-WL-011 specimens:**

- **True positive (INTEGRAL):** `if isinstance(self.data, dict): object.__setattr__(self, "data", MappingProxyType(dict(self.data)))` — type guard skips freezing for non-dict types.
- **True positive (INTEGRAL):** `if not isinstance(self.data, tuple):` guard that skips freeze for tuples (tuple of mutable dicts passes unfrozen).
- **True negative (INTEGRAL):** Unconditional `freeze_fields(self, "data")` call with no `isinstance` guard.
- **True negative:** `isinstance` check in `__post_init__` that is not a freeze guard (e.g., validation logic).
- **Adversarial false positive:** `isinstance` check in `__post_init__` of a non-frozen dataclass.

---

## 3. Excluded From This RFC

### 3.1 Frozen annotation enforcement

Detecting mutable container type annotations (`list[...]`, `dict[...]`, `set[...]`) on fields of frozen dataclasses — where `__post_init__` converts them to immutable types but the type annotation still permits mutation through the type checker — is a **type-system hygiene concern**, not a trust-tier concern.

The invariant being enforced ("type annotations must not lie about mutability") is orthogonal to the wardline's enforcement surface. It does not depend on tier classification, taint state, or boundary declarations. It belongs in the type-checking layer (mypy plugin, ruff rule, or equivalent linter) rather than the wardline binding.

### 3.2 Layer-boundary import enforcement

While ELSPETH's layer dependency enforcement (L0 contracts → L1 core → L2 engine → L3 plugins) maps to wardline's Group 6 (Layer Boundaries), this RFC does not propose changes to Group 6. The existing Group 6 annotation vocabulary and enforcement consequences are sufficient — ELSPETH's layer enforcement would be a consumer of Group 6, not a change to it.

---

## 4. Implementation Sketch

This section is non-normative. It describes one possible implementation path for a wardline Python binding scanner implementing the proposed rules.

### 4.1 WL-009 implementation

1. **Manifest loader** reads `boundary_declarations` and identifies entries with `serialization_boundary: true` (or equivalent flag — see §2.1.6).
2. **Annotation discovery** identifies functions with `@integral_read` and `@restoration_boundary`.
3. **Coherence check** cross-references: for each `@integral_read` function whose data source is a declared serialization boundary, verify that the function either (a) co-declares `@restoration_boundary` with at least one evidence category, or (b) its inputs trace through a declared restoration boundary within two-hop scope.
4. **Finding emission** produces SARIF output with `ruleId: "WL-009"`, the function's location, and remediation guidance pointing to §5.3.

### 4.2 PY-WL-010 / PY-WL-011 implementation

Both rules are intraprocedural AST visitors scoped to `__post_init__` methods of frozen dataclasses. The existing scanner infrastructure for rule visitors (the `BaseRule` class and `ScanContext`) is sufficient — no new scanner architecture is required.

1. **Class identification:** Walk AST for `ClassDef` nodes with `@dataclass(frozen=True)` in their decorator list.
2. **Method scoping:** Locate `__post_init__` method within the class body.
3. **PY-WL-010:** Within `__post_init__`, match `ast.Call` nodes where `func` resolves to `MappingProxyType` and the argument pattern matches `self.<field>` or `dict(self.<field>)`.
4. **PY-WL-011:** Within `__post_init__`, match `isinstance` calls where the first argument is `self.<field>`, any checked type is in the freeze-guard type set, and the `isinstance` appears as the test of an `if` statement whose body contains `object.__setattr__` calls.
5. **Taint context:** The enclosing class's tier is determined by module-level manifest assignment. The severity matrix cell is looked up from the rule's matrix row and the effective taint state.

---

## 5. Summary of Proposed Spec Changes

| Change | Spec location | Appendix | Type | Impact |
|--------|--------------|----------|------|--------|
| WL-009 rule definition | §7.1 (rule table) | A.1 | Normative | New row in rule table |
| WL-009 structural verification | §7.2 | A.2 | Normative | New paragraph after WL-008 |
| WL-009 severity matrix | §7.3 (severity matrix) | A.3 | Normative | New row: E/U across all states; update preamble |
| Rule count update | §7 (intro paragraph) | A.4 | Normative | "eight rules" → "nine rules"; "Two structural" → "Three structural" |
| Deep immutability note | §4.1 | A.5 | Non-normative | Binding guidance paragraph |
| WL-009 static analysis | §8.1 (requirements list) | A.6 | Normative | New bullet after WL-008 |
| WL-009 conformance | §14.2 criterion 3 | A.7 | Normative | Add WL-009 to structural verification criterion |
| WL-009 manifest schema | §13 / overlay schema | A.8 | Normative (minimal) | Optional `serialization_boundary` flag |
| WL-009 severity rationale | §7.5 (worked examples) | A.9 | Non-normative | Update WL-007/WL-008 paragraph to include WL-009 |
| SARIF presentation | §10.1 | A.10 | Normative | Add WL-009 to taint-state omission guidance |
| Python binding conformance | Part II-A (criterion table) | A.11 | Binding-normative | Add PY-WL-012 to criterion 3 row |
| PY-WL-010 rule definition | Part II-A | — | Binding-normative | New Python-specific rule |
| PY-WL-011 rule definition | Part II-A | — | Binding-normative | New Python-specific rule |
| PY-WL-010/011 severity matrix | Part II-A | — | Binding-normative | New rows in Python binding matrix |
| PY-WL-010/011 corpus specimens | Part II-A / corpus/ | — | Binding-normative | ~12 specimens across both rules |

---

## 6. Open Questions

1. **WL-009 manifest mechanism.** Should serialization boundaries be declared via a new `serialization_boundary: true` flag on existing `boundary_declarations` entries, or via a new top-level manifest section? The former is minimal; the latter may be cleaner if the set of serialization boundaries grows to include caches, message queues, and other storage media.

2. **WL-009 scope for `@integral_construction`.** Should WL-009 also fire when `@integral_construction` (the T2→T1 promotion) reads from a serialization boundary without restoration evidence? The construction decorator implies the function is *creating* a new Tier 1 artefact from Tier 2 inputs, but if those inputs come from a serialized store, the same evidence gap exists. The conservative answer is yes — any Tier 1 promotion from a serialized source requires restoration evidence.

3. **PY-WL-010/011 deep-freeze function recognition.** How should the scanner identify "equivalent recursive-immutability functions" beyond a hardcoded `deep_freeze` name? Options: (a) manifest-declared function paths, (b) decorator-based (`@idempotent_freeze`), (c) naming convention. Option (a) is consistent with the manifest-driven design; option (b) adds to the annotation surface; option (c) is fragile.

4. **PY-WL-010 scope beyond `MappingProxyType`.** Should the rule also detect `tuple(self.x)` in `__post_init__` where `self.x` is a `list` of mutable containers? This is the same shallow-freeze problem — `tuple` prevents list mutation but the tuple's elements (if they are dicts or lists) remain mutable. The detection broadens but so does the false-positive surface.

---

## Appendix A: Proposed Spec Text (Exact Deltas)

This appendix provides the exact text changes to each affected spec section. Insertions are marked with `[+]` prefix. Unchanged surrounding text is included for placement context.

### A.1 §7.1 — Rule table: Insert WL-009 row

**Location:** After the WL-008 row in the §7.1 rule table.

**Current text (last row):**

| **WL-008** | Semantic validation without prior shape validation | Data reaching a declared semantic-validation boundary ... |

**Insert after WL-008:**

[+] | **WL-009** | Integral-read on serialization path without restoration evidence | A function declared `@integral_read` (or binding equivalent) whose data source is a manifest-declared serialization boundary, and which neither co-declares `@restoration_boundary` with at least one evidence category nor traces its inputs through a declared restoration boundary within the two-hop analysis scope (§8.1). The function claims Tier 1 authority for deserialized data on assertion alone — the restoration model (§5.3) requires evidence-backed provenance claims, but no evidence is declared, no governance review of the restoration act is possible, and the scanner cannot verify that write-side invariants are re-established on read. This is the restoration-symmetry failure: construction paths include validation that serialization strips, and the read side re-stamps the deserialized representation as authoritative without re-verifying. WL-009 enforces that the restoration model is *invoked*, not that the evidence is *sufficient* — sufficiency remains a governance-reviewed claim (§12, residual risk 10). |

---

### A.2 §7.2 — Structural verification: Add WL-009 paragraph

**Location:** After the WL-008 paragraph in §7.2.

**Insert after the WL-008 paragraph:**

[+] **WL-009: Integral-read paths through serialization boundaries MUST declare restoration evidence.** This is a topology constraint, not a body-content check. The scanner cross-references `@integral_read` annotations against manifest-declared serialization boundaries and `@restoration_boundary` annotations. WL-009 fires when all three conditions hold: (1) a function is declared `@integral_read` (Group 1); (2) the function's data source is a manifest-declared serialization boundary (identified via `boundary_declarations` entries with `serialization_boundary: true`, or via `dependency_taint` entries for serialized stores); (3) the function does not co-declare `@restoration_boundary` (Group 17) with at least one evidence category, and its inputs do not trace through a declared restoration boundary with sufficient evidence within the two-hop analysis scope (§8.1). Functions whose `@integral_read` data source is not manifest-declared as a serialization boundary (e.g., in-memory Tier 1 reads) are not subject to WL-009. The manifest dependency is intentional: serialization boundaries are part of the trust topology and are declared in the manifest, not inferred by the scanner.

---

### A.3 §7.3 — Severity matrix: Insert WL-009 row

**Location:** After the WL-008 row in the §7.3 severity matrix.

**Current text (last rows):**

| **WL-007** | Validation with no rejection path | E/U | E/U | E/U | E/U | E/U | E/U | E/U | E/U |
| **WL-008** | Semantic validation without shape validation | E/U | E/U | E/U | E/U | E/U | E/U | E/U | E/U |

**Insert after WL-008:**

[+] | **WL-009** | Integral-read without restoration evidence | E/U | E/U | E/U | E/U | E/U | E/U | E/U | E/U |

**Update the paragraph above the matrix** (currently reads "WL-007 and WL-008 are structural verification rules"):

**Current:**
> WL-007 and WL-008 are structural verification rules (not pattern rules) and apply only to declared boundary functions, but are shown in the matrix for completeness. Their severity is UNCONDITIONAL across all contexts because they are framework invariants rather than context-dependent judgements.

**Proposed:**
> [+] WL-007, WL-008, and WL-009 are structural verification rules (not pattern rules) and apply only to declared boundary functions, but are shown in the matrix for completeness. Their severity is UNCONDITIONAL across all contexts because they are framework invariants rather than context-dependent judgements.

---

### A.4 §7 introductory paragraph: Update rule count

**Current (§7, first sentence):**
> This section defines eight rules in two categories. **Six pattern rules** (WL-001 through WL-006) detect syntactic proxies for semantic violations in declared semantic contexts. ... **Two structural verification rules** (WL-007 and WL-008) enforce invariants on declared boundary functions ...

**Proposed:**
> [+] This section defines nine rules in two categories. **Six pattern rules** (WL-001 through WL-006) detect syntactic proxies for semantic violations in declared semantic contexts. ... **Three structural verification rules** (WL-007, WL-008, and WL-009) enforce invariants on declared boundary functions ...

---

### A.5 §4.1 — Non-normative note on deep immutability

**Location:** After the "Verification gradient" paragraph (which ends with "...it narrows the residual governance surface but does not change which conformance profile applies."), before the paragraph beginning "Semantic validation is always comprehensive within its scope".

**Insert:**

[+] *Non-normative.* Language-level immutability mechanisms may provide only shallow guarantees — preventing attribute reassignment or reference rebinding while permitting mutation of contained values (nested dictionaries, lists, or equivalent mutable containers). When Tier 1 data structures rely on such mechanisms to establish their integrity contract, the declared immutability may be a false structural guarantee: downstream code — including the enforcement tool — treats the structure as immutable, but its contents can be silently modified after construction. Bindings whose target language has shallow immutability mechanisms SHOULD define supplementary rules that detect Tier 1 and Tier 2 data structures whose declared immutability does not achieve deep immutability. The detection criteria are necessarily language-specific — Python's `frozen=True` on dataclasses, Java's `final` on reference fields, Kotlin's `val` on collection properties — and belong in the binding, not the framework. The framework principle is: if a data structure's immutability declaration is part of its Tier 1 integrity contract, the declaration must be truthful. Bindings for languages where the immutability mechanism is inherently deep (e.g., Rust's ownership model, Haskell's persistent data structures) need not define such rules — the language guarantee is sufficient.

---

### A.6 §8.1 — Static analysis requirements: Add WL-009 bullet

**Location:** After the WL-008 enforcement bullet in the §8.1 requirements list.

**Current (WL-008 bullet):**
> - MUST enforce validation ordering (WL-008): data reaching a declared semantic-validation boundary MUST have passed through a declared shape-validation boundary *(framework invariant)*. Combined validation boundaries (T4→T2) satisfy this requirement internally

**Insert after:**

[+] - MUST enforce restoration symmetry (WL-009): a function declared `@integral_read` whose data source is a manifest-declared serialization boundary MUST co-declare `@restoration_boundary` with at least one evidence category, or its inputs MUST trace through a declared restoration boundary within the two-hop analysis scope *(framework invariant)*. WL-009 is a topology check on the annotation surface and does not require body-content analysis beyond what taint-flow tracing already provides

---

### A.7 §14.2 — Conformance criterion 3: Update to include WL-009

**Current:**
> 3. Structural verification: WL-007 is enforced on all validation boundary functions (shape, semantic, combined, and restoration) and WL-008 (validation ordering) is enforced on semantic-validation boundaries (§7.2, §8.1)

**Proposed:**
> [+] 3. Structural verification: WL-007 is enforced on all validation boundary functions (shape, semantic, combined, and restoration), WL-008 (validation ordering) is enforced on semantic-validation boundaries, and WL-009 (restoration symmetry) is enforced on `@integral_read` functions whose data sources are manifest-declared serialization boundaries (§7.2, §8.1)

---

### A.8 §13 — Manifest overlay schema: Add serialization_boundary flag

**Location:** Within the `boundary_declarations` entry schema in the overlay manifest schema.

**Proposed addition to boundary entry properties:**

```yaml
serialization_boundary:
  type: boolean
  default: false
  description: >
    Whether this boundary involves serialization/deserialization
    (database read/write, file I/O, message queue, cache).
    When true, @integral_read functions using this boundary
    as a data source are subject to WL-009 (restoration
    symmetry). In-memory data sources should not set this flag.
```

This is a backward-compatible, optional addition. Existing manifests without `serialization_boundary` entries are unaffected — WL-009 only fires when a serialization boundary is positively declared.

---

### A.9 §7.5 — Worked example rationale: Update WL-007/WL-008 paragraph

**Location:** §7.5 (or §7.4 depending on section numbering), the paragraph explaining structural verification severity.

**Current:**
> WL-007 (validation boundary structural verification) and WL-008 (semantic validation ordering) are both UNCONDITIONAL across all eight states. A validation function that contains no rejection path is structurally unsound regardless of context. Semantic validation applied to structurally unverified data is a category error regardless of context. These are framework invariants, not context-dependent judgements.

**Proposed:**
> [+] WL-007 (validation boundary structural verification), WL-008 (semantic validation ordering), and WL-009 (restoration symmetry) are all UNCONDITIONAL across all eight states. A validation function that contains no rejection path is structurally unsound regardless of context. Semantic validation applied to structurally unverified data is a category error regardless of context. An integral-read function on a serialization path without declared restoration evidence is a topology deficiency regardless of context. These are framework invariants, not context-dependent judgements.

---

### A.10 §10.1 — SARIF finding presentation: Update taint-state omission guidance

**Location:** §10.1, the paragraph on taint state omission for structural verification rules.

**Current:**
> For WL-007 (validation boundary integrity) and WL-008 (restoration boundary integrity), bindings SHOULD omit the taint state from primary finding messages. These structural verification rules are UNCONDITIONAL across all eight effective states — the taint state of the enclosing context is irrelevant to the finding because the rule fires on structural properties (boundary declaration completeness, rejection-path reachability) rather than data-flow properties.

**Proposed:**
> [+] For WL-007 (validation boundary integrity), WL-008 (validation ordering integrity), and WL-009 (restoration symmetry), bindings SHOULD omit the taint state from primary finding messages. These structural verification rules are UNCONDITIONAL across all eight effective states — the taint state of the enclosing context is irrelevant to the finding because the rule fires on structural properties (boundary declaration completeness, rejection-path reachability, restoration evidence presence) rather than data-flow properties.

**Current (next paragraph):**
> Including the taint state in WL-007/WL-008 primary messages trains developers to believe it matters for structural verification, creating a false mental model of how these rules operate.

**Proposed:**
> [+] Including the taint state in WL-007/WL-008/WL-009 primary messages trains developers to believe it matters for structural verification, creating a false mental model of how these rules operate.

---

### A.11 Part II-A — Python binding conformance table: Update criterion 3

**Location:** Part II-A, conformance criterion table, row 3.

**Current:**
> | 3 | Structural verification WL-007/WL-008 | PY-WL-008 (rejection path), PY-WL-009 (validation ordering) | `wardline corpus verify`; engine L3 integration tests |

**Proposed:**
> [+] | 3 | Structural verification WL-007/WL-008/WL-009 | PY-WL-008 (rejection path), PY-WL-009 (validation ordering), PY-WL-012 (restoration symmetry) | `wardline corpus verify`; engine L3 integration tests |

**Note:** WL-009 maps to PY-WL-012 in the Python binding (PY-WL-010 and PY-WL-011 are allocated to the shallow freeze rules proposed in §2.3 of this RFC). The binding rule numbering follows the existing Python binding convention where framework structural rules are renumbered into the binding's rule sequence.
