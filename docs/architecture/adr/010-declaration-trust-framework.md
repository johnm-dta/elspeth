# ADR-010: Declaration-trust framework — generalized contract protocol for plugin declarations

**Date:** 2026-04-19
**Status:** Accepted (amended 2026-04-20 — see Amendment A3 below for H2 cluster)
**Deciders:** John Morrissey (author); five-reviewer panel (solution-architect, systems-thinker, python-engineer, quality-engineer, security-architect) per session `/home/john/.claude/plans/elspeth-track-2-phase-2a-declaration-framework.review.json`
**Supersedes:** (partial, clause-level) — see "Supersession map" below
**Tags:** framework, declaration-contract, audit-evidence, tier-1, registry, audit-integrity
**Review date:** 2026-10-19 (six months from acceptance; ADR-010 §Consequences must be re-evaluated against observed 2B/2C experience by that date)
**Filigree epic:** `elspeth-a3ac5d88c6` (Track 2 — Declaration-trust framework Phase 2B/2C; hard SLA 2026-07-18)

---

> **Amendment A3 — 2026-04-20 (H2 cluster: `elspeth-425047a599`, `elspeth-10dc0b747f`, `elspeth-60890a7388`, `elspeth-f52d7c5a47`, `elspeth-5fc876138d`, `elspeth-b513c01cff`, `elspeth-121b268aec`, `elspeth-5dae105959`).**
>
> **What changed.** The 2A single-site `DeclarationContract` Protocol is replaced by a nominal 4-site ABC (C+B′ synthesis per panel comment #408). The dispatcher adopts **audit-complete** semantics (comment #417 anchor — see §Semantics below). The `EXPECTED_CONTRACTS: frozenset[str]` manifest becomes `EXPECTED_CONTRACT_SITES: Mapping[str, frozenset[DispatchSiteName]]`; the CI scanner gains `MC3a`/`MC3b`/`MC3c` rules. A pre-emission dispatch site is added (F2). The inline sink check is reclassified as `SinkTransactionalInvariantError` (F3, distinct from the future `SinkRequiredFieldsContract`). Rule-of-three resets per dispatch site (F4, §Adoption State). Aggregate violation semantics (`AggregateDeclarationContractViolation`, N3) are adopted. See §Semantics + §Adoption State + §H2 landing scope below for details.
>
> **Banner authority.** The sections below (Semantics, Adoption State, H2 landing scope) supersede the original 2A framing where they overlap. The original prose is preserved verbatim for audit-trail purposes.

> **Amendment A1 — 2026-04-20 (issue `elspeth-3f320398f1` / M6).**
> The Supersession-map row for **ADR-008 §Explicit scope boundary** claims the 2A framework provides what each future declaration adopts. The accurate statement is narrower: the *scope* statement is preserved, but `static_check` was deferred entirely to Phase 2B's walker refactor (see §Decision 3). Under 2A the protocol carries only `runtime_check` — new declarations adopting the 2A framework land their runtime-VAL contract today; walker-side checks wait for the Phase 2B shape decision (issue `elspeth-425047a599` / H2). The original row stands for audit-trail purposes; readers should treat this banner as the authoritative reading of the ADR-008 scope-boundary row under 2A. Precedent: ADR-009 §Alternatives Considered #4.

> **Amendment A2 — 2026-04-20 (issue `elspeth-3f320398f1` / M4).**
> The reversibility claim in §Consequences → §Neutral ("Reversibility: the framework is additive — removing it would require unpicking all decorator usages and re-introducing the hand-written `TIER_1_ERRORS` tuple. Non-trivial but not destructive.") is accurate at 2A's single-adopter state but becomes materially weaker as Phase 2B/2C adopters register. After six adopters land (four 2B transforms + two 2C boundaries), reversal requires a coordinated rollback of the six adopter modules, the dispatcher, the registry and its manifest (C2), the dispatch-site manifest (follow-up issue `elspeth-10dc0b747f`), and — if first-fire shadowing is fixed by then (issue `elspeth-60890a7388`) — the aggregate-violation machinery as well. Review date 2026-10-19 remains the formal reversibility checkpoint, but the practical window closes earlier: if reversal is being seriously considered it must happen before the first 2B adopter lands, not after. The original prose stands for audit-trail purposes; this banner is the authoritative reversibility reading as of the 2026-04-20 post-panel cleanup. Precedent: ADR-009 §Alternatives Considered #4.

---

## Supersession map

This ADR amends (not replaces) the single-declaration pattern from 007/008/009:

| Prior clause | Status under ADR-010 |
|--------------|----------------------|
| ADR-007 §Decision 1-3 | Remain normative for `passes_through_input`. New declarations use the ADR-010 framework. |
| ADR-007 §Negative Consequences #2 (duplicated walkers) | Resolved by ADR-009 §Clause 1. ADR-010 does not re-visit. |
| ADR-008 §"cross-check applies per-row in executor" | Generalised — the ADR-010 dispatcher replaces direct executor calls. Scope per-row preserved. |
| ADR-008 §Explicit scope boundary | Normatively remains; ADR-010 provides the framework each future declaration adopts, but each new declaration still requires its own ADR. |
| ADR-009 §Clause 1 (shared propagation primitives) | Remains normative. ADR-010 does NOT include `static_check` in the 2A protocol — the walker refactor is deferred entirely to Phase 2B. |
| ADR-009 §Clause 2 (runtime cross-check batch path) | Remains normative. ADR-010's dispatcher preserves the single-token and batch-flush call sites. |
| ADR-009 §Clause 3 (empty-emission carve-out + 90-day SLA) | Remains normative. The Track 2 filigree epic (ref above) anchors the SLA. |
| ADR-009 §Clause 4 (invariant harness) | ADR-010 extends — harness iterates registered contracts and exercises every `negative_example`. |

## Context

ADRs 007, 008, and 009 established a single-declaration implementation of the "declare + statically trust + runtime-verify" pattern for `passes_through_input`. The pattern generalises: each plugin declaration the DAG validator trusts today (`creates_tokens`, `declared_output_fields`, `declared_required_fields`, `output_schema_config.mode`, source `guaranteed_fields`, sink `required_fields`) has the same gap ADR-008 closed for pass-through — static analysis trusts with no runtime backstop; mis-annotation silently corrupts Tier-1 audit data.

ADR-008 §Explicit scope boundary deferred the wider class. ADR-010 generalises the pattern into a framework that future declarations adopt in ~200 LOC per declaration, without duplicating the static/runtime/invariant trio.

**Rule-of-three satisfaction:** ADR-009 §Alternatives Considered #1 rejected "DeclarationContract protocol up front" as premature abstraction. ADR-010 satisfies rule-of-three by paper-sketching `DeclaredOutputFieldsContract` against the protocol during Task 0 (before Task 1 implementation) — if the sketch exposes protocol gaps, they are closed in Tasks 1–5. The protocol shape in 2A reflects what that sketch validated. The old `CreatesTokensContract` preview used during 2A exploration was later retired by ADR-015 once the codebase confirmed `creates_tokens` is a permission flag, not an obligation.

## Decision

Three new framework primitives in L0 `contracts/`, one L2 dispatcher, migrations of the four existing Tier-1 exceptions into the decorator-registry pattern, and nominal AuditEvidence widening of `NodeStateGuard`.

### Decision 1 — `AuditEvidenceBase` nominal abstract base class

Replaces the bespoke `PluginContractViolation.to_audit_dict()` discriminator. Violation classes MUST inherit `AuditEvidenceBase` explicitly to contribute structured context via `NodeStateGuard.__exit__`. Nominal (not structural) because a `@runtime_checkable` Protocol with one method admits accidental duck-type matches — this was the Critical finding from the security review (spoofing audit evidence via incidental `to_audit_dict` methods on unrelated classes). CI scanner `enforce_audit_evidence_nominal.py` enforces at build time.

### Decision 2 — `@tier_1_error(reason=...)` factory decorator + frozen registry

Replaces the hand-maintained `TIER_1_ERRORS` tuple. Three safeguards beyond decoration-is-registration:
1. **Required `reason` kwarg** (justification is queryable; grep-visible; ADR-cross-referenceable).
2. **Module-prefix allowlist** — only `elspeth.contracts.*`, `elspeth.engine.*`, `elspeth.core.*` callers. Plugin authors cannot unilaterally elevate their exceptions to Tier-1.
3. **`freeze_tier_registry()`** — called at end of orchestrator bootstrap; registration after freeze raises `FrameworkBugError`.

`errors.TIER_1_ERRORS` is a module `__getattr__` returning the live `tuple(tier_registry._REGISTRY)` — not a snapshot. Call sites doing `from elspeth.contracts.errors import TIER_1_ERRORS` at import time see the tuple at the moment of access (caveat: Python's `from X import Y` binds the local name to the value produced, which for a module `__getattr__` is a fresh call — the `except TIER_1_ERRORS:` clause evaluates the name `TIER_1_ERRORS` on each `except` entry, not at import, so this works correctly).

### Decision 3 — `DeclarationContract` protocol + frozen registry

Runtime-checkable protocol carrying:
- `name: str`
- `applies_to(plugin) -> bool`
- `runtime_check(inputs, outputs) -> None`
- `payload_schema: type[TypedDict]` — enforced at violation construction (deny-by-default) and as defence-in-depth at Landscape serialization (scrubber)
- `negative_example() -> tuple[RuntimeCheckInputs, RuntimeCheckOutputs]` (classmethod; harness asserts it fires the contract's violation)

`static_check` is NOT in the 2A protocol. Walker refactor is fully Phase 2B scope — designing a `static_check` shape without a second real walker-aware declaration would be premature abstraction (the ADR-009 §Alternatives #1 concern). Phase 2B's first declaration will validate the walker-side extension.

Registry freezes at bootstrap; orchestrator asserts equality between the registered per-contract dispatch-site map and the `EXPECTED_CONTRACT_SITES` manifest at the same point (prevents silent runtime VAL disable — reviewer B9's concern; the weaker non-empty check was replaced first by the C2 manifest gate and then tightened to per-site equality by Amendment A3/N1).

`DeclarationContractViolation` inherits `AuditEvidenceBase`. Every subclass declares its own `payload_schema` (TypedDict). At construction, `__init__` rejects any payload carrying undeclared keys or missing a required key (issue `elspeth-3956044fb7` / H5 Layer 1 — see **Payload-schema enforcement** below). After validation, the payload is deep-frozen; on `to_audit_dict` it is passed through `scrub_payload_for_audit` (H5 Layer 2 defence-in-depth).

## Consequences

### Positive

- Phase 2B (transform-level declaration VALs) and Phase 2C (boundary VALs) each become ~200-LOC PRs that register a new contract; no executor, validator, or harness plumbing.
- Registry-driven invariant harness exercises every contract uniformly. New declaration = new contract = automatic invariant coverage.
- `negative_example()` requirement closes the dormant-`runtime_check` failure mode — a silently-returning `runtime_check` fails CI immediately.
- Audit evidence is contract-orthogonal. A future non-plugin Tier-1 exception (e.g., checkpoint-integrity violation) can contribute structured context without subclassing `PluginContractViolation`.
- TIER_1 registration is greppable at the class declaration site, requires a `reason`, and rejects plugin-module callers.
- Secret-scrubbing on payload serialization closes the audit-legal-record-as-secret-leak vector (reviewer F-4) defensively. As of issue `elspeth-3956044fb7` / H5 the scrubber is the _second_ of a two-layer defence: violation `__init__` now rejects any undeclared payload key up-front (H5 Layer 1), so an unknown secret format cannot reach the scrubber at all. The expanded scrubber (Azure SAS, DB conn strings, basic-auth URLs, bearer/session key names) remains as defence-in-depth for cases where a schema declares a `str` field whose value happens to carry a secret (H5 Layer 2).

### Negative

- Indirection cost: per-row check path routes through `run_runtime_checks` iterating the registry. Negligible today (one registered contract; `applies_to` short-circuits). Task 12 benchmarks establish a dispatcher-overhead budget; the ADR-008 25 µs / 50 µs gate still applies to the total. See **NFR derivation** below for the bounded scaling law as 2B/2C adopters register.
- Registration ordering: contracts register on module import. A test clearing the registry must re-import the modules or use the snapshot-restore pattern demonstrated in `test_framework_accepts_second_contract.py`. `_clear_registry_for_tests` raises in production.
- Nominal AuditEvidenceBase requires author discipline: new audit-evidence classes MUST inherit explicitly. CI scanner is the backstop.
- The v0 empty-emission carve-out (ADR-009 §Clause 3) still stands; ADR-010 does not tighten it. The 90-day SLA trigger (2026-07-18) remains the safeguard for moving to `can_drop_rows` before the next `passes_through_input` transform with external calls registers.

### Neutral

- `PassThroughContractViolation` continues to exist as a dedicated subclass with its 9-key payload; it co-exists with the generic `DeclarationContractViolation` to preserve triage SQL that filters on `exception_type = 'PassThroughContractViolation'`. New contracts may reuse the generic form or introduce their own subclass.
- Reversibility: the framework is additive — removing it would require unpicking all decorator usages and re-introducing the hand-written `TIER_1_ERRORS` tuple. Non-trivial but not destructive. Review date 2026-10-19 is the formal reversibility checkpoint.

### NFR derivation (dispatcher overhead) — issue `elspeth-5dae105959` / H1

**Claim.** Per-row dispatcher overhead at N registered contracts and M applicable contracts is bounded by

    budget_median(N, M) = 27 µs + (M − 1) × 25 µs + (N − M) × 1.5 µs
    budget_p99(N, M)    = 2 × budget_median(N, M)

where the 27 µs N=1 baseline = 25 µs (ADR-008 direct `verify_pass_through` on a 200-field row) + 2 µs (dispatcher's own structural cost: `registered_declaration_contracts()` tuple materialisation + loop entry).

**Per-apply and per-skip cost derivation.** For N registered contracts with M applicable, the dispatcher runs `applies_to` N times and invokes M runtime checks. The 1.5 µs per-skip budget covers the cheap `applies_to=False` short-circuit path. The 25 µs per-apply budget matches the established 200-field pass-through happy-path gate and prevents the benchmark from measuring only skipped contracts while production rows exercise several applicable contracts.

**Parametric enforcement.** `tests/performance/benchmarks/test_cross_check_overhead.py::test_dispatcher_overhead_scales_with_registered_and_applicable_contracts` is parametrised over `(N_registered, M_applicable)` and registers one real `PassThroughDeclarationContract`, additional applicable contracts that observe 200-field emitted rows, and skipped contracts. The scaling law is asserted at every tuple, so the benchmark gates both registration-loop skip cost and real apply cost as 2B/2C adopters register.

**Aggregate throughput bound.** At 20 000 rows/sec per worker and the current gate ceiling of N = 16, M = 5, per-row dispatch overhead is ≤ (27 + 4 × 25 + 11 × 1.5) µs = 143.5 µs. This is the worst-case benchmark envelope before the review date (2026-10-19 / adoption-evidence triggers) re-evaluates the framework against observed experience.

**Amendment policy.** New contracts that register MUST preserve the scaling law. A contract with an intrinsically expensive `applies_to` (e.g. requiring a deep attribute walk) must either (a) cache its decision at registration time and short-circuit on a scalar, or (b) amend this derivation with a justification and tighten the per-skip budget. The CI benchmark is the enforcement mechanism.

### Payload-schema enforcement (deny-by-default) — issue `elspeth-3956044fb7` / H5

**Claim.** Every `DeclarationContractViolation` payload is validated against its subclass's `payload_schema` at construction time, BEFORE deep-freeze and BEFORE any serialization path. Undeclared keys and missing-required keys raise immediately.

**Why the gate is at construction.** The Landscape audit record is the legal record. The `scrub_payload_for_audit` helper is a closed-set regex/key-name list — it can only redact secret formats it has been taught about. A new contract author who accidentally includes an undeclared field (`debug_connection_string`, `raw_response`, `parent_url`) could slip a secret format past the scrubber. The construction-time gate flips the posture: a violation whose payload carries undeclared keys cannot be instantiated, so unknown secret formats never reach the scrubber. The scrubber remains as defence-in-depth for keys that ARE declared but carry a value that matches a known pattern.

**Mechanism.** `DeclarationContractViolation.payload_schema` is a `ClassVar[type]` defaulting to an empty `_EmptyPayload` TypedDict. Subclasses MUST override with a purpose-built TypedDict. `__init__` resolves `get_type_hints(schema, include_extras=True)` so `NotRequired[...]` / `Required[...]` wrappers are respected even when the defining module uses `from __future__ import annotations` (the metaclass-populated `__required_keys__` / `__optional_keys__` are unreliable under future annotations). Keys are classified via `typing.get_origin(annotation)`:

| Origin | Classification |
|--------|----------------|
| `Required` | required |
| `NotRequired` | optional |
| `None` | follows the class-level `total` flag (default `True` → required) |

Validation asserts `payload.keys() ⊆ required ∪ optional` AND `required ⊆ payload.keys()`. Violations raise `ValueError` with the schema name and the offending key set in the message — no debugger round-trip needed for triage.

**Layer 2 patterns.** The scrubber's `_PATTERNS` tuple and `_SECRET_KEY_NAMES` frozenset were extended in the same issue to cover Azure SAS tokens (`sig=…`), ODBC- and URL-style database connection strings (`postgres(ql)?://u:p@`, `mysql://`, `mongodb(+srv)?://`, case-insensitive `Password=` / `PWD=`), HTTP(S) basic-auth URLs (with a required `user:pass@` discriminator so credential-free endpoint URIs pass through), and bearer/session key names (`session_token`, `access_token`, `refresh_token`, `auth_cookie`, `sas_token`, `connection_string`, `conn_string`). Whole-string replacement is retained — partial redaction leaks structure.

**Adopter obligation.** Each 2B / 2C declaration contract's ADR MUST name the concrete violation subclass and its `payload_schema` TypedDict in the "Violation" subsection. CI does not yet scan for missing schemas (subclass discoverability is weaker than contract discoverability); the review gate is the ADR itself.

## Alternatives Considered

### Alternative 1: leave the pattern as a template, re-type per declaration

Phase 2B/2C engineers copy ADR-008's pattern and implement `_cross_check_<declaration>` ad-hoc. Each copy drifts. The duplicate L1/L3 walker issue ADR-009 closed would recur at scale — across declarations instead of layers. The registry is the structural fix.

### Alternative 2: generate contracts from declarative YAML

A YAML schema enumerates declarations; a code-generator produces stubs. Rejected for the same reason ADR-007 §Alternative 1 rejected AST auto-detect: magic is inappropriate for audit-grade code. The generated contract becomes the source of truth but the declaration author doesn't read it.

### Alternative 3: `AuditEvidence` as a structural `@runtime_checkable` Protocol

Rejected. The five-reviewer panel's security review identified this as a Critical Spoofing (STRIDE S) finding: any class exposing `to_audit_dict()` would satisfy the structural protocol, including accidental matches from third-party libraries, test helpers, or unrelated plugin code. Nominal `AuditEvidenceBase` requires explicit author declaration and closes the spoofing vector.

### Alternative 4: defer the entire `DeclarationContract` protocol to Phase 2B

Considered seriously in review. Rejected because: (a) the nominal `AuditEvidenceBase` widening and `@tier_1_error` decorator have immediate cleanup value independent of the protocol; (b) Task 0's paper-sketch pre-flight validates the protocol shape before code is written, satisfying rule-of-three; (c) the `DeclaredOutputFieldsContract` sketch named in §Context already exercised the shape before implementation, and the later adopters recorded in §Adoption State confirmed the shape remained viable after ADR-015 retired the exploratory `CreatesTokensContract` preview. The remaining residual risk (protocol reshape in 2B) is bounded by the small 2A protocol surface.

## References

- Plan: `/home/john/.claude/plans/elspeth-track-2-phase-2a-declaration-framework.md`
- Reviewer verdicts: `.review.json` alongside the plan file (5-reviewer panel, 2026-04-19)
- Predecessor ADRs: 007 (pass-through propagation), 008 (runtime cross-check), 009 (pathway fusion)
- Successor ADRs (Phase 2B/2C): each declaration gets its own ADR per §Supersession map ADR-008 reference
- CLAUDE.md §Three-Tier Trust Model, §Plugin Ownership, §Frozen Dataclass Immutability, §Defensive Programming Forbidden
- Track 2 filigree epic: `elspeth-a3ac5d88c6`; ADR-009 §Clause 3 SLA hard trigger 2026-07-18
- H2 cluster landing (2026-04-20): `elspeth-425047a599` (H2), `elspeth-10dc0b747f` (N1), `elspeth-60890a7388` (N3), `elspeth-f52d7c5a47` (F2), `elspeth-5fc876138d` (F3), `elspeth-b513c01cff` (F4), `elspeth-121b268aec` (F5), `elspeth-5dae105959` (H1 amendment)
- H2 design sketch: `docs/plans/2026-04-20-h2-amendment-design.md`
- H2 decision anchor: comment #417 on `elspeth-425047a599` — ADR-010 §Semantics audit-complete decision record

---

## §Semantics — Audit-complete dispatch (Amendment A3 anchor)

Added 2026-04-20 per comment #417 on `elspeth-425047a599`. This section is
load-bearing for the H2 cluster's F2/F3/F4/N1/N3 acceptance bullets; every
future reviewer who re-litigates "fail-fast vs audit-complete" MUST arrive
at the same conclusion or file a new comment on H2 to document disagreement.

**Decision.** The declaration-contract dispatcher adopts **audit-complete**
semantics. On a row that would violate multiple registered contracts'
invariants, the dispatcher continues iteration past a raised
`DeclarationContractViolation` (or `PluginContractViolation` subclass, per
the `PassThroughContractViolation` legacy), collects all violations
applicable to a single (row, call-site) tuple, and raises
`AggregateDeclarationContractViolation` when M > 1. Single-violation
(M = 1) cases raise the original violation unchanged via reference equality
(`raise violations[0]`).

**Rationale (verbatim for audit-trail posterity).**

> ELSPETH's CLAUDE.md "Auditability Standard" makes "I don't know what
> happened" structurally impermissible for any output. Under fail-fast
> first-fire semantics, the audit trail's silence on a second contract's
> evaluation is indistinguishable from "checked and passed" — a Repudiation
> surface (STRIDE) the auditor cannot resolve. Under audit-complete
> semantics, every applicable contract's method runs; every violation is
> recorded; absence-of-violation in the audit trail means "checked and
> passed," not "skipped because an earlier contract fired." The
> performance cost (M-applicable worst case) is bounded and quantified by
> H1's amended NFR derivation; the audit-completeness benefit is
> qualitative and load-bearing.
>
> A third option — fail-fast with pre-evaluation audit log — was
> considered (Security Architect S2-004) and rejected: emitting a separate
> audit record for "contracts evaluated" creates an attribution-confusion
> vector (which record is authoritative?) that audit-complete avoids by
> design. It also introduces a dual-write ordering problem: the
> pre-evaluation record must land before evaluation, so a crash between
> pre-record and evaluation leaves the audit trail permanently misleading.

**Scope of this decision.**

- Applies to all four dispatch sites named in the H2 cluster:
  `pre_emission_check`, `post_emission_check`, `batch_flush_check`,
  `boundary_check`. Each site uses the shared collect-then-raise helper
  `_dispatch` in `src/elspeth/engine/executors/declaration_dispatch.py`.
- F2 (pre-emission call site), F3 (sink-inline reclassification), F4
  (rule-of-three per site), N1 (per-site manifest) inherit this posture.
- `AggregateDeclarationContractViolation` is a **sibling** class of
  `DeclarationContractViolation` on `(AuditEvidenceBase, RuntimeError)`,
  **not a subclass**. Triage SQL: `WHERE is_aggregate = true`. Generic
  `except DeclarationContractViolation` does NOT absorb aggregates by
  design — see the catch-site survey in the H2 landing PR.

**Registry-order shadowing — closed.** The 2A dispatcher raised the first
violation immediately, terminating the loop. Under import-order changes,
refactors, or conditional imports, the "first-registered" contract could
silently vary, causing non-deterministic attribution of multi-violation
rows. Audit-complete + per-site registry eliminates this class of bug:
iteration order still exists, but every applicable contract's method runs
before any violation raises, and the aggregate carries ALL violations
regardless of registration order. (See `elspeth-60890a7388` / N3
§Acceptance for the order-independence test `pytest -p no:randomly
--forked`.)

---

## §Adoption State — per dispatch surface (Amendment A3, F4)

Added 2026-04-20 per `elspeth-b513c01cff` (F4), updated 2026-04-20 after
Phase 2B Tasks 3-7 and the paired Phase 2C boundary adopters landed. The 2A rule-of-three criterion treated the
single-site registry as one surface; under H2 each dispatch site is its
own surface with its own rule-of-three gate. A contract's landing on a
new surface does NOT reset the rule for sites that already counted;
conversely, one boundary adopter does NOT prove the boundary subtype —
see the paired-landing rule below.

| Surface | Adopter count | Rule-of-three satisfied? | Adopters |
|---------|:-------------:|:------------------------:|----------|
| `pre_emission_check` | 1 | NO — 2 more needed | `DeclaredRequiredFieldsContract` (Phase 2B provisional adopter; batch-aware transforms remain fail-closed until a fifth site is ADR-approved) |
| `post_emission_check` | 4 | YES — closed at 4/3 | `PassThroughDeclarationContract`, `DeclaredOutputFieldsContract`, `SchemaConfigModeContract`, `CanDropRowsContract` |
| `batch_flush_check` | 4 | YES — closed at 4/3 | `PassThroughDeclarationContract`, `DeclaredOutputFieldsContract`, `SchemaConfigModeContract`, `CanDropRowsContract` |
| `boundary_check` | 2 | NO — 1 more needed | `SourceGuaranteedFieldsContract`, `SinkRequiredFieldsContract` |

**ADR-015 note (2026-04-20).** `CreatesTokensContract` is rejected as a
Phase 2B production adopter. `creates_tokens` remains a permission flag, not
a "must expand" runtime declaration. Remove it from future adopter-candidate
lists unless a later ADR changes the flag's semantics first.

**Paired-landing rule (boundary subtype).** The boundary subtype lands
with BOTH `SourceGuaranteedFieldsContract` AND `SinkRequiredFieldsContract`
in a single commit/PR. Staggered landings of one boundary adopter without
the other are rejected at review. Landing a single boundary adopter would
validate the subtype's shape against exactly one example, which is too
weak to satisfy the F4 sharpening (Security Architect).

**Per-surface governance.** A new contract adopting an existing surface
counts toward that surface's rule-of-three. A new contract introducing a
new surface (not one of the four named above) would require an ADR
amendment to name the surface, update `DispatchSite`, and restart the
rule-of-three gate at that surface.

**Pre-emission governance note (ST-4).** The pre-emission site's zero
baseline meant the rule-of-three gate provided weaker early coverage
there. Designs that plausibly fit `pre_emission_check` SHOULD prefer
that surface to build its evidential weight instead of defaulting to the
already-closed post-emission surfaces.

---

## §H2 landing scope (Amendment A3)

The H2 cluster (comment #415 on `elspeth-425047a599`) landed as a single
PR per H2 §Acceptance "ADR amendment landing manifest (H2-B)". W9 escape
hatch not exercised — no reviewer objection warranted a split. Landing
manifest:

- **H2** (`elspeth-425047a599`): nominal ABC + 4 bundle types +
  `@implements_dispatch_site` decorator (L0) + registry restructure.
- **N1** (`elspeth-10dc0b747f`): `EXPECTED_CONTRACT_SITES` per-site
  manifest + `MC3a`/`MC3b`/`MC3c` CI rules in
  `scripts/cicd/enforce_contract_manifest.py`.
- **N3** (`elspeth-60890a7388`): collect-then-raise dispatcher +
  `AggregateDeclarationContractViolation` sibling class + catch-site
  survey. Preserves triage SQL compatibility for the N=1 reference-
  equality case.
- **F2** (`elspeth-f52d7c5a47`): pre-emission dispatch site in
  `TransformExecutor` between input validation and `transform.process()`.
  Shared `_dispatch` helper, not parallel implementation.
- **F3** (`elspeth-5fc876138d`): `SinkTransactionalInvariantError`
  reclassification at `_validate_sink_input` — distinct from the future
  Phase 2C `SinkRequiredFieldsContract`. Two-layer architecture
  documented at the site.
- **F4** (`elspeth-b513c01cff`): per-surface rule-of-three criteria (see
  §Adoption State).
- **F5** (`elspeth-121b268aec`): E2E Landscape round-trip acceptance
  incorporated into each relevant ticket. Aggregate round-trip test
  added at
  `tests/integration/audit/test_declaration_contract_landscape_serialization_roundtrip.py`.
- **H1 amendment** (`elspeth-5dae105959`): NFR derivation extended to
  account for per-site dispatch (≤4 call sites × per-site N) — existing
  parametrised benchmark unchanged, applies per site.

**`override_input_fields` removal.** Panel F1 (comment #408) flagged the
2A `RuntimeCheckInputs.override_input_fields: frozenset | None` as the
B-antipattern in miniature. The H2 refactor replaces it with
`effective_input_fields: frozenset[str]` on each bundle type; callers
(TransformExecutor, `_cross_check_flush_output`) derive the set once
via `derive_effective_input_fields` and pass it in. Contracts no longer
re-derive — the fabrication surface is closed at the caller boundary.

**Triage SQL compatibility.** A row where a single pass-through contract
fires still produces `exception_type = 'PassThroughContractViolation'` in
the audit table (N=1 reference-equality fast path). A row where multiple
contracts fire produces `exception_type =
'AggregateDeclarationContractViolation'` with `is_aggregate = true` and a
`violations` list. Queries filtering on a specific contract's exception
type must be updated to also match inside the aggregate's `violations`
list when they want all occurrences.
