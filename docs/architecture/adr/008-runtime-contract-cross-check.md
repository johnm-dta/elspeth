# ADR-008: Runtime contract cross-check in TransformExecutor

**Date:** 2026-04-19
**Status:** Accepted
**Deciders:** Architecture Critic (SME agent), Systems Thinker (SME agent), Python Code Reviewer (SME agent), Quality Engineer (SME agent), Claude (synthesis/lead)
**Tags:** engine, executor, schema-contract, tier-1, audit-integrity, validation

> **Amended by [ADR-009](009-pass-through-pathway-fusion.md) on 2026-04-19.**
> ADR-009 extends §Decision scope from "after `transform.process()` returns"
> to also cover the batch aggregation flush path in
> `processor._process_batch_aggregation_node` and `handle_timeout_flush`
> (Clause 2). The "cross-check applies per-row in executor" claim in
> ADR-007 table line 53 is correct after ADR-009 lands. The shared
> `verify_pass_through` function in `engine/executors/pass_through.py`
> replaces the method-scoped implementation in `TransformExecutor`; both
> call sites now import the same primitive.

> **Amended by ADR-010 (Declaration-trust framework, 2026-04-19).**
> Normative in this ADR: §Explicit scope boundary (each new declaration still requires its own ADR).
> Superseded: the direct-call executor integration pattern — replaced by ADR-010 §Decision 3's `run_runtime_checks` dispatcher. Cross-check scope "per-row in executor" remains correct; the mechanism is now registry-driven.

## Context

ADR-007 establishes opt-in propagation declared via `BaseTransform.passes_through_input`. Static DAG analysis now trusts that declaration — the validator walks through annotated transforms and propagates predecessor guarantees downstream, mirroring runtime behaviour.

But static analysis alone is not enough. The declaration is a plugin-author claim; a mis-annotation (class marked `True` whose `process()` actually drops fields in some row path) would silently produce wrong audit data:

- Static validator accepts the pipeline.
- Runtime runs the transform, drops a field, emits a row.
- Downstream nodes record the row without the dropped field.
- The audit trail now contains evidence that looks processed but isn't — an auditor asking "why does row 42 lack field X?" gets a confident wrong answer.

This is the bug-class the original v2 plan was reaching for but left implicit: "the DAG static analysis trusts plugin declarations". ADR-007 trusts the declaration for pass-through specifically; ADR-008 adds the runtime backstop that catches mis-annotations before they corrupt audit integrity.

## Decision

Add a per-row runtime cross-check to `TransformExecutor.execute_transform`:

1. After `transform.process()` returns a successful `TransformResult`, if `transform.passes_through_input` is True, compute `input_fields = frozenset(input_row.contract.fields)` for every emitted row.
2. Compute `runtime_observed = frozenset(emitted_row.contract.fields) & frozenset(emitted_row.keys())` — the **intersection** of the emitted row's contract-set and its payload-set. `PipelineRow.__init__` accepts any `dict` and any `SchemaContract` as independent references and does not enforce `data.keys() ⊆ contract.fields`, so a field is "kept" at runtime iff the row simultaneously declares it in its contract AND carries it in its payload. Reading either side alone creates a one-sided blind spot: a buggy plugin can shrink the contract while keeping the payload (caught by the contract side), or shrink the payload while reusing the input contract (caught by the payload side). Using the intersection catches both vectors. The payload-side cost is `frozenset(emitted_row.keys())`, which reads the frozen `MappingProxyType` directly — no `deep_thaw` — so the NFR budget (median ≤ 25 µs / P99 ≤ 50 µs on a 200-field row) remains comfortable.
3. If `divergence_set = input_fields - runtime_observed` is non-empty, raise `PassThroughContractViolation` with the full set of audit fields (transform, node_id, run_id, row_id, token_id, static_contract, runtime_observed, divergence_set, message).
4. Before raising, increment `pass_through_cross_check_violations_total{transform=...}` — a telemetry counter acquired at `TransformExecutor.__init__`. This is the operational signal SRE sees even when Landscape recording itself fails.

### TIER_1 registration is load-bearing

`PassThroughContractViolation` is registered in `TIER_1_ERRORS` alongside `AuditIntegrityError`, `FrameworkBugError`, and `OrchestrationInvariantError`. This registration is not cosmetic — it is the mechanism that prevents `on_error` routing from silently absorbing audit-integrity violations.

Without the registration: a transform with `on_error="quarantine_sink"` would catch the violation via the executor's `except Exception:` block, route the row to the error sink, and continue. The Landscape would record a row-level FAILED state, the quarantine sink would accept the evidence, and the mis-annotation would survive to corrupt the next row. With the registration: the executor's `except TIER_1_ERRORS: raise` fires first, the `NodeStateGuard.__exit__` auto-completes the state as FAILED with the full structured context, and the exception propagates out of `execute_transform` so the orchestrator sees the crash.

### Audit-recording path

`NodeStateGuard.__exit__` (L2 engine) now populates the new `ExecutionError.context` field (L0 contract) from `PluginContractViolation.to_audit_dict()` when the raised exception is a `PluginContractViolation` or subclass. The `isinstance` check is a Tier-2/Tier-1 boundary discriminator — not defensive programming — and it benefits every `PluginContractViolation` subclass that defines `to_audit_dict()`, not just the pass-through case. The full 9-key structured payload (transform, transform_node_id, run_id, row_id, token_id, static_contract, runtime_observed, divergence_set, exception_type) reaches the Landscape and is queryable via `json_extract(error_data, '$.context.<key>')`.

### Tier placement for cross-check data

The cross-check crosses trust tiers. Each tier is explicit per CLAUDE.md's Three-Tier Trust Model:

| Input / Output | Tier | Handling |
|---|---|---|
| `transform.passes_through_input` (class attribute) | System code (outside tier model) | Read without defensive guards; missing/wrong type is a framework bug → `FrameworkBugError` |
| `input_row.contract.fields` | Tier 2 (elevated trust) | Expect types correct. `contract is None` → `FrameworkBugError` (framework invariant violation) |
| `emitted_row.contract.fields` | Tier 2 (elevated trust) | Same rule. `contract is None` → `FrameworkBugError` |
| `emitted_row.keys()` (payload-set) | Tier 2 (elevated trust) | Keys come from the frozen `MappingProxyType` wrapper that `PipelineRow.__init__` creates via `deep_freeze`. No thaw, no coercion — the key-set is the authoritative runtime payload shape. |
| Computed `divergence_set: frozenset[str]` | Tier 2 at computation, Tier 1 at event boundary | Deep-frozen into sorted lists in `to_audit_dict()` for canonical JSON serialization |
| Landscape event payload (9 fields) | Tier 1 (full trust) | Canonicalizable; crash on any anomaly during serialization |

`None` contracts at the Tier-2 boundary raise `FrameworkBugError` rather than silently skipping. Silent skip would mean a violated pipeline writes no violation event — evidence destruction.

### Explicit scope boundary (deliberate deferral)

This ADR fixes runtime VAL **only** for `passes_through_input` declarations. The wider class — "DAG static analysis trusts ALL plugin declarations" (including `creates_tokens`, `declared_output_fields`, `declared_required_fields`, schema-config modes) — is **deliberately deferred**. Each declaration would benefit from a similar runtime backstop, but extending the cross-check pattern across all declarations at once would couple this ADR to a much wider scope and delay the `elspeth-87f6d5dea5` fix.

Future ADRs may extend the pattern. This ADR establishes the architectural template (annotation + static trust + runtime VAL + TIER_1 escalation + invariant framework) which is reusable:

1. Declarative per-plugin attribute (auditable, greppable).
2. Static analysis trusts the declaration.
3. Runtime verifies the declaration per-row and raises a TIER_1-registered exception.
4. Invariant test framework exercises the declaration with diverse probe rows.

## Consequences

### Positive Consequences

- Audit trail recovers structured violation evidence. `ExecutionError.context` carries the full `to_audit_dict()` payload into `error_data`, queryable via `json_extract` for triage.
- Production mis-annotations crash loudly instead of corrupting Tier 1 data. The failure mode shifts from silent evidence tampering to a conspicuous pipeline crash at the first violated row.
- Telemetry counter `pass_through_cross_check_violations_total{transform=...}` provides operational visibility for SRE. Cardinality bounded by the annotated-transform set (short, known at startup). Counter fires before the raise so the metric survives downstream serialization failure.
- Pattern is reusable for future declaration-VAL gates. When the next declaration (e.g., `creates_tokens`) needs runtime VAL, the template is already proven.

### Negative Consequences

- Per-row overhead on the executor hot path. Bounded by NFR gate: median ≤ 25 µs, P99 ≤ 50 µs on a 200-field input row (measured via `pytest-benchmark` in `tests/performance/benchmarks/test_cross_check_overhead.py`). Only fires for annotated transforms — non-annotated transforms pay zero.
- `TIER_1_ERRORS` membership change affects ~40+ `isinstance()` call sites. Verified non-load-bearing by the §Verification grep step: no caller hardcodes the tuple length; all use `isinstance(exc, TIER_1_ERRORS)` which accepts the expanded tuple transparently.
- `ExecutionError` extension (new `context` field) ripples through any custom serializer of audit error data. Mitigated by the field being optional with `None` default — pre-existing serializers continue to emit the same keys they did before.

### Neutral Consequences

- `NodeStateGuard.__exit__` gains an `isinstance(exc_val, PluginContractViolation)` discriminator. This is a Tier-2/Tier-1 boundary type-check, not defensive programming (CLAUDE.md permits `isinstance` at trust boundaries).
- Landscape-unavailable failure mode: when DB recording itself fails, `__exit__` raises `AuditIntegrityError` chaining the original violation. Triage SQL filtering on `error_exception_type = 'PassThroughContractViolation'` returns zero rows in this scenario — the telemetry counter (incremented before the raise) is the reliable secondary signal. Documented in §Observability of the implementation plan.
- Cross-check is skipped entirely when `transform.passes_through_input` is False. Non-annotated transforms pay exactly the cost of one attribute read per row — negligible.

## Alternatives Considered

### Alternative 1: Development-only assertion gated on a debug flag

**Description:** Gate the cross-check behind `ELSPETH_DEBUG` or similar; production runs skip it entirely.

**Rejected because:** Production mis-annotations would silently reach Tier 1 audit data. Debug flags are by definition not enabled in production, so the backstop that catches the worst failure mode would be absent in the environment that matters.

### Alternative 2: Pure test-layer verification via invariant framework only

**Description:** The Hypothesis-driven invariant framework exercises every annotated transform with diverse probe rows; runtime cross-check is unnecessary.

**Rejected because:** Tests cover configurations the suite exercises; they miss configurations that only appear in operator-composed pipelines. The runtime cross-check is the production backstop for the long tail of pipelines the test suite never sees.

### Alternative 3: Do not register in TIER_1_ERRORS — let `on_error` route the violation

**Description:** `PassThroughContractViolation` remains a subclass of `PluginContractViolation` (which is not a TIER_1 error); `on_error` routing absorbs it like any other plugin error.

**Rejected because:** A pass-through-annotation lie is a framework contract violation, not a row-level data error. The static validator was told the transform emits a superset of input; runtime observed otherwise. Routing the row to `on_error` would record a misleading audit entry (implying this was a row-data problem) and continue the pipeline with a mis-annotated transform — future rows would keep producing silently wrong results. TIER_1 registration is the only mechanism that enforces the crash.

### Alternative 4: Cross-check across ALL declarations (`creates_tokens`, schema modes, etc.)

**Description:** Extend the cross-check to every declaration the DAG validator trusts.

**Rejected because:** Scope creep. ADR-008 establishes the template; future ADRs can extend it. Bundling the wider scope here would delay the `elspeth-87f6d5dea5` fix and couple the ADR to decisions not yet made about the other declarations.

## References

- Plan: `/home/john/.claude/plans/elspeth-87f6d5dea5-snazzy-swing.md`
- Companion ADR: `ADR-007: Pass-through contract propagation — declaration, semantics, and composer parity`
- Related bug report: `elspeth-87f6d5dea5` (composer/runtime schema-contract divergence)
- CLAUDE.md §Three-Tier Trust Model (tier boundary rules)
- CLAUDE.md §Plugin Ownership (plugin bugs must crash)
