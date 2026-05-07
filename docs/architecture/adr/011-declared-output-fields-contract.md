# ADR-011: Declared output fields contract

**Date:** 2026-04-20
**Status:** Accepted
**Deciders:** Codex synthesis from the Phase 2B panel-reviewed plan
**Tags:** declaration-contract, transform, post-emission, batch-flush, tier-1, audit-integrity
**Supersedes:** None
**Depends on:** [ADR-010](010-declaration-trust-framework.md)

## Context

`BaseTransform.declared_output_fields` is trusted by DAG/schema propagation to
mean "every emitted row exposes these fields." That trust currently has only a
static backstop: the DAG builder checks that the declaration is wired into
schema propagation, but runtime does not prove the emitted rows actually carry
the declared fields.

That gap is audit-significant. A transform that advertises
`declared_output_fields={"new_a", "new_b"}` but emits rows carrying only
`new_a` silently corrupts downstream lineage:

- downstream nodes believe `new_b` is guaranteed,
- required-field reasoning is performed on false premises,
- the audit trail records a materially incorrect output-shape claim.

ADR-010 established the generalized declaration-trust framework. ADR-011 lands
the first non-pass-through transform adopter on that framework.

## Decision

Introduce `DeclaredOutputFieldsContract`, a `DeclarationContract` adopter that
registers on:

- `post_emission_check`
- `batch_flush_check`

The contract applies to transforms where `bool(plugin.declared_output_fields)` is
true. `applies_to` uses direct attribute access. A plugin missing
`declared_output_fields` is a framework bug and must crash rather than silently
skip runtime validation.

### Runtime invariant

For every emitted row:

```python
runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)
runtime_payload_fields = frozenset(emitted.keys())
runtime_observed = runtime_contract_fields & runtime_payload_fields
missing = frozenset(plugin.declared_output_fields) - runtime_observed
```

Raise iff `missing` is non-empty.

The contract validates each emitted row independently. This matches the meaning
of `declared_output_fields` in schema propagation: the declaration is a
per-emitted-row guarantee, not a union-across-batch promise.

Empty emission is a no-op in ADR-011. Zero-emission governance remains owned by
the later `can_drop_rows` contract; this ADR does not re-litigate ADR-009
Clause 3.

### Violation

`DeclaredOutputFieldsViolation` subclasses `DeclarationContractViolation` and is
registered Tier 1 via `@tier_1_error`.

Payload schema:

```python
class DeclaredOutputFieldsPayload(TypedDict):
    declared: Required[list[str]]
    runtime_observed: Required[list[str]]
    missing: Required[list[str]]
```

All three fields are sorted lists for canonical audit serialization.

### Tier classification

Tier 1. A lie in `declared_output_fields` corrupts downstream lineage and
required-field reasoning. `on_error` routing must not absorb that evidence.

## Consequences

### Positive

- `declared_output_fields` now has a runtime backstop at both transform success
  surfaces that matter in 2B: single-token post-emission and batch flush.
- Aggregate dispatch with `PassThroughDeclarationContract` is now meaningful:
  a transform can be caught simultaneously for dropping inherited fields and
  for failing to emit newly declared fields.
- The rule-of-three count for both `post_emission_check` and
  `batch_flush_check` advances from 1/3 to 2/3.

### Negative

- Every live-registry test stub that reaches the dispatcher must now expose
  `declared_output_fields`, because `applies_to` intentionally reads the
  attribute directly.
- Empty-emission governance is still split across ADR-009 Clause 3 and the
  future `can_drop_rows` ADR until that contract lands.

### Neutral

- The contract does not require a `BaseTransform` API change. It reuses the
  existing `declared_output_fields` runtime attribute.

## Reversibility

Reversal is a coordinated rollback, not a scalar delete:

1. Flip `DeclaredOutputFieldsContract.applies_to` to always return `False` if an
   emergency soft-disable is needed.
2. Remove the `EXPECTED_CONTRACT_SITES["declared_output_fields"]` manifest entry.
3. Remove the bootstrap import from
   `engine/executors/declaration_contract_bootstrap.py`.
4. Remove `DeclaredOutputFieldsViolation` and the adopter module.

This ADR does not introduce a new `BaseTransform` attribute, so there is no
protocol-level rollback work outside the contract itself.

Triage-SQL signatures introduced by this ADR:

- `exception_type = 'DeclaredOutputFieldsViolation'`
- aggregate child filter:
  `json_extract(error_data, '$.context.violations[*].exception_type')`

## Scrubber-audit

Payload keys are structural only:

- `declared`
- `runtime_observed`
- `missing`

Each carries field-name lists, not row samples, config dicts, or free-form
payloads. No scrubber extension is required in this ADR. Forbidden payload keys
for this contract include `raw_schema_config`, `config_dict`, `options`, and
`sample_row`.

## References

- [ADR-010](010-declaration-trust-framework.md)
- `src/elspeth/engine/executors/declared_output_fields.py`
- `src/elspeth/contracts/errors.py`
- `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
