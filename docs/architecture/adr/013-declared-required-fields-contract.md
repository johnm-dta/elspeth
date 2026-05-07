# ADR-013: Declared required input fields contract

**Date:** 2026-04-20
**Status:** Accepted
**Deciders:** Codex synthesis from the Phase 2B panel-reviewed plan
**Tags:** declaration-contract, transform, pre-emission, tier-1, audit-integrity
**Supersedes:** None
**Depends on:** [ADR-010](010-declaration-trust-framework.md)

## Context

Transforms can already declare required input fields in configuration via
`TransformDataConfig.required_input_fields`, and the DAG validator uses that
surface to catch some upstream-missing-field mistakes before execution.
Runtime, however, there is no normalized transform attribute carrying the same
declaration, and there is no declaration-trust adopter proving that the input
row presented to `process()` actually satisfies the declaration.

That gap is attribution-significant. If a transform declares it requires
`customer_id` and `account_id` but is invoked on a row that only exposes
`account_id`, any later plugin crash or output record is attributed on false
premises. The framework needs to distinguish:

- "the transform body crashed for its own reasons" from
- "the framework invoked the transform on input that violated the transform's
  declared preconditions."

ADR-010 created the `pre_emission_check` surface specifically for this kind of
precondition contract. ADR-013 is the first adopter on that surface.

## Decision

Introduce `DeclaredRequiredFieldsContract`, a `DeclarationContract` adopter that
registers on:

- `pre_emission_check`

The contract applies to transforms where `bool(plugin.declared_input_fields)` is
true. `declared_input_fields` is a new runtime attribute on `BaseTransform` and
`TransformProtocol`, normalized from `TransformDataConfig.required_input_fields`
at construction time.

### Runtime attribute naming

The transform-side runtime attribute is named `declared_input_fields`, not
`declared_required_fields`.

Sinks already use `declared_required_fields` for a different contract surface:
fields required at the sink write boundary. Reusing the same name for transform
preconditions would make grep-time reasoning ambiguous and increase the chance
of future protocol drift.

### Runtime invariant

For a single-row pre-emission call:

```python
missing = plugin.declared_input_fields - inputs.effective_input_fields
```

Raise iff `missing` is non-empty.

`inputs.effective_input_fields` is the authoritative comparison source. It is
caller-derived by the executor from the input row's runtime contract per
ADR-010's F1 resolution. The plugin is not allowed to be its own witness.

### Batch scope

ADR-013 deliberately does **not** cover batch-aware transforms.

There is no `batch_pre_emission_check` dispatch site in ADR-010 today. Adding
one would be an ADR-010 amendment decision because it would extend
`DispatchSite`, widen the manifest/MC3 rule surface, and start a new
rule-of-three count from zero for the new site.

Phase 2B therefore chooses the lighter, honest path:

- `DeclaredRequiredFieldsContract` is `PRE_EMISSION` only.
- Batch-aware transforms that declare `declared_input_fields` fail closed.
- Failure is mechanical in two places:
  - construction-time normalization on `BaseTransform`
  - `DeclaredRequiredFieldsContract.applies_to()`

Silent skip is explicitly rejected because it would make "not checked" and
"checked and passed" indistinguishable in the audit trail.

### Violation

`DeclaredRequiredInputFieldsViolation` subclasses
`DeclarationContractViolation` and is registered Tier 1 via `@tier_1_error`.

Payload schema:

```python
class DeclaredRequiredInputFieldsPayload(TypedDict):
    declared: Required[list[str]]
    effective_input_fields: Required[list[str]]
    missing: Required[list[str]]
```

All three fields are sorted lists for canonical audit serialization.

### Tier classification

Tier 1. If runtime input does not satisfy the transform's declared
preconditions, the framework can no longer honestly attribute downstream
behaviour to the transform body alone. This is an audit-integrity problem, not
a row-level data-quality error, and `on_error` routing must not absorb it.

## Consequences

### Positive

- Transform required-input declarations now have a runtime backstop before
  `process()` executes.
- The runtime declaration surface is normalized onto a single immutable
  attribute (`declared_input_fields`) shared across transforms.
- Batch-aware misconfiguration becomes loud and mechanical instead of silently
  implying coverage the framework cannot actually provide.

### Negative

- Every transform construction path that validates config must now call the
  shared normalization helper so `declared_input_fields` is populated at
  runtime.
- The `pre_emission_check` surface remains provisional after this ADR:
  one adopter is not enough to satisfy ADR-010's rule-of-three guidance.

### Neutral

- Aggregate round-tripping with `PassThroughDeclarationContract` is
  structurally inapplicable here because pass-through only claims
  `post_emission_check` and `batch_flush_check`, while ADR-013 is
  `pre_emission_check` only.
- Shared aggregate mechanics remain covered by the framework's dispatcher tests
  and by multi-violation tests on same-site adopters.

## Reversibility

Reversal is coordinated, not scalar:

1. Remove `declared_input_fields` from `BaseTransform` and
   `TransformProtocol`.
2. Remove the `EXPECTED_CONTRACT_SITES["declared_required_fields"]` manifest
   entry.
3. Remove the bootstrap import from
   `engine/executors/declaration_contract_bootstrap.py`.
4. Remove `DeclaredRequiredFieldsContract` and
   `DeclaredRequiredInputFieldsViolation`.
5. Remove the construction-time normalization calls from transform
   instantiation paths.

Triage-SQL signatures introduced by this ADR:

- `exception_type = 'DeclaredRequiredInputFieldsViolation'`
- aggregate child filter:
  `json_extract(error_data, '$.context.violations[*].exception_type')`

## Scrubber-audit

Payload keys are structural only:

- `declared`
- `effective_input_fields`
- `missing`

Each carries field-name lists, not row samples, config dicts, or free-form
payloads. No scrubber extension is required in this ADR. Forbidden payload keys
for this contract include `raw_schema_config`, `config_dict`, `options`, and
`sample_row`.

## References

- [ADR-010](010-declaration-trust-framework.md)
- `src/elspeth/engine/executors/declared_required_fields.py`
- `src/elspeth/contracts/errors.py`
- `src/elspeth/plugins/infrastructure/base.py`
- `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
