# ADR-012: `can_drop_rows` governance contract

**Date:** 2026-04-20
**Status:** Accepted
**Deciders:** Codex synthesis from the Phase 2B panel-reviewed plan
**Tags:** declaration-contract, transform, pass-through, empty-emission, governance, tier-2, audit-integrity
**Supersedes:** None
**Depends on:** [ADR-010](010-declaration-trust-framework.md)

## Context

ADR-009 Clause 3 left pass-through runtime verification with an explicit
empty-emission carve-out:

```python
if not emitted_rows:
    return
```

That kept honest filter-style transforms from tripping the pass-through
cross-check, but it also meant the framework had no first-class declaration for
zero-emission success. Under audit-complete posture, "no emitted rows" must be
distinguished mechanically between:

- a transform that is allowed to drop rows, and
- a transform that silently behaved like a filter despite claiming
  `passes_through_input=True`.

Without an explicit declaration and a queryable terminal state, legitimate zero
emission is indistinguishable from a missed audit record.

## Decision

Introduce `CanDropRowsContract`, a `DeclarationContract` adopter that registers
on:

- `post_emission_check`
- `batch_flush_check`

Add `can_drop_rows: bool = False` to `BaseTransform` and `TransformProtocol`.

The contract applies exactly when:

```python
plugin.passes_through_input and not plugin.can_drop_rows
```

`applies_to` uses direct attribute access. A plugin missing either attribute is
a framework bug and must crash loudly rather than silently skipping runtime
governance.

### Runtime invariant

When the contract applies, `len(outputs.emitted_rows)` must be non-zero.

Raise `UnexpectedEmptyEmissionViolation` when:

```python
len(outputs.emitted_rows) == 0
```

Payload schema:

```python
class UnexpectedEmptyEmissionPayload(TypedDict):
    passes_through_input: Required[bool]
    can_drop_rows: Required[bool]
    emitted_count: Required[int]
```

Violation class:

```python
class UnexpectedEmptyEmissionViolation(DeclarationContractViolation):
    payload_schema: ClassVar[type] = UnexpectedEmptyEmissionPayload
```

### Clause-3 retirement

ADR-012 retires the ADR-009 Clause 3 carve-out mechanically, not just by
documentation.

`verify_pass_through(...)` no longer treats empty emission as an unconditional
no-op. Its rule is now:

- if `emitted_rows` is empty and `can_drop_rows=True`: no-op
- if `emitted_rows` is empty and `can_drop_rows=False`: raise
  `PassThroughContractViolation`

This keeps empty-emission governance on one declaration surface while still
allowing aggregate evidence bundles when both contracts apply.

### Terminal state

A legitimate zero-emission success is recorded as:

- `RowOutcome.DROPPED_BY_FILTER`

This state is terminal, queryable in Landscape, and distinct from `FAILED`.
Batch-flush zero emission records the same outcome for every buffered token.

### Batch-flush semantics

`CanDropRowsContract` claims `batch_flush_check`, so zero-emission batch flushes
must still enter the dispatcher. Passthrough flushes with zero emitted rows
therefore dispatch a batch-level empty-emission check rather than skipping the
dispatcher because 1:1 pairing is unavailable.

## Tier classification

Tier 2, not Tier 1.

A `can_drop_rows=False` transform that emits zero rows is a plugin declaration
bug and must fail loudly, but it does not fabricate or corrupt Tier-1 framework
state. The row still receives an auditable terminal outcome, and the evidence is
preserved in the legal record.

## Consequences

### Positive

- Empty-emission governance is now a first-class runtime declaration instead of
  a hidden carve-out in pass-through logic.
- Honest filter-style transforms can declare their semantics explicitly via
  `can_drop_rows=True`.
- Legitimate zero emission is queryable through `DROPPED_BY_FILTER` rather than
  being inferred from silence.
- Aggregate evidence with `PassThroughDeclarationContract` is now possible when
  a mis-declared pass-through transform both drops all rows and drops inherited
  input fields.

### Negative

- Every dispatcher-reaching test double with `passes_through_input=True` must
  now also expose `can_drop_rows`.
- Batch-flush passthrough empty emission needs an explicit batch-level dispatch
  path because there is no 1:1 emitted-row pairing to reuse.

### Neutral

- `can_drop_rows` does not change the meaning of `creates_tokens`; ADR-015 still
  treats that flag as permission, not an output-cardinality promise.
- `DeclaredOutputFieldsContract` remains orthogonal. Zero emitted rows are not
  modeled as "missing declared fields."

## Reversibility

Reversal is coordinated rollback:

1. Remove `can_drop_rows` from `BaseTransform` and `TransformProtocol`.
2. Remove `EXPECTED_CONTRACT_SITES["can_drop_rows"]`.
3. Remove the bootstrap import from
   `engine/executors/declaration_contract_bootstrap.py`.
4. Remove `CanDropRowsContract` and `UnexpectedEmptyEmissionViolation`.
5. Reintroduce the old unconditional empty-emission short-circuit in
   `pass_through.py` if the project intentionally returns to ADR-009 Clause 3.

Git history preserves the pre-ADR-012 behaviour if emergency rollback is ever
required.

## Scrubber-audit

No scrubber extension is required. Payload keys are structural only:

- `passes_through_input`
- `can_drop_rows`
- `emitted_count`

All values are bools or ints. Forbidden payload keys for this contract include
`raw_schema_config`, `config_dict`, `options`, and `sample_row`.

## References

- [ADR-009](009-pass-through-pathway-fusion.md)
- [ADR-010](010-declaration-trust-framework.md)
- `src/elspeth/engine/executors/can_drop_rows.py`
- `src/elspeth/engine/executors/pass_through.py`
- `src/elspeth/contracts/errors.py`
- `src/elspeth/engine/processor.py`
- `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
