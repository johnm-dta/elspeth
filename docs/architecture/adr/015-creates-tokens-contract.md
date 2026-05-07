# ADR-015: `creates_tokens` remains a permission flag, not a production declaration contract

**Date:** 2026-04-20
**Status:** Accepted
**Deciders:** Codex synthesis from the Phase 2B panel-reviewed plan
**Tags:** declaration-contract, transform, deaggregation, semantics, governance
**Supersedes:** None
**Depends on:** [ADR-010](010-declaration-trust-framework.md)

## Context

`creates_tokens` currently means:

- when `False`, multi-row output is treated as pass-through batching,
- when `True`, the processor is permitted to create child tokens when the
  transform returns multi-row output.

That is a permission model, not a "must expand" promise. The current code and
protocol docs already treat `creates_tokens=True` as compatible with a
single-row success result.

The old framework proof in
`tests/invariants/test_framework_accepts_second_contract.py` sketched a
`CreatesTokensContract` that interpreted the flag as "must emit multiple rows."
That proof served its purpose in 2A: it showed the framework could host a
second contract shape. It is not semantically honest enough to ship as a
production declaration contract.

## Decision

Choose **Path 1** from the Phase 2B plan:

`creates_tokens=True` means **multi-row expansion is permitted**, not required.

Therefore:

- there is **no production `CreatesTokensContract`** in Phase 2B,
- the proof-only `CreatesTokensContract` is retired,
- the framework shape-diversity proof is re-pointed at the real production
  adopter `DeclaredOutputFieldsContract`.

## Rationale

### Why the production contract is rejected

The dispatcher sees emitted `PipelineRow` objects and a token identity anchor.
It does not see a trustworthy semantic signal that distinguishes:

- "single-row result is correct; expansion was merely permitted" from
- "single-row result is incorrect; expansion was required."

Encoding "must expand" into a runtime contract would therefore over-promise at
the audit boundary. A false positive here would be worse than no contract: the
audit trail would claim the framework proved something the current semantics do
not actually guarantee.

### Alternative runtime mechanism

No replacement runtime declaration contract is needed today.

If the product later needs a real runtime invariant here, it should be framed as
a different contract with explicit semantics, for example:

- a transform-specific cardinality contract tied to a concrete config field, or
- a dedicated deaggregation contract whose declaration explicitly means "must
  emit N>1 rows under this success path."

That would be a new ADR, not a reinterpretation of `creates_tokens`.

## Consequences

### Positive

- Code, docs, and runtime semantics now agree on `creates_tokens`.
- Phase 2B avoids building a production contract on an impossible semantic
  premise.
- The framework shape-diversity proof remains, but now points at a real
  production adopter.

### Negative

- `creates_tokens` still has no runtime declaration-VAL backstop. This is an
  explicit non-goal of the current semantics, not an accidental gap.

### Neutral

- `BaseTransform.creates_tokens` and `TransformProtocol.creates_tokens` remain
  unchanged.
- No dispatcher surface or manifest entry is added or removed by this ADR.

## Reversibility

Reversal would mean choosing Path 2 later:

1. Tighten the documented meaning of `creates_tokens` from permission to
   obligation.
2. Update processor/protocol docs and tests in the same PR.
3. Introduce a new production contract only after the semantics change is
   complete.

That is not a scalar rollback; it is a deliberate semantic change requiring a
new ADR amendment or successor ADR.

## References

- [ADR-010](010-declaration-trust-framework.md)
- [ADR-011](011-declared-output-fields-contract.md)
- `src/elspeth/plugins/infrastructure/base.py`
- `src/elspeth/contracts/plugin_protocols.py`
- `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
