# ADR-014: Schema config mode contract

**Date:** 2026-04-20
**Status:** Accepted
**Deciders:** Codex synthesis from the Phase 2B panel-reviewed plan
**Tags:** declaration-contract, transform, schema, post-emission, batch-flush, tier-1, audit-integrity
**Supersedes:** None
**Depends on:** [ADR-010](010-declaration-trust-framework.md)

## Context

Transforms that expose `_output_schema_config` are making a runtime declaration
about emitted schema semantics, not just a DAG-planning hint. The declaration is
expressed in real repo types:

- config surface: `SchemaConfig.mode` with values `fixed`, `flexible`, `observed`
- runtime surface: `SchemaContract.mode` with values `FIXED`, `FLEXIBLE`, `OBSERVED`
- runtime lock state: `SchemaContract.locked`

Before ADR-014, ELSPETH trusted `_output_schema_config` for downstream
propagation while several runtime output-contract builders preserved input mode
or hard-coded `OBSERVED`. That drift is audit-significant:

- a transform can declare `fixed` output semantics,
- emit a row whose contract advertises `OBSERVED`,
- and still leave the audit trail looking internally consistent unless an
  auditor manually compares config intent to emitted runtime contract shape.

That is a fabrication surface. The transform's declared schema semantics and
the contract attached to emitted rows must agree.

## Decision

Introduce `SchemaConfigModeContract`, a `DeclarationContract` adopter that
registers on:

- `post_emission_check`
- `batch_flush_check`

The contract applies exactly when `plugin._output_schema_config is not None`.
`applies_to` is intentionally O(1): a single direct attribute read. A plugin
missing `_output_schema_config` crashes rather than silently skipping runtime
validation.

### Runtime invariant

For every emitted row, compare `plugin._output_schema_config` to the emitted
`PipelineRow.contract` using the existing schema-factory mapping logic:

```python
declared_mode = output_schema_config.mode          # fixed / flexible / observed
expected_mode = map_schema_mode(declared_mode)     # FIXED / FLEXIBLE / OBSERVED
declared_locked = True
observed_mode = emitted.contract.mode.lower()
observed_locked = emitted.contract.locked
```

Raise when:

- `observed_mode != declared_mode`, or
- `observed_locked != True`, or
- declared mode is `fixed` and undeclared output fields appear.

For `fixed`, undeclared extras are computed from the union of runtime contract
fields and runtime payload keys, then compared against the set of fields
declared by `SchemaConfig.fields`, `guaranteed_fields`, and `audit_fields`.
Using the union rather than only `contract ∩ payload` fails closed on one-sided
drift: a field that appears in only one runtime witness is still a runtime
schema-semantic leak.

### Mode-specific semantics

- `fixed`: emitted contract must report `FIXED`, must be locked, and must not
  expose undeclared extras.
- `flexible`: emitted contract must report `FLEXIBLE` and must be locked;
  extras are allowed.
- `observed`: emitted contract must report `OBSERVED` and must be locked;
  field shape is discovered at runtime, but the emitted row still must carry a
  locked runtime contract once it leaves the transform.

### Why `locked` belongs in this contract

ADR-014 keeps `locked` inside the contract payload rather than splitting it into
another adopter. At the dispatcher's observation point, an emitted row contract
is no longer a mutable builder artifact; it is the runtime contract auditors
query. An unlocked contract at that point means the transform leaked an
intermediate construction state into the audit surface.

### Violation

`SchemaConfigModeViolation` subclasses `DeclarationContractViolation` and is
registered Tier 1 via `@tier_1_error`.

Payload schema:

```python
class SchemaConfigModePayload(TypedDict):
    declared_mode: Required[str]
    observed_mode: Required[str]
    declared_locked: Required[bool]
    observed_locked: Required[bool]
    undeclared_extra_fields: NotRequired[list[str]]
```

Forbidden payload keys for this contract include:

- `raw_schema_config`
- `config_dict`
- `options`
- `sample_row`

The payload is structural only: mode strings, bools, and optional field-name
lists.

## Consequences

### Positive

- `_output_schema_config` now has a runtime backstop at both transform-success
  surfaces that matter in Phase 2B.
- Runtime output-contract builders are normalized to the transform's declared
  mode before rows leave the transform, shrinking the drift surface rather than
  merely documenting it.
- `post_emission_check` and `batch_flush_check` each advance another adopter
  toward ADR-010's per-surface rule-of-three closure.

### Negative

- Test doubles that reach the live dispatcher must expose `_output_schema_config`
  honestly, because `applies_to` reads it directly.
- Mode drift that previously hid behind permissive `OBSERVED` contracts now
  fails loudly as Tier 1.

### Neutral

- ADR-014 does not introduce a new config enum. The config surface remains the
  existing `fixed` / `flexible` / `observed` triad; `locked` remains runtime
  state, not user config.

## Tier classification

Tier 1. A transform declaring one schema mode while emitting another corrupts
the contract semantics the auditor queries by. That is an audit-surface lie,
not a recoverable row-level data issue, and `on_error` routing must not absorb
it.

## Reversibility

Reversal is coordinated rollback:

1. Remove `SchemaConfigModeContract` registration and its
   `EXPECTED_CONTRACT_SITES["schema_config_mode"]` manifest entry.
2. Remove the bootstrap import from
   `engine/executors/declaration_contract_bootstrap.py`.
3. Remove `SchemaConfigModeViolation`.
4. Remove the transform-side output-contract alignment helper if the project
   elects to stop treating `_output_schema_config` as a runtime declaration.

Git history preserves the pre-ADR-014 builder behavior if an emergency restore
is needed.

## Scrubber-audit

No scrubber extension is required. Payload keys are structural only:

- `declared_mode`
- `observed_mode`
- `declared_locked`
- `observed_locked`
- `undeclared_extra_fields`

No row snapshots, arbitrary config dicts, or options payloads are recorded.

## References

- [ADR-010](010-declaration-trust-framework.md)
- `src/elspeth/engine/executors/schema_config_mode.py`
- `src/elspeth/contracts/errors.py`
- `src/elspeth/plugins/infrastructure/base.py`
- `docs/plans/2026-04-20-phase-2b-declaration-trust.md`
