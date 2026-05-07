# ADR-017: Sink Required Fields Contract

**Status:** Accepted
**Date:** 2026-04-20
**Supersedes:** none
**Depends on:** ADR-010

## Context

Sinks already had an inline transactional backstop
(`SinkTransactionalInvariantError`) that checks required fields at the commit
boundary. That backstop is intentionally late and exists to catch state
divergence. It does not provide the Layer 1 intent attribution ELSPETH needs
when a row arrives at a sink already missing a declared required field.

## Decision

Introduce `SinkRequiredFieldsContract` as a `boundary_check` adopter under
ADR-010.

- Violation class: `SinkRequiredFieldsViolation`
- Payload schema: `SinkRequiredFieldsPayload`
- Tier: 1
- Runtime observation: row payload membership (`row_data.keys()`)
- Optional context: use `row_contract` only for richer coalesce-merge
  annotation on the primary sink path
- Call posture: run before `_validate_sink_input()` and before sink I/O on both
  primary and failsink paths

## Two-Layer Architecture

- Layer 1: `SinkRequiredFieldsViolation`
  This is the dispatcher-owned pre-write contract.
- Layer 2: `SinkTransactionalInvariantError`
  This remains the inline transactional backstop for divergence between Layer 1
  evaluation and commit.

The two signals are intentionally distinct and must not be merged.

## Consequences

- Missing sink-required fields are attributed before schema validation can mask
  them as generic plugin validation failures.
- Failsinks are first-class sink boundaries and run the same Layer 1 contract.
- The Layer 2 transactional backstop remains in place for the rarer
  post-Layer-1 divergence case.
