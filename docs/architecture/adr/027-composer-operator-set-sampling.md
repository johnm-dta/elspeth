# ADR-027: Composer Sampling Is Operator-Set Configuration

**Date:** 2026-06-04
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** composer, web-config, llm, telemetry, audit

## Context

The web composer previously hardcoded LLM sampling parameters in the
composer service: temperature `0.0`, and seed `42` only when LiteLLM reported
support for the OpenAI `seed` parameter. A guard comment treated sampling
configurability as a Tier 2 decision that needed an ADR before reading from
settings or environment.

That design failed against the current default composer model, `gpt-5.5`, a
reasoning-model default that rejects any non-default temperature. Keeping a
hardcoded temperature forced capability inference and reactive retry logic to
work around a value the operator should have owned directly.

The inference also created a second policy problem for `seed`: the system
predicted per-model provider capability instead of sending the operator's
configured intent and surfacing rejection as a configuration error.

## Decision

Composer LLM sampling is now explicit operator configuration.

- Add nullable `WebSettings.composer_temperature` and `composer_seed` fields.
- Default both to `None`, meaning the field is omitted from the provider
  request.
- Send configured values verbatim across all composer LLM paths: main compose
  loop, advisor, diagnostics text path, guided chain solver, guided chat solver,
  and auto-title.
- Delete per-model sampling inference helpers and hardcoded sampling constants.
- Record the configured value, including `None`, on per-call
  `ComposerLLMCall` audit records so the audit row mirrors the actual request
  shape.
- Validate the effective sampling config at boot when
  `composer_boot_probe_enabled` is true. A provider bad request is a fatal
  operator config error; transient/auth/network failures are graceful so
  non-LLM web features can still boot.
- Emit `composer.boot_config` telemetry at boot with the effective model and
  sampling values.

We are not adding separate advisor or auto-title sampling knobs. Those paths
share the composer sampling values.

## Consequences

### Positive Consequences

- Out-of-box composer requests omit sampling parameters, which is compatible
  with the reasoning-model default.
- Operators using models that accept deterministic sampling can set
  `composer_temperature=0.0` and `composer_seed=42` explicitly.
- Misconfigured sampling fails early at boot with a precise error instead of
  failing on the first user compose request.
- Per-call audit rows remain truthful: they show the values actually sent, not
  inferred defaults.

### Negative Consequences

- The old deterministic default is no longer automatic. Operators who want it
  must configure it.
- Provider rejection of configured sampling is now an operator-owned boot
  failure, not a hidden retry/omit behavior.

### Neutral Consequences

- `composer_boot_probe_enabled` exists for tests and offline development where
  a real provider call during lifespan would be inappropriate.
- The boot config is telemetry, not a Landscape audit record. Per-call LLM
  audit rows remain the durable record for actual composer requests.

## Alternatives Considered

### Alternative 1: Model-aware reactive retry

**Description:** Send the hardcoded temperature first, detect provider
temperature rejection by exception class/prose, retry without temperature, and
cache observed rejecting models.

**Rejected because:** It preserves the wrong ownership model. The system would
still infer model capability and mutate request shape instead of letting the
operator state intent.

### Alternative 2: Keep hardcoded deterministic sampling

**Description:** Continue forcing temperature `0.0` and seed `42` where
possible.

**Rejected because:** The current reasoning-model default rejects non-default
temperature, so this default is no longer coherent out of the box.

## Related Decisions

- Supersedes:
  `docs/superpowers/specs/2026-05-31-composer-model-aware-temperature-design.md`
- Implements:
  `docs/superpowers/specs/2026-06-03-composer-operator-set-sampling-config-design.md`

## References

- `docs/superpowers/plans/2026-06-03-composer-operator-set-sampling-config.md`
- `docs/superpowers/specs/2026-06-03-composer-operator-set-sampling-config-design.md`
