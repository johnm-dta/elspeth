# ADR-025: Audited Inline Blob Content

**Date:** 2026-05-24
**Status:** Accepted
**Deciders:** ELSPETH maintainer
**Tags:** composer, blob-store, audit, late-binding, rc5

## Context

ELSPETH already has two late-binding forms in pipeline configuration:

- `{secret_ref: NAME}` resolves sensitive values through the secrets resolver
  and records only HMAC fingerprints.
- `{blob_ref: ID}` binds a source plugin to a blob-backed file path and links
  the blob to the run for lifecycle protection.

The composer now needs to support long-form public or non-secret content in
arbitrary plugin config fields: LLM system prompts, prompt templates, SQL,
regex libraries, JSON templates, public certs, allowlists, and denylists. These
values are not secrets and should not inherit secret redaction semantics, but
they still need audit-grade provenance. A run must prove which blob content was
resolved, what hash was pinned, where it was injected, and that lifecycle
pinning protected the blob while the run used it.

The prior design question was whether to add a third ref form, such as
`{blob_content_ref: ID}`, or widen the existing `blob_ref` shape. The wider
decision has long-term consequences: every new ref form adds validation,
composer, runtime, audit, and lifecycle surfaces.

The cap decision also needs data. Unbounded config content reads are not
acceptable, but arbitrary caps are likely to be revisited when operators hit
them. The measured evidence is recorded in
[`blob-inline-content-cap-2026-05-24.md`](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/evidence/blob-inline-content-cap-2026-05-24.md).

## Decision

ELSPETH will widen `blob_ref` with an explicit `mode` discriminator instead of
adding a sibling `blob_content_ref` form.

The authoritative inline-content marker is:

```yaml
system_prompt:
  blob_ref: <uuid>
  mode: inline_content
  sha256: <64-char lowercase sha256 hex>
  encoding: utf-8
```

Rules:

- `mode` is required and must be one of `bind_source` or `inline_content`.
- `mode: bind_source` preserves source path-binding semantics and is source
  only.
- `mode: inline_content` may appear in plugin option fields where a string
  value is expected.
- `sha256` is required for `inline_content` and forbidden for `bind_source`.
- Runtime fetches blob bytes, verifies the pinned hash, decodes using a closed
  encoding set, and fails closed on mismatch or decode failure.
- The validate path checks metadata only through `get_blob()`: existence,
  status, declared size, MIME type, and stored hash. It does not read content or
  write audit rows because no run exists yet.
- Runtime preflight writes lifecycle `blob_run_links` with
  `direction='input'` for every unique referenced blob before dereferencing
  content.
- Runtime records every successful inline-content resolution in a dedicated
  audit table before plugin instantiation can observe the resolved string.
- Composer exposes blob metadata to the LLM, not bytes. Composer-authored
  markers pin `sha256` from `BlobRecord.content_hash`.
- No new late-binding ref form may be introduced without amending this ADR or
  creating a successor ADR.

Caps:

- Per-ref upper cap: `256 KiB`.
- Aggregate per-config cap: `1 MiB`.
- Soft lower threshold: composer-authored refs below `256 B` warn because they
  are usually an anti-pattern, but validation does not hard-reject a ref solely
  because it is small.

The lower threshold is intentionally soft. Current examples include short
regexes, allowlist-like values, contact strings, and prompt snippets; a hard
minimum would reject small but legitimate public content while adding little
safety.

## Consequences

### Positive Consequences

- Public/non-secret content can be late-bound without extending the secrets
  system or weakening secret handling.
- `blob_ref` remains the single blob-backed late-binding family, reducing
  future validation and composer surface area.
- Runtime hash pinning gives an auditor the exact content identity used for a
  run.
- `blob_run_links` continues to protect input blobs from deletion during active
  runs.
- The cap values are tied to current evidence rather than a guessed constant.

### Negative Consequences

- The widened marker requires coordinated changes across contracts, core
  resolver code, validation, runtime preflight, audit persistence, composer
  tools, and tests.
- `blob_ref` now has two semantic modes, so marker recognition must be strict
  and mechanically tested.
- Operators who want larger single prompt artifacts than `256 KiB` must split
  them or revisit this ADR with new evidence.

### Neutral Consequences

- Source `blob_ref` path binding remains conceptually separate from
  inline-content substitution even though both use the same marker family.
- Validation and runtime intentionally differ: validation uses metadata only;
  runtime reads and verifies bytes because it has a run to audit.
- Frontend polish for browsing and swapping inline refs can remain separate
  from the audit-grade API/runtime capability.

## Alternatives Considered

### Add `{blob_content_ref: ID}`

**Description:** Introduce a third ref shape alongside `secret_ref` and
`blob_ref`.

**Rejected because:** It duplicates the blob-backed late-binding concept while
sharing only the blob table. That would require separate validation, composer,
runtime, and audit paths for the same content-addressed primitive.

### Extend `secret_ref` for non-secret content

**Description:** Treat public content as named values resolved by the secrets
resolver, with selective redaction disabled.

**Rejected because:** Secrets discipline exists because values are sensitive.
Using it for public content would either over-redact audit evidence or weaken
the secret path with exceptions.

### Inline content directly in composer-authored config

**Description:** Let the composer write long prompts, SQL, and templates
directly into persisted configuration fields.

**Rejected because:** This hides content provenance inside mutable config state,
does not use blob lifecycle pinning, and gives poorer hash-level audit evidence
than a content-addressed blob reference.

### Use a hard lower size rejection

**Description:** Reject inline refs below `256 B` to prevent overuse.

**Rejected because:** Measured config and example data contain legitimate short
public strings. A composer warning discourages overuse without making direct
YAML brittle.

## Related Decisions

- ADR-003: Schema Validation Lifecycle
- ADR-005: Declarative DAG Wiring
- ADR-010: Declaration Trust Framework
- ADR-021: Sources and Sinks Are Uniformly Boundary by Architecture
- ADR-023: Custom Python Static Analyzer for ELSPETH-Specific CI Invariants

## References

- [Audited Content Injection widened `blob_ref` design](../../../docs-archive/2026-06-28-docs-cleanout/docs/superpowers/specs/2026-05-03-config-content-ref-design.md)
- [Blob inline content cap evidence](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/evidence/blob-inline-content-cap-2026-05-24.md)
- Filigree: `elspeth-fdebcaa79a`
