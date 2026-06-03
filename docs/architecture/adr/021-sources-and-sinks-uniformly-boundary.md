# ADR-021: Sources and Sinks Are Uniformly Boundary by Architecture

**Date:** 2026-05-18
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** audit-readiness, trust-tier, boundary-classification, supersedes-trust.py

## Context

The audit-readiness panel's plugin-trust row classifies every plugin in
a composition as either **boundary** (crosses a Tier-3 trust boundary,
must be visible to auditors) or **internal** (operates only on Tier-2
pipeline data). The classification drives the panel's "external-boundary
plugin(s) recorded" line and the per-component detail rows.

### The pre-Phase-7A state

Before Phase 7A, classification lived in `elspeth.web.audit_readiness.trust`,
a curated module that hand-maintained two facts per builtin plugin:

1. A `data_trust_tier` class attribute on each plugin.
2. An `_INTERNAL_PLUGIN_CLASSES` allowlist that explicitly *excluded* a
   small set of sinks (notably `csv` and `json`) from boundary
   classification. The rationale at the time: local-filesystem writes
   "don't really cross a trust boundary" in the same operational sense
   that a network sink does.

The exclusion list was load-bearing but undocumented: a reviewer of any
PR adding a new sink had no signal that the sink's classification was a
design decision rather than a registration detail. The list also drifted:
adding `chroma_sink` and `database` to the catalog required no edit to
`trust.py`, so they silently inherited "internal" status until someone
noticed.

### The Phase 7A widening

Phase 7A discharged the `data_trust_tier` attribute under the
no-legacy-code policy and replaced classification with a predicate
derived from `(kind, determinism)`:

```python
kind in ("source", "sink") or
plugin_cls.determinism in _AUDIT_FLAGGED_DETERMINISMS
```

The predicate is in `elspeth.web.audit_readiness.service._build_plugin_trust_row`;
the expected partition is pinned in
`elspeth.web.audit_readiness.boundary_expectations`. The parity test in
`tests/unit/web/audit_readiness/test_boundary_predicate_parity.py`
asserts they match the live builtin catalog.

The predicate widens the classification: **every Source and every Sink
is now uniformly boundary, regardless of whether the destination is a
local file or a remote service.** The csv/json sink exclusion is gone.

### The structural critique

The widening was deliberate but the rationale lived only in inline
comments inside `boundary_expectations.py`. A compliance-focused
reviewer (the Linda persona in `project_composer_personas`) now sees
every local CSV write annotated as "crosses an external boundary" and
has no discoverable record of why that classification was chosen over
the prior, narrower one. Without an ADR, the next maintainer asking
"should this sink be excluded?" has no governance artifact to consult
and no audit-trail anchor for the decision.

## Decision

Classify every Source and every Sink as boundary, unconditionally. The
predicate `kind in ("source", "sink")` short-circuits the
`_AUDIT_FLAGGED_DETERMINISMS` check for these kinds; the determinism
declaration on a Source or Sink does not affect its boundary
classification.

The architectural premise: **a Source reads external data into the
pipeline and a Sink writes pipeline data out — both cross Tier-3 by
definition.** The destination's locality (local file vs remote service)
is operationally significant but does not change the trust-tier
classification. A local CSV write still produces a payload outside the
pipeline's deterministic control surface; recovering that payload's
provenance is the auditor's question, and the readiness panel surfaces
the question regardless of where the bytes landed.

Transforms remain conditionally classified: only those whose declared
`Determinism` is `EXTERNAL_CALL` or `NON_DETERMINISTIC` are boundary
(see `_AUDIT_FLAGGED_DETERMINISMS`).

### What this is NOT

This is not a claim that local-filesystem and remote-service sinks are
operationally identical. They have different failure modes, different
retention semantics, different exfiltration risks, and different
regulatory exposure. The readiness panel's plugin-trust row makes one
specific claim about them: both produce a Tier-3 crossing whose
provenance an auditor will eventually want to trace.

## Consequences

### Positive Consequences

- A new Source or Sink is classified correctly at registration time
  without an explicit allowlist edit. The classification cannot
  silently drift the way `_INTERNAL_PLUGIN_CLASSES` did.
- The classification rule is one line of code (the predicate) plus the
  `_AUDIT_FLAGGED_DETERMINISMS` set. An auditor reading the source can
  see the rule in full without tracing through a curation history.
- Adding a new Tier-3 crossing requires a production-code diff in
  `boundary_expectations.py` (the parity test fails otherwise), so a PR
  reviewer always sees the audit-relevant change.
- The catalog and the readiness panel diverge intentionally on
  kind-default-determinism suppression (see
  `web/catalog/service.py:_derive_audit_characteristics` for the
  reciprocal docstring). The two surfaces answer different questions
  about the same plugin; treating them as separate classifications is
  honest.

### Negative Consequences

- The readiness panel's per-component detail text now reads
  `[sink] my_csv (csv) — crosses an external boundary` for a sink that
  writes only to a local file path. A compliance-focused user may read
  this as "this run exfiltrates data to a remote service" when the
  classification is making a narrower trust-tier claim.

  This is a UX wording question, not a classification question. A
  future readiness-panel iteration could distinguish "writes to local
  filesystem" from "writes to remote service" at the detail-row
  rendering layer without changing the underlying boundary
  classification. The distinction would draw on the sink's
  configuration (the resolved destination path) rather than its class,
  since the same `csv` sink can be configured to write to either.

  Captured as a follow-up UX nit, not a blocker for this ADR.
- Operators familiar with the pre-Phase-7A behaviour may briefly read
  the widened classification as a defect. The
  `boundary_expectations.py` docstring and this ADR are the
  discoverable answer.

### Neutral Consequences

- The `_AUDIT_FLAGGED_DETERMINISMS` set name now reflects only the
  Transform classification axis; Sources and Sinks no longer consult
  it. The set name is preserved because renaming it would touch every
  reader and the existing meaning is correct for the consulting site.
- The rule version is pinned in
  `_BOUNDARY_RULE_VERSION = "phase-7a-v1"` next to the predicate, so a
  future persistence phase (where derived audit characteristics enter
  the Landscape as a legal record) can stamp the version verbatim
  alongside each persisted verdict.

## Alternatives Considered

### Alternative 1: Preserve the csv/json exclusion in the new predicate

**Description:** Keep an `_INTERNAL_SINK_NAMES = frozenset({"csv", "json"})`
exclusion alongside `_AUDIT_FLAGGED_DETERMINISMS`, restoring the
pre-Phase-7A behaviour.

**Rejected because:** The exclusion would need to enumerate every
local-filesystem-only sink in the catalog, which is a moving target
(adding `database`, `chroma_sink`, `azure_blob`, `dataverse` each
required a separate judgement). The exclusion list is also misleading:
a `csv` sink configured with an HTTP `path:` writes remotely, and an
`azure_blob` sink configured with a local mount path writes locally.
Classification by class name cannot answer the operational question
correctly in either direction.

### Alternative 2: Read the resolved destination configuration

**Description:** Inspect each sink instance's configured destination at
readiness-build time and classify based on whether the destination
resolves to a local filesystem path or a remote URI.

**Rejected because:** Classification would then depend on runtime
configuration state, which the readiness service does not currently
have access to (it sees `state.outputs` from the composition, not the
instantiated sink). This is a larger architectural change and conflates
the design-time classification question ("what does this plugin do?")
with the runtime configuration question ("where will this instance
write?"). The UX wording fix in Negative Consequences above is the
right place for that distinction, not the trust-tier classification.

### Alternative 3: Add a per-plugin `is_remote_boundary` attribute

**Description:** Require each Source/Sink to declare a boolean
indicating whether it crosses a network boundary, and use that
attribute in the predicate.

**Rejected because:** Same problem as Alternative 1 — the answer
depends on configuration, not on the class. A `csv` sink could be
configured to POST to a webhook; an `azure_blob` sink could be
configured to write to a mounted local path. A per-class boolean would
encode the wrong abstraction level.

## Related Decisions

- Supersedes: the curation discipline previously documented in
  `elspeth.web.audit_readiness.trust` (module deleted Phase 7A).
- Related: ADR-010 (declaration-trust framework) — the boundary
  classification consumes `Determinism` declarations enforced by the
  declaration-trust framework's `__init_subclass__` gate.

## References

- `src/elspeth/web/audit_readiness/service.py` —
  `_build_plugin_trust_row`, `_AUDIT_FLAGGED_DETERMINISMS`,
  `_BOUNDARY_RULE_VERSION`.
- `src/elspeth/web/audit_readiness/boundary_expectations.py` — the
  expected partition pinned for parity tests.
- `tests/unit/web/audit_readiness/test_boundary_predicate_parity.py` —
  parity test that fails when a catalog change drifts the partition.
- `src/elspeth/web/catalog/service.py` —
  `_derive_audit_characteristics` cross-references this ADR for the
  related-but-distinct catalog-card classification question.

## Notes

The follow-up UX question (distinguishing local-filesystem from
remote-service sinks in the detail-row rendering) is intentionally not
in scope for this ADR. It is a separate decision about the readiness
panel's presentation layer; the trust-tier classification is the
structural decision recorded here.
