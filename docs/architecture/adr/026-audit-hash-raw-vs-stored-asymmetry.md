# ADR-026: Audit Hashes Fingerprint What Arrived vs What Was Stored — the Raw/Sanitized Asymmetry Is Deliberate

**Date:** 2026-05-30
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** trust-tier, audit-hashing, canonical, tier-3-boundary, cicd-judge-allowlist

## Context

ELSPETH hashes row data into the Landscape audit trail at several write
sites. Two of those sites compute a hash over **raw Tier-3 input**; one
computes a hash over the **sanitized row** that was actually stored. The
two are deliberately different, but until now the rationale lived only in
inline comments and in cicd-judge allowlist rationales — an enforcement
artifact, not a discoverable design record. This ADR makes the decision
discoverable, because the absence of it produced a concrete review
incident (below).

### The two hashing shapes

There are two distinct shapes in the codebase for "hash a row that may be
non-canonical" (e.g. carrying `NaN`/`Infinity` floats that passed source
type-validation but are not canonical-JSON serializable):

1. **Dual-semantics fallback** (audit-write sites in
   `src/elspeth/core/landscape/data_flow_repository.py`):

   ```python
   try:
       row_hash = stable_hash(row_data)
       row_data_json = canonical_json(row_data)
   except (ValueError, TypeError) as e:
       row_hash = repr_hash(row_data)          # load-bearing fallback
       row_data_json = json.dumps(NonCanonicalMetadata.from_error(row_data, e).to_dict(), allow_nan=False)
   ```

   Present at `create_row` (~lines 415–439), `record_validation_error`
   (~lines 1493–1502), and the transform-errors path (~lines 1652–1662).
   These sites hash data that has **not** been sanitized.

2. **Sanitize-then-single-hash** (the orchestrator telemetry/quarantine
   path, `src/elspeth/engine/orchestrator/core.py`):

   ```python
   source_item = replace(source_item, row=sanitize_for_canonical(source_item.row))   # :1787
   ...
   quarantine_content_hash = stable_hash(source_item.row)                            # :1851  (no fallback)
   ```

   The row is sanitized in place (`NaN`/`Inf` → `None`) **before** both
   the audit `create_row` write and the telemetry `content_hash`, so
   `stable_hash` cannot raise and no `repr_hash` fallback is needed. The
   cicd-judge **blocked** a `repr_hash` fallback here precisely because
   it would be dead code after the pre-sanitization.

### The review incident this ADR resolves

A reify-campaign agent, removing the (genuinely dead) orchestrator
fallback, observed the surviving dual-semantics fallbacks in
`data_flow_repository.py` and asked: *should the audit paths also
sanitize-then-single-hash, so the audit and telemetry hashes of one row
agree and the audit trail never carries two hash semantics under one
field name?* The framing assumed a **telemetry-vs-audit divergence** for
the same row.

Two independent SME reviews (a solution-design reviewer and a
systems-thinking pattern-recognizer) traced the call paths and reached
the same conclusion, recorded here. Their full verdicts are attached to
Filigree issue `elspeth-a90f68e076`.

### What the call-path trace established

- **The telemetry-vs-audit divergence does not exist.** `core.py:1787`
  reassigns `source_item` to its *sanitized* form before **both**
  `create_row` (the audit `data_hash`, via `create_quarantine_token` →
  `engine/tokens.py:176`) **and** the telemetry `content_hash`
  (`core.py:1851`). For a quarantined row, audit `source_data_hash` ==
  telemetry `content_hash` — one algorithm, no divergence. Pinned by
  `tests/integration/pipeline/orchestrator/test_quarantine_routing.py:727`.
  No cross-channel hash-agreement invariant exists anywhere in `src/` or
  `tests/`; none is relied upon.

- **There is a real divergence, on a different axis, and it is correct.**
  `validation_errors.row_hash` is computed from **raw** Tier-3 data — the
  validation error is recorded during source iteration, *before*
  sanitization (`core.py:1781` pops a pending error keyed to the raw
  row) — while `rows.data_hash` is computed from the **sanitized** row.
  The two records are explicitly linked (`link_validation_error_to_row`,
  `engine/tokens.py:184-189`). For a `NaN`-bearing row the two hashes
  differ by construction (`NaN` → `None` changes the canonical bytes).

The divergence is between *two audit records of two different facts about
one row*, not between two channels recording the same fact.

## Decision

**The raw/sanitized hashing asymmetry is deliberate and correct. It MUST
NOT be unified.**

- An **error-path** audit hash (`validation_errors.row_hash`,
  `transform_errors.row_hash`) fingerprints **the raw external input that
  arrived** — evidence of *what the Tier-3 source actually sent*,
  including the non-canonical value that made the row a quarantine/error
  case in the first place.

- A **stored-row** audit hash (`rows.data_hash`) and the telemetry
  `content_hash` fingerprint **the sanitized row that was actually
  persisted** — evidence of *what entered the pipeline's deterministic
  control surface*.

These answer different audit questions. **Sanitizing the error-path hash
would be fabrication** under CLAUDE.md's fabrication test: the audit
record would attest to a normalized value (`None`) the external system
never sent, erasing the very anomaly the error record exists to capture.

The three `repr_hash` fallbacks in `data_flow_repository.py` **stay**.
They are load-bearing: they hash raw Tier-3 data that genuinely can
contain `NaN`/`Inf`, and the doctrine is that losing the audit record is
worse than recording a repr-based hash. The cicd-judge ACCEPTED them on
that basis, and that judgement is affirmed here.

### What this is NOT

- This is **not** an endorsement of the "sanitize-then-single-hash
  everywhere" proposal. That proposal is rejected (see Alternatives).
- This is **not** a claim that the error-path hash and the stored-row
  hash *should* agree. They should differ; the difference is information.
- This is **not** a claim that the current implementation has no
  remaining defect. It has one — the post-purge ambiguity below — which
  is a *separate, narrower* issue tracked independently, **not** resolved
  by this ADR.

## Consequences

### Positive Consequences

- The asymmetry is now discoverable. A maintainer (or a future cicd-judge
  reaudit) who sees three audit sites with a `repr_hash` fallback and one
  telemetry site without no longer reads "inconsistent" and "fixes" it by
  unifying — which would erase audit fidelity. This ADR is the governance
  anchor the review incident lacked.
- An auditor following the `validation_errors` → `rows` link and seeing
  two different hashes for one row has a recorded explanation: one
  fingerprints the raw input, the other the stored row.
- The four signed cicd-judge allowlist entries in
  `config/cicd/enforce_tier_model/core.yaml` (the `create_row`,
  `record_validation_error`, and transform-errors fallbacks) can be
  back-linked to this ADR, turning their inline rationales into a
  traceable decision.

### Negative Consequences

- **The post-purge ambiguity remains open.** When a `repr_hash` fallback
  fires, *which* algorithm produced the hash is recorded only in
  `NonCanonicalMetadata` inside the **purgeable** `row_data_json`. After a
  payload-retention purge, a `repr_hash` and a `stable_hash` are
  indistinguishable in the hash column — which partially undermines the
  "hashes survive payload deletion" guarantee for the narrow slice of
  `NaN`/`Inf`-bearing error rows. This ADR records *why the hashes
  differ*; it does **not** make the hash *self-describing*. The fix — a
  durable, non-purgeable `hash_algorithm`/`is_non_canonical` column
  adjacent to each `*_hash` column, forward-only with no hash-value
  migration — is tracked in **`elspeth-a90f68e076`** and is out of scope
  here.
- **Proliferation risk persists.** The systems-thinking review classified
  the replicated fallback as *Drift Toward Low Standardization*: a
  reinforcing loop where each new Tier-3 audit-write copies the nearest
  signed sibling, the judge ACCEPTs it in isolation, and the signed entry
  reads as precedent for the next copy. The tier-model gate operates
  per-finding, never per-concept, so nothing structurally caps the number
  of independent re-implementations of "hash a possibly-non-canonical
  row." The structural fix (a single shared `audit_hash` primitive both
  subsystems call, plus a cross-site lint) is the higher-leverage
  follow-up in `elspeth-a90f68e076`. This ADR does not install it; it only
  records that the *semantic* asymmetry the primitive must preserve is
  intentional.

### Neutral Consequences

- The stale-rationale note: `create_row`'s `repr_hash` fallback is dead
  code at its sole live caller (`create_quarantine_token`, which
  pre-sanitizes). It is **kept** as a contract-defense guard — a future
  caller passing `quarantined=True` without sanitizing would otherwise
  lose a Tier-3 audit record. Its allowlist entry should be re-justified
  on contract-defense grounds rather than "data may contain NaN/Inf"
  (false at the live site). Tracked under `elspeth-a90f68e076`.

## Alternatives Considered

### Alternative 1: Sanitize-then-single-hash at every audit site

**Description:** Apply `sanitize_for_canonical` before hashing in
`record_validation_error`, the transform-errors path, and `create_row`,
removing the `repr_hash` fallbacks so every audit hash uses one
algorithm — matching the orchestrator telemetry path.

**Rejected because:** It would sanitize the row **before** the error-path
hash, making `validation_errors.row_hash` attest to a normalized value
the external system never sent — fabrication under the CLAUDE.md test, and
the destruction of the raw-anomaly fingerprint the error record exists to
preserve. It would also change emitted hash *values* (a schema-semantics
migration that breaks historical reproducibility, baselines, and fixtures)
and require re-signing the four judge-signed allowlist entries — all to
fix a cross-channel divergence that, per the call-path trace, does not
exist. Highest-risk, negative-leverage option.

### Alternative 2: Leave the asymmetry undocumented (status quo ante)

**Description:** Keep the inline comments and allowlist rationales as the
only record of why the asymmetry exists; write no ADR.

**Rejected because:** This is exactly the state that produced the review
incident. An allowlist rationale is an enforcement artifact scoped to one
suppression; it is not a discoverable design record, and the next
maintainer (or judge reaudit) asking "why are these inconsistent?" has no
governance anchor and is one step from proposing Alternative 1.

### Alternative 3: Remove the `repr_hash` fallbacks as dead code

**Description:** Since the orchestrator pre-sanitizes, delete the
`repr_hash` fallbacks from `data_flow_repository.py` as unreachable.

**Rejected because:** Only the `create_row` fallback is dead at its
*current* live caller. The `record_validation_error` and transform-errors
sites receive **raw** row data from source/executor call sites that do not
pre-sanitize, so their fallbacks are live and load-bearing. Even the
`create_row` fallback is retained as a contract-defense guard (see Neutral
Consequences). Removing them would crash the audit write on the first
`NaN`-bearing error row — losing the record the doctrine says to preserve.

## Related Decisions

- Related: ADR-010 (declaration-trust framework) and ADR-021 (sources and
  sinks uniformly boundary) — the trust-tier classification machinery that
  defines the Tier-3 boundary this ADR's hashes straddle.
- Follow-up: Filigree `elspeth-a90f68e076` — durable hash-algorithm
  discriminator column (closes the post-purge ambiguity), a shared
  `audit_hash` primitive, and a cross-site lint (closes the proliferation
  loop). Both SME verdicts are attached there.

## References

- `src/elspeth/core/landscape/data_flow_repository.py` — the three
  dual-semantics fallback sites: `create_row` (~415–439),
  `record_validation_error` (~1493–1502), transform-errors (~1652–1662).
- `src/elspeth/engine/orchestrator/core.py` — `sanitize_for_canonical` at
  `:1787`, quarantine `content_hash` at `:1851`, pending-error pop at
  `:1781`.
- `src/elspeth/engine/tokens.py` — `create_quarantine_token` (~176) and
  `link_validation_error_to_row` (~184–189).
- `src/elspeth/core/canonical.py` — `stable_hash`, `repr_hash`,
  `sanitize_for_canonical`, `canonical_json`.
- `config/cicd/enforce_tier_model/core.yaml` — the four signed cicd-judge
  allowlist entries for the fallback sites.
- `tests/integration/pipeline/orchestrator/test_quarantine_routing.py:727`
  — pins the orchestrator's sanitized single-hash (`emitted_hash !=
  repr_hash(nan_row)`).

## Notes

This ADR records only the **semantic decision** — that the raw/stored
hashing asymmetry is intentional and must be preserved. The companion
**auditability fix** (making the hash self-describing via a durable
discriminator column) and the **structural fix** (a shared hashing
primitive + cross-site lint to stop proliferation) are tracked in
`elspeth-a90f68e076` and are deliberately out of scope here. Any
implementation of the shared primitive MUST be a value-preserving router
that keeps each site's existing semantics — normalizing would change
emitted hash values and break historical reproducibility, which is the
opposite of what an audit trail requires.

Editing `data_flow_repository.py` to land the follow-up will invalidate
the whole-file `file_fingerprint` of the signed `core.yaml` entries in
that file, requiring an operator-held HMAC re-sign (a keyless agent cannot
self-serve it). This is friction, not data risk, and should be planned
into the follow-up.
