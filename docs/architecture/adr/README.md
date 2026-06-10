# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for ELSPETH.

## What is an ADR?

An Architecture Decision Record (ADR) captures an important architectural decision made along with its context and consequences. ADRs help us:

- Document the reasoning behind architectural choices
- Understand trade-offs and alternatives considered
- Provide context for future maintainers
- Enable informed decisions about changes

## Format

We use a modified version of Michael Nygard's ADR template. See `000-template.md` for the structure.

## Index of Decisions

| ADR | Title | Date | Status |
|-----|-------|------|--------|
| [000](000-template.md) | ADR Template | - | Template |
| [001](001-plugin-level-concurrency.md) | Plugin-Level Concurrency | 2026-01-22 | **Accepted** |
| [002](002-routing-copy-mode-limitation.md) | Routing Copy Mode Limitation | 2026-01-24 | **Accepted** |
| [003](003-schema-validation-lifecycle.md) | Schema Validation Lifecycle | 2026-01-24 | **Accepted** |
| [004](004-adr-explicit-sink-routing.md) | Explicit Sink Routing | 2026-02-09 | **Accepted** |
| [005](005-adr-declarative-dag-wiring.md) | Declarative DAG Wiring | 2026-02-09 | **Accepted** |
| [006](006-layer-dependency-remediation.md) | Layer Dependency Remediation | 2026-02-22 | **Accepted** |
| [007](007-pass-through-contract-propagation.md) | Pass-Through Contract Propagation | - | **Accepted** |
| [008](008-runtime-contract-cross-check.md) | Runtime Contract Cross-Check | - | **Accepted** |
| [009](009-pass-through-pathway-fusion.md) | Pass-Through Pathway Fusion | - | **Accepted** |
| [010](010-declaration-trust-framework.md) | Declaration Trust Framework | - | **Accepted** |
| [011](011-declared-output-fields-contract.md) | Declared Output Fields Contract | - | **Accepted** |
| [012](012-can-drop-rows-contract.md) | Can-Drop-Rows Contract | - | **Accepted** |
| [013](013-declared-required-fields-contract.md) | Declared Required Fields Contract | - | **Accepted** |
| [014](014-schema-config-mode-contract.md) | Schema Config Mode Contract | - | **Accepted** |
| [015](015-creates-tokens-contract.md) | Creates-Tokens Contract | - | **Accepted** |
| [016](016-source-guaranteed-fields-contract.md) | Source Guaranteed Fields Contract | - | **Accepted** |
| [017](017-sink-required-fields-contract.md) | Sink Required Fields Contract | - | **Accepted** |
| [018](018-producer-site-outcome-discrimination.md) | Producer-Site Outcome Discrimination | 2026-05-02 | **Accepted** |
| [019](019-two-axis-terminal-model.md) | Two-Axis Terminal Model — Lifecycle, Outcome, and Path | 2026-05-04 | **Accepted** |
| [020](020-retire-batch-llm-transforms.md) | Retire Batch-LLM Transforms (`azure_batch_llm`, `openrouter_batch_llm`) | 2026-05-06 | **Accepted** |
| [021](021-sources-and-sinks-uniformly-boundary.md) | Sources and Sinks Are Uniformly Boundary by Architecture | 2026-05-18 | **Accepted** |
| [022](022-shareable-reviews.md) | Shareable Reviews — Completion Gestures, Signed Tokens, and the Composer Completion Events Table | 2026-05-19 | **Accepted** |
| [023](023-custom-python-ci-analyzer.md) | Custom Python Static Analyzer for ELSPETH-Specific CI Invariants (the `elspeth-lints` Package) | 2026-05-19 | **Accepted** |
| [024](024-delivery-governance-for-single-maintainer-mode.md) | Delivery Governance for Single-Maintainer Mode | 2026-05-19 | **Accepted** |
| [025](025-audited-inline-blob-content.md) | Audited Inline Blob Content | 2026-05-24 | **Accepted** |
| [025](025-multi-source-ingestion.md) | Multi-Source Ingestion — Source Surface Is Plural | 2026-05-23 | **Accepted** |
| [026](026-audit-hash-raw-vs-stored-asymmetry.md) | Audit Hashes Fingerprint What Arrived vs What Was Stored — the Raw/Sanitized Asymmetry Is Deliberate | 2026-05-30 | **Accepted** |
| [026](026-durable-token-scheduler.md) | Durable Token Scheduler | 2026-05-23 | **Accepted** (with stated preconditions) |
| [027](027-composer-operator-set-sampling.md) | Composer Sampling Is Operator-Set Configuration | 2026-06-04 | **Accepted** |
| [028](028-queue-vs-coalesce-not-duplicates.md) | QUEUE and COALESCE Are Not Duplicates — Leave Them Separate | 2026-06-11 | **Accepted** |
| [029](029-journal-is-barrier-buffer-truth.md) | Scheduler Journal Is the Single Source of Barrier-Buffer Truth | 2026-06-11 | **Accepted** |

## Status Definitions

- **Proposed:** Decision is under discussion
- **Accepted:** Decision has been made and is in effect
- **Deprecated:** Decision is no longer recommended but still in use
- **Superseded:** Decision has been replaced by a newer ADR

## Creating a New ADR

1. Copy `000-template.md` to a new file with the next number: `NNN-short-title.md`
2. Fill in the template sections
3. Submit for review
4. Update this README with the new ADR entry
5. Mark status as "Proposed" until accepted

## Related Documentation

- [Architecture Overview](../overview.md)
- [Requirements](../requirements.md)
- [CLAUDE.md Guidelines](../../../CLAUDE.md)
