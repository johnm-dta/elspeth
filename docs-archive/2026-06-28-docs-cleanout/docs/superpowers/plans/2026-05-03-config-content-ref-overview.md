# Audited Content Injection (Widened `blob_ref`) — Plan Overview

> **For agentic workers:** This is a multi-phase plan. Each phase is a separate plan file; execute them sequentially. REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each phase plan.

**Spec:** `docs/superpowers/specs/2026-05-03-config-content-ref-design.md`
**Epic:** [elspeth-fdebcaa79a](filigree:elspeth-fdebcaa79a) — Audited content injection: widen `blob_ref` to support `inline_content` mode for plugin config fields
**Parent release:** [elspeth-d0ef7cbf54](filigree:elspeth-d0ef7cbf54) — RC 5

**Goal:** Operators can place long-form content (LLM system prompts, SQL queries, regex libraries, JSON templates, public certs, allow/denylists) into any plugin config field by reference, with content resolved at runtime from the existing blob store and recorded in a new Tier-1 audit table. The widened `blob_ref` marker carries a `mode` discriminator (`bind_source` keeps the existing source-data semantics; `inline_content` is the new content-binding form) and a composer-pinned `sha256` hash that closes the audit-fraud window between submit time and run time.

**Architecture:** Six sequential phases plus a deferred frontend phase. P1 ships the ADR and VAL-data-grounded numeric caps as markdown. P2 adds L0 contracts and L1 resolver functions with no behaviour change. P2b extracts the canonical DB-form ↔ YAML-form adapter (`generate_pipeline_dict`) so any walker traversing a `composition_state` consumes the same dict shape regardless of whether it came from a typed `CompositionState` or a raw DB row — closing architectural gap [`elspeth-be405bac87`](filigree:elspeth-be405bac87) before P3's walker extension lands. P3 wires the resolver into the runtime path, fail-closed, with the `blob_inline_resolutions` audit table and lifecycle pinning via the existing `blob_run_links` infrastructure — nothing emits inline-content refs yet. P4 opens the composer authorship path: `_prevalidate_plugin_options` strip-before-validate is extended for the widened marker, and Shape 9 is added to the agreement-suite registry with three sub-pins. P5 adds the MCP tools (`list_composer_blobs` for LLM blob discovery, `wire_blob_inline_ref` for authorship) and updates `set_source_from_blob` to emit explicit `mode: bind_source` so every persisted marker round-trips through recognition. P6 (frontend SecretsPanel-equivalent UI) is filed as a follow-up issue and is **not** on the RC5 critical path.

**Phase ordering pivot:** runtime side lands BEFORE composer side. Earlier drafts proposed validation parity in P3 followed by runtime preflight in P4 — that would deliberately ship composer-green / runtime-red as an intermediate state, which is the exact divergence Shape 1 and Shape 8 in `tests/integration/pipeline/test_composer_runtime_agreement.py` exist to close. Revision 1 inverts: runtime fail-closed lands first; composer authorship opens only when runtime is ready to honour it.

**Tech Stack:** Python 3.13, SQLAlchemy 2.x (sync `Engine`), Pydantic 2.x, FastAPI, structlog, OpenTelemetry, Hypothesis (property tests), pytest. Mirrors the secret-ref resolver pattern in `src/elspeth/core/secrets.py` and the blob-service pattern in `src/elspeth/web/blobs/service.py`.

---

## Phase plans

| Phase | File | What it delivers | Risk profile |
|---|---|---|---|
| P1 | `2026-05-03-config-content-ref-phase-1-adr.md` | ADR-021 with VAL-data-grounded numeric caps; closure rule "no new ref forms without ADR amendment"; H4 LLM visibility model; M2 encoding decision; `direction='input'` reuse rationale | Markdown only |
| P2 | `2026-05-03-config-content-ref-phase-2-l0-l1.md` | `contracts/blobs_inline.py` (L0) + `core/blobs_inline.py` (L1) three-function resolver; `AllowedMimeType` move from L3 to L0 | No behaviour change; unit tests only |
| P2b | `2026-05-03-config-content-ref-phase-2b-state-adapter.md` | Extract `generate_pipeline_dict(state) -> dict` from `yaml_generator.generate_yaml`; round-trip identity property test (`yaml.safe_load(generate_yaml(state)) == generate_pipeline_dict(state)` and `state == state_from_record(record_for(state))`); explicit YAML-shape snapshot pin; migrate `delete_blob`'s pre-link active-run guard to consume the adapter; closes architectural gap `elspeth-be405bac87` | Shape-preserving refactor; unit tests only |
| P3 | `2026-05-03-config-content-ref-phase-3-runtime-preflight.md` | `blob_inline_resolutions` table + service method; resolver wired into `_run_pipeline`; lifecycle pinning extended; audit-write primacy; bug-verification at the runtime-resolver and audit-write fix sites; direction-reuse + unique-constraint verification gate; walker consumes `generate_pipeline_dict(state_from_record(record))` from P2b | Fail-closed; nothing emits inline-content refs yet |
| P4 | `2026-05-03-config-content-ref-phase-4-composer-parity.md` | `blob_inline_ref_keys` strip-before-validate extension; `_validate_blob_content_refs` wired into `validate_pipeline`; Shape 9 sub-pins (A/B/C) added to the agreement-suite registry; bug-verification at the composer-parity-strip fix site | Composer authoring path opens; runtime is already fail-closed |
| P5 | `2026-05-03-config-content-ref-phase-5-composer-tool.md` | MCP tools `list_composer_blobs` + `wire_blob_inline_ref`; H4 visibility shape enforced as the tool's return contract; composer-side rejection rules from spec §7.2; `set_source_from_blob` emits explicit `mode: bind_source` | LLM has the affordance |
| P6 (filed as follow-up issue, NOT this epic) | not in this plan tree | SecretsPanel-equivalent UI for blob refs; click-through upload/swap; resolved-content-byte preview gated on operator role | Optional — not on the RC5 critical path |

## Cross-phase dependencies

- P2 depends on no other phase (pure foundation).
- P2b depends on P2 (phase ordering keeps the merge train linear; P2b touches `yaml_generator` and the existing `delete_blob` guard, neither of which depends on P2's L0/L1 modules — but P2b consumes the same `CompositionState` shape that P2 unit tests fix in place).
- P3 depends on P2b (consumes `generate_pipeline_dict` as the canonical input to the widened lifecycle walker; the per-implementer DB-vs-YAML-shape caveat that P3 previously carried is retired by P2b's adapter).
- P4 depends on P3 (validation parity is meaningful only when the runtime side is fail-closed; otherwise composer-green / runtime-red is the very Shape 9 footgun this work closes).
- P5 depends on P4 (the MCP tool's authorship path goes through composer pre-validation, which only recognises the marker after P4).
- P6 (follow-up) depends on P5 (UI surfaces the same MCP-tool-authored state).

## Done conditions across all phases

The whole work closes when:

1. P1–P5 PRs are merged. P6 is filed but not blocking the RC5 ship gate.
2. CI is green on RC5-UX (or successor) including `enforce_tier_model.py check` and `enforce_freeze_guards.py`.
3. Shape 9 (sub-pins A, B, C) passes in `tests/integration/pipeline/test_composer_runtime_agreement.py`.
4. The lifecycle-pinning round-trip integration test passes (a referenced blob cannot be GC'd while the run is pending/running).
5. The hash-determinism property test passes (audit-row hash equals re-derived hash from `BlobServiceImpl.read_blob_content`).
6. ADR-021 is committed, with VAL-data citations for every numeric cap in spec §1.4.
7. The OTel counter post-conditions hold across the property-test campaign (`composer.blob_inline.hash_mismatch_total == 0`, `composer.blob_inline.audit_row_tier1_violation_total == 0`).
8. P2b PR is merged, retiring `elspeth-be405bac87` (canonical `composition_state` adapter shipped as `generate_pipeline_dict`).

VAL — "the operator can actually inline an LLM system prompt and verify the audit trail" — is owned by an operator-acceptance ticket filed alongside P5 closure.

## Filed-during-implementation follow-ups

- **F-1**: file Filigree ticket for the P6 frontend recovery surface (SecretsPanel-equivalent UI). Cite ID in the P5 PR description.
- **F-2**: file Filigree ticket for `blob_inline_resolutions` retention CLI extension (should the CLI's `purge --retention-days` cover this table, or does Tier-1 audit retention diverge from payload-store retention?). Cite ID in the P3 PR description.
- **F-3**: file Filigree ticket for OpenTelemetry alert thresholds (`composer.blob_inline.hash_mismatch_total`, `composer.blob_inline.audit_row_tier1_violation_total`). Cite ID in the P3 PR description.

## Ground rules common to every phase

- **TDD throughout.** Each phase's tasks follow Red → Green → Refactor → Commit. Plans show the failing test first, the minimal implementation second, and the commit message last.
- **Real databases in tests.** §8 of the spec forbids mocking `BlobServiceImpl.read_blob_content` or any `_insert_*` helper. Schema tests run against in-memory SQLite (`sqlalchemy.create_engine("sqlite:///:memory:")` + `metadata.create_all()`); integration tests use the project's existing testcontainer infrastructure where applicable.
- **Path conventions.** `web/...`, `core/...`, `contracts/...` are shorthand for `src/elspeth/web/...`, `src/elspeth/core/...`, `src/elspeth/contracts/...`. Plans use the full path in `Files:` blocks but may use the shorthand in prose.
- **No defensive programming.** No `getattr`, `hasattr`, `.get()` for typed-dataclass field access. Tier-1 audit data crashes on anomaly. Tier-3 boundaries (LLM responses, route inputs) coerce/quarantine; the resolver's substitute-bytes step records violations via `BlobContentResolutionError`.
- **No legacy code.** Pre-release: direct cutover, no compatibility shims. P5 updates `set_source_from_blob` to emit explicit `mode` in the same PR; the recognition function rejects mode-less markers as malformed.
- **No new ref forms without ADR amendment.** ADR-021 (P1) carries this closure rule. Future late-binding needs widen the existing model or are rejected.
- **Frequent commits.** Each task ends with a commit. Where a task has multiple subtasks, each subtask ends with a commit.

---

## How to use this plan

1. Open the relevant phase file.
2. Inside that phase, work tasks in numerical order.
3. After completing each task's commit step, mark the checkbox.
4. After completing every task in a phase, open the PR.
5. Move to the next phase only after the previous phase is merged.

Each phase plan is self-contained: it cites the spec sections it relies on and lists every file it touches. You should not need to consult earlier-phase plans while executing a later phase, because the previous phase's contributions are now part of the codebase reality.
