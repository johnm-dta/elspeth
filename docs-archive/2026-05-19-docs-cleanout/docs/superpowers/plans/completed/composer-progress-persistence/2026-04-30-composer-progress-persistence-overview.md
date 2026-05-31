# Composer Progress Persistence — Plan Overview

> **For agentic workers:** This is a multi-phase plan. Each phase is a separate plan file; execute them sequentially. REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each phase plan.

**Spec:** `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` (revision 5, 3608 lines; manifest-keyed redaction). The Phase 1 plan contains reviewed corrections that supersede several stale spec snippets; Phase 1 must either amend the spec in-repo or mark those sections explicitly superseded before Phase 2 or Phase 3 begins.

**Goal:** When a long-running compose request fails, persist the LLM's accumulated tool-call breadcrumbs and partial draft to the database, expose them via an HTTP endpoint, and render a recovery panel that lets the user pick the work back up.

**Architecture:** Four sequential phases. Phase 1 establishes the data layer and the synchronous transaction primitive. Phase 2 adds the manifest-keyed redaction primitive (`ToolRedaction` dataclass + module-level `MANIFEST`), promotes sensitive-touching tools to type-driven `argument_model` entries via `Sensitive[T]` annotations, and retains the declarative `ToolRedactionPolicy` shape for remaining tools. Phase 3 wires the compose loop through the new persistence boundary, including the per-turn tool-call cap. Phase 4 first normalizes Phase 3's composer-service implementation shape, then builds the frontend recovery panel. Each phase is one PR.

**Tech Stack:** Python 3.12 and 3.13 (matching supported CI), SQLAlchemy 2.x (sync `Engine`), Pydantic 2.x, FastAPI, structlog, OpenTelemetry, Hypothesis (property tests), pytest with chaos fixtures, React/TypeScript for the frontend.

---

## Phase plans

| Phase | File | What it delivers |
|---|---|---|
| 1 | `2026-04-30-composer-progress-persistence-phase-1-data-layer.md` | Schema additions (`writer_principal`, `provenance`, `audit_access_log`), private concrete `SessionServiceImpl.persist_compose_turn` sync primitive plus protocol-public `SessionServiceProtocol.persist_compose_turn_async` dispatcher, advisory-lock + sequence-reservation helpers, SQLite same-session serialization, route-layer caller migrations, stale-state compare-and-set protection, and DB-level same-session enforcement for tool-row parent links. |
| 2 | `2026-04-30-composer-progress-persistence-phase-2-redaction.md` | `ToolRedaction` manifest dataclass, module-level `MANIFEST` keyed by tool name, `Sensitive[T]` promotion wave for sensitive-touching tools, declarative `ToolRedactionPolicy` + `HandlesNoSensitiveDataReason` for remaining tools, shared traversal iterator, `RedactionTelemetry` Protocol, four-assertion adequacy guard, broadened policy-hash snapshot, label-gate CI step. |
| 3 | `2026-04-30-composer-progress-persistence-phase-3-compose-loop.md` | Compose-loop integration: gather tool outcomes async, dispatch one sync write per turn, enforce the per-turn tool-call cap, surface `failed_turn` + `tool_responses_persisted` on error responses, expose `include_tool_rows=true` with audit-grade access logging. |
| 4 | `2026-04-30-composer-progress-persistence-phase-4-frontend.md` | Phase 4A removes Phase 3's module-tail `ComposerServiceImpl` monkeypatching while preserving behaviour; Phase 4B adds the frontend/client recovery surface: preserve `partial_state` + `failed_turn` in `ApiError`, fetch the audit-grade transcript with `include_tool_rows=true`, render `RecoveryPanel.tsx`, `RecoveryDiff.tsx`, `RecoveryTranscript.tsx`, and wire local-only Apply/Discard through `sessionStore` with a compose-start `compositionState.version` guard and accessibility hooks. |

## Cross-phase dependencies

- Phase 2 depends on Phase 1's schema (the `writer_principal` column is required to insert any chat_messages row).
- Phase 3 depends on Phase 2's redaction primitives (the compose loop uses `redact_tool_call_arguments`, `redact_tool_call_response`, and the module-level `MANIFEST`; the rev-4 `lookup_tool_class` helper is removed per rev-5 §5.7.5).
- Phase 4 depends on Phase 3's response shape (`failed_turn`, `tool_responses_persisted`) and the `include_tool_rows` query parameter. The current backend returns `failed_turn.transcript_url = null`; Phase 4 derives transcript fetches from the active session id instead of requiring a URL field.
- Phase 3 depends on Phase 1's async dispatcher contract: composer routes and services call `await service.persist_compose_turn_async(...)` through `SessionServiceProtocol`, never the concrete `_run_sync` bridge and never the sync primitive directly.
- Phase 3 remains mandatory before anyone claims recovery fidelity. Phase 1 creates the persistence primitive but does not yet make the live compose loop persist assistant/tool rows and composition state atomically.
- Phase 4 begins with Phase 4A cleanup of Phase 3's structural debt: move the `_run_one_turn_for_test`, `_serialize_response_via_walker`, `_state_payload_for_compose_turn_for_test`, and constructor initialization monkeypatches into normal `ComposerServiceImpl` code before adding the frontend panel.
- Each PR description must cite the previous phase's commit as a dependency. Reviewers can merge in order without re-reviewing earlier phases.

## Done conditions across all phases

The whole work closes when:

1. All four phase PRs are merged.
2. CI is green on RC5-UX (or successor) including `enforce_tier_model.py` and `enforce_freeze_guards.py`.
3. All seventeen `CL-PP-*` integration scenarios pass (CL-PP-11 against testcontainer PostgreSQL), and any Docker-enabled testcontainer CI job is included in the aggregate `ci-success` gate.
4. The schema-level backward-direction test (`tests/integration/web/test_inv_audit_ahead_backward.py`) passes.
5. The OTel-counter post-conditions in spec §8.3.2 hold across the property-test campaign, and before Phase 3 ships the Tier-1 audit counters have an alert route, dashboard visibility, and runbook entry.
6. The redaction-policy snapshot file is committed.
7. The staging-deploy runbook documents the session-DB archive/delete/restart recreation procedure. Row-level `DELETE FROM chat_messages` / `DELETE FROM composition_states` is forbidden for this schema change because it cannot add required columns or constraints.
8. The Phase 3 compose-loop PR proves recovery fidelity end-to-end: a failed compose turn persists assistant/tool breadcrumbs and state in one synchronous transaction, and the Phase 4 recovery panel reads that persisted state without relying on hidden chain-of-thought.

VAL — "the user can actually recover from a failure" — is owned by [elspeth-90b4542b63](filigree:elspeth-90b4542b63), the Composer progress persistence feature. That ticket must carry the current release-blocking label, `release:rc5`, so the VER/VAL split remains honest without pointing at a closed issue or nonexistent label.

## Filed-during-implementation follow-ups

- **OQ-1**: file Filigree ticket for `chat_messages` and `audit_access_log` retention CLI extension. Cite ID in the Phase 1 PR description.
- **OQ-3**: file Filigree ticket for `chat_messages` integrity-hash chain (mechanism sketched in spec §10). Cite ID in the Phase 3 PR description.
- **OQ-4**: confirm staging runbook captures the session-DB archive/delete/restart procedure before Phase 1 lands. The older pre-deploy row-DELETE framing is explicitly superseded.

## Ground rules common to every phase

- **TDD throughout.** Each phase's tasks follow Red → Green → Refactor → Commit. Plans show the failing test first, the minimal implementation second, and the commit message last.
- **Real databases in tests.** §8.6 of the spec forbids mocking `SessionServiceImpl.persist_compose_turn` or any `_insert_*` helper. Session persistence tests must use `create_session_engine(..., poolclass=StaticPool)` plus `initialize_session_schema()` so SQLite FK/connect hooks and schema validation match production. Bare `sqlalchemy.create_engine("sqlite:///:memory:")` + `metadata.create_all()` is banned for this surface. CL-PP-11 uses testcontainer PostgreSQL.
- **Path conventions.** `web/...` is shorthand for `src/elspeth/web/...`. Plans use the full path in `Files:` blocks but may use the shorthand in prose.
- **No defensive programming.** No `getattr`, `hasattr`, `.get()` for typed-dataclass field access. Tier-1 audit data crashes on anomaly. Tier-3 boundaries (LLM responses, route inputs) coerce/quarantine; tool-execution boundary records via tool rows.
- **No legacy code.** Pre-release: direct cutover, no compatibility shims. The existing `add_message` signature changes (new required `writer_principal` kwarg); Phase 1 updates every caller in the same PR.
- **Frequent commits.** Each task ends with a commit. Where a task has multiple subtasks, each subtask ends with a commit.

---

## How to use this plan

1. Open the relevant phase file.
2. Inside that phase, work tasks in numerical order.
3. After completing each task's commit step, mark the checkbox.
4. After completing every task in a phase, open the PR.
5. Move to the next phase only after the previous phase is merged.

Each phase plan is self-contained: it cites the spec sections it relies on and lists every file it touches. You should not need to consult earlier-phase plans while executing a later phase, because the previous phase's contributions are now part of the codebase reality.
