# Prompt — Composer Progress Persistence Phase 3 plan authoring

> Drop this whole document into a fresh Claude Code session. The agent has zero prior context; this prompt is its only briefing.

---

Brief: Phase 3 (composer compose-loop persistence + tool-call cap + audit-grade transcript endpoint) — write the implementation plan from scratch.

## What you're being asked to produce

**One artefact:** `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-3-compose-loop.md` — a fresh implementation plan, structurally analogous to the rev-5 Phase 2 plan at `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.md`. After you produce it, surface to the operator for a four-reviewer plan-review pass (`/review-plan`) before any code work begins. Do NOT start implementation. Do NOT run `gh pr create`.

The bar for the rewrite is APPROVED or APPROVED_WITH_WARNINGS on the plan-review pass. CHANGES_REQUESTED means another iteration. Phase 2's plan-review JSON at `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json` is your reference template for what "good plan-review hygiene" looks like — read it before writing.

## Project context

ELSPETH is a domain-agnostic framework for auditable Sense/Decide/Act (SDA) pipelines. Auditability is the headline property: every decision must be traceable to source data, configuration, and code version, and the Landscape audit DB is the legal record. CLAUDE.md at `/home/john/elspeth/.worktrees/composer-progress-1a/CLAUDE.md` is load-bearing — read it in full before drafting. Key invariants you will be subject to:

- **Three-tier trust model.** Tier-1 audit DB writes must be pristine; bad data crashes immediately. Tier-3 LLM input (tool-call arguments, response content) is zero-trust and must not crash the audit path.
- **Plugins are system code, not user extensions.** A defective plugin that silently produces wrong results is worse than a crash because it pollutes the audit trail.
- **No legacy code, no compatibility shims** — WE HAVE NO USERS YET; deferring breaking changes is forbidden.
- **Defensive programming forbidden.** `hasattr` is unconditionally banned; `.get()` / `getattr(..., default)` / `isinstance(...)` for suppression is forbidden. Use direct typed-attribute access; assert preconditions; let crashes be informative.
- **Frozen dataclasses with container fields** must call `freeze_fields(self, ...)` from `elspeth.contracts.freeze` in `__post_init__`. CI enforces.
- **4-layer architecture** (L0 contracts → L1 core → L2 engine → L3 plugins+composer+web+mcp+tui). Imports flow downward only. CI enforces.
- **Logging primacy.** Audit (Landscape) first, telemetry (OTel) second. Slog only for audit-system / telemetry-system failures. NEVER recommend slog as a diagnostic channel for pipeline activity.
- **No git stash, no `--no-verify` on code commits** (markdown-only commits may use `--no-verify`).

The composer (`src/elspeth/web/composer/`) is the LLM-driven pipeline-authoring tool. It runs an LLM via LiteLLM, exposes pipeline-construction tools through six function-pointer dispatch dicts at `src/elspeth/web/composer/tools.py:5250–5314`, and is destined to persist composition turns to a sessions DB for resumption and audit. The compose loop at `src/elspeth/web/composer/service.py:573-onward` is the recursive call site that takes a user goal, drives the LLM, dispatches tools, and (after Phase 3 lands) persists each turn.

## Where Phase 3 sits in the four-phase plan

Spec `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` (revision 5) splits delivery across four phases. The structural reading you need:

| Phase | Status | What it ships |
|---|---|---|
| **Phase 1A / 1B / 1C** | **Landed** on the umbrella branch (HEAD `f5115fd5` or later) | `chat_messages` schema with `tool_call_id`, `parent_assistant_id`, `sequence_no`, `writer_principal`; `composition_states` with `provenance` discriminator; `audit_access_log` table; `SessionsService.persist_compose_turn` sync primitive at `web/sessions/service.py`; `persist_compose_turn_async` async dispatcher; `AuditOutcome` dataclass at `src/elspeth/contracts/...`. Read `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1A-...md` and `...phase-1B-...md` and `...phase-1C-...md` for what's actually on disk. |
| **Phase 2** | Plan committed (commits `23509c87` spec rev-5 + `7338e4e2` plan rewrite); not yet implemented; `/review-plan` pending | Manifest-keyed redaction primitives (`MANIFEST: Mapping[str, ToolRedaction]`); `Sensitive[T]` annotations; promoted ~6–8 sensitive-touching tools to Pydantic argument models with `Model.model_validate` dispatch validation; declarative manifest entries for the rest; shared traversal iterator; four-assertion adequacy guard; policy-hash snapshot; label-gate CI step. |
| **Phase 3** | **No plan exists. THIS is what you are writing.** | Compose-loop modifications: per-turn tool-call cap (RSK-13); accumulate tool outcomes in async-land; redact via Phase 2's walker; dispatch `persist_compose_turn_async` once per turn; raise `ComposerPluginCrashError` AFTER audit completes; honour `AuditOutcome.unwind_audit_failed`; let `AuditIntegrityError` propagate (Tier-1). Add `failed_turn` field to 422/500 response bodies in `web/sessions/routes.py`. Extend `GET /api/sessions/{sid}/messages` with `?include_tool_rows=true&since=...` parameter and the audit-grade access-log emission per spec §6.3. |
| **Phase 4** | Plan exists at `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-4-frontend.md` | Frontend recovery panel. Reads Phase 3's `failed_turn` JSON shape and the `?include_tool_rows=true` endpoint. |

**Phase 3 is the structural keystone.** Without it, Phase 2's redaction layer has no callers, Phase 4's recovery panel has nothing to render, and Phase 1's persistence primitive sits unwired. The umbrella branch becomes runnable only when Phase 3 lands.

## Phase 3's contract on both sides — these are pinned

You do NOT get to redesign these. Phase 1 has shipped them; Phase 2 ships them; Phase 4 consumes them. Phase 3 must connect them as specified.

### Below (Phase 1 + Phase 2 → Phase 3)

- **`SessionsService.persist_compose_turn_async`**: signature, semantics, cancellation discipline. Read `src/elspeth/web/sessions/service.py` for the actual signature. Spec §5.2.2 / §5.7.1 / §5.7.2 cover the contract. Spec §5.2.4 cancellation table is load-bearing.
- **`AuditOutcome`** dataclass — two valid shapes per spec §13 glossary entry: (1) success with `assistant_id` populated and `unwind_audit_failed=False`; (2) tool failed AND audit unwind failed with `assistant_id=None` and `unwind_audit_failed=True`. There is NO `tier1_violation` field — Tier-1 audit-write failures raise `AuditIntegrityError` directly inside the sync worker.
- **`AuditIntegrityError`** — registered in `TIER_1_ERRORS`; `except Exception` blocks cannot swallow it. Phase 3 must let it propagate. Spec §4.5 / §5.2.2.
- **`redact_tool_call_arguments` / `redact_tool_call_response`** (Phase 2) — accept already-decoded `dict[str, Any]` (M3: single parse) plus a `RedactionTelemetry` instance; return the redacted dict. Spec §4.2.6 boundary table. **NB: Phase 2 is not yet implemented; you are writing a plan that depends on Phase 2's primitives existing. The Phase 3 plan's preflight section must require Phase 2 landed and gate-green.**
- **Existing ARG_ERROR pattern** at `src/elspeth/web/composer/service.py:1836–1870` — JSON decode, non-dict, missing-required-paths. Phase 3 preserves this exactly; the new audit-row insertion happens in addition, not in place of.
- **Existing `except` clauses around `execute_tool()`** per spec §2 Context: `except ToolArgumentError` (line ~867 — Tier-3 boundary), `except (AssertionError, MemoryError, RecursionError, SystemError)` (line ~907 — fail-fast), `except Exception as tool_exc` (line ~942 — converts to `ComposerPluginCrashError.capture(...)`). Phase 3 preserves the wrap-and-raise discipline; the audit-row insertion happens BEFORE the capture-and-raise, not in place of it.

### Alongside (Phase 2 ↔ Phase 3)

- **`MANIFEST[tool_name]`** lookup is the redaction dispatch root. Phase 3 calls `redact_tool_call_arguments(tool_name, decoded_args, telemetry=...)` once per tool call before dispatching to the persistence primitive.
- **Unknown tool name routing**: spec rev-5 §4.2.6 + §5.7.5 establish that an LLM-supplied unknown tool name is Tier-3 input. The dispatcher's existing failure-`ToolResult` path (fall-through at `tools.py:5481`: `return _failure_result(state, f"Unknown tool: {tool_name}")`) is the correct shape; Phase 3 routes the failure through the same audit-record path the existing ARG_ERROR sites use. The walker is NOT invoked for unknown tool names because dispatch never succeeds. There is no `MissingToolError` exception class.

### Above (Phase 3 → Phase 4)

The `failed_turn` JSON shape Phase 4 already consumes:

```typescript
// src/elspeth/web/frontend/src/components/recovery/recoveryTypes.ts (Phase 4)
export interface FailedTurn {
  assistant_message_id: string;
  tool_calls_attempted: number;
  tool_responses_persisted: number;
  transcript_url: string;
}
```

The transcript URL query-parameter shape Phase 4 already consumes: `?include_tool_rows=true` plus an optional `since=` cursor. Example URL from Phase 4's tests: `/api/sessions/s/messages?since=u_1&include_tool_rows=true`.

Phase 3's done-when criterion **must** include: the four `failed_turn` fields are populated correctly on every 422/500 response from the three failure-path route helpers; the `?include_tool_rows=true` endpoint returns the audit-grade transcript with the `audit_access_log` row written before the response body returns.

Read `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-4-frontend.md` to confirm — Phase 4's tests are the load-bearing definition of what Phase 3 must produce.

## Branch state

- Worktree: `/home/john/elspeth/.worktrees/composer-progress-1a`
- Branch: `feat/composer-progress-persistence-1a`
- HEAD at time of writing this prompt: `7338e4e2` (Phase 2 plan rewrite). Re-confirm with `git log -1`.
- Pre-Phase-3 gate: GREEN at the time of writing this prompt. Confirm before drafting:
  - `.venv/bin/python -m pytest tests/unit -q` passes
  - `.venv/bin/python -m pytest tests/integration -q -m "not testcontainer"` passes
  - `.venv/bin/python -m mypy src/` clean
  - `.venv/bin/python -m ruff check src/` clean
  - `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` clean
  - `.venv/bin/python scripts/cicd/enforce_freeze_guards.py` clean
- venv: `.venv/bin/python` is Python 3.13 (memory: `project_tier_model_python_version` — version skew triggers ~300 spurious tier-model false positives).
- No git stash. No `--no-verify` on code commits (markdown-only commits may use `--no-verify`).

## Spec sections that drive Phase 3

Read in this order, in full:

1. **§1.4 Quantified NFRs** — per-turn DB write overhead bounds; redacted-transcript fidelity NFR; INV-AUDIT-AHEAD invariant rate; tool-call cap; tier1_violation SLO. Phase 3's done-when references several of these.
2. **§2 Context — What Already Exists** — particularly the "Open" rows for `Only final assistant text is persisted` and `No tool-result rows in chat history`. Phase 3 closes these. Also the "Existing exception-handling around `execute_tool()`" subsection — Phase 3 must preserve every clause.
3. **§3 Approach Decisions (mini-ADRs)** — the "Transaction primitive shape" and "Atomicity grain" rows. Phase 3 inherits per-turn atomicity (one transaction per assistant turn).
4. **§4.1.1** `chat_messages` schema (read-only — Phase 1 owns it; Phase 3 writes rows that conform). Particularly the constraint table (partial unique index on `tool_call_id`, `writer_principal` CHECK, `composition_state_id` FK semantics).
5. **§4.1.2** `composition_states.provenance` discriminator. Phase 3 writes rows with `provenance='tool_call'`, `provenance='convergence_persist'`, `provenance='plugin_crash_persist'`, `provenance='preflight_persist'`. The route helpers map failure type → provenance value.
6. **§4.5** IntegrityError + OperationalError disposition table. Phase 3's per-turn write is the call site that surfaces these.
7. **§5.2 Insertion sites in the compose loop** — **the load-bearing section**. §5.2.1 Loop shape (single-sync-block-per-turn) is the structural pattern Phase 3 implements. §5.2.2 the sync write function. §5.2.3 What this design eliminates. §5.2.4 Cancellation semantics table. §5.2.5 Why per-turn (not per-tool-row) atomicity.
8. **§5.3 Bidirectional audit-ahead-of-state invariant (INV-AUDIT-AHEAD)** — the load-bearing correctness invariant. Phase 3 must satisfy both the forward direction (chat_messages may be ahead of composition_states, never behind) and the backward direction (composition_states must not be ahead of chat_messages for `provenance='tool_call'` rows). §8.3 property test verifies; Phase 3's done-when must require the property test green.
9. **§5.4 `partial_state` redaction symmetry** — partial-state writes go through the redaction layer too. Spec §5.4 + §4.7 cover.
10. **§5.5 Failure mode interaction** — table of failure-mode × counter behaviour. Phase 3's wiring exhibits each row.
11. **§5.6 Atomicity grain — per turn, not per tool row.**
12. **§5.7.5 `MANIFEST` lookup and redaction-layer integration (rev 5)** — the dispatch shape for the walker. Phase 3 calls the walker exactly here.
13. **§6 HTTP response shape** — read in full. Particularly §6.1 `failed_turn` field shape (must match Phase 4's TypeScript interface), §6.2 the three failure-path route helpers in `web/sessions/routes.py`, §6.3 the `?include_tool_rows=true` parameter and the audit-grade access-log discipline.
14. **§8.2 Backend integration test scenarios** — the CL-PP-* matrix. CL-PP-1 through CL-PP-13 are integration test cases Phase 3 must add; spec maps them to the new audit-row insertion paths.
15. **§8.3 Backend property test (bidirectional INV-AUDIT-AHEAD)** — §8.3.1 strategy contracts (`@example` decorators, cancellation arrival times); §8.3.2 post-conditions (forward direction, backward direction, OTel counter assertions).
16. **§8.5 Verification scope (VER) — explicit VER/VAL boundary** — what Phase 3 verifies vs what is owned by other tickets.
17. **§8.6 Test path integrity — explicit composer rule** — Phase 3 tests must use the production code paths (no mocked DB, no production-bypass shortcut).
18. **§9 Risks and Mitigations** — particularly RSK-04, RSK-07, RSK-10, RSK-11, RSK-12, RSK-13, RSK-15, RSK-17. Phase 3 mitigations live in the loop body.
19. **§11 Phase 3 scope and done-when** — the canonical statement of what Phase 3 ships and when it lands.
20. **§12.1 Revision 4 reviewer-finding traceability** + **§12.2 Revision 5 reviewer-finding traceability** — for plan-review hygiene; the Phase 3 plan should reference §12.x findings the loop changes touch (particularly C-1, C-2, C-3, C-5, F-1, F-2, F-4, F-7).

Spec total length: ~3608 lines. You will read most of it.

## Required structure of the Phase 3 plan

Mirror the rev-5 Phase 2 plan at `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.md`. Sections:

1. **Header** — goal, architecture (Phase 3 is L3 except for the route-helper changes which are also L3 in `web/sessions/routes.py`), tech stack, spec sections cited, statement of supersession (none — first plan for Phase 3).
2. **Preflight** — gate state required to begin. Must include: Phase 1A/1B/1C landed; **Phase 2 implemented and gate-green** (Phase 3 cannot start before Phase 2 lands); spec rev-5 on disk; pre-Phase-3 gate green.
3. **File Structure** — files to create / modify / NOT touched.
4. **Tasks** — TDD-shaped, one mechanism per task, dependencies between tasks called out explicitly. Use checkbox steps. Each task body cites the spec section that drives it.
5. **Summary** — what Phase 3 ships and what it deliberately does not.
6. **Spec / Depends on / Out of scope (later phases) / Test plan / Phase 3 Done When** — same shape as Phase 2.
7. **Appendix A — plan-review finding closure** — preemptive table mapping anticipated reviewer concerns to closing tasks. (Phase 3's plan-review hasn't run yet, but you can preempt the Phase 2 review's recurring patterns: walker-vs-guard divergence type concerns, Tier-3 vs Tier-1 boundary discipline, calendar-keyed vs content-keyed gates, label vs CODEOWNERS controls.)

## Tasks Phase 3 must include — proposed structure (refine as you draft)

**Foundational** (build the per-turn data structures before the loop changes):

1. `_TurnAccumulator` (or equivalent) async-land data structure that collects tool outcomes during a turn. Holds the ordered list of `(tool_call, decoded_arguments, redacted_arguments, dispatch_result, redacted_response, pre_version, post_version, status, error_class)` tuples. Pure data; no I/O.
2. Per-turn tool-call cap enforcement (RSK-13). New `ComposerConvergenceError(reason="tool_call_cap_exceeded")` reason code; CL-PP-12 covers; route helper handles.
3. Tool-call iteration shape: refactor the existing `for tool_call in assistant_message.tool_calls:` loop body into a function that yields the per-turn outcomes without persisting (Step 1 of §5.2.1). This is purely a refactor — no behavioural change yet.

**Walker integration** (Phase 2 walker called from the loop):

4. After the for-loop produces the outcomes list, call `redact_tool_call_arguments` for each tool call's decoded arguments and `redact_tool_call_response` for each tool's response. Build the `redacted_assistant_tool_calls` and `redacted_tool_rows` structures per spec §5.2.1 Step 2.
5. Wire the `RedactionTelemetry` instance through the loop. The instance is built once at compose-loop entry; pass it through to every walker call. Telemetry counter assertions verify §1.4 NFR.

**Persistence dispatch**:

6. Single dispatch site: `await persist_compose_turn_async(redacted_assistant_tool_calls, redacted_tool_rows, ...)`. Spec §5.2.1 Step 2 + §5.2.2 single-sync-block discipline. The async dispatcher is shielded from caller cancellation per Phase 1's contract.
7. Handle `AuditOutcome` per §5.2.1 Step 3. Two valid shapes; `unwind_audit_failed=True` increments the unwind-path counter; in the success shape, the `assistant_id` is recorded for INV-AUDIT-AHEAD.
8. Let `AuditIntegrityError` propagate. Tier-1 violation; registered in `TIER_1_ERRORS`; `except Exception` cannot swallow.
9. After audit completes, raise `ComposerPluginCrashError` if a tool exception was captured during the loop. Wrap-and-raise discipline preserved per spec §2 Context.

**Route-helper extensions** (`web/sessions/routes.py`):

10. Extend `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure` to populate `failed_turn` on 422/500 response bodies. Field shape mirrors Phase 4's TypeScript interface exactly.
11. Add or extend the failure-path partial-state persistence sites with `provenance='convergence_persist'` / `'plugin_crash_persist'` / `'preflight_persist'` per §4.1.2.

**Audit-grade transcript endpoint**:

12. Extend `GET /api/sessions/{sid}/messages` with `?include_tool_rows=true&since=...` parameters. Default behaviour preserved (the live chat panel does not regress). When `include_tool_rows=true` is set, the response includes `role='tool'` rows; the route helper writes an `audit_access_log` row before returning. Spec §6.3.
13. RSK-17 mitigation: integration test asserts the `audit_access_log` row exists for every audit-grade query.

**Verification surface**:

14. CL-PP-1 through CL-PP-13 integration scenarios per §8.2. Each is a separate test; spec subsection names each scenario.
15. Property test `tests/integration/web/test_inv_audit_ahead_property.py` per §8.3. Hypothesis strategies; both INV-AUDIT-AHEAD directions; OTel counter post-conditions; cancellation arrival times.
16. Latency sanity bound test per §1.4 (p95 ≤ 250 ms with N ≤ 8 tool calls per assistant turn).
17. Backward-direction post-condition test (`tests/integration/web/test_inv_audit_ahead_backward.py`) — schema-level introspection per §1.4 / §8.3.2.

**Final**:

18. Final gate run.
19. Operator decides PR-open. No `gh pr create`.

That's ~19 tasks. Subdivide further if any task body grows beyond ~150 lines of plan text. Each task follows TDD shape: write failing test, run, implement, run pass, gate slice, commit.

## Hard constraints

- CLAUDE.md at `/home/john/elspeth/.worktrees/composer-progress-1a/CLAUDE.md` — load-bearing. Re-read before writing.
- No git stash, no `--no-verify` on code commits.
- No PR-open without operator confirmation.
- No defensive `.get()` / `getattr(..., default)` / `hasattr` / `isinstance(...)` for suppression. Direct typed-attribute access; assert preconditions.
- Frozen dataclasses with container fields must call `freeze_fields(self, ...)` from `elspeth.contracts.freeze`.
- 4-layer architecture — Phase 3 work is L3 (composer + sessions/routes); imports flow downward only.
- Tier-1 audit invariant — redaction failures and audit-write failures crash, not silently degrade.
- Phase 3 cannot start before Phase 2 lands and is gate-green. State this in preflight.
- Phase 3 must produce the exact `failed_turn` JSON shape that Phase 4 consumes; deviating breaks the frontend.
- Spec rev-5 §5.7.5 — no `lookup_tool_class`, no `MissingToolError`. The walker dispatches via `MANIFEST[tool_name]`. An LLM-supplied unknown tool name is Tier-3 and routes through the dispatcher's existing failure-`ToolResult` path.

## Decisions you may need to surface to the operator before drafting

The Phase 2 brief I worked from contained two structural decisions that needed operator input before drafting (the Pydantic-promotion scope and the CODEOWNERS-team-existence finding). The Phase 3 brief may contain similar latent decisions; surface them via `AskUserQuestion` BEFORE drafting if they arise. Anticipated candidates:

1. **Scope of `failed_turn.transcript_url` content.** Phase 4 expects the URL to point to `/api/sessions/{sid}/messages?since={cursor}&include_tool_rows=true`. Question: is the cursor the user-message ID that started the failed turn, or the assistant message ID, or something else? Spec §6 may pin this; if not, surface.
2. **`provenance='tool_call'` write timing.** Spec §5.2 has the per-turn single-sync-block. But within that sync block, what is the order of writes — assistant row first, then tool rows, then state rows; or one transaction with all rows? Spec §5.2.2 should pin this; if not clear, surface before drafting the persistence-dispatch task.
3. **Latency sanity bound infra assumption.** §1.4 NFR p95 ≤ 250 ms is "on standard infra" but our CI runners may be noisier. Phase 3's done-when needs a concrete CI environment commitment or an explicit "tracking observation, not a build break" framing per the §1.4 split.
4. **Cancellation semantics edge case.** §5.2.4 has a cancellation table; verify it covers every arrival time (before LLM call, during LLM call, between tool calls, during a tool call, between tool call and audit, during audit, after audit). Any gap is an operator-level decision before drafting.

## Reference materials

- `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` — spec rev-5 (the load-bearing input)
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md` — what Phase 1A landed (schema)
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1B-compose-turn-primitive-audit-semantics.md` — what Phase 1B landed (`persist_compose_turn` primitive)
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1C-postgresql-ci-operational-proof.md` — what Phase 1C landed (Postgres CI lane)
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.md` — Phase 2 plan (rev-5 rewrite). Phase 3 depends on Phase 2's primitives.
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json` — Phase 2 plan-review verdict. Read for tone, format, and reviewer convergence patterns.
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-4-frontend.md` — Phase 4 plan. Read the `failed_turn` interface and the transcript-URL examples; Phase 3 must produce these.
- `src/elspeth/web/composer/service.py:573-onward` — the existing `_compose_loop`. Phase 3 modifies this body in-place.
- `src/elspeth/web/composer/service.py:1836-1870` — existing ARG_ERROR pattern. Preserve.
- `src/elspeth/web/composer/tools.py:5250-5314` — six function-pointer dispatch dicts. Read-only for Phase 3.
- `src/elspeth/web/composer/tools.py:5481` — fall-through "Unknown tool" failure-`ToolResult` shape. Spec §5.7.5 references.
- `src/elspeth/web/sessions/service.py` — `SessionsService.persist_compose_turn` and `persist_compose_turn_async`. Read the actual signatures; spec is correct as of rev-5 but signatures evolved during Phase 1.
- `src/elspeth/web/sessions/routes.py` — three failure-path route helpers (`_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`). Phase 3 extends.
- `src/elspeth/contracts/composer_audit.py` — `ComposerToolInvocation`, `ComposerToolStatus`, `ComposerToolRecorder`. Phase 3 produces invocations into the recorder.
- `src/elspeth/contracts/freeze.py` — `freeze_fields`, `deep_freeze`. For any new dataclass.
- `~/.claude/projects/-home-john-elspeth/memory/MEMORY.md` — operator memory. Particularly `feedback_no_slog_recommendations`, `feedback_no_git_clone_for_isolation`, `feedback_correctness_beats_performance`, `feedback_no_calendar_shipping_commitments`, `project_phase1c_implementation_complete`, `project_phase2_plan_review_verdict`. Read all of these.

## Success criterion

The drafted plan should pass a four-reviewer plan-review pass (`/review-plan`) at APPROVED or APPROVED_WITH_WARNINGS. CHANGES_REQUESTED on the rewrite means another iteration. Approved plan triggers an operator decision on whether to begin Phase 3 implementation.

After producing the plan, surface the deliverable to the operator and trigger `/review-plan`. Do not start any code work.

---

End of brief.
