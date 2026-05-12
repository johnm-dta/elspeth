# Handover — Composer Guided Mode (Phase 5 complete; Phase 6 onward)

## TL;DR

Phase 5 is **complete and green**. The backend feature set is closed:
auto-drop on solver-exhausted, progressive-disclosure transition prompt on
mode shift, full session-level audit emission coverage. Demo path runs
end-to-end with audit events backing every protocol decision.

Phases 6–10 ship the frontend. Phase 6 is the foundational layer:
TypeScript types mirroring the backend protocol, `apiClient` methods for
`/guided/start` and `/guided/respond`, and a `sessionStore` slice for
guided session state.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode` (36 commits ahead of RC5.2 as of Phase 5 close; top commit `8c0f7527`)
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — Python-version mismatch corrupts `enforce_tier_model.py` results.
- **Package manager:** `uv` only.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` (Phase 6 starts line 3866, ends line 4189)
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (Phase 6 relevant: §6.2 endpoints, §7.1–§7.3 component layout / state management, §11 out-of-scope)

## What's delivered in Phase 5

Seven commits on top of Phase 4 closure (`89286d0d`):

| Commit | Task | What |
|--------|------|------|
| `8c132210` | 5.1 | Auto-drop on solver-exhausted — repair attempt via `solve_chain(repair_context=…)`; on repair-also-fails → `mark_solver_exhausted` + `emit_dropped_to_freeform`; HTTP 200 with terminal |
| `84705d3c` | 5.1 | Spec-§9.1 fix: `guided_dropped_to_freeform` payload field renamed `step_index` → `prev_step` (audit-naming bug pre-existed; Task 5.1 first call site to surface it); `validation_result` always populated on solver_exhausted |
| `49be61ae` | 5.1 | Code-quality polish: `assert` → `raise RuntimeError` (production invariants survive `-O`); `_DATA_ERROR_KEY` constant in `tools.py`; docstring trims |
| `8bfd9f66` | 5.2 | Progressive-disclosure transition prompt: `build_mode_transition_system_prompt`; `_load_freeform_skill` (lru-cached); `transition_consumed` field on `GuidedSession`; gate + flip in `send_message` handler |
| `407cb089` | 5.2 | `Any | None` → `TerminalState | None` at 4 protocol/service sites (TYPE_CHECKING import); recompose handler symmetric gap closure (gate + flip + `guided_session` in `composer_meta`) |
| `21d970d7` | 5.2 | Code-quality polish: hoist lazy imports in `build_messages`; docstring trim; test imports normalised |
| `8c0f7527` | 5.3 | Full-session audit emission tests: 6 tests across recipe-match happy path + auto-drop path; assert spec §9.1 contract |

### Gate state at handover

- **1748 tests pass** across `tests/integration/web/composer/`, `tests/unit/web/composer/`, `tests/unit/web/sessions/`
- **163 guided tests** specifically (up from 141 at Phase 4 close — 22 new across Tasks 5.1-5.3)
- mypy clean on `src/elspeth/web/composer/` and `src/elspeth/web/sessions/routes.py`
- ruff clean
- `enforce_tier_model.py check`: clean (FPs rotated three times this phase — at Task 5.1 polish, Task 5.2 initial, Task 5.2 type/recompose fix)
- `enforce_freeze_guards.py check`: clean

### End-to-end demo capability (post Phase 5)

The wizard now drives:
1. Step 1: Source resolution (CSV / JSON / etc) via `_execute_set_source`
2. Step 2: Sink resolution (JSONL / JSON / CSV / etc) via `_execute_set_output`
3. Step 2.5: Recipe pre-match against (source, sink) — if match, apply recipe + auto-advance to COMPLETED
4. Step 3: LLM chain proposal via `solve_chain` (when no recipe match) → user accept → commit + COMPLETED
5. Step 3 failure modes: chain validation fails → one LLM repair attempt → if repair also fails, auto-drop with `TerminalReason.SOLVER_EXHAUSTED`
6. Manual exit via `control_signal=exit_to_freeform` at any step
7. Progressive disclosure: any subsequent freeform chat turn after any non-None terminal (including COMPLETED) uses the layered guided-skill + transition-message + freeform-skill prompt for ONE turn, then reverts to freeform-only

All seven flows emit the spec §9.1 audit events: `guided_turn_emitted`, `guided_turn_answered`, `guided_step_advanced`, `guided_dropped_to_freeform`.

## What's pending — Phase 6 onward

Phase 6 is the frontend foundation. Read the plan body at lines 3866–4189 but expect drift — the same pattern as Phases 4 and 5.

### Task 6.1 — TypeScript types mirroring the backend protocol (plan line 3872)

Create `src/elspeth/web/frontend/src/types/guided.ts` with TypeScript mirrors of:
- `TurnType` enum (`inspect_and_confirm | single_select | multi_select_with_custom | schema_form | propose_chain | recipe_offer`)
- `Turn`, `TurnResponse`, `ControlSignal` types
- `GuidedSession`, `TerminalState`, `TerminalKind`, `TerminalReason`
- `GuidedRespondResponse`, `GetGuidedResponse`, `GuidedStartResponse` (the HTTP response shapes — read from `routes.py`'s pydantic response models around lines 3863+ to find canonical shapes)

Add a vitest type-assertion test at `src/elspeth/web/frontend/src/types/guided.test.ts`.

**Reality check needed before starting:** the backend's HTTP response shapes are pydantic models (e.g., `GuidedRespondResponse` at routes.py). Open them and verify exact field names + nullability before drafting the TS mirrors. The spec §6.2 example is approximate; the running pydantic models are the canonical contract.

### Task 6.2 — API client methods (plan line 4016)

Add `postGuidedStart(sessionId)` and `postGuidedRespond(sessionId, turnResponse)` methods to the existing frontend `apiClient`. Existing API methods in `apiClient` are the pattern reference. Vitest tests with `msw` (already in use) for mocking the HTTP calls.

### Task 6.3 — `sessionStore` guided slice (plan line 4096)

Add a `GuidedSlice` to the existing zustand store:
```typescript
interface GuidedSlice {
  guidedSession: GuidedSession | null;
  startGuided: (sessionId: string) => Promise<void>;
  respondGuided: (turnResponse: TurnResponse) => Promise<void>;
  exitToFreeform: (reason: ExitReason) => Promise<void>;
}
```

**No optimistic updates** — server is authoritative. Every action posts to the corresponding endpoint and replaces the local `guidedSession` with the server response.

### Phase 6 exit criterion (plan line 3870)

> "Vitest store-slice tests pass; types compile cleanly; manual smoke against the running backend (curl the start endpoint, assert the store updates)."

The store-slice tests are unit-level (no real HTTP). Manual smoke is a one-off check against the running `elspeth.foundryside.dev` staging service or a locally-running `elspeth-web` (per project memory `project_staging_deployment.md`).

## Plan-vs-reality gotchas observed in Phase 5 (use these to gut-check Phase 6 plan body)

The same drift patterns from Phase 4 continued into Phase 5. Phase 6 will likely show:

| Plan says | Reality (probable) |
|-----------|---------------------|
| `chaosllm_stub` fixture | Does not exist (Phases 4+5 confirmed). Frontend tests will be vitest + msw, not chaosllm. |
| Endpoint module paths in `service.py` | Endpoints live in `routes.py`; `service.py` carries the LLM-orchestration logic. |
| Lazy imports with `# avoid cycle` comments | Banned per CLAUDE.md "Shifting the Burden" — check for actual cycles first; almost never present. Verified in Phase 5 (zero new lazy imports needed in the code we wrote). |
| `audit_recorder.guided_events()` | Does not exist. Audit-event tests use `service.get_messages(...)` + `role=="tool"` + filter by `invocation.tool_name`. Pattern established in `test_auto_drop.py` and `test_audit_emission.py`. |
| Type annotations using `Any | None` for protocol parameters | Anti-pattern — silently disables mypy checking. Use `TerminalState | None` (or whatever concrete type) with a TYPE_CHECKING import. `from __future__ import annotations` is already in effect across the composer codebase. |
| Plan body specifies one file but the change actually spans multiple files | Common — Phase 5's Task 5.2 brief said "service.py" but the work spanned `composer/prompts.py`, `composer/guided/prompts.py`, `composer/state_machine.py`, `composer/protocol.py`, `composer/service.py`, `routes.py` plus two test files. Phase 6's frontend work will likely span types, apiClient, sessionStore, and component files. |

## Open follow-ups carried out of Phase 5

These are flagged for future work — NOT blocking Phase 6 start:

1. **`_replace_dc` 4-site lazy-import normalisation** — `routes.py` has 4 instances of `from dataclasses import replace as _replace_dc` inside function bodies. The pre-existing pattern was kept (out of Task 5.2 scope per CLAUDE.md "don't refactor beyond what the task requires"). One-pass cleanup recommended in a future hygiene ticket.
2. **`ValidationSummary.to_dict()` consolidation** — three call sites construct `{"is_valid": ..., "errors": [e.to_dict() for e in ...]}` independently. Code-quality reviewer recommended adding a method on `ValidationSummary`. Out of Task 5.1 scope (would touch `tools.py` and several call sites); flagged for future cleanup.
3. **`emit_turn_emitted` / `emit_turn_answered` field-name parity audit** — Task 5.1 fixed `step_index` → `prev_step` on `guided_dropped_to_freeform` only. The other two `step_index`-using event types are spec-correct (per spec §9.1 they describe a turn within a step, not a transition), but worth a pass to confirm.
4. **Filigree observation `elspeth-obs-2bbdb1ddba`** — filed by the Task 5.2 implementer about the recompose handler gap, but the gap was fixed in commit `407cb089`. The observation is stale; can be dismissed by the next session via `mcp__filigree__dismiss_observation`.

## Conventions discovered in Phase 5 (follow these)

### Backend dispatcher (`routes.py::_dispatch_guided_respond`)

- Returns `(state, session, next_turn)` tuple.
- For mid-dispatch terminal transitions (e.g., auto-drop), emit the audit event INLINE inside `_dispatch_guided_respond` — the outer handler's directive-fan-out (around routes.py:3960-3989) only handles directives from `step_advance()`, not mid-dispatch terminations.
- Use `mark_solver_exhausted(...)` to construct the terminal + directives, then iterate the directive list and call the matching `emit_*` helper (the only directive type at the auto-drop site is `guided_dropped_to_freeform`, but mirroring the outer-handler pattern keeps the audit-emission shape consistent).

### Audit-emission tests

- Read events via `service.get_messages(session_id, limit=None)` filtered by `role=="tool"`.
- For each tool message, walk `msg.tool_calls` and inspect each `tc["invocation"]`.
- `invocation["tool_name"]` is the discriminator (one of `{guided_turn_emitted, guided_turn_answered, guided_step_advanced, guided_dropped_to_freeform}`).
- `invocation["arguments_canonical"]` is a JSON-encoded payload dict — `json.loads(...)` it to inspect arguments.
- Constants for the four guided event discriminators are typed `frozenset[str]` (see `test_audit_emission.py:37-44`).

### Stubbing LiteLLM

- Patch path is `elspeth.web.composer.guided.chain_solver._litellm_acompletion` for the Step 3 chain-solver path.
- Patch path is `elspeth.web.composer.service._litellm_acompletion` for the freeform composer path.
- Use `unittest.mock.AsyncMock` (NOT `MagicMock` — async functions need `AsyncMock`).
- Fake response shape is attribute-access SimpleNamespace (NOT dict): `SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name=..., arguments=json.dumps({...})))]))])`. The `arguments` field is a JSON-encoded STRING, not a dict.

### Type discipline

- Protocol parameters MUST have concrete types, never `Any | None`. Use TYPE_CHECKING imports when a concrete type would cause a runtime circular dependency. `from __future__ import annotations` is already enabled across the composer codebase, so all annotations are strings at runtime — TYPE_CHECKING imports are sufficient.
- `assert` is for type narrowing only. For runtime invariants that must survive `-O`, use explicit `if … raise RuntimeError(...)`.

### Frozen dataclass schema additions

- Adding a field to a frozen dataclass like `GuidedSession`: add the field with a default (e.g., `transition_consumed: bool = False`), update `to_dict()` to serialize it, update `from_dict()` to read it WITHOUT a backward-compat default (per `project_db_migration_policy.md`: delete the DB on schema change).
- If the dataclass has `freeze_fields(...)` in `__post_init__`, scalar fields don't need to be added to the freeze list. Only container fields (`dict`, `list`, `Mapping`, `Sequence`) require deep-freeze enforcement.

## CI gates (run before declaring any task done)

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
.venv/bin/python -m pytest tests/integration/web/composer/guided/ tests/unit/web/composer/guided/ -v
.venv/bin/python -m pytest tests/unit/web/sessions/ tests/unit/web/composer/ tests/integration/web/composer/ -q
.venv/bin/python -m mypy src/elspeth/web/composer/ src/elspeth/web/sessions/routes.py
.venv/bin/python -m ruff check src/elspeth/web/composer/ src/elspeth/web/sessions/ tests/integration/web/composer/guided/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
```

For frontend Phase 6 work, also:

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode/src/elspeth/web/frontend
npm test -- --run                # vitest
npx tsc --noEmit                  # TypeScript type-check
npx eslint src/                   # frontend linting
```

## First action when you start

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline RC5.2..HEAD | head -10                  # expect 36 commits, top = 8c0f7527
.venv/bin/python -m pytest tests/integration/web/composer/guided/ tests/unit/web/composer/guided/ tests/unit/web/sessions/ -q   # expect 1742 passed
```

If the baseline isn't clean (1748 across the broader suite, 163 in `tests/integration/web/composer/guided/`), **stop and investigate** before starting Phase 6 — the handover may be stale or someone else may have touched the branch.

Then: read Phase 6 in the plan (lines 3866–4189), spot-check the plan-vs-reality gotchas table above (especially the canonical pydantic response model shapes in `routes.py`), and dispatch the Task 6.1 implementer with a brief that overrides the plan body where it diverges from reality.

Expect to dispatch 3 implementer subagents (one per Phase 6 task) plus reality-verification grep work between each.

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** No Alembic; no migration scripts; no `from_dict` backward-compat defaults.
- **Default to worktree.** You're already in one; stay here.
- **No git stash.** Commit work to a branch if you need to preserve it.
- **Don't recommend `slog` as a diagnostic channel.** Audit/telemetry first per CLAUDE.md primacy order.
- **No calendar shipping commitments.** ELSPETH ships work-until-done; ADR SLAs are governance devices, not deadlines.
- **Correctness beats performance always.** Don't frame correctness work as a perf tradeoff.
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
- **Type annotations must be concrete.** `Any | None` for protocol parameters is forbidden — silently disables mypy. Use TYPE_CHECKING imports for forward references.
