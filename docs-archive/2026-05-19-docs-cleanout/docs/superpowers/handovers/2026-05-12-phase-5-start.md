# Handover — Composer Guided Mode (Phase 5 onward)

## TL;DR

You're inheriting a green branch with Phase 4 just landed. Phases 1–4 are complete (protocol + state machine + HTTP endpoints + LLM chain solver + Step 3 wiring). Phases 5–10 remain. **Phase 5 is the backend closer**: three tasks delivering auto-drop on solver-exhausted, progressive-disclosure system prompt on mode transition, and a full-session audit emission test. After Phase 5 the backend is feature-complete and Phases 6+ ship the frontend.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode` (29 commits ahead of RC5.2 as of Phase 4 close)
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — it'll fail tier-model checks because Python versions can mismatch (`project_tier_model_python_version.md`).
- **Package manager:** `uv` (never `pip` directly). Editable install: `uv pip install -e ".[all]" --python /home/john/elspeth/.worktrees/composer-guided-mode/.venv/bin/python`.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` (Phase 5 starts line 3685, ends 3863)
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (relevant: §5.3 manual exit, §5.4 auto-drop, §8.2 progressive disclosure, §9.1 audit events)

## Read these BEFORE writing any code

1. **The plan's Errata block** — lines 15–158 of the plan. Ten corrections (C1–C10) some of which still apply (C3 routes location, C5 module-level frontend functions, C6 sessionStore single atomic guidedTurn — relevant to Phase 6 only).
2. **CLAUDE.md at the repo root.** Especially Three-Tier Trust Model, Plugin Ownership, Defensive Programming Forbidden / Offensive Programming Encouraged, and No-Legacy-Code policy.
3. **Recent project memory** — `MEMORY.md` in your auto-memory dir. Especially `project_phase4_implementation_complete.md` (predecessor handover with architectural notes), `project_phase3_implementation_complete.md`, `project_tier_model_python_version.md`, `feedback_default_is_fix_not_ticket.md`, `feedback_no_scope_dumping.md`, `feedback_correctness_beats_performance.md`.

## What's delivered (Phase 4 — don't redo)

5 commits on top of Phase 3 closure (`795477d2`):

| Commit | Task | What |
|--------|------|------|
| `e0217edc` | 4.1 | Guided skill prompt (80 lines exactly) + `prompts.py` loader with `@lru_cache(maxsize=1)` |
| `94c768ac` | 4.2 | `build_step_3_context_block(*, source, sink, recipe_match) -> str` in `prompts.py` |
| `c0867f58` | 4.3 | `src/elspeth/web/composer/guided/chain_solver.py` — `async def solve_chain(*, source, sink, recipe_match=None) -> ChainProposal` wrapping `_litellm_acompletion` |
| `c87633be` | 4.4 | `handle_step_3_chain_accept` in `steps.py` — atomic commit via `_execute_set_pipeline` with source.on_success rewired to `"chain_in"` |
| `89286d0d` | 4.5 | Step 3 dispatcher wiring in `routes.py` at 3 seams + 33-FP rotation in `web.yaml` |

Gate state at handover:
- **495 tests pass** (136 guided + 359 sessions)
- mypy clean on `src/elspeth/web/composer/`
- ruff clean
- `enforce_tier_model.py check`: clean
- `enforce_freeze_guards.py check`: clean

**Demo path works end-to-end**: CSV source → JSON sink (no recipe match) → LLM-proposed transform chain → user accept → terminal=COMPLETED with rendered YAML.

## What's still pending — Phase 5 scope

Phase 5 has **three tasks**. Read the plan body at lines 3685–3863 but expect drift (see "Plan-vs-reality gotchas" below).

### Task 5.1 — Auto-drop on solver-exhausted (plan lines 3691–3729)

**Trigger:** User accepts a chain proposal at Step 3 (chosen=["accept"]) but `handle_step_3_chain_accept` returns `tool_result.success=False` — the chain failed `_execute_set_pipeline`'s internal validation.

**Current behavior (Phase 4):** `routes.py` raises `HTTPException(400, "Step 3 chain commit failed: ...")`. This is the seam Task 5.1 replaces.

**Phase 5 behavior:**
1. On preview failure, capture the validation error from `tool_result`.
2. Call `solve_chain` again with the error injected into the GUIDED CONTEXT block (a "repair this — previous attempt failed with: …" addendum).
3. Run `handle_step_3_chain_accept` against the repair proposal.
4. If the repair also fails: call `mark_solver_exhausted(session, validation_result=...)` (already exists at `state_machine.py:583`) and emit the `guided_dropped_to_freeform` audit event via `emit_dropped_to_freeform` (already exists at `audit.py`). The session's `terminal` becomes `TerminalState(kind=EXITED_TO_FREEFORM, reason=SOLVER_EXHAUSTED, pipeline_yaml=None)`.
5. Return the terminal in the HTTP response (200, not 4xx — the wizard concluded cleanly via auto-drop).

**Architectural decision the next session must make:** how does the repair attempt inject the validation error into the prompt? Two options:
- **(a)** Extend `solve_chain` signature with `repair_context: str | None = None` and have `prompts.py` build a separate context block. Cleaner separation; touches the chain_solver public surface.
- **(b)** Build a wrapper function in `routes.py` that constructs the repair prompt inline and passes it as an extra system message to `_litellm_acompletion`. Keeps `chain_solver.py` simple but duplicates the prompt-building logic.

My recommendation is **(a)** — chain_solver is the right home for "talk to the LLM about a chain", and adding one optional parameter is cheap. But verify against spec §5.4 before committing; it may specify the prompt structure.

### Task 5.2 — Progressive-disclosure transition prompt (plan lines 3730–3814)

**Trigger:** Any freeform chat message sent after `composition_state.guided_session.terminal is not None` — the user is now in freeform mode but this is their first turn after exit.

**What to build:**
1. `build_mode_transition_system_prompt(*, terminal_reason: str) -> str` in `prompts.py`. Returns the layered prompt: `[guided skill] + [transition message] + [freeform skill]`. Plan body at lines 3776–3792 gives the structure verbatim — the transition message is short, mentions `LIFTED` protocol restrictions, and includes the terminal reason. Spec §8.2 line 515–528 has the canonical text.
2. `_load_freeform_skill()` helper in `prompts.py` — loads `web/composer/skills/pipeline_composer.md` (the existing 1747-line freeform skill). `@lru_cache(maxsize=1)`.
3. Wire into the freeform chat endpoint. The endpoint is in `routes.py` (NOT `service.py` — plan body is wrong about location). When `composition_state.guided_session != None` AND `guided_session.terminal != None` AND we haven't yet emitted the transition prompt for this terminal, swap the freeform skill for `build_mode_transition_system_prompt(...)` for **the next chat turn only**.
4. Track "transition consumed" — the spec says first-turn-after-transition only. Use a flag in the session record OR check message history for the transition marker. The cleanest is to add a `transition_consumed: bool` field to `GuidedSession` (requires schema migration awareness — see `project_db_migration_policy.md`: delete the DB, no Alembic).

**Reality check needed:** find the freeform chat endpoint. Search `routes.py` for `send_message` and `ComposerServiceImpl.compose`. The existing freeform skill is loaded somewhere in that path — that's the swap site.

### Task 5.3 — Phase 5 closure: full audit emission test (plan lines 3816–3860)

**Test only — no implementation.** Two integration tests:
1. **Recipe match happy path** — drive a full session through recipe acceptance, assert the recorder contains `guided_turn_emitted`, `guided_turn_answered`, `guided_step_advanced`, and NOT `guided_dropped_to_freeform`.
2. **Auto-drop path** — drive a session through Step 3 with a chain solver that fails twice (Task 5.1 path), assert `guided_dropped_to_freeform` appears with `drop_reason="solver_exhausted"`.

The plan uses `audit_recorder.guided_events()` — that method may not exist. Use the actual `audit_recorder.invocations` API and filter for `tool_name in {"guided_turn_emitted", "guided_turn_answered", "guided_step_advanced", "guided_dropped_to_freeform"}`. The audit shape is `ComposerToolInvocation` per errata C4 (no separate guided audit primitive).

### Phase 5 exit criterion (plan line 3689)

> "A failing chain-solver path drops to freeform with the partial pipeline carried; a freeform chat turn after transition includes the `[freeform skill]` content + transition message and the LLM emits a non-guided tool call."

The freeform-tool-emission half requires a live LLM (or a careful stub). The stubbed version is sufficient for Phase 5 closure — verify the LLM is given the right system prompt; the LLM's actual response shape isn't load-bearing for closure.

## Plan-vs-reality gotchas observed in Phase 4 (use these to gut-check Phase 5 plan body)

The plan body had substantial drift on every Phase 4 task. Phase 5's body has similar patterns — verify before quoting.

| Plan says | Reality |
|-----------|---------|
| `chaosllm_stub` fixture | **Does not exist.** Use `patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", new_callable=AsyncMock, return_value=fake_response)`. Plan still references `chaosllm_stub` in Phase 5. |
| Routes mount / dispatcher in `service.py` | Dispatcher is `_dispatch_guided_respond` in `src/elspeth/web/sessions/routes.py:~1554` (line may shift; grep for the function name) |
| `update-fingerprints` subcommand of `enforce_tier_model.py` | **Does not exist.** Tool only has `check` and `dump-edges`. FP rotation is manual in `web.yaml` |
| Plain LiteLLM response is dict-shaped | **Wrong.** Attribute access via `_FakeLLMResponse` shape. See `chain_solver.py:_extract_tool_call` for the canonical parser |
| `audit_recorder.guided_events()` | Probably **does not exist.** Use `recorder.invocations` and filter on `tool_name` |
| `_execute_*(state, args)` positional shape | **Wrong order.** Real: `_execute_*(args, state, catalog, data_dir, *, session_engine?, session_id?) -> ToolResult` |
| Lazy import with "# avoid cycle" comment | **Banned** by CLAUDE.md ("Shifting the Burden"). Check for actual cycle first; almost never present |

## Phase 5–specific seams already documented in the code

Task 5.1's seam is at **routes.py STEP_3_TRANSFORMS dispatcher branch**. Find the `if not handler_result.tool_result.success:` block in the new Step 3 ACCEPT branch (added in commit `89286d0d`). That's where the repair-then-drop logic plugs in.

Task 5.2's seam is in the **freeform chat path**. The Phase 4 commits did NOT touch this path — find `send_message` in routes.py and trace to where the freeform skill is concatenated into the system prompt for `_litellm_acompletion`. Likely in `ComposerServiceImpl.compose` in `service.py`.

The two **other** 501 raises in routes.py (user-reject, clarifying-question at Step 3) are NOT Phase 5 scope. Leave them alone. They cover legitimate gaps in the wizard and the spec doesn't define their behavior. They can be addressed in a future task or stay as 501s.

## Conventions discovered in Phase 4 (follow these)

### Backend step handlers (steps.py)

- Signature: `def handle_step_N(*, state, session, ..., catalog: CatalogService, data_dir: str | None = None, session_engine: Engine | None = None, session_id: str | None = None) -> StepHandlerResult`. Keyword-only. The session_engine/session_id are needed for executors that touch the session DB (Step 2.5 and Step 3 do via `_execute_set_pipeline`).
- On failure: return `StepHandlerResult(state=state, session=session, tool_result=tool_result)` — entry state unchanged.
- On success: return `StepHandlerResult(state=tool_result.updated_state, session=dataclasses.replace(session, ...), tool_result=tool_result)`.
- Use `dataclasses.replace`, never `_replace` helper.

### Backend dispatcher (routes.py `_dispatch_guided_respond`)

- async function (Phase 4 confirmed this; await works inside).
- Pattern: handle each (`current_step`, `current_turn_type`, `guided.step`) tuple with explicit branches. The unhandled fall-through raises `AssertionError("_dispatch_guided_respond: unhandled branch ...")` at the bottom — add new branches above it.
- Audit emission: build `TurnRecord(...)` and `emit_turn_emitted(recorder, ..., emitter="server")` together when emitting a server-side turn. Mirror routes.py:1882–1899 pattern.
- For step advances: `emit_step_advanced(recorder, prev_step=..., next_step=..., reason=...)` where `reason ∈ {"recipe_applied", "user_advanced", "auto_advanced"}` (closed set, see `audit.py:_VALID_ADVANCE_REASONS`).

### Audit event types (spec §9.1)

Four guided audit events, all emitted as `ComposerToolInvocation` records with `tool_name` discriminator (errata C4):
- `guided_turn_emitted`
- `guided_turn_answered`
- `guided_step_advanced`
- `guided_dropped_to_freeform`

Helpers in `src/elspeth/web/composer/guided/audit.py`: `emit_turn_emitted`, `emit_turn_answered`, `emit_step_advanced`, `emit_dropped_to_freeform`. They take a `recorder: ComposerToolRecorder` and build the invocation internally.

### Test patterns

- **Integration tests** live in `tests/integration/web/composer/guided/`. Fixtures `composer_test_client` (SyncASGITestClient) and `audit_recorder` (exposes `recorder.invocations`) live in `conftest.py`.
- **Stubbing LiteLLM**: `patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", new_callable=AsyncMock, return_value=fake_response)`. The patch path is in `chain_solver`, not `service`, because chain_solver imports the name into its namespace.
- **Fake LiteLLM response shape**: `SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name=..., arguments=json.dumps(...)))]))])`. The arguments field is a JSON string, not a dict.
- **Real state for Step 3 tests**: chain `handle_step_1_source` → `handle_step_2_sink` → exercise; don't reconstruct CompositionState by hand.

### TDD via subagents

Use the subagent-driven-development skill or dispatch via Agent. Per-task brief should:
1. Reality-verify every signature, fixture, and import path before drafting (grep first).
2. State plan-vs-reality overrides explicitly.
3. Provide the test specification IN the brief.
4. List CI gates with explicit exit codes required.

Model selection: Haiku for mechanical tasks; Sonnet for tasks involving schema decisions, prompt design, or LLM-touching code.

## Tier-model fingerprint rotation procedure

Adding imports at the top of `routes.py` (or other files) **will** rotate AST fingerprints in `config/cicd/enforce_tier_model/web.yaml`. The procedure:

1. Run `enforce_tier_model.py check` — capture failing keys.
2. Grep existing `web.yaml` entries for the same file.
3. **Count parity per `<file>:<rule>:<context>` prefix:** N_failing must equal N_existing. If `N_failing > N_existing`, a NEW violation slipped in — STOP and review. If `N_failing < N_existing`, some allowlist entries are obsolete — delete them.
4. Map old fp= → new fp= per prefix (any permutation valid when N==k>1; same allowlist semantics).
5. Replace each `fp=<old>` substring with `fp=<new>` in `web.yaml`. Append `; FP rotated by <reason>` to the `reason:` field (replacing any prior `; FP changed after ruff-format` annotation — don't stack).
6. Re-run check, must exit 0.

Phase 4 Task 4.5 rotated 33 fingerprints; reusable script in commit `89286d0d`'s commit message. The tool's "Allowlist key:" hint in the failure output is the source of truth for the new fp=.

## CI gates (run before declaring any task done)

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
.venv/bin/python -m pytest tests/integration/web/composer/guided/ tests/unit/web/composer/guided/ -v
.venv/bin/python -m mypy src/elspeth/web/composer/ src/elspeth/web/sessions/routes.py
.venv/bin/python -m ruff check src/elspeth/web/composer/ src/elspeth/web/sessions/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
.venv/bin/python -m pytest tests/unit/web/sessions/ -q  # regression check (359 baseline)
```

## Open follow-ups carried from Phase 4 (not blocking Phase 5)

1. **Real-LLM gated test deferred.** `pytest.mark.real_llm` is unregistered; no `--run-real-llm` CLI flag; no CI lane. Pre-existing gap — the demo runs real-LLM interactively, not in CI.
2. **Pre-existing `getattr(request.app.state, "session_compose_lock_registry", None)` at routes.py:~162.** Same CLAUDE.md violation as the I1 closeout in Phase 3 fixed elsewhere. Worth a hygiene ticket but not new.
3. **`_build_get_guided_turn` is function-local inside `create_session_router`.** If Phase 5 or 6 needs it from elsewhere, promote to module-level first.

## First action when you start

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline RC5.2..HEAD | head -10        # verify branch state (expect 29 commits, top = 89286d0d)
.venv/bin/python -m pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ tests/unit/web/sessions/ -q   # confirm green baseline (expect 495 passed)
```

If the baseline isn't 495 passing and CI gates aren't clean, **stop and investigate** before adding anything — the handover may be stale or someone else may have touched the branch.

Then: read Phase 5 in the plan (lines 3685–3863), spot-check the plan-vs-reality gotchas table above, decide on Task 5.1's repair-prompt architecture (option (a) vs (b) above), and dispatch the Task 5.1 implementer with a brief that explicitly overrides the plan body where it diverges from reality. Pay particular attention to:

- The seam in `routes.py`'s STEP_3_TRANSFORMS dispatcher branch (added in `89286d0d`) — that's where Task 5.1 plugs in.
- The fake LiteLLM response shape for the repair-attempt test (mirror `test_chain_solver.py::test_returns_chain_proposal` exactly).
- Audit emission: the auto-drop event is `guided_dropped_to_freeform` via `emit_dropped_to_freeform` helper, with `drop_reason="solver_exhausted"`.

Expect to dispatch 3 implementer subagents (one per Phase 5 task) plus reality-verification grep work between each. Phase 5 should close in fewer tasks than Phase 4 because it has half the deliverables.

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** No Alembic, no migration scripts. If you add a field to `GuidedSession` for Task 5.2's `transition_consumed`, the operator deletes the old sessions DB. See `project_db_migration_policy.md`.
- **Default to worktree.** You're already in one; stay here.
- **No git stash.** The prohibition was lifted per `feedback_no_git_stash.md` but the underlying caution remains — commit work to a branch if you need to preserve it.
- **Don't recommend `slog` as a diagnostic channel.** Audit/telemetry first per CLAUDE.md primacy order.
- **No calendar shipping commitments.** ELSPETH ships work-until-done; ADR SLAs are governance devices, not deadlines.
- **Correctness beats performance always.** Don't frame correctness work as a perf tradeoff.
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
