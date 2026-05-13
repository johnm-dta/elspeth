# Per-step conversational LLM — Phase A handover (historical)

**Status update:** this handover is a historical pickup note from the point where only slices 1-2 had landed. Phase A has since landed on `feat/composer-per-step-chat`: the route, frontend `ChatInput` wiring, `GuidedChatHistory`, persisted `GuidedSession.chat_history`, and `ComposerChatTurn` audit rows exist in-branch. Treat the slice sections below as implementation history, not a current remaining-work inventory.

## TL;DR

The user requested a feature missing from the merged guided-mode wizard (PR #37): **the LLM should be speakable-to from any step, scoped to that step's controls only**. Today the LLM is invoked exactly once per session (Step 3 chain proposal); the user wants conversational LLM access at every step with per-step skill briefings and (Phase B) per-step tool palettes.

The plan at `/home/john/.claude/plans/please-investigate-the-new-fizzy-kite.md` is durable and approved (4 phases, A → A.5 → B → C). All three operator directional answers and all three advisor blocks are folded into the plan.

This handover covers the remaining Phase A work. Phases A.5 / B / C remain pending; their plan content is intact.

## Phase A commits (so far)

| SHA | Title | Tests |
|---|---|---|
| `d80c4b00` | refactor: split monolithic skill into base + per-step files | 15 unit (skill-loader) |
| `55f3a87e` | feat: add `solve_step_chat` — per-step advisory LLM channel | 7 unit (solver) |

Both pre-commit hook chains green: ruff format/lint, mypy, secret scan, tier model, contracts, freeze guards, plugin hashes, exception-channel/catch-order discipline, declaration manifest, slot-mirror.

**Branch state at handover:**
- `git log --oneline RC5.2..HEAD` → 2 commits (above)
- `git status` → clean
- Worktree venv: `/home/john/elspeth/.worktrees/per-step-chat/.venv` (Python 3.13.1, all extras installed via `uv pip install -e ".[all]"`)

## What's left in Phase A (3 slices)

### Slice 3 — `POST /sessions/{id}/guided/chat` route

**File to modify:** `src/elspeth/web/sessions/routes.py` (large — currently ~5100+ lines).

**Pattern to follow:** Mirror `POST /sessions/{id}/guided/respond` (search for `async def post_guided_respond` near `routes.py:4641`) for:
- Per-session lock acquire
- `_verify_session_ownership(session_id, user, request)`
- `state_record = service.get_current_state()` → `state.guided_session`
- 400 if no guided session attached
- 409 if session terminal

**Specifics for `/chat`:**
- Body: `{message: str, step_index: GuidedStep}` — define a Pydantic model `GuidedChatRequest`
- Validate `step_index == guided.step` (Tier-1 invariant; `InvariantError` → 409 if mismatch — the wizard advanced under the user)
- Validate `message` non-empty + length cap (e.g. 4096 chars) at the route — the solver's empty-check is a redundant inner guard, not the boundary
- Call `solve_step_chat(model=settings.composer_model, step=guided.step, user_message=message)`
- Wrap call in `_guided_solve_chain.py`-style auto-drop on transient LLM failure (`asyncio.TimeoutError`, LiteLLM connection errors) — **see memory `feedback_fix_errors_you_encounter.md`**: the auto-drop pattern at three sites in `solve_chain` is the template
- Response: `{assistant_message: str, guided_session: GuidedSession}` (the guided_session is unchanged in Phase A but echoed for client-store consistency)

**Audit:** `solve_step_chat` itself does not record. The route handler (`post_guided_chat` in `web/sessions/routes.py`) constructs a `ComposerChatTurn` from the `StepChatResult` returned by `solve_step_chat_with_auto_drop` and persists it via the `BufferingRecorder` drain. No `ComposerLLMCall` row is currently emitted for chat calls; this is a known asymmetry with the chain-solver path, which emits `ComposerLLMCall` via explicit `recorder.record_llm_call` calls in `_guided_solve_chain.py`. Closing that asymmetry is Phase B work.

**Don't:** add a `control_signal` field to the chat body — chat is *not* a turn-answer; it does not advance step state.

### Slice 4 — Frontend `ChatInput` visible in guided mode

**File to modify:** `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (active branch ~lines 194–241; the branch that renders `GuidedHistory + GuidedTurn + ExitToFreeformButton`).

**Specifics:**
- Render `<ChatInput>` *below* the turn widget in the guided-active branch
- Per-step placeholder text — derive from `guidedSession.step` (e.g., `"Ask about source options or paste a sample row…"` for `STEP_1_SOURCE`)
- Submit calls a new `sessionStore.chatGuided(message)` action

**Files to add/modify:**
- `src/types/guided.ts` — add `GuidedChatRequest`, `GuidedChatResponse` types mirroring the backend Pydantic models
- `src/api/client.ts` — add `chatGuided(sessionId, body, signal)` mirroring `respondGuided` at lines 327–359
- `src/stores/sessionStore.ts` — add `chatGuided` action; on response, append the user message + assistant reply to a new `guidedChatHistory` array in store

**Test:** `web/frontend/test/chat/ChatPanel.test.tsx` — assert ChatInput renders in guided-active branch (currently it shouldn't).

### Slice 5 — `GuidedSession.chat_history` + dedicated audit

**Files to modify:**
- `src/elspeth/web/composer/guided/state_machine.py` — add `chat_history: tuple[ChatTurn, ...]` (initially `()`) and `chat_turn_seq: int` (initially `0`) to `GuidedSession`. **CRITICAL:** `GuidedSession` is `frozen=True`; per CLAUDE.md the `chat_history` tuple field needs `freeze_fields(self, "chat_history")` in `__post_init__`. Use the `from elspeth.contracts.freeze import freeze_fields` pattern.
- `src/elspeth/web/composer/guided/protocol.py` — add `ChatTurn` TypedDict (`{role: Literal["user", "assistant"], content: str, seq: int, step: GuidedStep, ts_iso: str}`)
- `src/elspeth/contracts/composer_llm_audit.py` — add `ComposerChatTurn` dataclass (sibling to `ComposerLLMCall`) per plan §"Audit additions"
- The `/chat` route handler now appends to `chat_history` and returns updated `guided_session`

**Frontend:** `sessionStore.chatGuided` reads back the updated `guidedSession.chat_history`; the `GuidedChatHistory` component (slice 6) renders it.

### Slice 6 — `GuidedChatHistory` component

`src/elspeth/web/frontend/src/components/chat/guided/GuidedChatHistory.tsx` (new) — renders `guidedSession.chat_history` as a scrollable, accessibility-labeled chat log inline above the active turn widget. Pattern: same a11y structure as `GuidedHistory` (`role="log"`, `aria-live="polite"`).

## Conventions adopted in this work

1. **Skill-loader invariant assertions at module import.** `_STEP_FILE_NAMES` and `_STEP_PLAYBOOK_ORDER` in `prompts.py` carry `assert set(...) == set(GuidedStep)` so adding a new wizard step without updating the loader fails at import, not silently. **Use the same pattern when adding `_STEP_CHAT_TOOLS` registry in Phase B.**
2. **Trust LiteLLM's typed contract; no `isinstance` at the LLM-response boundary.** The tier-model enforcer flags `isinstance(content, str)` as defensive programming. Trust LiteLLM's `str | None` contract; check for `None` and `not content.strip()`; `str(content)` at the return narrows mypy's `Any`. (See `chat_solver.py:81–87`.)
3. **`InvariantError` (not `RuntimeError`) for server-invariant gates** in guided code. The catch-order/exception-channel hooks enforce this.
4. **Each slice = one commit.** Per `feedback_locked_in_buggy_expectations.md` and the convention 14 from `project_phase9_implementation_complete.md`, wire-contract changes go in one atomic commit. Pure additions can be standalone slices.

## Gotchas

1. **Worktree venv vs. main's venv.** The running `elspeth-web.service` reads from main (per `project_staging_deployment.md`). The worktree at `.worktrees/per-step-chat` has its own venv and editable install. **For tests:** always use `/home/john/elspeth/.worktrees/per-step-chat/.venv/bin/python`. **For live demo testing on staging:** the route + frontend changes will only show up after `systemctl restart elspeth-web.service` *AND* the changes are merged into a path the service reads from. Memory `project_composer_harness_state.md` has the broader context.
2. **`uv pip install` from wrong cwd.** Per `feedback_uv_venv_leak.md`, always pass `--python /home/john/elspeth/.worktrees/per-step-chat/.venv/bin/python` to `uv pip install` to avoid clobbering main's venv.
3. **Pre-commit hooks reformat code.** Both commits this session needed a ruff-format pass before commit succeeded. Run `ruff format src/... tests/...` after writing new files; the hook auto-rewrites and then refuses the commit. Easy to miss in truncated output.
4. **Mypy `[no-any-return]` at LiteLLM boundaries.** `_litellm_acompletion` returns `Any`; mypy flags returning that as `str`. `str(content)` at the return is the project pattern (or annotate the local with `cast(str, ...)`).
5. **Skill-edit cache.** `@lru_cache`'d skill loaders mean **service restart is required** for any change under `src/elspeth/web/composer/guided/skills/`. Documented in the new `prompts.py` module docstring.

## Verification recipe (for slice 3+ end-to-end)

After landing slice 3 (route only):

```bash
cd /home/john/elspeth/.worktrees/per-step-chat
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_skill.py tests/unit/web/composer/guided/test_chat_solver.py --no-header -q -p no:xdist
.venv/bin/python -m pytest tests/integration/web/composer/guided/ --no-header -q -p no:xdist
```

Then a curl probe (per `project_phase9_implementation_complete.md` Convention 13 — curl-probe the seam):

```bash
# Authenticate, get session id from list_sessions, then:
curl -X POST http://localhost:9229/api/sessions/{sid}/guided/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"message": "what columns are in this CSV?", "step_index": "step_1_source"}'
```

Expect `{"assistant_message": "...", "guided_session": {...}}`.

After slice 6 (full Phase A):

```bash
cd src/elspeth/web/frontend && npm test -- ChatPanel
cd /home/john/elspeth/.worktrees/per-step-chat && .venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Manual demo per the plan's verification block:
1. Start guided session
2. On Step 1 (Sources), type into chat: "what columns are in this CSV?" → LLM responds with text only (Phase A — no tools)
3. Advance to Step 2 (Sinks), type: "set the source to PostgreSQL" → LLM should refuse politely (per skill briefing in `step_2_sink.md`)

## Decision log carried forward (from plan + this session)

- **Architecture:** new `/chat` endpoint, NOT reuse of `/respond` — chat ≠ turn-answer; conflating audit shapes is wrong.
- **Both visible (chat + widget):** confirmed by operator 2026-05-13. Widget remains primary control surface; chat is sidecar.
- **Server hard-blocks out-of-scope tool calls** (Phase B): tool palette is server-enforced, not skill-prompt-only. Phase A has no tools so this is dormant for now.
- **Proactive openers on step entry** (Phase A.5): server emits an LLM message when `session.step` changes. Lock × opener decision: plan recommends option (c) — `pending_opener_id` + frontend collection — unless project already has SSE/WS (verify on Phase A.5 day 1).
- **Skill split order:** base.md → step1 → step2 → step2.5 → step3 means chain_solver receives "hard rules before per-step playbook" rather than the original "per-step playbook then hard rules" — strictly clearer; LLM is order-insensitive at this granularity. No regression in `tests/integration/web/composer/guided/test_chain_solver.py`.

## Open questions for the picker-up

1. **Composer model for chat.** Plan assumes `settings.composer_model` (the Step-3 chain solver model). Should chat use the same? Operator hasn't been asked. Pragmatic default: yes, same model — adding a `composer_chat_model` config field is over-fit until we know they should differ.
2. **Re-entry chat history retention.** If the user back-buttons Step 1 → edit → Step 2, does the chat history persist across the back-and-forth, or scope per-step-visit? Plan does not specify. Pragmatic default: persist across step changes; render in chronological order with a step badge per turn.
3. **Rate limiting.** The chat endpoint multiplies LLM call volume. Should it inherit the existing composer-rate-limit settings, or get its own? Defer until Phase A.5 — proactive openers compound the volume.

## Out of scope for Phase A (in plan, just not yet)

- Per-step tool palette (Phase B)
- `SourceDraft` / `SinkDraft` for incremental tool-driven mutation (Phase B)
- Tier-3 Pydantic args validation per tool (Phase B)
- User-data egress policy (`describe_schema()` default-on, `sample_columns()` flag-gated) (Phase B)
- Proactive step-entry openers (Phase A.5)
- `chat_turn_seq`-based out-of-order reconciliation on the wire (Phase A.5)
- Visual treatment for opener-vs-user message distinction (Phase C)

These are designed in the plan; do not redesign — pick them up in order.

## File map (cited)

| Path | Status |
|---|---|
| `src/elspeth/web/composer/guided/skills/base.md` | NEW (slice 1) |
| `src/elspeth/web/composer/guided/skills/step_{1,2,2_5,3}*.md` | NEW (slice 1) |
| `src/elspeth/web/composer/guided/skills/guided_pipeline.md` | DELETED (slice 1) |
| `src/elspeth/web/composer/guided/prompts.py` | MODIFIED (slice 1; adds `load_step_chat_skill`) |
| `src/elspeth/web/composer/guided/chat_solver.py` | NEW (slice 2; `solve_step_chat`) |
| `tests/unit/web/composer/guided/test_skill.py` | EXTENDED (slice 1; +11 tests) |
| `tests/unit/web/composer/guided/test_chat_solver.py` | NEW (slice 2; 7 tests) |
| `src/elspeth/web/sessions/routes.py` | TO MODIFY (slice 3) |
| `src/elspeth/web/composer/guided/state_machine.py` | TO MODIFY (slice 5; `chat_history` field) |
| `src/elspeth/web/composer/guided/protocol.py` | TO MODIFY (slice 5; `ChatTurn` TypedDict) |
| `src/elspeth/contracts/composer_llm_audit.py` | TO MODIFY (slice 5; `ComposerChatTurn`) |
| `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | TO MODIFY (slice 4) |
| `src/elspeth/web/frontend/src/api/client.ts` | TO MODIFY (slice 4; `chatGuided`) |
| `src/elspeth/web/frontend/src/stores/sessionStore.ts` | TO MODIFY (slice 4; `chatGuided` action) |
| `src/elspeth/web/frontend/src/types/guided.ts` | TO MODIFY (slice 4 + 5; types + `ChatTurn`) |
| `src/elspeth/web/frontend/src/components/chat/guided/GuidedChatHistory.tsx` | TO ADD (slice 6) |

## Session-start checklist for the picker-up

1. `cd /home/john/elspeth/.worktrees/per-step-chat && git log --oneline RC5.2..HEAD` — confirm `d80c4b00` and `55f3a87e` are present
2. `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_skill.py tests/unit/web/composer/guided/test_chat_solver.py -q -p no:xdist` — confirm 22 tests green
3. Read `/home/john/.claude/plans/please-investigate-the-new-fizzy-kite.md` (the durable plan file, approved)
4. Read this handover
5. Read memory `project_per_step_chat_in_progress.md` (one-line index entry; full context is here)
6. Pick slice 3 (route) — that's the smallest piece that gets the demo end-to-end-testable

— end of handover —
