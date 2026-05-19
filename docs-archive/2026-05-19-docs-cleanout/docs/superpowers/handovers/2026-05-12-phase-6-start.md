# Handover — Composer Guided Mode (Phase 6 onward)

## TL;DR

You're inheriting a green branch with Phase 5 just landed. Phases 1–5 are complete (backend feature-complete: protocol + state machine + HTTP endpoints + LLM chain solver + auto-drop + progressive disclosure + audit-emission tests). Phases 6–10 ship the frontend.

**Phase 6 is the frontend foundation**: three tasks delivering TypeScript types mirroring the backend protocol, two `apiClient` functions for the guided endpoints, and a guided slice added to the existing `sessionStore`. Phase 6 is the smallest of the frontend phases — Phases 7+ build the React widget surface on top of it.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode` (36 commits ahead of RC5.2 as of Phase 5 close)
- **Top commit:** `8c0f7527`
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — Python-version mismatch corrupts `enforce_tier_model.py` results.
- **Frontend toolchain:** `npm` (NOT pnpm/yarn). Vitest is the test runner; Playwright drives e2e but isn't needed for Phase 6. The TypeScript checker is `tsc --noEmit` (NOT a wrapper script). Lint is `eslint src/`.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` (Phase 6 starts line 3866, ends line 4189)
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (Phase 6 relevant: §6.2 endpoints, §7.1–§7.3 component layout / state management)
- **Inbound handover:** `docs/superpowers/handovers/2026-05-12-phase-5-complete.md` — read it first; covers Phase 5 deliverables and the conventions the backend established.

## Read these BEFORE writing any code

1. **CLAUDE.md** at the worktree root. Frontend code is held to the same standards as backend: No-Legacy-Code policy applies, no `try/catch` swallowing errors, no defensive `?.` chains over fields you control, etc. The Three-Tier Trust Model translates to the frontend as: backend response shapes are Tier 1 (trust the pydantic strict-validated response) and user input is Tier 3 (validate before posting).

2. **The canonical pydantic response models** at `src/elspeth/web/sessions/schemas.py` lines 213–296. **These are the ground truth for the wire shapes** — your TypeScript types must mirror them exactly, not the plan body's illustrative samples (the plan body has drift; see below).

3. **The plan body's Phase 6 sample code (lines 3866–4189) contains substantial drift.** Trust the actual code paths over the plan samples. Section "Plan-vs-reality" below enumerates every divergence I've spotted; verify the rest by reading before drafting.

4. **Recent project memory** — your auto-memory dir. Especially `project_phase5_implementation_complete.md` (predecessor handover with architectural notes), `project_phase4_implementation_complete.md`, `feedback_default_is_fix_not_ticket.md`, `feedback_no_scope_dumping.md`.

## What's delivered (Phases 1–5 — don't redo)

Phase 5 (this branch's most recent work) added 7 commits on top of Phase 4 closure (`89286d0d`):

| Commit | Task | What |
|--------|------|------|
| `8c132210` | 5.1 feat | Auto-drop on solver-exhausted |
| `84705d3c` | 5.1 fix | `step_index` → `prev_step` audit-payload spec-§9.1 fix |
| `49be61ae` | 5.1 polish | `assert` → `raise`; `_DATA_ERROR_KEY` constant; docstring trims |
| `8bfd9f66` | 5.2 feat | Progressive-disclosure transition prompt |
| `407cb089` | 5.2 fix | `Any` → `TerminalState` typing; recompose handler gap closure |
| `21d970d7` | 5.2 polish | Hoist lazy imports; docstring trim |
| `8c0f7527` | 5.3 | Full-session audit emission tests (6 tests) |

**Gate state at handover:**
- 1748 backend tests pass (163 in `tests/integration/web/composer/guided/`)
- mypy clean on `src/elspeth/web/composer/` and `src/elspeth/web/sessions/routes.py`
- ruff clean
- `enforce_tier_model.py check`: clean
- `enforce_freeze_guards.py check`: clean

**Demo path runs end-to-end** through the backend: CSV source → JSON sink → (recipe match OR LLM-proposed chain with auto-drop on failure) → COMPLETED terminal with rendered YAML. Audit events back every protocol decision per spec §9.1.

## Frontend project structure (orient first)

```
src/elspeth/web/frontend/
├── package.json                         # npm scripts: "test" (vitest run), "test:watch", "test:e2e"
├── src/
│   ├── api/
│   │   ├── client.ts                    # 719 lines; module of EXPORTED FREE FUNCTIONS (NOT a class)
│   │   └── websocket.ts                 # WebSocket bindings (not relevant for Phase 6)
│   ├── stores/
│   │   ├── sessionStore.ts              # 575 lines; single zustand store (NOT a slice composition pattern)
│   │   ├── sessionStore.test.ts         # existing pattern for store tests
│   │   ├── executionStore.ts            # another zustand store — read for convention reference
│   │   ├── authStore.ts                 # auth state
│   │   ├── blobStore.ts                 # blob handling
│   │   ├── secretsStore.ts
│   │   └── subscriptions.ts
│   ├── types/
│   │   ├── api.ts                       # existing API type definitions (your new guided.ts joins here)
│   │   ├── index.ts                     # re-exports
│   │   └── runStatus.test.ts            # vitest pattern for type tests
│   ├── components/                      # Phase 7+ work (not Phase 6)
│   ├── hooks/
│   ├── styles/
│   └── ...
└── tests/playwright/                    # E2E (Phase 9 work, not Phase 6)
```

## What's still pending — Phase 6 scope

Phase 6 has **three tasks**. Plan body at lines 3866–4189 but expect drift on every task (see "Plan-vs-reality" below).

### Task 6.1 — TypeScript types mirroring the backend protocol (plan lines 3872–4014)

Create `src/elspeth/web/frontend/src/types/guided.ts` with TS mirrors of the pydantic models. Add a vitest type-assertion test at `src/elspeth/web/frontend/src/types/guided.test.ts`.

**Canonical sources of truth** (READ THESE before drafting types):

- `src/elspeth/web/sessions/schemas.py:213-296` — pydantic response models:
  - `TurnRecordResponse`, `TerminalStateResponse`, `GuidedSessionResponse`, `TurnPayloadResponse`, `GetGuidedResponse`, `GuidedRespondRequest`, `GuidedRespondResponse`
- `src/elspeth/web/composer/guided/protocol.py:77-110` — backend enums:
  - `TurnType` (6 values), `ControlSignal` (3 values), `GuidedStep` (4 values)
- `src/elspeth/web/composer/guided/state_machine.py:29-50` — `TerminalKind` (2 values), `TerminalReason` (3 values: `user_pressed_exit | protocol_violation | solver_exhausted` — NOT `completed_pipeline`; that's a prompt-rendering string, not a wire value)

**Critical wire-shape correctness items:**

1. `Turn.step_index` is **`int`**, NOT a string like `"step_1_source"`. The wire transmits 1, 2, 3, or some integer mapping. **VERIFY** by reading `build_*_turn` helpers (e.g., `build_step_1_source_inspect_turn` in routes.py around line 3000+) — find one such function and check what int it puts into `step_index`. Mirror exactly in TS. (Plan body shows `step_index: string` — that's WRONG.)

2. `GuidedSessionResponse` wire fields are **only `step`, `history`, `terminal`**. It does NOT carry `step_1_result`, `step_2_result`, or `step_3_proposal` (those are internal server state). Plan body's `GuidedSession` interface (lines 3971–3984) is WRONG on this — strip those fields.

3. `TerminalState.reason` is `TerminalReason | null`. The four-value union in the plan body (`"user_pressed_exit" | "protocol_violation" | "solver_exhausted" | null`) is correct. **Do NOT include `"completed_pipeline"`** — that's a prompt-rendering literal used in `build_mode_transition_system_prompt`, not a wire value. When `kind=COMPLETED`, `reason` is `null`.

4. `GuidedRespondRequest.control_signal` is `string | null` on the wire (per the pydantic schema's comment: "stale clients sending an unknown signal value fail gracefully"). On the TS side, type it `ControlSignal | null` to enforce client-side discipline; the server still validates.

5. `step` field on `GuidedSessionResponse` is the string enum (`"step_1_source"` etc.), NOT an int. Unlike `Turn.step_index` (int), the session's current step pointer comes back as the string value of `GuidedStep`.

**There is NO `GuidedStartResponse` shape.** Plan body uses one; reality has only `GetGuidedResponse` (returned from `GET /api/sessions/{id}/guided`) and `GuidedRespondResponse` (returned from `POST /api/sessions/{id}/guided/respond`). There is no `/guided/start` endpoint — see Task 6.2 reality below.

### Task 6.2 — API client methods (plan lines 4016–4094)

Add two functions to `src/elspeth/web/frontend/src/api/client.ts`. The plan body shows class-based methods (`apiClient.postGuidedStart`); **reality is a module of exported free functions**. Mirror the existing pattern from `sendMessage` (client.ts:277) and `recompose` (client.ts:298).

**Critical endpoint-path correctness:**

| Plan body says | Reality |
|----------------|---------|
| `POST /composer/guided/start` | **DOES NOT EXIST.** Initial fetch is `GET /api/sessions/{session_id}/guided` (returns `GetGuidedResponse`) — no separate "start" endpoint. |
| `POST /composer/guided/respond` | Wrong path. Reality: `POST /api/sessions/{session_id}/guided/respond` |
| `apiClient.postGuidedStart` class method | Reality: bare `export async function getGuided(sessionId)` returning `Promise<GetGuidedResponse>` |
| `apiClient.postGuidedRespond` class method | Reality: bare `export async function respondGuided(sessionId, turnResponse)` returning `Promise<GuidedRespondResponse>` |

**Convention to mirror (from `sendMessage` at client.ts:277):**

```typescript
export async function getGuided(
  sessionId: string,
  signal?: AbortSignal,
): Promise<GetGuidedResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/guided`, {
    method: "GET",
    headers: authHeaders(),
    signal,
  });
  return parseResponse<GetGuidedResponse>(response);
}

export async function respondGuided(
  sessionId: string,
  turnResponse: TurnResponse,
  signal?: AbortSignal,
): Promise<GuidedRespondResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/guided/respond`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(turnResponse),
    signal,
  });
  return parseResponse<GuidedRespondResponse>(response);
}
```

**VERIFY** by reading the actual request schema: the POST `/guided/respond` body shape is `GuidedRespondRequest` (schemas.py:264), which has the same fields as `TurnResponse` (chosen, edited_values, custom_inputs, accepted_step_index, edit_step_index, control_signal). So passing `turnResponse` directly as the body is correct IF the TS `TurnResponse` type matches the pydantic `GuidedRespondRequest` shape exactly (which it should after Task 6.1).

**Helpers already in client.ts:**

- `authHeaders(contentType?: string): HeadersInit` at line 47 — injects auth bearer token. Pass `"application/json"` for POST/PUT bodies; pass nothing for GET.
- `parseResponse<T>(response: Response): Promise<T>` at line 73 — handles non-OK responses by throwing `ApiError`, parses JSON otherwise.

**Test pattern** (mirror `sessionStore.test.ts` or look for `client.test.ts` if it exists):

The plan body uses `vi.spyOn(globalThis, "fetch")` — that's the right pattern. Mock the response shape, assert the URL + body + headers, assert the parsed return value.

### Task 6.3 — `sessionStore` guided slice (plan lines 4096–4186)

The plan body assumes a **zustand slice composition pattern** (separate `GuidedSlice` interface merged into a multi-slice store). **Reality is a single flat zustand store**: `useSessionStore = create<SessionState>((set, get) => ({...}))` at sessionStore.ts:120.

**Real approach:** Add fields to the existing `SessionState` interface (sessionStore.ts:74) and actions to the existing `create()` body. Don't introduce a new "slice" abstraction — it'll be inconsistent with the rest of the store.

**Fields to add to `SessionState`:**

```typescript
guidedSession: GuidedSessionResponse | null;
guidedNextTurn: TurnPayloadResponse | null;   // server's next-turn, replaced atomically
guidedTerminal: TerminalStateResponse | null; // mirror of guidedSession.terminal for fast access
```

Note the types are the **wire response types** (`GuidedSessionResponse`, `TurnPayloadResponse`), not the spec-level abstract types. The store holds what the server sent.

**Actions to add:**

```typescript
startGuided: (sessionId: string) => Promise<void>;       // calls getGuided (GET /guided)
respondGuided: (turnResponse: TurnResponse) => Promise<void>;  // calls respondGuided POST
exitToFreeform: () => Promise<void>;                     // shortcut: respondGuided with control_signal
```

**Implementation guidance:**

- After each call, **replace** `guidedSession`, `guidedNextTurn`, and `guidedTerminal` from the server response. **No optimistic updates** — server is authoritative (spec §7.3).
- `exitToFreeform` is sugar for `respondGuided({ control_signal: "exit_to_freeform", chosen: null, edited_values: null, ... })`. All other fields null.
- The existing `activeSessionId` field in `SessionState` is where `respondGuided` reads the session ID. If it's null, throw — that's an offensive programming check (invariant: no respond without an active session).
- Per CLAUDE.md offensive programming, **don't use `?.` chains** to defend against null sessionId — throw a `RuntimeError`-equivalent (`throw new Error("respondGuided called without active session")`).

**Test pattern** (look at `sessionStore.test.ts` for convention):

- Use `vi.spyOn(apiClient, "getGuided")` and `respondGuided` — but `apiClient` is the module namespace. Import like:
  ```typescript
  import * as apiClient from "../api/client";
  vi.spyOn(apiClient, "getGuided").mockResolvedValue(...);
  ```
- Reset store state in `beforeEach` via `useSessionStore.setState({ guidedSession: null, ... })`.
- Assert state shape after each action.

### Phase 6 exit criterion (plan line 3870)

> "Vitest store-slice tests pass; types compile cleanly; manual smoke against the running backend (curl the start endpoint, assert the store updates)."

The "manual smoke" is a one-off check against the running `elspeth.foundryside.dev` staging service or local `elspeth-web` (per project memory `project_staging_deployment.md`). Not load-bearing for Phase 6 closure — the vitest + tsc gates are the mechanical criterion.

## Plan-vs-reality gotchas observed (verified before this handover)

| Plan body says | Reality |
|----------------|---------|
| Endpoint paths `/composer/guided/start` and `/composer/guided/respond` | `GET /api/sessions/{id}/guided` and `POST /api/sessions/{id}/guided/respond`. There is no separate "start" endpoint — GET handles initial fetch. |
| `class ApiClient` with `this._post<T>(...)` helper | `client.ts` is a module of free `export async function fooBar()` — `_post` does NOT exist. Use `fetch(...)` + `authHeaders()` + `parseResponse<T>()`. |
| `apiClient.postGuidedStart(...)` import as object | Bare named import: `import { getGuided, respondGuided } from "@/api/client"`. |
| `Turn.step_index: string` (`"step_1_source"`) | `int` on the wire. Verify with `build_*_turn` helpers in routes.py. |
| `GuidedSession.step_1_result / step_2_result / step_3_proposal` on the wire | These DO NOT cross the boundary. `GuidedSessionResponse` has only `step`, `history`, `terminal`. |
| `TerminalState.reason` includes `"completed_pipeline"` | NO. Wire enum is `user_pressed_exit | protocol_violation | solver_exhausted | null`. `"completed_pipeline"` is a prompt-rendering literal. |
| zustand "slice" pattern with separate `GuidedSlice` interface | `sessionStore.ts` is a flat `create<SessionState>(...)` — add fields directly to `SessionState`, don't introduce slice composition. |
| `useSessionStore.getState().startGuided("sess-1")` in tests | Real store-test pattern uses `useSessionStore.setState({...})` for setup and `useSessionStore.getState()` for invocations — verify in `sessionStore.test.ts`. |
| Test sample showing `vi.spyOn(apiClient, "postGuidedStart")` | Need `import * as apiClient from "../api/client"` since the module exports free functions, not a class instance. |

## Phase 6–specific conventions to follow

### TypeScript type discipline

- **Use the canonical pydantic response model field names verbatim** in TS interfaces (e.g., `guided_session` not `guidedSession` for wire shapes; `step_index` not `stepIndex`). The frontend reads JSON-decoded objects directly without camelCase transformation.
- If you want camelCase at the consumer boundary (component code), provide a converter — but that's Phase 7+ work, not Phase 6.
- For enum mirrors, use **string-literal union types** matching the backend's `StrEnum` values exactly:
  ```typescript
  export type ControlSignal = "exit_to_freeform" | "request_advisor" | "reject";
  ```
- `unknown` for genuinely opaque payloads (e.g., the `payload` inside `TurnPayloadResponse` which is shaped per turn type). Avoid `any` — it disables checking silently.

### API function convention

- Always pass `signal?: AbortSignal` as the last parameter, even if unused — mirrors `sendMessage` / `recompose` for retry-cancellation consistency.
- Use `authHeaders()` for GET, `authHeaders("application/json")` for POST/PUT with bodies.
- Always `return parseResponse<T>(response)` — never check `response.ok` manually. `parseResponse` throws on non-2xx.

### zustand store convention

- All mutations via `set({ ... })` — never mutate fields directly.
- All reads via `get()` inside actions or via selector hooks at the call site.
- Per CLAUDE.md offensive programming: throw on invariant violations (e.g., `activeSessionId` null when it shouldn't be), don't `?.` over them.

### Test conventions

- Vitest + `vi.spyOn` for HTTP mocks (no msw needed for these unit-level tests).
- `beforeEach` resets store state to known clean values.
- Each test exercises a single behavior; descriptive names that point to the broken behavior on failure.
- Test imports at module top (mirrors `sessionStore.test.ts` — not per-method).

## CI gates (run before declaring any task done)

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode/src/elspeth/web/frontend
npm test -- --run                                                  # vitest unit tests
npx tsc --noEmit                                                    # TypeScript type-check (no emit)
npx eslint src/                                                     # ESLint

# Plus the backend gates (must remain green — Phase 6 should not touch backend):
cd /home/john/elspeth/.worktrees/composer-guided-mode
.venv/bin/python -m pytest tests/integration/web/composer/guided/ tests/unit/web/composer/guided/ tests/unit/web/sessions/ -q
.venv/bin/python -m mypy src/elspeth/web/composer/ src/elspeth/web/sessions/routes.py
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
```

Phase 6 is frontend-only — the backend gates should stay green throughout (1748 tests, mypy/ruff/tier-model/freeze-guards all clean). If a backend gate flips red, something is wrong; investigate before continuing.

The frontend lacks an enforce_tier_model-equivalent — TypeScript/ESLint coverage is shallower than the backend's CI scaffolding. Be especially vigilant about:
- `any` types creeping in (eslint will warn but the warning is sometimes suppressed)
- Missing null checks where pydantic shapes have `| None`
- Unused imports / dead code

## Open follow-ups (NOT blocking Phase 6)

1. **`_replace_dc` 4-site lazy-import normalisation in routes.py.** Pre-existing pattern, out of Phase 5 scope. Future hygiene ticket.
2. **`ValidationSummary.to_dict()` consolidation.** Three call sites duplicate the shape; recommended in Phase 5 code review. Out of scope for Task 5.1, deferred.
3. **Stale filigree observation `elspeth-obs-2bbdb1ddba`.** Filed during Phase 5 about the recompose handler gap, but the gap was fixed in commit `407cb089`. Dismiss it via `mcp__filigree__dismiss_observation`.
4. **Frontend types directory needs a re-export update.** When you add `types/guided.ts`, check `types/index.ts` and decide whether to re-export the new module from there. Look at how `types/api.ts` is exposed for the convention.

## First action when you start

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline RC5.2..HEAD | head -10                # expect 36 commits, top = 8c0f7527
.venv/bin/python -m pytest tests/integration/web/composer/guided/ tests/unit/web/composer/guided/ tests/unit/web/sessions/ -q   # expect ~1742 passed

cd src/elspeth/web/frontend
npm test -- --run                                       # expect all existing frontend tests passing
npx tsc --noEmit                                        # expect clean
```

If any baseline gate fails, **stop and investigate** before starting Phase 6 — the handover may be stale or someone else may have touched the branch.

Then:
1. Read Phase 6 in the plan (lines 3866–4189) **alongside** the canonical sources of truth (pydantic schemas + backend enums). Note every plan-body divergence before writing TS.
2. **Verify `Turn.step_index` actual wire value** — find a `build_*_turn` helper and see what `int` it returns. This is the one item I haven't verified end-to-end in this handover; the plan body has it wrong but I'm not certain whether the wire value is the int position (1, 2, 3) or some other mapping.
3. Dispatch the Task 6.1 implementer (types + test) with a brief that explicitly overrides the plan body's interface samples with the pydantic shapes. Pay particular attention to:
   - `Turn.step_index: number` (not string)
   - `GuidedSession` shape stripped to `{ step, history, terminal }` only
   - `TerminalReason` union excludes `"completed_pipeline"`
   - No `GuidedStartResponse` type — only `GetGuidedResponse` and `GuidedRespondResponse`

Expect to dispatch 3 implementer subagents (one per Phase 6 task). Each task is small (≤200 lines of new code) but each has multiple reality-overrides that the plan body won't surface.

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** No Alembic; no migration scripts; no `from_dict` backward-compat defaults. (Phase 6 doesn't touch the DB — but the constraint applies if you somehow find yourself there.)
- **Default to worktree.** You're already in one; stay here.
- **No git stash.** Commit work to a branch if you need to preserve it.
- **Don't recommend `slog` as a diagnostic channel** (on the backend). The frontend has its own logging conventions — read existing components for the established pattern (probably `console.log`/`console.warn` for dev-only diagnostics).
- **No calendar shipping commitments.** ELSPETH ships work-until-done.
- **Correctness beats performance always.**
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
- **Type annotations must be concrete.** `any` in TypeScript is the moral equivalent of `Any | None` on the backend — silently disables checking. Use `unknown` for genuinely opaque payloads; use precise union types otherwise.
- **No optimistic updates in the store** — server is authoritative for guided session state (spec §7.3).
- **Don't camelCase wire field names in interfaces.** The pydantic shapes use `snake_case` and the JSON decoder doesn't transform them. Mirror exactly.
