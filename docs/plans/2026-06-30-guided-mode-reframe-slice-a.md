# Slice A — Surface Cleanup + Silent-Compute Indicator (Implementation Plan)

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans. Frontend tasks use vitest, not pytest — adapt the TDD template. See `2026-06-30-guided-mode-reframe-overview.md` for the shared gate + constraints.

**Goal:** Remove the two genuinely-dead protocol branches (classified safe by evidence review) and reuse the existing `ComposingIndicator` in the guided chat surface. No happy-path behaviour change. **No sessions-DB wipe.**

**Issue:** `elspeth-b30e59bfa3` (task). **Architecture:** subtractive + one additive reuse; all changes are non-persisted.

**Scope guard — DO NOT TOUCH (retained, not vestigial):**
- `TurnType.RECIPE_OFFER` / `GuidedStep.STEP_2_5_RECIPE_MATCH` (`protocol.py:33/159`, legal matrix, `_helpers.py:2436`, 409 orphan guards `guided.py:372-377`/`1208-1213`, `test_error_paths.py::TestOrphanedStep2_5Recovery`).
- `coaching` / `recipe_match` profile flags (`profile.py:33/35`) — persisted in `GuidedSession.profile` blob (`state_machine.py:423`). Removal forces `GUIDED_SESSION_SCHEMA_VERSION` (`state_machine.py:44`) + `SESSION_SCHEMA_EPOCH` (`models.py:132`) bump + wipe → **out of Slice A**.
- `inspect_and_confirm` — dormant emission but the step-1→step-2 transition turn (`state_machine.py:637`); structural spine.

---

## Task A1 — Remove the dead `interpretation_review` frontend `TurnType`

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/guided.ts:28` (remove union member `'interpretation_review'`)
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx:140-146` (remove the dead `null`-dispatch case; keep the `:171` exhaustiveness `never` check intact)
- Test: `src/elspeth/web/frontend/src/types/guided.test.ts:27-51` ("TurnType union has exactly 8 values")

**Why safe (evidence):** `interpretation_review` is a frontend-only `TurnType` member with a self-labelled "Dead dispatch path" rendering `null`. It is NOT in `InterpretationKind` (`src/elspeth/contracts/composer_interpretation.py:74-85`), NOT a backend `TurnType` (`protocol.py:25-34`), so it is never emitted or persisted and has no DB CHECK. Distinct from the real `request_interpretation_review` tool, the `interpretation_review_disabled` DB column, and `CHECK_INTERPRETATION_REVIEW` — none of those are touched.

**Step 1 — update the test to the target shape (RED surfaces at `tsc`, not vitest).** Per the reality review, `interpretation_review`/the count appears in **THREE** places in `guided.test.ts`: the compile-time `Equals<TurnType, …>` union literal (~`:28-38`), the runtime `all` array (~`:39-48`), and `toHaveLength(8)` (~`:50`). Update **all three** to the 7-value target.

Run: `npm run build` (tsc). NOTE: `npm run test` (vitest/esbuild) does NOT typecheck — the runtime array alone would pass after editing — so the RED gate is the build/typecheck, where `Equals<TurnType, …>` fails because the live union still has 8 members.
Expected: FAIL — `tsc` reports the `Equals<>` mismatch.

**Step 2 — remove the member + dispatch case (GREEN).** Delete `'interpretation_review'` from the `TurnType` union (`guided.ts:28`) and remove its `case 'interpretation_review':` block in `GuidedTurn.tsx:140-146`. Confirm the `switch` still compiles against the narrowed union (the `default`/`never` exhaustiveness check at `:171` now covers one fewer case).

Run: `npm run test -- guided.test.ts && npm run build`
Expected: PASS; `tsc` clean (no unreachable/`never` errors).

**Step 3 — commit.**
```bash
git add src/elspeth/web/frontend/src/types/guided.ts \
        src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx \
        src/elspeth/web/frontend/src/types/guided.test.ts
git commit -m "refactor(web/guided): drop dead interpretation_review TurnType + null dispatch

Frontend-only union member with a self-labelled dead null dispatch; not an
InterpretationKind, not a backend TurnType, never emitted/persisted. Interpretation
events surface via AcknowledgementStack, unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Definition of Done:**
- [ ] all THREE `guided.test.ts` spots updated (Equals union + `all` array + `toHaveLength`); passes
- [ ] `GuidedTurn.tsx` exhaustiveness `never` check intact; `tsc`/build clean
- [ ] mirror hook passes (slot_type untouched)
- [ ] no backend file touched

---

## Task A2 — Remove the `chosen:["reject"]`→501 dead branch

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py:3511-3515` (remove the `if chosen == ["reject"]: raise HTTPException(501, ...)` branch)
- Modify: `src/elspeth/web/sessions/routes/_helpers.py:3516-3519` (the adjacent fall-through 400 message still advertises `['reject']` — drop `reject` from "must have chosen=['accept'] or chosen=['reject']")
- Test: `tests/integration/web/composer/guided/test_step_3_e2e.py:241` (`test_csv_to_json_step_3_reject_returns_501`)

**Why safe:** the 501 is raised *before* any state mutation — not persisted, no DB CHECK. It is distinct from `ControlSignal.REJECT` (`protocol.py:131`, "re-roll the LLM"), which is a separate live mechanism and is NOT touched. After removal, `chosen:["reject"]` falls through to the standard 400 (invalid `chosen`). **Positive finding (arch review): `chosen:["reject"]` is UI-unreachable** — the frontend Reject affordance (`ProposeChainTurn.tsx:126`) sends `control_signal:"reject"`, never `chosen:["reject"]` — so this change is invisible to conforming clients (pure dead-code removal).

**Step 0 — precheck (quality review).** `grep -rn "Step 3 chain rejection" tests/ src/` to confirm the unique 501 message is asserted ONLY by the gate-path integration test (so no stray assertion elsewhere breaks unnoticed).

**Step 1 — restate the test for the new behaviour (RED).** Rename/rewrite `test_csv_to_json_step_3_reject_returns_501` to `test_csv_to_json_step_3_reject_returns_400` and assert `status_code == 400` plus the corrected message (no `reject` advertised).

Run: `pytest tests/integration/web/composer/guided/test_step_3_e2e.py -k step_3_reject -q`
Expected: FAIL — endpoint still returns 501.

**Step 2 — remove the branch + fix the message (GREEN).** Delete the 501 branch; update the 400 message to `"must have chosen=['accept']"`.

Run: `pytest tests/integration/web/composer/guided/test_step_3_e2e.py -k step_3_reject -q`
Expected: PASS (400 + corrected message).

**Step 3 — commit.**
```bash
git add src/elspeth/web/sessions/routes/_helpers.py \
        tests/integration/web/composer/guided/test_step_3_e2e.py
git commit -m "refactor(web/guided): drop dead chosen=['reject']->501 branch

Pre-mutation 'not yet implemented' branch; non-persisted. Falls through to the
standard 400 for invalid chosen; the 400 message no longer advertises reject.
ControlSignal.REJECT (re-roll) is unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Definition of Done:**
- [ ] `chosen:["reject"]` returns 400 with a message that does not advertise `reject`
- [ ] no persisted state touched; `ControlSignal.REJECT` untouched
- [ ] backend guided test set green

---

## Task A3 — Reuse `ComposingIndicator` in the guided chat surface (§5.5)

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (guided chat region near `:1354`; the guided branch currently only disables the box at `:1340` and shows a thin `guided-current-decision-pending` line at `:1500-1502`)
- Reuse: `src/elspeth/web/frontend/src/components/chat/ComposingIndicator.tsx` (`export function ComposingIndicator` `:146`; props `:6-10`: `latestRequest?`, `compositionState?`, `composerProgress?`; falls back to `heuristicWorkingView` when no progress snapshot `:151-153`)
- Test: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx` (or a focused `ChatPanel.guided.test.tsx`)

**Why no red-green logic test:** this is additive reuse of an existing, already-tested component (`ComposingIndicator.test.tsx`). Verify by a render assertion, not new logic.

**Step 1 — mount the indicator (additive).** In the guided chat region (~`ChatPanel.tsx:1354`), render the indicator while a guided build is in flight, gated on `guidedChatPending` (store, `:549`):
```tsx
{guidedChatPending && (
  <ComposingIndicator
    latestRequest={guidedLatestRequest}   // see note — NOT a real var yet; source the guided last user message, else null
    compositionState={compositionState}
    composerProgress={composerProgress}
  />
)}
```
**Note (reality review):** `lastGuidedChatMessage` does not exist; the freeform branch uses `latestRequest={activeComposerMessage?.content ?? null}` (`ChatPanel.tsx:1775`). `latestRequest` is OPTIONAL — the indicator falls back to `heuristicWorkingView(latestRequest, compositionState)` (`ComposingIndicator.tsx:151-153`). Source the guided last user message (e.g. from `guidedSession` history) or pass `null`; do not invent a variable. Source `compositionState`/`composerProgress` from `useSessionStore` as the freeform branch does (`:1773-1779`). Keep the existing `disabled={guidedChatPending || guidedResponsePending}` on the input (do not regress the 409-race fix).

**Step 2 — add a render test (GREEN).** Assert the indicator appears in the guided branch when `guidedChatPending` is true and is absent when false. **Also assert (409-race hygiene, per quality review) the input stays `disabled` while `guidedChatPending` is true** — a regression guard, since this change sits next to the prior guided-resend 409 fix.

Run: `npm run test -- ChatPanel`
Expected: PASS.

**Step 3 — verify styling + build.**
Run: `npm run lint:css && npm run build`
Expected: PASS (ComposingIndicator ships its own styles; no new CSS needed — confirm no stylelint regression).

**Step 4 — commit.**
```bash
git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel*.test.tsx
git commit -m "feat(web/guided): show ComposingIndicator during guided build

Guided previously only disabled the input during a /guided/chat build with no
'thinking' affordance. Reuse the freeform ComposingIndicator gated on
guidedChatPending. Additive; input-disable behaviour unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Definition of Done:**
- [ ] Indicator renders during a guided build, absent otherwise; input stays disabled during build (409-race fix preserved)
- [ ] no new component; reused existing one

---

## Slice A — overall Definition of Done

- [ ] A1+A2+A3 committed atomically
- [ ] Frontend gate green: `npm run test && npm run lint && npm run lint:css && npm run build`
- [ ] Backend gate green: `pytest tests/unit/web/composer/guided tests/integration/web/composer/guided -q`
- [ ] **No `data/sessions.db` wipe performed or required** (the only wipe-forcing items stay out of scope)
- [ ] Mirror hook green
- [ ] `elspeth-b30e59bfa3` closed with the commit SHAs; the spec §5.6 classification matches what shipped
