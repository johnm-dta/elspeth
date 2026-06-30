# Slice B — Wire-Stage Advisor Dead-End (Implementation Plan) — v2

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans. Mixed backend (pytest) + frontend (vitest). See the overview for the shared gate + constraints. **v2 incorporates plan-review findings (reality + arch + quality).**

**Goal:** End the wire-stage advisor dead-end (`elspeth-7b0f75e90e`). Render the `advisor_findings`/`signoff_outcome` the backend already emits; gate the affordances on the outcome; disclose the pass cost on the **explicit** "Ask advisor" action; stop the silent repeat-burn from the client. **Frontend rendering + one additive backend field at the re-emit sites.**

**Issue startability:** bug at `triage`; set `severity` then advance `triage→confirmed→fixing`.

**Tier-3 egress (unchanged):** "commit failed" detail at `_helpers.py:2925/3045/3231` (STEP_1/STEP_2) — NOT on the wire path. Confirm untouched by `git diff` at the gate.

---

## The outcome → affordance contract (ALL six rows are in scope and tested)

`SignoffOutcome` (`signoff.py:34-41`): `complete`, `revise`, `blocked_flagged`, `blocked_unavailable`, `escape_unavailable`. Live emit confirmed: `_helpers.py` emits `decision.outcome.value`.

| `signoff_outcome` | meaning | UI renders | cost copy? |
|---|---|---|---|
| *(absent)* | not checked yet (initial turn) **OR advisor-off tutorial** | "Confirm wiring" | **NO** (gate on `passes_remaining !== undefined`) |
| `revise` | FLAGGED, budget remains | findings + "Ask advisor" + "Exit to freeform"; **NO bare Confirm** | YES — "spends 1 of {passes_remaining}" |
| `blocked_flagged` | FLAGGED, exhausted | terminal: findings + "Exit to freeform"; no budget-burning button | n/a |
| `blocked_unavailable` | service/budget missing | explanation + "Exit to freeform" | n/a |
| `escape_unavailable` | advisor unreachable, exhausted | explanation + **"Complete without sign-off"** (ONLY here) | n/a |
| `complete` | clean — re-emitted via `request_advisor`; backend does NOT auto-complete | **actionable "Confirm wiring"** to finalize (NOT a dead-end) | n/a |
| *(any unknown)* | **default branch** | explanation + "Exit to freeform" (never a dead-end) | n/a |

**Governance invariant (server-enforced; do NOT violate in UI):** `complete_without_signoff` is honoured ONLY when `signoff_outcome === "escape_unavailable"` (`advisor_signoff_escape_offered` set solely on ESCAPE_UNAVAILABLE, `signoff.py:174`; pinned by `test_signoff_classifier.py::test_flagged_never_yields_an_escape`). The frontend gating is **defense-in-depth, not the boundary** — a non-conforming client sending `complete_without_signoff` on a FLAG just re-runs the (non-completing) sign-off. Never render that button on `revise`/`blocked_flagged`.

**Silent-burn fix = frontend (decision, with explicit risk-acceptance).** The repeat-burn is each *repeat* plain `chosen:["confirm"]` re-running `run_wire_signoff` (+1 at the increment inside `run_wire_signoff`, dispatched from `_helpers.py:3816`). After a FLAG the UI renders **no bare Confirm**, so no further silent spend is sent. The first Confirm legitimately runs one sign-off pass (the act of confirming) — expected, not the defect. For a non-conforming client the burn is **bounded** (`run_wire_signoff` no-ops once `passes_used>=max_passes`, `signoff.py:133-158` — `passes_delta=0`, no provider call), never unbounded, never a bypass. **Backend burn-guard (option b) is a deliberate non-goal here** — surface to the operator as a defensive follow-up ("backend should no-op a plain confirm when the live turn already carries a FLAGGED/REVISE outcome").

---

## Task B0 — (backend, additive) `passes_remaining` at the RE-EMIT sites only

**Files:**
- Modify: `src/elspeth/web/composer/guided/emitters.py` (`build_step_4_wire_turn` `:417`) — add a `passes_remaining: int | None = None` parameter; fold it into the payload **only when not None** (alongside the existing findings/outcome folding `:433-436`). The emitter stays dumb — it does NOT compute the value (it has no `session`/`settings` in scope).
- Modify: the **re-emit call sites where `max_passes` is actually assigned** — ONLY `_helpers.py:3703` (the `request_advisor` re-emit) and `_helpers.py:3858` (the auto-path re-emit). Compute `passes_remaining = max_passes - session.advisor_checkpoint_passes_used` and pass it. (`max_passes` = `settings.composer_advisor_checkpoint_max_passes`, `guided.py:1459`; the local is assigned at `:3671`/`:3815`.)
- **DO NOT** compute it at `_helpers.py:3655` / `:3783` (per arch re-review N1): those are inside the `composer_service is None or advisor_checkpoint_max_passes is None or … <= 0` branches (`:3645`/`:3777`) — `max_passes` is unassigned there, so the subtraction would `NameError`. Those emit `BLOCKED_UNAVAILABLE` turns that need no cost copy → leave `passes_remaining=None`. Confirm the set with `grep -n build_step_4_wire_turn src/elspeth/web` before editing.
- **Do NOT** touch the initial-turn emits (`_emit_wire_turn` `_helpers.py:2658`, `guided.py:275` rebuild) — they have no `max_passes` in scope, and the initial turn deliberately carries no cost copy (see decision above). This keeps B0 a bounded change and makes the advisor-off tutorial naturally omit `passes_remaining`.
- Modify: `src/elspeth/web/frontend/src/types/guided.ts` (`WireStageData`, near `:388-389`) — add `passes_remaining?: number`.
- Test (ROUTE-level, not just emitter): `tests/unit/web/sessions/routes/` — assert a **re-emitted flagged** wire RESPONSE (through `post_guided_respond` → `_turn_payload_response`, `guided.py:1607/127-153`) carries `passes_remaining`, and the **initial** wire response does NOT. (The serializer copies all payload keys, so the field rides through.) An emitter-only unit test cannot prove end-to-end delivery.

**Step 1 — failing route-level test (RED).** Drive a session to a FLAGGED re-emit; assert the response payload carries a **concrete** `passes_remaining` (per quality re-review, pin an integer to catch off-by-one — e.g. `max_passes=3`, after one flagged pass `passes_used=1` ⇒ assert `passes_remaining == 2`, NOT the formula `max_passes - passes_used`, which is true for either the pre- or post-increment snapshot and so can't catch an off-by-one); assert the initial wire response omits the field.

Run: `pytest tests/unit/web/sessions/routes -k wire -q`
Expected: FAIL — `passes_remaining` absent.

**Step 2 — thread it (GREEN).** Add the emitter param + compute-and-pass at the re-emit sites.

Run: `pytest tests/unit/web/composer/guided/test_wire_payload.py tests/unit/web/sessions/routes -k "wire or signoff" -q`
Expected: PASS.

**Step 3 — commit.** `feat(web/guided): expose passes_remaining on re-emitted wire turns` (+ Co-Authored-By).

**DoD:** re-emitted flagged wire response carries `passes_remaining`; initial + advisor-off tutorial responses omit it; serializer pass-through proven by the route test.

---

## Task B1 — frontend silent-burn fix (part of B2's branch; no separate commit)

On `revise`/`blocked_*`/`escape_unavailable` the UI renders no bare Confirm → no further plain `confirm` is sent → no silent repeat-burn. Implemented in B2.

---

## Task B2 — (frontend) render findings + the full outcome→affordance table

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx` (props `:81-85`; render `:99-146`) — read `advisor_findings`, `signoff_outcome`, `passes_remaining` from `data`; render findings when present; switch the action area on the table, **with a safe `default` branch** (unknown outcome → explanation + "Exit to freeform", never an empty action area).
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx:147-164` (`confirm_wiring` case) — pass advisor fields; wire `onAskAdvisor` → `{control_signal:"request_advisor", chosen:null, ...}`; `onExitToFreeform` → existing exit; `onCompleteWithoutSignoff` → `{chosen:["complete_without_signoff"], ...}` rendered ONLY when `signoff_outcome === "escape_unavailable"`. Mirror `ProposeChainTurn.tsx:137`'s Ask-advisor wiring.
- Cost copy: show "Ask advisor (spends 1 of {passes_remaining})" **only when `passes_remaining !== undefined`**; disable the button when `passes_remaining === 0`. (Off-by-one: `passes_remaining` already reflects budget *after* the flag's pass; "1 of N" uses it as N directly. Disable-at-0 is a defensive/live-unreachable guard — `revise` only emits with budget remaining — label it so in the test.)
- Test: `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx`
- a11y: add the new buttons to the FIXED enumerated suite at **`src/test/a11y/components.a11y.test.tsx`** (WireStageTurn is already listed; the new Ask-advisor / escape / complete-without-signoff buttons + findings block need explicit coverage).

**Backend reality (no change):** `request_advisor` (`ControlSignal.REQUEST_ADVISOR` `protocol.py:130`, accepted `_helpers.py:2280`, handled `:3638-3717`), `complete_without_signoff` (accepted `_helpers.py:3729`), and `exit_to_freeform` (pre-dispatch `state_machine.py:586`) all already exist — B2 only renders + dispatches them.

**Step 1 — write failing render tests covering ALL SIX rows + default (RED).**
1. `revise` → findings + "Ask advisor (spends 1 of N)" + Exit; **no bare Confirm**.
2. `blocked_flagged` → findings + Exit; no budget-burning button.
3. `blocked_unavailable` → explanation + Exit.
4. `escape_unavailable` → "Complete without sign-off" present.
5. `complete` → an **actionable "Confirm wiring"** button present (reachable via `request_advisor` on a clean verdict; backend does not auto-complete, so the user must Confirm to finalize) — assert the working Confirm is present, NOT a dead-end.
6. unknown outcome → default: explanation + Exit (never empty).
7. governance: "Complete without sign-off" ABSENT on `revise` and `blocked_flagged`.
8. cost copy ABSENT when `passes_remaining === undefined` (covers initial turn AND advisor-off tutorial — tutorial-honesty); PRESENT when defined.
9. disable-at-0 (labeled defensive).

Run: `npm run test -- WireStageTurn`
Expected: FAIL — component renders only the bare Confirm today.

**Step 2 — implement the branch (GREEN).** Render findings; switch on outcome with the safe default; gate cost copy + the escape button as specified. Do not introduce any new "commit failed" detail; do not touch the Tier-3 sites.

Run: `npm run test -- WireStageTurn GuidedTurn && npm run build`
Expected: PASS.

**Step 3 — a11y + commit.** Run the a11y suite (`npm run test -- components.a11y`); then:
```bash
git add src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx \
        src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx \
        src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx \
        src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx \
        src/elspeth/web/frontend/src/types/guided.ts
git commit -m "fix(web/guided): end the wire-stage advisor dead-end

Render advisor_findings/signoff_outcome the backend already emits; gate actions on
the outcome with a safe default (Ask-advisor with disclosed cost on revise;
exit-to-freeform on every flag/block/unknown; complete-without-signoff ONLY on
escape_unavailable). Remove the bare budget-burning Confirm after a flag. Cost copy
gated on passes_remaining so the advisor-off tutorial shows none. No FLAG bypass;
sign-off runner unchanged.

Closes elspeth-7b0f75e90e (frontend dead-end).
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Definition of Done:**
- [ ] All six outcomes + a safe default branch rendered and tested; no outcome yields an empty action area
- [ ] Findings always shown on flag/block; user always sees *why*
- [ ] "Complete without sign-off" appears ONLY on `escape_unavailable` (asserted absent on `revise`/`blocked_flagged`)
- [ ] No bare Confirm on `revise`/`blocked_*`; "Ask advisor" discloses cost only when `passes_remaining` defined, disables at 0
- [ ] Advisor-off tutorial shows NO cost copy (tutorial-honesty test)
- [ ] `exit_to_freeform` reachable from every flag/block/unknown state
- [ ] new buttons covered in `src/test/a11y/components.a11y.test.tsx`
- [ ] `git diff` shows `_helpers.py:2925/3045/3231` untouched

---

## Slice B — overall Definition of Done

- [ ] B0+B2 committed
- [ ] Frontend gate green (vitest/eslint/stylelint/build)
- [ ] **Backend gate (corrected) green** — includes the route-level invariant suites: `pytest tests/unit/web/composer/guided tests/integration/web/composer/guided tests/unit/web/sessions/routes -q` (the escape/gate suites `test_request_advisor_escape.py`, `test_wire_stage_signoff_gate.py`, `test_wire_signoff_audit_and_blocked.py` live under `routes/` and MUST run)
- [ ] Backend invariants intact: `test_signoff_classifier.py` (incl. flagged-never-escapes), `test_wire_signoff_runner.py` green — runner unmodified
- [ ] Backend burn-guard (option b) surfaced to operator as a follow-up, not implemented
- [ ] `elspeth-7b0f75e90e` closed with the commit SHA
