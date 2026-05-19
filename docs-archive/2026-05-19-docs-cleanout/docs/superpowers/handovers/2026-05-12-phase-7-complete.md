# Handover — Composer Guided Mode (Phase 7 complete; Phase 8 onward)

## TL;DR

Phase 7 is **complete and green**. The frontend widget surface is closed: six leaf turn-type widgets (`SingleSelectTurn`, `InspectAndConfirmTurn`, `MultiSelectWithCustomTurn`, `SchemaFormTurn`, `ProposeChainTurn`, `RecipeOfferTurn`), the `GuidedTurn` dispatcher routing all six, plus three peer components (`ExitToFreeformButton`, `GuidedHistory`, `CompletionSummary`). Twenty commits on top of `cc074420` deliver +193 frontend tests (211 → 404 cumulative), no critical or important issues, four cross-layer follow-ups durably tracked in Filigree, three deferred-polish observations, and zero unresolved blockers.

Phase 8 (ChatPanel integration) wires these widgets into the freeform-vs-guided mode discriminator established in Phase 6. Phase 9 lands the Playwright E2E + demo-SLA assertions. Phase 10 closes documentation and CHANGELOG.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode` (62 commits ahead of `RC5.2` as of Phase 7 close; top commit `acf712d2`)
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — Python-version mismatch corrupts `enforce_tier_model.py` results.
- **Frontend toolchain:** `npm` (NOT pnpm/yarn). `vitest` is the test runner. The canonical typecheck is `npx tsc -p tsconfig.app.json --noEmit` — NOT `npx tsc --noEmit` at the root (which trips on pre-existing tsconfig composite-project misconfiguration).
- **ESLint is broken** on this worktree due to v9 flat-config migration debt — not Phase 7's job to fix. Don't run `npx eslint src/`; rely on vitest + tsc for the per-task gate.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` (Phase 8 starts line ~4457)
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (Phase 8 relevant: §7.2 ChatPanel discriminator)
- **Inbound handover:** `docs/superpowers/handovers/2026-05-12-phase-6-complete.md` — the Phase 7 brief; references back to Phase 5 close.

## What's delivered in Phase 7

Twenty commits on top of Phase 6 closure (`cc074420` + docs `d60f7797`):

### Per-task commit summary

| Task | Component | Commits | Notes |
|------|-----------|---------|-------|
| 7.2 | `SingleSelectTurn` (template) | `6703e47d` + `ee125918` (fix-up) | Establishes chip-group conventions: `<fieldset>`+`<legend>`, `aria-pressed`, `useId()` per-instance scoping, shared `nullResponse()` fixture, prefers-reduced-motion shared block. |
| 7.3 | `InspectAndConfirmTurn` | `8c0427f3` + `afdbc245` (fix-up) | Establishes single-nullable-struct state pattern (collapsed `editorState: { columns } \| null`); focus management on view toggle via `firstRunRef`; aria-described warnings region relying on parent ChatPanel `aria-live`. |
| 7.4 | `MultiSelectWithCustomTurn` | `0c5baeff` + `539a21df` (fix-up) | Reuses chip-group family with `aria-pressed` toggle visual extension. **Escape button deferred** — backend wire-shape contradiction (see Filigree follow-ups below). Custom-chip removal with focus restoration. |
| 7.5 | `SchemaFormTurn` (largest widget) | `b0b88b8c` + `acd6c7f2` (self-fix-up) + `9548ef3f` + `283726d3` (3 review fix-ups) | Auto-generates form fields from Pydantic JSON Schema. Required-vs-advanced split with `aria-controls` resolving cross-state via `hidden` attribute (the canonical fix that GuidedHistory inherits). JSON-fallback for non-scalar types. Tier 2 numeric coercion (`""`/NaN → `null`). |
| 7.6 | `ProposeChainTurn` (Accept-only) | `f27bf2e5` + `e54372fc` (fix-up) | Card-list layout introducing `guided-propose-step-card` (later shared with 7.7). **3 of 4 buttons deferred** — backend Step-3 handler partial (see follow-ups). Plugin name as `<h3>` for SR landmark navigation. |
| 7.7 | `RecipeOfferTurn` | `375d5df2` | Card layout reusing `guided-propose-step-card`. **Wire-shape correction:** plan body said `edited_values: null` for Apply, backend at `routes.py:1965-1967` requires the recipe-data echo. Documented at three sites. |
| 7.1 | `GuidedTurn` dispatcher | `a0fdea4b` + `48b485c7` (nullResponse alignment) | Routes six turn types via switch + exhaustiveness assertion. **Option A** (per-case `as` casts) chosen over Option B (discriminated union) — cascade rationale documented; refactor candidate filed as observation. |
| 7.8 | `ExitToFreeformButton` | `e101b9ff` + `c6ed5795` (tautology removal) | Smallest widget (36 LOC). Self-contained; reads `exitToFreeform` from store directly; no-confirmation contract pinned with negative-space tests for `alertdialog`/`dialog`/`/cancel/i`. First widget to read action directly from store. |
| 7.9 | `GuidedHistory` (step+type+emitter only) | `904bb4f6` + `1a578f5b` (tracker swap) | Collapsible read-only list. Inherits the 7.5 I2 `aria-controls`+`hidden` cross-state pattern (the canonical reason 7.5 I2 was fixed). **Rich summaries deferred** — backend `TurnRecordResponse` is hash-only by design (audit-trail integrity). |
| 7.10 | `CompletionSummary` (FINAL) | `acf712d2` | Outer/inner component split for React hooks-rules compliance. Two buttons with **identical wire** semantics (verified at `state_machine.py:382-398`); UX-distinct labels with the wire-identity invariant pinned by tests. `prism-react-renderer` integration matches `YamlView.tsx` pattern. |

### Gate state at close

- **Vitest:** **404 / 404 pass** (39 test files; +193 from Phase 6 baseline of 211)
- **`npx tsc -p tsconfig.app.json --noEmit`:** clean
- **Backend pytest** (narrow guided + sessions slice): unchanged from Phase 6 close (Phase 7 is frontend-only — backend file modifications are zero)
- **Backend mypy** on `src/elspeth/web/composer/` + `src/elspeth/web/sessions/routes.py`: clean (no Phase-7 changes)
- **ESLint:** not run — broken on this worktree (v9 flat-config migration debt); pre-existing infra issue unrelated to Phase 7

### Per-task review iteration count (data point for future planning)

| Task | Implementer dispatches | Why iteration was needed |
|------|------------------------|---------------------------|
| 7.2 | 2 (initial + fix-up) | Code-quality reviewer caught template-quality items: missing `prefers-reduced-motion`, document-global IDs (`useId()` introduction), dead-defence keydown duplication, `nullResponse()` extraction to shared fixture. Critical to fix in template before 5 siblings copied. |
| 7.3 | 2 (initial + fix-up) | Two-state encoding (`editing` boolean + `editedColumns` array) collapsed to single nullable struct (per CLAUDE.md "make-illegal-states-unrepresentable"); focus management on view toggle established the pattern; warnings `aria-live` relationship documented. |
| 7.4 | 2 (initial + fix-up) | **Operator-adjudication mid-task:** backend mismatch on escape-button wire shape required Option C (drop the button + queue follow-up issue). Then standard fix-up: tracker-ID citation, focus-on-removal a11y, distinctness test rigor. |
| 7.5 | 4 (initial + self-fix-up + review fix-up + I2 completion fix-up) | Most complex widget: implementer caught their own defensive `??` and tautological `errorId` in the second commit. Review surfaced Tier 2 numeric coercion bug (empty optional numeric → `""` to numeric slot) and `aria-controls`-references-non-existent-element bug. The I2 completion required a TDD Red→Green verification cycle that the reviewer independently confirmed by reverting and re-running. |
| 7.6 | 2 (initial + fix-up) | **Operator-adjudication mid-task:** backend Step-3 handler is incomplete (consumes only `chosen: ["accept"]`); 3 of 4 planned buttons dropped + tracked. Then tracker-ID swap (observation → promoted issue) + heading semantics (`<span>` → `<h3>`). |
| 7.7 | 1 (initial only) | Cleanest review pass: implementer corrected the plan-body's `edited_values: null` Apply submit to send the verified backend-required echo. Cross-task discipline. |
| 7.1 | 2 (initial + nullResponse alignment) | Convention nit only: forwarding test used flat object literal instead of `...nullResponse()` spread-FIRST. Mechanical 5-line refactor. |
| 7.8 | 2 (initial + tautology removal) | One genuine improvement: dropped the 9th test asserting "two simultaneous instances render distinct DOM nodes" (always true in React). Tests should earn their keep. |
| 7.9 | 2 (initial + tracker swap) | Mid-task scope reduction (rich summaries → step+type+emitter only); fix-up swapped observation citation to promoted Filigree issue. |
| 7.10 | 1 (initial only) | Wire-identity decision documented thoroughly; outer/inner hooks-rules split applied cleanly. No fix-up needed. |

Pattern: every Phase 7 task either landed cleanly or needed exactly one iteration cycle. SchemaFormTurn (4 cycles) is the outlier — combinatorial surface (5 field types × 2 visibility states × prefill precedence × focus management × JSON-fallback) genuinely warrants the iteration. Three tasks (7.7, 7.10) cleared on first review pass — convention internalization paid off in the second half of the phase.

## End-to-end frontend capability (post Phase 7)

Phase 7 lands the full widget surface but does NOT integrate it into ChatPanel. After Phase 7:

1. All six per-`TurnType` React widgets exist with matching prop signatures: `{ payload: <PerTypePayload>; onSubmit: (body: GuidedRespondRequest) => void }` (sync onSubmit; widgets construct full 6-field wire bodies).
2. `GuidedTurn` dispatcher routes by `turn.type` to the matching widget; exhaustiveness assertion catches future TurnType additions at compile time.
3. `ExitToFreeformButton` is rendered as a peer in ChatPanel (Phase 8 work); it reads `exitToFreeform` from the store and emits a parameterless action call.
4. `GuidedHistory` is a peer that consumes `guidedSession.history` (the array of `TurnRecord` from Phase 6 store fields).
5. `CompletionSummary` is a peer that renders when `terminal.kind === "completed"` and shows the YAML preview with two freeform-transition buttons.
6. All widgets pass full `tsc -p tsconfig.app.json --noEmit` clean and 404/404 vitest tests.

The chat surface (`ChatPanel.tsx`) is **unchanged**; guided mode is reachable only programmatically until Phase 8 wires it into the mode discriminator described in spec §7.2.

## What's pending — Phase 8 onward

Phase 8 wires the dispatcher + peer components into `ChatPanel.tsx` via the top-level mode discriminator. Plan body starts at line ~4457 (`Task 8.1: ChatPanel mode discriminator`).

The integration sketch (per spec §7.2):

```tsx
if (guidedSession && !guidedSession.terminal) {
  return (
    <div className="chat-panel guided-mode">
      <GuidedHistory history={guidedSession.history} />
      {guidedNextTurn && (
        <GuidedTurn turn={guidedNextTurn} onSubmit={(body) => respondGuided(body)} />
      )}
      <ExitToFreeformButton />
    </div>
  );
}
if (guidedSession?.terminal?.kind === "completed") {
  return <CompletionSummary terminal={guidedSession.terminal} />;
}
return <ExistingChatPanelBody />;
```

Phase 9 lands Playwright E2E + the demo-SLA assertion. Phase 10 closes docs + CHANGELOG.

## Cross-layer follow-ups (active Filigree issues)

Three durably-tracked issues for Phase 5 backend completion work that surfaced during Phase 7 widget implementation:

| Issue | Surface | Phase impact |
|---|---|---|
| **`elspeth-5e905f3c9d`** | `MultiSelectWithCustomTurn` escape button — wire shape (`{edited_values: {schema_mode, required_fields}}` per plan) contradicts backend `_advance_step_2` (reads `outputs[]` shape). Widget defers rendering. | Phase 9 demo path requires Step 2 escape if user wants to skip required-fields collection. Resolve before Phase 9 E2E. |
| **`elspeth-2c08408170`** | `ProposeChainTurn` Step-3 backend handler completion (Reject + per-step Edit + Ask advisor). Backend `routes.py:2030-2137` consumes only `chosen: ["accept"]` (success) and `["reject"]` (501 stub). 3 of 4 planned buttons absent. | Phase 9 E2E cannot exercise Reject/Edit/Ask-advisor flows until backend lands. Hand-built path test (Task 9.2) will need the Edit button. |
| **`elspeth-611fc01d94`** | `GuidedHistory` rich step summaries — wire `TurnRecordResponse` is hash-only by audit-trail design. Decision needed: wire extension (`summary` field) or payload-fetch endpoint. | Phase 8 ChatPanel integration ships hash-only history; Phase 9 demo can describe but not show "Step 1: CSV with cols [price, qty]" until this lands. |

All three are likely best resolved as a single bundled "Phase 5 backend completion" task with three sub-deliverables. Phase 8 dispatch should consider whether to surface this bundling decision before starting widget integration.

## Deferred-polish observations (non-blocking, 14-day expiry unless promoted)

| Observation | Description |
|---|---|
| **`elspeth-obs-510a4fbdeb`** | `TurnPayload` discriminated-union refactor (Option B). Eliminates 6 `as` casts in `GuidedTurn.tsx`. Cascade cost: test fixtures in `client.guided.test.ts:71` and `sessionStore.guided.test.ts:54-56` use shorthand `payload: {options: [...]}` literals that need rewriting. Defer to post-Phase-8 cleanup. |
| **`elspeth-obs-0a1002de6d`** | Phase 8 ChatPanel must suppress persistent `ExitToFreeformButton` when `terminal.kind === "completed"` — `CompletionSummary`'s "Drop to freeform to keep editing" button has wire-identical behavior; rendering both produces three identical-action buttons. Phase 8 implementer alert. |
| **`elspeth-obs-f9e991f517`** | `useNonInitialEffect` hook extraction candidate. `firstRunRef` initial-mount-skip pattern landed in 7.3/7.4/7.5 (rule of three met). Cross-task reviewer recommended deferring past Phase 8 because 7.4's per-instance focus-target Map complicates a clean hook shape. |

The first two should remain visible to the Phase 8 implementer; the third is a polish item. Promote any to durable issues as needed.

## Phase 8 starting brief

Read this in conjunction with the plan file's Phase 8 section.

### Convention reminders inherited from Phase 7

1. **Widget prop signature:** every leaf widget accepts `{ payload: <PerTypePayload>; onSubmit: (body: GuidedRespondRequest) => void }`. Sync onSubmit (NOT Promise — widgets construct bodies; the store action awaits). The dispatcher's prop is `{ turn: TurnPayload; onSubmit: (body: GuidedRespondRequest) => void }`.
2. **Store-action delegation:** peer components (`ExitToFreeformButton`) read store actions directly via `useSessionStore((s) => s.actionName)`. Do NOT construct wire bodies in peer components — the store action owns the wire shape.
3. **Wire-shape verification convention:** before locking any widget's wire body to the plan body, verify against the actual backend handler (`state_machine.py:_advance_step_*` or `routes.py`). Three of the four cross-layer gaps in Phase 7 surfaced because the plan was written aspirationally before backend implementation pinned the contracts. Phase 8 ChatPanel integration touches the store action wiring — same discipline applies.
4. **`aria-controls` + `hidden` pattern:** any collapsible region MUST render the id-bearing container unconditionally with `hidden={!expanded}`; only children are gated. The `{expanded && <div id={...}>...}` pattern is BROKEN — `aria-controls` becomes a dangling reference in collapsed state. See `SchemaFormTurn.tsx:600` and `GuidedHistory.tsx:112-118` for the canonical pattern.
5. **Suppress `ExitToFreeformButton` when `terminal.kind === "completed"`** — observation `elspeth-obs-0a1002de6d` flags this; otherwise three identical-action buttons appear. The discriminator JSX in spec §7.2 should add an explicit branch.
6. **`<h3>` for primary entity headings:** Task 7.6 M3 set the precedent (plugin name → `<h3>`). RecipeOfferTurn (recipe_name) and CompletionSummary follow. ChatPanel may want an `<h2>` parent above the turn surface for SR landmark hierarchy.
7. **`useTheme()` for Prism integration:** `CompletionSummary.tsx:67-68` matches `YamlView.tsx:164`. Any new Prism use should follow the convention.
8. **Distinctness pin:** any element-level IDs use `useId()` per-instance scoping. Tests assert `expect(document.getElementById(id0)).not.toBe(document.getElementById(id1))` — node identity, not just string distinctness (Task 7.4 I4 lesson).

### Plan-vs-reality drift expected in Phase 8

The plan body has shown drift in every prior phase. Expect the same in Phase 8:

- The plan likely shows `GuidedTurn`/`ExitToFreeformButton` import paths that don't match the actual `@/components/chat/guided/...` structure. Override with verified paths.
- The plan likely shows `respondGuided` returning `Promise<void>` (correct) but the dispatcher wrapper as `(body) => respondGuided(body)` — make sure the void-promise floating warning is silenced (e.g., `(body) => void respondGuided(body)`).
- The plan may not call out the `terminal.kind === "completed"` branch having to suppress ExitToFreeformButton (per `elspeth-obs-0a1002de6d`).

The implementer dispatch brief for each Phase 8 task should pre-emptively override these.

### First action when you start Phase 8

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline cc074420..HEAD                          # expect 20 commits, top = acf712d2
cd src/elspeth/web/frontend
npm test -- --run                                          # expect 404 / 404 pass
npx tsc -p tsconfig.app.json --noEmit                      # expect clean
```

If any baseline gate fails, **stop and investigate** before starting Phase 8.

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** No Alembic; no migration scripts; no `from_dict` backward-compat defaults. (Phase 7 didn't touch the DB — but the constraint applies if Phase 8 finds itself there.)
- **Default to worktree.** Stay here; do not work in `/home/john/elspeth` main.
- **No git stash.** Commit work to a branch if preservation is needed.
- **No calendar shipping commitments.** ELSPETH ships work-until-done.
- **Correctness beats performance always.**
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
- **`any` is forbidden in TypeScript.** Use `unknown` for opaque, closed unions or interfaces otherwise.
- **No optimistic updates.** Server is authoritative.
- **snake_case wire field names in interfaces; camelCase store-internal fields.** Established in Phase 6.
- **ESLint is broken on this worktree.** Don't try to fix it inside Phase 8 unless a separate task is opened for the v9 flat-config migration.
- **No PR-open at the end of Phase 7.** This handover + project memory entry close Phase 7. PR is deferred per project convention; the branch will accumulate Phases 8-10 before PR-open consideration.
