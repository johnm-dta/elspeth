# Composer UX Review — Action List

**Audit date:** 2026-05-14
**Branch reviewed:** `RC5.2` (working tree at `/home/john/elspeth/src/elspeth/web/frontend/`)
**Method:** Source-level review of every component under `src/components/`, `App.tsx`, `Layout.tsx`, `App.css`, and `index.html`, dispatched as five parallel competency audits (freeform chat, guided mode, inspector, application shell, visual design system).
**Reviewer perspective:** Multi-competency UX (visual / IA / interaction / accessibility) against the project's stated first principles: (1) truthful representation, (2) recoverable mistakes, (3) progress legibility.

**Note on overlap with in-flight work.** The umbrella PR `feat/composer-guided-mode` (#37) is already retargeted onto RC5.2. Active work continues on `feat/composer-per-step-chat` (Phase A complete; per-step chat work in progress). Some guided-mode actions below may already be partly addressed on the in-progress branch — verify before starting.

---

## How to use this document

Each action is a checkbox. Each has:

- **Location** — file path(s) to edit
- **Effort** — rough estimate, single developer
- **Acceptance** — observable criterion that proves the action is done
- **First principle** — which of (1) truth / (2) recoverability / (3) legibility / (a) accessibility / (v) visual the action serves

Sections are ordered: **Critical → Major → Minor → Testing → Open questions**.

---

## Critical (Fix Before Public Demo)

These create fraud risk, silent data loss, or block primary accessibility paths.

### C-1. Reconcile the two `CompletionSummary` buttons

- [ ] Decide whether **"Save and exit"** and **"Drop to freeform to keep editing"** should have distinct server behaviour.
  - If yes: implement the wire difference (backend exposes two endpoints; frontend calls them with different signals).
  - If no: collapse to one button (`"Save and exit guided mode"`).
- **Location:** `src/components/chat/guided/CompletionSummary.tsx`; backend session-store handler.
- **Effort:** 30 min (merge) or ~3 hours (differentiate, including server).
- **Acceptance:** No two visible buttons in the app have wire-identical handlers; OR the comment in `CompletionSummary.tsx` no longer admits "wire-identical".
- **First principle:** (1) truth.

### C-2. Confirm before `ExitToFreeformButton` destroys in-progress turn state

- [ ] Wrap the button in `ConfirmDialog` when the current `GuidedTurn` has user-modified state that hasn't been submitted (e.g. typed-but-not-added MultiSelect custom, partly-filled SchemaForm, expanded ProposeChain card with unsubmitted answer).
- **Location:** `src/components/chat/guided/ExitToFreeformButton.tsx`; `src/components/chat/ChatPanel.tsx` (mounting point).
- **Effort:** ~1 hour.
- **Acceptance:** Clicking "Exit to freeform" with unsubmitted state opens a `ConfirmDialog`: *"Exit guided mode? Your in-progress {turn type} answer will be discarded."* Confirm proceeds; Cancel returns focus to the in-progress field.
- **First principle:** (2) recoverability.

### C-3. Surface the audit-trail truth around `RecipeOfferTurn` secret inputs

- [ ] Keep `<input type="text">` on `api_key_secret` slot (the existing comment correctly argues `type="password"` would conceal what gets recorded). **But** add a visible lock icon and microcopy: *"Secret values are written to the audit trail exactly as typed. They will appear in operator logs."*
- **Location:** `src/components/chat/guided/RecipeOfferTurn.tsx`; `App.css` for icon styling.
- **Effort:** ~30 min.
- **Acceptance:** Any slot whose name matches `*_secret` / `*_password` / `*_token` / `*_key` renders the icon + warning microcopy.
- **First principle:** (1) truth.

### C-4. Rename `ProposeChainTurn` "Accept proposal" to reflect actual capability

- [ ] As long as backend returns 501/400 for Reject / Edit-step / Ask-advisor paths (tracked: `elspeth-2c08408170`), the single button is misleading. Either:
  - Ship the deferred paths (large effort; outside this review's scope), or
  - Rename to **"Accept proposed chain (no per-step changes available)"** and remove any UI that implies reviewability.
- **Location:** `src/components/chat/guided/ProposeChainTurn.tsx`.
- **Effort:** ~15 min for the rename.
- **Acceptance:** Either Reject/Edit/Ask-advisor paths are wired, or the visible button text matches what the user can actually do.
- **First principle:** (1) truth.

### C-5. Verify `.btn-primary` and `.btn-danger` contrast in both themes

- [ ] Run an actual contrast checker (Stark, axe-core's `color-contrast`, or hand-computed) against:
  - `.btn-primary`: teal text `#14b0ae` on `rgba(20,176,174,0.2)` over `#0f2d35` (dark) and over `#fafcfc` (light).
  - `.btn-danger`: red text `#e85653` on `rgba(232,86,83,0.12)` over both backgrounds.
- [ ] If any pair is below WCAG AA (4.5:1 for normal text), solidify the button: switch to solid background + contrasting text, or raise foreground brightness.
- [ ] Extend `src/styles/tokens.test.ts` to assert the contrast ratios so future drift is caught.
- **Location:** `src/App.css` token block (L11–177); `src/styles/tokens.test.ts`.
- **Effort:** ~1 hour including the test.
- **Acceptance:** Test asserts ≥4.5:1 for primary/danger button text in both themes.
- **First principle:** (a) accessibility / (v) visual.

### C-6. Add focus trap to `CommandPalette`

- [ ] Wire up the existing `useFocusTrap` hook (used by `ConfirmDialog`, `RecoveryPanel`, `SecretsPanel`, `ShortcutsHelp`, `CatalogDrawer`) to the palette modal.
- **Location:** `src/components/common/CommandPalette.tsx`.
- **Effort:** ~30 min.
- **Acceptance:** With the palette open, Tab cycles only through the input + visible options + footer hints; Shift+Tab wraps backwards; Esc closes; focus restores to the prior element on close.
- **First principle:** (a) accessibility.

---

## Major (Fix Before Wider Public Ship)

Significant usability or accessibility issues; not demo-blocking individually but they compound.

### Guided-mode mechanics

### M-1. Add step-of-N progress to guided mode

- [ ] Backend: add an estimated remaining-steps count to the guided-turn payload (the chain solver already knows roughly how many steps are needed).
- [ ] Frontend: render `Step {n} of approximately {m}` at the top of every `GuidedTurn` dispatcher render. Use `approximately` until the backend estimate is precise.
- **Location:** server-side guided handler; `src/components/chat/guided/GuidedTurn.tsx`.
- **Effort:** ~3 hours (server contract + frontend).
- **Acceptance:** Every guided turn renders a step indicator above the prompt.
- **First principle:** (3) legibility.

### M-2. Add back/revise to guided mode

- [ ] Server: support replay-with-rollback (the user discards the last submitted answer and the previous turn is re-emitted with the prior answer pre-populated).
- [ ] Frontend: add a **"← Back to previous step"** button to every `GuidedTurn` except the first; disabled while a round-trip is in flight.
- **Location:** server-side guided handler; `src/components/chat/guided/GuidedTurn.tsx`.
- **Effort:** ~1 day (server work dominates).
- **Acceptance:** From any non-first turn, the user can return to the prior turn, see their previous answer, change it, and re-submit.
- **First principle:** (2) recoverability.

### M-3. Show user's answer in `GuidedHistory`, not the `turn_type` string

- [ ] Backend: return a `summary` field per turn (e.g. `"Selected: PostgreSQL"`, `"Added 3 custom fields: email, region, segment"`).
- [ ] Frontend: render `summary` in history rows; fall back to "(no summary)" only when backend omits it.
- [ ] Already tracked as `elspeth-611fc01d94`.
- **Location:** server-side; `src/components/chat/guided/GuidedHistory.tsx`.
- **Effort:** ~2 hours (server + frontend).
- **Acceptance:** Expanded history shows the user's actual answer, not the turn type.
- **First principle:** (3) legibility.

### M-4. Replace `SingleSelectTurn` button chips with native radios

- [ ] Use `<input type="radio">` with shared `name` attribute inside the existing `<fieldset><legend>`; style chips via `:checked` sibling selector.
- **Location:** `src/components/chat/guided/SingleSelectTurn.tsx`; corresponding CSS.
- **Effort:** ~2 hours including test updates.
- **Acceptance:** Screen reader announces "radio, 1 of N, {label}, selected/not selected"; arrow keys traverse natively within the group.
- **First principle:** (a) accessibility.

### M-5. Replace `MultiSelectWithCustomTurn` `aria-pressed` chips with native checkboxes

- [ ] Use `<input type="checkbox">` inside the same fieldset/legend; preserve the existing chip styling via sibling selectors.
- [ ] Custom-field add/remove flow unchanged.
- **Location:** `src/components/chat/guided/MultiSelectWithCustomTurn.tsx`; corresponding CSS.
- **Effort:** ~2 hours including test updates.
- **Acceptance:** Screen reader announces "checkbox, checked/unchecked"; existing focus-restore-on-remove behaviour preserved.
- **First principle:** (a) accessibility.

### M-6. Add per-field required-field errors in `SchemaFormTurn`

- [ ] On blur of an empty required field, show inline `<p role="alert" id="{field}-error">{Field name} is required</p>`; associate with the input via `aria-describedby` / `aria-invalid="true"`.
- [ ] Keep disabled-Continue as a redundant signal; do not remove it.
- **Location:** `src/components/chat/guided/SchemaFormTurn.tsx`.
- **Effort:** ~2 hours.
- **Acceptance:** A blurred-but-empty required field shows a visible error; screen reader announces the error; the field's `aria-invalid` flips to `true`.
- **First principle:** (a) accessibility / (3) legibility.

### M-7. Validate column names in `InspectAndConfirmTurn`

- [ ] Block "Apply edits" when any column name is empty or duplicates another (case-insensitive).
- [ ] Inline error per offending field.
- **Location:** `src/components/chat/guided/InspectAndConfirmTurn.tsx`.
- **Effort:** ~1 hour.
- **Acceptance:** Submit with empty or duplicate names is disabled; error message identifies the offender.
- **First principle:** (2) recoverability.

### M-8. Add widget-level in-flight state to every guided turn

- [ ] During the submit round-trip, set `aria-busy="true"` on the form, disable the submit button, and show a discrete spinner.
- [ ] On round-trip failure, surface a retry button with `role="alert"` text.
- **Location:** all seven turn components under `src/components/chat/guided/`; likely cleanest as a wrapper hook.
- **Effort:** ~3 hours.
- **Acceptance:** Submitting any guided turn visibly indicates "submitting…"; failure offers retry without losing input.
- **First principle:** (2) recoverability / (3) legibility.

### M-9. Humanise `RecipeOfferTurn` slot names

- [ ] Backend: add `display_label` (and optional `display_description`) per slot.
- [ ] Frontend: render `display_label` when present; fall back to a snake_case→Title-Case humaniser otherwise.
- **Location:** server-side recipe definitions; `src/components/chat/guided/RecipeOfferTurn.tsx`.
- **Effort:** ~2 hours.
- **Acceptance:** No raw `api_key_secret`-style identifier visible to users; humanised label shown instead.
- **First principle:** (3) legibility.

### M-10. Render `ProposeChainTurn` option values structurally

- [ ] Replace `JSON.stringify` fallback for non-scalars with nested `<dl>` or indented list rendering.
- **Location:** `src/components/chat/guided/ProposeChainTurn.tsx`.
- **Effort:** ~1 hour.
- **Acceptance:** A propose-chain with nested option objects shows them as readable hierarchies, not JSON dumps.
- **First principle:** (3) legibility.

### Freeform mode

### M-11. Add visible role labels to `MessageBubble`

- [ ] Render small-caps "You" and "ELSPETH" labels above user and assistant bubbles, matching the screen-reader prefix pattern already used in `GuidedChatHistory`.
- **Location:** `src/components/chat/MessageBubble.tsx`; `App.css` for the label class.
- **Effort:** ~1 hour.
- **Acceptance:** A screen-off user can tell who said what without relying on bubble color/position alone.
- **First principle:** (a) accessibility / (3) legibility.

### M-12. Enforce `rel="noopener noreferrer"` on markdown links

- [ ] Pass a custom `a` component to `ReactMarkdown` that adds `rel="noopener noreferrer"` and `target="_blank"`; block `javascript:` URLs explicitly.
- **Location:** `src/components/chat/MarkdownRenderer.tsx`.
- **Effort:** ~30 min including a unit test that asserts the attributes.
- **Acceptance:** Every external link rendered from markdown has `rel="noopener noreferrer"`.
- **First principle:** (1) truth (security adjunct).

### M-13. Add per-turn diff disclosure under assistant messages

- [ ] Reuse `RecoveryDiff` (already implemented for the recovery panel) inside a `<details>` collapsible labelled "What changed?" beneath each assistant message that mutated the composition state.
- **Location:** `src/components/chat/MessageBubble.tsx`; `src/components/recovery/RecoveryDiff.tsx`.
- **Effort:** ~3 hours.
- **Acceptance:** Each pipeline-mutating assistant turn surfaces its delta on demand.
- **First principle:** (3) legibility.

### Visual / shell

### M-14. Promote `--font-size-base` to 16px

- [ ] Increase the token; re-snapshot Playwright tests; verify no layout overflow at any breakpoint.
- **Location:** `src/App.css` (L tokens); affected components and snapshots.
- **Effort:** ~3 hours including snapshot review.
- **Acceptance:** Body text is 16px; long-form reading no longer requires zoom; visual regression suite green.
- **First principle:** (v) visual / (a) accessibility.

### M-15. Add `min-height: 44px` to base `.btn`

- [ ] Add the rule to the base class; remove the ~30 per-site duplicates.
- [ ] Adjust padding tokens if necessary to preserve visual balance.
- **Location:** `src/App.css`.
- **Effort:** ~2 hours.
- **Acceptance:** All buttons meet the 44×44 target without per-site overrides.
- **First principle:** (a) accessibility.

### M-16. Auto-respect `prefers-color-scheme` on first paint

- [ ] In `/theme-init.js` (already present, runs in `<head>`), default to OS preference when localStorage has no theme set.
- **Location:** `public/theme-init.js` (and ensure `useTheme` honours the same logic on first render).
- **Effort:** ~30 min.
- **Acceptance:** A first-time visitor with macOS dark mode sees dark theme without toggling.
- **First principle:** (v) visual.

### M-17. Verify the Confirm-button-default placement in `RecoveryPanel`

- [ ] Initial focus currently lands on **Discard recovery** (destructive). Recommend focusing **Apply partial draft** (productive) and making Discard the explicit secondary.
- [ ] Backdrop click currently discards — change to no-op or to "stay in dialog" so a misclick can't destroy work.
- **Location:** `src/components/recovery/RecoveryPanel.tsx`.
- **Effort:** ~1 hour including review.
- **Acceptance:** Initial focus on the productive button; backdrop click doesn't discard.
- **First principle:** (2) recoverability.

---

## Minor (Polish / Future)

Each is small and independent. Triage as time permits.

### m-1. Update `document.title` from `useHashRouter`

- [ ] Set to `${session.name} · ELSPETH` when a session is active; revert to `ELSPETH` otherwise.
- **Location:** `src/hooks/useHashRouter.ts`.
- **Effort:** 15 min.

### m-2. Add favicon and meta description

- [ ] Ship a favicon (SVG + ICO fallback); add `<meta name="description">` to `index.html`.
- **Location:** `index.html`; `public/`.
- **Effort:** 30 min.

### m-3. Remove inner `role="status"` from `ComposingIndicator`

- [ ] The parent `role="log" aria-live="polite"` already handles announcement; nested role risks double-announce.
- **Location:** `src/components/chat/ComposingIndicator.tsx`.
- **Effort:** 10 min.

### m-4. Add Esc → blur to `ChatInput`

- [ ] Esc blurs the input (do not clear; clearing is destructive).
- **Location:** `src/components/chat/ChatInput.tsx`.
- **Effort:** 15 min.

### m-5. Rename `src/styles/` to `src/tokens/`

- [ ] Directory contains only TypeScript token + tests; current name is misleading.
- **Location:** `src/styles/` → `src/tokens/`; update imports.
- **Effort:** 30 min including grep-and-replace.

### m-6. Add explicit client-side sort to `RunsView`

- [ ] Don't trust backend ordering; sort by `created_at` descending (or by status priority) defensively.
- **Location:** `src/components/inspector/RunsView.tsx`.
- **Effort:** 30 min.

### m-7. Add filter/search to runs list

- [ ] Show a search/filter input once the session crosses ~10 runs.
- **Location:** `src/components/inspector/RunsView.tsx`.
- **Effort:** ~2 hours.

### m-8. Add filter/search to run outputs

- [ ] Same pattern when an output count crosses a threshold.
- **Location:** `src/components/inspector/RunOutputsPanel.tsx`.
- **Effort:** ~1 hour.

### m-9. Transcript virtualisation

- [ ] Once a session crosses ~200 messages, swap the message list to `react-virtuoso` or similar.
- **Location:** `src/components/chat/ChatPanel.tsx`.
- **Effort:** ~4 hours.

### m-10. Move `SecretsPanel` inline styles into `App.css` tokens

- [ ] Restore token-system consistency.
- **Location:** `src/components/settings/SecretsPanel.tsx`; `src/App.css`.
- **Effort:** ~1 hour.

### m-11. Add a discoverable Help affordance

- [ ] Footer link or sidebar icon that opens `ShortcutsHelp` (currently only accessible via `?` keypress).
- **Location:** `src/components/common/Layout.tsx` or sidebar toolbar.
- **Effort:** ~30 min.

### m-12. Add a sender-side update flow for user-scoped secrets

- [ ] Currently the only way to change a user secret is delete + re-create, which splits the audit trail into two events. Add an explicit "Update value" that records a single update event.
- **Location:** `src/components/settings/SecretsPanel.tsx`; backend secrets handler.
- **Effort:** ~2 hours (depends on backend support).

### m-13. Add per-session run-status indicator in sidebar

- [ ] Beyond the active-run dot, show last-run status (✓ / ✗ / —) on each session entry.
- **Location:** `src/components/sessions/SessionSidebar.tsx`.
- **Effort:** ~1 hour.

---

## Testing Actions

These are pre-demo verification gates, independent of fix-list completion.

### T-1. Keyboard-only end-to-end walkthrough

- [ ] Start a new session, pick a template, complete a guided flow end-to-end, validate, execute, inspect outputs — **using only the keyboard**.
- [ ] Note every Tab stop, every dead-end, every place where focus disappears.
- **Effort:** ~2 hours.

### T-2. Screen-reader walkthrough

- [ ] NVDA on Windows and VoiceOver on macOS through the same path.
- [ ] Validate: chip groups announce as radio/checkbox (after M-4/M-5), step prompts announced on turn advance, live regions don't double-announce.
- **Effort:** ~3 hours (real-time pace).

### T-3. Contrast audit for every `.btn-*` variant in both themes

- [ ] Use axe-core's `color-contrast` rule or Stark.
- [ ] Add unit assertions for any pair that passes.
- **Effort:** ~2 hours.

### T-4. Colorblind simulation on Runs view and Graph view

- [ ] Use Chrome DevTools rendering panel (deuteranopia, tritanopia, achromatopsia).
- [ ] Confirm every status remains distinguishable via icon + label, not color alone.
- **Effort:** ~30 min.

### T-5. Mobile/tablet smoke test

- [ ] 768px (tablet portrait) and 600px (large phone). Verify no horizontal scroll and that key actions remain reachable.
- **Effort:** ~1 hour.

### T-6. Misclick / backdrop-click audit

- [ ] For every modal: does backdrop click destroy state? Catalogue inconsistencies (`RecoveryPanel` currently discards on backdrop; most others cancel).
- **Effort:** ~1 hour.

### T-7. Multi-tab `document.title` smoke test (post m-1)

- [ ] Open three tabs, three sessions. Can you tell them apart from the OS tab bar?
- **Effort:** 5 min.

---

## What This Review Could Not Assess

These require a running instance and are flagged as test-required, not as findings:

- Actual rendered contrast (depends on theme + system rendering)
- Animation feel at 60Hz vs 120Hz monitors
- Real screen-reader behaviour (NVDA/VoiceOver implementation quirks)
- Performance under load (long transcripts, large pipelines, slow LLMs)
- Touch-screen behaviour (chip-group fat-finger collisions)
- Cross-tab session-sync edge cases

---

## Open Questions for the Operator

These need a human decision before some actions can be planned.

1. **C-1 split or merge?** Should `CompletionSummary` have two semantically distinct exit paths (save-and-close vs save-and-continue) or one?
2. **M-2 backend scope.** Is replay-with-rollback in scope for the current demo prep, or should "back" be limited to "exit guided mode → restart"?
3. **C-4 ProposeChain.** Is shipping Reject / Edit / Ask-advisor on the demo critical path? If not, the rename is sufficient.
4. **m-9 virtualisation.** Have any demo sessions actually crossed the 200-message threshold? If not, defer.
5. **T-5 mobile.** Is the demo audience expected to use mobile? If desktop-only, mobile findings drop in priority.

---

## Already-Tracked Observations

These overlap with existing filigree observations. Cross-reference when filing issues:

- `elspeth-611fc01d94` — GuidedHistory shows turn_type not answer (action M-3).
- `elspeth-2c08408170` — ProposeChainTurn deferred actions (action C-4 / M-?).
- `elspeth-5e905f3c9d`, `elspeth-611fc01d94`, `elspeth-2c08408170`, `obs-0a1002de6d` — Phase 7 follow-ups; review for overlap.
- `elspeth-5ea21f94af` — focus-on-step-advance (related to M-1's UX promise).
- `elspeth-06854f0842` — Phase 10 follow-up (review for relevance).
- `elspeth-f626607b13` — RecipeOfferTurn editable slots (related to C-3).

---

## Suggested Sequencing

If acting on this list, a sane order:

1. **Day 1:** C-5 (contrast), M-15 (button min-height), M-14 (16px body) — visual foundation. These will surface layout fallout that subsequent fixes need to handle.
2. **Day 2:** C-1, C-2, C-3, C-4, C-6 — the truth/recoverability cluster. Each is small; together they remove the "the UI is lying" risks before any external eyes.
3. **Day 3–4:** M-4, M-5, M-6, M-7, M-8 — the guided-mode semantic correctness pass. Mechanical work; high accessibility ROI.
4. **Day 5:** M-1, M-2, M-3 — the guided-mode legibility pass. Highest user-visible win; requires backend changes so plan accordingly.
5. **Ongoing:** Minor items as polish; testing actions before each demo rehearsal.
