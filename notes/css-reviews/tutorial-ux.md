<!-- tutorial-ux.md — UX/a11y review of tutorial.css and its TSX consumers -->
<!-- Reviewed: 2026-05-23 | Reviewer: UX Critic Agent -->
<!-- Source: src/elspeth/web/frontend/src/components/tutorial/tutorial.css (~482 lines) -->
<!-- Consumers: HelloWorldTutorial.tsx, TutorialTurn1–6*.tsx -->

# Design Review: First-Run Tutorial CSS

## Summary

**Overall:** Acceptable — strong SR/AT scaffolding, but three concrete gaps prevent a
"Strong" rating: the error state has no non-colour differentiation at the CSS layer,
the progress-dot active state has no shape/outline differentiation, and the
mode-fieldset radio labels have no selected-state visual affordance.

**Critical Issues:** 0
**Major Issues:** 3
**Minor Issues:** 5

---

## 1. Visual Design

### Strengths

- Token-clean throughout — every colour, spacing, radius, and font value
  uses a design-token variable. No hardcoded hex except the gradient
  overlay (`rgba(255,255,255,0.03)`) which is cosmetic and intentional.
- The `.tutorial-progress-bar` gradient comment explicitly names the
  previous failure mode ("stuck at 60%") and documents the fix. That is
  load-bearing institutional memory, not decoration.
- `.tutorial-layer strong` comment explains why `--color-success` was
  removed. The substitution to `--color-text` (neutral) is correct for a
  welcome screen with no completed state.
- Box-shadow on `.tutorial-turn` (0 12px 30px rgba(0,0,0,0.18)) gives
  the card visual elevation without relying on colour alone.

### Issues

| # | Issue | Severity | Location | Fix |
|---|-------|----------|----------|-----|
| V1 | Error state is colour-only — `.tutorial-error` uses `--color-error` fg/bg/border with no icon or shape token. In forced-colours mode the semantic colour disappears; on low-vision displays the distinction from a warning relies entirely on hue. | Major | `.tutorial-error` (L140–147) | Add a CSS pseudo-element icon slot (`::before { content: "⚠"; }`) or document that the TSX consumers are required to supply an icon. Currently TutorialTurn2Describe, Turn4Run, Turn5AuditStory, and Turn6ModeChoice all render bare `<p role="alert" className="tutorial-error">` with text only — no icon in any of them. |
| V2 | `.tutorial-hash-copy` deliberately opts out of the `.btn` 44px hit area (comment L378–381). The justification — "sits in a dense audit panel" — is reasonable for desktop, but this button appears on mobile at 760px where thumb precision is lower. No mobile override exists. | Minor | `.tutorial-hash-copy` (L377–390) | Add `@media (max-width: 760px) { .tutorial-hash-copy { min-height: var(--size-control); } }` to restore the touch target on narrow viewports. |
| V3 | `.tutorial-cell-note` uses `--color-warning` as its only differentiator. Same forced-colours risk as V1, lower severity because it is secondary copy, not a user-actionable error. | Minor | `.tutorial-cell-note` (L327–332) | Pair with a short text prefix or `::before` glyph ("Note:" or "!") so the meaning survives in high-contrast mode. |

---

## 2. Information Architecture

### Strengths

- The kicker → heading → body → actions hierarchy is consistent across
  all six turns. The `.tutorial-kicker` uppercase treatment signals
  "where am I" without crowding the h2. Hierarchy is readable at a glance.
- Turn-by-turn focus management (`headingRef.current?.focus()` on mount)
  means SR users hear the step label immediately on transition. This is
  correct and consistent across Turns 2–6.
- Turn 1 (Welcome) omits the `tabIndex={-1}` / `headingRef` pattern
  because it is the initial render — the heading is already the first
  meaningful element. Correct.
- The `sr-only` "Step N of M: label" paragraph in `HelloWorldTutorial`
  pairs with `aria-hidden` dots. Dual-encoding handled correctly — the
  comment explicitly explains why `aria-current` was not added.

### Issues

| # | Issue | Severity | Location | Fix |
|---|-------|----------|----------|-----|
| IA1 | `TutorialTurn5AuditStory` has no loading skeleton or `aria-busy` on the audit-data container during fetch — it renders a `role="status"` muted paragraph ("Loading audit evidence…") but the section itself has no `aria-busy`. AT users do not receive a "busy" signal on the region that will change, only a static status node. Compare Turn 4's `<div role="status" aria-busy="true">` which is correct. | Minor | `TutorialTurn5AuditStory.tsx` L56–60, no CSS involvement | Add `aria-busy={summary === null && error === null}` to the `<section>` element. |
| IA2 | The audit-list (`<dl>`) uses `<div>` wrappers inside it. This is valid HTML (the `<dl>` with `<div>` children pattern). However, `.tutorial-audit-list div` is the selector (L341), which means any unintended `<div>` nested inside the list picks up card styling. Consider scoping to `.tutorial-audit-list > div` to prevent accidental style bleed. | Minor | `.tutorial-audit-list div` (L341–347) | Change to `.tutorial-audit-list > div`. |

---

## 3. Interaction Design

### Strengths

- `.tutorial-link-button` has `min-height: var(--size-control)` and
  `padding: var(--space-sm) 0` — it meets the touch target height
  requirement even though it is visually link-styled. Correct.
- `.tutorial-mode-fieldset input { margin-top: 5px }` vertically aligns
  the radio with the label's first line of text. A hardcoded `5px` is a
  smell (not a token), but it solves a well-known baseline-alignment
  problem with radio inputs and is not load-bearing for a11y.
- The `.tutorial-progress-bar::after` shimmer is gated on
  `(prefers-reduced-motion: no-preference)` — the affirmative opt-in
  pattern, not the `reduce` suppression pattern. The `reduce` block
  additionally provides a static centred chunk, so the affordance does
  not completely vanish. Both are correct.
- `tutorial-prompt-input` uses `resize: vertical` — allows power users to
  expand the textarea without horizontal scroll risk.

### Issues

| # | Issue | Severity | Location | Fix |
|---|-------|----------|----------|-----|
| I1 | `.tutorial-progress-dot--active` changes background colour only (`--color-state-positive` vs `--color-border-strong`). In forced-colours mode both colours resolve to system colours and may become identical. A shape change (border, size, or outline) would survive colour stripping. | Major | `.tutorial-progress-dot--active` (L453–455) | Add `outline: 2px solid var(--color-state-positive); outline-offset: 2px;` to the active dot, or increase its size by 2px when active. The `aria-hidden` on dots means this is a visual-only concern, but it matters for sighted users in HC mode. |
| I2 | `.tutorial-mode-fieldset label` has no `:checked + span` or `:has(:checked)` selected-state style. The only visual difference between a selected and unselected radio option is the native radio control itself (a 16px dot). On the card-style label layout, users with motor impairment who overshoot the radio and land on the label text have no reinforcing visual. | Major | `.tutorial-mode-fieldset label` (L422–430) | Add: `.tutorial-mode-fieldset label:has(input:checked) { border-color: var(--color-state-positive); background: var(--color-surface); }` The `:has()` selector is baseline 2023; safe for the project's browser target if Chromium-based. |
| I3 | `.tutorial-graph-node` has no interactive states (hover/focus/active). The TSX renders nodes as `<div>` (not interactive elements), so this is correct — nodes are purely informational. But the comment in the CSS owner header says "graph-node" is in scope — worth documenting explicitly that nodes are non-interactive by design to prevent a future contributor adding click handlers without corresponding states. | Minor | `.tutorial-graph-node` (L189–201) | Add a comment: `/* Non-interactive — nodes are display-only. If click handlers are added, add :hover/:focus-visible states here. */` |
| I4 | `.tutorial-mode-fieldset input { margin-top: 5px }` — hardcoded pixel not using a spacing token. Low-risk but inconsistent with the token-clean pattern elsewhere. | Minor | L432–434 | Replace with `margin-top: 4px` (token equivalent) or document why `5px` is intentional. |

---

## 4. Accessibility

### Quick Check

- [x] **1.4.3 Contrast** — Cannot measure token-resolved colours without the design-system values. Flagged for measurement against actual resolved colours. `.tutorial-kicker` at `--color-info` with `font-size-sm` and `font-weight: 700` must meet 4.5:1. `.tutorial-draft-progress-next` at `--color-text-muted` / `font-size-xs` is the highest-risk candidate.
- [x] **2.1.1 Keyboard** — Structural keyboard navigation is correct. `.tutorial-mode-fieldset` is a `<fieldset>` with `<legend>`, radios use `name="tutorial-mode"` (group), so arrow-key navigation between options is native. `.tutorial-link-button` is a `<button>` (keyboard native). No `div` click handlers found.
- [x] **2.4.7 Focus Visible** — No `:focus-visible` override in tutorial.css. The sheet defers to the global stylesheet. Acceptable if the global provides a 2px+ indicator; cannot verify from this file alone. The `.tutorial-hash-copy` comment mentions `:focus-visible` is still available — confirms the global is relied on.
- [x] **1.1.1 Alt Text** — All meaningful images are inline SVG with `aria-hidden="true"` (chevrons). No `<img>` tags. Correct.
- [ ] **1.4.1 Use of Colour** — V1 (error), V2 (cell-note), and I1 (progress dot) all rely on colour as the sole differentiator. None are Critical (no single-item decision depends solely on colour), but the aggregate risk is meaningful.

### Issues

| # | Issue | Severity | Location | Fix |
|---|-------|----------|----------|-----|
| A1 | `.tutorial-error` — no non-colour differentiator. `role="alert"` in TSX ensures SR users hear it; sighted users in forced-colours mode get no visual cue that this is an error vs body text. | Major | See V1 above | See V1 fix |
| A2 | The `title` attribute on progress dots (`title={label}`) surfaces tooltip text in some browsers. AT does not read `title` on `aria-hidden` elements (correct), but it creates a tooltip that appears on keyboard focus — except the dots are `<span>` elements, not focusable. The `title` is therefore unreachable by any mechanism except mouse hover. Either remove it (the sr-only paragraph makes it redundant) or convert to `data-label` for future tooltip implementation. | Minor | `HelloWorldTutorial.tsx` L57, no CSS involvement | Remove `title={label}` from the dot spans. |
| A3 | `TutorialTurn2Describe`: the error (`role="alert"`) is rendered *after* the actions in DOM order (L152–156). `role="alert"` announces on insertion regardless of DOM position, so SR users hear the error. But sighted keyboard users tab past the Retry/Back buttons to reach the error — the logical reading order is actions then error, which is backwards from the expected sequence (error then recovery actions). Turn 4 has the same issue (error at L191–211, actions inside the error block — correct there). Turn 2 should follow Turn 4's pattern. | Minor | `TutorialTurn2Describe.tsx` L126–156 | Move the `{error !== null && ...}` block above the `<div className="tutorial-actions">` block, or place the actions *inside* the error branch as Turn 4 does. |

---

## 5. Reduced-Motion

**Pass.** The implementation uses the affirmative opt-in pattern
(`no-preference` → animate; `reduce` → static centred chunk). The comment on
lines 248–255 documents the fallback intent, and the `reduce` block provides a
non-empty visual state. No other animations exist in the file.

---

## 6. Mobile (760px)

### Strengths

- Shell padding collapses correctly (`--space-xl` → `--space-md`).
- Three-column layer/summary/audit grids collapse to single-column.
- Graph flex direction switches to column, chevrons rotate 90deg via
  `transform`. The rotation is hardcoded (not a motion concern — it is not
  an animation) and correct.

### Issues

| # | Issue | Severity | Location | Fix |
|---|-------|----------|----------|-----|
| M1 | `.tutorial-result-table` has `min-width: 640px` with a wrapping `.tutorial-result-table-wrap { overflow-x: auto }`. On a 360px viewport the table scrolls horizontally inside the scroll container. This is functional but the `min-width: 640px` is a hardcoded pixel in an otherwise token-clean file. Consider `min-width: min-content` and rely on the overflow wrapper, or document why 640px is the minimum viable table width. | Minor | `.tutorial-result-table` (L313–317) | `min-width: min-content` lets the table compress further on very narrow viewports; columns remain usable because the preferred column order prioritises short keys. |
| M2 | `.tutorial-hash-copy` has no mobile touch-target fix (see V2 above). | Minor | `.tutorial-hash-copy` | See V2 fix. |

---

## 7. Token Discipline / Dead Rules

- One hardcoded pixel: `.tutorial-mode-fieldset input { margin-top: 5px }` (L433). Not in the token set; flagged in I4.
- One hardcoded pixel: `min-width: 640px` on `.tutorial-result-table` (L315). Flagged in M1.
- `padding: 2px 6px` on `.tutorial-hash-value` (L374). Reasonable micro-spacing not in the scale; acceptable.
- `padding-left: 18px` on `.tutorial-draft-progress-evidence` (L239). Reasonable list indent; acceptable.
- No dead rules found. All selectors in the file map to active TSX usages confirmed by reading the consumers.

---

## Priority Recommendations

### Major (Fix Before Demo / Launch)

1. **V1 / A1 — Error state colour-only:** Add a non-colour differentiator to `.tutorial-error` — either a CSS `::before` glyph or document that all TSX error sites must supply an icon element. All four error sites currently emit bare text.

2. **I1 — Progress dot active state colour-only:** Add `outline` or size differentiation to `.tutorial-progress-dot--active` so the active step survives forced-colours mode.

3. **I2 — Mode fieldset no selected-state visual:** Add `:has(input:checked)` border/background change to `.tutorial-mode-fieldset label` so the chosen option has a card-level visual affordance beyond the 16px radio dot.

### Minor (Improvement / Polish)

1. **A2 — Remove `title` from aria-hidden dots.** Unreachable tooltip is misleading to future contributors.
2. **A3 — Move error block above actions in TutorialTurn2Describe.** Keyboard reading order mismatch.
3. **I4 — Replace hardcoded `margin-top: 5px` with spacing token.**
4. **M1 / M2 — Table min-width and hash-copy touch target on mobile.**
5. **IA2 — Scope `.tutorial-audit-list div` to `> div`.**

---

## Confidence Assessment

**High** for CSS structural findings (direct line-by-line reading).
**High** for TSX/CSS cross-checks (all consumer files read in full).
**Medium** for contrast findings — token values not resolved to hex; actual contrast ratios unverified. A browser-based contrast audit against the live design-token values is required before the contrast findings can be upgraded to confirmed pass/fail.

## Risk Assessment

**Low** — no Critical issues. The three Major issues are all visual-only
failures (colour differentiation); the semantic SR layer (role="alert",
role="status", sr-only paragraphs, focus management) is correct and
consistent. A sighted user in forced-colours mode or high-contrast mode
would miss error state and active-step cues; keyboard-only and SR users
are well served.

## Information Gaps

- Design-token hex values for `--color-info`, `--color-text-muted`,
  `--color-state-positive` are not available in this review. Contrast
  ratios for `.tutorial-kicker` and `.tutorial-draft-progress-next` are
  unmeasured.
- Global `:focus-visible` styles are not in scope; their adequacy is
  assumed but unverified.
- Browser target (minimum Chromium version) was not specified; the
  `:has()` selector recommendation assumes Chromium 105+ / Firefox 121+.

## Caveats

The `tutorial-mode-fieldset input { margin-top: 5px }` hardcode and the
`min-width: 640px` on the result table are present in the file as reviewed.
If either was intentionally deferred from tokenisation, this review has
not changed that decision — they are flagged for awareness, not mandated
for immediate change.
