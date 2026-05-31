# Design Review: inspector.css

**File:** `src/elspeth/web/frontend/src/components/inspector/inspector.css`
**Consumer scope:** `HeaderVersionSelector.tsx`, `GraphView.tsx`
**Reviewer:** UX Critic Agent
**Date:** 2026-05-23

---

## Summary

**Overall:** Needs Work
**Critical Issues:** 0
**Major Issues:** 6
**Minor Issues:** 3

The react-flow custom-property plumbing is correct and light-theme coverage is intentionally partial (non-overridden vars cascade from themed tokens, which is sound). The focus-visible rule on the controls button is well-formed. The failures cluster in three areas: (1) the validation dot is colour-only with no shape/label alternative; (2) touch targets are under-floor on both the close button and the react-flow controls; (3) forced-colors coverage misses the graph nodes and the validation dot.

---

## Visual Design

### Strengths

- Dark/light react-flow overrides use semantic tokens throughout — no raw hex inside the react-flow block.
- `--xy-controls-box-shadow-default` is correctly re-themed for light at line 141 (shadow lightens to `rgba(15,45,53,0.10)`).
- `--xy-selection-border-default: 1px dashed var(--color-focus-ring)` gives a non-colour selection cue on the canvas.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Hardcoded shadow rgba values differ from the themed controls shadow | Minor | Line 38 (dropdown), line 213 (config panel) | Replace `rgba(0,0,0,0.15)` and `rgba(0,0,0,0.28)` with a shadow token, or add a `[data-theme="light"]` override block that matches the one at line 141 |

---

## Information Architecture

### Strengths

- `NodeConfigPanel` renders as `<aside role="complementary" aria-label="…">` — correct landmark usage, heading hierarchy (`h3`/`h4`) is sound.
- The config panel reflows to a bottom sheet at 760px breakpoint — the approach is correct.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Config panel at 760px `min-width` not constrained — `.version-selector-dropdown` has `min-width: 260px` but no narrow-viewport override; can clip at 320px | Minor | Line 29 | Add `@media (max-width: 320px) { .version-selector-dropdown { min-width: calc(100vw - 2*var(--space-sm)); } }` |
| 260px graph nodes have no responsive scale on narrow viewports | Minor | GraphView.tsx `NODE_WIDTH = 260` | Not a CSS-only fix — flag for JS: reduce `NODE_WIDTH` below 480px or add horizontal scroll affordance |

---

## Interaction Design

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.graph-config-close` touch target is 32x32px — below the 44x44px iOS floor and 48x48dp Android floor | Major | Lines 240–241 | Set `min-width: 44px; min-height: 44px` |
| React Flow controls buttons have no size override — vendor default is ~26–30px; `:focus-visible` rule exists (line 144) but no `min-width`/`min-height` | Major | Lines 144–147 | Add `.react-flow__controls-button { min-width: 44px; min-height: 44px; }` |
| `.version-selector-item--focused` keyboard-focus indicator is a background-colour swap only — no outline, no border, no leading marker | Major | Lines 58–60 | Add `outline: 2px solid var(--color-focus-ring); outline-offset: -2px;` to `.version-selector-item--focused` (or scoped to `:focus-within` on the listbox item) |
| `prefers-contrast: more` block in themes.css does not strengthen `--color-surface-hover` — the focused item's background cue disappears further in high-contrast mode | Major | themes.css lines 12–36 | In the `prefers-contrast: more` block, override `.version-selector-item--focused { outline: 2px solid var(--color-text); }` directly |

**Confirmed not a CSS gap:** `.version-selector-trigger` (line 181 of HeaderVersionSelector.tsx) uses `className="btn version-selector-trigger"` — the base `.btn` class provides the 44px floor. `.version-selector-revert-btn` similarly applies `.btn`. Touch-target gap for the trigger/revert button is closed by the base component class; the comment on lines 93–99 is accurate.

**Confirmed not a CSS gap:** `.version-selector-item--current` appends `(current)` as text inside `.version-selector-item-tag` (HeaderVersionSelector.tsx line 230–233) and sets `aria-label` with "(current)" annotation. The current-version indicator is not colour-only — text is present. No finding raised.

---

## Accessibility

### Quick Check

- [x] 1.4.3 Contrast: Pass (semantic tokens only; raw badge values are well-saturated in dark; light theme has dedicated badge overrides in tokens.css)
- [x] 2.1.1 Keyboard: Partial — listbox has `tabIndex=0` + `handleListKeyDown`; graph container is `role="img"` (read-only, not interactive, so keyboard nav of nodes is not required); `.graph-config-close` button is keyboard-reachable
- [ ] 2.4.7 Focus Visible: Fail — `.version-selector-item--focused` state has no visible outline; controls buttons have `:focus-visible` (pass); `.graph-config-close` inherits base button focus styles (needs verification)
- [ ] 1.4.1 Use of Color: Fail — `.graph-validation-dot` is colour-only; no shape, label, or pattern alternative

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.graph-validation-dot` (lines 192–197) is colour-only: 8x8px circle, background-color set by inline style per state (valid/warning/error). No shape, ARIA role, or visible label alternative. `title` attribute is mouse-hover only — screen readers receive it as tooltip, not as persistent description. Fails WCAG 1.4.1. | Major | Lines 192–197 + GraphView.tsx lines 498–519 | Replace or augment with shape: e.g. use `clip-path` or a character (✕/! / ✓) inside the dot; add `role="img"` and `aria-label` on the `<span>`; for error state use a different shape (diamond or triangle via CSS border-trick) rather than relying on colour alone |
| Forced-colors: `.graph-validation-dot` is not in themes.css forced-colors block. Its `background-color` will be stripped, leaving an invisible 8x8px circle. The dot will vanish in Windows High Contrast mode. | Major | themes.css lines 38–96 (omission) | Add to the `@media (forced-colors: active)` block: `.graph-validation-dot { forced-color-adjust: none; border: 1.5px solid CanvasText; }` plus per-state shape differentiation |
| Forced-colors: graph nodes use inline `backgroundColor: "var(--color-surface-elevated)"` (GraphView.tsx line 531). Forced-colors strips inline background — nodes will collapse to Canvas colour, losing all visual distinction from the canvas background. No forced-colors selector in themes.css covers `.react-flow__node` or the graph node container. | Major | themes.css (omission); GraphView.tsx line 531 | Add `.react-flow__node { forced-color-adjust: none; border: 1px solid CanvasText; }` to the `@media (forced-colors: active)` block in themes.css |

---

## Platform-Specific Notes

**Touch / Mobile (line 309 breakpoint)**

The `@media (max-width: 760px)` block correctly reflows the config panel to a bottom sheet. However:

1. React Flow controls buttons remain ~26–30px on touch — the CSS fix (min-width/min-height on `.react-flow__controls-button`) also resolves the touch-target gap.
2. The minimap threshold is `nodes.length > 5` — on a small screen the minimap (`120x80px`) plus the bottom-docked config panel will overlap; no z-index or margin coordination exists.

**Light theme parity**

Lines 138–142 override only 3 react-flow vars. This is intentional and correct: all remaining react-flow vars use `var(--color-*)` references that are already themed by tokens.css. No gap exists here.

---

## Priority Recommendations

### Major (Fix Before Merge)

1. `.graph-validation-dot` — add shape and ARIA alternatives; colour alone fails WCAG 1.4.1. Add `role="img"` and `aria-label` on the span. Consider distinct shapes per state via CSS (diamond border-trick for error, round for valid).
2. Forced-colors: `.graph-validation-dot` — add `forced-color-adjust: none; border: 1.5px solid CanvasText` in the `@media (forced-colors: active)` block in `themes.css`.
3. Forced-colors: React Flow graph nodes — add `.react-flow__node { forced-color-adjust: none; border: 1px solid CanvasText; }` to the `@media (forced-colors: active)` block in `themes.css`.
4. `.graph-config-close` — raise `min-width` and `min-height` from 32px to 44px (inspector.css lines 240–241).
5. `.react-flow__controls-button` — add `min-width: 44px; min-height: 44px` to the existing focus-visible rule block (inspector.css lines 144–147).
6. `.version-selector-item--focused` — add an outline or border cue; background-colour swap alone is insufficient at any contrast level (inspector.css lines 58–60). Also add override in `themes.css prefers-contrast: more` block.

### Minor (Polish)

1. Replace hardcoded `rgba(0,0,0,0.15)` and `rgba(0,0,0,0.28)` shadows with tokens or add a light-theme override block matching line 141.
2. Add a narrow-viewport (≤320px) `min-width` override for `.version-selector-dropdown`.
3. Document (in a comment) that the minimap overlaps with the bottom-docked config panel on narrow touch screens — flag for JS-side z-index coordination or conditional minimap suppression.

---

## Confidence Assessment

**High confidence** on: touch-target measurements (file evidence is direct), forced-colors gaps (themes.css read in full, no `.graph-validation-dot` or `.react-flow__node` entry found), validation-dot colour-only finding (8x8 circle, inline background-color only, confirmed in GraphView.tsx), version-selector-trigger `.btn` floor (confirmed in HeaderVersionSelector.tsx line 181).

**Medium confidence** on: `.graph-config-close` focus ring inheritance — depends on global `button:focus-visible` rule not inspected here; if present, the focus ring is covered by theme globals.

## Risk Assessment

**High risk if not fixed:** Forced-colors gaps affect Windows High Contrast users — graph nodes and validation dots will be invisible. WCAG 1.4.1 failure on the validation dot is a documented conformance issue.

**Low risk if deferred:** The hardcoded shadow and narrow-viewport dropdown clip are visual inconsistencies, not functional blockers.

## Information Gaps

1. Whether a global `button:focus-visible` rule in `base.css` or similar provides the `.graph-config-close` focus ring — if present, item 4 in the major list is reduced from Major to Minor for that element.
2. Whether the react-flow controls button size is configurable via a `controlsButtonStyle` prop — if so, the CSS min-width approach may conflict with the prop; prefer prop.

## Caveats

- react-flow vendor stylesheet (`@xyflow/react/dist/style.css`, imported in GraphView.tsx line 27) may define control button dimensions that override this file. The `:root .react-flow.react-flow` specificity pattern (confirmed intentional by file header comment) is already used for other overrides and should work here.
- This review is static; no runtime contrast measurement was performed. Badge colour contrast (e.g. `#e8a030` transform badge on `rgba(232,160,48,0.15)` background) was not numerically verified — a dedicated accessibility audit is recommended for badge contrast ratios.
