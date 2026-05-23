# UX/A11y Review: animations.css

**File:** `src/elspeth/web/frontend/src/styles/animations.css`
**Reviewer:** UX Critic Agent
**Date:** 2026-05-23
**Branch:** css-split

---

## Summary

**Overall:** Acceptable — solid reduced-motion discipline with two structural gaps and one dead keyframe.
**Critical Issues:** 0
**Major Issues:** 2
**Minor Issues:** 4

---

## 1. Reduced-Motion Coverage

### Strengths

- All four animated classes declared in this file (`.composing-dot`, `.progress-bar-stripe`, `.spinner`, `.progress-bar-complete`) have explicit silences inside `@media (prefers-reduced-motion: reduce)`.
- `.spinner` gets a corrective `border-top-color` reset so the static ring looks complete rather than visually broken.
- The `react-flow__edge.animated path` catch-all is a sound defensive measure for any animated SVG paths injected by the graph library.
- `pulse-dot` is defined here but consumed by `.status-badge-icon--cancelling` in `shared.css`. That consumer carries its own reduced-motion block in `shared.css` (lines 130-135), so the cross-file pattern is correctly handled.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.composing-dot` silence sets `opacity: 0.7` but leaves the element visible with no static replacement for the bounce signal — a user who needs reduced motion gets three grey dots with no "composing" meaning. | Minor | `animations.css` line 125-128 | Add a static text fallback via `::after` or rely on the parent `ComposingIndicator` wrapper text ("Working on...") being sufficient — verify with AT user that the `.composing-working-view` text reads without the dots. |

---

## 2. Animation Durations

### Assessment

| Animation | Duration | Judgment |
|-----------|----------|----------|
| `composing-bounce` | 1.2s | Acceptable — slow stagger, not aggressive. |
| `progress-stripe` | 0.8s | Borderline. The stripe travels 40px in 800ms. At `linear` easing this produces visible lateral motion that can trigger vestibular responses. 1.0-1.2s would be safer. |
| `spin` | 0.6s | Within tolerance (>400ms). Marginally fast but on a 14px element the arc distance is small. |
| `pulse-dot` (defined here, consumed in `shared.css`) | 1.2s | Fine. |
| `.progress-bar-complete` transition | 150ms (`--transition-normal`) | Correct — matches the token. |

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `progress-stripe` at 0.8s `linear` produces continuous lateral motion across the full viewport width of the bar. Vestibular guidelines recommend avoiding fast-moving large-area horizontal patterns; 0.8s sits below the informal 1s safety floor for wide striped bars. | Major | `animations.css` line 81 | Increase to `animation: progress-stripe 1.2s linear infinite` and consider reducing `background-size` from 40px to 24px so the stripe pitch is less prominent. |

---

## 3. Easing Consistency

### Assessment

Three distinct timing functions are in use:

- `ease-in-out` — composing-bounce, pulse-dot
- `linear` — progress-stripe, spin
- `ease` — `.progress-bar-complete` transition (via `--transition-normal: 150ms ease`)

The split is intentional and appropriate: `ease-in-out` for organic bounce/pulse; `linear` for mechanical rotation and background-position scroll. There is no ad-hoc inconsistency here.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `--transition-normal` is defined in `tokens.css` as `150ms ease`. The `ease` function is different from the `ease-in-out` used by the surrounding animations. This is not wrong but it is undocumented — a future author adding a transition here may reach for `ease-in-out` by analogy with the keyframes and produce a mismatch. | Minor | `tokens.css:193`, `animations.css:88` | Add a comment to the `.progress-bar-complete` rule: `/* Uses --transition-normal (150ms ease) — distinct easing from keyframe animations; see tokens.css */` |

---

## 4. Indeterminate Progress Affordance (WCAG SC 2.2.2)

WCAG 2.2.2 (Pause, Stop, Hide) requires that users can pause, stop, or hide moving content that starts automatically, lasts more than 5 seconds, and is presented in parallel with other content. Indeterminate spinners and striped bars run indefinitely, so they require either (a) a mechanism to pause them, or (b) the animation being essential. Both animated elements here are accompanied by text status and the reduced-motion media query silences them — meaning they are not essential and should be pausable or silenced. The reduced-motion path covers this mechanically.

SC 4.1.2 (Name, Role, Value) applies to the `role="progressbar"` in `ProgressView.tsx`. An indeterminate progressbar should carry `aria-valuenow` omitted (correct) but MUST still have `aria-valuemin` and `aria-valuemax` per WAI-ARIA 1.2 (`progressbar` role spec). `ProgressView.tsx` declares `role="progressbar"` and `aria-label` but omits `aria-valuemin` and `aria-valuemax`.

### Spinner aria-live audit (across consumers)

| Consumer | Spinner wrapped in live region? | Verdict |
|----------|--------------------------------|---------|
| `AuthGuard.tsx` | `role="status"` + `aria-label` on wrapper div | Pass |
| `LoginPage.tsx` | `role="status"` + `aria-label` on wrapper; spinner `aria-hidden="true"` | Pass |
| `ExecuteButton.tsx` | `role="status"` + `aria-label` on spinner `<span>` | Pass |
| `InterpretationReviewInlineMessage.tsx` | `role="status"` live region present in component | Pass |
| `InterpretationReviewTurn.tsx` | `role="status"` live region present | Pass |
| `PluginCard.tsx` | `role="status"` + `aria-live="polite"` on wrapper; spinner `aria-hidden="true"` | Pass |
| `SaveForReviewDialog.tsx` | `role="status"` on wrapper div | Pass |

All spinner usages carry an appropriate live region. The `composing-dot` animation delegates announcement to the parent `ChatPanel` `role="log"` region, which is the correct architectural choice (confirmed by `ComposingIndicator.test.tsx` line 120-123 explicitly testing that the indicator adds no redundant `aria-live`).

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `role="progressbar"` in `ProgressView.tsx` is missing `aria-valuemin="0"` and `aria-valuemax="100"`. WAI-ARIA 1.2 requires these on `progressbar` even in indeterminate mode (where `aria-valuenow` is omitted to signal indeterminate). Without them, some AT cannot distinguish "indeterminate" from "broken". | Major | `ProgressView.tsx` line 84 | Add `aria-valuemin={0} aria-valuemax={100}` to the `<div role="progressbar">`. Do not add `aria-valuenow` — its absence is the indeterminate signal. |

---

## 5. `will-change` / GPU Hints

No `will-change` declarations exist anywhere in the CSS files under review. For the three animated properties in use:

- `transform` (composing-bounce, spin) — browsers promote elements with CSS `animation` targeting `transform` automatically in most modern engines. Explicit `will-change: transform` would be redundant.
- `background-position` (progress-stripe) — this does NOT benefit from GPU promotion (background-position is composited on CPU in most browsers). Adding `will-change: background-position` would waste a compositing layer for no gain.
- `opacity` (pulse-dot, composing-bounce) — browsers already promote opacity animations.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.spinner` animates `transform: rotate()` on a 14px element. Most browsers auto-promote this, but some older WebKit versions do not. Adding `will-change: transform` to `.spinner` would guarantee GPU compositing at the cost of one small compositing layer. Low priority — only worth doing if paint jank is observed on low-end devices. | Minor | `animations.css` line 111 | Optional: add `will-change: transform` to `.spinner` rule. |

---

## 6. Dead Keyframes

**`@keyframes pulse-dot` is defined in `animations.css` but is not consumed by any class in `animations.css`.** It is consumed by `.status-badge-icon--cancelling` in `shared.css` (line 127).

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `@keyframes pulse-dot` is a dead export — no class in `animations.css` references it; its only consumer is in `shared.css`. The file header comment lists `pulse-dot` as a keyframe defined here, which is accurate, but readers will search `animations.css` for the consuming rule and not find it. This creates a maintenance trap: if `animations.css` is refactored, `pulse-dot` could be deleted as "unused" and silently break `shared.css`. | Minor | `animations.css` line 91; `shared.css` line 127 | Add a comment to `@keyframes pulse-dot`: `/* Consumed by .status-badge-icon--cancelling in shared.css — do not delete without checking shared.css */`. Consider moving the `.status-badge-icon--cancelling` animation rule into `animations.css` with the other driven rules, keeping all consumers co-located with their keyframes. |

---

## Priority Recommendations

### Critical (Fix Immediately)
None.

### Major (Fix Before Launch)

1. **`progress-stripe` duration too fast for vestibular safety:** Change `animation: progress-stripe 0.8s linear infinite` to `1.2s`. Consider reducing `background-size` from 40px to 24px.
2. **`role="progressbar"` missing `aria-valuemin`/`aria-valuemax`:** Add `aria-valuemin={0} aria-valuemax={100}` to `ProgressView.tsx` line 84.

### Minor (Improvement)

1. **`@keyframes pulse-dot` cross-file dependency is undocumented:** Add a comment preventing accidental deletion.
2. **`ease` vs `ease-in-out` easing divergence is undocumented:** Add a clarifying comment to `.progress-bar-complete`.
3. **`progress-stripe` pitch:** Consider reducing `background-size` from 40px to 24px even at corrected duration — smaller pitch is less vestibularly aggressive.
4. **`will-change: transform` on `.spinner`:** Optional, only if paint jank is observed.

---

## Confidence Assessment

**Confidence: High (85%).**

Direct code inspection of all six focus areas. Consumer survey was exhaustive for production components (all `.spinner`, `.composing-dot`, and `.progress-bar` usages verified). The `pulse-dot` cross-file finding is verified from both the definition site and the consumer.

**Lowering factors:**
- Cannot measure actual contrast of `--color-info` and `--color-success` tokens against the `--color-surface-elevated` background without token resolution — no contrast findings are made.
- Cannot observe real AT behaviour (screen reader announcement timing for `role="status"` wrappers) from static analysis alone.

## Risk Assessment

**Medium risk overall.** The missing `aria-valuemin`/`aria-valuemax` on the progressbar is an objective ARIA conformance gap (WCAG 4.1.2, AA). The `progress-stripe` speed is a vestibular concern that may affect users in real conditions. The dead-keyframe documentation gap is a low-probability maintenance hazard that will not affect current users.

## Information Gaps

- Token values for `--color-info`, `--color-success`, `--color-error`, `--color-warning` are not inspected for contrast against `--color-surface-elevated`. A separate contrast audit should verify `progress-bar-stripe` and `progress-bar-complete` colours pass 3:1 (non-text UI component threshold).
- `tutorial-progress-bar` class (used in `TutorialTurn2Describe.tsx` and `TutorialTurn4Run.tsx`) is not defined in `animations.css` — it appears to be styled elsewhere. If it animates, its reduced-motion coverage is out of scope for this review.

## Caveats

- This review covers `animations.css` and its direct consumers only. `shared.css` is reviewed only at the `pulse-dot` intersection point.
- WCAG criterion citations are against WCAG 2.1 AA unless noted.
- WAI-ARIA `progressbar` spec citation is WAI-ARIA 1.2 (current at review date).
