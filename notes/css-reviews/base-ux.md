# base.css — UX/Accessibility Review

**Reviewer**: ux-critic agent
**Date**: 2026-05-23
**File**: `src/elspeth/web/frontend/src/styles/base.css` (~130 lines)
**Tokens consulted**: `tokens.css` (`:root` dark defaults + `[data-theme="light"]` overrides)

---

## Summary

**Overall**: Acceptable — one Major issue, two Minor issues, no Critical blockers.

| Severity | Count |
|----------|-------|
| Critical | 0 |
| Major | 1 |
| Minor | 2 |

The core accessibility primitives (focus ring, sr-only, skip link visibility) are correctly implemented. Contrast is strong across both themes. Two issues need fixing before launch: scrollbar thumb contrast is below the WCAG 1.4.11 floor in both themes, and the skip link uses `:focus` instead of `:focus-visible`.

---

## Visual Design

### Strengths

- Body font-size resolves to `16px` (`--font-size-base`) — meets the 16px+ mobile minimum.
- Line-height resolves to `1.5` (`--line-height-normal`) — meets the 1.5x body standard.
- `-webkit-font-smoothing: antialiased` is appropriate for dark backgrounds and does not affect WCAG compliance.

### Issues

| Issue | Severity | Location | Evidence |
|-------|----------|----------|----------|
| Scrollbar thumb contrast below 3:1 in both themes | Major | Lines 56–78, tokens.css lines 129–131 and 327–329 | Dark theme: blended thumb `rgb(34,68,75)` vs track `#0f2d35` = **1.38:1**; hover = 1.86:1. Light theme: blended thumb `rgb(202,211,213)` vs track `#f4f8f9` = **1.42:1**; hover = 1.84:1. WCAG 1.4.11 requires 3:1 for UI components. |

**Fix**: Raise `--color-scrollbar-thumb` alpha from 0.15 → ~0.38 dark / ~0.40 light, and `--color-scrollbar-thumb-hover` from 0.28 → ~0.55 dark / ~0.58 light. Alternatively switch from alpha blending to opaque token values computed to hit 3:1. Verify with `colorContrast.test.ts`.

---

## Interaction Design

### Strengths

- `:focus-visible` correctly uses `outline` (not `box-shadow`) — renders correctly under Windows Forced Colors / High Contrast without a separate `@media (forced-colors)` block.
- Focus ring contrast is excellent across every surface in both themes: white ring on dark surfaces ranges 11.65:1–17.10:1; dark ring `#0f2d35` on light surfaces ranges 12.33:1–14.51:1. Both massively exceed WCAG 2.4.11's 3:1 floor.
- `outline-offset: 2px` prevents the ring from touching element borders, improving readability on bordered controls.
- Skip link correctly hides via `top: -100%` (not `display:none` or `visibility:hidden`), preserving keyboard reachability.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Skip link uses `:focus` instead of `:focus-visible` | Minor | Lines 114–116 | `:focus` fires on mouse click too, causing a brief visual flash when a mouse user clicks an element above the fold. Change `.skip-to-content:focus { top: 0; }` to `.skip-to-content:focus-visible { top: 0; }`. Skip links are interactive elements — `:focus-visible` is the correct selector. |
| Skip link target height is 40px, 4px under the WCAG 2.5.5 AAA floor | Minor | Lines 100–112 | Height = 16px text × 1.5 line-height + 8px+8px padding = 40px. Passes WCAG 2.5.8 (AA, 24px minimum). To reach 44px AAA, add `min-height: 44px` and `display: flex; align-items: center` to `.skip-to-content`. |

---

## Accessibility

### Quick Check

- [x] **1.4.3 Contrast (text)**: PASS — skip link: dark 5.32:1 (AA), light 7.50:1 (AAA). Body text tokens resolve well above 4.5:1.
- [ ] **1.4.11 Non-text contrast**: FAIL — scrollbar thumb 1.38:1–1.86:1 in both themes (need 3:1).
- [x] **2.1.1 Keyboard nav**: PASS — focus ring present, skip link keyboard-reachable, no `outline:none` without `:focus-visible` guard.
- [x] **2.4.7 Focus visible**: PASS — 2px white/dark outline with offset, high contrast against all surface variants.
- [x] **1.1.1 Alt text**: N/A — no images in this file.
- [x] **Forced colors**: PASS — outline-based ring survives forced-colors mode without override.

### sr-only Pattern

The implementation matches the functional canonical pattern. One cosmetic gap:

```css
/* Current (base.css lines 119–129) */
clip: rect(0, 0, 0, 0);        /* CSS2, deprecated but functional */

/* Modern canonical addition — not present */
clip-path: inset(50%);         /* CSS3, should accompany clip: rect() */
```

`clip: rect(0,0,0,0)` still works in all current browsers. Adding `clip-path: inset(50%)` alongside it removes the reliance on a deprecated property. This is a hygiene improvement, not a compliance gap — `clip:rect()` is not yet removed from any shipping engine.

No `display:none` or `visibility:hidden` — correct. Padding is `0` — correct (non-zero padding with `overflow:hidden` would expose a clickable area).

---

## Universal Selector Hygiene

The Firefox scrollbar block (lines 75–78) applies `scrollbar-width:thin` to `*`. The property is ignored on non-scrolling elements, so there is no practical breakage. Encapsulated Shadow DOM components are not affected (outer `*` rules do not pierce shadow boundaries). This is the idiomatic Firefox scrollbar reset pattern and does not warrant a change.

The leading `*,*::before,*::after { box-sizing: border-box }` block (lines 7–11) is the established global reset. Pseudo-element inclusion is correct. No breakage risk for Shadow DOM components since encapsulated shadow trees ignore this rule.

---

## Scrollbar Theme Parity

Both themes alias `--color-scrollbar-track` to `--color-bg`, and use matching alpha-on-dark / alpha-on-light teal/navy composites for thumb and hover. The structural parity is correct. The shared defect is the alpha values being too low in both themes — fixing one requires fixing the other in the same commit.

---

## Reduced Motion

No transitions are defined directly in `base.css` — the skip link repositioning is instant (`top: 0`), which is correct. Transition tokens (`--transition-fast/normal/slow`) are defined in `tokens.css` but not applied here. Reduced-motion enforcement is the responsibility of consuming component files, not `base.css`. No action required in this file.

---

## Priority Recommendations

**Major (Fix Before Launch):**

1. **Scrollbar contrast** — raise `--color-scrollbar-thumb` and `--color-scrollbar-thumb-hover` alpha values in both themes to achieve at least 3:1 against `--color-scrollbar-track`. Target blended values of approximately `rgb(68,110,119)` dark / `rgb(145,162,167)` light for the resting state. Verify against `colorContrast.test.ts`.

**Minor (Fix Before Launch):**

2. **Skip link selector** — change `.skip-to-content:focus` to `.skip-to-content:focus-visible` (line 114) to suppress the mouse-click flash.

**Hygiene (Improvement):**

3. **sr-only `clip-path`** — add `clip-path: inset(50%);` alongside `clip: rect(0, 0, 0, 0);` (after line 126) to replace reliance on deprecated CSS2 clipping syntax.

---

## Confidence Assessment

**High** for all measurements. Contrast ratios are computed directly from token hex values in `tokens.css` using WCAG-spec linearization. The scrollbar figures use correct alpha-compositing against the track color. The focus ring assessment covers all surface token variants. No runtime rendering was observed — computed values could theoretically differ if a browser applies gamma correction differently to alpha-composited scrollbar renders, but the gap (1.38:1 vs. 3:1 required) is large enough that measurement uncertainty does not affect the verdict.

## Risk Assessment

**Scrollbar contrast**: Medium business risk. Scrollbars are a visible UI component used throughout the composer. Failing 1.4.11 is a WCAG AA violation. In a demo context this is unlikely to be auditor-scrutinized, but it is a genuine compliance gap.

**Skip link selector**: Low risk. Affects only mouse users who happen to click something that triggers `:focus` on the skip link — an edge case. No functional keyboard regression.

## Information Gaps

- Actual rendered pixel size of the skip link was not verified via browser DevTools. The 40px figure is calculated from token values and assumes no override by a parent stylesheet. If `index.css` or another cascade layer sets `line-height` differently on the body before this rule is applied, the actual height may differ.
- The `--z-skip-link: 1000` value was verified in `tokens.css`. Whether any component in the consumer tree (`src/elspeth/web/frontend/src/`) creates a stacking context above z=1000 was not checked. If such a context exists, the skip link could be visually obscured despite being keyboard-accessible.

## Caveats

Scrollbar contrast calculations use alpha-compositing against a flat background. Real browser scrollbar rendering may apply OS-level compositing that differs slightly. The 1.38:1 figure should be treated as an upper bound for the actual contrast — the rendered value is unlikely to be higher.
