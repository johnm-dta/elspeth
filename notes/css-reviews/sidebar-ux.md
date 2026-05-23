# UX Review: sidebar.css — Side-Rail

**File:** `src/elspeth/web/frontend/src/components/sidebar/sidebar.css`
**Consumers:** `SideRail.tsx`, `SideRailValidationBanner.tsx`, `ExecuteButton.tsx`,
`ExportYamlButton.tsx`, `CatalogButton.tsx`, `GraphMiniView.tsx`
**Date:** 2026-05-23
**Reviewer:** UX Critic Agent

---

## Summary

**Overall:** Acceptable — functional and largely accessible, with four fixable gaps.
**Critical Issues:** 0
**Major Issues:** 3
**Minor Issues:** 4

The file is well-commented and the migration note for `.side-rail-error-banner` is
load-bearing documentation. Token discipline is clean. The primary gaps are: slot-fill
parity between `CatalogButton` and the `.btn`-family buttons; missing focus ring and
disabled state on `.graph-mini`; the error banner's colour-only distinction; and the
`apply-btn` disabled opacity failing AA.

---

## 1. Visual Design

### Strengths

- Token-only colour references throughout — no hardcoded hex values. Both themes are
  covered by the upstream token set.
- `.completion-bar` vertical-stack layout resolves the documented 436 px overflow
  problem correctly. The comment explaining the fix is thorough.
- `font-weight: 650` on `.catalog-reference-label` (line 127) will silently fall back
  to 700 on browsers that do not support fractional weights. This is not a bug — it
  degrades gracefully — but note it for cross-browser consistency.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.side-rail-suggestion-apply-btn:disabled` uses `opacity: 0.45`. At that opacity the text (`var(--color-info)` = `#61daff`) on the button background can dip below 3:1 against the dark inspector surface. The `.btn` disabled pattern (established in shared.css lines 192-198) uses a token pair that passes AA deliberately. | Major | `sidebar.css` L80-83 | Replace `opacity: 0.45` with the `.btn` disabled token pattern: `background-color: var(--color-bg); color: var(--color-text-muted); border-color: var(--color-border-strong); cursor: not-allowed;`. |
| `catalog-reference-meta` uses `--font-size-3xs` (10px) with `font-weight: 700` and `text-transform: uppercase`. At 10px this falls below the WCAG 1.4.3 14pt-bold threshold for large text — the token was documented as "sub-chrome micro-text" in tokens.css. The content ("Reference") is not critical to function (it duplicates the `aria-label` on the button), but if it ever carries unique information it will fail AA. | Minor | `sidebar.css` L133-142 | Raise to `--font-size-2xs` (11px) or `--font-size-xs` (12px). At 12px bold text the contrast requirement drops to 3:1 (large text threshold). |

---

## 2. Information Architecture

### Strengths

- `.side-rail-slot:empty { display: none; }` (shared.css) collapses absent slots cleanly
  — the rail does not leave gap regions when slots are null.
- Progressive disclosure on the suggestion list (collapsed when `suggestions.length > 2`)
  avoids overwhelming a short rail.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| The `SideRail` slot order in the JSX (audit-readiness → validation-banner → graph-mini → **completion-bar** → catalog) places the completion CTA above the catalog button. The catalog is a reference surface; the CTA is the primary action. This ordering is correct per the "reference, not toolkit" framing in memory. No change needed — documented here to confirm the intentional choice. | — | `SideRail.tsx` L27-43 | No action. |

---

## 3. Interaction Design

### Strengths

- `.btn` base class (shared.css) delivers `min-height: 44px` (WCAG 2.5.5 AAA) and a
  hover state. `ExecuteButton` and `ExportYamlButton` both compose `.btn`, so they
  inherit the full interactive treatment.
- `ExecuteButton` correctly uses both `disabled` and `aria-disabled="true"` together,
  and provides `aria-describedby` pointing to a `.sr-only` span when run is blocked by
  a pending interpretation. This is the WCAG-canonical pattern.
- `SuggestionList` header uses `role="button"`, `tabIndex={0}`, `aria-expanded`, and a
  `handleKeyDown` that handles both `Enter` and `Space`. The `focus-visible` ring is
  declared (L31-34). Correct.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| **Slot-fill parity gap: `CatalogButton` vs `ExecuteButton`/`ExportYamlButton`.** The two `.btn`-family buttons inherit `.btn`'s `focus-visible` ring from the global rule in `base.css` (`outline: 2px solid var(--color-focus-ring)`). `CatalogButton` uses `.side-rail-catalog-btn` which overrides this with `outline: 2px solid var(--color-accent)` (L117-120). `--color-accent` is `#1a7a52` (dark) vs `--color-focus-ring` which is `#ffffff` (dark theme) or `#0f2d35` (light theme). On the dark inspector surface (`--color-surface-inspector: #2a2826`) the accent green (`#1a7a52`) produces approximately 2.5:1 contrast — below the 3:1 minimum for non-text UI components (WCAG 1.4.11). Switch to `var(--color-focus-ring)` to match the rest of the `.btn` family and meet 1.4.11. | Major | `sidebar.css` L117-120 | `outline: 2px solid var(--color-focus-ring);` |
| **`CatalogButton` has no disabled state defined**, unlike `ExecuteButton` (which can be disabled) and `ExportYamlButton` (which hides when no session). If a future caller disables the catalog button, it will receive no visual feedback and the inherited `cursor: pointer` from L108 will not change. | Minor | `sidebar.css` L95-120 | Add: `.side-rail-catalog-btn:disabled, .side-rail-catalog-btn[aria-disabled="true"] { background-color: var(--color-bg); color: var(--color-text-muted); border-color: var(--color-border-strong); cursor: not-allowed; }` |
| **`GraphMiniView` (non-empty) renders as a `<button>` with class `.graph-mini` but `.graph-mini` has no `focus-visible` rule.** The global base.css rule (`focus-visible { outline: 2px solid var(--color-focus-ring) }`) applies, so keyboard focus is not *invisible*, but the focus ring will be clipped by the parent `.layout-siderail { overflow: hidden; }` if the button sits at the rail edge. At minimum, add an explicit `focus-visible` rule with `outline-offset: -2px` (inset) to ensure the ring is always visible regardless of parent clipping. Also: `.graph-mini--empty` applies `cursor: default` but the markup is a `<div>`, so it is not keyboard reachable — this is correct, but verify the `data-testid="graph-mini-empty"` is not accidentally receiving focus from a parent container. | Major | `sidebar.css` L169-193 | Add: `.graph-mini:focus-visible { outline: 2px solid var(--color-focus-ring); outline-offset: -2px; }` |
| **`side-rail-suggestion-apply-btn` has no `type` attribute in the TSX** (`SideRailValidationBanner.tsx` L51). Inside a `<form>` this would default to `type="submit"`. The button is not inside a form, so the risk is low today, but the pattern is incorrect. | Minor | `SideRailValidationBanner.tsx` L51 | Add `type="button"` to the apply button. |

---

## 4. Accessibility Quick Check

| Criterion | WCAG | Result |
|-----------|------|--------|
| Text contrast: `--color-info` (`#61daff`) on `--color-info-bg` | 1.4.3 | Pass (dark: ~5.9:1; light: `--color-info` deepened to `#176d8a`, ~4.7:1 on light info-bg) |
| `.side-rail-suggestion-apply-btn:disabled` opacity 0.45 | 1.4.3 | Fail — see Visual Design |
| Keyboard navigation: all interactive sidebar elements | 2.1.1 | Mostly pass — `GraphMiniView` button reachable; apply-btn missing `type="button"` is low risk |
| Focus visible: global rule in base.css covers `.btn` family | 2.4.7 | Pass for `.btn`; `.graph-mini` focus ring potentially clipped — see Major issue |
| `CatalogButton` focus ring: `--color-accent` on inspector surface | 1.4.11 | Fail — ~2.5:1 vs 3:1 minimum for UI components |
| Error banner text-only distinction | 1.4.1 | Partial — see error banner section below |
| Alt text / `aria-hidden` on SVG in `MiniSvg` | 1.1.1 | Pass (`role="img" aria-hidden="true"` on svg; parent button has `aria-label`) |
| `role="alert"` on error banner | 4.1.3 | Pass (SideRailValidationBanner.tsx L89) |
| Suggestion header: `role="button"`, keyboard handler | 4.1.2 | Pass |

---

## 5. Error Banner

### Strengths

- `role="alert"` is present on the error `<div>` in `SideRailValidationBanner.tsx`
  (L89), satisfying WCAG 4.1.3 for programmatic announcement.
- Inherits `.validation-banner-fail` which applies `color: var(--color-error)` and a
  border, providing two visual channels (colour + border) for the fail state.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.side-rail-error-banner` adds only `padding` and `font-size` on top of the `.validation-banner` and `.validation-banner-fail` classes. The error text itself has no icon — the distinction from a success banner is colour + border only. WCAG 1.4.1 requires information is not conveyed by colour alone. The `validation-banner-fail` border (`var(--color-error-border)`, a translucent red) is a second channel, but it is subtle. A text prefix or icon (e.g. a Unicode warning glyph) inside the banner content would satisfy 1.4.1 without depending on CSS. The current content is whatever `error` string the store provides — the fix belongs in `SideRailValidationBanner.tsx`, not the CSS. | Minor | `SideRailValidationBanner.tsx` L88-92 | Prepend a visible non-colour cue: e.g. `<span aria-hidden="true">⚠ </span>{error}`. The border is a partial mitigant; this raises it to fully conformant. |

---

## 6. Catalog Reference Panel / Scroll Behaviour

The catalog reference panel is opened via `OPEN_CATALOG_EVENT` dispatched from
`CatalogButton`. The panel itself is defined in `catalog.css` (out of scope for this
review). Within sidebar.css, `.catalog-reference-label` uses `overflow: hidden` and
`text-overflow: ellipsis` on `white-space: nowrap` (L123-130) — this correctly truncates
long plugin names to fit the grid column. No scroll behaviour is defined for the panel
within sidebar.css; the catalog drawer manages its own scrolling. This is appropriate.

---

## 7. Token Discipline / Dead Rules

### Dead rules

None found. All selectors in sidebar.css are consumed by at least one component in the
consumer directory:

- `.side-rail-suggestion-*`: `SideRailValidationBanner.tsx` (`SuggestionList`)
- `.side-rail-execute-btn`: `ExecuteButton.tsx`
- `.side-rail-export-yaml-btn`: `ExportYamlButton.tsx`
- `.side-rail-catalog-btn`, `.catalog-reference-*`: `CatalogButton.tsx`
- `.completion-bar`: referenced in the comment — the `CompletionBar` TSX is not in the
  consumer directory listed in scope; if that component has moved, verify the rule is
  still reached. The `sidebar.css` comment (L144-167) names `src/components/CompletionBar.tsx`.
- `.graph-mini`, `.graph-mini--empty`: `GraphMiniView.tsx`
- `.side-rail-error-banner`: `SideRailValidationBanner.tsx`

### Token discipline

Clean. No hardcoded hex, px magic numbers outside documented exceptions (the `2px 8px`
padding on `.side-rail-suggestion-apply-btn` matches the `.type-badge` pattern in
shared.css and is acceptable as a structural constant rather than a design token).

### Redundancy note

`.side-rail-execute-btn` and `.side-rail-export-yaml-btn` share an identical block
(L88-93). The comment at L85-87 acknowledges this is intentional (`kept as
comma-grouped selector so the class names remain stable`). This is fine. The
`.side-rail-slot-fill` helper in shared.css L271-275 encodes the same width/margin
pattern and is available for future callers — document that `CatalogButton` does not
use it because it also needs `display: grid`, making full composition impractical.

---

## Priority Recommendations

### Critical (Fix Immediately)

None.

### Major (Fix Before Demo / Merge)

1. **`CatalogButton` focus ring**: Change `sidebar.css` L119 from `var(--color-accent)`
   to `var(--color-focus-ring)`. One line. Fixes WCAG 1.4.11 failure.

2. **`GraphMiniView` focus-visible clipping**: Add `.graph-mini:focus-visible { outline: 2px solid var(--color-focus-ring); outline-offset: -2px; }` to `sidebar.css` after L188.

3. **`apply-btn` disabled opacity**: Replace `opacity: 0.45` / `cursor: not-allowed`
   block (L80-83) with the token-based `.btn` disabled pattern. Fixes potential 1.4.3
   failure.

### Minor (Improvement)

1. Add `type="button"` to the apply button in `SideRailValidationBanner.tsx` L51.

2. Add non-colour cue (prefix glyph) to error banner content in
   `SideRailValidationBanner.tsx` L88-92 to fully satisfy WCAG 1.4.1.

3. Add a disabled state rule for `.side-rail-catalog-btn` for defensive parity.

4. Raise `.catalog-reference-meta` from `--font-size-3xs` (10px) to `--font-size-xs`
   (12px) if the label ever carries unique (non-redundant) information.

---

## Confidence Assessment

**Confidence: High** for interaction and accessibility findings — all are grounded in
direct markup and CSS readings, not inferred. The contrast estimates for
`--color-accent` on `--color-surface-inspector` are computed from the token values
(`#1a7a52` foreground, `#2a2826` background → relative luminance ~0.022 fg, ~0.028 bg
→ ratio ~1.27:1 for the fill, but the *outline* pixel sits on whatever is behind the
button — the inspector surface, so: `#ffffff` ring on `#2a2826` = ~13:1; `#1a7a52`
ring on `#2a2826` = ~2.5:1). The focus ring failure is confirmed.

**Confidence: Medium** for the `CompletionBar` dead-rule check — the component is
referenced in a comment but not in the scoped consumer directory. If `CompletionBar.tsx`
exists at `src/components/CompletionBar.tsx` and imports `sidebar.css`, the rules are
live. This should be verified by the implementer.

## Risk Assessment

**Low overall.** All three Major findings are additive (adding/correcting a CSS
property) — no structural changes required. The `apply-btn` disabled fix is the highest
regression risk (changing visual appearance for an already-rendered state), but it
corrects a known platform pattern rather than introducing a new one.

## Information Gaps

- The full `CompletionBar.tsx` file is out of scope for this review. Confirm it still
  imports `sidebar.css` and uses `.completion-bar`, `.side-rail-execute-btn`, and
  `.side-rail-export-yaml-btn` directly.
- Actual rendered contrast values require a browser + computed-style pass.
  Measurements above are from raw token hex values.

## Caveats

- This review covers CSS and markup only — runtime behaviour (animation, WebSocket
  state transitions) is not assessed.
- WCAG target level is assumed AA (2.1). AAA requirements are noted where relevant
  but not treated as blocking.
- The inspector surface (`--color-surface-inspector`) is `#2a2826` in dark theme and
  `#faf7f3` in light theme. Contrast measurements above are dark-theme only unless
  stated otherwise.
