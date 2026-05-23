## Design Review: guided.css — Guided-mode Widget Family

### Summary
**Overall:** Needs Work
**Critical Issues:** 2
**Major Issues:** 5
**Minor Issues:** 4

---

### Visual Design

**Strengths:**

- Token discipline is consistent — no hardcoded colours anywhere in the file; every colour, spacing, and radius value draws from a CSS custom property.
- The reduce-motion block at lines 1378–1402 is comprehensive: every element that carries a `transition` in the file appears in the guard list. No animated property escapes the `prefers-reduced-motion: reduce` override.
- Focus-visible rings are uniform across all interactive elements: `2px solid var(--color-focus-ring)` with `outline-offset: 2px`. No element is missing a focus rule.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Pressed-chip visual is identical to primary CTA visual | Major | Lines 485–494 | `[aria-pressed="true"]` uses `--color-accent` + `--color-text-inverse` — the same values as every primary action button. A screen-width chip strip of "selected" options is visually equivalent to a row of call-to-action buttons. Differentiate with an inset box-shadow, border-width increase, or a distinct pressed-accent token. |
| Wrong-semantic token fallbacks on warning and error colours | Major | Lines 291, 716, 731 | `var(--color-warning, var(--color-accent))` and `var(--color-error, var(--color-accent))` fall back to the accent (blue) colour if the semantic token is absent. A missing warning token renders a warning aside in blue, stripping the hazard signal. If the tokens are guaranteed to be defined, remove the fallbacks. If they are not guaranteed, choose a fallback that preserves the semantic meaning (e.g. a neutral grey that signals "something is here" without impersonating a success or information state). |
| Hardcoded `120px` column minimum repeated in three places | Minor | Lines 925, 1017, 1082 | `.guided-propose-option-key`, `.guided-recipe-slot-key`, and the `.guided-recipe-slot-row .guided-recipe-input-warning` offset calc all encode the same `120px` gutter independently. A single token (e.g. `--guided-kv-key-width: 120px`) lets the gutter be adjusted in one place. |
| `line-height: 1` on `.guided-multi-custom-chip-label` | Minor | Line 529 | A line-height of 1 clips descenders on labels containing letters like `g`, `j`, `p`, `y`. Use `1.2` at minimum. |

---

### Information Architecture

**Strengths:**

- The `guided-chip-instruction` / `guided-chip-fieldset` / `guided-chip-legend` structure cleanly encodes group semantics: `<fieldset>+<legend>` provides the group boundary and name; the `<p id=…>` instruction is associated via `aria-describedby` on the fieldset rather than duplicated per-chip.
- Progressive disclosure pattern for `guided-history` is sound: the collapsible region uses `[hidden]` (native browser `display:none`) rather than a CSS visibility toggle, so the `aria-controls` reference always resolves (documented at line 1209).

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| No visual distinction between single-select and multi-select chips | Major | Lines 163–194 vs 485–494 | Both widget types render `guided-chip-btn`. The only difference is presence of `aria-pressed`. Sighted users receive no checkbox-like or radio-like visual cue indicating whether one or many items can be chosen. Add a pseudo-element indicator: a circle-inside-circle for single-select (radio pattern) and a square with a check for multi-select (checkbox pattern), or use border-radius as a signal (full round = single, small round = multi). |
| Custom-input row visually indistinguishable from preset chips | Major | Lines 196–237 | `.guided-custom-row` / `.guided-custom-input` use the same surface colour and border style as chip buttons. The "or type your own" region needs a separator or background shift to signal it is a different input modality. At minimum: a `border-top: 1px solid var(--color-border)` with padding above the row, plus a muted section label above the input. |
| CTA placement is inconsistent across widgets | Minor | Multiple action rows | In InspectAndConfirm the primary action ("Looks right") is left-aligned. In MultiSelect the primary action ("Continue") is right-aligned after the escape button. In ProposeChain the primary action ("Accept all steps") is rightmost among three peers. Standardise: primary is always the first (leftmost) action button in each row, or always the last. |

---

### Interaction Design

**Strengths:**

- All interactive elements meet the WCAG 2.5.8 minimum target via `min-height: var(--size-control)` — noted explicitly in comments at lines 174, 323, 575, 787, 963, 1102, 1125, 1160, 1339, 1362.
- `.guided-custom-submit-btn` uses `:not(:disabled)` for the active state rather than inverting the disabled state — avoids the cascade specificity trap where disabling fails to reset appearance.
- Focus management on custom-chip removal (WCAG 2.4.3) is documented in MultiSelectWithCustomTurn.tsx and the CSS correctly does not interfere with the JavaScript restore path.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.guided-multi-custom-remove-btn` target is 24x24px — below minimum | Major | Lines 535–537 | The X removal button is `width: 24px; height: 24px`. WCAG 2.5.8 requires a minimum 24x24px target with 24px of spacing around it, and iOS HIG specifies 44x44px. The outer `.guided-multi-custom-chip` does not provide enough surrounding padding to compensate. Increase to at least 32x32px and add padding, or wrap the button in a hit-target expander using `::before` padding-hack. |
| No keyboard arrow-key affordance hinted in chip-group CSS | Minor | Lines 148–194 | The `guided-chip-instruction` text ("Select one. Choosing an option continues") mentions selection but provides no arrow-key guidance for users who navigate chip groups by keyboard. A visible `(use arrow keys to navigate)` hint should be conditionally shown when a chip receives focus. This is a CSS-adjacent concern — the CSS provides `.guided-chip-instruction` as the vehicle; the component should populate it conditionally on keyboard navigation. Flag for implementation. |

---

### Accessibility

**Quick Check:**

- [x] 1.4.3 Contrast: Cannot measure without token resolution — all colours are variables. No hardcoded values to fail.
- [ ] 2.1.1 Keyboard: Chips are `<button>` elements — natively keyboard-accessible. However the pressed-state visual difference is insufficient for keyboard-only users who rely on visual state to understand toggle status.
- [x] 2.4.7 Focus Visible: Pass — all interactive elements have explicit `focus-visible` rules.
- [x] 1.1.1 Alt Text: No `<img>` elements in scope. The warning icon SVG at line 1073–1078 uses `fill: currentColor` on a block `display` SVG — verify in the consumer that a containing `aria-label` or `aria-hidden` is applied (the CSS cannot enforce this).

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Schema-form error state covers only `textarea`, not `input` or `select` | Critical | Lines 715–717 | `.guided-schema-textarea--error` exists to show an error border on JSON fallback fields. No equivalent modifier class exists for `.guided-schema-input` or `.guided-schema-select`. Required text/number/enum fields that fail validation will not show any error border — WCAG 3.3.1 (Error Identification) and 3.3.3 (Error Suggestion). Add `.guided-schema-input--error { border-color: var(--color-error); }` and `.guided-schema-select--error { border-color: var(--color-error); }`. |
| `guided-inspect-warnings` is an `<aside>` without a heading | Minor | Lines 289–295 | The aside carries `aria-label="Data warnings"` in the TSX, but the `.guided-inspect-warnings` styles provide no heading role. The CSS comment says "announced via aria-label" which is correct for the landmark; however the warnings lack a visible heading that would let a sighted user skim that this block is distinct from the table. Consider adding a styled warning heading class. |

---

### Critical: Missing CSS Definitions (Class Drift)

This is the highest-severity finding in the file.

**`SchemaFormTurn.tsx` uses three classes undefined in any CSS file:**

- `.guided-turn-primary` (line 120, 129 in SchemaFormTurn) — the primary action button for "Apply recipe" / "Continue"
- `.guided-turn-secondary` (line 129, 231 in SchemaFormTurn) — "Build manually" and "Clear" buttons
- `.guided-schema-fields` (line 105 in SchemaFormTurn) — the field list wrapper

**`ProposeChainTurn.tsx` uses four classes undefined in any CSS file:**

- `.guided-propose-step-actions` (line 200) — per-card action row
- `.guided-propose-edit-btn` (line 203) — "Edit step N" button inside each card
- `.guided-propose-secondary-btn` (lines 218, 226) — "Reject" and "Ask advisor" buttons

**`RecipeContextHeader.tsx` uses `.recipe-context-header` — also undefined.**

A search across all frontend CSS files (`src/styles/`, `src/components/**/*.css`) confirms these selectors exist in no stylesheet. The SchemaFormTurn primary action button and the ProposeChainTurn edit/reject/advisor buttons are currently unstyled — they render as browser-default `<button>` chrome. This is a functional regression for every schema form and every propose-chain turn.

**Fix:** Define the missing classes in `guided.css`. Suggested shapes:

```css
/* Shared CTA classes referenced by SchemaFormTurn */
.guided-turn-primary {
  padding: var(--space-sm) var(--space-lg);
  background-color: var(--color-accent);
  color: var(--color-text-inverse);
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--font-size-base);
  font-family: inherit;
  min-height: var(--size-control);
  transition: background-color 0.1s ease;
}
.guided-turn-primary:hover:not(:disabled) {
  background-color: var(--color-btn-primary-bg-hover);
}
.guided-turn-primary:disabled {
  background-color: var(--color-surface-elevated);
  color: var(--color-text-muted);
  cursor: not-allowed;
}
.guided-turn-primary:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
}

.guided-turn-secondary {
  padding: var(--space-sm) var(--space-lg);
  background-color: transparent;
  color: var(--color-text);
  border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--font-size-base);
  font-family: inherit;
  min-height: var(--size-control);
  transition: background-color 0.1s ease, border-color 0.1s ease;
}
.guided-turn-secondary:hover:not(:disabled) {
  background-color: var(--color-surface-hover);
  border-color: var(--color-accent);
}
.guided-turn-secondary:disabled {
  color: var(--color-text-muted);
  cursor: not-allowed;
}
.guided-turn-secondary:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
}

.guided-schema-fields {
  display: flex;
  flex-direction: column;
}

/* ProposeChainTurn classes */
.guided-propose-step-actions {
  display: flex;
  gap: var(--space-sm);
  margin-top: var(--space-sm);
}

.guided-propose-edit-btn {
  padding: var(--space-xs) var(--space-sm);
  background-color: transparent;
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: var(--font-size-sm);
  font-family: inherit;
  min-height: var(--size-control-compact);
  transition: background-color 0.1s ease, border-color 0.1s ease;
}
.guided-propose-edit-btn:hover:not(:disabled) {
  background-color: var(--color-surface-hover);
  border-color: var(--color-accent);
}
.guided-propose-edit-btn:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
}

.guided-propose-secondary-btn {
  padding: var(--space-sm) var(--space-md);
  background-color: transparent;
  color: var(--color-text);
  border: 1px solid var(--color-border-strong);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: var(--font-size-base);
  font-family: inherit;
  min-height: var(--size-control);
  transition: background-color 0.1s ease, border-color 0.1s ease;
}
.guided-propose-secondary-btn:hover:not(:disabled) {
  background-color: var(--color-surface-hover);
  border-color: var(--color-accent);
}
.guided-propose-secondary-btn:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
}
```

Add corresponding entries to the `@media (prefers-reduced-motion: reduce)` block for `.guided-turn-primary`, `.guided-turn-secondary`, `.guided-propose-edit-btn`, and `.guided-propose-secondary-btn`.

---

### Dead Rules

The following classes exist in `guided.css` but have no matching usage in any `.tsx` or `.ts` file in the consumer directory:

| Class(es) | Lines | Status |
|-----------|-------|--------|
| `.guided-schema-required-section` | 736–739 | Dead — no consumer |
| `.guided-schema-optional-section` | 741–743 | Dead — no consumer |
| `.guided-schema-advanced-toggle` (+ hover, focus-visible variants) | 746–770 | Dead — no consumer |
| `.guided-schema-continue-btn` (+ modifier variants) | 779–805 | Dead — no consumer; SchemaFormTurn uses `.guided-turn-primary` instead |
| `.guided-schema-textarea--error` | 715–717 | Dead — no consumer applies this modifier; should be extended to input/select |

These suggest a planned "required/optional/advanced" section split and a separate continue-button that was superseded when `SchemaFormTurn` was implemented using `.guided-turn-primary`. Delete them or confirm they are planned for a future widget state; either way, the `@media (prefers-reduced-motion: reduce)` guard for `.guided-schema-advanced-toggle` and `.guided-schema-continue-btn` at lines 1392–1393 is also dead.

---

### Priority Recommendations

**Critical (Fix Immediately):**

1. **Define the 8 missing CSS classes** (`guided-turn-primary`, `guided-turn-secondary`, `guided-schema-fields`, `guided-propose-step-actions`, `guided-propose-edit-btn`, `guided-propose-secondary-btn`, plus `recipe-context-header` for RecipeContextHeader). Every SchemaFormTurn and ProposeChainTurn currently renders with browser-default unstyled buttons. This is a functional regression visible in the demo.
2. **Add `.guided-schema-input--error` and `.guided-schema-select--error`** modifier classes. Required field validation on text, number, and enum fields produces no visual error state. WCAG 3.3.1 violation.

**Major (Fix Before Demo):**

3. **Differentiate pressed-chip from primary-CTA visually** — add inset shadow or border-width to `[aria-pressed="true"]`.
4. **Add visual distinction between single-select and multi-select chip groups** — radio-like vs checkbox-like indicator via pseudo-element.
5. **Separate the custom-input row visually from preset chips** — border-top separator and a section label.
6. **Increase `.guided-multi-custom-remove-btn` to 32x32px minimum** or provide a hit-area expansion.
7. **Resolve the semantic-fallback issue** in warning/error token vars — remove fallbacks or choose a non-accent fallback colour.

**Minor (Cleanup):**

8. Extract the three `120px` occurrences into a single `--guided-kv-key-width` token.
9. Fix `line-height: 1` on `.guided-multi-custom-chip-label` to `1.2`.
10. Standardise CTA placement (primary always leftmost or always rightmost) across all action rows.
11. Remove the 5 confirmed-dead CSS rule blocks and their `@media` guard entries.

---

### Confidence Assessment

**High confidence** on the class-drift finding (confirmed by exhaustive `grep` across all frontend CSS and TSX files — 8 classes are referenced but defined nowhere). **High confidence** on the dead-rule finding (same grep method, confirmed absent from all TSX consumers). **Moderate confidence** on contrast and touch-target assessments — these require token resolution and runtime measurement to confirm exact values. The 24x24px measurement for the remove button is from the CSS source directly (`width: 24px; height: 24px`).

### Risk Assessment

The missing CSS class definitions affect two widget types that are central to the guided-mode flow: schema-form turns (all plugin configuration) and propose-chain turns (multi-step plan acceptance). Browser-default button styling for primary actions is a demo-critical regression. All other findings are polish or a11y improvements that would not block a demo but would fail a WCAG AA audit.

### Information Gaps

- Token values for `--color-warning`, `--color-error`, `--size-control`, `--size-control-compact` are not resolved here. Contrast ratios for warning text on `--color-surface-elevated` cannot be computed without these values.
- `RecipeContextHeader.tsx` uses `.recipe-context-header` with an unstyled `<h3>` and no CSS class — verify this is intentional (component-level inline styling) or add it to the findings.
- The `GuidedHistory` replay context (multiple turn instances coexisting) was not tested for ID collision under the `useId()` scoping. The pattern appears correct from source review.

### Caveats

This review assessed CSS structure and class-consumer alignment from static source. It did not involve runtime rendering, screen-reader testing, or contrast measurement against live token values. The accessibility findings (WCAG citations) are structural — they identify missing CSS facilities that make WCAG conformance impossible to achieve in the current implementation, regardless of how the JavaScript is written.
