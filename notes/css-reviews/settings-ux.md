# UX/A11y Review: settings.css (Secrets Panel)

**Reviewer:** UX Critic Agent
**Date:** 2026-05-23
**File:** `src/elspeth/web/frontend/src/components/settings/settings.css` (165 lines)
**Consumer:** `SecretsPanel.tsx`
**Overall:** Acceptable — solid security posture, good structural bones, five specific gaps require attention before demo.

---

## Confidence Assessment

**High confidence** on structural gaps (touch targets, missing states, forced-colors, dead selectors) — these are directly verifiable from the CSS and TSX source. **Medium confidence** on contrast ratios — no design token value file was available; findings assume token values are typical for this token name pattern but must be verified against the actual resolved palette.

## Risk Assessment

**Security-surface risk is the primary concern.** The delete button fires immediately without confirmation — on a secret-management panel, an accidental delete has no undo path. This is the highest-impact issue. All other findings are interaction polish or compliance gaps.

## Information Gaps

- Design token resolved values (e.g. `--color-text-secondary`, `--color-warning`, `--color-error`) are not available. Contrast ratio judgements below are conditional on token values meeting 4.5:1 against their respective backgrounds.
- No `SecretsStore.tsx` read — the store's `error` field type is assumed to be `string | null` based on usage pattern in `SecretsPanel.tsx`. This does not affect any CSS finding.

## Caveats

- The "no reveal toggle" design decision is explicitly documented in code comments as intentional security policy. This review does not treat its absence as a gap.
- `AvailabilityDot` and `ScopeBadge` styles are applied inline in TSX, not in settings.css. Issues with those elements are noted but attributed to the component, not the stylesheet.

---

## 1. Reveal / Hide Secret Affordance

**Finding: No reveal toggle exists — by design.**

The component security comment (line 79) states: "No 'show password' toggle is provided." The value input uses `type="password"`, cleared immediately on submit. This is a deliberate, documented security posture. No CSS work is needed here.

**Gap: The password input has no focus-visible style in settings.css.**

`.secrets-form-input` has no `:focus-visible` rule. Focus rings on inputs fall through to browser defaults, which are suppressed by many CSS reset patterns. There is no global `:focus-visible` override visible in this file.

| Issue | Severity | Element | Fix |
|-------|----------|---------|-----|
| No `:focus-visible` on `.secrets-form-input` | Major | `#secret-value`, `#secret-name` | Add `outline: 2px solid var(--color-accent); outline-offset: 2px;` under `.secrets-form-input:focus-visible` |

---

## 2. Copy Button

**Finding: No copy button exists.**

The inventory list shows name, availability dot, scope badge, and (for user-scoped secrets) a delete button. There is no copy affordance for the secret *name* (the key reference operators would paste into pipeline YAML). The focus of the brief assumed a copy-to-clipboard button; it is absent from both the CSS and the TSX.

This is either a scoped-out feature or an information gap. If it is in-scope for this branch, it is a **critical omission** on a secrets management surface — operators need to copy the exact key name string into their pipeline config.

| Issue | Severity | Element | Recommendation |
|-------|----------|---------|----------------|
| No copy-name affordance for secret references | Critical (if in scope) | `.secrets-list-item` | Add a copy button beside the name; use `aria-live="polite"` region for "Copied!" confirmation that auto-clears after 2s. Verify with operator whether this is in scope for this branch. |

---

## 3. Delete Action

**Critical gap: No confirmation step.**

`handleDelete` (SecretsPanel.tsx line 124–128) calls `deleteSecret(secretName)` directly. There is no "are you sure?" gate. On a secrets panel where the only recovery is re-entering the secret value, an accidental tap on the × button destroys a credential permanently.

The delete button visual treatment (`.secrets-delete-btn`) is correct in that it uses `var(--color-error)` for colour. However:

- The button label is the × character (`×`), which is identical to the modal close button's label. Screen readers reading the list will encounter multiple "×" buttons in sequence with only `aria-label` distinguishing them. The label `Delete secret ${secret.name}` (TSX line 313) is correctly set, so that gap is already closed. The visual symbol ambiguity with the close button remains for low-vision users.
- There is no `:hover` or `:focus-visible` state declared in settings.css for `.secrets-delete-btn`. Without hover feedback, the destructive button does not visually escalate on approach — a user may not realise they are hovering over a destructive control.
- `min-width: 44px` and `min-height: var(--size-control)` are present — touch target size meets iOS HIG.

| Issue | Severity | Element | Fix |
|-------|----------|---------|-----|
| No confirmation step before delete | Critical | SecretsPanel.tsx `handleDelete` | Add inline confirm state: on first click show "Confirm delete?" with a "Yes, delete" button and cancel; only call `deleteSecret` on second confirmation. |
| No `:hover` / `:focus-visible` on `.secrets-delete-btn` | Major | `.secrets-delete-btn` | Add `background-color: var(--color-error-bg)` on `:hover` and `outline: 2px solid var(--color-error); outline-offset: 2px` on `:focus-visible`. |
| × symbol reused for delete and close | Minor | `.secrets-delete-btn` | Consider a trash/bin icon or the word "Remove" at narrow widths; the `aria-label` is correct but visual disambiguation aids low-vision users who may not use AT. |

---

## 4. List Density and Scanability

**Strengths:**
- `.secrets-list-name` correctly applies `var(--font-mono)` — key names in monospace are scannable and copy-paste accurate.
- `word-break: break-all` prevents overflow on long names — correct.
- `gap: var(--space-xs)` between list items is compact but not cramped for a side-panel context.
- `.secrets-list-item` uses `border: 1px solid var(--color-border)` — each item has a clear boundary.

**Gaps:**

`.secrets-unavailable-reason` uses `var(--color-warning)` as its only distinguishing treatment. No icon is provided. In forced-colors mode (see Section 6), colour alone distinguishes unavailable state from the name text. This violates WCAG 1.4.1 (Use of Color) — the warning reason text relies on colour alone to convey "this secret is broken."

The `AvailabilityDot` component (TSX lines 28–57) does use dual cues (filled vs. hollow ring) with an `aria-label`. This dot addresses the availability signal correctly. The problem is specifically the inline warning text that follows it — text colour alone distinguishes it.

| Issue | Severity | Element | Fix |
|-------|----------|---------|-----|
| Warning reason text uses colour alone | Major | `.secrets-unavailable-reason` | Prepend a warning icon (e.g. `⚠` with `aria-hidden="true"`) in the TSX; the colour remains as a reinforcing cue, not the sole signal. |
| No truncation cue for very long secret names | Minor | `.secrets-list-name` | `word-break: break-all` wraps rather than truncates. For density, consider `overflow: hidden; text-overflow: ellipsis; white-space: nowrap` with a `title` attribute on the span for tooltip on hover. Long names currently reflow the list row unpredictably. |

---

## 5. Add Form

**Strengths:**
- Both inputs have explicit `<label>` elements with `htmlFor` associations — correct, not placeholder-only.
- `autoComplete="new-password"` on the value field suppresses credential manager autocomplete — correct security practice.
- The submit button is disabled until both fields are populated — correct validation gate.
- "Saving…" text during submission gives in-button feedback.

**Gaps:**

**Error state has no CSS class — it is fully inline-styled** (TSX lines 249–257). This means:
- The error banner cannot be targeted by forced-colors rules.
- The error banner cannot participate in any theming token swap.
- The inline `fontSize: 12` is a hardcoded px value outside the token system.

The error banner also lacks a distinct icon or border — it is distinguishable from normal text only by its background colour (`var(--color-error-bg)`). In forced-colors mode, `background-color` is overridden to `Canvas` — the error banner becomes invisible.

`.secrets-form-label` uses `var(--color-text-secondary)`. Depending on token values, secondary text can be as low as 3.5:1 contrast on a light surface. Labels at this contrast at `var(--font-size-xs)` (typically 11–12px) are at risk of failing WCAG 1.4.3 for small text (minimum 4.5:1). This must be verified against resolved token values.

The submit button uses `.btn.btn-primary` global classes plus `.secrets-submit-btn`. `.secrets-submit-btn` only sets `align-self` and `padding` — no disabled state styling is declared here. Disabled appearance falls to the global `.btn` styles, which may or may not provide sufficient contrast reduction to signal "not yet available." Verify the global `.btn:disabled` treatment.

| Issue | Severity | Element | Fix |
|-------|----------|---------|-----|
| Error banner fully inline-styled — invisible in forced-colors | Major | TSX error div (line 245) | Extract to `.secrets-error-banner` class in settings.css; add `outline: 2px solid var(--color-error)` so the border survives forced-colors. |
| Hardcoded `fontSize: 12` in error banner | Minor | TSX line 255 | Replace with `font-size: var(--font-size-xs)` via the CSS class. |
| Label contrast at `--color-text-secondary` + xs size — unverified | Major (conditional) | `.secrets-form-label` | Measure resolved token value; if below 4.5:1 at xs size, increase to `var(--color-text)` or raise the font size to 14px. |

---

## 6. Forced-Colors (Windows High Contrast / Contrast More)

**Critical gap: `AvailabilityDot` and `ScopeBadge` are entirely inline-styled.**

Both components apply all visual properties inline (TSX lines 39–55 and 12–25). In forced-colors mode, UA overrides `background-color`, `border-color`, `color`, and `box-shadow` on elements that are not explicitly mapped to system color keywords. The result:

- `AvailabilityDot`: The filled disc (`backgroundColor: #16a34a`) becomes `Canvas`. The hollow ring's `border` becomes `ButtonText`. Available and unavailable states collapse to the same visual representation — a hollow ring — because the fill is erased. The `aria-label` survives, so screen reader users are unaffected, but sighted high-contrast users lose the visual distinction.
- `ScopeBadge`: `backgroundColor` and `color` are both inline. The badge collapses to uncoloured text indistinguishable from the surrounding name.
- `AvailabilityDot`'s `box-shadow` halo (the "lit" cue) is zeroed by forced-colors.

**Fix path:** Move `AvailabilityDot` to a CSS class approach. Use `@media (forced-colors: active)` in settings.css to remap:

```css
@media (forced-colors: active) {
  .secrets-availability-dot[data-available="true"] {
    background-color: Highlight;
    border-color: Highlight;
    forced-color-adjust: none; /* opt-out only for this sentinel element */
  }
  .secrets-availability-dot[data-available="false"] {
    border-color: ButtonText;
    background-color: Canvas;
  }
  .secrets-scope-badge {
    outline: 1px solid ButtonText;
  }
}
```

| Issue | Severity | Element | Fix |
|-------|----------|---------|-----|
| AvailabilityDot available/unavailable states collapse in forced-colors | Critical | `AvailabilityDot` component | Add `data-available` attribute; style via CSS class with forced-colors media query block. |
| ScopeBadge loses visual distinction in forced-colors | Major | `ScopeBadge` component | Add `outline: 1px solid ButtonText` in forced-colors block as minimum identification. |
| Error banner invisible in forced-colors (background-only treatment) | Major | Error div in SecretsPanel.tsx | Add `outline: 2px solid var(--color-error)` in the extracted CSS class. |

---

## 7. Hardcoded Colours and Dead Selectors

**Hardcoded colours in TSX (not settings.css):**

The component ships several hardcoded hex values in inline styles:

| Location | Value | Impact |
|----------|-------|--------|
| TSX line 46 | `"var(--color-success, #16a34a)"` | Fallback ok; token-first is correct, fallback is a safety net. Acceptable. |
| TSX line 48 | `"var(--color-success, #16a34a)"` (border) | Same — acceptable. |
| TSX line 49 | `"var(--color-text-muted, #9ca3af)"` | Acceptable fallback pattern. |
| TSX line 51 | `"var(--color-success-bg, rgba(20, 176, 174, 0.12))"` | The fallback value `rgba(20, 176, 174, 0.12)` is teal, which does not match the green `#16a34a` used for the fill. This is a colour inconsistency — the fallback glow uses a different hue than the fallback fill. Low severity but visually incoherent when tokens are absent. |
| TSX line 161 | `backgroundColor: "var(--color-surface, #fff)"` | Acceptable fallback. |
| TSX line 163 | `boxShadow: "0 8px 32px rgba(0,0,0,0.25)"` | Hardcoded — not a token violation per se (shadows often aren't tokenised) but will not adapt to dark themes if a shadow token exists. Minor. |
| TSX line 249 | `marginTop: 12` (error banner) | Hardcoded spacing. Should be `var(--space-sm)` or similar. |
| TSX line 250 | `padding: "6px 10px"` | Hardcoded. |
| TSX line 251 | `borderRadius: 4` | Hardcoded. Should be `var(--radius-sm)`. |
| TSX line 255 | `fontSize: 12` | Hardcoded. Should be `var(--font-size-xs)`. |

**Dead selectors in settings.css:** None detected. All 16 class names defined in settings.css are referenced at least once in SecretsPanel.tsx. There are no orphaned selectors.

---

## 8. Accessibility Quick Check

| Criterion | WCAG | Status |
|-----------|------|--------|
| 1.4.3 Contrast — body text | AA | Unverified (token-dependent) |
| 1.4.3 Contrast — label text at xs | AA | At risk — verify `--color-text-secondary` on surface |
| 1.4.1 Use of Color — warning reason text | AA | **Fail** — colour alone distinguishes unavailable state |
| 2.1.1 Keyboard navigation | A | Pass — dialog, form, and list are keyboard-reachable; focus trap via `useFocusTrap` |
| 2.4.7 Focus Visible — inputs | AA | **Fail** — no `:focus-visible` in settings.css; falls to browser default which may be suppressed |
| 2.4.7 Focus Visible — delete button | AA | **Fail** — no `:focus-visible` on `.secrets-delete-btn` |
| 1.1.1 Alt Text — AvailabilityDot | A | Pass — `role="img"` with `aria-label` present |
| 3.3.2 Labels — form inputs | A | Pass — explicit `<label>` with `htmlFor` |
| 1.3.1 Info and Relationships — error | A | At risk — error banner is role="alert" (correct) but structurally role conveys semantics; verify AT announces it |

---

## Priority Recommendations

### Critical (Fix Before Demo)

1. **Add delete confirmation step.** The × button on a secrets panel deletes immediately and irreversibly. Implement a two-step confirm: first click transitions the button to "Confirm?" with a cancel option; second click calls `deleteSecret`. This is a TSX change, not a CSS change.

2. **Fix AvailabilityDot in forced-colors.** Move the dot to a CSS class with `data-available` attribute. Add a `@media (forced-colors: active)` block in settings.css that maps available/unavailable to `Highlight`/`ButtonText` system colors. Available and unavailable states are currently visually identical in high-contrast mode.

### Major (Fix Before Merge)

3. **Add `:focus-visible` to `.secrets-form-input` and `.secrets-delete-btn`.** Two focusable element types have no focus ring in this stylesheet. This fails WCAG 2.4.7 and makes keyboard navigation unusable if the global reset suppresses browser defaults.

4. **Extract the error banner to a CSS class.** The fully inline error div is invisible in forced-colors (background-only treatment). Extract to `.secrets-error-banner` with an `outline` that survives forced-colors.

5. **Add warning icon to `.secrets-unavailable-reason`.** Reason text uses colour alone to distinguish from normal name text. Prepend a `⚠` with `aria-hidden="true"` to add a non-colour cue.

6. **Add `:hover` / `:focus-visible` states to `.secrets-delete-btn`.** A destructive button with no hover escalation is invisible until the click event fires.

### Minor (Improvement)

7. **Resolve the glow colour mismatch** in `AvailabilityDot` (fallback shadow is teal, fill is green). Align fallback values so the dot is visually coherent when tokens are absent.

8. **Replace hardcoded spacing/radius/size values** in the error banner inline styles (`marginTop: 12`, `padding: "6px 10px"`, `borderRadius: 4`, `fontSize: 12`) with token references via the extracted CSS class.

9. **Add `ScopeBadge` outline in forced-colors block** so badge text remains identifiable as a badge element.

10. **Consider `text-overflow: ellipsis` on `.secrets-list-name`** for very long key names. Current `word-break: break-all` is safe but inflates row height unpredictably; truncation with a `title` tooltip may serve dense lists better.
