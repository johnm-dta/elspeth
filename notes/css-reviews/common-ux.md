# UX/a11y Review: common.css

**File reviewed:** `src/elspeth/web/frontend/src/styles/common.css` (498 lines)
**Primary consumers read:** `CommandPalette.tsx`, `ErrorBoundary.tsx`, `YamlDisplay.tsx`, `YamlView.tsx`, `GraphModal.tsx`, `ExportYamlModal.tsx`, `MarkdownRenderer.tsx`
**Tokens cross-referenced:** `tokens.css`
**Date:** 2026-05-23
**Reviewer:** ux-critic agent

---

## Summary

**Overall: Needs Work**
**Critical Issues: 3**
**Major Issues: 5**
**Minor Issues: 8**

The modal close buttons are the single worst problem: they have no `:focus-visible` or `:hover` declarations at all, which means keyboard users cannot see whether the only explicit close affordance (beyond Esc) is in focus. The command palette input strips `outline` and substitutes a 1px border-colour swap that is marginal at best. The code-block copy button becomes visible on keyboard focus but has no ring once it is. These three together create a meaningful WCAG 2.4.7 failure pattern across this file's three interactive surfaces.

---

## 1. Modal Accessibility

### Strengths

- Both `GraphModal` and `ExportYamlModal` correctly set `role="dialog"`, `aria-modal="true"`, and `aria-labelledby` wired to a `useId`-generated id. The title association is solid.
- Both modals use `useFocusTrap` with an explicit initial-focus selector (`.graph-modal-close`, `.yaml-modal-close`), so focus lands on a meaningful target on open.
- Both modals attach a `document.addEventListener("keydown")` handler that calls `setIsOpen(false)` on Escape — discoverable, correct.
- Backdrops carry `aria-hidden="true"` and `onClick` close — correct suppression of the decorative layer.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| Modal close buttons have zero interactive-state styles | Critical | `common.css:434–442, 483–491`; `GraphModal.tsx:55–62`, `ExportYamlModal.tsx:56–63` | `.graph-modal-close` and `.yaml-modal-close` declare `min-width`, `min-height`, `border`, `border-radius`, `background-color`, `color`, `cursor` — **no `:hover`, `:focus-visible`, or `:active` rules anywhere in the file**. `useFocusTrap` deposits focus on these buttons; the user has no visible indication it landed. WCAG 2.4.7 (Focus Visible, AA) failure. | Add at minimum `:focus-visible { outline: 2px solid var(--color-focus-ring); outline-offset: 2px; }` and a `:hover` background shift (e.g. `var(--color-surface-hover)`). These buttons should mirror the `.btn` interactive-state pattern. |
| No scroll-lock cue | Major | `GraphModal.tsx`, `ExportYamlModal.tsx` | Neither modal sets `overflow: hidden` on `<body>` or equivalent; no CSS scroll-lock class is applied. Background content remains scrollable while a modal is open. | Add a `body.modal-open { overflow: hidden; }` rule (in base.css or here) and toggle the class in the modal open/close effect. |
| Backdrop opacity inconsistency | Minor | `common.css:211, 403, 452` | Command-palette backdrop: `rgba(0,0,0,0.5)`. Modal backdrops (graph + yaml): `rgba(0,0,0,0.45)`. No functional impact but signals ad-hoc layering rather than tokenised intent. | Introduce `--opacity-modal-backdrop` and `--opacity-palette-backdrop` tokens (or decide one value fits all and unify). |
| `.graph-modal-body` lacks `overflow: auto` | Minor | `common.css:444–447` vs `494–498` | `.yaml-modal-body` has `overflow: auto`; `.graph-modal-body` does not. Graph relies on `GraphView` inner scroll; YAML relies on body scroll. Asymmetric contract makes future consumer changes fragile. | Document the difference in a comment if intentional; if not, align. |

---

## 2. Command Palette

### Strengths

- `role="dialog"` + `aria-modal="true"` + `aria-label="Command palette"` on the palette container — correct.
- `role="combobox"` + `role="listbox"` + `aria-selected` + `aria-activedescendant` wiring in `CommandPalette.tsx` is textbook ARIA combobox pattern.
- Footer `<kbd>` tags expose keyboard affordances visually and semantically.
- `scrollIntoView({ block: "nearest" })` on selection change ensures long lists stay usable without mouse.
- `useFocusTrap` funnels initial focus to the search input, which is the correct starting point.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| `command-palette-input:focus` strips `outline` with no adequate replacement | Critical | `common.css:249–252` | `outline: none` removes the UA focus ring; the replacement is a `border-color` change from `--color-border-strong` (dark: `rgba(143,200,200,0.25)`) to `--color-focus-ring` (dark: `#ffffff`). This is a 1px-wide colour swap. WCAG 1.4.11 requires UI components to have 3:1 contrast against adjacent colour **on the component boundary** — a 1px border in an otherwise rounded input is marginal evidence of state change. The palette input is the only focusable thing on mount; the indicator must be unambiguous. | Replace `outline: none` with `outline: 2px solid var(--color-focus-ring); outline-offset: 0px;` and keep or remove the border-color change as desired. Do not suppress the outline without a stronger replacement. |
| No `@media (prefers-reduced-motion: reduce)` for palette item transitions | Major | `common.css:290–291` | `.command-palette-item { transition: background-color var(--transition-fast); }` — `--transition-fast` is `100ms ease`. Short but still active for users who have set the OS reduce-motion preference. | Add `@media (prefers-reduced-motion: reduce) { .command-palette-item { transition: none; } }`. |
| `command-palette-item-selected` hover rule redundancy / specificity conflict | Minor | `common.css:293–301` | Line 293: `.command-palette-item:hover, .command-palette-item-selected { background-color: var(--color-surface-hover); }`. Line 298: `.command-palette-item-selected { background-color: var(--color-highlight); }`. The second rule overrides the first for `.command-palette-item-selected` (same specificity, later in cascade). Hover on a selected item resolves to `--color-surface-hover` not `--color-highlight` due to the list ordering. Likely unintentional. | Merge into `.command-palette-item:hover { background-color: var(--color-surface-hover); }` and `.command-palette-item-selected { background-color: var(--color-highlight); }` separately — no sharing. |
| Group headers not associated with their option groups | Minor | `common.css:275–281`; `CommandPalette.tsx:330,363,396` | `.command-palette-group-header` divs label logical groups visually but carry no `id`; the surrounding `div.command-palette-group` has no `role="group"` or `aria-labelledby`. Screen readers announce the items without group context. | Add `role="group"` and `aria-labelledby={groupHeaderId}` to each `.command-palette-group` div; add matching `id` to the header div. |

---

## 3. YAML View

### Strengths

- `.yaml-view-line-number { user-select: none }` correctly excludes line numbers from clipboard copy. Good defensive practice.
- `aria-hidden="true"` on the `<span>` wrapping line numbers in `YamlDisplay.tsx` means SR users skip the noise. Correct.
- Loading state uses `role="status" aria-live="polite"` — correct for non-interruptive progress.
- `aria-label` on the Copy button changes between `"Copy YAML to clipboard"` and `"Copied to clipboard"` — announces to SR users who have focus on the button.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| Syntax-highlight token contrast is outside this file's control | Major (info) | `common.css:40` + `YamlDisplay.tsx:74` | `prism-react-renderer` `vsDark`/`vsLight` themes are imported as JS objects and rendered as inline `style` attributes. The CSS in `common.css` only enforces `background-color: var(--color-surface) !important` on the `<pre>` — it does NOT control token colours. Token contrast against `--color-surface` (dark: `#122f37`, light: `#ffffff`) has not been verified in this review. | Run a dedicated Prism token contrast audit in both themes. Flag any token that falls below 4.5:1 on the respective background. This is an **information gap** in this review. |
| `[data-copied="true"]` success cue is CSS colour-only at the rule level | Major | `common.css:394–397` | The CSS rule alone changes `background-color` and `color` — which would be colour-only signalling (WCAG 1.4.1 fail). However, `YamlDisplay.tsx` simultaneously changes button text ("Copy" → "Copied!") and `aria-label`. The cue is **not** colour-only in practice. **The CSS rule carries an implicit dependency on the JSX text change to be accessible.** Document this in a comment so future CSS-only refactors don't silently break the contract. | Add a comment: `/* Success state — colour change pairs with textContent "Copied!" + aria-label change in YamlDisplay.tsx; do not rely on CSS alone. */` |
| `.yaml-toolbar-btn` has no minimum height declaration | Minor | `common.css:389–392` | `padding: var(--space-xs) 10px` (4px top/bottom) + font-size-xs (12px) ≈ 20px tall — below WCAG 2.5.8's 24×24 minimum. The `.btn` base class likely supplies `min-height` (not verified in this file), but `.yaml-toolbar-btn` adds no explicit height guard. | Verify `.btn` base supplies `min-height: var(--size-control-compact)` (36px). If it does, this is safe. If not, add `min-height: var(--size-control-compact)`. |

---

## 4. Markdown Rendering

### Strengths

- `<pre>` blocks have `overflow-x: auto` — horizontal scroll works correctly on narrow viewports.
- `word-break: break-word` on `.markdown-body` prevents horizontal overflow for long unbreakable strings.
- Mermaid diagrams carry `role="img"` + `aria-label="Mermaid diagram"` — adequate minimal a11y for decorative-complex figures (though a more descriptive label tied to diagram content would be better).
- External links in `SafeLink` get `target="_blank" rel="noopener noreferrer"` — correct security pattern.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| h4 font-size equals body text | Major | `common.css:87–90` | h1=`--font-size-lg` (18px), h2=`--font-size-base` (16px), h3=`--font-size-md` (15px), h4=`--font-size-sm` (13px) = body text size. h4 is differentiated from body only by `font-weight: 600`. Low hierarchy resolution; cognitive load for navigating deeply nested documents. | Either raise h4 to a distinct size (e.g. `--font-size-base`) or reduce h levels to 3 (h1–h3) and remove h4 styling. |
| `.mermaid-fallback` uses `--color-text-muted` | Major | `common.css:192–195` | Fallback renders the raw diagram source as text when mermaid parsing fails. `--color-text-muted` (dark: `#7a9a9a`, light: `#426069`) is lower contrast than body text — an error state is rendered more quietly than success. Users may not notice the failure. | Use `--color-text` or `--color-warning` to signal degraded state. |
| `.code-block-copy` has no `:focus-visible` focus ring | Critical | `common.css:154–167` | The button has `opacity: 0` at rest and `opacity: 1` on `:hover` or `:focus-visible`. The `:focus-visible` rule only brings the button into view — it does not add a focus ring. After the button becomes visible to a keyboard user, there is no ring to confirm which element is active. WCAG 2.4.7 fail. | Add to the `:focus-visible` selector: `outline: 2px solid var(--color-focus-ring); outline-offset: 2px;` |
| `blockquote` colour contrast unverified | Minor | `common.css:197–202` | `--color-text-secondary` in dark mode is `#a8d0d0` on `--color-surface` `#122f37`. Estimated ratio ~5:1 (passes AA). Light mode: `#3a5a64` on `#ffffff` — estimated ~7:1. Both appear safe but are not mechanically verified in this review. | Add these pairs to `colorContrast.test.ts` coverage if not already present. |
| `SafeLink` does not warn on new-tab open | Minor | `MarkdownRenderer.tsx:79–87` | External links open `_blank` without a SR-readable hint (e.g. `" (opens in new tab)"` via `aria-description` or visually-hidden span). Users relying on SR may be disoriented. | Add `aria-description="opens in new tab"` or a visually-hidden span to external `<a>` elements. |

---

## 5. Error Boundary

### Strengths

- `role="alert"` on the fallback container — screen readers announce the error immediately on render, which is correct for unexpected failure states.
- `aria-hidden="true"` on the warning icon — decorative glyph correctly suppressed.
- Error message renders `this.state.error?.message` directly — no generic "something went wrong" that strips actionable information.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| Error boundary is colour-only distinction at the container level | Major | `common.css:340–377` | `.error-boundary-fallback` uses `--color-text-muted`, `--color-warning` (icon), and `--color-text` (title) — all colour-based differentiation. There is no border, background colour change, or pattern that distinguishes the error state from an empty/loading panel at the container level. Sighted users in low-colour-discrimination contexts may miss it. `role="alert"` covers SR; the visual layer needs a secondary cue. | Add `background-color: var(--color-error-bg); border: 1px solid var(--color-error-border);` to `.error-boundary-fallback`, or at minimum a left-border accent (e.g. `border-left: 3px solid var(--color-error)`). |
| No copy-error affordance | Minor | `ErrorBoundary.tsx:47–70` | The detail text includes the raw error message which developers need to copy for debugging. No copy button is provided. The `role="alert"` region is not selectable by default on all platforms. | Consider adding a "Copy error details" button (similar to `YamlDisplay`'s copy pattern) alongside Retry. |

---

## 6. Toolbar Buttons — [data-copied] State

### Strengths

- `aria-label` swap in `YamlDisplay.tsx` (line 81–82) and `FencedCodeBlock` (line 171) ensures SR users who focus the button post-click hear a meaningful state change.
- 2000ms timeout is deliberate and documented in `MarkdownRenderer.tsx` — long enough for users with higher cognitive load.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| No `aria-live` region for off-focus copy confirmation | Minor | `YamlDisplay.tsx:79–84`; `MarkdownRenderer.tsx:167–173` | If a pointing-device user clicks Copy and focus moves elsewhere (or never lands on the button), the textContent change to "Copied!" is not announced. `aria-label` attribute changes fire SR events only when the element is focused. For mouse/touch users, the state change is silent to SR. | Add a visually-hidden `aria-live="polite"` region adjacent to each copy button, updated to "Copied to clipboard" on success and cleared after 2 seconds. The `data-copied` CSS cue covers sighted users; the live region covers SR users regardless of focus position. |
| Copy button `aria-live` dependency is implicit | Minor | `common.css:394–397` | Documented above under YAML View. Repeat here for discoverability. | Comment in CSS: the colour swap alone is not sufficient for a11y — it depends on the sibling textContent change. |

---

## 7. Token Discipline

### Raw value instances (not using tokens where tokens exist)

| Line | Value | Available token | Notes |
|------|-------|-----------------|-------|
| 76 | `line-height: 1.5` | `--line-height-normal` | Direct token match; no justification for literal. |
| 142 | `line-height: 1.4` | None exact (`--line-height-tight: 1.3`, `--line-height-normal: 1.5`) | Gap in token set — 1.4 sits between two tokens. Flag as token-set gap, not a usage failure. |
| 165 | `opacity: 0` | N/A | Fine — opacity transitions are not tokenised. |
| 392 | `10px` padding | Token gap — 10px is between `--space-sm` (8px) and `--space-md` (12px) | Flag as token-set gap. |
| 409, 459 | `inset: 32px` | `--space-2xl: 32px` | Should be `inset: var(--space-2xl)`. |
| 199, 137, 138 | `10px 12px`, `8px 0`, `8px` margins/padding | Some map to `--space-sm`/`--space-md`; raw literals should use tokens | Audit margins/padding in the markdown-body block; most raw values have token equivalents. |
| 55, 59 | `width: 3ch` | N/A | Not a spacing token context — `ch` units for character-width alignment are appropriate. |
| 310, 329 | `padding: 2px 6px`, `1px` | `--space-2xs: 2px`, `--space-xs: 4px` | 6px and 1px have no tokens. Flag as token-set gaps. |

### Dead or unreferenced rules

- `.yaml-loading` (lines 382–387): used in `YamlView.tsx` loading branch. Not dead.
- `.yaml-view-display`: referenced in `YamlDisplay.tsx` line 77 as `className="yaml-view-display"` — but **there is no `.yaml-view-display` rule in `common.css`**. This class is applied to the root wrapper of `YamlDisplay` but has no styling. Either a rule was removed during the split refactor and forgot this class name, or it is intentionally unstyled (relying on `yaml-view` parent context). Verify.

---

## 8. Reduced Motion

No `@media (prefers-reduced-motion: reduce)` block exists anywhere in `common.css`. Affected transitions:

- `.command-palette-item { transition: background-color var(--transition-fast); }` (line 291)
- `.code-block-copy { transition: opacity var(--transition-fast); }` (line 166)
- `.yaml-toolbar-btn` inherits transition from `.btn` base — not visible in this file.

WCAG 2.3.3 (AAA) covers this; WCAG 2.1 AA does not mandate it. However, `prefers-reduced-motion` support is now standard practice and is particularly relevant for users with vestibular disorders. Recommend adding a block.

---

## Priority Recommendations

### Critical (Fix Before Merge)

1. **Add `:focus-visible` to `.graph-modal-close` and `.yaml-modal-close`** — keyboard users cannot see focus on the only explicit close button. `outline: 2px solid var(--color-focus-ring); outline-offset: 2px;` minimum.
2. **Replace `outline: none` on `.command-palette-input:focus`** with a proper `outline: 2px solid var(--color-focus-ring)` declaration. The 1px border swap is insufficient as the sole indicator.
3. **Add `:focus-visible` outline to `.code-block-copy`** beyond opacity reveal — making the button visible is not the same as showing a focus ring.

### Major (Fix Before Demo / Production)

4. **Add secondary visual cue to `.error-boundary-fallback`** beyond colour — `border-left: 3px solid var(--color-error)` minimum.
5. **Replace `--color-text-muted` on `.mermaid-fallback`** with `--color-text` or `--color-warning` — failure state must be at least as visible as success state.
6. **Fix h4 font-size** — `--font-size-sm` (13px) equals body text; differentiate by size not weight alone.
7. **Implement body scroll-lock** when modals open — currently background scrolls under open modals.
8. **Add `aria-live` region** for copy-confirmation announcements to cover mouse/touch SR users.

### Minor (Improvement)

9. Add `@media (prefers-reduced-motion: reduce)` blocks for `.command-palette-item` and `.code-block-copy` transitions.
10. Fix `.command-palette-item-selected` hover/selected rule split (line 293–301) — hover on selected resolves incorrectly.
11. Add `role="group"` + `aria-labelledby` to command palette category groups.
12. Inline-comment the `[data-copied]` CSS rule to document the JSX textContent dependency.
13. Replace `inset: 32px` with `inset: var(--space-2xl)` on both modal rules (lines 409, 459).
14. Replace `line-height: 1.5` on `.markdown-body` with `var(--line-height-normal)`.
15. Investigate `.yaml-view-display` — class applied in `YamlDisplay.tsx` but no rule exists in `common.css`.
16. Add `aria-description="opens in new tab"` to `SafeLink` for external links.

---

## Confidence Assessment

**High confidence:** Modal close button interactive-state gaps (directly verifiable from CSS + TSX). Command palette focus-ring suppression (directly observable). `.code-block-copy` focus-ring gap. Token misuse / missing tokens. Structural ARIA pattern correctness.

**Medium confidence:** Contrast ratios for `--color-text-secondary` on blockquote background — estimated from token hex values but not mechanically measured.

**Low confidence (requires tooling):** Prism syntax-highlight token contrast in both themes — colours are supplied by `prism-react-renderer` JS objects, not by `tokens.css` or `common.css`. Cannot be assessed from file review alone.

## Risk Assessment

**High risk:** The three critical focus-visible failures affect the only keyboard-navigable close affordance across two modal types and the copy button in markdown renderer. In a demo context where a presenter uses keyboard shortcuts (Ctrl+Shift+G, Ctrl+Shift+Y) to open modals, the broken close-button focus state is likely to be visible to observers.

**Medium risk:** The `aria-live` gap for copy confirmation is invisible in manual testing but detectable by SR users and automated a11y tools (axe-core).

**Low risk:** Token discipline lapses are cosmetic and do not affect functionality.

## Information Gaps

1. **Prism token contrast** — `vsDark`/`vsLight` token colours are external to this codebase and have not been measured against `--color-surface` / `--color-surface-elevated` in either theme.
2. **`.btn` base class interactive states** — `common.css` does not define `.btn`. It is assumed to supply `min-height`, `hover`, and `focus-visible` rules. If `.btn` is deficient, `.yaml-toolbar-btn` and `.error-boundary-retry` inherit those gaps. Verify in `base.css` or `shared.css`.
3. **`useFocusTrap` Tab/Shift-Tab loop** — focus trapping is implemented in JS (`useFocusTrap` hook), not CSS. Correctness of the trap loop (wrapping from last to first focusable element and vice versa) is out of scope for this CSS review.
4. **`--color-success` on `--color-success-bg` measured contrast** — the `[data-copied="true"]` state uses these tokens. Approximate dark-mode values: `#14b0ae` on `rgba(20,176,174,0.12)` over `#122f37` ≈ effective background `#162f35`-ish. Ratio not precisely calculated. Add to `colorContrast.test.ts`.

## Caveats

- This review covers CSS rules and direct component consumers. It does not cover: (a) runtime behaviour of `useFocusTrap` under rapid open/close cycles, (b) iOS VoiceOver vs macOS VoiceOver differences in combobox announcement, (c) forced-colors / Windows High Contrast mode (no `@media (forced-colors)` block exists anywhere in the file — a known gap project-wide).
- WCAG level references are AA unless stated. AAA items (reduced-motion, new-tab warning) are flagged as recommended practice, not compliance failures.
- The `colorContrast.test.ts` suite already enforces some pairings mechanically. Items not in that suite are estimated from hex values.
