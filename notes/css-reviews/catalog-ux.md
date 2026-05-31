# Design Review: catalog.css — Catalog Drawer & Plugin Browser

**Reviewed:** 2026-05-23
**Files:** `catalog.css` (544 lines) + `CatalogDrawer.tsx`, `PluginCard.tsx`, `FilterChipStrip.tsx`, `AuditCharacteristicIcon.tsx`, `InlineChatSourceEntry.tsx`
**Tokens from:** `tokens.css`

---

## Summary

**Overall:** Acceptable — the fundamentals are solid and the architecture is
clearly intentional, but five issues need fixing before the demo. Two are
accessibility hard failures; the rest are confidence risks.

**Critical Issues:** 2
**Major Issues:** 4
**Minor Issues:** 5

---

## 1. Drawer Accessibility

### Strengths

- Focus trap is implemented via `useFocusTrap` with correct Tab/Shift+Tab
  wrapping and focus restoration on close. This is the hardest part of drawer
  a11y and it is done correctly.
- `role="dialog"` + `aria-modal="true"` + `aria-labelledby="catalog-drawer-title"`
  — full modal semantics, correctly wired to the visible title element.
- Escape key closes the drawer (in the `useEffect` keydown handler). The
  handler runs only when `isOpen` is true so there is no stale listener risk.
- Backdrop `onClick` closes the drawer. Correct.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Backdrop not keyboard-accessible | Critical | `CatalogDrawer.tsx:343-347` | The backdrop `<div>` handles `onClick` but carries no `role`, `tabIndex`, or `onKeyDown`. A keyboard user who somehow bypasses the focus trap lands on the backdrop with no way to activate it. Add `role="button"`, `aria-label="Close plugin catalog"`, `tabIndex={0}`, and an `onKeyDown` that fires `onClose` on Enter/Space. This is belt-and-suspenders because the trap *should* prevent it, but WCAG 2.1.1 requires any click-triggered functionality to be keyboard-accessible on its own. |
| `if (!isOpen) return <div style={{ display: "none" }} />` | Major | `CatalogDrawer.tsx:338` | This renders a hidden element into the DOM when the drawer is closed. An `aria-hidden="true"` element with `display:none` is invisible to the AT, so the practical impact is low. But the empty `<div>` is unnecessary clutter and the right pattern for a conditional panel is a null return or conditional mount at the call site. If persistence is needed (schema cache) consider moving the cache up to the parent. |
| Focus does not auto-advance to search field on open | Major | `useFocusTrap.ts:35` | `useFocusTrap` focuses the first focusable element in the container, which is the close button (top-right). For a search-first catalog the more useful initial target is the search input. Pass `initialFocusSelector="input[type='text']"` to route initial focus to the search field, saving the user a Tab press. This is not a hard failure but it is a meaningful efficiency regression for keyboard users. |

---

## 2. Search Field

### Strengths

- `aria-label="Search plugins"` on the input — correctly labelled (no
  visible label element is present, so this is necessary and correct).
- Clear button: `min-width: 28px; min-height: 28px` is documented as
  intentionally above the WCAG 2.5.8 AA 24×24 floor. The comment in the CSS
  explains the rationale. Good.
- Clear button carries `aria-label="Clear search"` and returns focus to the
  search input after clearing — the correct UX pattern.
- Focus ring on `.catalog-search-clear:focus-visible` is present.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Search input has no visible focus ring in CSS | Major | `catalog.css:324-332` | `.catalog-search-input` defines border, background, padding, and font but no `:focus-visible` state. The browser default outline may be suppressed by the project's global reset. Check `base.css`; if `outline: none` or `outline: 0` appears on `input`, add an explicit `:focus-visible` rule with `outline: 2px solid var(--color-focus-ring); outline-offset: 2px` to `.catalog-search-input:focus-visible`. WCAG 2.4.7 Level AA. |
| Clear button 28×28 sits inside padded search box (overlap risk) | Minor | `catalog.css:334-355` | The right padding on `.catalog-search-input` is `28px`, matching the clear button's max width. This is tight: browser zoomed or with a large system font can cause the clear button to visually clip the typed text. Consider raising the right padding to `34px` when the clear button is present, or implementing this as an inline-end padding toggle via a `.has-clear` class on the container. |

---

## 3. Tab Strip + Filter Chips

### Strengths

- `role="tablist"` + `aria-label="Plugin type tabs"` on the strip container.
  Individual tabs use `role="tab"` + `aria-selected`. This is correct ARIA
  tablist semantics.
- Filter chips use `aria-pressed` (they are toggle buttons, not tabs). This
  is the correct role; `aria-selected` would be semantically wrong here.
- `:focus-visible` ring on `.filter-chip` uses the design token
  (`--color-focus-ring`) with 2px offset. Correct.
- Per-tab filter state (one `CatalogFilters` per `CatalogTab`) — the design
  decision is documented in the component and in memory. Not a UX issue.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Tabs do not support arrow-key navigation | Critical | `CatalogDrawer.tsx:419-447` | ARIA pattern for `role="tablist"` (APG Tabs pattern) requires Left/Right arrow keys to move between tabs, Tab to move focus to the active tab panel. The current implementation relies entirely on Tab key, which forces keyboard users to Tab through all three tab buttons every time they change tabs. Add an `onKeyDown` on the `tablist` div that intercepts ArrowLeft/ArrowRight, updates `activeTab`, and calls `.focus()` on the newly-active tab button. Additionally, inactive tab buttons should carry `tabIndex={-1}` and the active tab `tabIndex={0}` (the roving tabindex pattern) so Tab skips past them. Without this the component does not conform to WCAG 2.1.1. |
| `.filter-chip-active` has no dedicated focus style | Minor | `catalog.css:461-465` | An active chip already uses `border-color: var(--color-accent)` and `background-color: var(--color-info-bg)`. When the chip is also focused the `:focus-visible` outline at 2px white on the muted teal border is correct, but `filter-chip-active:focus-visible` has no rule — it inherits the base `:focus-visible` block from line 456. Verify the compound state is tested; if the active-chip border and the focus ring visually merge, add an explicit compound selector. |
| `.catalog-tab-count` contrast: `--inactive` variant | Minor | `catalog.css:391-394` | `catalog-tab-count--inactive` renders `--color-text-muted` text on `--color-surface-elevated` background. Dark mode: `#7a9a9a` on `#1a3d47`. Approximate contrast: ~3.1:1. For a 10px (`font-size-3xs`) badge, WCAG 1.4.3 requires 4.5:1 at small sizes. This fails AA for the count badge text. Either raise the muted text to `--color-text-secondary` on inactive tabs, or accept the badge as decorative and add `aria-hidden="true"` to the count `<span>` (the label already announces the tab name). |
| Filter chip group label is plain text with no `id`/`aria` association | Minor | `FilterChipStrip.tsx:109-115` | `.filter-chip-group-label` is a `<span>`, not associated with the chip `<div>` via `aria-labelledby` or `role="group"`. Screen reader users traversing the chip buttons will not hear "Capability:" or "Audit:" as a group label. Wrap each `ChipGroup` in a `<fieldset>` with a `<legend>`, or add `role="group"` and `aria-labelledby` wiring. |

---

## 4. Plugin Card

### Strengths

- Audit icon strip uses `.plugin-card-audit-strip` with `aria-label="Audit
  characteristics"` as a labelled container. Each `AuditCharacteristicIcon`
  carries a `title` attribute for tooltip. Glyphs are `aria-hidden`.
- `plugin-card-desc` uses `-webkit-line-clamp: 2` with a `title` attribute
  for full-text disclosure. Correct pattern for truncated content.
- `max-width: 100%` on `.audit-icon` prevents overflow from the card edge.
  Deliberate and working.
- Disclosure/Details buttons carry `aria-expanded` and descriptive
  `aria-label` (includes plugin name). Correct.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.plugin-card-kind` contrast: `--color-text-muted` at `font-size-3xs` | Major | `catalog.css:69-75` | Dark mode: `#7a9a9a` on `#122f37` (~3.3:1). At 10px uppercase this fails WCAG 1.4.3 (needs 4.5:1 at small text, 3:1 only applies at 18px+ or bold 14px+). Uppercase does not exempt small text from the 4.5:1 requirement. Raise to `--color-text-secondary` (`#a8d0d0`, ~5.8:1 on surface dark) or bump the kind label size to 11px bold and verify it clears 3:1 as large text. |
| `plugin-card-detail-toggle` and `plugin-card-disclosure` min-height: 30px | Minor | `catalog.css:139-143` | Both buttons set `min-height: 30px`. The project's `--size-control-compact` is 36px (WCAG 2.5.8 AA). A 30px button sits below the AA compact floor. Raise to `min-height: var(--size-control-compact)` or add the `.btn-compact` class. For a reference surface (not a primary action) 30px may be an acceptable design tradeoff, but it should be documented in a comment identical to the pattern used for `.catalog-search-clear`. |
| `plugin-card-variant-label` and `plugin-card-variant` carry no CSS rule | Minor | `catalog.css` (absent) | `PluginCard.tsx` renders `.plugin-card-variants`, `.plugin-card-variant`, and `.plugin-card-variant-label` (lines 203-212) but none of these selectors appear in `catalog.css`. They inherit generic body styles. Document whether these are intentionally unstyled or add a minimal grid/gap/border rule to visually separate discriminated schema variants. |

---

## 5. Empty / Loading / Error States

### Strengths

- Loading state: `role="status"` + `aria-live="polite"` on the catalog list
  loading message. Schema loading inside PluginCard also carries these
  attributes. Correct live-region pattern.
- Empty-state message changes copy based on whether filters are active
  ("No plugins match the active filters." vs "No plugins available."). This is
  correct — it prevents users from thinking the catalog is empty when they have
  active filters.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Error state missing `role="alert"` | Major | `CatalogDrawer.tsx:461-470` | The fetch-error `<div>` with `className="catalog-status-message catalog-status-message--error"` has no live-region role. A user waiting for the catalog to load will not hear the failure announcement when it appears. Add `role="alert"` or `aria-live="assertive"` to the error element. The Retry button is present and correctly labelled. |
| Loading and empty states share identical visual treatment | Minor | `catalog.css:480-495` | `.catalog-status-message`, `.catalog-status-message--error`, and `.catalog-status-message--center` are the only variants. Loading (polite) and empty (centre-aligned) are visually identical except for text content. A user with low vision cannot distinguish "still loading" from "zero results." Consider a spinner glyph for loading (the `<span className="spinner">` pattern already exists in PluginCard) or a distinct text treatment (e.g. `font-style: italic` for empty, `color: --color-text` at full brightness for error). |

---

## 6. Shortcut Help Block

The `kbd` styling for the shortcut help block (`.shortcuts-group`, `.shortcuts-list`) is
not defined in `catalog.css`. The comment at the top of the file notes that
`kbd` descendant styling was moved to `common.css`. This is confirmed in scope.

### Strengths

- `.shortcuts-list-item` uses `display: flex` with `justify-content: space-between`
  to pair shortcut keys against their descriptions. Readable table-like layout.
- `dd` uses `--color-text-secondary` which in dark mode is `#a8d0d0` on
  `#122f37` (~5.8:1). Passes AA.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `dt` has no minimum width floor enforcement on mobile | Minor | `catalog.css:34-37` | `.shortcuts-list-item dt` sets `flex-shrink: 0; min-width: 80px`. At very narrow viewports (320px), 80px for the `dt` column leaves only 240px for the `dd`. This is workable but may feel tight. No critical failure; worth a visual check at 320px. |

---

## 7. Mobile: Drawer Behaviour on Narrow Viewport

### Strengths

- `width: min(440px, calc(100% - 24px))` — the drawer caps at 440px wide and
  leaves 24px of backdrop visible on narrow screens. This is the correct
  swipe-to-dismiss affordance gap. Well handled.
- `position: absolute` on both backdrop and drawer means the drawer is
  scoped to its positioned ancestor (the inspector panel), not the full
  viewport. This is an architectural decision that limits full-screen overlay
  behaviour; it is appropriate for a reference panel embedded in a split layout.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| 24px backdrop gap exposes the parent UI but has no touch swipe dismiss | Minor | `catalog.css:260-267` | On touch devices, tapping the 24px sliver of backdrop is the only non-button close affordance. The keyboard Escape handler works for keyboard users. A `touchstart`/`touchmove` swipe gesture on the drawer panel toward the right would be the conventional mobile close pattern. This is a JS concern, not a CSS concern, but should be noted for the CatalogDrawer implementation. |
| Filter chip strip may overflow on very narrow drawer | Minor | `catalog.css:401-431` | The `.filter-chip-row` uses `flex-wrap: wrap`. At 320px total width the drawer is 296px (minus 24px backdrop gap). The `.filter-chip-group` uses a fixed `grid-template-columns: 74px minmax(0, 1fr)`, leaving 222px for chips. Long chip labels will wrap which is correct, but verify at 320px that the first chip in each row isn't orphaned at a line break that makes the group look empty. |

---

## 8. Token Discipline

### Strengths

- All colours reference design tokens. No literal hex values in `catalog.css`.
- Spacing uses token steps (`--space-xs`, `--space-sm`, `--space-md`) except
  for a handful of hard-coded pixel values (2px, 4px, 6px) which align to the
  token steps but use literals. The 2px gap in `.plugin-card-header-row` margin
  and `.plugin-card-prose-section` gap could use `--space-2xs` for consistency.
- Z-index values use named tokens (`--z-catalog-backdrop`, `--z-catalog`).
  The stacking order (backdrop 38, catalog 40) is well below dialogs (200) and
  the palette (300).

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `font-weight: 650` is non-standard | Minor | `catalog.css:107` | `.audit-icon` sets `font-weight: 650`. Variable fonts may support numeric weights between 600 and 700, but 650 is not a CSS keyword and will silently round to 600 or 700 in non-variable fonts. Inter (the project's declared body font) is a variable font and does support 650, but this is fragile. Use 600 (`semibold`) or 700 (`bold`) and rely on the declared font stack. |
| Literal `4px` gap values in three places | Minor | `catalog.css:93, 430` | `.plugin-card-audit-strip` and `.filter-chip-row` both use `gap: 4px` literally. `--space-xs` is 4px. Use the token for consistency; the value is identical but the indirection is the point. |
| `inline-chat-source-entry-try` min-height 30px | Minor | `catalog.css:542` | Same as `plugin-card-detail-toggle` — 30px is below `--size-control-compact` (36px). Document if intentional or raise. |

---

## Accessibility Quick Check

| Criterion | WCAG | Result |
|-----------|------|--------|
| 1.4.3 Text contrast — body/secondary text | AA (4.5:1 small) | Pass (dark: ~5.8:1, light tokens per colorContrast.test.ts) |
| 1.4.3 Text contrast — muted text (kind label, tab count) | AA | **FAIL** — `--color-text-muted` at 10px is ~3.1:1–3.3:1 |
| 1.4.11 UI component contrast | AA (3:1) | Pass (borders, focus rings) |
| 2.1.1 Keyboard: all functionality accessible | AA | **FAIL** — tab strip arrow-key pattern missing, backdrop click not keyboard-wired |
| 2.4.7 Focus visible | AA | Partial — chips and clear button have explicit rings; search input ring unverified |
| 1.1.1 Alt text / non-text content | AA | Pass — glyphs are aria-hidden, audit icons have title attributes |
| 3.3.2 Labels/instructions | AA | Partial — filter chip groups lack group labelling |
| 4.1.3 Status messages | AA | **FAIL** — fetch error state missing live region role |

---

## Priority Recommendations

### Critical (Fix Before Demo)

1. **Tab strip arrow-key navigation.** Add `ArrowLeft`/`ArrowRight` handler on
   the `tablist` and implement roving `tabIndex`. This is the standard ARIA
   Tabs pattern; AT users will attempt it. (`CatalogDrawer.tsx`)

2. **Fetch-error live region.** Add `role="alert"` to the error
   `catalog-status-message--error` element so screen readers announce the
   failure. One attribute. (`CatalogDrawer.tsx`)

### Major (Fix Before Launch)

3. **Search input `:focus-visible` ring.** Verify global reset is not
   suppressing the browser default; add an explicit rule if needed.
   (`catalog.css`)

4. **`.plugin-card-kind` contrast.** At `font-size-3xs` (10px), `--color-text-muted`
   fails 1.4.3. Raise to `--color-text-secondary`. (`catalog.css:69-75`)

5. **Backdrop keyboard accessibility.** Add `role="button"`, `tabIndex={0}`,
   `aria-label`, `onKeyDown` to the backdrop `<div>`. (`CatalogDrawer.tsx`)

6. **Initial focus target.** Pass `initialFocusSelector="input[type='text']"`
   to `useFocusTrap` so the drawer opens with focus on search. (`CatalogDrawer.tsx`)

### Minor (Polish)

7. Filter chip group labelling via `role="group"` + `aria-labelledby`.
8. Document or raise `plugin-card-detail-toggle` and `inline-chat-source-entry-try`
   30px height to `--size-control-compact`.
9. Replace `font-weight: 650` with 600 or 700.
10. Replace literal `4px`/`2px` gap values with `--space-xs`/`--space-2xs` tokens.
11. Add CSS rules for `.plugin-card-variants`/`.plugin-card-variant`/`.plugin-card-variant-label`
    or document intentional unstyled state.

---

## Confidence Assessment

**High confidence:** Structural HTML/ARIA correctness, focus trap implementation,
live-region gaps, tab strip keyboard pattern (these are verifiable from source
without runtime).

**Medium confidence:** Contrast ratios — calculated from raw token hex values
using the WCAG relative-luminance formula. Alpha-blended tokens (borders, chip
backgrounds) depend on the actual rendered stack which was not measured at
runtime. `colorContrast.test.ts` is cited in the token file as the project's
verification harness; that test should be treated as the ground truth.

**Lower confidence:** Mobile viewport behaviour at 320px — not verified in a
real browser. The CSS is structurally sound but wrap behaviour at the clip point
has not been witnessed.

## Risk Assessment

**Highest risk:** The arrow-key tab navigation gap. AT users testing keyboard
access will hit this on first tab-strip interaction. This is the most visible
demo risk.

**Second risk:** Fetch error live-region absence. If the catalog API is slow or
fails during the demo and the operator is using a screen reader, the failure is
silent.

## Information Gaps

- The project's `colorContrast.test.ts` and `statusBadgeAccessibility.test.ts`
  test alpha-blended tokens against rendered backgrounds. The results of those
  tests for catalog-specific tokens (audit icon variants, tab count badges)
  were not available to this review.
- Whether the global `base.css` reset suppresses `input` focus outlines is not
  confirmed. If it does not, `.catalog-search-input` may inherit a visible
  browser default ring that covers the gap identified in finding #4.
- The `useFocusTrap` hook does not set `aria-modal="true"` — the component
  does this directly. Confirm the trap correctly handles Safari VoiceOver's
  modal containment (VoiceOver does not respect `aria-modal` on all versions
  and may read outside the drawer).

## Caveats

This review is CSS and JSX static analysis only. Runtime contrast, touch
behaviour, and AT compatibility (particularly VoiceOver on iOS in the
`position: absolute` scoped drawer) require manual testing. The focus trap
implementation is correct per the spec but was not tested with a screen reader
traversing the drawer at runtime.
