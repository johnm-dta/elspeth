# header.css — UX/Accessibility Review

**Reviewer**: ux-critic agent
**Date**: 2026-05-23
**File**: `src/elspeth/web/frontend/src/components/header/header.css` (311 lines)
**Consumers read**: `AppHeader.tsx`, `UserMenu.tsx`, `HeaderSessionSwitcher.tsx`,
`HeaderVersionSelector.tsx` (in `components/header/`), `RunsHistoryDrawer.tsx`,
`App.tsx` (alert-banner instances)
**Tokens consulted**: `tokens.css` (`:root` dark defaults + `[data-theme="light"]` overrides)

---

## Summary

**Overall**: Needs Work — two Critical issues, five Major issues, four Minor issues.

| Severity | Count |
|----------|-------|
| Critical | 2 |
| Major | 5 |
| Minor | 4 |

The two critical issues are hard functional gaps: six CSS classes rendered by
`HeaderSessionSwitcher.tsx` and four by `RunsHistoryDrawer.tsx` have no rules
anywhere in the codebase, so filter controls, error alerts, and the drawer
itself are entirely unstyled. The major issues cover focus consistency (`:focus`
vs `:focus-visible`), destructive-action colour-only signalling, run-status
indistinguishability, mobile absence, and a load-bearing undefined token. Minor
issues are tokenisation gaps and the un-tokenised box-shadow.

---

## Visual Design

### Strengths

- Almost all colours, spacings, radii, and font sizes are tokenised correctly —
  the deviations are enumerated in Issues below and are few.
- `.run-failure-detail` signals error with border colour, thickened left accent,
  and tinted background — three independent channels, not colour alone.
- Font sizes within the header use `var(--font-size-xs)` / `var(--font-size-sm)`,
  appropriate for the chrome-row register.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| `var(--color-bg-hover, …)` uses an undefined token — the fallback `rgba(143, 200, 200, 0.08)` is load-bearing | Major | Line 128 (session-switcher hover/focus) | `tokens.css` defines no `--color-bg-hover`. The fallback fires on every render. This is a silent token-gap: adding `--color-bg-hover` to another file at a different value would silently override the intent here. | Either define `--color-bg-hover` in `tokens.css` (recommended) or rewrite the selector to use an already-defined token such as `var(--color-surface-hover)`. |
| `z-index: 100` (session-switcher menu) is a literal — `--z-overlay: 100` exists in tokens | Minor | Line 106 | `tokens.css` line 201: `--z-overlay: 100`. | Replace with `var(--z-overlay)`. |
| `z-index: 50` (user-menu list) is a literal — below `--z-overlay: 100` but above `--z-panel-controls: 10` with no named tier | Minor | Line 190 | No token for this layer exists. | Define `--z-header-dropdown: 50` in tokens or reuse `--z-overlay`. |
| `box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18)` (user-menu list) is un-tokenised | Minor | Line 198 | All other shadow usage in the codebase is either absent or inline. | Extract to `--shadow-dropdown` in tokens or accept as a deliberate one-off with a comment. |
| `var(--font-mono, monospace)` uses a redundant fallback — `--font-mono` is always defined | Minor | Line 292 (`run-failure-detail pre`) | `tokens.css` line 150 defines `--font-mono`. | Drop `, monospace`. |

---

## Information Architecture

### Strengths

- `.header-session-switcher-menu` limits height with `max-height: 60vh` and
  `overflow-y: auto` — long session lists won't push off-screen.
- The `min-width: 240px` / `200px` constraints on both dropdown menus give
  adequate reading room without unconstrained growth.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| Six `HeaderSessionSwitcher` classes have zero CSS rules anywhere in the codebase: `header-session-switcher-controls`, `header-session-switcher-filter`, `header-session-switcher-show-archived`, `header-session-switcher-archive-error`, `header-session-switcher-rename-error`, plus the wrapping `header-session-switcher-item-session` class | Critical | `HeaderSessionSwitcher.tsx` lines 280–301, 370–376 | `grep -r header-session-switcher-controls src/` returns nothing from CSS files. The filter/show-archived controls and both inline error alerts (`role="alert"`) render as unstyled raw HTML. Error text will be invisible or unstyled in the menu. | Add rules to `header.css`. At minimum: layout for `header-session-switcher-controls` (flex row, gap); input sizing for `header-session-switcher-filter`; error-banner styling for `archive-error` and `rename-error` (mirror `.alert-banner` semantics). |
| Four `RunsHistoryDrawer` classes have zero CSS rules: `runs-history-drawer`, `runs-history-drawer-header`, `runs-history-drawer-body`, `runs-history-list`, `runs-history-item` | Critical | `RunsHistoryDrawer.tsx` lines 119–145 | Same grep — no CSS file defines these. The `role="dialog"` drawer renders without layout, positioning, scroll containment, or visual chrome. Focus trap is wired (`Tab` key), but the drawer is likely invisible or zero-width. | Add rules to `header.css` (per the file's tenant comment: "Tenants rendered from the header session-switcher dropdown"). The drawer needs at minimum: fixed/absolute positioning with correct z-index, background surface, overflow-y scroll on the body region, and a visible close affordance area. |
| No mobile breakpoint — five-element left cluster plus right user menu will overflow on narrow viewports | Major | Entire file | `header.css` has zero `@media` rules. Peer files (`chat.css`, `inspector.css`) use 760px and 520px breakpoints. `.app-header { height: 40px }` is fixed. | Add `@media (max-width: 760px)` block: consider collapsing session switcher to icon-only or hamburger; ensure version selector remains accessible. |

---

## Interaction Design

### Strengths

- `.user-menu-action:focus-visible` correctly adds `2px solid var(--color-focus-ring)` with `outline-offset: -2px` — WCAG 2.4.7 passes for the user menu.
- `.user-menu-trigger:focus-visible` also adds a `2px` ring with `outline-offset: 2px` — correct outward placement for a contained trigger.
- `--size-control-compact: 36px` is explicitly applied to session-switcher actions and explicitly commented (lines 147–148) as meeting WCAG 2.5.8 (AA ≥24px). Intentional and documented.
- `.user-menu-action` uses `--size-control: 44px` (the WCAG 2.5.5 AAA floor) — full target size for the user dropdown items.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| Session-switcher item uses `:focus` not `:focus-visible` | Major | Lines 124–129 (`.header-session-switcher-item:focus`) | The rule fires on mouse click, painting a hover-identical background. Mouse users see no state change; keyboard users see no distinct keyboard-focus indicator (no outline, only the hover background). Contrast with `.user-menu-action:focus-visible` which adds a 2px ring. WCAG 2.4.7 requires visible focus. | Replace `:focus` with `:focus-visible` and add `outline: 2px solid var(--color-focus-ring)` so keyboard nav is visually distinct from hover. The roving-tabindex model in `HeaderSessionSwitcher.tsx` (line 74: `itemRefs.current[focusIndex]?.focus()`) means keyboard focus reliably lands on the correct item — the ring just needs to paint. |
| `user-menu-action--danger` (Sign out) is distinguished by colour only | Major | Lines 233–241 | `.user-menu-action--danger { color: var(--color-error) }` — no separator above it, no icon, no prefix text. Fails WCAG 1.4.1 (use of colour) for users with colour-vision deficiency. The in-file comment (lines 228–232) justifies subordinate treatment but doesn't address the colour-only concern. | Add a visual divider (`border-top: 1px solid var(--color-border)` on the `<li>`) and/or a destructive icon (e.g. `⚠` with `aria-hidden="true"`) before "Sign out". The `UserMenu.tsx` icon approach is already used for the theme toggle (line 107). |
| `app-header` uses `height: 40px` (fixed) rather than `min-height` | Minor | Line 62 | Fixed height will clip content if a long session name or scaled font causes any item to overflow. The brand span and buttons should be able to grow the container. | Change to `min-height: 40px` (keep `flex-shrink: 0` to prevent collapse in the column layout). |

---

## Accessibility

### Quick Check

- [x] 1.4.3 Contrast (text): Alert-banner text uses semantic colour tokens (`--color-error`, `--color-info`) on 12%-alpha tinted backgrounds composited over body. **Unverified at pixel level** — the project's `colorContrast.test.ts` should be extended to cover `--color-error` on `--color-error-bg` composited over `--color-surface-nav` (the header surface). Flag for testing.
- [ ] 1.4.1 Use of Colour: FAILS for `user-menu-action--danger` (colour-only distinction) and for run status display (see Focus #4 below).
- [x] 2.1.1 Keyboard: Session-switcher (`role="menu"`, roving tabindex, ArrowDown/Up/Home/End/Enter/Escape) and version-selector (`role="listbox"`, ArrowDown/Up/Escape) are both keyboard navigable. User menu uses Tab+Escape disclosure pattern (per in-file comment — correct choice).
- [x] 2.4.7 Focus Visible: PASSES for user-menu (`:focus-visible` + 2px ring). FAILS for session-switcher items (`:focus` background-only, no ring).
- [x] 1.1.1 Alt Text: No images in this file's scope.
- [x] 3.3.2 Labels: Rename input uses `aria-label="Rename session"`. Filter input uses `aria-label="Find a session…"`. Both are explicit labels.
- [ ] 4.1.3 Status Messages: Archive-error and rename-error use `role="alert"` — correct. However these nodes currently have no CSS, so they may not be visually rendered. A `role="alert"` region that is invisible to sighted users but announced to screen readers creates an asymmetric experience.

### Issues

| Issue | Severity | Location | Evidence | Fix |
|-------|----------|----------|----------|-----|
| Run status indicators have no visual differentiation beyond plain text | Major | `RunsHistoryDrawer.tsx` lines 147–150; `runs-history-item-summary` grid | `run.status.replace(/_/g, " ")` renders as unstyled text inside `.runs-history-item-status`. No CSS class variants for `failed`, `cancelled`, `completed`. There is no colour-only risk (WCAG 1.4.1) because no colour is applied — but there is also no visual hierarchy. A failed run reads identically to a completed one. | Add per-status modifier classes (`.run-status--failed`, `.run-status--completed`, `.run-status--cancelled`) with both colour token and an icon or text prefix (e.g. `"[FAILED]"`). The status token set already exists: `--color-status-failed`, `--color-status-completed`, `--color-status-cancelled`. |
| Alert-banner distinguishability from surrounding chrome relies on role alone | Major | Lines 16–49 (`alert-banner`, `alert-banner--info`) | The error variant (red-tinted bg) is visually distinct. The info variant (`--color-info-bg`: 10% alpha) against `--color-surface-nav` (`#0a1d2e`) may produce very low background-level contrast. Neither variant includes an icon or prefix label, so the distinction between error and info relies on colour tone alone when text is not read. | Add a leading icon (e.g. `⚠` for error, `ℹ` for info, both `aria-hidden="true"`) rendered via the `::before` pseudo-element or a `<span aria-hidden>` in the TSX. No TSX change is needed for the CSS pseudo-approach. |

---

## Platform-Specific Notes

**Mobile**: No breakpoints exist in `header.css`. The app-header renders five
items in the left cluster (brand, session switcher, separator, version selector)
plus the user menu on the right, all within a fixed 40px bar. At viewport widths
below approximately 480px this will overflow. The 760px breakpoint used by
`chat.css` and `inspector.css` establishes a project convention — match it here.

**Keyboard**: Focus management across all three dropdowns is present and wired
in TSX. The CSS gap is that the session-switcher's focus indicator (`:focus`
background-only) will be invisible under Windows High Contrast mode where
background overrides are suppressed. Adding a 2px `outline` would survive forced
colors.

---

## Priority Recommendations

### Critical (Fix Immediately)

1. **Unstyled session-switcher controls and error alerts** (`header-session-switcher-controls`, `header-session-switcher-filter`, `header-session-switcher-show-archived`, `header-session-switcher-archive-error`, `header-session-switcher-rename-error`): Add layout and error-alert styles to `header.css`. These are rendered in production and are currently invisible or unstyled.

2. **Unstyled `RunsHistoryDrawer`** (`runs-history-drawer`, `runs-history-drawer-header`, `runs-history-drawer-body`, `runs-history-list`, `runs-history-item`): Add positioning, surface background, scroll containment, and visual chrome to `header.css`. The `role="dialog"` element is currently rendered without any layout.

### Major (Fix Before Launch)

3. **Session-switcher `:focus` → `:focus-visible` + ring**: Replace bare `:focus` with `:focus-visible` and add `outline: 2px solid var(--color-focus-ring)` to pass WCAG 2.4.7 on keyboard nav.

4. **Sign-out colour-only distinction**: Add a separator line above the Sign out menu item and/or a leading `aria-hidden` icon to satisfy WCAG 1.4.1.

5. **Run status visual differentiation**: Add `.run-status--failed/completed/cancelled` modifier classes using both `--color-status-*` tokens and a text or icon prefix.

6. **Alert-banner icon supplement**: Add `aria-hidden` leading icons to error and info variants so information-type is not conveyed by colour alone.

7. **Mobile breakpoint missing**: Add `@media (max-width: 760px)` to `header.css` to collapse or adapt the five-element left cluster on narrow viewports.

### Minor (Improvement)

8. **Define `--color-bg-hover` in tokens.css** or replace the undefined token with `var(--color-surface-hover)` — the current fallback is load-bearing and fragile.

9. **Tokenise z-index literals**: Replace `z-index: 100` with `var(--z-overlay)` (line 106) and define `--z-header-dropdown` for the `z-index: 50` user-menu layer (line 190).

10. **`app-header` height → min-height**: Change `height: 40px` to `min-height: 40px` to prevent content clipping under font scaling or long session names.

11. **Remove redundant `var(--font-mono, monospace)` fallback** (line 292) — the token is always defined.

---

## Confidence Assessment

**High confidence** on structural issues (missing CSS classes, z-index literals, `:focus` vs `:focus-visible`): these are directly observable from file comparison and grep.

**Medium confidence** on alert-banner contrast at the composited pixel level: the 12%-alpha backgrounds blended over `--color-surface-nav` need measurement against actual rendered values. The project's `colorContrast.test.ts` is the right verification tool.

**Low confidence** on run-status visual requirements beyond the current rendering: the reviewer has not seen whether run status is also displayed elsewhere in the UI with richer treatment.

## Risk Assessment

**Highest risk**: The two unstyled-classes critical issues are live in production — the drawer and the session-switcher error alerts exist in rendered HTML with no visual representation. Users who trigger archive/rename failures or open the past-runs drawer encounter a broken experience.

**Second highest**: The mobile viewport gap is a layout breakage for any narrow-screen user. It is not recoverable via any existing style.

## Information Gaps

- Actual rendered contrast of `--color-error` text on composited `--color-error-bg` + `--color-surface-nav` has not been measured; `colorContrast.test.ts` should be extended.
- It is unclear whether run status is displayed with richer treatment in any other part of the UI (e.g. `InspectorPanel`). If it is, the gap here may be a lower priority.
- `--color-bg-hover` is not in `tokens.css`. A search found no definition anywhere in the frontend source. The fallback is working but the intent of the token is not documented.

## Caveats

This review is based on static analysis of CSS and TSX source. No live rendering, no browser DevTools measurement, and no assistive-technology screen-reader testing was performed. Contrast ratios cited are estimates based on token values; composited alpha blending over layered surfaces can differ from naively computed values. The `colorContrast.test.ts` file is the authoritative verification mechanism for this codebase.
