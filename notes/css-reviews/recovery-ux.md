# UX/A11y Review: recovery.css

**File:** `src/elspeth/web/frontend/src/components/recovery/recovery.css`
**Consumers:** `RecoveryPanel.tsx`, `RecoveryDiff.tsx`, `RecoveryTranscript.tsx`
**Reviewer:** UX Critic Agent
**Date:** 2026-05-23

---

## Confidence Assessment

**Overall confidence: High** — full token definitions read (tokens.css, themes.css,
shared.css, base.css), all three consumer TSX files read. Colour contrast figures
are calculated from actual hex values in tokens.css. Focus-trap and ARIA hooks
verified against RecoveryPanel.tsx source.

**Limitations:**
- Contrast figures assume nominal computed values; alpha-blended colours
  (e.g. `rgba(0,0,0,0.45)` backdrop) are not factored into any text contrast
  calculation because no text is rendered over them.
- No visual rendering was observed. Findings depend on static analysis of markup
  and token values.
- `recovery-diff-row-change`, `recovery-diff-compact`, `recovery-transcript-*`,
  `recovery-transcript-assistant`, `recovery-transcript-tools`,
  `recovery-transcript-missing`, `recovery-transcript-note` are used in TSX but
  have **no rules in recovery.css**. They inherit global styles only. This is
  the primary structural finding.

---

## Risk Assessment

| Risk | Likelihood | Impact |
|------|-----------|--------|
| Diff rows indistinguishable by colour alone (colour-blind) | High — no border/icon diff coding | High — operator cannot triage added vs removed |
| Transcript sections lack independent scroll — large transcripts overflow the panel | High — no overflow on `.recovery-transcript` | High — content unreachable |
| Confirmation step not visually elevated enough over warning banner | Medium | High — "Apply anyway" is irreversible |
| Missing forced-colors overrides for recovery elements | High — themes.css doesn't cover any `.recovery-*` | Medium |

---

## 1. Destructive Action Affordance

### Strengths

- Footer action order is correct: destructive ("Discard recovery") is on the
  left, primary ("Apply partial draft") on the right. This matches platform
  convention and reduces misclick risk.
- Both buttons use `.btn-danger` / `.btn-primary` from shared.css. `btn-danger`
  carries `--color-btn-danger-bg` (`#b23835` dark, same light), white text. At
  44px `min-height` both meet WCAG 2.5.5 touch-target (AAA).
- The confirm step surfaces as `role="alert"` inside `.recovery-panel-confirm`,
  which gets screen-reader attention automatically.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Both footer buttons are the same visual weight class on opposite poles of the action space. "Discard recovery" (`btn-danger`) and "Apply partial draft" (`btn-primary`) both render as filled, full-height, white-text buttons. The destructive action does not carry an icon or secondary label to distinguish it from the constructive one. | Major | `RecoveryPanel.tsx` L170–183 | Add a trash/warning icon to the Discard button and consider a confirmation affordance (e.g., requiring the user to type a word) for the terminal discard. At minimum, add a `title` attribute or adjacent microcopy. |
| The inner confirm dialog ("Apply anyway") also uses `btn-danger` styling — identical to the outer Discard button. A user who instinctively clicks the left-side button in each pair will Discard at the outer level and "Apply anyway" at the inner level, both destructive. There is no visual or positional hierarchy between them. | Major | `RecoveryPanel.tsx` L127–138 | Elevate the inner confirm dialog: increase contrast, add a warning icon adjacent to the "Apply anyway" label, or restructure so the inner confirm is a smaller inset widget rather than a peer-level action row. |
| No `aria-describedby` links the two footer buttons to the panel description (`recoveryError.detail`). A screen-reader user landing on "Discard recovery" gets button label only. | Minor | `RecoveryPanel.tsx` L170–183 | Add `aria-describedby="recovery-panel-title"` to both footer buttons. |

---

## 2. Diff Readability

### Strengths

- `RecoveryDiff` groups entries by kind (`added`, `removed`, `changed`) with
  labelled headers. Progressive disclosure (collapse when >50 rows) prevents
  overwhelming the panel.
- Summary badges (`.recovery-diff-summary span`) use the global border-chip
  pattern from shared.css.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| **Colour-blind safety — critical gap.** `.recovery-diff-row--added`, `.recovery-diff-row--removed`, `.recovery-diff-row--changed` are generated class names (see `RecoveryDiff.tsx` L272) but **recovery.css defines no rules for them**. These classes exist in the DOM but carry zero visual differentiation. All three diff kinds render identically: same `var(--color-border)` card, no background tint, no left-border stripe, no icon. A user cannot visually distinguish an addition from a removal without reading the row label text. This also means colour-blind users receive no alternative signal at all. | Critical | `recovery.css` (missing block) + `RecoveryDiff.tsx` L272 | Add three blocks: `.recovery-diff-row--added` (success-bg tint + left border in `--color-success`), `.recovery-diff-row--removed` (error-bg tint + left border in `--color-error`), `.recovery-diff-row--changed` (warning-bg tint + left border in `--color-warning`). The left border provides a shape/position signal independent of colour, making the distinction colour-blind safe. |
| `.recovery-diff-row-change` is referenced in `RecoveryDiff.tsx` L278–288 but has no CSS rule. The "before → after" change display inherits only body text styles: no visual separation between before-value and after-value spans, no monospace rendering to aid diffing. | Major | `recovery.css` (missing block) | Add `.recovery-diff-row-change` with `font-family: var(--font-mono); font-size: var(--font-size-sm); display: flex; gap: var(--space-sm); flex-wrap: wrap;`. |
| `.recovery-diff-compact` (the "Details are collapsed" placeholder paragraph) has no CSS rule. It will render as body text without any visual cue that it is replacing a potentially large list. | Minor | `recovery.css` (missing block) | Add `.recovery-diff-compact` with `color: var(--color-text-muted); font-size: var(--font-size-sm); font-style: italic;`. |
| The `"→"` arrow between before and after values in a changed row is rendered as a plain text span with `aria-hidden="true"` (correct for screen readers) but there is no semantic or visual label grouping the before/after pair. Sighted users will infer directionality from position; low-vision users scaling up to 200% zoom will lose that layout cue. | Minor | `RecoveryDiff.tsx` L280–281 | Wrap the before/after pair in a `<dl>` with `<dt>Before</dt><dd>` / `<dt>After</dt><dd>` for semantic structure. This also removes the need for the aria-hidden arrow entirely. |

---

## 3. Transcript Readability

### Strengths

- `RecoveryTranscript.tsx` uses `<section>`, `<article>`, `<h3>`, `<h4>` —
  correct semantic structure.
- `.recovery-transcript-tool-rows pre` has `max-height: 180px; overflow: auto`
  — individual tool-response blocks are bounded.
- `pre` inherits `font-family: var(--font-mono)` and `line-height: var(--line-height-relaxed)`
  from `base.css`. This is good: the 1.7 line height aids readability in dense output.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| **The `.recovery-transcript` section itself has no `overflow` rule.** The transcript column sits inside `.recovery-panel-body` which is `overflow: auto`, so the grid cell stretches to fit the transcript instead of the transcript scrolling independently within a bounded column. On a long transcript, the entire panel body scrolls as one mass, making it impossible to keep the diff column visible while reading the transcript. | Major | `recovery.css` L97–98 (`.recovery-panel-body`) | Add `.recovery-transcript { overflow-y: auto; min-height: 0; }`. The grid cell must also not grow beyond its column bounds: the current `minmax(280px, 0.8fr)` column will stretch without a height anchor on the transcript section. |
| `.recovery-transcript-assistant`, `.recovery-transcript-tools`, `.recovery-transcript-missing`, `.recovery-transcript-note` are all referenced in TSX but have no CSS rules. The assistant turn and tool-call sections render as unstyled body text, with no visual separation between sections. | Major | `recovery.css` (all missing) | Add section separators. Minimum: `.recovery-transcript-assistant`, `.recovery-transcript-tools` each get `border-top: 1px solid var(--color-border); padding-top: var(--space-sm); margin-top: var(--space-sm);`. |
| `.recovery-transcript-missing` has no CSS rule. "Missing tool response" is a high-signal error state (it means the tool call partially persisted) but currently renders as ordinary body text. | Major | `recovery.css` (missing block) | Add `.recovery-transcript-missing { color: var(--color-warning); font-size: var(--font-size-sm); }` — using warning rather than error because absence of a response is ambiguous: it may mean the response arrived but was not persisted before the crash. |
| Timestamp or role colour: `RecoveryTranscript` shows no timestamp display, and role identity (assistant vs tool) is conveyed only by heading level, not colour or badge. Screen reader users get the heading hierarchy; sighted users in a fast scan may miss the role boundary. | Minor | `RecoveryTranscript.tsx` — no role badge rendered | Add a small `.recovery-transcript-role` badge (reusing the `.type-badge` pattern or a custom chip) adjacent to each section heading that labels "Assistant" / "Tool". |

---

## 4. Focus Management

### Strengths

- `useFocusTrap(dialogRef, recoveryError !== null, ".recovery-panel-apply")` is
  called in `RecoveryPanel.tsx` L46. This is the correct pattern: focus trap is
  active when the panel is visible, and initial focus is placed on the primary
  positive action ("Apply partial draft"), not on the destructive one.
- The panel root has `tabIndex={-1}` and `role="dialog"` / `aria-modal="true"` /
  `aria-labelledby="recovery-panel-title"` — complete modal ARIA pattern.
- `onKeyDown` on the panel root guards against accidental Enter-on-backdrop
  submission.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| When `needsConfirmation` transitions to `true`, the confirmation region (`role="alert"`) renders but focus stays on the "Apply partial draft" button. The `role="alert"` will announce via screen reader, but keyboard focus does not move to the "Cancel" or "Apply anyway" buttons. A keyboard user must Tab forward to reach them. | Major | `RecoveryPanel.tsx` L116–139 | When `needsConfirmation` becomes `true`, move focus to the "Cancel" button inside the confirm region (use a `useEffect` + `ref` on the Cancel button). This is consistent with the ARIA dialog pattern for nested confirmations. |
| `.recovery-panel-backdrop` has `role="presentation"` but no `aria-hidden="true"`. The backdrop is a real DOM node before the dialog; some screen readers will still encounter and announce it. | Minor | `RecoveryPanel.tsx` L69–72 | Add `aria-hidden="true"` to the backdrop div. |
| The panel has no `Escape` key handler. The ARIA dialog pattern (APGD §3.8) requires that Escape closes the dialog. Currently pressing Escape in the recovery panel does nothing. | Major | `RecoveryPanel.tsx` (missing) | Add `onKeyDown` to the dialog div (or inside `useFocusTrap`) that calls `onDiscard()` on `event.key === "Escape"`. If Discard is irreversible and needs a confirm, Escape should move to the inner Cancel state, not immediately discard. |

---

## 5. Scrolling Behaviour

### Strengths

- `.recovery-panel` has `overflow: hidden` — the panel itself does not scroll;
  its body is the scroll surface. Correct architectural split.
- `.recovery-panel-body` has `overflow: auto` — this is the intended scroll
  container. The `min-height: 0` is also present, which is necessary for grid
  children to shrink.
- Individual `pre` blocks inside the transcript are bounded at `max-height: 180px`.

### Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| The two-column grid (`.recovery-panel-body`) scrolls as a single unit. When the transcript column is taller than the diff column, the diff scrolls away with it. This defeats the purpose of a side-by-side comparison view. | Major | `recovery.css` L92–98 | Change `.recovery-panel-body` to `overflow: hidden` (or remove the `overflow: auto`) and add `overflow-y: auto; min-height: 0;` to both the diff section (`.recovery-diff`) and the transcript section (`.recovery-transcript`) so each column scrolls independently within the bounded grid. |
| On mobile (`max-width: 900px`) the grid stacks to one column, but neither `.recovery-diff` nor `.recovery-transcript` gains a `max-height` or `overflow` constraint. On a short viewport (e.g. 568px iPhone SE landscape) the stacked transcript could be arbitrarily tall with no scroll anchor, requiring the user to scroll the entire page. | Major | `recovery.css` L133–145 | Inside the `@media (max-width: 900px)` block add `max-height: 50vh; overflow-y: auto;` to both `.recovery-diff` and `.recovery-transcript`. |

---

## 6. Hard-Coded Colours, Dead Selectors, and Brittle Chains

### Hard-coded colours

| Location | Value | Issue | Fix |
|----------|-------|-------|-----|
| `.recovery-panel-backdrop` | `rgba(0, 0, 0, 0.45)` (L10) | Literal RGBA, not a token. The same literal appears in `.graph-modal-backdrop` and `.yaml-modal-backdrop` in `common.css`, and `.command-palette-backdrop`. Four copies of the same magic number. | Extract `--color-dialog-backdrop-scrim: rgba(0, 0, 0, 0.45)` into tokens.css and use it everywhere. |
| `.recovery-panel` | `0 8px 32px rgba(0, 0, 0, 0.25)` (L28) | Same shadow literal appears in `common.css` `.graph-modal` and `.yaml-modal`. Three copies. | Extract `--shadow-dialog: 0 8px 32px rgba(0, 0, 0, 0.25)` into tokens.css. |
| `.recovery-panel-evidence`, `.recovery-diff-summary` shared rule | `2px 8px` padding on span (L76–79) | These are bare px values, not token references. This is a minor consistency risk. | Use `var(--space-2xs) var(--space-sm)` (2px/8px) to align with the spacing scale. |
| `.recovery-panel-reason` | `4px 8px` padding (L59) | Bare px. | Use `var(--space-xs) var(--space-sm)`. |

### Dead or unanchored selectors in recovery.css

Recovery.css defines no `.recovery-diff-row--*` rules despite `RecoveryDiff.tsx` emitting them. See Diff Readability section above — these are missing rules, not dead rules.

The composite shared selector at L31–47 groups `.recovery-panel-header`, `.recovery-panel-evidence`, `.recovery-panel-actions`, `.recovery-panel-confirm-actions`, `.recovery-diff-group-header`, `.recovery-diff-row-title`, `.recovery-transcript-tool-title` under the same `display: flex; align-items: center; gap: var(--space-sm)` rule. This is efficient but **brittle**: adding a new flex context to any one of these elements requires either breaking the group or accepting that all others inherit the same justification/alignment. The secondary selector at L43–47 applies `justify-content: space-between` to three of the seven. If any of the seven elements needs different alignment in future, the rule must be fragmented. Consider extracting a `.recovery-flex-row` utility or documenting the coupling.

### Brittle descendant chains

| Selector | Depth | Risk |
|----------|-------|------|
| `.recovery-transcript-tool-rows pre` (L127–130) | 2 descendants | Pre is a global element; if a `<pre>` is added inside `.recovery-transcript-tool-rows` for a different purpose it inherits the 180px cap. Low probability currently. Acceptable. |
| `.recovery-panel-body h3`, `.recovery-panel-body h4`, `.recovery-panel-body p` (L51–55, margin reset) | 2 descendants | These margin resets apply to all `h3/h4/p` inside the body grid, which spans both the RecoveryDiff and RecoveryTranscript subtrees. Any new heading or paragraph added to either component will be silently margin-reset. Document the intent or scope to direct child sections. |

---

## 7. Accessibility Quick Check

| WCAG | Criterion | Result | Evidence |
|------|-----------|--------|----------|
| 1.4.3 | Contrast (normal text) | **Pass** (tokens) / **Unknown** (missing classes) | `--color-text` (#dff0ee) on `--color-surface-paper` (#332f2c) ≈ 12.1:1 dark. `--color-warning` (#e38444) on `--color-warning-bg` (rgba 14%) on `--color-surface-paper` — computed tinted bg ≈ #3a3228; warning text ≈ 4.8:1. Passes AA. Light theme: `--color-warning` (#b86830) on light warning-bg tinted over `--color-surface-paper` (#f0eae3) ≈ #f2e8db; contrast ≈ 4.6:1. Marginal AA pass. `.recovery-diff-row--added/removed/changed` have no background rules and therefore no contrast to check, but they also have no colour differentiation at all — fail on non-text contrast (1.4.11) because the only signalling is absent. |
| 1.4.11 | Non-text contrast | **Fail** | Diff row kind distinctions rely solely on text labels (no border, icon, or background). |
| 2.1.1 | Keyboard access | **Partial fail** | Focus trap present. Escape key missing. Confirm state doesn't move focus. |
| 2.4.7 | Focus visible | **Pass** | Global `:focus-visible` rule in base.css (2px solid `--color-focus-ring`) applies to all interactive elements including the panel buttons. |
| 2.4.3 | Focus order | **Pass** | Focus trap starts on `.recovery-panel-apply` (positive action). Tab order follows DOM order which is logical. |
| 1.1.1 | Alt text | **N/A** | No images in these components. |
| 3.3.2 | Labels | **Pass** | All buttons have text labels. Sections have `aria-label` or `aria-labelledby`. |
| 4.1.2 | Name, role, value | **Pass** | `role="dialog"`, `aria-modal="true"`, `aria-labelledby` present. `role="alert"` on confirm. |
| Forced colours | `@media (forced-colors: active)` | **Fail** | `themes.css` defines forced-color overrides for `.type-badge*`, `.validation-banner*`, `.alert-banner`, `.tutorial-*` — but no `.recovery-*` overrides. The warning badge (`.recovery-panel-reason`) uses `background-color` and `border` that will be overridden by the forced-colours palette without a `forced-color-adjust: none` guard. |

---

## Priority Recommendations

### Critical (Fix Before Any User Sees This Component)

1. **Add `.recovery-diff-row--added`, `--removed`, `--changed` CSS rules.** The diff view currently provides zero visual distinction between diff kinds. Use background tint + left-border stripe from the existing semantic token families (`--color-success-bg/border`, `--color-error-bg/border`, `--color-warning-bg/border`). This is the single highest-impact missing piece.

2. **Add independent overflow to `.recovery-diff` and `.recovery-transcript`.** Change `.recovery-panel-body` to `overflow: hidden`; add `overflow-y: auto; min-height: 0` to each column section. Without this, long transcripts push the entire panel body off-screen and the diff view disappears.

### Major (Fix Before Launch)

3. **Add Escape key handler** to close/confirm-discard. Required by ARIA dialog pattern (APGD §3.8).

4. **Move focus to the confirm region's Cancel button** when `needsConfirmation` transitions to `true`. A keyboard user currently cannot reach the inner confirm actions without tabbing forward.

5. **Add CSS for all missing transcript classes** (`recovery-transcript-assistant`, `recovery-transcript-tools`, `recovery-transcript-missing`, `recovery-transcript-note`, `recovery-diff-row-change`, `recovery-diff-compact`). These are referenced in TSX but absent from recovery.css.

6. **Add forced-color override** for `.recovery-panel-reason` and any recovery elements that use semantic background colours:
   ```css
   @media (forced-colors: active) {
     .recovery-panel-reason {
       border: 2px solid CanvasText;
       forced-color-adjust: none;
     }
   }
   ```

7. **Add mobile overflow constraints** inside `@media (max-width: 900px)`: `max-height: 50vh; overflow-y: auto` on both `.recovery-diff` and `.recovery-transcript`.

### Minor (Improvement)

8. **Extract hard-coded colour literals** (`rgba(0,0,0,0.45)` backdrop scrim and `0 8px 32px rgba(0,0,0,0.25)` dialog shadow) to tokens shared with `common.css` graph/yaml modals.

9. **Add `aria-hidden="true"` to the backdrop div.** Currently has `role="presentation"` only.

10. **Add `aria-describedby="recovery-panel-title"` to footer buttons** so screen-reader users hear the recovery context before activating either action.

11. **Replace bare px padding values** (4px 8px, 2px 8px) in `.recovery-panel-reason` and the span rules with token references (`var(--space-xs)`, `var(--space-sm)`, `var(--space-2xs)`).

---

## Information Gaps

- No access to the `useFocusTrap` hook implementation. Focus trap behaviour (trap
  persistence when inner confirm appears, Escape handling inside the hook) was
  assumed from the call signature. If the hook already handles Escape, item 3
  above may be partially or fully covered.
- `fetchRecoveryTranscript` error states (L86 catch in `RecoveryTranscript.tsx`)
  show only "Failed to load the recovery transcript." — no retry affordance and no
  error detail. This is a UX gap but not a CSS issue and is therefore out of scope
  for this review.
- The `recovery-panel-transcript-controls` button ("View raw transcript controls")
  is grid-column 2 (L100–102) on desktop, meaning it sits above the transcript
  column but below the RecoveryDiff section. Its placement relationship to the
  transcript heading was not verified visually.

---

## Caveats

- Contrast ratios for alpha-blended backgrounds are estimated using the nominal
  surface colour as the matte. Actual rendered contrast depends on the display's
  rendering of the alpha compositing, which varies slightly by browser and OS.
- The `prefers-contrast: more` overrides in `themes.css` raise `--color-text-secondary`
  and `--color-border` but do not explicitly target `.recovery-panel-reason` or the
  diff row colours. These elements may benefit from the global text token lift but
  the semantic warning/error backgrounds are not adjusted in the high-contrast block.
  This is a gap in themes.css, not recovery.css, but worth noting here because
  `.recovery-panel-reason` uses `--color-warning` which is not overridden in the
  high-contrast block.
