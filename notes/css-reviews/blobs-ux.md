# Design Review: blobs.css — Blob Row & Blob Manager

**File:** `src/elspeth/web/frontend/src/components/blobs/blobs.css`
**Consumers:** `BlobRow.tsx`, `BlobManager.tsx`
**Reviewer:** UX Critic Agent
**Date:** 2026-05-23

---

### Summary

**Overall:** Needs Work
**Critical Issues:** 2
**Major Issues:** 5
**Minor Issues:** 4

The structural foundation is sound — token usage is clean, the filename truncation pattern is correctly implemented, and the empty-state copy is functional. The critical failures are both interactive-state gaps: `.blob-row-container` has no hover, focus, selected, or keyboard-navigation CSS whatsoever, and the action buttons (`.blob-action-btn`) are completely absent from the CSS file, making their size, focus indicator, and touch-target compliance unknowable without runtime inspection.

---

### Confidence Assessment

- **High confidence:** All findings below derive directly from the CSS source, the two TSX consumers, and WCAG 2.1 AA/APCA baselines. No inferred behaviour.
- **Gap:** No design tokens file was read; token values (e.g., `--size-control-compact`, `--font-size-xs`) are unresolved. Where a token is critical to a finding, this is called out explicitly.
- **Gap:** No computed theme values were measured against contrast ratios. Findings about contrast are structural (status dot is color-only with no text fallback in CSS) rather than numerically measured.

---

### Risk Assessment

- **High risk:** Missing `.blob-action-btn` rule means button sizing, focus ring, and touch-target compliance are invisible to CSS review and may fall below WCAG 2.4.7 / APCA minimums.
- **High risk:** Status dot conveys status with color alone — a WCAG 1.4.1 violation for users with color-vision deficiency. The `title` attribute in TSX partially mitigates this for pointer users but provides no benefit for touch or keyboard users without a visible text label.
- **Medium risk:** `blob-manager-loading` and `blob-manager-empty` share a rule block with identical styling; the empty-state CTA is plain text with no visual affordance to guide next action.

---

### 1. Visual Design

**Strengths:**
- Token usage is consistent throughout: `--color-text-muted`, `--color-surface-elevated`, `--color-border`, `--font-mono` etc. are all used correctly. No hard-coded color values.
- `blob-row-preview-pre` correctly pairs `font-family: var(--font-mono)` with `white-space: pre-wrap` and `word-break: break-word` — prevents overflow while preserving intent.
- `blob-manager-category-header` uses `text-transform: uppercase` + `letter-spacing: 0.5px` + `font-weight: 600` for visual hierarchy of section labels; tasteful and consistent with the rest of the design system.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Status dot is color-only: no visible text label accompanies `.blob-row-status-dot`; `title` tooltip requires pointer hover and is inaccessible on touch. | Critical | `.blob-row-status-dot` / `BlobRow.tsx:102–108` | Add a visually hidden `<span>` with `.sr-only` containing `status.label` next to the dot, or replace the dot with a labeled badge. CSS-side: ensure the label variant is hidden from layout but present for screen readers. |
| `--font-size-xs` drives both `.blob-row-size` and `.blob-manager-title` — if that token resolves below 11px, both fail WCAG 1.4.4 at 100% zoom on low-DPI displays. | Major | `.blob-row-size` (L36), `.blob-manager-title` (L104) | Verify token resolves to ≥11px; prefer ≥12px for secondary metadata. |
| `blob-row-preview-pre` has `line-height: 1.4` on preformatted code. WCAG 1.4.12 requires 1.5 for body text; code blocks are exempt only if they are genuinely non-wrapping, but `white-space: pre-wrap` here means lines reflow, so 1.5 is correct. | Minor | `.blob-row-preview-pre` (L62–75) | Set `line-height: 1.5`. |

---

### 2. Information Architecture

**Strengths:**
- Category headers ("Source files", "Output files", "Other files") provide a correct three-level mental model. Categories without blobs are suppressed (`if (categoryBlobs.length === 0) return null`), so the list stays clean.
- File count in header `Files (N)` is a compact, effective live count.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Empty state (`blob-manager-empty`) is a plain centered text node with no visual affordance. The copy "Upload a file to get started" implies action but provides no tappable link to trigger the upload. | Major | `.blob-manager-empty` (L129–133), `BlobManager.tsx:138–140` | Replace text-only empty state with a button or link styled to call out the upload action — same `fileInputRef.current?.click()` handler used by the header button. CSS: `.blob-manager-empty` should accept a `.blob-manager-empty-action` button child with appropriate visual styling. |
| No timestamp column is rendered. The focus area requested "timestamp metadata" — neither `BlobRow.tsx` nor `blobs.css` includes a created/modified time field. If `BlobMetadata` carries a timestamp, it is being suppressed at render time. | Major | `BlobRow.tsx` (all) | Confirm whether `BlobMetadata` exposes `created_at`; if so, add a `.blob-row-timestamp` element with `font-family: var(--font-mono)` and `color: var(--color-text-muted)` so it aligns with the size field. See "size / timestamp" note below. |
| `blob-manager-container` is `max-height: 280px`. With a populated list across three categories (headers + rows), this collapses into a dense scrolling pane with no scroll affordance. No shadow or gradient is applied to indicate overflowed content. | Minor | `.blob-manager-container` (L85–91), `.blob-manager-list` (L123–126) | Add a fade-gradient pseudo-element at the bottom of `.blob-manager-list` when overflow is present, or add `box-shadow: inset 0 -8px 8px -4px var(--color-surface)` to signal scrollability. |

---

### 3. Interaction Design

**Strengths:**
- `aria-label` on every action button correctly includes the filename (`aria-label={`Download ${blob.filename}`}`), making screen reader announcements unambiguous even when multiple rows are visible.
- `aria-expanded` on the preview toggle button is correctly implemented, matching the `previewOpen` state.
- The hidden file `<input>` correctly uses `aria-hidden="true"` and `tabIndex={-1}`.
- `role="alert"` on the error div in `BlobManager.tsx:125–129` gives live-region semantics for upload errors.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.blob-action-btn` is **not defined in blobs.css**. The class is applied to all four action buttons in `BlobRow.tsx` but has no corresponding rule. Without a rule, button sizing, focus ring, and padding are determined entirely by the global `.btn` default (if one exists) or browser UA styles. Touch target compliance (≥24px, ideally 44px) cannot be verified. This is the single highest-risk gap in the file. | Critical | `BlobRow.tsx:131–168`, `blobs.css` (absent) | Add `.blob-action-btn` rule to `blobs.css` specifying `min-width: 28px; min-height: 28px; padding: 2px 6px; display: inline-flex; align-items: center; justify-content: center;` as a minimum. Extend to `44px` square if the panel width permits. |
| `.blob-row-container` has no hover, focus-within, selected, or keyboard-navigation states. Users cannot tell which row is active via keyboard. No `:hover` background, no `:focus-within` outline, no selected state exists. | Critical | `.blob-row-container` (L7–13) | Add at minimum: `:hover { background-color: var(--color-surface-hover); }` and `:focus-within { outline: 2px solid var(--color-focus); outline-offset: -2px; }`. |
| The preview toggle button renders as the Unicode eye emoji (`👁 U+1F441`). Emoji rendering is font-dependent and may differ significantly across platforms — on some systems the glyph is very small, on others it clips the hit area. The aria-label is correct but the visual affordance is brittle. | Major | `BlobRow.tsx:139` | Replace the emoji with an SVG icon (consistent with the rest of the design system) or an icon font glyph with known sizing. If emoji must be used, wrap in a `<span aria-hidden="true">` and explicitly set `font-size` and `line-height` to stabilize hit-area calculation. |
| Creator badge (`.blob-row-creator`) renders emoji characters (inbox tray, robot, gear) with no text fallback visible in the layout. On high-contrast Windows mode, emoji may render as outlined glyphs with no color. The `title` attribute provides a tooltip for pointer users only. | Major | `.blob-row-creator` / `BlobRow.tsx:111–113` | Add a visually hidden text label alongside each emoji via `.sr-only` pattern, or replace with SVG icons that respond to `currentColor`. |
| Action buttons for non-ready blobs are conditionally rendered rather than disabled-and-visible, meaning a blob in `pending` or `error` status shows only a delete button. Users have no indication that other actions exist but are unavailable. | Minor | `BlobRow.tsx:130–159` | Render Download and Use-as-Input buttons with `disabled` attribute and `aria-disabled="true"` when `blob.status !== "ready"`, rather than omitting them. Style `.blob-action-btn:disabled` with reduced opacity in CSS. |

---

### 4. Accessibility

**Quick Check:**

- [ ] 1.4.3 Contrast: **Unverifiable** — token values not resolved; structural gap (status dot color-only) is a fail on 1.4.1 for all users regardless of contrast ratio
- [ ] 1.4.1 Use of Color: **Fail** — status dot uses color as the sole indicator of meaning (ready/pending/error) with no co-located text
- [ ] 2.1.1 Keyboard Nav: **Partial** — action buttons have `aria-label` and are focusable (native `<button>`), but no visible row focus state exists in CSS
- [ ] 2.4.7 Focus Visible: **Unknown** — `.blob-action-btn` has no CSS rule, focus ring styling is unverified
- [ ] 1.1.1 Alt Text: **N/A for images**, but emoji used as icons (creator badge, action buttons) lack text equivalents in CSS; TSX `title` provides partial mitigation for pointer users only
- [ ] 3.3.2 Labels: **Pass** — upload button has `aria-label="Upload file"`; action buttons all have explicit `aria-label` with filename

**Issues:**

| Issue | Severity | WCAG | Location | Fix |
|-------|----------|------|----------|-----|
| Status dot conveys ready/pending/error by color alone | Critical | 1.4.1 | `.blob-row-status-dot`, `BlobRow.tsx:102–108` | Add visually hidden text label or visible badge text alongside the dot |
| No visible focus indicator for `.blob-action-btn` | Critical | 2.4.7 | `blobs.css` (missing rule) | Define `.blob-action-btn:focus-visible { outline: 2px solid var(--color-focus); outline-offset: 2px; }` in the new rule |
| Creator badge emoji have no accessible text in layout | Major | 1.1.1 | `BlobRow.tsx:111–113` | Add `.sr-only` span with label text alongside each emoji |
| `.blob-manager-empty` CTA is plain text, not an interactive element | Major | 2.1.1 | `BlobManager.tsx:139` | Make the CTA a focusable `<button>` so keyboard users can trigger upload without tabbing back to the header |

---

### 5. Size / Timestamp Metadata (Focus Area 3)

Neither `blobs.css` nor `BlobRow.tsx` includes a timestamp field. The CSS has `.blob-row-size` correctly styled with `font-size: var(--font-size-xs)` and `color: var(--color-text-muted)`, but there is no `.blob-row-timestamp` rule and no corresponding TSX element. If `BlobMetadata` exposes `created_at`, this is a gap — tabular alignment of size and timestamp using `font-family: var(--font-mono)` would allow users to visually sort by recency. The monospace treatment on `.blob-row-size` is not yet present; add it if numeric alignment matters.

---

### 6. File-Type Recognisability (Focus Area 2)

There is no icon next to filenames. The creator badge (emoji) indicates provenance, not file type. `.blob-row-filename` renders the raw filename string with no MIME-type icon or extension label alongside it. PREVIEWABLE_MIME_TYPES is used to gate the preview button, but file type is otherwise visually invisible. For a file manager surface, file-type iconography (even a simple extension badge) is standard and reduces cognitive load. This is recorded as a minor finding.

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| No file-type icon or extension badge alongside filename | Minor | `.blob-row-filename`, `BlobRow.tsx:115–121` | Add a `.blob-row-file-type` span before filename showing the extension (`.csv`, `.json`, etc.) derived from `blob.filename` or `blob.mime_type`. CSS: monospace, muted, fixed-width to avoid layout jitter. |

---

### 7. Token Discipline and Dead Rules

**Token discipline: Good.** All color, spacing, font, and radius values use design tokens. No raw hex, no `px` values except `6px` padding in `.blob-row-container` (L11) and `.blob-manager-header` (L97) — both should use `var(--space-xs)` or similar for consistency if that token resolves to 6px.

**Dead rules: None confirmed.** All 16 rules in the file map to elements rendered in BlobRow or BlobManager. No orphaned selectors detected.

**Missing rules (not dead, but absent):**
- `.blob-action-btn` — used in TSX, absent in CSS (Critical gap, noted above)
- `.blob-row` — applied as a class alongside `.blob-row-container` in `BlobRow.tsx:96`, but has no CSS rule. If it exists in a global stylesheet, that's fine; if not, it's a phantom class.

---

### Priority Recommendations

**Critical (Fix Immediately):**
1. Add `.blob-action-btn` to `blobs.css` with explicit `min-height`, `min-width`, padding, and `:focus-visible` outline. Without this rule, touch-target and focus-ring compliance is unverifiable.
2. Add hover and focus-within states to `.blob-row-container` so keyboard users can identify the active row.
3. Replace color-only status dot with a paired text label (visually hidden for pointer users acceptable, but the text must exist in the DOM).

**Major (Fix Before Launch):**
1. Empty state: convert the "Upload a file to get started" text into an interactive button that triggers `fileInputRef.current?.click()`, and add CSS for `.blob-manager-empty-action`.
2. Creator badge emoji: add `.sr-only` text or replace with SVG icons that inherit `currentColor` so they work in forced-color / high-contrast mode.
3. Preview toggle button: replace `👁` emoji with a sized SVG or icon-font glyph; stabilize hit area.
4. Conditionally hidden action buttons: show Download and Use-as-Input as `disabled` when `blob.status !== "ready"` rather than omitting them.

**Minor (Improvement):**
1. Replace the two `6px` literal padding values with the appropriate spacing token.
2. Add a scroll fade-gradient to `.blob-manager-list` to signal overflowed content.
3. Add `.blob-row-timestamp` rule and corresponding TSX element if `BlobMetadata` exposes a creation time.
4. Add a `.blob-row-file-type` extension badge to make file type scannable without reading the full filename.

---

### Information Gaps

- Design token values for `--size-control-compact`, `--font-size-xs`, `--font-size-2xs`, and `--color-focus` were not read. Minimum font-size and focus-ring color compliance cannot be confirmed numerically.
- Whether `.blob-row` (the bare class applied alongside `.blob-row-container`) has a rule elsewhere in the system is unknown.
- `BlobMetadata` type definition was not read; timestamp field presence is unconfirmed.

---

### Caveats

- Contrast ratios are not measured numerically; contrast findings are structural (color-only use of status, unresolved token values) rather than tool-measured. Run a WCAG contrast checker against computed theme values before closing the color-only status finding.
- If `.blob-action-btn` is defined in a shared stylesheet loaded before blobs.css, the Critical gap narrows to a documentation/locality issue rather than a missing rule. Verify before implementing the fix.
