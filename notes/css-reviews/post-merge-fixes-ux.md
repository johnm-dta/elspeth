# UX/A11y Review — Post-merge CSS fixes (RC5.2)

**Scope:** Three change sites in commits c426b4225..HEAD on RC5.2.
**Files reviewed:** `composer/composer.css` (new, 88 lines), `chat/chat.css` (line-height + max-height changes), `composer/SaveForReviewDialog.tsx` (consumer, lines 100-202), `styles/tokens.css`, `styles/base.css`.
**Date:** 2026-05-23

---

## Status: NEEDS-FIX

**Critical:** 0
**Major:** 2
**Minor:** 3

---

## Findings

### MAJOR — 1: Scrollable `<pre>` is keyboard-unreachable

**File:** `chat.css:976-990` (`.tool-call-details pre`)
**Change:** `max-height: 320px; overflow-y: auto` added.

A scrollable region whose content overflows is only reachable by keyboard if the region itself has a `tabindex` value that puts it in the tab order. `<pre>` is not natively focusable, so keyboard users cannot scroll past the 320px fold to read the clipped content of long `set_pipeline` / `propose-chain` responses before accepting or rejecting the proposal.

WCAG 2.1.1 (Keyboard, Level A).

**Fix — two-part, CSS + TSX co-land required:**

In `chat.css`, add to `.tool-call-details pre`:
```css
.tool-call-details pre {
  /* existing rules ... */
  max-height: 320px;
  overflow-y: auto;
  /* add: */
  outline: none;          /* suppress outline on programmatic focus; scrollable region, not an action */
}
.tool-call-details pre:focus-visible {
  outline: 1px solid var(--color-border-strong);
  outline-offset: -1px;
}
```

In the TSX that renders `.tool-call-details pre` (not in scope for this review, but must co-land): add `tabIndex={0}` to the `<pre>` element and a screen-reader-only label, e.g. `aria-label="Tool call details"`, so the scrollable region is announced meaningfully when focused.

---

### MAJOR — 2: `aria-modal="true"` on an inline panel has no matching modal behaviour

**File:** `SaveForReviewDialog.tsx:113`

The element declares `role="dialog" aria-modal="true"`, which tells screen-reader virtual cursors to restrict navigation to the dialog's subtree (hiding the rest of the DOM). In practice the element is rendered inline below the chat textarea — there is no backdrop, no focus trap, and no `z-index` elevation. Mouse and keyboard users can freely interact with the composer behind it; screen reader users experience a navigation-restricted region that misrepresents the actual interaction model.

This is not a regression from this CSS change (the attribute predates this merge), but the composer.css addition now provides layout context that makes the mismatch observable: this is styled as a content block, not a modal.

**Fix (minimal, no refactor):** Remove `aria-modal="true"` and `role="dialog"` from the element; it already has `aria-labelledby` on the heading, which is sufficient for a non-modal expansion panel. If the intent is a true modal in a future phase, the CSS will also need a backdrop, focus trap, and `z-index: var(--z-dialog)`.

Do not add a backdrop or focus trap in this patch; the change would be scope-creep and could break the composer's interaction model.

---

### MINOR — 3: URL input touch target is undersized

**File:** `composer.css:58` (`.save-for-review-url-row input[type="text"]`)

Padding is `var(--space-xs) var(--space-sm)` = 4px top/bottom. At `--font-size-sm` (13px) with a normal line-height the rendered input height is approximately 21px — well below the WCAG 2.5.8 24px floor and the project's own `--size-control-compact` (36px) used for compact controls.

This is a text input, not a button — WCAG 2.5.8 targets interactive controls, and text inputs have softer guidance — but operators will routinely click into this input on touch devices to trigger the `onFocus → select()` copy gesture.

**Fix:** Change `var(--space-xs)` to `var(--space-sm)` on the vertical axis:
```css
.save-for-review-url-row input[type="text"] {
  padding: var(--space-sm) var(--space-sm);   /* was: var(--space-xs) var(--space-sm) */
  ...
}
```
This moves rendered height to ~29px. Still compact but materially more tappable.

---

### MINOR — 4: `.save-for-review-url-row a` ("Open in new tab") has no touch sizing

**File:** `composer.css:67-70`

The anchor is a bare text node with no `min-height` or `display` override. Hit target is the text's line-height only (~18px at `--font-size-sm`). The Copy button adjacent to it covers the primary affordance, so this is low risk, but parity with the project's touch target standard is worth a one-liner.

**Fix:**
```css
.save-for-review-url-row a {
  flex: 0 0 auto;
  color: var(--color-link);
  /* add: */
  display: inline-flex;
  align-items: center;
  min-height: var(--size-control-compact);
}
```

---

### MINOR — 5: `line-height: 1.35` is a magic number between two tokens

**File:** `chat.css:24` (`.bubble`)

`tokens.css` defines `--line-height-tight: 1.3` and `--line-height-normal: 1.5`. The new value 1.35 sits between them with no named token. The change itself is visually defensible — 1.42 → 1.35 tightens body copy without cramping (body text typically reads well down to 1.3). No existing rule in chat.css depends on the old 1.42 value numerically (no paired `min-height`, no explicit `calc()` using 1.42). The `3px` block spacing on `.bubble .markdown-body p/ul/ol` is independent of line-height and is unaffected.

Context: chat.css already hardcodes 1.4, 1.45, and 1.35 in several places (lines 359, 1103, 1244, 1253, 1258), so this is consistent with an existing pattern of intermediate values not covered by the three-level token set.

**Fix (optional, not blocking):** Either accept the magic number as consistent with the established pattern, or add `--line-height-snug: 1.35` to tokens.css and use it here and in the other five occurrences. The latter option improves future maintainability. Not a merge blocker.

---

## Focus Visibility — Confirmed Not an Issue

**Share URL input:** `base.css:83-86` provides a universal `:focus-visible` rule (`outline: 2px solid var(--color-focus-ring); outline-offset: 2px`) that applies to all elements including inputs. The `.save-for-review-url-row input` will receive a correct focus ring from the global rule. No per-element override needed. The per-component overrides in `chat.css` (e.g. `.chat-input-cancel-btn:focus-visible`) exist to add border-color changes on hover/focus, not to supply a focus ring the global rule omits.

---

## Token Discipline

All colour, spacing, radius, and font values in `composer.css` use `var(--…)` tokens. No magic colour literals. The one dimensional literal is the 320px `max-height` in `chat.css:988`, which is explicitly justified by the comment on lines 984-987 and is appropriate as a layout constraint rather than a semantic colour or spacing value.

---

## Error Block Contrast

`.save-for-review-error` uses `--color-error` text on `--color-error-bg`. The effective dark-mode background (12% red alpha over `--color-surface-elevated` #1a3d47) yields approximately #2c4047; `--color-error` is #e85653 — estimated ~3.0:1 contrast. This is the same token pair used by `.chat-panel-error` and `.chat-input-upload-alert`. If `colorContrast.test.ts` passes for those, it passes here. Not a new risk introduced by this change.

---

## `max-height: 320px` on `.tool-call-details pre` — Sizing Rationale

320px is approximately 16 lines at a 20px computed line-height, or ~20 lines at 16px. For typical `set_pipeline` YAML responses this is enough to show the structure and signature while keeping the Accept/Reject action row on-screen without scrolling the chat panel. The value is defensible. The keyboard accessibility gap (finding 1) is the only concern.
