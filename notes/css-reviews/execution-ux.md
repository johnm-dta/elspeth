# Design Review: execution.css — Execution Progress Panel

**Reviewed:** 2026-05-23
**File:** `src/elspeth/web/frontend/src/components/execution/execution.css` (124 lines)
**Primary consumer:** `ProgressView.tsx`

---

### Summary

**Overall:** Needs Work
**Critical Issues:** 3
**Major Issues:** 5
**Minor Issues:** 4

The file is lean and well-scoped. The token discipline is sound. The critical
failures are all accessibility gaps that will cause real harm: status states
are colour-only, the cancel button cannot meet touch-target minimums with the
current padding, and the error list lacks semantic structure and copy
affordance. Several major issues compound — the WS banner has only one visual
state (warning) for semantically distinct conditions, counter values have no
numeric alignment lock, and the routing summary has no accessible labels.

---

### Visual Design

**Strengths:**

- Token discipline is consistent throughout. Every measurement, colour, radius,
  and spacing value references a design token. No raw magic numbers except the
  hard-coded `6px 10px` on `.progress-ws-banner` and `10px` on
  `.progress-cancel-btn` (see Minor issues).
- `progress-counter-value` at `--font-size-xl / font-weight: 700` gives the
  numbers genuine visual weight appropriate to a live dashboard.
- Separating `progress-bar-outer` (structure, this file) from the animation
  classes (animations.css) is correct decomposition.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Status state is text-only with no icon or shape cue | Critical | `.progress-status-label` | Add a leading state icon (SVG or CSS-generated) per status. Text transform + colour alone fail colour-blind users and users in high-ambient-light. |
| Failed-tokens counter colour applied only via inline `style` prop | Major | `ProgressView.tsx` lines 138-143 | Move to a modifier class (e.g. `.progress-counter-value--error`) so it can be audited and overridden in themes without hunting JSX. |
| WS banner uses `--color-warning-*` for all states | Major | `.progress-ws-banner` | Add `.progress-ws-banner--connected` and `.progress-ws-banner--reconnecting` variants with distinct colour tokens (success / warning respectively). Currently the banner looks identical whether you're disconnected and giving up or actively reconnecting. |
| Raw pixel values leak through token system | Minor | `.progress-ws-banner` (`6px 10px`) and `.progress-cancel-btn` (`10px`) | Replace with token equivalents (`var(--space-xs) var(--space-sm)` or similar). |

---

### Information Architecture

**Strengths:**

- The vertical ordering (WS banner → status header → progress bar → counters →
  routing summary → result/error messages) follows a sensible scan path:
  connection health at top, current state, quantitative detail, and then
  outcome below.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Status label has no per-status visual identity | Critical | `.progress-status-label` | "pending", "running", "cancelling", "completed", "failed" are all uppercase text in the same colour. Add at minimum a shape/icon prefix so the state is distinguishable without colour. A `data-status` attribute on the element + CSS `content` pseudo-element is clean. |
| Routing summary items have no semantic labels — screen readers announce raw numbers | Major | `.progress-routing-summary` | Wrap each `<span>` in a `<dl>/<dt>/<dd>` pair or add `aria-label` to each span. "247 routed success" is parseable visually; "247" alone in a screen reader is not. |
| Errors title provides count but no severity context | Minor | `.progress-errors-title` | If severity levels exist on error items, surface the highest severity in the title (e.g. "Recent errors (12) — 3 critical"). |

---

### Interaction Design

**Strengths:**

- Cancel flow goes through `ConfirmDialog` with a "danger" variant and explicit
  confirmation label — intent is confirmed before action fires. Good.
- `aria-label="Cancel pipeline execution"` on the cancel button is present.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Cancel button almost certainly fails 44×44 px touch target | Critical | `.progress-cancel-btn` | Current rule: `padding: var(--space-xs) 10px; font-size: var(--font-size-xs)`. At typical token scales this produces a button well under 44 px tall. Add `min-height: 44px; min-width: 44px` explicitly. The button already carries `btn btn-danger` — check whether the base `btn` class establishes min-height; if not, add it there globally. |
| Error items have no copy affordance | Major | `.progress-error-item` | Errors are in a monospace scrollable box, but there is no copy-to-clipboard trigger. Users triaging failures will manually select text. Add a copy button per item (or a "Copy all errors" at title level). |
| Error list has no `role` or `aria-live` region | Major | `.progress-errors-container` | Errors arrive live (WebSocket). The container should carry `role="log"` and `aria-label="Recent errors"` so screen reader users hear new errors as they arrive without needing to navigate to the list. |
| Error items have no focus-visible treatment or keyboard-accessible copy path | Minor | `.progress-error-item` | If copy buttons are added, ensure `focus-visible` outline is present per WCAG 2.4.7. |
| `progress-bar-outer` has no `aria-valuenow` / `aria-valuetext` update during progress | Minor | `.progress-bar-outer` in ProgressView.tsx line 86 | `role="progressbar"` is present but indeterminate — add `aria-valuetext="Running"` updated to the status string so AT users know when it transitions to complete/failed without needing to navigate to the status label. |

---

### Accessibility Quick Check

| Criterion | WCAG | Result |
|-----------|------|--------|
| 1.4.3 Contrast — status label | 1.4.3 AA | Cannot verify without token values; colour-only state is a failure pattern regardless |
| 1.4.1 Use of Colour — status states | 1.4.1 | **Fail** — no non-colour distinguisher |
| 1.4.1 Use of Colour — WS banner single-state | 1.4.1 | **Fail** — all three connection states render identically |
| 1.4.1 Use of Colour — error token count colouring | 1.4.1 | **Fail** — `var(--color-error)` inline on `tokens_failed > 0` counter has no shape/icon companion |
| 2.1.1 Keyboard nav | 2.1.1 | Pass (cancel button is a `<button>`; confirm dialog receives focus) |
| 2.4.7 Focus visible — cancel button | 2.4.7 | Unknown — inherited from `btn btn-danger`; verify outline exists |
| 2.4.7 Focus visible — error items | 2.4.7 | Not applicable until copy affordance added |
| 4.1.3 Status messages — WS banner | 4.1.3 | Partial — `role="status"` present on banner and cancelled-msg; **absent on error list** |
| 4.1.3 Status messages — error list | 4.1.3 | **Fail** — no `aria-live` region; new errors arrive silently for AT users |
| 1.3.1 Info and Relationships — routing summary | 1.3.1 | **Fail** — number-label pairs expressed as unstructured text spans |
| 2.5.5 Target Size — cancel button | 2.5.5 | Likely **Fail** — padding-only sizing at `xs` font insufficient |
| 1.1.1 Alt Text | 1.1.1 | N/A — no `<img>` elements |

---

### Platform-Specific Notes

**Web / Panel:**

- The panel is nested inside a larger composer UI. `max-height: 200px` on
  `.progress-errors-container` is appropriate for a constrained panel, but
  verify that at the smallest panel width the monospace lines do not produce
  horizontal overflow without `overflow-x: auto` — currently absent.
- `flex-wrap: wrap` on `.progress-counters` is correct for responsive panel
  widths. However, without `font-variant-numeric: tabular-nums` on
  `.progress-counter-value`, live-updating numbers (e.g. going from "9" to
  "10") will cause layout jitter as digit widths differ. This is the counter
  alignment issue described below.

---

### Token Discipline / Dead Rules

**Token discipline:** Sound. All spatial and colour values reference custom
properties. No dead rules detected — all 19 classes present in the CSS are
referenced in `ProgressView.tsx`. No orphans.

**Two raw pixel values to tokenise:**
- `.progress-ws-banner`: `padding: 6px 10px` — use `var(--space-xs) var(--space-sm)` (or whatever maps to those values in the token scale)
- `.progress-cancel-btn`: `padding: ... 10px` — same horizontal token

---

### Priority Recommendations

**Critical (Fix Immediately):**

1. **Status pill has no non-colour identifier.** Add a `data-status` attribute
   to `.progress-status-label` in the JSX and CSS `content` pseudo-elements
   that render a distinct Unicode symbol or SVG icon per status
   (pending: ○, running: ◎, completed: ✓, failed: ✗, cancelled: ⊘). This
   satisfies WCAG 1.4.1 and also helps in high-ambient-light on mobile.
2. **Cancel button touch target.** Add `min-height: 44px` to
   `.progress-cancel-btn` (or to the `btn` base class if it is globally
   undersized). Confirm `min-width: 44px` or `min-width: fit-content` with
   adequate horizontal padding is also in effect.
3. **Error list `role="log"` and `aria-live`.** Add `role="log"` and
   `aria-label="Recent errors"` to `.progress-errors-container` in the JSX.
   This is a single-line change and unblocks screen reader users from receiving
   live error notifications.

**Major (Fix Before Launch):**

1. **WS banner state differentiation.** The current markup only conditionally
   shows or hides the banner based on `wsDisconnected`; it has one visual
   treatment. Add a reconnecting-vs-disconnected distinction by passing a
   connection phase prop and applying `.progress-ws-banner--reconnecting`
   (warning colour, e.g. "Reconnecting...") vs `.progress-ws-banner--lost`
   (error colour, e.g. "Connection lost. Trying to reconnect...").
2. **Counter numeric alignment.** Add `font-variant-numeric: tabular-nums` to
   `.progress-counter-value`. This prevents layout jitter as digit counts
   change during live updates without requiring fixed-width hacks.
3. **Error list copy affordance.** Add a "Copy all" button above the error
   container, and optionally a per-item copy icon (visible on hover/focus).
   Users debugging pipeline failures need this.
4. **Routing summary semantic structure.** Replace the bare `<span>` items
   inside `.progress-routing-summary` with `<dl>/<dt>/<dd>` pairs so screen
   readers announce both the label and value. The flex layout can be preserved
   on the `<dl>` container.
5. **Tokens-failed inline colour.** Move the conditional error colour on the
   failed-tokens counter from an inline `style` prop in JSX to a modifier CSS
   class (`.progress-counter-value--error`). Keeps styling auditable and
   theme-overridable.

**Minor (Improvement):**

1. **Tokenise raw pixel values** in `.progress-ws-banner` and
   `.progress-cancel-btn` padding.
2. **`aria-valuetext` on progress bar.** Update `aria-valuetext` to reflect
   the current status string so AT users receive state changes without
   navigating to the status label.
3. **`overflow-x: auto` on `.progress-errors-container`.** Guard against
   horizontal overflow for long single-line error messages in monospace.
4. **Error severity in title.** If error items carry severity data, surface the
   highest level in the count title for faster triage scanning.

---

### Confidence Assessment

**Confidence: High** for interaction and accessibility findings — the JSX is
fully readable and the structural gaps are unambiguous. **Moderate** for colour
contrast findings — token values are not resolved in this review; actual
contrast ratios require inspecting the design token definitions or measuring
in-browser. The 1.4.1 colour-only failures are evidence-based regardless of
token values.

### Risk Assessment

**High risk:** The three critical issues (status colour-only, touch target,
error list live region) will trigger WCAG A/AA failures in any formal
accessibility audit. They are also the issues most likely to surface in
usability testing with screen reader or motor-impaired users.

**Medium risk:** The WS banner single-state issue will cause operator confusion
during network instability — all states look identical.

**Low risk:** The token discipline and dead-rule cleanliness mean there is no
structural CSS debt to untangle during fixes.

### Information Gaps

- Design token values (especially `--font-size-xs`, `--space-xs`,
  `--color-warning`, `--color-error`) are not resolved here. Contrast ratios
  for `.progress-status-label`, `.progress-ws-banner`, and
  `.progress-errors-title` against their backgrounds require measuring actual
  computed values.
- The `btn btn-danger` base classes are not reviewed here. The cancel button
  touch-target finding assumes the base class does not already enforce
  `min-height: 44px` — verify before adding a redundant rule.
- Error item severity field: the review assumes severity data may exist on
  error items but does not confirm this from the type definitions. Check the
  `recent_errors` item shape.

### Caveats

This review is based on static analysis of the CSS and its primary JSX
consumer. It does not cover runtime rendering, actual computed pixel sizes
in the browser, theme-mode contrast, or the behaviour of `ConfirmDialog`
(which handles the cancel confirmation flow and is out of scope for this
file's review). Routing summary and WS banner behaviour in the reconnecting
state is inferred from the JSX; the actual `useWebSocket` hook state machine
was not reviewed.
