# UX Review: chat.css
**Date:** 2026-05-23
**Reviewer:** UX Critic Agent (lyra-ux-designer)
**File:** `src/elspeth/web/frontend/src/components/chat/chat.css` (1,408 lines)
**Status:** Needs Work — two critical issues and five major issues require remediation before demo release

---

## Summary

The file carries substantial well-considered work: bubble role distinction uses three independent cues (alignment, background, typography — not colour alone), the touch overlay rule fires correctly via `@media (hover: none)`, and the `composing-dot` reduced-motion guard in `animations.css` is correctly maintained. Token discipline is largely good but has a cluster of raw literals. Two layout bugs are critical because they affect every message: action overlays occlude bubble content at all times for touch users and at hover for pointer users, and tool-call JSON details panels have no scroll cap. Three further issues land as major.

---

## Visual Design

**Strengths:**
- Bubble role triple-encoding is solid: user (`align-self: flex-end`, green-tinted background, right-aligned row), assistant (`align-self: flex-start`, neutral surface, left-aligned row), system (centred, full-width, italic, muted text). No single-cue reliance on colour.
- Tool-call card status uses border-left accent (`--color-warning` / `--color-success` / `--color-error`) AND the heading string ("Proposed:", "Applied:", "Rejected:") in `ToolCallCard.tsx` line 38–42. Passes WCAG 1.4.1.
- `.pending-proposals-banner` and `.inline-source-fallback-prompt` both use border-left accent plus a text header, not colour alone.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Raw `z-index: 5` on `.pending-overlay-pill` ignores the project's `--z-*` token ladder. The token ladder in tokens.css lines 197–206 defines `--z-panel-controls: 10` as the floor for floating panel controls; `5` undercuts it without documented justification. | Minor | Line 1157 | Replace with `var(--z-panel-controls)` or add a dedicated `--z-overlay-pill: 5` token with an explanation. |
| `.template-card-description` and `.template-card-sda dd` (lines 1358–1365, 1389–1395) use `-webkit-line-clamp` without the now-standard `line-clamp`. The `-webkit-` form works in all current engines but the pairing is the canonical pattern documented by the CSS spec. | Minor | Lines 1358, 1389 | Add `line-clamp: 2;` alongside the `-webkit-line-clamp: 2;` declarations. |
| Template-card `aspect-ratio: 1/1` (line 1320) is retained at the 520px single-column breakpoint. One full-width square tile per viewport on a 375px phone occupies roughly 345px height, consuming the entire visible area for one card. The user must scroll past one card to see a second. | Major | Lines 1320, 1403 | At the 520px breakpoint, replace `aspect-ratio: 1/1` with `min-height: 140px` so cards are usably compact without enforcing a square constraint. |

---

## Information Architecture

**Strengths:**
- `.message-tools-toggle` as a shared visual class between the collapsible tool-calls group (a `<button>`) and the static sources-created heading (a `<div>`) is a deliberate design decision documented in the CSS comment at line 828 and in `MessageBubble.tsx` lines 269–273. The `cursor: default` override on `.message-sources-created-heading` correctly signals non-interactivity.
- `PendingProposalsBanner` co-locates proposal actions with the input area by CSS positioning within `.chat-panel`, avoiding forced upward scroll. Logical grouping is correct.
- `InlineSourceFallbackPrompt` stacks vertically at 760px (line 1144), converting a horizontal flex row to column, keeping both the copy and action buttons reachable.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.spec-pending-proposal` and `.runs-pending-proposal` (lines 1169–1183, 1185–1188) are dead rules. No TSX file in the chat/ subtree or across the full frontend `src/` applies either class name (grep confirms zero hits against `*.tsx` / `*.ts`). The shared visual styling is partially covered by `.yaml-pending-summary` (used in `inspector/YamlView.tsx`) but the spec and runs variants are orphaned. | Minor | Lines 1169, 1185 | Delete `.spec-pending-proposal` and `.runs-pending-proposal`. If the spec or runs surfaces are expected to appear in the chat panel in a future phase, create the rules in the relevant component CSS when the TSX ships, not speculatively. |

---

## Interaction Design

**Strengths:**
- `.bubble-copy-btn` / `.bubble-edit-btn` touch visibility: `@media (hover: none)` sets `opacity: 0.6` (lines 86–91), making the buttons persistently visible on touch devices. This follows the CSS spec recommendation for always-accessible controls on touch. The 36px `--size-control-compact` minimum size is WCAG 2.5.8 AA-compliant.
- `.bubble-copy-btn:focus-visible` and `.bubble-edit-btn:focus-visible` reveal the buttons at `opacity: 1` (line 82–85), so keyboard users trigger visibility correctly. The global `:focus-visible` rule in `common.css` line 83 provides the 2px focus ring.
- `.chat-input-icon-btn` and `.chat-input-send-btn` use `min-height: var(--size-control)` (44px) — WCAG 2.5.5 AAA touch target.
- Chat input `aria-describedby` wired to `hintId` (`ChatInput.tsx` line 249) — the "Shift+Enter for new line" hint is NOT `aria-hidden`, so screen readers receive it after the textarea label. This is the correct pattern.
- `scroll-to-bottom-btn` uses `aria-label="Scroll to bottom"` (ChatPanel.tsx line 1468) and has `min-height: var(--size-control-compact)` — meets AA floor.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| **CRITICAL — Overlay occluding bubble content.** `.bubble-action-overlay` is `position: absolute; top: 0` with right-pinned offsets (copy at `right: 0`, edit at `right: 44px`), inside `.message-bubble-content` which has no `padding-right` reservation (line 649–656). For touch users, both buttons are permanently visible at `opacity: 0.6`. For pointer users, both buttons appear on hover. In either case, the 44px-wide buttons sit directly over the top-right corner of the bubble text. A short user message (one line) or any short assistant reply has its content partially occluded. | Critical | Lines 662–686, 86–91 | Add `padding-top: calc(var(--size-control) + var(--space-xs))` to `.message-bubble-content` when overlay buttons are present, OR move the overlay container outside the bubble box (a sibling `position: absolute` on the `.message-row` wrapper). The second approach is more robust because it does not grow the bubble height. |
| **CRITICAL — Tool-call details panel unbounded.** `.tool-call-details pre` (lines 998–1006) has no `max-height` and no `overflow` property. The `set_pipeline` composer tool serialises full pipeline YAML into its arguments. A routine proposal can produce 150–300 lines of JSON. The pre block will grow to fill and push siblings off screen. The review task explicitly requires "scrollable details when long." | Critical | Lines 998–1006 | Add `max-height: 320px; overflow-y: auto;` to `.tool-call-details pre`. Consider also adding `overflow-x: auto` since JSON argument objects can contain long unbroken strings. |
| **MAJOR — Scroll-to-bottom `bottom: 80px` brittle at mobile wrap.** At 760px and below, `.chat-input-row` wraps (`flex-wrap: wrap` at line 469), the textarea takes full width with `min-height: 68px`, and the icon buttons wrap to a second row. The total input zone height exceeds 80px, causing `.scroll-to-bottom-btn` to visually overlap the send button. The existing `:has(.inline-run-results)` rule (lines 114–116) shows this mechanism is available but does not cover the bare wrapped state. | Major | Lines 96–99, 325–329 | Add a `@media (max-width: 760px)` override on `.scroll-to-bottom-btn` that raises `bottom` to at least `140px` to clear the wrapped input row. Alternatively, scope the button to be positioned relative to `.chat-panel-messages` rather than `.chat-panel`, isolating it from input zone height changes. |
| **MAJOR — "Change my default" button inline styles.** `ChatPanel.tsx` lines 1251–1264 render `.chat-panel-change-default` with `style={{ minHeight: 24, fontSize: 12, ... }}` — seven raw inline style properties for a button that has a CSS class. The class is named in the file but carries no rules in chat.css. The inline `minHeight: 24` falls below `--size-control-compact` (36px) and fails WCAG 2.5.8 AA. | Major | ChatPanel.tsx 1251–1264; chat.css (missing rule) | Add a `.chat-panel-change-default` rule to chat.css with `min-height: var(--size-control-compact)` and the remaining style properties. Remove all inline style declarations from the TSX. |
| **MAJOR — `chat-input-send-btn` no focus-visible differentiation on disabled.** The send button is always present in the DOM; in disabled state it uses `var(--color-surface-elevated)` background with `var(--color-text-muted)` text. The global `:focus-visible` rule from `common.css` still draws a 2px white outline. However, when enabled, the button switches to `var(--color-accent)` background — a deep green (#1a7a52 dark / #156048 light) against which a white focus ring is adequate, but against the elevated surface background the white ring is the only visible interactive cue. There is no `:hover` rule on the enabled send button, so the only state change indication at pointer is the `cursor: pointer` change. | Major | Lines 443–459 | Add `.chat-input-send-btn:not(:disabled):hover { background-color: var(--color-btn-primary-bg-hover); }` to make the hover state visible. Add `.chat-input-send-btn:not(:disabled):focus-visible { outline-offset: 3px; }` to ensure the focus ring clears the filled green background. |

---

## Accessibility

**Quick Check:**
- [x] 1.4.3 Contrast: Passes — dark theme `--color-text` (#dff0ee) on `--color-surface` (#122f37) and all bubble backgrounds verified. `--color-text-muted` (#7a9a9a) on `--color-bubble-assistant` is the border case; project's `colorContrast.test.ts` is the authoritative gate.
- [x] 2.1.1 Keyboard: Passes for overlay buttons (`:focus-visible` makes them visible). Fails on send button (no hover cue, no differentiation beyond cursor).
- [x] 2.4.7 Focus Visible: Passes — global rule in `common.css` line 83 covers all focusable elements including chat buttons.
- [x] 1.1.1 Alt Text: Passes — `ChatInputIcon` SVGs are `aria-hidden="true"` (ChatInput.tsx line 48); all interactive buttons have explicit `aria-label`. Composing pulse dots are `aria-hidden="true"` (ComposingIndicator.tsx line 166). Template card icons are `aria-hidden="true"` (TemplateCards.tsx line 51); the `<article>` carries an `aria-label` combining domain and description.
- [x] 1.4.11 Non-text Contrast: Passes for tool-call status (border-left + text heading dual-encoding). Passes for bubble distinction (alignment + background + font style).

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `.composing-terminal-mark` (lines 1224–1235) is marked `aria-hidden="true"` in ComposingIndicator.tsx line 162. The terminal state text ("Updated", "Failed", "Stopped") is therefore invisible to screen readers. The `role="status"` on the outer `.composing-indicator` wrapper would normally announce content changes, but since the text is hidden the SR receives no terminal signal when the composing phase ends. The working-view text below is not hidden, but it describes the *most recent update*, not the terminal state itself. | Major | ComposingIndicator.tsx 162; chat.css 1224 | Remove `aria-hidden="true"` from `.composing-terminal-mark` so the terminal label is announced by the `role="status"` parent, OR add a `<span class="sr-only">` sibling with the terminal text that is not aria-hidden. |
| `.bubble-system` has no explicit border (contrast cue) or prefix icon. Its role distinction relies on centre alignment, full width, italic text, and muted colour — three of four are non-colour cues, which is acceptable, but the task specification lists "border" as an expected distinguisher. Users who override or can't perceive font-style (e.g., dyslex-specific stylesheets that remove italic) lose the typographic signal and must rely on position alone. | Minor | Lines 52–58 | Add `border: 1px solid var(--color-border)` (the same border strength used by `.tool-call-ribbon`) to `.bubble-system` to provide a structural cue independent of font-style. |

---

## Token Discipline

**Strengths:**
- Control size tokens used consistently: `--size-control` (44px) on primary interactive buttons, `--size-control-compact` (36px) on overlay and chrome buttons.
- Spacing tokens cover most of the layout: `--space-xs`, `--space-sm`, `--space-md`, `--space-lg`, `--space-xl`, `--space-2xl` all appear correctly.
- Transitions: no raw `ms` literals in chat.css (transition values use the token ladder or the inline `0.15s ease` on the overlay opacity which is within the project's documented fast tier).

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `min-width: 44px` appears raw at lines 121 (`.blob-action-btn`) and 397 (`.chat-input-icon-btn`). `min-width: 56px` appears raw at line 422 (`.chat-input-cancel-btn`). Per tokens.css comment lines 183–188, new buttons must compose `var(--size-control)` rather than redeclaring the literal. | Minor | Lines 121, 397, 422 | Replace raw `44px` with `var(--size-control)`, `36px` with `var(--size-control-compact)`. The 56px cancel button minimum is non-standard; document it or resolve to the nearest token. |
| Magic numbers cluster in the `composing-bubble` block: `padding: 10px 14px` (line 1201), `max-width: min(640px, calc(100vw - 48px))` (line 1208), `min-width: 34px` on `.composing-pulse` (line 1215), `padding-top: 5px` (line 1217). The 10/14px padding is between `--space-sm` (8px) and `--space-md` (12px); no intermediate token exists, but `10px` is not a token value. | Minor | Lines 1201, 1208, 1215, 1217 | Use `--space-sm` / `--space-md` for the composing bubble padding (the visual difference from 10/14 to 8/12 is imperceptible). The 640px max-width and 48px clearance are layout-specific values that can remain as named CSS custom properties on `.composing-bubble` if global tokens would be too broad. |
| Six raw `6px` values appear across the file at lines 101, 269, 772, 826, 1240, 1323. The token ladder has `--space-xs: 4px` and `--space-sm: 8px` but no 6px entry. Using `6px` is not strictly wrong, but inconsistency in the gap between `xs` and `sm` creates drift. | Minor | Lines 101, 269, 772, 826, 1240, 1323 | Either add `--space-2xs-plus: 6px` to tokens.css (risky — adding a token between two others suggests the ladder is wrong) OR normalise to `--space-xs` or `--space-sm` depending on context. Normalising to `--space-sm` is the safer bet for most of these use cases (bubble padding, section gaps). |

---

## Platform-Specific Notes

### Mobile (760px and 520px breakpoints)

- Bubble max-width of 85% (`--bubble`, line 19) and message-bubble-content max-width of 80% (line 650) are appropriate for mobile readability.
- At 760px the input row wraps (line 469) but does not remove the hint text below the input. The `Shift+Enter for new line` hint (`--font-size-2xs: 11px`) remains visible but at 11px may be at the edge of comfortable reading on a small display. No immediate fix required but monitor with real device testing.
- Template card grid collapses 4-column → 2-column at 760px (line 1397) and 2-column → 1-column at 520px (line 1403). The `aspect-ratio: 1/1` is retained at both breakpoints — see the major finding above.
- `inline-run-results` is `max-height: min(300px, 36vh)` at 760px (line 330). On a 667px viewport (iPhone SE landscape) 36vh is 240px — adequate.

### Keyboard Navigation

- Tool-call `<details>` / `<summary>` expand/collapse is native HTML and keyboard-accessible by default. The `.tool-call-details summary` (line 993) has `cursor: pointer` and the global `summary:focus-visible` rule in `common.css` line 93 provides the focus ring. Pass.
- The `.message-tools-toggle` button carries `aria-expanded` wired in `MessageBubble.tsx` line 209 and the triangle symbols `▼ / ▶` as visual state indicators alongside the text. The aria-expanded is the accessible state; the symbol is reinforcement. Pass.

---

## Priority Recommendations

**Critical (Fix Immediately):**

1. **Action overlay occludes bubble content.** Add padding reservation to `.message-bubble-content`, or relocate the overlay container outside the bubble box, so the copy and edit buttons do not overlap text for touch users (always visible at opacity 0.6) and pointer users (visible on hover). Lines 662–686 and 86–91.

2. **Tool-call details pre has no scroll cap.** Add `max-height: 320px; overflow-y: auto;` to `.tool-call-details pre` (lines 998–1006). A full `set_pipeline` payload will overflow the chat panel without this constraint.

**Major (Fix Before Demo):**

3. **"Change my default" inline styles violate token discipline and fail WCAG 2.5.8.** Add `.chat-panel-change-default` rules to chat.css with `min-height: var(--size-control-compact)`. Remove the seven inline style properties from `ChatPanel.tsx` lines 1251–1264.

4. **Terminal composing mark is aria-hidden.** Remove `aria-hidden="true"` from `.composing-terminal-mark` in `ComposingIndicator.tsx` line 162, or add a visually-hidden `<span>` carrying the terminal state label. Screen readers do not receive the "Updated"/"Failed"/"Stopped" state change through the `role="status"` parent because the text is hidden.

5. **Scroll-to-bottom button sits below the wrapped input row on mobile.** At 760px the input wraps to two rows exceeding 80px; the button overlaps input controls. Add a `@media (max-width: 760px)` override raising `bottom` to at least `140px`.

**Minor (Improvement):**

6. **Template card `aspect-ratio: 1/1` retained at 520px single-column breakpoint.** One full-width square card per viewport on a 375px phone. Replace with `min-height: 140px` at the 520px breakpoint.

7. **`.spec-pending-proposal` and `.runs-pending-proposal` are dead rules** — zero TSX consumers. Delete both (lines 1169, 1185).

8. **Raw `min-width: 44px` / `min-height: 24px` literals.** Replace with `var(--size-control)` / `var(--size-control-compact)` at lines 121, 397, 422, and in ChatPanel.tsx.

9. **`.bubble-system` lacks a structural cue.** Add `border: 1px solid var(--color-border)` so the system role is distinguishable without relying on italic font style alone.

---

## Confidence Assessment

**High confidence** on findings 1–5 (critical and major): each is supported by direct code evidence with line numbers and verified against the TSX consumers.

**Medium confidence** on finding 3 (button contrast): the exact contrast ratio of the focus ring against `--color-accent` (#1a7a52) was not computationally verified in this session; the project's `colorContrast.test.ts` is the authoritative gate and should be updated to assert focus-ring contrast against the filled send button background.

**Lower confidence** on the line-height assessment for `.bubble`: `line-height: 1.42` (line 24) is between the project tokens `--line-height-tight: 1.3` and `--line-height-normal: 1.5`. WCAG SC 1.4.12 requires that body text line height be at least 1.5x. Bubble content is the primary reading surface. The value 1.42 is close but technically sub-threshold. This was not raised as a finding because the project's line-height tokens were not explicitly verified as compliant and because the spec wording is "at least 1.5x *without loss of content or functionality*" — bubbles use `word-break: break-word` which prevents content loss. Recommend verifying against SC 1.4.12 in a separate accessibility audit.

## Risk Assessment

**Highest risk for demo:** Finding 1 (overlay occlusion) affects every message exchange. On touch devices — which the demo may use — every message has permanently-visible buttons overlapping its top-right corner. This is visually obvious and will be noticed by a live audience.

**Second highest risk for demo:** Finding 2 (unbounded tool-call pre). The canonical demo prompt ("create a list of 5 government web pages and rate how cool they are") triggers a `set_pipeline` tool call. The JSON arguments will expand in the UI and push the accept/reject actions off screen if the demo operator opens the details disclosure.

## Information Gaps

- Exact contrast ratios for `--color-text-muted` on bubble backgrounds were not computed in this session. The project's `colorContrast.test.ts` is assumed to cover these pairings.
- The `--color-bubble-user-border` green tint (rgba(40,130,100,0.35)) against `--color-surface` was not verified to meet 3:1 for WCAG 1.4.11 (UI component contrast). This is a low-risk gap because the border is primarily aesthetic rather than conveying unique information (bubble identity is also conveyed by alignment).
- Light theme bubble contrast was not independently verified here; `colorContrast.test.ts` is the authoritative source.

## Caveats

This review is based on static analysis of `chat.css` and the primary TSX consumers in `components/chat/` (excluding `guided/`). Dynamic states (e.g., the exact rendered height of the wrapped input row at 760px) were inferred from CSS rules, not measured in a browser. Recommend verifying findings 3 and 5 on physical device or browser DevTools before implementing fixes.

The `guided/` subdirectory was excluded per the task scope. Guided-mode CSS may contain related issues not captured here.
