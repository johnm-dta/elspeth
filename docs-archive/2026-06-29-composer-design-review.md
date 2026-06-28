# Design Review: ELSPETH Web Composer

**Surface:** `src/elspeth/web/frontend` — React 18 + TypeScript SPA (~19.5k LOC components, ~8.5k LOC hand-written CSS over a CSS-custom-property design-token system; no Tailwind). An AI copilot for building data pipelines (chat + guided wizard + catalog + pipeline graph + execution/audit).

## Summary

**Overall assessment: Strong, accessibility-mature baseline with a concentrated cluster of real gaps.** This is not a codebase that "forgot about a11y" — it ships a full token system, `:focus-visible`, a skip-link, `forced-colors`/`prefers-contrast`/`prefers-reduced-motion` overrides, a 22-component `axe-core` test net, and a `colorContrast.test.ts` gate. The remaining issues are the *subtle, distributed* ones a mature surface still carries: the pipeline graph is effectively sighted-mouse-only, message authorship isn't programmatic, a handful of destructive actions lack confirmation, and a few real contrast/token-adherence gaps slipped past the existing test net.

- **Critical issues: 4**
- **Major issues: 14**
- **Minor issues: 47** (4 severity-adjusted down from major + 43 surfaced)
- **Refuted by verification: 0** (severity adjusted on 5; see method note)

### Method & evidence (and its limits)

This review combined four evidence sources so findings rest on data, not impression:
1. **Deterministic WCAG contrast matrix** — pure-math relative-luminance computation over every `tokens.css` pairing (not LLM eyeballing). This is the ground truth for all contrast claims.
2. **8 parallel review agents** (one per surface + one measurable-a11y baseline), each finding **adversarially verified** against the cited code and the contrast matrix. Every Critical/Major was re-read at file:line by a skeptical second agent; the 4 Criticals were then re-verified by hand.
3. **`axe-core` coverage map** — which components are/aren't in the automated net.
4. **16 live staging screenshots** (`elspeth.foundryside.dev`, dark + light, desktop + 640px + 375px) for the visual/reflow assessment.

**Honest limits:** Contrast/keyboard/ARIA are assessed precisely from code + math. Visual-hierarchy and reflow are grounded in real screenshots but at a few states only (most live sessions were stuck on the first-run tutorial overlay, so a fully-populated chat surface was reviewed from code, not pixels). **No live screen-reader or manual keyboard walkthrough was performed** — those are listed under Testing Recommendations and are the natural complement to this pass.

> Note on "0 refuted": the adversarial verifier downgraded 4 findings (major→minor) and upgraded 1 (major→critical) but refuted none outright. A zero-refutation run warrants suspicion, so the 4 Criticals were independently re-read against source (all held); one matrix-level contrast claim (focus-ring-on-primary-button) **was** caught as a false alarm by the baseline agent and removed.

---

## What's genuinely strong (lead with this)

A design review that only lists faults misrepresents this surface. Verified, evidence-cited strengths:

- **Keyboard & focus discipline is first-class.** `useFocusTrap` (save → move → wrap Tab/Shift+Tab → **restore on unmount**) is used consistently across GraphModal, CatalogDrawer, ConfirmDialog, SecretsPanel, ComposerPreferencesPanel, ExplainDialog. A grouped command palette (`Ctrl+K`, correct `role=combobox`+`listbox`+`aria-activedescendant`), a comprehensive shortcuts modal (`?`), and a global "Escape closes dialog/drawer" convention. Visible focus rings throughout.
- **Honest AI-trust design.** Generation is interruptible (Stop → `AbortController.abort`); users can **edit/fork a prior turn**; `ToolCallCard` discloses tool name + plain-language summary + a "Why:" rationale + "Affects:" list + a truncated `audit_event_id`; an atomic-reveal gate hides half-assembled turns so streaming never races the final answer; LLM egress is gated by a fan-out **preview-then-confirm** ConfirmDialog; placeholders are honest ("…did not include a summary field") rather than fabricated.
- **Color is rarely the sole signal.** Graph node validity = colored border **+ glyph** (×/!/✓) + `aria-label`; catalog audit tags render the **word** ("Network call"), with the emoji `display:none`; `AvailabilityDot` carries filled-vs-hollow shape + `role=img`/label; `AuditReadinessRow` adds a visually-hidden status label.
- **Near-total tokenization.** Zero hardcoded hex across ~7k lines of component CSS; JS-side inline styles reference `var(--…)`; documented target-size floors (`--size-control: 44px` / `-compact: 36px`) composed rather than re-declared.
- **Reduced-motion is thorough** — the global guard *plus* per-file silencing in chat/guided/tutorial CSS, including an honest static fallback for the indeterminate progress bar.

---

## Critical Issues (fix before launch)

| # | Issue | WCAG | Evidence | Recommendation |
|---|-------|------|----------|----------------|
| C1 | **Message authorship (user vs assistant vs system) is presentation-only.** `MessageBubble` distinguishes turns solely via CSS class (`message-bubble--${role}`, `bubble-user/assistant`, alignment + color + edge accent). No sender label, role text, or `sr-only` attribution. A screen-reader user hears a flat run of messages with no idea who said what — on an *AI* surface, where "did ELSPETH or I say this?" is load-bearing. | 1.3.1 (A) + AI-trust Legibility | `MessageBubble.tsx:114-120`; verified no `sr-only`/`aria-label` conveys author | Add a programmatic author label per turn — `<span class="sr-only">ELSPETH said:</span>` / `You said:` / `System:` inside each bubble (or `aria-label` on the row keyed off `message.role`). |
| C2 | **The pipeline graph hides its content from assistive tech.** The entire React Flow tree is wrapped in `role="img"` with an accessible name that is only a *count* ("Pipeline graph with N components"). `role=img` makes the subtree presentational — every node name, type, and validity state is invisible to AT. *(Mitigation: a `YamlView` text alternative exists for structure — but per-node validation state is graph-only.)* | 1.1.1 (A) | `GraphView.tsx:822-833` | Drop `role="img"` from the interactive container; expose a visually-hidden ordered list mirroring the DAG (node name, type, validity) as the accessible equivalent, or wire AT to the YAML view incl. validity. |
| C3 | **No keyboard path to inspect a node.** The inspector/config panel is driven by `selectedNodeId`, set *only* by `onNodeClick` (mouse). `GraphView` has no `onKeyDown`, no `onSelectionChange`, no keyboard node selection — so a keyboard-only user can never open a node's config. | 2.1.1 (A) | `GraphView.tsx:401-406, 849`; no `onSelectionChange`/`onKeyDown` present | Wire selection to a keyboard-inclusive source (`onSelectionChange`, or focusable nodes + Enter/Space) so the config panel opens without a mouse. |
| C4 | **Filter input + "Show archived" checkbox are keyboard-unreachable inside the open session menu.** The menu's `onKeyDown` handles `case "Tab"` with `preventDefault(); closeAndReturnFocus()` and **no `e.shiftKey` branch**, so *both* Tab and Shift+Tab close the menu; focus is forced onto a menuitem on open, and the controls live outside the roving index. | 2.1.1 (A) | `HeaderSessionSwitcher.tsx:191-194` | Include the filter input + checkbox in the Tab/roving sequence (let Shift+Tab from the first menuitem move *into* the controls), or move initial focus to the filter input. |

> C2 and C3 share a root cause — **the pipeline graph is a sighted-mouse-only artifact.** They're listed separately because they fail different success criteria (AT can't *read* it; keyboard can't *operate* it) and need different fixes. The YAML view softens C2 but does not resolve C3.

---

## Accessibility

### WCAG 2.2 AA compliance (composite across surfaces)

- [~] **1.4.3 Contrast (Text, AA):** *Partial.* Body/secondary/link/button text all pass (verified by math). **Fails:** `--color-danger` undefined → `#b00020` fallback (~1.6:1 on dark) in error text (M11); semantic colours as **banner body text** on their own tint (success/error/warning 3.4–4.4:1) (M12); muted text on **elevated/paper** surfaces (3.84/4.37:1), ungated by the contrast test (M14).
- [x] **2.1.1 Keyboard:** *Fails* — C3 (graph node inspect), C4 (session-menu controls).
- [x] **2.4.7 Focus Visible:** *Pass* globally (`:focus-visible` 2px ring, `outline-offset:2px`). Minor: command-palette input suppresses it to a 1px border change.
- [✓] **2.4.11 Focus Not Obscured (2.2):** *Pass* — modals/drawers manage focus and aren't obscured by sticky chrome in the states reviewed.
- [✓] **2.5.7 Dragging Movements (2.2):** *Pass* — graph nodes are `nodesDraggable={false}`; no drag-only operations found.
- [~] **2.5.8 Target Size Minimum (2.2, 24px):** *Mostly pass* (documented floors), with a few sub-floor controls (node-config close 32×32 is fine; custom-chip remove + opt-out link + a settings close button dip below) — minor.
- [~] **3.2.6 Consistent Help (2.2):** *Partial* — shortcuts/help exist but there's **no visible help affordance in the shell chrome** (discovery is keyboard-only via `?`).
- [x] **3.3.7 Redundant Entry (2.2):** *Fails* — secret name field is wiped on a *failed* save, forcing re-entry (M-settings).
- [✓] **3.3.8 Accessible Authentication (2.2):** *Pass* — LoginPage uses real labels, `autoComplete`, `type=password` with **no paste-blocker** and no cognitive-function test.
- [~] **4.1.2 Name/Role/Value:** *Partial* — C1 (authorship); SaveForReviewDialog declares `aria-modal` but isn't a real modal; disabled-reason `aria-describedby` unreachable because `disabled` removes the button from AT; Tabs primitive lacks `aria-controls`.
- [x] **4.1.3 Status Messages:** *Fails* — catalog load-failure/empty/result-count (M-catalog), run start/completion (M-inspector), and YAML copy success/failure are not announced.

### Accessibility issues (Major)

| Issue | Severity | Evidence | Fix |
|-------|----------|----------|-----|
| Error text uses **undefined `--color-danger`** → always falls back to `#b00020` (~1.6:1 on dark) | Major (1.4.3) | `InlineOptOutCheckbox.tsx:74`, `ComposerPreferencesPanel.tsx:91`; token never defined (grep-confirmed) | Replace with `var(--color-error)` (already passes), or define `--color-danger` in both themes. |
| **Semantic colour as banner body text** on its own 12% tint fails AA | Major (1.4.3) | `header.css:18-22` `.alert-banner{color:var(--color-error);background:var(--color-error-bg)}` @13px; same for warning/validation | Add deepened on-tint text tokens, or use `--color-text` on the tint and reserve the hue for the border/icon. |
| **Muted text on elevated/paper surface = 3.84:1** at 10px (audit label, catalog meta) — ungated | Major (1.4.3) | `catalog.css:103-105`, `sidebar.css:136-138`; matrix-confirmed | Use `--color-text-secondary` on elevated/paper, and extend `colorContrast.test.ts` to gate these pairings. |
| **Input boundary is the sole affordance and < 3:1** at rest (1px `border-strong` on elevated) | Major (1.4.11) | `shared.css:302-309`, `guided.css:753-757` | Add a `--color-input-border` ≥3:1 (mirror the `--color-chip-border` rationale). |
| **Catalog status changes not announced** (load-failure, empty, result count) | Major (4.1.3) | `CatalogDrawer.tsx:511-536` — only the *loading* branch has `role=status` | `role=alert` on error; polite live region for empty/result-count. |
| **No heading/list semantics in the catalog** — ~26 plugins are a flat run of unlabelled `<div>`s | Major (1.3.1) | `PluginCard.tsx:148`, `CatalogDrawer.tsx:538-553` | `role=list`/`listitem` (or `ul/li`); promote plugin name to a heading. |
| **Run start/completion not announced** (only `cancelled` is a live region) | Major (4.1.3) | `ProgressView.tsx:96-101, 160-194` | Polite live region announcing each terminal status + counts. |
| **RunsHistoryDrawer traps focus but never restores** it to the opener | Major (2.4.3) | `RunsHistoryDrawer.tsx:80-117` (bespoke trap, no `previouslyFocused`) | Reuse `useFocusTrap` (as GraphModal does). |
| **SaveForReviewDialog declares `aria-modal` but is an in-flow panel** — no trap, no initial focus, no backdrop | Major (4.1.2) | `SaveForReviewDialog.tsx:110-116`, `composer.css:9-15` | Add `useFocusTrap` + backdrop, or drop the modal ARIA and treat it as an inline region. |
| **Blob status (ready/pending/error) is colour-hue alone** | Major (1.4.1) | `BlobRow.tsx:110-117` — bare `<span>` dot, no role/label/shape | Mirror `AvailabilityDot`: `role=img`+label + non-colour shape/icon. |

---

## Interaction Design

### Issues (Major)

| Issue | Severity | Evidence | Fix |
|-------|----------|----------|-----|
| **Deleting a stored secret/API key is immediate — no confirm, no undo** | Major (3.3.4) | `SecretsPanel.tsx:155-160, 350-359` ("×" glyph → `deleteSecret`, no guard) | Route through the existing `ConfirmDialog` (danger variant) naming the secret. |
| **Deleting a file in the Blob Manager is immediate — no confirm, no undo** | Major (3.3.4) | `BlobRow.tsx:170-177`, `BlobManager.tsx:78-84` | Gate behind `ConfirmDialog` (danger) or provide a visible undo window. Files are pipeline inputs/outputs — data-loss risk. |
| **Secret name field wiped on a *failed* save** — defeats the inline error it just set | Major (3.3.7) | `SecretsPanel.tsx:144-150` (unconditional `finally` clears `name`) | Preserve `name` on failure (keep the security-motivated value-clear); clear `name` only on success. |
| **`SchemaFormTurn` has no required-field indication and no inline validation** — only signal is a disabled Continue button | Major (3.3.2) | `SchemaFormTurn.tsx:243-411` (six input branches, no `required`/`aria-required`) | Mark required fields (asterisk + text + `aria-required`); add `aria-invalid`+`aria-describedby` inline errors. |

### Notable minors

- Accept-proposal commits a pipeline mutation in **one tap with no in-surface undo** (AI-trust Reversibility).
- Danger `ConfirmDialog` **auto-focuses the destructive Confirm button**, pre-arming Enter (used for Archive/Revert/Execute-egress). Standard pattern is to focus Cancel for `variant=danger`.
- YAML Copy success and clipboard failure are silent; run-results collapse toggle lacks `aria-expanded`.

---

## Information Architecture

- **No back/revise across committed wizard steps** (minor, 3.2.3): `ControlSignal` has no `back`/`previous`; the stepper items are non-interactive `<li>`. A user who picked the wrong source must abandon guided mode. *Recommend a per-step Back control.*
- **No `<main>` landmark** around the primary content region (minor, 1.3.1) — and the skip-link target `#chat-main` only exists inside ChatPanel, so the bypass resolves on just 1 of 5 tutorial turns (minor, 2.4.1).
- **No visible help/shortcuts affordance in the chrome** (minor, 3.2.6) — discovery is keyboard-only (`?`). Catalog cards redundantly repeat the plugin-type label already conveyed by the selected tab.
- *Strength:* the catalog's per-tab filter state, `role=tablist` model, and disclosure pattern are textbook; the command palette's ACTIONS/NAVIGATION/SESSIONS grouping is excellent.

---

## Visual Design (grounded in live screenshots + the contrast matrix)

- **Visual hierarchy is clear and consistent** — the first-run welcome card, SENSE/DECIDE/ACT chunking, catalog drawer, and command palette all show strong grouping, whitespace, and a single emphasized primary action ("Let's go" / "New Session"). Dark and light themes both render cleanly.
- **The pipeline graph wastes most of the canvas** (minor): a single top-to-bottom node column floats in large empty horizontal margins (confirmed in `D2-graph-dark-true.png`). Consider horizontal/auto-layout or centering with denser nodes.
- **Muted helper text reads dim on dark surfaces** — corroborated visually (the "This step calls the configured LLM…" sub-text) and by math (muted-on-elevated 3.84:1). The 10px `--font-size-3xs` used for *content* labels (not just chrome), compounded by muted color, is a 1.4.4/1.4.3 concern.
- **The "Spec tab was removed — Showing Graph instead" redirect banner** is low-contrast light-blue on dark teal and persistent; it also uses `role=alert` (assertive) for a low-priority informational redirect.
- *Corrected false alarm:* the matrix flagged the light-theme focus ring at 2.75:1 on the primary-button fill — but `outline-offset:2px` lands the ring on the page background (13.57:1). **Not a defect.**

---

## AI Trust Stack

- [✓] **Legibility:** *Mostly pass* — tool calls are disclosed with rationale; streaming is visually + programmatically distinct (atomic-reveal + `role=status` ComposingIndicator). **Gap:** message authorship isn't programmatic for AT (C1).
- [✓] **Grounding:** *Pass* — `ToolCallCard` "Why:"/"Affects:"/audit-id; `InterpretationReviewTurn` surfaces LLM assumptions before the pipeline runs; honest "no summary" placeholders instead of fabrication.
- [✓] **Steering:** *Pass* — interrupt (Stop/abort), edit/fork prior turn, opt-out to freeform, exit-to-freeform.
- [✓] **Refusal & Recovery:** *Pass* — run failures render Retry + `role=alert`; interpretation amendment ("Change it: I meant…").
- [~] **Reversibility:** *Partial* — LLM egress and pipeline cancel are preview-then-confirm (strong), but Accept-proposal and secret/blob deletion are one-tap with no undo.
- [~] **Calibration:** *Partial* — prompt-shield trust caveats are surfaced honestly; but tutorial run-phase narration is driven by **fixed timers, not real progress**, in an audit-first product (minor).

---

## Platform / Responsive

- **Reflow passes** at 640px and 375px — the welcome cards and tutorial layers stack to a single column with no clip or horizontal overflow (verified in `13-reflow-640.png`, `14-mobile-375.png`). The guided stepper collapses to 2 columns under 640px.
- At 375px the header version selector + Account crowd the right edge — minor.

---

## Priority Recommendations

### Critical (before launch)
1. **Make the pipeline graph operable & legible without a mouse** (C2+C3): drop `role=img`, expose a visually-hidden DAG list with per-node validity, and wire keyboard node selection to the config panel.
2. **Add programmatic message authorship** to every chat turn (C1).
3. **Fix the session-menu keyboard trap** so the filter + archived toggle are reachable (C4).

### Major (fix soon)
1. **Confirm destructive actions:** secret deletion, blob deletion → `ConfirmDialog` (danger) or undo window.
2. **Close the contrast gaps:** define/replace `--color-danger`; fix semantic-colour-as-banner-text; switch muted→secondary on elevated/paper; add a ≥3:1 input border — and **extend `colorContrast.test.ts`** to gate the muted-on-elevated/paper and on-tint pairings so they can't regress.
3. **Announce status changes:** catalog load/empty/result-count, run start/completion, YAML copy (live regions).
4. **Fix focus restoration** (RunsHistoryDrawer) and the **fake modal** (SaveForReviewDialog).
5. **Required-field affordance + inline validation** in `SchemaFormTurn`; preserve the secret name on failed save.
6. **Catalog list/heading semantics** for the ~26-plugin browse list.

### Minor (polish)
- Focus Cancel (not Confirm) for `variant=danger` dialogs; add an in-surface undo for Accept-proposal.
- Add a `<main>` landmark + stable skip-link target; add a visible help affordance in the chrome.
- Graph auto-layout to use canvas width; raise 10px content labels; quieten the redirect toast to `role=status`.
- Per-tab count `aria-label` override (catalog), Tabs-primitive `aria-controls`/roving, disabled-reason reachability.

---

## Testing Recommendations

- [ ] **Manual screen-reader pass** (NVDA + VoiceOver) over chat, the graph, and the guided wizard — the highest-value follow-up; this review establishes the targets but did not run live AT.
- [ ] **Keyboard-only walkthrough** of the graph node-inspect path and the session-switcher menu (confirm C3/C4 fixes).
- [ ] **Extend the `axe-core` net** to the unaudited heavyweights: `ChatPanel`, `MessageBubble`, `GraphView`, `RunOutputsPanel`, execution/* , `SecretsPanel`, `CommandPalette` (note: axe-in-jsdom can't see contrast/focus/reflow — pair with the contrast test + manual).
- [ ] **Colorblind simulation** on the graph + status surfaces (already strong; confirm).
- [ ] **200%/400% browser-zoom reflow** beyond the viewport-narrowing proxy used here.
