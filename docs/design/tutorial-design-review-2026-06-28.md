# Design Review — First-run tutorial / guided composer (2026-06-28)

**Scope:** the in-app first-run tutorial (`HelloWorldTutorial` → `TutorialGuidedShell`
→ embedded `ChatPanel` guided surface) and the design-system migration that landed
alongside it in the last ~48h (`d56197fde` DS merge, `4cbfcb6aa` staged guided
composer, `f2ff4e912` ui/Input wiring).

**Method:** static cross-check of every class the guided/tutorial components emit
against the app CSS cascade (`styles/index.css` barrel), plus live Playwright
screenshots of the rendered surfaces against staging (`elspeth.foundryside.dev`,
fresh build `index-BDLZqn8p.js`) at desktop (1366px) and mobile (390px), driving the
real LLM walk through source → output stages.

> Provenance note: staging served a **stale** bundle (built Jun 27 22:35, before the
> 48h tutorial work). The frontend was rebuilt for this review so screenshots reflect
> current `release/0.7.0` source.

## Summary

**Overall:** Needs Work. The *static chrome* (welcome bookend, tokens, focus rings,
reduced-motion, ARIA) is genuinely strong. The **in-flow guided surface is janky**:
broken concatenated labels, unstyled buttons, a mobile layout that wraps stepper
labels one character per line, approvals placed below the input that gates them, and
tutorial-inappropriate controls leaking through.

- **Critical:** 0
- **Major:** 6
- **Minor:** 7
- **Latent (DS wiring debt):** 3

Most defects are **not** new — class-name drift between the guided widgets and
`guided.css` predates the 48h window (git `-S` traces to `a1bbbe60a` / widget
creation). The 48h work (tutorial now exercising the full guided engine + the DS
migration) is what made them *visible*. Attribution aside, they are real and on the
first-run path.

---

## Remediation status (applied 2026-06-28, build `index-CQJbu4S9`, live on staging)

**Fixed (additive CSS / tutorial-only guard — no behaviour change to non-tutorial flows):**
- **M3** mobile stepper — `repeat(6…)`→`repeat(5…)`, `@media(max-width:640px)` 2-col, `overflow-wrap:break-word`. *Verified on staging: labels whole, no shatter.*
- **M4** freeform opt-out now gated `{!isTutorial && <InlineOptOutCheckbox/>}`.
- **M1** `guided-chat-history-*` styled (role/step header line + middot separator) — "YouSource" → "You · Source".
- **M2** `guided-turn-primary`/`guided-turn-secondary` defined → schema-form buttons styled (44px target restored).
- **Propose secondary buttons** (`guided-propose-edit-btn`/`-secondary-btn`/`-step-actions`) defined → transforms-stage buttons styled.
- **Interpretation-review card** (`guided-interpretation-review-*`) text/layout/opt-out-link/amend-textarea styled (buttons already used `.btn`).
- **m1** login field spacing (`<form>` column gap). *Verified on staging.*
- **m5** `.tutorial-guided-shell` / `.tutorial-sample-loading` defined.
- **L1–L3** primitive port: `.card`/`.card-paper`, `.btn-ghost`, `.alert-banner--warning/--success` ported from `website/tokens/primitives.css`.

Build clean (`tsc + vite` exit 0); touched-area unit tests pass (321/323 — the 2
failures are **pre-existing on HEAD**, see below). After the fixes, the guided
widgets' undefined-class count dropped 42 → ~6 (all decorative wrapper hooks with no
visible effect).

**Fixed in the second pass (operator-approved 2026-06-28):**
- **M5** approvals now render **above** the intent box (`GuidedInterpretationReviews` moved above `.guided-step-chat`). Operator chose *reorder only* — the input is deliberately **not** gated on pending approvals (it never was; only the turn widget is).
- **M6** "Decisions so far" — frontend now de-dups to one row per step (last wins) and drops the widget-type fallback (shows the summary or a neutral "Decided", never "Single select"). *Backend `summary`-per-turn population still recommended* so rows read richer than "Decided".
- **m3** raw-JSON knobs render **read-only** in tutorial mode (`SchemaFormTurn` → `KnobFieldRenderer` `readOnly={isTutorial}`): the prefilled schema stays visible (transparency) but is no longer an editable JSON editor.
- **m4** dual affordance — the manual "Current decision" section is de-emphasised in tutorial with a note ("You don't need to fill this in — pressing Send above builds this step for you"); widgets stay functional (confirm steps still use them).
- **m2** welcome preamble retensed ("This step calls the configured LLM and fetches pages over the network").
- **m6** locked-prompt textarea now sizes to its content (no more clipped URLs).
- **`.visually-hidden` leak** (found in the back-half filmstrip): the class was undefined, so SR-only status text ("… needs review") rendered visibly above every approval card. Aliased onto `.sr-only` in `base.css` — fixes 4 components.
- **Tests updated** for the intentional behaviour changes (`GuidedHistory` fallback→"Decided" + de-dup; stepper 6→5 columns + mobile breakpoint); full touched-area suite green (565/565).

**Remaining (small / needs a separate call):**
- **M6 backend** — emit a human `summary` per completed turn so the recap reads richer than "Decided".
- **m7** `null` source still appears in the first-run picker. Now lower-impact (the picker is de-emphasised by m4 and the learner uses Send); a clean fix is a backend/catalog tutorial-filter rather than a frontend hack — not done.
- **m8** `.tutorial-run-discarded` (one undefined class on the run turn's discarded-rows note) — minor, not yet styled.

**Coverage boundary:** visual evidence covers welcome → source → output (the LLM walk
driver stalls at the output `single_select` — a harness limitation, not a product
bug). The run → audit → graduation tail was assessed by **static cross-check**: those
turn components are CSS-complete except **`.tutorial-run-discarded`** (new m8, minor).
A full back-half filmstrip needs a driver that clicks `single_select` chips (offered).

**Concurrent foreign work (not this review's, left untouched):**
`TutorialTurn7Graduation.tsx` + `.test.tsx` were modified in the working tree by
another session (a change that lands the graduate ON the pipeline they just built
instead of a fresh empty composer — a good UX fix, verified live in the filmstrip).
Its tests transiently failed mid-edit during this review but are **green again** now
that session completed (full touched-area suite 565/565). This review never touched
those files; flagged only so the overlap is on record.

**Mandated gates:** `wardline` (CLAUDE.md trust-boundary gate) is **N/A** — every
change here is presentational frontend TS/CSS (class wiring, layout, copy); none
touches external-input parsing or a taint sink.

---

## Major issues

| # | Issue | Evidence | Recommendation |
|---|-------|----------|----------------|
| **M1** | Chat-history role labels render concatenated: **"YouSource" / "AssistantSource"**. The role span and step span have no separator and both classes are undefined, so they collapse. (SR is unaffected — a separate `visually-hidden` "You said:" prefix carries the role.) | `GuidedChatHistory.tsx:88-93` emits `<span.guided-chat-history-role>You</span><span.guided-chat-history-step>Source</span>`; neither class exists in any app CSS. Screenshot `film-03`. | Define the two classes in `guided.css` as a badge row (role pill + step, with gap), or restructure the markup to a single formatted label. |
| **M2** | **Schema-form action buttons render as raw browser buttons** ("Continue", "Apply recipe", "Build manually", "Clear …") on the Output-required, Recipe, and source-`schema` stages. The buttons' only class is `guided-turn-primary` / `guided-turn-secondary`, which are undefined. Also fails WCAG **2.5.8** (unstyled buttons ≈21px tall < 24px target). | `SchemaFormTurn.tsx:137,147,248`; classes absent from cascade (`guided.css` still defines the *old* `guided-schema-continue-btn`). | Either point the buttons at the existing `.btn`/`.btn-primary` (as `InterpretationReviewTurn` already does), or add `.guided-turn-primary`/`.guided-turn-secondary` to `guided.css`. Prefer `.btn` reuse for one button identity. |
| **M3** | **Mobile: workflow stepper labels wrap one character per line** ("S/o/u/r/c/e", "T/r/a/n/s/f/o/r/m/s"). The stepper is a fixed `repeat(6, minmax(0,1fr))` grid with **no responsive breakpoint**; at ≤~430px the columns are too narrow and `overflow-wrap:anywhere` shatters the labels. It is also a **6-column grid for 5 steps** (empty trailing column). | `guided.css:29-36`. Screenshot `film-04-guided-source-mobile`. | Add a `@media (max-width: ~640px)` rule: collapse to a horizontal scroll row or 2–3 columns / icon-only steps; correct the column count to the step count. |
| **M4** | **Freeform opt-out leaks into the tutorial.** "Always start new sessions in freeform mode" checkbox shows on every tutorial stage, contradicting the tutorial's purpose (graduation owns the default-mode save). The sibling `ExitToFreeformButton` *is* correctly hidden in tutorial. | `ChatPanel.tsx:1434` gates `ExitToFreeformButton` on `!isTutorial`, but **1438 renders `<InlineOptOutCheckbox/>` unconditionally**. Screenshots `film-04/05`. | Gate it: `{!isTutorial && <InlineOptOutCheckbox/>}`. One-line fix, matches the line above. |
| **M5** | **"Approvals appear below the input that gates them."** The LLM approval/alert cards (`GuidedInterpretationReviews`) render inside `.guided-current-decision`, which is *below* the "Describe what you want" box. The user must approve before proceeding, so the approvals should sit **above** the input. (Operator-directed.) | `ChatPanel.tsx:1371` step-chat first, `1418` reviews second. | Move `<GuidedInterpretationReviews>` (and any pending alerts) above the `.guided-step-chat` section; consider disabling the chat input while approvals pend. |
| **M6** | **"Decisions so far" recap is meaningless / duplicated.** Rows show the *widget type* ("Single select", "Schema form") instead of the decision, and the same step appears multiple times ("Source — Single select", "Source — Schema form", "Source — Configured: json"). | `GuidedHistory.tsx:76` `turn.summary ?? TURN_TYPE_LABELS[...]` — falls back to widget type when backend `summary` is null; backend emits null summaries + multiple turns per step. Screenshots `film-03/05`. | Backend: populate a human `summary` per completed turn. Frontend: drop the widget-type fallback (show nothing or "—" rather than "Single select"); de-duplicate to one row per step. |

---

## Minor issues

| # | Issue | Evidence | Recommendation |
|---|-------|----------|----------------|
| m1 | **Login fields cramped** — Username/Password inputs collide with no vertical gap; "Sign in" tight beneath. The `Input` primitive's wrapper `<div>` has no margin and `LoginPage`'s `<form>` has no gap. | `Input.tsx:38-48` (bare `<div>`); `LoginPage.tsx:173-202`. Screenshot `05-login-desktop`. | Add a field gap (form `display:flex;flex-direction:column;gap` or a `.field` wrapper margin). |
| m2 | **Welcome preamble has wrong tense/placement** — "This is calling the configured LLM and fetching the URLs the composer chose." shows on the static welcome screen before anything runs. | `tutorial.css:124-129` `.tutorial-preamble` ("above the Turn 4 run indicator"), rendered on welcome. Screenshot `01-welcome`. | Reword to future/explanatory ("When you start, ELSPETH will call the configured LLM…") or move to the run turn. |
| m3 | **Raw JSON schema editor exposed to a passive learner** — the source `schema` stage shows a `{ "mode":"observed", ... }` JSON textarea + "schema: {mode: observed}" hint. | `SchemaFormTurn.tsx` json-object branch; screenshot `film-03`. | For the passive tutorial, prefill+lock or hide the raw-JSON knob; keep the worked-example caveat copy. |
| m4 | **Dual competing affordance** — every tutorial stage shows the locked "Describe what you want" chat (the intended Send path) *and* the fully interactive manual widget (chips/form) below. A learner can stray onto the manual path. | `ChatPanel.tsx:1371-1427` renders both unconditionally in guided. Screenshots `film-03/04/05`. | Product decision: in tutorial, visually de-emphasise or disable the manual widget so "Send" is the single obvious action. |
| m5 | **Unstyled tutorial wrapper classes** — `.tutorial-guided-shell` and `.tutorial-sample-loading` are referenced but undefined (no visible break today; the section inherits the centered column). | `TutorialGuidedShell.tsx:175,206`; absent from `tutorial.css`. | Add the rules (or reuse `.tutorial-shell` framing for the guided wrapper) so the shell has intentional padding/width. |
| m6 | **Locked prompt textarea clips** multi-line content (the 3 sample URLs) without an obvious scroll affordance. | `ChatInput` fixed height; screenshots `film-03/05`. | Auto-grow to content (capped) or show it's scrollable; the locked prompt is informational, not edited. |
| m7 | **`null` source exposed in first-run picker** — "null — A source that yields no rows." appears among the source options shown to a brand-new user. | Screenshot `03-guided-source`. | Catalog/UX: consider hiding `null` (and other degenerate sources) from the first-run/tutorial picker. |

---

## Latent — design-system wiring debt (not visible yet; will break the next adopter)

The `f2ff4e912` fix ported only the **input** block of the `components/ui` primitive
CSS from the standalone `website/` tree into the app bundle. The rest of the library's
classes remain undefined in the app:

| # | Primitive | Undefined class(es) | Impact |
|---|-----------|---------------------|--------|
| L1 | `Card` / `CardHeader` | `.card`, `.card-paper` | Renders as an unstyled `<div>` the moment any app screen uses `<Card>`. |
| L2 | `Button variant="ghost"` | `.btn-ghost` | Ghost variant silently degrades (no class → base `.btn`). |
| L3 | `AlertBanner` tones | `.alert-banner--warning`, `.alert-banner--success` | Only `--info` is defined (in `header.css`); warning/success fall back to the base (error-styled) banner. |

Currently only `LoginPage` consumes primitives, and only `Input` / `Button[primary]`
/ `AlertBanner[error]` — all styled — so the blast radius is zero **today**.
Recommendation: port the remaining primitive blocks (`.card*`, `.btn-ghost`,
`.alert-banner--warning/success`) into `shared.css` to finish the migration and add
an `inputPrimitives.test.ts`-style guard covering them.

---

## What's genuinely good (keep)

- **Token discipline:** every tutorial.css / guided.css value resolves to a defined token (verified); no hardcoded colours.
- **Accessibility baseline is strong:** visible `:focus-visible` rings throughout (2.4.7 ✓), `var(--size-control)` 44px targets on guided buttons (2.5.8 ✓ — except the M2 unstyled ones), `@media (prefers-reduced-motion: reduce)` coverage, sr-only "Step N of M" + role-prefix announcements, `role="region"`/`role="log"` landmarks, exhaustive ARIA on the interpretation-review widget.
- **Welcome bookend & SENSE/DECIDE/ACT cards** are clean and responsive (desktop + mobile both good).
- **AI trust posture:** the per-stage interpretation-review gate (surfacing LLM assumptions to approve/amend) is a model calibration/grounding pattern; "Assistant said:" framing keeps model output legible (undercut visually by M1).

## WCAG 2.2 AA quick verdict
- 1.4.3 Contrast: Pass (token-based; colorContrast.test.ts guards).
- 2.1.1 / 2.4.7 Keyboard + Focus visible: Pass.
- 2.5.8 Target size: **Fail on M2** (unstyled schema-form buttons); pass elsewhere.
- 4.1.2 Name/Role/Value: Pass.
- 3.2.6 Consistent Help: Pass (Skip/▾ consistent).

## Remediation order
1. **Quick CSS/markup wins (low risk, ship first):** M4 (gate checkbox), M1 (role-label CSS), M2 (button class → `.btn`), M3 (mobile stepper media query), m1 (login gap), L1–L3 (finish primitive port).
2. **Operator-directed IA:** M5 (move approvals above the input).
3. **Product decisions (confirm first):** m3 (JSON knob in tutorial), m4 (dual affordance), m6/m7, M6 backend summary.
