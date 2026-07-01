# Slice C — Live Graph as the Verification Surface (Implementation Plan) — v2

> **For Claude:** REQUIRED SUB-SKILL: superpowers:executing-plans. Primarily frontend (vitest). Mount/layout work is reuse — no forced red-green; the gloss + validation-summary helpers ARE red-green-able. **v2 incorporates plan-review findings (arch CRITICAL + quality MAJOR + reality).**

**Goal:** Make the pipeline graph the novice's verification surface in guided mode by **mounting existing store-coupled components** in the column shared by live-guided and the tutorial, plus two small net-new legibility pieces (a one-line gloss + a plain-language validation summary). Additive; no flow change. **D1: zero source rows.**

**Issue:** `elspeth-aabb519a49` (feature; blocks D).

### Corrections from review (the v1 plan was wrong on these — do NOT regress to it)
1. **`<GraphModal/>` at `App.tsx:408` is UNCONDITIONAL** (outside the `showTutorial` ternary that closes `:405`) — it is already mounted for BOTH the tutorial and non-tutorial trees. **Do NOT add a second `GraphModal`.** A tutorial `GraphMiniView` click dispatches `OPEN_GRAPH_MODAL_EVENT`; the existing App-root modal catches it.
2. **`GraphMiniView` has NO per-node markers** — it renders aggregate lanes ("1 src", "3 tx", "sink"), `aria-hidden`, no `validationResult` coupling (`GraphMiniView.tsx:64-133`). Per-node markers live only in `GraphView` (inside the modal, `GraphView.tsx:373/421-437`). So the column gets its "is it OK?" signal from a **new small `PipelineValidationSummary`** reading the same `validationResult` store; full per-node markers are available **on expand** (the modal). The C3 marker test targets the modal `GraphView` + the summary — NOT `GraphMiniView`. **Do not weaken/delete the marker assertion to get green.**
3. **`GraphMiniView` already renders in the live-guided SideRail** (`App.tsx:391`, ungated). To avoid a duplicate thumbnail, the **column** `GraphMiniView` is gated to the tutorial (which has no rail); live-guided keeps using the rail thumbnail. The gloss + validation summary are added to the column in BOTH surfaces.

---

## Task C1 — Mount the guided verification panel in `.guided-scroll`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (guided branch `:1358-1518`; the `guided-current-decision` block `:1455-1505`)
- Reuse (unchanged): `GraphMiniView` (`sidebar/GraphMiniView.tsx`; populated state is a `<button aria-label="Pipeline graph (click to expand)">` `:51-60`; empty state `data-testid="graph-mini-empty"`), App-root `GraphModal` (`App.tsx:408`).
- New (Tasks C2): `PipelineGloss`, `PipelineValidationSummary`.

**Design — the column shows the verification SIGNAL; the full marker graph is one click away:**
```tsx
// inside .guided-scroll, guided branch of ChatPanel
<section className="guided-graph-panel" aria-label="Pipeline so far">
  <PipelineGloss compositionState={compositionState} />          {/* C2 */}
  <PipelineValidationSummary />                                   {/* C2b — reads validationResult */}
  {isTutorial && <GraphMiniView />}   {/* tutorial has no SideRail; live-guided uses the rail thumbnail */}
</section>
{/* NO new GraphModal — the App-root one (App.tsx:408) is unconditional and serves both surfaces */}
```
Retain the rationale prose (demote, don't delete): the graph/summary is now the canonical "what I built"; the prose becomes supporting context.

**Step 1 — mount + render test.** Add the panel. Tests: in the guided branch the panel + gloss + summary render; in the tutorial branch a `GraphMiniView` (query by `aria-label="Pipeline graph (click to expand)"`, populated state) is present; in live-guided the column does NOT add a second `GraphMiniView` (rely on the rail). No new `GraphModal` is mounted.

Run: `npm run test -- ChatPanel && npm run build`
Expected: PASS; build clean.

**Commit:** `feat(web/guided): mount the guided verification panel (gloss + validation summary) in .guided-scroll` (+ Co-Authored-By).

**DoD:**
- [ ] Panel renders in the guided column for BOTH live-guided and tutorial (gloss + validation summary in both; thumbnail in tutorial)
- [ ] **No second `GraphModal` mounted** (App-root one serves both; expand works from the rail in live-guided and the column thumbnail in tutorial)
- [ ] shared `GraphView`/`GraphMiniView` rendering UNCHANGED (live composer unaffected)
- [ ] rationale prose retained (demoted, not deleted)

---

## Task C2 — Plain-language gloss + validation summary (net-new, minimal; justified by spec §8)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/guided/PipelineGloss.tsx` + pure helper `pipelineGloss.ts` (derives one sentence from `compositionState`)
- Create: `src/elspeth/web/frontend/src/components/chat/guided/PipelineValidationSummary.tsx` (reads `useExecutionStore.validationResult` AND `useSessionStore.compositionState`; maps each finding's `component_id` → plain node name via the SAME node→plain-phrase map the gloss uses; renders plain-language status)
- Tests: `pipelineGloss.test.ts`, `PipelineValidationSummary.test.tsx`
- a11y: add `PipelineGloss` + `PipelineValidationSummary` to the FIXED list at `src/test/a11y/components.a11y.test.tsx`

**Gloss design:** "This pipeline reads your data, rates each row, and writes a CSV" — derived from the ordered nodes (`node_type` + plugin → plain phrase). Lives in the guided wrapper only; do NOT humanize labels inside the shared `GraphView` (would bleed into the live composer). Per-node plain labels are an explicit follow-up, not this slice.

**Validation summary design:** read `useExecutionStore.validationResult` (the same store the modal markers use; auto-populated mode-agnostically, `subscriptions.ts:264-276`). Render plain status: "✓ Looks good" / "⚠ 1 warning: 'rate each row' — {message}" / "✕ Error in 'write CSV': {message}". This is the in-column verification signal that compensates for `GraphMiniView` lacking markers.

**Plain node names need `compositionState` (quality re-review):** `validationResult.errors/warnings` carry only `component_id: string | null` + `message` (`types/index.ts:342/388`) — **no human node name**. So wire `compositionState` into `PipelineValidationSummary` and reuse the gloss's `component_id`→plain-phrase mapping to render "'rate each row'" rather than the raw id. `PipelineValidationSummary.test.tsx` MUST assert the PLAIN node name, not the raw `component_id` (and a finding whose `component_id` has no composition match falls back to a generic phrase, never a crash).

**N3 (arch re-review) — register overlap, decided:** in live-guided the rail `SideRailValidationBanner` already shows validation status in a *technical* register. `PipelineValidationSummary` is the *plain-language, novice* register and is intentionally kept in-column for BOTH surfaces — the guided column is the novice's focal point, so the overlap is accepted (rail = technical detail; column = plain-language signal). Suppressing the rail banner in guided would be a `Layout`/`SideRail` gating change (more blast radius) — deferred, not in this slice.

**Step 1 — failing helper/summary tests (RED).**
- `pipelineGloss`: source→llm→csv ⇒ expected sentence; empty/partial composition ⇒ safe fallback ("Your pipeline is taking shape…").
- `PipelineValidationSummary`: valid `validationResult` ⇒ "Looks good"; with a warning ⇒ the warning text + the PLAIN node name (mapped from `component_id` via `compositionState`); with an error ⇒ the error text + plain node name; a finding with an unmappable `component_id` ⇒ generic fallback phrase (no crash); empty ⇒ neutral.

Run: `npm run test -- pipelineGloss PipelineValidationSummary`
Expected: FAIL — components/helper don't exist.

**Step 2 — implement (GREEN).** Pure functions over `compositionState` / `validationResult`; no store mutation, no API calls.

Run: `npm run test -- pipelineGloss PipelineValidationSummary ChatPanel`
Expected: PASS.

**Step 3 — a11y + commit.** Run `npm run test -- components.a11y`; commit `feat(web/guided): plain-language gloss + validation summary above the guided graph` (+ Co-Authored-By).

**DoD:**
- [ ] gloss renders with a safe fallback; pure helper unit-tested
- [ ] validation summary reflects `validationResult` (valid/warning/error) in plain language; unit-tested
- [ ] both added to the a11y suite
- [ ] no change to shared `GraphView` labels

---

## Task C3 — D1 guard + tutorial-parity verification

**Files:**
- Test: a focused vitest for D1 (store-only, no source-data fetch) + tutorial parity
- Doc: record the manual D1 check (spec §9) in the issue

**D1 (load-bearing):** the panel builds from `compositionState` + `validationResult` (store) — **zero source rows**. Confirmed: `validate_pipeline` is metadata-only (`src/elspeth/web/execution/validation.py:709`, "validate checks metadata only" `:749`); `preview_pipeline`'s blob read is gated on `options["blob_ref"]` (consumable source → none → zero reads) and doesn't feed the frontend graph anyway.

**Step 1 — D1 store-only test.** Mount the guided panel with a mocked store (compositionState + validationResult). Assert it renders gloss + summary (+ tutorial thumbnail) **without calling source-DATA endpoints**. Spy scope: source-data APIs only (preview / upload / source-rows) — **NOT** a blanket `api.*` (the mode-agnostic auto-validate fires `api.validate`, which is metadata-only and D1-safe; a blanket spy would false-fail).

**Step 2 — tutorial parity test.** Render the tutorial guided surface; assert (a) `PipelineValidationSummary` reflects `validationResult` (the in-column signal — auto-validate is version-keyed/mode-agnostic, so it fires in the tutorial), and (b) the `GraphMiniView` thumbnail is present and its click dispatches `OPEN_GRAPH_MODAL_EVENT` (caught by the App-root modal). The per-node **marker** assertion targets the modal `GraphView` (or is exercised via the existing `GraphView.test.tsx` marker coverage) — NOT `GraphMiniView`. **If a marker test against the chosen surface fails, that is a true signal — fix the surface, do not delete the assertion.**

Run: `npm run test -- ChatPanel pipelineGloss PipelineValidationSummary`
Expected: PASS.

**Step 3 — manual D1 verification (record).** Per spec §9: compose a consumable-source pipeline; confirm no source row is read for the panel (built from `validationResult` + `compositionState` only). Record pass/fail in `elspeth-aabb519a49`.

**Step 4 — commit (if separate test file).** `test(web/guided): D1 store-only panel + tutorial validation-signal parity` (+ Co-Authored-By).

**DoD:**
- [ ] store-only test green; spy proves no source-DATA fetch (auto-validate allowed)
- [ ] tutorial shows the gloss + validation summary (signal parity) + a working expand thumbnail
- [ ] manual D1 check performed and recorded

---

## Slice C — overall Definition of Done

- [ ] C1+C2+C3 committed
- [ ] Frontend gate green (vitest/eslint/stylelint/build)
- [ ] Guided column shows gloss + plain validation status in BOTH surfaces; full per-node markers reachable via expand (App-root modal); thumbnail in tutorial, rail thumbnail in live-guided (no duplicate)
- [ ] **D1 explicit:** panel renders from store only, zero source rows (test scoped to source-data APIs + manual check)
- [ ] Live composer unaffected (shared `GraphView`/`GraphMiniView` rendering unchanged; no second modal)
- [ ] No guided flow change (additive surface; conversational-primary reroute is Slice D)
- [ ] `elspeth-aabb519a49` closed with the commit SHAs; unblocks D
