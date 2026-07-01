# Guided Mode Reframe — Conversational Builder + Live-Graph Verification

- **Date:** 2026-06-30
- **Target:** `release/0.7.0` — the **last component** of the web UX reframe (operator-confirmed; not a next-cycle epic).
- **Status:** spec / design (start). Decisions in §2 are locked; §6 slices are the implementation plan.
- **Sizing:** *not* a major refactor. With sample-execution parked (§7) and the verification surface being the **already-existing, already-store-coupled** graph (§4), this is predominantly **reuse + reroute + copy**. The only genuinely new design work is back-navigation semantics (Slice E). The "from-the-studs / major refactor" framing applied to the *earlier* proposal (sample-execution previews + a big-bang flow rewrite), both of which are out of scope here.
- **Provenance:** 2026-06-29/30 first-principles UX eval of `src/elspeth/web` guided mode (operator-prompted: "the UX is actually not very good"). Companion to `docs-archive/2026-06-29-composer-design-review.md`.

---

## 1. Problem (condensed verdict)

Guided mode is the **novice on-ramp** (default for new sessions; experts opt into freeform). Its job-to-be-done: take *data + goal* → a runnable pipeline **without** the user learning plugins or YAML. It nails **translation** (the chat→build path is the crown jewel) but botches **elicitation** and **verification** because it **can't decide what it is** — it stacks three competing interaction models in one column:

- a **wizard** (stepper + confirm widgets),
- a **chat** (per-step "describe what you want" box), and
- a **review surface** (read-only summaries, "current decision" = the LLM's prose).

Consequences, all verified in code:
- **Two input loci** (chat box *and* widget wizard) with opposite philosophies — the user never knows whether to type, pick, or confirm, nor where the result appears.
- **No back-navigation** — `BACK` is a summary string with no state-machine transition.
- **Steering is all secondary** — `SchemaFormTurn` opens read-only (Edit hidden); `propose_chain` Reject only re-rolls the LLM; no take-over.
- **A wire-stage advisor dead-end** (filed as **`elspeth-7b0f75e90e`**, P1): a FLAGGED sign-off re-emits an identical Confirm with no reason, no escape, and silently burns the per-compose pass budget.
- **The tell:** the team special-cased the tutorial composer to the *top* to make it teachable. A layout you must invert to teach a novice is failing that novice. (That special case was removed 2026-06-30, commit `4ed6a4b37` — tutorial composer now docks at the bottom like the live composer.)

---

## 2. Locked decisions (this session)

### D1 — Verification surface = the pipeline **GRAPH**, not sample output. (Sample-execution preview is PARKED — see §7.)
**Rationale (a domain invariant, not a cost trade):** in many real deployments the **data source is consumable / non-replayable** — RNG draws, satellite telemetry, streaming feeds. Reading "sample rows" to preview *consumes data that is then lost to the real run*: **test rows represent lost data.** This is also *why the system logs failures so aggressively* — when a row seizes, the failure log is the record of "this is the data that was lost, and we know where it was lost." It follows that the correct moment to verify is **before** you spend irreplaceable rows, against the pipeline's **structure**, not by trial-running real data. Structural verification is therefore the *only correct* verification surface for this data model — not merely the cheaper one.

The graph already exists and already carries the legibility we need (see §4): a React Flow DAG over the live `CompositionState` with **per-node validation markers** (valid / warning / error).

### D2 — Target `release/0.7.0`, last component of the web UX reframe.
Land in **incremental, independently shippable slices** (§6) on the release train rather than as one big-bang change. Each slice is reversible.

### D3 — Demote the widget wizard to an assistant-offered **fallback**.
Conversational chat is the **primary** (and only default) input locus. The plugin-pick / schema-form widgets become a fallback the assistant offers **only when it cannot resolve intent** — and the place expert knobs live. (Operator agreed in the verdict.)

---

## 3. Target interaction model

One input locus, one verification surface, plain-language throughout:

1. **Input — conversational (primary).** A single chat box, docked at the bottom (consistent with `4ed6a4b37`). The user describes data + goal in their own terms. The assistant elicits intent in user terms, never asks them to "pick from a list you could fill yourself" (this is what `base.md` already tells the model to do — the reframe makes the *UI* honour it).
2. **Translate — the crown jewel (unchanged).** The chat→build path composes into `CompositionState` via the existing guided solver + state machine.
3. **Verify — the live graph.** A persistent graph panel (reusing `GraphView` / `GraphMiniView`) renders the DAG as it is built, annotated with per-node validation status + plain-language node labels. This is the novice's judgement surface: "source → rate each row → write CSV — yes, that's what I wanted."
4. **Steer — promoted from secondary.** Back-navigation, edit-a-node, and reject/redo are first-class, not buried. When the assistant can't resolve intent, it *offers* the wizard/form fallback (D3).
5. **Sign-off — honest.** The wire-stage advisor outcome is surfaced with reasons + an escape (the `elspeth-7b0f75e90e` fix), not a silent dead-end.

---

## 4. What we reuse (crown jewels — do NOT rebuild)

| Asset | Location | Role in the reframe |
|---|---|---|
| chat→build path | `composer/guided/{chat_solver,chain_solver}.py`, state machine | The translate step. Untouched behaviourally; must not be destabilised. |
| `GraphView` | `frontend/.../components/inspector/GraphView.tsx` | React Flow (`@xyflow/react`) + dagre DAG over `CompositionState`, read-only, pan/zoom, **per-node validation markers**. The verification surface. |
| `GraphMiniView` / `GraphModal` | `frontend/.../components/sidebar/` | Compact lane view + modal expand; store-coupled to `useSessionStore.compositionState`. Candidate for the docked guided panel. |
| `preview_pipeline` (dry-run) | `composer/tools/generation.py:1685` (`_execute_preview_pipeline`) | Returns validation, edge/semantic contracts, proof diagnostics, node/output overview **without executing**. **Correction (2026-06-30): the live frontend graph markers are NOT fed by `preview_pipeline`** — they come from `POST /api/sessions/{id}/validate` (`validate_pipeline`, metadata-only), via the mode-agnostic version-keyed auto-validate subscription. `preview_pipeline` is a separate composer/MCP discovery surface (its `proof_diagnostics` reads an uploaded blob *only* when a source has `options["blob_ref"]`; a consumable source has none → zero reads). Both paths are row-free, so D1 holds. |
| guided state machine + turn protocol | `composer/guided/{state_machine,protocol,steps}.py` | The orchestration spine. Reframe changes *flow + which affordances are primary*, not the spine. |

**Key economic finding:** the verification surface is **store-coupled and already exists**. Guided mode already composes into the same `useSessionStore.compositionState` the graph renders. Surfacing the graph in guided mode is mostly *mounting an existing component in a new column*, not building a visualisation. This collapses the reframe's risk to **flow + layout + copy**.

---

## 5. What changes (frontend-led)

- **5.1 Layout.** Primary chat docked bottom + a persistent live-graph panel in the guided column (mount `GraphView` or `GraphMiniView`+`GraphModal`, reading the existing store). The graph replaces the "current decision = LLM prose" review block as the canonical "what I built" surface. **Mount point (2026-06-30 review): inside `ChatPanel`'s `.guided-scroll` (~`ChatPanel.tsx:1455-1505`) — the one surface shared by live-guided AND the tutorial, which renders `ChatPanel` *without* the `SideRail` where the live `GraphMiniView`+`GraphModal` already run. Markers auto-populate from `validationResult` (version-keyed auto-validate), already firing in guided — no new data plumbing.**
- **5.2 Demote the wizard (D3).** `single_select` / `multi_select_with_custom` / `schema_form` / `propose_chain` widgets become assistant-*offered* fallbacks, not the default path. Keep `SchemaFormTurn`'s summary-first + Edit-toggle for the fallback/expert path.
- **5.3 Back-navigation.** Add real BACK transitions to the state machine (today `BACK` is a no-op summary string). Define per-step semantics; prefer **"fork/redo from here"** over destructive mutate-in-place when a decided step has downstream dependents.
- **5.4 Wire-stage dead-end** — implement **`elspeth-7b0f75e90e`**: surface `advisor_findings` / `signoff_outcome` (declared-but-unused at `types/guided.ts:388-389`), add an "Ask advisor" (budget-aware) action and a "complete without sign-off" escape with explanation, and stop the silent budget burn on `BLOCKED_FLAGGED`.
- **5.5 Silent-compute indicator.** Reuse `ComposingIndicator` in the guided branch (today it renders only in the freeform branch); guided currently just disables the box while the model builds.
- **5.6 Kill vestigial vocabulary — CLASSIFIED (corrected 2026-06-30 by evidence review; full map in the Slice A plan).** Only **two** items are genuinely safe to remove — non-persisted, no DB CHECK, **no sessions-DB wipe**: the frontend-only `interpretation_review` `TurnType` member + its self-labelled dead `null` dispatch (`GuidedTurn.tsx`), and the `chosen:["reject"]`→501 branch (`_helpers.py`, raised pre-mutation). The rest are **retained, not vestigial**, each for a concrete reason: `step_2_5_recipe_match`/`recipe_offer` land in **persisted audit rows**, shift `step_index` ordering, and back a 409 orphan-recovery guard (operator-gated keep — see `project_passive_tutorial_e2e_green`); `coaching`/`recipe_match` profile flags live in the **persisted `GuidedSession.profile` blob** (removal forces a `GUIDED_SESSION_SCHEMA_VERSION`+`SESSION_SCHEMA_EPOCH` bump + a `data/sessions.db` wipe per the `entry_seed` precedent — out of Slice A scope); `inspect_and_confirm` has test-only *emission* but is the documented **step-1→step-2 transition turn** (`state_machine.py`) — structural spine, not vocab. `BACK` belongs to Slice E (back-nav), not cleanup. Net: **Slice A needs NO DB wipe.** (The `SlotType / guided.ts mirror` hook validates only `RecipeSlotInput.slot_type` — a no-op gate for the `TurnType` edit.)
- **5.7 Tutorial honesty.** The tutorial must not be *easier* than the live surface it teaches (today it hides Edit/Reject/Ask-advisor and bypasses the advisor gate — the inverse of onboarding). Align it with the reframed model. (Bottom-dock already landed in `4ed6a4b37`.)

---

## 6. Slices (land incrementally on `release/0.7.0`)

Ordered so each is independently shippable, reversible, and low-blast-radius first. Sizing per slice is annotated — most are reuse/reroute/copy, not new architecture:

- **Slice A — surface cleanup (S, low risk).** §5.6 vestigial-vocab removal + §5.5 silent-compute indicator (reuse `ComposingIndicator`). No happy-path behaviour change; shrinks the surface.
- **Slice B — wire-stage dead-end (`elspeth-7b0f75e90e`) (M).** §5.4. Direction-independent, highest user-harm; ships even if later slices slip. Mostly rendering data the backend already emits.
- **Slice C — live graph as verification surface (S–M).** §5.1 (graph half). Read-only **reuse** of `GraphView`/`GraphMiniView`; additive panel, no flow change yet.
- **Slice D — conversational-primary + wizard-as-fallback (M).** §5.1 (input half) + §5.2. The behavioural heart, but a **reroute** of existing turn affordances, not new widgets.
- **Slice E — back-nav + steering (M–L, the only genuinely new design).** §5.3. New state-machine transitions + fork-from-here semantics; spec this before building.
- **Slice F — tutorial alignment + honesty (S).** §5.7.

---

## 7. Non-goals / parked

- **Sample-execution / live-output preview — PARKED (D1).** Consumable/non-replayable sources mean test rows = lost data; aggressive failure logging (not preview) is the audit answer for "what was lost, where." If a *replayable-source* subclass is ever introduced, revisit as an **additive, opt-in** behaviour gated behind the spend guard, SSRF/egress controls, and the fanout-guard fail-open fix (`elspeth-35bb0ca6f6`) — explicitly out of scope here.
- **Backend advisor-checkpoint redesign — separate (`elspeth-dac6602a2b`).** That feature reshapes the *backend* sign-off (deterministic checkpoints, field-contract checker, two-tier models). This spec **consumes** whatever wire-stage outcome shape it produces; §5.4 fixes the *frontend* dead-end against today's shape and tracks any new shape.

---

## 8. Risks / open questions

- **Graph legibility for true novices.** Is a DAG self-explanatory to a non-technical operator? Likely needs plain-language node labels + a one-line "what this pipeline does" gloss beside it, not just the raw graph.
- **Regression risk to the crown jewel.** Making chat the primary locus must not destabilise the working chat→build path. Treat the existing path as a fixed contract.
- **Back-nav semantics with downstream dependents.** Re-opening step 1 after step 3 is built is non-trivial; "fork from here" may be safer than mutate-in-place. Needs explicit state-machine design before Slice E.
- **Tutorial parity vs. brutality.** Making the tutorial honest about the advisor gate without making first-run onboarding punishing.

---

## 9. Verification (per slice)

- `vitest` + `eslint` + `stylelint` + production build green per slice.
- Staging tutorial e2e (Playwright vs live staging) after slices that touch the tutorial.
- **D1 guard (manual):** compose a consumable-source pipeline and confirm **no source row is read for preview** — the graph is built from `preview_pipeline` (dry-run) + `CompositionState` only.
