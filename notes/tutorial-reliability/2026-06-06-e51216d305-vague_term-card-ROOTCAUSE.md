# elspeth-e51216d305 — root cause: it is NOT a harness gap, it is a backend order-dependent bug

**Date:** 2026-06-06 · **Investigator:** claude-debug · **Method:** systematic-debugging (primary-source: failing-run Playwright snapshots + backend source)

## TL;DR (the issue is misdiagnosed)

The issue says: *"Tutorial-reliability harness can't drive the vague_term review card (amend UI)… Make the accept loop resolve vague_term cards… NOT a product bug."*

**Every load-bearing claim in that framing is wrong:**

1. The harness **already** drives/accepts the vague_term card. Its accept button `aria-label` is
   `"Accept the LLM's interpretation of <term>"`, which **matches** the harness locator
   `getByRole("button", { name: /^Accept /i })` (verified empirically, both apostrophe variants).
2. At the moment of timeout the vague_term card is **already gone (resolved)**. The card that is
   **stuck** is the **LLM prompt template** card, showing a **"Stale review — This prompt template has
   already changed. Reload the session…"** alert (Playwright a11y snapshot, runs 3 & 6 identical).
3. It **is** a product bug. A real tutorial user hitting the same card-resolution order is **also**
   permanently stuck (see "Why reload can't save it").

The harness is doing its job: it classifies this `tutorial_fault / frontend-state-machine` and
writes the record. It is reporting a **true** fault that lowers the *real* tutorial pass rate. A
harness "fix" that makes this run green would be cache-as-fakery (cf. elspeth-63001e4134) and is the
exact thing "DON'T thrash the harness accept-loop" warns against.

## Evidence

- Failing runs: `.harness-results/fix-verify2-2026-06-06/run-03.json`, `run-06.json`
  (`outcome=tutorial_fault`, `fault=frontend-state-machine`, `error="assumptions never all became
  acceptable (continue stayed disabled)"`, `turn_reached=2`).
- Playwright a11y snapshot at failure (`test-results/…run-3…/error-context.md`, run-6 identical):
  - `1 assumption to review`
  - `region "LLM prompt template"` with `alert: Stale review / This prompt template has already changed.`
  - `button "Accept LLM prompt template" [enabled]` (clicking it re-issues the 422 forever)
  - `button "Looks good" [disabled]`
- Control: `run-01.json` **passed** with a vague_term present (5 cards) — so "vague_term appears" is
  **not** the predictor. **Resolution order** is.

## Mechanism (backend)

The composer stages, on one LLM node, both:
- a `vague_term` review ("visually impressive"), and
- an `llm_prompt_template` review of the rendered prompt that contains that term.

Accept loop resolves cards in DOM order (= `created_at`, `TutorialTurn2bShowBuilt`). Two paths:

- **prompt-template accepted BEFORE vague_term** → both succeed (run-01 path).
- **vague_term accepted BEFORE prompt-template** → prompt-template card is **bricked** (run-03/06 path):

  1. `resolve_interpretation_event` (`web/sessions/service.py:3025`) for the vague_term calls
     `_resolve_vague_term` → `_patch_llm_transform_prompt` (`service.py:854,862-863`), which
     **rewrites** `options.prompt_template` (substitutes the term).
  2. The pending `llm_prompt_template` event is **untouched** — `resolve_interpretation_event` only
     mutates the one event it resolves; it never regenerates/supersedes siblings. Its `llm_draft`
     (the *pre-bake* rendered prompt) is now frozen and stale.
  3. The frontend store (`stores/interpretationEventsStore.ts` `resolveEvent`) updates
     **incrementally** — removes only the resolved event, never re-fetches siblings — so the stale
     card keeps rendering.
  4. Accepting it hits the gate in **`_resolve_prompt_template_review` (`service.py:1092`)**:

     ```python
     if accepted_value != prompt_template:          # accepted_value = frozen llm_draft (pre-bake)
         raise InterpretationPlaceholderConsumedError # prompt_template = live (post-bake) → never equal
     ```

     → 422 `interpretation_placeholder_unavailable` → "Stale review". **Permanent**: the frozen draft
     can never again equal the post-bake template.

### This contradicts the code's own design intent

`_resolve_prompt_template_review` (`service.py:1104-1112`) states the prompt-template review approves
the prompt **skeleton/structure**, anchored to `prompt_structure_hash_from_options(...)`, and is
**"invariant under vague-term resolution (which rewrites the rendered prompt)."** But the **gate** at
line 1092 uses **full-text equality of the rendered prompt**, which is *not* invariant under
vague-term resolution. The attestation anchor (structure hash) and the acceptance gate (rendered
full-text) disagree. **That inconsistency is the root cause.**

## (a)/(b)/(c) determination — settled from source

After a sibling vague_term bake the prompt-template event is **(c) pending-but-unacceptable**, frozen
permanently. It is **not** regenerated and **not** superseded. Therefore **refresh/reload-based
recovery is futile** (re-fetch returns the same stale `llm_draft`; accept → same 422).

### Why reload can't save it (and why a real user is stuck too)

`HelloWorldTutorial` holds its step in a client `useReducer` starting at `"welcome"` with **no
persistence/restore on mount**. `page.reload()` (the product's own "Reload the session" advice)
resets the tutorial to Turn 1 and discards `showBuilt` state. In `TutorialTurn2bShowBuilt`,
`showOptOut={false}` (no opt-out escape) and "Edit prompt" dumps back to Describe. So a real user who
resolves the vague_term before the prompt-template has **no recovery** inside the tutorial.

## Recommended dispositions (operator decides scope)

- **Reframe/close e51216d305** — it is not a harness gap. Do **not** add a refresh/reorder band-aid;
  refresh is futile (c) and reorder is fragile and hides a real user-facing fault.
- **File a backend product bug** (the real fix). Cleanest candidate: make the line-1092 acceptance
  gate validate against the prompt **structure/skeleton** (`prompt_structure_hash_from_options`) — the
  design's own invariant — instead of full rendered-text equality, so a vague-term bake no longer
  bricks the prompt-template review. Alternatives: regenerate the prompt-template event's `llm_draft`
  on bake; or model the dependency (don't surface the prompt-template review as an independent peer
  while its slot vague-terms are pending). All three are real backend changes with audit implications
  → operator scoping.

## UPDATE 2026-06-06 — backend fix committed (22168c227) is PARTIAL; staging found Case B

Operator chose "fix backend now". Landed the skeleton-hash gate (commit 22168c227,
`_resolve_prompt_template_review` compares surfacing vs live `prompt_structure_hash`
instead of rendered-text equality). Unit-proven (RED→GREEN). Then ran a staging
verification battery (8 runs, reusing the valid saved token, globalSetup skipped) vs
the restarted staging (service ActiveEnter 22:35:20 > commit 22:29:05, so fix WAS live).

Result — the fix is PARTIAL. Two distinct shapes of the vague_term + prompt-template
interaction exist; the composer is non-deterministic about which it produces:

- **Case A** (slot already wired into `prompt_template_parts` when the prompt-template
  review surfaced): a vague-term *value* bake changes the rendered prompt but NOT the
  skeleton. Old gate bricked (text differs); new gate accepts (skeleton equal). The fix
  flips this. **Unit-proven only** — passing staging runs retain no Playwright trace
  (`trace: retain-on-failure`), so I could NOT confirm the fix flipped any *real* run
  (runs 01/02 cleared Turn 2b but could be lucky order A1, not brick-flipping A2).

- **Case B** (run-05, staging, vague_term "visually impressive"): the composer surfaced
  the prompt-template review against skeleton **S1** (frozen `llm_draft` = the short
  prompt ending "...rating, justification.", NO slot — confirmed from the trace's
  interpretation-events list, event `75a38eee…`, surfacing state `7ab564a9…`). It THEN
  grew the skeleton to **S2** by adding fixed framing text `"\n\nOverall visual polish
  meaning: "` + an `interpretation_ref` to the vague-term requirement
  `visual_impressive_semantics` + `"."`. The vague_term surfaced at a DIFFERENT state
  (`72f9559b…`). At resolve, live skeleton S2 ≠ surfacing skeleton S1 → the new gate
  rejects (correctly: the user reviewed S1 and never saw S2's composer-authored framing
  bytes). Bricks under BOTH old and new gates, regardless of resolve order. The fix does
  NOT address Case B.

Staging vague_term tally (8-run batch, stopped early): run-01 graduated; run-02/06
infra-timeout (cleared/limited by provider latency, not the brick); **run-05 BRICKED
(Case B)**. So elspeth-e51216d305 is NOT resolved by 22168c227 alone.

### Case B is a composer staging-ORDER bug (fix at source)

Root cause of the residual: the composer surfaces the `llm_prompt_template` review
BEFORE it finishes wiring the vague-term slot + framing into the prompt structure, then
mutates the skeleton afterward — invalidating a review it already surfaced. The backend
gate is behaving correctly (the only unreviewed bytes in S2 are the composer's framing
`"\n\nOverall visual polish meaning: "` + `"."`; the vague *value* itself WAS approved
via its own vague_term review). Relaxing the backend to bless S2 silently would record
the user as approving framing text they never saw — an audit-integrity problem.

Candidate fixes (each "at source", different layer — operator to choose scope):
1. **Composer:** wire all vague-term slots + framing into the prompt BEFORE surfacing the
   prompt-template review (preferred, but ordering is LLM-driven → reliability question).
2. **Backend:** when a pending prompt-template review's skeleton ≠ live, supersede/
   regenerate it against the current structure so the user re-reviews what runs (may
   subsume A and B; but a regenerated card won't appear in the tutorial store without a
   refresh — the same incremental-store gap noted above).
3. **Backend tolerance:** accept when the added live parts are `interpretation_ref`s to
   already-resolved requirements (rejected as primary — silently blesses composer-authored
   framing text the user never reviewed; audit-integrity).
4. **Composer:** fold the framing text into the vague-term slot's reviewed content,
   collapsing B into A.

The committed fix (22168c227) is kept: correct for Case A, masks nothing (Case B still
surfaces), compatible with all four paths.

## UPDATE 2026-06-06 — SME panel (solution-architect + systems-thinker + llm-specialist) + advisor

Three SMEs converged; advisor pressure-tested. Consensus:
- KEEP 22168c227 (unanimous; correct for Case A, masks nothing).
- Candidate 1 (skill-prompt staging order) is EMPIRICALLY DEAD: pipeline_composer.md:600-640
  ALREADY instructs exactly that ordering, and the composer violated it in run-05.
  LLM-driven ordering is not enforcement.
- The surfacing-time guard (mirror vague_term_wiring_count, sessions.py:1279) does NOT
  work for Case B: the review surfaces against a clean S1, then the skeleton grows to S2
  AFTER. Must enforce at the mutation/finalization boundary, not surfacing.
- Candidates 3/4 fail (prompt_structure_hash hashes text AND slot requirement-ids →
  folding framing into the slot still changes the hash; tolerance blesses unreviewed bytes).
- The fix is a DETERMINISTIC BACKEND INVARIANT at the compose finalization seam
  (composer/service.py `_try_terminate_no_tools` ~2209 / `_missing_pending_interpretation_review_sites`
  ~1293 / `_orphaned_interpretation_review_validation` ~757; precedent = commit 33f05f186).

CODE FINDINGS that decide the remaining FORK:
- The `llm_prompt_template` REQUIREMENT is auto-staged on EVERY llm node (mutation-time
  auto-stager, interpretation_state.py:639,659) → a prompt-template review is a
  DETERMINISTIC, non-selective obligation, and its draft IS the backend-held rendered prompt.
- The EVENT is currently surfaced by the LLM via `request_interpretation_review`
  (service.py:622,3261) — so backend-derived surfacing means re-routing that.
- NO clean supersede primitive: InterpretationChoice has ABANDONED ("session ended without
  resolution") but no SUPERSEDED. Detect-and-repair needs a NEW audit-governed enum value +
  DB CheckConstraint (governed contract change, NOT free).

THE FORK (operator scope decision; plan-not-implement):
- **Path 2 — backend-derived surfacing (eliminate-the-class; advisor's + my lean):** the
  backend auto-surfaces the prompt-template review at turn finalization against the FROZEN
  final skeleton; make `request_interpretation_review` reject kind=llm_prompt_template so the
  LLM can't surface it early. No staleness by construction, NO supersede primitive, no
  ordering dependency, prevents future "Case C". Cost: re-route surfacing (reuse
  create_pending_interpretation_event) + skill change. Bigger conceptual change, smaller
  long-run surface.
- **Path 1 — detect-and-repair at finalization (panel's seam pick):** reuse 33f05f186
  fail-closed + repair machinery; supersede the stale S1 review + re-surface S2. Smaller
  seam change but ADDS a governed SUPERSEDED audit-enum + supersede semantics; inherently
  more failure surface; doesn't prevent Case C.
- **Defer:** keep 22168c227, file Case B with this analysis, stop.

Frontend: NO change needed for either backend path IF reconciliation completes server-side
before the turn returns — the tutorial surfaces reviews via a BATCH refreshAll AFTER the
compose turn (TutorialTurn2Describe.tsx:222; architect-verified). Confirm the freeform
compose path refreshes too (parity).

Case-B unit test still deferred — expected behavior differs by chosen path.

## Files touched while investigating

Backend fix 22168c227 (service.py + test). Staging verification read-only (temp config
deleted). Case B fix pending operator path choice — case-B unit test deferred until then
(expected behavior differs by chosen path).
