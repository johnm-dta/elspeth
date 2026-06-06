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

## Files touched while investigating

Read-only. No code changed. Issue claimed by `claude-debug` (in_progress) pending operator decision.
