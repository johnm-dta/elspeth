# Frontend intent-primary guided surface + remove tutorial freeform dead-end — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax.

**Goal:** Make every guided phase lead with a plain-English intent box rendered ABOVE the structured form, recaptioned per phase, and make a tutorial session structurally incapable of reaching the panel-less freeform surface (concern B).

**Architecture:** This is the `p2` slice of the 4-plan LLM-primary series (`docs/superpowers/specs/2026-06-25-llm-primary-guided-creation-design.md`). It is a pure-frontend, presentation-only change to one React component (`ChatPanel.tsx`) plus the tutorial shell (`TutorialGuidedShell.tsx`): (1) reorder the existing `guided-step-chat` "intent box" above the `guided-current-decision` form and recaption it per phase; (2) thread a client-only `isTutorial` React prop that suppresses both freeform exits and redirects the discriminator's fall-through to a guided placeholder. **The intent box's *apply* behavior is NOT this plan's work** — it rides the unchanged store action `chatGuided` whose backend `/guided/chat` apply contract is delivered by `p1`. **Dependencies:** p2 consumes p1's `/guided/chat` apply contract (behavioral, same wire shape — see Consumes blocks); p4 (tutorial scenario) depends on p2's concern-B `isTutorial` behavior. p1 and p3 can start independently; p2 can be written/tested before p1 lands because the reorder + concern-B are observable without the apply behavior.

**Tech Stack:** React 18 + TypeScript, Zustand store (`sessionStore`), Vitest + Testing Library (`vitest run`), ESLint, `tsc --noEmit`. No backend, no wire/schema/DB change.

## Global Constraints

- All work lands on `release/0.7.0` (the named release branch), NOT a feature
  branch. Verify `git branch --show-current` before committing; feature branches
  get orphaned.
- The agent SIGNS NOTHING. The operator holds the HMAC key and pushes per the
  release-train process. Do not proactively re-sign tier-model fingerprints or
  plugin hashes; surface owed re-signs as an operator chore.
- Editing a plugin file (e.g. `src/elspeth/plugins/transforms/llm/transform.py`)
  trips TWO CI gates, both operator-owed re-sign chores: (a) the plugin
  `source_file_hash` gate (`plugin-contract-plugin-hashes`) — refresh via
  `scripts/cicd/plugin_hash.py` (`compute_source_file_hash`/`fix_source_file_hash`);
  (b) the tier-model fingerprint cascade (`trust-tier-model`; adding imports
  shifts `Module.body` indices) — allowlists `config/cicd/enforce_tier_model/plugins.yaml`
  (plugin files) and `.../web.yaml` (web files: interpretation_state.py, state.py),
  rotated via `elspeth_lints.rules.trust_tier.tier_model.rotate`
  (scripts/cicd/rotate_tier_model_fingerprints.py). Co-land the fingerprint/hash
  updates with the source change; the operator re-signs.
- The canonical tutorial prompt couples FOUR things in lockstep: the backend
  constant `CANONICAL_SEED_PROMPT` (`web/preferences/tutorial_cache.py`), its
  byte-identical FRONTEND MIRROR `CANONICAL_TUTORIAL_PROMPT`
  (`frontend/src/components/tutorial/tutorialMachine.ts`, byte-identity enforced
  by `test_canonical_seed_matches_frontend_constant`), the `composer_skill_hash`
  re-bake (`PIPELINE_COMPOSER_SKILL_HASH` in `composer/prompts.py` +
  `assert_skill_hash_unchanged_on_disk`) when the live `pipeline_composer.md`
  skill changes, AND a live-prompt SERVICE RESTART. Editing the prompt constant
  alone needs the mirror + the two value-assert tests (NO restart). Editing the
  live skill/recipe needs the re-baked hash + restart (the 5-input
  `tutorial_model_id` invalidates the cache). Do not conflate the two.
- Prompt-shield reviews and advisor/checkpoint reviews are ADVISORY and NEVER
  hard-block: emitted into `validate()` `warnings` at "medium", excluded from the
  blocking contract. Do not promote them to errors.
- `entry_seed` (the tutorial framing/dataset seed) is SERVER-SIDE ONLY and never
  rides the wire: it is redacted from `WorkflowProfileResponse`. Do not add it to
  any wire/GET shape, and do not infer "tutorial" from the wire profile booleans
  (use the client-only `isTutorial` React prop).
- Existing tests that assert about-to-change behavior must be UPDATED to the new
  behavior, NOT reverted. A wave of failures after a structural change is the
  change landing visibly; update the assertions, do not roll back the change.
- For code commits use `git commit` with
  `SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier` (the
  operator-owed-re-sign gates) — NEVER a blanket `--no-verify`. Doc-only commits
  may use `--no-verify`. Reconcile the full slice diff at the slice boundary.
- NEVER `git add -A` / `git add .`; stage only the files this plan owns.

> **p2 SCOPE NOTE (read before any task).** Every file this plan touches is
> TypeScript/TSX under `src/elspeth/web/frontend/`. The Python re-sign gates in
> the constraints block above (`SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier`,
> plugin-hash, tier-model) **do not fire on `.tsx`/`.ts` files** — so this plan's
> commit steps use a plain `git commit` (no `SKIP=`). The relevant local
> verification before commit is the frontend gate set, run from the frontend
> directory:
> ```
> cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npm run lint && npx vitest run src/components/chat src/components/tutorial
> ```
> The constraints block is reproduced verbatim because it is the shared
> cross-plan contract; the Python-specific lines simply do not apply to p2's diff.

> **DO NOT (cross-plan boundaries, pinned):**
> - Do NOT modify the store action `chatGuided` (`src/elspeth/web/frontend/src/stores/sessionStore.ts:1333-1392`).
>   The intent box already routes through it; p1 makes it apply-capable on the
>   backend. p2 keeps the wiring unchanged.
> - Do NOT modify `respondGuided` or the `/guided/respond` path — the structured
>   form's submit stays as-is.
> - Do NOT add a wire/GET field, a new `WorkflowProfile` boolean, or any persisted
>   tutorial flag. `isTutorial` is a client-only React prop. (A wire flag was
>   considered and REJECTED — ground truth Q4.)
> - Do NOT infer "tutorial" from `profile.bookends` / `profile !== null`. Only the
>   client-only prop is the discriminator.
> - Do NOT gate or alter `<InlineOptOutCheckbox />` — it flips the account-level
>   default-mode preference; it is NOT a freeform exit. Touching it is scope creep.

---

## Shared interfaces this plan CONSUMES (from the contract)

From `/tmp/.../llm-primary-contract.md` §2.1 (p1 PRODUCES, p2 CONSUMES) — **wire
shapes UNCHANGED; the change is behavioral**:

- **`chatGuided`** (`sessionStore.ts:1333-1392`) → `POST /guided/chat`. Already
  merges `next_turn ?? prev` / `terminal ?? prev` / `composition_state ?? prev`.
  After p1: an actionable submit returns a re-rendered current-phase `next_turn`
  populated from the applied `step_N_result`, `step` UNCHANGED (apply-in-place);
  a non-actionable submit returns advisory prose with `next_turn=null`. **p2
  requires no change to consume this** — the existing merge already re-renders the
  form when `next_turn` arrives. p2's only job is layout + concern-B.
- **`respondGuided`** (`sessionStore.ts:1235-1280`) → `POST /guided/respond`.
  Unchanged; the structured form's explicit-confirm/advance seam.

## Shared interfaces this plan PRODUCES (consumed by p4)

- **`ChatPanel` gains an optional `isTutorial?: boolean` prop** (default
  `undefined`/falsy). When truthy: (i) `<ExitToFreeformButton/>` is suppressed,
  (ii) the `CompletionSummary` "Open freeform editor" button is suppressed, (iii)
  the discriminator's fall-through to the freeform body is replaced by a guided
  placeholder surface (`data-testid="tutorial-guided-loading"`). p4's tutorial
  worked example relies on this to guarantee "tutorial never reaches freeform."
- **`TutorialGuidedShell` renders `<ChatPanel isTutorial />`** — the only render
  site that passes the prop truthy. `App.tsx:385` keeps `<ChatPanel />` (prop
  absent → all current non-tutorial behavior preserved).

---

## Task ordering & test-impact map (READ FIRST)

There are TWO independent change-axes; keep them separate so the right tests
break for the right reason:

- **Axis 1 — reorder + recaption (Task 1).** This is UNIVERSAL (every guided
  phase, tutorial or not). It moves the `guided-step-chat` section above the
  `guided-current-decision` form and flips the advisory captions/placeholders to
  intent copy. The `GUIDED_CHAT_PLACEHOLDERS` map has FIVE entries; FOUR of them
  change (step_1_source, step_2_sink, step_2_5_recipe_match, step_3_transforms)
  and step_4_wire stays a confirm. **Tests that genuinely break here:** the
  per-step placeholder tests `ChatPanel.test.tsx:684-735` (which today pin only
  STEP_1_SOURCE / STEP_2_SINK / STEP_4_WIRE) PLUS two NEW per-step placeholder
  tests this task adds for STEP_2_5_RECIPE_MATCH and STEP_3_TRANSFORMS (so every
  changed entry is pinned, not just two of four) — these are *tests-to-update /
  tests-to-add* to the new intent placeholders. The
  DOM-order regression test 964-1001 keeps passing (it asserts the log region
  wraps GuidedTurn and the exit button is OUTSIDE it — both still true after the
  move; the section just moves earlier in document order).
- **Axis 2 — concern-B `isTutorial` suppression (Tasks 2–4).** This is ADDITIVE:
  `isTutorial` defaults falsy, so **all non-tutorial tests stay GREEN unchanged**
  — `ChatPanel.test.tsx:430-462` (exit button present), `931-962` (exited_to_freeform
  falls through to freeform body), `964-1001` (exit button outside log region).
  Concern-B coverage is NEW tutorial-variant sibling tests, NOT rewrites of the
  green non-tutorial tests. **If any of 430-462 / 931-962 / 964-1001 goes red,
  the `isTutorial` default is wrong — fix the default, do not edit those tests.**

Execute Task 1 → Task 2 → Task 3 → Task 4 in order.

---

### Task 1: Reorder the intent box above the form + recaption per phase

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
  - `GUIDED_CHAT_PLACEHOLDERS` map (`:463-469`) — recaption to per-phase intent copy.
  - Guided-active branch (`:1264-1382`) — move the `guided-step-chat` `<section>`
    (`:1365-1378`) to ABOVE the `guided-current-decision` `<section>` (`:1310-1340`);
    update the `<h2>` heading (`:1370`) and the section `aria-label` (`:1368`).
  - The "Per-step conversational chat input (Phase A slice 4)" comment at
    `:1346-1364` (deleted with the section it precedes) and the "ABOVE the wizard
    turn" comment at `:1299-1308` (rewritten to describe the new order).
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`
  - The three EXISTING per-step placeholder tests (`:684-703`, `:705-719`,
    `:721-735`): update STEP_1_SOURCE and STEP_2_SINK to the new intent copy;
    leave STEP_4_WIRE unchanged.
  - ADD two NEW per-step placeholder tests for the two intermediate phases whose
    copy also changes — STEP_2_5_RECIPE_MATCH and STEP_3_TRANSFORMS — so all four
    changed `GUIDED_CHAT_PLACEHOLDERS` entries are pinned (only step_4_wire is
    unchanged and stays uncovered by a dedicated test, matching its today state).
  - Add one new top-to-bottom DOM-order assertion.

**Interfaces:**
- Consumes: `chatGuided` and `respondGuided` store actions (UNCHANGED — see
  Consumes block above); `GuidedChatHistory`, `GuidedHistory`, `GuidedTurn`,
  `GuidedInterpretationReviews`, `ChatInput`, `ExitToFreeformButton`,
  `InlineOptOutCheckbox`, `InlineRunResults` (all existing imports, unchanged);
  `GUIDED_CHAT_MESSAGE_MAX_LENGTH`, `GUIDED_STEP_PURPOSES` (`:1723-1729`),
  `guidedSession.step: GuidedStep`.
- Produces: the new top-to-bottom order of the guided-active branch (consumed by
  Task 2's suppression edits, which gate elements within this layout).

**New DOM order (top → bottom) of the guided-active branch** (this is the
contract Task 2 layers onto):
1. `<GuidedWorkflowStepper>` (unchanged, `:1271`)
2. error banner (unchanged, `:1272-1274`)
3. `<GuidedHistory>` (unchanged, `:1298`)
4. `<GuidedChatHistory>` (unchanged, `:1309`)
5. **`<section className="guided-step-chat">` — the intent box (MOVED UP)** — at
   `:1365-1378`. The CSS classnames (`guided-step-chat` /
   `guided-step-chat-heading`) are KEPT UNCHANGED: they are presentation-internal
   identifiers that `guided.css` (lines 82, 95, 116 — dashed border, elevated
   background, heading typography) keys on, and the contract (§2.1) requires
   recaptioning the heading TEXT + `aria-label`, not renaming the CSS identifier.
   Renaming the classnames without touching `guided.css` would silently strip the
   styling and no automated gate (tsc/ESLint/jsdom) would catch it — so this plan
   leaves the classnames alone and changes only the visible heading text and the
   `aria-label`.
6. `<section className="guided-current-decision">` — the editable form (`:1310-1340`).
7. `<ExitToFreeformButton/>` (`:1341`) — Task 2 gates this on `!isTutorial`.
8. `<InlineOptOutCheckbox/>` (`:1345`) — untouched.
9. `<InlineRunResults/>` (`:1379`).

- [ ] **Step 1: Update + add the per-step placeholder tests to the new intent copy (failing first).**
  Four of the five `GUIDED_CHAT_PLACEHOLDERS` entries change; only `step_4_wire`
  stays a confirm. Today only THREE per-step tests exist (STEP_1_SOURCE,
  STEP_2_SINK, STEP_4_WIRE). Update the two changing ones AND add two NEW tests
  so every changed entry is pinned by a failing-first assertion (important review
  finding: don't let step_2_5_recipe_match / step_3_transforms slip in untested).

  (a) Update the two EXISTING changing tests. New copy:
  - STEP_1_SOURCE (`:700-702`): `"Describe the source you want — e.g. a CSV, a store query, or pages to scrape…"`
  - STEP_2_SINK (`:716-718`): `"Describe the output you want — the shape and fields the pipeline should produce…"`
  - STEP_4_WIRE (`:732-734`): `"Confirm how the steps connect, then continue."` (UNCHANGED — wire is a confirm, not an intent phase; spec §"Wire phase"). Leave this test as-is.

  Apply each edit:
  ```
  // :700-702
      expect(chatInput.dataset.placeholder).toBe(
        "Describe the source you want — e.g. a CSV, a store query, or pages to scrape…",
      );
  ```
  ```
  // :716-718
      expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
        "Describe the output you want — the shape and fields the pipeline should produce…",
      );
  ```

  (b) ADD two NEW per-step placeholder tests, mirroring the existing structure
  (set `guidedSession.step`, render `<ChatPanel />`, assert the chat-input
  placeholder). Use the NEW intent copy for the assertion — these are red-first
  because the map still holds the OLD strings
  (`"Ask about the suggested recipe or alternatives…"` and
  `"Ask about the proposed transform chain…"`, verified at ChatPanel.tsx:466-467)
  until Step 2 recaptions the map. Append after the STEP_4_WIRE test
  (`:721-735`), inside the same describe block:
  ```ts
    it("renders the per-step placeholder for STEP_2_5_RECIPE_MATCH", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: { ...activeGuidedSession(), step: "step_2_5_recipe_match" },
        guidedNextTurn: singleSelectTurn(),
      });

      render(<ChatPanel />);

      expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
        "Describe how this recipe should change, or accept it as proposed…",
      );
    });

    it("renders the per-step placeholder for STEP_3_TRANSFORMS", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: { ...activeGuidedSession(), step: "step_3_transforms" },
        guidedNextTurn: singleSelectTurn(),
      });

      render(<ChatPanel />);

      expect(screen.getByTestId("chat-input").dataset.placeholder).toBe(
        "Describe what each row should become, or how to fix the proposed transforms…",
      );
    });
  ```

  Run to verify the FOUR changing placeholder tests FAIL (the map still holds
  the old strings):
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx -t "per-step placeholder"
  ```
  Expected: `STEP_1_SOURCE`, `STEP_2_SINK`, `STEP_2_5_RECIPE_MATCH`,
  `STEP_3_TRANSFORMS` FAIL with `expected "Ask about …" to be "Describe …"`;
  `STEP_4_WIRE` PASSES.

- [ ] **Step 2: Recaption the placeholder map.**
  In `ChatPanel.tsx`, edit `GUIDED_CHAT_PLACEHOLDERS` (`:463-469`) to the intent
  copy (recaption is the spec's "leads with a plain-English intent box"; the
  per-phase captions invite plain English — spec §"Core model" point 1):
  ```ts
  const GUIDED_CHAT_PLACEHOLDERS: Record<GuidedStep, string> = {
    step_1_source:
      "Describe the source you want — e.g. a CSV, a store query, or pages to scrape…",
    step_2_sink:
      "Describe the output you want — the shape and fields the pipeline should produce…",
    step_2_5_recipe_match:
      "Describe how this recipe should change, or accept it as proposed…",
    step_3_transforms:
      "Describe what each row should become, or how to fix the proposed transforms…",
    step_4_wire: "Confirm how the steps connect, then continue.",
  };
  ```

  Run to verify the placeholder tests now PASS:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx -t "per-step placeholder"
  ```
  Expected: 5 passed (STEP_1_SOURCE, STEP_2_SINK, STEP_2_5_RECIPE_MATCH,
  STEP_3_TRANSFORMS, STEP_4_WIRE).

- [ ] **Step 3: Move the intent `<section>` above the form and recaption the heading/aria-label.**
  In the guided-active branch of `ChatPanel.tsx`:

  (a) DELETE the entire `guided-step-chat` `<section>` block currently at
  `:1365-1378` (from `<section\n          className="guided-step-chat"` through its
  closing `</section>`), INCLUDING the comment block at `:1346-1364` that
  precedes it.

  (b) INSERT, immediately AFTER the `<GuidedChatHistory chatHistory={...} />`
  line (`:1309`) and BEFORE the `<section className="guided-current-decision">`
  line (`:1310`), the recaptioned intent section:
  ```tsx
        {/*
          Intent box (LLM-primary, spec §"Core model" point 1). This is the
          PRIMARY input for the phase and renders ABOVE the editable form.
          It routes the learner's plain English through `chatGuided`
          (/guided/chat), which p1 makes apply-capable: an actionable submit
          re-proposes + applies this phase's config IN PLACE (step pointer
          unchanged) and the form below re-renders from the new step_N_result;
          a question / prose / failure falls back to advisory prose with no
          mutation. p2 owns only the layout + caption; the apply behavior is
          p1's backend contract (chatGuided is unchanged here).

          The caption is keyed on the live `guidedSession.step` via
          GUIDED_CHAT_PLACEHOLDERS (closed list at module top), recaptioned
          per phase to invite intent ("Describe the source you want…") rather
          than advisory Q&A.

          `disabled={guidedChatPending}` blocks rapid double-submits while a
          /guided/chat round-trip is in flight.
        */}
        <section
          className="guided-step-chat"
          role="region"
          aria-label="Describe what you want"
        >
          <h2 className="guided-step-chat-heading">Describe what you want</h2>
          <ChatInput
            onSend={(content) => void chatGuided(content)}
            disabled={guidedChatPending}
            inputRef={inputRef}
            placeholder={GUIDED_CHAT_PLACEHOLDERS[guidedSession.step]}
            maxLength={GUIDED_CHAT_MESSAGE_MAX_LENGTH}
          />
        </section>
  ```
  > **Keep the CSS classnames (`guided-step-chat` / `guided-step-chat-heading`)
  > UNCHANGED.** The recaption the contract (§2.1) asks for is the heading TEXT
  > ("Ask about this step" → "Describe what you want") and the `aria-label`
  > ("Ask about this step" → "Describe what you want") — NOT the CSS class
  > identifiers. `guided.css` keys on `.guided-step-chat` (lines 82, 116: dashed
  > border + elevated background) and `.guided-step-chat-heading` (line 95:
  > heading typography). Renaming the classnames here without also editing
  > `guided.css` would silently strip the intent box's styling, and NO automated
  > gate (`tsc --noEmit`, ESLint, Vitest/jsdom) inspects CSS class names — the
  > regression would ship invisible. This plan therefore does NOT touch
  > `guided.css` at all. (The section MOVE does not shift styling: the three
  > rules are plain class selectors, not sibling/`nth-` combinators — verified
  > against guided.css:82-119.)

  (c0) Fix one stale comment in the TEST file that the reorder falsifies. In
  `ChatPanel.test.tsx`, the guided-active test at `:430` has a comment at
  `:464-468` describing the ChatInput as "rendered INSIDE the guided-active
  branch (below GuidedTurn + ExitToFreeformButton)". After the move it is ABOVE
  the form. The test itself only asserts presence (`:470`, no positional check),
  so it stays green — but fix the comment so the file does not ship a false
  claim. Replace the comment text (`:464-468`) with:
  ```
      // Phase A slice 4 / LLM-primary reorder: the intent ChatInput is rendered
      // INSIDE the guided-active branch, now ABOVE the GuidedTurn form (it is
      // the primary input). This test asserts presence only; per-step
      // placeholder + onSend wiring are exercised in the dedicated tests below.
  ```

  (c) Rewrite the now-stale "ABOVE the wizard turn" comment at `:1299-1308`
  (it described GuidedChatHistory's placement relative to the wizard turn; that
  is still accurate, but it also said the ChatInput "at the bottom of the branch
  is where they reply" — that is no longer true). Replace its last sentence so it
  reads:
  ```tsx
        {/*
          Per-step chat log (Phase A slice 6). Placed ABOVE the editable
          form. GuidedChatHistory carries its OWN role="log" + aria-live so
          new chat turns are announced independently of wizard turn advances.
          The intent box (the reply surface) now renders directly below this
          log and above the form — see the "Intent box" section. Empty-state
          returns null; no DOM contribution before the first chat exchange.
        */}
  ```

  Run typecheck + the full ChatPanel suite (the intent-box tests at `:737-759`
  and `:761-...` that assert `chatGuided` wiring + maxLength must still pass —
  they query by the `chat-input` testid, which is position-independent):
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npx vitest run src/components/chat/ChatPanel.test.tsx
  ```
  Expected: all pass. (If 964-1001 fails, the move accidentally nested the intent
  section inside the log region — re-check it sits as a sibling between
  `GuidedChatHistory` and `guided-current-decision`.)

- [ ] **Step 4: Add a DOM-order assertion pinning intent-above-form.**
  Append to `ChatPanel.test.tsx`, inside the same `describe` block that holds the
  guided-active tests (after the placeholder tests; reuse the existing
  `activeGuidedSession()` and `singleSelectTurn()` helpers in this file):
  ```ts
    it("renders the intent box ABOVE the editable form (LLM-primary layout)", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: activeGuidedSession(),
        guidedNextTurn: singleSelectTurn(),
      });

      const { container } = render(<ChatPanel />);

      // Classnames are unchanged (Option B): the intent box keeps
      // `.guided-step-chat`; only its heading text + aria-label were recaptioned.
      const intent = container.querySelector(".guided-step-chat");
      const form = container.querySelector(".guided-current-decision");
      expect(intent).not.toBeNull();
      expect(form).not.toBeNull();
      // Document-order: intent precedes form. compareDocumentPosition returns
      // DOCUMENT_POSITION_FOLLOWING (4) when `form` follows `intent`.
      expect(
        intent!.compareDocumentPosition(form!) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
      // The intent box's recaptioned heading is present.
      expect(
        screen.getByRole("region", { name: "Describe what you want" }),
      ).toBeInTheDocument();
    });
  ```

  Run to pass:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx -t "intent box ABOVE"
  ```
  Expected: 1 passed.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npm run lint && npx vitest run src/components/chat/ChatPanel.test.tsx
  cd /home/john/elspeth && git branch --show-current   # expect: release/0.7.0
  git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx && git commit -m "$(cat <<'EOF'
feat(composer/guided-ui): lead each guided phase with an intent box above the form

Move the per-phase chat input (the intent box) ABOVE the structured
form in the guided-active branch and recaption it per phase to invite
plain English ("Describe the source you want…") rather than advisory
Q&A. This is the frontend half of LLM-primary guided creation: the
intent box is the primary input and the form is the editable result
beneath it. The box still routes through the unchanged `chatGuided`
store action; p1 makes /guided/chat apply-capable on the backend.

Update the existing per-step placeholder tests (source/sink) to the new
intent copy, add two new ones (recipe-match/transforms) so all four
changed entries are pinned, and add a DOM-order assertion pinning
intent-above-form. The CSS classnames are kept unchanged (only the
heading text + aria-label are recaptioned), so guided.css is untouched.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 2: Add the `isTutorial` prop and suppress both freeform exits

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
  - `ChatPanelProps` interface (`:471-480`) — add `isTutorial?: boolean`.
  - `ChatPanel({...})` destructure (`:488-491`) — destructure `isTutorial`.
  - Guided-active branch: gate `<ExitToFreeformButton/>` (`:1341`) on `!isTutorial`.
  - Completed branch (`:1247-1262`): pass `isTutorial` into `<CompletionSummary/>`.
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/CompletionSummary.tsx`
  - `CompletionSummaryProps` (`:31-33`) — add `isTutorial?: boolean`.
  - Thread it through to `CompletionSummaryInner` and gate the "Open freeform
    editor" button (`:111-117`) on `!isTutorial`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx` (ADD
  tutorial-variant sibling tests; do NOT edit the green non-tutorial tests).
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/CompletionSummary.test.tsx`
  (ADD a tutorial-variant test).

**Interfaces:**
- Consumes: the Task 1 layout; `CompletionSummary` (existing import,
  `ChatPanel.tsx:` — verify with grep below); `TerminalState`.
- Produces: `ChatPanel`'s `isTutorial?: boolean` prop with exit-suppression
  semantics (consumed by Task 3's fall-through redirect and by p4's tutorial).

- [ ] **Step 1: Add the failing tutorial-variant test for ExitToFreeformButton suppression.**
  Append to `ChatPanel.test.tsx` (same describe block as Task 1; reuse
  `activeGuidedSession()` / `singleSelectTurn()`):
  ```ts
    it("suppresses ExitToFreeformButton when isTutorial (concern B)", () => {
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: activeGuidedSession(),
        guidedNextTurn: singleSelectTurn(),
      });

      render(<ChatPanel isTutorial />);

      // The form still renders (manual path is never gated)...
      expect(
        screen.getByRole("group", { name: "Which source plugin should we use?" }),
      ).toBeInTheDocument();
      // ...but the freeform exit affordance is gone — a tutorial must never
      // expose a switch-to-freeform path (spec §"Frontend" concern B).
      expect(
        screen.queryByRole("button", { name: "Exit to freeform" }),
      ).toBeNull();
    });
  ```

  Run to fail (the prop does not exist yet → TS error AND the button still renders):
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx -t "suppresses ExitToFreeformButton"
  ```
  Expected: FAIL — `<ChatPanel isTutorial />` is a type error (`isTutorial` not in
  `ChatPanelProps`) and/or the `Exit to freeform` button is found.

- [ ] **Step 2: Add the prop and gate the exit button.**
  In `ChatPanel.tsx`, extend `ChatPanelProps` (`:471-480`):
  ```ts
  interface ChatPanelProps {
    onOpenSecrets?: () => void;
    onOpenComposerPreferences?: () => void;
    // Concern B (LLM-primary spec §"Frontend"): a TUTORIAL session must never
    // reach a freeform surface. This client-only flag is passed truthy ONLY by
    // TutorialGuidedShell. It is deliberately NOT a wire/profile field — there
    // is no tutorial discriminator on the wire (ground truth Q2/Q4), and
    // inferring tutorial from profile booleans is fragile. When true it (i)
    // suppresses ExitToFreeformButton, (ii) suppresses CompletionSummary's
    // "Open freeform editor" button, and (iii) redirects the discriminator's
    // freeform fall-through to a guided placeholder (Task 3).
    isTutorial?: boolean;
  }
  ```
  Update the destructure (`:488-491`):
  ```ts
  export function ChatPanel({
    onOpenSecrets,
    onOpenComposerPreferences,
    isTutorial,
  }: ChatPanelProps) {
  ```
  Gate the exit button (`:1341`) — replace `<ExitToFreeformButton />` with:
  ```tsx
        {!isTutorial && <ExitToFreeformButton />}
  ```

  Run to verify the new test passes AND the non-tutorial exit-button test
  (`:430-462`) stays green:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npx vitest run src/components/chat/ChatPanel.test.tsx -t "ExitToFreeformButton"
  ```
  Expected: `suppresses ExitToFreeformButton when isTutorial` PASS; `renders
  guided-active surface (GuidedTurn + ExitToFreeformButton) …` PASS.

- [ ] **Step 3: Add the failing CompletionSummary tutorial test.**
  Append to `CompletionSummary.test.tsx`. It reuses the module-level const
  `COMPLETED_TERMINAL` (`:48`, a `kind: "completed"` terminal with a non-null
  `pipeline_yaml`) that the existing tests already use:
  ```ts
    it("hides 'Open freeform editor' when isTutorial (concern B)", () => {
      render(<CompletionSummary terminal={COMPLETED_TERMINAL} isTutorial />);
      // The summary still renders. Bind to the SEMANTIC heading element (an
      // <h3>, CompletionSummary.tsx:87), matching the file's existing pattern
      // (CompletionSummary.test.tsx:95 uses getByRole("heading")) — getByText
      // would still pass if the heading were demoted to a paragraph.
      expect(
        screen.getByRole("heading", { name: "Pipeline ready" }),
      ).toBeInTheDocument();
      // ...but the freeform exit is suppressed in a tutorial.
      expect(
        screen.queryByRole("button", { name: "Open freeform editor" }),
      ).toBeNull();
      // The two non-freeform actions remain (exact names verified against
      // CompletionSummary.tsx:123,131). Pin BOTH presence AND the surviving
      // button count, so a regression that drops a non-freeform button can't
      // slip past an absent-button-only check.
      expect(
        screen.getByRole("button", { name: "Review YAML" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "Validate pipeline" }),
      ).toBeInTheDocument();
      expect(screen.getAllByRole("button")).toHaveLength(2);
    });
  ```
  > The load-bearing contract is: exactly TWO buttons survive under tutorial
  > ("Review YAML", "Validate pipeline"), and "Open freeform editor" is gone.
  > Button names are verified against CompletionSummary.tsx:116 ("Open freeform
  > editor"), :123 ("Review YAML"), :131 ("Validate pipeline" — note: NOT
  > "Validate"). Contract 3 (`:115`) already pins the non-tutorial "Open freeform
  > editor" presence — leave it green.

  Run to fail:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/guided/CompletionSummary.test.tsx -t "isTutorial"
  ```
  Expected: FAIL — `isTutorial` not in props (type error) and the button is found.

- [ ] **Step 4: Add the prop to CompletionSummary and gate the button.**
  In `CompletionSummary.tsx`, extend props (`:31-33`):
  ```ts
  interface CompletionSummaryProps {
    terminal: TerminalState;
    // Concern B: in a tutorial the "Open freeform editor" action is suppressed
    // (the only path out of a tutorial is graduation, never freeform).
    isTutorial?: boolean;
  }
  ```
  Thread it through the outer component (`:37-44`) into the inner one:
  ```ts
  export function CompletionSummary({ terminal, isTutorial }: CompletionSummaryProps) {
    if (terminal.kind !== "completed" || terminal.pipeline_yaml === null) {
      return null;
    }
    return (
      <CompletionSummaryInner yaml={terminal.pipeline_yaml} isTutorial={isTutorial} />
    );
  }
  ```
  Extend `CompletionSummaryInnerProps` (`:52-54`) and the inner signature (`:56`):
  ```ts
  interface CompletionSummaryInnerProps {
    yaml: string;
    isTutorial?: boolean;
  }

  function CompletionSummaryInner({ yaml, isTutorial }: CompletionSummaryInnerProps) {
  ```
  Gate the button (`:111-117`) — wrap it:
  ```tsx
          {!isTutorial && (
            <button
              type="button"
              className="guided-completion-save-btn"
              onClick={handleExit}
            >
              Open freeform editor
            </button>
          )}
  ```

  Now thread `isTutorial` from `ChatPanel`'s completed branch. In `ChatPanel.tsx`
  (`:1258`), replace `<CompletionSummary terminal={guidedSession.terminal} />`
  with:
  ```tsx
          <CompletionSummary terminal={guidedSession.terminal} isTutorial={isTutorial} />
  ```

  Run to verify CompletionSummary's tutorial test + the green Contract-3 test
  both pass, and ChatPanel typechecks:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npx vitest run src/components/chat/guided/CompletionSummary.test.tsx
  ```
  Expected: all pass (including the green `clicking 'Open freeform editor' calls
  exitToFreeform once` test, which renders WITHOUT `isTutorial`).

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npm run lint && npx vitest run src/components/chat
  cd /home/john/elspeth && git branch --show-current   # expect: release/0.7.0
  git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx src/elspeth/web/frontend/src/components/chat/guided/CompletionSummary.tsx src/elspeth/web/frontend/src/components/chat/guided/CompletionSummary.test.tsx && git commit -m "$(cat <<'EOF'
feat(composer/guided-ui): suppress freeform exits under a tutorial (concern B)

Add a client-only `isTutorial` prop to ChatPanel (and thread it into
CompletionSummary). When set, both freeform-exit affordances are
suppressed: the persistent ExitToFreeformButton in the guided-active
branch and the "Open freeform editor" action on the completed surface.
A tutorial must never expose a switch-to-freeform path. The flag is NOT
a wire/profile field (there is no tutorial discriminator on the wire);
it is passed truthy only by TutorialGuidedShell (next task). Default-off
keeps all non-tutorial behavior unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 3: Redirect the discriminator fall-through to a guided placeholder under tutorial

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
  - Insert a tutorial guard immediately BEFORE the final freeform `return` (`:1384`).
  - Update the discriminator doc-comment (`:1225-1246`) to document the guard.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx` (ADD
  two tutorial-variant fall-through tests; do NOT edit the green `931-962` test).

**Interfaces:**
- Consumes: `isTutorial` (Task 2); the discriminator branches (`:1247`, `:1264`).
- Produces: a guided placeholder surface (`data-testid="tutorial-guided-loading"`)
  that replaces the freeform body for tutorials in BOTH the startup-flash
  (`guidedSession===null` / `guidedNextTurn===null`) and `exited_to_freeform`
  cases (consumed by p4's "tutorial never reaches freeform" guarantee).

**Why ONE guard covers both cases (ground truth Q3 §3):** the final `return` at
`:1384` is reached whenever neither the completed branch (`:1247`) nor the
guided-active branch (`:1264`) matched — i.e. (a) the TutorialGuidedShell
startup flash where `guidedSession`/`guidedNextTurn` are transiently null
(`TutorialGuidedShell.tsx:61-81`), and (b) any `exited_to_freeform` terminal.
A single `if (isTutorial) return <placeholder>` placed just before `:1384`
catches both. It is positioned AFTER the completed branch so a tutorial
completion still graduates normally (the completed branch returns first).

- [ ] **Step 1: Add the failing test for the startup-flash redirect.**
  Append to `ChatPanel.test.tsx`:
  ```ts
    it("renders a guided placeholder (not the freeform body) when isTutorial and the session is still loading (concern B startup flash)", () => {
      // TutorialGuidedShell clears guidedSession/guidedNextTurn to null before
      // the async start resolves (TutorialGuidedShell.tsx:61-81). Without the
      // tutorial guard, ChatPanel would fall through to the panel-less freeform
      // body during that window.
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: null,
        guidedNextTurn: null,
      });

      render(<ChatPanel isTutorial />);

      // Guided placeholder present...
      expect(
        screen.getByTestId("tutorial-guided-loading"),
      ).toBeInTheDocument();
      // ...and the freeform composer input is NOT rendered.
      expect(screen.queryByTestId("chat-input")).toBeNull();
    });
  ```

  Run to fail (no guard → freeform body renders, `chat-input` present, placeholder absent):
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx -t "guided placeholder"
  ```
  Expected: FAIL — `Unable to find element by testid: tutorial-guided-loading`.

- [ ] **Step 2: Add the tutorial guard before the freeform return.**
  In `ChatPanel.tsx`, immediately BEFORE the final `return (` at `:1384`, insert:
  ```tsx
    // ── Concern B: a tutorial must NEVER reach the panel-less freeform body ──
    //
    // Reaching this point means neither the completed branch nor the
    // guided-active branch matched. For a non-tutorial session that is the
    // legitimate freeform surface (below). For a TUTORIAL session it is one of
    // two states that must NOT show freeform:
    //   (a) the TutorialGuidedShell startup flash, where guidedSession /
    //       guidedNextTurn are transiently null before the async start resolves
    //       (TutorialGuidedShell.tsx:61-81); and
    //   (b) an `exited_to_freeform` terminal (which a tutorial can no longer
    //       trigger after Task 2 removed the exit affordances, but is guarded
    //       here defensively in case a stale persisted session carries it).
    // Both are caught by this single guard; the completed branch above returns
    // first, so a tutorial completion still graduates normally.
    //
    // The rail reflects the ACTUAL session step when one is available
    // (the exited_to_freeform case carries a real `guidedSession.step`); it
    // falls back to "step_1_source" ONLY for the startup-flash case where
    // `guidedSession === null` (no step exists yet). Hardcoding step_1 in the
    // non-null case would show the wrong step in the rail — a fidelity gap.
    if (isTutorial) {
      const placeholderStep: WorkflowStepId = guidedSession?.step ?? "step_1_source";
      return (
        <div
          id="chat-main"
          className="chat-panel chat-panel--guided"
          aria-label="Guided composer"
          data-testid="tutorial-guided-loading"
        >
          <GuidedWorkflowStepper activeStep={placeholderStep} />
          <p role="status" className="guided-loading-status">
            Preparing your guided pipeline…
          </p>
        </div>
      );
    }

  ```
  > `GuidedWorkflowStepper` is defined locally in `ChatPanel.tsx`
  > (`function GuidedWorkflowStepper({ activeStep }: { activeStep: WorkflowStepId })`,
  > `:1743`) and already used in the guided-active branch at `:1271` — no import
  > is needed. `guidedSession.step` is a `GuidedStep`, which is the same string
  > union the stepper's `activeStep: WorkflowStepId` accepts (the active branch
  > passes `activeStep={guidedSession.step}` at `:1271`, so the type is already
  > proven assignable). `"step_1_source"` is the null-session (startup-flash)
  > default only.

  Run to verify the startup-flash test passes AND the non-tutorial fall-through
  test (`931-962`) stays green:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npx vitest run src/components/chat/ChatPanel.test.tsx -t "guided placeholder|freeform body when terminal"
  ```
  Expected: `guided placeholder …` PASS; `falls through to the freeform body when
  terminal.kind === 'exited_to_freeform'` PASS (it renders `<ChatPanel />` with no
  prop, so the guard is skipped).

- [ ] **Step 3: Add the failing test for the exited_to_freeform redirect under tutorial.**
  Use a NON-step_1 fixture step (`step_3_transforms`) so the test actually
  exercises the `guidedSession?.step ?? "step_1_source"` fix from Step 2: if the
  guard regressed to a hardcoded `step_1_source`, the rail's `aria-current` step
  would no longer match the session's real step and this test would catch it.
  Append to `ChatPanel.test.tsx`:
  ```ts
    it("renders a guided placeholder (not the freeform body) when isTutorial and terminal is exited_to_freeform (concern B defensive)", () => {
      const terminal: TerminalState = {
        kind: "exited_to_freeform",
        reason: "user_pressed_exit",
        pipeline_yaml: null,
      };
      useSessionStore.setState({
        activeSessionId: "session-guided",
        sessions: [guidedSessionFixture],
        messages: [],
        guidedSession: {
          step: "step_3_transforms",
          history: [],
          terminal,
          chat_history: [],
          chat_turn_seq: 0,
          profile: null,
        },
        guidedNextTurn: null,
        guidedTerminal: terminal,
      });

      render(<ChatPanel isTutorial />);

      expect(screen.getByTestId("tutorial-guided-loading")).toBeInTheDocument();
      expect(screen.queryByTestId("chat-input")).toBeNull();
      // The placeholder rail reflects the ACTUAL session step, not a hardcoded
      // step_1 (GuidedWorkflowStepper marks the current step with
      // aria-current="step", ChatPanel.tsx:1759). The transform-step rail item
      // must be the current one.
      const current = screen
        .getByTestId("tutorial-guided-loading")
        .querySelector('[aria-current="step"]');
      expect(current).not.toBeNull();
      expect(current).toHaveTextContent(/transform/i);
    });
  ```
  > Confirm the transform step's rail LABEL contains "transform" against
  > `GUIDED_WORKFLOW_STEPS` (the list the stepper renders, ChatPanel.tsx near
  > `:1743`) before committing; if the visible label differs, match the actual
  > label text. The load-bearing assertion is `aria-current="step"` pointing at
  > the transform step — that is what proves the rail tracks the real step.

  Run to pass (the Step-2 guard already handles this — `exited_to_freeform` falls
  past both branches, hits the `isTutorial` guard):
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx -t "exited_to_freeform (concern B"
  ```
  Expected: 1 passed.

- [ ] **Step 4: Update the discriminator doc-comment.**
  In `ChatPanel.tsx`, extend the precedence list in the comment at `:1227-1246`.
  After the existing point 3, add a point 4:
  ```
  //   4. (tutorial only) when `isTutorial` is set, the fall-through in (3) is
  //      replaced by a guided placeholder surface instead of the freeform body,
  //      so a tutorial can never land on a panel-less freeform screen (concern
  //      B). The completed branch (1) still wins for a tutorial completion.
  ```

  Run the full ChatPanel suite to confirm no regression:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx
  ```
  Expected: all pass.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npm run lint && npx vitest run src/components/chat
  cd /home/john/elspeth && git branch --show-current   # expect: release/0.7.0
  git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx && git commit -m "$(cat <<'EOF'
fix(composer/guided-ui): tutorial never falls through to the freeform body (concern B)

When isTutorial, replace the discriminator's freeform fall-through with
a guided placeholder surface. A single guard before the freeform return
covers both reachable cases: the TutorialGuidedShell startup flash
(guidedSession/guidedNextTurn transiently null) and a defensive
exited_to_freeform terminal. The completed branch still returns first,
so a tutorial completion graduates normally. Non-tutorial sessions are
unchanged (the guard is skipped when the prop is absent).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 4: Pass `isTutorial` from TutorialGuidedShell and pin it

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx`
  - The `<ChatPanel />` render (`:149`) → `<ChatPanel isTutorial />`.
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx`
  - The ChatPanel module mock (`:30-32`) → capture props.
  - The "mounts the ChatPanel guided surface" test (`:78-88`) → assert `isTutorial`.

**Interfaces:**
- Consumes: `ChatPanel`'s `isTutorial?: boolean` prop (Task 2).
- Produces: the only render site that passes `isTutorial` truthy — the seam p4
  relies on for "tutorial never reaches freeform." `App.tsx:385` keeps
  `<ChatPanel />` (prop absent) so non-tutorial guided/freeform is unchanged.

- [ ] **Step 1: Make the shell test capture the prop and assert it (failing first).**
  In `TutorialGuidedShell.test.tsx`, change the ChatPanel mock (`:30-32`) to
  record the prop:
  ```ts
  vi.mock("@/components/chat/ChatPanel", () => ({
    ChatPanel: (props: { isTutorial?: boolean }) => (
      <div
        data-testid="chat-panel-stub"
        data-is-tutorial={String(props.isTutorial)}
      />
    ),
  }));
  ```
  Update the "mounts the ChatPanel guided surface" test (`:78-88`) to assert the
  prop reaches ChatPanel:
  ```ts
    it("mounts the ChatPanel guided surface with isTutorial set", async () => {
      // ChatPanel is stubbed at the module boundary (see vi.mock above). The
      // tutorial shell MUST pass isTutorial so ChatPanel suppresses the
      // freeform exits and never falls through to the freeform body (concern B).
      render(
        <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
      );
      const stub = await screen.findByTestId("chat-panel-stub");
      expect(stub).toBeInTheDocument();
      expect(stub.dataset.isTutorial).toBe("true");
    });
  ```

  Run to fail (shell still renders `<ChatPanel />` propless → `data-is-tutorial`
  is `"undefined"`):
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/tutorial/TutorialGuidedShell.test.tsx -t "isTutorial set"
  ```
  Expected: FAIL — `expected "undefined" to be "true"`.

- [ ] **Step 2: Pass the prop from the shell.**
  In `TutorialGuidedShell.tsx`, change the render (`:149`):
  ```tsx
        <ChatPanel isTutorial />
  ```

  Run to pass:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx vitest run src/components/tutorial/TutorialGuidedShell.test.tsx
  ```
  Expected: all pass.

- [ ] **Step 3: Verify App.tsx still renders ChatPanel WITHOUT the tutorial prop.**
  This is a read-only guard against accidentally making the whole app a tutorial.
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && grep -n "<ChatPanel" src/App.tsx
  ```
  Expected: the `App.tsx` render (around `:385`) has NO `isTutorial` — only
  `onOpenSecrets`/`onOpenComposerPreferences` (or no props). If it accidentally
  carries `isTutorial`, that is a defect — App.tsx is the non-tutorial surface and
  must NOT pass it. (No edit expected; this step only confirms.)

- [ ] **Step 4: Full slice verification — the whole chat + tutorial surface.**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npm run lint && npx vitest run src/components/chat src/components/tutorial
  ```
  Expected: all pass, zero type errors, zero lint errors. This is the slice
  boundary; reconcile any unexpected red here (a concern-B test going red on the
  non-tutorial path means an `isTutorial` default regressed — fix the default,
  not the test).

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && git branch --show-current   # expect: release/0.7.0
  git add src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx && git commit -m "$(cat <<'EOF'
feat(tutorial): pass isTutorial to the embedded ChatPanel (concern B)

TutorialGuidedShell renders the real ChatPanel with isTutorial set, so a
tutorial session suppresses both freeform-exit affordances and never
falls through to the panel-less freeform body. App.tsx keeps the propless
render for the non-tutorial guided/freeform surface. Update the shell's
ChatPanel module mock to capture and assert the prop pass-through.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Verification checklist (slice complete)

- [ ] `git branch --show-current` = `release/0.7.0` for all four commits.
- [ ] `cd src/elspeth/web/frontend && npm run typecheck` → clean.
- [ ] `npm run lint` → clean.
- [ ] `npx vitest run src/components/chat src/components/tutorial` → all green.
- [ ] Non-tutorial ChatPanel tests `430-462`, `931-962`, `964-1001` are GREEN and
      UNCHANGED (additive concern-B did not break them).
- [ ] Intent box renders ABOVE the form on every guided phase; placeholders are
      the new per-phase intent copy.
- [ ] `guided.css` is UNTOUCHED and the intent box's CSS classnames
      (`.guided-step-chat` / `.guided-step-chat-heading`) are UNCHANGED — only the
      heading text + `aria-label` were recaptioned. **Manual visual check (no
      automated gate inspects CSS class names — tsc/ESLint/jsdom cannot catch a
      styling loss):** the intent box still shows its dashed border and elevated
      background, and sits above the form, in a browser. Belt-and-braces under
      Option B (classnames unchanged ⇒ no styling can be lost), but verify once
      against the pre-change baseline since the section moved in document order.
- [ ] No change to `sessionStore.ts`, `respondGuided`, any wire/GET schema, or
      `InlineOptOutCheckbox`.
- [ ] No `GUIDED_SESSION_SCHEMA_VERSION` / `SESSION_SCHEMA_EPOCH` bump (this plan
      is presentation-only — contract §3 p2: NO bump).

## Out of scope (other plans)

- `/guided/chat` apply behavior (mutation rule, apply-in-place, no auto-advance):
  **p1**. p2 consumes the unchanged `chatGuided` action.
- Per-phase drivers (source scrape routing, sink driver, transform revise): **p1**.
- 3-state prompt-shield surface: **p3**.
- Tutorial synthetic pages, base-URL/SSRF, scenario constants, prompt-shield
  State-C copy, staging harness: **p4** (depends on this plan's `isTutorial`).
